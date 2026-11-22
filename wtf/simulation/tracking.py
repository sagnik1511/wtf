"""Bounded personal sightings and anonymous ground-scent fields; no policy/RNG use."""

from collections import OrderedDict, deque

SECTORS = ((0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
SECTOR_NAMES = (
    "here",
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
)


def scent_contract(template):
    spec = template.get("scent_observation")
    if spec is None:
        return None
    if spec.get("mode", "separated") == "blended":
        return dict(mode="blended", radius=spec.get("radius", 2))
    return dict(
        species=list(spec["species"]),
        radius=spec.get("radius", 2),
        separate_sexes=spec.get("separate_sexes", True),
    )


def scent_channels(template):
    spec = scent_contract(template)
    if spec is None:
        return []
    if spec.get("mode") == "blended":
        return [("blended", "any")]
    return [
        (name, sex)
        for name in spec["species"]
        for sex in (("female", "male") if spec["separate_sexes"] else ("any",))
    ]


def tracking_width(template):
    steps = template.get("movement_history_steps", 0)
    return (
        2 * template.get("observation_top_k", 1) * (7 + 3 * steps) if steps else 0
    ) + 9 * len(scent_channels(template))


def tracking_names(template):
    names = []
    steps = template.get("movement_history_steps", 0)
    if steps:
        for category in ("Prey", "Predator"):
            for slot in range(1, template.get("observation_top_k", 1) + 1):
                prefix = f"{category} track {slot}"
                names.extend(
                    f"{prefix} {part}"
                    for part in (
                        "last dx",
                        "last dy",
                        "velocity x",
                        "velocity y",
                        "age",
                        "confidence",
                        "visible",
                    )
                )
                for lag in range(1, steps + 1):
                    names.extend(
                        f"{prefix} lag {lag} {part}" for part in ("dx", "dy", "seen")
                    )
    for species, sex in scent_channels(template):
        prefix = (
            "Scent blended"
            if species == "blended"
            and scent_contract(template).get("mode") == "blended"
            else f"Scent {species} {sex}"
        )
        names.extend(f"{prefix} {side}" for side in SECTOR_NAMES)
    return names


class VisualMemory:
    def __init__(self):
        self.frames = {}

    def observe(self, animal, tick, radius, k, steps, targets):
        frames = self.frames.setdefault(animal.id, deque())
        while frames and frames[0][0] < tick - steps:
            frames.popleft()
        current = {target_id: (category, x, y) for category, target_id, x, y in targets}
        if frames and frames[-1][0] == tick:
            frames[-1] = (tick, current)
        else:
            frames.append((tick, current))
        # Newest personal sighting wins. Hidden/dead targets are never looked up
        # in world state, and targets first seen now have no backfilled history.
        latest = {}
        for when, seen in reversed(frames):
            for target_id, point in seen.items():
                latest.setdefault(target_id, (when, point))
        by_tick = dict(frames)
        features = []
        for category in (0, 1):
            ids = [i for i, (kind, _, _) in current.items() if kind == category]
            ids += [
                i
                for i, (_, (kind, _, _)) in latest.items()
                if kind == category and i not in current
            ]
            for slot in range(k):
                if slot >= len(ids):
                    features.extend([0.0] * (7 + 3 * steps))
                    continue
                target_id = ids[slot]
                when, (_, x, y) = latest[target_id]
                previous = next(
                    (
                        (t, p)
                        for t, seen in reversed(frames)
                        if t < when and (p := seen.get(target_id)) is not None
                    ),
                    None,
                )
                vx = vy = 0.0
                if previous is not None:
                    t, (_, px, py) = previous
                    vx, vy = (x - px) / ((when - t) * radius), (y - py) / (
                        (when - t) * radius
                    )
                age = tick - when
                features.extend(
                    (
                        (x - animal.x) / radius,
                        (y - animal.y) / radius,
                        vx,
                        vy,
                        age / steps,
                        1 - age / (steps + 1),
                        float(target_id in current),
                    )
                )
                for lag in range(1, steps + 1):
                    p = by_tick.get(tick - lag, {}).get(target_id)
                    features.extend(
                        ((p[1] - animal.x) / radius, (p[2] - animal.y) / radius, 1.0)
                        if p is not None
                        else (0.0, 0.0, 0.0)
                    )
        return features

    def checkpoint(self):
        return [
            [i, [[t, [[j, *p] for j, p in seen.items()]] for t, seen in frames]]
            for i, frames in self.frames.items()
        ]

    @classmethod
    def restore(cls, state):
        result = cls()
        result.frames = {
            i: deque(
                (t, {j: (kind, x, y) for j, kind, x, y in points})
                for t, points in frames
            )
            for i, frames in state
        }
        return result


class ScentField:
    def __init__(self, config, width):
        self.width = width
        self.species = sorted(config["species"])
        self.groups = {
            (name, sex): 2 * i + j
            for i, name in enumerate(self.species)
            for j, sex in enumerate(("female", "male"))
        }
        self.channels = 2 * len(self.species)
        self.lifetime = config["scent_trails"]["lifetime_steps"]
        # Sorted by most recent deposit: refreshes replace, rather than append,
        # so stationary animals do not accumulate a queue of duplicate records.
        self.cells = OrderedDict()
        self._plans = OrderedDict()

    def advance(self, world, dead=()):
        tick = world.tick
        while self.cells and tick - next(iter(self.cells.values())) >= self.lifetime:
            self.cells.popitem(last=False)
        for animals in (world.animals.values(), dead):
            for animal in animals:
                if world.blocked(animal.position) or (
                    world.geography and world.geography.depth(animal.position)
                ):
                    continue
                cell = animal.y * self.width + animal.x
                key = cell * self.channels + self.groups[animal.species, animal.sex]
                self.cells[key] = tick
                self.cells.move_to_end(key)

    def sample(self, world, animal, template):
        spec = template.get("scent_observation")
        if spec is None:
            return []
        blended = spec.get("mode", "separated") == "blended"
        key = (
            tuple(spec.get("species", ())),
            spec.get("separate_sexes", True),
            spec.get("radius", 2),
            blended,
        )
        plan = self._plans.get(key)
        if plan is None:
            plan = (
                (tuple(range(self.channels)),)
                if blended
                else tuple(
                    tuple(
                        self.groups[name, group_sex]
                        for group_sex in (
                            ("female", "male") if sex == "any" else (sex,)
                        )
                    )
                    for name, sex in scent_channels(template)
                )
            )
            self._plans[key] = plan
            if len(self._plans) > 16:
                self._plans.popitem(last=False)
        self._plans.move_to_end(key)
        features = [0.0] * (9 * len(plan))
        start = animal.position
        seen = {start}
        queue = deque([(start, 0)])
        # Ground scent follows locally reachable dry cells. It does not cross a
        # sealed wall or river, and is not a simulated airborne odour plume.
        while queue:
            (x, y), distance = queue.popleft()
            if world.blocked((x, y)) or (
                world.geography and world.geography.depth((x, y))
            ):
                continue
            dx, dy = x - animal.x, y - animal.y
            sector = SECTORS.index(((dx > 0) - (dx < 0), (dy > 0) - (dy < 0)))
            cell = (y * self.width + x) * self.channels
            for index, groups in enumerate(plan):
                combined = 0.0
                for group in groups:
                    when = self.cells.get(cell + group)
                    if when is not None:
                        strength = max(0.0, 1 - (world.tick - when) / self.lifetime)
                        combined = (
                            min(1.0, combined + strength)
                            if blended
                            else max(combined, strength)
                        )
                offset = index * 9 + sector
                features[offset] = max(features[offset], combined)
            if distance < key[2]:
                for p in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
                    if p not in seen:
                        seen.add(p)
                        queue.append((p, distance + 1))
        return features


class Tracking:
    def __init__(self, config, width):
        self.visual = VisualMemory()
        self.history_enabled = any(
            b.get("movement_history_steps", 0) for b in config["brains"].values()
        )
        self.scent = (
            ScentField(config, width)
            if config.get("scent_trails", {}).get("enabled", False)
            else None
        )

    @staticmethod
    def enabled(config):
        return config.get("scent_trails", {}).get("enabled", False) or any(
            b.get("movement_history_steps", 0) for b in config["brains"].values()
        )

    def checkpoint(self):
        result = {"version": 1, "visual": self.visual.checkpoint()}
        if self.scent is not None:
            result["scent"] = list(map(list, self.scent.cells.items()))
        return result

    @classmethod
    def restore(cls, config, width, state):
        if state is None or state.get("version") != 1:
            raise ValueError("Checkpoint has missing or unsupported sensory_state")
        result = cls(config, width)
        if state is not None:
            result.visual = VisualMemory.restore(state.get("visual", []))
            if result.scent is not None:
                result.scent.cells = OrderedDict(state.get("scent", []))
        return result
