"""Physical state and simultaneous action resolution; no learning dependencies."""

from wtf.native import kernels
from wtf.simulation.actions import (
    WALK_OFFSETS,
    WALK_ACTIONS,
    MOVE_ACTIONS,
    ATTEMPT_ACTIONS,
)


import copy
import math
import random
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from wtf.simulation.perception import SpatialIndex
from wtf.simulation.tracking import Tracking
from wtf.simulation.geography import Geography, resolve_map
from wtf.simulation.day_night import sunlight, effective_body

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
VISIBILITY_CACHE_LIMIT = 2048
_TRIAL_STATE_FIELDS = frozenset(
    (
        "config",
        "seed",
        "width",
        "height",
        "tick",
        "rng",
        "obstacles",
        "geography",
        "animals",
        "resources",
        "events",
        "deaths",
        "total_energy_gained",
        "kills",
        "lifespans",
        "births",
        "mutations",
        "next_animal_id",
    )
)


@dataclass
class Animal:
    id: int
    species: str
    x: int
    y: int
    heading: int
    energy: float
    health: float
    age: int = 0
    last_action: str = "rest"
    last_reward: float = 0.0
    parent_id: int = -1
    generation: int = 0
    last_reproduced_age: int = -1
    kills: int = 0
    energy_gained: float = 0.0
    statistics_since_age: int = 0
    reproduction_due_age: int = -1
    mutated: bool = False
    mutation_count: int = 0
    mutation_weights: int = 0
    sex: str = "female"
    second_parent_id: int = -1
    drift_x: float = 0.0
    drift_y: float = 0.0
    mating_intent_until_tick: int = -1
    last_mating_failure: str = ""

    @property
    def position(self):
        return self.x, self.y


@dataclass
class Resource:
    x: int
    y: int
    quantity: float


@dataclass(frozen=True)
class Event:
    kind: str
    actor: int
    target: int = -1
    amount: float = 0.0
    cause: str = ""
    mate: int = -1
    position: list = None


@dataclass
class StepResult:
    observations: dict
    rewards: dict
    dead_ids: set
    events: list


class World:
    def __init__(self, config, seed):
        self.config = copy.deepcopy(config)
        if "terrain" in self.config:
            self.config["terrain"] = resolve_map(self.config["terrain"])
        config = self.config
        self.seed = seed
        self.tick = 0
        self.rng = random.Random(seed ^ 0x12345678)
        layout = random.Random(seed)
        w = config["world"]
        self.width, self.height = w["width"], w["height"]
        cells = [(x, y) for y in range(self.height) for x in range(self.width)]
        layout.shuffle(cells)
        o = config["obstacles"]
        self.obstacles = (
            set(map(tuple, o["explicit_cells"]))
            if o["generation"] == "explicit"
            else set(cells[: int(len(cells) * o["occupied_fraction"])])
        )
        self.geography = (
            Geography(config["terrain"], self.width, self.height)
            if "terrain" in config
            else None
        )
        if self.geography:
            self.obstacles |= self.geography.obstacles
        free = [
            p
            for p in cells
            if p not in self.obstacles
            and not (self.geography and self.geography.depth(p))
        ]
        r = config["resources"]
        placement = config.get("placement")
        if placement:
            from wtf.simulation.placement import resource_pools, resource_counts, zone

            pools = resource_pools(free, placement)
            resource_cells = [
                p
                for side, count in resource_counts(
                    r["patch_count"], placement["resources"]
                ).items()
                for p in layout.sample(pools[side], count)
            ]
        else:
            resource_cells = layout.sample(free, r["patch_count"])
        self.resources = {
            p: Resource(*p, r["initial_quantity_per_patch"]) for p in resource_cells
        }
        layout.shuffle(free)
        if placement:
            spawn_pools = {
                side: [
                    p for p in free if zone(p, placement["enclosure_bounds"]) == side
                ]
                for side in ("inside", "outside")
            }
        self.animals = {}
        for species_index, (species, s) in enumerate(config["species"].items()):
            # Balanced founders, shuffled independently of layout/actions/learning.
            sexes = [
                "female" if n % 2 == 0 else "male"
                for n in range(s["initial_population"])
            ]
            random.Random(config["seed"] ^ 0x534558 ^ (species_index * 7919)).shuffle(
                sexes
            )
            for sex in sexes:
                x, y = (
                    spawn_pools[placement["species"][species]] if placement else free
                ).pop()
                i = len(self.animals)
                self.animals[i] = Animal(
                    i,
                    species,
                    x,
                    y,
                    layout.randrange(4),
                    s["body"]["initial_energy"],
                    s["body"]["max_health"],
                    sex=sex,
                )
        self.next_animal_id = len(self.animals)
        self.births = defaultdict(int)
        self.mutations = defaultdict(int)
        self.events = []
        self.deaths = defaultdict(int)
        self.total_energy_gained = defaultdict(float)
        self.kills = defaultdict(int)
        self.lifespans = []
        self._tracking = (
            Tracking(config, self.width) if Tracking.enabled(config) else None
        )
        if self._tracking is not None and self._tracking.scent is not None:
            self._tracking.scent.advance(self)

    def body(self, animal):
        cycle = self.config.get("day_night")
        if not cycle or not cycle.get("enabled", False):
            return self.config["species"][animal.species]["body"]
        if getattr(self, "_body_tick", None) != self.tick:
            self._refresh_cycle_bodies(cycle)
        return self._cycle_bodies[animal.species]

    def _refresh_cycle_bodies(self, cycle):
        light = sunlight(cycle, self.tick)
        species = self.config["species"]
        self._cycle_bodies = {
            name: effective_body(spec, light) for name, spec in species.items()
        }
        self._cycle_specs = None
        self._body_tick = self.tick

    def _metabolism_species(self):
        cycle = self.config.get("day_night")
        if not cycle or not cycle.get("enabled", False):
            return self.config["species"]
        if getattr(self, "_body_tick", None) != self.tick:
            self._refresh_cycle_bodies(cycle)
        if self._cycle_specs is None:
            # Build only when flat native metabolism actually needs this view.
            self._cycle_specs = {
                name: (
                    dict(spec, body=self._cycle_bodies[name])
                    if self._cycle_bodies[name] is not spec["body"]
                    else spec
                )
                for name, spec in self.config["species"].items()
            }
        return self._cycle_specs

    def diet(self, animal):
        return self.config["species"][animal.species]["diet"]

    def front(self, animal):
        dx, dy = DIRECTIONS[animal.heading]
        return animal.x + dx, animal.y + dy

    def movement_destination(self, animal, action):
        dx, dy = DIRECTIONS[(animal.heading + WALK_OFFSETS.get(action, 0)) % 4]
        return animal.x + dx, animal.y + dy

    def blocked(self, p):
        x, y = p
        return not (0 <= x < self.width and 0 <= y < self.height) or p in self.obstacles

    def terrain_move_cost(self, animal, destination, action):
        if not self.geography:
            return 0.0
        rules = self.config["species"][animal.species].get("locomotion", {})
        gain = max(
            0.0,
            self.geography.elevation(destination)
            - self.geography.elevation(animal.position),
        )
        cost = gain * rules.get("uphill_energy_cost", 1.0)
        if action == "swim":
            depth = max(
                self.geography.depth(animal.position), self.geography.depth(destination)
            )
            cost += depth * rules.get("swim_energy_cost", 1.5)
        return cost

    def movement_block_reason(self, animal, destination, action, occupied=None):
        if self.blocked(destination):
            return "obstacle"
        if occupied is not None and destination in occupied:
            return "occupied"
        if not self.geography:
            return "not_near_water" if action == "swim" else ""
        rules = self.config["species"][animal.species].get("locomotion", {})
        source_depth = self.geography.depth(animal.position)
        target_depth = self.geography.depth(destination)
        if action in WALK_ACTIONS and (source_depth or target_depth):
            return "requires_swim"
        if action == "swim":
            depth = max(source_depth, target_depth)
            if not depth:
                return "not_near_water"
            minimum = (
                rules.get("swim_min_health", self.body(animal)["max_health"] * 0.5)
                * depth
            )
            if animal.health < minimum:
                return "health"
        gain = self.geography.elevation(destination) - self.geography.elevation(
            animal.position
        )
        if gain > rules.get("max_climb", 1.0):
            return "slope"
        cost = self.terrain_move_cost(animal, destination, action)
        if cost and animal.energy <= cost:
            return "energy"
        return ""

    def _terrain_charge(self, animal, amount, events, cause):
        if amount:
            animal.energy = max(0.0, animal.energy - amount)
            events.append(Event("energy_spent", animal.id, amount=amount, cause=cause))

    def _river_drift(self, events):
        if not self.geography:
            return
        paths = {}
        for i, a in self.animals.items():
            water = self.geography.water.get(a.position)
            if not water:
                a.drift_x = a.drift_y = 0.0
                continue
            tx, ty = a.drift_x + water[1], a.drift_y + water[2]

            def whole(v):
                return math.floor(v + 1e-12) if v >= 0 else math.ceil(v - 1e-12)

            nx, ny = whole(tx), whole(ty)
            a.drift_x = 0.0 if abs(tx - nx) < 1e-12 else tx - nx
            a.drift_y = 0.0 if abs(ty - ny) < 1e-12 else ty - ny
            n = max(abs(nx), abs(ny))
            if n:
                paths[i] = [
                    (
                        round(k * nx / n) - round((k - 1) * nx / n),
                        round(k * ny / n) - round((k - 1) * ny / n),
                    )
                    for k in range(1, n + 1)
                ]
        for substep in range(max(map(len, paths.values()), default=0)):
            occupied = {a.position for a in self.animals.values()}
            claims = defaultdict(list)
            for i, path in list(paths.items()):
                if substep >= len(path):
                    continue
                a = self.animals[i]
                if not self.geography.depth(a.position):
                    a.drift_x = a.drift_y = 0.0
                    del paths[i]
                    continue
                dx, dy = path[substep]
                destination = a.x + dx, a.y + dy
                corner = (
                    dx
                    and dy
                    and (self.blocked((a.x + dx, a.y)) or self.blocked((a.x, a.y + dy)))
                )
                if corner or self.movement_block_reason(
                    a, destination, "drift", occupied
                ):
                    events.append(Event("drift_blocked", i))
                    del paths[i]
                    continue
                claims[destination].append(i)
            for destination, ids in claims.items():
                winner = self.rng.choice(ids)
                a = self.animals[winner]
                self._terrain_charge(
                    a, self.terrain_move_cost(a, destination, "drift"), events, "uphill"
                )
                a.x, a.y = destination
                events.append(Event("drifted", winner, cause="current"))
                if not self.geography.depth(destination):
                    a.drift_x = a.drift_y = 0.0
                for i in ids:
                    if i != winner:
                        events.append(
                            Event("drift_blocked", i, cause="competing_claim")
                        )
                        paths.pop(i, None)

    def visible(self, start, end):
        # Integer Bresenham ray. The start cell does not occlude itself.
        x, y = start
        tx, ty = end
        dx, dy = abs(tx - x), -abs(ty - y)
        sx, sy = (1 if x < tx else -1), (1 if y < ty else -1)
        error = dx + dy
        while (x, y) != (tx, ty):
            e2 = 2 * error
            if e2 >= dy:
                error += dy
                x += sx
            if e2 <= dx:
                error += dx
                y += sy
            if (x, y) in self.obstacles:
                return False
        return True

    def _perception_visible(self, start, end):
        # Only used after _observation_context has reconciled obstacle changes.
        # Keep ordered endpoints: Bresenham tie behavior is not assumed symmetric.
        key = start, end
        cache = self._visibility_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        maps = getattr(self, "_native_sight", None)
        if (
            maps is not None
            and kernels.available("ray")
            and getattr(self.visible, "__func__", None) is World.visible
            and all(
                0 <= p[0] < self.width and 0 <= p[1] < self.height for p in (start, end)
            )
        ):
            result, wet = kernels.native.ray(
                *start, *end, *maps, self.width, self.height
            )
            self._water_sight_cache[key] = wet
            if len(self._water_sight_cache) > VISIBILITY_CACHE_LIMIT:
                self._water_sight_cache.popitem(last=False)
        else:
            result = self.visible(start, end)
        cache[key] = result
        if len(cache) > VISIBILITY_CACHE_LIMIT:
            cache.popitem(last=False)
        return result

    def _water_sight_cells(self, start, end):
        # Static map data only; ordered endpoints match the visibility ray.
        key = start, end
        cache = self._water_sight_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        maps = getattr(self, "_native_sight", None)
        if (
            maps is not None
            and kernels.available("ray")
            and all(
                0 <= p[0] < self.width and 0 <= p[1] < self.height for p in (start, end)
            )
        ):
            _, count = kernels.native.ray(*start, *end, *maps, self.width, self.height)
            cache[key] = count
            if len(cache) > VISIBILITY_CACHE_LIMIT:
                cache.popitem(last=False)
            return count
        x, y = start
        tx, ty = end
        dx, dy = abs(tx - x), -abs(ty - y)
        sx, sy = (1 if x < tx else -1), (1 if y < ty else -1)
        error = dx + dy
        count = 0
        while (x, y) != (tx, ty):
            e2 = 2 * error
            if e2 >= dy:
                error += dy
                x += sx
            if e2 <= dx:
                error += dx
                y += sy
            if (x, y) != (tx, ty) and self.geography.depth((x, y)):
                count += 1
        cache[key] = count
        if len(cache) > VISIBILITY_CACHE_LIMIT:
            cache.popitem(last=False)
        return count

    def _observation_context(self):
        occupied, food, prey, predators = {}, [], [], []
        species = self.config["species"]
        for i, animal in self.animals.items():
            position = animal.position
            occupied[position] = animal
            (
                prey if species[animal.species]["diet"] == "resource" else predators
            ).append((i, position))
        for position, patch in self.resources.items():
            if patch.quantity > 0.001:
                food.append(position)
        if not hasattr(self, "_perception"):
            self._perception = SpatialIndex(), SpatialIndex(), SpatialIndex()
            self._observed_obstacles = None
            self._visibility_cache = OrderedDict()
            self._water_sight_cache = OrderedDict()
            self._visibility_impl = None
        visibility_impl = getattr(self.visible, "__func__", self.visible)
        visibility_changed = visibility_impl is not self._visibility_impl
        if visibility_changed:
            self._visibility_cache.clear()
            self._visibility_impl = visibility_impl
        obstacles_changed = self._observed_obstacles != self.obstacles
        if obstacles_changed:
            self._observed_obstacles = self.obstacles.copy()
            self._visibility_cache.clear()
        if (
            obstacles_changed
            or visibility_changed
            or not hasattr(self, "_native_sight")
        ):
            self._native_sight = None
            self._native_sight_callback = World._perception_visible
            if (
                kernels.available("nearest")
                and visibility_impl is World.visible
                and all(
                    0 <= x < self.width and 0 <= y < self.height
                    for x, y in self.obstacles
                )
            ):
                blocked = bytearray(self.width * self.height)
                water = bytearray(len(blocked))
                for x, y in self.obstacles:
                    blocked[y * self.width + x] = 1
                if self.geography:
                    for x, y in self.geography.water:
                        water[y * self.width + x] = 1
                self._native_sight = bytes(blocked), bytes(water)
        for index, points in zip(
            self._perception, ([(None, p) for p in food], prey, predators)
        ):
            index.sync(
                points, self.animals.keys(), obstacles_changed or visibility_changed
            )
        if any(
            b.get("observation_version", 1) >= 3 or b["type"] == "maskable_ppo"
            for b in self.config["brains"].values()
        ):
            self._prepare_mating_context(occupied)
        else:
            self._mate_context = None
        if self._tracking is not None and self._tracking.history_enabled:
            self._visual_positions = {}
            for animal in self.animals.values():
                self._visual_positions.setdefault(animal.position, animal.id)
        return occupied, *self._perception

    def _prepare_mating_context(self, occupied=None):
        """Fresh decision-phase data; never implicitly reuse it after state edits."""
        if occupied is None:
            occupied = {a.position: a for a in self.animals.values()}
        populations, buckets, status = Counter(), defaultdict(list), {}
        for a in self.animals.values():
            populations[a.species] += 1
            buckets[a.species, a.sex, a.x // 8, a.y // 8].append(a)
            spec = self.config["species"][a.species]
            rules = spec.get("reproduction")
            if (
                not rules
                or not rules["enabled"]
                or "reproduce" not in spec["allowed_actions"]
            ):
                status[a.id] = (0.0, 0.0, 0.0)
                continue
            due = self._reproduction_due(a, rules)
            maturity = due if a.last_reproduced_age < 0 else rules["min_age_steps"]
            progress = (
                1.0
                if a.last_reproduced_age >= 0
                else min(1.0, a.age / max(1, maturity))
            )
            cooldown = (
                min(
                    1.0, max(0, due - (a.age + 1)) / max(1, due - a.last_reproduced_age)
                )
                if a.last_reproduced_age >= 0
                else 0.0
            )
            status[a.id] = (
                progress,
                cooldown,
                float(self._reproduction_eligible(a, rules)),
            )
        self._mate_context = occupied, populations, buckets, status
        return self._mate_context

    def _mating_features(self, a, include_intent=False):
        occupied, populations, buckets, status = self._mate_context
        spec = self.config["species"][a.species]
        rules = spec.get("reproduction", {})
        body = self.body(a)
        radius = body["vision_radius"]
        bonus = (
            body.get("water_vision_bonus", 0)
            if self.geography and self.geography.water
            else 0
        )
        limit = radius + bonus
        scale = max(1, limit)
        opposite = "male" if a.sex == "female" else "female"
        best, pair, consenting_pair = None, False, False
        for by in range((a.y - limit) // 8, (a.y + limit) // 8 + 1):
            for bx in range((a.x - limit) // 8, (a.x + limit) // 8 + 1):
                for mate in buckets.get((a.species, opposite, bx, by), ()):
                    dx, dy = mate.x - a.x, mate.y - a.y
                    squared = dx * dx + dy * dy
                    if squared > limit * limit:
                        continue
                    allowed = (
                        radius
                        + min(bonus, self._water_sight_cells(a.position, mate.position))
                        if bonus
                        else radius
                    )
                    if squared > allowed * allowed or not self._perception_visible(
                        a.position, mate.position
                    ):
                        continue
                    key = squared, mate.id
                    if best is None or key < best[0]:
                        best = key, mate
                    if (
                        (not pair or (include_intent and not consenting_pair))
                        and status[a.id][2]
                        and status[mate.id][2]
                        and populations[a.species] < rules["max_population"]
                        and squared <= rules.get("mate_radius", 2) ** 2
                        and self._mating_land_route(
                            a, mate, rules.get("mate_radius", 2)
                        )
                        and self._has_birth_space(a, mate, occupied)
                    ):
                        pair = True
                        consenting_pair = (
                            consenting_pair or self._intent_fraction(mate) > 0
                        )
        features = [1.0 if a.sex == "male" else -1.0, *status[a.id]]
        if best is None:
            features += [0.0] * 5
        else:
            mate = best[1]
            features += [
                (mate.x - a.x) / scale,
                (mate.y - a.y) / scale,
                math.sqrt(best[0][0]) / scale,
                1.0,
                status[mate.id][2],
            ]
        features += [float(pair)]
        if include_intent:
            features += [
                self._intent_fraction(a),
                self._intent_fraction(best[1]) if best is not None else 0.0,
                float(consenting_pair),
            ]
        return features

    def _intent_fraction(self, animal):
        steps = (
            self.config["species"][animal.species]
            .get("reproduction", {})
            .get("intent_steps", 1)
        )
        return min(1.0, max(0, animal.mating_intent_until_tick - self.tick + 1) / steps)

    def observe(self, a, context=None):
        body = self.body(a)
        radius = body["vision_radius"]
        template = self.config["brains"][self.config["species"][a.species]["brain"]]
        top_k = template.get("observation_top_k", 1)
        occupied, food, prey, predators = (
            self._observation_context() if context is None else context
        )
        features = [
            a.energy / body["max_energy"],
            a.health / body["max_health"],
            *DIRECTIONS[a.heading],
        ]
        for offset in range(4):
            dx, dy = DIRECTIONS[(a.heading + offset) % 4]
            p = a.x + dx, a.y + dy
            features.append(
                float(bool(self.movement_block_reason(a, p, "move_forward", occupied)))
                if self.geography
                else float(self.blocked(p) or p in occupied)
            )
        bonus = (
            body.get("water_vision_bonus", 0)
            if self.geography and self.geography.water
            else 0
        )
        if bonus:

            def in_range(start, end, squared):
                if squared <= radius * radius:
                    return True
                allowed = radius + min(bonus, self._water_sight_cells(start, end))
                return squared <= allowed * allowed

            in_range._native_sight_range = (radius, bonus)
            for index in (food, prey, predators):
                features += index.nearest_many(
                    a,
                    radius + bonus,
                    self._perception_visible,
                    top_k,
                    in_range,
                    (radius, bonus),
                )
        else:
            for index in (food, prey, predators):
                features += index.nearest_many(
                    a, radius, self._perception_visible, top_k
                )
        version = template.get("observation_version", 1)
        if version >= 2:
            features += (
                self.geography.features(a, DIRECTIONS) if self.geography else [0.0] * 12
            )
        if version >= 3:
            features += self._mating_features(a, include_intent=version >= 4)
        history_steps = template.get("movement_history_steps", 0)
        if history_steps:
            targets = []
            for category in (0, 1):
                offset = 8 + 4 * top_k * (category + 1)
                for slot in range(top_k):
                    dx, dy, _, present = features[
                        offset + 4 * slot : offset + 4 * slot + 4
                    ]
                    if present:
                        x, y = round(a.x + dx * (radius + bonus)), round(
                            a.y + dy * (radius + bonus)
                        )
                        targets.append((category, self._visual_positions[x, y], x, y))
            features += self._tracking.visual.observe(
                a, self.tick, radius + bonus, top_k, history_steps, targets
            )
        if "scent_observation" in template:
            features += self._tracking.scent.sample(self, a, template)
        return tuple(features)

    def observe_all(self):
        # Reconcile authoritative state once per phase. Dynamic bucket contents
        # update cached windows without rebuilding their overlapping interior.
        context = self._observation_context()
        if self._tracking is not None:
            for animal_id in self._tracking.visual.frames.keys() - self.animals.keys():
                del self._tracking.visual.frames[animal_id]
        return {i: self.observe(a, context) for i, a in self.animals.items()}

    def step(self, actions):
        if set(actions) != set(self.animals):
            raise ValueError("Exactly one action required for every living animal")
        for i, action in actions.items():
            if (
                action
                not in self.config["species"][self.animals[i].species][
                    "allowed_actions"
                ]
            ):
                raise ValueError(f"Animal {i}: unsupported action {action}")
        events = []
        # Destinations occupied at the beginning of this phase remain blocked.
        # Swaps and following into a just-vacated cell are deliberately disallowed.
        occupied = {a.position for a in self.animals.values()}
        claims = defaultdict(list)
        for i, a in self.animals.items():
            action = actions[i]
            a.last_action = action
            if action == "turn_left":
                a.heading = (a.heading - 1) % 4
            elif action == "turn_right":
                a.heading = (a.heading + 1) % 4
            elif action in MOVE_ACTIONS:
                destination = self.movement_destination(a, action)
                reason = self.movement_block_reason(a, destination, action, occupied)
                if reason:
                    cause = (
                        ("swim:" + reason)
                        if action == "swim"
                        else reason if self.geography else ""
                    )
                    if action in WALK_ACTIONS and action != "move_forward":
                        cause = action + ":" + reason
                    events.append(Event("movement_blocked", i, cause=cause))
                else:
                    claims[destination].append(i)
        for destination, ids in claims.items():
            winner = self.rng.choice(ids)
            a = self.animals[winner]
            if self.geography:
                self._terrain_charge(
                    a,
                    self.terrain_move_cost(a, destination, actions[winner]),
                    events,
                    "swim" if actions[winner] == "swim" else "uphill",
                )
            a.x, a.y = destination
            events.append(
                Event(
                    "moved",
                    winner,
                    cause=actions[winner] if actions[winner] != "move_forward" else "",
                )
            )
            events.extend(
                Event(
                    "movement_blocked",
                    i,
                    cause=(
                        "swim:competing_claim"
                        if actions[i] == "swim"
                        else (
                            actions[i] + ":competing_claim"
                            if actions[i] != "move_forward"
                            else "competing_claim"
                        )
                    ),
                )
                for i in ids
                if i != winner
            )
        self._river_drift(events)
        priority, competition_deaths, challenge_successes = [], set(), set()
        has_challenges = "challenge" in actions.values()
        if has_challenges:
            from wtf.simulation.competition import resolve

            priority, competition_deaths, challenge_successes = resolve(
                self, actions, events
            )
        # Feeding uses seeded contention order; resources may share animal cells.
        eaters = [i for i in self.animals if actions[i] == "eat"]
        self.rng.shuffle(eaters)
        if priority:
            # A winning challenger gets one normal bite, not an extra feeding turn.
            eaters = priority + [i for i in eaters if i not in priority]
        for i in eaters:
            if i in competition_deaths:
                continue
            a = self.animals[i]
            b = self.body(a)
            candidates = [self.resources.get(p) for p in (a.position, self.front(a))]
            patch = next(
                (r for r in candidates if r is not None and r.quantity > 0), None
            )
            conversion = b["energy_per_food_unit"]
            if patch is not None and conversion > 0:
                amount = min(
                    patch.quantity,
                    b["bite_size"],
                    (b["max_energy"] - a.energy) / conversion,
                )
                patch.quantity -= amount
                gained = amount * conversion
                a.energy += gained
                if gained > 0:
                    events.append(
                        Event(
                            "energy_gained",
                            i,
                            amount=gained,
                            cause=(
                                "resource"
                                if actions[i] == "eat"
                                else "competition_food"
                            ),
                        )
                    )
        # Attacks are simultaneous. Multiple attackers share actual damage;
        # one seeded attacker receives energy for a kill, never multiple rewards.
        positions = {a.position: a for a in self.animals.values() if a.health > 0}
        attacks = defaultdict(list)
        for i, a in self.animals.items():
            if actions[i] == "attack" and a.health > 0:
                target = positions.get(self.front(a))
                if (
                    target is not None
                    and self.diet(target) == "resource"
                    and self.body(a)["attack_damage"] > 0
                ):
                    attacks[target.id].append(i)
        for target_id, ids in attacks.items():
            target = self.animals[target_id]
            attempted_damage = sum(
                self.body(self.animals[i])["attack_damage"] for i in ids
            )
            damage = min(target.health, attempted_damage)
            # Attribute only actual damage, proportional to each attacker's
            # contribution. Overkill cannot multiply the hit reward.
            for i in ids:
                contribution = damage * (
                    self.body(self.animals[i])["attack_damage"] / attempted_damage
                )
                events.append(
                    Event("damage_dealt", i, target_id, contribution, "attack")
                )
            target.health -= damage
            events.append(
                Event("damage_taken", target_id, amount=damage, cause="attack")
            )
            if target.health <= 0:
                killer = self.animals[self.rng.choice(ids)]
                b = self.body(killer)
                gained = min(b["energy_per_kill"], b["max_energy"] - killer.energy)
                killer.energy += gained
                events.append(
                    Event(
                        "kill",
                        killer.id,
                        target_id,
                        position=(
                            list(target.position)
                            if self.geography or has_challenges
                            else None
                        ),
                    )
                )
                events.append(
                    Event("energy_gained", killer.id, target_id, gained, "prey")
                )
        attack_deaths = {i for i in attacks if self.animals[i].health <= 0}
        drowned = set()
        rewards = {i: 0.0 for i in self.animals}
        native_metabolism = (
            not competition_deaths
            and not self.geography
            and kernels.available("flat_metabolism")
            and (
                getattr(kernels.native, "escape_movement", False)
                or not any(action in WALK_ACTIONS[1:] for action in actions.values())
            )
            and kernels.native.flat_metabolism(
                self.animals.values(),
                actions,
                self._metabolism_species(),
                rewards,
                events,
                Event,
                Animal,
            )
        )
        if not native_metabolism:
            for i, a in self.animals.items():
                b = self.body(a)
                cost = b["living_cost"]
                if actions[i] in MOVE_ACTIONS:
                    cost += b["movement_cost"]
                if actions[i] == "attack":
                    cost += b["attack_cost"]
                a.energy = max(0.0, a.energy - cost)
                rewards[i] += cost * self.config["species"][a.species]["rewards"].get(
                    "per_energy_cost", 0
                )
                a.age += 1
                if self.geography and a.health > 0:
                    depth = self.geography.depth(a.position)
                    if depth:
                        spec = self.config["species"][a.species]
                        rules = spec.get("locomotion", {})
                        self._terrain_charge(
                            a,
                            depth * rules.get("water_energy_cost", 0.25),
                            events,
                            "water",
                        )
                        minimum = (
                            rules.get("swim_min_health", b["max_health"] * 0.5) * depth
                        )
                        if (
                            "swim" not in spec["allowed_actions"]
                            or a.health < minimum
                            or a.energy <= 0
                        ):
                            damage = min(
                                a.health, depth * rules.get("drowning_damage", 2.0)
                            )
                            a.health -= damage
                            if damage:
                                events.append(
                                    Event(
                                        "damage_taken",
                                        i,
                                        amount=damage,
                                        cause="drowning",
                                    )
                                )
                            if a.health <= 0:
                                drowned.add(i)
                if a.energy <= 0 and a.health > 0:
                    damage = min(a.health, b["starvation_damage"])
                    a.health -= damage
                    events.append(
                        Event("damage_taken", i, amount=damage, cause="starvation")
                    )
        dead = {i for i, a in self.animals.items() if a.health <= 0}
        for i in sorted(dead):
            a = self.animals[i]
            cause = (
                "predation"
                if i in attack_deaths
                else (
                    "competition"
                    if i in competition_deaths
                    else "drowning" if i in drowned else "starvation"
                )
            )
            events.append(
                Event(
                    "died",
                    i,
                    cause=cause,
                    position=(
                        list(a.position) if self.geography or has_challenges else None
                    ),
                )
            )
            self.deaths[cause] += 1
            self.lifespans.append(
                {"id": i, "species": a.species, "age": a.age, "cause": cause}
            )
        successful = set(challenge_successes)
        for e in events:
            a = self.animals[e.actor]
            r = self.config["species"][a.species]["rewards"]
            if (
                e.kind == "moved"
                or e.kind == "damage_dealt"
                and e.cause != "competition"
                and e.amount > 0
                or e.kind == "energy_gained"
                and e.cause == "resource"
                and e.amount > 0
            ):
                successful.add(e.actor)
            if e.kind == "energy_spent":
                rewards[e.actor] += e.amount * r.get("per_energy_cost", 0)
            elif e.kind == "energy_gained":
                rewards[e.actor] += e.amount * r["per_energy_gained"]
                self.total_energy_gained[a.species] += e.amount
                a.energy_gained += e.amount
            elif e.kind == "damage_dealt":
                rewards[e.actor] += e.amount * r.get("per_damage_dealt", 0.0)
            elif e.kind == "damage_taken":
                rewards[e.actor] += e.amount * r["per_damage_taken"]
            elif e.kind == "died":
                rewards[e.actor] += r["death"]
            elif e.kind == "kill":
                self.kills[a.species] += 1
                a.kills += 1
        dead_animals = {i: self.animals.pop(i) for i in dead}
        before_birth = len(events)
        self._reproduce(events)
        for birth in events[before_birth:]:
            if birth.kind != "born":
                continue
            for parent_id in (birth.actor, birth.mate):
                parent = self.animals[parent_id]
                spec = self.config["species"][parent.species]
                successful.add(parent_id)
                rewards[parent_id] += spec["rewards"].get("per_offspring", 0)
                rewards[parent_id] += spec["reproduction"]["energy_cost"] * spec[
                    "rewards"
                ].get("per_energy_cost", 0)
        for i, action in actions.items():
            a = self.animals.get(i) or dead_animals[i]
            if action == "reproduce" and i in dead:
                a.last_mating_failure = "parent_died"
                events.append(Event("mating_failed", i, cause="parent_died"))
            rules = self.config["species"][a.species]["rewards"]
            if action in ATTEMPT_ACTIONS and i not in successful:
                penalty = rules.get("failed_action", 0)
                rewards[i] += penalty
                # Living bodies already record their action/reward. Only terminal
                # failed choices need an extra event for exact offline feedback.
                if i in dead and penalty:
                    events.append(
                        Event("action_failed", i, amount=penalty, cause=action)
                    )
            a.last_reward = rewards[i]
        r = self.config["resources"]
        native_growth = kernels.available(
            "resource_growth"
        ) and kernels.native.resource_growth(
            self.resources.values(),
            Resource,
            r["capacity_per_patch"],
            r["growth_per_step"],
        )
        if not native_growth:
            for patch in self.resources.values():
                patch.quantity = min(
                    r["capacity_per_patch"], patch.quantity + r["growth_per_step"]
                )
        self.tick += 1
        if self._tracking is not None and self._tracking.scent is not None:
            self._tracking.scent.advance(self, dead_animals.values())
        self.events = events
        return StepResult(self.observe_all(), rewards, dead, events)

    def _reproduction_interval(self, rules, first=False):
        center = rules["min_age_steps" if first else "cooldown_steps"]
        prefix = "age" if first else "cooldown"
        pct = rules.get(prefix + "_deviation_pct")
        # Half steps round up; preserve old saved configurations expressed in steps.
        deviation = (
            int(center * pct / 100 + 0.5)
            if pct is not None
            else rules.get(prefix + "_deviation_steps", 0)
        )
        # A zero-width interval must not consume RNG: legacy trajectories stay exact.
        return (
            center
            if not deviation
            else self.rng.randint(max(1, center - deviation), center + deviation)
        )

    @staticmethod
    def _has_reproduction_window(rules):
        return any(
            rules.get(
                prefix + "_deviation_pct", rules.get(prefix + "_deviation_steps", 0)
            )
            for prefix in ("age", "cooldown")
        )

    def _reproduction_ready(self, animal, rules, has_window=None):
        if has_window is None:
            has_window = self._has_reproduction_window(rules)
        if not has_window:
            # Preserve the original gates and RNG consumption for old environments.
            return animal.age >= rules["min_age_steps"] and (
                animal.last_reproduced_age < 0
                or animal.age - animal.last_reproduced_age >= rules["cooldown_steps"]
            )
        if animal.reproduction_due_age < 0:
            first = animal.last_reproduced_age < 0
            origin = 0 if first else animal.last_reproduced_age
            animal.reproduction_due_age = origin + self._reproduction_interval(
                rules, first
            )
        return animal.age >= animal.reproduction_due_age

    def _mating_land_route(self, first, second, radius):
        if not self.geography:
            return True
        if self.geography.depth(first.position) or self.geography.depth(
            second.position
        ):
            return False
        limit = min(
            self.config["species"][a.species]
            .get("locomotion", {})
            .get("max_climb", 1.0)
            for a in (first, second)
        )
        queue = deque([(first.position, 0)])
        seen = {first.position}
        while queue:
            p, hops = queue.popleft()
            if p == second.position:
                return True
            if hops >= 2 * radius:
                continue
            for dx, dy in DIRECTIONS:
                q = p[0] + dx, p[1] + dy
                if q in seen or self.blocked(q) or self.geography.depth(q):
                    continue
                if (
                    abs(self.geography.elevation(q) - self.geography.elevation(p))
                    > limit
                ):
                    continue
                seen.add(q)
                queue.append((q, hops + 1))
        return False

    def _birth_cell_candidates(self, mother, father, occupied):
        """Yield placeable cells in the established parent/direction order."""
        spec = self.config["species"][mother.species]
        for parent in (mother, father):
            for dx, dy in DIRECTIONS:
                position = parent.x + dx, parent.y + dy
                if self.blocked(position):
                    continue
                if self.geography and (
                    self.geography.depth(position)
                    or abs(
                        self.geography.elevation(position)
                        - self.geography.elevation(parent.position)
                    )
                    > spec.get("locomotion", {}).get("max_climb", 1.0)
                ):
                    continue
                if position not in occupied:
                    yield position

    def _birth_cells(self, mother, father, occupied):
        # Actual births require the complete ordered, unique choice sequence.
        return list(
            dict.fromkeys(self._birth_cell_candidates(mother, father, occupied))
        )

    def _has_birth_space(self, mother, father, occupied):
        # Availability needs neither deduplication nor a materialized list.
        return (
            next(self._birth_cell_candidates(mother, father, occupied), None)
            is not None
        )

    def _reproduction_due(self, a, rules):
        if a.reproduction_due_age >= 0:
            due = a.reproduction_due_age
        else:
            first = a.last_reproduced_age < 0
            center = rules["min_age_steps" if first else "cooldown_steps"]
            prefix = "age" if first else "cooldown"
            pct = rules.get(prefix + "_deviation_pct")
            deviation = (
                int(center * pct / 100 + 0.5)
                if pct is not None
                else rules.get(prefix + "_deviation_steps", 0)
            )
            due = (0 if first else a.last_reproduced_age) + max(1, center - deviation)
        return due

    def _reproduction_eligible(self, a, rules):
        return not self._personal_mating_failure(a, rules, a.age + 1)

    def _personal_mating_failure(self, animal, rules, age):
        if not rules or not rules["enabled"]:
            return "disabled"
        if age < self._reproduction_due(animal, rules):
            return "immature" if animal.last_reproduced_age < 0 else "cooldown"
        if animal.health <= rules["health_threshold"]:
            return "low_health"
        if animal.energy < rules["min_energy"] or animal.energy <= rules["energy_cost"]:
            return "low_energy"
        if self.geography and self.geography.depth(animal.position):
            return "in_water"
        return ""

    def _mating_failure_reason(self, animal, occupied, populations, paired):
        rules = self.config["species"][animal.species].get("reproduction")
        reason = self._personal_mating_failure(animal, rules, animal.age)
        if reason:
            return reason
        if populations[animal.species] >= rules["max_population"]:
            return "population_cap"
        radius = rules.get("mate_radius", 2)
        nearby = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                mate = occupied.get((animal.x + dx, animal.y + dy))
                if (
                    dx * dx + dy * dy <= radius * radius
                    and mate is not None
                    and mate.species == animal.species
                    and mate.sex != animal.sex
                ):
                    nearby.append(mate)
        if not nearby:
            return "no_nearby_mate"
        reachable = [
            mate
            for mate in nearby
            if self.visible(animal.position, mate.position)
            and self._mating_land_route(animal, mate, radius)
        ]
        if not reachable:
            return "blocked_route"
        if any(mate.id in paired for mate in reachable):
            return "mate_unavailable"
        ready = [
            mate
            for mate in reachable
            if not self._personal_mating_failure(mate, rules, mate.age)
        ]
        if not ready:
            return "mate_not_ready"
        willing = [mate for mate in ready if mate.mating_intent_until_tick >= self.tick]
        if not willing:
            return "awaiting_consent"
        return "no_birth_space"

    def reproduction_available(
        self, animal, occupied=None, populations=None, mating_context=None
    ):
        """Decision-time mask, without RNG/clock mutation or assuming consent.

        Current masks cannot predict combat or the other animal's next choice;
        final physical gates and mutual choices are checked again after actions.
        An explicit context is valid only in its unchanged decision phase; calls
        without one always read current state, including same-tick edits.
        """
        spec = self.config["species"][animal.species]
        rules = spec.get("reproduction")
        if (
            not rules
            or not rules["enabled"]
            or "reproduce" not in spec["allowed_actions"]
        ):
            return False

        if mating_context is None:
            eligible = lambda a: self._reproduction_eligible(a, rules)
        else:
            occupied, populations, _, status = mating_context
            eligible = lambda a: status[a.id][2]

        if not eligible(animal):
            return False
        if populations is None:
            populations = Counter(a.species for a in self.animals.values())
        if populations.get(animal.species, 0) >= rules["max_population"]:
            return False
        if occupied is None:
            occupied = {a.position: a for a in self.animals.values()}
        radius = rules.get("mate_radius", 2)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                mate = occupied.get((animal.x + dx, animal.y + dy))
                if (
                    mate is None
                    or mate.id == animal.id
                    or mate.species != animal.species
                    or mate.sex == animal.sex
                    or not eligible(mate)
                ):
                    continue
                if (
                    self.visible(animal.position, mate.position)
                    and self._mating_land_route(animal, mate, radius)
                    and self._has_birth_space(animal, mate, occupied)
                ):
                    return True
        return False

    def _reproduce(self, events):
        # Resolve after combat/starvation. Only animals present before this
        # phase can parent; newborns first act on the next simulation step.
        parents = list(self.animals.values())
        rules = {
            name: spec.get("reproduction")
            for name, spec in self.config["species"].items()
        }
        windowed = {
            name: self._has_reproduction_window(r) if r and r["enabled"] else False
            for name, r in rules.items()
        }
        eligible = []
        counts = defaultdict(int)
        for a in parents:
            counts[a.species] += 1
            r = rules[a.species]
            ready = (
                self._reproduction_ready(a, r, windowed[a.species])
                if r and r["enabled"]
                else False
            )
            if a.last_action == "reproduce":
                a.last_mating_failure = ""
                a.mating_intent_until_tick = (
                    self.tick + r.get("intent_steps", 1) - 1
                    if ready and not self._personal_mating_failure(a, r, a.age)
                    else -1
                )
            elif a.last_action != "rest":
                a.mating_intent_until_tick = -1
            if not ready or (r and self._personal_mating_failure(a, r, a.age)):
                a.mating_intent_until_tick = -1
            if (
                ready
                and a.mating_intent_until_tick >= self.tick
                and not (self.geography and self.geography.depth(a.position))
                and a.health > r["health_threshold"]
                and a.energy >= r["min_energy"]
                and a.energy > r["energy_cost"]
            ):
                eligible.append(a)
        self.rng.shuffle(eligible)
        occupied = {a.position for a in parents}
        males = {(a.species, a.x, a.y): a for a in eligible if a.sex == "male"}
        used = set()
        for mother in eligible:
            if mother.sex != "female":
                continue
            spec = self.config["species"][mother.species]
            r = spec["reproduction"]
            if counts[mother.species] >= r["max_population"]:
                continue
            radius = r.get("mate_radius", 2)
            candidates = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    mate = males.get((mother.species, mother.x + dx, mother.y + dy))
                    if (
                        mate is not None
                        and mate.id not in used
                        and self.visible(mother.position, mate.position)
                        and self._mating_land_route(mother, mate, radius)
                    ):
                        candidates.append(mate)
            if not candidates:
                continue
            # Find placeable pairs before committing either parent's resources.
            self.rng.shuffle(candidates)
            for father in candidates:
                free = self._birth_cells(mother, father, occupied)
                if free:
                    break
            else:
                continue
            x, y = self.rng.choice(free)
            child_id = self.next_animal_id
            self.next_animal_id += 1
            child = Animal(
                child_id,
                mother.species,
                x,
                y,
                self.rng.randrange(4),
                spec["body"]["initial_energy"],
                spec["body"]["max_health"],
                parent_id=mother.id,
                second_parent_id=father.id,
                sex=self.rng.choice(("female", "male")),
                generation=max(mother.generation, father.generation) + 1,
                mutation_count=max(mother.mutation_count, father.mutation_count),
            )
            self.animals[child_id] = child
            occupied.add(child.position)
            counts[mother.species] += 1
            for parent in (mother, father):
                used.add(parent.id)
                parent.mating_intent_until_tick = -1
                parent.last_mating_failure = ""
                parent.energy -= r["energy_cost"]
                parent.last_reproduced_age = parent.age
                if windowed[parent.species]:
                    parent.reproduction_due_age = (
                        parent.age + self._reproduction_interval(r)
                    )
            self.births[mother.species] += 1
            events.append(
                Event(
                    "born",
                    mother.id,
                    child_id,
                    cause="paired_reproduction",
                    mate=father.id,
                )
            )
        attempts = [
            a for a in parents if a.last_action == "reproduce" and a.id not in used
        ]
        if attempts:
            final_occupied = {a.position: a for a in self.animals.values()}
            for animal in attempts:
                reason = self._mating_failure_reason(
                    animal, final_occupied, counts, used
                )
                animal.last_mating_failure = reason
                events.append(Event("mating_failed", animal.id, cause=reason))

    def metrics(self):
        populations = Counter(a.species for a in self.animals.values())
        return dict(
            tick=self.tick,
            populations={s: populations[s] for s in self.config["species"]},
            food=sum(r.quantity for r in self.resources.values()),
            deaths=dict(self.deaths),
            energy_gained=dict(self.total_energy_gained),
            kills=dict(self.kills),
            births=dict(self.births),
            mutations=dict(self.mutations),
        )

    def snapshot(self, include_static=True, metrics=None):
        # Explicit report reuse is valid only within an unchanged-state phase.
        # Standalone snapshots always read current state; outputs own their maps.
        data = dict(
            tick=self.tick,
            resources=[vars(r).copy() for r in self.resources.values()],
            animals=[vars(a).copy() for a in self.animals.values()],
            events=[vars(e).copy() for e in self.events],
            metrics=self.metrics() if metrics is None else copy.deepcopy(metrics),
        )
        if include_static:
            return dict(
                width=self.width,
                height=self.height,
                resource_capacity=self.config["resources"]["capacity_per_patch"],
                obstacles=sorted(self.obstacles),
                **data,
            )
        return data

    def checkpoint(self):
        result = dict(
            config=self.config,
            seed=self.seed,
            snapshot=self.snapshot(),
            rng_state=self.rng.getstate(),
            lifespans=self.lifespans,
            next_animal_id=self.next_animal_id,
        )
        if self._tracking is not None:
            result["sensory_state"] = self._tracking.checkpoint()
        return result

    @classmethod
    def restore(cls, state, static_geography=None):
        # Construct directly: restoration must not depend on a new layout draw.
        w = cls.__new__(cls)
        w.config = copy.deepcopy(state["config"])
        return w._restore_state(state, static_geography)

    def _reset_trial(self, state):
        """Reset a sequential GRPO trial; retain only its private config/RNG."""
        geography = self.geography
        # Preserve authoritative attribute slots/hints; dict.clear() loses the
        # compact key-sharing layout and slows later world attribute access.
        for name in tuple(vars(self)):
            if name not in _TRIAL_STATE_FIELDS:
                delattr(self, name)
        # Release the large old collections before constructing their replacements.
        self.animals = self.resources = self.events = self.lifespans = (
            self.obstacles
        ) = None
        return self._restore_state(state, geography, _trial=True)

    def _restore_state(self, state, static_geography=None, _trial=False):
        w = self
        w.seed = state["seed"]
        s = state["snapshot"]
        w.width, w.height, w.tick = s["width"], s["height"], s["tick"]
        w.obstacles = set(map(tuple, s["obstacles"]))
        if static_geography is not None:
            if static_geography.definition != w.config.get("terrain"):
                raise ValueError("Shared geography does not match restored map")
            w.geography = static_geography
        else:
            w.geography = (
                Geography(w.config["terrain"], w.width, w.height)
                if "terrain" in w.config
                else None
            )
        w.animals = {}
        for original in s["animals"]:
            if _trial:
                # Current trial records need no legacy/default-field adaptation.
                # Preserve normal instance layout for hot attribute reads.
                animal = Animal(**original)
                w.animals[animal.id] = animal
                continue
            animal = dict(original)
            if "kills" not in animal:
                animal["statistics_since_age"] = animal["age"]
            animal.setdefault("sex", "female" if animal["id"] % 2 == 0 else "male")
            w.animals[animal["id"]] = Animal(**animal)
        w.resources = {(r["x"], r["y"]): Resource(**r) for r in s["resources"]}
        w.events = [Event(**e) for e in s["events"]]
        w.deaths = defaultdict(int, s["metrics"]["deaths"])
        w.total_energy_gained = defaultdict(float, s["metrics"]["energy_gained"])
        w.kills = defaultdict(int, s["metrics"]["kills"])
        w.lifespans = copy.deepcopy(state["lifespans"])
        w.births = defaultdict(int, s["metrics"].get("births", {}))
        w.mutations = defaultdict(int, s["metrics"].get("mutations", {}))
        if "next_animal_id" in state:
            w.next_animal_id = state["next_animal_id"]
        else:
            # Only legacy checkpoints need to reconstruct the identity bound.
            founding_count = sum(
                spec["initial_population"] for spec in w.config["species"].values()
            )
            highest_id = (
                max(list(w.animals) + [a["id"] for a in w.lifespans], default=-1) + 1
            )
            w.next_animal_id = max(founding_count, highest_id)

        def tuples(v):
            return tuple(tuples(x) for x in v) if isinstance(v, (list, tuple)) else v

        if not hasattr(w, "rng"):
            w.rng = random.Random()
        w.rng.setstate(state["rng_state"] if _trial else tuples(state["rng_state"]))
        w._tracking = (
            Tracking.restore(w.config, w.width, state.get("sensory_state"))
            if Tracking.enabled(w.config)
            else None
        )
        return w
