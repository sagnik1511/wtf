"""Opt-in collective sightings and bounded, coordinate-free consequence learning.

Policies never share parameters, rewards, optimizer state or PPO trajectories.
The shared predictor estimates observed outcomes, not action values or rewards.
"""

from collections import OrderedDict, defaultdict, deque
import heapq
import math

from wtf.simulation.actions import ACTIONS, MOVE_ACTIONS
from wtf.learning.consequences import Consequences

GROUPS = ("walk", "swim", "feed", "combat", "reproduce", "idle")
PARTS = ("confidence", "harm", "feeding", "birth")
GROUP = {
    **dict.fromkeys(ACTIONS[:4], 0),
    "swim": 1,
    "eat": 2,
    "attack": 3,
    "challenge": 3,
    "reproduce": 4,
    "turn_left": 5,
    "turn_right": 5,
    "rest": 5,
}


def social_contract(template):
    """Resolve the optional local social-observation contract."""

    # Exact types intentionally reject bool/subclass substitutions in persisted
    # numeric contracts.
    # pylint: disable=unidiomatic-typecheck

    if "social_observation" not in template:
        return None
    spec = template["social_observation"]
    if not isinstance(spec, dict) or set(spec) - {"history_steps", "max_neighbors"}:
        raise ValueError("Invalid collective social observation layout")
    result = {
        "history_steps": spec.get("history_steps", 4),
        "max_neighbors": spec.get("max_neighbors", 8),
    }
    if any(type(value) is not int or not 1 <= value <= 16 for value in result.values()):
        raise ValueError("Social history_steps/max_neighbors must be integers in 1..16")
    return result


def social_width(template):
    """Compute the optional social and consequence input width."""
    return (24 if social_contract(template) is not None else 0) + (
        48 if template.get("consequence_observation", False) else 0
    )


def social_names(template):
    """Name the optional social and consequence inputs in vector order."""
    names = []
    if social_contract(template) is not None:
        names += [f"Collective {group} {part}" for group in GROUPS for part in PARTS]
    if template.get("consequence_observation", False):
        names += [
            f"Consequence {action} {part}" for action in ACTIONS for part in PARTS
        ]
    return names


def enabled(config):
    """Return whether the optional feature is configured for this context."""
    return any(
        social_width(config["brains"][spec["brain"]])
        for spec in config["species"].values()
    )


def _template(world, animal):
    return world.config["brains"][world.config["species"][animal.species]["brain"]]


def _can_see(world, observer, target):
    # A cached immutable start/end sight description; no lookups of hidden bodies.

    # This internal collaborator intentionally inspects the owning component's private
    # state.
    # pylint: disable=protected-access

    x, y, radius, bonus = observer
    dx, dy = target[0] - x, target[1] - y
    if dx * dx + dy * dy > (radius + bonus) ** 2:
        return False
    allowed = radius + (
        min(bonus, world._water_sight_cells((x, y), target)) if bonus else 0
    )
    return dx * dx + dy * dy <= allowed * allowed and world.visible((x, y), target)


def _sight(world, animal):
    body = world.body(animal)
    bonus = (
        body.get("water_vision_bonus", 0)
        if world.geography and world.geography.water
        else 0
    )
    return animal.x, animal.y, body["vision_radius"], bonus


def _depth_bin(depth):
    return 0 if depth <= 0 else 1 if depth <= 0.5 else 2 if depth <= 1 else 3


def _local_context(world, animal, buckets):
    body = world.body(animal)
    sight = _sight(world, animal)
    diet = world.diet(animal)
    danger = any(
        other.id != animal.id
        and world.diet(other) != diet
        and abs(other.x - animal.x) <= 2
        and abs(other.y - animal.y) <= 2
        and _can_see(world, sight, other.position)
        for by in range((animal.y - 2) // 8, (animal.y + 2) // 8 + 1)
        for bx in range((animal.x - 2) // 8, (animal.x + 2) // 8 + 1)
        for other in buckets.get((bx, by), ())
    )
    return body, danger


def _context(world, animal, action, local):
    """Local physical context only; never absolute position or a hidden target state."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals

    body, danger = local
    rules = world.config["species"][animal.species].get("locomotion", {})
    source = animal.position
    target = (
        world.movement_destination(animal, action) if action in MOVE_ACTIONS else source
    )
    geo = world.geography
    sd, td = (geo.depth(source), geo.depth(target)) if geo else (0, 0)
    climb = geo.elevation(target) - geo.elevation(source) if geo else 0
    flow = geo.water.get(source, (0, 0, 0))[1:] if geo else (0, 0)
    blocked = world.blocked(target)
    # Adjacent terrain is already part of the ordinary terrain observation.
    food = world.resources.get(target) if not blocked else None
    spec = world.config["species"][animal.species]
    minimum = rules.get("swim_min_health", body["max_health"] * 0.5) * max(sd, td)
    return (
        ACTIONS.index(action),
        min(3, max(0, int(4 * animal.energy / body["max_energy"]))),
        min(3, max(0, int(4 * animal.health / body["max_health"]))),
        _depth_bin(sd),
        _depth_bin(td),
        max(-2, min(2, int(climb))),
        min(3, int(math.hypot(*flow) * 2)),
        int(animal.health >= minimum),
        int(bool(food and food.quantity > 0.1)),
        int(blocked),
        int(animal.age >= spec.get("reproduction", {}).get("min_age_steps", 0)),
        int(danger),
    )


class SocialState:
    """Track bounded visible outcomes and species consequence estimates."""

    # Keep authoritative sensory state and its bounded ephemeral caches with one owner.
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config):
        self.config = config
        self.training = True
        self.knowledge = (
            Consequences(config)
            if config.get("consequence_learning", {}).get("enabled", False)
            and any(
                config["brains"][spec["brain"]].get("consequence_observation", False)
                for spec in config["species"].values()
            )
            else None
        )
        self.sightings = {}  # observer -> deque[(tick, 6 * four integer counters)]
        self.pending = {}  # actor -> (species, cumulative labels, deque of contexts)
        self.before = None
        self.buckets = {}
        self._bucket_members = ()

    def prepare(self, world):
        # Standalone World.step is valid before the first observe_all call.
        """Prepare the inputs needed by the next processing phase."""

        # This internal collaborator intentionally inspects the owning component's
        # private state.
        # pylint: disable=protected-access

        if not hasattr(world, "_water_sight_cache"):
            world._water_sight_cache = OrderedDict()
        # Reconcile authoritative membership/order and bucket coordinates. Stored
        # objects expose current body/species fields, even for within-bucket moves.
        if len(self._bucket_members) == len(world.animals) and all(
            animal is previous and animal.x // 8 == bx and animal.y // 8 == by
            for animal, (previous, bx, by) in zip(
                world.animals.values(), self._bucket_members
            )
        ):
            return
        buckets = defaultdict(list)
        members = []
        for animal in world.animals.values():
            bx, by = animal.x // 8, animal.y // 8
            buckets[bx, by].append(animal)
            members.append((animal, bx, by))
        self.buckets = buckets
        self._bucket_members = members

    def begin(self, world, actions):
        """Capture pre-step visible subjects and private outcome-window contexts."""

        # The callback is consumed synchronously within this iteration; it never
        # escapes the loop.
        # pylint: disable=cell-var-from-loop

        positions = {a.id: a.position for a in world.animals.values()}
        self.prepare(world)
        buckets = self.buckets
        witnesses = {}
        for animal in world.animals.values():
            template = _template(world, animal)
            social = social_contract(template)
            if social is not None:
                sight = _sight(world, animal)
                radius = sight[2] + sight[3]
                candidates = (
                    other
                    for by in range(
                        (animal.y - radius) // 8, (animal.y + radius) // 8 + 1
                    )
                    for bx in range(
                        (animal.x - radius) // 8, (animal.x + radius) // 8 + 1
                    )
                    for other in buckets.get((bx, by), ())
                    if other.id != animal.id
                    and other.species == animal.species
                    and _can_see(world, sight, other.position)
                )
                nearest = heapq.nsmallest(
                    social["max_neighbors"],
                    candidates,
                    key=lambda a: ((a.x - animal.x) ** 2 + (a.y - animal.y) ** 2, a.id),
                )
                witnesses[animal.id] = [a.id for a in nearest]
            if (
                self.training
                and self.knowledge is not None
                and animal.species in self.knowledge.tables
            ):
                record = self.pending.setdefault(
                    animal.id, (animal.species, [0, 0, 0], deque())
                )
                record[2].append(
                    [
                        world.tick,
                        _context(
                            world,
                            animal,
                            actions[animal.id],
                            _local_context(world, animal, buckets),
                        ),
                        list(record[1]),
                    ]
                )
        self.before = positions, dict(actions), witnesses

    def finish(self, world, events, dead_animals):
        """Aggregate visible outcomes and close completed consequence windows."""

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals

        positions, actions, witnesses = self.before
        self.before = None
        all_animals = {**dead_animals, **world.animals}
        outcomes = defaultdict(lambda: [0, 0, 0])
        for event in events:
            if (
                event.kind == "damage_taken"
                and event.amount > 0
                or event.kind == "died"
            ):
                outcomes[event.actor][0] = 1
            elif event.kind == "energy_gained" and event.amount > 0:
                outcomes[event.actor][1] = 1
            elif event.kind == "born":
                outcomes[event.actor][2] = 1
                if event.mate >= 0:
                    outcomes[event.mate][2] = 1
        for observer_id, neighbors in witnesses.items():
            observer = world.animals.get(observer_id)
            if observer is None:
                continue
            sight = _sight(world, observer)
            contract = social_contract(_template(world, observer))
            counts = [0] * 24
            for subject_id in neighbors:
                subject = all_animals.get(subject_id)
                # Both ends must be seen. A disappearance alone is not a death.
                if (
                    subject is None
                    or not _can_see(world, sight, subject.position)
                    or (
                        subject.position != positions[subject_id]
                        and not _can_see(world, sight, positions[subject_id])
                    )
                ):
                    continue
                offset = GROUP[actions[subject_id]] * 4
                counts[offset] += 1
                labels = list(outcomes[subject_id])
                if labels[2]:
                    labels[2] = int(
                        any(
                            e.kind == "born"
                            and subject_id in (e.actor, e.mate)
                            and e.target in world.animals
                            and _can_see(world, sight, world.animals[e.target].position)
                            for e in events
                        )
                    )
                for i, value in enumerate(labels, 1):
                    counts[offset + i] += value
            history = self.sightings.setdefault(observer_id, deque())
            history.append((world.tick, counts))
            while history and history[0][0] <= world.tick - contract["history_steps"]:
                history.popleft()
        for animal_id in self.sightings.keys() - world.animals.keys():
            del self.sightings[animal_id]
        if self.training and self.knowledge is not None:
            for animal_id, (species, totals, queue) in list(self.pending.items()):
                for i, value in enumerate(outcomes[animal_id]):
                    totals[i] += value
                dead = animal_id in dead_animals
                while queue and (
                    dead or world.tick - queue[0][0] >= self.knowledge.horizon
                ):
                    _, key, opening = queue.popleft()
                    self.knowledge.learn(
                        species, tuple(key), [a - b for a, b in zip(totals, opening)]
                    )
                if dead:
                    del self.pending[animal_id]

    def observe(self, world, animal, template):
        """Append only the configured local social and consequence features."""
        features = []
        contract = social_contract(template)
        if contract is not None:
            totals = [0] * 24
            for tick, counts in self.sightings.get(animal.id, ()):
                if world.tick - contract["history_steps"] < tick <= world.tick:
                    totals = [a + b for a, b in zip(totals, counts)]
            for i in range(0, 24, 4):
                n = totals[i]
                features += [
                    n / (n + 4),
                    *(v / n if n else 0.0 for v in totals[i + 1 : i + 4]),
                ]
        if template.get("consequence_observation", False):
            allowed = world.config["species"][animal.species]["allowed_actions"]
            local = _local_context(world, animal, self.buckets)
            for action in ACTIONS:
                features += (
                    self.knowledge.predict(
                        animal.species, _context(world, animal, action, local)
                    )
                    if action in allowed
                    else [0.0] * 4
                )
        return features

    def checkpoint(self):
        """Serialize state required to resume this component."""
        return {
            "version": 1,
            "training": self.training,
            "knowledge": self.knowledge.checkpoint() if self.knowledge else None,
            "sightings": {
                str(i): list(history) for i, history in self.sightings.items()
            },
            "pending": {
                str(i): [
                    name,
                    list(totals),
                    [[tick, list(key), list(opening)] for tick, key, opening in queue],
                ]
                for i, (name, totals, queue) in self.pending.items()
            },
        }

    @classmethod
    def restore(cls, config, state):
        """Reconstruct the component from its persisted continuation state."""

        # Exact types intentionally reject bool/subclass substitutions in persisted
        # numeric contracts.
        # pylint: disable=unidiomatic-typecheck
        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-boolean-expressions
        # pylint: disable=too-many-branches

        result = cls(config)
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state.get("version") != 1
        ):
            raise ValueError("Missing/unsupported social checkpoint state")
        if (
            set(state) != {"version", "training", "knowledge", "sightings", "pending"}
            or type(state["training"]) is not bool
        ):
            raise ValueError("Invalid social checkpoint fields")
        if not isinstance(state["sightings"], dict) or not isinstance(
            state["pending"], dict
        ):
            raise ValueError("Invalid social checkpoint histories")
        for key, history in state["sightings"].items():
            if (
                not str(key).isdigit()
                or not isinstance(history, list)
                or len(history) > 16
            ):
                raise ValueError("Oversized social sighting history")
            last = -1
            for tick, counts in history:
                if (
                    type(tick) is not int
                    or tick <= last
                    or len(counts) != 24
                    or any(type(v) is not int or not 0 <= v <= 16 for v in counts)
                ):
                    raise ValueError("Invalid social sighting counters")
                if sum(counts[::4]) > 16 or any(
                    any(v > counts[i] for v in counts[i + 1 : i + 4])
                    for i in range(0, 24, 4)
                ):
                    raise ValueError("Inconsistent social sighting counters")
                last = tick
        if result.knowledge is None and (
            state["knowledge"] is not None or state["pending"]
        ):
            raise ValueError("Unexpected consequence state")
        for key, (name, totals, queue) in state["pending"].items():
            if (
                not str(key).isdigit()
                or result.knowledge is None
                or name not in result.knowledge.tables
                or len(totals) != 3
                or any(type(v) is not int or v < 0 for v in totals)
                or len(queue) > result.knowledge.horizon
            ):
                raise ValueError("Invalid pending consequence history")
            last = -1
            for tick, context, opening in queue:
                if (
                    type(tick) is not int
                    or tick <= last
                    or len(context) != 12
                    or any(type(v) is not int for v in context)
                    or not 0 <= context[0] < len(ACTIONS)
                    or len(opening) != 3
                    or any(
                        type(v) is not int or not 0 <= v <= totals[i]
                        for i, v in enumerate(opening)
                    )
                ):
                    raise ValueError("Invalid pending consequence context")
                last = tick
        result.training = state["training"]
        if result.knowledge is not None:
            result.knowledge = Consequences.restore(config, state["knowledge"])
        result.sightings = {
            int(i): deque((t, list(counts)) for t, counts in history)
            for i, history in state["sightings"].items()
        }
        result.pending = {
            int(i): (
                name,
                list(totals),
                deque((t, tuple(key), list(opening)) for t, key, opening in queue),
            )
            for i, (name, totals, queue) in state["pending"].items()
        }
        return result
