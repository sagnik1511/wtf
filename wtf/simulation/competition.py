"""Opt-in same-species contests. No learning, persistence or viewer dependencies."""


def food_patch(world, animal):
    body = world.body(animal)
    if (
        world.diet(animal) != "resource"
        or animal.energy >= body["max_energy"]
        or body["bite_size"] <= 0
        or body["energy_per_food_unit"] <= 0
    ):
        return None
    return next(
        (
            p
            for p in (animal.position, world.front(animal))
            if p in world.resources and world.resources[p].quantity > 0
        ),
        None,
    )


def opportunity(world, animal, positions):
    """Physical opportunity only; do not inspect future actions or consume RNG."""
    rules = world.config["species"][animal.species].get("competition", {})
    if (
        not rules.get("enabled", False)
        or animal.health <= 0
        or animal.energy <= rules.get("energy_cost", 1.0)
        or (world.geography and world.geography.depth(animal.position))
    ):
        return None, None

    def eligible(target):
        return (
            target is not None
            and target.id != animal.id
            and target.species == animal.species
            and target.health > 0
            and not (world.geography and world.geography.depth(target.position))
        )

    target = positions.get(world.front(animal))
    if eligible(target):
        patch = food_patch(world, animal)
        if (
            rules.get("food", True)
            and patch is not None
            and patch == food_patch(world, target)
        ):
            return target, "food"
        if rules.get("blocked_path", True):
            return target, "blocked_path"
        if rules.get("aggression", True):
            return target, "aggression"
    if rules.get("aggression", True):
        from wtf.simulation.world import DIRECTIONS

        nearby = (
            positions.get((animal.x + dx, animal.y + dy)) for dx, dy in DIRECTIONS
        )
        target = min(
            (a for a in nearby if eligible(a)), key=lambda a: a.id, default=None
        )
        if target is not None:
            return target, "aggression"
    return None, None


def _advance(world, winner, loser, events, Event):
    """Advance only the challenger; never swap or push through terrain/occupants."""
    destination = loser.position
    occupied = {a.position for a in world.animals.values() if a.health > 0}
    if world.movement_block_reason(
        winner, destination, "move_forward", occupied - {destination}
    ):
        return
    retreat = None
    if loser.health > 0:
        from wtf.simulation.world import DIRECTIONS

        choices = [(loser.x + dx, loser.y + dy) for dx, dy in DIRECTIONS]
        choices.sort(
            key=lambda p: (-((p[0] - winner.x) ** 2 + (p[1] - winner.y) ** 2), p)
        )
        retreat = next(
            (
                p
                for p in choices
                if not world.movement_block_reason(loser, p, "move_forward", occupied)
            ),
            None,
        )
        if retreat is None:
            return
    # Check both destinations before changing either body.
    for animal, target, kind in (
        (loser, retreat, "displaced"),
        (winner, destination, "contest_moved"),
    ):
        if target is None:
            continue
        cost = world.body(animal)["movement_cost"] + world.terrain_move_cost(
            animal, target, "move_forward"
        )
        world._terrain_charge(animal, cost, events, "competition_movement")
        animal.x, animal.y = target
        events.append(
            Event(kind, animal.id, position=list(target), cause="competition")
        )


def resolve(world, actions, events):
    """One contest per participant per step; return food priority, deaths, successes."""
    from wtf.simulation.world import Event

    positions = {a.position: a for a in world.animals.values()}
    challengers = [a for a in world.animals.values() if actions[a.id] == "challenge"]
    candidates = {}
    # Eligibility is checked before paying attempt cost; failed choices still pay
    # configured challenge effort. Opponent choices are inspected only here.
    for animal in challengers:
        target, kind = opportunity(world, animal, positions)
        rules = world.config["species"][animal.species].get("competition", {})
        if rules.get("enabled", False):
            world._terrain_charge(
                animal, rules.get("energy_cost", 1.0), events, "competition_attempt"
            )
        if kind == "food" and actions[target.id] not in ("eat", "challenge"):
            target, kind = (
                (target, "blocked_path")
                if rules.get("blocked_path", True)
                else (
                    (target, "aggression")
                    if rules.get("aggression", True)
                    else (None, None)
                )
            )
        candidates[animal.id] = target, kind
        events.append(
            Event(
                "challenge_attempt",
                animal.id,
                target.id if target else -1,
                cause=kind or "unavailable",
                position=list(animal.position),
            )
        )
    engaged, successful, deaths, priority = set(), set(), set(), []
    for animal in challengers:
        target, kind = candidates[animal.id]
        if target is None or animal.id in engaged or target.id in engaged:
            continue
        # Earlier displacements can invalidate a previously considered pairing.
        if (
            animal.health <= 0
            or target.health <= 0
            or (
                abs(animal.x - target.x) + abs(animal.y - target.y) != 1
                if kind == "aggression"
                else world.front(animal) != target.position
            )
        ):
            continue
        engaged.update((animal.id, target.id))
        for challenger, opponent in ((animal, target), (target, animal)):
            if (
                actions[challenger.id] == "challenge"
                and candidates[challenger.id][0] is opponent
            ):
                successful.add(challenger.id)
                events.append(
                    Event("challenge_resolved", challenger.id, opponent.id, cause=kind)
                )

        def strength(a):
            body = world.body(a)
            return a.health / body["max_health"] + a.energy / body["max_energy"]

        left, right = strength(animal), strength(target)
        winner = (
            animal
            if left > right
            else target if right > left else world.rng.choice((animal, target))
        )
        loser = target if winner is animal else animal
        rules = world.config["species"][winner.species]["competition"]
        damage = min(loser.health, rules.get("damage", 5.0))
        loser.health -= damage
        events.append(
            Event(
                "contest_won",
                winner.id,
                loser.id,
                damage,
                kind,
                position=list(winner.position),
            )
        )
        events.append(
            Event(
                "contest_lost",
                loser.id,
                winner.id,
                damage,
                kind,
                position=list(loser.position),
            )
        )
        if damage:
            events.append(
                Event("damage_dealt", winner.id, loser.id, damage, "competition")
            )
            events.append(
                Event("damage_taken", loser.id, winner.id, damage, "competition")
            )
        if loser.health <= 0:
            deaths.add(loser.id)
        if kind == "food":
            priority.append(winner.id)
        elif kind == "blocked_path" and winner is animal:
            _advance(world, winner, loser, events, Event)
    return priority, deaths, successful
