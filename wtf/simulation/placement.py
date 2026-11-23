"""Founder zones, adjacent same-species pairs and boundary-biased food placement."""

from collections import deque
import random


def zone(position, bounds):
    """Resolve the configured placement zone."""
    x, y = position
    left, top, right, bottom = bounds
    if left < x < right and top < y < bottom:
        return "inside"
    if x < left or x > right or y < top or y > bottom:
        return "outside"
    return "boundary"


def resource_pools(free, placement):
    """Build the eligible resource placement pools."""
    bounds = placement["enclosure_bounds"]
    left, top, right, bottom = bounds
    width = placement["resources"]["boundary_width"]
    pools = {"inside": [], "outside": [], "far": []}
    for p in free:
        side = zone(p, bounds)
        if side == "boundary":
            continue
        x, y = p
        distance = (
            min(x - left, right - x, y - top, bottom - y)
            if side == "inside"
            else max(left - x, x - right, top - y, y - bottom)
        )
        pools[side if distance <= width else "far"].append(p)
    return pools


def resource_counts(count, rules):
    """Compute the resource counts assigned to placement pools."""
    near = int(count * rules["boundary_fraction"] + 0.5)
    inside = int(near * rules["inside_fraction"] + 0.5)
    return {"inside": inside, "outside": near - inside, "far": count - near}


def validate_placement(config, unavailable):
    """Validate the configured animal and resource placement rules."""

    # This import is deferred until the operation runs, after both modules have
    # initialized.
    # pylint: disable=cyclic-import

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    from wtf.config import obj, num, fail

    p = config["placement"]
    obj(p, "placement", ("enclosure_bounds", "species", "resources"))
    if config["obstacles"]["generation"] != "explicit":
        fail("placement", "requires explicit obstacles for predictable zone capacity")
    bounds = p["enclosure_bounds"]
    if not isinstance(bounds, list) or len(bounds) != 4:
        fail("placement.enclosure_bounds", "expected [left, top, right, bottom]")
    w = config["world"]
    for i, n in enumerate(bounds):
        num(
            n,
            f"placement.enclosure_bounds[{i}]",
            0,
            w["width" if i % 2 == 0 else "height"] - 1,
            True,
        )
    if bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 2:
        fail("placement.enclosure_bounds", "must enclose at least one interior cell")
    obj(p["species"], "placement.species", tuple(config["species"]))
    for name, side in p["species"].items():
        if side not in ("inside", "outside"):
            fail("placement.species." + name, "expected inside or outside")
    r = p["resources"]
    obj(
        r,
        "placement.resources",
        ("boundary_fraction", "boundary_width", "inside_fraction"),
    )
    num(r["boundary_fraction"], "placement.resources.boundary_fraction", 0, 1)
    num(r["inside_fraction"], "placement.resources.inside_fraction", 0, 1)
    num(r["boundary_width"], "placement.resources.boundary_width", 1, 32, True)
    free = [
        (x, y)
        for y in range(w["height"])
        for x in range(w["width"])
        if (x, y) not in unavailable
    ]
    for side in ("inside", "outside"):
        demand = sum(
            s["initial_population"]
            for name, s in config["species"].items()
            if p["species"][name] == side
        )
        if demand > sum(zone(cell, bounds) == side for cell in free):
            fail("placement.species", f"insufficient {side} dry cells for founders")
    pools = resource_pools(free, p)
    for side, count in resource_counts(config["resources"]["patch_count"], r).items():
        if count > len(pools[side]):
            fail(
                "placement.resources",
                f"insufficient {side} dry cells for requested food allocation",
            )


def _pair_cells(world, species, occupied):
    """Find enough disjoint walkable edges when fixed founder anchors cannot fit."""
    # Keep the bounded augmenting-path state together; it enforces disjoint pairs.
    # pylint: disable=too-many-locals

    spec = world.config["species"][species]
    placement = world.config.get("placement")
    side = placement["species"][species] if placement else None
    climb = spec.get("locomotion", {}).get("max_climb", 1.0)
    cells = {
        (x, y)
        for y in range(world.height)
        for x in range(world.width)
        if (x, y) not in occupied
        and not world.blocked((x, y))
        and not (world.geography and world.geography.depth((x, y)))
        and (not placement or zone((x, y), placement["enclosure_bounds"]) == side)
    }
    left = sorted(p for p in cells if (p[0] + p[1]) % 2 == 0)
    rng = random.Random(world.seed ^ 0x50414952)
    rng.shuffle(left)
    matched_left, matched_right = {}, {}
    needed = spec["initial_population"] // 2
    for start in left:
        queue, seen, previous = deque([start]), {start}, {}
        found = None
        while queue and found is None:
            current = queue.popleft()
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                target = current[0] + dx, current[1] + dy
                if target not in cells or target in previous:
                    continue
                if (
                    world.geography
                    and abs(
                        world.geography.elevation(current)
                        - world.geography.elevation(target)
                    )
                    > climb
                ):
                    continue
                previous[target] = current
                if target not in matched_right:
                    found = target
                    break
                other = matched_right[target]
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        if found is None:
            continue
        while found is not None:
            current = previous[found]
            old = matched_left.get(current)
            matched_left[current] = found
            matched_right[found] = current
            found = old
        if len(matched_left) == needed:
            return list(matched_left.items())
    raise ValueError(
        f"species.{species}.spawn: insufficient adjacent dry, reachable "
        "cell pairs in its spawn zone"
    )


def place_adjacent_pairs(world):
    """Opt-in founder pairing before brain creation, without consuming world RNG."""
    # Placement owns this ordered phase; splitting it obscures reservations/rollback.
    # pylint: disable=too-many-locals,too-many-branches

    for index, (species, spec) in enumerate(world.config["species"].items()):
        if spec.get("spawn", {}).get("mode", "random") != "adjacent_pairs":
            continue
        founders = [a for a in world.animals.values() if a.species == species]
        females = [a for a in founders if a.sex == "female"]
        males = [a for a in founders if a.sex == "male"]
        if len(females) != len(males):
            raise ValueError(
                f"species.{species}.spawn: adjacent_pairs requires an even initial_population"
            )
        if not founders:
            continue
        originals = {a.id: a.position for a in founders}
        other_cells = {
            a.position for a in world.animals.values() if a.species != species
        }
        occupied = other_cells | {a.position for a in females}
        rng = random.Random(world.seed ^ 0x50414952 ^ (index * 7919))
        placement = world.config.get("placement")
        climb = spec.get("locomotion", {}).get("max_climb", 1.0)
        for female, male in zip(females, males):
            candidates = []
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                target = female.x + dx, female.y + dy
                if target in occupied or world.blocked(target):
                    continue
                if (
                    placement
                    and zone(target, placement["enclosure_bounds"])
                    != placement["species"][species]
                ):
                    continue
                if world.geography and (
                    world.geography.depth(target)
                    or abs(
                        world.geography.elevation(target)
                        - world.geography.elevation(female.position)
                    )
                    > climb
                ):
                    continue
                candidates.append(target)
            if not candidates:
                for a in founders:
                    a.x, a.y = originals[a.id]
                pairs = _pair_cells(world, species, other_cells)
                for (mother, father), (first, second) in zip(
                    zip(females, males), pairs
                ):
                    mother.x, mother.y = first
                    father.x, father.y = second
                break
            male.x, male.y = rng.choice(candidates)
            occupied.add(male.position)
