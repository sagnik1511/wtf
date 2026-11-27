"""Version-five coordinate encoding; no policy decisions or hidden target lookup."""

from wtf.simulation.actions import DIRECTIONS
from wtf.simulation.tracking import SECTORS, scent_channels

OPPORTUNITY_NAMES = ("Attack possible now", "Eat possible now", "Swim possible now")
BASE_WIDTHS = {1: 20, 2: 32, 3: 42, 4: 45, 5: 48}
# Four immutable cardinal permutations, shared by all animals and decision phases.
SCENT_ORDERS = tuple(
    tuple(SECTORS.index((-y * hx - x * hy, -y * hy + x * hx)) for x, y in SECTORS)
    for hx, hy in DIRECTIONS
)


def relative(dx, dy, heading):
    """Project a world vector onto the animal's forward/right basis."""
    hx, hy = DIRECTIONS[heading]
    return dx * hx + dy * hy, -dx * hy + dy * hx


def opportunities(world, animal, occupied, body):
    """Describe immediate affordances without masking or predicting other actions."""
    actions = world.config["species"][animal.species]["allowed_actions"]
    front = world.front(animal)
    target = occupied.get(front)
    attack = (
        "attack" in actions
        and world.diet(animal) == "prey"
        and target is not None
        and world.diet(target) == "resource"
        and body["attack_damage"] > 0
    )
    eat = (
        "eat" in actions
        and animal.energy < body["max_energy"]
        and body["bite_size"] > 0
        and body["energy_per_food_unit"] > 0
        and any(
            position in world.resources and world.resources[position].quantity > 0
            for position in (animal.position, front)
        )
    )
    swim = "swim" in actions and not world.movement_block_reason(
        animal, front, "swim", occupied
    )
    return float(attack), float(eat), float(swim)


def encode_v5(features, template, heading, available):
    """Convert the assembled v4 spatial blocks and insert three affordance values.

    Memory stays in world coordinates internally. Encoding is a projection of
    current personally observed/remembered inputs, never a fresh sight query.
    """
    k = template.get("observation_top_k", 1)
    terrain = 8 + 12 * k
    core = terrain + 25

    def rotate(index):
        features[index : index + 2] = relative(*features[index : index + 2], heading)

    for index in range(8, terrain, 4):
        rotate(index)
    rotate(terrain + 1)
    rotate(terrain + 16)  # Mate forward/right, after the 12-value terrain block.
    steps = template.get("movement_history_steps", 0)
    offset = core
    if steps:
        for _ in range(2 * k):
            rotate(offset)
            rotate(offset + 2)
            for lag in range(steps):
                rotate(offset + 7 + 3 * lag)
            offset += 7 + 3 * steps
    order = SCENT_ORDERS[heading]
    for _ in (
        () if template.get("observation_version") == 7 else scent_channels(template)
    ):
        values = features[offset : offset + 9]
        features[offset : offset + 9] = [values[index] for index in order]
        offset += 9
    features[core:core] = available
    return features


def raw_feature_names(signature):
    """Name the model inputs in their exact observation-vector order."""
    # Lazy imports keep encoding independent of controller construction.
    # pylint: disable=import-outside-toplevel
    from wtf.simulation.tracking import tracking_names
    from wtf.simulation.social import social_names

    names = [
        "Energy",
        "Health",
        "Heading x",
        "Heading y",
        "Blocked front",
        "Blocked right",
        "Blocked back",
        "Blocked left",
    ]
    spatial = (
        ("forward", "right")
        if signature.get("observation_version", 1) == 5
        else ("dx", "dy")
    )
    for category in ("Food", "Prey", "Predator"):
        for slot in range(1, signature.get("observation_top_k", 1) + 1):
            names.extend(
                f"{category} {slot} {part}"
                for part in (*spatial, "distance", "present")
            )
    version = signature.get("observation_version", 1)
    if version >= 2:
        names.extend(
            [
                "Water depth",
                *(
                    ["Flow forward", "Flow right"]
                    if version == 5
                    else ["Flow x", "Flow y"]
                ),
                "Elevation",
            ]
        )
        for side in ("Front", "Right", "Back", "Left"):
            names.extend([f"{side} water", f"{side} climb"])
    if version >= 3:
        names.extend(
            [
                "Sex",
                "Maturity",
                "Cooldown",
                "Self ready",
                "Mate forward" if version == 5 else "Mate dx",
                "Mate right" if version == 5 else "Mate dy",
                "Mate distance",
                "Mate present",
                "Mate ready",
                "Pair opportunity",
            ]
        )
    if version >= 4:
        names.extend(["Self consent", "Mate consent", "Consenting pair"])
    if version == 5:
        names.extend(OPPORTUNITY_NAMES)
    return names + tracking_names(signature) + social_names(signature)
