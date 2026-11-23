"""Shared action groups and replay movement labels; no world dependencies."""

ACTIONS = (
    "move_forward",
    "move_backward",
    "strafe_left",
    "strafe_right",
    "turn_left",
    "turn_right",
    "rest",
    "eat",
    "attack",
    "swim",
    "reproduce",
    "challenge",
)
WALK_OFFSETS = {
    "move_forward": 0,
    "move_backward": 2,
    "strafe_left": -1,
    "strafe_right": 1,
}
WALK_ACTIONS = tuple(WALK_OFFSETS)
MOVE_ACTIONS = (*WALK_ACTIONS, "swim")
ATTEMPT_ACTIONS = (*MOVE_ACTIONS, "attack", "eat", "reproduce", "challenge")


def movement_action(cause):
    """Decode new event labels and legacy unlabeled forward/swim events."""
    if cause.startswith("swim"):
        return "swim"
    candidate = cause.split(":", 1)[0]
    return candidate if candidate in WALK_OFFSETS else "move_forward"
