"""Deterministic sunlight and species modifiers, derived solely from world tick."""

import math


def sunlight(cycle, tick):
    """Compute sunlight from the configured saved-world time."""
    if not cycle or not cycle.get("enabled", False):
        return 1.0
    phase = (
        cycle.get("start_phase", 0.25)
        + (tick % cycle["cycle_steps"]) / cycle["cycle_steps"]
    ) % 1
    return math.cos(2 * math.pi * (phase - 0.5)) if 0.25 < phase < 0.75 else 0.0


def effective_body(species, light):
    """Apply configured day/night modifiers to the species body."""
    body = species["body"]
    rules = species.get("circadian")
    if not rules:
        return body
    result = dict(body)
    for field in ("vision_radius", "living_cost", "movement_cost"):
        night = rules.get("night_" + field + "_multiplier", 1.0)
        value = body[field] * (night + (1 - night) * light)
        result[field] = (
            max(1, min(256, math.floor(value + 0.5)))
            if field == "vision_radius"
            else value
        )
    return result
