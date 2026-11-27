"""Heading-relative visual field geometry, independent of range and occlusion."""

import math
from wtf.simulation.actions import DIRECTIONS

# cos(75 degrees)**2, using the same closed form in the JavaScript renderer.
HALF_ANGLE_COS2 = {150: (2 - math.sqrt(3)) / 4, 180: 0.0}


def in_field(start, end, heading, degrees=360):
    """Include the origin and angular boundaries for configured visual fields."""
    if degrees == 360:
        return True
    if degrees not in HALF_ANGLE_COS2:
        raise ValueError("Visual field must be 150, 180 or 360 degrees")
    hx, hy = DIRECTIONS[heading]
    dx, dy = end[0] - start[0], end[1] - start[1]
    dot = dx * hx + dy * hy
    cosine_squared = HALF_ANGLE_COS2[degrees]
    return dot >= 0 and (
        not cosine_squared or dot * dot >= (dx * dx + dy * dy) * cosine_squared
    )
