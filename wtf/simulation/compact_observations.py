"""Opt-in V7 numeric cues, bounded movement summaries and anonymous scent bearings."""

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from wtf.simulation.actions import ACTIONS
from wtf.simulation.observations import raw_feature_names, OPPORTUNITY_NAMES
from wtf.simulation.tracking import scent_channels


@dataclass(frozen=True, slots=True)
class CompactLayout:
    """Immutable complete ordering; indices refer to canonical V5 sensed values."""

    # Keep the immutable plan and its offsets in one bounded cache entry.
    # pylint: disable=too-many-instance-attributes

    names: tuple
    core_indices: tuple
    raw_core: int
    scent_start: int
    social_start: int
    consequence_start: int
    physical_width: int
    actions: tuple
    raw_names: tuple

    @property
    def width(self):
        """Number of recorded compact scalar values."""
        return len(self.names)


def compact_layout(template):
    """Compile only immutable observation-contract metadata, never animal state."""
    # Versioned contracts intentionally name the same optional sensory fields.
    # pylint: disable=duplicate-code
    keys = (
        "observation_top_k",
        "observation_actions",
        "movement_history_steps",
        "scent_observation",
        "social_observation",
        "consequence_observation",
        "consequence_scope",
    )
    return _layout(
        json.dumps({k: template[k] for k in keys if k in template}, sort_keys=True)
    )


@lru_cache(maxsize=128)
# Compile all related offsets in one pass, preserving source order.
# pylint: disable-next=too-many-locals
def _layout(key):
    template = json.loads(key)
    raw = raw_feature_names({**template, "observation_version": 5})
    k = template.get("observation_top_k", 1)
    core = 48 + 12 * (k - 1)
    actions = tuple(template.get("observation_actions", ACTIONS))
    if (
        not actions
        or any(
            not isinstance(action, str) or action not in ACTIONS for action in actions
        )
        or len(set(actions)) != len(actions)
    ):
        raise ValueError("Invalid V7 ordered action contract")
    opportunities = dict(zip(OPPORTUNITY_NAMES, ("attack", "eat", "swim")))
    indices = tuple(
        i
        for i, name in enumerate(raw[:core])
        if not name.endswith(" distance")
        and (name not in opportunities or opportunities[name] in actions)
    )
    names = [raw[i] for i in indices]
    h = template.get("movement_history_steps", 0)
    if h:
        for category in ("Prey", "Predator"):
            for slot in range(1, k + 1):
                names.extend(
                    f"{category} track {slot} {part}"
                    for part in (
                        "last forward",
                        "last right",
                        "trend forward",
                        "trend right",
                        "freshness",
                        "visible",
                        "motion reliability",
                    )
                )
    scent_start = len(names)
    for species, sex in scent_channels(template):
        names.extend(
            f"Scent {species} {sex} {part}"
            for part in (
                "direction forward",
                "direction right",
                "strength",
                "freshness",
            )
        )
    physical = len(names)
    raw_social = (
        core + (2 * k * (7 + 3 * h) if h else 0) + 9 * len(scent_channels(template))
    )
    if template.get("social_observation") is not None:
        names.extend(raw[raw_social : raw_social + 24])
    consequence_start = len(names)
    if template.get("consequence_observation", False):
        scope = template.get("consequence_scope", "context")
        prefixes = (
            ("Species action", "Consequence")
            if scope == "both"
            else ("Species action" if scope == "species_action" else "Consequence",)
        )
        for prefix in prefixes:
            names.extend(
                f"{prefix} {action} {part}"
                for action in actions
                for part in ("confidence", "harm", "feeding", "birth")
            )
    return CompactLayout(
        tuple(names),
        indices,
        core,
        scent_start,
        raw_social,
        consequence_start,
        physical,
        actions,
        tuple(raw),
    )


# Time, path and confidence terms describe one ordered numerical reduction.
# pylint: disable-next=too-many-locals
def movement_summary(block, steps):
    """Summarize personally seen points only; no hidden movement or extrapolation."""
    if not block[5]:
        return [0.0] * 7
    age = round(block[4] * steps)
    points = {-age: (block[0], block[1])}
    for lag in range(1, steps + 1):
        offset = 7 + 3 * (lag - 1)
        if block[offset + 2]:
            points[-lag] = (block[offset], block[offset + 1])
    times = sorted(points)
    vx = vy = reliability = 0.0
    if len(times) > 1:
        first, last = points[times[0]], points[times[-1]]
        span = times[-1] - times[0]
        dx, dy = last[0] - first[0], last[1] - first[1]
        vx, vy = dx / span, dy / span
        path = sum(
            math.hypot(points[b][0] - points[a][0], points[b][1] - points[a][1])
            for a, b in zip(times, times[1:])
        )
        straightness = min(1.0, math.hypot(dx, dy) / path) if path else 1.0
        reliability = (len(times) - 1) / steps * straightness
    return [block[0], block[1], vx, vy, block[5], block[6], reliability]


def encode_compact(values, template, scent=(), *, layout=None, social=None):
    """Remove derivable/static cues and append fixed-size sensory summaries."""
    layout = compact_layout(template) if layout is None else layout
    result = [values[i] for i in layout.core_indices]
    k, h = template.get("observation_top_k", 1), template.get(
        "movement_history_steps", 0
    )
    cursor = layout.raw_core
    if h:
        width = 7 + 3 * h
        for _ in range(2 * k):
            result.extend(movement_summary(values[cursor : cursor + width], h))
            cursor += width
    if len(scent) != 4 * len(scent_channels(template)):
        raise ValueError("V7 scent summary has the wrong width")
    result.extend(scent)
    if social is not None:
        result.extend(social)
        return result
    cursor = layout.social_start
    if template.get("social_observation") is not None:
        result.extend(values[cursor : cursor + 24])
        cursor += 24
    if template.get("consequence_observation", False):
        for _ in range(2 if template.get("consequence_scope") == "both" else 1):
            for action in layout.actions:
                offset = cursor + 4 * ACTIONS.index(action)
                result.extend(values[offset : offset + 4])
            cursor += 48
    return result


def consequence_rows(observation, template, *, layout=None):
    """Expand compact evidence to stable twelve-action target rows for private heads."""
    layout = compact_layout(template) if layout is None else layout
    start = layout.consequence_start
    width = 4 * len(layout.actions)
    blocks = []
    for cursor in range(start, layout.width, width):
        rows = [(0.0, 0.0, 0.0, 0.0)] * len(ACTIONS)
        for i, action in enumerate(layout.actions):
            rows[ACTIONS.index(action)] = tuple(
                observation[cursor + 4 * i : cursor + 4 * i + 4]
            )
        blocks.append(rows)
    return blocks
