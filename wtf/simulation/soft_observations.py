"""V6 soft semantic basis over V5 sensed values, with bounded immutable plans."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
import json
import math
from wtf.simulation.observations import raw_feature_names
from wtf.simulation.tracking import BODY_SECTOR_NAMES

DEFAULT_CATEGORIES = {
    "distance": (0.0, 0.3, 1.0),
    "energy": (0.0, 0.25, 0.75, 1.0),
    "health": (0.0, 0.5, 1.0),
    "level": (0.0, 0.5, 1.0),
    "depth": (0.0, 0.1, 1.0),
    "elevation": (0.0, 0.1, 1.0),
    "climb": (-1.0, -0.1, 0.0, 0.1, 1.0),
    "speed": (0.0, 0.25, math.sqrt(2)),
    "outcome": (0.0, 0.001, 0.01, 0.1, 1.0),
}
CATEGORY_LABELS = {
    "distance": ("near", "medium", "far"),
    "energy": ("critical", "hungry", "comfortable", "full"),
    "health": ("critical", "weak", "healthy"),
    "level": ("none", "partial", "strong"),
    "depth": ("dry", "shallow", "deep"),
    "elevation": ("low", "middle", "high"),
    "climb": ("steep-down", "gentle-down", "level", "gentle-up", "steep-up"),
    "speed": ("still", "gentle", "strong"),
    "outcome": ("zero", "very-low", "low", "moderate", "high"),
}
BEARINGS = BODY_SECTOR_NAMES[1:]


def soft_contract(template):
    """Resolve and validate fixed-size knot lists; no dynamic state enters them."""
    overrides = template.get("soft_categories", {})
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULT_CATEGORIES):
        raise ValueError("soft_categories contains unknown category groups")
    result = {}
    for key, default in DEFAULT_CATEGORIES.items():
        values = overrides.get(key, default)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != len(default)
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
        ):
            raise ValueError(
                f"soft_categories.{key} requires {len(default)} finite numeric knots"
            )
        if (
            any(a >= b for a, b in zip(values, values[1:]))
            or values[0] != default[0]
            or values[-1] != default[-1]
        ):
            raise ValueError(
                f"soft_categories.{key} knots must increase with fixed endpoints "
                f"{default[0]}, {default[-1]}"
            )
        result[key] = tuple(values)
    return result


def memberships(value, knots):
    """Adjacent linear hats sum to one; values outside the basis saturate."""
    result = [0.0] * len(knots)
    if value <= knots[0]:
        result[0] = 1.0
    elif value >= knots[-1]:
        result[-1] = 1.0
    else:
        upper = bisect_right(knots, value)
        right = (value - knots[upper - 1]) / (knots[upper] - knots[upper - 1])
        result[upper - 1], result[upper] = 1 - right, right
    return result


def direction(forward, right):
    """Two adjacent circular bearing memberships; zero vectors have no bearing."""
    result = [0.0] * 8
    if forward or right:
        sector = (math.atan2(right, forward) % (2 * math.pi)) / (math.pi / 4)
        index = int(sector) % 8
        fraction = sector - int(sector)
        result[index] = 1 - fraction
        result[(index + 1) % 8] = fraction
    return result


def bearing(values):
    """Recover circular angle using local unwrapping of the soft sectors."""
    if not any(values):
        return 0.0
    anchor = max(range(8), key=values.__getitem__)
    return (
        sum((anchor + ((i - anchor + 4) % 8 - 4)) * w for i, w in enumerate(values))
        * math.pi
        / 4
    )


@dataclass(frozen=True, slots=True)
class Field:
    """One immutable canonical-to-semantic field mapping."""

    raw: int
    count: int
    offset: int
    label: str
    kind: str
    knots: tuple
    names: tuple


@dataclass(frozen=True, slots=True)
class SoftLayout:
    """Fixed semantic layout, shared by compatible templates rather than animals."""

    # One layout owns all derived offsets; splitting it adds lookup objects.
    # pylint: disable=too-many-instance-attributes

    fields: tuple
    names: tuple
    raw_names: tuple
    core: int
    tracking: int
    zeroes: tuple
    confidence_offsets: tuple
    raw_indices: tuple

    @property
    def width(self):
        """Encoded scalar count."""
        return len(self.names)

    def offset(self, raw):
        """Find a block boundary without retaining animal-specific data."""
        if raw == len(self.raw_names):
            return self.width
        index = bisect_left(self.raw_indices, raw)
        if index == len(self.fields) or self.raw_indices[index] != raw:
            raise StopIteration
        return self.fields[index].offset


def _scalar_kind(name):
    direct = {"Energy": "energy", "Health": "health", "Elevation": "elevation"}
    if name in direct:
        return direct[name]
    if (
        name in ("Maturity", "Cooldown", "Self consent", "Mate consent")
        or name.endswith((" age", " confidence"))
        or name.startswith("Scent ")
    ):
        return "level"
    if name == "Water depth" or name.endswith(" water"):
        return "depth"
    if name.endswith(" climb"):
        return "climb"
    if name.startswith(
        ("Collective ", "Species action ", "Consequence ")
    ) and name.endswith((" harm", " feeding", " birth")):
        return "outcome"
    return "raw"


def soft_layout(template):
    """Use a bounded structural cache; it contains no observations or world state."""
    keys = (
        "observation_top_k",
        "movement_history_steps",
        "scent_observation",
        "social_observation",
        "consequence_observation",
        "consequence_scope",
        "soft_categories",
    )
    key = json.dumps(
        {k: template[k] for k in keys if k in template},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _layout(key)


@lru_cache(maxsize=128)
def _layout(key):
    # Keep input/output offsets and field classification in one compilation pass.
    # pylint: disable=too-many-locals
    template = json.loads(key)
    template["observation_version"] = 5
    knots = soft_contract(template)
    raw = raw_feature_names(template)
    fields = []
    names = []
    index = 0
    core = 48 + 12 * (template.get("observation_top_k", 1) - 1)
    # Canonical helpers are called with V5 so optional width/label logic never recurses.
    # pylint: disable=import-outside-toplevel
    from wtf.simulation.tracking import tracking_width

    tracking = tracking_width(template)
    while index < len(raw):
        name = raw[index]
        count = 1
        kind = _scalar_kind(name)
        label = name
        if name == "Flow forward" or name.endswith("velocity forward"):
            count = 2
            kind = "vector"
            group = "speed"
            label = name.rsplit(" ", 1)[0]
        elif name.endswith(" forward") and (
            name.startswith(("Food ", "Prey ", "Predator ", "Mate "))
        ):
            count = (
                3
                if index + 2 < len(raw) and raw[index + 2].endswith(" distance")
                else 2
            )
            kind = "position"
            group = "distance"
            label = name.rsplit(" ", 1)[0]
        if kind in ("position", "vector"):
            group = "distance" if kind == "position" else "speed"
            centers = knots[group]
            labels = tuple(
                f'{label} {"range" if kind=="position" else "speed"} {part}'
                for part in CATEGORY_LABELS[group]
            ) + tuple(f"{label} direction {part}" for part in BEARINGS)
        elif kind == "raw":
            centers = ()
            labels = (name,)
        else:
            centers = knots[kind]
            labels = tuple(f"{name} {part}" for part in CATEGORY_LABELS[kind])
        fields.append(
            Field(index, count, len(names), label, kind, tuple(centers), labels)
        )
        names.extend(labels)
        index += count
    zeroes = []
    confidence_offsets = []
    for field in fields:
        if field.kind == "raw":
            zeroes.append(0.0)
        elif field.kind in ("position", "vector"):
            zeroes.extend(
                [0.0] * 11
                if field.count == 3
                else memberships(0.0, field.knots) + [0.0] * 8
            )
        else:
            zeroes.extend(memberships(0.0, field.knots))
        if field.label.startswith(
            ("Species action ", "Consequence ")
        ) and field.label.endswith(" confidence"):
            confidence_offsets.append(field.offset)
    return SoftLayout(
        tuple(fields),
        tuple(names),
        tuple(raw),
        core,
        tracking,
        tuple(zeroes),
        tuple(confidence_offsets),
        tuple(field.raw for field in fields),
    )


@lru_cache(maxsize=4096)
def _basis(value, knots, negative_zero=False):
    """Share exact immutable coefficients; signed interior zero has its own key."""
    if negative_zero:
        value = -0.0
    return tuple(memberships(value, knots))


@lru_cache(maxsize=2048)
def _direction_entries(forward, right):
    """Reuse exact deterministic bearings, without retaining observation vectors."""
    return tuple(direction(forward, right))


def encode_soft(values, template):
    """Assemble exact V6 values from a shared zero basis and active fields only."""
    layout = soft_layout(template)
    result = list(layout.zeroes)
    for field in layout.fields:
        index, offset = field.raw, field.offset
        value = values[index]
        if field.kind == "raw":
            # Preserve original scalar types and signed zero in raw flags/headings.
            result[offset] = value
        elif field.kind in ("position", "vector"):
            if field.count == 3 and not values[index + 3]:
                continue
            forward, right = values[index], values[index + 1]
            if field.count != 3 and not (forward or right):
                continue
            distance = (
                values[index + 2] if field.count == 3 else math.hypot(forward, right)
            )
            n = len(field.knots)
            result[offset : offset + n] = _basis(distance, field.knots)
            result[offset + n : offset + n + 8] = _direction_entries(forward, right)
        elif value:
            result[offset : offset + len(field.knots)] = _basis(value, field.knots)
        elif math.copysign(1.0, value) < 0:
            # A negative zero at an interior knot can produce a -0 coefficient.
            result[offset : offset + len(field.knots)] = _basis(
                value, field.knots, True
            )
    return result


@lru_cache(maxsize=4096)
def _unbasis(weights, knots):
    """Reuse exact scalar reconstructions, especially published consequence rows."""
    return sum(a * b for a, b in zip(weights, knots))


def decode_soft(values, template, raw_start=0):
    """Decode against the template's immutable structural layout."""
    return decode_layout(values, soft_layout(template), raw_start)


def decode_layout(values, layout, raw_start=0):
    """Representative V5 values for display/auxiliary targets, never policy inputs."""
    result = []
    start = bisect_left(layout.raw_indices, raw_start)
    for field in islice(layout.fields, start, None):
        block = values[field.offset : field.offset + len(field.names)]
        if field.kind == "raw":
            result.append(block[0])
        elif field.kind in ("position", "vector"):
            n = len(field.knots)
            distance = _unbasis(tuple(block[:n]), field.knots)
            angle = bearing(block[n:])
            result.extend((distance * math.cos(angle), distance * math.sin(angle)))
            if field.count == 3:
                result.append(distance)
        else:
            result.append(_unbasis(tuple(block), field.knots))
    return result
