"""Static JSON geography and terrain-local observations; no learning dependencies."""

import json
import math
from pathlib import Path
from types import MappingProxyType


def resolve_map(value):
    """Load and resolve the configured geography source."""

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    if isinstance(value, dict) and "map_file" in value:
        if (
            set(value) != {"map_file"}
            or not isinstance(value["map_file"], str)
            or not value["map_file"].strip()
        ):
            raise ValueError(
                "terrain.map_file: use a nonempty path alone, or an inline map definition"
            )
        path = Path(value["map_file"]).expanduser()
        if not path.is_absolute():
            local = Path.cwd() / path
            if local.is_file():
                return json.loads(local.read_text(encoding="utf-8"))
            from wtf.resources import asset

            return json.loads(asset(path.as_posix()).read_text(encoding="utf-8"))
        return json.loads(path.read_text(encoding="utf-8"))
    return value


def validate_map(value, width, height):
    """Validate terrain dimensions and per-cell geography fields."""

    # Exact types intentionally reject bool/subclass substitutions in persisted
    # numeric contracts.
    # pylint: disable=unidiomatic-typecheck
    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches

    def fail(message):
        raise ValueError("terrain: " + message)

    def number(x, low, high):
        return type(x) in (int, float) and math.isfinite(x) and low <= x <= high

    if not isinstance(value, dict) or set(value) != {
        "map_version",
        "name",
        "width",
        "height",
        "rows",
        "legend",
    }:
        fail("expected map_version, name, width, height, rows and legend")
    if type(value["map_version"]) is not int or value["map_version"] != 1:
        fail("unsupported map_version")
    if not isinstance(value["name"], str) or not value["name"].strip():
        fail("name must be nonempty")
    if (
        type(value["width"]) is not int
        or type(value["height"]) is not int
        or (value["width"], value["height"]) != (width, height)
    ):
        fail("map dimensions must match world dimensions")
    rows, legend = value["rows"], value["legend"]
    if (
        not isinstance(rows, list)
        or len(rows) != height
        or any(not isinstance(r, str) or len(r) != width for r in rows)
    ):
        fail("rows must have exactly height strings of width characters")
    if not isinstance(legend, dict) or not legend:
        fail("legend must be nonempty")
    for symbol, cell in legend.items():
        if (
            not isinstance(symbol, str)
            or len(symbol) != 1
            or not 32 <= ord(symbol) <= 126
            or not isinstance(cell, dict)
        ):
            fail("legend uses single-character keys and objects")
        if set(cell) - {"elevation", "blocked", "water_depth", "flow"}:
            fail("unknown legend cell property")
        if not number(cell.get("elevation", 0), 0, 100):
            fail("elevation must be finite and between 0 and 100")
        if type(cell.get("blocked", False)) is not bool:
            fail("blocked must be boolean")
        if not number(cell.get("water_depth", 0), 0, 10):
            fail("water_depth must be between 0 and 10")
        flow = cell.get("flow", [0, 0])
        if (
            not isinstance(flow, list)
            or len(flow) != 2
            or any(not number(v, -4, 4) for v in flow)
        ):
            fail("flow must be [dx, dy], each between -4 and 4 cells/step")
        if cell.get("water_depth", 0) <= 0 and any(flow):
            fail("flow requires water_depth > 0")
        if cell.get("blocked", False) and cell.get("water_depth", 0) > 0:
            fail("a cell cannot be both blocked and water")
    if any(symbol not in legend for row in rows for symbol in row):
        fail("rows contain undefined legend symbols")
    return value


class Geography:
    """Sparse static lookups; only the compact definition is saved with config."""

    def __init__(self, definition, width, height):
        self.definition = resolve_map(definition)
        validate_map(self.definition, width, height)
        self.elevations = {}
        self.water = {}
        self.obstacles = set()
        for y, row in enumerate(self.definition["rows"]):
            for x, symbol in enumerate(row):
                cell = self.definition["legend"][symbol]
                p = (x, y)
                if cell.get("elevation", 0):
                    self.elevations[p] = cell["elevation"]
                if cell.get("blocked", False):
                    self.obstacles.add(p)
                if cell.get("water_depth", 0) > 0:
                    self.water[p] = (cell["water_depth"], *cell.get("flow", [0, 0]))
        self.elevations = MappingProxyType(self.elevations)
        self.water = MappingProxyType(self.water)
        self.obstacles = frozenset(self.obstacles)

    def elevation(self, position):
        """Return the elevation at the requested grid coordinate."""
        return self.elevations.get(position, 0)

    def depth(self, position):
        """Return the water depth at the requested grid coordinate."""
        return self.water.get(position, (0, 0, 0))[0]

    def features(self, animal, directions):
        """Build the local terrain observation features."""
        p = animal.position
        depth, fx, fy = self.water.get(p, (0, 0, 0))
        height = self.elevation(p)
        values = [depth / 10, fx / 4, fy / 4, height / 100]
        for offset in range(4):
            dx, dy = directions[(animal.heading + offset) % 4]
            q = (animal.x + dx, animal.y + dy)
            values.extend((self.depth(q) / 10, (self.elevation(q) - height) / 100))
        return values
