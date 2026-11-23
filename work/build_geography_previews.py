"""Render static geography diagrams from the editable maps, without training."""

# Keep independent numeric/compatibility or benchmark paths explicit to preserve their
# exact contracts.
# pylint: disable=duplicate-code


# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import argparse
import math

from html import escape
from pathlib import Path

from wtf.config import load

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "meandering_river",
    "river_delta",
    "alpine_pass",
    "lake_islands",
    "grasslands",
    "seaside",
    "rolling_hills",
    "wetlands",
    "canyon",
    "oasis",
)
TILE = 16


def preview(config):
    """Render a static preview of the configured geography."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals

    terrain = config["terrain"]
    width, height = terrain["width"], terrain["height"]
    maximum = max(cell.get("elevation", 0) for cell in terrain["legend"].values())
    canvas_width, canvas_height = width * TILE + 32, height * TILE + 116
    title = terrain["name"].replace("_", " ").title()
    light_rules = set()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" '
        f'height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}" '
        'role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        '<desc id="desc">Static map diagram. Green is dry land, brown is higher '
        "land, blue is water, gray is blocked rock. Arrows show configured "
        "currents. Animals and food are not shown.</desc>",
        '<rect width="100%" height="100%" fill="#111c21"/>',
        '<g font-family="system-ui, sans-serif" fill="#e1e9e8">',
        f'<text x="16" y="29" font-size="21">{escape(title)}</text>',
        f'<text x="16" y="49" font-size="12">Static geography · {width} × {height} cells · not a viewer recording</text>',
    ]
    for y, row in enumerate(terrain["rows"]):
        for x, symbol in enumerate(row):
            cell = terrain["legend"][symbol]
            depth, elevation = cell.get("water_depth", 0), cell.get("elevation", 0)
            if cell.get("blocked"):
                color = "#89949a"
                light_color = "#5e707e"
            elif depth:
                color = f"hsl(207,65%,{22 + min(depth, 3) * 5}%)"
                light_color = f"hsl(207,65%,{65 - min(depth, 3) * 5}%)"
            elif elevation:
                ratio = elevation / maximum
                color = f"hsl({85 - 50 * ratio},25%,{19 + 14 * ratio}%)"
                light_color = f"hsl({85 - 50 * ratio},25%,{84 - 20 * ratio}%)"
            else:
                color = "#24382e"
                light_color = "#d8e6cc"
            light_rules.add(f"[fill='{color}'] {{ fill: {light_color}; }}")
            px, py = 16 + x * TILE, 60 + y * TILE
            parts.append(
                f'<rect x="{px}" y="{py}" width="16" height="16" '
                f'fill="{color}" stroke="#ffffff" stroke-opacity=".05"/>'
            )
            fx, fy = cell.get("flow", (0, 0))
            if (fx or fy) and x % 2 == 0 and y % 3 == 0:
                length = math.hypot(fx, fy)
                ux, uy = fx / length, fy / length
                cx, cy, r = px + 8, py + 8, 5
                tip_x, tip_y = cx + ux * r, cy + uy * r
                parts.append(
                    f'<path d="M{cx-ux*r} {cy-uy*r} L{tip_x} {tip_y} '
                    f"M{tip_x-ux*3-uy*2} {tip_y-uy*3+ux*2} L{tip_x} {tip_y} "
                    f'L{tip_x-ux*3+uy*2} {tip_y-uy*3-ux*2}" '
                    'fill="none" stroke="#c0e9ff" stroke-width="1"/>'
                )
    bottom = 60 + height * TILE
    parts.extend(
        [
            f'<text x="16" y="{bottom+23}" font-size="12">Green: dry land · Brown: elevation · Gray: rock · Blue: water</text>',
            f'<text x="16" y="{bottom+42}" font-size="12">Arrows: currents · Blue shades show water depth · Swim required to enter water</text>',
            "</g>",
        ]
    )
    parts.append(
        "<style>@media (prefers-color-scheme: light) {"
        "[fill='#111c21'] {fill:#f1f5fa;} [fill='#e1e9e8'] {fill:#1c3543;} "
        "[stroke='#c0e9ff'] {stroke:#145b8f;} [stroke='#ffffff'] {stroke:#172c3c;} "
        + "".join(sorted(light_rules))
        + "}</style>"
    )
    parts.append("</svg>")
    parts.insert(
        1,
        "<style>"
        + (ROOT / "docs/assets/viewer/theme-palette.css").read_text()
        + "</style>",
    )
    return "\n".join(parts) + "\n"


def main():
    """Parse command-line options and run the requested operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "docs/assets/geography"
    destination.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        config = load(ROOT / "environments" / (name + ".json"))
        target = destination / (name + ".svg")
        expected = preview(config)
        if args.check:
            if not target.is_file() or target.read_text() != expected:
                raise SystemExit(f"Outdated geography preview: {target.name}")
        else:
            target.write_text(expected)
    print(
        f"{'Verified' if args.check else 'Rendered'} {len(NAMES)} static map previews in docs/assets/geography"
    )


if __name__ == "__main__":
    main()
