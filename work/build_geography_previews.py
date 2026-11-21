"""Render static geography diagrams from the editable maps, without training."""

from html import escape
from pathlib import Path

from ecosystem.config import load

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("meandering_river", "river_delta", "alpine_pass", "lake_islands")
TILE = 16


def preview(config):
    terrain = config["terrain"]
    width, height = terrain["width"], terrain["height"]
    maximum = max(cell.get("elevation", 0) for cell in terrain["legend"].values())
    canvas_width, canvas_height = width * TILE + 32, height * TILE + 116
    title = terrain["name"].replace("_", " ").title()
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
        '<text x="16" y="49" font-size="12">Static geography · 40 × 28 cells · not a viewer recording</text>',
    ]
    for y, row in enumerate(terrain["rows"]):
        for x, symbol in enumerate(row):
            cell = terrain["legend"][symbol]
            depth, elevation = cell.get("water_depth", 0), cell.get("elevation", 0)
            if cell.get("blocked"):
                color = "#89949a"
            elif depth:
                color = f"hsl(207,65%,{22 + min(depth, 3) * 5}%)"
            elif elevation:
                ratio = elevation / maximum
                color = f"hsl({85 - 50 * ratio},25%,{19 + 14 * ratio}%)"
            else:
                color = "#24382e"
            px, py = 16 + x * TILE, 60 + y * TILE
            parts.append(
                f'<rect x="{px}" y="{py}" width="16" height="16" '
                f'fill="{color}" stroke="#ffffff" stroke-opacity=".05"/>'
            )
            fx, fy = cell.get("flow", (0, 0))
            if (fx or fy) and x % 2 == 0 and y % 3 == 0:
                # Current vectors in this gallery all flow downstream.
                arrow = "↘" if fx > 0 else ("↙" if fx < 0 else "↓")
                parts.append(
                    f'<text x="{px+8}" y="{py+12}" text-anchor="middle" '
                    f'font-size="12" fill="#c0e9ff">{arrow}</text>'
                )
    bottom = 60 + height * TILE
    parts.extend(
        [
            f'<text x="16" y="{bottom+23}" font-size="12">Green: dry land · Brown: elevation · Gray: rock · Blue: water</text>',
            f'<text x="16" y="{bottom+42}" font-size="12">Arrows: currents · Lighter blue: deeper water · Swim required to enter water</text>',
            "</g></svg>",
        ]
    )
    parts.insert(
        1,
        "<style>"
        + (ROOT / "docs/assets/viewer/theme-palette.css").read_text()
        + "</style>",
    )
    return "\n".join(parts) + "\n"


def main():
    destination = ROOT / "docs/assets/geography"
    destination.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        config = load(ROOT / "environments" / (name + ".json"))
        (destination / (name + ".svg")).write_text(preview(config))
    print(f"Rendered {len(NAMES)} static map previews in docs/assets/geography")


if __name__ == "__main__":
    main()
