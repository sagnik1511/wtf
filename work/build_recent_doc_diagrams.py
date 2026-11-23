"""Rebuild recent-feature SVGs; local .mmd files remain ignored authoring caches."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import argparse
import html
import json

from xml.etree import ElementTree as ET

from work.build_doc_diagrams import ASSETS, ROOT, render

FLOWS = {
    "top-k-observation-pipeline": (
        "Top-k observation pipeline",
        "docs/observations.md",
        "visible-target-slots",
        """flowchart TB
  area["Species sight radius and water extension"] --> visible["Filter targets by range and wall visibility"]
  visible --> slots["Nearest k in each category; four values per slot; zero padding"]
  slots --> vector["Pack one fixed-width observation vector"]
  extras["Self and blocked moves: 8; terrain v2/3: 12; mating v3: 10"] --> vector
  vector --> sizes["k=4: version 1 / 2 / 3 has 56 / 68 / 78 inputs"]
  sizes --> brain["One private animal brain"]
  brain --> action["Choose this animal's next action"]
""",
    ),
    "inspector-current-and-sampled": (
        "Inspector: current and sampled state",
        "docs/recording.md",
        "visual-agent-inspector",
        """flowchart TB
  chunk["Decode the requested replay chunk"] --> body["Current body and static map at step t"]
  body --> sight["Current sight overlay and body fields labelled t"]
  chunk --> sample["Diagnostic sample s at or before t: saved inputs and scores"]
  sample --> panel["Radar, input gauges and action bars labelled s"]
  sample --> history["At most 64 selected-agent score vectors"]
  history --> dots["Prediction dots; discard future points on backward seek"]
""",
    ),
    "champion-chaining-between-runs": (
        "Champion chaining between runs",
        "docs/brain-library.md",
        "champion-chaining-between-runs",
        """flowchart TB
  learned["Private animals learn during a completed run"] --> compare["Compare within each species / sex / criterion"]
  incumbent["Best observed record so far; known typed warm starts seed it"] --> compare
  compare --> better{"Full rank strictly higher?"}
  better -->|Yes| replace["Retain the new snapshot and its provenance"]
  better -->|No or exact tie| keep["Keep incumbent weights, artifact ID and model generation"]
  replace --> archive["Write small run index; reuse identical binary weights"]
  keep --> archive
  archive --> selected["Choose the configured criterion for each species / sex"]
  selected --> next["Next world: independent bodies and fresh training state"]
  original["Original starting weights"] -->|No-chain mode| next
""",
    ),
    "shared-champion-weights": (
        "Shared champion weight storage",
        "docs/binary-storage.md",
        "champion-storage-measurement",
        """flowchart TB
  snapshot["Champion snapshot: signature, precision and parameters"] --> encode["Encode exact typed binary values"]
  encode --> hash["SHA-256 of the uncompressed model record"]
  hash --> exists{"Model blob already stored?"}
  exists -->|Yes| reuse["Reuse its relative reference; write no weight bytes"]
  exists -->|No| write["Write one compressed binary/XZ blob"]
  reuse --> pool["Shared champions/models pool"]
  write --> pool
  pool --> metadata["Small per-run JSON artifacts and index"]
  config["One JSON file per distinct configuration in that generation"] --> metadata
  metadata --> load["Validate hash, signature and shape; initialize private brains"]
""",
    ),
    "early-stopping-patience": (
        "Early-stopping patience",
        "docs/running.md",
        "stop-on-performance-plateaus",
        """flowchart TB
  baseline["Baseline: first completed run or a known warm-start rank"] --> finish["Finish the next whole training run"]
  finish --> watch["Check selected criteria for monitored species and both sexes"]
  watch --> improved{"Any monitored rank improved?"}
  improved -->|Yes| reset["Reset stagnant-run counter to zero"]
  improved -->|No or exact tie| count["Increment stagnant-run counter"]
  reset --> limit{"Patience K reached?"}
  count --> limit
  limit -->|No| next["Continue unless the configured run cap is reached"]
  limit -->|Yes| stop["Save this run, stop reason and final checkpoint; do not start another"]
  state["Checkpoint keeps ranks and counter"] -->|Resume| finish
""",
    ),
}


def inspector_example():
    """Controlled geometry and actual top-k inputs; invented, labelled scores."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel
    # This internal collaborator intentionally inspects the owning component's private
    # state.
    # pylint: disable=protected-access

    from wtf.config import load
    from wtf.simulation.world import Animal, Resource, World

    c = load(ROOT / "environments/river_crossing_strong_predator.json")
    rows = [
        "".join(
            "~" if 10 <= x <= 12 else "#" if x == 5 and 3 <= y <= 6 else "."
            for x in range(21)
        )
        for y in range(14)
    ]
    c["world"].update(width=21, height=14)
    c["terrain"] = {
        "map_version": 1,
        "name": "Inspector example",
        "width": 21,
        "height": 14,
        "rows": rows,
        "legend": {
            ".": {},
            "#": {"blocked": True},
            "~": {"water_depth": 1, "flow": [0, 0.4]},
        },
    }
    c["resources"]["patch_count"] = 0
    for b in c["brains"].values():
        b.update(observation_version=2, observation_top_k=4)
    for s in c["species"].values():
        s["body"].update(vision_radius=4, water_vision_bonus=3)
    w = World(c, 100)
    w.animals = {
        0: Animal(0, "predator", 6, 7, 1, 80, 55),
        1: Animal(1, "prey", 8, 5, 0, 60, 30),
        2: Animal(2, "prey", 9, 9, 0, 60, 30),
        3: Animal(3, "prey", 14, 7, 0, 60, 30),
        4: Animal(4, "predator", 3, 9, 0, 80, 55),
    }
    w.resources = {
        (x, y): Resource(x, y, 1) for x, y in ((6, 7), (8, 6), (7, 9), (14, 9))
    }
    old = w.observe_all()[0]
    w.animals[0].x = 7
    w._observation_context()
    actor = w.animals[0]
    visible = []
    for y in range(14):
        for x in range(21):
            allowed = 4 + min(3, w._water_sight_cells(actor.position, (x, y)))
            if (x - actor.x) ** 2 + (y - actor.y) ** 2 <= allowed**2 and w.visible(
                actor.position, (x, y)
            ):
                visible.append((x, y))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="700" viewBox="0 0 1040 700" role="img" aria-labelledby="title description">',
        '<title id="title">Agent inspector: current sight and sampled inputs</title>',
        '<desc id="description">Illustration, not an experiment result. The board uses current step 120 geometry; the radar uses genuine top-k features from a controlled earlier pose. Action probabilities are illustrative.</desc>',
        "<style>text{font-family:Arial,sans-serif;fill:#e6edf3}.muted{fill:#a4b5c5}</style>",
    ]

    def rect(x, y, width, height, fill, stroke=None, r=0):

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments

        parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{r}" fill="{fill}"'
            + (f' stroke="{stroke}"' if stroke else "")
            + "/>"
        )

    def text(x, y, label, size=15, color=None):
        parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}"'
            + (f' style="fill:{color}"' if color else "")
            + f">{html.escape(str(label))}</text>"
        )

    def circle(x, y, r, fill, stroke=None, dash=None):

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments

        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"'
            + (f' stroke="{stroke}"' if stroke else "")
            + (f' stroke-dasharray="{dash}"' if dash else "")
            + "/>"
        )

    rect(0, 0, 1040, 700, "#0b1118")
    text(26, 42, "Agent inspector: two clocks, one replay", 27)
    text(
        26,
        72,
        "Illustration • controlled inputs and example probabilities, not a captured experiment",
        15,
        "#a4b5c5",
    )
    text(26, 114, "BOARD • CURRENT FRAME 120", 16, "#8ac8ff")
    ox, oy, cell = 26, 132, 22
    rect(ox - 1, oy - 1, 464, 310, "#101a22", "#415569", 5)
    for y in range(14):
        for x in range(21):
            fill = (
                "#245d86"
                if 10 <= x <= 12
                else "#3a4b5c" if (x, y) in w.obstacles else "#101a22"
            )
            rect(ox + x * cell, oy + y * cell, cell - 1, cell - 1, fill)
    for x, y in visible:
        rect(ox + x * cell + 1, oy + y * cell + 1, cell - 3, cell - 3, "#254252")
    for x, y in w.resources:
        rect(ox + x * cell + 7, oy + y * cell + 7, 8, 8, "#a2c76b")
    for a in w.animals.values():
        x, y = ox + (a.x + 0.5) * cell, oy + (a.y + 0.5) * cell
        if a.species == "prey":
            circle(x, y, 5, "#7de2bd")
        else:
            parts.append(
                f'<path d="M{x},{y-7} L{x+6},{y+5} L{x-6},{y+5} Z" fill="'
                + ("#ffffff" if a.id == 0 else "#ff9a77")
                + f'" transform="rotate({a.heading*90} {x} {y})"/>'
            )
        if a.id == 0:
            circle(x, y, 10, "none", "#8ac8ff")
    text(26, 475, "Blue tiles: legal sight from the current position.", 15, "#8ac8ff")
    text(26, 501, "Walls occlude; river cells can extend the range.", 15, "#a4b5c5")
    text(26, 541, "The selected predator has moved since its last sample.", 14)
    text(26, 568, "A target visible now may be absent from the old radar.", 14)
    text(26, 612, "Other animals on the board remain visible to you;", 14, "#a4b5c5")
    text(26, 635, "the highlighted area shows what the animal can see.", 14, "#a4b5c5")
    rect(512, 105, 502, 548, "#111c27", "#415569", 8)
    text(532, 138, "INSPECTOR • RECORDED SAMPLE 110", 16, "#8ac8ff")
    text(532, 162, "Version 2 • k=4 • 68 inputs", 14, "#a4b5c5")
    cx, cy, r = 654, 279, 93
    circle(cx, cy, r, "#0b131c", "#415569")
    circle(cx, cy, r * 4 / 7, "none", "#8ac8ff", "4 5")
    parts.append(
        f'<path d="M{cx-r},{cy} H{cx+r} M{cx},{cy-r} V{cy+r}" stroke="#263341"/>'
    )
    text(cx - 5, cy - r - 9, "N", 12, "#a4b5c5")
    for i, (category, color) in enumerate(
        (("food", "#a2c76b"), ("prey", "#7de2bd"), ("predator", "#ff9a77"))
    ):
        count = 0
        for slot in range(4):
            dx, dy, _distance, present = old[
                8 + 16 * i + 4 * slot : 12 + 16 * i + 4 * slot
            ]
            if not present:
                continue
            count += 1
            x, y = cx + dx * r, cy + dy * r
            if category == "food":
                rect(x - 3, y - 3, 6, 6, color)
            else:
                circle(x, y, 4, color)
        text(785, 218 + i * 27, f"{category.title()}: {count}/4", 15, color)
    parts.append(f'<path d="M{cx},{cy} h17" stroke="#fff" stroke-width="2"/>')
    circle(cx, cy, 3, "#fff")
    text(785, 318, "Energy 67%", 14, "#a4b5c5")
    text(785, 343, "Health 100%", 14, "#a4b5c5")
    text(532, 401, "Action probabilities • example values", 17)
    values = (0.18, 0.09, 0.08, 0.07, 0.38, 0.12, 0.08)
    colors = (
        "#8ac8ff",
        "#c6a5ff",
        "#ffbd75",
        "#a1c8b0",
        "#7de2bd",
        "#ff9aab",
        "#e1d17c",
    )
    for i, (action, value, color) in enumerate(
        zip(
            (
                "Move forward",
                "Turn left",
                "Turn right",
                "Rest",
                "Attack",
                "Swim",
                "Reproduce",
            ),
            values,
            colors,
        )
    ):
        y = 422 + i * 28
        text(532, y + 12, action, 14)
        rect(656, y, 270, 12, "#0b131c", "#415569", 2)
        rect(657, y + 1, 268 * value, 10, color, r=1)
        if value == max(values):
            rect(653, y - 3, 276, 18, "none", "#e6edf3", 3)
        text(942, y + 12, f"{value:.0%}", 14)
    text(
        26,
        681,
        "Dots are recorded target slots, not every visible entity. Probabilities describe preferences, not guaranteed actions.",
        14,
        "#a4b5c5",
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

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest_path = ASSETS / "manifest.json"
    entries = json.loads(manifest_path.read_text())
    for name, (title, document, anchor, source) in FLOWS.items():
        image, nodes, edges = render(source, title, document)
        entry = {
            "title": title,
            "document": document,
            "anchor": anchor,
            "source": name + ".mmd",
            "asset": name + ".svg",
            "nodes": nodes,
            "edges": edges,
        }
        target = ASSETS / entry["asset"]
        found = next(
            (i for i, e in enumerate(entries) if e["asset"] == entry["asset"]), None
        )
        if args.check:
            assert target.read_text() == image, target
            assert found is not None and entries[found] == entry, entry
        else:
            (ASSETS / entry["source"]).write_text(source)
            target.write_text(image)
            if found is None:
                entries.append(entry)
            else:
                entries[found] = entry
    example = ROOT / "docs/assets/inspector/inspector-example.svg"
    illustration = inspector_example()
    ET.fromstring(illustration)
    if args.check:
        assert example.read_text() == illustration, example
    else:
        example.parent.mkdir(parents=True, exist_ok=True)
        example.write_text(illustration)
        manifest_path.write_text(json.dumps(entries, indent=2) + "\n")
    print(
        ("Checked" if args.check else "Rendered")
        + " five flow diagrams and one inspector illustration."
    )


if __name__ == "__main__":
    main()
