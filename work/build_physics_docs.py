"""Refresh physics documentation from World.step; no training or extra packages."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long
# This internal collaborator intentionally inspects the owning component's private
# state.
# pylint: disable=protected-access
# This public backend/layout identifier is established API rather than an uppercase
# scalar constant.
# pylint: disable=invalid-name


import copy
import json
import html
import math
import re
from xml.etree import ElementTree as ET
from pathlib import Path
from wtf.config import load, validate
from wtf.simulation.world import World, Animal

OUT = Path("docs/assets/physics")
OUT.mkdir(parents=True, exist_ok=True)
c = load("environments/enclosed_valleys.json")


def make_world(rows, legend, species, position, health=None):
    """Construct the controlled world used by the physics figures."""

    # Local fixture/geometry names describe a separate scope and do not consume the
    # outer binding.
    # pylint: disable=redefined-outer-name

    cfg = copy.deepcopy(c)
    cfg.pop("placement", None)
    cfg["world"].update(width=len(rows[0]), height=len(rows), max_steps=20)
    cfg["terrain"] = {
        "map_version": 1,
        "name": "physics_example",
        "width": len(rows[0]),
        "height": len(rows),
        "rows": rows,
        "legend": legend,
    }
    cfg["resources"]["patch_count"] = 0
    for name, s in cfg["species"].items():
        s["initial_population"] = int(name == species)
        s["reproduction"]["enabled"] = False
    w = World(validate(cfg), 100)
    b = cfg["species"][species]["body"]
    w.animals = {
        0: Animal(
            0,
            species,
            *position,
            1,
            b["initial_energy"],
            b["max_health"] if health is None else health,
        )
    }
    w.next_animal_id = 1
    return w


def collect(w, action, n):
    """Collect the controlled observations used by a physics example."""

    # Local fixture/geometry names describe a separate scope and do not consume the
    # outer binding.
    # pylint: disable=redefined-outer-name

    states = []
    body = w.animals[0]
    for tick in range(n + 1):
        a = w.animals.get(0, body)
        states.append(
            {
                "tick": tick,
                "x": a.x,
                "y": a.y,
                "energy": a.energy,
                "health": a.health,
                "dx": a.drift_x,
                "dy": a.drift_y,
                "alive": 0 in w.animals,
                "events": [
                    {"kind": e.kind, "cause": e.cause, "amount": e.amount}
                    for e in w.events
                ],
            }
        )
        if tick < n:
            w.step({0: action} if 0 in w.animals else {})
    return states


data = {
    "species": c["species"],
    "valley": c["terrain"],
    "swim": {},
    "elevation": {},
    "mating": {},
}
for name in c["species"]:
    data["swim"][name] = {}
    data["elevation"][name] = {}
    data["mating"][name] = {}
    rows = [".........~~~~......."] * 12
    legend = {".": {}, "~": {"water_depth": 1, "flow": [0.4, 0.75]}}
    for action in ("move_forward", "swim", "rest"):
        for weak in (False, True):
            start = (10, 3) if action == "rest" else (8, 3)
            w = make_world(rows, legend, name, start, 10 if weak else None)
            data["swim"][name][action + ("_weak" if weak else "")] = collect(
                w, action, 8
            )
    data["swimRows"] = rows
    data["swimLegend"] = legend
    for gain in (1, 2, -1):
        left, right = (1, 0) if gain < 0 else (0, gain)
        rows = ["a" * 8 + "b" * 12] * 10
        legend = {"a": {"elevation": left}, "b": {"elevation": right}}
        w = make_world(rows, legend, name, (7, 5))
        data["elevation"][name][str(gain)] = {
            "rows": rows,
            "legend": legend,
            "states": collect(w, "move_forward", 1),
        }
    for distance in range(1, 5):
        for both in (False, True):
            w = make_world(["." * 20] * 11, {".": {}}, name, (7, 5))
            spec = w.config["species"][name]
            spec["reproduction"]["enabled"] = True
            a = w.animals[0]
            a.sex = "female"
            a.age = 600
            b = spec["body"]
            w.animals[1] = Animal(
                1,
                name,
                7 + distance,
                5,
                3,
                b["initial_energy"],
                b["max_health"],
                age=600,
                sex="male",
            )
            w.next_animal_id = 2
            w.step({0: "reproduce", 1: "reproduce" if both else "rest"})
            births = [e for e in w.events if e.kind == "born"]
            data["mating"][name][f"{distance}_{int(both)}"] = {
                "born": bool(births),
                "child": vars(w.animals[births[0].target]) if births else None,
                "energies": [w.animals[i].energy for i in (0, 1)],
                "rewards": [w.animals[i].last_reward for i in (0, 1)],
            }

fragment = OUT / "explorer.fragment.html"
old_fragment = fragment.read_text()
new_fragment = re.sub(
    r"const DATA=.*?;\nconst root",
    lambda _: "const DATA=" + json.dumps(data, separators=(",", ":")) + ";\nconst root",
    old_fragment,
    count=1,
    flags=re.S,
)
assert new_fragment != old_fragment or "const DATA=" in new_fragment
fragment.write_text(new_fragment)
export = OUT / "explorer.html"
if export.exists():
    text = export.read_text()
    before = html.escape(old_fragment)
    assert before in text, "Re-export the standalone wrapper from the fragment first"
    text = text.replace(before, html.escape(new_fragment), 1)
    # These optional tooltip/icon libraries are unused by this explorer.
    text = re.sub(
        r"&lt;script\b(?:(?!&gt;).)*?\bsrc=&quot;https://unpkg\.com/.*?&gt;&lt;/script&gt;\n?",
        "",
        text,
    )
    export.write_text(text)

CSS = """svg{--bg:#ffffff;--fg:#1e293b;--rule:#d2d8df;--water:#2674ac;--normal:#25654d;--extra:#a26800;--prey:#25654d;--pred:#b04051;background:var(--bg);color:var(--fg);font:18px system-ui,sans-serif}text{fill:var(--fg)}.grid{stroke:var(--rule);stroke-width:.5}.wall{fill:var(--fg);fill-opacity:.5}.water{fill:var(--water);fill-opacity:.2}.land{fill:var(--fg);fill-opacity:.06}.normal{fill:var(--normal);fill-opacity:.14}.extra{fill:var(--extra);fill-opacity:.35}.title{font-size:24px;font-weight:500}.subtitle{font-size:18px}.ray{stroke:var(--fg);stroke-width:2;fill:none;stroke-dasharray:4 4}.trail{stroke:var(--normal);stroke-width:3;fill:none}.animal{fill:var(--bg);stroke:var(--fg);stroke-width:2}.food{fill:var(--normal)}@media(prefers-color-scheme:dark){svg{--bg:#171b22;--fg:#e4e8ee;--rule:#424952;--water:#74b7e8;--normal:#72c6a6;--extra:#e8b85d;--prey:#72c6a6;--pred:#f29ca8}}"""


def txt(x, y, text, cls="", anchor="start"):
    """Emit a text element for the physics figure."""

    # Local fixture/geometry names describe a separate scope and do not consume the
    # outer binding.
    # pylint: disable=redefined-outer-name

    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{html.escape(str(text))}</text>'


def map_svg(
    m, view, cell, observer=None, target=None, sight=None, trail=None, markers=None
):
    """Render the controlled grid and its visibility annotations as SVG."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-positional-arguments
    # Local fixture/geometry names describe a separate scope and do not consume the
    # outer binding.
    # pylint: disable=redefined-outer-name
    # This internal collaborator intentionally inspects the owning component's private
    # state.
    # pylint: disable=protected-access

    x0, y0, cols, rows = view
    out = []

    def xy(p):

        # Local fixture/geometry names describe a separate scope and do not consume
        # the outer binding.
        # pylint: disable=redefined-outer-name

        return ((p[0] - x0 + 0.5) * cell, (p[1] - y0 + 0.5) * cell)

    for y in range(y0, y0 + rows):
        for x in range(x0, x0 + cols):
            t = m["legend"][m["rows"][y][x]]
            px = (x - x0) * cell
            py = (y - y0) * cell
            cls = (
                "wall"
                if t.get("blocked")
                else (
                    "water"
                    if t.get("water_depth")
                    else "land" if t.get("elevation") else ""
                )
            )
            if cls:
                out.append(
                    f'<rect x="{px}" y="{py}" width="{cell}" height="{cell}" class="{cls}"/>'
                )
            if sight and not t.get("blocked"):
                w, base, bonus = sight
                a = observer
                p = (x, y)
                distance = math.dist(a, p)
                ok = w.visible(a, p)
                if ok and distance <= base:
                    cls = "normal"
                elif ok and distance <= base + min(bonus, w._water_sight_cells(a, p)):
                    cls = "extra"
                else:
                    cls = ""
                if cls:
                    out.append(
                        f'<rect x="{px}" y="{py}" width="{cell}" height="{cell}" class="{cls}"/>'
                    )
    for x in range(cols + 1):
        out.append(f'<path class="grid" d="M{x*cell} 0V{rows*cell}"/>')
    for y in range(rows + 1):
        out.append(f'<path class="grid" d="M0 {y*cell}H{cols*cell}"/>')
    if trail:
        pts = " ".join(f"{xy(p)[0]},{xy(p)[1]}" for p in trail)
        out.append(f'<polyline class="trail" points="{pts}"/>')
    if observer and target:
        a, b = xy(observer), xy(target)
        out.append(f'<path class="ray" d="M{a[0]} {a[1]}L{b[0]} {b[1]}"/>')
    marks = list(markers or [])
    if observer:
        marks.append((observer, "A"))
    if target:
        marks.append((target, "B"))
    for p, label in marks:
        x, y = xy(p)
        out.append(f'<circle cx="{x}" cy="{y}" r="10" class="animal"/>')
        out.append(txt(x, y + 6, label, anchor="middle"))
    return "".join(out)


def figure(name, title, panels, columns=None):
    """Assemble the documented multi-panel physics figure."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals
    # Local fixture/geometry names describe a separate scope and do not consume the
    # outer binding.
    # pylint: disable=redefined-outer-name

    columns = columns or min(3, len(panels))
    width = 1120
    gap = 22
    pw = (width - gap * (columns + 1)) / columns
    ph = 340
    height = 100 + math.ceil(len(panels) / columns) * ph
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}"><title>{html.escape(title)}</title><style>{CSS}</style>',
        txt(22, 36, title, "title"),
        txt(
            22,
            68,
            "Exact simulator rules · example positions unless labelled actual terrain",
            "subtitle",
        ),
    ]
    for i, (label, m, view, args, notes) in enumerate(panels):
        x = gap + (i % columns) * (pw + gap)
        y = 100 + (i // columns) * ph
        cell = min(22, (pw - 8) / view[2], 230 / view[3])
        out.append(f'<g transform="translate({x},{y})">')
        out.append(txt(0, 0, label))
        out.append(f'<g transform="translate(0,18)">{map_svg(m,view,cell,**args)}</g>')
        for j, note in enumerate(notes):
            out.append(txt(0, 265 + j * 25, note, "subtitle"))
        out.append("</g>")
    if name.startswith("visibility-"):
        out.append(
            txt(
                22,
                height - 16,
                "Normal sight: green · added sight: amber · water: blue · walls: dark · A: observer · B: target",
            )
        )
    out.append("</svg>")
    p = OUT / (name + ".svg")
    p.write_text("".join(out))
    ET.parse(p)


rows = [
    "".join(
        (
            "~"
            if 10 <= x <= 12
            else (
                "#"
                if x == 14 and (y <= 4 or y >= 8)
                else "h" if x < 5 and y < 4 else "."
            )
        )
        for x in range(23)
    )
    for y in range(13)
]
legend = {
    ".": {},
    "~": {"water_depth": 1},
    "#": {"blocked": True},
    "h": {"elevation": 3},
}
w = make_world(rows, legend, "prey", (7, 6))
w._observation_context()
example = w.config["terrain"]
view = (0, 0, 23, 13)
figure(
    "visibility-water",
    "Water extends sight only along the target ray",
    [
        (
            label,
            example,
            view,
            {"observer": (7, 6), "target": (16, 6), "sight": (w, 6, bonus)},
            notes,
        )
        for label, bonus, notes in [
            (
                "Before: radius 6",
                0,
                ["Distance 9 · hidden", "3 intervening water cells"],
            ),
            (
                "After: radius up to 9",
                3,
                ["Distance 9 · visible", "Normal sight + water-only extension"],
            ),
        ]
    ],
    2,
)
figure(
    "visibility-limits",
    "Dry directions, walls and height",
    [
        (
            "Dry direction",
            example,
            view,
            {"observer": (7, 6), "target": (0, 6), "sight": (w, 6, 3)},
            ["Distance 7 · no water · hidden", "Water elsewhere gives no bonus"],
        ),
        (
            "Wall occlusion",
            example,
            view,
            {"observer": (7, 6), "target": (15, 3), "sight": (w, 6, 3)},
            ["Distance 8.54 · water crossed", "Wall blocks the target"],
        ),
        (
            "High ground",
            example,
            view,
            {"observer": (7, 6), "target": (4, 3), "sight": (w, 6, 3)},
            ["Distance 4.24 · visible", "Height does not block sight"],
        ),
    ],
)
figure(
    "visibility-species",
    "Species keep different normal sight ranges",
    [
        (
            label,
            example,
            view,
            {"observer": (7, 6), "target": (17, 6), "sight": (w, base, 3)},
            [f"Distance 10 · normal {base} · max {base+3}", result],
        )
        for label, base, result in [
            ("Prey", 6, "Hidden: beyond maximum 9"),
            ("Predator", 8, "Visible across three water cells"),
        ]
    ],
    2,
)
actual = World(c, 100)
actual._observation_context()
figure(
    "visibility-valley",
    "Actual valley: one water cell grants one extra cell",
    [
        (
            label,
            c["terrain"],
            (0, 9, 20, 9),
            {"observer": (10, 13), "target": (3, 13), "sight": (actual, 6, bonus)},
            notes,
        )
        for label, bonus, notes in [
            (
                "Before: prey radius 6",
                0,
                ["Distance 7 · hidden", "Left gate crop; example animals"],
            ),
            (
                "After: effective radius 7",
                3,
                ["Distance 7 · visible", "One water cell, not an automatic +3"],
            ),
        ]
    ],
    2,
)
water_map = {"rows": data["swimRows"], "legend": data["swimLegend"]}
wv = (5, 1, 12, 9)


def water_panel(key, tick, label):
    """Render the water visibility and movement example panel."""

    # Local fixture/geometry names describe a separate scope and do not consume the
    # outer binding.
    # pylint: disable=redefined-outer-name

    states = data["swim"]["predator"][key]
    s = states[tick]
    return (
        label,
        water_map,
        wv,
        {
            "observer": (s["x"], s["y"]),
            "trail": [(a["x"], a["y"]) for a in states[: tick + 1]],
        },
        [
            f"Energy {s['energy']:.3f} · HP {s['health']:.0f}",
            f"Carry ({s['dx']:.2f}, {s['dy']:.2f})",
        ],
    )


figure(
    "swimming",
    "Water entry requires a swim choice and enough health",
    [
        water_panel("move_forward", 1, "Walk: entry blocked"),
        water_panel("swim", 1, "Swim: entry succeeds"),
        water_panel("swim_weak", 1, "10 HP: swim entry blocked"),
    ],
)
figure(
    "currents",
    "Current moves waterborne animals after their action",
    [
        water_panel("swim", 0, "Step 0: facing water"),
        water_panel("swim", 2, "Step 2: swim + drift"),
        water_panel("swim", 4, "Step 4: dry landing"),
    ],
)
figure(
    "drowning",
    "Resting in water still costs energy and can cause drowning",
    [
        water_panel("rest_weak", 0, "Step 0: 10 HP in water"),
        water_panel("rest_weak", 3, "Step 3: 4 HP remaining"),
        water_panel("rest_weak", 5, "Step 5: drowning death"),
    ],
)
height_panels = []
for gain, label in [
    (1, "Climb +1 succeeds"),
    (2, "Cliff +2 is blocked"),
    (-1, "Downhill has no extra cost"),
]:
    e = data["elevation"]["predator"][str(gain)]
    s = e["states"][1]
    cost = e["states"][0]["energy"] - s["energy"]
    height_panels.append(
        (
            label,
            e,
            (3, 1, 12, 8),
            {
                "observer": (s["x"], s["y"]),
                "trail": [(a["x"], a["y"]) for a in e["states"]],
            },
            [f"Total energy cost {cost:.3f}", f"HP unchanged at {s['health']:.0f}"],
        )
    )
figure("elevation", "Elevation changes movement cost, not visibility", height_panels)
plain = {"rows": ["." * 20] * 11, "legend": {".": {}}}
birth_panels = []
for d, both, label in [
    (2, 1, "Near + both choose: birth"),
    (2, 0, "Near + one choice: no birth"),
    (3, 1, "Too far: no birth"),
]:
    e = data["mating"]["prey"][f"{d}_{both}"]
    markers = [((7, 5), "F"), ((7 + d, 5), "M")]
    if e["child"]:
        markers.append(((e["child"]["x"], e["child"]["y"]), "C"))
    birth_panels.append(
        (
            label,
            plain,
            (4, 2, 10, 7),
            {"markers": markers},
            [
                f"Parent energy {e['energies'][0]:.2f} / {e['energies'][1]:.2f}",
                "Mature, healthy parents · radius 2",
            ],
        )
    )
figure(
    "reproduction",
    "Birth is a mutual nearby decision; no birth-specific HP cost",
    birth_panels,
)
# Full actual founder/resource layout; no trained controllers or saved run involved.
layout = map_svg(c["terrain"], (0, 0, 40, 28), 22)
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="780" viewBox="0 0 1000 780" role="img" aria-label="Actual enclosed-valley initial layout, seed 100"><title>Actual enclosed-valley initial layout, seed 100</title><style>{CSS}</style>',
    txt(28, 36, "Actual enclosed valley · founder layout · seed 100", "title"),
    txt(28, 66, "48 prey inside · 12 predators outside · two three-cell gates"),
    '<g transform="translate(45,94)">',
    layout,
]
for x, y in actual.resources:
    px, py = (x + 0.5) * 22, (y + 0.5) * 22
    parts.append(f'<path class="food" d="M{px} {py-4}l4 8h-8z"/>')
for a in actual.animals.values():
    px, py = (a.x + 0.5) * 22, (a.y + 0.5) * 22
    role = "prey" if actual.diet(a) == "resource" else "pred"
    parts.append(
        f'<circle cx="{px}" cy="{py}" r="5" style="fill:var(--{role})"/>'
        if role == "prey"
        else f'<rect x="{px-5}" y="{py-5}" width="10" height="10" style="fill:var(--pred)"/>'
    )
parts.extend(
    [
        "</g>",
        txt(
            28,
            745,
            "Circle: prey · square: predator · triangle: food · 30 inner + 30 outer boundary patches, 20 farther away",
        ),
        "</svg>",
    ]
)
(OUT / "valley-layout.svg").write_text("".join(parts))
for p in OUT.glob("*.svg"):
    ET.parse(p)
print(
    f'Generated {len(list(OUT.glob("*.svg")))} SVG figures and refreshed interactive scenario snapshots.'
)
