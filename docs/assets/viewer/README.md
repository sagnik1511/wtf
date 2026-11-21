# Viewer appearance assets

These controlled vector sheets support the [viewer appearance reference](../../viewer-visual-reference.md).
They contain no run IDs, measurements, recordings or saved model weights.

| Asset | Contents |
| --- | --- |
| [Board objects](board-objects.svg) | Sex/species shapes and colors, resource quantities, obstacles, selection, heading and event markers |
| [Terrain and sight](terrain-and-sight.svg) | Water depth/current, elevation, occlusion, directional water visibility and gates |
| [Inspector and brains](brain-views.svg) | Actual radar/output/network SVG components with illustrative data; deep-view summary |
| [Workspace map](workspace-map.svg) | Schematic panel positions with source-rendered board/radar examples |

## Sources and regeneration

From the project root, using Node (no external packages):

```sh
node work/build_viewer_visuals.mjs
node work/build_viewer_visuals.mjs --check
```

The [generator](../../../work/build_viewer_visuals.mjs) executes isolated board,
birth and death drawing functions from `viewer/app.js` through a minimal SVG
canvas adapter. It uses the actual species palette, terrain renderer and
observation overlay. The inspector radar, archive output chart and small-brain
structure come from their actual JavaScript SVG components through a small DOM
adapter. Checks verify expected component structure, palette assumptions and
byte-for-byte regeneration. No simulation or browser is started.

Board scenes use enlarged cells. Events freeze age zero and omit canvas shadow
glow. These sheets are identification references, not pixel-exact screenshots
or animation demonstrations. Workspace boxes and the deep-brain description are
schematic. All feature values, probabilities, activations and chart lines are
explicitly illustrative; they do not describe learned behavior. Keep local PNG
previews in ignored `.cache/`, and track only the intended SVG exports here.

When drawing functions, palette, inspector components or controls change, update
these specimens and their captions together. The check detects source-rendered
asset drift; schematic labels and layout still require review against the UI.
