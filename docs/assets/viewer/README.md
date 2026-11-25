# Viewer appearance assets

These controlled vector sheets support the [viewer appearance reference](../../viewer-visual-reference.md).
They contain no run IDs, measurements, recordings or saved model weights.

| Asset | Contents |
| --- | --- |
| [Interface themes](themes.svg) | Schematic light/dark UI specimens from the actual CSS palette tokens |
| [Day and night](day-night.svg) | Same objects under actual dawn/midday/dusk/night lighting and a deterministic sunlight curve |
| [Collective learning](social-learning.svg) | Actual inspector charts for pooled local outcomes, species consequence estimates and unknown evidence |
| [Object shadows](object-shadows.svg) | Morning/midday/evening/night projections and fixed reduced-motion footprints |
| [River and feeding phases](river-feeding.svg) | Saved-current ripples and bounded food-gain cues, derived from existing events |
| [Resource/obstacle phases](environment-objects.svg) | Quantity-driven plants, fixed faceted rocks, replay-tick sway/shimmer and reduced-motion snapshots |
| [Board objects](board-objects.svg) | Sex/species shapes and colors, resource quantities, obstacles, selection, heading, event markers and same-species competition |
| [Terrain and sight](terrain-and-sight.svg) | Water depth/current, elevation, occlusion, directional water visibility and gates |
| [Movement and scent](tracking.svg) | Actual sampled-history and blended anonymous scent heatmap components |
| [Inspector and brains](brain-views.svg) | Actual radar/output/network SVG components with illustrative data; deep-view summary |
| [Workspace map](workspace-map.svg) | Schematic panel positions with source-rendered board/radar examples |

## Sources and regeneration

From the project root, using Node (no external packages):

```sh
node work/build_viewer_visuals.mjs
node work/build_viewer_visuals.mjs --check
```

The [generator](../../../work/build_viewer_visuals.mjs) executes isolated board,
birth, death and competition drawing functions from `viewer/app.js` through a minimal SVG
canvas adapter. It uses the actual species palette, terrain renderer and
observation overlay. The inspector radar, archive output chart and small-brain
structure come from their actual JavaScript SVG components through a small DOM
adapter. Checks verify expected component structure, palette assumptions and
byte-for-byte regeneration. No simulation or browser is started.

Board scenes use enlarged cells. Events freeze age zero and omit canvas shadow
glow. These sheets are identification references, not pixel-exact screenshots
or simulation results. Workspace boxes and the deep-brain description are
schematic. All feature values, probabilities, activations and chart lines are
explicitly illustrative; they do not describe learned behavior. Keep local PNG
previews in ignored `.cache/`, and track only the intended SVG exports here.

When drawing functions, palette, inspector components or controls change, update
these specimens and their captions together. The check detects source-rendered
asset drift; schematic labels and layout still require review against the UI.

Theme CSS must belong to the outer SVG viewport. These sheets contain nested SVG
components, so appending a stylesheet at the first `</svg>` themes only that
component. The exporter attaches the palette before the final root closing tag;
a regression check rejects theme styles scoped to nested viewports. Validate the
complete sheet in both site palettes, not just a component or the CSS source.

## Animated vector demonstrations

The generator also uses [animation composition](../../../work/build_doc_animations.mjs)
to build eight four-second SVG loops, with eight discrete illustrative states:

| Asset | Source and meaning |
| --- | --- |
| [World playback](animation-simulation-playback.svg) | Current board and selected sight, authored positions |
| [Species metrics](animation-species-metrics.svg) | Schematic cumulative bars for three example species |
| [Day–night](animation-day-night.svg) | Current board lighting at authored cycle phases |
| [Animal brain](animation-animal-brain.svg) | Current inspector/network components, illustrative samples |
| [Archived brain](animation-archived-brain.svg) | Current explanation charts, illustrative comparison data |
| [River crossing](animation-river-crossing.svg) | Controlled small river map |
| [Highlands](animation-highlands.svg) | Controlled small elevation map |
| [Enclosed valleys](animation-enclosed-valleys.svg) | Controlled small enclosure and gate |

The label is inside each image so it survives direct sharing. These do not
claim to show real inference, learned trajectories, an exact preset map or the
whole interface. Archive chart values are authored independently of weights.
One SVG contains both board palettes; the outer viewport owns adaptive theme
styles. CSS selects one state at a time, and reduced motion disables the loop
and reveals the first state. No script, embedded raster or external font is used.
The site's `color-scheme` connects embedded SVGs to its palette toggle; standalone
images use the browser preference. Hosts that disable animation can show the
static first state. Regenerate and check with the commands above.
