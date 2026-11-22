# Board objects and viewer guide

[Playback and inspection](viewer/playback/index.md) · [Viewer and storage](viewer/index.md)

[Playback guide](recording.md) · [Visual guide](visual-guide.md) · [Asset sources](assets/viewer/README.md)

Use this catalog to identify board objects, overlays and inspector graphics.
The examples are controlled specimens rendered through current viewer code.
The workspace map is schematic; chart values are illustrative, not experiment
results. Existing [full-viewer clips](geography.md#recorded-terrain-previews)
show the interface in motion.

## Find a viewer element

| Looking for | Section |
| --- | --- |
| Animal shapes, female/male colors, food, obstacles and event markers | [Board pieces and their meaning](#animals-resources-and-event-markers) |
| Rivers, elevation and the selected animal's visible area | [Terrain and current sight](#terrain-and-current-sight) |
| Sunlight and the time dial | [Day and night](#day-and-night) |
| Light/dark interface colors | [Interface themes](#interface-themes) |
| Board, sidebar, timeline and metrics placement | [Workspace and charts](#workspace-and-charts) |
| Observation circle, predictions and network diagrams | [Inspector and brain views](#inspector-and-brain-views) |
| Header logo and loading animation | [Project header](#project-header) |

The figures are embedded below. Open this guide from **Viewer and storage →
Board objects and viewer guide** in the documentation navigation. Use the site
light/dark toggle to change adaptive figures; comparison sheets intentionally
show several lighting conditions together.

## Animals, resources and event markers

This is the board-piece legend: animal pointers, resources, obstacles, selection
and lifecycle/competition events. [Open the full-size SVG](assets/viewer/board-objects.svg)
for a closer look. The final row shows prey/predator challenges, a shared-food
contest and a competition death using the current viewer drawing functions.

![Twenty examples of animal shapes and sex colors, resources, obstacles, selection, birth, kill, natural death, heading and competition](assets/viewer/board-objects.svg)

| Appearance | Meaning |
| --- | --- |
| Circle / triangle | Resource-eating animal / prey-eating animal, regardless of species name |
| Mint / blue circles | First resource-eating species: female / male |
| Coral / gold triangles | First prey-eating species: female / male |
| Thin light body outline | Male; independent of the larger white selection ring |
| Small dark dot | Heading, rotating with the body |
| Green square | Resource; size and opacity increase with quantity relative to capacity |
| Slate square | Blocking obstacle |
| Large white ring | Selected animal; selection drives the inspector |
| Cyan ring, plus and BORN | Birth at the child's recorded position |
| Red ring, cross and KILL | Killed animal's recorded position; hunter gets a gold ring |
| Amber paired rings and connector, winner / damage label | Optional same-species challenge at the recorded fight locations; distinct from a hunting kill |
| Green / amber ring, cross and DIED | Natural death of a resource-eater / prey-eater |

The legend uses configured species names and female/male colors. Extra species
receive generated colors; optional `display.color` and `display.male_color`
overrides are described in [species appearance](species.md#viewer-colors-and-metrics).
Legacy animals without recorded sex use the base species color; that color does
not establish their sex. Body color identifies species/sex, not remaining health.

Event illustrations freeze age zero and omit the canvas glow. In playback,
birth/death rings expand and fade, while competition rings fade at their
recorded fight locations. Banners identify recent events. They do not insert a simulation or playback delay. Marker placement
requires a recorded position; these examples do not imply inferred positions
for older recordings.

### Optional social contests

![Food, path and aggression contexts with viewer highlights](assets/diagrams/competition.svg)

The [competition guide](competition.md) explains the opt-in learned action,
physical eligibility, costs and outcomes. The contest banner names both animals,
the context and actual damage; markers persist briefly without pausing playback.
Contest and kill/death labels share a stacked overlay inside the board, leaving
its size unchanged when labels appear or disappear. Births retain their lower
board label.
Aggression needs no food or path objective and changes only the participating
animals through effort, health and resulting reward feedback.

## Terrain and current sight

![Flat land, blue river and flow arrows, elevation colors, wall-limited sight, water-directed sight and an enclosure gate](assets/viewer/terrain-and-sight.svg)

Water uses blue tiles with depth-dependent shading; sparse arrows indicate flow
direction. Arrow length is a display convention, not a speed scale. Dry elevation
colors are scaled against the map's maximum height, with sparse numeric labels.
Compare heights using their values rather than assuming the same hue means the
same height on different maps.

The translucent blue wash is the selected animal's **current** sight. Walls
occlude visibility rays and water can extend permitted directions. The board
still displays objects outside that wash: it is an inspection overlay, not
fog of war. Heading is shown by the dot and does not create a sight cone.
Elevation and other animals do not occlude sight. The
[physics guide](physics.md) explains the corresponding movement and visibility
rules; the [map gallery](geography.md) shows maintained environments.

## Day and night

![Board lighting at dawn, midday, dusk and night with the sunlight curve](assets/viewer/day-night.svg)

The example boards use the selected interface palette in both light and dark mode.
Optional [day–night cycles](day-night.md) dim the board and warm the low-light
periods. The time label follows saved world steps; night remains legible.
The sun/moon dial moves with saved frames and freezes on pause. It is hidden
when the cycle is absent or disabled. Species sight overlays and inspector costs
follow configured light effects.

## Interface themes

![Light and dark interface palettes, including panels, input fields and theme controls](assets/viewer/themes.svg)

Use **Light mode / Dark mode** in the header. The preference is saved in the
browser and applies to controls, panels, dialogs, inspector views, chart axes
and the board grid/terrain. The selected theme appears first in the comparison; the labeled alternate
remains visible beside it. The board examples include midday and night in both
interface palettes.
Inspector radars, target legends, prediction histories, action bars, neuron
activations and archived-model sensitivity/route charts use theme-aware colors.
Cached or expanded charts recolor immediately on a theme toggle without seeking
or waiting for another diagnostic sample.

It is independent of simulated sunlight; daylight lights the ground and night
dims it in either interface theme.

## Workspace and charts

![Schematic placement of recording controls, current totals, board, agent inspector, timeline, history and supporting panels](assets/viewer/workspace-map.svg)

Choose a saved recording and run before playing. The board and current totals
follow the playhead. The population/food history and **Metrics dashboard** show
values over recorded time; the separate behavior analysis summarizes a run.
Species colors follow the legend. See [dashboard metrics](recording.md#per-step-species-dashboard)
for definitions and missing-data handling.

## Inspector and brain views

![Source-rendered sampled-observation radar, action score comparison and complete small-network structure, alongside a deeper-network summary](assets/viewer/brain-views.svg)

The observation circle plots sampled target slots relative to the animal; it
is separate from the current board sight overlay. The body step can advance
between recorded brain samples, so compare the displayed sample timestamps.
Inputs are normalized brain features, not necessarily physical units.

**Watch animal brain** follows the selected animal's recorded inputs, outputs and
sampled activations during playback. **Explain saved brain** instead evaluates
an archived species/sex champion on those inputs. Colored action bars show that
archive's inference; gray comparison bars show recorded scores. Masked actions
are labeled. Value-based brains show action values rather than probabilities.

Inputs and outputs lead both views. With default **rank threshold k = 2**, a
brain with at most two hidden processing stages shows its whole structure.
Deeper brains show a summary, with detailed maps/routes under **Advanced**.
Here rank means hidden-stage count, separate from observation top-k. Green/pink
units show positive/negative activations; gray units mean unavailable activity.
Structure links show connectivity, not causal importance or weight strength.
The [brain guide](explainability.md) includes current-viewer clips of both data
sources and explains recurrent memory, weighted routes,
local input sensitivity, what-if inputs and unavailable data.

## Project header

The header uses the [project symbol](branding.md) beside readable project text.
It switches between forest/lime and pale-mint/lime with the interface theme.
The SVG is primary, with transparent PNG fallbacks; simulated day/night does
not affect header branding.

While opening a recording or selecting a run, the four sections circulate around
the fixed window and the status badge says **Loading recording…**. The symbol
returns to its static form when loading completes or fails, and remains static
through playback and frame buffering. Reduced-motion preferences disable the
ribbon animation. See [motion assets and behavior](branding.md#motion-policy).

## Movement memory and scent

![Current inspector movement-memory and species/sex scent components](assets/viewer/tracking.svg)

With the optional [tracking senses](tracking.md) enabled, select an animal to see
its sampled movement history and anonymous local scent patterns. **Tracks & scent
charts** jumps to those panels. A selected-only board overlay shows personal
visual tracks, while **Run movement heatmap** opens arrival counts through the current replay step
in Behavior & learning. Scent itself stays off the main board. Hollow markers
are last-seen memories; dashed inspector lines are velocity estimates, and gaps remain
unknown. Each labeled scent grid belongs to a species/sex channel rather than
one animal. These controlled SVG specimens come from the actual inspector
renderers and adapt to interface light/dark palettes.
