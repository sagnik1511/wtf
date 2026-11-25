# Documentation assets

[All visual files](../visuals.md) · [Homepage click paths](../access-map.md)

These intentional documentation assets are tracked in Git; experiment results
remain ignored and original videos stay outside the repository.

## Light/dark preview behavior

Start at **Home → All visuals → visual name** for an embedded preview with the
Zensical palette toggle. The original-file column keeps the exact source asset
available separately. Opening a raw file leaves the site's theme controls.

| Asset kind | Preview behavior |
| --- | --- |
| Structural, physics, map and inspector SVGs | Adaptive palettes follow the site toggle |
| Theme/daylight comparison sheets | Day/night boards use the selected palette; interface comparison puts the selected theme first, alongside the labeled alternate |
| Animated vector demonstrations | One SVG adapts to the site theme; reduced motion freezes the first state |
| Logo SVG/PNG variants | Select the matching light/dark artwork; favicon PNG has a contrasting presentation surface |
| Full physics explorer / editable fragment | Embed the complete interactive explorer with inherited light/dark mode; the fragment remains an editable-source download |
| GitHub banner PNG | Fixed screenshot shown unchanged, with its purpose stated explicitly |

Preview pages are generated during the Zensical build and reuse the original media.
The catalog does not eagerly embed every image or animation. See the
[complete click sequences](../access-map.md) for each page, file and runtime copy.

## Project identity

The [branding guide](../branding.md) links the transparent SVG symbols, full
lockups and PNG fallbacks, including light/dark variants and usage rules.

## Animated playback demonstrations

Eight [animated SVG demonstrations](viewer/README.md#animated-vector-demonstrations)
replace the older raster clips: world playback, per-species metrics, day–night,
selected-animal brain activity, archived-model comparison, river crossing,
highlands and enclosed valleys. They are not experiment recordings or full UI
captures. Current board/inspector components render illustrative states; metrics
use schematic bars. One file supports both themes without color inversion or
embedded raster images. See the [terrain catalog](terrains/README.md).

## Physics diagrams and interactive explorer

[Physics asset catalog](physics/README.md) covers ten static SVG diagrams and the
standalone explorer, with provenance and regeneration commands. The
[illustrated guide](../physics.md) presents the cases in context. These contain
controlled physics snapshots and no training results or model weights.

## Animated terrain previews

[Landscape map diagrams](geography/README.md) show four additional maps directly
from their JSON: a meandering river, delta, alpine passes and lake islands. These
small vector diagrams use actual preset maps; the animations use controlled example maps.

[Terrain previews](terrains/README.md) documents three animated SVG examples.
The [geography gallery](../geography.md#recorded-terrain-previews) embeds them;
the section anchor is retained for existing links.

## Software and learning flows

[Flow diagram sources and SVGs](diagrams/README.md) cover simulation, private
learning, inheritance, saved state, storage, analysis and maintenance. The
[visual guide](../visual-guide.md) maps questions to these diagrams. All exports
are tracked vector assets; editable flowchart sources are local-only and ignored.
No new GIFs or recordings were needed for these conceptual explanations.

## Agent inspector example

The [inspector illustration](inspector/README.md) shows current sight alongside
older sampled inputs and action bars. Its provenance distinguishes controlled
observation values from invented example probabilities. No new recording is used.

## Playback data structures

The [storage illustration catalog](storage/README.md) covers the experiment
artifact map, binary chunk record layout, worked frame-delta/mask example and
model-weight reference/parameter structure. It also includes checkpoint file,
controller-state and restoration/access diagrams; see the
[checkpoint guide](../checkpoint-data-structures.md).
The [structure guide](../replay-data-structures.md) embeds them beside their
schemas and field tables. These are controlled code-verified illustrations.

## Selected runtime structures

The [runtime structure catalog](structures/README.md) covers observation-vector
packing, overlapping founder families, observation-window bucket references,
actual sampled animal activity, archived-brain analysis ownership and a worked
activation/sensitivity example.
These illustrations were selected for structural clarity; the
[visual guidance](../visual-guide.md#when-a-visual-helps) records where existing
figures, tables and prose remain the better choice.

## Viewer appearance catalog

[Source-rendered viewer specimens](viewer/README.md) identify animal/sex glyphs, resources, events, terrain, visibility and inspector/brain graphics. The [appearance reference](../viewer-visual-reference.md) embeds the seven sheets with their meaning; chart values are illustrative.

## CUDA architecture diagrams

[CUDA data flow and state residency](../cuda.md#where-the-cuda-path-saves-work)
show CPU/GPU ownership, exact-content upload caches, compact gradient reduction
and private PPO optimizer state across minibatches. These adaptive SVGs describe
mechanisms rather than attributing measured time savings. Sources live in
`work/build_recent_doc_diagrams.py`; regenerate with
`python -m work.build_recent_doc_diagrams --only cuda-dqn-dataflow --only cuda-ppo-residency`.
Local `.mmd` exports remain ignored.
