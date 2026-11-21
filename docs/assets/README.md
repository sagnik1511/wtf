# Documentation assets

[All visual files](../visuals.md) · [Homepage click paths](../access-map.md)

These intentional documentation assets are tracked in Git; experiment results
remain ignored and original videos stay outside the repository.

## Light/dark preview behavior

Start at **Home → All visuals → visual name** for an embedded preview with the
MkDocs palette toggle. The original-file column keeps the exact source asset
available separately. Opening a raw file leaves the site's theme controls.

| Asset kind | Preview behavior |
| --- | --- |
| Structural, physics, map and inspector SVGs | Adaptive palettes follow the site toggle |
| Theme/daylight comparison sheets | Outer labels/backgrounds adapt; explicit comparison samples keep their colors |
| Playback GIFs and light WebP variants | Select the matching recorded export, without color inversion |
| Logo SVG/PNG variants | Select the matching light/dark artwork; favicon PNG has a contrasting presentation surface |
| Full physics explorer / editable fragment | Embed the complete interactive explorer with inherited light/dark mode; the fragment remains an editable-source download |
| GitHub banner PNG | Fixed screenshot shown unchanged, with its purpose stated explicitly |

Preview pages are generated during the MkDocs build and reuse the original media.
The catalog does not eagerly embed every image or animation. See the
[complete click sequences](../access-map.md) for each page, file and runtime copy.

## Project identity

The [branding guide](../branding.md) links the transparent SVG symbols, full
lockups and PNG fallbacks, including light/dark variants and usage rules.

## Current playback clips

The playback previews are exported from the current viewer using saved replay
frames, rather than trimmed from an older desktop video. Each preserves the
whole viewer window (1280 × 720, or 1280 × 1440 for the taller XAI dialogs),
including its header and controls; there is no
board-only crop. The metrics clip opens the **Metrics dashboard**, while the
main playback clip selects an animal to show its sight and inspector.

| Asset | View | Encoding |
| --- | --- | --- |
| [simulation-playback.gif](simulation-playback.gif) | Playback with selected-animal inspector | 3.5 s, 14 frames, 4 fps, 128 colors |
| [species-metrics.gif](species-metrics.gif) | Per-species dashboard following the replay position | 3.5 s, 14 frames, 4 fps, 128 colors |
| [day-night.gif](day-night.gif) | Sunlight transitions and moving time dial | 3.5 s, 14 frames, 4 fps, 128 colors |
| [animal-brain.gif](animal-brain.gif) | Selected animal’s actual sampled inputs, outputs and network activity | 3.5 s, 14 frames, 4 fps, 128 colors |
| [archived-brain.gif](archived-brain.gif) | Fixed archived brain evaluated on changing replay observations | 3.5 s, 14 frames, 4 fps, 128 colors |

Ten simulation steps separate exported frames (20 for day–night), so these are time-compressed
previews rather than real-time screen recordings. Chart history stops at the
playhead. The small demonstration replays illustrate the interface and do not
establish learning quality. Source frames and temporary replays stay outside
Git; only optimized GIFs and [the provenance manifest](playback-manifest.json)
are tracked, along with the paired light-theme WebP animations under `themed/`.
Pages selects the export using its theme toggle; GitHub uses picture media.
See [the export workflow](terrains/README.md#refresh-the-clips).

## Physics diagrams and interactive explorer

[Physics asset catalog](physics/README.md) covers ten static SVG diagrams and the
standalone explorer, with provenance and regeneration commands. The
[illustrated guide](../physics.md) presents the cases in context. These contain
controlled physics snapshots and no training results or model weights.

## Full-viewer terrain previews

[Landscape map diagrams](geography/README.md) show four additional maps directly
from their JSON: a meandering river, delta, alpine passes and lake islands. These
small vector diagrams have a separate provenance from the recorded GIFs below.

[Terrain previews](terrains/README.md) documents three 3.5-second GIFs exported
from the current viewer. The [geography gallery](../geography.md#recorded-terrain-previews)
embeds them. Like the main clips, these retain the full viewer window and are
sampled from replay steps rather than captured desktop video.

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

[Source-rendered viewer specimens](viewer/README.md) identify animal/sex glyphs, resources, events, terrain, visibility and inspector/brain graphics. The [appearance reference](../viewer-visual-reference.md) embeds the six sheets with their meaning; chart values are illustrative.
