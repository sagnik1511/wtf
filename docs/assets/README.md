# Documentation assets

These intentional documentation assets are tracked in Git; experiment results
remain ignored and original videos stay outside the repository.

## Original demonstration clips

These two clips come from the project screen recording captured on
2026-11-17 at 11:48:02. The original MOV remains outside the repository.

| Asset | Source interval | Output |
| --- | --- | --- |
| `simulation-playback.gif` | 00:08–00:11.5 | 960 px wide, 12 fps, 96 colors, looping |
| `species-metrics.gif` | 00:26–00:29.5 | 880 px wide, 10 fps, 96 colors, looping |

Crops focus on the simulation board and metrics panel respectively. FFmpeg
palette generation and differential palette rendering keep the pair around
3 MB combined. GIF timing rounds the first clip to 3.58 seconds. These are viewer
demonstrations, not evidence of learning quality.

## Physics diagrams and interactive explorer

[Physics asset catalog](physics/README.md) covers ten static SVG diagrams and the
standalone explorer, with provenance and regeneration commands. The
[illustrated guide](../physics.md) presents the cases in context. These contain
controlled physics snapshots and no training results or model weights.

## Full-viewer terrain recordings

[Landscape map diagrams](geography/README.md) show four additional maps directly
from their JSON: a meandering river, delta, alpine passes and lake islands. These
small vector diagrams have a separate provenance from the recorded GIFs below.

[Recorded terrain previews](terrains/README.md) documents the three new
3.5-second GIFs, preserving the entire captured viewer screen. The
[geography gallery](../geography.md#recorded-terrain-previews) embeds them, and
[the manifest](terrains/manifest.json) records source timestamps, hashes and sizes.
Unlike the original board/metrics crops above, these have no spatial crop.

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
packing, overlapping founder families and observation-window bucket references.
These three diagrams were selected for structural clarity; the
[visual guidance](../visual-guide.md#when-a-visual-helps) records where existing
figures, tables and prose remain the better choice.
