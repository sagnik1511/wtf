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
