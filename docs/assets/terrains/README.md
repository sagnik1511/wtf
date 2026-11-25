# Animated terrain demonstrations

[Geography gallery](../../geography.md#recorded-terrain-previews) · [Asset index](../README.md)

| Vector demonstration | What it illustrates |
| --- | --- |
| [River crossing](../viewer/animation-river-crossing.svg) | Water, flow, selected sight and authored movement |
| [Highlands](../viewer/animation-highlands.svg) | Elevation shading and authored movement |
| [Enclosed valleys](../viewer/animation-enclosed-valleys.svg) | Walls, a gate and an authored route through the opening |

These are small controlled example maps, not exact preset maps, whole-viewer
captures or simulation results. They use the current board renderer. Paths are
authored to explain appearance, not computed policy decisions or physics tests.
Both light/dark palettes and reduced motion are handled inside each SVG.

## Refresh the clips

Run `node work/build_viewer_visuals.mjs`, then the same command with `--check`.
See [the generator contract](../viewer/README.md#animated-vector-demonstrations).
No experiment, saved replay, browser recording or FFmpeg conversion is required.
The older GIF/WebP exports and manifests have been retired from the published
documentation; the legacy export tools remain optional utilities for real captures.
