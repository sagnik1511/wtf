# Landscape map diagrams

[Geography guide](../../geography.md#landscape-inspired-maps) · [Asset index](../README.md)

The ten SVGs in this directory are generated from the actual editable
`maps/*.json` definitions selected by their corresponding environment presets.
They show dry land, elevation, blocked rock, water depth and sampled current
arrows. Water/elevation colors follow the viewer's terrain color formulas.
The background, grid and explanatory labels are documentation styling.

These are static map diagrams, not full-viewer screenshots or recordings.
They contain no animal positions, food placements, learned behavior or results.
No image-generation service or external geographic dataset was used.

Regenerate from the project root with no extra packages or training:

```sh
.venv/bin/python -m work.build_geography_previews
```

The helper reads its ten named environment presets, resolves and validates their
map references, and overwrites only those SVG assets. Use `--check` to verify
that saved previews match the current maps without writing files. Both light and
dark palettes preserve terrain/obstacle/current distinctions. Edit the map JSON first;
do not edit generated SVG cells by hand. The map JSON remains the source of
truth for both training and these diagrams. Replays embed their own map copy.
