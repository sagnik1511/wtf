# Recorded terrain previews

[Geography gallery](../../geography.md#recorded-terrain-previews) · [Asset index](../README.md)

These are clips of the **whole recorded viewer screen**, not generated diagrams
or board-only crops. The toolbar, counters, simulation board, sidebar and the
bottom chart area visible in the original capture are retained. Only time is
trimmed and the full frame resized from 3400 × 1906 to 1280 × 718.

Each GIF loops for **3.5 seconds**, with 42 frames at an average 12 fps and a
128-color palette. Playback timing is retained; reducing the source's 60 fps
samples frames without speeding up the video. The three GIFs total about 1.94 MB.

| Terrain / asset | Source recording, 2026-11-17 | Relative source interval |
| --- | --- | --- |
| [River crossing](river-crossing.gif) | `Screen Recording 2026-11-17 at 17.38.51.mov` | 00:10.0–00:13.5 |
| [Highlands](highlands.gif) | `Screen Recording 2026-11-17 at 17.40.05.mov` | 00:18.0–00:21.5 |
| [Enclosed valleys](enclosed-valleys.gif) | `Screen Recording 2026-11-17 at 17.41.08.mov` | 00:10.0–00:13.5 |

The highlands interval avoids the paused/loading opening. River/valley clips
include active movement and event highlights. All show run 1 of the corresponding
seed-100 experiments recorded for documentation. These are mid-run examples,
not initial-condition screenshots or evidence that the policies learned better.
The live metric dashboard modal was not opened in these excerpts; the ordinary
viewer sidebar and chart header remain exactly as captured.

The [manifest](manifest.json) stores source basenames and SHA-256 hashes, clip
intervals, experiment identifiers, dimensions, timing and encoded byte counts.
Original MOV files stay outside the repository; experiment results remain ignored.
No personal source-directory path is stored in the tracked assets or manifest.

## Recreate the clips

With the three original recordings available in one directory:

```sh
.venv/bin/python -m work.export_terrain_gifs --source-dir /path/to/recordings
```

The helper requires FFmpeg/ffprobe, uses per-clip palette generation and
rectangular differential rendering, and rewrites the GIFs and manifest. Temporary
palettes are cleaned up under ignored `.cache/`. No Python media packages,
training runs or model checkpoints are needed. Edit the helper's clip table to
choose different source files or intervals.
