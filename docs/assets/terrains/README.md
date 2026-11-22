# Current terrain playback previews

[Geography gallery](../../geography.md#recorded-terrain-previews) · [Asset index](../README.md)

These theme-aware previews show the **whole current viewer window**, including its project
header, toolbar, counters, board, sidebar and visible chart area. They are
assembled from exported replay frames; no desktop screen recording is needed.
The 1280 × 720 composition stays fixed throughout each clip, with no spatial
crop or fabricated UI.

Each loops for **3.5 seconds**, with 14 frames at 4 fps and a 128-color palette.
The viewer is stepped forward by ten simulation steps between images, so the
previews compress simulation time. They do not reproduce wall-clock playback
speed or every intervening animation frame.

| Asset | Maintained preset |
| --- | --- |
| [River crossing](river-crossing.gif) | `environments/river_crossing.json` |
| [Highlands](highlands.gif) | `environments/highlands.json` |
| [Enclosed valleys](enclosed-valleys.gif) | `environments/enclosed_valleys.json` |

The [terrain manifest](manifest.json) records preset/source hashes, seed,
replay sampling, dimensions, timing and encoded bytes. The
[complete playback manifest](../playback-manifest.json) also covers the main
playback, metrics, [day–night](../../day-night.md#playback-and-storage) and
[animal/archive brain views](../../explainability.md) GIFs. Source replays, checkpoints and intermediate images
stay local. These are interface demonstrations, not evidence of policy quality.

## Refresh the clips

1. Create short temporary replays from the maintained `baseline`,
   `river_crossing`, `highlands`, `enclosed_valleys` and `day_night` presets. Use one run,
   300 steps and seed 100; disable shared champion exports with `--no-library`.
   Put outputs under an ignored cache or temporary directory.
2. Open those replays in the current viewer. Use a consistent 1280 × 720 viewport,
   dark interface theme and the entire viewer window (1280 × 1440 for the XAI
   dialogs so the graphs fit). Export still frames at
   steps 0, 10, …, 130 (0, 20, …, 260 for day–night). Do not capture loading states. The main clip selects
   animal 1; the metrics clip opens the dashboard and chooses **Mean predicted
   Q value** for Chart 5, matching the baseline's Q learner. The `animal-brain`
   clip opens **Watch animal brain** for animal 1 at steps 0–130. The
   `archived-brain` clip opens **Explain saved brain** with **Follow replay
   samples** checked at steps 130–260, using the matching lifespan champion and
   **Highest predicted action**. These two views use different data sources.
   For the archive clip, allow the temporary replay to save champions: replace
   `--no-library` with `--library-dir /path/to/temporary-library` and clean that
   directory up with the replay afterward.
   Export the same samples in both viewer themes into separate light/dark roots.
3. Save each sequence as `000.jpg` through `013.jpg` in one of these folders:
   `simulation-playback/`, `species-metrics/`, `day-night/`, `animal-brain/`,
   `archived-brain/`, `river-crossing/`, `highlands/`, `enclosed-valleys/`. Each folder also needs `capture.json`:

   ```json
   {
     "steps": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
     "frames": 14,
     "duration_seconds": 3.5,
     "theme": "dark"
   }
   ```

   Use the corresponding 20-step sample list in the day–night metadata, and
   steps 130, 140, …, 260 for the archived-brain clip. Wait for archive requests
   to finish before exporting a frame; do not capture a stale/loading report.

4. Encode the sequences and refresh both manifests with FFmpeg/ffprobe installed:

   ```sh
   .venv/bin/python -m work.export_viewer_gifs --frames-dir /path/to/dark-frames \
     --light-frames-dir /path/to/light-frames
   ```

5. Check the first, middle and final GIF frames, timing and file sizes. Confirm
   the maps differ and dashboard values follow the playhead. Keep README and
   guide image links pointing to the same canonical GIFs. Remove temporary
   replays after verification; never add checkpoints or raw image sequences to Git.

The encoder uses palette generation and rectangular differential rendering;
it adds no viewer or simulator dependency. `work/export_terrain_gifs.py` remains
a legacy importer for supplied MOV files, not the refresh path for these
current-viewer exports.

The original `.gif` files remain dark-theme fallbacks and keep existing direct
asset URLs working. Light-theme clips use animated WebP after the same 128-color
quantization, avoiding another full set of GIFs. Both carry the same frame count,
step samples and duration. Page markup uses `<picture class="replay-preview">`:
GitHub selects its media source, while the Zensical preprocessor resolves nested paths and
emits image variants controlled by the site's explicit palette toggle.
