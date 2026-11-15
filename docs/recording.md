# Playback, recording format and memory

[Project overview](../README.md) · [Setup and runs](running.md) · [Analysis](analysis.md)

Commands in this guide run from the project root.

## Playback controls

The Python simulator and JavaScript viewer are independent. Run training first,
then start the viewer with `./run.sh`. Open <http://127.0.0.1:8000>, choose an
experiment and click **Play recording**. The server never loads a learning
checkpoint or advances the world.

- **Refresh list** discovers new saved experiments.
- Choose any saved run using the **Run** dropdown in the top toolbar. **Previous run** and **Next run** jump between runs; the frame slider, **Back**, and **Next frame** inspect the current timeline.
- Playback automatically continues into the next recorded run and stops after the final run.
- A recording opened during training discovers new committed frames/runs when it reaches the available end. It waits for new data and refreshes metadata at most once every two seconds.
- Click an animal to see its body, lineage and sampled brain diagnostics.
- Set **Frames / second** to change playback speed; simulation outcomes stay fixed.
- Kills have red rings, attacker highlights, KILL labels and a banner.
- Natural deaths use green rings for prey and amber rings for predators, with DIED labels.
- Births use cyan rings and plus signs, BORN labels, a parent/child banner and a counter.
- Playback continues through events at the chosen rate. Event animations remain visible.
- Event markers continue across sequential chunk boundaries and reset across runs.
- **Behavior & learning** analyzes the entire saved run, including later frames; see [metric definitions](analysis.md).

You can close the simulator before playing a recording. Older experiments made
before frame recording was introduced have only metrics/checkpoints and do not
appear in the playback catalog.

## Per-step species dashboard

Click **Metrics dashboard** in the toolbar. Its charts follow the replay position
while playing, stepping or scrubbing; no future frames are displayed. The modal
has its own play, next/back and seek controls, keeping the main board's size
unchanged. Choose all species or one species and a rolling window of 128, 512 or
1,024 recorded frames. Each of six charts has an independent metric selector.

Available trends include population, mean health/energy, births and deaths per
step, predation/starvation deaths, kills, attack/eat attempts, hunting/feeding
success, starving animals, reward, age, generations and spatial groups. Cards
show the current step, not episode totals. Hunting success is kills per attack;
prey feeding success is productive feeds per eat attempt. No attempts gives no
value. Reward averages surviving animals that acted; deceased actors' rewards
are omitted. Action coverage exposes missing terminal actions.

Prediction charts distinguish action probabilities from Q values, and also
include state value, policy entropy and Q exploration. Select an action for
probability/Q charts. Dots appear only at recorded diagnostic samples; values
are not filled forward or interpolated. These are averages on the sampled
agents' changing observations, not prediction accuracy or proof of improvement.
Health/energy/age averages are missing after extinction. Known event counts
remain zero when there were no events. If frames skip simulation steps, event
metrics and success rates are missing across the gap, not reconstructed.

Spatial groups are same-species connected components of animals linked within
3 Euclidean grid cells, ignoring obstacles. The group count includes components
of two or more animals; solitary animals are separate. Links are transitive,
and proximity does not imply cooperative behavior or shared brains.

The server derives telemetry from old or new replays on demand, in pages of at
most 128 frames, reading bounded chunks. The browser caches at most 12 pages of
scalar summaries; there are no additional result files, models or simulations.
Live recording pages refresh when additional frames are committed. The read-only
API is `GET /api/recordings/{id}/runs/{run}/telemetry/{page}` (zero-based page).
The **Behavior & learning** panel remains a separate whole-run summary.

## Recording settings

Each environment has a recording section:

```json
"recording": {
  "enabled": true,
  "frame_every_steps": 1,
  "frames_per_chunk": 64,
  "diagnostics_every_steps": 10,
  "encoding": "delta",
  "compression": "gzip",
  "compression_level": 6
}
```

Replay storage separates the data layout (`encoding`) from compression:

- `"delta"` (default): each chunk begins with a complete keyframe. Later frames
  store only changed top-level/metric fields and entity additions, removals and
  changed fields. Animal IDs and resource coordinates identify entities; small
  field masks avoid repeating field names. Unchanged food patches are omitted.
- `"snapshot"`: every frame is complete, useful for debugging and comparison.
- `"gzip"` compression (default level 6) wraps either layout in `*.json.gz`.
  Levels 1–9 are accepted; `"none"` writes uncompressed `*.json`.

Both layouts preserve every recorded frame, full-precision number, event and
sampled diagnostic. Updates store absolute values, avoiding cumulative rounding.
Every chunk has its own keyframe and field table: seeking reconstructs at most
one chunk, never the whole run. The recorder streams frames and retains only the
previous frame for computing changes. No database or new package is required.

Previous configurations without this section use the same defaults. Set `enabled` to `false` or pass `--no-record` for metrics/checkpoints only. Increase `frame_every_steps` to reduce disk usage; initial and final frames are always saved. Intermediate events between sampled frames are omitted, while sampled metrics retain cumulative outcomes. Set `diagnostics_every_steps` to `0` to omit brain diagnostics entirely. When enabled, diagnostics are also sampled at chunk boundaries so chunks can be inspected independently.

The recorder encodes one frame at a time into an open chunk file; it never collects the full playback history in RAM. Completed chunk files are renamed atomically, then advertised in the manifest. Cleanly interrupted runs retain their committed frames and are labeled `interrupted`; a hard crash can leave an unadvertised `.tmp` file. Such files are not served. The current manifest can be refreshed to discover additional committed chunks, although the intended workflow is playback after simulation finishes.

The browser retains one decoded world chunk at a time and loads the necessary chunk for seeking. It also keeps a small previous-frame position list to locate removed animals for death highlights. A direct seek can briefly read the preceding chunk to recover those positions, then release it. For older recordings that omit intermediate steps, natural deaths still show a banner, but no location is guessed if the immediately preceding state was not recorded. Chart history is loaded separately as at most 2,048 lightweight population/food samples for the current run, plus the exact current-frame point. It persists across world-chunk boundaries and only displays points up to the playback position. Very long histories are progressively downsampled, preserving the first and last recorded metric points. Switching runs resets the chart to that run. The server reads one requested chunk per response and has no recording cache. Neural weights, target networks, optimizer state and raw experience buffers are absent from playback files; recorded diagnostics contain observations and counts plus Q values/epsilon or PPO probabilities/entropy/value. The inspector identifies the diagnostic sample step, which may precede the displayed body state.

Training still needs the live brains and replay buffers. CLI checkpoint writing copies and encodes one brain at a time, then atomically publishes a compact compressed checkpoint ([format and migration](running.md#optimize-existing-checkpoint-files)). Programmatic `checkpoint()` still returns a detached in-memory snapshot of all brains; restoring a JSON checkpoint still parses that complete file. Recording avoids an additional full-run frame history; it does not make learning itself constant-memory or guarantee a measured peak-memory reduction. Disk usage grows with duration. `frames_per_chunk` is capped at 256, but individual frame size still depends on world/population size.

## Read-only HTTP API

| Route | Response |
| --- | --- |
| `GET /api/recordings` | Saved recording catalog |
| `GET /api/recordings/{id}` | Recording metadata and run list |
| `GET /api/recordings/{id}/runs/{run}/chunks/{index}` | One bounded frame chunk |
| `GET /api/recordings/{id}/runs/{run}/metrics` | Lightweight population/food history |
| `GET /api/recordings/{id}/runs/{run}/analysis` | Behavior and sampled learning statistics |

 The previous live simulation POST endpoints return 405. Use `./run.sh --results PATH` to view recordings stored under a custom CLI `--output` directory.


## Optimize older recordings

Restart the playback server and refresh the viewer after upgrading. Old full
snapshot arrays, compact delta chunks, plain JSON and gzip files remain readable,
including mixtures left by an interrupted conversion.

```sh
# Upgrade one completed experiment to delta encoding plus gzip:
.venv/bin/python -m ecosystem.compress_recordings results/EXPERIMENT --delta
# Upgrade all completed/interrupted experiments:
.venv/bin/python -m ecosystem.compress_recordings results --all --delta
# Gzip alone, preserving the existing serialized layout:
.venv/bin/python -m ecosystem.compress_recordings results --all
```

Conversion handles one committed chunk at a time and retains the smaller
representation. Delta conversion verifies SHA-256 of canonical expanded frames;
gzip-only conversion verifies decompressed bytes. Replacements are atomic,
originals survive failed verification, batch mode skips active recordings, and
conversion can be rerun safely. Checkpoints, saved champions, metrics and training
configuration are untouched. A single active recording is rejected; finish or
close training first. Metadata may say `mixed` if some original chunks were smaller.

For programmatic access use `ecosystem.replay_io.load_chunk(path)`: it returns
full frames for either layout/compression. Behavior analysis and fallback
population history use the same reader. The viewer retains one decoded world
chunk, so this storage change does not reduce the expanded chunk's memory.

The chunk HTTP endpoint returns a full-frame JSON array by default for existing
clients. The viewer sends `X-WTF-Replay: delta-v1` to receive the native stored
layout and decode it locally. Gzip-capable native requests receive stored gzip
bytes directly. Responses vary on both `Accept-Encoding` and `X-WTF-Replay`.
No simulation or learning runs during decoding.

A representative 400-step PPO replay (401 frames, seed 11, 64-frame chunks) used:

| Layout | Chunk bytes |
| --- | ---: |
| Full JSON snapshots | 9,276,835 |
| Full snapshots + gzip 6 | 851,601 |
| Delta JSON | 3,881,570 |
| Delta + gzip 6 | 722,700 |

Delta plus gzip saved another **15.1% versus gzip alone**, or **92.2% versus full
JSON**. This is one local workload; savings depend on population, changes and
diagnostics. Encoding the already generated frames took 0.55 seconds and parsing
plus delta reconstruction took 0.68 seconds across all 401 frames; these are not
end-to-end training or browser timings. Reducing recording frequency can save
more space but omits intermediate events; both layouts above preserve them.
