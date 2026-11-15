# Playback, recording format and memory

[Project overview](../README.md) · [Setup and runs](running.md) · [Analysis](analysis.md)

Commands in this guide run from the project root.

## Playback controls

The Python simulator and JavaScript viewer are independent. Run training first,
then start the viewer with `./run.sh`. Open <http://127.0.0.1:8000>, choose an
experiment and click **Play recording**. The server never loads a learning
checkpoint or advances the world.

- **Refresh list** discovers new saved experiments.
- Use the run selector, frame slider, **Back**, and **Next frame** to inspect the timeline.
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

## Recording settings

Each environment has a recording section:

```json
"recording": {
  "enabled": true,
  "frame_every_steps": 1,
  "frames_per_chunk": 64,
  "diagnostics_every_steps": 10
}
```

Previous configurations without this section use the same defaults. Set `enabled` to `false` or pass `--no-record` for metrics/checkpoints only. Increase `frame_every_steps` to reduce disk usage; initial and final frames are always saved. Intermediate events between sampled frames are omitted, while sampled metrics retain cumulative outcomes. Set `diagnostics_every_steps` to `0` to omit brain diagnostics entirely. When enabled, diagnostics are also sampled at chunk boundaries so chunks can be inspected independently.

The recorder encodes one frame at a time into an open chunk file; it never collects the full playback history in RAM. Completed chunk files are renamed atomically, then advertised in the manifest. Cleanly interrupted runs retain their committed frames and are labeled `interrupted`; a hard crash can leave an unadvertised `.tmp` file. Such files are not served. The current manifest can be refreshed to discover additional committed chunks, although the intended workflow is playback after simulation finishes.

The browser retains one decoded world chunk at a time and loads the necessary chunk for seeking. It also keeps a small previous-frame position list to locate removed animals for death highlights. A direct seek can briefly read the preceding chunk to recover those positions, then release it. For older recordings that omit intermediate steps, natural deaths still show a banner, but no location is guessed if the immediately preceding state was not recorded. Chart history is loaded separately as at most 2,048 lightweight population/food samples for the current run, plus the exact current-frame point. It persists across world-chunk boundaries and only displays points up to the playback position. Very long histories are progressively downsampled, preserving the first and last recorded metric points. Switching runs resets the chart to that run. The server reads one requested chunk per response and has no recording cache. Neural weights, target networks, optimizer state and raw experience buffers are absent from playback files; recorded diagnostics contain observations and counts plus Q values/epsilon or PPO probabilities/entropy/value. The inspector identifies the diagnostic sample step, which may precede the displayed body state.

Training still needs the live brains and replay buffers. CLI checkpoint writing copies and encodes one brain at a time, then atomically publishes compact JSON. Programmatic `checkpoint()` still returns a detached in-memory snapshot of all brains; restoring a JSON checkpoint still parses that complete file. Recording avoids an additional full-run frame history; it does not make learning itself constant-memory or guarantee a measured peak-memory reduction. Disk usage grows with duration. `frames_per_chunk` is capped at 256, but individual frame size still depends on world/population size.

## Read-only HTTP API

| Route | Response |
| --- | --- |
| `GET /api/recordings` | Saved recording catalog |
| `GET /api/recordings/{id}` | Recording metadata and run list |
| `GET /api/recordings/{id}/runs/{run}/chunks/{index}` | One bounded frame chunk |
| `GET /api/recordings/{id}/runs/{run}/metrics` | Lightweight population/food history |
| `GET /api/recordings/{id}/runs/{run}/analysis` | Behavior and sampled learning statistics |

 The previous live simulation POST endpoints return 405. Use `./run.sh --results PATH` to view recordings stored under a custom CLI `--output` directory.
