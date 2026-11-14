# Ecosystem Lab

A Python ecosystem simulation with individually learning prey and predators, JSON environments, reproducible experiments, and a JavaScript canvas viewer.

## Record first, replay later

The Python simulator and JavaScript viewer are independent. Simulation runs headlessly and streams playback data to disk. The viewer reads saved chunks; it never loads a learning checkpoint or advances the world.

```sh
cd /Users/tensored/wtf
.venv/bin/python -m ecosystem.cli environments/baseline.json --runs 1
./run.sh
```

Open <http://127.0.0.1:8000>, select a saved experiment, and click **Play recording**. Use the frame slider to seek, select a recorded run, or click an animal to inspect its recorded state and sampled diagnostics. **Refresh list** discovers newly recorded experiments. Kills have red board markers, an attacker highlight, an identity banner, and a dedicated cumulative counter. **Hold briefly on kills** adds a 0.9-second playback hold at recorded kill frames; uncheck it for uninterrupted playback. Markers persist briefly across sequential chunk boundaries and clear when switching runs. Existing recordings work without regeneration; events skipped by a reduced recording frequency cannot be highlighted.

You can close the simulator before starting playback. The playback server does not import the simulation or learning modules. Existing results created before playback recording was added have metrics/checkpoints only and do not appear in the recording catalog; run another experiment to create a recording.

The launcher uses the project-local `.venv` and stores bytecode caches in `.cache/`. For CLI commands using `python` below, activate the environment first. If recreating the environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Python 3.9+ is supported. There are no third-party application dependencies. Future Python packages should be installed inside `.venv`; JavaScript rendering uses native browser APIs and needs no packages or build step.

## JSON environments

Each JSON file is a complete experiment definition:

- `environments/baseline.json`: 24 prey, 6 predators, moderate food.
- `environments/scarce_food.json`: slower regrowth; otherwise the baseline settings.
- `environments/dense_obstacles.json`: more obstructed cells.
- `environments/predator_heavy.json`: 14 predators.

Copy a JSON file, edit its name and values, and pass its path to the Python CLI to add a scenario. The playback viewer shows the recorded configuration as read-only text.

All costs and growth rates are per simulation step. Playback speed changes only the rate of displaying saved frames. Configuration changes apply to the next CLI experiment; they cannot change an existing recording.

`ecosystem/config.py` validates required and unknown fields, numeric ranges, compatible diets and actions, referenced brain templates, replay settings, and world capacity. Invalid configurations are rejected before simulation begins. `environment.schema.json` provides an editor-friendly structural schema; Python validation also enforces cross-field constraints.

Use `obstacles.generation = "explicit"` with `explicit_cells = [[x, y], ...]` for hand-designed maps. In this mode `occupied_fraction` is ignored. Animals and resource patches may share cells; two animals cannot occupy one cell. Generation does not guarantee connected free space.

## Run experiments without the viewer

```sh
python -m ecosystem.cli environments/baseline.json
python -m ecosystem.cli environments/scarce_food.json --runs 3 --steps 1000 --seed 42
python -m ecosystem.cli environments/baseline.json --mode random --runs 3
```

Each invocation creates a unique folder inside `results/` containing:

- `environment.json`: exact effective configuration, including CLI overrides.
- `manifest.json`: mode, Python/platform version, checkpoint/observation versions.
- `run_seeds.json`: actual world seeds used.
- `metrics.csv`: per-run population, food, cumulative energy gained, hunts and deaths.
- `summary.json`: per-run outcomes, lifespans of dead animals, and ages of survivors. Survivor ages are censored observations, not completed lifespans.
- `checkpoints/`: scheduled learning checkpoints and a final checkpoint.
- `playback/recording.json`: versioned recording manifest and static layout per run.
- `playback/run_0001_chunk_000000.json`, etc.: small arrays of recorded frames.

Pass `--output PATH` to choose another result directory. For comparisons, use the same base seed and world seed policy across scenarios and repeat with several base seeds. Layouts differ when obstacle or population settings differ, even with the same seed.

Resume a checkpoint:

```sh
python -m ecosystem.cli --resume results/EXPERIMENT/checkpoints/run_0001.json
```

Checkpoints include the world, counters, private replay buffers, online/target weights, and policy/training/world random-generator states. They are JSON, not executable pickle files. Resume continues remaining runs from the saved experiment; a completed experiment stays completed.

Evaluate saved brains by restoring a checkpoint in evaluation mode:

```sh
python -m ecosystem.cli --resume results/EXPERIMENT/checkpoints/run_0001.json --mode evaluate
```

Use an intermediate checkpoint with remaining runs to evaluate future runs. Checkpoint restore is a CLI operation; the playback viewer reads frame recordings only.

## Recording and memory

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

The recorder writes each frame directly to an open chunk file; it never collects the full playback history in RAM. Completed chunk files are renamed atomically, then advertised in the manifest. Cleanly interrupted runs retain their committed frames and are labeled `interrupted`; a hard crash can leave an unadvertised `.tmp` file. Such files are not served. The current manifest can be refreshed to discover additional committed chunks, although the intended workflow is playback after simulation finishes.

The browser retains one decoded chunk at a time, loads the necessary chunk for seeking, and builds its chart from that chunk only. The server reads one requested chunk per response and has no recording cache. Neural weights, target networks, and experience buffers are absent from playback files; recorded diagnostics include only observation features, action values, exploration, and counts. The inspector identifies the diagnostic sample step, which may precede the displayed body state.

Training still needs the live brains and replay buffers. Learning checkpoint serialization also temporarily copies training state. Recording avoids an additional full-run frame history; it does not make learning itself constant-memory or guarantee a measured peak-memory reduction. Disk usage grows with duration. `frames_per_chunk` is capped at 256, but individual frame size still depends on world/population size.

The read-only server exposes `GET /api/recordings`, `GET /api/recordings/{id}`, and `GET /api/recordings/{id}/runs/{run}/chunks/{index}`. The previous live simulation POST endpoints return 405. Use `./run.sh --results PATH` to view recordings stored under a custom CLI `--output` directory.

## Simulation and learning

`world.py` owns physical state and emits events. `brain.py` owns individual learning. `runner.py` connects observations, actions, consequences and experiences. `recording.py` writes playback files; `server.py` only serves those files; `viewer/` renders saved data and controls playback.

The world collects all actions before changing animal state:

1. Resolve turns and movement. Walls and initial animal occupancy block moves. Swaps and following into cells vacated during the same step are disallowed. Competing claims to an empty cell use seeded random selection.
2. Resolve resource feeding with seeded contention order. Prey eat from their current cell, or the cell directly ahead. Eating stops at maximum energy.
3. Resolve simultaneous attacks against prey in the cell directly ahead. Combined damage is capped at remaining health. Exactly one seeded attacker gets energy for a kill. A prey's chosen feeding action can complete in the same step it is killed.
4. Charge living and action costs, including failed movement and attack attempts. At zero energy, apply starvation damage. There is no additional affordability restriction on actions.
5. Record deaths and individual rewards, remove dead bodies, regrow resources up to capacity, and construct the next observations.
6. Give each learning agent its own experience, including a terminal experience on death.

Food rewards use actual energy gained. Damage and death give configurable penalties. There is no reward for approaching targets, simply surviving a step, moving or resting. There is no reproduction, corpse persistence, health recovery, or population replenishment within a run.

The observation vector has 20 entries:

| Indices | Meaning |
| --- | --- |
| 0–1 | Energy and health as fractions of their maxima |
| 2–3 | Facing direction as dx/dy |
| 4–7 | Blocked neighbors: forward, right, backward, left |
| 8–11 | Nearest visible food: dx/radius, dy/radius, distance/radius, visible |
| 12–15 | Nearest visible prey: same four features |
| 16–19 | Nearest visible predator: same four features |

Distances use a circular vision radius, with obstacle occlusion by integer line rays. Direction deltas use world axes; there is no field-of-view angle limit. Missing targets have four zero values. The current animal is excluded from animal targets. Resources with negligible quantity are omitted from perception.

Brains use one small tanh hidden layer and linear action-value outputs, independent replay buffers, uniformly sampled mini-batches, Huber-loss gradients, plain SGD, target networks, and epsilon-greedy exploration. There is no Adam state or third-party ML framework. Animals of a species start from copied base weights but have separate policy and learning randomness. Target networks synchronize after the configured number of training updates.

The same brain template can be referenced by multiple species without sharing live weights. Learning updates occur at the configured cadence. Time limits are truncations and retain bootstrapped future value; death is terminal. Resetting a body can preserve its learned weights, exploration state and experience buffer, or reinitialize its brain according to configuration. Species base weights are reproducible across resets.

`train` explores and learns; `evaluate` chooses greedily and does not learn; `random` chooses random allowed actions and does not learn. A population that survives longer is not by itself evidence of learning. Compare frozen trained brains with fresh and random policies across multiple held-out seeds. The initial settings are experimental, not calibrated for stable coexistence. Independent simultaneous learners can be unstable.

The standard-library implementation prioritizes inspectability and small populations. Large worlds and many private networks will be slower than a vectorized implementation. CLI metrics stream to disk, and playback charts show only the currently loaded chunk. The simulation runner retains at most 2,000 sampled metric points per current run for checkpoint compatibility; this is separate from playback recording.

## Verification

```sh
python -m unittest discover -v
# Optional JS playback-engine checks (requires Node.js 20+):
node --test tests/playback.test.mjs
```

Tests cover strict configuration validation, resource limits, blocked and contested movement, feeding rewards, simultaneous hunts, starvation, occlusion, independent weights, positive-feedback learning, terminal versus truncated experience, evaluation behavior, deterministic runs, exact JSON checkpoint continuation, resets, streamed recording correctness, recording transparency, committed-chunk visibility, bounded playback chunk loading, seeking, and read-only HTTP endpoints.

The server binds only to `127.0.0.1`. Use `--port 8001` if the default port is occupied. Close the terminal process or press Ctrl-C to stop it.
