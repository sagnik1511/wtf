# WTF — Wildlife Trial Framework

**Reinforcement Learning Ecosystem Simulator**

A Python ecosystem simulation with individually learning prey and predators, JSON environments, reproducible experiments, and a JavaScript canvas viewer.

## Record first, replay later

The Python simulator and JavaScript viewer are independent. Simulation runs headlessly and streams playback data to disk. The viewer reads saved chunks; it never loads a learning checkpoint or advances the world.

```sh
cd /Users/tensored/wtf
.venv/bin/python -m ecosystem.cli environments/baseline.json --runs 1
./run.sh
```

Open <http://127.0.0.1:8000>, select a saved experiment, and click **Play recording**. Use the frame slider to seek, select a recorded run, or click an animal to inspect its recorded state and sampled diagnostics. **Refresh list** discovers newly recorded experiments. Kills have red board markers, an attacker highlight, an identity banner, and a dedicated cumulative counter. Natural deaths (currently starvation) use matching amber rings, a DIED marker, and a banner naming the animal and cause. **Hold briefly on deaths** adds a 0.9-second playback hold at kill or natural-death frames; uncheck it for uninterrupted playback. Markers persist briefly across sequential chunk boundaries and clear when switching runs. Existing recordings work without regeneration; events skipped by a reduced recording frequency cannot be highlighted.

You can close the simulator before starting playback. The playback server does not import the simulation or learning modules. Existing results created before playback recording was added have metrics/checkpoints only and do not appear in the recording catalog; run another experiment to create a recording.

The launcher uses the project-local `.venv` and stores bytecode caches in `.cache/`. For CLI commands using `python` below, activate the environment first. If recreating the environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Python 3.9+ is supported. There are no third-party application dependencies. Future Python packages should be installed inside `.venv`; JavaScript rendering uses native browser APIs and needs no packages or build step.

## JSON environments

Each JSON file is a complete experiment definition:

- `environments/baseline.json`: 48 prey, 12 predators, moderate food; predator damage 30 against prey health 30. Successful attacks are lethal, but reaching and attacking prey remain learned decisions.
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
- `checkpoints/`: scheduled intermediate learning checkpoints and a final checkpoint. The final run writes `final.json` once instead of duplicating the same state in a numbered file.
- `playback/recording.json`: versioned recording manifest and static layout per run.
- `playback/run_0001_chunk_000000.json`, etc.: small arrays of recorded frames.

Pass `--output PATH` to choose another result directory.

### Randomized and reproducible runs

Example environments default to `"world_seed_policy": "random"`. Each fresh CLI invocation gets a new random seed stream, and every run in that experiment gets a distinct seed. Positions, resource locations, obstacle layouts and conflict outcomes vary. Newly initialized species base brains and individual exploration streams vary too. Animals still begin from copied species base weights, and `retain_brains_between_runs: true` preserves learned brains across the different worlds.

The three policies are:

| `experiment.world_seed_policy` | Behavior |
| --- | --- |
| `random` | New random seed for each run and fresh invocation; JSON `seed` is not used |
| `increment` | Reproducible seeds `seed`, `seed + 1`, etc. |
| `fixed` | Same world seed on each run; retained learning can still change behavior |

Run normally to randomize:

```sh
.venv/bin/python -m ecosystem.cli environments/baseline.json --runs 3
```

For reproducible comparisons, `--seed 42` automatically selects `increment` unless you explicitly choose `--seed-policy fixed`. `--seed-policy random` explicitly enables randomization for an older fixed configuration. Combining `--seed` with `--seed-policy random` is rejected to avoid silently ignoring an explicit seed.

```sh
.venv/bin/python -m ecosystem.cli environments/baseline.json --runs 3 --seed 42
.venv/bin/python -m ecosystem.cli environments/baseline.json --runs 3 --seed 42 --seed-policy fixed
```

Actual world seeds are printed, written to `run_seeds.json` as runs start, and stored with recordings and summaries. `manifest.json` also records the experiment's randomization seed. The viewer's run selector displays each recorded world seed. Checkpoints preserve the seed generator state, so restoring continues the same future seed sequence and learning trajectory, rather than starting a fresh random sequence. Old deterministic checkpoints remain supported. To compare configurations, use the same explicit base seed and policy and repeat with several seeds. Layouts differ when obstacle or population settings differ, even with the same seed.

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

The recorder encodes one frame at a time into an open chunk file; it never collects the full playback history in RAM. Completed chunk files are renamed atomically, then advertised in the manifest. Cleanly interrupted runs retain their committed frames and are labeled `interrupted`; a hard crash can leave an unadvertised `.tmp` file. Such files are not served. The current manifest can be refreshed to discover additional committed chunks, although the intended workflow is playback after simulation finishes.

The browser retains one decoded world chunk at a time and loads the necessary chunk for seeking. It also keeps a small previous-frame position list to locate removed animals for death highlights. A direct seek can briefly read the preceding chunk to recover those positions, then release it. For older recordings that omit intermediate steps, natural deaths still show a banner, but no location is guessed if the immediately preceding state was not recorded. Chart history is loaded separately as at most 2,048 lightweight population/food samples for the current run, plus the exact current-frame point. It persists across world-chunk boundaries and only displays points up to the playback position. Very long histories are progressively downsampled, preserving the first and last recorded metric points. Switching runs resets the chart to that run. The server reads one requested chunk per response and has no recording cache. Neural weights, target networks, and experience buffers are absent from playback files; recorded diagnostics include only observation features, action values, exploration, and counts. The inspector identifies the diagnostic sample step, which may precede the displayed body state.

Training still needs the live brains and replay buffers. CLI checkpoint writing copies and encodes one brain at a time, then atomically publishes compact JSON. Programmatic `checkpoint()` still returns a detached in-memory snapshot of all brains; restoring a JSON checkpoint still parses that complete file. Recording avoids an additional full-run frame history; it does not make learning itself constant-memory or guarantee a measured peak-memory reduction. Disk usage grows with duration. `frames_per_chunk` is capped at 256, but individual frame size still depends on world/population size.

The read-only server exposes `GET /api/recordings`, `GET /api/recordings/{id}`, `GET /api/recordings/{id}/runs/{run}/chunks/{index}`, and `GET /api/recordings/{id}/runs/{run}/metrics`. The previous live simulation POST endpoints return 405. Use `./run.sh --results PATH` to view recordings stored under a custom CLI `--output` directory.

## Simulation and learning

`world.py` owns physical state and emits events. `brain.py` owns individual learning. `runner.py` connects observations, actions, consequences and experiences. `recording.py` writes playback files; `server.py` only serves those files; `viewer/` renders saved data and controls playback.

The world collects all actions before changing animal state:

1. Resolve turns and movement. Walls and initial animal occupancy block moves. Swaps and following into cells vacated during the same step are disallowed. Competing claims to an empty cell use seeded random selection.
2. Resolve resource feeding with seeded contention order. Prey eat from their current cell, or the cell directly ahead. Eating stops at maximum energy.
3. Resolve simultaneous attacks against prey in the cell directly ahead. Combined damage is capped at remaining health. Exactly one seeded attacker gets energy for a kill. A prey's chosen feeding action can complete in the same step it is killed.
4. Charge living and action costs, including failed movement and attack attempts. At zero energy, apply starvation damage. There is no additional affordability restriction on actions.
5. Record deaths and individual rewards, remove dead bodies, regrow resources up to capacity, and construct the next observations.
6. Give each learning agent its own experience, including a terminal experience on death.

Food rewards use actual energy gained. Predators also receive the configurable `species.predator.rewards.per_damage_dealt` reward for successful hits (default environment value: `0.05` per actual damage point). A 10-damage hit earns `0.5`. The baseline now uses 30 attack damage against 30 prey health, giving `1.5` damage reward for a healthy prey killed by one attack; a kill still earns the existing food reward, up to `4.0` for 40 energy at `0.1` per energy point. Other scenarios can keep weaker multi-hit predators. Hit rewards do not restore energy. Missed attacks earn no hit reward. When several predators attack the same prey, its actual health loss is split proportionally to their attack strengths, so overkill cannot multiply damage credit. Omitting `per_damage_dealt` preserves the previous zero-hit-reward behavior, including for old checkpoints. Damage taken and death give configurable penalties. There is no reward for approaching targets, simply surviving a step, moving or resting. There is no reproduction, corpse persistence, health recovery, or population replenishment within a run.

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

`train` explores and learns; `evaluate` chooses greedily and does not learn; `random` chooses random allowed actions and does not learn. A population that survives longer is not by itself evidence of learning. Compare frozen trained brains with fresh and random policies across multiple held-out seeds. The initial settings are experimental, not calibrated for stable coexistence. Independent simultaneous learners can be unstable. Each new CLI invocation without `--resume` starts fresh brains. With `--seed` or a deterministic seed policy, repeating a single run reproduces its outcome; it does not continue learning from a previous invocation. Randomized mode varies fresh runs but is not a substitute for retaining or resuming learned brains. Use several runs with retained brains or resume an intermediate checkpoint to continue training. Making successful attacks lethal increases predation opportunities; it is a physical rule change, not evidence that the model learned to hunt.

The standard-library implementation prioritizes inspectability and small populations. Large worlds and many private networks will be slower than a vectorized implementation. CLI metrics stream to disk, and playback charts retain the current run’s lightweight history across world-chunk changes. The simulation runner retains at most 2,000 sampled metric points per current run for checkpoint compatibility; this is separate from playback recording.

## Performance

The engine shares a per-observation-phase list of positions and resources instead of rebuilding it for each animal. Squared-distance filtering reduces distance calculations while preserving target order and visibility. Small networks use native multiplication iterators with the original summation order. Replay experiences use compact immutable tuples. Snapshot creation copies primitive entity fields without recursive dataclass conversion.

Recording buffers only one encoded frame. CLI checkpoint files are written atomically in compact JSON with one copied brain at a time. This avoids simultaneously holding every copied replay buffer and a large formatted JSON string. Checkpoint values and format version remain compatible with existing files. Scheduled intermediate checkpoints are preserved; the final checkpoint is written once. Population sizes, model dimensions, training cadence, recording frequency and diagnostics remain unchanged by these optimizations.

Playback reuses the loaded snapshot when rendering and avoids clearing/reallocating the canvas backing store or writing identical layout dimensions every frame. Chunk loading, chart history, selection and death effects remain available.

Local results (median of three separate-process measurements, 180 steps, 48 prey + 12 predators, Python 3.9.6):

| Measurement | Before | After |
| --- | ---: | ---: |
| Simulation | 3.09 s | 1.69 s |
| Simulation + recording | 3.67 s | 1.80 s |
| Checkpoint writing | 1.04 s | 0.15 s |
| Peak RSS, including checkpoint writing | 114.53 MiB | 29.25 MiB |

Every before/after run produced the same complete-state digest. Ordinary simulation RAM changed only slightly; the large memory reduction is specifically in checkpoint writing.

A local before/after benchmark and repeated measurements are saved under `work/performance/`. Run the optimized probe with:

```sh
.venv/bin/python -m work.performance.benchmark --case record --steps 180 --output work/performance/local-record.json
.venv/bin/python -m work.performance.benchmark --case checkpoint --steps 180 --output work/performance/local-checkpoint.json
```

The benchmark fixes seed 42 with the baseline 48 prey/12 predators and checks a digest of the complete world, brain, replay and random-generator state. It measures process peak RSS before constructing the comparison digest; checkpoint-case peak RSS includes both simulation and checkpoint saving. These are local measurements for a small workload, not guarantees for every environment. Large populations still incur independent-brain and perception costs, and the growing collection of completed-run summaries remains in memory for experiment reporting.

## Verification

```sh
python -m unittest discover -v
# Optional JS playback-engine checks (requires Node.js 20+):
node --test tests/playback.test.mjs
```

Tests cover strict configuration validation, resource limits, blocked and contested movement, feeding rewards, simultaneous hunts, starvation, occlusion, independent weights, positive-feedback learning, terminal versus truncated experience, evaluation behavior, deterministic and randomized runs, exact JSON checkpoint continuation including future random seeds, resets, streamed recording correctness, recording transparency, committed-chunk visibility, bounded playback chunk loading, seeking, and read-only HTTP endpoints.

The server binds only to `127.0.0.1`. Use `--port 8001` if the default port is occupied. Close the terminal process or press Ctrl-C to stop it.
