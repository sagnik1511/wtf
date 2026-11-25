# Command and override reference

[Customization handbook](customization.md) · [JSON fields](configuration-reference.md) · [Running guide](running.md)

Commands run from the checkout root. These options configure execution; do not
add their names as arbitrary JSON fields. `--help` prints the installed source's
current parser without starting training or a server.

## Installed command hub

`wtf run`, `wtf view`, `wtf analyze` dispatch to the parsers below. `wtf presets`
lists bundled environment names. `python -m wtf` offers the same hub;
`wtf-run`, `wtf-view`, `wtf-analyze` are direct aliases. See [packaging](packaging.md).

## Simulator

```sh
.venv/bin/python -m wtf.cli ENVIRONMENT.json [OPTIONS]
```

| Argument | Definition |
| --- | --- |
| `environment` | Complete environment JSON; defaults to `environments/baseline.json` when omitted |
| `--device cpu\|cuda` | CPU default; optional experimental Linux/NVIDIA dense forwards, PPO/masked-PPO and DQN/DDQN training. [Build and limits](cuda.md) |
| `--cuda-precision fp64\|fp32` | CUDA arithmetic; FP64 default. Optional FP32 changes learning numerics, not world physics/storage |
| `--cuda-inference auto\|cpu\|cuda` | Auto uses CPU actions for CUDA-trained PPO/DQN and CUDA actions for other supported dense policies |
| `--cuda-optimizer cpu\|cuda` | DQN/DDQN optimizer placement when CUDA is selected; default cuda publishes coherent CPU snapshots, cpu keeps gradient download and host Adam |
| `--cuda-profile` | CUDA events and host component timings; adds overhead, use separately from speed comparisons |
| `--cuda-batch-agents N` | Independent jobs per GPU batch; default 64, range 1–4096 |
| `--cuda-workspace-mb N` | Device workspace LRU budget in MiB; default 256, range 1–16384; excludes driver/host memory |
| `--mode train` | Choose actions with training exploration and perform learning; default for a fresh experiment |
| `--mode evaluate` | Use evaluation decisions without learning; this is still simulation, not saved replay |
| `--mode random` | Use random baseline decisions without training |
| `--runs N` | Override configured number of episodes for a fresh experiment |
| `--steps N` | Override maximum world steps per episode |
| `--seed N` | Set reproducible base seed; defaults seed policy to increment unless explicitly fixed |
| `--seed-policy random\|fixed\|increment` | Override world randomization; explicit `--seed` cannot be combined with random |
| `--resume PATH` | Restore a supported exact checkpoint; use saved environment/schedule rather than a new JSON |
| `--record-last-episodes K` | Current source: record final K scheduled episodes, capped at total; -1 records all (default), 0 none. Persists on resume; [early-stop/profile behavior](recording.md#record-only-the-final-scheduled-episodes) |
| `--recording-profile full\|compact\|training-only` | Training capture preset; evaluation/random resolve a configured training preset to full. Explicit compact/training-only CLI choices require train mode. Persisted on resume; see [profiles](recording.md#training-recording-profiles) |
| `--consequences-from PATH` | Import compatible shared consequence estimates from a result folder or `consequences.json.xz`; requires opted-in inputs, leaves individual policy weights independent and cannot override exact resume |
| `--consequence-update-schedule continuous\|episode` | Fresh enabled consequence-learning experiments: publish continuously (default) or before the next training episode; saved schedule governs resume |
| `--freeze-brains` | Evaluation only: offspring copy the same-sex parent’s model without crossover; no learning or mutation. Persisted in checkpoints |
| `--no-record` | Skip replay recording; metrics and exact checkpoints are still saved |
| `--progress`, `--no-progress` | Force or suppress tqdm display; otherwise automatic output handling applies |
| `--output DIR` | Parent output directory; a new experiment subfolder is created, default `results/` |
| `--library-dir DIR` | Optional shared champion library for selectors and extra copies; run-local archives remain separate |
| `--checkpoint-format binary\|compact\|json` | Binary/XZ `.bin.xz` (default), compact gzip `.json.gz`, or legacy `.json`; not a replay encoding setting |
| `--checkpoint-keep-last N` | Retain latest N newly written exact recovery checkpoints including final; default 2, zero retains all |
| `--checkpoint-every-steps K` | Full recovery checkpoint cadence within a run; default 500, zero disables periodic writes |
| `--brain-save-every-steps K` | Check periodically for strict rank improvement with changed weights; default 250, zero disables |
| `--brain-keep-last N` | Retain N most recent qualifying best-brain collections; default 2, zero retains all; all run-end archives/playbacks remain |
| `--brain-save-min-score-delta D` | Require absolute primary-score gain D for periodic saves; default zero; does not change champion selection |
| `--shared-checkpoint-metadata` | Opt in to shared, checksummed static metadata for v4 binary recovery files; keep the metadata folder with checkpoints |
| `--self-contained-checkpoints` | Override a shared-metadata JSON setting and embed metadata in standalone v3 binary output (the default) |
| `--no-library` | Disable saving new champion models; does not remove existing artifacts or disable exact checkpoints |
| `--brains-from RUN_NAME_OR_PATH` | Select an experiment's run-local champion archive or an explicit periodic index; directories default to their latest archived numbered run |
| `--brain-run N` | Choose an archived numbered run within `--brains-from` |
| `--brain SPECIES[:SEX]=MODEL` | Repeatable initial-model choice: criterion from the source experiment, artifact path, supported shared-library selector, or `fresh` |
| `--champion SPECIES[:SEX]=CRITERION` | Repeatable champion choice for subsequent chaining; does not load initial weights |
| `--chain-brains`, `--no-chain-brains` | Enable/disable later fresh episodes starting from retained matching champions |
| `--early-stop-runs K` | Stop after K consecutive completed runs without a monitored improvement; zero disables |
| `--early-stop-species NAME` | Repeatable species filter for early stopping; default monitors any configured species |
| `-h`, `--help` | Print parser help and exit |

`--resume` rejects configuration overrides such as runs, steps, seeds, model
selectors, champion/chaining and early-stop changes. It permits the supported
execution controls, including mode/output/recording/progress/checkpoint-output
options. Resuming a completed final checkpoint does not manufacture a new world.
See [resume semantics](running.md#resume-an-exact-experiment).

A complete example of fresh initialization and later-run lineage selection:

```sh
.venv/bin/python -m wtf.cli environments/maskable_ppo.json \
  --runs 3 --seed 100 --brain prey=fresh --brain predator=fresh \
  --champion prey=lineage --champion predator=lineage --progress
```

Species names above refer to this preset; substitute exact configured names for
custom species. Choices such as `lifespan`, `food`, `kills`, `health`, `balanced`
and `lineage` have different meanings and diet restrictions; see the
[champion definitions](brain-library.md), rather than treating them as algorithm names.

## Playback server

```sh
.venv/bin/python -m wtf.server --results results --port 8000
```

| Argument | Definition |
| --- | --- |
| `--results DIR` | Experiment root to list/read; default is `results/` in the current working directory |
| `--port N` | HTTP port, default 8000 |
| `--host ADDRESS` | Bind address, default `127.0.0.1`; Docker uses `0.0.0.0` inside its container |
| `--help` | Print options and exit without starting the server |

The server is read-only playback and explanation analysis; it never trains.
`./run.sh` forwards these options. Docker's launcher accepts `build`,
`train [CLI arguments]`, `view [port]` and `down`; see [Docker](docker.md).

## Saved-run analyzer

```sh
.venv/bin/python -m wtf.inspection.analyze results/EXPERIMENT --run 1 --window-steps 250
```

| Argument | Definition |
| --- | --- |
| `recording` | Experiment directory or supported playback directory |
| `--run N` | Analyze one recorded numbered run; omission analyzes the available runs |
| `--window-steps N` | World-step span for early/late behavior comparisons, not replay-frame count |
| `--output FILE` | Output report path; default is `analysis.json` under the supplied recording path |
| `--help` | Print options and exit |

Analysis reads recorded behavior without loading learner checkpoints or running
new simulation. Sparse recordings limit what can be reconstructed. Storage
conversion tools have separate commands documented in [recording](recording.md)
and [binary storage](binary-storage.md); preserve originals as those guides specify.
