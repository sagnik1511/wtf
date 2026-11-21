# WTF — Wildlife Trial Framework

A Python reinforcement-learning ecosystem with independently learning prey and
predators, growing resources, obstacles and paired male/female reproduction. Environments are JSON;
a JavaScript canvas viewer replays saved runs. Add named prey/predator species
through JSON; the presets start with one of each.

The simulator records first and the viewer plays later, keeping training and
rendering independent. Replay chunks use structured binary deltas and lossless XZ compression to reduce disk usage.

![Saved ecosystem playback with prey, predators, obstacles and event highlights](docs/assets/simulation-playback.gif)

A short replay of a saved experiment. See the [live species charts](docs/recording.md#per-step-species-dashboard)
for population, energy, births and hunting trends.

## Quick start

Requires Python 3.9+. Run from the project directory:

```sh
cd /path/to/wtf
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements.txt
.venv/bin/python -m ecosystem.cli environments/baseline.json --runs 1
./run.sh
```

Open <http://127.0.0.1:8000>, select the saved experiment and click **Play recording**.
You can close the simulator before starting playback. Python packages stay in
`.venv`; the viewer needs no packages or build step.

## Keep learned brains

Completed training runs archive health, kills, lifespan and food-gain champions
for each species/sex group inside `results/<run-name>/champions/`. Models keep their scores and ancestry.
Use the longest-lived prey champions and predator hunting champions to start a new generation:

```sh
.venv/bin/python -m ecosystem.cli environments/baseline.json --runs 1 \
  --brains-from PREVIOUS_RUN_NAME --brain prey=lifespan --brain predator=kills
```

Replace `PREVIOUS_RUN_NAME` with a folder name under `results/` containing compatible Q champions. See the
[saved-brain guide](docs/brain-library.md) to choose another criterion, algorithm
or exact model generation. Species-wide selectors resolve matching female and male champions separately.
With multiple runs, each group’s champion seeds the next run by default; add `--no-chain-brains` to restart from the original models.

## Documentation

| Guide | Contents |
| --- | --- |
| [Visual guide](docs/visual-guide.md) | Find diagrams for simulation, learning, inheritance, storage and workflows |
| [Setup and runs](docs/running.md) | Installation, CLI flags, seeds, progress and exact checkpoint resume |
| [Physics illustrated](docs/physics.md) | Interactive examples and diagrams for visibility, water, elevation and reproduction |
| [Geography and terrain](docs/geography.md) | Full-viewer terrain GIFs, river currents, swimming, elevation and map presets |
| [Adding species](docs/species.md) | Named prey/predator species, diets, colors and species/sex champions |
| [Animal observations](docs/observations.md) | Visibility, top-k target slots and brain input layouts |
| [Configuration](docs/configuration.md) | Presets, populations, food, obstacles and reproduction |
| [Learning algorithms](docs/algorithms.md) | Q-learning, PPO, A2C, REINFORCE, experimental GRPO and network settings |
| [Saved brains](docs/brain-library.md) | Champions, warm starts, compatibility and model generations |
| [Playback and recordings](docs/recording.md) | Viewer controls, visual agent inspector, recording format and HTTP API |
| [Behavior analysis](docs/analysis.md) | Feeding, hunting, energy, lifespan and learning statistics |
| [Evaluation](docs/evaluation.md) | Matched policies, controlled comparisons and interpretation |
| [Calibration](docs/calibration.md) | Current supported recommendations, scope and tradeoffs |
| [Architecture](docs/architecture.md) | Python/JS components, simulation rules, rewards and observations |
| [Performance](docs/performance.md) | Optimizations, memory tradeoffs and reproducible benchmarks |
| [Releases and rollouts](docs/releases/README.md) | Versioned commit history, compatibility, validation and release maintenance |
| [Project context](docs/project-context.md) | Design decisions, code navigation, agent instructions and reusable workflows |
| [Development](docs/development.md) | Tests and project conventions |

## Project layout

```text
ecosystem/     Python simulation, learning, archives and analysis
viewer/        JavaScript playback engine
environments/  JSON experiment presets
maps/          Editable river, highland and valley layouts
docs/          Detailed guides
tests/         Python and JavaScript checks
```

Generated `results/`, `brain_library/`, `.cache/` and `.venv/` remain local and are
ignored by Git. Observed survival and champion scores are useful diagnostics;
checking learning quality requires matched evaluation on unseen world seeds.
