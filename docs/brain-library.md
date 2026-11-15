# Saved champions and model generations

[Project overview](../README.md) · [Configuration](configuration.md) · [Runs and checkpoints](running.md)

Commands in this guide run from the project root.

Training automatically tracks four independent champions for every species:
`health` (highest HP), `kills` (most individual kills), `lifespan` (oldest age
reached) and `food` (most actual energy gained from food or prey). Prey cannot
kill under the current rules, so their kill score remains zero; use `food` for
a useful foraging champion. These are observed records, not validated estimates
of a model's quality on other worlds.

Champions capture the model at the step the record is set, after that step's
learning. Dead animals are considered before their controllers are discarded.
A previously healthy champion keeps the weights from its healthy moment; its
later starvation cannot overwrite that snapshot. Equal primary scores prefer
older animals, so an initially healthy untrained founder does not permanently
lock the health record. Kills and food gained are tracked per lifetime, and
newborns start those counters at zero. Raw HP is used for health, and age reached
also includes living survivors rather than treating their age as a completed
lifespan. The exact tie-breaking ranks are included with each model.

At each completed training run, compact artifacts are published atomically to:

```text
brain_library/gen_<timestamp>_<id>/<species>/health.json
brain_library/gen_<timestamp>_<id>/<species>/kills.json
brain_library/gen_<timestamp>_<id>/<species>/lifespan.json
brain_library/gen_<timestamp>_<id>/<species>/food.json
```

Every generation has its own immutable directory and `manifest.json`. A
run-local index is saved under `results/<experiment>/champions/run_0001.json`.
Both the library and results are ignored by Git. No model or replay files are
loaded by the viewer server. In-memory champion storage is bounded by four
models per species; snapshots omit replay, episode buffers and optimizer state.
Library publication adds no disk writes inside the simulation step loop.

To list models and their paths:

```sh
.venv/bin/python -m ecosystem.library
```

Configure a species to load the best compatible saved model:

```json
"initial_brain": {"selection": "best", "metric": "food"}
```

For predators, use `"metric": "kills"` or `"lifespan"`; `"health"` is available
for either species. `"selection": "latest"` uses that criterion's model from
the most recent compatible archived generation. Selectors are resolved when an
experiment starts, and each agent receives an independent copy. To pin a
specific generation, use the path printed by the library command:

```json
"initial_brain": {"path": "brain_library/gen_<timestamp>_<id>/prey/food.json"}
```

Relative paths resolve from the project root, regardless of the working
directory. Files must match species, algorithm, network shape, observation
version and action order. A Q model cannot initialize a PPO or GRPO controller.
Malformed or non-finite weights are rejected. A selector with no compatible
model fails clearly; it never silently substitutes a fresh brain.

Each newly initialized controller gets fresh body state, empty experience
buffers, fresh optimizer state and random streams, and the exploration settings
from its JSON.
Q target weights and GRPO's frozen reference policy start at the imported model.
Learning rates and other training settings can be changed without changing the
architecture. Ordinary `retain_brains_between_runs` still takes precedence for
later runs in the same experiment. With retention disabled, each run reloads
copies of the model selected at experiment startup. Exact checkpoints embed the
selected weights and champion book, so resuming does not need the original
library files and is unaffected by newer library generations.

Artifacts record scores, world seed, run, tick, animal ID, body generation,
training configuration and source artifact. `model_generation` counts warm-start
ancestry depth, separately from reproduction generations. Subsequent retained
runs also record their prior run number. Old archive directories stay available
for comparisons and branching new lineages. Legacy checkpoints that lack
individual kill/food counters begin counting those statistics at the resumed
age; `statistics_since_age` makes that limitation explicit. Past model weights
that were never saved cannot be reconstructed from a playback recording.

```sh
# First produce compatible Q champions, then train their next model generation.
.venv/bin/python -m ecosystem.cli environments/baseline.json --runs 1
.venv/bin/python -m ecosystem.cli environments/champion_lineage.json --runs 1
```

The lineage preset uses a prey `food` champion and a predator `kills` champion.
Omitting `initial_brain` preserves fresh initialization. `--library-dir PATH`
chooses another archive location for saving and automatic selectors; explicit
paths remain relative to the project root. `--no-library` skips publishing new
generations. Evaluation and random-control runs do not publish champion models.
