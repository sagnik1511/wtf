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

## Brains belong to their run

Each completed training run saves its own champions. There is no need to track
which separate model-library generation contains a result:

```text
results/<run-name>/champions/run_0001.json
results/<run-name>/champions/run_0001/manifest.json
results/<run-name>/champions/run_0001/prey/{health,kills,lifespan,food}.json
results/<run-name>/champions/run_0001/predator/{health,kills,lifespan,food}.json
```

The experiment folder is self-contained. Moving it keeps its models usable.
Experiments with several numbered runs keep separate `run_0002`, `run_0003`, etc.
By default loading chooses the latest archived numbered run; `--brain-run` pins
one. Results and optional legacy shared libraries remain ignored by Git.

List the available champions by run name:

```sh
.venv/bin/python -m ecosystem.library --run PREVIOUS_RUN_NAME
```

The model snapshots contain weights and provenance, with no replay or optimizer
buffers. Tracking retains at most four models per species and performs no disk
writes inside the simulation-step loop. Models are published atomically when
the numbered run completes.

## Choose models with CLI arguments

Reuse champions from a specific saved experiment without changing its environment JSON:

```sh
.venv/bin/python -m ecosystem.cli environments/ppo.json --runs 1 \
  --brains-from PREVIOUS_PPO_RUN_NAME \
  --brain prey=food --brain predator=kills
```

Replace the source name with its experiment folder name under project `results/`.
An explicit experiment path also works for a custom results directory.
`--brains-from` defaults to its latest archived run, with food champions for
resource eaters and kill champions for predators. Add `--brain-run 1` to pin a
particular run. Each selected model must match the target species and architecture.
The source artifacts stay unchanged, and the new experiment saves the resolved
paths in `environment.json` and their provenance in `manifest.json`.

Repeat `--brain SPECIES=MODEL` to select any configured species. `MODEL` accepts:

| Value | Selection |
| --- | --- |
| `health`, `kills`, `lifespan`, `food` | That champion from `--brains-from` |
| `best:food`, `best:kills`, etc. | Best compatible champion across the library |
| `latest:lifespan`, etc. | Criterion from the latest compatible library generation |
| Path to an artifact JSON | Exact saved model |
| `fresh` | Fresh initialization for that species, overriding any JSON warm start |

For example, choose different per-run champions or keep one species fresh:

```sh
.venv/bin/python -m ecosystem.cli environments/ppo.json --runs 1 \
  --brains-from PREVIOUS_PPO_RUN_NAME --brain prey=lifespan --brain predator=health
.venv/bin/python -m ecosystem.cli environments/ppo.json --runs 1 \
  --brains-from PREVIOUS_PPO_RUN_NAME --brain prey=fresh
```

Unknown species, duplicate selections, missing artifacts and incompatible models
fail before a new results folder is created. These options cannot be combined
with `--resume`, which restores the exact saved experiment instead.

## Optional JSON defaults

Explicit JSON paths are still supported. Earlier shared-library selectors also remain available for existing configurations:

```json
"initial_brain": {"selection": "best", "metric": "food"}
```

For predators, use `"metric": "kills"` or `"lifespan"`; `"health"` is available
for either species. `"selection": "latest"` uses that criterion's model from
the most recent compatible archived generation. Selectors are resolved when an
experiment starts, and each agent receives an independent copy. To pin a
specific generation, use the path printed by the library command:

```json
"initial_brain": {"path": "results/PREVIOUS_RUN_NAME/champions/run_0001/prey/food.json"}
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
# Train, then use the result folder name printed by the first command.
.venv/bin/python -m ecosystem.cli environments/ppo.json --runs 1
.venv/bin/python -m ecosystem.cli environments/ppo.json --runs 1 \
  --brains-from PREVIOUS_PPO_RUN_NAME --brain prey=food --brain predator=kills
```

The older `champion_lineage.json` preset uses shared-library selectors for a prey
`food` champion and a predator `kills` champion. The default workflow now uses
`--brains-from RUN_NAME` and per-species `--brain` criteria.
With no JSON or CLI initialization request, agents start fresh. `--library-dir PATH`
opts into an additional shared archive and selects the shared search location
for legacy `best:`/`latest:` selectors. Run-local models are still saved. Explicit
JSON paths remain relative to the project root. `--no-library` skips publishing new
generations. Evaluation and random-control runs do not publish champion models.


## Older shared archives

Older run indexes that point into `brain_library/` continue to load. To make one
self-contained, copy its existing models into the run with:

```sh
.venv/bin/python -m ecosystem.library --run PREVIOUS_RUN_NAME --localize
```

This preserves model bytes and artifact identities and keeps the original shared
copies. It can be repeated safely. The four existing archived experiments were
localized when this workflow was introduced, including
`ppo_20261115T044556Z_889391`.
