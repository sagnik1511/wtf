# Package layout and import migration

[Development](development.md) · [Architecture](architecture.md) · [Compatibility](compatibility.md)

The Python namespace is **`wtf`**. The installable distribution remains
**`wtf-wildlife-trial-framework`**; these are different names for importing and
installing the same project. Do not install an unrelated registry project named
only `wtf`. Python 3.12+ is still required.

Use the [source call/caller browser](source-graphs.md) to follow implementation
relationships and inspect the original source lines for each component.

## Start with the public API

```python
from wtf import Experiment, load_config

config = load_config("day_night")
config["world"]["max_steps"] = 100
config["experiment"].update(runs=1, world_seed_policy="increment")
experiment = Experiment(config)
while not experiment.done:
    experiment.step()
```

Importing `wtf` alone keeps training modules unloaded. Accessing `Experiment`
loads the experiment implementation on demand. The CLI records replay and
reports automatically; the small API example above only advances the experiment.

```sh
.venv/bin/python -m wtf run day_night --runs 1 --steps 1000 --seed 100 --progress
.venv/bin/python -m wtf view --results results
.venv/bin/python -m wtf analyze --help
```

`python -m wtf.cli` and `python -m wtf.server` remain short module entry points.
Installed commands are `wtf`, `wtf-run`, `wtf-view` and `wtf-analyze`.

## Responsibilities and boundaries

![The wtf root exposes commands and a lazy public API, with simulation, learning, experiment, storage, inspection and native subpackages](assets/structures/package-layout.svg)

| Package | Owns | Main modules |
| --- | --- | --- |
| `wtf` | Public API, configuration, resource lookup and command entry points | `config`, `resources`, `commands`, `cli`, `server`, `progress` |
| `wtf.simulation` | Bodies, physical actions, terrain, visibility and placement | `world`, `actions`, `geography`, `perception`, `placement`, `day_night` |
| `wtf.learning` | Independent models, updates, inheritance and activation samples | `brain`, `brains`, `ppo`, `policy_gradients`, `recurrent_ppo`, `qrdqn`, `contextual_ppo`, `grouped_rollouts`, `optimization`, `mutation`, `activation_diagnostics` |
| `wtf.experiment` | Episode scheduling, feedback, family ranking and plateau stopping | `runner`, `lineage`, `early_stopping` |
| `wtf.storage` | Artifact encoding, persistence, champion libraries and storage conversion | `binary_io`, `checkpoint_io`, `model_io`, `replay_io`, `replay_codec`, `recording`, `library`, `compress_recordings`, `optimize_storage`, `optimize_checkpoints` |
| `wtf.inspection` | Read-only behavior, telemetry and archived-model explanations | `analyze`, `telemetry`, `explainability`, `explanations` |
| `wtf.native` | Numeric buffer helpers, optional compiled kernels and Python fallback selection | `numeric`, `kernels`, private `_kernels` extension and headers |

The subpackages organize ownership; they do not introduce a new plugin registry,
shared species policy or live-training dependency in playback. JavaScript still
lives in `viewer/`, presets in `environments/` and terrain definitions in `maps/`.
Installed read-only runtime assets are bundled under `wtf/assets/`. Results,
checkpoints and learned-model directories remain writable working-directory data.

## Migrate existing imports

The former `ecosystem` namespace is no longer shipped. Update external scripts
using the table below; no compatibility aliases duplicate modules or class state.
Historical release notes retain the import names used at their source revisions.

| Previous import | Current import |
| --- | --- |
| `from ecosystem import Experiment, load_config` | `from wtf import Experiment, load_config` |
| `ecosystem.actions` | `wtf.simulation.actions` |
| `ecosystem.activation_diagnostics` | `wtf.learning.activation_diagnostics` |
| `ecosystem.analyze` | `wtf.inspection.analyze` |
| `ecosystem.binary_io` | `wtf.storage.binary_io` |
| `ecosystem.brain` | `wtf.learning.brain` |
| `ecosystem.brains` | `wtf.learning.brains` |
| `ecosystem.checkpoint_io` | `wtf.storage.checkpoint_io` |
| `ecosystem.cli` | `wtf.cli` |
| `ecosystem.commands` | `wtf.commands` |
| `ecosystem.compress_recordings` | `wtf.storage.compress_recordings` |
| `ecosystem.config` | `wtf.config` |
| `ecosystem.contextual_ppo` | `wtf.learning.contextual_ppo` |
| `ecosystem.day_night` | `wtf.simulation.day_night` |
| `ecosystem.early_stopping` | `wtf.experiment.early_stopping` |
| `ecosystem.explainability` | `wtf.inspection.explainability` |
| `ecosystem.explanations` | `wtf.inspection.explanations` |
| `ecosystem.geography` | `wtf.simulation.geography` |
| `ecosystem.grouped_rollouts` | `wtf.learning.grouped_rollouts` |
| `ecosystem.kernels` | `wtf.native.kernels` |
| `ecosystem.library` | `wtf.storage.library` |
| `ecosystem.lineage` | `wtf.experiment.lineage` |
| `ecosystem.model_io` | `wtf.storage.model_io` |
| `ecosystem.mutation` | `wtf.learning.mutation` |
| `ecosystem.numeric` | `wtf.native.numeric` |
| `ecosystem.optimization` | `wtf.learning.optimization` |
| `ecosystem.optimize_checkpoints` | `wtf.storage.optimize_checkpoints` |
| `ecosystem.optimize_storage` | `wtf.storage.optimize_storage` |
| `ecosystem.perception` | `wtf.simulation.perception` |
| `ecosystem.placement` | `wtf.simulation.placement` |
| `ecosystem.policy_gradients` | `wtf.learning.policy_gradients` |
| `ecosystem.ppo` | `wtf.learning.ppo` |
| `ecosystem.progress` | `wtf.progress` |
| `ecosystem.qrdqn` | `wtf.learning.qrdqn` |
| `ecosystem.recording` | `wtf.storage.recording` |
| `ecosystem.recurrent_ppo` | `wtf.learning.recurrent_ppo` |
| `ecosystem.replay_codec` | `wtf.storage.replay_codec` |
| `ecosystem.replay_io` | `wtf.storage.replay_io` |
| `ecosystem.resources` | `wtf.resources` |
| `ecosystem.runner` | `wtf.experiment.runner` |
| `ecosystem.server` | `wtf.server` |
| `ecosystem.telemetry` | `wtf.inspection.telemetry` |
| `ecosystem.world` | `wtf.simulation.world` |

## Reinstall entry points and rebuild native kernels

From a checkout, refresh the current environment's installed command scripts:

```sh
.venv/bin/python -m pip install -e .
.venv/bin/wtf --help
```

For a built wheel, reinstall the newly built artifact in the intended environment;
source changes alone do not update an older installed wheel. Restart Python and
viewer server processes after upgrading so they do not retain old imports.

Rebuild optional native kernels at their new location:

```sh
.venv/bin/python -m work.build_native
WTF_KERNELS=native .venv/bin/python -c 'from wtf.native.kernels import status; print(status())'
```

The extension is `wtf.native._kernels`; do not copy a binary from the old package
or another interpreter/machine. `WTF_KERNELS=python` selects the reference path.
No GPU, precision, learning, RNG, physics or replay-format changes accompany this
namespace/layout migration.

## Existing experiment data

Checkpoint, champion and replay formats use structured records, not Python
module pickle names. Existing supported files retain their decoders and model
compatibility rules; no file conversion or results-directory rename is required.
Exact continuation was compared with pre-migration fixtures for every supported
learner and a mixed-policy case under the same Python/native backend. This does
not extend numerical guarantees across Python versions or changed configurations.
See the [compatibility policy](compatibility.md) and [checkpoint structures](checkpoint-data-structures.md).

## Where to make changes

Follow [policy extensions](extending-policies.md), [species configuration](species.md)
and [world/action extensions](extending-worlds-and-actions.md). Their code examples
and registration paths use the current subpackages. Keep numerical kernels,
persistence signatures, offspring initialization and viewer diagnostics aligned
when changing a shared contract. New runtime assets still require the
[packaging allowlist](../packaging_assets.json).

The optional collective-sensing sidecar lives in `simulation/social.py`; the
bounded empirical outcome model and its import/export live in
`learning/consequences.py`. Neither owns or mutates private policy parameters.
See [collective learning](social-learning.md).
