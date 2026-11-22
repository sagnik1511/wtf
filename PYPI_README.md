# WTF — Wildlife Trial Framework

A Python ecosystem simulator with independently learning animals, multiple named
species, configurable JSON worlds, terrain, rivers, day/night cycles and paired
reproduction. A bundled JavaScript viewer replays saved experiments separately
from training. It requires no Node.js build step.

Supported learners include Q-network/DQN, PPO, A2C, REINFORCE, experimental GRPO,
recurrent PPO, QR-DQN, masked PPO and MAPPO. Current source additionally implements
Discrete SAC, TRPO and LSTM PPO; these additions are not in published 0.0.2.9.
Controllers, optimizers and RNGs are
private to each animal. These are experimental implementations, not claims of
biological realism or established policy improvement.

Requires Python **3.12 or newer**. Optional native kernels support CPython
3.12–3.13; newer interpreters use the Python backend.

## Install from PyPI

Use Python 3.12+ in a virtual environment:

```sh
python3.12 -m venv .venv
# Activate .venv using your platform's shell command.
python -m pip install wtf-wildlife-trial-framework==0.0.2.9
```

The distribution name is `wtf-wildlife-trial-framework`; the Python import is
`wtf`. Version [0.0.2.9](https://pypi.org/project/wtf-wildlife-trial-framework/0.0.2.9/)
is published with a portable wheel and source archive. The default wheel needs
no compiler and includes presets, terrain maps, schema and viewer assets.

## Training without playback, then recorded inference

```sh
wtf run ppo --runs 10 --recording-profile training-only --progress
wtf run results/TRAINING_RUN/environment.json \
  --brains-from results/TRAINING_RUN --mode evaluate --freeze-brains \
  --recording-profile full --runs 1 --progress
```

Replace `TRAINING_RUN` with the training output folder. Training-only retains
metrics, best brains and recovery checkpoints. Frozen inference disables learning,
mutation and birth crossover; offspring copy their same-sex parent's model.
Evaluation recordings describe fresh simulations, not the original training runs.
Use `--recording-profile compact` to keep training world playback with fewer
inspector samples and no activations. Full configured capture remains the default.

## Local package installation

To install your own checkout build instead, build and install its wheel:

```sh
python -m pip install build
python -m build --outdir dist/0.0.2.9
python -m pip install dist/0.0.2.9/wtf_wildlife_trial_framework-0.0.2.9-py3-none-any.whl
```

Run from any writable working directory:

```sh
wtf presets
wtf run maskable_ppo --runs 1 --steps 200 --seed 100 --progress
wtf view --results results
```

Open http://127.0.0.1:8000 and load the generated recording. Maintained presets,
terrain maps, schema and viewer assets are included. Results default to the
current directory's `results/`; they are not written into the installed package.
The default wheel is pure Python. Native CPU kernels are an optional source-build
choice for supported CPython versions; they are not required to install or run.

Use `wtf --help`, `wtf run --help`, `wtf view --help` and `wtf analyze --help`
for commands. `python -m wtf` provides the same command hub.

See the [documentation](https://sagnik1511.github.io/wtf/) and
[source repository](https://github.com/sagnik1511/wtf) for
installation, Docker, complete tutorials, all configuration definitions, species,
policy/action/geography extension guidance, storage and artifact compatibility.

MIT License. Copyright (c) 2026 Sagnik Roy.
