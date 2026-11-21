# WTF — Wildlife Trial Framework

A Python ecosystem simulator with independently learning animals, multiple named
species, configurable JSON worlds, terrain, rivers, day/night cycles and paired
reproduction. A bundled JavaScript viewer replays saved experiments separately
from training. It requires no Node.js build step.

Supported learners include Q-network/DQN, PPO, A2C, REINFORCE, experimental GRPO,
recurrent PPO, QR-DQN, masked PPO and MAPPO. Controllers, optimizers and RNGs are
private to each animal. These are experimental implementations, not claims of
biological realism or established policy improvement.

Requires Python **3.12 or newer**. Optional native kernels support CPython
3.12–3.13; newer interpreters use the Python backend.

## TestPyPI trial installation

Use Python 3.12+ in a virtual environment. Install the runtime dependency from
PyPI, then install the trial package from TestPyPI without mixing package indexes:

```sh
python -m venv .venv
# Activate .venv using your platform's shell command.
python -m pip install "tqdm>=4.66,<5"
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps wtf-wildlife-trial-framework==0.1.0
```

TestPyPI is a separate trial index. Availability there does not mean the package
has been published on the production PyPI index.

## Local package installation

For an unpublished checkout, build distributions and install the resulting wheel:

```sh
python -m pip install build
python -m build
python -m pip install dist/wtf_wildlife_trial_framework-0.1.0-py3-none-any.whl
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
for commands. `python -m ecosystem` provides the same command hub.

See the [documentation](https://sagnik1511.github.io/wtf/) and
[source repository](https://github.com/sagnik1511/wtf) for
installation, Docker, complete tutorials, all configuration definitions, species,
policy/action/geography extension guidance, storage and artifact compatibility.

MIT License. Copyright (c) 2026 Sagnik Roy.
