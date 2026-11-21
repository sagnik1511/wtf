# Building and verifying the Python distribution

[Installation](../INSTALL.md) · [Compatibility](compatibility.md) · [Linux setup](linux.md) · [Customization](customization.md)

The distribution is named `wtf-wildlife-trial-framework`; the Python import is
`ecosystem`. Version `0.1.0` is configured locally. Building and checking it does
not publish it or reserve the name on PyPI. No upload command or publishing
workflow is included in this preparation.

## Build a wheel and source archive

From a source checkout, install build tools into the project environment:

```sh
.venv/bin/python -m pip install build twine
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

The build backend is isolated setuptools with explicit package discovery and an
MIT SPDX license declaration. The default artifacts are:

```text
dist/
├── wtf_wildlife_trial_framework-0.1.0-py3-none-any.whl
└── wtf_wildlife_trial_framework-0.1.0.tar.gz
```

The default wheel is pure Python, portable across supported runtimes and needs
no compiler. The source archive can build the same wheel. Runtime dependencies
remain limited to tqdm. Build/Twine are development tools, not installed runtime
dependencies. The PyPI long description uses `PYPI_README.md`; it does not depend
on checkout-relative image/doc links resolving on a package index.

## What is included

[`packaging_assets.json`](../packaging_assets.json) is the explicit runtime asset
allowlist: maintained environment presets, terrain maps, viewer HTML/JS/CSS and
the configuration schema. The build hook copies these under
`ecosystem/assets/` in the wheel. Source files stay in their existing checkout
locations; asset lookup uses `importlib.resources` with a checkout fallback.

No experiment output, learned models, paper, personal/local JSONs, caches,
virtual environments, host native binaries or documentation GIFs enter the
runtime wheel. `MANIFEST.in` carries the required sources/assets into the source
archive and prunes development/experiment trees. Add a new **shipped** preset,
map or viewer module to the allowlist and test it; local exploration files stay
excluded. Neither Git tracking alone nor a wildcard makes a new resource ship.

## Installed usage without a checkout

Create an independent environment and install the wheel, then change to a
writable experiment directory:

```sh
python3 -m venv /path/to/test-env
/path/to/test-env/bin/python -m pip install /path/to/wtf/dist/wtf_wildlife_trial_framework-0.1.0-py3-none-any.whl
mkdir -p /path/to/experiments
cd /path/to/experiments
/path/to/test-env/bin/wtf presets
/path/to/test-env/bin/wtf run day_night --runs 1 --steps 200 --seed 100 --progress
/path/to/test-env/bin/wtf view --results results
```

`wtf run`, `wtf view`, `wtf analyze` dispatch to the existing parsers. Aliases
`wtf-run`, `wtf-view`, `wtf-analyze` and `python -m ecosystem` are also provided.
A simulator environment can be a bundled name (`ppo`), its historical
`environments/ppo.json` spelling, or an existing custom JSON path. Existing
local files take precedence. Relative external map paths resolve in the current
working directory before bundled/project assets. Outputs and default shared
library paths are relative to the working directory, never site-packages.

In Python:

```python
from ecosystem import Experiment, load_config

config = load_config("maskable_ppo")
config["world"]["max_steps"] = 100
config["experiment"].update(runs=1, world_seed_policy="increment")
experiment = Experiment(config)
while not experiment.done:
    experiment.step()
```

The small Python API above does not record automatically. Use the CLI or the
[complete recorder tutorial](tutorial.md) when you want saved playback and
champion exports. Templates and actions retain their normal compatibility rules.

## Optional native source build

On supported CPython 3.12–3.13 with a compiler and matching headers:

```sh
WTF_BUILD_NATIVE=1 python -m pip install --no-cache-dir --force-reinstall \
  /path/to/wtf_wildlife_trial_framework-0.1.0.tar.gz
```

Opt-in compilation uses the same strict floating-point flags as the reference
native build. A requested native build fails clearly on unsupported runtimes;
omit the flag to install the portable Python distribution. Do not ship a wheel
containing a binary copied from another machine. Prebuilt native wheels across
all platforms are not part of this initial packaging setup.

## Repeat the isolated installation check

The checkout tool below is intended to run through the **installed** environment,
from an empty writable directory outside the checkout:

```sh
cd /path/to/empty-check-directory
/path/to/test-env/bin/python -I /path/to/wtf/work/check_installation.py
```

It verifies the installed package origin, console scripts, every bundled preset,
a five-step training run, checkpoint restoration, replay diagnostics/chunks and
the viewer's static/HTTP endpoints. It starts a temporary localhost server on an
automatically assigned port, then closes it. It writes a small check report and
an experiment in that check directory; use temporary directories for repeated
checks. No Docker execution or publication is involved.

Run the same check after installing the source archive in another clean venv.
`-I` prevents checkout/PYTHONPATH imports from concealing missing package files.
Inspect distribution file lists as well as successful imports: local results or
host `.so` files must never leak into a portable wheel.

The Ubuntu workflow in [package.yml](../.github/workflows/package.yml) repeats
wheel/source checks on Python 3.12, 3.13 and 3.14 and opt-in native builds on
3.12/3.13. Native jobs run the full Python suite with both backend selections.
Those jobs execute only when CI runs; a locally prepared workflow is not evidence
of an Ubuntu pass. Keep platform support claims aligned with actual CI results.

## Current local verification

Python 3.12 is the minimum declared in distribution metadata. Fresh virtual
environments must use 3.12 or newer, including source-archive builds. Optional
native kernels are validated for CPython 3.12–3.13; other supported interpreters
use the portable backend. Local verification results for this migration are
recorded in [rollout 0.0.1.31](releases/0.0.1.31.md). Ubuntu jobs are configured;
their execution must be confirmed on the pushed commit.

The packaging jobs pin **Ubuntu 24.04 x64** rather than `ubuntu-latest`. The
Selected Python versions need interpreter builds for that exact runner OS; a
moving latest image may advance before compatible builds are provided. Native
jobs use the same pinned OS/architecture. Review GitHub's
[Python build manifest](https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json)
before changing this pin. A setup-python download failure occurs before package
installation tests; it is not evidence of a simulator/runtime failure.
