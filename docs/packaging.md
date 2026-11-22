# Building and verifying the Python distribution

[Installation](../INSTALL.md) · [Compatibility](compatibility.md) · [Linux setup](linux.md) · [Customization](customization.md)

The distribution is named `wtf-wildlife-trial-framework`; the Python import is
`wtf`. Version [0.0.2.9 is published on PyPI](https://pypi.org/project/wtf-wildlife-trial-framework/0.0.2.9/)
with a portable wheel and source archive.
Future builds/checks do not publish automatically. **TestPyPI** has separate
accounts, tokens and projects; see [the upload procedure](#testpypi-trial-release).

## Build a wheel and source archive

From a source checkout, install build tools into the project environment:

```sh
.venv/bin/python -m pip install build twine
.venv/bin/python -m build --outdir dist/0.0.2.9
.venv/bin/python -m twine check dist/0.0.2.9/*
```

The build backend is isolated setuptools with explicit package discovery and an
MIT SPDX license declaration. The default artifacts are:

```text
dist/0.0.2.9/
├── wtf_wildlife_trial_framework-0.0.2.9-py3-none-any.whl
└── wtf_wildlife_trial_framework-0.0.2.9.tar.gz
```

The default wheel is pure Python, portable across supported runtimes and needs
no compiler. The source archive can build the same wheel. Runtime dependencies
remain limited to tqdm. Build/Twine are development tools, not installed runtime
dependencies. The PyPI long description uses `PYPI_README.md`; it does not depend
on checkout-relative image/doc links resolving on a package index.

## What is included

All `wtf` subpackages are discovered explicitly with the `wtf.*` package pattern; the native source/headers belong to `wtf.native`. See [package layout](package-layout.md).

[`packaging_assets.json`](../packaging_assets.json) is the explicit runtime asset
allowlist: maintained environment presets, terrain maps, viewer HTML/JS/CSS and
the configuration schema. The build hook copies these under
`wtf/assets/` in the wheel and supplies the same allowlist to setuptools' source
collector for the source archive. Source files stay in their existing checkout
locations; asset lookup uses `importlib.resources` with a checkout fallback.

No experiment output, learned models, paper, personal/local JSONs, caches,
virtual environments, host native binaries or documentation GIFs enter the
runtime wheel. `MANIFEST.in` includes build metadata/native sources and prunes
development/experiment trees; runtime assets are not enumerated a second time.
Missing or out-of-checkout allowlisted files fail the build with their path. Add a new **shipped** preset,
map or viewer module to the allowlist and test it; local exploration files stay
excluded. Neither Git tracking alone nor a wildcard makes a new resource ship.

Validate the complete source-archive → wheel path with `python -m build`, rather
than only `python -m build --wheel` against a checkout. The packaging regression
check verifies every allowlisted source-archive member against its original
bytes; installed smoke checks also request the competition viewer module. This
catches a preset or JS dependency omitted from the source archive before CI
tries to rebuild its wheel.

## Installed usage without a checkout

Create an independent environment and install the wheel, then change to a
writable experiment directory:

```sh
python3 -m venv /path/to/test-env
/path/to/test-env/bin/python -m pip install /path/to/wtf/dist/0.0.2.9/wtf_wildlife_trial_framework-0.0.2.9-py3-none-any.whl
mkdir -p /path/to/experiments
cd /path/to/experiments
/path/to/test-env/bin/wtf presets
/path/to/test-env/bin/wtf run day_night --runs 1 --steps 200 --seed 100 --progress
/path/to/test-env/bin/wtf view --results results
```

`wtf run`, `wtf view`, `wtf analyze` dispatch to the existing parsers. Aliases
`wtf-run`, `wtf-view`, `wtf-analyze` and `python -m wtf` are also provided.
A simulator environment can be a bundled name (`ppo`), its historical
`environments/ppo.json` spelling, or an existing custom JSON path. Existing
local files take precedence. Relative external map paths resolve in the current
working directory before bundled/project assets. Outputs and default shared
library paths are relative to the working directory, never site-packages.

In Python:

```python
from wtf import Experiment, load_config

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
  /path/to/wtf_wildlife_trial_framework-0.0.2.9.tar.gz
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

## TestPyPI trial release

Build and validate a dedicated artifact directory so a trial upload cannot pick
up stale/native distributions from another build:

```sh
WTF_BUILD_NATIVE=0 .venv/bin/python -m build --outdir dist/testpypi-0.0.2.9
.venv/bin/python -m twine check --strict dist/testpypi-0.0.2.9/*
```

Inspect the wheel and source archive, test each in a fresh environment with
`work/check_installation.py`, and run the project checks before uploading.
Keep results, local research, credentials and host native binaries out of both
archives. Publish only the named version's two reviewed artifacts:

```sh
.venv/bin/python -m twine upload --repository-url https://test.pypi.org/legacy/ \
  dist/testpypi-0.0.2.9/wtf_wildlife_trial_framework-0.0.2.9-py3-none-any.whl \
  dist/testpypi-0.0.2.9/wtf_wildlife_trial_framework-0.0.2.9.tar.gz
```

Authenticate to TestPyPI with the `__token__` username and a TestPyPI API token
through Twine's password prompt or a locally configured keyring. Do not put a
token in a shell command, repository file, build log or chat. First-time project
creation may require an account-scoped token; after creation, prefer a token
scoped to this project. TestPyPI accounts/tokens are distinct from PyPI. See the
[official TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/).

Once uploaded, check the project page and install the exact trial from the index
in another fresh Python 3.12+ environment:

```sh
python -m pip install "tqdm>=4.66,<5"
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps \
  wtf-wildlife-trial-framework==0.0.2.9
```

Install dependencies from PyPI separately rather than using `--extra-index-url`.
Run the same installed-package smoke check against this download, including its
viewer assets, maps, presets and checkpoint/replay readers. Record the artifact
hashes and actual upload/installation outcome. A wheel validated locally is not
an uploaded release; keep status Pending until the index confirms the files.
Avoid reusing a version/filename after upload: indexes treat release files as
immutable. A TestPyPI trial does not authorize a production PyPI upload.
