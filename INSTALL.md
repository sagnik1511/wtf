# Install Wildlife Trial Framework

[Project overview](README.md) · [Python 3.12 rollout](docs/releases/0.0.1.31.md) · [Complete tutorial](docs/tutorial.md)

Choose a built Python wheel, a source checkout or Docker. Rollout **0.0.1.30**
adds package builds. TestPyPI trial uploads and production PyPI publication are
separate steps; see the [trial release procedure](docs/packaging.md#testpypi-trial-release). The distribution name is
`wtf-wildlife-trial-framework`, and the Python import is `wtf`.

For renamed imports and module locations, see [package layout and migration](docs/package-layout.md).

## Install from PyPI

Version [0.0.2.0](https://pypi.org/project/wtf-wildlife-trial-framework/0.0.2.0/) is available from production PyPI.
In a fresh working directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install wtf-wildlife-trial-framework==0.0.2.0
.venv/bin/wtf presets
.venv/bin/wtf run baseline --runs 1 --steps 200 --seed 100 --progress
.venv/bin/wtf view --results results
```

The portable wheel includes presets, maps and viewer assets; no compiler is needed.
For the optional native extension, use the [native build instructions](docs/packaging.md#optional-native-source-build).

## Install the built wheel

Build from the checkout in its development environment, then install the wheel
into a fresh environment:

```sh
.venv/bin/python -m pip install build twine
.venv/bin/python -m build --outdir dist/0.0.2.0
.venv/bin/python -m twine check dist/0.0.2.0/*
python3.12 -m venv /path/to/test-env
/path/to/test-env/bin/python -m pip install dist/0.0.2.0/wtf_wildlife_trial_framework-0.0.2.0-py3-none-any.whl
```

From any writable working directory, use `wtf presets`,
`wtf run maskable_ppo --runs 1 --steps 200 --seed 100 --progress`, then
`wtf view --results results` (activate that environment or use its full script
paths). Presets, maps and viewer assets are included. See
[package verification](docs/packaging.md) and [Linux/Ubuntu setup](docs/linux.md).
Do not confuse installing this wheel with publishing it, or substitute a package
named only `wtf` from a registry.

## Local Python installation

The project declares Python 3.12+. Python 3.12 matches the Docker runtime and is
suitable for the optional native backend. A modern browser runs the viewer;
Node.js, npm, PyTorch, NumPy and a GPU are not runtime requirements.

On macOS or Linux:

```sh
cd /path/to/wtf
python3.12 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements.txt
.venv/bin/python -m wtf.cli --help
```

Keep subsequent Python packages inside `.venv`. The only current runtime
dependency is `tqdm`; a compiler is unnecessary for the Python fallback.

Run one short experiment, then start the offline viewer:

```sh
.venv/bin/python -m wtf.cli environments/maskable_ppo.json \
  --runs 1 --steps 200 --seed 100 --progress
./run.sh
```

Open <http://127.0.0.1:8000>, choose the saved experiment and click
**Load recording**, then **Play recording**. Training and playback are separate
processes. The 200-step run is an installation check; increase the duration for
behavior evaluation and reproduction maturity. Press Ctrl+C to stop the viewer.
If port 8000 is occupied, use `./run.sh --port 8001`.

The launcher can create `.venv` but does not install `requirements.txt` for you.
You can also start playback directly:

```sh
.venv/bin/python -m wtf.server --results results --port 8000
```

On Windows PowerShell, use the same CLI with the virtual environment executable:

```powershell
cd C:\path\to\wtf
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m wtf.cli environments/maskable_ppo.json --runs 1 --steps 200 --seed 100 --progress
.\.venv\Scripts\python.exe -m wtf.server --results results --port 8000
```

Use the Python fallback where a native build is unavailable; the shell launchers
are optional on Windows.

## Optional native CPU kernels

On a supported CPython/compiler setup, install the build dependency into `.venv`
and build from the checkout root:

```sh
.venv/bin/python -m pip install setuptools
.venv/bin/python -m work.build_native
.venv/bin/python -c "from wtf.native.kernels import status; print(status())"
```

Current native dispatch supports CPython 3.12–3.13. Other supported Python
runtimes use the reference path. Imports do not invoke a compiler. Native files
are specific to Python, OS and architecture; rebuild rather than copying them
between machines. See [native kernels](docs/native-kernels.md) for compiler
requirements, exactness constraints and explicit fallback selection.

## Docker installation

Install Docker with its Compose plugin, then from the checkout:

```sh
sh docker/wtf.sh build
sh docker/wtf.sh train environments/maskable_ppo.json --runs 1 --steps 200 --seed 100 --progress
sh docker/wtf.sh view
```

Open <http://127.0.0.1:8000>. The launcher maps the host user and stores results
under the checkout's `results/`; the viewer mounts them read-only. Training is
not started automatically by the viewer service. Use `sh docker/wtf.sh view 8001`
for another port, and `sh docker/wtf.sh down` to remove Compose containers/network
without deleting results. With `WTF_BUILD_NATIVE=0`, the build skips compiler
installation and uses the Python backend. Details, direct Compose commands and
custom configurations are in the [Docker guide](docs/docker.md).

Docker files are supplied for this rollout, but images/containers were not built
or run during its preparation. Validate the build, networking and mounted output
on your Docker host before relying on that deployment path.

## Files and next steps

Runtime outputs belong in ignored `results/`, outside the source/configuration
files. Keep a complete experiment directory when transferring it, including
referenced champion model/config pools. Installing an updated source checkout
does not convert saved artifacts; consult the [compatibility policy](docs/compatibility.md).

- [Complete Markdown tutorial](docs/tutorial.md): configuration, all example code,
  recording, analysis and checkpoint restoration.
- [Run and resume guide](docs/running.md): full CLI, seeds and multi-run handling.
- [Viewer appearance](docs/viewer-visual-reference.md): objects, themes, sight,
  brain diagnostics and optional simulated day/night.
- [Customization handbook](docs/customization.md): all setup controls, complete
  field/command definitions and species/policy extension workflows.
- [Development setup](docs/development.md): formatters and tests; these are not
  needed just to run the simulator or viewer.
