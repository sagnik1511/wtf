# Linux and Ubuntu setup

[Installation](../INSTALL.md) · [Python package verification](packaging.md) · [Docker](docker.md)

The Python runtime, viewer server and browser assets have no macOS dependency.
Paths use pathlib, launchers use POSIX shell, and presets/maps ship inside the
wheel. The optional macOS/Xcode discovery code is guarded by
`sys.platform == "darwin"`; Linux builds use their normal compiler/sysconfig.
No display server is needed for training or serving saved replay data.

## Portable Python installation

On Ubuntu 24.04 (Python 3.12), install the interpreter's venv support if absent:

```sh
sudo apt-get update
sudo apt-get install python3-venv python3-pip
python3 -m venv .venv
.venv/bin/python -m pip install wtf-wildlife-trial-framework==0.0.2.9
.venv/bin/wtf run maskable_ppo --runs 1 --steps 200 --seed 100 --progress
.venv/bin/wtf view --results results
```

The portable release is available from [PyPI](https://pypi.org/project/wtf-wildlife-trial-framework/0.0.2.9/).
You can also install a locally built wheel using its path. An Ubuntu
Python newer than the native backend's supported range intentionally uses the
Python fallback; a compiler is not required for the portable wheel. Python 3.12+
is the declared package range. On older Ubuntu releases, select an installed
Python 3.12+ interpreter and its matching venv package explicitly; a system
Python 3.10/3.11 is insufficient. Keep results in a writable working directory;
installation itself should not write experiments into the system Python or
site-packages. Commands are case-sensitive on typical Linux filesystems.

For remote/headless servers, keep the viewer bound to localhost and forward it
through SSH if needed:

```sh
ssh -L 8000:127.0.0.1:8000 user@server
```

Then open `http://127.0.0.1:8000` in the client browser. The viewer uses ordinary
browser APIs; Node/npm are unnecessary. Binding `--host 0.0.0.0` is explicit,
not the normal local-server default. Docker uses that bind only inside its
container, with the host-published port restricted to localhost.

## Optional native backend

Native compilation currently targets CPython 3.12–3.13. With a compatible selected
interpreter, install its **matching** development headers and a compiler (for the
system interpreter, Ubuntu packages commonly include `python3-dev` and
`build-essential`). Do not assume those headers match a separately installed
Python. Use `WTF_BUILD_NATIVE=1` with the source archive as documented in
[packaging](packaging.md#optional-native-source-build).

C code uses Python's C API and standard math/string/integer headers. Compiler
flags preserve strict floating-point behavior; no Apple frameworks, CUDA,
architecture-specific instruction set or copied macOS shared library is required.
The default wheel contains no native binary. Cross-machine artifacts are numeric
data; rebuild native code for the target OS/Python/architecture.

## Verification status and CI

Local installation checks on the development host are separate from actual Linux
execution. [Ubuntu packaging CI](../.github/workflows/package.yml) is configured
for portable wheel/source installation on CPython 3.12, 3.13 and 3.14, plus native
3.12/3.13 installations and full reference/native Python tests. Each runs outside the checkout and checks presets, a short
simulation, checkpoints, replay chunks and viewer HTTP assets. It does not run
Docker or publish to PyPI. Confirm these jobs pass on the pushed commit before
claiming that release was executed successfully on Ubuntu.

Local Python 3.12/3.13 verification is described in [package verification](packaging.md). Archive checks
verify case-sensitive viewer imports and exclude host binaries from the portable
wheel. A Linux-platform guard test checks that compiler setup leaves the normal
Linux environment unchanged. These checks support the code review; they do not
replace the pending Ubuntu CI executions.

The packaging jobs pin **Ubuntu 24.04 x64** rather than `ubuntu-latest`. The
Each selected Python needs an interpreter build for that exact runner OS; a
moving latest image may advance before compatible builds are provided. Native
jobs use the same pinned OS/architecture. Review GitHub's
[Python build manifest](https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json)
before changing this pin. A setup-python download failure occurs before package
installation tests; it is not evidence of a simulator/runtime failure.
