# Published installation snapshot

[Historical reference](index.md) · [Current installation](../../INSTALL.md)

These commands preserve the tested 0.0.2.9 package snapshot. That release
predates the current source features and is not the current guide setup.

## Install from PyPI

Version [0.0.2.9](https://pypi.org/project/wtf-wildlife-trial-framework/0.0.2.9/) is available from production PyPI.
In a fresh working directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install wtf-wildlife-trial-framework==0.0.2.9
.venv/bin/wtf presets
.venv/bin/wtf run baseline --runs 1 --steps 200 --seed 100 --progress
.venv/bin/wtf view --results results
```

The portable wheel includes presets, maps and viewer assets; no compiler is needed.
For the optional native extension, use the [native build instructions](../packaging.md#optional-native-source-build).

## Install the built wheel

Build from the checkout in its development environment, then install the wheel
into a fresh environment:

```sh
.venv/bin/python -m pip install build twine
.venv/bin/python -m build --outdir dist/0.0.2.9
.venv/bin/python -m twine check dist/0.0.2.9/*
python3.12 -m venv /path/to/test-env
/path/to/test-env/bin/python -m pip install dist/0.0.2.9/wtf_wildlife_trial_framework-0.0.2.9-py3-none-any.whl
```

From any writable working directory, use `wtf presets`,
`wtf run maskable_ppo --runs 1 --steps 200 --seed 100 --progress`, then
`wtf view --results results` (activate that environment or use its full script
paths). Presets, maps and viewer assets are included. See
[package verification](../packaging.md) and [Linux/Ubuntu setup](../linux.md).
Do not confuse installing this wheel with publishing it, or substitute a package
named only `wtf` from a registry.

