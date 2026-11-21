# Complete experiment tutorial

[Project overview](../README.md) · [Runnable Python example](../examples/complete_experiment.py) · [Docker](docker.md)

Work through these guides from a checkout using its `.venv`. No notebook or
Jupyter installation is needed. All example code is included in Markdown;
the complete training script also lives under `examples/`.

| Guide | What you will do |
| --- | --- |
| [1. Configure the world](tutorials/01-configuration.md) | Choose a preset, seed and short one-run schedule |
| [2. Run, record and analyze](tutorials/02-run-record-analyze.md) | Use the complete Python script to save replay, champions, metrics and checkpoints |
| [3. Restore and view](tutorials/03-restore-and-view.md) | Load exact state, understand champion warm starts and open offline playback |

Quick run from the checkout root:

```sh
.venv/bin/python -m examples.complete_experiment --steps 200 --seed 100
./run.sh
```

The example creates a new folder under ignored `results/`. It never overwrites a
preset or existing experiment. One short run demonstrates the API workflow;
it does not establish learning or guarantee reproduction. Use the standard
CLI for large campaigns and streaming metrics. See [compatibility](compatibility.md)
for version/format and cross-runtime limits.

For changes beyond the tutorial, start with the [customization handbook](customization.md),
[complete field definitions](configuration-reference.md) and [command reference](command-reference.md).
