# Complete experiment tutorial

[Project overview](../README.md) · [Runnable Python example](../examples/complete_experiment.py) · [Docker](docker.md)

Use a [current source installation](../INSTALL.md#current-source-quick-start) or
a locally built artifact in an isolated `.venv`. No notebook or Jupyter installation is needed. All example code
is included in Markdown; installed-package users can save the Tutorial 2 code as
`complete_experiment.py`. The checkout also provides it under `examples/`.

| Guide | What you will do |
| --- | --- |
| [1. Configure the world](tutorials/01-configuration.md) | Choose a preset, seed and short one-run schedule |
| [2. Run, record and analyze](tutorials/02-run-record-analyze.md) | Use the complete Python script to save replay, champions, metrics and checkpoints |
| [3. Restore and view](tutorials/03-restore-and-view.md) | Load exact state, understand champion warm starts and open offline playback |
| [4. Train, then record inference](tutorials/04-training-and-inference.md) | Choose a recording profile, retain models/recovery data, and record new frozen-policy worlds |
| [5. Compare saved policy banks](tutorials/05-compare-saved-policies.md) | Evaluate two compatible banks on matched seeds and inspect paired outcomes |
| [6. Configure senses and consequence learning](tutorials/06-sensing-and-learning.md) | V7/history/scent profiles, optional PPO objectives, multi-episode commands and inspector interpretation |

Tutorials 1–3 teach the small Python API workflow. Tutorials 4–5 use the installed
CLI for larger campaigns and controlled evaluation. Tutorial 6 provides complete
local configurations for current source sensory/consequence features. Existing step-by-step guides
cover [adding species](species.md#add-entries-to-a-json-environment),
[editing maps](geography.md) and [extending policies](extending-policies.md);
use those instead of duplicating their recipes here.

Quick run from the checkout root (for an installed package, save the Tutorial 2
script and run `.venv/bin/python complete_experiment.py --steps 200 --seed 100`):

```sh
.venv/bin/python -m examples.complete_experiment --steps 200 --seed 100
.venv/bin/python -m wtf.server
```

The example creates a new folder under ignored `results/`. It never overwrites a
preset or existing experiment. One short run demonstrates the API workflow;
it does not establish learning or guarantee reproduction. Use the standard
CLI for large campaigns and streaming metrics. See [compatibility](compatibility.md)
for version/format and cross-runtime limits.

For changes beyond the tutorial, start with the [customization handbook](customization.md),
[complete field definitions](configuration-reference.md) and [command reference](command-reference.md).
