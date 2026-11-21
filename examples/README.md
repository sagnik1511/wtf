# Small complete examples

[Python script](complete_experiment.py) · [Complete tutorial](../docs/tutorial.md) · [Compatibility](../docs/compatibility.md) · [Docker](../docs/docker.md)

From the checkout root, using the project environment:

```sh
.venv/bin/python -m examples.complete_experiment --steps 200 --seed 100
```

The script uses `Experiment`, `Recorder`, champion export, checkpoint writing and
analysis directly. It runs exactly one short episode and creates a fresh folder
under ignored `results/` with:

```text
example_…/
├── environment.json
├── summary.json
├── metrics.json
├── analysis.json
├── checkpoints/final.bin.xz
├── champions/                 # Metadata and referenced model/config pools
└── playback/                  # Manifest and bounded binary/XZ chunks
```

It never overwrites an existing run or modifies a preset. `--environment`,
`--steps`, `--seed` and `--output` control the example; for instance:

```sh
.venv/bin/python -m examples.complete_experiment \
  --environment environments/day_night.json --steps 1000 --seed 100
./run.sh
```

Select the printed result folder in the viewer. The default 200-step example is
for the API workflow; it is too short for some maturity thresholds and does not
guarantee reproduction or establish learning. It writes the engine's bounded
metric history as JSON; use the normal CLI for long campaigns, streamed metrics
CSV, multiple runs, progress reporting and resume options.

The [Markdown tutorial](../docs/tutorial.md) includes the full script with
explanations, checkpoint restore code and playback instructions. No notebook or
Jupyter dependency is needed. Keep generated experiments under `results/`.
