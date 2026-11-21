# 2. Run, record and analyze

[Tutorial hub](../tutorial.md) · [Previous: configuration](01-configuration.md) · [Next: restore and view](03-restore-and-view.md)

The complete script below is also available as
[`examples/complete_experiment.py`](../../examples/complete_experiment.py).
Run it as a module from the checkout root so imports resolve consistently:

```sh
.venv/bin/python -m examples.complete_experiment --steps 200 --seed 100
```

## Complete Python code

The recorder commits bounded replay chunks instead of accumulating all frames.
Completion closes the recording manifest; exceptions mark a partial recording
as interrupted. The script exports species/sex champions, a final checkpoint,
readable metrics and an analysis report. It creates a fresh directory each time.

<!-- complete-example:start -->
```python
"""A short, one-run Python workflow with replay, champions, metrics and resume data.

From the checkout: python -m examples.complete_experiment --steps 200
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from wtf.inspection.analyze import analyze_recording
from wtf.config import load
from wtf.storage.library import export_champions
from wtf.storage.recording import Recorder
from wtf.experiment.runner import Experiment


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_experiment(environment=None, output=Path("results"), steps=200, seed=100):
    """Write a fresh, bounded, one-run experiment; return its result directory."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    config = load(environment or "maskable_ppo")
    config["seed"] = seed
    config["world"]["max_steps"] = steps
    config["experiment"].update(runs=1, world_seed_policy="increment")
    experiment = Experiment(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output) / f"example_{stamp}_{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "environment.json", experiment.config)
    recorder = Recorder(folder, experiment.config, experiment.mode)
    completed = False
    try:
        recorder.begin_run(experiment)
        while not experiment.done:
            experiment.step()
            recorder.capture(experiment)
        recorder.end_run(experiment)
        (folder / "checkpoints").mkdir()
        experiment.write_checkpoint(folder / "checkpoints/final.bin.xz")
        export_champions(
            experiment, folder / "champions/run_0001.json", label=folder.name
        )
        write_json(folder / "summary.json", experiment.summaries)
        # This intentionally short example uses the engine's bounded metric history.
        write_json(folder / "metrics.json", experiment.history)
        completed = True
    finally:
        recorder.close(completed=completed)
    write_json(folder / "analysis.json", analyze_recording(folder / "playback", run=1))
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    folder = run_experiment(args.environment, args.output, args.steps, args.seed)
    print(f"Saved experiment: {folder.resolve()}")
    print("Playback: ./run.sh --results " + str(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
```
<!-- complete-example:end -->

## Saved artifacts

```text
results/example_…/
├── environment.json          # Actual validated configuration
├── metrics.json              # Bounded step history for this short example
├── summary.json              # Completed-episode metrics
├── analysis.json             # Readable behavior report
├── checkpoints/final.bin.xz  # Exact completed training/world state
├── champions/                # Species/sex metadata plus referenced model pools
└── playback/                 # Manifest and independently loadable replay chunks
```

The example calls the existing library APIs directly. The final checkpoint
includes optimizer/RNG/controller state; champion exports are weight initializers
for fresh worlds. Replay files contain frames and sampled diagnostics, not
training checkpoints. Keep referenced metadata and model pools together.

Metrics use the engine's bounded history (up to 2,000 retained metric rows), which
is suitable for this short example. The standard CLI streams metrics CSV and
handles multiple runs, tqdm progress, interruption checkpoints and additional
campaign controls. Use it for longer work:

```sh
.venv/bin/python -m wtf.cli environments/day_night.json \
  --runs 1 --steps 1000 --seed 100 --progress
```

To call the example from your own Python session at the checkout root:

```python
from pathlib import Path
from examples.complete_experiment import run_experiment

folder = run_experiment(
    environment=Path("environments/day_night.json"),
    output=Path("results"),
    steps=1000,
    seed=100,
)
print(folder)
```

This is one experimental episode. Compare frozen policies on matched seeds
before claiming improvement; survival/population changes alone are insufficient.
