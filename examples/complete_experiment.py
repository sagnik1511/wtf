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
    print(
        "Playback: .venv/bin/python -m wtf.server --results "
        + str(Path(args.output).resolve())
    )


if __name__ == "__main__":
    main()
