"""Reproducible, dependency-free performance probe; run from project root.

Use separate processes per case. RSS excludes the final semantic digest step.
"""

import argparse
import cProfile
import hashlib
import json
import platform
import resource
import tempfile
import time
from pathlib import Path
from ecosystem.config import load
from ecosystem.runner import Experiment
from ecosystem.recording import Recorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=["simulate", "record", "checkpoint"], required=True
    )
    parser.add_argument(
        "--environment", type=Path, default=Path("environments/baseline.json")
    )
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    config = load(args.environment)
    config["seed"] = 42
    config["experiment"].update(runs=1, world_seed_policy="increment")
    config["world"]["max_steps"] = args.steps
    experiment = Experiment(config)
    profile = cProfile.Profile() if args.profile else None
    with tempfile.TemporaryDirectory(dir="work/performance") as tmp:
        folder = Path(tmp)
        start = time.perf_counter()
        if profile:
            profile.enable()
        writer = Recorder(folder, config, "train") if args.case == "record" else None
        if writer:
            writer.begin_run(experiment)
        while not experiment.done:
            experiment.step()
            if writer:
                writer.capture(experiment, force=experiment.finished_run)
        if writer:
            writer.end_run(experiment)
            writer.close()
        simulation_seconds = time.perf_counter() - start
        checkpoint_seconds = None
        if args.case == "checkpoint":
            start = time.perf_counter()
            path = folder / "checkpoint.json"
            if hasattr(experiment, "write_checkpoint"):
                experiment.write_checkpoint(path)
            else:
                path.write_text(
                    json.dumps(experiment.checkpoint(), indent=2, allow_nan=False)
                    + "\n"
                )
            checkpoint_seconds = time.perf_counter() - start
        if profile:
            profile.disable()
            profile.dump_stats(args.profile)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(
            case=args.case,
            environment=str(args.environment),
            steps=args.steps,
            python=platform.python_version(),
            platform=platform.system(),
            simulation_seconds=simulation_seconds,
            checkpoint_seconds=checkpoint_seconds,
            peak_rss_mib=rss / (1024**2 if platform.system() == "Darwin" else 1024),
            output_bytes=sum(
                p.stat().st_size for p in folder.rglob("*") if p.is_file()
            ),
            metrics=experiment.world.metrics(),
        )
        digest = hashlib.sha256()
        for piece in json.JSONEncoder(
            sort_keys=True, separators=(",", ":"), allow_nan=False
        ).iterencode(experiment.checkpoint()):
            digest.update(piece.encode())
        result["state_sha256"] = digest.hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
