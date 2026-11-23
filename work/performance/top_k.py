"""Compare only top-k dispatch in fresh processes; keep all other kernels enabled.

Outputs live under ignored caches. No run/champion artifacts are retained.
"""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time


def digest(value):
    """Compute a stable checksum for workload-equivalence comparisons."""
    result = hashlib.sha256()
    for part in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), allow_nan=False
    ).iterencode(value):
        result.update(part.encode())
    return result.hexdigest()


def worker(args):
    """Execute one controlled benchmark workload."""

    # Keep independent numeric/compatibility or benchmark paths explicit to preserve
    # their exact contracts.
    # pylint: disable=duplicate-code

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel
    # This internal collaborator intentionally inspects the owning component's private
    # state.
    # pylint: disable=protected-access
    # The same branch condition guards both construction and consumption of this local
    # result.
    # pylint: disable=possibly-used-before-assignment

    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.native import kernels
    from wtf.simulation.world import World
    from wtf.storage.recording import Recorder

    available = kernels.available
    if not available("nearest_many"):
        raise SystemExit("Build the current extension with python -m work.build_native")

    # Both versions incur the same wrapper, while every other kernel is retained.
    def selected(name):
        if name == "nearest_many":
            return args.worker == "native" and available(name)
        return available(name)

    kernels.available = selected
    config = load(args.environment)
    config["seed"] = args.seed
    config["world"]["max_steps"] = args.steps
    config["experiment"].update(runs=1, world_seed_policy="fixed")
    for i, species in enumerate(config["species"].values()):
        brain = copy.deepcopy(config["brains"][species["brain"]])
        brain["observation_top_k"] = (
            args.prey_k if species["diet"] == "resource" else args.predator_k
        )
        if brain["type"] == "reinforce":
            brain["max_episode_steps"] = max(brain["max_episode_steps"], args.steps)
        config["brains"][f"top_k_probe_{i}"] = brain
        species["brain"] = f"top_k_probe_{i}"
    with tempfile.TemporaryDirectory(dir=".cache") as temporary:
        gc.collect()
        start = time.perf_counter()
        if args.case == "observe":
            world = World(config, args.seed)
            observations = world.observe_all()
            for _ in range(args.steps):
                # Force query/ray-cache misses without changing authoritative state.
                for index in world._perception:
                    index.results.clear()
                world._visibility_cache.clear()
                world._water_sight_cache.clear()
                observations = world.observe_all()
            actual_steps = args.steps
            experiment = None
        else:
            experiment = Experiment(config)
            recorder = (
                Recorder(temporary, experiment.config, experiment.mode)
                if args.case == "record"
                else None
            )
            if recorder is not None:
                recorder.begin_run(experiment)
                recorder.capture(experiment)
            while not experiment.done:
                experiment.step()
                if recorder is not None:
                    recorder.capture(
                        experiment,
                        force=experiment.finished_run,
                        metrics=experiment.last_record,
                    )
            if recorder is not None:
                recorder.end_run(experiment)
                recorder.close()
            actual_steps = experiment.world.tick
        seconds = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if sys.platform == "darwin" else 1024
        )
        checksum = digest(
            experiment.checkpoint() if experiment is not None else observations
        )
        chunks = sorted(Path(temporary).glob("playback/run_*_chunk_*"))
        replay = hashlib.sha256()
        for path in chunks:
            replay.update(path.read_bytes())
        return {
            "seconds": seconds,
            "peak_rss_mib": rss,
            "state_sha256": checksum,
            "actual_steps": actual_steps,
            "replay_sha256": replay.hexdigest(),
            "replay_bytes": sum((p.stat().st_size for p in chunks)),
        }


def main():
    """Parse command-line options and run the requested operation."""

    # Keep independent numeric/compatibility or benchmark paths explicit to preserve
    # their exact contracts.
    # pylint: disable=duplicate-code

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", type=Path, default=Path("environments/competition.json")
    )
    parser.add_argument(
        "--case", choices=("observe", "train", "record"), default="train"
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--prey-k", type=int, choices=range(1, 17), default=3)
    parser.add_argument("--predator-k", type=int, choices=range(1, 17), default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker", choices=("reference", "native"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.steps < 1 or args.samples < 1:
        parser.error("steps and samples must be positive")
    Path(".cache").mkdir(exist_ok=True)
    if args.worker:
        print(json.dumps(worker(args)))
        return
    from wtf.native import kernels

    if not kernels.available("nearest_many"):
        parser.error("Build the current extension with python -m work.build_native")
    rows = {"reference": [], "native": []}
    for repetition in range(args.samples):
        for mode in (
            ("reference", "native") if repetition % 2 == 0 else ("native", "reference")
        ):
            command = [
                sys.executable,
                "-m",
                "work.performance.top_k",
                "--worker",
                mode,
                "--environment",
                str(args.environment),
                "--case",
                args.case,
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--prey-k",
                str(args.prey_k),
                "--predator-k",
                str(args.predator_k),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            rows[mode].append(json.loads(completed.stdout))
    for field in ("state_sha256", "replay_sha256", "replay_bytes", "actual_steps"):
        values = {row[field] for samples in rows.values() for row in samples}
        if len(values) != 1:
            raise RuntimeError(f"Before/after mismatch: {field}")
    result = {
        "environment": str(args.environment),
        "case": args.case,
        "steps": args.steps,
        "actual_steps": rows["native"][0]["actual_steps"],
        "seed": args.seed,
        "prey_k": args.prey_k,
        "predator_k": args.predator_k,
        "samples": args.samples,
        "python": platform.python_version(),
        "platform": platform.system(),
        "versions": rows,
    }
    result["median"] = {
        mode: {
            "seconds": statistics.median((row["seconds"] for row in samples)),
            "peak_rss_mib": statistics.median((row["peak_rss_mib"] for row in samples)),
        }
        for mode, samples in rows.items()
    }
    result["speedup"] = (
        result["median"]["reference"]["seconds"] / result["median"]["native"]["seconds"]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
