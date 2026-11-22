"""Matched GRPO reset and independent-actor batching probes in fresh processes."""

import argparse
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
    checksum = hashlib.sha256()
    for part in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), allow_nan=False
    ).iterencode(value):
        checksum.update(part.encode())
    return checksum.hexdigest()


def worker(args):
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.learning import grouped_rollouts
    from wtf.learning.batched import choose_actions
    from wtf.native import kernels
    from wtf.storage.recording import Recorder

    available = kernels.available
    if not available("mlp_predict_many"):
        raise SystemExit("Build current native kernels first")
    batched = args.worker in ("batch", "both")
    reuse = args.worker in ("trials", "both")

    def selected(name):
        return (
            batched and available(name)
            if name == "mlp_predict_many"
            else available(name)
        )

    kernels.available = selected
    collect = grouped_rollouts.collect_group
    group_calls = 0

    def group(world, brains, focal_id):
        nonlocal group_calls
        group_calls += 1
        return collect(world, brains, focal_id, reuse_world=reuse)

    grouped_rollouts.collect_group = group
    config = load(args.environment)
    config["seed"] = args.seed
    config["experiment"].update(runs=1, world_seed_policy="fixed")
    if args.case in ("train", "record"):
        config["world"]["max_steps"] = args.steps
        for brain in config["brains"].values():
            if brain["type"] == "reinforce":
                brain["max_episode_steps"] = max(brain["max_episode_steps"], args.steps)
    with tempfile.TemporaryDirectory(dir=".cache") as folder:
        if args.case in ("choose", "groups"):
            experiment = Experiment(config)
        gc.collect()
        start = time.perf_counter()
        last_group = None
        if args.case == "choose":
            for _ in range(args.steps):
                choose_actions(experiment.brains, experiment.observations)
            actual_steps = args.steps
        elif args.case == "groups":
            focal_id = next(
                i
                for i in experiment.world.animals
                if experiment.brains[i].config["type"] == "grpo"
            )
            for _ in range(args.steps):
                last_group = group(experiment.world, experiment.brains, focal_id)
            actual_steps = args.steps
        else:
            experiment = Experiment(config)
            recorder = (
                Recorder(folder, config, experiment.mode)
                if args.case == "record"
                else None
            )
            if recorder:
                recorder.begin_run(experiment)
                recorder.capture(experiment)
            while not experiment.done:
                experiment.step()
                if recorder:
                    recorder.capture(
                        experiment,
                        force=experiment.finished_run,
                        metrics=experiment.last_record,
                    )
            if recorder:
                recorder.end_run(experiment)
                recorder.close()
            actual_steps = experiment.world.tick
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if sys.platform == "darwin" else 1024
        )
        checksum = digest((experiment.checkpoint(), last_group))
        chunks = sorted(Path(folder).glob("playback/run_*_chunk_*"))
        replay = hashlib.sha256()
        for path in chunks:
            replay.update(path.read_bytes())
        return dict(
            seconds=elapsed,
            peak_rss_mib=rss,
            state_sha256=checksum,
            actual_steps=actual_steps,
            group_collections=group_calls,
            replay_sha256=replay.hexdigest(),
            replay_bytes=sum(p.stat().st_size for p in chunks),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", type=Path, default=Path("environments/grpo.json")
    )
    parser.add_argument(
        "--case", choices=("choose", "groups", "train", "record"), default="train"
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--components", action="store_true")
    parser.add_argument(
        "--worker",
        choices=("reference", "trials", "batch", "both"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.samples < 1:
        parser.error("steps and samples must be positive")
    Path(".cache").mkdir(exist_ok=True)
    if args.worker:
        print(json.dumps(worker(args)))
        return
    from wtf.native import kernels

    if not kernels.available("mlp_predict_many"):
        parser.error("Build current native kernels first")
    modes = (
        ("reference", "trials", "batch", "both")
        if args.components
        else ("reference", "both")
    )
    rows = {mode: [] for mode in modes}
    for repetition in range(args.samples):
        for mode in modes if repetition % 2 == 0 else modes[::-1]:
            command = [
                sys.executable,
                "-m",
                "work.performance.brain_workflow",
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
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise SystemExit(completed.stderr)
            rows[mode].append(json.loads(completed.stdout))
    for field in (
        "state_sha256",
        "replay_sha256",
        "replay_bytes",
        "actual_steps",
        "group_collections",
    ):
        if len({row[field] for samples in rows.values() for row in samples}) != 1:
            raise RuntimeError("Mismatch: " + field)
    result = dict(
        environment=str(args.environment),
        case=args.case,
        steps=args.steps,
        actual_steps=rows["both"][0]["actual_steps"],
        seed=args.seed,
        samples=args.samples,
        python=platform.python_version(),
        platform=platform.system(),
        versions=rows,
    )
    result["median"] = {
        mode: dict(
            seconds=statistics.median(row["seconds"] for row in samples),
            peak_rss_mib=statistics.median(row["peak_rss_mib"] for row in samples),
        )
        for mode, samples in rows.items()
    }
    result["speedup"] = (
        result["median"]["reference"]["seconds"] / result["median"]["both"]["seconds"]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
