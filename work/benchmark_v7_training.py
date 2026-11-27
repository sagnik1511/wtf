"""Exact matched V7 training and replay comparison against a preserved source tree."""

# The replay check deliberately follows the public Recorder lifecycle verbatim.
# pylint: disable=duplicate-code

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time

from work.benchmark_v6 import digest

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    """Isolate code/native imports; hash full state outside the timed training loop."""
    # Source selection must precede optional simulator imports.
    # pylint: disable=import-outside-toplevel,too-many-locals
    sys.path.insert(0, str(args.source.resolve()))
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.storage.recording import Recorder
    from wtf.storage.replay_io import load_chunk, chunk_path
    from wtf.native import kernels

    config = load(ROOT / "environments/seaside_150_v7.json")
    config["seed"] = 100
    config["experiment"].update(
        runs=2 if args.job == "published" else 1, world_seed_policy="fixed"
    )
    config["world"]["max_steps"] = (
        args.steps // 2 if args.job == "published" else args.steps
    )
    config["recording"].update(
        record_last_episodes=-1, diagnostics_every_steps=10, frames_per_chunk=32
    )
    for brain in config["brains"].values():
        brain["type"] = args.algorithm
    experiment = Experiment(config)
    if args.job == "published":
        while not experiment.finished_run:
            experiment.step()
        experiment.next_run()
    if args.job == "replay":
        stream = hashlib.sha256()
        frames = hashlib.sha256()
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(folder, config, "train")
            recorder.begin_run(experiment)
            while not experiment.done:
                experiment.step()
                recorder.capture(experiment, force=experiment.finished_run)
            recorder.end_run(experiment)
            recorder.close()
            metadata = json.loads(
                (Path(folder) / "playback/recording.json").read_text()
            )
            for i in range(metadata["runs"][0]["chunk_count"]):
                path = chunk_path(Path(folder) / "playback", 1, i)
                stream.update(path.read_bytes())
                for frame in load_chunk(path):
                    frames.update(bytes.fromhex(digest(frame)))
        return {
            "job": args.job,
            "algorithm": args.algorithm,
            "state": digest(experiment.checkpoint()),
            "chunks": stream.hexdigest(),
            "frames": frames.hexdigest(),
        }
    started = time.perf_counter()
    while not experiment.done:
        experiment.step()
    seconds = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    return {
        "job": args.job,
        "algorithm": args.algorithm,
        "seconds": seconds,
        "peak_rss_bytes": peak,
        "steps": experiment.world.tick,
        "updates": experiment.total_updates,
        "state": digest(experiment.checkpoint()),
        "native_late": kernels.available("late_ppo_batch"),
    }


def main():
    """Alternate source order; retain ignored results only after exact comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--job", choices=["first", "published", "replay"])
    parser.add_argument("--algorithm", choices=["ppo", "maskable_ppo"], default="ppo")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["ppo", "maskable_ppo"],
        default=["ppo", "maskable_ppo"],
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/v7-training-optimization.json"
    )
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1:
        parser.error("steps must be >=2 and repeats >=1")
    if args.job:
        print(json.dumps(worker(args)))
        return
    if args.baseline is None:
        parser.error("--baseline is required")
    rows = []
    for algorithm in args.algorithms:
        for job in ("first", "published", "replay"):
            for repeat in range(1 if job == "replay" else args.repeats):
                pair = []
                sources = [("before", args.baseline), ("after", ROOT)]
                if repeat % 2:
                    sources.reverse()
                for label, source in sources:
                    command = [
                        sys.executable,
                        "-m",
                        "work.benchmark_v7_training",
                        "--source",
                        str(source),
                        "--algorithm",
                        algorithm,
                        "--job",
                        job,
                        "--steps",
                        str(args.steps),
                    ]
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=True,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    )
                    row = {
                        "label": label,
                        "repeat": repeat,
                        **json.loads(completed.stdout),
                    }
                    rows.append(row)
                    pair.append(row)
                    print(
                        algorithm,
                        job,
                        label,
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("state", "chunks", "frames")
                        },
                        flush=True,
                    )
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(rows, indent=2) + "\n")
                for key in ("state", "steps", "updates", "chunks", "frames"):
                    if key in pair[0]:
                        assert pair[0][key] == pair[1][key], (algorithm, job, key)
        for job in ("first", "published"):
            medians = {
                label: statistics.median(
                    r["seconds"]
                    for r in rows
                    if r["algorithm"] == algorithm
                    and r["job"] == job
                    and r["label"] == label
                )
                for label in ("before", "after")
            }
            print(
                algorithm,
                job,
                medians,
                "speedup",
                medians["before"] / medians["after"],
                flush=True,
            )
    print("Complete states, counts and compressed replay chunks matched.", flush=True)


if __name__ == "__main__":
    main()
