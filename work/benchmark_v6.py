"""Exact V6 allocation/training comparison in isolated processes; no saved runs."""

# The replay probe deliberately uses the standard Recorder lifecycle verbatim.
# pylint: disable=duplicate-code

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    """Stable complete-state hash, including every weight and optimizer field."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def worker(args):
    """Time one workload; memory capture is a separate untimed pass."""
    # Isolated source imports keep a dirty-tree baseline independent and reproducible.
    # pylint: disable=import-outside-toplevel,too-many-locals,too-many-branches,too-many-statements
    sys.path.insert(0, str(args.source.resolve()))
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.simulation.world import World
    from wtf.simulation import soft_observations as soft
    from wtf.storage.recording import Recorder
    from wtf.storage.replay_io import load_chunk, chunk_path
    import random

    config = load(ROOT / "environments/seaside_150_v6.json")
    config["seed"] = 100
    config["experiment"].update(runs=1, world_seed_policy="fixed")
    config["world"]["max_steps"] = args.steps
    if args.job == "encoding":
        templates = {
            s: config["brains"][spec["brain"]] for s, spec in config["species"].items()
        }
        raw_config = json.loads(json.dumps(config))
        for b in raw_config["brains"].values():
            b["observation_version"] = 5
        world = World(raw_config, 100)
        rng = random.Random(71)
        bank = []
        for tick in range(64):
            for identity, values in world.observe_all().items():
                species = world.animals[identity].species
                v = list(values)
                # Published rows repeat within species and eight-step windows.
                for row in range(24):
                    start = len(v) - 96 + row * 4
                    v[start : start + 4] = [
                        0.5 if tick // 8 % 2 else 0.75,
                        0.001 * (row % 4),
                        0.1 * (row % 3),
                        0.0001 * (row % 2),
                    ]
                bank.append((species, tuple(v)))
            world.step(
                {
                    i: rng.choice(config["species"][a.species]["allowed_actions"])
                    for i, a in world.animals.items()
                }
            )

        def encode():
            return [tuple(soft.encode_soft(v, templates[s])) for s, v in bank]

        encode()
        started = time.perf_counter()
        for _ in range(args.passes):
            encoded = encode()
        seconds = time.perf_counter() - started
        checksum = hashlib.sha256()
        for row in encoded:
            checksum.update(struct.pack("<" + "d" * len(row), *row))
        del encoded
        row = None
        # Include newly allocated structural/value caches in the retained-bank probe.
        for name in ("_basis", "_direction_entries", "_unbasis", "_layout"):
            cache = getattr(soft, name, None)
            if cache is not None:
                cache.cache_clear()
        tracemalloc.start()
        encoded = encode()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "job": args.job,
            "seconds": seconds,
            "observations": len(bank) * args.passes,
            "retained_python_bytes": retained,
            "peak_python_bytes": peak,
            "state": checksum.hexdigest(),
        }
    if args.job in ("training", "training-published"):
        if args.job == "training-published":
            config["experiment"]["runs"] = 2
            config["world"]["max_steps"] = args.steps // 2
        # No I/O in timed work; canonical full state catches policy/optimizer/RNG drift.
        e = Experiment(config)
        if args.job == "training-published":
            while not e.finished_run:
                e.step()
            e.next_run()
        started = time.perf_counter()
        while not e.done:
            e.step()
        seconds = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        if args.job == "training-published":
            assert any(brain.auxiliary.updates for brain in e.brains.values())
        return {
            "job": args.job,
            "seconds": seconds,
            "steps": e.world.tick,
            "updates": e.total_updates,
            "state": digest(e.checkpoint()),
            "peak_rss_bytes": peak,
        }
    config["recording"].update(
        enabled=True,
        record_last_episodes=-1,
        frames_per_chunk=32,
        diagnostics_every_steps=10,
    )
    e = Experiment(config)
    replay = hashlib.sha256()
    binary = hashlib.sha256()
    with tempfile.TemporaryDirectory() as folder:
        recorder = Recorder(folder, config, "train")
        recorder.begin_run(e)
        while not e.done:
            e.step()
            recorder.capture(e, force=e.finished_run)
        recorder.end_run(e)
        recorder.close()
        entry = json.loads((Path(folder) / "playback/recording.json").read_text())[
            "runs"
        ][0]
        for i in range(entry["chunk_count"]):
            path = chunk_path(Path(folder) / "playback", 1, i)
            binary.update(path.read_bytes())
            for frame in load_chunk(path):
                replay.update(bytes.fromhex(digest(frame)))
    return {
        "job": args.job,
        "state": digest(e.checkpoint()),
        "replay": replay.hexdigest(),
        "chunk_bytes_sha256": binary.hexdigest(),
    }


def main():
    """Alternate baseline/current timings and require exact work/state agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument(
        "--job", choices=["encoding", "training", "training-published", "replay"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/v6-optimization.json"
    )
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1 or args.passes < 1:
        parser.error("steps must be >= 2; repeats and passes must be positive")
    if args.job:
        print(json.dumps(worker(args)))
        return
    if args.baseline is None:
        parser.error("--baseline is required")
    rows = []
    for job in ("encoding", "training", "training-published", "replay"):
        for repeat in range(args.repeats if job != "replay" else 1):
            pair = []
            for label, source in (
                (("before", args.baseline), ("after", ROOT))
                if repeat % 2 == 0
                else (("after", ROOT), ("before", args.baseline))
            ):
                command = [
                    sys.executable,
                    "-m",
                    "work.benchmark_v6",
                    "--source",
                    str(source),
                    "--job",
                    job,
                    "--steps",
                    str(args.steps),
                    "--passes",
                    str(args.passes),
                ]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    cwd=ROOT,
                )
                row = {"label": label, "repeat": repeat, **json.loads(result.stdout)}
                pair.append(row)
                rows.append(row)
                print(
                    job,
                    label,
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("state", "replay", "chunk_bytes_sha256")
                    },
                    flush=True,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
            for key in (
                "state",
                "replay",
                "chunk_bytes_sha256",
                "steps",
                "updates",
                "observations",
            ):
                if key in pair[0]:
                    assert pair[0][key] == pair[1][key], (job, key)
    for job in ("encoding", "training", "training-published"):
        before = [
            r["seconds"] for r in rows if r["job"] == job and r["label"] == "before"
        ]
        after = [
            r["seconds"] for r in rows if r["job"] == job and r["label"] == "after"
        ]
        print(
            job,
            "median speedup",
            statistics.median(before) / statistics.median(after),
            flush=True,
        )
    print(
        "Exact values, complete states, updates and replay bytes matched.", flush=True
    )


if __name__ == "__main__":
    main()
