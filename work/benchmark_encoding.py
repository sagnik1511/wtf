"""Fresh-process encoding/copy benchmarks, with exact state/replay comparisons."""

import argparse
import copy
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
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve()))
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.storage.retention import ArtifactRetention
    from wtf.storage.recording import Recorder
    from wtf.storage.replay_io import load_chunk
    from wtf.storage.checkpoint_io import read_checkpoint

    c = load(ROOT / "environments/tracking.json")
    c["seed"] = 100
    c["experiment"].update(runs=1, world_seed_policy="increment")
    c["world"]["max_steps"] = 32
    if args.job == "metadata":
        e = Experiment(c)
        prepare = (
            (lambda: e._checkpoint_metadata(detached=False))
            if hasattr(e, "_checkpoint_metadata")
            else (lambda: e.checkpoint(include_brains=False))
        )
        start = time.perf_counter()
        for _ in range(20):
            metadata = prepare()
            del metadata
        seconds = time.perf_counter() - start
        tracemalloc.start()
        metadata = prepare()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result = dict(seconds=seconds, python_peak_bytes=peak, state=digest(metadata))
    elif args.job == "encoding":
        e = Experiment(c)
        seconds = 0
        with tempfile.TemporaryDirectory() as folder:
            policy = ArtifactRetention(
                Path(folder) / "experiment", dict(brain_save_every_steps=1)
            )
            for tick in range(10):
                if tick:
                    e.champions.entries = copy.deepcopy(e.champions.entries)
                    for groups in e.champions.entries.values():
                        for entries in groups.values():
                            for entry in entries.values():
                                entry["rank"][0] += 1
                e.world.tick = tick
                start = time.perf_counter()
                saved = policy.step_brains(e)
                seconds += time.perf_counter() - start
                assert saved == (tick == 0)
            result = dict(seconds=seconds, state=digest(e.checkpoint()))
    else:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "experiment"
            start = time.perf_counter()
            e = Experiment(c)
            recorder = Recorder(root, c, "train")
            recorder.begin_run(e)
            policy = ArtifactRetention(
                root,
                dict(
                    checkpoint_keep_last=2, brain_save_every_steps=8, brain_keep_last=2
                ),
            )
            while not e.done:
                e.step()
                recorder.capture(e, force=e.finished_run, metrics=e.last_record)
                policy.step_brains(e)
                if e.world.tick % 16 == 0:
                    policy.checkpoint(e, f"step_{e.world.tick}")
            recorder.close()
            seconds = time.perf_counter() - start
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            checkpoint_paths = sorted((root / "checkpoints").glob("*.bin.xz"))
            restored = Experiment.restore(read_checkpoint(checkpoint_paths[-1]))
            assert restored.checkpoint() == e.checkpoint()
            paths = sorted((root / "playback").glob("*.bin.xz"))
            # Hash each bounded decoded chunk; do not materialize a whole long run.
            replay = hashlib.sha256()
            for path in paths:
                replay.update(digest(load_chunk(path)).encode())
            result = dict(
                seconds=seconds,
                state=digest(e.checkpoint()),
                replay=replay.hexdigest(),
                checkpoint_bytes=sum(p.stat().st_size for p in checkpoint_paths),
                replay_bytes=sum(p.stat().st_size for p in paths),
                rss_mib=rss / (1024**2 if sys.platform == "darwin" else 1024),
            )
    if "rss_mib" not in result:
        result["rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if sys.platform == "darwin" else 1024
        )
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source", default=str(ROOT))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--job", choices=("metadata", "encoding", "workflow"), default="workflow"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.baseline is None or args.repeats < 1:
        parser.error("Provide baseline package root and positive repeats")
    reports = {}
    for job in ("metadata", "encoding", "workflow"):
        rows = {"before": [], "after": []}
        for repeat in range(args.repeats):
            for variant in (
                ("before", "after") if repeat % 2 == 0 else ("after", "before")
            ):
                source = args.baseline.resolve() if variant == "before" else ROOT
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source",
                    str(source),
                    "--job",
                    job,
                ]
                value = json.loads(
                    subprocess.check_output(
                        command,
                        cwd=tempfile.gettempdir(),
                        env=dict(
                            os.environ,
                            WTF_KERNELS="native",
                            PYTHONDONTWRITEBYTECODE="1",
                        ),
                        text=True,
                    )
                )
                rows[variant].append(value)
                print(job, variant, repeat + 1, round(value["seconds"], 4), flush=True)
        for key in ("state", "replay", "checkpoint_bytes", "replay_bytes"):
            if key in rows["before"][0]:
                assert (
                    len({row[key] for values in rows.values() for row in values}) == 1
                ), (job, key, "mismatch")
        reports[job] = {
            variant: {
                key: statistics.median(row[key] for row in values)
                for key in values[0]
                if type(values[0][key]) in (int, float)
            }
            for variant, values in rows.items()
        }
        reports[job]["exact_match"] = True
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
