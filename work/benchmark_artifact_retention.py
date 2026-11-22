"""Bounded, fresh-process training/storage comparison; never touches results/."""

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

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve()))
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.storage.checkpoint_io import read_checkpoint

    c = load(ROOT / "environments/tracking.json")
    c["seed"] = 100
    c["experiment"]["world_seed_policy"] = "increment"
    c["world"]["max_steps"] = 160
    c["experiment"]["runs"] = 1
    if args.job == "training":
        start = time.perf_counter()
        e = Experiment(c)
        while not e.done:
            e.step()
        seconds = time.perf_counter() - start
        result = dict(seconds=seconds, state=digest(e.checkpoint()), steps=e.world.tick)
    elif args.job == "update":
        from wtf.learning.contextual_ppo import MaskablePPOBrain
        from wtf.learning.ppo import RolloutStep
        from wtf.native.kernels import status

        template = c["brains"][c["species"]["predator"]["brain"]]
        template.update(target_kl=0, epochs=4, batch_size=16)
        brain = MaskablePPOBrain(
            template,
            ["move_forward", "turn_left", "turn_right", "rest", "attack", "reproduce"],
            100,
            123,
        )
        n = brain.network.sizes[0]
        observation = tuple((i % 11 - 5) / 11 for i in range(n))
        brain.mask = (True, True, False, True, True, False)
        packed = brain._pack(observation)
        rollout = [RolloutStep(packed, 0, 0.2, -1.2, 0.1, 0.2, False, False)] * 64
        start = time.perf_counter()
        for _ in range(12):
            brain.rollout = list(rollout)
            brain._update()
        result = dict(
            seconds=time.perf_counter() - start,
            state=digest(brain.checkpoint()),
            backend=status(),
        )
    else:
        e = Experiment(c)
        total = 0
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoints").mkdir()
            if args.storage != "legacy":
                from wtf.storage.retention import ArtifactRetention

                policy = ArtifactRetention(
                    root,
                    dict(
                        checkpoint_keep_last=2,
                        checkpoint_shared_metadata=args.storage == "shared",
                    ),
                )
            for step in range(8):
                e.step()
                start = time.perf_counter()
                if args.storage == "legacy":
                    path = root / "checkpoints" / f"step_{step}.bin.xz"
                    e.write_checkpoint(path)
                else:
                    path = policy.checkpoint(e, f"step_{step}")
                total += time.perf_counter() - start
                restored = Experiment.restore(read_checkpoint(path))
                assert restored.checkpoint() == e.checkpoint()
            result = dict(
                seconds=total,
                state=digest(e.checkpoint()),
                bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
                files=len(list((root / "checkpoints").glob("*.bin.xz"))),
            )
    result["rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source", default=str(ROOT))
    parser.add_argument(
        "--job", choices=("training", "update", "storage"), default="training"
    )
    parser.add_argument(
        "--storage", choices=("legacy", "rolling", "shared"), default="legacy"
    )
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if not args.baseline or args.repeats < 1:
        parser.error("Provide a pre-change package root and positive repeats")
    reports = {}
    for job in ("update", "training", "storage"):
        variants = (
            ("legacy", "rolling", "shared") if job == "storage" else ("before", "after")
        )
        samples = {v: [] for v in variants}
        for repeat in range(args.repeats):
            for variant in variants if repeat % 2 == 0 else tuple(reversed(variants)):
                source = (
                    args.baseline.resolve() if variant in ("before", "legacy") else ROOT
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source",
                    str(source),
                    "--job",
                    job,
                ]
                if job == "storage":
                    command += ["--storage", variant]
                env = dict(
                    os.environ, WTF_KERNELS="native", PYTHONDONTWRITEBYTECODE="1"
                )
                value = json.loads(
                    subprocess.check_output(
                        command, cwd=tempfile.gettempdir(), env=env, text=True
                    )
                )
                samples[variant].append(value)
                print(job, variant, repeat + 1, round(value["seconds"], 3), flush=True)
        assert len({row["state"] for rows in samples.values() for row in rows}) == 1, (
            job,
            "state mismatch",
        )
        reports[job] = {
            variant: {
                key: statistics.median(row[key] for row in rows)
                for key in rows[0]
                if isinstance(rows[0][key], (int, float))
            }
            for variant, rows in samples.items()
        }
        reports[job]["exact_state_match"] = True
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
