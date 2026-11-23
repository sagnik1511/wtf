"""Fresh-process before/after checks for metadata caching and PPO sample reuse.

Baseline must be a saved pre-change package root, including its native module.
All artifacts are temporary; results/ and existing recordings are untouched.
"""

# Keep independent numeric/compatibility or benchmark paths explicit to preserve their
# exact contracts.
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
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    """Compute a stable checksum for workload-equivalence comparisons."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def worker(args):
    """Execute one controlled benchmark workload."""

    # Keep independent numeric/compatibility or benchmark paths explicit to preserve
    # their exact contracts.
    # pylint: disable=duplicate-code

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel
    # This internal collaborator intentionally inspects the owning component's private
    # state.
    # pylint: disable=protected-access

    sys.path.insert(0, str(args.source.resolve()))
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.storage.checkpoint_io import Encoder, read_checkpoint
    from wtf.storage.recording import Recorder
    from wtf.storage.replay_io import load_chunk
    from wtf.learning.brains import create_brain
    from wtf.learning.ppo import RolloutStep
    from wtf.native import kernels

    assert kernels.available("ppo_batch") and kernels.available("masked_ppo_batch")
    c = load(ROOT / "environments/ppo.json")
    c["seed"] = 100
    c["experiment"].update(runs=1, world_seed_policy="increment")
    c["world"]["max_steps"] = 128
    c["recording"]["record_last_episodes"] = -1
    if args.job == "metadata":
        e = Experiment(c)
        for _ in range(16):
            e.step()
        metadata = e._checkpoint_metadata(detached=False)

        def encode():
            encoder = Encoder(binary=True)
            value = getattr(encoder, "_encode_metadata", encoder.encode)(metadata)
            return {"metadata": value, "shared": encoder.shared}

        start = time.perf_counter()
        for _ in range(10):
            result = encode()
        seconds = time.perf_counter() - start
        tracemalloc.start()
        traced = encode()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert traced == result
        report = {
            "seconds": seconds,
            "python_peak_bytes": peak,
            "state": digest(result),
        }
    elif args.job in ("ppo", "masked"):
        if args.job == "ppo":
            template = c["brains"]["predator_ppo"]
            actions = c["species"]["predator"]["allowed_actions"]
        else:
            tracking = load(ROOT / "environments/tracking.json")
            spec = tracking["species"]["predator"]
            template = tracking["brains"][spec["brain"]]
            actions = spec["allowed_actions"]
        template.update(epochs=4, batch_size=16, target_kl=0)
        brain = create_brain(template, actions, 100, 101)
        inputs = brain.network.sizes[0]
        observations = [
            tuple(((j + i) % 19 - 9) / 10 for j in range(inputs)) for i in range(64)
        ]
        if args.job == "masked":
            observations = [obs + tuple(True for _ in actions) for obs in observations]

        def fill():

            # Keep independent numeric/compatibility or benchmark paths explicit to
            # preserve their exact contracts.
            # pylint: disable=duplicate-code

            brain.rollout = []
            for i, obs in enumerate(observations):
                outputs = brain.network.predict(obs)
                logp = brain.distribution(outputs, obs)[1][i % len(actions)]
                brain.rollout.append(
                    RolloutStep(
                        obs,
                        i % len(actions),
                        (i % 7 - 3) / 10,
                        logp,
                        outputs[-1],
                        0.1,
                        False,
                        i == 63,
                    )
                )

        seconds = 0.0
        for _ in range(64):
            fill()
            start = time.perf_counter()
            brain._update()
            seconds += time.perf_counter() - start
        fill()
        tracemalloc.start()
        brain._update()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report = {
            "seconds": seconds,
            "python_peak_bytes": peak,
            "state": digest(brain.checkpoint()),
        }
    else:
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            start = time.perf_counter()
            e = Experiment(c)
            recorder = Recorder(folder, c, "train")
            recorder.begin_run(e)
            checkpoints = []
            while not e.finished_run:
                e.step()
                recorder.capture(e, force=e.finished_run, metrics=e.last_record)
                if e.world.tick % 64 == 0:
                    path = folder / f"step_{e.world.tick}.bin.xz"
                    e.write_checkpoint(path)
                    checkpoints.append(path)
            recorder.end_run(e)
            recorder.close()
            seconds = time.perf_counter() - start
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            final = digest(e.checkpoint())
            restored = Experiment.restore(read_checkpoint(checkpoints[-1]))
            assert digest(restored.checkpoint()) == final
            replay = hashlib.sha256()
            chunks = sorted((folder / "playback").glob("*.bin.xz"))
            assert chunks
            for chunk in chunks:
                for frame in load_chunk(chunk):
                    replay.update(bytes.fromhex(digest(frame)))
            report = {
                "seconds": seconds,
                "state": final,
                "replay": replay.hexdigest(),
                "checkpoint_bytes": sum((p.stat().st_size for p in checkpoints)),
                "checkpoint_hashes": digest(
                    [hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints]
                ),
                "replay_bytes": sum((p.stat().st_size for p in chunks)),
                "rss_mib": rss / (1024**2 if sys.platform == "darwin" else 1024),
            }
    report.setdefault(
        "rss_mib",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
    )
    print(json.dumps(report))


def main():
    """Parse command-line options and run the requested operation."""

    # Keep independent numeric/compatibility or benchmark paths explicit to preserve
    # their exact contracts.
    # pylint: disable=duplicate-code

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--job", choices=("metadata", "ppo", "masked", "workflow"), default="workflow"
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.baseline is None or args.repeats < 1:
        parser.error("Provide --baseline and positive --repeats")
    reports = {}
    for job in ("metadata", "ppo", "masked", "workflow"):
        rows = {"before": [], "after": []}
        for repeat in range(args.repeats):
            for variant in (
                ("before", "after") if repeat % 2 == 0 else ("after", "before")
            ):
                source = args.baseline.resolve() if variant == "before" else ROOT
                value = json.loads(
                    subprocess.check_output(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--source",
                            str(source),
                            "--job",
                            job,
                        ],
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
                print(job, variant, repeat + 1, round(value["seconds"], 5), flush=True)
        for key in (
            "state",
            "replay",
            "checkpoint_hashes",
            "checkpoint_bytes",
            "replay_bytes",
        ):
            if key in rows["before"][0]:
                assert len({v[key] for values in rows.values() for v in values}) == 1, (
                    job,
                    key,
                )
        medians = {
            variant: {
                key: statistics.median(v[key] for v in values)
                for key in values[0]
                if isinstance(values[0][key], (int, float))
            }
            for variant, values in rows.items()
        }
        reports[job] = {"samples": rows, "medians": medians}
    target = ROOT / ".cache/checkpoint-ppo-reuse-benchmark.json"
    target.write_text(json.dumps(reports, indent=2) + "\n")
    print(target)
    print(json.dumps({k: v["medians"] for k, v in reports.items()}, indent=2))
    return None


if __name__ == "__main__":
    main()
