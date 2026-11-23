"""Fresh-process native-to-native LSTM benchmark with exact state/replay checks.

Save the pre-change package (including its native module) outside the import path
and pass --baseline. All workload artifacts are temporary, outside results/.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
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

    sys.path.insert(0, str(args.source.resolve()))
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.learning.brains import create_brain

    from wtf.learning.recurrent_ppo import RecurrentStep
    from wtf.learning.ppo import policy_distribution
    from wtf.storage.recording import Recorder
    from wtf.storage.replay_io import load_chunk
    from wtf.storage.checkpoint_io import read_checkpoint
    from wtf.native import kernels

    assert kernels.available("gru_step") and kernels.available("mlp_predict")
    c = load(ROOT / "environments/ppo.json")
    c["name"] = "native_lstm_benchmark"
    c["seed"] = 100
    c["world"]["max_steps"] = 32
    c["experiment"].update(runs=1, world_seed_policy="increment")
    c["recording"].update(record_last_episodes=-1, frames_per_chunk=8)
    for template in c["brains"].values():
        template.update(
            type="lstm_ppo",
            recurrent_size=16,
            sequence_length=8,
            rollout_steps=16,
            epochs=2,
            batch_size=16,
            target_kl=0,
        )
    spec = c["species"]["predator"]
    template = dict(c["brains"][spec["brain"]])
    actions = spec["allowed_actions"]
    rng = random.Random(37)
    brain = create_brain(template, actions, 100, 101)
    network = brain.network
    x = tuple(rng.uniform(-1, 1) for _ in range(network.inputs))
    memory = [rng.uniform(-0.5, 0.5) for _ in range(32)]
    if args.job == "forward":
        state = memory[:]
        start = time.perf_counter()
        for _ in range(500):
            outputs, state = network.step(x, state, tape=True)[:2]
        seconds = time.perf_counter() - start
        tracemalloc.start()
        sample = network.step(x, state, tape=True)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report = {
            "seconds": seconds,
            "python_peak_bytes": peak,
            "state": digest([outputs, state, sample]),
        }
    elif args.job == "backward":
        _, _, tape = network.step(x, memory, True)
        gradients = network.zeros()
        derivative = [(i - 3) / 10 for i in range(len(actions) + 1)]
        future = [0.01] * 32
        start = time.perf_counter()
        for _ in range(500):
            previous = network.backward(tape, derivative, future, gradients)
        seconds = time.perf_counter() - start
        fresh = network.zeros()
        tracemalloc.start()
        network.backward(tape, derivative, future, fresh)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report = {
            "seconds": seconds,
            "python_peak_bytes": peak,
            "state": digest(
                [
                    previous,
                    [{k: list(v) for k, v in layer.items()} for layer in gradients],
                ]
            ),
        }
    elif args.job == "update":
        template.update(rollout_steps=32)
        brain = create_brain(template, actions, 100, 101)
        observations = [
            tuple(((i + j) % 19 - 9) / 10 for j in range(network.inputs))
            for i in range(32)
        ]

        def fill():
            state = [0.0] * 32
            brain.rollout = []
            for i, obs in enumerate(observations):
                old = tuple(state)
                outputs, state = brain.network.step(obs, state)
                action = i % len(actions)
                logp = policy_distribution(outputs[:-1])[1][action]
                brain.rollout.append(
                    RecurrentStep(
                        obs,
                        action,
                        (i % 7 - 3) / 10,
                        logp,
                        outputs[-1],
                        0.1,
                        False,
                        i == 31,
                        old,
                    )
                )
            brain.hidden = state

        seconds = 0.0
        for _ in range(8):
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
                if e.world.tick % 16 == 0:
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
            assert len(chunks) > 1
            for path in chunks:
                for frame in load_chunk(path):
                    replay.update(bytes.fromhex(digest(frame)))
            report = {
                "seconds": seconds,
                "state": final,
                "replay": replay.hexdigest(),
                "checkpoint_hashes": digest(
                    [hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints]
                ),
                "replay_hashes": digest(
                    [hashlib.sha256(p.read_bytes()).hexdigest() for p in chunks]
                ),
                "checkpoint_bytes": sum((p.stat().st_size for p in checkpoints)),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--job",
        choices=("forward", "backward", "update", "workflow"),
        default="workflow",
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.baseline is None or args.repeats < 1:
        parser.error("Provide --baseline and positive --repeats")
    reports = {}
    for job in ("forward", "backward", "update", "workflow"):
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
            "replay_hashes",
            "checkpoint_bytes",
            "replay_bytes",
        ):
            if key in rows["before"][0]:
                assert len({r[key] for values in rows.values() for r in values}) == 1, (
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
    target = ROOT / ".cache/native-lstm-benchmark.json"
    target.write_text(json.dumps(reports, indent=2) + "\n")
    print(target)
    print(json.dumps({job: r["medians"] for job, r in reports.items()}, indent=2))
    return None


if __name__ == "__main__":
    main()
