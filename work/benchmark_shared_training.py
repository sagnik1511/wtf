"""Compare shared-work optimizations against a saved package with its native module.

Run from this checkout, with --baseline pointing to a directory containing wtf/.
All probes use fresh processes; complete workflow artifacts are temporary.
"""

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
CASES = (
    "decisions",
    "dqn_update",
    "ddqn_update",
    "scent",
    "social",
    "dqn_workflow",
    "ddqn_workflow",
)


def digest(value):
    """Stable exact-value checksum, outside the measured interval."""
    # Keep benchmark checksums self-contained and compatible with earlier probes.
    # pylint: disable=duplicate-code
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def worker(args):
    """One fixed workload; no profiling or experiment campaign."""
    # Benchmark dispatch keeps isolated imports and matching workloads in one place.
    # The recording loop intentionally matches the public workflow being measured.
    # pylint: disable=duplicate-code
    # pylint: disable=import-outside-toplevel,too-many-locals,too-many-statements,too-many-branches,protected-access
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.source.resolve()))
    from tests.test_advanced_algorithms import settings
    from tests.test_mutation import environment
    from tests.test_tracking import scene
    from tests.test_social_learning import config, controlled
    from wtf.learning.brains import create_brain
    from wtf.learning.brain import Experience
    from wtf.learning.batched import choose_actions
    from wtf.native import kernels
    from wtf.simulation.world import Animal
    from wtf.experiment.runner import Experiment
    from wtf.storage.recording import Recorder
    from wtf.storage.replay_io import load_chunk
    from wtf.storage.checkpoint_io import read_checkpoint

    assert kernels.available(
        "mlp_predict_many"
    ), "Both variants require existing native kernels"
    kind = "ddqn" if args.case.startswith("ddqn") else "dqn"
    c = settings(kind)
    c["network"]["hidden_layers"] = [64, 32, 16]
    c.update(
        observation_version=4,
        observation_top_k=10,
        social_observation={"history_steps": 4, "max_neighbors": 8},
        consequence_observation=True,
    )
    c["exploration"]["initial"] = 0.35
    actions = [
        "rest",
        "eat",
        "turn_left",
        "turn_right",
        "move_forward",
        "reproduce",
        "attack",
    ]
    brain = create_brain(c, actions, 1, 2)
    width = brain.network.sizes[0]
    observation = tuple(i / width for i in range(width))
    if args.case == "decisions":
        brains = {i: create_brain(c, actions, 1, i + 10) for i in range(32)}
        observations = {i: observation for i in brains}

        def probe():
            return choose_actions(brains, observations)

        iterations = 160

        def state():
            return {i: b.checkpoint() for i, b in brains.items()}

    elif args.case.endswith("_update"):
        c.update(
            batch_size=16,
            replay_capacity=128,
            learning_starts_after=16,
            train_every_steps=1,
            target_update_every_training_steps=8,
        )
        brain = create_brain(c, actions, 1, 2)
        counter = [0]

        def probe():
            counter[0] += 1
            i = counter[0]
            terminal = i % 9 == 0
            brain.learn(
                Experience(
                    observation,
                    actions[i % len(actions)],
                    i % 5 - 2,
                    None if terminal else observation,
                    terminal,
                    not terminal and i % 11 == 0,
                )
            )
            return brain.updates

        iterations = 96

        def state():
            return brain.checkpoint()

    elif args.case == "scent":
        world = scene(scent=True)
        field = world._tracking.scent
        world.obstacles.clear()
        world.animals = {
            i: Animal(
                i,
                "prey" if i % 2 else "predator",
                2 + i % 8,
                2 + i // 8,
                0,
                40,
                30,
                sex="male" if i % 3 else "female",
            )
            for i in range(48)
        }
        field.cells.clear()
        field.advance(world)
        template = {"scent_observation": {"mode": "blended", "radius": 4}}

        def probe():
            phase = field.prepare(world) if hasattr(field, "prepare") else None
            return [
                (
                    field.sample(world, animal, template, phase)
                    if phase is not None
                    else field.sample(world, animal, template)
                )
                for animal in world.animals.values()
            ]

        iterations = 160

        def state():
            return probe()

    elif args.case == "social":
        world = controlled(config())
        world.animals = {
            i: Animal(i, "prey", 2 + i % 8, 2 + i // 8, 0, 40, 30) for i in range(64)
        }
        social = world._social

        def probe():
            social.prepare(world)

        iterations = 12000

        def state():
            return {
                str(p): [a.id for a in animals] for p, animals in social.buckets.items()
            }

    else:
        environment_config = environment(kind, 0)
        environment_config["world"]["max_steps"] = 32
        environment_config["experiment"].update(runs=1, world_seed_policy="fixed")
        for spec in environment_config["species"].values():
            spec["initial_population"] = 16 if spec["diet"] == "resource" else 6
        for template in environment_config["brains"].values():
            template["network"]["hidden_layers"] = [64, 32, 16]
            template.update(batch_size=8, learning_starts_after=8, train_every_steps=4)
        environment_config["recording"].update(
            enabled=True,
            encoding="binary",
            compression="xz",
            frames_per_chunk=8,
            diagnostics_every_steps=4,
            record_activations=True,
        )
        with tempfile.TemporaryDirectory() as folder:
            start = time.perf_counter()
            experiment = Experiment(environment_config)
            recorder = Recorder(folder, experiment.config, experiment.mode)
            recorder.begin_run(experiment)
            while not experiment.done:
                experiment.step()
                recorder.capture(experiment)
            recorder.end_run(experiment)
            recorder.close()
            checkpoint = Path(folder) / "final.bin.xz"
            experiment.write_checkpoint(checkpoint, shared_metadata=True)
            seconds = time.perf_counter() - start
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            files = sorted(Path(folder).rglob("*"))
            artifacts = {
                str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
                if p.is_file() and p.suffix != ".json"
            }
            chunks = [
                load_chunk(p)
                for p in sorted((Path(folder) / "playback").rglob("*.bin.xz"))
            ]
            assert digest(read_checkpoint(checkpoint)) == digest(
                experiment.checkpoint()
            )
            return {
                "seconds": seconds,
                "rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "bytes": sum(p.stat().st_size for p in files if p.is_file()),
                "digest": digest(experiment.checkpoint()),
                "replay_digest": digest(chunks),
                "artifact_digest": digest(artifacts),
            }
    for _ in range(3):
        probe()
    start = time.perf_counter()
    for _ in range(iterations):
        probe()
    seconds = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Separate allocation probe; its work does not enter the timing/RSS result.
    tracemalloc.start()
    for _ in range(3):
        probe()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seconds": seconds,
        "iterations": iterations,
        "rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "traced_peak_bytes": peak,
        "digest": digest(state()),
    }


def main():
    """Alternate fresh baseline/current processes and require exact result hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".cache/optimization-0.0.2.23/benchmark.json",
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.source:
        print(json.dumps(worker(args)))
        return
    if not args.baseline or args.repeats < 1:
        parser.error("--baseline and a positive --repeats are required")
    report = {}
    for case in CASES:
        samples = {"before": [], "after": []}
        for repeat in range(args.repeats):
            order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
            for variant in order:
                source = args.baseline if variant == "before" else ROOT
                output = subprocess.check_output(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--source",
                        str(source.resolve()),
                        "--case",
                        case,
                    ],
                    cwd=ROOT,
                    env={**os.environ, "WTF_KERNELS": "native"},
                    text=True,
                )
                samples[variant].append(json.loads(output))
        for key in ("digest", "replay_digest", "artifact_digest"):
            if key in samples["before"][0]:
                assert len({s[key] for side in samples.values() for s in side}) == 1, (
                    case,
                    key,
                )
        summary = {
            side: {
                key: statistics.median(s[key] for s in rows)
                for key in rows[0]
                if isinstance(rows[0][key], (int, float))
            }
            for side, rows in samples.items()
        }
        summary["speedup"] = summary["before"]["seconds"] / summary["after"]["seconds"]
        report[case] = {"summary": summary, "samples": samples}
        print(case, json.dumps(summary), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
