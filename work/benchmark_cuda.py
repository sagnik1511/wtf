"""Compare complete CPU/CUDA Experiment.step loops in fresh isolated processes.

No recordings/checkpoints are written. This includes packing, transfers, world
physics and CPU optimizer work; it is not a kernel-only speed measurement.
"""

import argparse
import copy
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from wtf.native import kernels

KINDS = (
    "dqn",
    "ddqn",
    "ppo",
    "maskable_ppo",
    "a2c",
    "reinforce",
    "grpo",
    "mappo",
    "qrdqn",
    "trpo",
    "discrete_sac",
)


def configuration(kind, steps, seed, hidden=None):
    """Use maintained presets; derive non-preset learners from their PPO fields."""
    source = (
        "qrdqn"
        if kind in ("dqn", "ddqn")
        else "ppo" if kind in ("trpo", "discrete_sac") else kind
    )
    config = copy.deepcopy(load(source))
    used = {species["brain"] for species in config["species"].values()}
    config["brains"] = {
        key: value for key, value in config["brains"].items() if key in used
    }
    template = next(b for b in config["brains"].values() if b["type"] == source)
    config["brains"] = {
        key: copy.deepcopy(b if b["type"] == source else template)
        for key, b in config["brains"].items()
    }
    for brain in config["brains"].values():
        brain["type"] = kind
        if kind in ("dqn", "ddqn"):
            brain.pop("num_quantiles", None)
        if hidden is not None:
            brain["network"]["hidden_layers"] = list(hidden)
        if kind == "discrete_sac":
            # This explicit constructor-field list also appears in independent test fixtures.
            # pylint: disable=duplicate-code
            keep = (
                "type",
                "network",
                "learning_rate",
                "discount_factor",
                "max_grad_norm",
                "observation_version",
                "observation_top_k",
            )
            # pylint: enable=duplicate-code
            replacement = {key: brain[key] for key in keep if key in brain}
            replacement.update(
                batch_size=32,
                replay_capacity=2048,
                learning_starts_after=32,
                train_every_steps=1,
                temperature=0.1,
                target_tau=0.05,
            )
            brain.clear()
            brain.update(replacement)
        elif kind == "trpo":
            for key in ("clip_ratio", "epochs", "target_kl"):
                brain.pop(key, None)
            brain.update(
                value_epochs=2,
                max_kl=0.01,
                cg_iterations=10,
                cg_damping=0.1,
                cg_tolerance=1e-10,
                backtrack_steps=10,
                backtrack_coefficient=0.5,
                accept_ratio=0.1,
            )
    config["world"]["max_steps"] = steps
    config["experiment"].update(runs=1, world_seed_policy="fixed")
    config["seed"] = seed
    validate(config)
    return config


def worker(args):
    """Time initialization and the full scheduled loop separately in one process."""
    # Private accelerator status is intentionally inspected by this internal tool.
    # pylint: disable=protected-access
    config = configuration(args.algorithm, args.steps, args.seed, args.hidden)
    start = time.perf_counter()
    experiment = Experiment(config)
    experiment.set_training_device(
        args.worker,
        max_agents=args.batch_agents,
        workspace_mb=args.workspace_mb,
        precision=args.cuda_precision,
        profile=args.cuda_profile,
        inference=args.cuda_inference,
    )
    initialization = time.perf_counter() - start
    completed, warm_seconds = 0, 0.0
    initial_updates = experiment.total_updates
    start = time.perf_counter()
    while completed < args.steps:
        experiment.step()
        completed += 1
        if completed == args.warmup:
            warm_seconds = time.perf_counter() - start
        if experiment.finished_run:
            break
    total = time.perf_counter() - start
    status = experiment._cuda_training.status() if experiment._cuda_training else {}
    # ru_maxrss is bytes on macOS and KiB on Linux. Each worker is a fresh process.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    updates = experiment.total_updates - initial_updates
    if completed < args.warmup:
        warm_seconds = total
    result = {
        "algorithm": args.algorithm,
        "device": args.worker,
        "world_seed": experiment.world.seed,
        "steps": completed,
        "initialization_seconds": initialization,
        "cpu_backend": kernels.status(),
        "python": sys.version,
        "loop_seconds": total,
        "warmup_steps": min(args.warmup, completed),
        "warmup_seconds": warm_seconds,
        "steady_seconds": total - warm_seconds,
        "updates": updates,
        "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
        "accelerator": status,
        "config": config,
    }
    experiment.set_training_device("cpu")
    return result


def main(argv=None):
    """Run CPU and CUDA cases independently; report failures instead of fake timings."""
    # Keep benchmark cases, process results and report construction together.
    # pylint: disable=too-many-locals,too-many-statements
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--hidden", nargs="+", type=int, help="Optional matched hidden-layer sizes"
    )
    parser.add_argument("--cuda-precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument(
        "--cuda-inference",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CPU actions for CUDA-trained PPO/DQN; "
        "GPU for other dense policies",
    )
    parser.add_argument("--cuda-profile", action="store_true")
    parser.add_argument("--batch-agents", type=int, default=64)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument(
        "--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=KINDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.steps < 2 or not 0 <= args.warmup < args.steps or args.repeats < 1:
        parser.error("Require steps >= 2, 0 <= warmup < steps, repeats >= 1")
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    records, failures = [], []
    for kind in args.algorithms:
        for repeat in range(args.repeats):
            for device in args.devices:
                command = [
                    sys.executable,
                    "-m",
                    "work.benchmark_cuda",
                    "--worker",
                    device,
                    "--algorithm",
                    kind,
                    "--steps",
                    str(args.steps),
                    "--warmup",
                    str(args.warmup),
                    "--seed",
                    str(args.seed + repeat),
                    "--batch-agents",
                    str(args.batch_agents),
                    "--workspace-mb",
                    str(args.workspace_mb),
                ]
                command += [
                    "--cuda-precision",
                    args.cuda_precision,
                    "--cuda-inference",
                    args.cuda_inference,
                ]
                if args.cuda_profile:
                    command.append("--cuda-profile")
                if args.hidden:
                    command += ["--hidden", *map(str, args.hidden)]
                process = subprocess.run(
                    command, text=True, capture_output=True, check=False
                )
                if process.returncode:
                    failures.append(
                        {"algorithm": kind, "device": device, "error": process.stderr}
                    )
                    print(f"{kind} {device}: FAILED", file=sys.stderr, flush=True)
                    continue
                record = json.loads(process.stdout)
                record["repeat"] = repeat
                records.append(record)
                print(
                    f'{kind} {device}: {record["steps"]} steps, {record["updates"]} updates, '
                    f'{record["loop_seconds"]:.3f}s',
                    file=sys.stderr,
                    flush=True,
                )
    comparisons = []
    for kind in args.algorithms:
        cpu = [r for r in records if r["algorithm"] == kind and r["device"] == "cpu"]
        gpu = [r for r in records if r["algorithm"] == kind and r["device"] == "cuda"]
        matched = len(cpu) == len(gpu) == args.repeats and all(
            a["steps"] == b["steps"]
            and a["updates"] == b["updates"]
            and a["updates"] > 0
            for a, b in zip(cpu, gpu)
        )
        exercised = bool(gpu) and all(
            r["accelerator"].get("predictions", 0)
            + r["accelerator"].get("model_updates", 0)
            + r["accelerator"].get("ppo_model_updates", 0)
            > 0
            for r in gpu
        )
        comparisons.append(
            {
                "algorithm": kind,
                "matched_work_counts": matched,
                "cuda_exercised": exercised,
                "profiled": args.cuda_profile,
                "loop_speedup": (
                    (
                        statistics.median(r["loop_seconds"] for r in cpu)
                        / statistics.median(r["loop_seconds"] for r in gpu)
                    )
                    if matched and exercised and not args.cuda_profile
                    else None
                ),
            }
        )
    report = {
        "records": records,
        "comparisons": comparisons,
        "failures": failures,
        "caveat": "CUDA floating-point differences may change trajectories. "
        "Equal counts do not establish identical work. No replay I/O is timed.",
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
