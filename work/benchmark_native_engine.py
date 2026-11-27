"""Benchmark complete Python/C++ CLI workflows with exact saved-artifact checks.

Each sample runs in a fresh subprocess. Timings include backend initialization,
bounded native batch transfers, recording and binary/XZ checkpoints.
Verification and subprocess startup are reported outside the timed workflow.
Artifacts live in temporary folders; raw reports default to .cache/native-engine.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    """Encode complete values deterministically, retaining signed float zeros."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value):
    """Hash a complete logical value without sampling its contents."""
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    """Hash file bytes in bounded blocks."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def peak_rss_bytes():
    """Normalize macOS bytes and Unix kilobytes from getrusage."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def replay_report(folder, load_chunk):
    """Hash every physical replay chunk and every decoded frame in order."""
    frames = hashlib.sha256()
    chunks = []
    frame_count = 0
    for path in sorted((folder / "playback").glob("run_*_chunk_*")):
        chunks.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
        )
        for frame in load_chunk(path):
            frames.update(canonical(frame))
            frames.update(b"\n")
            frame_count += 1
    return {
        "chunks": chunks,
        "chunk_bytes_sha256": digest(chunks),
        "decoded_frames_sha256": frames.hexdigest(),
        "frame_count": frame_count,
        "chunk_count": len(chunks),
        "chunk_bytes": sum(chunk["bytes"] for chunk in chunks),
    }


def state_counts(state, replay):
    """Expose useful workload counts alongside the complete checkpoint hash."""
    snapshot = state["world"]["snapshot"]
    inherited = state.get("inherited_updates", {})
    return {
        "final_tick": snapshot["tick"],
        "animals": len(snapshot["animals"]),
        "resources": len(snapshot["resources"]),
        "brains": len(state["brains"]),
        "history_rows": len(state["history"]),
        "completed_runs": len(state["summaries"]),
        "run_seeds": state["run_seeds"],
        "total_updates": state["retired_updates"]
        + sum(
            brain["updates"] - inherited.get(key, 0)
            for key, brain in state["brains"].items()
        ),
        "replay_frames": replay["frame_count"],
        "replay_chunks": replay["chunk_count"],
        "final_metrics": snapshot["metrics"],
    }


def worker(args):
    """Run and verify one complete artifact-producing workflow."""
    # Imports and verification stay outside the timed CLI workflow.
    # pylint: disable=import-outside-toplevel,too-many-locals,too-many-statements
    sys.path.insert(0, str(args.source.resolve()))
    from wtf.cli import run_experiment
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.experiment.native import NativeExperiment
    from wtf.native import kernels
    from wtf.storage.checkpoint_io import read_checkpoint
    from wtf.storage.recording import Recorder
    from wtf.storage.replay_io import load_chunk

    config_path = Path(args.preset)
    if not config_path.is_file():
        config_path = (
            args.source / "environments" / (args.preset.removesuffix(".json") + ".json")
        )
    config = load(config_path)
    config["seed"] = args.seed
    config["world"]["max_steps"] = args.steps
    config["experiment"].update(runs=args.runs, world_seed_policy="fixed")
    config.setdefault("recording", {}).update(
        enabled=args.profile == "replay", profile="full"
    )
    with tempfile.TemporaryDirectory(prefix="wtf-native-engine-") as temporary:
        # Identical basename keeps ordinary artifact provenance comparable.
        folder = Path(temporary) / "run"
        folder.mkdir()

        def save(name, data):
            (folder / name).write_text(
                json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )

        with redirect_stdout(io.StringIO()):
            start = time.perf_counter()
            native = None
            if args.backend == "cpp":
                native = NativeExperiment(config, mode=args.mode)
                experiment = native.to_python()
            else:
                experiment = Experiment(config, mode=args.mode)
            recorder = (
                Recorder(folder, experiment.config, args.mode)
                if args.profile == "replay"
                else None
            )
            initialized = time.perf_counter()
            completed = False
            try:
                run_experiment(
                    experiment,
                    folder,
                    save,
                    recorder,
                    save_brains=False,
                    checkpoint_format="binary",
                    engine=args.backend,
                    native_session=native,
                )
                completed = True
            finally:
                if recorder is not None:
                    recorder.close(completed=completed)
            finished = time.perf_counter()
        workflow_rss = peak_rss_bytes()
        state = read_checkpoint(folder / "checkpoints" / "final.bin.xz")
        replay = replay_report(folder, load_chunk)
        files = [path for path in folder.rglob("*") if path.is_file()]
        engine_spec = importlib.util.find_spec("wtf.native._engine")
        report = {
            "backend": args.backend,
            "preset": args.preset,
            "profile": args.profile,
            "mode": args.mode,
            "seed": args.seed,
            "steps": args.steps,
            "runs": args.runs,
            "initialization_seconds": initialized - start,
            "run_and_artifacts_seconds": finished - initialized,
            "total_seconds": finished - start,
            "peak_rss_bytes": workflow_rss,
            "checkpoint_sha256": digest(state),
            "checkpoint_file_sha256": file_digest(
                folder / "checkpoints" / "final.bin.xz"
            ),
            "metrics_csv_sha256": file_digest(folder / "metrics.csv"),
            "counts": state_counts(state, replay),
            "replay": replay,
            "artifact_total_bytes": sum(path.stat().st_size for path in files),
            "artifact_file_count": len(files),
            "native_kernels": kernels.status(),
            "native_engine": {
                "available": engine_spec is not None,
                "path": None if engine_spec is None else engine_spec.origin,
            },
            "source": str(args.source.resolve()),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        }
        report["validation_peak_rss_bytes"] = peak_rss_bytes()
        return report


def atomic_report(path, report):
    """Preserve completed measurements after every worker and pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def compare_pair(pair):
    """Require exact logical state, counters, CSV and physical replay artifacts."""
    left, right = pair["samples"]["python"], pair["samples"]["cpp"]
    fields = (
        "checkpoint_sha256",
        "checkpoint_file_sha256",
        "metrics_csv_sha256",
        "counts",
        "replay",
        "artifact_total_bytes",
        "artifact_file_count",
    )
    return [field for field in fields if left[field] != right[field]]


def summarize(report):
    """Summarize paired medians for each independent workload."""
    summaries = []
    groups = {}
    for pair in report["pairs"]:
        key = tuple(pair[name] for name in ("preset", "profile", "mode", "seed"))
        groups.setdefault(key, []).append(pair)
    for key, pairs in groups.items():
        row = dict(zip(("preset", "profile", "mode", "seed"), key))
        for backend in ("python", "cpp"):
            samples = [pair["samples"][backend] for pair in pairs]
            row[backend] = {
                field: statistics.median(sample[field] for sample in samples)
                for field in (
                    "initialization_seconds",
                    "run_and_artifacts_seconds",
                    "total_seconds",
                    "peak_rss_bytes",
                )
            }
        row["speedup"] = row["python"]["total_seconds"] / row["cpp"]["total_seconds"]
        row["run_and_artifacts_speedup"] = (
            row["python"]["run_and_artifacts_seconds"]
            / row["cpp"]["run_and_artifacts_seconds"]
        )
        row["rss_ratio_python_over_cpp"] = (
            row["python"]["peak_rss_bytes"] / row["cpp"]["peak_rss_bytes"]
        )
        summaries.append(row)
    return summaries


def benchmark(args):
    """Alternate fresh subprocesses, persist raw evidence and verify each pair."""
    # This bounded harness coordinates a complete paired measurement transaction.
    # pylint: disable=too-many-locals,too-many-statements
    output = args.output or args.source / ".cache" / "native-engine" / "benchmark.json"
    report = {
        "schema_version": 1,
        "description": (
            "Complete CLI workflow including Python bootstrap and artifact persistence; "
            "verification excluded"
        ),
        "source": str(args.source.resolve()),
        "steps": args.steps,
        "runs": args.runs,
        "repeats": args.repeats,
        "pairs": [],
        "status": "running",
    }
    atomic_report(output, report)
    workloads = itertools.product(args.presets, args.profiles, args.modes, args.seeds)
    for preset, profile, mode, seed in workloads:
        for repeat in range(args.repeats):
            order = (
                ("python", "cpp")
                if len(report["pairs"]) % 2 == 0
                else ("cpp", "python")
            )
            pair = {
                "preset": preset,
                "profile": profile,
                "mode": mode,
                "seed": seed,
                "repeat": repeat,
                "order": order,
                "samples": {},
            }
            report["pairs"].append(pair)
            for backend in order:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source",
                    str(args.source.resolve()),
                    "--backend",
                    backend,
                    "--preset",
                    preset,
                    "--profile",
                    profile,
                    "--mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--runs",
                    str(args.runs),
                ]
                start = time.perf_counter()
                result = subprocess.run(
                    command,
                    cwd=args.source,
                    capture_output=True,
                    text=True,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                    check=False,
                )
                if result.returncode:
                    pair["error"] = {
                        "backend": backend,
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                        "stdout": result.stdout,
                    }
                    report["status"] = "worker_failed"
                    atomic_report(output, report)
                    raise RuntimeError(
                        f"{backend} worker failed; report: {output}\n{result.stderr}"
                    )
                try:
                    sample = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pair["error"] = {
                        "backend": backend,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                    report["status"] = "invalid_worker_output"
                    atomic_report(output, report)
                    raise
                sample["subprocess_seconds"] = time.perf_counter() - start
                pair["samples"][backend] = sample
                atomic_report(output, report)
            pair["mismatches"] = compare_pair(pair)
            pair["equivalent"] = not pair["mismatches"]
            if not pair["equivalent"]:
                report["status"] = "artifact_mismatch"
                atomic_report(output, report)
                raise RuntimeError(
                    f"Artifact mismatch: {pair['mismatches']}; report: {output}"
                )
            atomic_report(output, report)
            print(
                f"{preset} {profile} {mode} seed={seed} repeat={repeat + 1}: exact artifacts"
            )
    report["summaries"] = summarize(report)
    report["status"] = "complete"
    atomic_report(output, report)
    for row in report["summaries"]:
        python, cpp = row["python"], row["cpp"]
        print(
            f"{row['preset']} {row['profile']} {row['mode']} seed={row['seed']}: "
            f"Python {python['total_seconds']:.3f}s, C++ {cpp['total_seconds']:.3f}s, "
            f"speedup {row['speedup']:.2f}x; peak RSS "
            f"{python['peak_rss_bytes'] / 1024**2:.1f}/{cpp['peak_rss_bytes'] / 1024**2:.1f} MiB"
        )
    print(f"Raw report: {output}")


def main():
    """Parse explicit workload controls and dispatch parent or isolated worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--presets", nargs="+", default=["ppo"])
    parser.add_argument(
        "--profiles",
        choices=("headless", "replay"),
        nargs="+",
        default=["headless", "replay"],
    )
    parser.add_argument(
        "--modes",
        choices=("train", "evaluate"),
        nargs="+",
        default=["train", "evaluate"],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("python", "cpp"), help=argparse.SUPPRESS)
    parser.add_argument("--preset", help=argparse.SUPPRESS)
    parser.add_argument(
        "--profile", choices=("headless", "replay"), help=argparse.SUPPRESS
    )
    parser.add_argument("--mode", choices=("train", "evaluate"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.steps, args.runs, args.repeats) < 1:
        parser.error("steps, runs and repeats must be positive")
    args.source = args.source.resolve()
    if args.worker:
        if any(
            getattr(args, name) is None
            for name in ("backend", "preset", "profile", "mode", "seed")
        ):
            parser.error("worker requires backend, preset, profile, mode and seed")
        print(json.dumps(worker(args), allow_nan=False))
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
