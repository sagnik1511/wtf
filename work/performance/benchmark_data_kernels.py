"""Benchmark recording codecs, expansion and dashboard groups against a source tree.

Run after benchmark_kernels, not concurrently. Fixtures and results stay ignored.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import tempfile

ROOT = Path(".cache/native-v2")
CASES = ("binary_encode", "binary_decode", "replay_expand", "dashboard", "groups_500")


def prepare():
    from wtf.config import load
    from wtf.experiment.runner import Experiment
    from wtf.storage.recording import Recorder, RecordingStore
    from wtf.storage.replay_io import load_chunk

    c = load("environments/ppo.json")
    c["seed"] = 42
    c["experiment"].update(runs=1, world_seed_policy="increment")
    c["world"]["max_steps"] = 64
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        e = Experiment(c)
        recorder = Recorder(folder, c, "train")
        recorder.begin_run(e)
        while not e.done:
            e.step()
            recorder.capture(e, force=e.finished_run)
        recorder.end_run(e)
        recorder.close()
        frames = []
        for file in sorted((Path(folder) / "playback").glob("*.bin*")):
            frames.extend(load_chunk(file))
        if not frames:
            raise RuntimeError("No fixture frames")
    rng = random.Random(42)
    data = dict(
        config=c,
        frames=frames,
        animals=[dict(x=rng.randrange(40), y=rng.randrange(28)) for _ in range(500)],
    )
    (ROOT / "data-fixtures.json").write_text(json.dumps(data, separators=(",", ":")))


def worker(args):
    if args.reference:
        sys.path.insert(0, str(args.reference.resolve()))
    from wtf.storage.binary_io import Encoder, Decoder
    from wtf.storage.replay_codec import encode_chunk, decode_chunk
    from wtf.inspection.telemetry import summarize, groups

    data = json.loads((ROOT / "data-fixtures.json").read_text())
    payload = encode_chunk(data["frames"])
    encoded = Encoder().encode(payload)
    loops = dict(
        binary_encode=20,
        binary_decode=20,
        replay_expand=10,
        dashboard=3,
        groups_500=100,
    )[args.case]
    outputs = None
    start = time.perf_counter()
    for _ in range(loops):
        if args.case == "binary_encode":
            outputs = Encoder().encode(payload)
        elif args.case == "binary_decode":
            outputs = Decoder().decode(encoded)
        elif args.case == "replay_expand":
            outputs = decode_chunk(payload)
        elif args.case == "dashboard":
            outputs = [
                summarize(data["config"], f, data["frames"][i - 1] if i else None)
                for i, f in enumerate(data["frames"])
            ]
        else:
            outputs = groups(data["animals"])
    seconds = time.perf_counter() - start
    semantic = (
        bytes(outputs)
        if isinstance(outputs, (bytes, bytearray))
        else json.dumps(outputs, sort_keys=True, separators=(",", ":")).encode()
    )
    print(
        json.dumps(
            dict(
                case=args.case,
                seconds=seconds,
                loops=loops,
                sha256=hashlib.sha256(semantic).hexdigest(),
            )
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--case", choices=CASES)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.worker:
        return worker(args)
    if not args.reference:
        p.error("Provide --reference containing the previous native setup")
    ROOT.mkdir(parents=True, exist_ok=True)
    prepare()
    results = []
    for case in CASES:
        for i in range(args.repeats):
            for mode in (
                ("native", "reference") if i % 2 == 0 else ("reference", "native")
            ):
                cmd = [
                    sys.executable,
                    "-m",
                    "work.performance.benchmark_data_kernels",
                    "--worker",
                    "--case",
                    case,
                ]
                if mode == "reference":
                    cmd += ["--reference", str(args.reference)]
                r = subprocess.run(
                    cmd,
                    env=dict(
                        os.environ, WTF_KERNELS="native", PYTHONDONTWRITEBYTECODE="1"
                    ),
                    capture_output=True,
                    text=True,
                )
                if r.returncode:
                    raise RuntimeError(r.stderr)
                sample = json.loads(r.stdout)
                sample.update(mode=mode, repeat=i + 1)
                results.append(sample)
                print(case, mode, round(sample["seconds"], 4), flush=True)
    summary = {}
    for case in CASES:
        selected = [s for s in results if s["case"] == case]
        if len({s["sha256"] for s in selected}) != 1:
            raise RuntimeError("Data semantics differ: " + case)
        values = {
            m: statistics.median(s["seconds"] for s in selected if s["mode"] == m)
            for m in ("native", "reference")
        }
        summary[case] = dict(
            values,
            speedup=values["reference"] / values["native"],
            loops=selected[0]["loops"],
            sha256=selected[0]["sha256"],
        )
    report = dict(
        status="complete", repeats=args.repeats, summary=summary, samples=results
    )
    (ROOT / "data-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
