"""Verify and compact existing checkpoint files, leaving active experiments alone."""

import argparse
import hashlib
import json
from pathlib import Path
import uuid

from wtf.storage.checkpoint_io import read_checkpoint, write_checkpoint


def state_digest(state):
    """Compute a stable checksum of decoded continuation state."""
    result = hashlib.sha256()
    for part in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), allow_nan=False
    ).iterencode(state):
        result.update(part.encode())
    return result.digest()


def file_digest(path):
    """Compute a content checksum without retaining the entire file."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            result.update(block)
    return result.digest()


def optimize_experiment(folder):
    """Rewrite supported artifacts only after logical-equivalence validation."""

    # This import is deferred until the operation runs, after both modules have
    # initialized.
    # pylint: disable=cyclic-import

    # Keep independent numeric/compatibility or benchmark paths explicit to preserve
    # their exact contracts.
    # pylint: disable=duplicate-code

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    folder = Path(folder).resolve()
    playback = folder / "playback/recording.json"
    if (
        playback.is_file()
        and json.loads(playback.read_text()).get("status") == "writing"
    ):
        raise ValueError(
            "Experiment is still writing; finish training before optimizing checkpoints"
        )
    checkpoints = (folder / "checkpoints").resolve()
    if checkpoints.parent != folder:
        raise ValueError("Checkpoint directory outside experiment")
    if not checkpoints.is_dir():
        raise FileNotFoundError("No checkpoint directory")
    if not playback.is_file() and not any(
        (checkpoints / name).is_file()
        for name in ("final.json", "final.json.gz", "final.bin.gz", "final.bin.xz")
    ):
        raise ValueError("No completion marker; experiment may still be writing")
    if any(".bin" in p.suffixes for p in checkpoints.iterdir()):
        from wtf.storage.optimize_storage import optimize_experiment as optimize_binary

        result = optimize_binary(folder, replays=False, checkpoints=True)
        return {
            "folder": result["folder"],
            "before_bytes": result["before_bytes"],
            "after_bytes": result["after_bytes"],
            "saved_bytes": result["saved_bytes"],
            "checkpoints_converted": result["files_reduced"],
        }
    stems = sorted(
        {
            p.name[:-3] if p.name.endswith(".gz") else p.name
            for p in checkpoints.iterdir()
            if p.name.endswith((".json", ".json.gz"))
        }
    )
    before = after = converted = 0
    for name in stems:
        raw, packed = (checkpoints / name).resolve(), (
            checkpoints / (name + ".gz")
        ).resolve()
        if raw.parent != checkpoints or packed.parent != checkpoints:
            raise ValueError("Checkpoint path outside experiment")
        originals = [p for p in (raw, packed) if p.is_file()]
        before += sum(p.stat().st_size for p in originals)
        hashes = {p: file_digest(p) for p in originals}
        source = packed if packed.is_file() else raw
        state = read_checkpoint(source)
        expected = state_digest(state)
        for path in originals:
            if path != source and state_digest(read_checkpoint(path)) != expected:
                raise ValueError(f"Existing checkpoint copies differ: {name}")
        temporary = checkpoints / ("." + uuid.uuid4().hex + ".json.gz")
        try:
            metadata = {k: v for k, v in state.items() if k != "brains"}
            write_checkpoint(temporary, metadata, state["brains"].items())
            if state_digest(read_checkpoint(temporary)) != expected:
                raise ValueError(f"Lossless checkpoint verification failed: {name}")
            if any(file_digest(p) != value for p, value in hashes.items()):
                raise ValueError(f"Checkpoint changed during conversion: {name}")
            smallest = min(originals, key=lambda p: p.stat().st_size)
            if temporary.stat().st_size < smallest.stat().st_size:
                temporary.replace(packed)
                if raw.exists():
                    raw.unlink()
                converted += 1
                kept = packed
            else:
                kept = smallest
                for path in originals:
                    if path != kept:
                        path.unlink()
            after += kept.stat().st_size
        finally:
            if temporary.exists():
                temporary.unlink()
    return {
        "folder": str(folder),
        "before_bytes": before,
        "after_bytes": after,
        "saved_bytes": before - after,
        "checkpoints_converted": converted,
    }


def main():
    """Parse command-line options and run the requested operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="Experiment directory, or results root with --all"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Optimize completed/interrupted experiments; skip active ones",
    )
    args = parser.parse_args()
    folders = (
        sorted(p.parent for p in args.path.glob("*/checkpoints"))
        if args.all
        else [args.path]
    )
    totals = {
        "before_bytes": 0,
        "after_bytes": 0,
        "saved_bytes": 0,
        "checkpoints_converted": 0,
    }
    for folder in folders:
        try:
            result = optimize_experiment(folder)
        except ValueError as error:
            if args.all and (
                "still writing" in str(error) or "may still be writing" in str(error)
            ):
                print(f"Skipped active/unconfirmed experiment: {folder}", flush=True)
                continue
            raise
        print(json.dumps(result), flush=True)
        for key in totals:
            totals[key] += result[key]
    print("Total:", json.dumps(totals))


if __name__ == "__main__":
    main()
