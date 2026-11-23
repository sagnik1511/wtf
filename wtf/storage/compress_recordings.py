"""Losslessly compress existing committed replay chunks, one file at a time."""

# Keep independent numeric/compatibility or benchmark paths explicit to preserve their
# exact contracts.
# pylint: disable=duplicate-code


import argparse
import gzip
import hashlib
import json
from pathlib import Path
import uuid

from wtf.storage.recording import write_json
from wtf.storage.replay_codec import encode_chunk, decode_chunk, CODEC
from wtf.storage.replay_io import chunk_path, load_chunk_payload


def digest(path, compressed=False):
    """Compute a stable checksum for workload-equivalence comparisons."""
    result = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            result.update(block)
    return result.digest()


def compress_recording(folder, level=6, delta=False):
    """Compress saved replay chunks while preserving their decoded content."""

    # This import is deferred until the operation runs, after both modules have
    # initialized.
    # pylint: disable=cyclic-import

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-nested-blocks
    # pylint: disable=too-many-statements
    # Exact types intentionally reject bool/subclass substitutions in persisted
    # numeric contracts.
    # pylint: disable=unidiomatic-typecheck
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    if type(level) is not int or not 1 <= level <= 9:
        raise ValueError("Compression level must be an integer from 1 to 9")
    folder = Path(folder).resolve()
    playback = folder if (folder / "recording.json").is_file() else folder / "playback"
    playback = playback.resolve()
    manifest_path = playback / "recording.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("recording_version") != 1:
        raise ValueError("Unsupported recording version")
    if manifest.get("status") == "writing":
        raise ValueError(
            "Recording is still being written; finish or close training before compression"
        )
    if any(
        ".bin" in chunk_path(playback, run["run"], i).suffixes
        for run in manifest["runs"]
        for i in range(run["chunk_count"])
    ):
        from wtf.storage.optimize_storage import optimize_experiment

        result = optimize_experiment(playback.parent, replays=True, checkpoints=False)
        return {
            "folder": result["folder"],
            "before_bytes": result["before_bytes"],
            "after_bytes": result["after_bytes"],
            "saved_bytes": result["saved_bytes"],
            "chunks_converted": result["files_reduced"],
        }
    if delta:
        return optimize_chunks(playback, manifest_path, manifest, level)
    before = after = converted = 0
    formats = set()
    for run in manifest["runs"]:
        for index in range(run["chunk_count"]):
            stem = f"run_{run['run']:04d}_chunk_{index:06d}.json"
            raw, packed = (playback / stem).resolve(), (
                playback / (stem + ".gz")
            ).resolve()
            if raw.parent != playback or packed.parent != playback:
                raise ValueError("Chunk path outside recording folder")
            if not raw.is_file() and not packed.is_file():
                raise FileNotFoundError(f"Missing committed chunk: {stem}")
            before += sum(p.stat().st_size for p in (raw, packed) if p.is_file())
            if raw.is_file():
                temporary = playback / (stem + "." + uuid.uuid4().hex + ".gz.tmp")
                try:
                    if packed.exists():
                        # A previous conversion may have stopped before removing
                        # the redundant plain file. Never trust it without checking.
                        if digest(raw) != digest(packed, compressed=True):
                            raise ValueError(
                                f"Existing compressed chunk differs: {stem}"
                            )
                    else:
                        original = hashlib.sha256()
                        with (
                            raw.open("rb") as source,
                            gzip.open(temporary, "wb", compresslevel=level) as target,
                        ):
                            while True:
                                block = source.read(1024 * 1024)
                                if not block:
                                    break
                                original.update(block)
                                target.write(block)
                        if (
                            digest(temporary, compressed=True) != original.digest()
                            or digest(raw) != original.digest()
                        ):
                            raise ValueError(f"Lossless verification failed: {stem}")
                        if temporary.stat().st_size >= raw.stat().st_size:
                            temporary.unlink()
                        else:
                            temporary.replace(packed)
                    if packed.is_file() and packed.stat().st_size < raw.stat().st_size:
                        raw.unlink()  # Only after byte-for-byte SHA-256 verification.
                        converted += 1
                    elif packed.is_file():
                        packed.unlink()  # The verified plain copy is smaller.
                finally:
                    if temporary.exists():
                        temporary.unlink()
            after += sum(p.stat().st_size for p in (raw, packed) if p.is_file())
            formats.add("gzip" if packed.is_file() else "none")
    manifest["storage_compression"] = (
        next(iter(formats), "none") if len(formats) <= 1 else "mixed"
    )
    write_json(manifest_path, manifest)
    return {
        "folder": str(playback.parent),
        "before_bytes": before,
        "after_bytes": after,
        "chunks_converted": converted,
        "saved_bytes": before - after,
    }


def frame_digest(frames):
    """Compute a stable checksum of decoded recording frames."""
    result = hashlib.sha256()
    for frame in frames:
        result.update(
            json.dumps(
                frame, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        result.update(b"\n")
    return result.digest()


def optimize_chunks(playback, manifest_path, manifest, level):
    """Verify expanded frames before atomically replacing each individual chunk."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-nested-blocks

    before = after = converted = 0
    encodings, compressions = set(), set()
    for run in manifest["runs"]:
        for index in range(run["chunk_count"]):
            source = chunk_path(playback, run["run"], index)
            stem = f"run_{run['run']:04d}_chunk_{index:06d}.json"
            raw, packed = (playback / stem).resolve(), (
                playback / (stem + ".gz")
            ).resolve()
            if raw.parent != playback or packed.parent != playback:
                raise ValueError("Chunk path outside recording folder")
            existing = [p for p in (raw, packed) if p.is_file()]
            before += sum(p.stat().st_size for p in existing)
            hashes = {p: digest(p) for p in existing}
            payload = load_chunk_payload(source)
            frames = decode_chunk(payload)
            expected = frame_digest(frames)
            for path in existing:
                if (
                    path != source
                    and frame_digest(decode_chunk(load_chunk_payload(path))) != expected
                ):
                    raise ValueError(f"Existing compressed chunk differs: {stem}")
            temporary = playback / (stem + "." + uuid.uuid4().hex + ".tmp")
            try:
                # Already compact chunks are idempotent; do not recompress them.
                compact = isinstance(payload, dict) and payload.get("codec") == CODEC
                if not compact or source.suffix != ".gz":
                    candidate = encode_chunk(frames)
                    with gzip.open(
                        temporary, "wt", encoding="utf-8", compresslevel=level
                    ) as handle:
                        json.dump(
                            candidate, handle, separators=(",", ":"), allow_nan=False
                        )
                    with gzip.open(temporary, "rt", encoding="utf-8") as handle:
                        verified = frame_digest(decode_chunk(json.load(handle)))
                    if verified != expected:
                        raise ValueError(f"Lossless verification failed: {stem}")
                    if any(digest(path) != value for path, value in hashes.items()):
                        raise ValueError(f"Chunk changed during conversion: {stem}")
                    smallest = min(existing, key=lambda p: p.stat().st_size)
                    if temporary.stat().st_size < smallest.stat().st_size:
                        temporary.replace(packed)
                        if raw.is_file():
                            raw.unlink()
                        converted += 1
                    else:
                        # Retain the smaller verified original representation.
                        for path in existing:
                            if path != smallest:
                                path.unlink()
                elif len(existing) > 1:
                    raw.unlink()
                current = chunk_path(playback, run["run"], index)
                after += current.stat().st_size
                stored = load_chunk_payload(current)
                encodings.add("delta" if isinstance(stored, dict) else "snapshot")
                compressions.add("gzip" if current.suffix == ".gz" else "none")
            finally:
                if temporary.exists():
                    temporary.unlink()
    manifest["storage_format"] = (
        next(iter(encodings), "delta") if len(encodings) <= 1 else "mixed"
    )
    manifest["storage_compression"] = (
        next(iter(compressions), "none") if len(compressions) <= 1 else "mixed"
    )
    write_json(manifest_path, manifest)
    return {
        "folder": str(playback.parent),
        "before_bytes": before,
        "after_bytes": after,
        "chunks_converted": converted,
        "saved_bytes": before - after,
    }


def main():
    """Parse command-line options and run the requested operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="Experiment folder, or results root with --all"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all non-writing recordings under the results root",
    )
    parser.add_argument("--level", type=int, default=6, choices=range(1, 10))
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Store lossless field changes between chunk keyframes, then gzip",
    )
    args = parser.parse_args()
    folders = (
        sorted(p.parent.parent for p in args.path.glob("*/playback/recording.json"))
        if args.all
        else [args.path]
    )
    if not folders:
        parser.error("No recordings found")
    totals = {
        "before_bytes": 0,
        "after_bytes": 0,
        "saved_bytes": 0,
        "chunks_converted": 0,
    }
    for folder in folders:
        if args.all:
            metadata = json.loads((folder / "playback/recording.json").read_text())
            if metadata.get("status") == "writing":
                print(f"Skipped active recording: {folder}")
                continue
        result = compress_recording(folder, args.level, delta=args.delta)
        for key in totals:
            totals[key] += result[key]
        print(json.dumps(result), flush=True)
    print("Total:", json.dumps(totals))


if __name__ == "__main__":
    main()
