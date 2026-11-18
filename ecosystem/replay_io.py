"""Lossless replay chunk I/O shared by the recorder, server and analyzer."""

import gzip
import json
import lzma
from pathlib import Path
from .replay_codec import decode_chunk


def chunk_path(folder, run, index):
    folder = Path(folder).resolve()
    stem = f"run_{run:04d}_chunk_{index:06d}"
    for suffix in (".bin.xz", ".bin.gz", ".bin", ".json.xz", ".json.gz", ".json"):
        path = (folder / (stem + suffix)).resolve()
        if path.parent != folder:
            raise ValueError("Chunk path outside recording folder")
        if path.is_file():
            return path
    raise FileNotFoundError("Recorded chunk not found")


def read_chunk_bytes(path):
    path = Path(path)
    if path.suffix in (".gz", ".xz"):
        opener = gzip.open if path.suffix == ".gz" else lzma.open
        with opener(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def load_chunk_payload(path):
    path = Path(path)
    from .binary_io import MAGIC, replay_payload

    data = read_chunk_bytes(path)
    if data.startswith(MAGIC):
        return replay_payload(data)
    return json.loads(data)


def load_chunk(path):
    return decode_chunk(load_chunk_payload(path))
