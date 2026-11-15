"""Lossless replay chunk I/O shared by the recorder, server and analyzer."""
import gzip
import json
from pathlib import Path
from .replay_codec import decode_chunk


def chunk_path(folder, run, index):
    folder = Path(folder).resolve()
    stem = f'run_{run:04d}_chunk_{index:06d}'
    for suffix in ('.json.gz', '.json'):
        path = (folder / (stem + suffix)).resolve()
        if path.parent != folder:
            raise ValueError('Chunk path outside recording folder')
        if path.is_file():
            return path
    raise FileNotFoundError('Recorded chunk not found')


def read_chunk_bytes(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as handle:
            return handle.read()
    return path.read_bytes()


def load_chunk_payload(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def load_chunk(path):
    return decode_chunk(load_chunk_payload(path))
