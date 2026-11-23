"""Immutable run-local checkpoint metadata; numeric training state stays in checkpoints."""

import hashlib
import json
import lzma
from pathlib import Path
import re
import uuid

MAX_METADATA = 16 * 1024 * 1024


class MetadataPool:
    """Store shared immutable metadata by content-derived identity."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.cache = {}

    def store(self, value):
        """Store a shared immutable metadata value and return its reference."""
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > MAX_METADATA:
            raise ValueError("Checkpoint metadata exceeds size limit")
        digest = hashlib.sha256(raw).hexdigest()
        if digest in self.cache:
            return digest
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / (digest + ".json.xz")
        if path.exists():
            self.load(digest)
        else:
            temporary = self.folder / ("." + uuid.uuid4().hex + ".tmp")
            try:
                with lzma.open(temporary, "wb", preset=6) as stream:
                    stream.write(raw)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        self.cache[digest] = value
        return digest

    def load(self, digest):
        """Read persisted data and validate its supported contract."""
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid checkpoint metadata reference")
        if digest not in self.cache:
            path = self.folder / (digest + ".json.xz")
            if path.resolve().parent != self.folder.resolve():
                raise ValueError("Checkpoint metadata outside pool")
            try:
                with lzma.open(path, "rb") as stream:
                    raw = stream.read(MAX_METADATA + 1)
            except (OSError, lzma.LZMAError, EOFError) as error:
                raise ValueError("Missing or corrupt checkpoint metadata") from error
            if len(raw) > MAX_METADATA or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Checkpoint metadata checksum mismatch")
            self.cache[digest] = json.loads(raw)
        return self.cache[digest]
