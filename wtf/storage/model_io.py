"""Lossless, content-addressed champion weights; JSON metadata stays readable."""

import hashlib
import io
import lzma
from pathlib import Path
import re

from wtf.storage.binary_io import MAGIC, MAX_RECORD, RecordWriter, records


def model_path(artifact_path, reference):
    """V3 weights live in the shared models directory beside run generations."""
    digest = reference.get("sha256", "")
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or reference.get("weights_file") != f"../models/{digest}.bin.xz"
    ):
        raise ValueError("Invalid champion weight reference")
    root = Path(artifact_path).resolve().parents[2].parent
    path = (root / "models" / f"{digest}.bin.xz").resolve()
    if root not in path.parents:
        raise ValueError("Champion weights outside archive")
    return path


def write_model(model, staging, pool):
    buffer = io.BytesIO()
    RecordWriter(buffer).write(dict(kind="champion_model", model=model))
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    existing = Path(pool) / f"{digest}.bin.xz"
    target = Path(staging) / "models" / existing.name
    if existing.exists():
        try:
            with lzma.open(existing, "rb") as stream:
                if stream.read(len(payload) + 1) != payload:
                    raise ValueError("Existing champion weight checksum mismatch")
        except (lzma.LZMAError, EOFError) as error:
            raise ValueError("Invalid compressed champion weights") from error
    elif not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with lzma.open(target, "wb", preset=6) as stream:
            stream.write(payload)
    return dict(
        signature=model["signature"],
        precision=model.get("precision", "float64"),
        weights_file=f"../models/{digest}.bin.xz",
        sha256=digest,
    )


def publish_models(staging, pool):
    """Publish complete blobs before indexes; a failed export exposes no index."""
    staged = Path(staging) / "models"
    if staged.exists():
        Path(pool).mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            path.replace(Path(pool) / path.name)
        staged.rmdir()


def read_model(artifact_path, reference):
    try:
        with lzma.open(model_path(artifact_path, reference), "rb") as stream:
            payload = stream.read(MAX_RECORD + len(MAGIC) + 5)
    except (lzma.LZMAError, EOFError) as error:
        raise ValueError("Invalid compressed champion weights") from error
    if len(payload) > MAX_RECORD + len(MAGIC) + 4:
        raise ValueError("Champion model payload too large")
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError("Champion weight checksum mismatch")
    values = list(records(io.BytesIO(payload)))
    if len(values) != 1 or values[0].get("kind") != "champion_model":
        raise ValueError("Invalid champion model payload")
    model = values[0]["model"]
    if (
        model["signature"] != reference["signature"]
        or model.get("precision", "float64") != reference["precision"]
    ):
        raise ValueError("Champion model metadata mismatch")
    return model
