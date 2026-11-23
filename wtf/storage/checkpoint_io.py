"""Lossless portable numeric arrays and shared metadata for resume checkpoints.

Public loaders always return the original version-1 state. New compressed files
stream one brain at a time; plain JSON output stays compatible with old readers.
"""

from array import array
from collections import OrderedDict
import base64
import copy
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import sys

ENCODING = "wtf-checkpoint-v2"
TAGS = {"$ref", "$literal", "$f64", "$u32", "$meta"}
SHARED = {"config", "actions", "model", "signature"}


def dumps(value):
    """Encode checkpoint state into its supported binary representation."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


class Encoder:
    """Encode values into the versioned binary representation."""

    def __init__(self, binary=False, metadata_pool=None):
        self.pool = metadata_pool
        self.binary = binary
        self.shared = []
        self.indices = {}
        self._identity_refs = None

    def _encode_metadata(self, value):
        """Borrow stable metadata only for this synchronous traversal.

        Never memoize the streamed brain iterator: it can reuse/mutate yielded
        containers. Public encode() also keeps its mutation-sensitive behavior.
        """
        self._identity_refs = OrderedDict()
        try:
            return self.encode(value)
        finally:
            self._identity_refs = None

    def encode(self, value, key=None):
        """Encode a value using this component's binary representation."""
        if (
            self.pool is not None
            and key in {"config", "actions", "signature"}
            and isinstance(value, (dict, list, tuple))
        ):
            return {"$meta": self.pool.store(value)}
        if isinstance(value, (dict, list, tuple)) and (
            key in SHARED
            or isinstance(value, dict)
            and {"signature", "parameters"}
            <= set(value)
            <= {"signature", "parameters", "precision"}
        ):
            memo = self._identity_refs
            identity = id(value)
            if memo is not None and identity in memo:
                _, index = memo[identity]
                memo.move_to_end(identity)
                return {"$ref": index}
            fingerprint = hashlib.sha256(
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).digest()
            if fingerprint not in self.indices:
                # Model-shaped dictionaries would match again; bypass just this
                # node while recursively encoding its contents.
                encoded = self.contents(value)
                self.indices[fingerprint] = len(self.shared)
                self.shared.append(encoded)
            index = self.indices[fingerprint]
            if memo is not None:
                # Strong references prevent ID reuse. Metadata already owns the
                # values; retain at most 128 references, never serialized copies.
                memo[identity] = value, index
                if len(memo) > 128:
                    memo.popitem(last=False)
            return {"$ref": index}
        return self.contents(value)

    def contents(self, value):
        """Return the encoder's accumulated binary contents."""

        # Exact types intentionally reject bool/subclass substitutions in persisted
        # numeric contracts.
        # pylint: disable=unidiomatic-typecheck

        if isinstance(value, dict):
            if TAGS.intersection(value):
                return {"$literal": [[k, self.encode(v, k)] for k, v in value.items()]}
            return {k: self.encode(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            code = None
            if len(value) >= 16 and all(type(v) is float for v in value):
                if not all(math.isfinite(v) for v in value):
                    raise ValueError("Non-finite checkpoint number")
                code, tag = "d", "$f64"
            elif len(value) >= 16 and all(
                type(v) is int and 0 <= v < 2**32 for v in value
            ):
                code, tag = "I", "$u32"
            if code and self.binary:
                return value if isinstance(value, list) else list(value)
            if code:
                packed = array(code, value)
                if packed.itemsize != (8 if code == "d" else 4):
                    raise ValueError("Unsupported native numeric array width")
                if sys.byteorder != "little":
                    packed.byteswap()
                return {tag: base64.b64encode(packed.tobytes()).decode("ascii")}
            return [self.encode(v) for v in value]
        return value


class Decoder:
    """Read values from the versioned binary representation."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-few-public-methods

    def __init__(self, shared, metadata_pool=None):
        if not isinstance(shared, list):
            raise ValueError("Invalid checkpoint shared table")
        self.pool = metadata_pool
        self.shared, self.decoded, self.resolving = shared, {}, set()

    def decode(self, value):
        """Decode a value from this component's binary representation."""

        # Exact types intentionally reject bool/subclass substitutions in persisted
        # numeric contracts.
        # pylint: disable=unidiomatic-typecheck
        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-return-statements

        if isinstance(value, list):
            return [self.decode(v) for v in value]
        if not isinstance(value, dict):
            return value
        tags = TAGS.intersection(value)
        if not tags:
            return {k: self.decode(v) for k, v in value.items()}
        if len(value) != 1:
            raise ValueError("Invalid checkpoint tag")
        tag = next(iter(tags))
        data = value[tag]
        if tag == "$meta":
            if self.pool is None:
                raise ValueError("Checkpoint requires shared metadata pool")
            return copy.deepcopy(self.pool.load(data))
        if tag == "$ref":
            if type(data) is not int or not 0 <= data < len(self.shared):
                raise ValueError("Invalid checkpoint reference")
            if data in self.resolving:
                raise ValueError("Cyclic checkpoint reference")
            if data not in self.decoded:
                self.resolving.add(data)
                try:
                    self.decoded[data] = self.decode(self.shared[data])
                finally:
                    self.resolving.remove(data)
            return copy.deepcopy(self.decoded[data])
        if tag == "$literal":
            return {k: self.decode(v) for k, v in data}
        if not isinstance(data, str):
            raise ValueError("Invalid checkpoint numeric data")
        raw = base64.b64decode(data, validate=True)
        code, width = ("d", 8) if tag == "$f64" else ("I", 4)
        if len(raw) % width:
            raise ValueError("Invalid checkpoint array length")
        packed = array(code)
        if packed.itemsize != width:
            raise ValueError("Unsupported native numeric array width")
        packed.frombytes(raw)
        if sys.byteorder != "little":
            packed.byteswap()
        if code == "d" and not all(math.isfinite(v) for v in packed):
            raise ValueError("Non-finite checkpoint number")
        return packed.tolist()


def decode_checkpoint(payload, metadata_pool=None):
    """Decode a checkpoint while enforcing supported format constraints."""
    if payload.get("checkpoint_version") == 1:
        return payload
    if (
        payload.get("checkpoint_version") != 2
        or payload.get("checkpoint_encoding") != ENCODING
    ):
        raise ValueError("Unsupported checkpoint encoding")
    decoder = Decoder(payload["shared"], metadata_pool)
    state = decoder.decode(payload["metadata"])
    state["brains"] = {
        key: decoder.decode(value) for key, value in payload["brains"].items()
    }
    if state.get("checkpoint_version") != 1:
        raise ValueError("Invalid checkpoint metadata")
    return state


def read_checkpoint(path):
    """Read a supported checkpoint file and resolve its metadata."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    from wtf.storage.binary_io import MAGIC, records

    path = Path(path)
    if not path.is_file():
        stem = path.name
        for suffix in (".json.gz", ".json", ".bin.xz", ".bin.gz", ".bin"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        for suffix in (".bin.xz", ".bin.gz", ".json.gz", ".json", ".bin"):
            candidate = path.with_name(stem + suffix)
            if candidate.is_file():
                path = candidate
                break
    with path.open("rb") as source:
        magic = source.read(6)
        compressed = magic.startswith(b"\x1f\x8b")
    opener = (
        lzma.open if magic == b"\xfd7zXZ\x00" else gzip.open if compressed else open
    )
    with opener(path, "rb") as handle:
        magic = handle.read(len(MAGIC))
        handle.seek(0)
        if magic == MAGIC:
            values = records(handle)
            header = next(values, None)
            if (
                not isinstance(header, dict)
                or header.get("kind") != "checkpoint"
                or header.get("version") not in (3, 4)
            ):
                raise ValueError("Unsupported binary checkpoint")
            brains = {}
            shared = None
            for record in values:
                if record.get("kind") == "brain" and shared is None:
                    key = record["id"]
                    if key in brains:
                        raise ValueError("Duplicate binary brain")
                    brains[key] = record["state"]
                elif record.get("kind") == "shared" and shared is None:
                    shared = record["values"]
                else:
                    raise ValueError("Invalid binary checkpoint record")
            if shared is None:
                raise ValueError("Missing binary checkpoint shared table")
            from wtf.storage.metadata_pool import MetadataPool

            return decode_checkpoint(
                {
                    "checkpoint_version": 2,
                    "checkpoint_encoding": ENCODING,
                    "metadata": header["metadata"],
                    "brains": brains,
                    "shared": shared,
                },
                (
                    MetadataPool(path.parent / "_metadata")
                    if header["version"] == 4
                    else None
                ),
            )
        return decode_checkpoint(json.load(handle))


def write_checkpoint(path, metadata, brains, *, shared_metadata=False):
    """Write metadata and an iterable of (id, detached brain state) atomically."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel
    # This internal collaborator intentionally inspects the owning component's private
    # state.
    # pylint: disable=protected-access

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    binary = ".bin" in path.suffixes
    compact = path.suffix in (".gz", ".xz")
    from wtf.storage.metadata_pool import MetadataPool

    if shared_metadata and not binary:
        raise ValueError("Shared checkpoint metadata requires binary output")
    pool = MetadataPool(path.parent / "_metadata") if shared_metadata else None
    encoder = Encoder(binary=binary, metadata_pool=pool) if compact or binary else None
    try:
        opener = lzma.open if path.suffix == ".xz" else gzip.open if compact else open
        if binary:
            from wtf.storage.binary_io import RecordWriter

            with opener(
                temporary,
                "wb",
                **(
                    {
                        "filters": [
                            {"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": 1 << 20}
                        ]
                    }
                    if path.suffix == ".xz"
                    else {"compresslevel": 6} if compact else {}
                ),
            ) as handle:
                writer = RecordWriter(handle)
                writer.write(
                    {
                        "kind": "checkpoint",
                        "version": 4 if shared_metadata else 3,
                        "metadata": encoder._encode_metadata(metadata),
                    }
                )
                for animal_id, state in brains:
                    writer.write(
                        {
                            "kind": "brain",
                            "id": str(animal_id),
                            "state": encoder.encode(state),
                        }
                    )
                    del state
                writer.write({"kind": "shared", "values": encoder.shared})
            temporary.replace(path)
            return
        with opener(
            temporary,
            "wt",
            encoding="utf-8",
            **({"compresslevel": 6} if compact else {}),
        ) as handle:
            if compact:
                handle.write(
                    '{"checkpoint_version":2,"checkpoint_encoding":'
                    + dumps(ENCODING)
                    + ',"metadata":'
                )
                handle.write(dumps(encoder._encode_metadata(metadata)))
                handle.write(',"brains":{')
            else:
                handle.write(dumps(metadata)[:-1] + ',"brains":{')
            for index, (animal_id, state) in enumerate(brains):
                if index:
                    handle.write(",")
                handle.write(dumps(str(animal_id)) + ":")
                handle.write(dumps(encoder.encode(state) if compact else state))
                del state  # Release this detached brain before requesting the next one.
            handle.write("}")
            if compact:
                handle.write(',"shared":' + dumps(encoder.shared))
            handle.write("}\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
