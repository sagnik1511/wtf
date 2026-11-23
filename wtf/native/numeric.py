"""Packed model storage; Python float64 arithmetic and explicit snapshot boundaries.

Float32 rounds on storage, not on every arithmetic operation. Legacy checkpoints
without precision metadata retain their list-based float64 training behavior.
"""

from array import array
import copy
import math
from contextlib import contextmanager
from contextvars import ContextVar

_DEFAULT_PRECISION = ContextVar("brain_precision", default="float32")


@contextmanager
def precision_scope(precision):
    """Temporarily select the requested numeric precision contract."""
    if precision not in ("float32", "float64"):
        raise ValueError("Unsupported brain precision: " + str(precision))
    token = _DEFAULT_PRECISION.set(precision)
    try:
        yield
    finally:
        _DEFAULT_PRECISION.reset(token)


class Float32(array):
    """Compact floating-point storage compatible with the numeric helpers."""

    __slots__ = ()

    def __new__(cls, values=()):
        result = super().__new__(cls, "f", values)
        if result.itemsize != 4:
            raise RuntimeError("A four-byte float representation is required")
        if not all(math.isfinite(value) for value in result):
            raise ValueError("Non-finite float32 model value")
        return result

    def copy(self):
        """Copy this compact numeric storage into an independent array."""
        return type(self)(self)

    def __deepcopy__(self, memo):
        result = self.copy()
        memo[id(self)] = result
        return result


def packed(value, precision=None):
    """Copy a tree of parameter/optimizer lists into independent packed arrays."""
    precision = _DEFAULT_PRECISION.get() if precision is None else precision
    if precision not in ("float32", "float64"):
        raise ValueError("Unsupported brain precision: " + str(precision))
    if isinstance(value, dict):
        return {key: packed(values, precision) for key, values in value.items()}
    if isinstance(value, (list, tuple, array)):
        if value and isinstance(value[0], dict):
            return [packed(layer, precision) for layer in value]
        return Float32(value) if precision == "float32" else list(value)
    return copy.deepcopy(value)


def plain(value):
    """Detached JSON-compatible state, without leaking mutable model buffers."""
    if isinstance(value, dict):
        return {key: plain(values) for key, values in value.items()}
    if isinstance(value, array):
        return value.tolist()
    if isinstance(value, list):
        return [plain(values) for values in value]
    return copy.deepcopy(value)


def precision_of(brain):
    """Return the component's persisted numeric precision."""
    values = brain.weights if hasattr(brain, "weights") else brain.network.layers
    leaf = (
        next(iter(values.values()))
        if isinstance(values, dict)
        else next(iter(values[0].values()))
    )
    return "float32" if isinstance(leaf, Float32) else "float64"


def restore_numeric(brain, state):
    """Restore numeric trees with recorded precision (old checkpoints are f64)."""
    precision = state.get("precision", "float64")
    for key in ("weights", "target", "moments", "variances"):
        if key in state:
            setattr(brain, key, packed(state[key], precision))
    for key, attribute in (
        ("network", "network"),
        ("target_network", "target_network"),
        ("reference_network", "reference_network"),
    ):
        if key in state:
            getattr(brain, attribute).layers = packed(state[key], precision)
