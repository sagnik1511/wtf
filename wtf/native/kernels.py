"""Optional compiled CPU kernels; never compile or install during imports.

Use WTF_KERNELS=python for the reference path or native to require the extension.
Native float reductions match builtin sum() on CPython 3.12 and 3.13.
New interpreter versions fall back until numerical equivalence is validated.
"""

# This public backend/layout identifier is established API rather than an uppercase
# scalar constant.
# pylint: disable=invalid-name


import os
import sys
from array import array

_mode = os.environ.get("WTF_KERNELS", "auto")
if _mode not in ("auto", "python", "native"):
    raise ValueError("WTF_KERNELS must be auto, python or native")
native = None
_reason = "disabled by WTF_KERNELS=python"
if _mode != "python":
    if sys.implementation.name != "cpython" or not (3, 12) <= sys.version_info[:2] < (
        3,
        14,
    ):
        _reason = "native reductions currently support CPython 3.12–3.13"
    else:
        try:
            from wtf.native import _kernels as native
        except ImportError as error:
            _reason = str(error)
        else:
            _reason = None
    if _mode == "native" and native is None:
        raise RuntimeError(
            "Native kernels unavailable: "
            + _reason
            + ". Build with .venv/bin/python -m work.build_native"
        )


def status():
    """Describe the selected native backend and any fallback reason."""
    return {
        "backend": "native" if native is not None else "python",
        "requested": _mode,
        "fallback_reason": _reason,
        "native_api_version": (
            getattr(native, "api_version", 1) if native is not None else None
        ),
    }


def gradient_zeros(values):
    # Legacy list-backed networks preserve the reference representation too.
    """Allocate gradient storage compatible with the active numeric representation."""
    packed_gradient = isinstance(values, array) and (
        native is not None or values.typecode == "d"
    )
    return array("d", [0.0]) * len(values) if packed_gradient else [0.0] * len(values)


def available(name):
    """Check whether the loaded native backend supports the requested operation."""

    # Exact types intentionally reject bool/subclass substitutions in persisted
    # numeric contracts.
    # pylint: disable=unidiomatic-typecheck

    if native is None:
        return False
    version = getattr(native, "api_version", 1)
    return type(version) is int and version == 2 and hasattr(native, name)
