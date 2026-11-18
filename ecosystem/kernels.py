"""Optional compiled CPU kernels; never compile or install during imports.

Use WTF_KERNELS=python for the reference path or native to require the extension.
Python 3.12 changed float summation; keep its reference behavior until the native
reduction has a separately verified implementation for that interpreter.
"""
import os
import sys
from array import array

_mode = os.environ.get('WTF_KERNELS', 'auto')
if _mode not in ('auto', 'python', 'native'):
    raise ValueError('WTF_KERNELS must be auto, python or native')
native = None
_reason = 'disabled by WTF_KERNELS=python'
if _mode != 'python':
    if sys.implementation.name != 'cpython' or sys.version_info >= (3, 12):
        _reason = 'native reductions currently support CPython 3.9–3.11'
    else:
        try:
            from . import _kernels as native
        except ImportError as error:
            _reason = str(error)
        else:
            _reason = None
    if _mode == 'native' and native is None:
        raise RuntimeError('Native kernels unavailable: ' + _reason +
                           '. Build with .venv/bin/python -m work.build_native')


def status():
    return dict(backend='native' if native is not None else 'python',
                requested=_mode, fallback_reason=_reason,
                native_api_version=getattr(native,'api_version',1) if native is not None else None)


def gradient_zeros(values):
    # Legacy list-backed networks preserve the reference representation too.
    packed_gradient = isinstance(values, array) and (native is not None or values.typecode == 'd')
    return array('d', [0.0]) * len(values) if packed_gradient else [0.0] * len(values)


def available(name):
    if native is None:return False
    version=getattr(native, 'api_version', 1)
    return type(version) is int and version == 2 and hasattr(native, name)
