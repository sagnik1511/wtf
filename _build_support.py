"""Shared build-time compiler selection; no effect on runtime imports."""

import os
from pathlib import Path
import sys


def configure_compiler():
    # Use installed binaries/SDK directly on macOS; Apple's command shims can
    # fail during an Xcode update. Explicit user CC/SDKROOT choices still win.
    """Configure only the supported platform-specific native build flags."""
    if sys.platform == "darwin":
        candidates = (
            [Path(os.environ["DEVELOPER_DIR"])]
            if os.environ.get("DEVELOPER_DIR")
            else [
                Path("/Applications/Xcode.app/Contents/Developer"),
                Path("/Library/Developer/CommandLineTools"),
            ]
        )
        for developer in candidates:
            cc = developer / "Toolchains/XcodeDefault.xctoolchain/usr/bin/clang"
            if not cc.is_file():
                cc = developer / "usr/bin/clang"
            sdk = developer / "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
            if not sdk.is_dir():
                sdk = developer / "SDKs/MacOSX.sdk"
            if cc.is_file() and (sdk / "usr/include/stdio.h").is_file():
                os.environ.setdefault("CC", str(cc))
                cxx = cc.with_name("clang++")
                if cxx.is_file():
                    os.environ.setdefault("CXX", str(cxx))
                os.environ.setdefault("SDKROOT", str(sdk))
                break


def engine_extension(root):
    """Share strict native-engine build settings between wheels and local builds."""
    # Build dependencies stay optional for runtime imports.
    # pylint: disable=import-outside-toplevel
    from setuptools import Extension

    if sys.implementation.name != "cpython" or not (3, 12) <= sys.version_info[:2] < (
        3,
        14,
    ):
        raise RuntimeError("The native engine currently supports CPython 3.12–3.13")
    configure_compiler()
    root = Path(root)
    core = root / "wtf/native/core"
    return Extension(
        "wtf.native._engine",
        sorted(str(path.relative_to(root)) for path in core.glob("*.cpp")),
        language="c++",
        depends=sorted(str(path.relative_to(root)) for path in core.glob("*.hpp")),
        extra_compile_args=(
            ["/std:c++17", "/O2", "/fp:strict"]
            if os.name == "nt"
            else [
                "-std=c++17",
                "-O3",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-fno-builtin-pow",
            ]
        ),
    )
