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
                os.environ.setdefault("SDKROOT", str(sdk))
                break
