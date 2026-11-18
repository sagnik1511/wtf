"""Build optional CPU kernels with the current environment's Python/compiler.

Run from the project root: .venv/bin/python -m work.build_native
No runtime dependency is installed, and imports never trigger a compiler.
"""

import os
import sys
import shutil
import sysconfig
import tempfile
from pathlib import Path
from setuptools import Extension, setup


def main():
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        raise SystemExit("Run this command from the repository root")
    # Use installed binaries/SDK directly on macOS; Apple's command shims can
    # fail during an Xcode update. Explicit user CC/SDKROOT choices still win.
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
    setup(
        name="wtf-native-kernels",
        version="0.0.0",
        packages=["ecosystem"],
        ext_modules=[
            Extension(
                "ecosystem._kernels",
                ["ecosystem/_kernels.c"],
                depends=[str(p) for p in Path("ecosystem").glob("_kernels*.h")],
                extra_compile_args=[
                    "-O3",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                    "-fno-builtin-pow",
                ],
            )
        ],
        script_args=[
            "build_ext",
            "--force",
            "--build-temp",
            ".cache/native-build/temp",
            "--build-lib",
            ".cache/native-build/lib",
        ],
    )
    artifact = Path(".cache/native-build/lib/ecosystem") / (
        "_kernels" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    target = Path("ecosystem") / artifact.name
    # Publishing by rename preserves mappings in already-running processes.
    fd, name = tempfile.mkstemp(prefix="._kernels-", suffix=".so", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(artifact, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
