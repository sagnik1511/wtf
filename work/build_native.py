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
from _build_support import configure_compiler


def main():
    """Parse command-line options and run the requested operation."""
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        raise SystemExit("Run this command from the repository root")
    if sys.implementation.name != "cpython" or not (3, 12) <= sys.version_info[:2] < (
        3,
        14,
    ):
        raise SystemExit(
            "Native builds support CPython 3.12–3.13; use the Python backend on other interpreters"
        )
    configure_compiler()
    setup(
        name="wtf-native-kernels",
        version="0.0.0",
        packages=["wtf", "wtf.native"],
        ext_modules=[
            Extension(
                "wtf.native._kernels",
                ["wtf/native/_kernels.c"],
                depends=[str(p) for p in Path("wtf/native").glob("_kernels*.h")],
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
    artifact = Path(".cache/native-build/lib/wtf/native") / (
        "_kernels" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    target = Path("wtf/native") / artifact.name
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
