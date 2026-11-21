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
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        raise SystemExit("Run this command from the repository root")
    configure_compiler()
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
