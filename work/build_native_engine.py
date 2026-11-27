"""Explicitly compile the optional native-owned simulation and learning engine."""

import os
from pathlib import Path
import shutil
import sysconfig
import tempfile

from setuptools import setup

from _build_support import engine_extension


def main():
    """Build locally and atomically publish only this interpreter's extension."""
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        raise SystemExit("Run this command from the repository root")
    extension = engine_extension(root)
    setup(
        name="wtf-native-engine",
        version="0.0.0",
        packages=[],
        ext_modules=[extension],
        script_args=[
            "build_ext",
            "--force",
            "--build-temp",
            ".cache/native-engine/temp",
            "--build-lib",
            ".cache/native-engine/lib",
        ],
    )
    name = "_engine" + sysconfig.get_config_var("EXT_SUFFIX")
    artifact = Path(".cache/native-engine/lib/wtf/native") / name
    destination = Path("wtf/native") / name
    fd, name = tempfile.mkstemp(
        prefix="._engine-", suffix=destination.suffix, dir=destination.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(artifact, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        "Built native simulation and learning engine. Select explicitly with --engine cpp."
    )


if __name__ == "__main__":
    main()
