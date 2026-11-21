"""Explicit runtime-asset packaging and opt-in native compilation."""

import json
import os
import shutil
import runpy
from pathlib import Path
import sys

from setuptools import Extension, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
ASSETS = json.loads((ROOT / "packaging_assets.json").read_text())


class BuildPy(build_py):
    def run(self):
        target_package = Path(self.build_lib) / "ecosystem"
        if target_package.resolve() == (ROOT / "ecosystem").resolve():
            raise ValueError("Build directory must not be the source package")
        if (target_package / "assets").exists():
            shutil.rmtree(target_package / "assets")
        if not self.distribution.ext_modules:
            for pattern in ("_kernels*.so", "_kernels*.pyd"):
                for stale in target_package.glob(pattern):
                    stale.unlink()
        super().run()
        for relative in ASSETS:
            source = (ROOT / relative).resolve()
            if ROOT not in source.parents or not source.is_file():
                raise ValueError(f"Invalid packaged asset: {relative}")
            target = Path(self.build_lib) / "ecosystem/assets" / relative
            self.mkpath(str(target.parent))
            self.copy_file(str(source), str(target))

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "ecosystem/assets" / relative)
            for relative in ASSETS
        ]


extensions = []
mode = os.environ.get("WTF_BUILD_NATIVE", "0")
if mode not in ("0", "1"):
    raise ValueError("WTF_BUILD_NATIVE must be 0 or 1")
if mode == "1":
    if sys.implementation.name != "cpython" or not (3, 9) <= sys.version_info[:2] < (
        3,
        12,
    ):
        raise RuntimeError(
            "Native builds currently support CPython 3.9–3.11; omit WTF_BUILD_NATIVE for the portable wheel"
        )
    runpy.run_path(str(ROOT / "_build_support.py"))["configure_compiler"]()
    extensions = [
        Extension(
            "ecosystem._kernels",
            ["ecosystem/_kernels.c"],
            depends=[
                str(p.relative_to(ROOT))
                for p in (ROOT / "ecosystem").glob("_kernels*.h")
            ],
            extra_compile_args=(
                ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fno-builtin-pow"]
                if os.name != "nt"
                else ["/O2", "/fp:strict"]
            ),
        )
    ]

setup(cmdclass={"build_py": BuildPy}, ext_modules=extensions)
