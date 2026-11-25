"""Explicit NVCC build: python -m wtf.native.cuda_build (Linux/NVIDIA only)."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main(argv=None):
    """Compile a local, optional CUDA library; imports never invoke a compiler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument(
        "--arch", default="native", help="NVCC architecture, e.g. native or sm_75"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("_cuda_dqn.so")
    )
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error(
            "Build CUDA on Linux with an NVIDIA GPU/CUDA toolkit (e.g. Colab)."
        )
    compiler = shutil.which(args.nvcc)
    if compiler is None:
        parser.error("nvcc was not found; install the CUDA toolkit or set NVCC.")
    if args.arch != "native" and not (
        args.arch.startswith("sm_") and args.arch[3:].isdigit()
    ):
        parser.error("--arch must be native or sm_<compute capability>")
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="._cuda_dqn-", suffix=".so", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        command = [
            compiler,
            "-std=c++17",
            "-O3",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "--fmad=false",
            f"-arch={args.arch}",
            str(Path(__file__).with_name("_cuda_dqn.cu")),
            "-o",
            str(temporary),
        ]
        print("Building optional CUDA kernels:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Built {destination}. Select it with --device cuda.")


if __name__ == "__main__":
    main()
