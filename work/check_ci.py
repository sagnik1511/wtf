"""Run the Python CI gates locally: Black, full Pylint, and distribution builds."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from work.check_quality import ROOT, source_files


def run_command(command, root=ROOT, env=None):
    """Preserve each tool's output and report missing executables as failures."""
    print("+ " + " ".join(map(str, command)), flush=True)
    try:
        return subprocess.run(command, cwd=root, env=env, check=False).returncode
    except OSError as error:
        print(f"Could not run check: {error}", file=sys.stderr)
        return 1


def check_black(root=ROOT):
    """Check the shared maintained-source inventory, including untracked files."""
    files = [name for name in source_files(root) if name.endswith(".py")]
    if not files:
        print("No maintained Python files found.", file=sys.stderr)
        return 1
    return run_command(
        [sys.executable, "-m", "black", "--check", "--diff", *files], root
    )


def check_pylint(root=ROOT):
    """Check every maintained Python source with the full Pylint diagnostic set."""
    files = [name for name in source_files(root) if name.endswith(".py")]
    if not files:
        print("No maintained Python files found.", file=sys.stderr)
        return 1
    return run_command(
        [
            sys.executable,
            "-m",
            "pylint",
            "--persistent=n",
            "--reports=n",
            *files,
        ],
        root,
    )


def check_build(root=ROOT):
    """Build wheel via sdist and strictly check only this invocation's artifacts."""
    parent = root / ".cache/ci-builds"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="portable-", dir=parent) as folder:
        env = {**os.environ, "WTF_BUILD_NATIVE": "0", "WTF_BUILD_ENGINE": "0"}
        result = run_command(
            [sys.executable, "-m", "build", "--outdir", folder], root, env
        )
        if result:
            return result
        artifacts = sorted(Path(folder).glob("*"))
        if (
            len(artifacts) != 2
            or sum(p.suffix == ".whl" for p in artifacts) != 1
            or sum(p.name.endswith(".tar.gz") for p in artifacts) != 1
        ):
            print(
                "Expected exactly one wheel and one source archive from this build.",
                file=sys.stderr,
            )
            return 1
        return run_command(
            [sys.executable, "-m", "twine", "check", "--strict", *map(str, artifacts)],
            root,
        )


def main(argv=None):
    """Run selected gates, or all three by default, without hiding failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--black", action="store_true", help="Check all maintained Python with Black"
    )
    parser.add_argument(
        "--pylint",
        action="store_true",
        help="Run full Pylint on all maintained Python sources",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build portable wheel/sdist and run strict Twine metadata validation",
    )
    args = parser.parse_args(argv)
    checks = [("black", check_black), ("pylint", check_pylint), ("build", check_build)]
    selected = [
        (name, check) for name, check in checks if getattr(args, name)
    ] or checks
    failed = False
    for name, check in selected:
        result = check()
        print(f"{name}: exit {result}", flush=True)
        failed |= bool(result)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
