"""Run maintained-source formatters and syntax checks without importing project code.

Use --fix to format every maintained source, then run the same checks. This is
not a semantic linter or a substitute for tests and independent code review.
"""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYTHON = {".py"}
NATIVE = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cu"}
JAVASCRIPT = {".js", ".mjs", ".cjs"}
WEB = JAVASCRIPT | {".css", ".html", ".json"}
SUPPORTED = PYTHON | NATIVE | WEB | {".sh", ".toml"}
EXCLUDED = (
    ".cache/",
    ".venv/",
    "node_modules/",
    "results/",
    "brain_library/",
    "dist/",
    "build/",
    "site/",
    "docs/paper/",
    "environments/local/",
)


def source_files(root=ROOT):
    """Inventory maintained tracked and non-ignored source files."""
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    files = []
    for name in sorted(set(os.fsdecode(raw).split("\0"))):
        path = root / name
        if not name or name.startswith(EXCLUDED) or path.suffix not in SUPPORTED:
            continue
        # Generated visual exports use their producer's checks, not formatters.
        # The two maintained viewer palette CSS files are renderer inputs.
        if name.startswith("docs/assets/") and path.suffix != ".css":
            continue
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source symlink leaves the checkout: {name}")
        files.append(name)
    return files


def syntax_errors(files, root=ROOT):
    """Collect syntax errors without executing the maintained source code."""
    errors = []
    node = shutil.which("node")
    for name in files:
        path = root / name
        try:
            if path.suffix in PYTHON:
                compile(path.read_text(encoding="utf-8"), name, "exec")
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".toml":
                tomllib.loads(path.read_text(encoding="utf-8"))
            elif path.suffix in JAVASCRIPT:
                if node is None:
                    errors.append(f"{name}: node is unavailable for syntax checking")
                    continue
                # Browser .js entries are ES modules. Parse through stdin so a
                # checkout without package.json is not incorrectly treated as CJS.
                mode = "commonjs" if path.suffix == ".cjs" else "module"
                result = subprocess.run(
                    [node, f"--input-type={mode}", "--check"],
                    input=path.read_text(encoding="utf-8"),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=root,
                    check=False,
                )
                if result.returncode:
                    errors.append(f"{name}: {result.stdout.strip()}")
        except (OSError, ValueError, SyntaxError) as error:
            errors.append(f"{name}: {error}")
    return errors


def formatter_commands(files, fix=False):
    """Build formatter commands for the maintained source inventory."""
    prettier = ROOT / ".cache/formatters/node_modules/.bin/prettier"
    shfmt = Path(sys.executable).parent / ("shfmt.exe" if os.name == "nt" else "shfmt")
    return [
        (
            "black",
            [sys.executable, "-m", "black", *([] if fix else ["--check"])],
            [name for name in files if Path(name).suffix in PYTHON],
        ),
        (
            "clang-format",
            [
                shutil.which("clang-format"),
                *(["-i"] if fix else ["--dry-run", "--Werror"]),
            ],
            [name for name in files if Path(name).suffix in NATIVE],
        ),
        (
            "prettier",
            [
                str(prettier) if prettier.is_file() else shutil.which("prettier"),
                "--write" if fix else "--check",
            ],
            [name for name in files if Path(name).suffix in WEB],
        ),
        (
            "shfmt",
            [
                str(shfmt) if shfmt.is_file() else shutil.which("shfmt"),
                "-i",
                "2",
                "-w" if fix else "-d",
            ],
            [name for name in files if Path(name).suffix == ".sh"],
        ),
    ]


def main(argv=None):
    """Parse command-line options and run the requested operation."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-locals

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--fix",
        action="store_true",
        help="Format all maintained source files, then check",
    )
    modes.add_argument("--check", action="store_true", help="Check only (the default)")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print selected files without checking or changing them",
    )
    args = parser.parse_args(argv)
    files = source_files()
    if args.list:
        print("\n".join(files))
        return 0
    logs = ROOT / ".cache/quality"
    logs.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "BLACK_CACHE_DIR": str(ROOT / ".cache/black"),
    }
    failed = False
    for fix in [True, False] if args.fix else [False]:
        for name, command, targets in formatter_commands(files, fix):
            if not targets:
                continue
            log = logs / f'{name}-{"format" if fix else "check"}.log'
            if command[0] is None:
                log.write_text(
                    f"{name} is unavailable; see docs/development.md for tool setup.\n"
                )
                code = 1
            else:
                with log.open("w") as output:
                    result = subprocess.run(
                        command + targets,
                        cwd=ROOT,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                code = result.returncode
            print(
                f'{name} {"format" if fix else "check"}: {len(targets)} files; exit {code}; {log.relative_to(ROOT)}',
                flush=True,
            )
            if code:
                failed = True
                print(log.read_text()[-4000:])
    errors = syntax_errors(files)
    (logs / "syntax.log").write_text(
        "\n".join(errors)
        + ("\n" if errors else "Syntax/configuration parsing passed.\n")
    )
    print(
        f"Syntax/configuration parsing: {len(files)} maintained files; {len(errors)} errors",
        flush=True,
    )
    if errors:
        failed = True
        print("\n".join(errors))
    result = subprocess.run(
        ["git", "diff", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        failed = True
        print(result.stdout + result.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
