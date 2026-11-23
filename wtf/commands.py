"""Installed command hub; subcommands retain their own argument parsers."""

import argparse
import sys


def main():
    """Parse command-line options and run the requested operation."""

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(prog="wtf", description="Wildlife Trial Framework")
    parser.add_argument("command", choices=("run", "view", "analyze", "presets"))
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    if args.command == "presets":
        if remaining:
            parser.error("presets does not accept additional arguments")
        from wtf.resources import preset_names

        print("\n".join(preset_names()))
        return
    if args.command == "run":
        from wtf.cli import main as execute
    elif args.command == "view":
        from wtf.server import main as execute
    else:
        from wtf.inspection.analyze import main as execute
    original = sys.argv
    try:
        sys.argv = [f"wtf {args.command}", *remaining]
        execute()
    finally:
        sys.argv = original
