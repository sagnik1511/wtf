#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python 3.12+ is required; install it and recreate .venv")'
    python3 -m venv .venv
  fi
fi
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
exec .venv/bin/python -m ecosystem.server "$@"
