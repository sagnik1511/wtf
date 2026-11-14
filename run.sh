#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
exec .venv/bin/python -m ecosystem.server "$@"
