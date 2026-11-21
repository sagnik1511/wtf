#!/bin/sh
# Run from anywhere; all files/results belong to this checkout.
set -eu
cd "$(dirname "$0")/.."
export WTF_UID="${WTF_UID:-$(id -u)}"
export WTF_GID="${WTF_GID:-$(id -g)}"
usage() {
  echo 'Usage: docker/wtf.sh build | train [CLI arguments] | view [port] | down' >&2
}
case "${1:-}" in
build)
  shift
  docker compose build "$@"
  ;;
train)
  shift
  mkdir -p results
  if [ "$#" -eq 0 ]; then
    docker compose run --rm train
  else
    # Keep custom experiments in the mounted results folder as well.
    docker compose run --rm train "$@" --output /data/results
  fi
  ;;
view)
  shift
  if [ "$#" -gt 1 ]; then
    usage
    exit 2
  fi
  export WTF_PORT="${1:-${WTF_PORT:-8000}}"
  mkdir -p results
  docker compose up viewer
  ;;
down)
  shift
  docker compose down "$@"
  ;;
*)
  usage
  exit 2
  ;;
esac
