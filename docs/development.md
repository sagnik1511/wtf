# Development and verification

[Project overview](../README.md) · [Architecture](architecture.md) · [Performance](performance.md)

Commands in this guide run from the project root.

```sh
.venv/bin/python -m unittest discover -v
# Optional JS playback-engine checks (requires Node.js 20+):
node --test tests/*.test.mjs
```

Tests cover strict configuration validation, resource limits, blocked and contested movement, feeding rewards, simultaneous hunts, starvation, occlusion, independent weights, positive-feedback learning, terminal versus truncated experience, evaluation behavior, deterministic and randomized runs, exact JSON checkpoint continuation including future random seeds and mixed PPO/Q models, PPO finite-difference gradient checks and terminal/truncated rollouts, resets, streamed recording correctness, recording transparency, committed-chunk visibility, bounded playback chunk loading, seeking, and read-only HTTP endpoints.

## Local server

The server binds only to `127.0.0.1`. Use `--port 8001` if the default port is occupied. Close the terminal process or press Ctrl-C to stop it.

## Working conventions

Keep Python packages inside `.venv`. Simulation changes should be checked with
seeded runs and exact checkpoint continuation where relevant. Performance tests
compare observations and numerical operations against reference implementations;
timing thresholds are kept out of unit tests.

Generated `results/`, `brain_library/`, `.cache/`, bytecode and `.venv/` are
ignored. Benchmark output and local reference snapshots under `work/performance/`
are ignored; runnable benchmark scripts are retained. Champion publication is
disabled in CLI unit-test subprocesses with `--no-library`, while dedicated
library tests use temporary directories.

Tests also cover model compatibility, independent warm-started agents, ancestry,
pre-death model capture, missing-library checkpoint restores, analyzer coverage,
energy budgets and inherited learning-counter baselines.
