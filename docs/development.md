# Development and verification

[Project overview](../README.md) · [Architecture](architecture.md) · [Performance](performance.md)

Commands in this guide run from the project root.

For a new coding session, start with [AGENTS.md](../AGENTS.md) and the maintained
[project context](project-context.md). Repo-local workflows live under
[`.agents/skills/`](../.agents/skills/); their entrypoints are linked in AGENTS.md.
Read the matching skill for saved-run analysis, optimization or release work.
These files are tracked so future sessions can recover project decisions without
relying on chat history. They add no runtime dependencies.

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

## Release records

Keep [the rollout hub](releases/README.md) and its versioned child documents
current when preparing implementation changes. Use the [release template](releases/TEMPLATE.md),
record actual commit hashes only after they exist, and distinguish Pending work
from Committed history. Include behavior, compatibility/migration, validation
results and rollback considerations. Documentation-only hash reconciliation stays
attached to the rollout it records. Do not change the Python package version or
create Git tags merely to update this documentation index.

## Advanced algorithm checks

`tests/test_advanced_algorithms.py` checks GRU gradients through two timesteps,
quantile Huber and masked-policy derivatives, actor/critic information separation,
current/next critic contexts, terminal/truncated bootstrap, newborn memory,
mask preservation, bounded replay, mixed policies, GRPO trial isolation,
recording transparency, frozen evaluation weights, exact JSON/compact resume,
and CLI progress/archive/warm-start workflows. See the
[advanced algorithms guide](advanced-algorithms.md) for the experiment semantics.

Reproduction-window tests additionally cover inclusive bounds, per-animal spread,
positive-step clipping, one draw per clock, all delayed-birth gates, cooldowns
measured from actual births, legacy zero-deviation RNG behavior and pending-clock
preservation across JSON/compact checkpoints and delta replays.

`tests/test_advanced_efficiency.py` guards against duplicate GRU head passes,
unused bootstrap/target outputs, repeated QR-DQN diagnostic inference, duplicate
MAPPO contexts, critic inference during greedy MAPPO execution, discarded default
networks and per-agent trial occupancy/controller copies. These checks count
operations and validate numerical results, without timing thresholds or profiling.

`tests/test_binary_storage.py` verifies typed-column Python/JavaScript parity,
signed zero and large seeds, corruption rejection, playback seeking, binary
checkpoint continuation for all algorithms, recording transparency, atomic
verified conversion, legacy HTTP fallback and bounded/invalidation-safe telemetry
caching. [Binary storage](binary-storage.md) documents formats and migration.

`tests/test_simulation_caches.py` verifies the bounded per-world visibility LRU,
obstacle/implementation invalidation, checkpoint exclusion, current target
selection, reproduction timing flag reuse, one-pass populations and final-only
champion model copies against the former ranking path. Cache checks use call
counts and exact state comparisons rather than wall-clock thresholds.

`tests/test_mutation.py` covers sparse matrix-only changes for all algorithms,
parent/RNG independence, zero/disabled behavior, optimizer/target/reference
resets, update accounting, lineage inheritance, frozen evaluation, GRPO trial
isolation, readable mutation metrics, champion provenance and exact JSON/binary
resume. Forced events also pass the JavaScript binary replay decoder.
