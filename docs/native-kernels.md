# Optional CPU kernels

[Performance](performance.md) · [Running](running.md) · [Development](development.md)

The optional CPython C extension accelerates dense-layer inference/backpropagation,
chosen-action DQN gradients, SGD and clipped Adam updates. These kernels remove
Python iteration over parameters and avoid allocating a slice of every weight
row. They do not change model sizes, training schedules, policy sampling or
per-animal ownership. Python remains responsible for environment logic, replay,
rollouts, rewards, reproduction, mutation and recording.

The [second rollout report](native-kernels-v2.md) covers the current GRU,
PPO minibatch, perception, world, replay and dashboard extensions, with benchmarks
against the preceding native setup. The benchmark table later in this guide is
the historical first rollout's comparison against FP32 Python code.

## Build and select

From the repository root, with a C compiler and Python development headers:

```sh
.venv/bin/python -m work.build_native
.venv/bin/python -c 'from wtf.native.kernels import status; print(status())'
```

The current build command targets Clang/GCC toolchains. It uses the existing
virtual environment's setuptools; if absent, install it into `.venv` first:

```sh
.venv/bin/python -m pip install --cache-dir .cache/pip setuptools
```

No NumPy, PyTorch or Cython package is needed. Compilation is explicit, never
triggered by importing a module or starting a run. Build files live in ignored
`.cache/native-build/`; the ignored extension binary lives beside the Python
modules. Rebuild after C-source, Python ABI, compiler or platform changes. A
fresh clone works with the Python fallback before building.

`WTF_KERNELS` chooses the process backend:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use the extension if available on a supported interpreter; otherwise use Python |
| `python` | Use the Python reference implementation without importing the extension |
| `native` | Require the extension; report an actionable error if unavailable |

```sh
# Require native kernels for an ordinary training invocation:
WTF_KERNELS=native .venv/bin/python -m wtf.cli environments/ppo.json --runs 1 --progress
# Reference implementation for comparison:
WTF_KERNELS=python .venv/bin/python -m wtf.cli environments/ppo.json --runs 1 --progress
```

Acceleration supports CPython 3.12–3.13. Native reductions corresponding to
Python's builtin `sum()` use compensated summation matching those interpreters;
ordinary explicit accumulation loops keep their original order. See the
[Python summation change](https://docs.python.org/3.12/library/functions.html#sum).
The extension retains float32 parameter storage and double arithmetic. Newer
interpreters intentionally use Python until native equivalence is validated;
`WTF_KERNELS=native` fails explicitly outside the validated range. Rebuild the
extension after changing Python minor version. Legacy list-backed float64 parameter operations retain Python math; shared policy
probability/objective helpers use equivalent double arithmetic when available.

## Coverage and precision

| Path | Compiled work |
| --- | --- |
| DQN | Hidden/output inference, chosen output, replay gradient accumulation and SGD |
| PPO, A2C, REINFORCE, GRPO, Maskable PPO | Shared MLP forward/backward and Adam |
| MAPPO | Independent actor/critic MLPs and Adam |
| QR-DQN | Network forward/backward, pairwise quantile loss and Adam |
| Recurrent PPO | GRU gates/BPTT, output MLP and whole-network Adam |

Parameters and optimizer state keep their float32 storage. C expressions use
double arithmetic, matching the reference's Python floats. Temporary parameter
gradients use float64 buffers on the native path; the Python fallback retains
its list-based scratch storage to avoid buffer-indexing overhead. Kernels preserve product accumulation order,
add bias after the reduction, round moments/variances before using them in Adam,
and disable fast-math/FMA contraction. They own no RNG or global scratch data.
The GIL remains held: these are single-process CPU-loop improvements, not
threading or GPU acceleration. Perception, flat-world phases, replay and dashboard helpers are covered by
the [second rollout](native-kernels-v2.md); terrain-specific physics stays Python.

Checkpoints/champion/replay formats are unchanged by the kernels. Backend
selection is a process option, not part of saved learned state. Native and Python
can resume each other's checkpoints on the tested interpreter/platform; strict
parity tests and benchmark complete-state hashes cover this. Cross-platform
libm/compiler differences are still possible; bitwise agreement on one machine
is not a universal floating-point guarantee.

## Checks and benchmarks

```sh
PYTHONDONTWRITEBYTECODE=1 WTF_KERNELS=native .venv/bin/python -m unittest discover
PYTHONDONTWRITEBYTECODE=1 WTF_KERNELS=python .venv/bin/python -m unittest discover
.venv/bin/python -m work.performance.benchmark_kernels --steps 180 --repeats 3
```

The benchmark alternates separate Python/native processes for each preset and
checks SHA-256 of complete final checkpoints, including RNGs, buffers, champions,
world state and metrics. It keeps full preset founder populations, fixes seed
42, uses one episode and reports median wall time. Initialization is timed
separately; peak RSS is sampled before constructing the semantic digest. The
terrain-recording case includes binary/XZ output. No profiler, automatic library
publication or persistent experiment result directory is used.

Raw reports default to ignored `.cache/native-build/benchmark.json`. Use
`--cases ppo dqn` to narrow the workload, or `--reference /path/to/source-tree`
to include an earlier implementation containing `wtf/`. Source snapshots,
raw samples and compiler outputs stay ignored. The measured rollout report is
linked from [performance](performance.md).

## Benchmark results

Measured on 2026-11-18: an ARM64 Mac, macOS 26.5.2, CPython 3.9.6.
These compare the **preceding FP32 implementation** against compiled kernels;
the speedups do not include the earlier float32 storage change. Each case uses
seed 42, 48 prey + 12 predators, one 180-step episode and the original preset
architecture/learning settings. Values are medians of three separate processes
per implementation. Original/native processes were interleaved with auxiliary
Python-fallback runs; no benchmark workers ran concurrently. The terrain case
was repeated separately after adding map assets to the baseline snapshot.
Initialization and semantic-digest construction are excluded from these times.
Terrain recording includes normal binary/XZ encoding and compression.

| Preset | Original FP32 | C kernels | Speedup |
| --- | ---: | ---: | ---: |
| DQN | 0.874 s | 0.252 s | 3.46× |
| PPO | 3.515 s | 0.664 s | 5.30× |
| A2C | 1.288 s | 0.392 s | 3.29× |
| REINFORCE | 1.040 s | 0.352 s | 2.96× |
| GRPO + DQN preset | 1.201 s | 0.568 s | 2.11× |
| Recurrent PPO | 38.673 s | 25.330 s | 1.53× |
| QR-DQN | 28.548 s | 4.539 s | 6.29× |
| Maskable PPO | 9.154 s | 0.966 s | 9.48× |
| MAPPO | 16.028 s | 1.236 s | 12.97× |
| Larger predator PPO | 9.087 s | 0.831 s | 10.94× |
| Enclosed valley + recording | 2.924 s | 1.215 s | 2.41× |

All **66 original/native samples** have the same complete-state hash within
each case, including optimizer/RNG state and replay/rollout buffers. The GRPO
preset uses DQN prey and GRPO predators; the valley preset uses DQN prey and
PPO predators. Algorithm ratios describe their full preset simulations, not
isolated mathematical operations. Short throughput tests do not establish
better learning or predict every longer experiment. Recurrent PPO gains less
because its GRU cell loops are still Python.

Peak RSS was broadly similar and did not improve consistently: QR-DQN rose
from **26.3 to 30.6 MiB**, while recurrent PPO fell from 32.2 to 29.9 MiB in
these median samples. These are whole-process high-water measurements, not
live model sizes or evidence of a general memory reduction. Raw samples
include timing ranges and per-process RSS.

After retaining scratch lists on the disabled-kernel path, three-run fallback
checks measured DQN at 0.881 s and PPO at 3.566 s, versus original medians
of 0.874 s and 3.515 s. These small local differences are not a fallback
speedup claim. Numerical equivalence remains required in either mode.

The completed comparison is saved locally in
`.cache/native-build/benchmark.json`; the original staged-source snapshot and
its file hashes are under `.cache/native-build/fp32-reference/`. These ignored
artifacts are measurements rather than tracked experiment data. Reproduce a
comparison to that snapshot with:

```sh
.venv/bin/python -m work.performance.benchmark_kernels \
  --steps 180 --repeats 3 --backends native \
  --reference .cache/native-build/fp32-reference
```

Without the optional historical snapshot, use the default Python/native
comparison command above. The original timing batch also retained auxiliary
fallback samples; those are not used in the published original/native table.

Final validation: **385 Python tests passed with `WTF_KERNELS=native` and
385 with `WTF_KERNELS=python`; 50 JavaScript viewer tests passed**. Tests
also compare decoded native/Python binary recordings across chunk boundaries.
Documentation links and source/export diagram checks pass. The rollout is
[0.0.0.43](releases/0.0.0.43.md), committed in `baa5101`.
