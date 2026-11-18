# Second native-kernel rollout

[Native setup](native-kernels.md) · [Performance](performance.md) · [Rollout 0.0.1.0](releases/0.0.1.0.md)

The second rollout reduces work remaining after the first native kernels. It
compares against **commit `baa5101` with its existing C extension enabled**,
not against the older Python-only or pre-float32 implementations. Configurations,
precision, observations, action cadence, diagnostics, learning schedules and
recording compression are unchanged.

## Implemented paths

| Path | Change and ownership |
| --- | --- |
| Dense inference | One C call/workspace for all layers; selected critic/action heads still skip unused final outputs. No persistent per-animal scratch cache. |
| Plain PPO minibatches | Forward passes, PPO objectives and gradient accumulation run in one C call with workspace reused across the batch. Python keeps sampling/shuffling, KL-stop decisions and Adam cadence. Custom objective/network subclasses retain their existing paths. |
| Policy probabilities/loss | Ordered double softmax/log probabilities and the PPO objective have native helpers. A2C/REINFORCE/GRPO keep their custom objective logic with shared native probabilities/inference. |
| Recurrent PPO | All GRU gates, hidden updates and BPTT matrix loops run in C, retaining chronological truncation and the original tape shape. |
| QR-DQN | Pairwise quantile Huber loss/derivatives are compiled with original reduction order. |
| Perception | Native ordered candidate search, wall occlusion and water-ray counts use two bounded static byte maps per world. Existing per-observer dynamic caches and bucket revisions remain authoritative. |
| Physics | Resource growth and flat-world cost/age/starvation phases use C loops with Python numeric dispatch, preserving original int/float representation and min/max ties. Terrain-specific body physics stays Python. |
| Replay | Native varints, integer-array handling, type/signed-zero comparisons and detached primitive-tree cloning preserve existing bytes/tags, aliases within copies, chunk bounds and public readers. Unsupported tree types use deepcopy. |
| Dashboard | Native sparse bucket scans/union-find return identical spatial-group counts. |
| GRPO trials | Reuse nonfocal RNG objects and restore full states for each trial; avoid repeatedly seeding generators that are immediately overwritten. Focal seeds/draw order are unchanged. |

## Numerical and compatibility boundaries

The implementation uses the existing C extension with three included source
headers: `ecosystem/_kernels_math.h`, `_kernels_spatial.h`, and `_kernels_data.h`.
There is no C++/ML-framework dependency or change to model/checkpoint/replay formats.
All arithmetic/storage boundaries remain the same as the first native rollout.

CPython 3.9's compensated `math.hypot` can differ by one last bit from C libm's
`hypot`. Native nearest-target search therefore returns integer offsets and
Python performs the original final distance/normalization. Candidate ordering,
wall/range checks, moved/dead/newborn-target invalidation and changed-food
invalidation remain intact. Custom visibility callbacks use the reference path.
Static map masks are rebuilt on obstacle/visibility changes and never checkpointed.

Native API version 2 gates new entry points. An older extension continues using
its compatible first-generation kernels and falls back for new operations until
rebuilt. `WTF_KERNELS=python` retains the full Python path. Builds track included
headers and publish the finished extension by atomic rename so existing processes
retain their mapped library; restart training or the playback server to use the
new backend. No JavaScript refresh/protocol migration is required by this rollout.
The macOS builder can use installed compiler/SDK paths directly and respects
explicit `CC`, `SDKROOT`, and `DEVELOPER_DIR` selections.

## Matched simulation benchmarks

Measured on 2026-11-18, ARM64 Mac, macOS 26.7.3, CPython 3.9.6. Both versions
force `WTF_KERNELS=native`. Each row uses its unchanged preset with seed 42,
48 prey + 12 predators, one 180-step episode, and medians of three alternating
separate-process runs. Initialization and verification digests are outside the
reported loop time. Recording rows include ordinary binary/XZ writing. These
are local workload measurements, not universal speed guarantees.

| Preset | Previous native | New native | Additional speedup |
| --- | ---: | ---: | ---: |
| DQN | 0.258 s | 0.235 s | 1.10× |
| PPO | 0.679 s | 0.306 s | 2.22× |
| A2C | 0.383 s | 0.321 s | 1.19× |
| REINFORCE | 0.365 s | 0.298 s | 1.23× |
| GRPO + DQN preset | 0.578 s | 0.463 s | 1.25× |
| Recurrent PPO | 26.263 s | 1.230 s | 21.36× |
| QR-DQN | 4.472 s | 1.003 s | 4.46× |
| Maskable PPO | 1.007 s | 0.774 s | 1.30× |
| MAPPO | 1.246 s | 1.042 s | 1.19× |
| Larger predator PPO | 0.860 s | 0.485 s | 1.77× |
| Valley + recording | 1.206 s | 1.031 s | 1.17× |
| Flat PPO + recording | 1.253 s | 0.825 s | 1.52× |

All **72 simulation samples** have identical complete-state hashes within each
case, including private RNG, weights, optimizer/replay/rollout state, champions,
world and metrics. Both 180-step recording cases also have identical SHA-256 of
the actual compressed replay chunks, not only equivalent summaries. The GRPO
preset pairs DQN prey with GRPO predators; the valley pairs DQN prey with PPO
predators. These compare preset throughput rather than isolated algorithm math.

An additional **1,500-step PPO recording** comparison (one sample per version,
validation rather than a three-run timing claim) has identical final checkpoints
and identical compressed replay bytes across all 1,501 frames. Its timings are
retained in `.cache/native-v2/long-validation.json`.

Peak RSS does not consistently decrease: recurrent PPO measured approximately
31.9 → 35.2 MiB, while QR-DQN measured 34.7 → 33.7 MiB. High-water RSS varies
between processes; no general RAM reduction is claimed. Workspaces are local to
calls/updates and the added sight masks use two bytes per grid cell.

## Recording, analysis and dashboard benchmarks

These separate tests use the same saved 65-frame PPO fixture on both versions.
The group test uses 500 synthetic positions and is a scaling test, not a claim
that the default simulation contains 500 animals. Each timing is the median of
three separate processes and covers the loop count shown. Every encoded-byte or
expanded-result hash matches.

| Workload | Repeats per timed sample | Previous native | New native | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Binary encoding | 20 | 1.061 s | 0.874 s | 1.21× |
| Binary decoding | 20 | 1.989 s | 1.936 s | 1.03× |
| Replay frame expansion | 10 | 0.893 s | 0.263 s | 3.39× |
| Dashboard summaries | 3 | 0.046 s | 0.024 s | 1.90× |
| 500-position group metrics | 100 | 0.636 s | 0.032 s | 19.83× |

Raw binary decoding is essentially unchanged in this fixture (about 1.03×);
its object construction and remaining parser work are still Python. The larger
replay-read improvement is frame expansion, where recursive deepcopy work was
removed. Compression algorithms/levels and disk formats were not changed.

## Reproduce and verify

The ignored baseline copy, including its original native library and source
hashes, is `.cache/native-v2/baseline/`. Reports are
`.cache/native-v2/combined-benchmark.json`, `data-benchmark.json`, and
`long-validation.json`. The baseline is local test data, not a tracked duplicate
source tree. For a fresh checkout, supply an equivalent previous source/library
tree and include its `maps/` assets.

```sh
.venv/bin/python -m work.build_native
.venv/bin/python -m work.performance.benchmark_kernels \
  --steps 180 --repeats 3 --backends native \
  --reference .cache/native-v2/baseline --reference-backend native \
  --output .cache/native-v2/recheck.json
.venv/bin/python -m work.performance.benchmark_data_kernels \
  --reference .cache/native-v2/baseline
PYTHONDONTWRITEBYTECODE=1 WTF_KERNELS=native .venv/bin/python -m unittest discover
PYTHONDONTWRITEBYTECODE=1 WTF_KERNELS=python .venv/bin/python -m unittest discover
node --test tests/*.test.mjs
```

Final checks: **393 native-mode tests passed; the Python fallback suite passed
385 tests with eight v2-only tests skipped; all 50 JS tests passed**. Added tests
cover fused batch metrics/gradients, GRU tapes/BPTT, quantile accumulation, sparse
groups, exact wire bytes/types/signed zero, detached alias-preserving copies,
body/resource numeric types and starvation events, malformed inputs and stale
extension fallback. Existing suites cover observations/caches, terrain physics,
paired births/mutations, resumes, archives/chaining, playback seeking, animations,
chart data and chunk boundaries. No historical result was converted or retuned.
