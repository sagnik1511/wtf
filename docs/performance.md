# Performance and benchmarking

[Project overview](../README.md) · [Architecture](architecture.md) · [Development](development.md)

Commands in this guide run from the project root.

Champion exports now use lossless binary/XZ arrays and a content-addressed shared
weight directory. A measured deep-river archive fell from 8.66 MB to 0.62 MB
(92.8% smaller); a subsequent unchanged archive wrote zero new weight bytes.
See [measurement and compatibility](binary-storage.md#champion-storage-measurement).
An export-local identity cache reuses serialization references for shared,
immutable snapshots; it is bounded by distinct champion models in that export
and discarded afterward. It never caches live controller parameters or enters
checkpoints. No training throughput/RAM improvement is claimed for this change.

## Recent export and inspector work

Champion export now serializes and hashes each shared configuration object once
per export, alongside the existing immutable-model reference reuse. The cache key
is object identity; it keeps strong references until export returns, is bounded
by the distinct configurations in that generation and is then discarded. A later
export always reads current values. Different historical training configurations
remain separate, and equal serialized configurations still share the same file.
JSON bytes, model blobs, artifact identities and compatibility are unchanged.

The inspector builds its recent-prediction chart only when its disclosure opens.
It processes each sample once when building that chart and reuses the chart when
closed/reopened until the diagnostic sample changes. Closed panels retain only
the existing bounded score history, without allocating SVG dots for it. Collecting
samples validates finite scores without constructing full bar-layout objects.
Current radar and action bars, expanded-state persistence, the 64-score-vector
limit, chunk loading and backward-seek filtering are unchanged.

Operation-count tests cover one configuration serialization per identity, exact
configuration/model round trips and fresh values after a later export. Viewer
checks cover deferred rendering, one score pass, reopen reuse, mixed score types
and signed/zero scales. No profiling or throughput/peak-RAM benchmark was run,
so these are removed operations rather than measured speedup percentages.

A remaining training candidate is `SpatialIndex.nearest_many(k>1)`: candidate
filtering/sorting runs in Python rather than the fused native nearest-target
selector used for k=1. Visibility rays can already use native code. A future
top-k kernel needs exact distance/insertion ties, wall/water filtering and cache
invalidation parity; it is not part of this optimization.

## Earlier optimizations and measurements

Lineage accounting now allocates a founder-family dictionary only on first use.
Champion comparisons request rank tuples without diagnostic dictionaries; full,
detached family metadata is generated only for final lineage winners. No new
cache or checkpoint state is introduced. In a 60-animal accounting/ranking probe
(500 passes, seven alternating samples, baseline `65345bf`), median time fell
from 0.15828 s to 0.09859 s (37.7%). This measures that path, not end-to-end
training speed or peak RAM. Forty-step seed-42 checkpoint comparisons matched
across all nine algorithms and the mixed meandering-river preset. See
[rollout 0.0.1.5](releases/0.0.1.5.md) for scope and verification.

The latest [second native-kernel report](native-kernels-v2.md) compares against
commit `baa5101` **with the existing native backend enabled**. Median additional
180-step speedups include PPO 2.22×, recurrent PPO 21.36× and QR-DQN 4.46×.
Replay expansion is 3.39× and dashboard summarization 1.90× faster on their saved
fixtures. All matched checkpoints and recorded chunk bytes agree. Raw binary
decoding is essentially unchanged (1.03×); peak RAM is not consistently lower.
The report includes the complete tables, workload, memory tradeoffs and checks.

For implemented CPU acceleration, build/backend controls and numerical boundaries,
see [optional CPU kernels](native-kernels.md). The preceding
[native performance review](native-performance-review.md) records the original
assessment; its remaining GRU/quantile/visibility proposals are not implemented.

New experiments now keep network parameters, target/reference networks and Adam
moments/variances in packed **float32** buffers across all nine algorithms.
Python arithmetic, gradient accumulation, activations, losses, returns and world
physics retain float64 precision. That storage change reduces retained model
memory. The optional C kernels independently accelerate numerical loops. Parameter rounding
can change learned policies and ecosystem trajectories, even with the same seed.

Checkpoint precision is explicit. Older checkpoints without it keep float64
models, including later births and episode initialization; warm-starting their
champions into a new experiment converts those weights to float32. Exact resume
is distinct from a warm start. See [precision handling](running.md#numeric-precision)
and [rollout 42](releases/0.0.0.42.md).

The [native-kernel benchmark results](native-kernels.md#benchmark-results) compare
three-process medians against the preceding FP32 code on 11 matched 180-step
presets. Complete-state hashes match. Those first-rollout training-loop speedups range from
1.53× (recurrent PPO) to 12.97× (MAPPO); the valley with binary/XZ recording gains
2.41×. Peak RSS is not consistently lower; QR-DQN rose from 26.3 to 30.6 MiB in
these samples. See the linked report for settings, all timings, limitations and
build instructions. Native kernels are optional and do not change replay formats.

Historical measurements describe the implementation evaluated at each optimization stage. Rerun the probe for the current revision and your workload.

The engine shares a per-observation-phase list of positions and resources instead of rebuilding it for each animal. An incremental spatial index groups food and animals into nearby cells, so limited-vision agents skip distant candidates. Small target sets and wide-vision queries use a full scan. Squared-distance filtering preserves visibility and insertion-order ties, including across cells. Each animal keeps a spatial window: movement reuses its overlapping bucket references and updates only entering/leaving strips. Shared buckets receive target moves, births, deaths and food-presence changes once per phase. Stationary observers reuse nearest-target results while the relevant bucket versions stay unchanged. Movement and obstacle edits force visibility to be checked again because occlusion inside the overlap can change. Energy, health, heading and blocked neighbors are always current. Derived caches are rebuilt after checkpoint restoration and released for dead observers; authoritative state is reconciled each phase to detect manual edits too. The Python fallback uses native multiplication iterators with the original summation order. Training computes Q-network weight offsets once per neuron, and MLP backpropagation avoids checking the input-layer case for every weight. Optimizer mathematics and accumulation order remain unchanged. Replay experiences use compact immutable tuples. Snapshot creation copies primitive entity fields without recursive dataclass conversion.

Recording streams one encoded frame and retains one previous frame for computing chunk-local deltas. CLI checkpoint files are written atomically in compact JSON with one copied brain at a time. This avoids simultaneously holding every copied replay buffer and a large formatted JSON string. Checkpoint values remain exact; the CLI now writes compact version-2 envelopes while the loader still accepts existing version-1 files. Scheduled intermediate checkpoints are preserved; the final checkpoint is written once. Population sizes, model dimensions, training cadence, recording frequency and diagnostics remain unchanged by these optimizations.

Playback reuses the loaded snapshot when rendering and avoids clearing/reallocating the canvas backing store or writing identical layout dimensions every frame. Chunk loading, chart history, selection and death effects remain available.

Earlier optimization results (median of three separate-process measurements, 180 steps, 48 prey + 12 predators, Python 3.9.6):

| Measurement | Before | After |
| --- | ---: | ---: |
| Simulation | 3.09 s | 1.69 s |
| Simulation + recording | 3.67 s | 1.80 s |
| Checkpoint writing | 1.04 s | 0.15 s |
| Peak RSS, including checkpoint writing | 114.53 MiB | 29.25 MiB |

Every before/after run produced the same complete-state digest. Ordinary simulation RAM changed only slightly; the large memory reduction is specifically in checkpoint writing.

A subsequent pass with reproduction enabled (400 steps, seed 42, median of three
separate-process runs) reduced baseline simulation time from **4.62 s to 3.71 s**
and A2C simulation plus recording from **5.48 s to 4.71 s**. Complete-state
hashes matched in every comparison; A2C recording output sizes also matched.
Peak RSS increased from **38.51 to 39.46 MiB** for baseline and **23.69 to
26.11 MiB** for A2C. This pass improves runtime with a small memory tradeoff;
that earlier pass rebuilt temporary spatial lookups each observation phase.
No dependencies, model sizes, learning schedules or ecosystem settings changed.

Incremental-window reuse then reduced baseline time from **3.71 s to 3.33 s**
(another 10%) and A2C with recording from **4.74 s to 4.53 s** (another 4%),
using the same 400-step, seed-42, three-run comparison. Complete-state hashes
and recording output sizes matched. Peak RSS was **37.16 to 36.36 MiB** for
baseline and **26.31 to 27.13 MiB** for A2C; these measurements show no clear
memory improvement. Windows and nearest-result caches are bounded by the living
observers and spatial cells, rather than accumulating one entry per step.


A local before/after benchmark and repeated measurements are saved under `work/performance/`. Run the optimized probe with:

```sh
.venv/bin/python -m work.performance.benchmark --case record --steps 180 --output work/performance/local-record.json
.venv/bin/python -m work.performance.benchmark --environment environments/a2c.json --case record --steps 400 --output work/performance/local-a2c.json
.venv/bin/python -m work.performance.benchmark --case checkpoint --steps 180 --output work/performance/local-checkpoint.json
```

The benchmark fixes seed 42 with the baseline 48 prey/12 predators and checks a digest of the complete world, brain, replay and random-generator state. It measures process peak RSS before constructing the comparison digest; checkpoint-case peak RSS includes both simulation and checkpoint saving. These are local measurements for a small workload, not guarantees for every environment. Large populations still incur independent-brain and perception costs, and the growing collection of completed-run summaries remains in memory for experiment reporting.


Replay chunks use a complete keyframe followed by sparse field changes within
each chunk, then lossless gzip (default level 6). On the same 401-frame PPO data,
full snapshots used 9,276,835 bytes, gzip snapshots 851,601 bytes, and delta plus
gzip 722,700 bytes: another 15.1% reduction over gzip alone. Entity identity and
unchanged food cells are no longer repeated every frame. Every chunk decodes
independently, preserving random seeking and full recorded precision. The native
viewer receives compressed stored bytes; legacy API clients receive full frames.
Decoded viewer memory remains one chunk and live brain memory is unaffected.
See the [recording guide](recording.md#optimize-older-recordings) for format,
measurement limits and conversion commands.

Checkpoint storage now pools repeated configuration/actions/model metadata and
encodes large float64/uint32 arrays as portable little-endian binary values in
base64 before gzip. A measured PPO checkpoint fell from 7.03 MB plain JSON to
2.56 MB, versus 3.19 MB with gzip alone. Writing took 0.24 seconds in that one
local sample; this is not a general throughput guarantee. The writer still
streams one detached brain at a time. Restoring and offline conversion decode
one complete checkpoint in RAM; no peak-memory reduction is claimed for loading.
See [checkpoint storage](running.md#optimize-existing-checkpoint-files).

A subsequent code-review pass reduces training-loop work without changing
learning settings or arithmetic order:

- DQN replay loss computes only the chosen action's current Q value; policy and
  bootstrap inference still compute all action values. Action indices are
  precomputed, and target refreshes update existing arrays in place.
- Modern experiments release dead founders as well as descendants after terminal
  learning and champion capture. Their update counts are retained separately.
  Old modern checkpoints skip rebuilding already-dead controllers when restored.
  Legacy checkpoints that reuse founder controllers retain their prior behavior.
- DQN-only steps avoid the per-animal GRPO scan. Mode checks are outside the
  learning loop; balanced champion settings are constructed once per predator
  species per candidate pass rather than once per animal.

Validation compares DQN against the former full-output replay update through
buffer wraparound, target refreshes, terminal/truncated transitions and resume.
Lifecycle checks verify buffer release, identical living controller/world states,
champions, update totals and future chained runs. These changes were reviewed and
checked for correctness without profiling or timing benchmarks; no runtime or
peak-RAM percentage improvement is claimed. Living agents still own independent
networks and experience buffers, and completed episode summaries still accumulate.

The shared policy-training paths also avoid unnecessary work:

| Algorithms | Code-review optimization |
| --- | --- |
| PPO, A2C, REINFORCE, GRPO | Inference returns outputs without retaining a backpropagation activation tape. Full activations are retained only for actual gradient calculation. |
| PPO, A2C | Bootstrapping computes the critic output without evaluating unused policy output heads. |
| PPO, GRPO | Minibatches reuse gradient containers within one update call, resetting values before each minibatch. No scratch arrays are kept on every living brain. |
| PPO, A2C, REINFORCE | Fresh advantage vectors normalize in place. REINFORCE reuses its return vector after calculating the mean return. |
| GRPO | Shared trajectory advantages/weights are calculated once per trajectory. Zero-KL settings skip reference inference. Trial clones omit historical lifespan records while preserving the authoritative next-animal ID. |
| All checkpoint restores | Recorded next-animal IDs avoid rescanning death history; legacy files still reconstruct the bound. |

These preserve rewards, update schedules, sampling order, optimizer arithmetic,
rollout sizes and diagnostic contents. Correctness checks include exact serialized
state comparisons with the original implementations over 240 policy transitions,
including normalization variants, PPO/GRPO KL stopping, unequal trial lengths and
checkpoint resumes. Regression tests additionally compare inference with full
forward passes, reused/fresh gradients, and trial histories with/without old death
records. No profiler or timing/memory benchmark was used in this review, so these
are reductions in work and allocation rather than measured speedup claims.


## Additional algorithm review

The GRU, distributional and centralized-critic paths reduce repeated computation
without adding persistent per-animal caches:

| Path | Reduction in work or temporary allocations |
| --- | --- |
| Recurrent PPO backpropagation | The output MLP returns its input derivative while accumulating parameter gradients, replacing a separate second traversal through the same head. |
| Recurrent PPO bootstrapping | Evaluates only the scalar value head after the GRU/hidden layers; unused policy outputs are skipped. Gate dot products use native multiplication iterators with the same summation order. |
| QR-DQN diagnostics | One network prediction supplies both mean action values and return quantiles. |
| QR-DQN targets | Computes only the chosen action's target quantiles at the final layer. Online action selection still evaluates all actions. |
| MAPPO training | Reuses the post-physics global context for bootstrapping and the next decision. A pure MAPPO phase does not build an unused mask-occupancy map. |
| MAPPO greedy evaluation | Uses only the local actor; the critic remains available for recorded diagnostics. |
| PPO variant construction/restore | Constructs the actual network once. GRU and MAPPO also avoid discarded default optimizer arrays; Maskable PPO avoids its discarded default MLP. |
| GRPO with masked opponents | Shares one occupancy map per trial step across masked agents and evaluates frozen policies without copying controller objects. |

Correctness verification compared exact serialized world, brain, replay/rollout,
optimizer, RNG and champion state with revision `0ae8df0`: 207 checkpoint
comparisons across all nine algorithms and 23 further comparisons for mixed
GRPO/Maskable PPO. The workload uses two 11-step episodes per case, small
networks, reproduction windows, chained champions, a forced death in the
nine-algorithm comparison and checkpoint restoration in the mixed case. The
23rd comparison in each case also checks the already-completed lifecycle.

Regression tests check numerical input derivatives, selected-head equivalence,
forward-pass and context-build counts, direct network construction, value-only
GRU bootstrap and shared trial occupancy. Learning schedules, rewards, network
sizes, buffers, random draws and checkpoint formats are unchanged. No profiler,
time benchmark or peak-memory measurement was run, so no percentage speedup or
RAM reduction is claimed. The main dense arithmetic remains standard-library
Python and independent-agent costs still grow with population and model size.


## Binary storage and bounded replay caches

New presets store typed binary replay chunks with XZ compression. The records
retain full float64 precision and wrap the existing field deltas; homogeneous
entity/event records become columns, numeric arrays are raw bytes and repeated
strings use file-local dictionary entries. Binary version-3 checkpoints replace
the previous base64/JSON wrapper while retaining shared metadata/models and
one-controller-at-a-time writing. Readable metrics and champion archives retain
their existing formats. See [binary storage](binary-storage.md).

A verified conversion of the completed 20-episode experiment
`a2c_20261117T044059Z_e2b39a` measured these actual file sizes (decimal MB):

| Data | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| 480 replay chunks | 69.98 MB | 44.84 MB | 35.9% |
| 20 exact checkpoints | 32.68 MB | 29.43 MB | 9.9% |
| Whole experiment, including unchanged readable files | 106.18 MB | 77.80 MB | 26.7% |

All 500 replaced files passed expanded-frame/state verification, with smaller
replacements only. Actual converted first/last replay chunks also matched the
JavaScript decoder, and the converted final checkpoint restored all 20 completed
episodes. The measurement is disk size, not a timing or peak-memory benchmark.
LZMA2 uses a 1 MiB dictionary but has additional encoder memory and CPU costs.
Use gzip for a faster compression option; no throughput improvement is claimed.

Recording avoids sorting/copying static obstacles and dimensions into every
frame before discarding them. The playback server reads metadata once per
telemetry request and caches at most four derived 128-frame pages, protected by
a lock and invalidated using manifest/chunk stat fingerprints. Cached results
are detached from callers. No simulation-state or learned-prediction cache is
introduced, and no complete world-chunk cache is retained on the server.



## Observation cache dependencies

Dynamic targets and static geometry have different invalidation rules. An observer staying still is not enough to reuse a target result.

![Observation cache dependencies](assets/diagrams/observation-cache-dependencies.svg)

## Simulation caches and champion selection

The simulation uses simple in-memory dictionaries rather than a broad memoization
layer. An obstacle-only visibility LRU (`OrderedDict`) retains at most 2,048
ordered start/end ray results per world. Observation setup already reconciles
obstacle edits; changed obstacles or visibility implementations clear this cache
and invalidate nearest-target results. Changing animal positions or food does
not invalidate a geometric ray, but target selection still uses the current
spatial-index revisions. Direct `World.visible` calls remain uncached and current.
Restored worlds rebuild the cache; it is absent from checkpoints and replay data.
There is no shared global cache holding retired worlds or model controllers.

Champion selection resolves all final winning ranks before copying model
parameters. A transient dictionary copies each actual winning animal once per
consideration pass, sharing that detached snapshot across its winning criteria.
Temporary record breakers no longer trigger discarded copies. Snapshots are not
reused between steps because weights can change independently of observed rank
changes or optimizer counters.

Population metrics count animals in one pass instead of one scan per configured
species. Reproduction resolves species-wide timing-window flags once per phase
instead of once per animal and successful parent. Both remain derived from current
state on each call, so manual edits and lifecycle changes are visible immediately.

Verification passed 279 exact serialized-state comparisons with the previous
simulation/champion paths across all nine algorithms: two 15-step episodes per
case plus a completed-lifecycle check, with reproduction, forced mortality,
obstacle edits, checkpoint restoration and chained champions. Regression tests
cover cache hits/eviction, geometry invalidation, transient final-winner copies,
former ranking behavior and current population counts. The full suite passes
274 Python tests. No profiler or timing benchmark was run, so fewer calculations
and copies are verified without claiming a measured overall speedup or RAM drop.
