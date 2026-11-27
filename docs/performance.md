# Performance and benchmarking

[Project overview](../README.md) · [Architecture](architecture.md) · [Development](development.md)

Commands in this guide run from the project root.

Champion exports now use lossless binary/XZ arrays and a content-addressed shared
weight directory. Repeated unchanged models reuse the existing weight blobs
rather than storing another parameter copy.
See [measurement and compatibility](binary-storage.md#champion-storage-measurement).
An export-local identity cache reuses serialization references for shared,
immutable snapshots; it is bounded by distinct champion models in that export
and discarded afterward. It never caches live controller parameters or enters
checkpoints. No training throughput/RAM improvement is claimed for this change.

Preset defaults now use [V7 with history and scent](configuration.md#v7-sensory-preset-defaults).
Earlier benchmark tables retain their measured contracts, including legacy
153-/482-input workloads; those sizes do not describe the promoted presets.
Reproducing older timings requires the documented baseline revision/configuration,
while fresh comparisons must hold the full sensory settings equal.

## Native PPO episode engine

The optional [C++ core](native-engine.md) owns the complete CPU world/learning
loop, initialization and episode chaining for all 14 built-in learners. The
measurements here use ordinary PPO; other learners have differential correctness
coverage, without a performance guarantee inferred from these timings.

Local CPython 3.12.9/macOS ARM64 measurements compare the two engines with the
existing C numerical kernels enabled in the Python baseline. Three alternating
fresh-process samples per workflow use seed 100 and identical configuration,
private brains, recording cadence and saved formats. Timings include the
backend's initialization, bounded CLI transfers, metrics and final binary/XZ
checkpoint writing. Process startup and subsequent artifact verification are
outside timing. Champion-library publication is disabled; champion selection
and checkpointed champion records remain part of the workload.

| Workflow | Python | C++ | Speedup | Peak RSS Python / C++ |
| --- | ---: | ---: | ---: | ---: |
| Training, no playback | 15.952 s | 7.951 s | 2.01× | 71.8 / 159.9 MiB |
| Evaluation, no playback | 4.517 s | 1.391 s | 3.25× | 46.3 / 88.0 MiB |
| Training + playback | 21.377 s | 14.591 s | 1.47× | 82.9 / 163.9 MiB |
| Evaluation + playback | 6.655 s | 4.237 s | 1.57× | 57.7 / 99.0 MiB |

Values are per-workflow medians, including median process peak RSS. Training
completed 1,000 steps and 13,032 optimizer updates. Evaluation reached natural
extinction at step 816 under the same 1,000-step limit and performed no learning.
All 12 paired cases matched complete logical checkpoints, checkpoint file bytes,
metrics CSV, replay chunks and workload counts. Lifecycle tests separately
exercise forced births, rare mutation, deaths and restoration.

A second headless workload uses `land_bridge_collective`: elevated terrain,
day/night, movement history, scent, shared consequences and private PPO auxiliary
heads. Its three paired 256-step cases completed 3,320 optimizer updates and
matched all retained artifacts. Median time was **7.254 → 4.009 seconds
(1.81×)**; median peak RSS was **100.8 → 165.0 MiB**.
This is a separate workload, rather than a direct timing comparison with the
1,000-step table.

The native loop removes repeated per-animal Python dispatch, experience/input
packaging and boundary conversions around individual numerical operations.
Private typed model/optimizer buffers, reusable learner workspaces and shared
observation-phase views keep computation inside C++. See the
[ownership/data-flow diagram](native-engine.md#execution-and-ownership).

**This is a speed/RAM tradeoff.** Native state and detached Python artifact/query
projections coexist at boundaries; exporting complete ordinary-list checkpoints
also expands parameter representations temporarily. Packed internal FP32 buffers
reduce repeated model trees, but do not remove that peak. Recorded workflows
also spend time in the existing frame transport and encoding/compression layer.
The measured improvement does not establish lower memory use or a speedup for
every configuration. Linux and other interpreter results require their own
execution; their CI configuration is not local benchmark evidence.

Reproduce with:

```sh
.venv/bin/python -m work.build_native_engine
.venv/bin/python -m work.benchmark_native_engine \
  --steps 1000 --runs 1 --repeats 3 --seeds 100 \
  --profiles headless replay --modes train evaluate \
  --output .cache/native-engine/benchmark-full-core-1000.json
.venv/bin/python -m work.benchmark_native_engine \
  --presets land_bridge_collective --steps 256 --runs 1 --repeats 3 --seeds 100 \
  --profiles headless --modes train \
  --output .cache/native-engine/benchmark-full-core-rich-256.json
```

Raw reports stay in ignored caches. Each benchmark retains temporary artifacts
only long enough to verify them; no training campaign is added to `results/`.

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

The optional [native top-k selector](native-top-k.md) now handles candidate
collection, tuple ordering and visibility/range filtering for k > 1. Python still
owns invalidation and exact final feature arithmetic. Its matched measurements
compare against the reference selector with all other kernels already enabled.

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

Storage conversion is validated by comparing expanded replay frames and exact
checkpoint states before replacing a file. Only smaller verified replacements
are published. Python/JavaScript decoder parity, seeking and checkpoint restore
must remain intact. Disk reduction depends on entity counts, numeric ranges and
compression settings; it is not a fixed property of every recording. LZMA2 adds
encoder memory and CPU cost; gzip offers a faster compression option.

Recording avoids sorting/copying static obstacles and dimensions into every
frame before discarding them. The playback server reads metadata once per
telemetry request and caches at most four derived 128-frame pages, protected by
a lock and invalidated using manifest/chunk stat fingerprints. Cached results
are detached from callers. No simulation-state or learned-prediction cache is
introduced, and no complete world-chunk cache is retained on the server.



## Spatial window structure

![Overlapping observer windows reuse shared bucket objects while target revisions invalidate cached query results](assets/structures/spatial-window-cache.svg)

Each world has separate food/resource-eater/prey-eater indexes. Windows retain
bucket references, so a target change inside an overlapping bucket is visible
without copying the window contents. Revisions invalidate target selections;
current self/terrain/mating fields are assembled separately. Sparse/wide queries
can scan directly. The following dependency flow describes the invalidations.

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

## Decision-phase mating data and birth availability

Observation production builds one occupied-cell mapping and passes it into the
mating context. Population counts, sex/spatial buckets and personal readiness are
prepared once for that phase. Masked-PPO context refresh receives this data
explicitly instead of rebuilding the mapping/counts and repeating each mate's
health, energy and timing checks. GRPO's masked opponents reuse their own trial
world's phase data; real and hypothetical worlds never share it.

This is phase-local reuse, not a persistent LRU or a cache keyed only by position
or tick. The context is replaced during observation production and is bounded by
living animals. It is valid only while the observed physical state is unchanged.
Standalone mask/availability calls read current state; standalone runner refresh
prepares new data, so direct same-tick health, population or configuration edits
remain visible. Contexts are not serialized, and restore creates independent
ones. The post-movement/combat/cost birth phase rechecks actual eligibility.

Observation pair flags are not substituted for action-mask results. Mating
observations apply the configured observation range, whereas mask availability
keeps its existing radius/line-of-sight rules. Sharing personal readiness does
not merge these different contracts or predict another animal's consent.

Birth-cell predicates use a common ordered candidate iterator. Boolean
availability checks stop at the first valid cell without allocating a list or a
deduplication dictionary. Actual births still materialize the complete unique
sequence in the original parent/direction order, preserving seeded cell choice.
Terrain dryness, elevation, occupancy and boundaries use the same conditions.

[Regression tests](../tests/test_mating_context_efficiency.py) cover one readiness
preparation per masked phase, early exit, same-tick edits, RNG/state transparency,
water/cliffs, empty/full birth space, ordering, visibility differences and fresh
post-cost eligibility. Pre-change reference comparisons cover all supported
learners, consent versions, mixed GRPO/masked opponents, terrain, forced births
and mutation; checkpoint/observation values and recorded replay bytes match on
both automatic/native and Python-only paths. These checks establish removed
work and equivalence, not a measured throughput or peak-memory improvement.

## Founder initialization and shared reporting

Run initialization constructs one fresh controller per populated species/sex
rather than generating the same seeded network for every founder. A selected
warm-start model is validated and loaded once for that group. Remaining founders
clone that pristine template with their original individual seeds. Templates are
local to initialization and discarded before actions or learning; they never
hold a trained animal's optimizer or experiences. Each animal owns its parameters,
configuration, target/reference networks, optimizer, buffers, recurrent state and
independent action/training/GRPO RNGs. No controller state is pooled across animals,
sexes, species or runs. Existing legacy controller-retention behavior is preserved.

Completed-step metrics are produced after learning and birth mutation. When the
metrics and replay schedules coincide, CLI recording uses that existing report;
CSV, progress and final terminal output read it too. The final summary takes a
detached copy instead of rescanning the world. Initial CLI reporting similarly
shares one fresh report among CSV, recording and progress. Skipped replay samples
still return before computing metrics or building a frame; a replay-only step
computes its own current metrics.

This sharing uses explicit optional `metrics` arguments to `World.snapshot`,
`Recorder.begin_run`/`capture` and `RunProgress.start`, valid only while the reported
physical state is unchanged. There is no persistent reporting cache or tick-only
key. Standalone calls without a report calculate fresh values, including after
same-tick edits. Snapshots own copies of supplied metric maps; summaries remain
independent of history. Reporting templates and metrics add no checkpoint fields,
format changes, diagnostic samples or viewer requirements.

[Regression checks](../tests/test_founder_reporting_efficiency.py) count founder
construction/model loads, verify original seeded states and independent mutable
state across all nine learners, and check report scan counts, detached maps,
sampling and fresh standalone reads. Short pre-change reference comparisons
verify exact observations, checkpoints and replay bytes. Two-run workflow checks
also cover resume, champion chaining, disabled chaining and randomized worlds,
including identical CSV, summary, checkpoint and recording artifacts. These
checks demonstrate removed work without claiming a measured throughput gain.

### Measured initialization and reporting cost

A paired native-backend benchmark on macOS ARM64 with CPython 3.9.6 compares
0.0.1.24 against the saved pre-change implementation. Both versions include the
preceding mating-context optimization. The workload uses `maskable_ppo.json`,
48 prey/6 predators, seed 100, one fixed world and a 32-step limit; network sizes,
learning settings, metric/frame/diagnostic intervals and binary/XZ recording
settings retain the preset values. Each timed sample runs in a separate process,
after an untimed warmup, with alternating before/after order. Values below are
medians of five uninstrumented samples per version. Training/recording includes
initialization and chunk closure, but excludes checkpoint/champion exports,
post-run state hashing and process/import startup.

| Measurement | Before | After | Interpretation |
| --- | --- | --- | --- |
| Initialization | 179.7 ms | 71.1 ms | 60.5% less elapsed time; 2.53× speedup |
| Initialization plus 32 training/recording steps | 860.9 ms | 748.4 ms | 13.1% less elapsed time |
| Initialization peak traced Python allocation | 4.70 MiB | 4.69 MiB | Effectively unchanged |
| Training/recording peak traced Python allocation | 22.09 MiB | 22.07 MiB | Effectively unchanged |

Memory values are separate single-sample `tracemalloc` measurements; they exclude
untraced native/compressor allocations and are not total process RAM. Independent
per-animal weights, buffers and optimizer state remain necessary. The full-run
time reduction is approximately the initialization saving; these measurements
do not establish a sustained per-step throughput improvement or material memory
reduction. Do not extrapolate the short-run percentage to long episodes.

Both versions perform 432 optimizer updates and produce 53,297 bytes of playback
artifacts. Complete state and replay artifact hashes match for every timed pair.
Temporary recordings are deleted after each sample; measurements and the local
benchmark helper remain under ignored `.cache/`, outside `results/`.

## On-demand brain explanations

[Brain explanations](explainability.md) run only when the selected animal's
inspector window is opened, its controls change, or a new recorded observation
sample arrives while **Follow replay samples** is enabled. The archived-analysis
path adds no training callbacks, model snapshots or replay bytes. It never restores a world or a full
checkpoint to explain a champion. One stateless forward/reverse pass computes
activations and local input sensitivity, avoiding a forward pass per input.

The optimization review adopted these bounds:

- Decode only the selected published chunk, extract one animal's diagnostic and
  release the frame list. A two-entry context cache keys recording/run/frame/ID
  and chunk path/size/mtime/ctime. It never retains full chunks or later samples.
- Validate/prepare a champion once; retain packed double arrays to preserve
  legacy weights exactly. Cache at most two models with a combined one-million
  scalar-parameter limit, keyed by weight-blob stat identity and its reference,
  or artifact stat identity for inline models. Identical blobs are shared across
  criteria. Only weight loading/preparation is locked; arithmetic is read-only.
- The model cache limit bounds retained parameter payload, not total process RAM
  or concurrent response memory. Cache misses serialize model decompression;
  oversized/unsupported models fail with an explanation message. Layers are
  limited to 1,024 units and input vectors to 512 values.
- Reuse one manifest read per context request; avoid prefix-list allocation when
  finding the diagnostic. Cache ownership is playback-server-local and ephemeral.
- Send a bounded route graph rather than weight matrices: up to 12 selected
  hidden units per layer and three incoming visible links per displayed unit,
  plus layer activations and feature sensitivities. The browser retains one
  displayed response while open, skips identical samples and rejects stale
  responses. Following has one in-flight request and only the newest pending
  request; there is no unbounded queue or per-animation-frame inference. Errors
  stop automatic retries until a manual retry or group change. Closing/death/run
  changes invalidate pending work. Following writes no new artifacts or replay
  fields, and does not alter diagnostic capture frequency.

A CPython 3.9.6/macOS ARM64 check using stateless Python inference on a
153-input, 64/32-hidden-unit
masked-PPO archived actor (12,167 parameters) measured 30.7 ms for one cold store
request and 3.28 ms median across 20 subsequent action-target requests
(3.06–3.63 ms). Across all 21 requests there was one chunk decode and one model
load. The cached parameter arrays occupied 97,336 bytes; a representative JSON
response was 41,558 bytes with 43 graph nodes and 93 links. A separate warm
`tracemalloc` sample peaked at 226,727 bytes of transient traced Python allocation,
excluding already-cached arrays and untraced native allocations. Timings exclude
HTTP transport and browser rendering; they are not training-throughput numbers.
Temporary playback artifacts are local test fixtures and are removed after use.

Tests compare all nine algorithms' probabilities/Q means with actual model
prediction functions, check analytic derivatives against finite differences,
and cover packed-weight deduplication, stat invalidation, legacy inline archives,
current top-k/consent layouts, HTTP validation and asynchronous UI ownership.

## Sampled animal activations

[Actual brain activity](explainability.md#recorded-animal-activations) reuses the
existing diagnostic sampling schedule for every living animal. Each sampled
controller performs one diagnostic forward pass; its hidden values are detached
and rounded to FP32 for visualization storage only. Learning arithmetic, state,
RNG consumption and existing observation/prediction values are unchanged. No
inference is added on unsampled frames. Setting `recording.record_activations`
to `false` omits the new arrays and uses the previous diagnostic path.

Only hidden layers are added: inputs, action scores, values and quantiles already
exist in diagnostics. Recurrent PPO adds actual prior memory and the resulting
GRU activation; sampling never assigns to its live hidden state. MAPPO retains
only its actor's hidden tape, while its critic remains value-only inference.
No weights, optimizers, gradients, layer labels or weighted routes are repeated
in these samples. FP32 values use the existing lossless narrow-array codec;
there is no model or world arithmetic conversion.

The browser renders only the selected animal, keeps one world chunk and reuses
its activation DOM between diagnostic samples. The large window follows the
existing replay loop rather than starting a polling/inference loop. Closing it
releases its retained sample/view. Selection, death and recording/run changes
replace or clear the display. Inline maps show at most 64 units per layer; the
window shows all units. This saves display work without dropping recorded units.

A controlled 32-step native-backend check of the masked-PPO preset, seed 100,
with 48 prey/6 predators initially and its default binary/XZ recording cadence,
produced 48,660 bytes of replay chunks without activations and 74,776 bytes with
them: an additional 26,116 bytes. The 268 animal diagnostics contained 6,688 new
activation scalars (26,752 raw FP32 bytes). This is additional storage, not a disk
reduction compared with omitting the feature. The same scalars would occupy twice
as many bytes as FP64 before compression. Long-run costs depend on sample
cadence, populations and layer widths; this short check is not a throughput or
peak-RAM benchmark. Temporary fixture recordings are removed after verification.

Regression tests compare complete controller/world checkpoints and existing
recorded fields with sampling enabled/disabled across all nine learners, including
nonzero recurrent memory, and pass on both native and Python fallback paths.
Tests also verify one inference pass, no unused centralized critic tape, sampling
cadence, FP32 isolation, buffer ownership, seek/run boundaries and chart reuse.

## Rank-based explanation display

Input/output cards are the default. Whole-structure visualization uses the
existing complete activation arrays and declared stage sizes when hidden rank
is at most the viewer's k (default two). It does not request full matrices or
expand the saved replay payload. Every unit is represented; above 1,024 total
dense inter-stage links, connectivity is bundled instead of allocating one
element per edge. Deeper brains default to a stage-size summary.

Individual unit maps, recurrent details and selected weighted routes are lazy
Advanced disclosures. Threshold changes re-render only the selected view from
already loaded data; they do not issue explanation requests or change the model.
The threshold is separate from observation top-k and is not stored in environment
files or model/checkpoint signatures.

## Day–night rendering and metabolism

The board shares a single sunlight result during each draw: ground light,
night shading, dial and selected-animal body calculations consume that same
value. The overlay receives the effective body already used by the sight
caption, and passes it through to visibility geometry on a cache miss. These
are frame-local values, not a cross-frame cache; backward seeking and changing
recordings always read the current saved tick/configuration. Diagnostic radar
inputs still use their own sample tick. The existing overlay cache includes
effective radius and invalidates when that radius changes.

The dial retains only its last phase and SVG node references. An unchanged
phase returns before orbit trigonometry or SVG writes, with visibility still
updated when switching enabled/disabled recordings. Repeated status text is
not assigned again; disabled cycles skip hidden label construction. No browser
animation timer or replay storage is added.

In Python, effective bodies remain bounded by the species count and are rebuilt
once per tick. A native-metabolism configuration view reuses those body objects
and shallow-copies only affected species entries. It is allocated lazily only
for flat native metabolism, retained for that tick, and invalidated alongside
the body cache. Configuration is immutable during a run; loading another JSON
creates a new world. Views are read-only by convention and never checkpointed.
Restore/GRPO trial worlds reconstruct private caches. Terrain and unavailable
native kernels continue through the Python reference path.

Regression evidence includes exact native/reference checkpoint, observation,
event and reward equality across 120 steps, one sunlight computation per tick,
restore isolation, explicit frame-value reuse and byte-identical regeneration
of the viewer appearance sheets. These remove repeated operations; no measured
end-to-end speedup or RAM reduction is claimed.

## Champion capture and competition occupancy

Champion ranking and source metadata still update on each record improvement.
The champion book can reuse the detached model payload when its signature and
all packed FP32 parameter bytes match the previous capture for that controller.
Comparison is lazy: a changed optimizer counter forces a fresh snapshot without
building comparison buffers. An unchanged counter alone never permits reuse;
the cache compares `array.tobytes()` against the prior snapshot's packed values,
so direct edits, native optimizer writes, loading and signed-zero changes are
detected. The first reuse packs the old detached list once; subsequent reuses
retain its comparison bytes. Changes to recurrent memory or optimizer state alone do not alter
a warm-start model payload; exact resume checkpoints still retain that state.

The cache belongs to one champion book and retains only controllers that won a
criterion in the latest consideration pass. Its key includes controller identity
(verified through a weak reference), model signature and every parameter-buffer
byte in layer/field order. It holds immutable comparison bytes and detached
snapshots, never live model buffers. Non-winners are removed on the next pass;
restoration starts with an empty cache. Legacy FP64/list models use the existing
uncached capture path. Snapshot payloads shared among criteria remain immutable
by convention. No cache field enters checkpoints, brain artifacts or replays.

This replaces repeated Python float/list materialization with byte copying and
comparison; it is not a zero-copy cache. Retained comparison bytes cost four
bytes per cached FP32 parameter after reuse, plus small structure/weak-reference
overhead. Controllers updated between every capture retain no comparison bytes.
Existing champion records already retain the shared model snapshots. The cache
has no additional lifetime ownership of controllers, optimizers or replay buffers.

Within an enabled competition phase, live occupied positions are built lazily
on the first winning path challenge and then updated after every displacement
or contest death. Later contests see current occupancy without another full
population scan. The shared set is discarded when resolution returns; opportunity
targets remain selected from the original phase positions. Retreat order, seeded
ties, damage, movement effort and recorded event positions are unchanged.

Two controlled operation-count probes demonstrate the removed work. A 20-step
PPO fixture with reproduction disabled and rollout length 64, including founder
initialization, made 63 detached model captures before reuse and six afterward.
A six-animal phase resolving three path contests made five full animal-list
passes before reuse and three afterward (targets, challengers and one live set).
These are fixture-specific counts, not measured training speedups or peak-RAM
reductions. Exact state and replay comparisons are separate compatibility checks.

### Local timing and memory measurements

The following comparison uses CPython 3.12 on local macOS with the optional native
backend enabled. The baseline is the implementation before champion snapshot and
contest occupancy reuse (`4558cef`); the optimized version includes both changes.
Each workload used five fresh processes per version, alternating before/after
order. Values are medians. Complete-state hashes matched between versions.

| Workload | Before | After | Observed change |
| --- | ---: | ---: | --- |
| 1,000 champion-ranking passes over one unchanged 3,655-parameter predator brain from `ppo_large_predator.json` | 48.76 ms | 10.57 ms | 4.61× faster; 78.3% less elapsed time |
| PPO simulation, 60 founders, 120 steps | 199.00 ms | 197.78 ms | Essentially unchanged; 0.6% less elapsed time |
| Competition simulation, 54 founders, 120 steps | 716.08 ms | 682.37 ms | 4.7% less elapsed time |

The isolated champion probe increments the animal's age and source tick for each
consideration while retaining its weights; ranking, lineage and metadata handling
remain included. The simulation probes use [`ppo.json`](../environments/ppo.json)
and [`competition.json`](../environments/competition.json), seed 100, one fixed-seed
training run and a 120-step limit. Both reached step 120. Their timers include
experiment initialization and training, excluding imports, recording, rendering
and the final comparison digest. Other preset settings are unchanged.

The simulations are short: the small timing differences need longer measurements
before treating them as dependable gains. The 4.61× figure describes unchanged
champion capture, not whole-training acceleration or controllers that update
between every capture.

| Process peak RSS | Before | After |
| --- | ---: | ---: |
| Champion-ranking probe | 28.41 MiB | 28.25 MiB |
| PPO simulation | 30.83 MiB | 30.77 MiB |
| Competition simulation | 40.48 MiB | 40.50 MiB |

Peak RSS uses `resource.getrusage(RUSAGE_SELF).ru_maxrss`, sampled after the
workload and before building its comparison digest. It includes process imports
and initialization. These small differences do not establish a memory reduction;
peak RAM is effectively unchanged. The tested champion retains about **14.3 KiB**
of comparison bytes after reuse (3,655 × 4 bytes), plus container overhead.
The benefit is fewer repeated snapshot allocations rather than lower steady-state
memory in every workload. Legacy FP64 and Python-fallback timings were not
measured here; their compatibility was checked separately.

### Kernel assessment

Neither change requires a new native kernel: packed-byte extraction/comparison
and set membership already execute in CPython's compiled implementations. The
removed work was repeated object construction or full-world lookup preparation.
The existing optional kernels continue to cover numerical inference, gradients,
optimizers, visibility rays and spatial group metrics.

Top-k perception is implemented separately in the optional
[native selector](native-top-k.md). Its benchmark and exact-parity checks are
independent of the champion/competition optimization above.


## GRPO trial reuse and independent actor batches

The [workflow guide](training-workflow-performance.md) documents group-private
trial resets, immutable initial-observation reuse and independent dense-model
inference batches. It includes matched timing/RSS tables, component controls and
limits: recurrent/Q-learning decisions remain scalar, actor weights are never
merged, and native scratch exists only during a batch call.

## Rolling artifacts and masked-PPO batches

[Artifact retention](artifact-retention.md) documents separate policies for exact
recovery and recent qualifying best-brain milestones. All replay chunks remain.
The native masked-PPO batch handles local-input networks and stored masks in one
call, retaining the scalar path's floating-point order and per-animal learning
state. Invalid actions receive the same masked logit and zero derivative. FP64
models, subclasses or overridden objectives use the existing fallback; no GPU,
new dependency, shared policy or changed learning schedule is introduced.

Reproduce the bounded before/after measurements with a copy of the pre-change
package (including its built native module) outside the import path:

```sh
.venv/bin/python work/benchmark_artifact_retention.py \
  --baseline /path/to/baseline-package-root --repeats 3
```

The measured baseline package matches runtime commit `19085c5`; the comparison
includes the pending masked-PPO batch extension. The script uses fresh processes,
alternating order, matching seed 100, CPython
3.12 and native kernels. It compares full state hashes and restore round trips.
Temporary outputs are isolated from `results/`. The update workload applies 12
64-sample masked-PPO updates (four epochs, minibatches of 16) to the tracking
preset's then-current V4 482-input, 64/32-hidden-unit predator network. Training covers one
160-step tracking world, 48 prey and six predators, including initialization and
champion selection but excluding recording and disk writes. Storage writes eight
successive states of that setup and times the writers separately from stepping
and restore verification. Peak RSS includes imports/initialization; storage RSS
also includes explicit restore checks. Timings are local medians, not universal
speedup guarantees or process-independent RAM estimates.


| Workload | Before | After | Result |
| --- | ---: | ---: | --- |
| 12 masked-PPO rollout updates | 433.32 ms | 374.34 ms | 13.6% less elapsed time |
| 160-step tracking training | 2.629 s | 2.284 s | 13.1% less elapsed time |
| Eight self-contained recovery saves, retained disk | 5,707,800 B (eight files) | 1,455,864 B (latest two) | 74.5% less retained disk; older recovery points intentionally unavailable |
| Rolling two, optional shared static metadata | 1,455,864 B | 1,456,236 B including pool | 372 B larger; pooling is opt-in, not a claimed improvement |

The writer-only medians for eight saves were 7.969 s before, 8.096 s with rolling
retention and 8.014 s with shared metadata. Retention reduces disk, not compression
work: all eight snapshots were still serialized. This fixture does not establish
a faster checkpoint writer. Native batching improves learning, not checkpoint I/O.

Peak RSS medians were 39.95→37.33 MiB for update-only processes,
152.05→149.94 MiB for training, and 183.28/183.72/190.31 MiB for
legacy/rolling/shared storage probes. These are workload-specific process peaks,
including explicit restore verification in storage probes; they do not establish
a general RAM reduction. Most checkpoint bytes are dynamic numeric state, not
repeated static definitions. The three variants produced identical final state
hashes, and each stored checkpoint restored exactly. A separate regression test
checks identical decoded playback across enabled/disabled retention, including
chunk boundaries and multiple runs. No replay retention or recording-cadence
changes were used to obtain these savings.


## Shared model encoding and checkpoint copy reduction

See the [cache ownership and lifetime diagram](artifact-retention.md#save-time-memory-and-repeated-work)
and [checkpoint copy paths and file layout](checkpoint-data-structures.md#file-records-and-shared-values).

Periodic champion gates and exports now use one save-scoped model encoding
cache. A snapshot shared by several criteria is encoded once, and the same
canonical bytes/hash feed both the unchanged-weight gate and the weight writer.
A small PPO fixture with 16 criteria referencing three distinct snapshots needed
19 model encodings on first publication before this change and three afterward;
an unchanged-weight check with higher ranks needed 16 before and three afterward.
These are operation counts, not whole-training speedup factors.

The cache owns strong model references for one synchronous save, preventing
object-ID reuse. Its payload LRU is bounded to 4 MiB, in addition to one active
encoding and the writer's normal workspace; oversized or evicted payloads are
re-encoded and checked when needed. All cache data expires when the save returns,
including skipped saves. Later weight changes cannot reuse a previous save's
hash. No cache is added to a checkpoint or replay. Standalone model writes keep
their existing behavior and exact record bytes/checksums.

Synchronous checkpoint writing borrows existing retained-model/configuration/
report metadata instead of deep-copying those trees before deduplication. It
still constructs the world snapshot and streams one detached controller at a
time. The public `checkpoint()` API stays detached. The borrowed view is internal
to the blocking writer: simulation does not advance while it is consumed. This
is not a background checkpoint implementation or support for concurrent mutation.
Both standalone and optional shared-metadata file layouts remain unchanged.

Measurements below compare a saved copy of the pre-change **pending 0.0.2.8**
runtime against this implementation, on CPython 3.12/macOS with native kernels.
They use three alternating fresh-process samples per variant and report medians.
Reproduce with:

```sh
.venv/bin/python work/benchmark_encoding.py \
  --baseline /path/to/pre-change-package-root --repeats 3
```

| Workload | Before | After | Observed change |
| --- | ---: | ---: | --- |
| 20 checkpoint-metadata preparations | 660.80 ms | 11.56 ms | 98.3% less component time |
| One metadata preparation, peak traced Python allocations | 1,452,264 B | 126,464 B | 91.3% less component allocation |
| First periodic publication plus nine higher-rank/unchanged-weight checks | 831.41 ms | 188.06 ms | 77.4% less save/check time |
| Training + recording + periodic saves + recovery writes, 32 steps | 4.047 s | 3.703 s | 8.5% less complete workflow time |
| Complete workflow process peak RSS | 118.64 MiB | 99.23 MiB | 16.4% lower measured process peak |

The seed-100 fixture uses the tracking preset: 48 prey, six predators, the blended
scent layout and a 32-step limit. Metadata timing excludes `tracemalloc`; the
allocation row is one separately traced preparation. The periodic component's
higher-rank checks are controlled persistence probes; they do not simulate
learning improvements. Complete workflow timing includes founder initialization,
all simulation steps and diagnostic frames, brain checks every eight steps,
recording finalization and recovery writes at steps 16 and 32. Restore verification
and result hashing occur after the workflow timer/RSS measurement. No unbounded
run is loaded into memory; replay hashes are accumulated chunk by chunk.

Final state and decoded-replay hashes matched in every before/after sample.
Both variants retained exactly **3,777,620 checkpoint bytes** and **87,292 replay
bytes** in that workload. This change removes repeated work and transient copies;
it does not reduce recording detail, checkpoint contents or retained file counts.
The figures are local short-workload measurements, not guarantees for every
population, model depth or algorithm. They are separate from the earlier native
masked-PPO and rolling-retention benchmarks; do not add their percentages.


## Checkpoint metadata cache and PPO sample preparation

The synchronous checkpoint encoder now memoizes shared-container identity to the
already assigned internal `$ref` index during metadata traversal. A 128-entry LRU
owns strong references to existing metadata containers, not copies of their JSON
payloads. The key is object identity in this unchanged, synchronous traversal;
eviction recomputes the normal content hash. The cache clears on success or
failure **before the controller iterator is advanced**. Public `Encoder.encode`
remains mutation-sensitive. Separate equal objects still deduplicate by content,
and optional external `$meta` pooling retains its existing path. No cache survives
another save, enters a checkpoint, or changes restored ownership. See the
[checkpoint ownership diagram](checkpoint-data-structures.md#file-records-and-shared-values).

Native PPO prepares immutable observation/action/log-probability/advantage/target
rows once, when the first native minibatch is used. Masked PPO also prepares its
local observation and mask slices once. Minibatches gather the same row references
in their original shuffled order; gradients, current network weights, optimizer
steps and KL stopping retain their original behavior. The scratch list lives
only until `_update` returns and is absent from controller checkpoints. Scalar,
FP64 and customized paths that do not use these native kernels skip preparation.
For a complete 64-sample/four-epoch update, final sample-row construction falls
from 256 to 64; masked observation/mask slice construction falls by the same factor.
A regression fixture with eight references to one metadata object needs one JSON
fingerprint instead of eight, while keeping identical shared-table contents.

### Measurements and limits

Reproduce against a saved pre-change package root, including its native module:

```sh
.venv/bin/python work/benchmark_checkpoint_ppo_reuse.py \
  --baseline /path/to/pre-change-package-root --repeats 5
```

The baseline is the pending source after the ordinary-PPO calibration correction,
before these two optimizations, on committed base `2a15e0c`. Both sides use the
same current preset files. Measurements are five alternating fresh-process samples,
CPython 3.12/macOS, native kernels, seed 100. Raw reports stay in ignored `.cache/`;
all temporary model/replay files are removed after each worker. No existing result
or playback is modified. These are short local benchmarks, not general speedup
or RAM guarantees.

| Workload | Before median | After median | Observed difference |
| --- | ---: | ---: | --- |
| Ten metadata encodings after 16 training steps | 706.57 ms | 225.96 ms | 68.0% less component time |
| 64 ordinary-PPO rollout updates | 465.55 ms | 461.81 ms | 0.8% less; modest difference |
| 64 masked-PPO rollout updates | 1,204.12 ms | 1,189.02 ms | 1.3% less; modest difference |
| 128-step training + full recording + two recovery writes | 1.844 s | 1.772 s | 3.9% less complete workflow time |
| Complete workflow process peak RSS | 71.31 MiB | 68.50 MiB | 2.81 MiB lower measured peak |

The metadata component encodes the same borrowed metadata ten times with fresh
encoders, excluding world stepping and compression. Update-only workers use
64-sample rollouts, four epochs, minibatches of 16 and disabled KL stopping;
ordinary PPO uses 153 inputs and hidden `[64, 32]`, while masked PPO uses the
tracking preset's then-current V4 482 inputs and hidden `[64, 32]`. Synthetic rollout generation
and a separate traced allocation probe are outside the update timer. Neither
learning schedule nor production KL settings are changed by the benchmark.

The measured workflow used the then-current V4 ordinary-PPO preset: 32×24 cells, 64 prey and six
predators, 128 steps, full world capture every step, diagnostics every ten steps,
64 frames per chunk and binary/XZ recovery at steps 64 and 128. Its timer includes
initialization, training, recording/finalization and checkpoint writes; restore,
digest checks and cleanup are afterward. It excludes champion archive publication,
terminal progress and CLI CSV writing. The checkpoint optimization mainly explains
the save-heavy workflow gain; the update-only differences are too small to claim
substantial PPO training acceleration.

Retaining prepared samples reduces allocation churn, **not necessarily peak
memory**. A separate `tracemalloc` probe of one update measured ordinary PPO at
182,020→182,532 bytes and masked PPO at 581,252→768,772 bytes (+183.1 KiB).
One metadata encoding measured 515,700→513,596 bytes. RSS includes imports/native
allocations and should not be inferred from these Python-only peaks. Early KL
stopping may use only part of a prepared rollout, so its amortization can be lower.

All repeated workers matched full state hashes. Workflow checkpoint files were
byte-identical and restored to the exact final experiment state; decoded replay
hashes also matched across chunk boundaries. Both sides retained **3,055,004
checkpoint bytes** and **313,744 replay bytes**. These changes save repeated work,
not disk space or recording detail; retention and all playbacks remain unchanged.

## New-policy hotpath review

The generic [Discrete SAC, TRPO and LSTM PPO](additional-policies.md) implementations
already use existing dense inference/backpropagation and Adam helpers where
supported. Their new Python control/math paths have not been assigned measured
speedup claims. The review below identifies actual repeated work and candidates;
it is not a native-kernel benchmark. The subsequent
[native LSTM measurements](#native-lstm-forward-and-backward) cover the adopted cell kernels.

Implemented local reuse:

- TRPO partitions each Fisher direction into layer views once per Fisher product,
  rather than once for every rollout observation. With 64 samples this changes
  direction-slice preparation from 64 passes to one **per product**, not a 64×
  training speedup. Forward tapes/probabilities are already retained during the
  conjugate-gradient solve. Candidate line searches calculate one categorical
  distribution and skip gradient-vector construction when only the objective
  value and KL are needed.
- SAC prepares its actor/twin-critic gradient-list views once per sampled batch
  and hoists the constant retained-target coefficient out of parameter loops.
  Its current-state actor/critic and next-state target passes serve distinct
  mathematical purposes; they cannot all be removed as duplicate inference.
  Diagnostics reuse their existing policy probabilities/entropy for soft values.
- LSTM backpropagation computes each current-cell tanh once and slices the prior
  hidden vector once, rather than repeating them across gate-gradient rows.
  These values live only for that backward call.

| Priority | Remaining path | Existing/native opportunity | Required safeguards |
| --- | --- | --- | --- |
| 1 | SAC sampled-update network dispatch and target Polyak loops | First reuse independent actor/target inference batching; consider a fused discrete-SAC minibatch and target-update helper | Preserve sample/RNG order, terminal versus truncated targets, pre-update Q values, joint clipping and target update timing |
| 2 | TRPO Fisher-vector products and backtracking evaluation | Batched tangent/Fisher products and actor prediction across rollout states | Preserve damped CG residuals, candidate FP32 rounding, actual KL acceptance and exact actor rollback |

For a fully nonterminal SAC batch of size B, the straightforward implementation
performs six network traversals per sample: current actor, two current critics,
next-state actor and two next-state target critics. B=16 therefore has 96
traversals before backpropagation. This is an operation count, not 96 independent
Python-only matrix multiplies: existing native dense helpers already handle
eligible inner arithmetic. Actor decisions for the new separate-network bundles
currently retain scalar orchestration; they are not registered in the existing
cross-animal inference batch whitelist.

LSTM gate affine work scales as approximately `4h(I+h)` products per forward
step, before the head, for input width I and recurrent width h. TRPO repeats
Fisher products across conjugate-gradient iterations and rollout states. These
are stronger kernel opportunities than minor container cleanup. LSTM now uses
validated optional cell kernels; SAC/TRPO remain future candidates. Fallbacks
remain available without a compiler. Existing environment presets, recording
frequency and replay fields are unchanged.

Keep any future caches update-local or keyed by the full dependency set. Grid
position alone cannot cache policy output: energy, health, heading, visible
animals, LSTM state and learned parameters can all change at that position.
Training cadence, sequence length, batch size and lower precision are behavioural
or numerical tradeoffs, not transparent optimizations.


## Native LSTM forward and backward

The optional CPU extension now exposes `lstm_step` and `lstm_backward` under
native API v2. Forward computes all four gates plus hidden/cell updates in one
call. Backward computes gate derivatives, both recurrent memory gradients and
parameter-gradient accumulation. The existing native MLP head and Adam remain
unchanged; Python still owns sampling, chronological sequences, truncation, KL
stopping, birth/reset rules and optimizer cadence.

The kernels borrow FP32 parameter buffers and writable FP64 gradient buffers,
copy only observation/memory/tape vectors, and use call-local O(inputs + hidden)
workspace. No persistent per-animal cache, RNG or global scratch is added.
All cell-gradient shape/type checks and output allocation precede gradient
mutation. Forward retains the reference gate order; backward preserves the
reference left-to-right products, Python `sum` compensation and `** 2` libm
behaviour. Builds retain disabled fast-math and FMA contraction. Hidden/cell
values and arithmetic stay FP64; only existing parameter storage is FP32.

Each entry point is feature-detected independently. Existing extensions missing
them and FP64/list-backed models keep the Python cell. Rebuild and restart the
trainer to enable the new symbols; rebuilding does not replace already mapped
libraries in running processes:

```sh
.venv/bin/python -m work.build_native
```

The full reference backend remains selectable with `WTF_KERNELS=python`.
No checkpoint signature, field, replay codec, diagnostic frequency or viewer
change is needed. See [policy templates and memory ownership](additional-policies.md).

### Matched before/after measurements

Measured on local ARM64 macOS / CPython 3.12 with native kernels enabled on
**both** sides. The baseline is the pending `0.0.2.13` source before the LSTM
kernels, including its previous compiled extension, on committed base `1117b6d`.
Thus existing dense/Adam/world/replay kernels are retained in the baseline;
these are not comparisons with an entirely Python backend. Five samples per
variant run in separate fresh processes with alternating order. Reproduce using
a saved baseline package root containing its matching native module:

```sh
.venv/bin/python work/benchmark_native_lstm.py \
  --baseline /path/to/pre-lstm-kernel-package-root --repeats 5
```

| Workload | Before median | After median | Speedup |
| --- | ---: | ---: | ---: |
| 500 LSTM forward calls including MLP head and training tape | 316.63 ms | 11.18 ms | 28.33× |
| 500 backward calls including MLP head and nonzero recurrent gradients | 466.01 ms | 6.42 ms | 72.55× |
| Eight 32-sample LSTM-PPO updates | 815.86 ms | 25.32 ms | 32.23× |
| 32-step initialization + training + recording + two recovery writes | 8.154 s | 2.323 s | 3.51×; 71.5% less elapsed time |
| Complete workflow process peak RSS | 73.80 MiB | 70.78 MiB | 3.02 MiB lower measured peak |

Forward/backward fixtures use the predator network with 153 inputs, recurrent
size 16, six actions and an MLP head with hidden `[64, 32]`. Forward evolves
nonzero initial memory across 500 calls. Backward accumulates gradients over
500 calls to one fixed tape with nonzero future hidden/cell gradients; it does
not update weights. The update-only fixture uses 32 samples, sequence length 8,
two epochs, minibatches of 16 and disabled KL stopping. Synthetic rollout
construction is outside its timer; a separate ninth update measures traced
allocation and participates in the final-state equality check.

The workflow uses the current ordinary-PPO world's 32×24 cells, 64 prey and six
predators, with **both** controllers replaced by LSTM PPO for the benchmark only.
It retains source observation widths (69/153), head widths (`[16]` / `[64, 32]`)
and ecology, but sets recurrent size 16, 16-step rollouts, sequence length 8,
two epochs, minibatches of 16 and disabled KL stopping for a bounded measurement.
World seed is 100. It captures every frame, uses eight frames per chunk and
samples diagnostics every ten steps (plus chunk/final boundaries), then writes
binary/XZ checkpoints at steps 16 and 32. Initialization, stepping, recording
finalization and synchronous saves are timed. Restore/digest verification and
cleanup follow the timer. CLI CSV/progress and champion-file publication are
excluded. Maintained JSON presets are not modified.

Process RSS includes imports and fixture initialization; it is a workload-specific
peak, not a universal RAM reduction. Separate `tracemalloc` probes found one
forward call at 5,312→5,640 bytes, one backward call at 3,960→5,376 bytes and one
rollout update at 255,608→258,952 bytes. Native vector conversion/workspace can
slightly increase traced peak allocation even while reducing Python work.
Do not infer zero scratch allocation from the timing improvement.

All workers matched final model/optimizer/RNG/memory hashes. Workflow recovery
files and replay chunks were **byte-identical**, exact checkpoint restore passed,
and decoded replay hashes matched across chunk boundaries. Both sides retained
**7,745,060 checkpoint bytes** and **158,924 replay bytes**. The improvement is CPU
time, not disk compression or changed recording detail. Temporary artifacts are
removed after each worker; raw measurements stay in ignored `.cache/`. These
short controlled fixtures do not establish a speedup for every configuration or
better learning, and no Ubuntu/CPython 3.13 benchmark was performed locally.


## Resource and obstacle drawing

[Environment-object visuals](viewer-visual-reference.md#resource-and-obstacle-motion)
use one O(resources + obstacles) canvas pass during existing board draws.
Sway/shimmer is derived from the saved tick and coordinates, with no retained
per-frame animation history, per-object timers, image downloads or independent
animation loop. Paused frames consume no additional animation work. Reduced-motion
preferences are read once through a media-query object and redraw on preference
changes. The renderer preserves its canvas state and leaves replay inputs intact.

Faceted rocks and leaf curves require more canvas operations than the former
solid rectangles; this is a visual feature, not a claimed rendering speedup.
Simulation/learning work and replay storage are unchanged. The visual generator
reuses the same renderer, so controlled documentation specimens cannot drift into
a second implementation of resource or obstacle geometry.

### River and feeding overlays

River crests use map flow and the current saved tick; there is no separate timer
or water-history buffer. Feeding cues inspect only the latest four saved frames,
with at most 64 compact position records carried between chunks. Animal lookup
maps are created only for frames with confirmed food gains. These are viewer-only
allocations, with no new replay fields or training work. Random seeking rebuilds
short-lived cues from preceding chunks; legacy chunks smaller than four frames
can require up to three preceding reads. Normal playback retains one decoded
world chunk and the bounded cue list. Drawing costs increase slightly; no speedup
is claimed. See the [visual reference](viewer-visual-reference.md#river-flow-and-feeding-feedback).

### Sunlight-driven shadows

An enabled sunlit cycle adds one O(resources + obstacles + animals) geometry pass.
Direction, reach and opacity are computed once per board draw; silhouettes share
one filled path and avoid canvas blur/shadow filters. Disabled cycles and nighttime
return before building paths. No RNG, replay fields, timers or persistent caches
are introduced. This adds drawing work rather than accelerating rendering; the
Python simulation and replay size are unchanged. See [shadow behavior](day-night.md#decorative-object-shadows).

## Optional collective and consequence processing

The opt-in social path builds spatial buckets per observation/transition phase,
limits each observer to its nearest configured same-species neighbours and pools
only a short history. Context prediction reuses one local threat/body query across
all available actions. The consequence model stores bounded outcome-frequency
records, without neural optimizers or full experience replay, and trains each
animal's own physical transition once rather than once per witness.

Checkpoint state holds each species table once. Completed runs export a single
compressed aggregate file, rather than per-brain copies. Replay uses existing
sampled diagnostic vectors with 24/48 extra inputs; no additional world-frame
schema or event histories are introduced. Extra policy inputs still increase
network/buffer/sample size. The `seaside_social` example opts into compact capture
to limit diagnostic storage. No training speedup or net memory reduction is
claimed. See [configuration, retention and limits](social-learning.md).

## Collective learning hotpaths

Episode-scheduled consequence checkpoints pack the published and staged table
rows independently but copy their common physical contracts only once. Returned
rows and contracts remain detached from live model state. There is no retained
encoding cache, additional checkpoint field or artifact-layout change.

Witness processing checks both start and end positions for a moving subject.
When they are equal, one visibility query suffices within that same finish phase.
This reuses a pure result only for identical coordinates with the same observer,
world and sight parameters; it does not cache across steps or visibility changes.
Hidden starts of moving subjects still exclude their outcomes.

A local CPython 3.12 microbenchmark compared these two paths against `29d2c2f`
with the evaluation-freeze fix applied equally. Seven batches per implementation,
10 warmup calls, used 1,000 empty-checkpoint calls, 100 populated-checkpoint calls
or 100 witness-finish calls per batch. Values below are medians per call; traced
allocation peaks cover ten repeated calls after garbage collection, not process
RSS. Both species had published and staged tables in the populated case. The
stationary case used 35 animals with eight same-species witnesses each.

| Path | Before | After | Time reduction | Peak traced bytes, before → after |
| --- | --- | --- | --- | --- |
| Empty episode checkpoint | 51.54 µs | 27.73 µs | 46.2% | 8,736 → 5,384 |
| Episode checkpoint, 512 rows/species per table | 941.74 µs | 758.90 µs | 19.4% | 663,272 → 660,216 |
| Stationary witness finish | 341.64 µs | 247.48 µs | 27.6% | 125,088 → 125,088 |

An independent rerun reproduced exact state equality, visibility call counts and
allocation peaks. Its medians were 45.92 → 24.75 µs (empty checkpoint),
942.26 → 919.91 µs (populated checkpoint), and 392.22 → 380.94 µs (witness finish).
This timing variability limits the strength of speedup claims; the removed
copies and visibility calls are the stable structural improvements.

Visibility calls fell from 560 to 280 for the stationary fixture. Serialized
checkpoint/export values and witness recovery state matched the reference
exactly. Regressions separately check moving subjects with a hidden start and
snapshot independence. These are path-specific synthetic results, not measured
end-to-end training speedups; overall gains depend on checkpoint cadence,
movement and visibility costs. Replays, recorded observations, RNG use and policy
updates are unchanged. No disk reduction is claimed.

## Scalar DQN batches and shared sensory work

Scalar `dqn` and `ddqn` updates can use the optional native `dqn_batch` entry
point. It borrows each controller's private online/target buffers and accumulates
one sampled minibatch in the original order. Python still owns replay sampling,
epsilon decay, Adam/clipping, update counters and hard target synchronization.
DQN maximizes target values; DDQN selects online and evaluates the selected target
head. Terminal transitions omit bootstrapping; truncated transitions retain it.

The kernel validates and converts every sample and validates network/gradient
shapes before changing gradients. Its temporary converted samples and workspaces
are released at return. Old extensions missing the symbol, FP64 models and
custom controller/network hooks retain the scalar update. Rebuild/restart the
trainer with `.venv/bin/python -m work.build_native` to enable the new symbol;
no new dependency, model signature or recording format is introduced.

Scent reachability uses a per-world 128-entry LRU keyed by origin and radius.
The observation-phase preparation compares dimensions, obstacle coordinates and
water-cell membership, clearing geometry on topology changes. Only stock blocked
and depth methods qualify; custom callbacks and direct `sample` calls use the
uncached path. Geometry contains no animal positions or scent values. Each new
observation phase gets a separate dictionary of at most 4,096 combined cell
strengths, keyed by cell, ordered channel plan and blending mode. This preserves
fresh deposits/decay, species/sex channel ordering and blended versus max
reductions. Caches are excluded from checkpoints and rebuilt by restored worlds
and GRPO trials.

Social spatial buckets are reused only while ordered animal object identities
and bucket coordinates match. Within-bucket movement reads current animal
objects; replacements, reorderings, bucket crossings, births and deaths trigger
rebuilding. The owner retains at most one ordered population reference list and
one bucket map. No cached policy outputs or cross-step outcome estimates are
introduced by this optimization.

A candidate DQN/DDQN decision batch was tested separately. With 32 deep brains,
225 inputs, seven actions and epsilon 0.35, five alternating fresh-process
samples gave 75.66 → 75.20 ms for 160 decisions per animal (only 0.6%); the
three-call traced allocation probe increased from 5,400 to 11,568 bytes.
It was not adopted. Decisions continue through the existing scalar path.

### Reproducing the comparison

Save the pre-optimization package, including its compiled native library, in a
directory containing `wtf/`. Both variants must have the existing native kernels:

```sh
.venv/bin/python work/benchmark_shared_training.py \
  --baseline /path/to/pre-optimization-package --repeats 5 \
  --output .cache/shared-training-benchmark.json
```

The probe alternates separate baseline/current processes and requires identical
state hashes. Workflow cases also compare expanded replays and compressed binary
artifact hashes and verify checkpoint round trips. Artifacts live in temporary
directories. Microbenchmark timings exclude construction; complete workflows
include initialization, training, recording and a shared-metadata recovery write.
Traced peaks are measured separately after warmup, so they exclude already
retained caches. Process peak RSS includes setup. Caching trades bounded retained
memory for less repeated computation; no general RAM reduction is implied.

### Matched local measurements

Measured on ARM64 macOS / CPython 3.12.9, with existing native kernels enabled on
both sides. The baseline is the pre-optimization working tree on `22268d3`,
including pending adjacent-spawn and scalar DQN/DDQN support, saved with its
matching compiled module. Five fresh-process samples per variant alternate order.

| Workload | Before median | After median | Speedup |
| --- | ---: | ---: | ---: |
| 96 DQN transitions, 16-sample updates | 90.58 ms | 75.95 ms | 1.19× |
| 96 DDQN transitions, 16-sample updates | 116.97 ms | 99.89 ms | 1.17× |
| 160 scent phases, 48 stationary animals, radius 4 | 227.36 ms | 64.21 ms | 3.54× |
| 12,000 social preparations, 64 unchanged animals | 53.19 ms | 50.05 ms | 1.06× |
| 32-step DQN workflow with recording/recovery | 384.86 ms | 371.11 ms | 1.04× |
| 32-step DDQN workflow with recording/recovery | 388.13 ms | 376.76 ms | 1.03× |

Update probes use hidden widths `[64, 32, 16]`, 225 inputs and seven actions.
Workflow probes use 16 prey and six predators with three hidden layers, four-step
training/diagnostic intervals, eight-frame binary/XZ chunks and activations.
They include initialization, recording and final recovery writing; these short
workloads are not a forecast for large ecological experiments. Stationary scent
and unchanged social populations favor reuse; moving populations or topology
changes can reduce the benefit. The unchanged decision-path control measured
75.09 → 74.55 ms, illustrating small timing variation without a code improvement.

The three-call allocation probe for DQN/DDQN updates rose from 149,704 to 208,512
bytes (57.4 KiB extra peak) because native batches validate/convert samples before
mutating gradients. Scent rose from 7,608 to 12,808 bytes; unchanged social
preparation fell from 2,240 to 616 bytes. These are traced allocation peaks after
warmup, not retained cache sizes or process peaks. DQN workflow peak RSS was
71.69 → 71.66 MiB; DDQN was 71.45 → 71.56 MiB. This does not establish a RAM saving.

All complete-state, expanded-playback and compressed binary artifact checksums
matched. Total retained workload files were exactly 1,416,211 bytes for DQN and
1,417,652 bytes for DDQN on both sides. There is no disk-space reduction or data
loss from this computation-only rollout.


## Cumulative collective playback views

Collective-history aggregation is viewer-only and stores totals for living
animals and configured species, rather than a timeline of inputs. A detached
chunk-start copy supports local rewinds; distant jumps rebuild the earlier
prefix by streaming and releasing one older chunk at a time. Selection changes
reuse already accumulated living-animal totals. A temporary older chunk can
coexist with the selected chunk during a prefix rebuild; long recordings can
therefore take longer to seek without requiring whole-recording RAM.

Dashboard and inspector charts reuse their existing SVG nodes when averages
change. Species panels are updated only while the dashboard is open and their
species filter is visible. Closed dashboards retain aggregate numbers without
building per-step chart histories. No additional replay bytes are written and
training is untouched. Tests verify bounded live-agent state, constant chart
node identities, rewind/cancellation and read-only reconstruction. No timing or
peak-RAM improvement is claimed; this adds a bounded viewer feature.


## Published species/action summaries

State-independent consequence lookup builds one derived action-count summary per
species from its published bounded context table. Subsequent animals reuse it;
continuous learning invalidates only the updated species and episode publication
invalidates only changed species. Staged evidence remains separate. The cache is
bounded by species × 12 actions × four immutable rates and is omitted from checkpoints,
exports and replay. Restore/GRPO trials construct independent empty caches.
Species/action observation lookup also avoids constructing the querying animal's
local physical context. Context collection for real outcomes remains unchanged.
No speed or RAM benchmark improvement is claimed; tests verify reuse,
invalidation, publication isolation and exact restore.


## Combined consequence inputs

`consequence_scope: "both"` reuses the same published tables and general summary
cache, then appends context-matched estimates as a separate block. It collects
no additional events and introduces no persistent table or file-format version.
The general cache stores immutable rate tuples; fresh returned lists share their
float values across consumers, while list mutation cannot alter cached data.
Continuous learning/publication invalidation and cache restore boundaries are
unchanged. This avoids repeated divisions and duplicate live general-rate float
objects. No wall-time or peak-RAM improvement is claimed.

Combined mode adds 48 neural inputs per animal: first-layer parameters, captured
diagnostics and replay observations become wider. Cumulative viewer summaries
keep both sources independently in their bounded living-animal/species totals;
charts update existing SVG nodes. This is a configurable learning feature with
extra input/storage costs, not a behavior-preserving speed optimization.


## Reusing unchanged species playback means

Cumulative playback now reuses each species' instantaneous mean while its ordered
member objects and held social row objects remain identical. New diagnostics,
member replacement/order changes, births and deaths invalidate the relevant
species. Extinction removes its member references. Copied chunk-prefix totals
start with an empty derived cache, preserving independent ownership and seek
reconstruction. The cache holds two references per sampled living animal;
it does not retain another frame, duplicate row values or enter saved artifacts.

A Node.js 20.12.1 ARM64/macOS probe used 70 animals, two species, 1,600 frames,
combined consequence inputs and diagnostic samples every 50 steps. Five alternating
fresh-process samples compared the staged scope-26 implementation before this
reuse against the current code, on committed base `c01bf16`. Each process warms
once; timing includes replay aggregation and the probe's mean-build counters,
not simulation, rendering or disk/network access. A separate pass hashes every
frame's selected-animal and both-species cumulative view; all hashes agree.

| Measurement | Before | After |
| --- | ---: | ---: |
| Median aggregation time | 116.89 ms | 92.18 ms |
| Species mean reconstructions | 3,200 | 64 |
| Cached member pairs | 0 | 70 |

This is 1.27× throughput (21.1% less elapsed time) and 98% fewer mean
reconstructions in this sparse-diagnostic workload. Frequent samples or changing
populations reduce reuse. The extra member references trade bounded retained RAM
for fewer reductions/temporary rows; no peak-RAM, disk saving or end-to-end
viewer speedup is claimed. Reproduce with a saved pre-reuse package root:

```sh
node work/benchmark_collective_history.mjs \
  --baseline /path/to/pre-reuse-root --repeats 5 \
  --output .cache/collective-history-benchmark.json
```

Reference checks compare cached and forced-fresh reductions through new samples,
order changes, birth/death/extinction and independent copied states. Existing
chunk/seek, publication and dual-source charts retain their contracts.

## Reusing dashboard presentation nodes

Species metric cards and legend entries are retained by exact species name for
one recording/run. Every draw refreshes values and colors; filtering changes only
which existing nodes are attached. Recording/run changes clear these node maps.
Their size is bounded by configured species, with no per-frame DOM history or
cached observations. Compact species heatmaps are not repainted while a single
species is enlarged; returning to All species refreshes them at the current step.

Repeated metric choices in the six chart slots share one prepared series during
that draw only. The local map is discarded afterwards, so changing the action,
filter, recorded values, window or playback position never reuses stale series.
Chart bounds scan finite values directly instead of building flattened arrays
or spreading every species/time value into a function call. The population/food
overview similarly scans its existing run history. This also avoids the argument
limit that could break plots with many species. Movement accumulation builds
its shared prior-position map in one loop without filter/map staging arrays.

A three-species, 1,000-redraw DOM probe compared against the immediately preceding
uncommitted 0.0.2.27 viewer. It used the default six chart choices, one telemetry
point per redraw, species filtering, a theme change and a run reset; heatmap and
collective data were absent in this node-count probe and have separate tests.
DOM element construction fell from **62,105 to 143**, and `replaceChildren`
calls from **3,002 to 10**. Ordered DOM snapshots matched at every draw. These are
operation counts in a lightweight DOM/canvas harness, not browser timings,
whole-viewer speedups or measured peak-RAM reductions. Retained nodes deliberately
trade a bounded per-species cache for less repeated allocation.

Backward seeking immediately removes future telemetry from the visible trend
history before older pages finish loading. Theme/resize/filter redraws during
that request cannot redisplay later values. Regressions cover three exact
species, card reuse, missing/absent species, shared heatmap scales, filter/run
resets, independent draw-local series, and 150-species/1,024-point chart bounds.
Failed cross-chunk seeks cannot publish a snapshot from the wrong loaded chunk;
inspection requests require a successful retry. Passive redraws keep the committed
board during that failure. Sidebar heatmap repainting also checks frame, tick,
run and shared grid-container identity, including theme changes during pending
collective-history loads. A mismatched sidebar map stays hidden until the loaded
state is committed; no backup grid copy is retained. Simulation, recording bytes,
sampling cadence, numeric precision and normal replay controls are unchanged.

## Experimental CUDA backend

See [CUDA training](cuda.md) for optional dense action batches across eleven learners
and DQN/DDQN gradient batches. CPU remains the default. Real NVIDIA execution and
end-to-end speedups are unverified locally; the guide provides required hardware
tests and a matched full-loop benchmark, including host/device transfers.

The revised CUDA path removes per-sample parameter-gradient tensors, reuses host
staging and exact-content device weight caches, and batches complete private PPO
updates with device Adam. Default automatic inference keeps CUDA-trained PPO/DQN
actions on CPU. Optional FP32 and diagnostic timing/copy counters are described
in the [CUDA guide](cuda.md). Revised T4 performance remains unmeasured locally.

For the mechanism behind the CUDA changes, see the
[DQN/DDQN boundary diagram](cuda.md#where-the-cuda-path-saves-work) and
[PPO residency diagram](cuda.md#ppo-state-residency-and-publication). They show
removed transfers/intermediates and state ownership rather than timing charts.


### DQN gradient publication and optimizer placement

The [CUDA state-publication guide](cuda.md#dqn-state-residency-and-publication)
explains private device Adam/target state with eager CPU snapshots. The alternative
CPU-optimizer path widens FP32 gradients once in native code instead of per-value
Python conversion. A local 64-model `[20,64,64,32,10]` host-only conversion/unpacking
microbenchmark measured 15.295 ms → 0.455 ms per batch (33.6×), with byte-identical
gradients. This excludes GPU work, Adam and world simulation. Subsequent Kaggle
T4 results for 1,000 steps, three matched seeds, `[64,64,32]` hidden layers and a
32 MiB workspace favor GPU Adam in FP64 (DQN 1.051×, DDQN 1.118× versus V3 CPU Adam)
but CPU Adam in FP32. The latter yielded median loops of 8.637/9.173 seconds and
overall 1.745×/2.167× CPU-to-CUDA speedups. See the
[scoped V3 results](releases/0.0.2.30.md#kaggle-t4-optimizer-comparison) for
host-RAM and precision limits. No recording, checkpoint or viewer format changes.


## V6 semantic encoding and auxiliary work

V6 retains its complete expanded observation vectors and numerical precision.
The encoder starts from a compiled zero-input basis, fills current values and
reuses immutable scalar/bearing coefficient tuples. Each animal still receives
a fresh observation container. Shared numeric entries contain no animal identity,
world, model, optimizer or RNG state. Cache eviction changes work, never values.

![Exact V6 coefficient reuse and private learning](assets/diagrams/v6-coefficient-reuse.svg)

| Reused work | Bound | Key / ownership |
| --- | --- | --- |
| Structural plan, zero basis and block offsets | 128 layouts | Complete optional layout and resolved category configuration |
| Scalar memberships | 4,096 entries | Value, knot tuple and signed-zero marker |
| Circular bearings | 2,048 entries | Forward/right vector |
| Scalar reconstruction | 4,096 entries | Membership tuple and knot tuple |

Plans use slotted immutable records and indexed tail offsets. The auxiliary PPO
branch checks compiled confidence positions before decoding: no published
evidence means zero loss already, so target reconstruction and the masked input
copy can be skipped. Nonzero evidence uses the same target arithmetic, gradient
order and KL gate. Custom target hooks bypass this shortcut. This is independent
of action availability. All consequence input channels remain excluded from the
auxiliary prediction branch.

### Local matched benchmark

On macOS ARM64, Python 3.12.9, with the same CPU native kernels enabled in both
source snapshots, three isolated repetitions produced these medians. Baseline is
the immediately preceding uncommitted V6 implementation, not the older committed
runtime that has no V6. Ordinary PPO, seed 100 and `seaside_150_v6` were unchanged.

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 13,167 semantic encodes | 0.669 s | 0.427 s | 1.57× |
| First episode, 128 training steps | 5.390 s | 3.245 s | 1.66× |
| Published-evidence episode, 64 timed training steps | 3.835 s | 3.138 s | 1.22× |

The encoding fixture retains 4,389 observations from 64 canonical world steps,
with controlled species/action evidence windows added to cover repeated published
knowledge. A separate cold-cache allocation pass, including new caches and plans,
reduced retained traced Python memory from **47,120,590 to 28,097,054 bytes**
(**40.4%**). This is observation-bank allocation, not total training RAM. Median
process peak RSS was **99.2 → 96.1 MiB** in the first-episode test and
**112.0 → 95.0 MiB** in the published-evidence test. RSS includes interpreter and
native allocations; small differences depend on the allocator and workload.

The published-evidence measurement first completes 64 untimed steps and starts a
second episode. Its 2,175 total updates include that warmup; the first-episode
probe performs 2,116 updates. Initialization, checkpoint hashing and file I/O are
outside these training timings. No all-policy or GPU speedup is inferred.

Every paired encode hash, full checkpoint (world, weights, biases, optimizers
and RNG), work count and decoded replay matched. A separate 128-step recording
probe also matched the compressed binary+XZ chunk bytes exactly, including
sampled inspector data. Replays are retained and no storage format changed.
Tests additionally compare float64 bits, signed zeros, raw scalar types, custom
knots, partial decoding, cache bounds and instance/class target hooks.

Reproduce against a preserved pre-change source tree containing the same V6
preset and native extension; results stay in ignored local cache:

```sh
.venv/bin/python -m work.benchmark_v6 \
  --baseline /path/to/pre-change-source \
  --steps 128 --repeats 3 --passes 3 \
  --output .cache/v6-optimization.json
```

The harness alternates before/after order, rejects unequal state/work hashes and
uses temporary replay files. Its printed speedups are ratios of median times;
timing remains machine- and workload-dependent. No new kernel was needed:
these savings remove Python allocation and repeated pure calculations before
existing private learner/native arithmetic.


## V7 late-adapter training batches

The initial late consequence adapter used the per-sample reference PPO path.
That rebuilt evidence features for both the objective and adapter-gradient pass
on every epoch, and crossed the Python/native boundary separately for each actor
layer and sample. Smaller observations alone did not remove that overhead.

The revised stock PPO/masked-PPO path prepares each rollout sample's immutable
evidence features once, then computes physical-network forward/backward, bounded
logit residuals, final-policy loss and both gradients in one native CPU minibatch.
Features use contiguous **FP64** temporary buffers, preserving their original
values rather than quantizing them. This avoids retaining nested Python-float
matrices throughout the update. Current weights are read every minibatch; no
stale predictions or gradients are cached. Staging lives for one animal's update
and is omitted from checkpoints. The feature payload bound is
`len(rollout) × action_count × (12 × source_count) × 8` bytes
(normally bounded by configured `rollout_steps`), plus containers,
physical-input slices and native minibatch scratch. Sources are local/general/
context as configured. Sample object overhead and interpreter arenas are separate.

Learning rate, cadence, epochs, masks, RNG draws, FP32 weight storage and FP64
arithmetic are unchanged. Python retains advantage calculation, shuffle order,
KL rejection, auxiliary scheduling and Adam commits. Auxiliary computation can
use its separate [native minibatch](#v7-data-flow-and-auxiliary-batching). A rejected batch
commits neither actor nor adapter. Exact stock contracts are required: custom
instance/class adapter hooks and actor network overrides retain the reference
implementation. Custom actor or adapter optimizer overrides also bypass cached
preparation, including class-level overrides. Older or missing native builds work normally. This is a CPU
kernel, not new CUDA support.

### Local matched measurements

Python 3.12.9 on macOS ARM64, seed 100, the unchanged `seaside_150_v7` preset,
three isolated repetitions per policy/phase, alternating before/after source
order. Baseline is the immediately preceding V7 reference-adapter implementation
with its existing CPU extension. Both runs retain all configured learning,
tracking, scent and auxiliary behavior.

| Timed workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Ordinary PPO · first 128 steps | 6.485 s | 3.235 s | 2.00× |
| Ordinary PPO · published-evidence 64 steps | 3.995 s | 2.330 s | 1.71× |
| Masked PPO · first 128 steps | 6.741 s | 3.315 s | 2.03× |
| Masked PPO · published-evidence 64 steps | 4.342 s | 2.441 s | 1.78× |

Published-evidence tests first complete 64 untimed steps and start a second
64-step episode; only that second episode is timed. First-phase update counts
are 2,132 ordinary / 2,112 masked; published-phase total counts are 2,172 / 2,180
including warmup. Every paired complete checkpoint and update/step count matched.
Separate 128-step replay probes for both policies also matched decoded frames
and compressed binary+XZ chunk bytes exactly, including inspector samples.

Median host peak RSS before → after was 65.94 → 63.78 MiB for ordinary first,
67.73 → 65.88 MiB for ordinary published, 65.73 → 63.88 MiB for masked first,
and 65.69 → 66.00 MiB for masked published. Memory is roughly similar, not a
uniform RAM reduction. These short local runs establish a training-time gain for
the measured setup, not a guaranteed speedup for every configuration.


Reproduce from the checkout against a preserved pre-optimization V7 source tree
with its original compiled CPU extension. Both sources must use the same preset,
interpreter and native build flags:

```sh
.venv/bin/python -m work.build_native
.venv/bin/python -m work.benchmark_v7_training \
  --baseline /path/to/pre-optimization-v7-source \
  --steps 128 --repeats 3 \
  --output .cache/v7-training-optimization.json
```

The harness alternates source order in fresh processes, excludes initialization,
I/O and checkpoint hashing from training times, and rejects mismatched complete
checkpoint hashes or work counts. A separate recording check requires identical
decoded frames and compressed binary+XZ chunks. Host RSS is captured before
checkpoint hashing; allocator/OS variance limits small peak-RSS comparisons.
Local software timings do not establish other-OS/GPU performance or biological
improvement. Replay storage/layout, diagnostics cadence and sampled data stay
unchanged. The separately requested inspector cleanup removes verbose V7 text
and uses a small scent compass; it does not change recorded observations.


This new late-adapter CPU kernel is specific to `ppo` and `maskable_ppo`. Compact
V7 observations and ordinary consequence-input mode support all fourteen
algorithms. The [algorithm matrix](observations.md#algorithm-support-for-v7-and-consequence-handling)
distinguishes those capabilities and explains why each other learner needs its
own loss/target/recurrent integration before late placement can be enabled.


## V7 data flow and auxiliary batching

V7 observation assembly sends compact social action rows directly to the final
vector. It no longer pads scent to nine values, builds unsupported-action rows,
or copies a complete V5-sized social tail only to discard those entries. Physical
and visual-history calculations retain their established ordered projection;
this is not a new observation version or a precision change. Published general
consequence rates are immutable shared tuples, invalidated by species publication
(or each continuous update); final observation tuples remain independently owned.
Custom social and consequence providers retain the reference call contract.

Each stock PPO/masked-PPO update prepares auxiliary targets and masked physical
inputs once per rollout, after its first accepted KL gate. Preparation is bounded
by the owning rollout, recomputed for the next update and never checkpointed.
Current network outputs, gradients and Adam steps are **not** cached across epochs.
V7 avoids allocating zero-filled evidence padding that the physical trunk would
immediately discard. V1–V6 continue using their existing target semantics.

An optional native CPU auxiliary minibatch reuses model buffer views and one
activation workspace across samples. It runs after actor/adapter gradients, in
the original sample and reduction order, preserving exact additive gradients
and KL-gated commits. Missing/older extensions, FP64 parameter archives and
custom numerical/evidence hooks retain Python execution. This does not add a
CUDA kernel or share animals' weights, optimizer state or trajectories.

Recorded stock late-adapter diagnostics compute one residual instead of three,
reusing it for final probabilities and inspector fields. No diagnostic sample,
activation, replay frame, event, metric or playback is removed. Checkpoint and
replay formats are unchanged; rebuild CPU kernels and start a new Python process
for native auxiliary batching.

### Matched data-flow measurements

Python 3.12.9, macOS ARM64, seed 100, unchanged `seaside_150_v7`, three isolated
alternating repetitions against the immediately preceding native late-PPO setup.
Use the same `work.benchmark_v7_training` command above with a preserved baseline
containing its original native extension. The first phase times 128 steps; the
published phase warms up a 64-step episode and times the next 64-step episode.
Initialization, recording I/O and checkpoint hashing are outside these timings.

| Workload | Before median | After median | Speedup | Median peak host RSS before → after |
| --- | ---: | ---: | ---: | ---: |
| Ordinary PPO, first episode | 3.276 s | 2.932 s | 1.12× | 66.61 → 62.77 MiB |
| Ordinary PPO, published evidence | 2.363 s | 1.715 s | 1.38× | 69.61 → 69.20 MiB |
| Masked PPO, first episode | 3.529 s | 3.483 s | 1.01× | 59.30 → 62.20 MiB |
| Masked PPO, published evidence | 2.441 s | 1.772 s | 1.38× | 69.42 → 65.19 MiB |

Complete checkpoint hashes and step/update counts matched in every pair.
Separate 128-step recording probes matched decoded frames and compressed chunk
bytes exactly for both policies. All 28 benchmark records passed. First-episode
masked PPO is effectively unchanged; most speed benefit arrives when published
consequence targets become available. Peak RSS does not show a consistent large
reduction; Python allocator variation and extra bounded target staging matter.
These short measurements are not a promise of the same speedup on larger worlds,
other machines, other learners or GPU execution. Storage size remains unchanged.
