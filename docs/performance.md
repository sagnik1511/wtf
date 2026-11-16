# Performance and benchmarking

[Project overview](../README.md) · [Architecture](architecture.md) · [Development](development.md)

Commands in this guide run from the project root.

Historical measurements describe the implementation evaluated at each optimization stage. Rerun the probe for the current revision and your workload.

The engine shares a per-observation-phase list of positions and resources instead of rebuilding it for each animal. An incremental spatial index groups food and animals into nearby cells, so limited-vision agents skip distant candidates. Small target sets and wide-vision queries use a full scan. Squared-distance filtering preserves visibility and insertion-order ties, including across cells. Each animal keeps a spatial window: movement reuses its overlapping bucket references and updates only entering/leaving strips. Shared buckets receive target moves, births, deaths and food-presence changes once per phase. Stationary observers reuse nearest-target results while the relevant bucket versions stay unchanged. Movement and obstacle edits force visibility to be checked again because occlusion inside the overlap can change. Energy, health, heading and blocked neighbors are always current. Derived caches are rebuilt after checkpoint restoration and released for dead observers; authoritative state is reconciled each phase to detect manual edits too. Small networks use native multiplication iterators with the original summation order. Training computes Q-network weight offsets once per neuron, and MLP backpropagation avoids checking the input-layer case for every weight. Optimizer mathematics and accumulation order remain unchanged. Replay experiences use compact immutable tuples. Snapshot creation copies primitive entity fields without recursive dataclass conversion.

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
