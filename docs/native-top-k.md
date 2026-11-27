# Native top-k perception

[CPU kernels](native-kernels.md) · [Performance](performance.md) · [Observations](observations.md)

The optional C extension fuses candidate collection, distance ordering and
visibility/range filtering for `observation_top_k > 1`. It accepts the same
incremental spatial-index iterators as the reference selector, so it does not
rebuild a full-world target list for each observer. All other learning and world
kernels remain independently available.

![Python cache ownership, native candidate selection and Python feature construction](assets/diagrams/native-top-k.svg)

## Exact behavior and ownership

Candidates sort by the same tuple as Python: squared distance, insertion order,
x and y. Self identity is excluded; a food patch under the observer remains
present. Sorted candidates receive the existing integer visibility/water ray
check until k visible, in-range targets are found. Occluding target cells and
water counts that exclude both ray endpoints retain their existing semantics.
The kernel returns integer offsets only. Python still computes `math.hypot`,
normalizes by the effective radius, adds presence flags and pads unused slots.
There is no change to observation dimensions, ordered actions or model signatures.

Dynamic bucket revisions, observer windows and result caches remain Python-owned.
Their keys and invalidation for movement, food, births/deaths, terrain/walls,
vision radius and directional water range are unchanged. C storage belongs to
one query and is freed before returning, including error paths. It holds four
`Py_ssize_t` fields per candidate (32 bytes on a 64-bit build), with geometric
capacity growth. There is no persistent native cache or per-animal C allocation.

Only the recognized world visibility callback and supported water-range tag
activate the fused selector. Custom callbacks stay Python. Nearest-only queries
keep the existing k=1 kernel. Missing extensions, `WTF_KERNELS=python`, or an older
API-2 extension without `nearest_many` use the reference selector. The extension
API remains backward compatible; rebuild explicitly to enable the new symbol:

```sh
.venv/bin/python -m work.build_native
```

New imports never build or install an extension. No replay, checkpoint, champion
or viewer format changes are required. Existing recordings do not need conversion.

## Reproduce the comparison

The benchmark toggles only `nearest_many` dispatch; other native kernels stay
enabled in both versions. Each sample uses a fresh process, alternating reference
and native order. State hashes must match across all samples; recording cases
also require byte-identical compressed replay chunks and byte counts. Temporary
recordings are cleaned after each sample and results stay in ignored `.cache/`.

```sh
WTF_KERNELS=native .venv/bin/python -m work.performance.top_k \
  --environment environments/competition.json --case train \
  --steps 240 --samples 5 --prey-k 3 --predator-k 10 \
  --output .cache/top-k-train.json
```

Use `--case observe` for forced query/ray-cache misses on unchanged world geometry,
`--case train` for initialization plus simulation/learning, and `--case record`
for simulation plus replay capture/compression. Observation probes are isolated
selection workloads, not whole-training speedups. Timings exclude imports and
final digest construction. Peak process RSS includes imports/initialization and
is sampled before hashing. Numeric reductions, policies, seeds, dimensions and
recording cadence otherwise remain the same in each matched pair.

## Local measurements and validation

Local macOS / CPython 3.12.9 results below use five fresh-process samples per
version, alternating order, seed 100, 240 phases/steps and prey k=3 / predator
k=10. Champion/competition reuse and every other native kernel are enabled on
both sides. The reference column disables only the new selector; it is not an
all-Python baseline. Times are medians.

| Workload | Reference selector | Native top-k | Speedup |
| --- | ---: | ---: | ---: |
| Observation cache-miss probe, flat competition map | 0.4547 s | 0.2906 s | 1.56× |
| Observation cache-miss probe, strong-predator river map | 0.5277 s | 0.2499 s | 2.11× |
| Training, flat competition map | 1.2888 s | 1.1952 s | 1.08× |
| Training, strong-predator river map | 2.3656 s | 2.2570 s | 1.05× |
| Training + recording, day/night map | 1.9196 s | 1.8326 s | 1.05× |

The maps are `competition.json`, `river_crossing_strong_predator_deep.json` and
`day_night.json`; the runner assigns the stated k values to diet-specific brain
copies without modifying preset files. All training probes reached step 240.
Observation probes perform 240 forced-miss phases after initialization, without
advancing world time. These short local measurements show about 4.6–7.3% less
whole-training time and 4.5% less training-plus-recording time; they are not
universal speedup guarantees. Do not attribute the isolated 2.11× perception
result to the whole simulation or to already-cached observations.

| Peak process RSS | Reference selector | Native top-k |
| --- | ---: | ---: |
| Flat observation probe | 28.95 MiB | 28.84 MiB |
| River observation probe | 29.00 MiB | 28.80 MiB |
| Flat training | 42.16 MiB | 42.47 MiB |
| River training | 54.28 MiB | 54.53 MiB |
| Day/night training + recording | 50.83 MiB | 50.89 MiB |

Peak RAM is not consistently lower. Compact query-local candidates reduce Python
object construction, but these process-level figures do not establish a memory
reduction. The recording probe produced exactly **515,708 compressed replay
bytes** per sample on both paths, including diagnostics and activations.

Validation covers randomized target ordering/ties, own-cell food, self exclusion,
water range and occlusion, unsupported input cleanup/error propagation, k=1
routing, custom callbacks and missing-symbol fallback. All fourteen controllers have
matched per-step checkpoints and byte-identical short replay chunks across chunk
boundaries. Dynamic world comparisons include walls, food disappearance, deaths,
terrain, restored worlds and different day/night observation radii. Existing
observation/cache tests additionally cover movement and births.

Full test, formatting and documentation audits are recorded in
[rollout 0.0.2.3](releases/0.0.2.3.md).
