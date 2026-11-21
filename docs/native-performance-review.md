# Native kernels and runtime data widths

[Performance guide](performance.md) · [Binary storage](binary-storage.md) · [Private learning loop](algorithms.md#private-learning-loop)

Code-level assessment, 2026-11-18. The backend already runs on CPython. The useful
next step is optional Cython/C kernels over packed buffers, rather than compiling
the whole simulator indiscriminately. No native backend, precision change,
profiling run, speed benchmark or new training experiment was introduced by this
review. Priorities below are inferred from the code and operation counts.

Subsequent implementation: [rollout 42](releases/0.0.0.42.md) adds packed float32
model/optimizer storage and lossless narrow binary arrays. The table below is
the preceding assessment, not an inventory of work still pending.
[Optional CPU kernels](native-kernels.md) now implement shared dense math,
DQN gradients/SGD and Adam. The [second rollout](native-kernels-v2.md) adds GRU,
quantile, fused PPO/inference, perception/ray, world/replay/dashboard helpers.
Packed rollout columns remain a proposal. Double arithmetic is retained.

## Most useful boundaries

| Priority | Current code | Candidate change | Coverage and constraints |
| --- | --- | --- | --- |
| 1 | `MLP.forward`, `_predict`, `accumulate_gradient` in [ppo.py](../ecosystem/ppo.py); Q `_hidden`/`learn` in [brain.py](../ecosystem/brain.py) | Persistent contiguous float64 weights/gradients plus compiled affine/activation/backprop loops | Shared MLP benefits the policy algorithms, QR-DQN and recurrent head. Q has a separate implementation needing an adapter. Preserve each animal's independent state. |
| 1 | `_adam_step` in [optimization.py](../ecosystem/optimization.py) | Compile clipping/norm, moment and parameter-update loops with reusable buffers | Shared by eight algorithms; legacy Q uses SGD. Do not change gradient scaling, update order or optimizer counters. |
| 2 | `Experience`, `RolloutStep`, `PolicyStep`, `RecurrentStep` and their buffers | Packed per-controller replay/rollout columns; float64 numeric payload, byte-sized bounded actions/flags | Removes retained Python objects across all algorithm families. Replay observations already share adjacent tuple references; avoid storing a new duplicate next-observation row unnecessarily. |
| 2 | `quantile_huber` in [qrdqn.py](../ecosystem/qrdqn.py) | Compile the pairwise quantile loop and chosen-head gradient path | Work scales with prediction-quantiles × target-quantiles per sampled transition. Whole-MLP backprop currently also visits zero-gradient action heads; a sparse-head path could avoid those operations. |
| 2 | `GRUNetwork.step` / `backward` in [recurrent_ppo.py](../ecosystem/recurrent_ppo.py) | Compile gates and recurrent gradient loops on packed matrices | Removes repeated row slices and gate-temporary allocations. Preserve chronological order, stored hidden states and truncation boundaries. |
| 3 | `visible`, `_water_sight_cells` in [world.py](../ecosystem/world.py); `SpatialIndex.nearest` in [perception.py](../ecosystem/perception.py) | Fuse eligible wall/water ray traversals; later compile rays/candidate scans against compact masks | A cold extended-water query can currently walk the same Bresenham line twice. Keep wall occlusion, ordered endpoints, tie-breaking and complete dynamic invalidation. |
| Later | [binary_io.py](../ecosystem/binary_io.py) | Direct buffer-aware numeric encoding, compiled varints/table loops if needed | Existing `struct` numeric packing and XZ/gzip are already native operations. Boxing, repeated type scans and `*values` expansion remain Python overhead. Do not change the replay wire format casually. |

The present dense layers slice each weight row before `sum(map(mul, ...))`.
That already puts part of iteration in CPython's C implementation, but it still
creates a row list and Python-number products. A native kernel can read the
original contiguous matrix using offsets, accumulate in C doubles and avoid those
objects. Compiling code that still operates on Python lists/objects leaves much
of this overhead intact.

For QR-DQN, preserve the chosen head's reduction order when skipping zeros and
verify signed-zero behavior. A sparse gradient update must not accidentally skip
nonzero optimizer moments from earlier updates to another action head.

## Packing often matters more than lower precision

On the current 64-bit CPython 3.9.6 interpreter, an independently allocated Python
float object occupies 24 bytes, before its container reference. The following
`sys.getsizeof` measurement includes the container and retained independent float
objects; it is **not process RSS or a runtime speed measurement**.

| 1,024 values generated as `float(i) / 7` | Retained bytes |
| --- | ---: |
| Python list plus individual float objects | 33,432 |
| `array('d')`, float64 | 8,256 |
| `array('f')`, float32 | 4,160 |

Packing retains about one quarter of this list representation's footprint even
without reducing numeric precision. Float32 then roughly halves the packed
payload again, but rounds values. Results depend on allocator/container sizing;
Python lists can share immutable floats, especially initial zero-filled biases
and optimizer arrays. These numbers are not a universal fourfold RAM reduction.

An additional one-time measurement of 60 initial 32-feature observation tuples
retained 52,912 bytes including shared references; a flattened float64 buffer used
15,968 bytes. That buffer excludes row metadata and does not preserve the
original integer-versus-float element types, so it is not a drop-in serialization
replacement. No steps or training were run for this object-layout measurement.

These parameter-only estimates illustrate scale after updates have produced
independent float objects. They exclude container headers, experience buffers,
activations, configuration and Python runtime memory:

| Current predator model | Trainable parameters | Persistent copies counted | List payload estimate | Packed float64 payload |
| --- | ---: | --- | ---: | ---: |
| Enclosed-valley PPO | 664 | weights + Adam moments/variances | 63,744 B | 15,936 B |
| Large PPO | 3,655 | weights + Adam moments/variances | 350,880 B | 87,720 B |
| Recurrent PPO | 9,511 | weights + Adam moments/variances | 913,056 B | 228,264 B |
| QR-DQN | 6,592 | online + target + Adam moments/variances | 843,776 B | 210,944 B |

These are arithmetic estimates from the current preset shapes, not measured live
controller heaps. GRPO's frozen reference is additional. Saved checkpoints and
replays already pack numeric arrays; shrinking **live** data would primarily
help training RAM, not yield the same reduction again on disk.

## Dtypes to consider

| Data | First choice | Reason / boundary |
| --- | --- | --- |
| Weights, gradients, optimizer moments, observations and returns | Packed float64 | Remove boxing before changing numerical behavior. |
| Action index, heading, terminated/truncated/mask flags | uint8 / byte flags | Current action indices fit in 0–7; heading in 0–3. Pack flags together when useful. Q currently stores shared action-name references; the string is not copied per transition. |
| Valid stored x/y coordinates | uint8 is sufficient for current 256-cell dimension cap; uint16 is a simple general-purpose choice | Neighbor probes and differences must use signed/wider arithmetic for −1/256 boundaries. Merely changing a Python annotation does not shrink a Python int. |
| Static blocked/water-present masks | byte buffers or bitsets | Presence is Boolean; elevation, depth and flow may be fractional and should remain float64 initially. |
| Dense per-step occupancy indices | int32 with −1 for empty | Use a dense index into live animals, not the unbounded lifetime animal ID. |
| Animal IDs, seeds, counters, parent IDs | Keep existing exact representation until bounds are explicitly enforced | Birth IDs grow over time; −1 sentinels and large RNG values preclude blindly shrinking everything to uint16/uint32. |
| Float32 policy/replay state | Explicit experimental precision mode | Can alter probabilities, advantages, clipping and action outcomes; needs dtype metadata and separate comparisons. |

Float32 is particularly inappropriate as a silent change to health, energy,
resource growth or fractional river carry. Threshold comparisons and accumulated
rounding can change death, reproduction or drift steps. Equal seeds would not
imply the previous trajectory. Float16 is not a first target for these small CPU
models and sensitive optimizer statistics.

## Integration details that determine the outcome

1. **Keep buffers native between calls.** Rebuilding NumPy arrays from lists for
   every tiny 16-neuron inference can overwhelm useful compute. Pure-Python loops
   over `array('d')` still box values when reading them and can trade speed for RAM;
   the paired native-kernel change is the stronger candidate.
2. **Avoid unnecessary precision loss.** A double-buffer native backend should
   preserve accumulation order and disable reassociation/fast-math/FMA contraction
   where needed. The current affine expression sums products before adding bias;
   starting the accumulator at the bias is not necessarily equivalent. Validate
   behavior against the supported Python runtime rather than assuming any C
   implementation reproduces Python arithmetic across versions/platforms.
3. **Preserve state ownership and sampling.** Each animal keeps private parameters,
   optimizer and RNG streams. Packed ring buffers must preserve logical sample
   order, overwrite cursors and terminal/truncation flags. Adjacent Q observations
   currently share tuples; a naive two-row-per-transition layout can duplicate data.
4. **Adapt the boundaries.** Current checkpoint/model code expects lists/tuples,
   uses list `.copy()` and slice assignments, and often requires exact integer/
   float types. New internal buffers need explicit lossless serialization adapters
   and compatible public snapshots; replacing lists blindly will break this.
5. **Keep the pure-Python fallback.** Native kernels can be optional. Moving only
   arithmetic/rays leaves orchestration, reproduction, event order, progress and
   configuration inspectable. Build dependencies should be isolated from normal
   runtime installation.
6. **Do not add two authoritative worlds.** Packed occupancy/terrain structures
   must replace an existing representation or be bounded derived caches with
   explicit invalidation. Dynamic obstacle edits, moved/dead/newborn animals and
   changed food still need immediate correct observations.

`Animal`/`Resource`/`Event` instance dictionaries are another allocation source,
but slots are a lower-priority change for small populations. Python 3.12 now
supports dataclass slots, but several `vars(...)` snapshot paths still require a
serialization refactor before adding `dataclass(slots=True)`.
GRPO also rebuilds `random.Random` objects for every agent/trial. Preserve exact
streams if pursuing that path; network and buffer work is the clearer shared
starting point.

## Proposed rollout order

1. Packed float64 network/optimizer state plus optional compiled MLP, Q and Adam
   kernels; keep numeric/reference/finite-difference and exact-resume checks.
2. Packed replay/rollout columns and reusable activation/gradient scratch buffers
   across all algorithms; verify ring wraparound, child independence and RNG order.
3. Specialized quantile/GRU kernels and fused wall/water ray checks; verify current
   dynamic-cache and Python/JS replay parity tests.
4. Consider opt-in float32 only after a stable lossless backend, with recorded
   precision, compatible checkpoint handling and matched fresh/frozen-policy trials.

No end-to-end speedup or total-RAM percentage is claimed from this review.

## References

Standard-library [numeric arrays](https://docs.python.org/3.9/library/array.html)
expose a buffer interface; actual widths should be checked through `itemsize`.
Cython [typed memoryviews](https://docs.cython.org/en/stable/src/userguide/memoryviews.html)
can access buffer-backed data in compiled code without requiring NumPy storage.
Those interfaces support the proposed design; they do not measure this project's
performance.
