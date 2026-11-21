---
name: wtf-optimize
description: Review or optimize Wildlife Trial Framework training loops, observation caches and replay/checkpoint storage while preserving simulation and playback behavior.
---

# Optimize WTF computation and storage

Read [performance](../../../docs/performance.md), [binary storage](../../../docs/binary-storage.md)
and [development checks](../../../docs/development.md) for the affected path.
Inspect code first. Respect requests to avoid profiling; benchmarks are optional
and must be within scope. Do not launch long training as a default verification.

Trace work across algorithms, rather than assuming a DQN-only fix. Look for
repeated policy/value inference, context construction, deep copies, discarded
networks, per-step allocations and champion snapshots before adding abstraction.
Use the existing numerical/call-count reference tests to establish equivalence.
Changing learning cadence, precision, observations, RNG consumption or diagnostic
frequency is a behavior tradeoff, not a transparent optimization.

For caches, write down the cached value, complete key, invalidation trigger,
size bound and owner lifetime. Favor an in-memory dict/LRU when appropriate.
Observer position alone is insufficient: targets can move, die or be born and
food changes. Keep ephemeral caches out of checkpoints; ensure trial worlds do
not share mutable caches. Check observations after these changes against an
uncached/reference path, including a stationary observer.

For storage, reuse existing binary codecs and compression readers. Preserve
independent chunk decoding, seeking, run selection, event animation carry,
metrics/prediction samples, exact numeric round trips and old readers. Decode
only needed chunks; avoid experiment-wide materialization. Model checkpoints
must retain state needed for exact continuation, including optimizer and RNG
state. Champion warm starts have different requirements; inspect the format.

Do not convert active recordings. If conversion is requested, use the existing
verified/atomic storage tooling and retain originals on verification failure.
See `ecosystem.optimize_storage --help` before choosing flags. Confirm logical
round trips, Python/JS codec parity and playback across a chunk boundary.

Choose relevant suites from development docs (cache, training-loop efficiency,
binary storage, checkpoint, replay, mutation and JS playback tests). Record only
measured byte/time/RAM results with a baseline/workload; otherwise describe the
removed operation and expected benefit without invented speedup claims. Update
applicable guides and use the rollout workflow for an adopted implementation
change. Keep individual experiment IDs and ecological observations out of guides
and releases; reusable benchmark methods and verified implementation properties
may document the optimization.
