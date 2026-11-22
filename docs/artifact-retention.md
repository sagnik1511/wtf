# Best-brain milestones and rolling recovery

[Run and resume](running.md) · [Binary storage](binary-storage.md) · [Champion selection](brain-library.md)

With recording enabled, large experiments keep **every replay chunk, metrics row
and numbered run champion archive**. Training-only profiles skip replay creation;
retention does not delete existing playback. Follow the
[training/inference tutorial](tutorials/04-training-and-inference.md) to choose
capture and retention independently.
Retention affects only new recovery files and periodic best-brain milestones owned
by the current invocation. Existing experiments are not scanned, converted or pruned.

![Recovery checkpoints and best-brain milestones have separate retention and shared metadata](assets/storage/artifact-retention.svg)

## Two independent policies

| Purpose | Default | What is retained |
| --- | --- | --- |
| Exact recovery | Every 500 steps, configured run boundaries, and final | Latest two successful full checkpoints, including final; each contains changing world, controllers, optimizers, buffers, RNGs and run progress |
| Best-brain milestones | Check every 250 steps | Two most recent **qualifying best** collections, rather than arbitrary recent snapshots; shared lossless weights within this periodic archive |
| Playback and reports | Existing recording cadence | Every run; no retention deletion or sampling changes |
| Numbered run champion archives | End of each training run | Existing complete per-species/sex/criterion archives; unchanged chaining and selection |

A periodic best-brain collection advances only if at least one species/sex/criterion
has a strictly better lexicographic rank **and different model weights** than its
last saved milestone. Optional `brain_save_min_score_delta` also requires that
absolute gain in the first ranking coordinate. This is a persistence gate, not a
learning reward, selection change or evidence of generalization across seeds.
Unchanged groups carry their previously saved entries into the new collection.
The first collection establishes the baseline. Zero margin still requires strict
rank improvement and changed weights. Larger margins use each criterion's own
abstract primary-score units; one margin need not suit all criteria equally.

Champions are still considered every simulation step, including terminal learning
before dead controllers are released. K controls disk publication, not discovery.
A crash can lose up to one publication interval of new champion improvements.
Run-end archives continue to use all normal selection criteria and are not gated
by the periodic margin. Periodic exports do not modify chaining provenance.

## Configuration and commands

Add these optional fields under `experiment`:

```json
{
  "checkpoint_every_steps": 500,
  "checkpoint_keep_last": 2,
  "checkpoint_shared_metadata": false,
  "brain_save_every_steps": 250,
  "brain_keep_last": 2,
  "brain_save_min_score_delta": 0
}
```

`brain_keep_last` counts qualifying best collections; `brain_save_every_steps`
is the interval between checks. `checkpoint_keep_last` counts recovery files,
which serve a different purpose. A keep count of zero retains all new files;
a step interval of zero disables the corresponding periodic writes. Existing
`checkpoint_every_runs` still schedules run-boundary recovery writes.

```sh
.venv/bin/python -m wtf run tracking --runs 10 --seed 100 --progress \
  --checkpoint-every-steps 500 --checkpoint-keep-last 2 \
  --brain-save-every-steps 250 --brain-keep-last 2
```

CLI storage overrides also work with `--resume` and do not change the saved
simulation configuration. `--no-library` disables periodic and run-end champion
exports; it does not disable recovery checkpoints. These controls belong to the
CLI's disk writer; direct `Experiment.step()` calls do not perform disk I/O.

The latest periodic index is `champions/steps/latest.json`. Restore its saved
weights into a compatible fresh environment with:

```sh
.venv/bin/python -m wtf run tracking --runs 1 --seed 100 \
  --brains-from /path/to/experiment/champions/steps/latest.json
```

An explicit index cannot be combined with `--brain-run`. An experiment directory
still selects its latest numbered run by default. Best-brain warm starts reset
optimizer, buffers and RNG state; use a recovery checkpoint for exact continuation.

## Common metadata and portability

Optional shared binary checkpoints use header version 4 and `$meta` references to
`checkpoints/_metadata/<sha256>.json.xz`. Configuration, action order and model
signatures are immutable and shared across files. Their original mapping order
is preserved. Checksums, bounded decompression and path checks protect restoration.

Changing tensors, optimizers, rollout/replay buffers, RNGs, world state and
champion weights remain in each recovery checkpoint. They cannot be reconstructed
from weights alone. Existing within-file sharing still deduplicates model values.
Sharing static metadata is a separate, smaller saving than limiting checkpoint
history, especially with large neural networks or replay buffers.

Self-contained version-3 output remains the default: in the measured short
retention workload, pooling static metadata did not improve compressed disk
usage. Enable it explicitly with `checkpoint_shared_metadata: true` or
`--shared-checkpoint-metadata` when appropriate for your metadata workload.
See [the measurements](performance.md#rolling-artifacts-and-masked-ppo-batches).

Copy a shared checkpoint **with its `_metadata/` sibling directory**. Missing or
corrupt blobs fail explicitly. Keep counts never remove metadata needed for a
retained checkpoint. Use `checkpoint_shared_metadata: false` or
`--self-contained-checkpoints` for standalone version-3 binary output; direct
`Experiment.write_checkpoint()` also remains self-contained unless explicitly
passed `shared_metadata=True`. Older JSON, compact and version-3 binary files stay
readable. Older runtimes cannot read version-4 shared checkpoints. Existing
storage conversion can flatten a completed shared checkpoint to self-contained
output through the common reader; do not convert an active experiment.

Periodic best brains have their own `champions/steps/models/` pool. Only blobs
created by the current writer and no longer referenced by a retained periodic
collection are removed. Numbered archives and their separate weight pool are
never garbage-collected by this policy. A new invocation owns no older files.


## Save-time memory and repeated work

![One synchronous save shares bounded canonical model encodings between the improvement gate and exporter, then releases the cache](assets/storage/model-encoding-cache.svg)

The periodic gate and exporter share a bounded, save-local model encoding cache.
Criteria pointing at the same detached model reuse its canonical payload and
hash. The 4 MiB payload LRU is released after each check/publication; oversized
or evicted payloads use the normal exact encoder again. This does not change
rank thresholds, criteria, hashes, file formats or retained snapshots. See the
[before/after measurements](performance.md#shared-model-encoding-and-checkpoint-copy-reduction).

The synchronous recovery writer also avoids cloning retained metadata before
encoding it. Public `Experiment.checkpoint()` results remain detached, while
file writes consume an internal borrowed metadata view before another step can
run. Controllers are still detached and serialized one at a time. Every replay
chunk and numbered run archive remains retained.
