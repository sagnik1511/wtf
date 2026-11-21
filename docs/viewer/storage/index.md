# Stored artifacts

[Documentation home](../../index.md) · [Complete directory](../../directory.md)

A recording reproduces past frames, a champion supplies learned parameters to a
new controller, and a checkpoint restores a complete experiment state. Keep the
files required for the operation you intend to perform.

| Task | Guide |
| --- | --- |
| [Replay files and data structures](../../replay-data-structures.md) | Artifact map, manifests, chunks, frame deltas and champion weight records. |
| [Checkpoint state and restoration](../../checkpoint-data-structures.md) | World state, optimizer, RNG, buffers and restore/access paths. |
| [Binary formats and compression](../../binary-storage.md) | Typed arrays, compression, deduplication and legacy encodings. |
| [Load or select champions](../../brain-library.md) | Archive criteria, species/sex groups and CLI initialization. |
| [Check format compatibility](../../compatibility.md) | Package/runtime support and restore guarantees. |
| [Change recording detail](../../recording.md#recording-settings) | Sampling cadence and stored diagnostic information. |

To use these artifacts visually, return to [Playback and inspection](../playback/index.md).
