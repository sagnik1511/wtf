# Playback, model and checkpoint structure illustrations

[Data-structure guide](../../replay-data-structures.md) · [Checkpoint structures](../../checkpoint-data-structures.md) · [Asset index](../README.md)

| SVG source | Purpose |
| --- | --- |
| [Experiment artifacts](experiment-artifacts.svg) | Connected directory/file blocks, with dashed references to pooled champion dependencies |
| [Binary chunk layout](binary-chunk-layout.svg) | Nested prefix/length/payload blocks and physical entity-table columns |
| [Model-weight structure](model-weight-storage.svg) | Shared artifact references, one-record binary blob, signature/precision and MLP layer arrays |
| [Checkpoint file layout](checkpoint-record-layout.svg) | Borrowed synchronous metadata versus detached public snapshots; physical records and optional v4 metadata references |
| [Model encoding cache](model-encoding-cache.svg) | Save-local snapshot identities, bounded payload/hash reuse, gate/export consumers and cache release |
| [Private controller state](checkpoint-controller-state.svg) | Nested network, optimizer, experience, RNG and algorithm-specific state blocks |
| [Restore and runtime access](checkpoint-restore-access.svg) | Decode/restore stages, object ownership and decision/learning consumers |
| [Frame delta example](frame-delta-example.svg) | Before/after field cells, highlighted mask bits, update-tuple blocks and reconstruction |

These are original, directly editable SVG sources with accessible titles and
descriptions. They do not require Mermaid, JavaScript, remote fonts or raster
images. They are schematic and not drawn to byte scale. Raster previews used for
review belong under ignored `.cache/docs-viz/storage/`.

The paths and fields were checked against the current recorder, delta codec,
binary codec, world snapshots and viewer. The mask-20 example was passed through
`DeltaEncoder` and decoded to recover the exact next frame. A binary header,
keyframe, delta and footer were round-tripped through `RecordWriter`, XZ and
`records`; the four-row column example was checked for tag 11 and exact decode.
These are codec examples, not measurements or captured ecology trajectories.

The SVGs themselves are the maintained source. When the storage schema changes,
update their labels and the guide together, repeat those codec checks and review
raster previews for readable labels and clipping. No recorded files are copied
into these assets.

The model diagram follows `model_snapshot`, `write_model`/`read_model` and the PPO
MLP implementation. Its example shapes (68 → 16 → 8, 1,240 scalars) were checked
against a constructed controller without training. Two writes reused one blob;
SHA-256, exact readback and loading into a fresh controller were verified. Other
algorithms' parameter trees differ, as the guide explains.

Checkpoint diagrams follow the I/O writer/reader, `Experiment.restore`, world
restore and each algorithm's checkpoint/restore methods. A controlled Q/PPO check
verified record order, shared-value detachment, nonempty buffers, nonzero Adam
state, restored RNGs and identical next steps. The existing binary continuation
test passed across all nine algorithms with XZ and gzip. The diagrams describe
state ownership/access, not a changed simulation execution order.
