# Historical feature contracts and settings

[Documentation home](../index.md) · [Current sensing handbook](../learning/features.md) · [Release history](../releases/README.md)

Primary guides describe the current source behavior and maintained presets.
This area preserves superseded layouts and illustrations only for old artifacts,
explicit comparisons or historical questions. It is not the default setup.

| Historical material | Current replacement |
| --- | --- |
| [V1–V6 observation layouts](observation-layouts.md) | [V7 inputs and exact offsets](../learning/v7-observations.md) |
| [Old sensory inspector sheets](sensory-viewer.md) | [Current viewer guide](../viewer-visual-reference.md) |
| [Published-package installation snapshot](published-installation.md) | [Current source installation](../../INSTALL.md#current-source-quick-start) |
| [Earlier feedback audit](feedback-audit.md) | [Current feedback contracts](../feedback-review.md) |
| [V3 mating inputs and earlier river setup](mating-inputs-v3.md) | [Current mating inputs](../mating-observations.md) |

## Archived settings without changing saved artifacts

`baseline` remains a minimal V1 control; `champion_lineage` is an archived-weight
initialization example. The explicitly versioned `seaside_150_v5` and
`seaside_150_v6` JSONs remain available as comparison controls. The V7 reference
is current but separately enables optional late/auxiliary learning heads.
Keeping these names resolvable avoids breaking saved workflows; primary source
examples use current V7 presets instead.

For other historical settings, use the original run's captured `environment.json`
and compatible source/interpreter, or retrieve the relevant configuration from
its implementation commit. Exact checkpoint resume already restores its captured
configuration. Do not alter model metadata to force old weights into current
inputs. See [artifact compatibility](../compatibility.md) and
[rollout records](../releases/README.md) for reader/version boundaries.

Historical benchmark results retain the measured workload and source revision.
They are not fresh measurements of the current presets. Biological observations
remain in the local paper/results workspace, not this package reference.
