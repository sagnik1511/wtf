# Selected runtime structure illustrations

[Visual guide](../../visual-guide.md#when-a-visual-helps) · [Asset index](../README.md)

| Maintained SVG source | Guide | Structure explained |
| --- | --- | --- |
| [Package boundaries](package-layout.svg) | [Package layout](../../package-layout.md) | Root API and six responsibility-based subpackages, separate from JavaScript and writable artifacts |
| [Observation vector](observation-vector-layout.svg) | [Input layout](../../observations.md#input-layout) | Contiguous offsets and four-field slots, with optional appended sections |
| [Founder families](founder-family-membership.svg) | [Lineage credit](../../brain-library.md#founder-membership-and-family-credit) | Parent relationships, set union and separate overlapping ledgers |
| [Actual animal activation replay](animal-activation-replay.svg) | [Recorded animal activity](../../explainability.md#recorded-animal-activations) | Current animal state, compact diagnostic arrays and selected-only replay display |
| [Archived brain analysis](brain-explanation-flow.svg) | [Explanation data flow](../../explainability.md#what-is-being-explained) | Selected diagnostic and champion weights remain distinct from recorded prediction scores |
| [Activation example](brain-activation-example.svg) | [Worked network](../../explainability.md#worked-activation-example) | Weighted sums, biases, tanh, softmax and reverse-mode input sensitivity |
| [Spatial window](spatial-window-cache.svg) | [Cache structure](../../performance.md#spatial-window-structure) | Reused bucket references, revision keys and cache lifetime |

These original SVGs are directly editable source assets with accessible titles
and descriptions. They require no external fonts/images, scripts or Mermaid.
Raster previews and local drafting helpers remain under ignored `.cache/`.

The vector layout follows `World.observe` and `observation_inputs`: v3/k=4 has
78 fields, with terrain at 56–67 and mating at 68–77. The diagram expands a target
slot's schema; it is not a captured animal observation.

The pedigree is a controlled same-species example of repeated ancestry. Actual
`LineageBook` checks confirmed roots `{A,B}`, generation 2, four members in each
family and +2 kills per family when E gains two kills. Reobserving does not add
that credit again, and the candidate rank uses one family rather than a sum.
Physical mating gates and selection qualification remain separate rules.

The bucket drawing is schematic, not a visibility map or measured cache trace.
A controlled `SpatialIndex` check verified shared object identity in an overlap,
last-query reuse, and invalidation when a target moves while its observer stays
still. Existing observation tests cover target changes, ties, births, deaths,
food changes, walls and small/wide scan paths. No new cache was implemented.

These examples explain existing structures; they do not measure performance or
establish ecological learning. Keep numerical comparisons in their guide tables
rather than producing figures without a new structural question.

The archived-brain flow describes the selected-only `ExplanationStore` data path
and the separate historical prediction reference. It does not depict exact
historical weight recovery or a live simulation connection.

The activation example is a hand-authored mathematical slice with two active
input columns and two tanh units. Its weights/biases are illustrative, not learned
parameters. Production `prepare_model`/`explain_model` verified its forward scores
and input gradients; central finite differences agree within 1e-9. The drawing
omits inactive input columns and the zero-sensitivity value head. All numerical
labels are rounded for readability. No experiment artifact is needed to edit or
understand these directly maintained SVG sources.
