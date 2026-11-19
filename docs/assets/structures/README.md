# Selected runtime structure illustrations

[Visual guide](../../visual-guide.md#when-a-visual-helps) · [Asset index](../README.md)

| Maintained SVG source | Guide | Structure explained |
| --- | --- | --- |
| [Observation vector](observation-vector-layout.svg) | [Input layout](../../observations.md#input-layout) | Contiguous offsets and four-field slots, with optional appended sections |
| [Founder families](founder-family-membership.svg) | [Lineage credit](../../brain-library.md#founder-membership-and-family-credit) | Parent relationships, set union and separate overlapping ledgers |
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
