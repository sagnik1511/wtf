# Physics asset catalog

[Illustrated guide](../../physics.md) · [Asset index](../README.md)

These small, intentional documentation assets are tracked. They are controlled
physics examples generated from the repository, not training recordings or
learned-model artifacts. SVGs are vector images with light/dark styles and need
no raster-image dependency. No third-party network scripts are loaded by the HTML.

| Asset | Cases |
| --- | --- |
| [explorer.html](explorer.html) | Standalone interactive explorer; both species, all views/actions/steps |
| [explorer.fragment.html](explorer.fragment.html) | Editable explorer source with small embedded scenario snapshots |
| [valley-layout.svg](valley-layout.svg) | Actual seed-100 founder positions, gate geometry and food placement |
| [visibility-water.svg](visibility-water.svg) | Three-cell river before/after water credit |
| [visibility-limits.svg](visibility-limits.svg) | Dry range, walls, high terrain without sight occlusion |
| [visibility-species.svg](visibility-species.svg) | Different prey/predator sight limits |
| [visibility-valley.svg](visibility-valley.svg) | Actual left-gate crop; one-cell river gives +1 |
| [swimming.svg](swimming.svg) | Blocked walking, successful swimming, weak entry failure |
| [currents.svg](currents.svg) | Fractional current, post-action drift and dry landing |
| [drowning.svg](drowning.svg) | Resting weak animal: health loss and death |
| [elevation.svg](elevation.svg) | +1 climb, blocked +2 cliff, downhill |
| [reproduction.svg](reproduction.svg) | Mutual nearby birth, missing choice and out-of-range failure |

## Regenerate

From the repository root, using the existing environment:

```sh
.venv/bin/python -m work.build_physics_docs
node work/check_physics_docs.cjs
```

The builder runs explicit actions through `World.step`; no neural policy is
trained. It refreshes the embedded snapshots, all ten SVGs and the matching
fragment inside the standalone wrapper. It reads current enclosed-valley body,
locomotion, reproduction and map settings. Review captions and expected outcomes
when changing those settings; some example thresholds/positions are intentional.
Generated figures are portable and contain no personal filesystem paths.

The standalone wrapper was exported with the visualize skill's render utility;
its unused optional tooltip/icon CDN imports were removed because this explorer
uses native controls and custom data marks. Exported host access is limited to
widget state storage, with the wrapper's local fallback. Keep source and export
in sync; the builder checks that the previous fragment exists in the wrapper
before updating it. A structural wrapper change requires a fresh export.

The Node check exercises visibility toggles, the actual valley crop, blocked
walking, swimming, cliffs and mutual/in-range reproduction, and catches missing
control IDs. It is a logic smoke check, not a replacement for browser visual QA.
