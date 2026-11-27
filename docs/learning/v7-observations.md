# V7 observations: layout and examples

[Sensing handbook](features.md) · [Version reference](../observations.md#observation-version-7) · [Memory and scent](memory-and-scent.md)

V7 is a numerical encoding of sensed information. It removes redundant distance
fields, summarizes observed motion and compresses each scent channel to a bearing
and two scalar values. It is not a smaller neural network architecture, a new RL
algorithm or a categorical action rule.

## Current representation

Visible targets use forward/right coordinates with explicit presence. Motion uses
bounded personally observed summaries; scent uses a bearing, strength and
freshness. Consequence blocks follow the configured action contract rather than
allocating permanently unavailable actions. The blocks below describe the current
layout in full.

Earlier representations and their field comparisons are in the
[historical observation reference](../archive/observation-layouts.md). They remain
loadable under their own metadata; their weights are not interchangeable with V7.

## Vector blocks and width

The following order is exact. Offsets depend on the template and do not change
when the population grows, an animal dies or fewer targets are visible.

| Ordered block | V7 width | Interpretation |
| --- | ---: | --- |
| Own state | 8 | Energy/health fractions; heading x/y; blocked front/right/back/left |
| Food, prey-role, predator-role slots | `9k` | Each category has k slots of forward/right/presence |
| Terrain | 12 | Current depth/flow/elevation and four adjacent water/climb pairs; flat worlds are padded |
| Mating and consent | 12 | Sex, maturity, cooldown, own readiness; mate position/presence/readiness; pair and consent cues |
| Immediate opportunities | `O` | Attack/eat/swim affordances only when the action is configured |
| Visual history, if enabled | `14k` | k prey-role and k predator-role tracks, seven values each |
| Scent, if enabled | `4C` | One four-value summary per configured scent channel |
| Local witnesses, if enabled | 24 | Six action groups × confidence/harm/feeding/birth |
| General consequences, if enabled | `4A` | One row per configured action |
| Context consequences, if enabled | `4A` | One row per configured action |

Here k is `observation_top_k`, A is the ordered action count, O is the number of
configured actions among attack/eat/swim, and C is the scent-channel count.
Opportunity fields retain attack/eat/swim order after filtering; consequence
rows instead follow `observation_actions`. Blended scent has C=1.
Scope `context` or `species_action` adds one consequence
block; `both` adds general first, then context.

For enabled visual history H>0, the numerical width is:

```text
32 + 9k + O + 14k + 4C + optional local/evidence blocks
```

If H=0, omit `14k`. If scent is absent, C=0. Changing H from 3 to 6 leaves the
V7 vector width unchanged but changes its meaning and model signature.
A larger window still increases the internal history and computation.

## Worked offsets: the ordinary PPO preset

The current flat `ppo` preset has no local/shared consequence inputs or optional
outcome heads. Its prey have k=3 and nine actions, including eat but not attack
or swim. Its predators have k=10 and six actions, including attack but not eat
or swim. Both have H=3 and one blended scent channel.

| Block | Prey offsets `[start, end)` | Predator offsets `[start, end)` |
| --- | --- | --- |
| Own state | `[0, 8)` | `[0, 8)` |
| Three target categories | `[8, 35)` | `[8, 98)` |
| Terrain | `[35, 47)` | `[98, 110)` |
| Mating/consent | `[47, 59)` | `[110, 122)` |
| Opportunity | `[59, 60)` | `[122, 123)` |
| Movement summaries | `[60, 102)` | `[123, 263)` |
| Blended scent | `[102, 106)` | `[263, 267)` |
| Total | **106** | **267** |

Adding features can increase inputs over an older V1/V4 preset even though V7
is compact relative to the richer representation. Do not compare 106 with an
old 20-input model and infer smaller training/storage costs.

The separate `seaside_150_v7` reference includes swim, local witnesses, both
consequence scopes, auxiliary prediction and late placement. Its **recorded**
widths are 211 prey / 348 predator, while its physical trunks consume only
107 / 268. [Late routing](consequence-pipeline.md#late-consequence-adjustment)
explains why recorded inputs and trunk width can differ.

## Reading a target slot

These values are illustrative, not an experiment result:

```text
forward = +0.50, right = -0.25, present = 1
```

The target is ahead and to the animal's left in its current heading frame.
Coordinates are normalized by the observation range used for the sight query;
they are not world-grid coordinates. A target behind the animal can have a
negative forward component when the configured visual field permits it.

The corresponding absent slot is `[0, 0, 0]`. Presence prevents confusing an
absent target with a real animal/patch at zero relative displacement. A distance
can be reconstructed from coordinates, with possible last-bit rounding
differences from the old independently calculated distance field.

Visible target slots are sorted by distance within diet-based categories, not
by configured species name. A two-prey-species world does not get an independent
k budget for each named prey species. Scent and consequence grouping have their
own rules; see [ownership](features.md#what-is-independent-and-what-is-shared).

## Why the action list belongs to the input contract

V7 evidence rows and opportunity fields depend on the exact ordered
`allowed_actions`. Validation resolves and stores `observation_actions` on each
brain template. Reordering actions can change model semantics even when the
width is equal.

Species may share a template only when their action lists have the same order.
If prey can eat while predators can attack, use two configuration entries even
when architecture and optimizer settings are identical. Those entries are
settings, not species-wide neural weights: every animal still gets a private
controller. Never edit saved model metadata to force a contract match.

## Inspect the exact names in Python

This code prints the implementation's resolved order without training or
creating result artifacts:

```python
from wtf import load_config
from wtf.inspection.explainability import feature_names

config = load_config("ppo")
for species, settings in config["species"].items():
    template = config["brains"][settings["brain"]]
    names = feature_names(template)
    print(f"{species}: {len(names)} inputs")
    for index, name in enumerate(names):
        print(f"  {index:3d}: {name}")
```

For viewer behavior, use the [V7 inspector specimen](../viewer-visual-reference.md#compact-v7-and-late-consequence-adjustment).
For a runnable configuration, continue to the [tutorial](../tutorials/06-sensing-and-learning.md).
