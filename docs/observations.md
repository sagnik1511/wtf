# Animal observations

[Configuration](configuration.md) · [Geography](geography.md) · [Saved brains](brain-library.md)

Each living animal receives a numerical observation before choosing its own
next action. The sight area determines which targets it can detect. The brain
does not receive an image or control the other animals.

## Visible target slots

![Visibility filtering, top-k target slots and fixed-width brain inputs](assets/diagrams/top-k-observation-pipeline.svg)

Set `observation_top_k` on a brain template to choose how many visible targets
are retained **per category**: food patches, resource-eating animals and
prey-eating animals. Categories follow diet, not configured species names.
The setting accepts integers 1–16 and defaults to 1 for old configurations.
Prey and predators can reference different templates with different k values.

```json
"observation_version": 2,
"observation_top_k": 4
```

These fields belong alongside `type`, `network` and training settings inside
`brains.<template>`. The enhanced [river preset](../environments/river_crossing_strong_predator.json)
and its [deeper-network variant](../environments/river_crossing_strong_predator_deep.json)
set k=4 for both species' templates.

Targets are ordered by distance within each category. Equal-distance targets
retain authoritative insertion order, matching the original nearest-only rule.
Each slot contains normalized relative x, relative y, distance, and presence
(1 for a target, 0 for padding). Empty slots contain four zeros. A food patch at
the animal's own cell has distance zero and presence one. An animal excludes
itself from animal targets. Food must exceed the existing presence threshold.

The first k **visible** targets are selected: a hidden near target does not
consume a slot. Sight radius, directional water extension and wall occlusion
still apply. Target coordinates/distances are normalized by the query radius
(base sight plus configured water bonus on a wet map), as with k=1. Positions
use world x/y axes; nearby blocked directions are relative to heading.

This exposes several possible targets instead of only the closest. It still
limits information to k per category and does not expose target health, target
brain state, unique identity or a complete occupancy grid. Slot identities can
change as distance ordering changes. Brains choose existing actions such as
turn/move/swim/attack; an observation slot is not a direct attack-target command.

## Input layout

![Contiguous self, target, terrain and mating blocks with four-field target slots](assets/structures/observation-vector-layout.svg)

The illustrated version-3, k=4 vector has terrain at indexes 56–67 and mating
at 68–77. With k=1 those blocks start at 20 and 32. More target slots shift the
following blocks; they do not duplicate terrain or mating inputs.

| Section | Number of values |
| --- | ---: |
| Own energy, health and heading vector | 4 |
| Blocked walking ahead, right, behind and left | 4 |
| k food slots, k resource-eater slots, k prey-eater slots | 12 × k |
| Version 2/3: current depth, flow x/y, elevation; four adjacent depth/elevation differences | 12 |
| Version 3: sex, maturity, cooldown, readiness and nearest eligible-category mate/pair information | 10 |

Mating inputs keep their existing single nearest-visible-mate summary, separate
from the k general animal slots. Version 2/3 terrain inputs are zero-padded on
flat maps. General target slots do not distinguish sex or exact species.

| Observation version | Default k=1 | k=4 |
| --- | ---: | ---: |
| 1: base | 20 | 56 |
| 2: terrain | 32 | 68 |
| 3: terrain + mating | 42 | 78 |
| 4: mating + visible consent | 45 | 81 |

The general formula is the original version width plus `12 * (k - 1)`.
Changing k changes the input layout and required model signature. Start fresh,
or load a model with the same algorithm, architecture, actions, version and k.
Old models without a top-k field keep their original k=1 signature; explicitly
setting k=1 is compatible too. Exact checkpoints retain their saved setting.

## Optional movement memory and scent

[Movement history and scent trails](tracking.md) append personal sighting and
blended anonymous scent or legacy species/sex ground-scent inputs after the existing versioned blocks.
Both default off. The base widths above remain unchanged; enabled movement adds
`2 * K * (7 + 3 * H)` values and scent adds nine values per configured channel.
Their layout fields participate in model compatibility. The inspector displays
sampled tracks and a blended heatmap (or legacy local scent grids) without storing complete world scent maps.

## Runtime and playback

Multi-target selection reuses the world's spatial buckets and visibility cache.
Candidates inside the maximum radius are distance-sorted; sight checks stop
when k visible targets have been found. The original native nearest-target path
remains available for k=1; the optional [native top-k selector](native-top-k.md)
fuses multi-target ranking and sight filtering for recognized world callbacks.
Final feature arithmetic remains Python, and custom callbacks or older/missing
extensions use the reference selector.

One result per observer and category is retained, keyed by position, target
bucket revisions, radius, k and directional range settings. Moves, births,
deaths, food-presence changes, obstacle edits and visibility implementation
changes invalidate relevant results. Derived caches are world-local, omitted
from checkpoints and rebuilt on restore. They are not shared mutably by GRPO
trial worlds. Higher k increases inputs, training memory and observation work;
workload-specific measurements are in the native selector guide.

Replays still store world frames/events/diagnostics in the existing binary chunk
format. They do not store a separate observation vector for every animal per
frame. Playback and chart interfaces do not need a format change.

The [visual inspector](recording.md#visual-agent-inspector) draws current sight on
the board and the recorded input slots in a radar, alongside probability/Q bars
and bounded prediction-sample history. The observation and prediction timestamps
are shown separately from the current body/frame.

## Visible consent

[Version 4](escape-and-consent.md#observation-version-4) appends own consent,
nearest visible mate consent and a proximal consenting opportunity flag. It
preserves versions 1–3 and uses a distinct saved-model signature.
