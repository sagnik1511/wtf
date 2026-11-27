# Archived observation layouts: V1–V6

[Historical reference](index.md) · [Current guide](../observations.md)

This page preserves earlier layouts and display contracts for old artifacts or
explicit comparison settings. It is not the current preset setup. Current guides
use V7 and the maintained feature pipeline; runtime readers still support these
older contracts. Historical measurements are not new evaluations.

## Visible target slots

![Visibility filtering, top-k target slots and fixed-width brain inputs](../assets/diagrams/top-k-observation-pipeline.svg)

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
`brains.<template>`. The enhanced [river preset](../../environments/river_crossing_strong_predator.json)
and its [deeper-network variant](../../environments/river_crossing_strong_predator_deep.json)
set k=4 for both species' templates.

Targets are ordered by distance within each category. Equal-distance targets
retain authoritative insertion order, matching the original nearest-only rule.
In numeric versions 1–5, each slot contains normalized relative x, relative y, distance, and presence
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

## Forward visual field

Set this on each species body to use front-facing vision:

```json
"field_of_view_degrees": 180
```

Supported values are **150**, **180** and **360**; omission preserves the original 360°
view. The 150° option uses a ±75° forward cone. The 180° setting keeps targets in the front half-plane relative to the
animal's current heading, including both side boundaries and its own cell.
Turning changes the field immediately, even without movement. Radius, wall
occlusion, night-time range and the directional water bonus still constrain
sight. The water bonus cannot reveal a target behind the animal.

This filters fresh food/prey/predator slots, visual mate/pair/consent cues and
local witnessed outcomes. Top-k remains a maximum per category; missing targets
are zero-padded. Target coordinates remain world-axis offsets in versions 1–4; V5 uses
forward/right offsets. Input-vector
lengths and policy signatures do not change, although perceived states do.
Personal movement memory can briefly retain an older rear sighting with fading
confidence; it never receives unseen new movement. Learned species knowledge is
also retained. Scent remains omnidirectional within its smell radius.

Own body state, immediate adjacent collision/terrain affordances and physical
action eligibility are not the visual target sensor. They retain their existing
rules, including backward movement and proximal mating. A physical mate behind
an animal is omitted from fresh visual mate cues; this does not enlarge or
shrink the reproduction radius or bypass mutual consent.

The selected-agent board overlay shows the current forward sector. The inspector
radar shades the configured sector using the **recorded sample's heading**, which may
lag the current board. Both follow the viewer theme. Old recordings without the
setting retain their circular overlay. [Appearance examples](../viewer-visual-reference.md#forward-vision).

`land_bridge_collective` enables 180° for both species. The earlier bridge
presets retain 360°; compare with matching vision settings when attributing a
learning effect. Existing compatible policy weights can transfer between fields,
but they must be evaluated under the new perception. The shared consequence
predictor's saved physical contract includes the species body, so 360° knowledge
exports are not silently imported into a different 150°/180° body contract.

The `land_bridge_collective_150` preset keeps the collective bridge setup with
`field_of_view_degrees: 150` for both species; use it to compare a narrower
forward cone with `land_bridge_collective` (180°). Other settings remain the same.

`seaside_150` applies 150° to both species and enables visible social outcomes,
both general and context consequence inputs, durable species/action totals and
private auxiliary consequence heads (coefficient `0.1`). Evidence is published
between episodes. It retains the ordinary `seaside` map, body/reproduction
settings and PPO update settings; its observation/model contract is larger.
Start fresh rather than loading weights from the earlier plain `seaside_150`
configuration. Brains remain independent. See [collective learning](../social-learning.md).

```sh
.venv/bin/python -m wtf run seaside_150 --runs 5 --seed 100 --progress
```

## Input layout

![Contiguous self, target, terrain and mating blocks with four-field target slots](../assets/structures/observation-vector-layout.svg)

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
| 5: body-relative spatial values + immediate opportunities | 48 | 84 |

The general formula is the original version width plus `12 * (k - 1)`.
Changing k changes the input layout and required model signature. Start fresh,
or load a model with the same algorithm, architecture, actions, version and k.
Old models without a top-k field keep their original k=1 signature; explicitly
setting k=1 is compatible too. Exact checkpoints retain their saved setting.

## Observation version 5

Set `brains.<template>.observation_version: 5` to opt in. Versions 1–4 retain
exactly their existing widths and world-coordinate meanings. V5 keeps the same
self, top-k target, terrain and mating blocks as V4, projects directional values
into the animal's current forward/right basis, and inserts three opportunity
values **after consent and before optional tracking/social inputs**.

![V4 and V5 input blocks and the corresponding inspector radar](../assets/viewer/observation-v5.svg)

For heading `(hx, hy)` and a normalized world vector `(dx, dy)`:

```text
forward = dx * hx + dy * hy
right   = -dx * hy + dy * hx
```

Negative right means left; negative forward means behind. Heading x/y remain
world-relative self information. Target/mate distances, presence, normalization
radii, top-k ordering, sight limits, physical rules and consent are unchanged.

| Block | V5 encoding |
| --- | --- |
| Food, prey, predator and mate offsets | Forward/right relative to the observer |
| River current | Forward/right components, with the existing flow scale |
| Personal visual memory | Last position, velocity and each remembered position rotate using the observer's current heading; stored memory stays in world coordinates |
| Ground scent | Same anonymous channels and strengths, reordered as here/front/front-right/right/back-right/back/back-left/left/front-left |
| Social and consequence estimates | Same scalar meanings and configured widths |

| New input | Meaning at the decision snapshot |
| --- | --- |
| Attack possible now | Attack is allowed, the observer is a predator with positive attack damage, and prey occupies the cell directly ahead |
| Eat possible now | Eat is allowed, the animal has energy capacity and positive bite/food conversion, and a nonempty resource is here or directly ahead |
| Swim possible now | Swim is allowed and the front transition passes the existing occupancy, terrain, health and effort gates |

Unsupported actions produce zero. These are sensed immediate affordances,
**not masks**: ordinary PPO still assigns probability to every allowed action.
They do not predict another animal's simultaneous move, reserve a resource or
guarantee the eventual outcome. Existing pair-opportunity and consent inputs
continue to describe reproduction; V5 does not add duplicate mating flags.

With k=10, the base width changes from V4's 153 to **156**, before optional
tracking/social blocks. Architectures and top-k are unchanged. Old V4 weights
cannot be silently relabeled V5; create fresh compatible brains. Exact checkpoints
restore their saved version. The [compatibility policy](../compatibility.md) and
[viewer reference](../viewer-visual-reference.md#observation-v5) cover readers.

`seaside_150_v5` is the retained V5 comparison control. The common
`seaside_150` preset now uses V7 plus history and scent. Those two files no
longer differ only in representation; use the comparison helper below to derive
older version arms from one common configuration with other settings held fixed.

```sh
.venv/bin/python -m wtf run seaside_150_v5 --runs 5 --seed 100 --progress
```

To derive a controlled V4-versus-V5 comparison without large artifacts:

```sh
.venv/bin/python -m work.compare_observation_versions \
  --environment environments/seaside_150.json \
  --seeds 100 200 300 --episodes 3 --steps 1500 \
  --evaluation-seeds 900 901 --progress \
  --output .cache/v4-v5-comparison.json
```

Each version starts fresh with the same seeds and all other configuration values.
The helper verifies common configuration and initial world hashes, trains without
recording, and freezes same-species/sex selected champions on shared holdout seeds.
It reports greedy and seeded stochastic evaluation separately and asserts that
weights and consequence evidence remain frozen. Births in frozen evaluation copy
same-sex parent weights without crossover or mutation. The report retains compact
summaries, not model/replay files. Input widths change initialization shapes, so
matched seeds do not imply identical initial weights, trajectories or work counts.
Repeated runs and both species changing prevent treating individual births/kills
as proof of a general improvement. V5 remains opt-in; its representation is not
a survival or reproduction guarantee. Keep experiment reports local.

## Observation version 6

V6 is an opt-in **soft semantic basis** over the already sensed V5 vector. It
keeps each top-k entity slot and independent ordinary PPO brains. It changes
what the network receives; it does not change actions, physics, proximal mating,
mutual consent or the publication schedule of consequence knowledge.

### What changed from V5 to V6

![The same sensed data represented as earlier V5 numbers and current V6 soft categories](../assets/viewer/observation-v5-v6.svg)

V5 supplied normalized numbers; V6 expands selected numbers into overlapping
category memberships, while exact flags and top-k slot identities remain intact.
The comparison shows the same scene on both sides, plus all optional block
widths. This is a change in representation, not additional sensory information
or a smaller byte format. In-range scalar memberships retain reconstructable
values; saturation outside the configured endpoints remains explicit.

The later V6 allocation/training optimization changes **none of these inputs**:
values, order, widths and recorded playback are identical before and after that
optimization. Only repeated computation and allocation are reduced.

### Reading the memberships

![V6 range, bearing, energy and health memberships with the actual inspector decoder](../assets/viewer/observation-v6.svg)

Instead of a single normalized distance, a slot carries near/medium/far
memberships. Directions use eight circular body-relative sectors. Adjacent
categories overlap linearly: a distance of `0.09` with knots `[0, 0.3, 1]` is
`near=0.7, medium=0.3, far=0`. Bearings interpolate between adjacent sectors across
the front-left/front wrap too. A zero vector has no bearing; a presence flag
keeps an animal/resource in the observer's own cell distinct from an absent slot.

| Sensed field | Semantic channels |
| --- | --- |
| Energy | Critical, hungry, comfortable, full |
| Health | Critical, weak, healthy |
| Each target/mate position | Near/medium/far; eight forward-relative bearing memberships; exact presence |
| Current and remembered velocity | Still/gentle/strong speed and eight bearings |
| Local/adjacent water | Dry/shallow/deep |
| Elevation | Low/middle/high |
| Adjacent climb | Steep-down/gentle-down/level/gentle-up/steep-up |
| Maturity, cooldown, consent duration, memory age/confidence, scent strength and evidence confidence | None/partial/strong, interpreted in the named field's existing normalized scale |
| Harm/feeding/birth tendency | Zero/very-low/low/moderate/high, with extra knots near zero to preserve rare-outcome distinctions |

Sex, world heading, collision indicators, presence/seen flags, readiness,
pair opportunity, consenting-pair opportunity and attack/eat/swim opportunities
retain their exact existing values. No new hidden positions, identities, rewards
or future outcomes enter the vector. Unknown evidence has confidence `none=1`;
that remains distinct from a well-supported low outcome rate.

### V6 input layout

![Ordered V6 input blocks and one expanded top-k target slot](../assets/viewer/observation-v6-layout.svg)

The input is one ordered flat vector. Each target keeps its own top-k slot;
range/bearing memberships are concatenated with its exact presence flag. Core
blocks precede optional movement history, scent, local visible outcomes, general
consequences and context consequences, in that order. Disabled blocks contribute
no channels. The figure labels actual zero-based ranges for the maintained
`seaside_150_v6` prey template and checks those boundaries against the current
decoder. Its predator template uses a different k and therefore different offsets.
The zoomed values are illustrative; the blocks are not drawn to scale.

### Soft-category configuration

Set `observation_version: 6` on selected brain templates. Optional
`soft_categories` overrides membership centers on **normalized V5 scales**, not
absolute cell distances or physical thresholds:

```json
{
  "observation_version": 6,
  "soft_categories": {
    "distance": [0, 0.2, 1],
    "energy": [0, 0.2, 0.7, 1]
  }
}
```

Every list has a fixed length, strictly increasing finite values and fixed
endpoints. Omitted groups use the defaults below. Category names stay fixed.
Changing the centers changes the saved-model signature and requires matching
weights. Physical health, swimming and reproduction thresholds remain in their
existing species settings; soft labels never replace eligibility gates.

| Group | Default centers |
| --- | --- |
| distance | `0, 0.3, 1` |
| energy | `0, 0.25, 0.75, 1` |
| health / level | `0, 0.5, 1` |
| depth / elevation | `0, 0.1, 1` |
| climb | `-1, -0.1, 0, 0.1, 1` |
| speed | `0, 0.25, sqrt(2)` |
| outcome | `0, 0.001, 0.01, 0.1, 1` |

Values outside a group's represented range **saturate at its end category**.
This is particularly relevant to remembered positions after movement or range
changes. Scalar values inside the range can be reconstructed from their linear
memberships; this is not hard quantization. Category-derived inspector positions
are representative values, especially after saturation. Exact world replay
positions, event data, physics and V1–V5 observations remain unchanged.

### Width, storage and private auxiliary learning

V6 makes information more semantic, **not necessarily smaller**. Its base width
is `94 + 36*k`: 130 at k=1, 202 at k=3 and 454 at k=10. Optional blocks add:

- Personal history: `2*k*(29 + 12*H)` values when H is enabled.
- Scent: 27 values per configured anonymous channel.
- Local social observations: 108 values.
- Consequences: 216 values per scope; `both` adds 432.

The maintained `seaside_150_v6` preset therefore has **742 prey / 994 predator
inputs**, versus V5's 192/276 with the same optional settings. The larger first
layer and rollout vectors can cost more CPU, RAM and replay space. V6 templates
are limited to 4096 encoded inputs; reduce top-k/history/channels if needed.
Compiled structural layouts use a bounded 128-entry cache with no animal state.
Bounded immutable coefficient caches reuse repeated scalar/bearing values and
auxiliary reconstruction; fresh observation containers preserve independent
ownership. Unknown consequence samples avoid zero-loss auxiliary decoding.
Exact-value contracts and measured CPU/memory results are in
[V6 performance](../performance.md#v6-semantic-encoding-and-auxiliary-work).

The private auxiliary head keeps its 36 physical-outcome outputs. It decodes
published consequence targets from V6 memberships and zeros **every encoded
consequence channel** on its auxiliary branch. Independent ownership, the PPO
KL gate and on-policy rollouts are unchanged.

```sh
.venv/bin/python -m wtf run seaside_150_v6 --runs 5 --seed 100 --progress
```

This preset changes only the V5 preset's observation version and descriptive
identity. It retains ordinary PPO and the same auxiliary/consequence settings;
V5 and all earlier presets stay available. Start with fresh compatible brains.

For a bounded comparison with compact local summaries:

```sh
.venv/bin/python -m work.compare_observation_versions \
  --versions 5 6 --environment environments/seaside_150_v5.json \
  --seeds 100 200 300 --episodes 3 --steps 1000 \
  --evaluation-seeds 900 901 --progress \
  --output .cache/v5-v6-comparison.json
```

The helper retains V4/V5 defaults unless `--versions` is supplied. V6-only knot
settings are removed from older arms and recorded separately as treatment
metadata; they are excluded from the common-configuration hash. Both arms use
the same remaining configuration and starting world seeds, with fresh compatible
weights. Frozen greedy and stochastic holdouts are reported separately; no
replays or model archives are written. Input shapes and initial parameters differ;
matched seeds do not establish identical trajectories or a runtime speedup.
Keep study outcomes in local reports, and evaluate ecological benefit before
promoting this opt-in representation.

## Optional movement memory and scent

[Movement history and scent trails](../tracking.md) append personal sighting and
blended anonymous scent or legacy species/sex ground-scent inputs after the existing versioned blocks.
Both default off. The base widths above remain unchanged; enabled movement adds
`2 * K * (7 + 3 * H)` values and scent adds nine values per configured channel.
Their layout fields participate in model compatibility. The inspector displays
sampled tracks and a blended heatmap (or legacy local scent grids) without storing complete world scent maps.
