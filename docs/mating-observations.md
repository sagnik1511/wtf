# Mating readiness and guarded inheritance

[Saved brains](brain-library.md) · [Rewards](rewards.md) · [Geography](geography.md)

The meandering-river preset now exposes mating opportunities and uses longer,
more frequent predator GRPO trials. Reward coefficients and physical mating
rules are unchanged. These changes have not yet been validated as an ecological
improvement in a long experiment.

## Observation version 3

Set `observation_version: 3` on any algorithm's brain template. All nine
controllers support it on flat and terrain maps. With default k=1, its **42
inputs** contain 20 local inputs, 12 terrain inputs (zero on flat maps) and ten
mating values. Each extra target slot adds 12 inputs: k=4 has 78. The
[vector block diagram](observations.md#input-layout) shows how these sections
shift. Versions 1/2 retain their signatures when k is unchanged.

The indexes below assume k=1. Add `12 * (k - 1)` to each mating index for
a larger target count; the ten-field mating summary itself stays the same.

| Zero-based index | Feature | Meaning |
| --- | --- | --- |
| 32 | Sex | -1 female, +1 male |
| 33 | Maturity progress | Age / due maturity age, capped at 1; already-reproduced animals stay at 1 |
| 34 | Cooldown remaining | Remaining cooldown for the next action phase / cooldown interval, from 0 to 1; zero before first birth |
| 35 | Self readiness | Personal age, strict HP, energy and dry-cell gates allow attempting reproduction next step |
| 36–37 | Nearest mate dx, dy | World-axis displacement / maximum sight range, including water bonus |
| 38 | Nearest mate distance | Euclidean distance / the same sight scale |
| 39 | Mate visible | Visible opposite-sex animal of the same species exists |
| 40 | Mate readiness | That nearest visible animal passes the same personal gates |
| 41 | Pair opportunity | At least one visible, eligible nearby mate has line of sight, dry route, population capacity and reachable free birth space |

Mate position/presence/readiness values are zero without a visible mate.
Nearest-mate ties prefer lower IDs. A nearer unready mate can be reported while
another nearby ready mate makes pair opportunity true. These inputs do not
predict another controller's action: both animals must still choose reproduction.
Movement, combat, expenditure or contention can invalidate an opportunity.

Readiness shares the action mask's read-only eligibility calculation. Births
resolve after aging, so readiness checks `age + 1`. Before a timing deviation is
sampled, it uses the earliest possible due age; once sampled by physical
simulation, it uses the actual due age. Thus initial/newborn opportunities can
be optimistic. Observing never samples or changes a clock or RNG state.

Mate sight uses ordinary radius, directional water bonus and wall occlusion.
There is no distant/invisible mate state or future-consent input. Phase-local
species/sex spatial buckets, counts and readiness are rebuilt once per observation
phase, including stationary animals whose health, energy or age changed. These
derived lookups stay out of checkpoints.

## Guard provisional lineage winners

Fresh experiments enable `experiment.guard_lineage_chaining` by default. A new
unqualified lineage fallback cannot replace an already qualified starting model
for the same species and sex. Retention preserves weights, model generation and
source identity. Provenance records `retained_after_run` and `retention_reason`.

Fallbacks remain archived. Without a previously qualified model, provisional
selection can still initialize later runs. Qualified replacements are promoted
normally. This is a qualification guard, not held-out evaluation or a guarantee
that every replacement improves. Other criteria and sex isolation are unchanged.
Set `experiment.guard_lineage_chaining: false` to disable it.

The effective guard is saved in checkpoint and CLI manifest metadata without
mutating input configuration. Legacy checkpoints lacking both a saved effective
flag and configuration flag preserve their original unguarded behavior on exact
resume. Their observation versions and trial settings also remain unchanged.

## Updated meandering-river experiment

| Setting | Previous | Updated |
| --- | --- | --- |
| Both observation versions | 2 / 32 inputs | 3 / 42 inputs |
| Predator GRPO trial horizon | 4 | 16 |
| Predator GRPO training interval | 128 | 32 |
| Maximum GRPO group updates per world step | 1 | 1 |
| Lineage promotion guard | Unconditional fallback promotion | Enabled |

The preset retains A2C prey and GRPO predators. Longer, more frequent trials
increase computation; their runtime has not been benchmarked here. Other presets
retain their observation versions and learning cadence; version 3 is opt-in.

Start fresh with lineage selection:

```sh
.venv/bin/python -m ecosystem.cli environments/meandering_river.json \
  --runs 10 --seed 100 --progress \
  --champion prey=lineage --champion predator=lineage
```

Existing 20/32-input weights cannot initialize 42-input brains; signature checks
reject them. Use fresh initialization or version-3-compatible models. Existing
recordings remain playable, and legacy resume uses its embedded configuration.
The viewer handles dynamic diagnostic arrays without a replay-format change.
