# Mating readiness and guarded inheritance

[Mating intent](mating-intent.md) · [V7 observations](learning/v7-observations.md) · [Champion selection](brain-library.md)

The current V7 input includes twelve mating/consent values, after the terrain
block. They help an animal react to observed opportunities without granting
automatic reproduction or access to hidden mate state.

## Current mating input block

| Values | Meaning |
| --- | --- |
| Sex, maturity, cooldown, own readiness | Own body/readiness state; timing windows use the configured personal clocks |
| Mate forward/right and presence | Observed mate position in the current heading frame, with explicit absence |
| Mate readiness and pair opportunity | Current observed eligibility; a positive cue is not a future birth guarantee |
| Own consent, observed mate consent, consenting pair | Existing active intent, subject to physical and visibility rules |

The redundant mate-distance field is omitted in V7. A hidden mate's state is not
added just because it is physically nearby. Radius, directional field, water
sight and walls govern visual mate cues; smell does not reveal a consenting mate.
Own maturity and cooldown expose current progress, not future RNG samples.
Observation production does not advance reproduction clocks or consume RNG.

Both parents must have active consent, same species/opposite sex, eligible
health/energy/age/cooldown, dry reachable proximity and birth space/capacity.
`reproduce` grants or refreshes eligible consent; configured `rest` steps can
retain it. Other actions cancel it under the current implementation. See
[consent state and failed attempts](mating-intent.md).

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


## Presets and saved brains

The ordinary and masked PPO profiles keep proximal prey/predator radii 2/3.
The meandering-river preset uses V7 inputs with A2C prey and GRPO predators;
its trial horizon and training interval are still 16 and 32. Its current k=1
sensory vectors have 61 values, including history and blended scent.
Use matching complete model contracts or start fresh. Exact checkpoint resumes
retain their captured versions, clocks, private training state and knowledge.

<a id="observation-version-3"></a>
<a id="updated-meandering-river-experiment"></a>

Earlier V3 layouts and river settings are preserved only in the
[historical mating-input reference](archive/mating-inputs-v3.md).
