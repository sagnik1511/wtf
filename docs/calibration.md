# Current calibration and tuning decisions

[Configuration](configuration.md) · [Rewards](rewards.md) · [Geography](geography.md) · [Evaluation](evaluation.md)

This guide records the current recommended settings and their supported scope.
Update a recommendation when evaluation supports a better choice; keep campaign
logs, individual run identifiers, observation tables and rejected candidates out
of this guide. This is a living decision reference, not a chronological history.
Environment JSON remains authoritative for the settings actually used.

## Current survival-oriented flat-world profile

For the flat escape/consent setup with trained, compatible species/sex brain
banks, use a lower initial predator density while retaining proximal mating.
The maintained profile is
[`maskable_ppo.json`](../environments/maskable_ppo.json). It already contains the
recommended population, observation and body settings; no density override or
experiment-specific JSON is required. Superseded exploratory configurations
remain local and are not package presets.

| Setting | Current recommendation | Reason or constraint |
| --- | --- | --- |
| Initial populations | 48 prey, 6 predators | Lower starting pressure supports predator feeding and prey replenishment |
| Population caps | 96 prey, 24 predators | Leave room for growth; keep caps separate from starting density |
| Predator mating radius | 3 cells | Preserve physical proximity, visibility and dry-route gates |
| Prey mating radius | 2 cells | Retain proximal pairing |
| Predator kill energy | 60 | Improve feeding support, subject to the existing 120-energy cap |
| Observation top-k | Predator 10, prey 3 | Retain the compatible observation/model contract |
| Observation and controller | Version 4, masked PPO | Local consent signals and immediate physical action masks |
| Escape actions | Existing prey backward/sideways actions | Provide learned evasion without automatic escape |
| Child inheritance | Existing scalar crossover | A reliable ecological advantage from whole-parent copying is not established |
| Mutation | Rare child-only mutation, probability 0.0001 | Preserve the existing exploration mechanism without treating it as proven improvement |

To create another scenario, copy this maintained JSON and edit the intended
fields. Keep comparisons explicit rather than reusing a filename whose settings
no longer match the intended control.

Retain the profile's other body, food, reproduction, reward and learning
settings when applying this recommendation. In particular, predator health is
55, initial energy 80, living cost 0.15, movement cost 0.2, attack damage 30 and
attack cost 1. Prey health is 40. Mutual five-step resting consent, maturity,
cooldowns and species/sex brain ownership remain unchanged. The common preset
retains its randomized-world lifecycle and lifespan-based champion defaults;
lineage selection remains an explicit configuration/CLI choice.

**Supported objective:** reduced starvation and better founder/offspring survival
under the evaluated flat-world policy configuration. **Unresolved objective:**
deeper predator generations. A lower founding population supplies fewer lineage
branches and potential mates; better individual persistence does not establish
better reproductive continuity or long-term coexistence. Do not label this the
best reproductive calibration or extrapolate it to all policies and maps.

## Terrain-specific settings

Terrain changes travel costs, accessible food, encounters and water risk. Keep
its calibration separate from the flat profile. The maintained terrain presets
currently distinguish maintenance costs from opening reserves:

| Preset | Initial energy, prey / predator | Living cost, prey / predator | Movement cost, prey / predator | Predator attack cost |
| --- | --- | --- | --- | --- |
| [River crossing](../environments/river_crossing.json) | 60 / 80 | 0.05 / 0.075 | 0.10 / 0.14 | 0.25 |
| [Highlands](../environments/highlands.json) | 75 / 100 | 0.07 / 0.10 | 0.10 / 0.14 | 0.35 |
| [Enclosed valleys](../environments/enclosed_valleys.json) | 60 / 80 | 0.05 / 0.075 | 0.10 / 0.14 | 0.25 |

These are current preset values, not evidence that the same choice is optimal
after changing geography, policy, placement or observation contracts. Higher
initial energy also increases the energy granted to every newborn. Keep that
subsidy explicit when interpreting survival, rather than crediting it to learning.

## What to measure before changing a recommendation

- **Energy:** food-energy coverage, starving animal-step fraction and starvation
  deaths per animal-step. Initial stores and newborn grants are separate inputs.
- **Predation:** prey survival to maturity, predator kill rate and feeding
  distribution. More efficient hunting can still deplete the prey population.
- **Reproduction:** age/health/energy eligibility, proximal mate opportunities,
  mutual consent, birth space, offspring that become parents and lineage depth.
- **Persistence:** both species' populations, censoring at the horizon and
  repeated-seed consistency. Raw births alone are insufficient.

Reproduction spends energy, not health directly. Starvation damages health after
energy is exhausted, and there is no health recovery. A brief energy shortage can
therefore permanently remove an animal's breeding eligibility. More health only
extends the margin; it does not supply food or restore already lost health.

## Updating this guide

After a relevant experiment, assess whether it changes the supported calibration.
Replace the current recommendation only when matched evidence supports the
intended objective without an unacceptable regression. Record the affected
settings, applicable terrain/controller/initialization conditions, reason and
remaining uncertainty. If no improvement is established, keep the current choice
and its limitations rather than appending another trial report.

Keep mating radius proximal unless an explicit objective authorizes changing it.
Compare one intervention at a time, retain the control, use held-out seeds for
promotion decisions and distinguish training from frozen-policy evaluation.
Adopted JSON changes must be validated; an experiment by itself is not a release.

## Preset ownership and compatibility

Promote supported settings into the common preset for the evaluated controller
and geography. Do not retain a separate public JSON for every trial variant or
copy a flat-world result into unrelated algorithm/terrain presets without evidence.
Keep superseded controls local; small test fixtures may retain old input/action
contracts solely for regression coverage.

The maintained masked-PPO profile uses version-4 observations, prey k=3 and
predator k=10, with prey escape actions. Old nearest-only/version-1 weight banks
are incompatible with those signatures. Start fresh or load compatible weights;
exact checkpoints and recordings continue using their captured configurations.
The common name is stable, while its current supported defaults can improve.
