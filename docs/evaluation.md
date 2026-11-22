# Evaluating behavior and policy changes

[Analysis](analysis.md) · [Calibration](calibration.md) · [Saved brains](brain-library.md)

Use matched evaluation to distinguish a useful policy or configuration change
from changes caused by placement, starting reserves, population or opponent
behavior. This guide describes the method; experiment observations and run
histories do not belong in package documentation or release notes.

For an executable walkthrough, follow [Tutorial 5: compare saved policy banks](tutorials/05-compare-saved-policies.md).

## Match the starting conditions

Capture the effective environment and select compatible species/sex brain banks.
Keep geometry, food, physics, observation/action contracts and opposing policies
constant unless one is the intended intervention. Use the same held-out world
seeds across conditions and verify initial bodies and model hashes.

For population changes, a balanced subset of a common founding cohort can
preserve retained animals' IDs, positions, sexes and initial action/training RNGs.
Recompute observations and masks after removing animals. Equal starting seeds do
not guarantee equal future random-event schedules: trajectories and RNG draws
naturally diverge after actions, births and deaths differ.

## Specify what remains adaptive

Frozen evaluation disables optimizer updates and birth mutation. Categorical
sampling and greedy actions are different evaluation policies; choose deliberately
and keep the choice consistent across conditions. Verify zero updates and
unchanged parameters throughout each animal's life.

Reproduction can remain active during frozen evaluation. Ordinary crossover
creates a new child parameter set even when each individual never learns.
Use `--mode evaluate --freeze-brains` to copy same-sex parent weights at birth
without crossover. Ordinary evaluation without the flag retains crossover.
Describe an opponent as a fixed starting bank unless descendants also preserve
those weights; do not imply unchanged descendant policies otherwise.

Training comparisons keep the learning budget, optimizer resets, inheritance and
mutation settings consistent except for the chosen intervention. Keep initial
founder banks fixed between independent trials; chained worlds answer a different
question. Offspring inherit weights with fresh learning state, not parental
experience buffers or optimizer history.

## Report comparable measures

| Question | Useful measures |
| --- | --- |
| Are animals feeding effectively? | Realized food energy, energy coverage, kill/feed success and rates per animal-step |
| Are they surviving? | Starving exposure, cause-specific death rates, initial-cohort restricted mean lifetime and extinction |
| Are descendants viable? | Offspring exposure, kills, survivors and offspring that become parents |
| Is reproduction sustained? | Qualified lineage depth, birth rate and the fraction reaching the target generation |
| Is the ecosystem supported? | Both species' population and food trends, not one species' isolated score |

Counts can rise solely because more animals are alive. Normalize exposure and
report missing denominators rather than replacing them with zero. Alive animals
are censored at the horizon; offspring born late have less observation time.
Generation-zero bodies in a fresh world may carry older learned weights, so
biological generation and model provenance must remain distinct.

Compare paired seed differences and report uncertainty at the independent unit.
Worlds sharing one source policy bank are not independent training replicas.
Different selection criteria or training exposure prevent a simple causal claim
that generation or crossover alone caused an observed difference.

## Artifacts and promotion

Keep effective configurations, seeds, source/model hashes, endpoint metrics and
bounded scalar traces under ignored results. Save replay or exact checkpoints
only when needed for the task. Existing `work/` utilities expose their inputs and
outputs through `--help`; study-specific settings are tool inputs, not package
recommendations or release records.

Update the calibration guide when a supported recommendation changes. Update
normal documentation and release records only for adopted implementation,
configuration or interface changes, including their verification and limits.
