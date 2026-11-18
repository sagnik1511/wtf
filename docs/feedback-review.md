# Generic environment feedback review

[Reward rules](rewards.md) · [Algorithms](algorithms.md) · [Geography](geography.md) · [Configuration](configuration.md)

Subsequent improvement: [version-3 mating inputs and guarded promotion](mating-observations.md)
address the observation and fallback limitations identified here without changing
the reward formula. This review describes the earlier 20/32-input baseline.

Reviewed on 2026-11-18 against all 24 environment presets, the shared world
transition, native flat-world metabolism, algorithm delivery, GRPO trial worlds
and offline feedback accounting. No missing or double-applied reward was found
in the reviewed paths. Runtime formulas and preset coefficients are unchanged.
This is a code/contract review, not a claim that the presets are calibrated or
that every policy learns the desired behavior.

## One environmental reward, different learning algorithms

`World.step` produces per-animal feedback from actual physical costs and events.
It looks up `species.<name>.rewards`; no map filename or algorithm selects a
different environmental reward function. Species identity is arbitrary; diet
controls feeding behavior, while species/sex determine mating eligibility.

The conceptual reward is:

```text
food energy gained × food coefficient
+ actual damage dealt × hit coefficient
+ actual damage taken × harm coefficient
+ death penalty, if dead
+ total energy demand × effort coefficient
+ failure penalty, once if the chosen attempted action failed
+ offspring reward, once for each participating parent
```

This is a description of the existing accounting, not an instruction to change
floating-point summation order. Physical energy is debited separately; learning
feedback does not spend the same energy again. Food intake is capped by available
capacity, damage by the target's remaining health, and energy *demand* is counted
before zero-flooring reserves. Thus exhausted animals do not get free attempts.

| Controller | How it uses the feedback |
| --- | --- |
| DQN (`q_network`) | Individual reward in a bootstrapped TD target |
| QR-DQN | Individual reward in quantile TD targets |
| PPO | Individual reward in value targets and advantages |
| A2C | Individual reward in actor/critic returns and advantages |
| Recurrent PPO | The same individual reward, with recurrent state and chronological updates |
| Maskable PPO | The same individual reward; physical action masks restrict available choices |
| REINFORCE | Individual discounted episodic returns; no critic bootstrap |
| GRPO | Individual discounted returns from matched cloned-world trials, normalized within a group |
| MAPPO | Intentional exception: species-mean reward for training, with a centralized critic |

Death terminates an individual's transition. A world time limit truncates it;
value-based/actor-critic algorithms keep their existing bootstrap rules.
REINFORCE and finite-horizon GRPO use their existing return definitions. An
algorithm's policy/value loss is calculated by the learner, not the world.
Evaluation and random mode still produce environmental feedback without learning.

MAPPO's recorded `last_reward` remains the **individual environmental reward**.
Its learner receives the species mean over animals present at the start of that
step, including any that die. Raw dashboard/offline reward is therefore not a
MAPPO optimizer objective. Existing analysis also reports missing terminal action
coverage when a recording does not identify the deceased animal's action; it
does not recover unrecorded data by assumption.

## Every map uses physical consequences

| Geography | Consequences already included in feedback |
| --- | --- |
| Flat presets, including density/food variants | Living and attempted movement/attack demand, feeding, combat, starvation, failed choices and paired births |
| Straight river and meandering river | Swim attempt demand, successful depth-dependent swim effort, water maintenance, drowning and blocked entry/strokes |
| Branching delta | The same river rules, sampled at the animal's actual position after vector drift |
| Lake islands | The same swimming/water costs, with zero currents; no passive crossing bonus |
| Highlands and alpine passes | Successful uphill cost, maximum-climb restrictions and failed blocked moves; no swim output in dry presets |
| Enclosed valleys | Obstacle/occupancy/gate restrictions, local height/water costs and proximal reproduction eligibility |

Successful voluntary movement determines move/swim success. River drift cannot
turn a blocked voluntary action into a success. Blocked strokes pay their normal
attempt cost but not a successful-stroke or uphill charge. Current transport
has no invented directional reward. There is no extra height, river-crossing,
proximity or lifespan bonus.

An attack succeeds on positive damage, even without a kill. Simultaneous attacks
share actual hit credit and only one killer receives prey energy. A full predator
can still receive damage-dealt reward while getting no food energy. This is an
intentional hunting incentive, which can favor hunting beyond immediate hunger.

Reproduction succeeds only after both parents choose it and the physical gates
pass. Each parent receives its own offspring credit and reproduction energy
feedback. Failed reproduction has no birth cost/cooldown, but does incur normal
living demand and the failure penalty. Newborn energy is a body initialization
grant, not a food reward; the child has no transition to learn on its birth step.

## Current coefficients and scale

All 24 presets currently share these per-diet coefficients. The JSON format
already allows a species in any environment to choose other valid values.

| Reward coefficient | Resource-eating species | Prey-eating species |
| --- | --- | --- |
| `per_energy_gained` | 0.1 | 0.1 |
| `per_damage_dealt` | 0 | 0.05 |
| `per_damage_taken` | -0.2 | -0.2 |
| `death` | -10 | -10 |
| `per_energy_cost` | -0.01 | -0.01 |
| `failed_action` | -0.05 | -0.05 |
| `per_offspring` | 1 | 1 |

For the meandering/river PPO physical settings, a dry successful prey uphill
move of one elevation unit demands `0.1 + 0.15 + 1 = 1.25` energy, yielding
`-0.0125` effort feedback. A successful prey birth contributes
`1 - 0.01 × (10 + 0.1) = 0.899` including that step's living cost. A predator
birth similarly yields `1 - 0.01 × (15 + 0.15) = 0.8485`, absent other events.

A predator kill dealing 30 actual damage and gaining 40 energy yields
`1.5 + 4 - 0.01 × (1 + 0.15) = 5.4885`, absent other events. At full energy,
the same hit can still yield `1.4885` because damage reward is independent of
food. These examples expose relative incentives; they do not prescribe a new
calibration or guarantee reproduction.

## Learning limits that correct accounting does not solve

- Mutual mating has a sparse coordination reward. Current 20/32-input contracts
  do not explicitly expose age, sex, cooldown or an eligible mate's readiness.
  A failure penalty alone cannot tell a policy which hidden gate failed.
- Identical coefficients do not make terrains equally easy. Extra swimming,
  uphill effort, route length and separation change the reward distribution.
  Calibration should adjust explicit JSON parameters against comparable seeds
  and measured survival/reproduction, rather than branch by map name in code.
- Short GRPO trials can observe immediate costs while missing a later feeding
  or reproduction benefit. Its training interval and trial horizon matter even
  though trial rewards use the same physical rules. Equal-return trials provide
  no relative policy signal.
- MAPPO averages individual outcomes; an individual's death is diluted across
  its species. This is its documented cooperative objective, not lost feedback.
- Lineage curation chooses inherited models; it does not add a family reward to
  the controller's loss or promise long-term evolutionary improvement.

Observation changes, reward shaping or calibration experiments are separate
behavior changes. None are introduced by this review.

## Regression coverage

The new `tests/test_feedback_contract.py` crosses every preset with all nine
controllers: **216 combinations**, each checking living feedback and a terminal
starvation transition, including MAPPO's species mean and individual terminal
flags. It deliberately renames species and retains map/placement configuration.
Small populations, forced valid resting choices and lightweight controller
templates make this a deterministic contract check rather than an experiment.
REINFORCE's buffer limit is matched to the configured world horizon; GRPO trials
are deferred so the matrix isolates real-transition delivery.

Existing numerical/event tests additionally cover partial and shared kills,
feeding capacity, failed/successful reproduction, both parents' costs, optional
legacy coefficients, terrain effort/drowning, GRPO trial isolation and
event-derived terminal/birth feedback. Run the focused review checks with:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_feedback_contract tests.test_reproduction_decisions \
  tests.test_hit_rewards tests.test_geography
```

These checks establish accounting/delivery contracts, not generalization,
population stability or the optimal coefficient values for each landscape.
