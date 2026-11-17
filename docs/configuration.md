# Environment configuration and reproduction

[Project overview](../README.md) · [Runs](running.md) · [Algorithms](algorithms.md)

Commands in this guide run from the project root.

Each JSON file is a complete experiment definition:

| Preset | Prey | Predators | Prey / predator brain |
| --- | ---: | ---: | --- |
| [a2c.json](../environments/a2c.json) | 48 | 12 | a2c / a2c |
| [dqn.json](../environments/dqn.json) | 48 | 12 | q_network / q_network |
| [baseline.json](../environments/baseline.json) | 48 | 12 | q_network / q_network |
| [champion_lineage.json](../environments/champion_lineage.json) | 48 | 12 | q_network / q_network |
| [dense_obstacles.json](../environments/dense_obstacles.json) | 24 | 6 | q_network / q_network |
| [grpo.json](../environments/grpo.json) | 48 | 12 | q_network / grpo |
| [mixed_brains.json](../environments/mixed_brains.json) | 48 | 12 | q_network / ppo |
| [ppo.json](../environments/ppo.json) | 48 | 12 | ppo / ppo |
| [ppo_large_predator.json](../environments/ppo_large_predator.json) | 48 | 12 | ppo [16] / ppo [64, 32] |
| [recurrent_ppo.json](../environments/recurrent_ppo.json) | 48 | 12 | recurrent_ppo / recurrent_ppo |
| [qrdqn.json](../environments/qrdqn.json) | 48 | 12 | qrdqn / qrdqn |
| [maskable_ppo.json](../environments/maskable_ppo.json) | 48 | 12 | maskable_ppo / maskable_ppo |
| [mappo.json](../environments/mappo.json) | 48 | 12 | mappo / mappo |
| [predator_heavy.json](../environments/predator_heavy.json) | 24 | 14 | q_network / q_network |
| [reinforce.json](../environments/reinforce.json) | 48 | 12 | reinforce / reinforce |
| [scarce_food.json](../environments/scarce_food.json) | 24 | 6 | q_network / q_network |

`champion_lineage.json` requires archived compatible Q models; the other presets start fresh unless you add `initial_brain`.

Copy a JSON file, edit its name and values, and pass its path to the Python CLI to add a scenario. The playback viewer shows the recorded configuration as read-only text.

All costs and growth rates are per simulation step. Playback speed changes only the rate of displaying saved frames. Configuration changes apply to the next CLI experiment; they cannot change an existing recording.

`ecosystem/config.py` validates required and unknown fields, numeric ranges, compatible diets and actions, referenced brain templates, replay settings, and world capacity. Invalid configurations are rejected before simulation begins. `environment.schema.json` provides an editor-friendly structural schema; Python validation also enforces cross-field constraints.

Use `obstacles.generation = "explicit"` with `explicit_cells = [[x, y], ...]` for hand-designed maps. In this mode `occupied_fraction` is ignored. Animals and resource patches may share cells; two animals cannot occupy one cell. Generation does not guarantee connected free space.

## Common changes

| Change | JSON field |
| --- | --- |
| Starting populations | `species.prey.initial_population`, `species.predator.initial_population` |
| Body health/energy | `species.<name>.body` |
| Perception radius | `species.<name>.body.vision_radius` |
| Food supply/regrowth | `resources` |
| Map size and duration | `world.width`, `world.height`, `world.max_steps` |
| Learning rate per species | `species.<name>.learning_rate_multiplier` (template rate × multiplier) |
| Model and training algorithm | `species.<name>.brain` and the referenced `brains` template |
| Predator survival + kill selection | `species.predator.champion_selection` ([scoring rule](brain-library.md#balance-predator-kills-and-survival)) |
| Cross-run inheritance | `experiment.chain_best_brains` (default `true`), or `--no-chain-brains` |
| Initial learned weights | `species.<name>.initial_brain` ([guide](brain-library.md)) |
| Reproduction | `species.<name>.reproduction` |
| Rare newborn weight mutation | `species.<name>.mutation` ([guide](mutation.md)) |
| Randomization and repetitions | `experiment` ([guide](running.md#randomized-and-reproducible-runs)) |
| Disk recording cadence | `recording` ([guide](recording.md)) |

For per-experiment model choices without editing JSON, use the [CLI model selectors](brain-library.md#choose-models-with-cli-arguments). The resolved choices are saved with the new experiment.

Predator health in the supplied presets is 55 in the main scenarios and 44 in
the dense-obstacle, scarce-food and predator-heavy scenarios (a 10% increase).
Reproduction thresholds remain at 80% of maximum health. With starvation damage
of 1, the extra health provides roughly 5 or 4 additional starving steps,
respectively, assuming no other damage or recovery. This is a tuning choice;
actual survival also depends on energy use and hunting.

## Learning rates per species

All supplied presets use `learning_rate_multiplier: 1.0` for prey and `1.25`
for predators. Set this optional field inside each species object:

```json
"learning_rate_multiplier": 1.25
```

The effective rate is the referenced `brains` template's `learning_rate` times
this multiplier. For a template rate of `0.001`, prey use `0.001` and predators
use `0.00125`. Referencing a common template still creates independent models;
this setting does not modify the template or another species' rate. Set both to
`1.0` for equal rates; lower values reduce update size. Multipliers must be
between 0.01 and 10, and the effective rate must remain within the algorithm's
validated range (0.000001–1 for Q-learning, 0.000001–0.1 for policy algorithms).
Omitting the field defaults to 1.0, preserving old configurations.

This applies to every supported algorithm, including QR-DQN and the PPO variants. It changes update size,
not update frequency, rollout length, exploration or reward. REINFORCE still
updates only when an episode finishes; PPO/GRPO clipping and gradient clipping
still apply. Policy/value networks that share an optimizer both use the rate.
Larger updates do not guarantee faster or better learning and can destabilize it;
the asymmetry is an experimental choice rather than a biological rule. Compare
matched seeds and evaluation outcomes using the [analysis guide](analysis.md).

Newborns inherit their parent's effective rate. New runs, including champion
warm starts and chaining, apply the current environment's multiplier once.
Exact checkpoint resumes retain the stored controller configuration; editing an
environment does not retune an already saved checkpoint or recording.

## Larger predator PPO network

`environments/ppo_large_predator.json` keeps the world, rewards, prey network
(`[16]`) and training settings from `ppo.json`, while using two hidden layers
(`[64, 32]`, tanh) for the predator. Both species use PPO and explicitly select
`champion_selection.criterion: "lifespan"` for cross-run inheritance. The original
`ppo.json` defaults to lifespan for prey and kills for predators, so matching
those selection criteria is also necessary to isolate network capacity.

Start a fresh experiment from the project root:

```bash
.venv/bin/python -m ecosystem.cli environments/ppo_large_predator.json \
  --runs 20 --seed 100 --progress \
  --brain prey=fresh --brain predator=fresh
```

`fresh` controls initial weights; it does not select a winner criterion. This
preset starts with untrained models and later episodes automatically chain the
previous episode's longest-lived champions for both species. Lifespan means
individual age in steps, including dead individuals recorded before episode end.
Add `--no-chain-brains` to restart each episode from fresh weights. The original
preset remains available as a baseline; align selection criteria as well as
seeds for a network-size comparison. Old `[16]` predator champions cannot
initialize the `[64, 32]` network because saved weights require matching
architectures. This preset increases model and optimizer memory per predator
and computation per step; more capacity does not guarantee better hunting.
To tune capacity, edit
`brains.large_predator_ppo.network.hidden_layers` in the new JSON.

The [advanced algorithms guide](advanced-algorithms.md) covers GRU memory,
quantile regression, immediate action masking and centralized critics. All four
new presets select lifespan champions for both species.

## Reproduction

All supplied environments enable single-parent reproduction for both species.
Edit `species.prey.reproduction` or `species.predator.reproduction` in your JSON:

```json
"reproduction": {
  "enabled": true,
  "health_threshold": 24,
  "min_age_steps": 300,
  "age_deviation_pct": 10,
  "cooldown_steps": 225,
  "cooldown_deviation_pct": 10,
  "min_energy": 30,
  "energy_cost": 20,
  "max_population": 96
}
```

Health must be **strictly greater** than `health_threshold` (absolute health,
not a percentage). After combat and starvation, a surviving parent must also
meet the age and energy requirements, afford the energy cost with some energy
left, and have waited since its last successful birth. A species' living
population cannot exceed `max_population`. Each eligible parent produces at
most one child per step, in a free cardinally adjacent cell. If the cap or space
blocks a birth, no energy is charged and no cooldown starts. Contention uses the
world's seeded random generator. Cooldowns measure simulation steps, not playback
time. The presets use 80% of maximum health as their threshold and randomize
reproduction timing with approximately 10% deviation:

| Species | First eligible age | Cooldown after an actual birth |
| --- | --- | --- |
| Prey | `300 ± 30` = 270–330 steps | `225 ± 23` = 202–248 steps |
| Predator | `100 ± 10` = 90–110 steps | `200 ± 20` = 180–220 steps |

Each animal draws one uniformly distributed integer from the inclusive first-age
window on its first reproduction check. After a successful birth it draws one
cooldown interval, measured from that actual birth age. Newborns have their own
first-age draw; they do not inherit their parent's clock. These are per-animal
age windows, not global simulation ticks, and timing is independent of the brain
algorithm. The example `min_age_steps` is the center of the first-age window,
so a nonzero deviation permits maturity before that center.

Timing is never redrawn just because a step passes or health, energy, capacity
or space prevents birth. Once the sampled age is reached, eligibility persists
until the birth succeeds, even beyond the window's upper edge; the health and
energy gates still apply. The window randomizes when eligibility begins, rather
than defining a short fertility period or a per-step birth probability.

Both deviation fields are optional percentages from 0 to 100, defaulting to
zero. `10` means ±10%; fractional percentages such as `2.5` are supported.
The step deviation is `floor(center * percentage / 100 + 0.5)` (half steps
round up). For example, a 225-step cooldown with 10% deviation rounds to ±23
steps. Editing the center automatically scales its window.
Zero restores the original fixed minimum-age and cooldown rules and consumes no
extra timing randomness. Windows extending below one step are clipped to one.
The world seed controls timing draws, and checkpoints retain pending clocks and
random-generator state for exact continuation. Editing a preset affects new
experiments; an exact resume uses its saved rules and clock.

Older saved configurations with `age_deviation_steps` or
`cooldown_deviation_steps` remain readable with their original absolute-step
semantics. New presets use percentages. Do not specify both units for the same
clock; validation rejects that ambiguity.

The [optional mutation rule](mutation.md) can reset training state after a rare
newborn weight change; ordinary births follow the inheritance behavior below.

A child starts at age zero, with the same species' `body.max_health` (also the
founder's initial health) and `body.initial_energy`. It inherits an independent
copy of its parent's current policy, critic/target model, optimizer state,
exploration setting, learned baseline and training counters, after the parent
receives that step's feedback. Experience buffers and pending decisions start
empty; random streams are independent. Reproduction has no mating requirement; optional newborn mutation is described separately.
Recurrent PPO children start with zero GRU memory; masks and centralized critic
context are derived for the child from the current world. Children first act on
the next step and must reach their own sampled eligible age.
Starting health/energy are supplied by the environment; the energy cost is a
separate parent charge, so this model does not conserve energy across births.

Frames/checkpoints include `parent_id` (-1 for founders), `generation`,
`last_reproduced_age` and `reproduction_due_age`. The latter is an absolute
individual age, initially -1 until a randomized clock is scheduled. It stays -1
with fixed timing. The inspector displays a scheduled eligible age when present. Birth events identify parent (`actor`) and child (`target`);
metrics, CSV and summaries count births per species. The viewer emphasizes births with cyan rings that follow newborns, plus signs,
BORN labels, a parent/child banner, and a highlighted birth counter. Markers
last briefly across sequential chunk boundaries and reset when switching runs.
These effects work with existing recordings that contain birth events. IDs are never reused within a run.
Dead descendants' controllers are released. Later runs initialize from the
previous run's champions by default; see [run chaining](brain-library.md#automatically-chain-runs). Each new run starts with the
configured initial population. GRPO trial worlds include births with independent
child random streams and frozen inherited policies, with independent copies for
rare mutated trial children.

Set `enabled` to false to disable births. Omitting `reproduction` also disables
it, keeping older environment files and checkpoints compatible. Changes affect
new experiments; resuming uses the saved configuration. Reproduction can delay
extinction but does not guarantee a stable ecosystem: offspring still need food,
and parents with low health cannot reproduce.

## Rare newborn mutations

All supplied presets enable `species.<name>.mutation` with
`probability_per_birth: 0.0001`, `max_changed_weights: 3` and `noise_std: 0.005`.
This applies only to training births. A selected child receives small Gaussian
perturbations on 1–3 matrix weights, fresh optimizer state and synchronized
algorithm targets/references. The parent is unchanged. Omit the block or disable
it to retain ordinary inheritance. See the [mutation guide](mutation.md) for
configuration, probability semantics, lineage tracking and exact resume behavior.
