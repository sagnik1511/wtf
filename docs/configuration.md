# Environment configuration and reproduction

[Project overview](../README.md) · [Runs](running.md) · [Algorithms](algorithms.md)

Commands in this guide run from the project root.

Each JSON file is a complete experiment definition:

| Preset | Prey | Predators | Prey / predator brain |
| --- | ---: | ---: | --- |
| [a2c.json](../environments/a2c.json) | 48 | 12 | a2c / a2c |
| [baseline.json](../environments/baseline.json) | 48 | 12 | q_network / q_network |
| [champion_lineage.json](../environments/champion_lineage.json) | 48 | 12 | q_network / q_network |
| [dense_obstacles.json](../environments/dense_obstacles.json) | 24 | 6 | q_network / q_network |
| [grpo.json](../environments/grpo.json) | 48 | 12 | q_network / grpo |
| [mixed_brains.json](../environments/mixed_brains.json) | 48 | 12 | q_network / ppo |
| [ppo.json](../environments/ppo.json) | 48 | 12 | ppo / ppo |
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
| Model and training algorithm | `species.<name>.brain` and the referenced `brains` template |
| Initial learned weights | `species.<name>.initial_brain` ([guide](brain-library.md)) |
| Reproduction | `species.<name>.reproduction` |
| Randomization and repetitions | `experiment` ([guide](running.md#randomized-and-reproducible-runs)) |
| Disk recording cadence | `recording` ([guide](recording.md)) |

## Reproduction

All supplied environments enable single-parent reproduction for both species.
Edit `species.prey.reproduction` or `species.predator.reproduction` in your JSON:

```json
"reproduction": {
  "enabled": true,
  "health_threshold": 24,
  "min_age_steps": 300,
  "cooldown_steps": 225,
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
time. The presets use 80% of maximum health as their threshold. Prey first reproduce
at age 300 and wait 225 steps between births; predators mature at age 100 and
wait 200 steps. These settings slow prey growth while retaining the same caps.

A child starts at age zero, with the same species' `body.max_health` (also the
founder's initial health) and `body.initial_energy`. It inherits an independent
copy of its parent's current policy, critic/target model, optimizer state,
exploration setting, learned baseline and training counters, after the parent
receives that step's feedback. Experience buffers and pending decisions start
empty; random streams are independent. There is no mutation or mating requirement.
Children first act on the next step and must reach the minimum age themselves.
Starting health/energy are supplied by the environment; the energy cost is a
separate parent charge, so this model does not conserve energy across births.

Frames/checkpoints include `parent_id` (-1 for founders), `generation`, and
`last_reproduced_age`. Birth events identify parent (`actor`) and child (`target`);
metrics, CSV and summaries count births per species. The viewer emphasizes births with cyan rings that follow newborns, plus signs,
BORN labels, a parent/child banner, and a highlighted birth counter. Markers
last briefly across sequential chunk boundaries and reset when switching runs.
These effects work with existing recordings that contain birth events. IDs are never reused within a run.
Dead descendants' controllers are released; only founding controllers persist
when `retain_brains_between_runs` is enabled. Each new run starts with the
configured initial population. GRPO trial worlds include births with independent
child random streams and frozen inherited policies.

Set `enabled` to false to disable births. Omitting `reproduction` also disables
it, keeping older environment files and checkpoints compatible. Changes affect
new experiments; resuming uses the saved configuration. Reproduction can delay
extinction but does not guarantee a stable ecosystem: offspring still need food,
and parents with low health cannot reproduce.
