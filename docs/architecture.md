# Architecture and simulation rules

[Project overview](../README.md) · [Algorithms](algorithms.md) · [Configuration](configuration.md)

Commands in this guide run from the project root.

## Components

| Location | Responsibility |
| --- | --- |
| `ecosystem/world.py` | Bodies, resources, movement, combat, rewards and reproduction |
| `ecosystem/perception.py` | Incremental spatial windows and cache invalidation |
| `ecosystem/brain.py`, `ppo.py`, `policy_gradients.py` | Independent learned models |
| `ecosystem/grouped_rollouts.py` | Matched GRPO trial worlds |
| `ecosystem/runner.py` | Run lifecycle, feedback, checkpoints and champion selection |
| `ecosystem/library.py` | Compact model artifacts, compatibility and model ancestry |
| `ecosystem/recording.py` | Frame streaming and read-only recording access |
| `ecosystem/analyze.py` | Offline behavior analysis from saved frames |
| `ecosystem/server.py` | Read-only playback HTTP server |
| `viewer/` | JavaScript canvas playback and inspection |
| `environments/` | JSON experiment presets |

## Species identities

Species are arbitrary nonempty keys under the JSON `species` object. Diet values
`resource` and `prey` define prey/predator feeding roles. Multiple species may
share a role and a brain template while keeping independent controllers,
reproduction pairs, species/sex champion pools and metrics. Observation target
indexes combine animals by diet; they do not add a feature block per species.
See [adding species](species.md) for behavior and configuration details.

## Simulation steps and observations

`world.py` owns physical state and emits events. `brain.py` owns individual learning. `runner.py` connects observations, actions, consequences and experiences. `recording.py` writes playback files; `server.py` only serves those files; `viewer/` renders saved data and controls playback.

The world collects all actions before changing animal state:

1. Resolve turns and movement. Walls and initial animal occupancy block moves. Swaps and following into cells vacated during the same step are disallowed. Competing claims to an empty cell use seeded random selection.
2. Resolve resource feeding with seeded contention order. Prey eat from their current cell, or the cell directly ahead. Eating stops at maximum energy.
3. Resolve simultaneous attacks against prey in the cell directly ahead. Combined damage is capped at remaining health. Exactly one seeded attacker gets energy for a kill. A prey's chosen feeding action can complete in the same step it is killed.
4. Charge living and action costs, including failed movement and attack attempts. At zero energy, apply starvation damage. There is no additional affordability restriction on actions.
5. Record deaths and individual rewards, remove dead bodies, resolve eligible births, regrow resources up to capacity, and construct the next observations.
6. Give each learning agent its own experience, including a terminal experience on death, then mix both parents’ model parameters into independent newborn controllers.

Food rewards use actual energy gained. Predators also receive the configurable `species.predator.rewards.per_damage_dealt` reward for successful hits (default environment value: `0.05` per actual damage point). A 10-damage hit earns `0.5`. The baseline now uses 30 attack damage against 30 prey health, giving `1.5` damage reward for a healthy prey killed by one attack; a kill still earns the existing food reward, up to `4.0` for 40 energy at `0.1` per energy point. Other scenarios can keep weaker multi-hit predators. Hit rewards do not restore energy. Missed attacks earn no hit reward. When several predators attack the same prey, its actual health loss is split proportionally to their attack strengths, so overkill cannot multiply damage credit. Omitting `per_damage_dealt` preserves the previous zero-hit-reward behavior, including for old checkpoints. Damage taken and death give configurable penalties. There is no reward for approaching targets, simply surviving a step, moving or resting. Reproduction follows the configured maturity, health, energy, cooldown and population rules in the [configuration guide](configuration.md#reproduction). There is no corpse persistence or health recovery.

The observation vector has 20 entries:

| Indices | Meaning |
| --- | --- |
| 0–1 | Energy and health as fractions of their maxima |
| 2–3 | Facing direction as dx/dy |
| 4–7 | Blocked neighbors: forward, right, backward, left |
| 8–11 | Nearest visible food: dx/radius, dy/radius, distance/radius, visible |
| 12–15 | Nearest visible prey: same four features |
| 16–19 | Nearest visible predator: same four features |

Distances use a circular vision radius, with obstacle occlusion by integer line rays. Direction deltas use world axes; there is no field-of-view angle limit. Missing targets have four zero values. The current animal is excluded from animal targets. Resources with negligible quantity are omitted from perception.

Q-learning brains use one small tanh hidden layer and linear action-value outputs, independent replay buffers, uniformly sampled mini-batches, Huber-loss gradients, plain SGD, target networks, and epsilon-greedy exploration. PPO uses the [actor-critic implementation](algorithms.md), including its own Adam state. A2C, REINFORCE and GRPO share the small-network/optimizer infrastructure described in the [algorithm guide](algorithms.md). No learning algorithm requires a third-party ML framework. Fresh founders start from sex-specific copied base weights but have separate policy and learning randomness. Target networks synchronize after the configured number of training updates.

The same brain template can be referenced by multiple species without sharing live weights. Learning updates occur at the configured cadence. Death is terminal. Time-limit truncations retain bootstrapped future value for Q/PPO/A2C; actor-only REINFORCE and GRPO use their finite returns. Resetting a body can preserve its learned weights, exploration state and experience buffer, or reinitialize its brain according to configuration. Species base weights are reproducible across resets.

`train` explores and learns; `evaluate` chooses greedily and does not learn; `random` chooses random allowed actions and does not learn. A population that survives longer is not by itself evidence of learning. Compare frozen trained brains with fresh and random policies across multiple held-out seeds. The initial settings are experimental, not calibrated for stable coexistence. Independent simultaneous learners can be unstable. Each new CLI invocation starts fresh controllers, optionally with saved weights selected through `initial_brain`. Exact continuation uses `--resume`; see [saved models](brain-library.md). With `--seed` or a deterministic seed policy, repeating a single run reproduces its outcome; it does not continue learning from a previous invocation. Randomized mode varies fresh runs but is not a substitute for retaining or resuming learned brains. Use multiple runs with champion chaining or resume an intermediate checkpoint to continue training. Making successful attacks lethal increases predation opportunities; it is a physical rule change, not evidence that the model learned to hunt.

The standard-library implementation prioritizes inspectability and small populations. Large worlds and many private networks will be slower than a vectorized implementation. CLI metrics stream to disk, and playback charts retain the current run’s lightweight history across world-chunk changes. The simulation runner retains at most 2,000 sampled metric points per current run for checkpoint compatibility; this is separate from playback recording.
