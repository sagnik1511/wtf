# Configuration field reference

[Customization handbook](customization.md) · [Configuration guide](configuration.md) · [Schema](../environment.schema.json)

Generated from schema definitions with `python -m work.build_config_reference`. Use `--check` to detect missing definitions or stale exports. Every declared field is listed, including each algorithm-specific variant.

A `<name>` component means a user-chosen key. Required status is relative to the containing object/variant, not a requirement to enable that optional feature. A schema default describes omission; it does not rewrite your JSON. A dash means no default is declared here: copy a complete supported preset and consult its feature guide for runtime behavior.

The Python validator also checks cross-field rules (map dimensions, population capacity, diet/body/actions, reproduction limits and model signatures). Passing a JSON editor's schema check alone is not sufficient. Physical values are abstract simulation units; time is measured in world steps.

## `config`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `terrain` | Optional map geometry, either an inline terrain map or a map_file reference. | variant | Optional; — |
| `schema_version` | Environment document schema version, not a package or checkpoint version. | value; exact: `1` | Required; — |
| `name` | Stable environment or map name; experiment folder names use the environment name. | string; pattern: `"^[a-zA-Z0-9_-]+$"` | Required; — |
| `description` | Human-readable purpose of this preset; does not affect physics. | string | Required; — |
| `seed` | Base seed for layout, controllers and reproducible run-seed generation; CLI overrides can replace it. | integer; min: `100`; max: `4294967295` | Required; — |
| `world` | Grid dimensions, boundary rule and maximum episode duration. | object | Required; — |
| `obstacles` | Blocking-cell generation or explicit coordinates. | object | Required; — |
| `resources` | Shared food patch placement, capacity and regrowth. | object | Required; — |
| `species` | Named species definitions at the root, species spawn-zone assignments under placement, or monitored species names under early_stopping; keys/names must match the environment. | object; min properties: `1` | Required; — |
| `brains` | Named controller templates; a template name is separate from its algorithm type. | object; min properties: `1` | Required; — |
| `experiment` | Run schedule, seeding, model chaining, metric cadence and checkpoint behavior. | object | Required; — |
| `recording` | Optional replay cadence, chunk format/compression and sampled brain diagnostics. | object | Optional; — |
| `placement` | Optional founder zones and boundary-biased resource placement for an enclosure. | object | Optional; — |
| `day_night` | Optional repeating sunlight cycle; omitted or disabled leaves static body values and lighting. | object | Optional; — |

## `config.terrain (oneOf variant 1)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `map_version` | Terrain map format version, independent of environment schema_version. | value; exact: `1` | Required; — |
| `name` | Stable environment or map name; experiment folder names use the environment name. | string; min length: `1` | Required; — |
| `width` | Grid width in cells; must match the map width when terrain is supplied. | integer; min: `3`; max: `256` | Required; — |
| `height` | Grid height in cells; must match the map height when terrain is supplied. | integer; min: `3`; max: `256` | Required; — |
| `rows` | Equal-width strings of terrain symbols, one per map row. | array; min items: `3`; max items: `256`; items: string; min length: `3`; max length: `256` | Required; — |
| `legend` | Map symbol definitions for elevation, blocking, water depth and flow. | object; min properties: `1`; key constraints: `{"pattern": "^[ -~]$"}` | Required; — |

## `config.terrain (oneOf variant 1).legend.<name>`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `elevation` | Absolute tile elevation used for climbing differences and rendering. | number; min: `0`; max: `100` | Optional; — |
| `blocked` | Whether this terrain symbol contributes a blocking obstacle. | boolean | Optional; — |
| `water_depth` | Water depth at this tile; zero means dry land. | number; min: `0`; max: `10` | Optional; — |
| `flow` | Water-current vector [dx, dy] in grid cells per step; fractional drift is retained. | array; min items: `2`; max items: `2`; items: number; min: `-4`; max: `4` | Optional; — |

## `config.terrain (oneOf variant 2)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `map_file` | Path to a terrain map JSON; relative paths resolve from the current working directory first, then bundled/project assets. | string; min length: `1` | Required; — |

## `config.world`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `width` | Grid width in cells; must match the map width when terrain is supplied. | integer; min: `3`; max: `256` | Required; — |
| `height` | Grid height in cells; must match the map height when terrain is supplied. | integer; min: `3`; max: `256` | Required; — |
| `boundary` | World boundary behavior; current worlds use closed edges. | value; exact: `"closed"` | Required; — |
| `max_steps` | Maximum simulation steps per episode, not wall-clock seconds. | integer; min: `1`; max: `10000000` | Required; — |

## `config.obstacles`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `generation` | Choose scattered random obstacle generation or an explicit obstacle list. | value; choices: `["scattered", "explicit"]` | Required; — |
| `occupied_fraction` | Fraction of world cells selected as scattered obstacles. | number; min: `0`; max: `0.8` | Required; — |
| `explicit_cells` | Obstacle coordinates as [x, y] pairs; used by explicit generation. | array; unique items: `true`; items: array; min items: `2`; max items: `2`; items: integer; min: `0`; max: `255` | Required; — |

## `config.resources`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `patch_count` | Number of resource patches placed in free eligible cells. | integer; min: `0`; max: `65536` | Required; — |
| `capacity_per_patch` | Maximum food quantity held by each patch. | number; min: `0.001` | Required; — |
| `initial_quantity_per_patch` | Starting food quantity in each patch; cannot exceed capacity. | number; min: `0` | Required; — |
| `growth_per_step` | Food quantity regenerated per patch per simulation step, capped by capacity. | number; min: `0` | Required; — |

## `config.species.<name>`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `locomotion` | Optional swimming, water exposure and elevation movement rules. | object | Optional; — |
| `display` | Optional stable species/sex colors for the viewer, separate from physics and brain signatures. | object | Optional; — |
| `initial_brains` | Optional female and male initial champion selectors resolved independently. | object | Optional; — |
| `champion_selection_by_sex` | Female/male champion criterion overrides within this exact species. | object | Optional; — |
| `initial_brain` | Species-wide champion selector used when a sex-specific initializer does not override it. | variant | Optional; — |
| `initial_population` | Number of founder animals of this species. | integer; min: `0`; max: `10000` | Required; — |
| `diet` | Resource-eating animals consume food patches; prey-eating animals hunt resource-eating animals across species. | value; choices: `["resource", "prey"]` | Required; — |
| `allowed_actions` | Ordered action names available to this species; order is part of saved-model compatibility. | array; min items: `1`; unique items: `true`; items: value; choices: `["move_forward", "turn_left", "turn_right", "rest", "eat", "attack", "swim", "reproduce", "move_backward", "strafe_left", "strafe_right"]` | Required; — |
| `body` | Physical energy, health, sight and diet-specific feeding/combat parameters. | variant | Required; — |
| `brain` | Key of the brain template used to initialize this species; each animal receives independent live state. | string | Required; — |
| `champion_selection` | Default criterion and scoring parameters used to curate this species' champions. | object | Optional; — |
| `learning_rate_multiplier` | Multiply the selected template learning rate for this species. | number; min: `0.01`; max: `10` | Optional; `1.0` |
| `rewards` | Coefficients converting physical events, energy costs and failed choices into learning feedback. | object | Required; — |
| `mutation` | Rare newborn-only weight perturbation settings; does not mutate the whole species. | object | Optional; — |
| `reproduction` | Same-species opposite-sex eligibility, consent window, maturity, cost and population limits. | object | Optional; — |
| `circadian` | Species-specific night multipliers, used only when the world day_night cycle is enabled. | object | Optional; — |

## `config.species.<name>.locomotion`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `swim_min_health` | Minimum health required for active swimming under terrain rules. | number; min: `0` | Optional; — |
| `swim_energy_cost` | Additional energy effort for a swim action. | number; min: `0`; max: `100` | Optional; — |
| `water_energy_cost` | Additional energy charged for occupying water, scaled by depth. | number; min: `0`; max: `100` | Optional; — |
| `drowning_damage` | Depth-scaled health damage under the configured unsafe-water conditions. | number; min: `0`; max: `100` | Optional; — |
| `uphill_energy_cost` | Additional energy charged per positive elevation difference traversed. | number; min: `0`; max: `100` | Optional; — |
| `max_climb` | Maximum permitted positive elevation change for ordinary movement. | number; min: `0`; max: `100` | Optional; — |

## `config.species.<name>.display`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `color` | Female/base species color as a six-digit hex string. | string; pattern: `"^#[0-9A-Fa-f]{6}$"` | Optional; — |
| `male_color` | Male species color as a six-digit hex string; male body outlines remain a separate cue. | string; pattern: `"^#[0-9A-Fa-f]{6}$"` | Optional; — |

## `config.species.<name>.initial_brains`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `female` | Override for this species' female initialization or champion-selection group. | variant | Optional; — |
| `male` | Override for this species' male initialization or champion-selection group. | variant | Optional; — |

## `config.species.<name>.initial_brains.female (anyOf variant 2) (oneOf variant 1)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `path` | Path to archived champion metadata, not a replay or exact checkpoint file. | string; min length: `1` | Required; — |

## `config.species.<name>.initial_brains.female (anyOf variant 2) (oneOf variant 2)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `selection` | Shared-library selection strategy for the requested champion metric. | value; choices: `["best", "latest"]` | Required; — |
| `metric` | Champion ranking criterion to select from the requested library/archive context. | value; choices: `["health", "kills", "lifespan", "food", "balanced", "lineage"]` | Required; — |

## `config.species.<name>.initial_brains.male (anyOf variant 2) (oneOf variant 1)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `path` | Path to archived champion metadata, not a replay or exact checkpoint file. | string; min length: `1` | Required; — |

## `config.species.<name>.initial_brains.male (anyOf variant 2) (oneOf variant 2)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `selection` | Shared-library selection strategy for the requested champion metric. | value; choices: `["best", "latest"]` | Required; — |
| `metric` | Champion ranking criterion to select from the requested library/archive context. | value; choices: `["health", "kills", "lifespan", "food", "balanced", "lineage"]` | Required; — |

## `config.species.<name>.champion_selection_by_sex`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `female` | Override for this species' female initialization or champion-selection group. | value; choices: `["health", "kills", "lifespan", "food", "balanced", "lineage"]` | Optional; — |
| `male` | Override for this species' male initialization or champion-selection group. | value; choices: `["health", "kills", "lifespan", "food", "balanced", "lineage"]` | Optional; — |

## `config.species.<name>.initial_brain (oneOf variant 1)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `path` | Path to archived champion metadata, not a replay or exact checkpoint file. | string; min length: `1` | Required; — |

## `config.species.<name>.initial_brain (oneOf variant 2)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `selection` | Shared-library selection strategy for the requested champion metric. | value; choices: `["best", "latest"]` | Required; — |
| `metric` | Champion ranking criterion to select from the requested library/archive context. | value; choices: `["health", "kills", "lifespan", "food", "balanced", "lineage"]` | Required; — |

## `config.species.<name>.body (oneOf variant 1)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `initial_energy` | Energy granted to a founder or newborn of this species. | number; min: `0` | Required; — |
| `max_energy` | Upper bound on stored animal energy. | number; >: `0` | Required; — |
| `max_health` | Initial health and health cap; current reproduction eligibility uses absolute health units. | number; >: `0` | Required; — |
| `living_cost` | Energy charged each step regardless of selected action, before circadian scaling. | number; min: `0` | Required; — |
| `movement_cost` | Additional energy charged for movement actions, before circadian scaling and geography surcharges. | number; min: `0` | Required; — |
| `starvation_damage` | Health lost per step while energy is exhausted. | number; min: `0` | Required; — |
| `vision_radius` | Base radial sight distance in grid cells before optional sunlight and water modifiers. | integer; min: `1`; max: `256` | Required; — |
| `bite_size` | Maximum food units consumed by one successful eat action. | number; min: `0` | Required; — |
| `energy_per_food_unit` | Energy granted per consumed food unit, subject to the animal energy cap. | number; min: `0` | Required; — |
| `water_vision_bonus` | Maximum additional sight cells permitted along open-water rays. | integer; min: `0`; max: `32` | Optional; — |

## `config.species.<name>.body (oneOf variant 2)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `initial_energy` | Energy granted to a founder or newborn of this species. | number; min: `0` | Required; — |
| `max_energy` | Upper bound on stored animal energy. | number; >: `0` | Required; — |
| `max_health` | Initial health and health cap; current reproduction eligibility uses absolute health units. | number; >: `0` | Required; — |
| `living_cost` | Energy charged each step regardless of selected action, before circadian scaling. | number; min: `0` | Required; — |
| `movement_cost` | Additional energy charged for movement actions, before circadian scaling and geography surcharges. | number; min: `0` | Required; — |
| `starvation_damage` | Health lost per step while energy is exhausted. | number; min: `0` | Required; — |
| `vision_radius` | Base radial sight distance in grid cells before optional sunlight and water modifiers. | integer; min: `1`; max: `256` | Required; — |
| `attack_damage` | Health damage contributed by an attack; simultaneous attackers share kill credit rules. | number; min: `0` | Required; — |
| `attack_cost` | Additional energy cost of selecting attack. | number; min: `0` | Required; — |
| `energy_per_kill` | Energy supplied by a kill, subject to the killer energy cap and shared-kill accounting. | number; min: `0` | Required; — |
| `water_vision_bonus` | Maximum additional sight cells permitted along open-water rays. | integer; min: `0`; max: `32` | Optional; — |

## `config.species.<name>.champion_selection`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `criterion` | Champion ranking objective; permitted values and role restrictions are validated separately. | value; choices: `["health", "kills", "lifespan", "food", "balanced", "lineage"]` | Required; — |
| `survival_steps_per_kill` | Age normalization for the survival bonus in balanced predator selection. | number; min: `1`; max: `10000000` | Optional; `300` |
| `max_survival_bonus` | Maximum survival bonus added to kills by balanced predator scoring. | number; min: `0`; max: `100` | Optional; `2` |
| `lineage_min_age_steps` | Lineage maturity gate; defaults to reproduction.min_age_steps (at least one step). | integer; min: `1`; max: `10000000` | Optional; — |

## `config.species.<name>.rewards`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `per_energy_gained` | Reward per actual energy unit gained from feeding/hunting. | number; min: `-100000`; max: `100000` | Required; — |
| `per_energy_cost` | Reward coefficient on charged energy demand; normally negative to penalize effort. | number; min: `-100000`; max: `0` | Optional; `0` |
| `failed_action` | Additional reward/penalty when an attempted action fails. | number; min: `-100000`; max: `0` | Optional; `0` |
| `per_offspring` | Reward credited to a successful parent for producing offspring. | number; min: `0`; max: `100000` | Optional; `0` |
| `per_damage_taken` | Reward coefficient per health damage received; normally negative. | number; min: `-100000`; max: `100000` | Required; — |
| `death` | Terminal reward/penalty when the animal dies. | number; min: `-100000`; max: `100000` | Required; — |
| `per_damage_dealt` | Optional reward per actual damage point credited to this attacker; omitted means zero. | number; min: `0`; max: `100000` | Optional; `0` |

## `config.species.<name>.mutation`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `enabled` | Enable the containing optional feature; meaning and required companion fields depend on that feature. | boolean | Required; — |
| `probability_per_birth` | Probability of attempting a mutation on an individual newborn. | number; min: `0`; max: `1` | Required; — |
| `max_changed_weights` | Maximum distinct eligible matrix weights selected in a successful mutation event. | integer; min: `1`; max: `1024` | Required; — |
| `noise_std` | Standard deviation of additive Gaussian mutation noise. | number; min: `0`; max: `1` | Required; — |

## `config.species.<name>.reproduction`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `enabled` | Enable the containing optional feature; meaning and required companion fields depend on that feature. | boolean | Required; — |
| `mate_radius` | Proximal mate distance in cells, combined with same-species, physical-route and birth-space checks. | integer; min: `1`; max: `10` | Optional; — |
| `intent_steps` | Consent duration including the reproduce step. Rest preserves intent; other actions cancel it. Default 1 retains same-step consent. | integer; min: `1`; max: `128` | Optional; — |
| `health_threshold` | Each parent's health must be strictly above this absolute value to reproduce. | number; min: `0` | Required; — |
| `min_age_steps` | Central age in steps at which an animal can first become eligible to reproduce. | integer; min: `1`; max: `10000000` | Required; — |
| `cooldown_steps` | Central number of age steps required between successful reproductions. | integer; min: `1`; max: `10000000` | Required; — |
| `age_deviation_pct` | First-maturity window half-width as a percentage of min_age_steps; takes precedence over legacy absolute deviation. | number; min: `0`; max: `100` | Optional; — |
| `age_deviation_steps` | Legacy first-maturity window half-width in absolute steps; prefer age_deviation_pct. | integer; min: `0`; max: `10000000` | Optional; — |
| `cooldown_deviation_pct` | Cooldown window half-width as a percentage of cooldown_steps; takes precedence over legacy absolute deviation. | number; min: `0`; max: `100` | Optional; — |
| `cooldown_deviation_steps` | Legacy cooldown window half-width in absolute steps; prefer cooldown_deviation_pct. | integer; min: `0`; max: `10000000` | Optional; — |
| `min_energy` | Minimum current energy required from each eligible parent. | number; min: `0` | Required; — |
| `energy_cost` | Energy charged to each successful parent at reproduction; not a full health reset. | number; min: `0.001` | Required; — |
| `max_population` | Living population cap for this exact species; includes founders and children. | integer; min: `0`; max: `10000` | Required; — |

## `config.species.<name>.circadian`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `night_vision_radius_multiplier` | Night-time base sight multiplier; blends toward one at midday, then rounds to a valid grid radius. | number; min: `0.05`; max: `4` | Optional; — |
| `night_living_cost_multiplier` | Night-time living-cost multiplier; blends toward one with sunlight. | number; min: `0.05`; max: `4` | Optional; — |
| `night_movement_cost_multiplier` | Night-time movement-cost multiplier; blends toward one with sunlight. | number; min: `0.05`; max: `4` | Optional; — |

## `config.brains.<name> (q_network)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"q_network"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `1`; items: integer; min: `1`; max: `256` | Required; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `batch_size` | Samples per optimizer batch; exact interpretation depends on replay or rollout algorithm. | integer; min: `1`; max: `1000000` | Required; — |
| `replay_capacity` | Maximum stored experience transitions for replay-based learners. | integer; min: `1`; max: `1000000` | Required; — |
| `learning_starts_after` | Minimum collected experiences before replay-based optimization begins. | integer; min: `1`; max: `1000000` | Required; — |
| `train_every_steps` | Decision-step interval between replay updates or GRPO trial requests, depending on algorithm. | integer; min: `1`; max: `1000000` | Required; — |
| `target_update_every_training_steps` | Optimizer-update interval between target-network synchronizations. | integer; min: `1`; max: `1000000` | Required; — |
| `exploration` | Epsilon-greedy action-exploration schedule for value-based learners. | object | Required; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (q_network).exploration`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `initial` | Initial epsilon random-action probability. | number; min: `0`; max: `1` | Required; — |
| `minimum` | Minimum epsilon random-action probability. | number; min: `0`; max: `1` | Required; — |
| `decay_per_decision` | Multiplicative epsilon decay per decision, bounded by minimum. | number; min: `0`; max: `1` | Required; — |

## `config.brains.<name> (ppo)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"ppo"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `network` | Neural architecture used by this controller; each animal has its own live parameters. | object | Required; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `0.1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `gae_lambda` | Lambda controlling the temporal bias/variance balance of generalized advantage estimates. | number; min: `0`; max: `1` | Required; — |
| `clip_ratio` | PPO/GRPO policy-ratio clipping width around one. | number; min: `0.001`; max: `1` | Required; — |
| `entropy_coefficient` | Weight of the entropy term encouraging action diversity in the policy objective. | number; min: `0`; max: `1` | Required; — |
| `value_coefficient` | Weight of the value-prediction loss for actor-critic methods. | number; min: `0`; max: `10` | Required; — |
| `rollout_steps` | Collected transition count at which an on-policy rollout update is triggered. | integer; min: `1`; max: `4096` | Required; — |
| `epochs` | Optimizer passes over a collected PPO/GRPO rollout or trial group. | integer; min: `1`; max: `20` | Required; — |
| `batch_size` | Samples per optimizer batch; exact interpretation depends on replay or rollout algorithm. | integer; min: `1`; max: `4096` | Required; — |
| `max_grad_norm` | Global gradient-norm clipping threshold before optimizer updates. | number; min: `1e-05`; max: `100` | Required; — |
| `normalize_advantages` | Whether the supported learner normalizes its collected advantages. | boolean | Required; — |
| `target_kl` | Optional KL early-stop threshold during repeated policy optimization. | number; min: `0`; max: `1` | Optional; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (ppo).network`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mlp"` | Required; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `activation` | Hidden-layer activation function; current supported architecture uses tanh. | value; exact: `"tanh"` | Optional; — |

## `config.brains.<name> (a2c)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"a2c"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `network` | Neural architecture used by this controller; each animal has its own live parameters. | object | Required; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `0.1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `entropy_coefficient` | Weight of the entropy term encouraging action diversity in the policy objective. | number; min: `0`; max: `1` | Required; — |
| `max_grad_norm` | Global gradient-norm clipping threshold before optimizer updates. | number; min: `1e-05`; max: `100` | Required; — |
| `gae_lambda` | Lambda controlling the temporal bias/variance balance of generalized advantage estimates. | number; min: `0`; max: `1` | Required; — |
| `value_coefficient` | Weight of the value-prediction loss for actor-critic methods. | number; min: `0`; max: `10` | Required; — |
| `rollout_steps` | Collected transition count at which an on-policy rollout update is triggered. | integer; min: `1`; max: `4096` | Required; — |
| `normalize_advantages` | Whether the supported learner normalizes its collected advantages. | boolean | Required; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (a2c).network`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mlp"` | Required; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `activation` | Hidden-layer activation function; current supported architecture uses tanh. | value; exact: `"tanh"` | Optional; — |

## `config.brains.<name> (reinforce)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"reinforce"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `network` | Neural architecture used by this controller; each animal has its own live parameters. | object | Required; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `0.1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `entropy_coefficient` | Weight of the entropy term encouraging action diversity in the policy objective. | number; min: `0`; max: `1` | Required; — |
| `max_grad_norm` | Global gradient-norm clipping threshold before optimizer updates. | number; min: `1e-05`; max: `100` | Required; — |
| `max_episode_steps` | REINFORCE trajectory flush horizon; align with the episode schedule as appropriate. | integer; min: `1`; max: `10000000` | Required; — |
| `baseline_decay` | Moving-baseline decay factor used by REINFORCE. | number; min: `0`; max: `1` | Required; — |
| `normalize_advantages` | Whether the supported learner normalizes its collected advantages. | boolean | Required; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (reinforce).network`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mlp"` | Required; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `activation` | Hidden-layer activation function; current supported architecture uses tanh. | value; exact: `"tanh"` | Optional; — |

## `config.brains.<name> (grpo)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"grpo"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `network` | Neural architecture used by this controller; each animal has its own live parameters. | object | Required; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `0.1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `entropy_coefficient` | Weight of the entropy term encouraging action diversity in the policy objective. | number; min: `0`; max: `1` | Required; — |
| `max_grad_norm` | Global gradient-norm clipping threshold before optimizer updates. | number; min: `1e-05`; max: `100` | Required; — |
| `clip_ratio` | PPO/GRPO policy-ratio clipping width around one. | number; min: `0.001`; max: `1` | Required; — |
| `kl_coefficient` | GRPO penalty weight for divergence from its reference policy. | number; min: `0`; max: `10` | Required; — |
| `group_size` | Number of matched GRPO trials compared at an update opportunity. | integer; min: `2`; max: `8` | Required; — |
| `rollout_horizon` | Simulation-step horizon of each GRPO trial. | integer; min: `1`; max: `16` | Required; — |
| `train_every_steps` | Decision-step interval between replay updates or GRPO trial requests, depending on algorithm. | integer; min: `1`; max: `1000000` | Required; — |
| `epochs` | Optimizer passes over a collected PPO/GRPO rollout or trial group. | integer; min: `1`; max: `20` | Required; — |
| `batch_size` | Samples per optimizer batch; exact interpretation depends on replay or rollout algorithm. | integer; min: `1`; max: `128` | Required; — |
| `target_kl` | Optional KL early-stop threshold during repeated policy optimization. | number; min: `0`; max: `1` | Optional; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (grpo).network`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mlp"` | Required; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `activation` | Hidden-layer activation function; current supported architecture uses tanh. | value; exact: `"tanh"` | Optional; — |

## `config.brains.<name> (recurrent_ppo)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"recurrent_ppo"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `network` | Neural architecture used by this controller; each animal has its own live parameters. | object | Required; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `0.1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `gae_lambda` | Lambda controlling the temporal bias/variance balance of generalized advantage estimates. | number; min: `0`; max: `1` | Required; — |
| `clip_ratio` | PPO/GRPO policy-ratio clipping width around one. | number; min: `0.001`; max: `1` | Required; — |
| `entropy_coefficient` | Weight of the entropy term encouraging action diversity in the policy objective. | number; min: `0`; max: `1` | Required; — |
| `value_coefficient` | Weight of the value-prediction loss for actor-critic methods. | number; min: `0`; max: `10` | Required; — |
| `rollout_steps` | Collected transition count at which an on-policy rollout update is triggered. | integer; min: `1`; max: `4096` | Required; — |
| `epochs` | Optimizer passes over a collected PPO/GRPO rollout or trial group. | integer; min: `1`; max: `20` | Required; — |
| `batch_size` | Samples per optimizer batch; exact interpretation depends on replay or rollout algorithm. | integer; min: `1`; max: `4096` | Required; — |
| `max_grad_norm` | Global gradient-norm clipping threshold before optimizer updates. | number; min: `1e-05`; max: `100` | Required; — |
| `normalize_advantages` | Whether the supported learner normalizes its collected advantages. | boolean | Required; — |
| `target_kl` | Optional KL early-stop threshold during repeated policy optimization. | number; min: `0`; max: `1` | Optional; — |
| `recurrent_size` | GRU memory width for recurrent PPO; part of model compatibility. | integer; min: `1`; max: `128` | Required; — |
| `sequence_length` | Maximum recurrent training sequence length. | integer; min: `1`; max: `128` | Required; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (recurrent_ppo).network`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mlp"` | Required; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `activation` | Hidden-layer activation function; current supported architecture uses tanh. | value; exact: `"tanh"` | Optional; — |

## `config.brains.<name> (maskable_ppo)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"maskable_ppo"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `network` | Neural architecture used by this controller; each animal has its own live parameters. | object | Required; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `0.1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `gae_lambda` | Lambda controlling the temporal bias/variance balance of generalized advantage estimates. | number; min: `0`; max: `1` | Required; — |
| `clip_ratio` | PPO/GRPO policy-ratio clipping width around one. | number; min: `0.001`; max: `1` | Required; — |
| `entropy_coefficient` | Weight of the entropy term encouraging action diversity in the policy objective. | number; min: `0`; max: `1` | Required; — |
| `value_coefficient` | Weight of the value-prediction loss for actor-critic methods. | number; min: `0`; max: `10` | Required; — |
| `rollout_steps` | Collected transition count at which an on-policy rollout update is triggered. | integer; min: `1`; max: `4096` | Required; — |
| `epochs` | Optimizer passes over a collected PPO/GRPO rollout or trial group. | integer; min: `1`; max: `20` | Required; — |
| `batch_size` | Samples per optimizer batch; exact interpretation depends on replay or rollout algorithm. | integer; min: `1`; max: `4096` | Required; — |
| `max_grad_norm` | Global gradient-norm clipping threshold before optimizer updates. | number; min: `1e-05`; max: `100` | Required; — |
| `normalize_advantages` | Whether the supported learner normalizes its collected advantages. | boolean | Required; — |
| `target_kl` | Optional KL early-stop threshold during repeated policy optimization. | number; min: `0`; max: `1` | Optional; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (maskable_ppo).network`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mlp"` | Required; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `activation` | Hidden-layer activation function; current supported architecture uses tanh. | value; exact: `"tanh"` | Optional; — |

## `config.brains.<name> (mappo)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mappo"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `network` | Neural architecture used by this controller; each animal has its own live parameters. | object | Required; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `0.1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `gae_lambda` | Lambda controlling the temporal bias/variance balance of generalized advantage estimates. | number; min: `0`; max: `1` | Required; — |
| `clip_ratio` | PPO/GRPO policy-ratio clipping width around one. | number; min: `0.001`; max: `1` | Required; — |
| `entropy_coefficient` | Weight of the entropy term encouraging action diversity in the policy objective. | number; min: `0`; max: `1` | Required; — |
| `value_coefficient` | Weight of the value-prediction loss for actor-critic methods. | number; min: `0`; max: `10` | Required; — |
| `rollout_steps` | Collected transition count at which an on-policy rollout update is triggered. | integer; min: `1`; max: `4096` | Required; — |
| `epochs` | Optimizer passes over a collected PPO/GRPO rollout or trial group. | integer; min: `1`; max: `20` | Required; — |
| `batch_size` | Samples per optimizer batch; exact interpretation depends on replay or rollout algorithm. | integer; min: `1`; max: `4096` | Required; — |
| `max_grad_norm` | Global gradient-norm clipping threshold before optimizer updates. | number; min: `1e-05`; max: `100` | Required; — |
| `normalize_advantages` | Whether the supported learner normalizes its collected advantages. | boolean | Required; — |
| `target_kl` | Optional KL early-stop threshold during repeated policy optimization. | number; min: `0`; max: `1` | Optional; — |
| `critic_hidden_layers` | MAPPO centralized critic widths; the local actor remains a separate route. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (mappo).network`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mlp"` | Required; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `activation` | Hidden-layer activation function; current supported architecture uses tanh. | value; exact: `"tanh"` | Optional; — |

## `config.brains.<name> (qrdqn)`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"qrdqn"` | Required; — |
| `observation_version` | Input-layout version; affects feature meanings, width and saved-model compatibility. | value; choices: `[1, 2, 3, 4]`; max: `4` | Optional; — |
| `learning_rate` | Base optimizer update rate before the species multiplier. | number; min: `1e-06`; max: `0.1` | Required; — |
| `discount_factor` | Future-reward discount gamma, applied per simulation transition. | number; min: `0`; max: `1` | Required; — |
| `batch_size` | Samples per optimizer batch; exact interpretation depends on replay or rollout algorithm. | integer; min: `1`; max: `1000000` | Required; — |
| `replay_capacity` | Maximum stored experience transitions for replay-based learners. | integer; min: `1`; max: `1000000` | Required; — |
| `learning_starts_after` | Minimum collected experiences before replay-based optimization begins. | integer; min: `1`; max: `1000000` | Required; — |
| `train_every_steps` | Decision-step interval between replay updates or GRPO trial requests, depending on algorithm. | integer; min: `1`; max: `1000000` | Required; — |
| `target_update_every_training_steps` | Optimizer-update interval between target-network synchronizations. | integer; min: `1`; max: `1000000` | Required; — |
| `exploration` | Epsilon-greedy action-exploration schedule for value-based learners. | object | Required; — |
| `network` | Neural architecture used by this controller; each animal has its own live parameters. | object | Required; — |
| `num_quantiles` | Number of return quantiles per action in QR-DQN. | integer; min: `2`; max: `128` | Required; — |
| `huber_threshold` | Huber transition threshold for quantile-regression loss. | number; min: `1e-05`; max: `100` | Required; — |
| `max_grad_norm` | Global gradient-norm clipping threshold before optimizer updates. | number; min: `1e-05`; max: `100` | Required; — |
| `observation_top_k` | Maximum visible target slots per food/prey/predator category; unused slots are zero-filled. | integer; min: `1`; max: `16` | Optional; `1` |

## `config.brains.<name> (qrdqn).exploration`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `initial` | Initial epsilon random-action probability. | number; min: `0`; max: `1` | Required; — |
| `minimum` | Minimum epsilon random-action probability. | number; min: `0`; max: `1` | Required; — |
| `decay_per_decision` | Multiplicative epsilon decay per decision, bounded by minimum. | number; min: `0`; max: `1` | Required; — |

## `config.brains.<name> (qrdqn).network`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `type` | Algorithm identifier for a brain, or architecture identifier inside network; allowed values depend on the object. | value; exact: `"mlp"` | Required; — |
| `hidden_layers` | Hidden processing-layer widths in order; location differs between legacy Q-network and network-based templates. | array; min items: `1`; max items: `4`; items: integer; min: `1`; max: `256` | Required; — |
| `activation` | Hidden-layer activation function; current supported architecture uses tanh. | value; exact: `"tanh"` | Optional; — |

## `config.experiment`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `runs` | Maximum number of episodes in this experiment; early stopping may finish sooner. | integer; min: `1`; max: `1000000` | Required; — |
| `chain_best_brains` | Initialize later fresh episodes from matching retained champion groups when enabled. | boolean | Optional; `true` |
| `retain_brains_between_runs` | Legacy controller-retention compatibility flag; modern champion chaining is governed by chain_best_brains. | boolean | Required; — |
| `world_seed_policy` | Choose random, fixed or incrementing world seeds across runs; CLI seed overrides are recorded. | value; choices: `["random", "fixed", "increment"]` | Required; — |
| `record_metrics_every_steps` | Simulation-step interval for readable training metric records; final metrics are forced. | integer; min: `1`; max: `1000000` | Required; — |
| `checkpoint_every_runs` | Completed-run interval for intermediate exact checkpoints in the CLI; final state is saved separately. | integer; min: `1`; max: `1000000` | Required; — |
| `grpo_updates_per_step` | Maximum GRPO animal trial updates scheduled in one real-world step. | integer; min: `1`; max: `8` | Optional; — |
| `guard_lineage_chaining` | Retain the previous qualified matching-sex lineage brain when a new episode only provides an unqualified fallback. | boolean | Optional; `true` |
| `promote_only_improved` | Retain best-so-far species/sex/criterion champions; ties keep the incumbent. | boolean | Optional; `true` |
| `early_stopping` | Optional consecutive-run stagnation limit and monitored species. | object | Optional; — |

## `config.experiment.early_stopping`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `patience_runs` | Consecutive completed runs without a selected champion rank improvement; zero disables. | integer; min: `0`; max: `1000000` | Required; — |
| `species` | Configured species to monitor. Omit to watch all; any monitored sex/species improvement resets patience. | array; min items: `1`; unique items: `true`; items: string | Optional; — |

## `config.recording`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `enabled` | Enable the containing optional feature; meaning and required companion fields depend on that feature. | boolean | Required; — |
| `frame_every_steps` | World-step interval between recorded playback frames; sparse frames omit intermediate events. | integer; min: `1`; max: `1000000` | Required; — |
| `frames_per_chunk` | Maximum committed replay frames per independently loadable chunk. | integer; min: `1`; max: `256` | Required; — |
| `encoding` | Replay representation: legacy full snapshots, JSON deltas or structured binary according to allowed values. | value; choices: `["binary", "delta", "snapshot"]` | Optional; `"binary"` |
| `compression` | Replay transport compression; independent of the logical encoding. | value; choices: `["gzip", "xz", "none"]` | Optional; `"xz"` |
| `compression_level` | Compression effort level used for the selected replay compressor. | integer; min: `1`; max: `9` | Optional; `6` |
| `diagnostics_every_steps` | Step interval for sampled brain inputs/outputs/activations, with chunk-start/final capture rules; zero disables all diagnostic capture. | integer; min: `0`; max: `1000000` | Required; — |
| `record_activations` | Include compact sampled hidden activations for actual-animal playback; does not add weights. | boolean | Optional; `true` |

## `config.placement`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `enclosure_bounds` | Rectangle [left, top, right, bottom] defining inside, outside and boundary zones. | array; min items: `4`; max items: `4`; items: integer; min: `0`; max: `255` | Required; — |
| `species` | Named species definitions at the root, species spawn-zone assignments under placement, or monitored species names under early_stopping; keys/names must match the environment. | object | Required; — |
| `resources` | Shared food patch placement, capacity and regrowth. | object | Required; — |

## `config.placement.resources`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `boundary_fraction` | Fraction of resource patches allocated near the enclosure boundary. | number; min: `0`; max: `1` | Required; — |
| `inside_fraction` | Fraction of boundary-near resource patches placed on the inside. | number; min: `0`; max: `1` | Required; — |
| `boundary_width` | Width in cells of the boundary-near resource band. | integer; min: `1`; max: `32` | Required; — |

## `config.day_night`

| Field | Definition | Type and constraints | Required / schema default |
| --- | --- | --- | --- |
| `enabled` | Enable the containing optional feature; meaning and required companion fields depend on that feature. | boolean | Required; — |
| `cycle_steps` | Number of world steps in a full simulated day, independent of playback speed. | integer; min: `4`; max: `10000000` | Required; — |
| `start_phase` | Initial cycle fraction: zero midnight, 0.25 sunrise, 0.5 midday, 0.75 sunset. | number; min: `0`; max: `1` | Optional; — |

Field entries documented: **317**.
