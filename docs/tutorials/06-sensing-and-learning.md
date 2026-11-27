# Tutorial 6: Configure, train and inspect sensory learning

[Tutorial hub](../tutorial.md) · [Sensing handbook](../learning/features.md) · [Previous: compare saved policies](05-compare-saved-policies.md)

This walkthrough creates a complete local JSON, validates it, runs multiple
episodes and explains what to inspect. It uses current source with V7 support;
the published 0.0.2.9 package predates these additions. Follow
[source installation](../../INSTALL.md) if the options are unavailable.
No GPU, notebook or external ML framework is required.

## 1. Choose a feature profile

| Profile below | Sensory inputs | Learning additions |
| --- | --- | --- |
| `senses` | V7 + history + blended scent | Existing ordinary PPO only |
| `input` | Above + local/general/context evidence | Evidence is read as normal network inputs |
| `auxiliary` | Same inputs as `input` | Additional private outcome-prediction head |
| `late` | Same recorded evidence | Evidence bypasses physical trunk; private late adapter and auxiliary head enabled |

The last two profiles require ordinary PPO or masked PPO. Their objectives and
model signatures differ from the first two; they are not alternative display
settings. A2C, DQN, GRPO and the other learners can use the sensory/input features
without acquiring an unsupported auxiliary/late head.

This example starts from `ppo`, keeping its learner, architecture, geography,
health/energy costs, mating radii, mutual intent and rewards. Changing the profile
does not make reproduction automatic.

## 2. Generate a complete local configuration

Run this Python block from a writable working directory. It creates only an
ignored local environment JSON when used from the repository; it does not start
training. Set `PROFILE` to one row from the table and rerun to create another arm.

```python
import json
from pathlib import Path

from wtf import load_config
from wtf.config import validate

PROFILE = "input"  # senses, input, auxiliary, or late
if PROFILE not in {"senses", "input", "auxiliary", "late"}:
    raise ValueError("Unknown sensory-learning profile")

config = load_config("ppo")
config["name"] = f"ppo_{PROFILE}_sensing"
config["seed"] = 100
config["world"]["max_steps"] = 1500
config["experiment"].update(runs=5, world_seed_policy="increment")
config["recording"].update(
    enabled=True,
    profile="full",
    record_last_episodes=-1,
    diagnostics_every_steps=10,
    record_activations=True,
)
config["scent_trails"] = {"enabled": True, "lifetime_steps": 32}

if PROFILE == "senses":
    config.pop("consequence_learning", None)
else:
    config["consequence_learning"] = {
        "enabled": True,
        "max_contexts": 512,
        "horizon_steps": 16,
        "update_schedule": "episode",
        "persistent_action_totals": True,
    }

for template in config["brains"].values():
    template.update(
        observation_version=7,
        movement_history_steps=3,
        scent_observation={"mode": "blended", "radius": 2},
    )
    for key in (
        "social_observation",
        "consequence_observation",
        "consequence_scope",
        "auxiliary_consequence",
        "late_consequence",
    ):
        template.pop(key, None)
    if PROFILE != "senses":
        template.update(
            social_observation={"history_steps": 4, "max_neighbors": 8},
            consequence_observation=True,
            consequence_scope="both",
        )
    if PROFILE in {"auxiliary", "late"}:
        if template["type"] not in {"ppo", "maskable_ppo"}:
            raise ValueError("This objective needs PPO or masked PPO")
        template["auxiliary_consequence"] = {"coefficient": 0.1}
    if PROFILE == "late":
        template["late_consequence"] = {"max_adjustment": 2.0}

# Resolve each exact ordered action contract after configuration edits.
for species in config["species"].values():
    config["brains"][species["brain"]]["observation_actions"] = list(
        species["allowed_actions"]
    )
validate(config)

path = Path("environments/local") / f"{config['name']}.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2) + "\n")
print(f"Validated {PROFILE} profile: {path}")
```

Validation rejects a V7 template shared by species with different ordered action
lists. The maintained PPO preset already has separate templates. If adapting
this recipe to your own multi-species configuration, split such templates first;
assigning the last species' action order is not a valid workaround.

The coefficient 0.1 and adjustment bound 2.0 are supported starting values,
not established universal optima. Read [the calibration scope](../calibration.md)
before retuning physiology or interpreting a profile comparison as better biology.

## 3. Run multiple episodes and retain playback

For `PROFILE = "input"`, the generated path is:

```sh
.venv/bin/python -m wtf run environments/local/ppo_input_sensing.json \
  --runs 5 --seed 100 --seed-policy increment \
  --record-last-episodes -1 --progress
```

For another profile replace the filename with `ppo_senses_sensing.json`,
`ppo_auxiliary_sensing.json` or `ppo_late_sensing.json`.

The explicit recording selector retains all five episode playbacks. It costs
more disk than recording only the tail; choose `--record-last-episodes 2` for a
larger campaign when interim playback is unnecessary. Metrics, private learning,
champion selection and knowledge publication still run on uncaptured episodes.
A short `--steps 64` is useful for a technical smoke test, but it is shorter than
prey maturity in this preset and cannot assess full reproductive behavior.

With episode publication, the first fresh episode begins without shared
consequence evidence. Completed windows accumulate privately in staged species
tables, then become available when episode two starts. The policy itself still
learns from its own transitions during episode one.

## 4. Inspect the saved data

```sh
.venv/bin/python -m wtf view --results results
```

Open the printed viewer address and select the generated recording/episode.
Then select an animal and press Play.

| Viewer area | What to inspect | What it does not establish |
| --- | --- | --- |
| Observation panel | Targets, readiness, movement trend and V7 scent compass | Hidden target routes or future positions |
| Action predictions | How probabilities change at saved sample ticks | A guaranteed action or a conscious plan |
| Late-adjustment panel, `late` profile | Base probabilities, signed residuals, final probabilities | Shared species policy weights |
| Sampled hidden activity | Actual recorded per-animal activations | Activity at ticks that were never sampled |
| Species dashboard | Population, outcomes and cumulative consequence summaries | That every displayed average was an NN input |
| Movement heatmaps | Occupancy accumulated through the replay step | The scent field or a sensory channel |

If shared rows stay fixed within an episode, first check `update_schedule`.
If the graph holds a value between ticks, check diagnostic cadence. “Explain
saved brain” uses archived weights on a saved observation; it is a separate
analysis from actual recorded activity. Details are in the
[inspection guide](../learning/consequence-pipeline.md#inspect-the-result-in-playback).

## 5. Compare profiles without mixing contracts

Generate each profile using the same seed schedule and physical configuration,
then run each with the same number of episodes. Compare matched world seeds,
not expected identical trajectories: representation and objectives can change
initial networks and subsequent actions. Do not load one profile's weights
into a different signature to make the initialization look equivalent.

Report energy coverage, successful feeding/attacks, completed lifespan and
survivor censoring, deaths by cause, births, biological generations and mating
failure reasons. Also retain the champion selection/chaining settings: best-ever
champions can repeatedly restore older private weights while shared evidence
continues accumulating. Neither lower entropy nor more table samples alone
proves useful behavior.

Keep run-specific tables in results or the paper workspace. General package
guides should retain reusable mechanics and current calibration decisions.

## 6. Reuse weights or knowledge deliberately

A compatible consequence export can initialize population evidence while the
new campaign starts fresh private brains:

```sh
.venv/bin/python -m wtf run environments/local/ppo_input_sensing.json \
  --runs 5 --seed 100 --consequences-from /path/to/previous-run --progress
```

The source must have compatible consequence contracts. Adding `--brains-from`
is a separate choice and requires the complete matching species/sex model
signature, including optional heads. Exact checkpoint resume restores its
captured configuration and pending state; editing a new JSON does not alter it.
See [restoration](03-restore-and-view.md) and
[compatibility](../compatibility.md#compact-v7-models).

For read-only inference with no private/shared learning, use evaluation. Add
`--freeze-brains` when offspring must copy matching parent policies without
crossover/mutation. Scent, visual sightings and physical events can still evolve
while learning is disabled. See [training and inference](04-training-and-inference.md).
