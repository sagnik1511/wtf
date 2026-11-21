# 1. Configure a small world

[Tutorial hub](../tutorial.md) · [Next: run and record](02-run-record-analyze.md)

Run commands from the checkout root after installing `requirements.txt` into
`.venv`, as in [setup](../running.md). The examples use current source APIs;
installation from PyPI is not yet part of this workflow.

Load a complete preset instead of constructing partial configuration dictionaries:

```python
from ecosystem.config import load, validate

config = load("environments/maskable_ppo.json")
config["seed"] = 100
config["world"]["max_steps"] = 200
config["experiment"].update(runs=1, world_seed_policy="increment")
validate(config)
print({name: spec["initial_population"] for name, spec in config["species"].items()})
```

Changes affect this in-memory object, not the preset file. A fixed seed and an
incrementing world-seed policy make the one-run example repeatable within the
supported runtime. Species still have independent controllers; a shared brain
template does not share live animal weights.

| Choice | Meaning |
| --- | --- |
| `maskable_ppo.json` | Prey and predator masked PPO with current supported configuration |
| `day_night.json` | Adds opt-in sunlight and species circadian effects |
| `max_steps: 200` | Short workflow demonstration; some animals will not reach reproduction maturity |
| `max_steps: 1000` | More time for maturity and interaction, with more compute/output |
| `runs: 1` | One episode; use the CLI for multi-run campaigns and champion chaining |

Presets contain paired reproduction, rare birth mutation and supported physical
rules. You do not need to widen mating distance for this tutorial. Birth requires
physical eligibility and consenting decisions; it is not guaranteed. Consult
[configuration](../configuration.md), [observations](../observations.md) and
[day–night](../day-night.md) before changing these contracts.

For definitions of every supported field and a map of all editable components,
see the [configuration reference](../configuration-reference.md) and
[customization handbook](../customization.md).
