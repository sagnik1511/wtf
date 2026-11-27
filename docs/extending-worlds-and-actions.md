# Custom geography, physics and animal actions

[Customization handbook](customization.md) · [Every JSON field](configuration-reference.md) · [Existing physics](physics.md)

The configuration supports new map shapes and per-species action subsets today.
A new terrain **mechanism** or action **implementation** is a source extension;
there is no external physics/action plugin registry. This guide separates those
cases so an accepted JSON string is never confused with implemented behavior.

## Create a map using existing geography

Use `terrain.rows` to lay out symbols and `terrain.legend` to define each symbol.
Existing tile properties are `blocked`, `elevation`, `water_depth` and `flow`.
New symbol characters can combine those properties without Python changes.
Coordinates are zero-based `[x, y]`; row index is y and character index is x.

This complete configuration recipe starts from a terrain-capable preset, so its
brain inputs and swimming actions already match the world:

```python
import json
from pathlib import Path
from wtf.config import load, validate

config = load("environments/river_crossing.json")
config["name"] = "my_river"
config["world"].update(width=20, height=14, max_steps=200)
config["resources"]["patch_count"] = 16
config["species"]["prey"]["initial_population"] = 6
config["species"]["predator"]["initial_population"] = 2
config["obstacles"].update(generation="explicit", explicit_cells=[])
config.pop("placement", None)
rows = ["." * 10 + "~" + "." * 9 for _ in range(14)]
config["terrain"] = {
    "map_version": 1,
    "name": "my_river_map",
    "width": 20,
    "height": 14,
    "rows": rows,
    "legend": {
        ".": {"elevation": 0},
        "~": {"water_depth": 1.2, "flow": [0, 0.4]},
    },
}
validate(config)
output = Path("environments/local/my_river.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(config, indent=2) + "\n")
```

Run it with `wtf run environments/local/my_river.json --runs 1 --seed 100`, or
use `python -m wtf.cli` from the checkout. Inline maps are portable with the
configuration. A `map_file` reference is also supported; relative references
first look in the working directory and then in packaged/project assets. Keep
world/map dimensions equal, enough dry/unblocked founder cells, compatible
terrain observation versions and valid placement zones. Elevation uses physical
height differences, not image pixels. Fractional currents accumulate drift.

See [the geography guide](geography.md#select-or-edit-a-map) for layout choices,
water rules, height costs, placement and observation details. New preset/map
files intended for installed users must also enter the package asset manifest;
local exploratory maps do not belong in the distribution.

## Give species different existing actions

`allowed_actions` is an **ordered** per-species list. You can remove an unwanted
choice, add a supported choice, or give different species different subsets.
Keep role restrictions (`eat` for resource-eaters, `attack` for prey-eaters),
terrain requirements and masked-policy fallback actions valid. For example:

```python
config["species"]["prey"]["allowed_actions"] = [
    "move_forward", "move_backward", "strafe_left", "strafe_right",
    "turn_left", "turn_right", "rest", "eat", "swim", "reproduce",
]
validate(config)
```

This snippet assumes the terrain-capable configuration above. It adds escape
choices but does not invent new movement semantics. Actions are described in
[the handbook](customization.md#action-definitions). The learner outputs one
score per ordered action; changing membership/order changes the saved-brain
signature and requires matching weights or a fresh start.

## Implement an entirely new action

Define its semantics before naming it: eligibility, simultaneous resolution,
energy/health costs, failed attempts, events and whether it is masked. Then trace
all affected paths:

| Integration point | What to update |
| --- | --- |
| Validation | `simulation/actions.py:ACTIONS` (re-exported by `config.py`) and diet/terrain-specific rules; `environment.schema.json` action enum/descriptions |
| Movement classification | `actions.py:WALK_OFFSETS`, `WALK_ACTIONS`, `MOVE_ACTIONS`, `ATTEMPT_ACTIONS` only if the new semantics belong in those groups |
| Physics | `world.py:step`, destination/block checks and interaction resolution; respect simultaneous occupancy and event order |
| Masking | `contextual_ppo.py:action_mask`; use pre-step physical eligibility, not future information or extra RNG draws |
| Feedback | Charge actual costs, record success/failure consistently, map events through generic reward accounting |
| Observation | Add a new version only if the action needs new input meanings/widths; update Python and JS layouts together |
| Model compatibility | Ordered action signature, output dimension, archives/checkpoints and existing-weight rejection |
| Native path | Review `_kernels*` action dispatch. Until equivalence is implemented, explicitly select the Python reference path for the unsupported action; do not let native code silently treat it as rest |
| Replay/analysis | Event labels and codecs if needed, telemetry action attempts/success, analyzer reconstruction and dashboard action choices |
| Viewer | Labels, inspector score graphs, action masks, event markers and any source-rendered appearance examples |

Add tests for valid/invalid attempts, blocked/conflicting movement, costs/rewards,
terminal steps, action masks, checkpoint continuation and Python/native parity.
Check birth inheritance and saved-model incompatibility when the output list
changes. Existing kernels are optional; building a new kernel is not required
to prototype a Python action, but selecting the correct fallback is required.

## Add a new environmental mechanism

Editing symbols, flow values and heights uses current physics. Temperature,
weather, cover, new resource types or selective food webs need additional logic:

1. Add strictly defined configuration/schema fields and defaults, then define
   initialization and update order. State whether the feature is opt-in.
2. Keep static map data in metadata. Put changing world state in checkpoint and,
   only where visualization needs it, compact replay records or tick-derived data.
3. Route effects through shared physical helpers such as effective bodies,
   movement checks and observations so every learner sees the same environment.
4. Include new effects in masks, effort/reward accounting and failed reasons.
   Preserve proximal reproduction unless changing it intentionally.
5. Version any new observation or binary-record contract. Extend both readers
   and inspector interpretation rather than reusing an old field for a new meaning.
6. Define cache keys/invalidation for changing visibility, geometry/resources
   and body modifiers. Keep trial-world caches private; test stationary observers
   after relevant changes, and exact continuation from mid-cycle checkpoints.
7. Add corresponding viewer rendering, theme/day-night contrast and random-seek
   checks. Keep training independent of the browser.

Use the existing [physics illustrations](physics.md), [runtime architecture](architecture.md),
[storage structures](replay-data-structures.md) and [compatibility policy](compatibility.md)
as contracts. Regenerate field definitions after adding schema properties and
add new shipped resources explicitly to the packaging asset list. Document
supported behavior and defaults, not one experiment's outcomes.

## Optional social action contracts

When adding an action identifier, also review `wtf/simulation/social.py` and
`viewer/social.mjs`: their stable action order, collective group mapping, context
projection and input labels are part of the optional social/consequence model
contract. Change the versioned contract and persistence validation when changing
that layout. Never label hidden outcomes or copy another agent's reward/rollout
into a private learner. See [collective learning](social-learning.md).


## Native backend admission

The optional [native CPU path](native-engine.md) has an explicit world/action
support boundary. A Python geography, action or execution hook needs its own
native implementation and parity tests before that backend can admit it.
Preserve resolution/RNG order, rewards, observations, births and captured events;
do not silently omit a new mechanism to keep an experiment running in C++.
The reference backend remains available for the full Python feature set.
