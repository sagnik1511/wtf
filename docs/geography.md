# Geography, rivers and elevation

[Configuration](configuration.md) · [Species](species.md) · [Playback](recording.md)

Terrain is optional. Existing environments remain flat unless they select a map.
Four optional presets cover three maps, keeping one prey species and one predator species:

| Environment | Map |
| --- | --- |
| [river_crossing.json](../environments/river_crossing.json) | A river bisects the world, with a fast deep channel and a slower shallow ford |
| [river_crossing_ppo.json](../environments/river_crossing_ppo.json) | The same river map with PPO for both species |
| [highlands.json](../environments/highlands.json) | Terraced highlands, a steep central plateau, a climbable north ramp and obstacle approaches |
| [enclosed_valleys.json](../environments/enclosed_valleys.json) | Walled valleys, narrow gates, a river crossing and a completely secluded pocket |

```sh
.venv/bin/python -m ecosystem.cli environments/river_crossing.json --runs 10 --progress
# Alternatives: environments/highlands.json or environments/enclosed_valleys.json
./run.sh
```

The original three terrain presets use Q brains; `river_crossing_ppo.json` uses
PPO for both prey and predators. All nine algorithms support
terrain: when copying another algorithm's complete template, retain
`observation_version: 2` and the appropriate species' action list. The river and enclosed-valley Q presets share model signatures/actions.
Dry highlands omits `swim`, so its action signature differs. Transfer requires
matching action lists in the same order. The PPO preset needs compatible terrain PPO weights rather than Q weights. Older 20-input brains cannot directly initialize the 32-input terrain
models; their shapes and observation contracts differ.

## Use PPO on the river map

```sh
.venv/bin/python -m ecosystem.cli environments/river_crossing_ppo.json --runs 10 --progress
```

Both species use small PPO actor-critic networks with 32 terrain inputs. World,
map, health, populations, reproduction, currents and swim/climb settings match
the Q river preset. Brains remain private per animal and champions remain
species/sex-specific. No saved weights are requested by this preset; later runs
chain the corresponding PPO champions by default.

## Select or edit a map

An environment can reference a project-relative map JSON:

```json
"terrain": {"map_file": "maps/river_crossing.json"}
```

Absolute map paths also work. On loading, the map definition replaces this
reference in the resolved configuration. Results and checkpoints embed it, so
replay/resume does not read the source map again or follow later edits. The map
files under [maps/](../maps/) are intended to be edited and committed.

A map contains `map_version: 1`, `name`, `width`, `height`, `rows` and `legend`.
Dimensions must match `world.width`/`world.height`. `rows` is a list of strings,
each exactly `width` printable ASCII characters; there must be `height` rows.
Each character names a legend entry. Its properties are:

| Legend property | Meaning and default |
| --- | --- |
| `elevation` | Height units from 0 to 100; default 0 |
| `blocked` | Impassable, vision-occluding obstacle; default false |
| `water_depth` | Relative depth from 0 to 10; default 0 (dry land) |
| `flow` | `[dx, dy]` current vector in cells per simulation step; default `[0, 0]`, each component between −4 and 4 |

For example, the river map uses `.`, `~`, `%` and `=` for land, ordinary river,
deep channel and shallow ford. A custom river legend entry can be:

```json
"~": {"water_depth": 1, "flow": [0.4, 0.75]}
```

Coordinates start at the top-left: positive x goes right and positive y goes
down. A cell cannot be both blocked and water. Map obstacles are combined with
`obstacles.explicit_cells` or scattered obstacles; use explicit/no additional
obstacles to retain designed gates. Extra scattered barriers can block routes.
Founders, resources and newborns spawn on dry, unblocked cells. Initial capacity
is validated against dry space; disconnected regions are intentional and need
not contain food or both sexes in every randomized run.

## Swimming and currents

Add `swim` to a species' `allowed_actions` to make it capable of swimming. Normal
`move_forward` cannot enter **or leave** water. A swim stroke moves one cell in
the facing direction and works only while in water or facing a water cell; turn
toward the river first. Other attempted uses are blocked. Maskable PPO removes
unavailable strokes from its distribution; other algorithms can attempt blocked
actions just as they can attempt blocked walking or missed attacks.

Both health and energy constrain swimming. A stroke requires:

- Health at least `swim_min_health × max(current_depth, destination_depth)`.
- Energy strictly greater than swimming plus uphill costs.
- An unblocked/unoccupied destination and a climb within `max_climb`.

Every animal already in water drifts after voluntary movement, including an
animal that cannot swim or is too weak to stroke. The vector is sampled at its
post-action cell for that step. Fractional movement accumulates independently
on x and y: a 0.25-cell component moves one cell every four steps, rather than
rounding away. A stroke and current can reinforce or oppose each other.

Fast currents are traced through intermediate cells, with seeded contention,
no following/swapping into occupied starting cells, no wall tunneling and no
diagonal corner cutting. Blocked integer displacement is discarded rather than
stored up into a later leap. Fractional remainder persists while in water and
resets on reaching land. Flow stops at the first dry cell; heading is unchanged.
A current can therefore wash an animal onto another bank without a second action.

## Species locomotion costs

Optional `species.<name>.locomotion` values control abilities and costs:

```json
"locomotion": {
  "swim_min_health": 18,
  "swim_energy_cost": 1.5,
  "water_energy_cost": 0.25,
  "drowning_damage": 2,
  "uphill_energy_cost": 1,
  "max_climb": 1
}
```

`swim_min_health` is absolute health, not a percentage. With a threshold of 18,
a depth-1 cell needs 18 HP, a shallow depth-0.5 cell needs 9 HP, and depth 1.6
needs 28.8 HP. Predators in the supplied terrain presets use threshold 25.

A successful stroke spends `swim_energy_cost × maximum touched depth`, plus any
uphill cost. Failed strokes incur no extra terrain movement charge; the existing
body `movement_cost` still applies to attempted walks/strokes. An animal ending
the step in water spends `water_energy_cost × depth` each step. If it lacks swim
ability, falls below the depth-adjusted health threshold, or runs out of energy,
it also takes `drowning_damage × depth`. Existing starvation damage can also
apply to an exhausted animal. Drowning is recorded as a separate death cause.

Moving uphill spends `positive elevation gain × uphill_energy_cost`, and a
single rise above `max_climb` is blocked. Costs are charged only to successful
moves and require energy to spare. Downhill movement has no extra cost or refund;
there is no falling-injury model. Currents respect climb limits and pay uphill
costs too. Cliffs can therefore be inaccessible from below while ramps permit
incremental climbs.

Omitted fields default to swimming health of half maximum health, stroke cost
1.5, water cost 0.25, drowning damage 2, uphill cost 1 and maximum climb 1.
Locomotion costs have no effect on flat environments without terrain. Parents
must be on land with a bounded dry route between them; reproduction cannot mate
across a river or through sealed obstacle corners. Newborn placement also respects
local cliff limits.

## Terrain observations and model compatibility

Flat observation version 1 retains its 20 features. Version 2 appends 12 values:

| Indices | Meaning |
| --- | --- |
| 20 | Current water depth / 10 |
| 21–22 | Current flow x/y components / 4 |
| 23 | Current elevation / 100 |
| 24–25 | Forward water depth / 10 and elevation difference / 100 |
| 26–27 | Right water depth and elevation difference, same scales |
| 28–29 | Backward water depth and elevation difference, same scales |
| 30–31 | Left water depth and elevation difference, same scales |

Neighbor directions are relative to facing; flow components use world axes.
Scales are fixed across maps. The original neighbor-blocked features additionally
reflect normal walking's terrain restrictions in version 2. Terrain maps require
version 2 on referenced templates; a version-2 controller on a flat world receives
zero terrain features. Fractional current carry is checkpointed body state, not
an additional observation feature.

PPO-family networks, Q/QR-DQN, GRUs, masks and MAPPO critics use the configured
input size. MAPPO still adds its separate 16 critic-only context values. Model
signatures record the version/input count and action order; incompatible saved
weights are rejected instead of being truncated or padded.

## Playback, metrics and storage

Restart the playback server after updating Python telemetry code, then refresh
the browser. The viewer shades elevation, labels occasional heights and draws river/current
arrows. Its inspector shows current height, water depth and flow; its observation
panel displays the actual feature count. Drowning uses the existing natural-death
animation, with recorded final coordinates after drift. Kill markers likewise
use recorded positions when supplied.

The dashboard adds swim attempts, current-driven cell moves, terrain energy
demand and drowning deaths per step. Whole-run analysis separates swim success
from walking success and accounts for terrain energy. Terrain CSV files include
a `drowning` column. Forced drift is not counted as a voluntary successful move.
These are physical/behavioral measurements, not proof of learned navigation.

ASCII geometry is stored once in recording configuration, not in every frame.
Replay chunks retain only dynamic bodies, events and fractional drift state.
Seeking still loads one chunk at a time. Checkpoints rebuild sparse terrain
lookups from their embedded map; GRPO trials share read-only static lookups while
retaining independent obstacles, bodies, random streams and learning state.

## Learning feedback

[Reward accounting](rewards.md) explains direct energy-demand and failed-action
penalties, alongside existing drowning, damage and death penalties. A blocked
climb or swim receives a failed-action penalty; normal attempt costs still apply.
Current drift is forced motion and cannot make a blocked voluntary action count
as successful.
