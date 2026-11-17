# Adding prey and predator species

[Configuration](configuration.md) · [Saved brains](brain-library.md) · [Playback](recording.md)

Species are the named entries in an environment's `species` object. The supplied
presets still contain exactly one prey species (`prey`) and one predator species
(`predator`). You can add more entries without changing Python or JavaScript.
Use stable, nonempty names such as `rabbits`, `deer`, `foxes` and `wolves`.

## Species identity and diet

| Field | Meaning |
| --- | --- |
| Key under `species`, e.g. `deer` | Distinct species identity used for mating, metrics, controllers and champion archives |
| `diet: "resource"` | Prey/herbivore: eats the shared resource patches |
| `diet: "prey"` | Predator: hunts any resource-eating animal, across all prey species |
| `brain` | Key of a brain template under `brains` |

A species name is not its diet: `wolves` with `diet: "prey"` is a predator.
Predators do not hunt other predators. Prey species compete for the same food,
and all animals share the grid and obstacles. Selective food webs and
predator-on-predator hunting are not currently modeled.

## Add entries to a JSON environment

Copy the complete existing prey or predator object under a new key, then edit
its population, body, learning-rate multiplier, reproduction settings and brain
reference. Keep every required field. A reused brain template initializes private
controllers; it does not share live weights between species or animals.

This optional command creates a complete four-species configuration from the
baseline. It does not change the baseline itself:

```sh
.venv/bin/python - <<'PY'
import copy
import json
from pathlib import Path

config = json.loads(Path("environments/baseline.json").read_text())
config["name"] = "my_world"
for name, template in (("deer", "prey"), ("wolves", "predator")):
    spec = copy.deepcopy(config["species"][template])
    spec["initial_population"] = 8
    spec.pop("initial_brain", None)
    spec.pop("initial_brains", None)
    config["species"][name] = spec
Path("environments/my_world.json").write_text(json.dumps(config, indent=2) + "\n")
PY
.venv/bin/python -m ecosystem.cli environments/my_world.json --runs 1 --progress
```

You can instead make these copies directly in your JSON editor. Adjust world
size, food supply and population caps for the combined population. Python
validation checks initial grid capacity and rejects invalid settings. Different
species can use different supported learning algorithms or network sizes.

Remove copied saved-brain selectors when founding a new species: an artifact
for a differently named species is intentionally incompatible. See
[brain initialization](brain-library.md#species-and-sex-selection).

## Reproduction and best brains

Only an eligible female/male pair of the **same species** can reproduce. A deer
cannot mate with a rabbit, even though both eat resources. Children retain that
species, mix both parents' weights and keep the rare birth-mutation handling.

Every species has its own female and male health, kills, lifespan and food
champions; predator sexes also have balanced champions. Archives and CLI
selectors use the exact configured name:

```sh
.venv/bin/python -m ecosystem.cli environments/my_world.json --runs 5 \
  --brains-from PREVIOUS_RUN_NAME --brain deer=lifespan --brain wolves:male=kills
```

## Viewer colors and metrics

The board legend is generated from the recorded species, with actual species
names and female/male colors. Prey remain circles and predators triangles;
males have light outlines. The first prey/predator retain the familiar palette,
and additional species receive generated colors without cycling a fixed palette.
Colors remain fixed while replaying that recording.

Optionally add this object inside a species definition for fixed colors across
configurations and ordering changes:

```json
"display": {"color": "#b4e580", "male_color": "#ca94ff"}
```

`color` is the female/base/chart color; `male_color` is the male body color. Both
are optional six-digit hex values. These are presentation settings and do not
change observations, model signatures or learning.

The top prey/predator cards show totals by diet; hover for per-species counts.
Population chart lines, dashboard choices, analysis, CSV columns and champion
folders remain separate for each species.

The observation vector stays at 20 values: nearest resource, nearest prey across
all prey species, and nearest predator across all predator species. Brains do not
receive a species-name ID or separate feature block per species. Adding species
therefore keeps existing model architectures compatible, without adding explicit
species recognition. MAPPO's team context groups allies by exact species and
summarizes all other species together in its non-ally block.
