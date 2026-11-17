# Physics and visibility, illustrated

[Project overview](../README.md) · [Geography and configuration](geography.md) · [Rewards](rewards.md)

The [interactive physics explorer](assets/physics/explorer.html) contains all four
views—visibility, swimming/current, elevation and reproduction—with both prey
and predator settings. Open the HTML locally in a browser. GitHub displays HTML
source instead of running it; the SVG figures below render directly in the docs.
The explorer needs no ChatGPT session, Python backend or network libraries.

If you prefer a local URL, from the repository root:

```sh
.venv/bin/python -m http.server 8002 --directory docs/assets/physics
```

Then open [the explorer](http://127.0.0.1:8002/explorer.html). Stop that temporary
server with Ctrl-C. It is separate from the simulation viewer on port 8000.

For full recordings of the actual viewer, see the
[terrain GIF gallery](geography.md#recorded-terrain-previews). The diagrams below
are controlled physics illustrations, not screen recordings.

For the training, archive, storage and cache diagrams, see the
[documentation visual guide](visual-guide.md).

## Actual valley layout

![Actual valley founder positions, three-cell gates and food allocation](assets/physics/valley-layout.svg)

This is the current map and initial layout with world seed 100: 48 prey inside,
12 predators outside, two three-cell gates, and food split into 30 inner-boundary,
30 outer-boundary and 20 farther-away patches. These are starting zones, not
permanent movement restrictions. The river is outside the left wall.

## Sight before and after water credit

![Normal sight compared with water-extended sight across a three-cell river](assets/physics/visibility-water.svg)

Prey have normal radius 6. Three intervening water cells grant three extra cells,
so a target nine cells away becomes visible. Only rays crossing water receive
credit; enabling the bonus does not increase the range equally in every direction.

![Dry direction limits, wall occlusion and visible elevated terrain](assets/physics/visibility-limits.svg)

Dry rays retain the base radius. Walls block line of sight even across water.
Elevation currently changes movement cost but does **not** occlude sight; other
animals also do not occlude it. Vision is 360°, independent of facing. Range is
Euclidean distance; occlusion and water credit follow integer Bresenham rays.

![Prey and predator visibility at the same target distance](assets/physics/visibility-species.svg)

Prey use radius 6, up to 9 across water; predators use radius 8, up to 11. The
bonus is capped at 3 and counts only water strictly between observer and target.
Water at the observer or target alone adds no credit.

![The actual valley's left gate and one-cell river before and after sight credit](assets/physics/visibility-valley.svg)

This crop uses the actual valley geometry with example animal positions. Its
river is one cell wide, so it grants **one** extra cell along this ray, not an
automatic three. The three-cell river in the earlier figures is a controlled
example used to show the maximum extension.

The brain does not receive this whole shaded visibility grid. It receives one
nearest accepted target vector for each of food, prey and predator, plus body,
heading, blocked-neighbor and local-terrain inputs. Terrain observations remain
32 values. With the water bonus enabled on a wet map, nearest-target vectors are
normalized by the maximum possible radius. See [the detailed rule](geography.md#directional-sight-across-water).

## Walking, swimming and currents

![Walking into water fails, swimming succeeds, and weak swimmers cannot enter](assets/physics/swimming.svg)

These panels use predator settings. Walking cannot enter or leave water; animals
must choose `swim`. At depth 1, a predator needs at least 25 HP and energy strictly
greater than the stroke plus uphill cost. Low-health animals cannot enter by
swimming. Blocked attempts still pay normal living/movement-attempt costs and
receive the configured failed-action reward penalty.

![Fractional currents move an animal after swimming and stop at dry land](assets/physics/currents.svg)

Current displacement is separate from the chosen action. The example current is
(+0.4, +0.75) cells per step. Fractional components accumulate until whole-cell
movement is possible. Intermediate cells, walls, occupied cells, climb limits
and blocked diagonal corners constrain drift. Dry landing clears the remainder.
Normal movement also forbids swaps and following into a cell vacated in the same
phase; competing claims use seeded random resolution.

![A weak animal resting in water loses health and dies from drowning](assets/physics/drowning.svg)

Resting in water still incurs living and water-maintenance costs, and the current
still moves the animal. Weak, exhausted or nonswimming animals take drowning
damage. Here a predator starts at 10 HP and loses 2 HP per step at depth 1.
The explorer also includes healthy/weak variants of each water action and every
intermediate step, rather than just the selected frames shown here.

## Height and energy

![An allowed one-unit climb, a blocked two-unit cliff and a downhill move](assets/physics/elevation.svg)

A predator pays 1.3 additional energy per height unit gained. A +1 climb therefore
costs 1.515 total energy here: 1.3 uphill + 0.14 movement + 0.075 living. A +2 cliff
exceeds `max_climb: 1` and blocks movement; ordinary attempt costs still apply.
Downhill has no extra cost or refund. Prey use their own configured coefficients.

There is no passive HP decay or health recovery. Living/actions spend energy;
starvation, attacks and drowning deal damage. Food restores energy. Attacks reach
the cell ahead; resource eating checks the current cell and then the cell ahead.

## Mutual, proximal reproduction

![Successful mutual reproduction, missing consent and an out-of-range pair](assets/physics/reproduction.svg)

All panels use mature, healthy, sufficiently energized opposite-sex prey. Both
must choose `reproduce` in the same step, within radius **2**, with line of sight,
a dry reachable route, population capacity and a free child cell. Proximity
alone does not create a child. Failure spends no birth energy and starts no
cooldown; ordinary living cost and failed-action feedback still apply.

A successful birth costs each prey parent 10 energy (predator: 15), with no direct
HP deduction. The child receives configured initial energy and full configured
health. The training lifecycle mixes both parents' weights, resets child training
state and may apply the existing rare child-only mutation. The figures show body
physics; they do not display brain weights or demonstrate trained behavior.

## Provenance and maintenance

These are controlled examples, not experiment replays or evidence of learning.
Movement, energy, health, drift and birth snapshots come directly from `World.step`
with explicit actions. Visibility uses the same ray, range and occlusion rules;
the earlier interactive version was cross-checked against Python on 1,916 rays.
The static layout is generated from the current map/config. No model weights,
checkpoints, run recordings or screen captures are embedded.

Asset sources and regeneration instructions are in the
[physics asset catalog](assets/physics/README.md). The gallery covers every named
case in the explorer; target coordinates and species remain adjustable in HTML.
