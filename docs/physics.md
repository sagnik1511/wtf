# Physics and simulation rules, illustrated

[Project overview](../README.md) · [Geography and configuration](geography.md) · [Rewards](rewards.md)

This guide explains what physically happens during a simulation step, separately
from what an animal observes, which action its policy selects, and the reward
used for learning. The world is a discrete grid with authored movement, energy,
combat and reproduction rules—not a continuous rigid-body physics engine.
One step advances simulation time; viewer playback speed does not change it.

## Rules at a glance

| Mechanism | Simulation rule | Configure or inspect |
| --- | --- | --- |
| Movement and collisions | One-cell voluntary moves; occupied starting cells block entry, swaps and following; competing empty-cell claims use seeded resolution | Species actions/body; [escape actions](escape-and-consent.md#movement-decisions) |
| Rivers | Water entry/exit requires swimming; fractional current drift follows voluntary movement | Terrain/map and species locomotion; [geography](geography.md) |
| Elevation | Climb limits gate motion; successful uphill motion costs additional energy | Species locomotion and elevation map |
| Sight | Circular range, wall occlusion and optional directional water credit; facing and elevation do not occlude sight | Body vision, water bonus and [observation slots](observations.md) |
| Food and health | Food/kill gains replenish energy up to capacity; starvation, attacks and drowning damage health | Resources, body and [cost/reward accounting](rewards.md) |
| Reproduction | Mutual consent plus proximal physical eligibility; both parents pay energy | [Consent windows and failure reasons](mating-intent.md) |
| Day and night | Optional sunlight changes species-specific sight and energy costs | [Day–night configuration](day-night.md) |

Parameters belong to each environment and species. Numerical examples below
illustrate the supplied controlled scenarios; they are not universal defaults
or a claim that every maintained preset still uses those exact values.

## What happens in one step

The experiment gathers one action from each living animal before advancing the
world. Physical resolution then proceeds in this order:

1. **Turn and move.** Resolve voluntary movement, terrain eligibility, destination
   contention and successful swimming/uphill charges.
2. **Apply currents.** Accumulate fractional flow and resolve permitted drift.
   Subsequent interactions use the resulting positions.
3. **Feed and attack.** Resource eaters feed; attacks are collected and resolved
   together. Kill energy goes to one seeded credited attacker. An animal can
   finish feeding in the step it is killed.
4. **Charge demand and harm.** Spend ordinary living/action energy, advance age,
   charge water maintenance and apply drowning/starvation damage where eligible.
5. **Remove deaths and resolve births.** Only surviving eligible pairs can
   reproduce. Consent, distance, route, capacity and free child space are checked
   at this phase, after movement, combat and energy spending.
6. **Finish the transition.** Account for rewards/failures, regrow resources up to
   their caps, advance world time and prepare the next observations. The experiment
   updates parent controllers before creating independent inherited child brains.
   A newborn does not receive an experience update on its birth step.

This ordering matters: an available decision is not a guaranteed outcome after
other animals act. The viewer renders saved outcomes; its animations do not
apply another movement, attack or energy charge.

## Interactive and visual examples

The [interactive physics explorer](assets/physics/explorer.html) contains all four
views—visibility, swimming/current, elevation and reproduction—with both prey
and predator settings. On the Zensical website, follow the link to open the interactive page directly.
When browsing the repository on GitHub, HTML is shown as source; download/open
the file locally instead. The SVG figures below render in both places.
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

This controlled layout snapshot uses world seed 100: 48 prey inside,
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

The brain does not receive this whole shaded visibility grid. It receives up to
`observation_top_k` visible target vectors for each of food, resource-eating animals
and prey-eating animals, ordered by proximity, plus body, heading, blocked-neighbor
and version-specific inputs. At the default top-k of 1, terrain observation
version 2 has 32 values; each additional target slot adds 12. Other observation
versions have their own input layouts; see [animal observations](observations.md).
With the water bonus enabled on a wet map, target vectors are normalized by the
maximum possible radius. See [the detailed rule](geography.md#directional-sight-across-water).

## Walking, swimming and currents

Walking moves one cell relative to heading. Optional `move_backward`,
`strafe_left` and `strafe_right` preserve heading and use the same collision,
slope and walking-effort rules as `move_forward`. They are learned choices,
not automatic escape. Movement resolves before attacks, so a successful escape
can leave an attack cell. All these actions cancel pending mating intent, even
when blocked. [Escape and visible consent](escape-and-consent.md) explains the
input/action compatibility boundary.

![Relative walking directions and local consent observations](assets/diagrams/escape-and-consent.svg)


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

## Food, attacks, starvation and death

There is no passive HP decay or health recovery. Living/actions spend energy;
starvation, attacks and drowning deal damage. Food restores energy rather than repairing HP. Resource consumption is bounded
by available food and remaining energy capacity; resource growth is capped by
configuration. Attacks reach the cell ahead; resource eating checks the current
cell and then the cell ahead. Attack damage is capped at remaining target health,
and credited kill energy is capped by the attacker's energy capacity. Dead
animals are removed; their bodies do not remain as edible resources.

Blocked walking/swimming and failed attacks still incur their applicable normal
attempt costs. A failed swim does not pay the successful stroke/climb surcharge.
Physical energy spending and the scalar learning reward are separate: the
reward coefficient does not spend the body's energy a second time. See
[reward accounting](rewards.md#physical-costs-and-learning-feedback) for the
complete contract.

## Mutual, proximal reproduction

![Successful mutual reproduction, missing consent and an out-of-range pair](assets/physics/reproduction.svg)

All panels use mature, healthy, sufficiently energized opposite-sex prey. Both
must consent to `reproduce`, within radius **2** in these illustrated cases, with line of sight,
a dry reachable route, population capacity and a free child cell. Proximity
alone does not create a child. Same-step consent is the default; optional
`intent_steps` preserves consent across a short window while resting. Other
actions, expiry or personal ineligibility clear it. Both parents must still be
eligible when the birth resolves; the window never expands mating radius.
Failure spends no birth energy and starts no cooldown; ordinary living cost
and failed explicit-attempt feedback still apply. [Mating intent](mating-intent.md)
lists the recorded failure reasons and their priority.

A successful birth costs each prey parent 10 energy (predator: 15), with no direct
HP deduction. The child receives configured initial energy and full configured
health. The training lifecycle mixes both parents' weights, resets child training
state and may apply the existing rare child-only mutation. The figures show body
physics; they do not display brain weights or demonstrate trained behavior.

## Daylight and what the brain can sense

![Dawn, midday, dusk and night lighting with the sunlight cycle](assets/viewer/day-night.svg)

Day–night behavior is opt-in per environment. Sunlight rises from sunrise to
midday, then fades toward sunset; species-specific night multipliers interpolate
vision radius, living cost and movement cost. Water-vision credit is still
additive. Mating radius, attack damage and resource growth keep their configured
rules. Interface light/dark mode is independent of simulation time.

Reduced sight can expose fewer targets, but does not reduce configured
`observation_top_k`, change input-vector length or add a clock/light input.
Missing target slots remain zero-padded. Current visibility, the policy's sampled
observation and the final physical outcome are different things; the inspector
shows the body step and diagnostic-sample step separately. See
[day–night rules](day-night.md), [animal inputs](observations.md) and
[agent inspection](recording.md#visual-agent-inspector).

## Change or extend the rules

Use [the field reference](configuration-reference.md) for valid values and
[geography](geography.md) for map symbols, water, flow and elevation settings.
JSON can select existing actions and mechanisms; implementing a new physical
rule requires coordinated changes to world resolution, costs/rewards, masks,
observations, persistence and viewer semantics. Follow
[world and action extensions](extending-worlds-and-actions.md), and use
[calibration guidance](calibration.md) when choosing body/energy settings.

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
