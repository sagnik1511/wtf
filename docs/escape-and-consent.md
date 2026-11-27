# Escape actions and visible consent

[Mating intent](mating-intent.md) · [Observations](observations.md) · [Feedback](rewards.md)

Optional walking actions let a policy move away from threats without turning.
Current V7 observations expose local mating readiness and consent. These mechanisms
retain ordinary collision, effort and reproduction eligibility rules; populations,
radii, health and learning settings belong to the chosen environment JSON.

![Relative escape moves and local consent inputs](assets/diagrams/escape-and-consent.svg)

## Movement decisions

| Action | One-cell destination, relative to heading | Changes heading? |
| --- | --- | --- |
| `move_forward` | Ahead | No |
| `move_backward` | Behind | No |
| `strafe_left` | Left | No |
| `strafe_right` | Right | No |

All walking attempts debit the configured movement effort in addition to living
cost, including blocked attempts. Unsuccessful moves receive the configured
failed-action reward. Successful uphill movement also pays its terrain cost.
Backward/sideways moves cannot enter or travel through water; use `swim` where
configured. Slope, boundaries, obstacles, initial occupancy and seeded destination
contention work exactly as for forward walking. Swaps and following into a
just-vacated cell remain blocked. Movement resolves before attacks, so a chosen
escape may evade a directional attack. This is an available learned action,
not automatic predator avoidance, a speed boost or an invulnerability period.
All three escape actions cancel mating intent, even if blocked.

Any species can opt in by
adding them to `allowed_actions`. The policy's action-order signature includes
them; existing weight files with another action vocabulary cannot initialize it.
Native and Python metabolism charge the same effort. Rebuild native kernels with
`.venv/bin/python -m work.build_native` after updating. With an older binary,
steps containing escape choices use Python metabolism to avoid undercharging.
Other supported native kernels remain available.

## Current consent observations

V7 includes own consent, observed mate consent and consenting-pair cues inside
the twelve-field mating block. They describe current intent, not a partner's
future choice, and never bypass radius, visibility, health/energy or birth-space
rules. See [current mating inputs](mating-observations.md) and
[exact V7 offsets](learning/v7-observations.md).

<a id="observation-version-4"></a>

The earlier V4 append-only layout is kept in the
[historical observation reference](archive/observation-layouts.md), not as a
current preset example. Matching older recordings remain readable.

Choose parameters through the [configuration guide](configuration.md); consult
[current calibration](calibration.md) for recommendations and their scope.
