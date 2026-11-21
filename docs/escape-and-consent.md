# Escape actions and visible consent

[Mating intent](mating-intent.md) · [Observations](observations.md) · [Feedback](rewards.md)

Optional walking actions let a policy move away from threats without turning.
Observation version 4 exposes current local mating consent. These mechanisms
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

## Observation version 4

Version 4 preserves the version-3 vector and appends three values:

| Index at k=1 | Signal | Range |
| --- | --- | --- |
| 42 | Own consent time remaining / configured intent duration | 0–1 |
| 43 | Nearest visible same-species, opposite-sex mate's remaining consent | 0–1 |
| 44 | A visible, physically eligible proximal consenting mate with birth space exists, and self is ready | 0 or 1 |

Add `12 * (k - 1)` to these indices for larger top-k. Inputs total
`45 + 12 * (k - 1)`: 69 inputs at k=3 and 45 at k=1.
All nine controller implementations accept the same version.
The inspector labels the recorded signals and shows escape action predictions.

Signals describe current consent only. They do not read a partner's upcoming
choice or sample a policy. Visibility, range, walls and dry routes still apply;
occluded or out-of-sight mates reveal no consent. The nearest mate can be
unwilling while another eligible nearby mate makes the last flag true. No signal
identifies an invisible distant mate or bypasses mutual consent. Reads do not
advance cooldowns, consent, RNG state or recurrent memory.

Versions 1–3 keep their original vector contracts. Version-3 intent remains
inspector-only; use version 4 to expose it to a brain. Existing models from
another observation version are incompatible with this contract.
Use fresh or compatible saved models. Checkpoints, replay chunks and viewer
seeking retain their current formats; new action event labels are decoded by
updated readers, while old recordings remain readable. Walking statistics now
include backward/sideways attempts and reconstruct their effort and failures.

Choose parameters through the [configuration guide](configuration.md); consult
[current calibration](calibration.md) for recommendations and their scope.
