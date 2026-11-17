# Rare mutations at birth

[Configuration](configuration.md) · [Algorithms](algorithms.md) · [Saved brains](brain-library.md) · [Playback](recording.md)

During training, a newborn can receive a small sparse change after inheriting its
parents' mixed learned weights. Both parents are unchanged by crossover/mutation. Founders and existing adults
are not mutated. All nine algorithms support this optional mechanism.

## Configuration

Each species accepts:

```json
"mutation": {
  "enabled": true,
  "probability_per_birth": 0.0001,
  "max_changed_weights": 3,
  "noise_std": 0.005
}
```

All supplied presets enable these defaults. Omit `mutation`, set `enabled` to
`false`, or set probability to `0` to disable it. Zero noise also produces no
mutations. Reproduction retains its normal health, energy, timing, free-cell and
population requirements; mutation is checked only after a successful birth.

`0.0001` means 0.01%, approximately one selected birth in 10,000. It is a
probability, not a schedule: experiments can legitimately contain no mutations.
For a controlled feature test, probability `1.0` mutates every training newborn;
restore the rare value afterward for normal experiments.

On a selected birth, choose an integer count uniformly from 1 through
`max_changed_weights`, capped by the matrix parameter count. Sample that many
distinct matrix weights and add independent zero-mean Gaussian noise with the
configured standard deviation. `noise_std` is an absolute weight scale, not a
percentage. Biases are excluded. Shared policy/value trunks, separate MAPPO
critic matrices and GRU input/recurrent matrices are eligible, so a mutation does
not necessarily change the next action probability immediately.

If floating-point rounding leaves every selected value unchanged, the birth is
not counted as mutated and its training state is preserved. Non-finite mutated
weights are rejected. No reward, health or energy bonus is attached to mutation.
The child continues its normal learning algorithm afterward. Mating eligibility
is physical only; mutation does not determine whether a pair can reproduce.

## Randomness and training state

The mutation stream is derived independently from the world seed and child ID
using the existing offspring-seed rule and a mutation-specific salt. Mutation
checks/noise do not consume world, action, training-minibatch or GRPO-group RNG
streams. Identical seeded birth identities reproduce their mutation choices.
Changing policies can still change which births occur and later world outcomes.

A mutated child receives fresh optimizer moments, training counters and baseline,
with empty experience buffers and pending decisions. Q-learning/QR-DQN target
weights and GRPO reference weights synchronize to its new online weights.
Recurrent memory starts at zero. The learning rate and Q exploration level inherited from the mother remain
unchanged. All children already start with fresh temporal training state after
[paired crossover](configuration.md#reproduction), even if no mutation occurs.

Evaluation and random-action runs keep inherited weights frozen and do not apply
mutations. During training, GRPO trial worlds apply the same birth rule: all
trial descendants receive independent mixed policies, optionally mutated. These hypothetical mutations cannot alter real
controllers, real counters or real-world events and are not trained in trials.

## Tracking and compatibility

Animal state records three small fields:

| Field | Meaning |
| --- | --- |
| `mutated` | This individual's brain changed at its own birth |
| `mutation_weights` | Number of matrix weights actually changed at that birth |
| `mutation_count` | Number of mutation events along this biological lineage |

An unmutated descendant retains its parent's lineage count, with `mutated=false`
and `mutation_weights=0`. A mutated child increments that count once. Lineage
counts describe the current world's family tree; cross-run model ancestry remains
in the saved champion artifact chain rather than being added to founder counters.

A `mutated` replay event identifies the parent (`actor`), child (`target`) and
changed weight count (`amount`). The inspector shows mutated/inherited lineage
information even between diagnostic samples. CSV metrics add `<species>_mutations`
columns; summaries and world metrics contain cumulative per-species counts.
Per-step telemetry and the behavior analyzer also count mutation events, and the
analyzer totals changed weights. Champion provenance includes mutation metadata
for selected mutated lineages. No full weight arrays are added to replays.

Plain and binary checkpoints preserve the settings, counters and lineage metadata
for exact continuation. Older checkpoints without these fields restore with zero
counts and mutation disabled unless their saved configuration explicitly enables
it. Model signatures are unchanged, so compatible old champion weights remain
usable. Binary replay field tables carry the new scalar fields through the
existing browser decoder without changing recording cadence or chunk loading.

Mutation introduces exploration across descendants; it does not guarantee
improvement. Compare frozen descendants against suitable parent/baseline models
on matched held-out worlds to assess effects beyond a single unusual trajectory.

`mutation_count` follows the maximum parental lineage count plus one if this
child mutates. It is a lineage-depth measure, not a count of unique mutation
events across both family trees, which can have shared ancestors.
