# Collective observations and consequence learning

[Observations](observations.md) · [Tracking](tracking.md) · [Rewards](rewards.md) · [Playback and inspection](viewer-visual-reference.md#collective-learning)

Two independent opt-ins add social information while keeping every animal's
policy parameters, optimizer, exploration and learning experiences independent.
Local collective observations pool visible same-species behavior. A separate
bounded species predictor retains population experience across episodes. Neither
feature copies another animal's reward into an individual's learning update.
Existing algorithm-specific semantics, including MAPPO's species-mean training
reward, remain unchanged.

![Local collective sensing, shared species predictions and independent policies](assets/diagrams/social-learning.svg)

## Run the complete example

```sh
.venv/bin/python -m wtf run seaside_social --runs 3 --seed 100 --progress
```

The editable [environment](../environments/seaside_social.json) uses ordinary PPO
for both species on the existing [seaside map](../maps/seaside.json). It retains
the seaside ecology, action sets and proximal mating radii. Only the two optional
input blocks and consequence settings are enabled. Its `compact` recording profile
retains world playback but samples diagnostics at least 50 steps apart and omits
activation traces. Use `--recording-profile full` for the preset's normal configured
diagnostic/activation cadence. Existing presets stay opted out.

## Local collective observations

Add this object to each participating brain template, preserving its other fields:

```json
"social_observation": {
  "history_steps": 4,
  "max_neighbors": 8
}
```

Each observer samples at most eight nearest visible animals of the **same exact
species**, including either sex. Different species remain separate even if they
share a diet or brain template. Ordinary sight radius, water sight extension,
night modifiers and obstacle occlusion apply. A subject must be seen at the start
and end of its action. Disappearing beyond sight is not an observed death, and
hidden actions/outcomes are never backfilled. Birth feedback requires the newborn
to be visible too.

The observer receives six pooled groups, without individual identities:

| Group | Observed actions |
| --- | --- |
| Walk | Forward/backward walking and sideways escape actions |
| Swim | Swimming |
| Feed | Eating |
| Combat | Attack and same-species challenge |
| Reproduce | Reproduction choice |
| Idle | Resting and turning |

Each group adds **confidence, harm rate, feeding rate and birth rate**, for
**24 inputs**. Harm means visible injury or death; feeding requires an actual
positive food/hunting energy gain; birth requires an actual birth. Rates describe
outcomes during the observed step, rather than another animal's numerical reward.
Counters are pooled across the recent `history_steps` (1–16). `max_neighbors`
(1–16) bounds sampling per observer and step.

Confidence is an evidence-size heuristic, `n / (n + 4)`, rather than calibrated
statistical certainty. No samples gives confidence zero; the viewer labels that
state **unknown** rather than presenting it as evidence of safety.

## Persistent consequence predictor

Enable a top-level population-learning block:

```json
"consequence_learning": {
  "enabled": true,
  "max_contexts": 512,
  "horizon_steps": 16,
  "update_schedule": "continuous"
}
```

Then add this to every consuming brain template:

```json
"consequence_observation": true
```

The first implementation is a compact empirical outcome table, not another
neural policy. There is one bounded table per exact species, pooling both sexes.
It records animals' **own physical experiences**, including animals that die.
It does not train from exact rewards or duplicate experiences for every witness.

A context contains the action identifier, coarse own energy/health ranges,
current/destination depth, uphill difference, current strength, swimming health
eligibility, adjacent food/obstruction, maturity and a nearby visible threat
indicator. It contains no absolute coordinate or target identity, allowing matching
situations at different locations to reuse evidence. Physical differences outside
this coarse representation can still matter.

For each action, the model estimates whether **harm, feeding or a birth** occurs
within the next `horizon_steps` (1–64). Death closes pending windows. A living
animal at an episode's time limit has incomplete windows discarded; it is not
labelled safe beyond the observed interval. Overlapping windows are correlated,
so these rates are associations and are not causal estimates. Consequences
beyond the selected horizon, such as much later starvation, do not label the
initial action harmful.

Each of the twelve stable action identifiers contributes confidence and three
outcome rates: **48 inputs**. Unsupported actions have zero-filled slots.
Unknown contexts also have zero confidence. The predictor supplies information;
it does not force an action, change action masks, issue a reward or merge policy
weights. Each learner decides how to use the estimates through its own algorithm.

Updates occur only in training. Evaluation/random modes consult the retained
model without updating it, while local witnessing can still update observations.
GRPO trial worlds have private copies of this state; trials cannot update the
real world's knowledge. Local histories and incomplete windows reset between
episodes; the species tables survive independently of champion selection and
`--no-chain-brains`.

## Publication timing

`update_schedule` defaults to `continuous`: finished outcome windows update the
predictor during training. Choose `episode` to collect the same bounded evidence
while keeping the published predictor frozen until the next training episode's
first observations:

```sh
.venv/bin/python -m wtf run seaside_social --runs 3 --seed 100 \
  --consequence-update-schedule episode --progress
```

The CLI override applies to fresh experiments with consequence learning enabled;
`--resume` restores its saved schedule. Local witness inputs and individual
policy learning keep their existing cadence. Collection still uses rolling
horizon windows, rather than whole-episode trajectories. The first episode has
unknown estimates unless compatible population knowledge was imported.

Episode mode lazily copies a species table on its first new evidence, keeping
published rows separate from staged updates. This bounds staging by one extra
capacity-limited table per updated species. Changing the schedule does not change
input width or require different policy weights. Evaluation and random modes
neither collect evidence nor publish staged training evidence at episode changes;
that evidence remains available for a later training continuation.

## Save, restore and reuse

Recovery checkpoints include the species tables **once**, plus the short personal
witness histories and pending outcome windows needed for exact continuation.
No knowledge table is copied into each animal's brain checkpoint or champion.
Continuous mode retains version-1 model state. Episode mode uses version-2
recovery state with both published and staged tables, so interruption does not
publish evidence early or lose it. Completed aggregate exports remain version 1
and include finished-window evidence from the final episode without changing
that episode's recorded predictions. Either schedule accepts compatible exports.

A completed CLI experiment also writes one `consequences.json.xz` beside its
summary. This is a compressed aggregate table with physical contracts, not full
trajectories, world frames, policy weights or individual identities. Recovery
checkpoints remain the way to recover an interrupted experiment's latest state.

```sh
# Start fresh policy weights while reusing compatible population knowledge.
.venv/bin/python -m wtf run seaside_social --runs 3 --seed 200 \
  --consequences-from results/EXPERIMENT --progress

# Matching policy brains can be loaded separately when desired.
.venv/bin/python -m wtf run seaside_social --runs 3 --seed 200 \
  --brains-from results/EXPERIMENT \
  --consequences-from results/EXPERIMENT --progress
```

The import accepts either a result folder or the specific file. Species names,
physical species contracts and prediction horizon must match. Geography may
change, because coordinates are excluded, but unseen contexts still need new
evidence. Rates also depend on the policies and encounter conditions that
produced the samples; they are not guaranteed outcomes under a different policy.
Changing reward coefficients alone does not invalidate physical outcome
knowledge. A smaller capacity retains the newest compatible context entries.
Exact `--resume` uses its checkpoint's own model and rejects `--consequences-from`.

## Input and storage compatibility

These blocks append **after** existing observation, movement-history and scent
inputs. Enabling both adds 72 inputs; the example has 141 prey inputs and 225
predator inputs. The optional fields enter model signatures. Old weights without
the blocks cannot initialize these wider policies; unchanged presets and legacy
checkpoints retain their original layouts and behavior.

Tables retain at most `max_contexts` entries (16–8192) per participating species,
using deterministic update-order LRU eviction. There are no unlimited experience
archives. Personal histories are bounded by recent steps and neighbour count;
pending windows are bounded by living animals and the horizon. Model reads and
repeated observation calls neither train the predictor nor duplicate sightings.

Replay storage adds **no new frame/event fields**. Existing sampled diagnostic
vectors contain the added inputs, so inspectors need no decoded-run knowledge
model. Extra inputs still enlarge policy weights, training buffers and recorded
diagnostic samples; compact capture is useful for large experiments. No reduction
in total RAM/disk usage is claimed from enabling this feature.

## Playback inspection

Select a living animal in a recording with these inputs. Open its agent inspector
and find **Collective learning**:

- **Local witnesses:** harm, feeding and birth rates grouped by observed action.
- **Species consequence estimates:** corresponding rates for the selected animal's
  available actions in its sampled physical context.

Charts show the recorded sample step, evidence through opacity/tooltips and
unknown contexts explicitly. They update when replay reaches a new diagnostic
sample, remain fixed while paused and follow backward seeking. There is no
independent animation timer. The panel labels the configured publication mode.
In episode mode, bars may change as the animal's sampled context changes even
though the underlying predictor stays frozen throughout that episode. They read
saved values only, with no future model
state or omniscient reconstruction from board positions. Both themes use the
shared inspector palette. Older recordings without these inputs omit the panel.

![Actual inspector components with illustrative collective and predictor inputs](assets/viewer/social-learning.svg)

## Biological motivation and limits

| Mechanism | Motivation and approximation |
| --- | --- |
| Local observed feeding | Guppies learned food routes by swimming with trained conspecifics; our grouped rates replace detailed individual social interactions. [Laland & Williams, 1997](https://pubmed.ncbi.nlm.nih.gov/9236013/) |
| Observed harm | Rhesus monkeys acquired snake fear observationally; this motivates danger learning, but does not validate our specific drowning labels or numerical rules. [Mineka et al., 1984](https://communities.pacificu.edu/wp-content/uploads/sites/1523/2023/02/Mineka-Davidson-Cook-Keir-1984.-Observational-conditioning-of-snake-fear-in-Rhesus-monkeys.pdf) |
| Population traditions | Foraging traditions spread in wild great tits, with social conformity; our retained tables approximate accumulated information rather than modelling that diffusion mechanism. [Aplin et al., 2015](https://www.nature.com/articles/nature13998) |

**The persistent population model is a training abstraction.** All participating
animals of a species can consult pooled experience, including experience from
far-away individuals. There is no simulated local communication/diffusion delay.
Its estimates must not be described as biologically realistic messages between
distant animals. Same-species-only pooling is a configurable-model boundary here,
not a claim that animals never learn across species.

Likewise, PPO and scalar rewards are computational learning mechanisms, and
copying/mixing learned weights at birth is evolutionary computation rather than
biological inheritance of acquired neural weights. Health, energy, swimming and
climbing thresholds use abstract units without species-specific empirical
calibration. This is biologically inspired learning with explicit approximations,
rather than a validated biological ecosystem.

Assess usefulness with matched-seed, held-out coast/inland evaluations: drowning,
energy coverage, survival, reproduction and successful crossings matter alongside
training reward. More cautious behaviour alone does not establish improvement.
