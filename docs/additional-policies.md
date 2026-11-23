# Discrete SAC, TRPO and LSTM PPO

[Algorithms](algorithms.md) · [Policy extensions](extending-policies.md) · [Configuration fields](configuration-reference.md) · [Evaluation](evaluation.md)

These are reusable policy implementations, available to any species in any
compatible environment. The bundled `lstm_ppo` preset selects LSTM PPO for both
species; Discrete SAC and TRPO remain selectable through custom JSON templates.
They require current source; the published 0.0.2.9 distribution predates them.
Every animal owns its parameters, optimizer, buffers and random streams. None
uses shared species training, new automatic mating or action masking.

![Independent actor/critic updates, trust-region policy steps and LSTM state](assets/diagrams/additional-policy-structures.svg)

| JSON `type` | What changes | Main cost/limit |
| --- | --- | --- |
| `discrete_sac` | Categorical actor, twin Q critics, bounded replay and slowly updated target critics | Five networks per animal including targets; fixed configurable entropy temperature |
| `trpo` | Categorical actor updated by a damped natural gradient and KL-constrained backtracking; independent value critic | Repeated rollout passes for conjugate gradients and line search; empirical constraints do not guarantee ecosystem improvement |
| `lstm_ppo` | PPO with a forget-gated LSTM before its MLP head | Stores hidden **and** cell state; chronological truncated BPTT is slower than feedforward PPO |

These provide different learning mechanisms, not evidence of better survival,
mating or descendant performance. Compare frozen policies on matched unseen
seeds, retain proximal reproduction gates and measure offspring that themselves
become parents. Changing selection criteria and the learner together confounds
that comparison.

## Run the bundled LSTM preset

[`environments/lstm_ppo.json`](../environments/lstm_ppo.json) uses the ordinary
PPO preset's flat-world calibration: 64 prey, six predators, 32×24 cells and
proximal mating radii 2/3. Both species use recurrent size 16 and sequence length
8; their MLP heads remain `[16]` and `[64, 32]`, with observation top-k 3/10.
This is an LSTM starting configuration, not evidence that PPO's calibration is
optimal for a recurrent learner. Lifespan champion chaining and full-detail
playback of only the final two scheduled episodes remain enabled.

```sh
.venv/bin/python -m wtf run lstm_ppo --runs 20 --seed 1000 --progress
```

Episodes default to 1,500 steps. Both species start fresh unless you explicitly
load compatible LSTM champions. Add `--record-last-episodes -1` to capture every
episode or `--no-chain-brains` to reset founder weights between episodes.

## Discrete SAC

The actor minimizes the exact categorical expectation of `temperature × log π −
min(Q1, Q2)`. Both critics fit the reward plus discounted target soft value,
computed by summing over every configured action. Target critics follow current
critics by `target_tau` after each update. Death does not bootstrap; time-limit
truncation does. Each animal has a bounded circular replay buffer and samples
with its private training RNG. One Adam step applies the actor/critic gradients
with joint gradient-norm clipping and independent per-parameter moments.

Temperature is fixed and positive: this implementation does not automatically
tune entropy. It supports discrete configured actions, not continuous actions.
The formulation follows [Discrete SAC](https://arxiv.org/abs/1910.07207).

Complete brain-template example:

```json
{
  "type": "discrete_sac",
  "network": {"type": "mlp", "hidden_layers": [32], "activation": "tanh"},
  "learning_rate": 0.001,
  "discount_factor": 0.98,
  "max_grad_norm": 0.5,
  "batch_size": 16,
  "replay_capacity": 512,
  "learning_starts_after": 32,
  "train_every_steps": 4,
  "temperature": 0.1,
  "target_tau": 0.01,
  "observation_version": 4,
  "observation_top_k": 3
}
```

Keep replay capacity conservative with large populations. It is allocated per
animal, not once per species. Checkpoints include the replay cursor, experiences,
both target critics and optimizer state. Warm-start champions contain actor and
both online critics; target critics synchronize to the loaded online critics.

## TRPO

The actor maximizes an importance-weighted advantage surrogate with the optional
entropy term. The exact categorical Fisher-vector product uses forward tangents
and reverse gradients, without finite-difference perturbations of FP32 weights.
A damped conjugate-gradient solve gives the direction. Backtracking accepts only
stored candidate weights with positive sufficient surrogate improvement and
empirical `KL(old || new) <= max_kl`. A rejected search restores the actor exactly.

The scalar value critic has separate parameters and is fitted afterward with
Adam, so value fitting cannot undo the accepted actor KL bound. `learning_rate`,
`batch_size`, `value_epochs`, `value_coefficient` and `max_grad_norm` control that
critic fit. Actor step length is governed by `max_kl` and backtracking, not Adam's
learning rate. This is a small practical implementation of
[TRPO](https://proceedings.mlr.press/v37/schulman15.html), not a claim of monotonic
improvement in a changing multi-agent ecosystem.

```json
{
  "type": "trpo",
  "network": {"type": "mlp", "hidden_layers": [32], "activation": "tanh"},
  "learning_rate": 0.001,
  "discount_factor": 0.98,
  "gae_lambda": 0.95,
  "entropy_coefficient": 0.03,
  "value_coefficient": 0.5,
  "rollout_steps": 64,
  "batch_size": 16,
  "normalize_advantages": true,
  "max_grad_norm": 0.5,
  "value_epochs": 3,
  "max_kl": 0.01,
  "cg_iterations": 10,
  "cg_damping": 0.1,
  "cg_tolerance": 1e-10,
  "backtrack_steps": 10,
  "backtrack_coefficient": 0.5,
  "accept_ratio": 0.1,
  "observation_version": 4,
  "observation_top_k": 3
}
```

Inspector training metrics include `empirical_kl`, `policy_accepted`,
`surrogate_improvement` and `value_loss`. `updates` counts critic optimizer steps;
`policy_updates` separately counts accepted actor steps. Rollout data and pending
choices are checkpointed. There is no replay buffer or PPO clipping parameter.

## LSTM PPO

`lstm_ppo` retains PPO's clipped objective, GAE, entropy term and optimizer. Its
recurrent cell uses input, forget and output sigmoid gates and a tanh candidate.
Hidden and cell vectors each have `recurrent_size` values. The forget bias starts
at one. Read about [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735); memory enables
history-dependent policies but does not guarantee useful long-term behaviour.

```json
{
  "type": "lstm_ppo",
  "network": {"type": "mlp", "hidden_layers": [32], "activation": "tanh"},
  "learning_rate": 0.001,
  "discount_factor": 0.98,
  "gae_lambda": 0.95,
  "clip_ratio": 0.2,
  "entropy_coefficient": 0.03,
  "value_coefficient": 0.5,
  "rollout_steps": 64,
  "epochs": 4,
  "batch_size": 16,
  "max_grad_norm": 0.5,
  "normalize_advantages": true,
  "target_kl": 0.03,
  "recurrent_size": 16,
  "sequence_length": 8,
  "observation_version": 4,
  "observation_top_k": 3
}
```

`batch_size` must cover at least one sequence. Sequence interiors stay
chronological; gradients are truncated at sequence boundaries. Both memory
vectors persist across ordinary steps and checkpoint restore, and reset at death,
time-limit truncation and birth. Evaluation advances memory without learning;
diagnostics read it without changing it. GRPO trials keep private copies of both
vectors and never advance the real animal's memory.

## Select a learner without changing maintained presets

Create the ignored directory with `mkdir -p environments/local`.
Save one complete JSON template above as `environments/local/brain-template.json`.
Then copy an environment into that same ignored local directory and assign the
new template. This example preserves each source species' network widths and
observation/tracking layout for a cleaner learner comparison:

```python
import copy
import json
from pathlib import Path
from wtf.config import load, validate

config = load("ppo")
new = json.loads(Path("environments/local/brain-template.json").read_text())
brains = {}
for name, species in config["species"].items():
    old = config["brains"][species["brain"]]
    brain = copy.deepcopy(new)
    brain["network"] = copy.deepcopy(old["network"])
    for key in ("observation_version", "observation_top_k",
                "movement_history_steps", "scent_observation"):
        brain.pop(key, None)
        if key in old:
            brain[key] = copy.deepcopy(old[key])
    brains[name] = brain
    species["brain"] = name
    species.pop("initial_brain", None)
    species.pop("initial_brains", None)
config["brains"] = brains
config["name"] = new["type"] + "_comparison"
validate(config)
path = Path("environments/local") / (config["name"] + ".json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2) + "\n")
print(path)
```

For example, after choosing the SAC template:

```sh
.venv/bin/python -m wtf run environments/local/discrete_sac_comparison.json \
  --runs 3 --seed 100 --progress
```

To mix algorithms, assign different valid templates to different species; there
is no requirement that prey and predator use the same learner. Preserve source
world/body/reward settings when isolating algorithm effects.

## Inheritance, storage and viewer contracts

- New algorithm identifiers have distinct brain signatures. Existing PPO/GRU
  weights cannot initialize them merely because some dimensions match.
- Sexual crossover mixes all online parameter arrays, including separate
  critics. Newborns receive fresh optimizers, empty buffers, zero counters and
  independent RNGs. SAC targets synchronize after crossover/mutation; LSTM
  hidden and cell state start at zero. Frozen evaluation copies the same-sex
  parent under the existing rules.
- Champions remain species/sex-specific. All online networks are included in
  lossless weight storage; exact recovery additionally includes training state,
  SAC targets/replay and LSTM memory. No artifact format version changes.
- Viewer predictions remain **action probabilities**. SAC Q estimates are
  diagnostic values, not probabilities. Actor-route diagrams exclude separate
  critics; LSTM activity is labelled as LSTM. Archived LSTM explanations use
  zero hidden/cell state and say so; sampled replay activity shows actual memory.
- These implementations use existing scalar/native dense helpers. They add no
  GPU path, dependency or claim of native cross-animal batching. LSTM now has
  optional [native cell kernels](performance.md#native-lstm-forward-and-backward);
  older extensions and FP64 models retain the Python cell.
  TRPO and LSTM can be substantially slower than ordinary PPO; no performance
  ranking or ecological advantage has been established.
