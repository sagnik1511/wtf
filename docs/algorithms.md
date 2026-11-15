# Learning algorithms and networks

[Project overview](../README.md) · [Configuration](configuration.md) · [Analysis](analysis.md)

Commands in this guide run from the project root.

The `species.<name>.brain` field selects a template in `brains`. Each animal receives its own model, optimizer, random streams and experience storage; referencing a common template never shares live weights between animals. The supplied baseline remains the existing Q-network comparison.

```sh
.venv/bin/python -m ecosystem.cli environments/ppo.json --runs 1
.venv/bin/python -m ecosystem.cli environments/mixed_brains.json --runs 1
```

Per-species `learning_rate_multiplier` scales a template's learning rate without
changing training frequency. Presets use prey 1.0× and predator 1.25×; see
[species learning rates](configuration.md#learning-rates-per-species).

Supported algorithms:

| Template `type` | Model and training |
| --- | --- |
| `q_network` | Existing one-hidden-layer action-value network, replay, target network, epsilon-greedy policy and SGD |
| `ppo` | Categorical policy and scalar critic sharing an MLP, recent on-policy rollouts, GAE, clipped policy loss, entropy bonus and Adam |
| `a2c` | Independent advantage actor-critic, one gradient update per rollout, GAE and entropy regularization |
| `reinforce` | Actor-only episodic Monte Carlo policy gradient with a running baseline from previous episodes |
| `grpo` | Experimental actor-only clipped policy updates using comparable simulator trajectory groups and a frozen reference policy |

A PPO template (placed under `brains.small_ppo`) is:

```json
{
  "type": "ppo",
  "network": {
    "type": "mlp",
    "hidden_layers": [
      16
    ],
    "activation": "tanh"
  },
  "learning_rate": 0.001,
  "discount_factor": 0.98,
  "gae_lambda": 0.95,
  "clip_ratio": 0.2,
  "entropy_coefficient": 0.01,
  "value_coefficient": 0.5,
  "rollout_steps": 64,
  "epochs": 4,
  "batch_size": 16,
  "max_grad_norm": 0.5,
  "normalize_advantages": true,
  "target_kl": 0.03
}
```

Set either species' `brain` to `"small_ppo"` to use that template. All policy-based templates (`ppo`, `a2c`, `reinforce`, `grpo`) accept one to four tanh layers in `network.hidden_layers`. Change `[16]` to `[32, 16]`, for example, to experiment with a deeper DNN. The network architecture is separate from the PPO algorithm. The legacy Q-network configuration retains its single-layer format and behavior.

PPO samples actions from its categorical policy in training; greedy evaluation and the random-action baseline do not train. It stores at most `rollout_steps` transitions per animal and consumes them after each update. Buffers also flush when an animal dies or a run reaches its step limit, including short partial rollouts. Death disables value bootstrapping; time-limit truncation bootstraps the final observation but breaks the episode trace. Advantages are normalized only within that individual's rollout when enabled. Critic targets are the GAE lambda returns; critic loss is un-clipped mean squared error with the conventional one-half factor. `epochs` and `batch_size` control repeated minibatch passes. The shared gradient is clipped by `max_grad_norm`. Adam uses beta1 0.9, beta2 0.999 and epsilon 1e-8. A positive `target_kl` stops remaining minibatches if their approximate policy KL exceeds the cap; zero disables that check. Extreme log-ratios are bounded to ±60 for numerical safety.

The implementation follows the clipped PPO objective and GAE structure described in the [PPO paper](https://arxiv.org/abs/1707.06347) and [Spinning Up's PPO reference](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py). It is a small independent standard-library implementation, with numerical gradient checks, rather than a wrapper around PyTorch or an established training library.

PPO checkpoints include model and Adam state, partial rollouts with old log probabilities/values, pending action data and random-generator states. Mixed-algorithm experiments and legacy Q checkpoints restore through the same runner. Retaining brains preserves model/optimizer state across runs, while PPO's consumed rollout data is discarded. The viewer shows action probabilities, entropy and critic value for PPO; Q-learning recordings continue to show action values and epsilon. Existing recordings without an algorithm label remain supported.

PPO needs multiple passes over recent data and maintains optimizer moments, so it is not automatically cheaper or better at hunting than Q-learning. Small layers, short rollouts and modest epoch counts keep this implementation suitable for CPU experiments. Test algorithms with held-out seeds and matched physical configurations; an implementation test or a few kills is not evidence of a trained policy outperforming the baseline. The additional algorithms below share the same configuration, checkpoint and playback interfaces.

## A2C and REINFORCE

```sh
.venv/bin/python -m ecosystem.cli environments/a2c.json --runs 1 --progress
.venv/bin/python -m ecosystem.cli environments/reinforce.json --runs 1 --progress
```

A2C uses the same configurable shared policy/value MLP as PPO, with one full-rollout advantage actor-critic update instead of repeated clipped-objective epochs. GAE targets, terminal handling, entropy regularization and gradient clipping are retained. Its structure follows the advantage actor-critic family represented by the [Baselines A2C reference](https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py), with independent per-animal models rather than shared population weights.

REINFORCE has no critic. It collects one animal's complete finite episode, computes discounted rewards-to-go and the summed discounted log-policy gradient, and applies one Adam update. The scalar baseline is derived only from earlier episodes, then updated using `baseline_decay` after the current update. Optional advantage normalization is disabled in the example. This is an episodic implementation of the [REINFORCE policy-gradient family](https://link.springer.com/article/10.1007/BF00992696), with entropy regularization and gradient clipping. Time limits define finite episodes for this actor-only method; they are not bootstrapped from a value network. To bound memory without silently treating arbitrary fragments as complete episodes, `max_episode_steps` must be at least `world.max_steps`. Increase both when requesting longer REINFORCE runs. The selected completed-episode policy can seed the next run through champion chaining; the new controller starts with a fresh baseline. Its policy does not change midway through a collected episode.

## Experimental GRPO

```sh
.venv/bin/python -m ecosystem.cli environments/grpo.json --runs 1 --progress
```

This is a finite-horizon simulator adaptation of the group-relative clipped policy objective in [DeepSeekMath](https://arxiv.org/html/2402.03300v3), rather than the original language-model training setup. Each focal animal samples `group_size` alternative trajectories from the **same current world state**. Environment RNG and other agents' action RNG streams restart from matching states for every trial. Other policies remain frozen during the group, and no real controller's `choose`/`learn` methods are called inside a trial. Only the focal policy uses different trial action samples. Its observations remain local; rewards come from the ordinary environment rules.

Discounted returns are standardized within that group, never across unrelated animals. Every step in a trajectory receives its trajectory's group-relative advantage. The objective is averaged per trajectory and then per group, so shorter trajectories are not accidentally underweighted. Clipped importance ratios use the sampling policy's old log probabilities. Because there are few discrete actions, the reference penalty uses exact categorical KL against a frozen copy of the animal's initial policy. There is no critic. Equal-return groups are filtered out entirely, including regularization, to avoid updates without a relative reward signal; the inspector reports group counts and reward spread. A single real transition advances scheduling; GRPO's learning signal comes from the matched trial outcomes.

The default example uses four trials, four steps per trial, `train_every_steps: 128` per agent, and at most one group per real world step (`experiment.grpo_updates_per_step: 1`). Groups are scheduled in rotating order. Trials run sequentially; only one cloned world, small previous RNG states, and bounded trial observations are retained. No other agent's replay buffer or network is copied for a trial. Larger groups/horizons can substantially increase training time. The theoretical extra world-step budget per real step is at most `grpo_updates_per_step * group_size * rollout_horizon`, with fewer steps when no agents are due or trials terminate early.

Trial worlds do not advance the real simulation or contribute kills, deaths, food, or frame recordings to it. Progress and playback counts describe only the real timeline. Checkpoints preserve the frozen reference policy, group/training RNG states and update scheduling, so a restored run reproduces its future trials. The example keeps prey on Q-learning to limit overhead; either species can reference a GRPO template.

All five algorithms can coexist through species-to-brain references. Their checkpoints include the relevant partial rollout/episode buffers and optimizer state. The inspector shows action probabilities for policy-based methods, Q-values for Q-learning, critic values only where a critic exists, and episode/group details where applicable. Algorithm correctness tests do not establish better ecosystem performance; compare multiple held-out seeds before drawing that conclusion.

See [saved champions](brain-library.md) for transferring compatible learned weights to a fresh experiment, and [analysis](analysis.md) for checking whether updates lead to useful behavior.
