"""Independent A2C, episodic REINFORCE, and experimental branched GRPO."""

import copy
import math
import random
from typing import NamedTuple
from wtf.learning.activation_diagnostics import activation_sample
from wtf.native.numeric import packed, plain, precision_of, restore_numeric
from wtf.learning.brain import observation_inputs
from wtf.learning.optimization import AdamOptimizer
from wtf.learning.ppo import MLP, PPOBrain, advantages_and_returns, policy_distribution


def actor_loss_gradient(
    logits,
    action,
    advantage,
    entropy_coefficient,
    old_logp=None,
    clip_ratio=0.2,
    reference_logits=None,
    kl_coefficient=0.0,
):
    """Compute the categorical actor loss and its logit derivative."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-positional-arguments

    probabilities, logs = policy_distribution(logits)
    entropy = -sum(p * logp for p, logp in zip(probabilities, logs))
    approximate_kl = 0.0
    if old_logp is None:
        policy_loss, derivative = -logs[action] * advantage, -advantage
    else:
        log_ratio = max(-60.0, min(60.0, logs[action] - old_logp))
        ratio = math.exp(log_ratio)
        clipped = max(1.0 - clip_ratio, min(1.0 + clip_ratio, ratio))
        policy_loss = -min(ratio * advantage, clipped * advantage)
        inactive = (advantage > 0 and ratio > 1.0 + clip_ratio) or (
            advantage < 0 and ratio < 1.0 - clip_ratio
        )
        derivative = 0.0 if inactive else -advantage * ratio
        approximate_kl = (ratio - 1.0) - log_ratio
    gradient = [
        derivative * ((1.0 if i == action else 0.0) - p)
        + entropy_coefficient * p * (logs[i] + entropy)
        for i, p in enumerate(probabilities)
    ]
    reference_kl = 0.0
    if reference_logits is not None and kl_coefficient:
        reference_logs = policy_distribution(reference_logits)[1]
        differences = [lp - rp for lp, rp in zip(logs, reference_logs)]
        reference_kl = sum(p * d for p, d in zip(probabilities, differences))
        for i, p in enumerate(probabilities):
            gradient[i] += kl_coefficient * p * (differences[i] - reference_kl)
    loss = policy_loss - entropy_coefficient * entropy + kl_coefficient * reference_kl
    return (
        loss,
        gradient,
        {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "approximate_kl": approximate_kl,
            "reference_kl": reference_kl,
        },
    )


def normalized(values, *, inplace=False):
    """Normalize advantages; fresh local vectors can reuse their own storage."""
    if len(values) <= 1:
        return values
    mean = sum(values) / len(values)
    scale = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values) + 1e-8)
    if inplace:
        for i, value in enumerate(values):
            values[i] = (value - mean) / scale
        return values
    return [(v - mean) / scale for v in values]


class A2CBrain(PPOBrain):
    """Advantage actor-critic using its configured rollout update schedule."""

    algorithm = "a2c"

    def _update(self):
        c = self.config
        advantages, targets = advantages_and_returns(
            self.rollout, c["discount_factor"], c["gae_lambda"]
        )
        if c["normalize_advantages"]:
            advantages = normalized(advantages, inplace=True)
        gradients = self.network.zeros()
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for step, advantage, target in zip(self.rollout, advantages, targets):
            outputs, activations = self.network.forward(step.observation)
            _, derivative, metrics = actor_loss_gradient(
                outputs[:-1], step.action, advantage, c["entropy_coefficient"]
            )
            error = outputs[-1] - target
            derivative.append(c["value_coefficient"] * error)
            self.network.accumulate_gradient(activations, derivative, gradients)
            totals["policy_loss"] += metrics["policy_loss"]
            totals["entropy"] += metrics["entropy"]
            totals["value_loss"] += 0.5 * error * error
        self._adam_step(gradients, len(self.rollout))
        self.last_metrics = {
            key: value / len(self.rollout) for key, value in totals.items()
        }
        self.rollout_updates += 1
        self.rollout.clear()


class PolicyStep(NamedTuple):
    """One policy transition including the action log probability."""

    observation: tuple
    action: int
    reward: float
    log_probability: float
    terminated: bool
    truncated: bool


class ActorBrain(AdamOptimizer):
    """Common categorical policy and checkpointing, without a value network."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-many-instance-attributes

    # Subclasses select the algorithm; only GRPO creates its guarded extra state.
    algorithm: str
    reference_network: MLP
    group_rng: random.Random

    def __init__(self, config, actions, base_seed, individual_seed):
        self.config, self.actions = copy.deepcopy(config), list(actions)
        self.action_indices = {name: i for i, name in enumerate(actions)}
        self.network = MLP(
            config["network"]["hidden_layers"],
            len(actions),
            base_seed,
            value_head=False,
            inputs=observation_inputs(config),
        )
        self.rng = random.Random(individual_seed)
        self.training_rng = random.Random(individual_seed ^ 0x5F3759DF)
        self.moments, self.variances = packed(self.network.zeros()), packed(
            self.network.zeros()
        )
        self.trajectory, self.pending = [], None
        self.steps = self.updates = self.rollout_updates = 0
        self.baseline = 0.0
        self.last_metrics = {}

    def action_probabilities(self, observation):
        """Return the categorical action probabilities for an observation."""
        return policy_distribution(self.network.predict(observation))[0]

    def choose(self, observation, explore=True, random_baseline=False):
        """Choose an action using this animal's private policy and exploration state."""
        if random_baseline:
            return self.rng.choice(self.actions)
        logits = self.network.predict(observation)
        return self._choose_outputs(observation, logits, explore)

    def _choose_outputs(self, observation, logits, explore):
        probabilities, logs = policy_distribution(logits)
        if not explore:
            return self.actions[
                max(range(len(probabilities)), key=probabilities.__getitem__)
            ]
        draw, cumulative, action = self.rng.random(), 0.0, len(self.actions) - 1
        for i, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative:
                action = i
                break
        self.pending = (tuple(observation), action, logs[action])
        return self.actions[action]

    def diagnostics(self, observation, include_activations=False):
        """Return predictions and optional activation samples for inspection."""
        if include_activations:
            outputs, layers = self.network.forward(observation)
        else:
            outputs = self.network.predict(observation)
        probabilities, logs = policy_distribution(outputs)
        info = {
            "algorithm": self.algorithm,
            "features": observation,
            "action_probabilities": dict(zip(self.actions, probabilities)),
            "entropy": -sum((p * lp for p, lp in zip(probabilities, logs))),
            "updates": self.updates,
            "rollout_updates": self.rollout_updates,
            "buffer_kind": "episode" if self.algorithm == "reinforce" else "group",
            "buffer_size": len(self.trajectory),
            "training_metrics": copy.deepcopy(self.last_metrics),
        }
        if self.algorithm == "reinforce":
            info["return_baseline"] = self.baseline
        else:
            info.update(
                group_size=self.config["group_size"],
                rollout_horizon=self.config["rollout_horizon"],
            )
        return activation_sample(info, layers[1:-1]) if include_activations else info

    def checkpoint(self):
        """Serialize state required to resume this component."""
        state = {
            "algorithm": self.algorithm,
            "precision": precision_of(self),
            "config": copy.deepcopy(self.config),
            "actions": self.actions[:],
            "network": self.network.checkpoint(),
            "moments": plain(self.moments),
            "variances": plain(self.variances),
            "trajectory": [s._asdict() for s in self.trajectory],
            "pending": self.pending,
            "steps": self.steps,
            "updates": self.updates,
            "rollout_updates": self.rollout_updates,
            "baseline": self.baseline,
            "last_metrics": copy.deepcopy(self.last_metrics),
            "rng_state": self.rng.getstate(),
            "training_rng_state": self.training_rng.getstate(),
        }
        if self.algorithm == "grpo":
            state.update(
                reference_network=self.reference_network.checkpoint(),
                group_rng_state=self.group_rng.getstate(),
                last_group_step=self.last_group_step,
            )
        return state

    @classmethod
    def restore(cls, state):
        """Reconstruct the component from its persisted continuation state."""

        # This cache or restored state is rebuilt at its owning lifecycle boundary,
        # not eagerly.
        # pylint: disable=attribute-defined-outside-init

        b = cls(state["config"], state["actions"], 0, 0)
        for key in (
            "steps",
            "updates",
            "rollout_updates",
            "baseline",
            "last_metrics",
        ):
            setattr(b, key, copy.deepcopy(state[key]))
        restore_numeric(b, state)
        b.trajectory = [
            PolicyStep(
                tuple(s["observation"]),
                s["action"],
                s["reward"],
                s["log_probability"],
                s["terminated"],
                s["truncated"],
            )
            for s in state["trajectory"]
        ]
        pending = state["pending"]
        b.pending = None if pending is None else (tuple(pending[0]), *pending[1:])

        def tuples(value):
            return (
                tuple(tuples(v) for v in value)
                if isinstance(value, (list, tuple))
                else value
            )

        b.rng.setstate(tuples(state["rng_state"]))
        b.training_rng.setstate(tuples(state["training_rng_state"]))
        if b.algorithm == "grpo":
            b.group_rng.setstate(tuples(state["group_rng_state"]))
            b.last_group_step = state["last_group_step"]
        return b


class ReinforceBrain(ActorBrain):
    """Episode-return policy gradient with a private running baseline."""

    algorithm = "reinforce"

    def learn(self, experience):
        """Consume a transition and apply the configured learning schedule."""
        if len(self.trajectory) >= self.config["max_episode_steps"]:
            raise ValueError("REINFORCE episode exceeded max_episode_steps")
        action = self.action_indices[experience.action]
        if self.pending is not None and self.pending[:2] == (
            tuple(experience.observation),
            action,
        ):
            old_logp = self.pending[2]
        else:
            old_logp = policy_distribution(
                self.network.predict(experience.observation)
            )[1][action]
        self.pending = None
        self.trajectory.append(
            PolicyStep(
                tuple(experience.observation),
                action,
                experience.reward,
                old_logp,
                experience.terminated,
                experience.truncated,
            )
        )
        self.steps += 1
        if experience.terminated or experience.truncated:
            self._update()

    def _update(self):

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-locals

        c = self.config
        returns, total = [0.0] * len(self.trajectory), 0.0
        for i in range(len(self.trajectory) - 1, -1, -1):
            total = self.trajectory[i].reward + c["discount_factor"] * total
            returns[i] = total
        mean_return = sum(returns) / len(returns)
        advantages = returns
        for i, value in enumerate(advantages):
            advantages[i] = (c["discount_factor"] ** i) * (value - self.baseline)
        if c["normalize_advantages"]:
            advantages = normalized(advantages, inplace=True)
        gradients = self.network.zeros()
        totals = {"policy_loss": 0.0, "entropy": 0.0}
        for step, advantage in zip(self.trajectory, advantages):
            logits, activations = self.network.forward(step.observation)
            _, derivative, info = actor_loss_gradient(
                logits, step.action, advantage, c["entropy_coefficient"]
            )
            self.network.accumulate_gradient(activations, derivative, gradients)
            for key in totals:
                totals[key] += info[key]
        count = len(self.trajectory)
        # Episodic REINFORCE uses the summed discounted policy gradient.
        self._adam_step(gradients, 1)
        self.baseline = (
            c["baseline_decay"] * self.baseline
            + (1.0 - c["baseline_decay"]) * mean_return
        )
        self.last_metrics = {key: value / count for key, value in totals.items()}
        self.last_metrics["mean_return"] = mean_return
        self.rollout_updates += 1
        self.trajectory.clear()


class GRPOBrain(ActorBrain):
    """Experimental GRPO for finite simulator branches, not unrelated agents."""

    algorithm = "grpo"

    def __init__(self, config, actions, base_seed, individual_seed):
        super().__init__(config, actions, base_seed, individual_seed)
        self.reference_network = MLP(
            config["network"]["hidden_layers"],
            len(actions),
            base_seed,
            value_head=False,
            inputs=observation_inputs(config),
        )
        self.group_rng = random.Random(individual_seed ^ 0x0BADC0DE)
        self.last_group_step = 0

    @property
    def needs_group(self):
        """Return whether the GRPO controller is due for another trial group."""
        return self.steps - self.last_group_step >= self.config["train_every_steps"]

    def learn(self, experience):
        # Training data comes from matched simulator trials collected by the
        # runner. A single real transition is not a comparable outcome group.
        """Consume a transition and apply the configured learning schedule."""

        # The callback keeps its caller's keyword signature; this branch does not
        # consume every input.
        # pylint: disable=unused-argument

        self.steps += 1
        self.pending = None

    def update_group(self, trajectories):
        """Apply one grouped policy update from matched trial trajectories."""

        # Keep independent numeric/compatibility or benchmark paths explicit to
        # preserve their exact contracts.
        # pylint: disable=duplicate-code

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-locals

        c = self.config
        if len(trajectories) < 2 or any(not t for t in trajectories):
            raise ValueError("GRPO needs at least two nonempty matched trajectories")
        returns = [
            sum(
                (c["discount_factor"] ** t) * step.reward
                for t, step in enumerate(trajectory)
            )
            for trajectory in trajectories
        ]
        mean = sum(returns) / len(returns)
        deviation = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
        self.last_group_step = self.steps
        self.rollout_updates += 1
        self.last_metrics = {
            "group_returns": returns,
            "mean_return": mean,
            "return_std": deviation,
            "zero_variance": deviation < 1e-08,
            "early_stopped": False,
        }
        if deviation < 1e-8:
            return
        count = sum(len(t) for t in trajectories)
        samples = []
        for trajectory, result in zip(trajectories, returns):
            advantage = (result - mean) / (deviation + 1e-8)
            weight = count / (len(trajectories) * len(trajectory))
            samples.extend((step, advantage, weight) for step in trajectory)
        gradients = None
        indices = list(range(count))
        for _ in range(c["epochs"]):
            self.training_rng.shuffle(indices)
            for offset in range(0, count, c["batch_size"]):
                batch = indices[offset : offset + c["batch_size"]]
                if gradients is None:
                    gradients = self.network.zeros()
                else:
                    self.network.clear_gradients(gradients)
                approximate_kl = 0.0
                for index in batch:
                    step, advantage, weight = samples[index]
                    logits, activations = self.network.forward(step.observation)
                    reference = (
                        self.reference_network.predict(step.observation)
                        if c["kl_coefficient"]
                        else None
                    )
                    _, derivative, info = actor_loss_gradient(
                        logits,
                        step.action,
                        advantage,
                        c["entropy_coefficient"],
                        old_logp=step.log_probability,
                        clip_ratio=c["clip_ratio"],
                        reference_logits=reference,
                        kl_coefficient=c["kl_coefficient"],
                    )
                    self.network.accumulate_gradient(
                        activations, [value * weight for value in derivative], gradients
                    )
                    approximate_kl += info["approximate_kl"]
                target_kl = c.get("target_kl", 0.03)
                if target_kl and approximate_kl / len(batch) > target_kl:
                    self.last_metrics["early_stopped"] = True
                    return
                self._adam_step(gradients, len(batch))
