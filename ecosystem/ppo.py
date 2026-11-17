"""Independent categorical PPO with GAE, clipped objectives and Adam.

A small standard-library implementation for inspectable CPU experiments.
Each animal owns all network, optimizer, rollout, and random-generator state.
"""

import copy
import math
import random
from operator import mul
from typing import NamedTuple
from .brain import INPUTS
from .optimization import AdamOptimizer


class RolloutStep(NamedTuple):
    observation: tuple
    action: int
    reward: float
    log_probability: float
    value: float
    next_value: float
    terminated: bool
    truncated: bool


def policy_distribution(logits):
    maximum = max(logits)
    exponentials = [math.exp(z - maximum) for z in logits]
    normalizer = sum(exponentials)
    log_normalizer = math.log(normalizer)
    log_probabilities = [(z - maximum) - log_normalizer for z in logits]
    probabilities = [v / normalizer for v in exponentials]
    return probabilities, log_probabilities


def advantages_and_returns(rollout, discount, gae_lambda):
    advantages = [0.0] * len(rollout)
    trace = 0.0
    for i in range(len(rollout) - 1, -1, -1):
        step = rollout[i]
        bootstrap = 0.0 if step.terminated else step.next_value
        delta = step.reward + discount * bootstrap - step.value
        continuation = 0.0 if step.terminated or step.truncated else 1.0
        trace = delta + discount * gae_lambda * continuation * trace
        advantages[i] = trace
    return advantages, [a + step.value for a, step in zip(advantages, rollout)]


def loss_and_gradient(outputs, action, old_log_probability, advantage, target, config):
    """Loss and output derivatives, separated for numerical gradient checks."""
    probabilities, logs = policy_distribution(outputs[:-1])
    log_ratio = max(-60.0, min(60.0, logs[action] - old_log_probability))
    ratio = math.exp(log_ratio)
    clip = config["clip_ratio"]
    clipped_ratio = max(1.0 - clip, min(1.0 + clip, ratio))
    clipped = (advantage > 0 and ratio > 1.0 + clip) or (
        advantage < 0 and ratio < 1.0 - clip
    )
    policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
    d_logp = 0.0 if clipped else -advantage * ratio
    entropy = -sum(p * logp for p, logp in zip(probabilities, logs))
    gradient = [
        d_logp * ((1.0 if j == action else 0.0) - p)
        + config["entropy_coefficient"] * p * (logs[j] + entropy)
        for j, p in enumerate(probabilities)
    ]
    error = outputs[-1] - target
    value_loss = 0.5 * error * error
    gradient.append(config["value_coefficient"] * error)
    loss = (
        policy_loss
        + config["value_coefficient"] * value_loss
        - config["entropy_coefficient"] * entropy
    )
    info = dict(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approximate_kl=(ratio - 1.0) - log_ratio,
        clip_fraction=float(abs(ratio - 1.0) > clip),
    )
    return loss, gradient, info


class MLP:
    def __init__(self, hidden_layers, actions, seed, value_head=True, inputs=INPUTS):
        self.sizes = [inputs, *hidden_layers, actions + int(value_head)]
        rng = random.Random(seed)
        self.layers = []
        for index, (inputs, outputs) in enumerate(zip(self.sizes, self.sizes[1:])):
            scale = math.sqrt(6 / (inputs + outputs))
            weights = [rng.uniform(-scale, scale) for _ in range(inputs * outputs)]
            if index == len(self.sizes) - 2:
                # Near-uniform initial policy; the last output is the critic.
                for i in range(inputs * actions):
                    weights[i] *= 0.01
            self.layers.append(dict(weights=weights, bias=[0.0] * outputs))

    def forward(self, observation):
        activations = [tuple(observation)]
        for index, layer in enumerate(self.layers):
            previous = activations[-1]
            inputs = self.sizes[index]
            weights = layer["weights"]
            values = [
                bias + sum(map(mul, previous, weights[j * inputs : (j + 1) * inputs]))
                for j, bias in enumerate(layer["bias"])
            ]
            activations.append(
                [math.tanh(v) for v in values]
                if index < len(self.layers) - 1
                else values
            )
        return activations[-1], activations

    def _predict(self, observation, value_only=False, output_range=None):
        """Inference keeps one layer at a time; no backpropagation tape."""
        previous = tuple(observation)
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            inputs, weights = self.sizes[index], layer['weights']
            if value_only and index == last:
                output = len(layer['bias']) - 1
                return layer['bias'][output] + sum(map(mul, previous, weights[output*inputs:(output+1)*inputs]))
            if output_range is not None and index == last:
                start, stop = output_range
                return [layer['bias'][j] + sum(map(mul, previous, weights[j*inputs:(j+1)*inputs]))
                        for j in range(start, stop)]
            values = [bias + sum(map(mul, previous, weights[j*inputs:(j+1)*inputs]))
                      for j,bias in enumerate(layer['bias'])]
            previous = [math.tanh(v) for v in values] if index < last else values
        return previous

    def predict(self, observation):
        return self._predict(observation)

    def value(self, observation):
        """Final output only, used for actor-critic value bootstrapping."""
        return self._predict(observation, value_only=True)

    def predict_range(self, observation, start, stop):
        """Evaluate only the requested final-layer heads, preserving their arithmetic."""
        if not 0 <= start <= stop <= self.sizes[-1]:
            raise ValueError('Invalid output range')
        return self._predict(observation, output_range=(start, stop))

    @staticmethod
    def clear_gradients(gradients):
        # Local to one update call, never a persistent per-animal buffer.
        for layer in gradients:
            for values in layer.values():
                values[:] = [0.0] * len(values)

    def zeros(self):
        return [
            {key: [0.0] * len(values) for key, values in layer.items()}
            for layer in self.layers
        ]

    def accumulate_gradient(self, activations, output_gradient, gradients, return_input_gradient=False):
        delta = output_gradient
        for index in range(len(self.layers) - 1, -1, -1):
            previous = activations[index]
            inputs = self.sizes[index]
            weights = self.layers[index]["weights"]
            dw, db = gradients[index]["weights"], gradients[index]["bias"]
            if index or return_input_gradient:
                previous_gradient = [0.0] * inputs
                for j, derivative in enumerate(delta):
                    db[j] += derivative
                    offset = j * inputs
                    for i, value in enumerate(previous):
                        parameter = offset + i
                        dw[parameter] += derivative * value
                        previous_gradient[i] += derivative * weights[parameter]
                if not index:
                    return previous_gradient
                delta = [
                    g * (1.0 - value * value)
                    for g, value in zip(previous_gradient, previous)
                ]
            else:
                # Input features need no propagated gradient. Keep that branch
                # outside the loop that visits every input-layer weight.
                for j, derivative in enumerate(delta):
                    db[j] += derivative
                    offset = j * inputs
                    for i, value in enumerate(previous):
                        dw[offset + i] += derivative * value

    def checkpoint(self):
        return [
            {key: values.copy() for key, values in layer.items()}
            for layer in self.layers
        ]


class PPOBrain(AdamOptimizer):
    algorithm = "ppo"

    def __init__(self, config, actions, base_seed, individual_seed):
        self.config = copy.deepcopy(config)
        self.actions = list(actions)
        self.action_indices = {name: i for i, name in enumerate(actions)}
        self.network = self._new_network(config, len(actions), base_seed)
        self.rng = random.Random(individual_seed)
        self.training_rng = random.Random(individual_seed ^ 0x5F3759DF)
        self.moments = self.network.zeros()
        self.variances = self.network.zeros()
        self.rollout = []
        self.pending = None
        self.steps = self.updates = self.rollout_updates = 0
        self.last_metrics = {}

    def _new_network(self, config, actions, seed):
        return MLP(config['network']['hidden_layers'], actions, seed)

    def distribution(self, outputs, observation):
        return policy_distribution(outputs[:-1])

    def objective(self, outputs, step, advantage, target):
        return loss_and_gradient(outputs, step.action, step.log_probability,
                                 advantage, target, self.config)

    def action_probabilities(self, observation):
        return self.distribution(self.network.predict(observation), observation)[0]

    def choose(self, observation, explore=True, random_baseline=False):
        if random_baseline:
            return self.rng.choice(self.actions)
        outputs = self.network.predict(observation)
        probabilities, logs = self.distribution(outputs, observation)
        if not explore:
            return self.actions[
                max(range(len(probabilities)), key=probabilities.__getitem__)
            ]
        draw, cumulative = self.rng.random(), 0.0
        action = len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative:
                action = index
                break
        self.pending = (tuple(observation), action, logs[action], outputs[-1])
        return self.actions[action]

    def learn(self, experience):
        action = self.action_indices[experience.action]
        if self.pending is not None and self.pending[:2] == (
            tuple(experience.observation),
            action,
        ):
            _, _, old_logp, value = self.pending
        else:
            outputs = self.network.predict(experience.observation)
            old_logp = self.distribution(outputs, experience.observation)[1][action]
            value = outputs[-1]
        self.pending = None
        if experience.terminated:
            next_value = 0.0
        else:
            if experience.next_observation is None:
                raise ValueError(
                    "PPO requires a next observation for nonterminal experience"
                )
            next_value = self.network.value(experience.next_observation)
        self.rollout.append(
            RolloutStep(
                tuple(experience.observation),
                action,
                experience.reward,
                old_logp,
                value,
                next_value,
                experience.terminated,
                experience.truncated,
            )
        )
        self.steps += 1
        if (
            len(self.rollout) >= self.config["rollout_steps"]
            or experience.terminated
            or experience.truncated
        ):
            self._update()

    def _update(self):
        c = self.config
        advantages, targets = advantages_and_returns(
            self.rollout, c["discount_factor"], c["gae_lambda"]
        )
        if c["normalize_advantages"] and len(advantages) > 1:
            mean = sum(advantages) / len(advantages)
            deviation = math.sqrt(
                sum((v - mean) ** 2 for v in advantages) / len(advantages) + 1e-8
            )
            for i, value in enumerate(advantages):
                advantages[i] = (value - mean) / deviation
        totals, samples, stopped = {}, 0, False
        gradients = None
        indices = list(range(len(self.rollout)))
        for _ in range(c["epochs"]):
            self.training_rng.shuffle(indices)
            for offset in range(0, len(indices), c["batch_size"]):
                batch = indices[offset : offset + c["batch_size"]]
                if gradients is None:
                    gradients = self.network.zeros()
                else:
                    self.network.clear_gradients(gradients)
                batch_metrics = {}
                for i in batch:
                    step = self.rollout[i]
                    outputs, activations = self.network.forward(step.observation)
                    _, derivative, info = self.objective(
                        outputs, step, advantages[i], targets[i]
                    )
                    self.network.accumulate_gradient(activations, derivative, gradients)
                    for key, value in info.items():
                        batch_metrics[key] = batch_metrics.get(key, 0.0) + value
                # Stop before an excessive policy update; value and policy
                # share a trunk, so the whole minibatch is skipped on this cap.
                target_kl = c.get("target_kl", 0.03)
                if (
                    target_kl
                    and batch_metrics["approximate_kl"] / len(batch) > target_kl
                ):
                    stopped = True
                    break
                self._adam_step(gradients, len(batch))
                samples += len(batch)
                for key, value in batch_metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
            if stopped:
                break
        self.rollout_updates += 1
        self.last_metrics = (
            {key: value / samples for key, value in totals.items()} if samples else {}
        )
        self.last_metrics["early_stopped"] = stopped
        self.rollout.clear()  # Old-policy samples must never enter another rollout.

    def diagnostics(self, observation):
        outputs = self.network.predict(observation)
        probabilities, logs = self.distribution(outputs, observation)
        return dict(
            algorithm=self.algorithm,
            buffer_kind="rollout",
            buffer_size=len(self.rollout),
            features=observation,
            action_probabilities=dict(zip(self.actions, probabilities)),
            state_value=outputs[-1],
            entropy=-sum(p * logp for p, logp in zip(probabilities, logs)),
            updates=self.updates,
            rollout_updates=self.rollout_updates,
            rollout_size=len(self.rollout),
            training_metrics=self.last_metrics.copy(),
        )

    def checkpoint(self):
        return dict(
            algorithm=self.algorithm,
            config=copy.deepcopy(self.config),
            actions=self.actions[:],
            network=self.network.checkpoint(),
            moments=copy.deepcopy(self.moments),
            variances=copy.deepcopy(self.variances),
            rollout=[step._asdict() for step in self.rollout],
            pending=self.pending,
            steps=self.steps,
            updates=self.updates,
            rollout_updates=self.rollout_updates,
            last_metrics=self.last_metrics.copy(),
            rng_state=self.rng.getstate(),
            training_rng_state=self.training_rng.getstate(),
        )

    @classmethod
    def restore(cls, state):
        b = cls(state["config"], state["actions"], 0, 0)
        b.network.layers = copy.deepcopy(state["network"])
        for key in (
            "moments",
            "variances",
            "steps",
            "updates",
            "rollout_updates",
            "last_metrics",
        ):
            setattr(b, key, copy.deepcopy(state[key]))
        b.rollout = [
            RolloutStep(
                tuple(s["observation"]),
                s["action"],
                s["reward"],
                s["log_probability"],
                s["value"],
                s["next_value"],
                s["terminated"],
                s["truncated"],
            )
            for s in state["rollout"]
        ]
        pending = state.get("pending")
        b.pending = None if pending is None else (tuple(pending[0]), *pending[1:])

        def tuples(value):
            return (
                tuple(tuples(v) for v in value)
                if isinstance(value, (list, tuple))
                else value
            )

        b.rng.setstate(tuples(state["rng_state"]))
        b.training_rng.setstate(tuples(state["training_rng_state"]))
        return b
