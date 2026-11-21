"""Small independent Q network, replay, target network and clipped loss.

Packed float32 parameters and standard-library math keep installation self-contained.
Networks are small deliberately; immutable replay tuples reduce overhead.
"""

from wtf.learning.activation_diagnostics import activation_sample
from wtf.native import kernels
from wtf.native.kernels import gradient_zeros
from wtf.native.numeric import Float32, packed, plain, precision_of, restore_numeric


import copy
import math
import random
from operator import mul
from typing import NamedTuple, Optional

INPUTS = 20
TERRAIN_INPUTS = 32
MATING_INPUTS = 42
INTENT_INPUTS = 45


def observation_inputs(config):
    base = {1: INPUTS, 2: TERRAIN_INPUTS, 3: MATING_INPUTS, 4: INTENT_INPUTS}[
        config.get("observation_version", 1)
    ]
    return base + 12 * (config.get("observation_top_k", 1) - 1)


class Experience(NamedTuple):
    observation: tuple
    action: str
    reward: float
    next_observation: Optional[tuple]
    terminated: bool
    truncated: bool = False


class Brain:
    def __init__(self, config, actions, base_seed, individual_seed):
        self.config = copy.deepcopy(config)
        self.actions = list(actions)
        self.action_indices = {action: i for i, action in enumerate(self.actions)}
        self.hidden = config["hidden_layers"][0]
        self.inputs = observation_inputs(config)
        self.rng = random.Random(individual_seed)
        self.training_rng = random.Random(individual_seed ^ 0x5F3759DF)
        init = random.Random(base_seed)
        self.weights = {
            "w1": [
                init.uniform(-1, 1) * math.sqrt(6 / (self.inputs + self.hidden))
                for _ in range(self.inputs * self.hidden)
            ],
            "b1": [0.0] * self.hidden,
            "w2": [
                init.uniform(-1, 1) * math.sqrt(6 / (self.hidden + len(actions)))
                for _ in range(self.hidden * len(actions))
            ],
            "b2": [0.0] * len(actions),
        }
        self.weights = packed(self.weights)
        self.target = {k: v.copy() for k, v in self.weights.items()}
        self.replay = []
        self.cursor = self.steps = self.updates = 0
        self.epsilon = config["exploration"]["initial"]

    def _hidden(self, observation, weights):
        w1 = weights["w1"]
        hidden = self.hidden
        if kernels.native is not None and isinstance(w1, Float32):
            return kernels.native.affine(
                observation, w1, weights["b1"], True, 0, hidden
            )
        return [
            math.tanh(
                bias
                + sum(
                    map(mul, observation, w1[j * self.inputs : (j + 1) * self.inputs])
                )
            )
            for j, bias in enumerate(weights["b1"])
        ]

    def forward(self, observation, weights=None):
        w = self.weights if weights is None else weights
        h = self._hidden(observation, w)
        w2, hidden = w["w2"], self.hidden
        if kernels.native is not None and isinstance(w2, Float32):
            return h, kernels.native.affine(h, w2, w["b2"], False, 0, len(self.actions))
        # Preserve summation order for policies, diagnostics and target values.
        q = [
            bias + sum(map(mul, h, w2[a * hidden : (a + 1) * hidden]))
            for a, bias in enumerate(w["b2"])
        ]
        return h, q

    def choose(self, observation, explore=True, random_baseline=False):
        if random_baseline or (explore and self.rng.random() < self.epsilon):
            return self.rng.choice(self.actions)
        q = self.forward(observation)[1]
        return self.actions[max(range(len(q)), key=q.__getitem__)]

    def learn(self, experience):
        c = self.config
        if len(self.replay) < c["replay_capacity"]:
            self.replay.append(experience)
        else:
            self.replay[self.cursor] = experience
        self.cursor = (self.cursor + 1) % c["replay_capacity"]
        self.steps += 1
        self.epsilon = max(
            c["exploration"]["minimum"],
            self.epsilon * c["exploration"]["decay_per_decision"],
        )
        if (
            len(self.replay) < max(c["learning_starts_after"], c["batch_size"])
            or self.steps % c["train_every_steps"]
        ):
            return
        gradients = {k: gradient_zeros(v) for k, v in self.weights.items()}
        gw1, gb1, gw2, gb2 = (gradients[k] for k in ("w1", "b1", "w2", "b2"))
        w2, hidden = self.weights["w2"], self.hidden
        for e in self.training_rng.sample(self.replay, c["batch_size"]):
            h = self._hidden(e.observation, self.weights)
            a = self.action_indices[e.action]
            # Only Q(s, action_taken) contributes to this replay loss. Other
            # output heads are still computed for policy and target inference.
            offset = a * hidden
            value = (
                kernels.native.affine(h, w2, self.weights["b2"], False, a, a + 1)[0]
                if kernels.native is not None and isinstance(w2, Float32)
                else self.weights["b2"][a]
                + sum(map(mul, h, w2[offset : offset + hidden]))
            )
            expected = e.reward
            if not e.terminated:
                expected += c["discount_factor"] * max(
                    self.forward(e.next_observation, self.target)[1]
                )
            delta = max(-1.0, min(1.0, value - expected))
            if kernels.native is not None and isinstance(w2, Float32):
                kernels.native.q_backward(
                    e.observation, h, w2, a, gw1, gb1, gw2, gb2, delta
                )
            else:
                gb2[a] += delta
                for j in range(self.hidden):
                    gw2[a * hidden + j] += delta * h[j]
                    dh = delta * w2[a * hidden + j] * (1 - h[j] ** 2)
                    gb1[j] += dh
                    offset = j * self.inputs
                    for i, x in enumerate(e.observation):
                        gw1[offset + i] += dh * x
        scale = c["learning_rate"] / c["batch_size"]
        for k, gradient in gradients.items():
            if kernels.native is not None and isinstance(self.weights[k], Float32):
                kernels.native.sgd(self.weights[k], gradient, scale)
                continue
            for i, value in enumerate(gradient):
                self.weights[k][i] -= scale * value
        self.updates += 1
        if self.updates % c["target_update_every_training_steps"] == 0:
            for key, values in self.weights.items():
                self.target[key][:] = values

    def diagnostics(self, observation, include_activations=False):
        hidden, outputs = self.forward(observation)
        info = dict(
            algorithm="q_learning",
            features=observation,
            action_values=dict(zip(self.actions, outputs)),
            epsilon=self.epsilon,
            updates=self.updates,
            replay_size=len(self.replay),
        )
        return activation_sample(info, [hidden]) if include_activations else info

    def checkpoint(self):
        return dict(
            precision=precision_of(self),
            config=copy.deepcopy(self.config),
            actions=self.actions[:],
            weights=plain(self.weights),
            target=plain(self.target),
            replay=[
                dict(
                    observation=tuple(e.observation),
                    action=e.action,
                    reward=e.reward,
                    next_observation=(
                        tuple(e.next_observation)
                        if e.next_observation is not None
                        else None
                    ),
                    terminated=e.terminated,
                    truncated=e.truncated,
                )
                for e in self.replay
            ],
            cursor=self.cursor,
            steps=self.steps,
            updates=self.updates,
            epsilon=self.epsilon,
            rng_state=self.rng.getstate(),
            training_rng_state=self.training_rng.getstate(),
        )

    @classmethod
    def restore(cls, state):
        b = cls(state["config"], state["actions"], 0, 0)
        for k in ("cursor", "steps", "updates", "epsilon"):
            setattr(b, k, copy.deepcopy(state[k]))
        restore_numeric(b, state)
        b.replay = [
            Experience(
                tuple(e["observation"]),
                e["action"],
                e["reward"],
                None if e["next_observation"] is None else tuple(e["next_observation"]),
                e["terminated"],
                e["truncated"],
            )
            for e in state["replay"]
        ]

        def tuples(value):
            return (
                tuple(tuples(x) for x in value)
                if isinstance(value, (list, tuple))
                else value
            )

        b.rng.setstate(tuples(state["rng_state"]))
        b.training_rng.setstate(tuples(state["training_rng_state"]))
        return b
