"""Independent categorical PPO with GAE, clipped objectives and Adam.

A small standard-library implementation for inspectable CPU experiments.
Each animal owns all network, optimizer, rollout, and random-generator state.
"""

import copy
from array import array
from itertools import chain
import math
import random
from operator import mul
from typing import NamedTuple
from contextvars import ContextVar
from wtf.learning.activation_diagnostics import activation_sample
from wtf.native import kernels
from wtf.native.kernels import gradient_zeros
from wtf.native.numeric import Float32, packed, plain, precision_of, restore_numeric
from wtf.learning.brain import INPUTS, observation_inputs
from wtf.learning.late_consequence import LateConsequence, policy_input_width
from wtf.learning.optimization import AdamOptimizer


# Transient, thread/task-local execution context; never retained by a brain/checkpoint.
PPO_ACCELERATOR = ContextVar("ppo_accelerator", default=None)
# Retain the stock optimizer identity even when callers replace a class method.
# This private mixin method is deliberately captured as the native eligibility contract.
# pylint: disable-next=protected-access
_STANDARD_ADAM_STEP = AdamOptimizer._adam_step


class RolloutStep(NamedTuple):
    """One PPO sample with value, log probability and ending flags."""

    observation: tuple
    action: int
    reward: float
    log_probability: float
    value: float
    next_value: float
    terminated: bool
    truncated: bool


def policy_distribution(logits):
    """Convert logits into probabilities and stable log probabilities."""

    # The optional CPython API is checked by its wrapper and native/reference
    # equivalence tests.
    # pylint: disable=c-extension-no-member

    if kernels.available("distribution"):
        return kernels.native.distribution(logits)
    maximum = max(logits)
    exponentials = [math.exp(z - maximum) for z in logits]
    normalizer = sum(exponentials)
    log_normalizer = math.log(normalizer)
    log_probabilities = [(z - maximum) - log_normalizer for z in logits]
    probabilities = [v / normalizer for v in exponentials]
    return probabilities, log_probabilities


def advantages_and_returns(rollout, discount, gae_lambda):
    """Compute PPO advantage and return targets with ending flags."""
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

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-positional-arguments
    # The optional CPython API is checked by its wrapper and native/reference
    # equivalence tests.
    # pylint: disable=c-extension-no-member

    if kernels.available("ppo_loss"):
        return kernels.native.ppo_loss(
            outputs,
            action,
            old_log_probability,
            advantage,
            target,
            config["clip_ratio"],
            config["entropy_coefficient"],
            config["value_coefficient"],
        )
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
    info = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approximate_kl": ratio - 1.0 - log_ratio,
        "clip_fraction": float(abs(ratio - 1.0) > clip),
    }
    return loss, gradient, info


def _prepare_native_samples(rollout, advantages, targets, masked_inputs=None):
    """Update-local immutable inputs; weights and shuffle order stay live."""
    if masked_inputs is None:
        return [
            (step.observation, step.action, step.log_probability, advantage, target)
            for step, advantage, target in zip(rollout, advantages, targets)
        ]
    return [
        (
            step.observation[:masked_inputs],
            step.action,
            step.log_probability,
            advantage,
            target,
            tuple(step.observation[masked_inputs:]),
        )
        for step, advantage, target in zip(rollout, advantages, targets)
    ]


class MLP:
    """Small dense network supporting inference and explicit backpropagation."""

    def __init__(self, hidden_layers, actions, seed, value_head=True, inputs=INPUTS):
        self.sizes = [inputs, *hidden_layers, actions + int(value_head)]
        rng = random.Random(seed)
        self.layers = []
        for index, (layer_inputs, outputs) in enumerate(
            zip(self.sizes, self.sizes[1:])
        ):
            scale = math.sqrt(6 / (layer_inputs + outputs))
            weights = [
                rng.uniform(-scale, scale) for _ in range(layer_inputs * outputs)
            ]
            if index == len(self.sizes) - 2:
                # Near-uniform initial policy; the last output is the critic.
                for i in range(layer_inputs * actions):
                    weights[i] *= 0.01
            self.layers.append(packed({"weights": weights, "bias": [0.0] * outputs}))

    def forward(self, observation):
        """Evaluate the network and retain intermediates needed by backpropagation."""

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        activations = [tuple(observation)]
        for index, layer in enumerate(self.layers):
            previous = activations[-1]
            inputs = self.sizes[index]
            weights = layer["weights"]
            if kernels.native is not None and isinstance(weights, Float32):
                values = kernels.native.affine(
                    previous,
                    weights,
                    layer["bias"],
                    index < len(self.layers) - 1,
                    0,
                    len(layer["bias"]),
                )
                activations.append(values)
                continue
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

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        previous = tuple(observation)
        if kernels.available("mlp_predict") and isinstance(
            self.layers[0]["weights"], Float32
        ):
            start, stop = (
                (self.sizes[-1] - 1, self.sizes[-1])
                if value_only
                else output_range if output_range is not None else (0, self.sizes[-1])
            )
            result = kernels.native.mlp_predict(self.layers, previous, start, stop)
            return result[0] if value_only else result
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            inputs, weights = self.sizes[index], layer["weights"]
            if kernels.native is not None and isinstance(weights, Float32):
                start, stop = (
                    (len(layer["bias"]) - 1, len(layer["bias"]))
                    if value_only and index == last
                    else (
                        output_range
                        if output_range is not None and index == last
                        else (0, len(layer["bias"]))
                    )
                )
                previous = kernels.native.affine(
                    previous, weights, layer["bias"], index < last, start, stop
                )
                if value_only and index == last:
                    return previous[0]
                continue
            if value_only and index == last:
                output = len(layer["bias"]) - 1
                return layer["bias"][output] + sum(
                    map(mul, previous, weights[output * inputs : (output + 1) * inputs])
                )
            if output_range is not None and index == last:
                start, stop = output_range
                return [
                    layer["bias"][j]
                    + sum(map(mul, previous, weights[j * inputs : (j + 1) * inputs]))
                    for j in range(start, stop)
                ]
            values = [
                bias + sum(map(mul, previous, weights[j * inputs : (j + 1) * inputs]))
                for j, bias in enumerate(layer["bias"])
            ]
            previous = [math.tanh(v) for v in values] if index < last else values
        return previous

    def predict(self, observation):
        """Compute outputs without performing a learning update."""
        return self._predict(observation)

    def value(self, observation):
        """Final output only, used for actor-critic value bootstrapping."""
        return self._predict(observation, value_only=True)

    def predict_range(self, observation, start, stop):
        """Evaluate only the requested final-layer heads, preserving their arithmetic."""
        if not 0 <= start <= stop <= self.sizes[-1]:
            raise ValueError("Invalid output range")
        return self._predict(observation, output_range=(start, stop))

    @staticmethod
    def clear_gradients(gradients):
        # Local to one update call, never a persistent per-animal buffer.
        """Reset the supplied gradient storage for reuse."""
        for layer in gradients:
            for values in layer.values():
                values[:] = gradient_zeros(values)

    def zeros(self):
        """Allocate zero gradients matching the ordered parameter layout."""
        return [
            {key: gradient_zeros(values) for key, values in layer.items()}
            for layer in self.layers
        ]

    def accumulate_gradient(
        self, activations, output_gradient, gradients, return_input_gradient=False
    ):
        """Accumulate derivatives into the supplied gradient storage."""

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-locals
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        delta = output_gradient
        for index in range(len(self.layers) - 1, -1, -1):
            previous = activations[index]
            inputs = self.sizes[index]
            weights = self.layers[index]["weights"]
            dw, db = gradients[index]["weights"], gradients[index]["bias"]
            if (
                kernels.native is not None
                and isinstance(weights, Float32)
                and not isinstance(dw, list)
            ):
                previous_gradient = kernels.native.backward(
                    previous,
                    delta,
                    weights,
                    dw,
                    db,
                    bool(index or return_input_gradient),
                )
                if not index:
                    return previous_gradient
                delta = [
                    g * (1.0 - value * value)
                    for g, value in zip(previous_gradient, previous)
                ]
                continue
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
        return None

    def checkpoint(self):
        """Serialize state required to resume this component."""
        return [
            {key: list(values) for key, values in layer.items()}
            for layer in self.layers
        ]


# Auxiliary fusion must not bypass later class-level numerical customizations.
_AUXILIARY_MLP_HOOKS = {
    name: getattr(MLP, name) for name in ("forward", "accumulate_gradient")
}


class PhysicalMLP(MLP):
    """Evidence travels with rollout samples but enters only the late adapter."""

    def forward(self, observation):
        return super().forward(observation[: self.sizes[0]])

    def _predict(self, observation, value_only=False, output_range=None):
        return super()._predict(observation[: self.sizes[0]], value_only, output_range)


class PPOBrain(AdamOptimizer):
    """Independent clipped policy-gradient learner with private rollout state."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-many-instance-attributes

    algorithm = "ppo"

    def __init__(self, config, actions, base_seed, individual_seed):
        self.config = copy.deepcopy(config)
        self.actions = list(actions)
        self.action_indices = {name: i for i, name in enumerate(actions)}
        self.network = self._new_network(config, len(actions), base_seed)
        self.rng = random.Random(individual_seed)
        self.training_rng = random.Random(individual_seed ^ 0x5F3759DF)
        self.moments = packed(self.network.zeros())
        self.variances = packed(self.network.zeros())
        self.rollout = []
        self.pending = None
        self.steps = self.updates = self.rollout_updates = 0
        self.last_metrics = {}
        if "late_consequence" in config:
            self.late = LateConsequence(config, actions)
        if "auxiliary_consequence" in config:
            # Optional only: constructing the head never consumes policy RNGs.
            # pylint: disable=import-outside-toplevel,cyclic-import
            from wtf.learning.auxiliary_consequence import AuxiliaryConsequence

            self.auxiliary = AuxiliaryConsequence(
                config, self.network.sizes[-2], base_seed
            )

    def _new_network(self, config, actions, seed):
        cls = PhysicalMLP if "late_consequence" in config else MLP
        return cls(
            config["network"]["hidden_layers"],
            actions,
            seed,
            inputs=policy_input_width(config),
        )

    def distribution(self, outputs, observation):
        """Convert policy outputs into the available action distribution."""

        # The callback keeps its caller's keyword signature; this branch does not
        # consume every input.
        # pylint: disable=unused-argument

        if hasattr(self, "late"):
            outputs = self.late.adjusted(outputs, observation)
        return policy_distribution(outputs[:-1])

    def objective(self, outputs, step, advantage, target):
        """Compute the policy objective and gradients for a stored sample."""
        if hasattr(self, "late"):
            outputs = self.late.adjusted(outputs, step.observation)
        return loss_and_gradient(
            outputs, step.action, step.log_probability, advantage, target, self.config
        )

    def action_probabilities(self, observation):
        """Return the categorical action probabilities for an observation."""
        return self.distribution(self.network.predict(observation), observation)[0]

    def choose(self, observation, explore=True, random_baseline=False):
        """Choose an action using this animal's private policy and exploration state."""
        if random_baseline:
            return self.rng.choice(self.actions)
        outputs = self.network.predict(observation)
        return self._choose_outputs(observation, outputs, explore)

    def _choose_outputs(self, observation, outputs, explore):

        # Keep independent numeric/compatibility or benchmark paths explicit to
        # preserve their exact contracts.
        # pylint: disable=duplicate-code

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
        """Consume a transition and apply the configured learning schedule."""
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

        # Keep the KL gate and both accepted optimizer commits in one transaction.
        # pylint: disable=too-many-statements
        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel
        # Exact types intentionally reject bool/subclass substitutions in persisted
        # numeric contracts.
        # pylint: disable=unidiomatic-typecheck
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        # This import is deferred until the operation runs, after both modules have
        # initialized.
        # pylint: disable=cyclic-import

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
        accelerator = PPO_ACCELERATOR.get()
        if self.rollout and accelerator is not None and accelerator.accepts_ppo(self):
            accelerator.submit_ppo(self, advantages, targets)
            return
        totals, samples, stopped = {}, 0, False
        gradients = None
        indices = list(range(len(self.rollout)))
        from wtf.learning.contextual_ppo import (
            LocalMLP,
            MaskablePPOBrain,
            _NATIVE_MASKED_OBJECTIVE,
        )

        prepared_samples = None
        prepared_auxiliary = None
        masked_batch = (
            type(self) is MaskablePPOBrain
            and not hasattr(self, "late")
            and type(self.network) is LocalMLP
            and "forward" not in vars(self.network)
            and getattr(self.objective, "__func__", None) is MaskablePPOBrain.objective
            and kernels.available("masked_ppo_batch")
        )
        late_batch = (
            kernels.available("late_ppo_batch")
            and type(self) in (PPOBrain, MaskablePPOBrain)
            and type(self.network) in (PhysicalMLP, LocalMLP)
            and getattr(self.objective, "__func__", None)
            is (
                _NATIVE_MASKED_OBJECTIVE
                if type(self) is MaskablePPOBrain
                else _NATIVE_PPO_OBJECTIVE
            )
            and getattr(self.objective, "__module__", "")
            in ("wtf.learning.ppo", "wtf.learning.contextual_ppo")
            and "_adam_step" not in vars(self)
            and getattr(self._adam_step, "__func__", None) is _STANDARD_ADAM_STEP
            and hasattr(self, "late")
            and self.late.native_compatible()
            and all(
                name not in vars(self.network)
                for name in (
                    "forward",
                    "accumulate_gradient",
                    "zeros",
                    "clear_gradients",
                )
            )
            and all(
                getattr(getattr(type(self.network), name), "__module__", "")
                in ("wtf.learning.ppo", "wtf.learning.contextual_ppo")
                for name in (
                    "forward",
                    "accumulate_gradient",
                    "zeros",
                    "clear_gradients",
                )
            )
            and isinstance(self.network.layers[0]["weights"], Float32)
            and isinstance(self.late.network.layers[0]["weights"], Float32)
        )
        if late_batch:
            width = self.network.sizes[0]
            masks = type(self) is MaskablePPOBrain
            inputs = observation_inputs(self.config)
            prepared_late = [
                (
                    step.observation[:width],
                    step.action,
                    step.log_probability,
                    advantage,
                    target,
                    (
                        tuple(step.observation[inputs:])
                        if masks
                        else (True,) * len(self.actions)
                    ),
                    array(
                        "d",
                        chain.from_iterable(self.late.feature_rows(step.observation)),
                    ),
                )
                for step, advantage, target in zip(self.rollout, advantages, targets)
            ]
        for _ in range(c["epochs"]):
            self.training_rng.shuffle(indices)
            for offset in range(0, len(indices), c["batch_size"]):
                batch = indices[offset : offset + c["batch_size"]]
                if gradients is None:
                    gradients = self.network.zeros()
                else:
                    self.network.clear_gradients(gradients)
                batch_metrics = {}
                late_gradients = (
                    self.late.network.zeros() if hasattr(self, "late") else None
                )
                if late_batch:
                    batch_metrics = kernels.native.late_ppo_batch(
                        self.network.layers,
                        gradients,
                        self.late.network.layers[0]["weights"],
                        late_gradients[0]["weights"],
                        [prepared_late[i] for i in batch],
                        c["clip_ratio"],
                        c["entropy_coefficient"],
                        c["value_coefficient"],
                        self.late.bound,
                    )
                elif (
                    masked_batch
                    or (
                        kernels.available("ppo_batch")
                        and type(self.network) is MLP
                        and getattr(self.objective, "__func__", None)
                        is PPOBrain.objective
                    )
                ) and isinstance(self.network.layers[0]["weights"], Float32):
                    if prepared_samples is None:
                        prepared_samples = _prepare_native_samples(
                            self.rollout,
                            advantages,
                            targets,
                            self.network.sizes[0] if masked_batch else None,
                        )
                    samples_for_batch = [prepared_samples[i] for i in batch]
                    batch_kernel = (
                        kernels.native.masked_ppo_batch
                        if masked_batch
                        else kernels.native.ppo_batch
                    )
                    batch_metrics = batch_kernel(
                        self.network.layers,
                        gradients,
                        samples_for_batch,
                        c["clip_ratio"],
                        c["entropy_coefficient"],
                        c["value_coefficient"],
                    )
                else:
                    for i in batch:
                        step = self.rollout[i]
                        outputs, activations = self.network.forward(step.observation)
                        _, derivative, info = self.objective(
                            outputs, step, advantages[i], targets[i]
                        )
                        self.network.accumulate_gradient(
                            activations, derivative, gradients
                        )
                        if late_gradients is not None:
                            self.late.accumulate(
                                step.observation, derivative, late_gradients
                            )
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
                auxiliary = getattr(self, "auxiliary", None)
                auxiliary_proposal = None
                if auxiliary is not None:
                    if (
                        auxiliary.preparation_compatible()
                        and getattr(self._adam_step, "__func__", None)
                        is _STANDARD_ADAM_STEP
                    ):
                        if prepared_auxiliary is None:
                            prepared_auxiliary = auxiliary.prepare(
                                [step.observation for step in self.rollout],
                                self.network.sizes[0],
                            )
                        auxiliary_metrics, auxiliary_proposal = (
                            auxiliary.accumulate_prepared(
                                self.network,
                                [prepared_auxiliary[i] for i in batch],
                                gradients,
                            )
                        )
                    else:
                        auxiliary_metrics, auxiliary_proposal = auxiliary.accumulate(
                            self.network,
                            [self.rollout[i].observation for i in batch],
                            gradients,
                        )
                    batch_metrics.update(auxiliary_metrics)
                self._adam_step(gradients, len(batch))
                if late_gradients is not None:
                    self.late.apply(late_gradients, len(batch))
                if (
                    auxiliary_proposal is not None
                    and auxiliary_metrics["auxiliary_coverage"]
                ):
                    auxiliary.apply(auxiliary_proposal, len(batch))
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

    def diagnostics(self, observation, include_activations=False):
        """Return predictions and optional activation samples for inspection."""
        if include_activations:
            forward = getattr(self.network, "activation_forward", self.network.forward)
            outputs, layers = forward(observation)
        else:
            outputs = self.network.predict(observation)
        probabilities, logs, late_info = self._diagnostic_distribution(
            outputs, observation
        )
        info = {
            "algorithm": self.algorithm,
            "buffer_kind": "rollout",
            "buffer_size": len(self.rollout),
            "features": observation,
            "action_probabilities": dict(zip(self.actions, probabilities)),
            "state_value": outputs[-1],
            "entropy": -sum((p * logp for p, logp in zip(probabilities, logs))),
            "updates": self.updates,
            "rollout_updates": self.rollout_updates,
            "rollout_size": len(self.rollout),
            "training_metrics": self.last_metrics.copy(),
        }
        if late_info is not None:
            info["late_consequence"] = late_info
        return activation_sample(info, layers[1:-1]) if include_activations else info

    def _diagnostic_distribution(self, outputs, observation):
        """Reuse one stock residual; custom distribution/adapter hooks stay authoritative."""
        # Contextual PPO imports this module, so defer its stock hook identities.
        # Exact stock classes prevent bypassing custom distribution semantics.
        # pylint: disable=import-outside-toplevel,cyclic-import,unidiomatic-typecheck
        from wtf.learning.contextual_ppo import (
            MaskablePPOBrain,
            _NATIVE_MASKED_DISTRIBUTION,
        )

        if not hasattr(self, "late"):
            return (*self.distribution(outputs, observation), None)
        distribution = getattr(self.distribution, "__func__", None)
        stock = (
            type(self) is PPOBrain and distribution is _NATIVE_PPO_DISTRIBUTION
        ) or (
            type(self) is MaskablePPOBrain
            and distribution is _NATIVE_MASKED_DISTRIBUTION
        )
        if stock and self.late.native_compatible():
            adjustments = self.late.residuals(observation)
            final = [a + b for a, b in zip(outputs[:-1], adjustments)]
            if type(self) is MaskablePPOBrain:
                mask = observation[observation_inputs(self.config) :]
                if len(mask) != len(self.actions) or not any(mask):
                    raise ValueError("Missing or empty action mask")
                logits = [v if valid else -1e30 for v, valid in zip(final, mask)]
            else:
                logits = final
            probabilities, logs = policy_distribution(logits)
        else:
            probabilities, logs = self.distribution(outputs, observation)
            adjustments = self.late.residuals(observation)
            final = self.late.adjusted(outputs, observation)[:-1]
        return (
            probabilities,
            logs,
            {
                "base_logits": list(outputs[:-1]),
                "adjustments": adjustments,
                "final_logits": final,
            },
        )

    def checkpoint(self):
        """Serialize state required to resume this component."""

        # Keep independent numeric/compatibility or benchmark paths explicit to
        # preserve their exact contracts.
        # pylint: disable=duplicate-code

        state = {
            "algorithm": self.algorithm,
            "precision": precision_of(self),
            "config": copy.deepcopy(self.config),
            "actions": self.actions[:],
            "network": self.network.checkpoint(),
            "moments": plain(self.moments),
            "variances": plain(self.variances),
            "rollout": [step._asdict() for step in self.rollout],
            "pending": self.pending,
            "steps": self.steps,
            "updates": self.updates,
            "rollout_updates": self.rollout_updates,
            "last_metrics": self.last_metrics.copy(),
            "rng_state": self.rng.getstate(),
            "training_rng_state": self.training_rng.getstate(),
        }
        if hasattr(self, "auxiliary"):
            state["auxiliary"] = self.auxiliary.checkpoint()
        if hasattr(self, "late"):
            state["late"] = self.late.checkpoint()
        return state

    @classmethod
    def restore(cls, state):
        """Reconstruct the component from its persisted continuation state."""

        # Keep independent numeric/compatibility or benchmark paths explicit to
        # preserve their exact contracts.
        # pylint: disable=duplicate-code

        b = cls(state["config"], state["actions"], 0, 0)
        for key in (
            "steps",
            "updates",
            "rollout_updates",
            "last_metrics",
        ):
            setattr(b, key, copy.deepcopy(state[key]))
        restore_numeric(b, state)
        if hasattr(b, "late"):
            b.late.restore_state(state["late"], precision_of(b))
        if hasattr(b, "auxiliary"):
            b.auxiliary.restore_state(state["auxiliary"])
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


# Preserve the original late-batch loss contract across class-level wrappers.
_NATIVE_PPO_OBJECTIVE = PPOBrain.objective

_NATIVE_PPO_DISTRIBUTION = PPOBrain.distribution
