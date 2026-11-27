"""Private bounded PPO logit residuals over saved published physical evidence."""

import math
from wtf.learning.optimization import AdamOptimizer
from wtf.native.numeric import packed, plain
from wtf.simulation.compact_observations import compact_layout
from wtf.simulation.social import GROUP
from wtf.learning.brain import observation_inputs


class LateConsequence(AdamOptimizer):
    """Learn evidence utility, conditioned on needs; no fixed action preference."""

    # Private parameters, optimizer and their immutable evidence contract share a lifecycle.
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config, actions):
        # Defer construction until PPO has defined MLP; no import-time callback occurs.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from wtf.learning.ppo import MLP

        self.config = {key: config[key] for key in ("learning_rate", "max_grad_norm")}
        self.layout = compact_layout(config)
        self.actions = tuple(actions)
        self.scope = config.get("consequence_scope", "context")
        self.local = config.get("social_observation") is not None
        self.bound = config["late_consequence"].get("max_adjustment", 2.0)
        if (
            type(self.bound) not in (int, float)
            or not math.isfinite(self.bound)
            or not 0 < self.bound <= 10
        ):
            raise ValueError("Invalid late consequence adjustment bound")
        if self.actions != self.layout.actions:
            raise ValueError("Late consequence action and observation orders differ")
        self.sources = int(self.local) + (2 if self.scope == "both" else 1)
        self.features = 12 * self.sources
        self.network = MLP([], len(actions), 0, value_head=False, inputs=self.features)
        # Initially neutral; independent actor initialization/RNG is unchanged.
        for layer in self.network.layers:
            for values in layer.values():
                for i, _ in enumerate(values):
                    values[i] = 0.0
        self.moments = packed(self.network.zeros())
        self.variances = packed(self.network.zeros())
        self.updates = 0
        self.ready_index = self.layout.names.index("Self ready")

    def native_compatible(self):
        """Custom evidence/gradient hooks retain their reference implementation."""
        # Exact stock types prevent bypassing subclass evidence/gradient semantics.
        # pylint: disable-next=unidiomatic-typecheck
        return type(self) is LateConsequence and all(
            getattr(getattr(self, name), "__func__", None) is function
            for name, function in _NATIVE_HOOKS.items()
        )

    def feature_rows(self, observation):
        """Use the recorded evidence snapshot, never live knowledge during replay."""
        context = (
            1.0,
            max(0.0, 1 - observation[0]),
            max(0.0, 1 - observation[1]),
            observation[self.ready_index],
        )
        rows = []
        for action in self.actions:
            sources = []
            if self.local:
                offset = self.layout.physical_width + 4 * GROUP[action]
                sources.append(observation[offset : offset + 4])
            index = self.layout.actions.index(action)
            for block in range(2 if self.scope == "both" else 1):
                offset = self.layout.consequence_start + 4 * (
                    block * len(self.layout.actions) + index
                )
                sources.append(observation[offset : offset + 4])
            rows.append(
                tuple(
                    confidence * rate * need
                    for confidence, *rates in sources
                    for rate in rates
                    for need in context
                )
            )
        return rows

    def residuals(self, observation):
        """Confidence-zero evidence produces an exact zero residual, not a veto."""
        weights = self.network.layers[0]["weights"]
        return [
            self.bound
            * math.tanh(
                sum(
                    w * v
                    for w, v in zip(
                        weights[i * self.features : (i + 1) * self.features], row
                    )
                )
            )
            for i, row in enumerate(self.feature_rows(observation))
        ]

    def adjusted(self, outputs, observation):
        """Adjust actor logits only; critic output remains the private base value."""
        return [a + b for a, b in zip(outputs[:-1], self.residuals(observation))] + [
            outputs[-1]
        ]

    def accumulate(self, observation, derivative, gradients):
        """Chain the final PPO objective through the bounded residual."""
        weights = self.network.layers[0]["weights"]
        target = gradients[0]["weights"]
        for i, row in enumerate(self.feature_rows(observation)):
            offset = i * self.features
            value = math.tanh(
                sum(
                    w * v for w, v in zip(weights[offset : offset + self.features], row)
                )
            )
            scale = derivative[i] * self.bound * (1 - value * value)
            for j, feature in enumerate(row):
                target[offset + j] += scale * feature

    # Keep the chain-rule factors beside the matching evidence offsets.
    # pylint: disable-next=too-many-locals
    def input_gradient(self, observation, output_derivative):
        """Exact local derivative of the residual branch for archived inspection."""
        gradient = [0.0] * len(observation)
        weights = self.network.layers[0]["weights"]
        needs = (
            1.0,
            max(0.0, 1 - observation[0]),
            max(0.0, 1 - observation[1]),
            observation[self.ready_index],
        )
        for i, (action, row) in enumerate(
            zip(self.actions, self.feature_rows(observation))
        ):
            w = weights[i * self.features : (i + 1) * self.features]
            value = math.tanh(sum(a * b for a, b in zip(w, row)))
            scale = output_derivative[i] * self.bound * (1 - value * value)
            offsets = (
                [self.layout.physical_width + 4 * GROUP[action]] if self.local else []
            )
            for block in range(2 if self.scope == "both" else 1):
                offsets.append(
                    self.layout.consequence_start
                    + 4
                    * (
                        block * len(self.layout.actions)
                        + self.layout.actions.index(action)
                    )
                )
            for source, offset in enumerate(offsets):
                confidence = observation[offset]
                for outcome in range(3):
                    rate = observation[offset + 1 + outcome]
                    start = source * 12 + outcome * 4
                    utility = sum(w[start + j] * need for j, need in enumerate(needs))
                    gradient[offset] += scale * rate * utility
                    gradient[offset + 1 + outcome] += scale * confidence * utility
                    if observation[0] < 1:
                        gradient[0] -= scale * confidence * rate * w[start + 1]
                    if observation[1] < 1:
                        gradient[1] -= scale * confidence * rate * w[start + 2]
                    gradient[self.ready_index] += (
                        scale * confidence * rate * w[start + 3]
                    )
        return gradient

    def checkpoint(self):
        """Private learned adapter and Adam state; no copy of world knowledge."""
        # This stable state shape intentionally matches other private heads.
        # pylint: disable=duplicate-code
        return {
            "network": self.network.checkpoint(),
            "moments": plain(self.moments),
            "variances": plain(self.variances),
            "updates": self.updates,
        }

    def restore_state(self, state, precision):
        """Restore packed private adapter state after controller reconstruction."""
        self.network.layers = packed(state["network"], precision)
        self.moments = packed(state["moments"], precision)
        self.variances = packed(state["variances"], precision)
        self.updates = state["updates"]

    def apply(self, gradients, count):
        """Commit only after the actor's minibatch passes its KL gate."""
        self._adam_step(gradients, count)


def policy_input_width(config):
    """Separate recorded observations from the base network's physical inputs."""
    return (
        compact_layout(config).physical_width
        if "late_consequence" in config
        else observation_inputs(config)
    )


# Capture originals so both instance and class extension hooks bypass native fusion.
_NATIVE_HOOKS = {
    name: getattr(LateConsequence, name)
    for name in (
        "feature_rows",
        "residuals",
        "adjusted",
        "accumulate",
        "apply",
        "_adam_step",
    )
}
