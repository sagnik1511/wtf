"""Private outcome head; supervised evidence never becomes an actor trajectory."""

import math
from wtf.learning.optimization import AdamOptimizer
from wtf.native import kernels
from wtf.native.numeric import Float32, packed, plain, precision_of
from wtf.simulation.actions import ACTIONS
from wtf.simulation.compact_observations import compact_layout, consequence_rows
from wtf.simulation.soft_observations import decode_layout, soft_layout


# Keep private optimizer, head, scope and versioned target layout together.
# pylint: disable-next=too-many-instance-attributes
class AuxiliaryConsequence(AdamOptimizer):
    """Predict physical outcomes without copying the target-bearing input block."""

    def __init__(self, config, hidden, seed):
        # Deferred import avoids the PPO/controller construction cycle.
        # pylint: disable=import-outside-toplevel,cyclic-import
        from wtf.learning.ppo import MLP

        self.config = {key: config[key] for key in ("learning_rate", "max_grad_norm")}
        self.network = MLP(
            [], len(ACTIONS) * 3, seed ^ 0x434F4E534551, value_head=False, inputs=hidden
        )
        self.moments = packed(self.network.zeros())
        self.variances = packed(self.network.zeros())
        self.updates = 0
        self.coefficient = config["auxiliary_consequence"]["coefficient"]
        self.scope = config.get("consequence_scope", "context")
        if config.get("observation_version") == 7:
            from copy import deepcopy

            self.compact_template = deepcopy(config)
            self.compact_plan = compact_layout(self.compact_template)
        self.semantic_layout = (
            soft_layout(config) if config.get("observation_version", 1) == 6 else None
        )

    # Keep scope selection and target exclusion in one version-aware transaction.
    # pylint: disable-next=too-many-locals
    def targets(self, observation, inputs):
        """Use published context estimates when known; otherwise general evidence."""
        if hasattr(self, "compact_template"):
            layout = self.compact_plan
            blocks = consequence_rows(observation, self.compact_template, layout=layout)
            general, context = (
                (blocks[0], blocks[1])
                if self.scope == "both"
                else (
                    (blocks[0], ())
                    if self.scope == "species_action"
                    else ((), blocks[0])
                )
            )
            targets = [
                (
                    context[i]
                    if context and context[i][0] > 0
                    else general[i] if general else (0.0, 0.0, 0.0, 0.0)
                )
                for i in range(len(ACTIONS))
            ]
            end = min(inputs, layout.consequence_start)
            physical = tuple(observation[:end]) + (0.0,) * (inputs - end)
            return physical, targets
        width = 96 if self.scope == "both" else 48
        start = inputs - width
        if self.semantic_layout is not None:
            layout = self.semantic_layout
            raw_start = len(layout.raw_names) - width
            start = layout.offset(raw_start)
            block = decode_layout(observation, layout, raw_start)
            width = inputs - start
        else:
            block = observation[start:inputs]
        general, context = (
            (block[:48], block[48:])
            if self.scope == "both"
            else ((block, ()) if self.scope == "species_action" else ((), block))
        )
        targets = []
        for action in range(len(ACTIONS)):
            offset = action * 4
            row = (
                context[offset : offset + 4]
                if context and context[offset] > 0
                else general[offset : offset + 4]
            )
            targets.append(tuple(row) if row else (0.0, 0.0, 0.0, 0.0))
        # Do not let either estimate block leak into its own prediction target.
        physical = tuple(observation[:start]) + (0.0,) * width
        return physical, targets

    def _has_evidence(self, observation, inputs):
        """Conservatively skip zero-confidence samples before decoding/allocating."""
        if hasattr(self, "compact_template"):
            layout = self.compact_plan
            return any(
                observation[i] for i in range(layout.consequence_start, layout.width, 4)
            )
        if self.semantic_layout is not None:
            # Confidence has fixed zero-first knots; only the other hats carry mass.
            return any(
                observation[i + 1] or observation[i + 2]
                for i in self.semantic_layout.confidence_offsets
            )
        width = 96 if self.scope == "both" else 48
        return any(observation[i] for i in range(inputs - width, inputs, 4))

    def preparation_compatible(self):
        """Only stock pure evidence decoders may be staged across update epochs."""
        # Exact identities preserve custom instance/class extension hooks.
        # pylint: disable-next=unidiomatic-typecheck
        return type(self) is AuxiliaryConsequence and all(
            getattr(getattr(self, name), "__func__", None) is method
            for name, method in _PREPARATION_HOOKS.items()
        )

    def prepare(self, observations, inputs):
        """Own immutable targets for one update; retain no live world reference."""
        standard = getattr(self.targets, "__func__", None) is _STANDARD_TARGETS
        result = []
        for observation in observations:
            if standard and not self._has_evidence(observation, inputs):
                result.append(None)
                continue
            physical, targets = self.targets(observation, inputs)
            result.append(
                (physical, tuple(targets)) if sum(row[0] for row in targets) else None
            )
        return result

    def accumulate(self, policy, observations, gradients):
        """Prepare a standalone batch while preserving the public extension hook."""
        return self.accumulate_prepared(
            policy, self.prepare(observations, policy.sizes[0]), gradients
        )

    def accumulate_prepared(self, policy, prepared, gradients):
        """Recompute predictions at current weights; targets stay update-local."""
        # The update-local view aliases only this animal's private trunk gradients.
        # Native entry points are optional and availability-gated below.
        # pylint: disable=import-outside-toplevel,too-many-locals,c-extension-no-member
        from wtf.learning.ppo import MLP, _AUXILIARY_MLP_HOOKS

        view = MLP.__new__(MLP)
        view.layers = policy.layers[:-1] + self.network.layers
        view.sizes = policy.sizes[:-1] + [len(ACTIONS) * 3]
        auxiliary_gradients = gradients[:-1] + self.network.zeros()
        loss = 0.0
        samples = 0
        active = [sample for sample in prepared if sample is not None]
        stock_mlp = all(
            getattr(MLP, name) is method
            for name, method in _AUXILIARY_MLP_HOOKS.items()
        )
        native_buffers = all(
            isinstance(layer["weights"], Float32) for layer in view.layers
        ) and all(
            not isinstance(layer["weights"], list) for layer in auxiliary_gradients
        )
        if (
            active
            and self.preparation_compatible()
            and kernels.available("auxiliary_batch")
            and stock_mlp
            and native_buffers
        ):
            metrics = kernels.native.auxiliary_batch(
                view.layers, auxiliary_gradients, active, self.coefficient
            )
            return metrics, [auxiliary_gradients[-1]]
        for physical, targets in active:
            samples += 1
            outputs, tape = view.forward(physical)
            derivative = []
            for index, (weight, *rates) in enumerate(targets):
                for outcome, target in enumerate(rates):
                    logit = outputs[index * 3 + outcome]
                    probability = 1 / (1 + math.exp(-max(-60.0, min(60.0, logit))))
                    scale = self.coefficient * weight / (3 * len(ACTIONS))
                    # Stable binary cross entropy for soft empirical targets.
                    loss += scale * (
                        max(logit, 0)
                        - logit * target
                        + math.log1p(math.exp(-abs(logit)))
                    )
                    derivative.append(scale * (probability - target))
            view.accumulate_gradient(tape, derivative, auxiliary_gradients)
        return {"auxiliary_loss": loss, "auxiliary_coverage": samples}, [
            auxiliary_gradients[-1]
        ]

    def apply(self, gradients, count):
        """Apply the head proposal only after the owning PPO minibatch accepts it."""
        self._adam_step(gradients, count)

    def load_weights(self, layers, precision):
        """Validate private-head dimensions before adopting portable weights."""
        if (
            not isinstance(layers, list)
            or len(layers) != 1
            or not isinstance(layers[0], dict)
            or set(layers[0]) != {"weights", "bias"}
        ):
            raise ValueError("Invalid auxiliary network")
        for key, expected in self.network.layers[0].items():
            values = layers[0][key]
            if (
                not isinstance(values, list)
                or len(values) != len(expected)
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in values
                )
            ):
                raise ValueError("Invalid auxiliary parameter array")
        self.network.layers = packed(layers, precision)

    def checkpoint(self):
        """Compact private head and optimizer state for exact continuation."""
        return {
            "network": self.network.checkpoint(),
            "moments": plain(self.moments),
            "variances": plain(self.variances),
            "updates": self.updates,
            "precision": precision_of(self),
        }

    def restore_state(self, state):
        """Restore private arrays; no live table or other animal is referenced."""
        # Exact integers reject bools in persisted optimizer counters.
        # pylint: disable=unidiomatic-typecheck
        precision = state.get("precision", "float64")
        if type(state.get("updates")) is not int or state["updates"] < 0:
            raise ValueError("Invalid auxiliary optimizer counter")
        self.load_weights(state["network"], precision)
        for key in ("moments", "variances"):
            # Optimizer arrays have the same fixed dimensions as the head.
            values = state[key]
            reference = self.network.layers
            self.load_weights(values, precision)
            setattr(self, key, self.network.layers)
            self.network.layers = reference
        if any(value < 0 for values in self.variances[0].values() for value in values):
            raise ValueError("Invalid auxiliary optimizer variance")
        self.updates = state["updates"]


# Class/instance extension hooks must not be mistaken for the stock decoder.
_STANDARD_TARGETS = AuxiliaryConsequence.targets

# Capture hook identities before callers install class-level customizations.
_PREPARATION_HOOKS = {
    name: getattr(AuxiliaryConsequence, name)
    for name in (
        "targets",
        "_has_evidence",
        "prepare",
        "accumulate",
        "accumulate_prepared",
        "_adam_step",
    )
}
