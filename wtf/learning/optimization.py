"""Shared gradient clipping and per-agent Adam optimizer state."""

import math
from typing import Any
from wtf.native import kernels
from wtf.native.numeric import Float32


class AdamOptimizer:
    # Concrete brain constructors own these independent per-animal values.
    """Share Adam update logic while concrete brains own optimizer state."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-few-public-methods

    moments: list
    variances: list
    network: Any
    config: dict
    # A concrete brain initializes its own counter; this also exposes the mixin
    # default to static analysis of the augmented assignment below.
    updates: int = 0

    def _adam_step(self, gradients, count):

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-locals
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        if kernels.native is not None:
            rows = [
                (
                    parameters,
                    self.moments[j][key],
                    self.variances[j][key],
                    gradients[j][key],
                )
                for j, layer in enumerate(self.network.layers)
                for key, parameters in layer.items()
            ]
            if all(
                isinstance(p, Float32) and not isinstance(g, list)
                for p, m, v, g in rows
            ):
                kernels.native.adam(
                    rows,
                    count,
                    self.config["max_grad_norm"],
                    self.config["learning_rate"],
                    self.updates + 1,
                )
                self.updates += 1
                return
        norm = math.sqrt(
            sum(
                (value / count) ** 2
                for layer in gradients
                for values in layer.values()
                for value in values
            )
        )
        factor = min(1.0, self.config["max_grad_norm"] / (norm + 1e-12)) / count
        self.updates += 1
        correction1, correction2 = 1.0 - 0.9**self.updates, 1.0 - 0.999**self.updates
        rate = self.config["learning_rate"]
        for layer_index, layer in enumerate(self.network.layers):
            for key, parameters in layer.items():
                first = self.moments[layer_index][key]
                second = self.variances[layer_index][key]
                for i, gradient in enumerate(gradients[layer_index][key]):
                    g = gradient * factor
                    first[i] = 0.9 * first[i] + 0.1 * g
                    second[i] = 0.999 * second[i] + 0.001 * g * g
                    parameters[i] -= (
                        rate
                        * (first[i] / correction1)
                        / (math.sqrt(second[i] / correction2) + 1e-8)
                    )
