"""Shared gradient clipping and per-agent Adam optimizer state."""

import math


class AdamOptimizer:
    def _adam_step(self, gradients, count):
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
