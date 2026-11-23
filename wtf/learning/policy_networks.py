"""Independent categorical actor and critic networks with one storage contract."""

from wtf.learning.ppo import MLP
from wtf.native.kernels import gradient_zeros


class PolicyBundle:
    """Expose actor and critic parameters through one checkpoint interface."""

    def __init__(self, config, actions, seed, inputs, critics=1, critic_outputs=1):

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments

        self.inputs = inputs
        widths = config["network"]["hidden_layers"]
        self.actor = MLP(widths, actions, seed, value_head=False, inputs=inputs)
        self.critics = [
            MLP(
                widths,
                critic_outputs,
                seed ^ (0xA341316C + i),
                value_head=False,
                inputs=inputs,
            )
            for i in range(critics)
        ]
        self.block = len(self.actor.layers)

    @property
    def layers(self):
        """Expose the ordered trainable parameter layers."""
        return self.actor.layers + [
            layer for critic in self.critics for layer in critic.layers
        ]

    @layers.setter
    def layers(self, layers):
        """Expose the ordered trainable parameter layers."""
        if len(layers) != self.block * (1 + len(self.critics)):
            raise ValueError("Invalid policy bundle layer count")
        self.actor.layers = layers[: self.block]
        for i, critic in enumerate(self.critics):
            critic.layers = layers[(i + 1) * self.block : (i + 2) * self.block]

    def zeros(self):
        """Allocate zero gradients matching the ordered parameter layout."""
        return [
            {k: gradient_zeros(v) for k, v in layer.items()} for layer in self.layers
        ]

    clear_gradients = staticmethod(MLP.clear_gradients)

    def value(self, observation):
        """Evaluate the state-value head for the supplied observation."""
        return (
            self.critics[0].predict(observation)[0] if len(self.critics) == 1 else 0.0
        )

    def predict(self, observation):
        """Compute outputs without performing a learning update."""
        return self.actor.predict(observation) + [self.value(observation)]

    def forward(self, observation):
        """Evaluate the network and retain intermediates needed by backpropagation."""
        logits, tape = self.actor.forward(observation)
        return logits + [self.value(observation)], tape

    def activation_forward(self, observation):
        """Evaluate the network with intermediate activation samples."""
        return self.forward(observation)

    def checkpoint(self):
        """Serialize state required to resume this component."""
        return [{k: list(v) for k, v in layer.items()} for layer in self.layers]
