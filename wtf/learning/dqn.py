"""Multilayer scalar DQN with private replay, target network and Huber TD loss."""

import copy
from wtf.learning.qrdqn import QRDQNBrain
from wtf.learning.ppo import MLP
from wtf.native import kernels
from wtf.native.numeric import Float32
from wtf.learning.activation_diagnostics import activation_sample


def huber_td(value, target, threshold):
    """Return scalar Huber TD loss and its derivative with respect to value."""
    error = value - target
    absolute = abs(error)
    loss = (
        0.5 * error * error
        if absolute <= threshold
        else threshold * (absolute - 0.5 * threshold)
    )
    return loss, max(-threshold, min(threshold, error))


class DQNBrain(QRDQNBrain):
    """Reuse bounded value-controller state, with scalar max-target DQN updates."""

    algorithm = "dqn"

    def __init__(self, config, actions, base_seed, individual_seed):
        # The shared value-controller constructor needs an output multiplicity;
        # scalar DQN exposes no quantile configuration or distributional objective.
        storage_config = dict(config, num_quantiles=1)
        super().__init__(storage_config, actions, base_seed, individual_seed)
        self.config = copy.deepcopy(config)

    def learn(self, experience):
        """Fit scalar Bellman targets from independent bounded replay samples."""
        # Keep sample/target/update order together, independent of QR-DQN's loss.
        # pylint: disable=too-many-locals,duplicate-code
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
            len(self.replay) < max(c["batch_size"], c["learning_starts_after"])
            or self.steps % c["train_every_steps"]
        ):
            return
        gradients = self.network.zeros()
        samples = self.training_rng.sample(self.replay, c["batch_size"])
        total_loss = self._batch_loss(samples, gradients)
        self._adam_step(gradients, c["batch_size"])
        if self.updates % c["target_update_every_training_steps"] == 0:
            for target_layer, online in zip(
                self.target_network.layers, self.network.layers
            ):
                for key, values in online.items():
                    target_layer[key][:] = values
        self.last_metrics = {"td_loss": total_loss / c["batch_size"]}

    def _batch_loss(self, samples, gradients):
        """Update-local batching; private sample order and gradient sums are exact."""
        # Keep scalar target/loss/gradient order explicit as the reference backend.
        # pylint: disable=too-many-locals,c-extension-no-member
        c = self.config
        if self._native_batchable():
            return kernels.native.dqn_batch(
                self.network.layers,
                self.target_network.layers,
                gradients,
                [
                    (
                        item.observation,
                        self.action_indices[item.action],
                        item.reward,
                        item.next_observation,
                        item.terminated,
                    )
                    for item in samples
                ],
                c["discount_factor"],
                c["huber_threshold"],
                self.algorithm == "ddqn",
            )
        total_loss = 0.0
        for item in samples:
            outputs, tape = self.network.forward(item.observation)
            target = item.reward
            if not item.terminated:
                if item.next_observation is None:
                    raise ValueError("DQN needs a next observation unless terminated")
                target += c["discount_factor"] * self.bootstrap(item.next_observation)
            action = self.action_indices[item.action]
            loss, derivative = huber_td(outputs[action], target, c["huber_threshold"])
            total_loss += loss
            gradient = [0.0] * len(outputs)
            gradient[action] = derivative
            self.network.accumulate_gradient(tape, gradient, gradients)
        return total_loss

    def _native_batchable(self):
        """Preserve custom numeric/controller hooks and older native libraries."""
        # Exact stock types protect caller overrides and legacy FP64 arithmetic.
        # pylint: disable=unidiomatic-typecheck
        return (
            type(self) in (DQNBrain, DDQNBrain)
            and kernels.available("dqn_batch")
            and not {"bootstrap", "values", "_mean_values"}.intersection(vars(self))
            and all(
                type(network) is MLP
                and not {
                    "forward",
                    "predict",
                    "predict_range",
                    "accumulate_gradient",
                }.intersection(vars(network))
                and isinstance(network.layers[0]["weights"], Float32)
                for network in (self.network, self.target_network)
            )
        )

    def bootstrap(self, observation):
        """DQN maximizes the target network's next-state values."""
        return max(self.values(observation, self.target_network))

    def _mean_values(self, values):
        """Scalar outputs already contain one Q-value per action."""
        return values

    def diagnostics(self, observation, include_activations=False):
        """Expose scalar action values and actual dense activations in playback."""
        # Scalar telemetry intentionally matches the legacy value-viewer contract.
        # pylint: disable=duplicate-code
        if include_activations:
            outputs, layers = self.network.forward(observation)
        else:
            outputs = self.network.predict(observation)
        info = {
            "algorithm": self.algorithm,
            "features": observation,
            "action_values": dict(zip(self.actions, outputs)),
            "epsilon": self.epsilon,
            "updates": self.updates,
            "replay_size": len(self.replay),
            "training_metrics": self.last_metrics.copy(),
        }
        return activation_sample(info, layers[1:-1]) if include_activations else info


class DDQNBrain(DQNBrain):
    """Scalar Double DQN: online action selection and separate target evaluation."""

    algorithm = "ddqn"

    def bootstrap(self, observation):
        """Evaluate the online-greedy next action with the target network."""
        values = self.values(observation)
        best = max(range(len(values)), key=values.__getitem__)
        return self.target_network.predict_range(observation, best, best + 1)[0]
