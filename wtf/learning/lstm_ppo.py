"""PPO with a forget-gated LSTM and chronological truncated BPTT.

Memory is [hidden, cell], each recurrent_size wide; both are checkpointed and
reset at episode boundaries and births. Archived explanations use zero memory.
"""

from array import array
from wtf.native import kernels
import math
import random
from wtf.learning.brain import observation_inputs
from wtf.learning.ppo import MLP, policy_distribution
from wtf.learning.recurrent_ppo import GRUNetwork, RecurrentPPOBrain, sigmoid
from wtf.learning.activation_diagnostics import activation_sample
from wtf.native.numeric import Float32, packed


class LSTMNetwork(GRUNetwork):
    def __init__(self, config, actions, seed):
        self.hidden_size = h = config["recurrent_size"]
        self.inputs = observation_inputs(config)
        rng = random.Random(seed ^ 0x6A09E667)
        self.cell = packed(
            dict(
                input_weights=[
                    rng.uniform(-1, 1) * math.sqrt(6 / (self.inputs + h))
                    for _ in range(4 * h * self.inputs)
                ],
                recurrent_weights=[
                    rng.uniform(-1, 1) * math.sqrt(3 / h) for _ in range(4 * h * h)
                ],
                bias=[0.0] * h + [1.0] * h + [0.0] * (2 * h),
            )
        )
        self.head = MLP(config["network"]["hidden_layers"], actions, seed, inputs=h)

    def step(self, observation, hidden, tape=False, value_only=False):
        h = self.hidden_size
        if len(hidden) != 2 * h:
            raise ValueError("LSTM memory must contain hidden and cell state")
        old, cell = hidden[:h], hidden[h:]
        wx = self.cell["input_weights"]
        wh = self.cell["recurrent_weights"]
        bias = self.cell["bias"]
        if kernels.available("lstm_step") and all(
            isinstance(v, Float32) for v in (wx, wh, bias)
        ):
            state, parts = kernels.native.lstm_step(
                observation, hidden, wx, wh, bias, tape
            )
            current = state[:h]
            if tape:
                outputs, head_tape = self.head.forward(current)
                return (
                    outputs,
                    state,
                    (tuple(observation), tuple(hidden), *parts, head_tape),
                )
            return (
                self.head.value(current) if value_only else self.head.predict(current)
            ), state
        gates = []
        for group, fn in enumerate((sigmoid, sigmoid, math.tanh, sigmoid)):
            values = []
            for j in range(h):
                row = group * h + j
                value = (
                    bias[row]
                    + sum(
                        observation[i] * wx[row * self.inputs + i]
                        for i in range(self.inputs)
                    )
                    + sum(old[i] * wh[row * h + i] for i in range(h))
                )
                values.append(fn(value))
            gates.append(values)
        incoming, forget, candidate, output = gates
        current_cell = [
            f * c + i * g for f, c, i, g in zip(forget, cell, incoming, candidate)
        ]
        current = [o * math.tanh(c) for o, c in zip(output, current_cell)]
        state = current + current_cell
        if tape:
            outputs, head_tape = self.head.forward(current)
            return (
                outputs,
                state,
                (
                    tuple(observation),
                    tuple(hidden),
                    incoming,
                    forget,
                    candidate,
                    output,
                    current_cell,
                    head_tape,
                ),
            )
        return (
            self.head.value(current) if value_only else self.head.predict(current)
        ), state

    def backward(self, tape, output_gradient, future_gradient, gradients):
        x, old, incoming, forget, candidate, output, cell, head_tape = tape
        h = self.hidden_size
        derivative = self.head.accumulate_gradient(
            head_tape, output_gradient, gradients[1:], return_input_gradient=True
        )
        if (
            kernels.available("lstm_backward")
            and isinstance(self.cell["recurrent_weights"], Float32)
            and all(
                isinstance(v, array) and v.typecode == "d"
                for v in gradients[0].values()
            )
        ):
            return kernels.native.lstm_backward(
                (
                    x,
                    old,
                    incoming,
                    forget,
                    candidate,
                    output,
                    cell,
                    derivative,
                    future_gradient,
                ),
                self.cell["recurrent_weights"],
                gradients[0],
            )
        dh = [a + b for a, b in zip(derivative, future_gradient[:h])]
        tanh_cell = [math.tanh(v) for v in cell]
        prior_hidden = old[:h]
        dc = [
            future_gradient[h + j] + dh[j] * output[j] * (1 - tanh_cell[j] ** 2)
            for j in range(h)
        ]
        deltas = [
            [dc[j] * candidate[j] * incoming[j] * (1 - incoming[j]) for j in range(h)],
            [dc[j] * old[h + j] * forget[j] * (1 - forget[j]) for j in range(h)],
            [dc[j] * incoming[j] * (1 - candidate[j] ** 2) for j in range(h)],
            [dh[j] * tanh_cell[j] * output[j] * (1 - output[j]) for j in range(h)],
        ]
        gradient = gradients[0]
        wh = self.cell["recurrent_weights"]
        previous = [
            sum(
                deltas[g][j] * wh[(g * h + j) * h + i]
                for g in range(4)
                for j in range(h)
            )
            for i in range(h)
        ]
        for g, delta in enumerate(deltas):
            for j, d in enumerate(delta):
                row = g * h + j
                gradient["bias"][row] += d
                for i, v in enumerate(x):
                    gradient["input_weights"][row * self.inputs + i] += d * v
                for i, v in enumerate(prior_hidden):
                    gradient["recurrent_weights"][row * h + i] += d * v
        return previous + [dc[j] * forget[j] for j in range(h)]


class LSTMPPOBrain(RecurrentPPOBrain):
    algorithm = "lstm_ppo"

    def __init__(self, config, actions, base_seed, individual_seed):
        super().__init__(config, actions, base_seed, individual_seed)
        self.reset_memory()

    def _new_network(self, config, actions, seed):
        return LSTMNetwork(config, actions, seed)

    def reset_memory(self):
        self.hidden = [0.0] * (2 * self.config["recurrent_size"])

    def diagnostics(self, observation, include_activations=False):
        if include_activations:
            outputs, current, tape = self.network.step(
                observation, self.hidden, tape=True
            )
        else:
            outputs, current = self.network.step(observation, self.hidden)
        p, logs = policy_distribution(outputs[:-1])
        h = self.config["recurrent_size"]
        info = dict(
            algorithm=self.algorithm,
            features=observation,
            action_probabilities=dict(zip(self.actions, p)),
            state_value=outputs[-1],
            entropy=-sum(v * lp for v, lp in zip(p, logs)),
            updates=self.updates,
            rollout_updates=self.rollout_updates,
            buffer_kind="sequence",
            buffer_size=len(self.rollout),
            recurrent_size=h,
            recurrent_cell="lstm",
            memory_norm=math.sqrt(sum(v * v for v in self.hidden[:h])),
            cell_norm=math.sqrt(sum(v * v for v in self.hidden[h:])),
            training_metrics=self.last_metrics.copy(),
        )
        return (
            activation_sample(info, [current[:h]] + list(tape[-1][1:-1]), self.hidden)
            if include_activations
            else info
        )
