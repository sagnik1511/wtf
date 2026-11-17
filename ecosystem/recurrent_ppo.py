"""GRU PPO with chronological truncated backpropagation through time."""

import copy
import math
import random
from typing import NamedTuple
from .brain import INPUTS
from .ppo import MLP, PPOBrain, advantages_and_returns, policy_distribution


class RecurrentStep(NamedTuple):
    observation: tuple
    action: int
    reward: float
    log_probability: float
    value: float
    next_value: float
    terminated: bool
    truncated: bool
    hidden: tuple


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def input_gradient(network, tape, gradient):
    """Derivative w.r.t. an MLP's inputs, without adding parameter gradients."""
    delta = list(gradient)
    for index in range(len(network.layers) - 1, -1, -1):
        inputs = network.sizes[index]
        weights = network.layers[index]['weights']
        prior = [sum(d * weights[j * inputs + i] for j, d in enumerate(delta)) for i in range(inputs)]
        delta = [g * (1.0 - v*v) for g, v in zip(prior, tape[index])] if index else prior
    return delta


class GRUNetwork:
    def __init__(self, config, actions, seed):
        self.hidden_size = h = config['recurrent_size']
        rng = random.Random(seed ^ 0x6A09E667)
        self.cell = dict(
            input_weights=[rng.uniform(-1, 1) * math.sqrt(6 / (INPUTS + h)) for _ in range(3*h*INPUTS)],
            recurrent_weights=[rng.uniform(-1, 1) * math.sqrt(3 / h) for _ in range(3*h*h)],
            bias=[0.0] * (3*h))
        self.head = MLP(config['network']['hidden_layers'], actions, seed, inputs=h)

    @property
    def layers(self):
        return [self.cell] + self.head.layers

    @layers.setter
    def layers(self, values):
        self.cell, self.head.layers = values[0], values[1:]

    def zeros(self):
        return [{k: [0.0] * len(v) for k, v in layer.items()} for layer in self.layers]

    clear_gradients = staticmethod(MLP.clear_gradients)

    def step(self, observation, hidden, tape=False):
        h = self.hidden_size
        wx, wh, bias = (self.cell[k] for k in ('input_weights', 'recurrent_weights', 'bias'))
        def gate(offset, previous, activation):
            return [activation(bias[offset+j]
                    + sum(x*w for x, w in zip(observation, wx[(offset+j)*INPUTS:(offset+j+1)*INPUTS]))
                    + sum(x*w for x, w in zip(previous, wh[(offset+j)*h:(offset+j+1)*h]))) for j in range(h)]
        z, r = gate(0, hidden, sigmoid), gate(h, hidden, sigmoid)
        rh = [a*b for a, b in zip(r, hidden)]
        candidate = gate(2*h, rh, math.tanh)
        current = [(1-a)*n + a*old for a, n, old in zip(z, candidate, hidden)]
        if tape:
            outputs, ht = self.head.forward(current)
            return outputs, current, (tuple(observation), tuple(hidden), z, r, candidate, rh, ht)
        return self.head.predict(current), current

    def backward(self, tape, output_gradient, future_gradient, gradients):
        x, old, z, r, candidate, rh, ht = tape
        h = self.hidden_size
        derivative = input_gradient(self.head, ht, output_gradient)
        self.head.accumulate_gradient(ht, output_gradient, gradients[1:])
        dh = [a+b for a, b in zip(derivative, future_gradient)]
        dz = [d*(v-n)*a*(1-a) for d, v, n, a in zip(dh, old, candidate, z)]
        dn = [d*(1-a)*(1-n*n) for d, a, n in zip(dh, z, candidate)]
        wh = self.cell['recurrent_weights']
        drh = [sum(dn[j]*wh[(2*h+j)*h+i] for j in range(h)) for i in range(h)]
        dr = [d*v*a*(1-a) for d, v, a in zip(drh, old, r)]
        previous_gradient = [dh[i]*z[i] + drh[i]*r[i]
            + sum(dz[j]*wh[j*h+i] + dr[j]*wh[(h+j)*h+i] for j in range(h)) for i in range(h)]
        gradient = gradients[0]
        for offset, delta, previous in ((0, dz, old), (h, dr, old), (2*h, dn, rh)):
            for j, d in enumerate(delta):
                row = offset+j
                gradient['bias'][row] += d
                for i, v in enumerate(x):
                    gradient['input_weights'][row*INPUTS+i] += d*v
                for i, v in enumerate(previous):
                    gradient['recurrent_weights'][row*h+i] += d*v
        return previous_gradient

    def checkpoint(self):
        return [{k: v.copy() for k, v in layer.items()} for layer in self.layers]


class RecurrentPPOBrain(PPOBrain):
    algorithm = 'recurrent_ppo'

    def __init__(self, config, actions, base_seed, individual_seed):
        super().__init__(config, actions, base_seed, individual_seed)
        self.network = GRUNetwork(config, len(actions), base_seed)
        self.moments, self.variances = self.network.zeros(), self.network.zeros()
        self.hidden = [0.0] * config['recurrent_size']

    def reset_memory(self):
        self.hidden = [0.0] * self.config['recurrent_size']

    def action_probabilities(self, observation):
        outputs, _ = self.network.step(observation, self.hidden)
        return policy_distribution(outputs[:-1])[0]

    def choose(self, observation, explore=True, random_baseline=False):
        before = tuple(self.hidden)
        outputs, self.hidden = self.network.step(observation, before)
        probabilities, logs = policy_distribution(outputs[:-1])
        if random_baseline:
            return self.rng.choice(self.actions)
        if not explore:
            return self.actions[max(range(len(probabilities)), key=probabilities.__getitem__)]
        draw, cumulative = self.rng.random(), 0.0
        action = len(probabilities)-1
        for i, p in enumerate(probabilities):
            cumulative += p
            if draw < cumulative:
                action = i
                break
        self.pending = (tuple(observation), action, logs[action], outputs[-1], before)
        return self.actions[action]

    def learn(self, experience):
        action = self.action_indices[experience.action]
        if self.pending is None or self.pending[:2] != (tuple(experience.observation), action):
            raise ValueError('Recurrent PPO requires choose() before learn()')
        obs, action, logp, value, hidden = self.pending
        self.pending = None
        next_value = 0.0
        if not experience.terminated:
            if experience.next_observation is None:
                raise ValueError('Recurrent PPO requires a nonterminal next observation')
            next_value = self.network.step(experience.next_observation, self.hidden)[0][-1]
        self.rollout.append(RecurrentStep(obs, action, experience.reward, logp, value, next_value,
                                          experience.terminated, experience.truncated, hidden))
        self.steps += 1
        if len(self.rollout) >= self.config['rollout_steps'] or experience.terminated or experience.truncated:
            self._update()
        if experience.terminated or experience.truncated:
            self.reset_memory()

    def _update(self):
        c = self.config
        advantages, targets = advantages_and_returns(self.rollout, c['discount_factor'], c['gae_lambda'])
        if c['normalize_advantages'] and len(advantages) > 1:
            mean = sum(advantages) / len(advantages)
            std = math.sqrt(sum((a-mean)**2 for a in advantages)/len(advantages) + 1e-8)
            advantages = [(a-mean)/std for a in advantages]
        length = c['sequence_length']
        sequences = [list(range(i, min(i+length, len(self.rollout)))) for i in range(0, len(self.rollout), length)]
        per_batch = max(1, c['batch_size']//length)
        gradients, totals, samples, stopped = self.network.zeros(), {}, 0, False
        for _ in range(c['epochs']):
            self.training_rng.shuffle(sequences)
            for offset in range(0, len(sequences), per_batch):
                self.network.clear_gradients(gradients)
                batch_metrics, count = {}, 0
                for indices in sequences[offset:offset+per_batch]:
                    hidden = self.rollout[indices[0]].hidden
                    tapes = []
                    for i in indices:
                        step = self.rollout[i]
                        outputs, hidden, tape = self.network.step(step.observation, hidden, tape=True)
                        _, derivative, info = self.objective(outputs, step, advantages[i], targets[i])
                        tapes.append((tape, derivative))
                        for key, value in info.items():
                            batch_metrics[key] = batch_metrics.get(key, 0.0)+value
                        count += 1
                    future = [0.0] * self.config['recurrent_size']
                    for tape, derivative in reversed(tapes):
                        future = self.network.backward(tape, derivative, future, gradients)
                if c.get('target_kl', 0.03) and batch_metrics['approximate_kl']/count > c.get('target_kl', 0.03):
                    stopped = True
                    break
                self._adam_step(gradients, count)
                samples += count
                for key, value in batch_metrics.items():
                    totals[key] = totals.get(key, 0.0)+value
            if stopped:
                break
        self.rollout_updates += 1
        self.last_metrics = {k: v/samples for k, v in totals.items()} if samples else {}
        self.last_metrics['early_stopped'] = stopped
        self.rollout.clear()

    def diagnostics(self, observation):
        outputs, _ = self.network.step(observation, self.hidden)
        probabilities, logs = policy_distribution(outputs[:-1])
        return dict(algorithm=self.algorithm, features=observation,
                    action_probabilities=dict(zip(self.actions, probabilities)), state_value=outputs[-1],
                    entropy=-sum(p*lp for p, lp in zip(probabilities, logs)), updates=self.updates,
                    rollout_updates=self.rollout_updates, buffer_kind='sequence', buffer_size=len(self.rollout),
                    recurrent_size=len(self.hidden), memory_norm=math.sqrt(sum(v*v for v in self.hidden)),
                    training_metrics=self.last_metrics.copy())

    def checkpoint(self):
        state = super().checkpoint()
        state['hidden'] = self.hidden[:]
        return state

    @classmethod
    def restore(cls, state):
        # Reuse PPO restoration for optimizer/RNG fields, then restore sequence state.
        compatible = dict(state, rollout=[], pending=None)
        b = super().restore(compatible)
        b.hidden = list(state['hidden'])
        b.rollout = [RecurrentStep(tuple(s['observation']), s['action'], s['reward'], s['log_probability'],
                    s['value'], s['next_value'], s['terminated'], s['truncated'], tuple(s['hidden']))
                    for s in state['rollout']]
        pending = state.get('pending')
        b.pending = None if pending is None else (tuple(pending[0]), *pending[1:4], tuple(pending[4]))
        return b
