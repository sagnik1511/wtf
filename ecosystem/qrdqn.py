"""Quantile-regression Double DQN with bounded per-animal replay and Adam."""

import copy
import random
from .brain import Experience
from .optimization import AdamOptimizer
from .ppo import MLP


def quantile_huber(predictions, targets, kappa):
    """Mean pairwise quantile Huber objective and prediction derivatives."""
    n, m = len(predictions), len(targets)
    loss, gradient = 0.0, [0.0] * n
    for i, value in enumerate(predictions):
        tau = (i + 0.5) / n
        for target in targets:
            delta = target - value
            weight = abs(tau - float(delta < 0))
            absolute = abs(delta)
            huber = 0.5 * delta * delta if absolute <= kappa else kappa * (absolute - 0.5 * kappa)
            loss += weight * huber / kappa
            gradient[i] -= weight * max(-kappa, min(kappa, delta)) / kappa
    scale = n * m
    return loss / scale, [g / scale for g in gradient]


class QRDQNBrain(AdamOptimizer):
    algorithm = 'qrdqn'

    def __init__(self, config, actions, base_seed, individual_seed):
        self.config = copy.deepcopy(config)
        self.actions = list(actions)
        self.action_indices = {a: i for i, a in enumerate(actions)}
        self.quantiles = config['num_quantiles']
        self.network = MLP(config['network']['hidden_layers'], len(actions) * self.quantiles,
                           base_seed, value_head=False)
        self.target_network = copy.deepcopy(self.network)
        self.moments, self.variances = self.network.zeros(), self.network.zeros()
        self.rng = random.Random(individual_seed)
        self.training_rng = random.Random(individual_seed ^ 0x5F3759DF)
        self.replay = []
        self.cursor = self.steps = self.updates = 0
        self.epsilon = config['exploration']['initial']
        self.last_metrics = {}

    def values(self, observation, network=None):
        values = (network or self.network).predict(observation)
        return self._mean_values(values)

    def _mean_values(self, values):
        n = self.quantiles
        return [sum(values[i:i+n]) / n for i in range(0, len(values), n)]

    def choose(self, observation, explore=True, random_baseline=False):
        if random_baseline or (explore and self.rng.random() < self.epsilon):
            return self.rng.choice(self.actions)
        q = self.values(observation)
        return self.actions[max(range(len(q)), key=q.__getitem__)]

    def learn(self, experience):
        c = self.config
        if len(self.replay) < c['replay_capacity']:
            self.replay.append(experience)
        else:
            self.replay[self.cursor] = experience
        self.cursor = (self.cursor + 1) % c['replay_capacity']
        self.steps += 1
        self.epsilon = max(c['exploration']['minimum'], self.epsilon * c['exploration']['decay_per_decision'])
        if len(self.replay) < max(c['batch_size'], c['learning_starts_after']) or self.steps % c['train_every_steps']:
            return
        gradients, total_loss = self.network.zeros(), 0.0
        n = self.quantiles
        for e in self.training_rng.sample(self.replay, c['batch_size']):
            outputs, tape = self.network.forward(e.observation)
            start = self.action_indices[e.action] * n
            targets = [e.reward] * n
            if not e.terminated:
                if e.next_observation is None:
                    raise ValueError('QR-DQN needs a next observation unless terminated')
                q = self.values(e.next_observation)
                best = max(range(len(q)), key=q.__getitem__)
                bootstrap = self.target_network.predict_range(e.next_observation, best*n, (best+1)*n)
                targets = [e.reward + c['discount_factor'] * v for v in bootstrap]
            loss, derivative = quantile_huber(outputs[start:start+n], targets, c['huber_threshold'])
            total_loss += loss
            gradient = [0.0] * len(outputs)
            gradient[start:start+n] = derivative
            self.network.accumulate_gradient(tape, gradient, gradients)
        self._adam_step(gradients, c['batch_size'])
        if self.updates % c['target_update_every_training_steps'] == 0:
            for target, online in zip(self.target_network.layers, self.network.layers):
                for key, values in online.items():
                    target[key][:] = values
        self.last_metrics = {'quantile_loss': total_loss / c['batch_size']}

    def diagnostics(self, observation):
        outputs, n = self.network.predict(observation), self.quantiles
        return dict(algorithm=self.algorithm, features=observation,
                    action_values=dict(zip(self.actions, self._mean_values(outputs))),
                    return_quantiles={a: outputs[i*n:(i+1)*n] for i, a in enumerate(self.actions)},
                    epsilon=self.epsilon, updates=self.updates, replay_size=len(self.replay),
                    training_metrics=self.last_metrics.copy())

    def checkpoint(self):
        return dict(algorithm=self.algorithm, config=copy.deepcopy(self.config), actions=self.actions[:],
                    network=self.network.checkpoint(), target_network=self.target_network.checkpoint(),
                    moments=copy.deepcopy(self.moments), variances=copy.deepcopy(self.variances),
                    replay=[e._asdict() for e in self.replay], cursor=self.cursor,
                    steps=self.steps, updates=self.updates, epsilon=self.epsilon,
                    last_metrics=self.last_metrics.copy(), rng_state=self.rng.getstate(),
                    training_rng_state=self.training_rng.getstate())

    @classmethod
    def restore(cls, state):
        b = cls(state['config'], state['actions'], 0, 0)
        b.network.layers = copy.deepcopy(state['network'])
        b.target_network.layers = copy.deepcopy(state['target_network'])
        for key in ('moments', 'variances', 'cursor', 'steps', 'updates', 'epsilon', 'last_metrics'):
            setattr(b, key, copy.deepcopy(state[key]))
        b.replay = [Experience(tuple(e['observation']), e['action'], e['reward'],
                              None if e['next_observation'] is None else tuple(e['next_observation']),
                              e['terminated'], e['truncated']) for e in state['replay']]
        def tuples(x):
            return tuple(tuples(v) for v in x) if isinstance(x, (list, tuple)) else x
        b.rng.setstate(tuples(state['rng_state']))
        b.training_rng.setstate(tuples(state['training_rng_state']))
        return b
