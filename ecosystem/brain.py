"""Small independent Q network, replay, target network and clipped loss.

Lists and standard-library math keep installation self-contained. Networks are
small deliberately; immutable replay tuples and native map operations reduce overhead.
"""
import copy
import math
import random
from operator import mul
from typing import NamedTuple, Optional

INPUTS = 20


class Experience(NamedTuple):
    observation: tuple
    action: str
    reward: float
    next_observation: Optional[tuple]
    terminated: bool
    truncated: bool = False


class Brain:
    def __init__(self, config, actions, base_seed, individual_seed):
        self.config = copy.deepcopy(config)
        self.actions = list(actions)
        self.hidden = config['hidden_layers'][0]
        self.rng = random.Random(individual_seed)
        self.training_rng = random.Random(individual_seed ^ 0x5F3759DF)
        init = random.Random(base_seed)
        self.weights = {
            'w1': [init.uniform(-1, 1) * math.sqrt(6 / (INPUTS + self.hidden)) for _ in range(INPUTS * self.hidden)],
            'b1': [0.] * self.hidden,
            'w2': [init.uniform(-1, 1) * math.sqrt(6 / (self.hidden + len(actions))) for _ in range(self.hidden * len(actions))],
            'b2': [0.] * len(actions),
        }
        self.target = {k: v.copy() for k, v in self.weights.items()}
        self.replay = []
        self.cursor = self.steps = self.updates = 0
        self.epsilon = config['exploration']['initial']

    def forward(self, observation, weights=None):
        w = self.weights if weights is None else weights
        w1, w2, hidden = w['w1'], w['w2'], self.hidden
        # map(mul) avoids millions of Python generator/index operations while
        # preserving the exact multiplication and summation order.
        h = [math.tanh(bias + sum(map(mul, observation, w1[j * INPUTS:(j + 1) * INPUTS]))) for j, bias in enumerate(w['b1'])]
        q = [bias + sum(map(mul, h, w2[a * hidden:(a + 1) * hidden])) for a, bias in enumerate(w['b2'])]
        return h, q

    def choose(self, observation, explore=True, random_baseline=False):
        if random_baseline or (explore and self.rng.random() < self.epsilon):
            return self.rng.choice(self.actions)
        q = self.forward(observation)[1]
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
        if len(self.replay) < max(c['learning_starts_after'], c['batch_size']) or self.steps % c['train_every_steps']:
            return
        gradients = {k: [0.] * len(v) for k, v in self.weights.items()}
        gw1, gb1, gw2, gb2 = (gradients[k] for k in ('w1', 'b1', 'w2', 'b2'))
        w2, hidden = self.weights['w2'], self.hidden
        for e in self.training_rng.sample(self.replay, c['batch_size']):
            h, q = self.forward(e.observation)
            a = self.actions.index(e.action)
            expected = e.reward
            if not e.terminated:
                expected += c['discount_factor'] * max(self.forward(e.next_observation, self.target)[1])
            delta = max(-1., min(1., q[a] - expected))
            gb2[a] += delta
            for j in range(self.hidden):
                gw2[a * hidden + j] += delta * h[j]
                dh = delta * w2[a * hidden + j] * (1 - h[j] ** 2)
                gb1[j] += dh
                for i, x in enumerate(e.observation):
                    gw1[j * INPUTS + i] += dh * x
        scale = c['learning_rate'] / c['batch_size']
        for k, gradient in gradients.items():
            for i, value in enumerate(gradient):
                self.weights[k][i] -= scale * value
        self.updates += 1
        if self.updates % c['target_update_every_training_steps'] == 0:
            self.target = {k: v.copy() for k, v in self.weights.items()}

    def diagnostics(self, observation):
        return dict(algorithm='q_learning', features=observation, action_values=dict(zip(self.actions, self.forward(observation)[1])), epsilon=self.epsilon, updates=self.updates, replay_size=len(self.replay))

    def checkpoint(self):
        return dict(config=copy.deepcopy(self.config), actions=self.actions[:], weights={k: v.copy() for k, v in self.weights.items()}, target={k: v.copy() for k, v in self.target.items()}, replay=[dict(observation=tuple(e.observation), action=e.action, reward=e.reward, next_observation=tuple(e.next_observation) if e.next_observation is not None else None, terminated=e.terminated, truncated=e.truncated) for e in self.replay], cursor=self.cursor, steps=self.steps, updates=self.updates, epsilon=self.epsilon, rng_state=self.rng.getstate(), training_rng_state=self.training_rng.getstate())

    @classmethod
    def restore(cls, state):
        b = cls(state['config'], state['actions'], 0, 0)
        for k in ('weights', 'target', 'cursor', 'steps', 'updates', 'epsilon'):
            setattr(b, k, copy.deepcopy(state[k]))
        b.replay = [Experience(tuple(e['observation']), e['action'], e['reward'], None if e['next_observation'] is None else tuple(e['next_observation']), e['terminated'], e['truncated']) for e in state['replay']]
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value
        b.rng.setstate(tuples(state['rng_state']))
        b.training_rng.setstate(tuples(state['training_rng_state']))
        return b
