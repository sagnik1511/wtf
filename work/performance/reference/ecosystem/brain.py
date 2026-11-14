"""Small independent Q network, replay, target network and clipped loss.

Lists and standard-library math keep installation self-contained. Networks are
small deliberately; a future NumPy backend can implement the same interface.
"""
import copy
import math
import random
from dataclasses import dataclass
from typing import Optional

INPUTS = 20


@dataclass(frozen=True)
class Experience:
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
        self.target = copy.deepcopy(self.weights)
        self.replay = []
        self.cursor = self.steps = self.updates = 0
        self.epsilon = config['exploration']['initial']

    def forward(self, observation, weights=None):
        w = self.weights if weights is None else weights
        h = [math.tanh(w['b1'][j] + sum(x * w['w1'][j * INPUTS + i] for i, x in enumerate(observation))) for j in range(self.hidden)]
        q = [w['b2'][a] + sum(x * w['w2'][a * self.hidden + j] for j, x in enumerate(h)) for a in range(len(self.actions))]
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
        for e in self.training_rng.sample(self.replay, c['batch_size']):
            h, q = self.forward(e.observation)
            a = self.actions.index(e.action)
            expected = e.reward
            if not e.terminated:
                expected += c['discount_factor'] * max(self.forward(e.next_observation, self.target)[1])
            delta = max(-1., min(1., q[a] - expected))
            gradients['b2'][a] += delta
            for j in range(self.hidden):
                gradients['w2'][a * self.hidden + j] += delta * h[j]
                dh = delta * self.weights['w2'][a * self.hidden + j] * (1 - h[j] ** 2)
                gradients['b1'][j] += dh
                for i, x in enumerate(e.observation):
                    gradients['w1'][j * INPUTS + i] += dh * x
        scale = c['learning_rate'] / c['batch_size']
        for k, gradient in gradients.items():
            for i, value in enumerate(gradient):
                self.weights[k][i] -= scale * value
        self.updates += 1
        if self.updates % c['target_update_every_training_steps'] == 0:
            self.target = copy.deepcopy(self.weights)

    def checkpoint(self):
        from dataclasses import asdict
        return dict(config=copy.deepcopy(self.config), actions=self.actions[:], weights=copy.deepcopy(self.weights), target=copy.deepcopy(self.target), replay=[asdict(e) for e in self.replay], cursor=self.cursor, steps=self.steps, updates=self.updates, epsilon=self.epsilon, rng_state=self.rng.getstate(), training_rng_state=self.training_rng.getstate())

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
