"""Independent A2C, episodic REINFORCE, and experimental branched GRPO."""
import copy
import math
import random
from typing import NamedTuple
from .optimization import AdamOptimizer
from .ppo import MLP, PPOBrain, advantages_and_returns, policy_distribution


def actor_loss_gradient(logits, action, advantage, entropy_coefficient, old_logp=None, clip_ratio=.2, reference_logits=None, kl_coefficient=0.):
    probabilities, logs = policy_distribution(logits)
    entropy = -sum(p * logp for p, logp in zip(probabilities, logs))
    approximate_kl = 0.
    if old_logp is None:
        policy_loss, derivative = -logs[action] * advantage, -advantage
    else:
        log_ratio = max(-60., min(60., logs[action] - old_logp))
        ratio = math.exp(log_ratio)
        clipped = max(1. - clip_ratio, min(1. + clip_ratio, ratio))
        policy_loss = -min(ratio * advantage, clipped * advantage)
        inactive = (advantage > 0 and ratio > 1. + clip_ratio) or (advantage < 0 and ratio < 1. - clip_ratio)
        derivative = 0. if inactive else -advantage * ratio
        approximate_kl = (ratio - 1.) - log_ratio
    gradient = [derivative * ((1. if i == action else 0.) - p) + entropy_coefficient * p * (logs[i] + entropy) for i, p in enumerate(probabilities)]
    reference_kl = 0.
    if reference_logits is not None and kl_coefficient:
        reference_logs = policy_distribution(reference_logits)[1]
        differences = [lp - rp for lp, rp in zip(logs, reference_logs)]
        reference_kl = sum(p * d for p, d in zip(probabilities, differences))
        for i, p in enumerate(probabilities):
            gradient[i] += kl_coefficient * p * (differences[i] - reference_kl)
    loss = policy_loss - entropy_coefficient * entropy + kl_coefficient * reference_kl
    return loss, gradient, dict(policy_loss=policy_loss, entropy=entropy, approximate_kl=approximate_kl, reference_kl=reference_kl)


def normalized(values):
    if len(values) <= 1:
        return values
    mean = sum(values) / len(values)
    scale = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values) + 1e-8)
    return [(v - mean) / scale for v in values]


class A2CBrain(PPOBrain):
    algorithm = 'a2c'

    def _update(self):
        c = self.config
        advantages, targets = advantages_and_returns(self.rollout, c['discount_factor'], c['gae_lambda'])
        if c['normalize_advantages']:
            advantages = normalized(advantages)
        gradients = self.network.zeros()
        totals = dict(policy_loss=0., value_loss=0., entropy=0.)
        for step, advantage, target in zip(self.rollout, advantages, targets):
            outputs, activations = self.network.forward(step.observation)
            _, derivative, metrics = actor_loss_gradient(outputs[:-1], step.action, advantage, c['entropy_coefficient'])
            error = outputs[-1] - target
            derivative.append(c['value_coefficient'] * error)
            self.network.accumulate_gradient(activations, derivative, gradients)
            totals['policy_loss'] += metrics['policy_loss']
            totals['entropy'] += metrics['entropy']
            totals['value_loss'] += .5 * error * error
        self._adam_step(gradients, len(self.rollout))
        self.last_metrics = {key: value / len(self.rollout) for key, value in totals.items()}
        self.rollout_updates += 1
        self.rollout.clear()


class PolicyStep(NamedTuple):
    observation: tuple
    action: int
    reward: float
    log_probability: float
    terminated: bool
    truncated: bool


class ActorBrain(AdamOptimizer):
    """Common categorical policy and checkpointing, without a value network."""
    def __init__(self, config, actions, base_seed, individual_seed):
        self.config, self.actions = copy.deepcopy(config), list(actions)
        self.action_indices = {name: i for i, name in enumerate(actions)}
        self.network = MLP(config['network']['hidden_layers'], len(actions), base_seed, value_head=False)
        self.rng = random.Random(individual_seed)
        self.training_rng = random.Random(individual_seed ^ 0x5F3759DF)
        self.moments, self.variances = self.network.zeros(), self.network.zeros()
        self.trajectory, self.pending = [], None
        self.steps = self.updates = self.rollout_updates = 0
        self.baseline = 0.
        self.last_metrics = {}

    def action_probabilities(self, observation):
        return policy_distribution(self.network.forward(observation)[0])[0]

    def choose(self, observation, explore=True, random_baseline=False):
        if random_baseline:
            return self.rng.choice(self.actions)
        logits, _ = self.network.forward(observation)
        probabilities, logs = policy_distribution(logits)
        if not explore:
            return self.actions[max(range(len(probabilities)), key=probabilities.__getitem__)]
        draw, cumulative, action = self.rng.random(), 0., len(self.actions) - 1
        for i, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative:
                action = i
                break
        self.pending = (tuple(observation), action, logs[action])
        return self.actions[action]

    def diagnostics(self, observation):
        probabilities, logs = policy_distribution(self.network.forward(observation)[0])
        info = dict(algorithm=self.algorithm, features=observation, action_probabilities=dict(zip(self.actions, probabilities)), entropy=-sum(p * lp for p, lp in zip(probabilities, logs)), updates=self.updates, rollout_updates=self.rollout_updates, buffer_kind='episode' if self.algorithm == 'reinforce' else 'group', buffer_size=len(self.trajectory), training_metrics=copy.deepcopy(self.last_metrics))
        if self.algorithm == 'reinforce':
            info['return_baseline'] = self.baseline
        else:
            info.update(group_size=self.config['group_size'], rollout_horizon=self.config['rollout_horizon'])
        return info

    def checkpoint(self):
        state = dict(algorithm=self.algorithm, config=copy.deepcopy(self.config), actions=self.actions[:], network=self.network.checkpoint(), moments=copy.deepcopy(self.moments), variances=copy.deepcopy(self.variances), trajectory=[s._asdict() for s in self.trajectory], pending=self.pending, steps=self.steps, updates=self.updates, rollout_updates=self.rollout_updates, baseline=self.baseline, last_metrics=copy.deepcopy(self.last_metrics), rng_state=self.rng.getstate(), training_rng_state=self.training_rng.getstate())
        if self.algorithm == 'grpo':
            state.update(reference_network=self.reference_network.checkpoint(), group_rng_state=self.group_rng.getstate(), last_group_step=self.last_group_step)
        return state

    @classmethod
    def restore(cls, state):
        b = cls(state['config'], state['actions'], 0, 0)
        b.network.layers = copy.deepcopy(state['network'])
        for key in ('moments', 'variances', 'steps', 'updates', 'rollout_updates', 'baseline', 'last_metrics'):
            setattr(b, key, copy.deepcopy(state[key]))
        b.trajectory = [PolicyStep(tuple(s['observation']), s['action'], s['reward'], s['log_probability'], s['terminated'], s['truncated']) for s in state['trajectory']]
        pending = state['pending']
        b.pending = None if pending is None else (tuple(pending[0]), *pending[1:])
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
        b.rng.setstate(tuples(state['rng_state']))
        b.training_rng.setstate(tuples(state['training_rng_state']))
        if b.algorithm == 'grpo':
            b.reference_network.layers = copy.deepcopy(state['reference_network'])
            b.group_rng.setstate(tuples(state['group_rng_state']))
            b.last_group_step = state['last_group_step']
        return b


class ReinforceBrain(ActorBrain):
    algorithm = 'reinforce'

    def learn(self, experience):
        if len(self.trajectory) >= self.config['max_episode_steps']:
            raise ValueError('REINFORCE episode exceeded max_episode_steps')
        action = self.action_indices[experience.action]
        if self.pending is not None and self.pending[:2] == (tuple(experience.observation), action):
            old_logp = self.pending[2]
        else:
            old_logp = policy_distribution(self.network.forward(experience.observation)[0])[1][action]
        self.pending = None
        self.trajectory.append(PolicyStep(tuple(experience.observation), action, experience.reward, old_logp, experience.terminated, experience.truncated))
        self.steps += 1
        if experience.terminated or experience.truncated:
            self._update()

    def _update(self):
        c = self.config
        returns, total = [0.] * len(self.trajectory), 0.
        for i in range(len(self.trajectory) - 1, -1, -1):
            total = self.trajectory[i].reward + c['discount_factor'] * total
            returns[i] = total
        advantages = [(c['discount_factor'] ** i) * (value - self.baseline) for i, value in enumerate(returns)]
        if c['normalize_advantages']:
            advantages = normalized(advantages)
        gradients = self.network.zeros()
        totals = dict(policy_loss=0., entropy=0.)
        for step, advantage in zip(self.trajectory, advantages):
            logits, activations = self.network.forward(step.observation)
            _, derivative, info = actor_loss_gradient(logits, step.action, advantage, c['entropy_coefficient'])
            self.network.accumulate_gradient(activations, derivative, gradients)
            for key in totals:
                totals[key] += info[key]
        count = len(self.trajectory)
        # Episodic REINFORCE uses the summed discounted policy gradient.
        self._adam_step(gradients, 1)
        mean_return = sum(returns) / count
        self.baseline = c['baseline_decay'] * self.baseline + (1. - c['baseline_decay']) * mean_return
        self.last_metrics = {key: value / count for key, value in totals.items()}
        self.last_metrics['mean_return'] = mean_return
        self.rollout_updates += 1
        self.trajectory.clear()


class GRPOBrain(ActorBrain):
    """Experimental GRPO for finite simulator branches, not unrelated agents."""
    algorithm = 'grpo'

    def __init__(self, config, actions, base_seed, individual_seed):
        super().__init__(config, actions, base_seed, individual_seed)
        self.reference_network = MLP(config['network']['hidden_layers'], len(actions), base_seed, value_head=False)
        self.group_rng = random.Random(individual_seed ^ 0x0BADC0DE)
        self.last_group_step = 0

    @property
    def needs_group(self):
        return self.steps - self.last_group_step >= self.config['train_every_steps']

    def learn(self, experience):
        # Training data comes from matched simulator trials collected by the
        # runner. A single real transition is not a comparable outcome group.
        self.steps += 1
        self.pending = None

    def update_group(self, trajectories):
        c = self.config
        if len(trajectories) < 2 or any(not t for t in trajectories):
            raise ValueError('GRPO needs at least two nonempty matched trajectories')
        returns = [sum((c['discount_factor'] ** t) * step.reward for t, step in enumerate(trajectory)) for trajectory in trajectories]
        mean = sum(returns) / len(returns)
        deviation = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
        self.last_group_step = self.steps
        self.rollout_updates += 1
        self.last_metrics = dict(group_returns=returns, mean_return=mean, return_std=deviation, zero_variance=deviation < 1e-8, early_stopped=False)
        if deviation < 1e-8:
            return
        count = sum(len(t) for t in trajectories)
        samples = [(step, (r - mean) / (deviation + 1e-8), count / (len(trajectories) * len(trajectory))) for trajectory, r in zip(trajectories, returns) for step in trajectory]
        indices = list(range(count))
        for _ in range(c['epochs']):
            self.training_rng.shuffle(indices)
            for offset in range(0, count, c['batch_size']):
                batch = indices[offset:offset + c['batch_size']]
                gradients, approximate_kl = self.network.zeros(), 0.
                for index in batch:
                    step, advantage, weight = samples[index]
                    logits, activations = self.network.forward(step.observation)
                    reference = self.reference_network.forward(step.observation)[0]
                    _, derivative, info = actor_loss_gradient(logits, step.action, advantage, c['entropy_coefficient'], old_logp=step.log_probability, clip_ratio=c['clip_ratio'], reference_logits=reference, kl_coefficient=c['kl_coefficient'])
                    self.network.accumulate_gradient(activations, [value * weight for value in derivative], gradients)
                    approximate_kl += info['approximate_kl']
                target_kl = c.get('target_kl', .03)
                if target_kl and approximate_kl / len(batch) > target_kl:
                    self.last_metrics['early_stopped'] = True
                    return
                self._adam_step(gradients, len(batch))
