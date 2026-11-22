"""Discrete SAC: exact categorical expectations, twin Q critics and soft targets.

Temperature is fixed/configurable. No continuous-action reparameterization or
shared species policy is used. Target critics bootstrap truncations, not deaths.
"""

import copy
from wtf.learning.brain import Experience, observation_inputs
from wtf.learning.ppo import PPOBrain, policy_distribution
from wtf.learning.policy_networks import PolicyBundle
from wtf.native.numeric import packed, precision_of


def soft_value(probabilities, logs, first, second, temperature):
    return sum(
        p * (min(a, b) - temperature * lp)
        for p, lp, a, b in zip(probabilities, logs, first, second)
    )


def actor_objective(logits, first, second, temperature):
    probabilities, logs = policy_distribution(logits)
    q = [min(a, b) for a, b in zip(first, second)]
    expected = sum(p * v for p, v in zip(probabilities, q))
    entropy = -sum(p * lp for p, lp in zip(probabilities, logs))
    loss = -expected - temperature * entropy
    gradient = [
        p * (temperature * (lp + entropy) - v + expected)
        for p, lp, v in zip(probabilities, logs, q)
    ]
    return loss, gradient, entropy


class DiscreteSACBrain(PPOBrain):
    algorithm = "discrete_sac"

    def __init__(self, config, actions, base_seed, individual_seed):
        super().__init__(config, actions, base_seed, individual_seed)
        self.replay = []
        self.cursor = 0
        self.target_critics = copy.deepcopy(self.network.critics)

    def _new_network(self, config, actions, seed):
        return PolicyBundle(
            config, actions, seed, observation_inputs(config), 2, actions
        )

    def synchronize_targets(self):
        for target, critic in zip(self.target_critics, self.network.critics):
            target.layers = copy.deepcopy(critic.layers)

    def action_probabilities(self, observation):
        return policy_distribution(self.network.actor.predict(observation))[0]

    def choose(self, observation, explore=True, random_baseline=False):
        if random_baseline:
            return self.rng.choice(self.actions)
        return self._choose_outputs(
            observation, self.network.actor.predict(observation) + [0.0], explore
        )

    def learn(self, experience):
        if not experience.terminated and experience.next_observation is None:
            raise ValueError("Discrete SAC requires a nonterminal next observation")
        self.pending = None
        if len(self.replay) < self.config["replay_capacity"]:
            self.replay.append(experience)
        else:
            self.replay[self.cursor] = experience
        self.cursor = (self.cursor + 1) % self.config["replay_capacity"]
        self.steps += 1
        if (
            len(self.replay)
            >= max(self.config["learning_starts_after"], self.config["batch_size"])
            and self.steps % self.config["train_every_steps"] == 0
        ):
            self._update()

    def _update(self):
        c = self.config
        alpha = c["temperature"]
        block = self.network.block
        batch = self.training_rng.sample(self.replay, c["batch_size"])
        gradients = self.network.zeros()
        actor_gradients = gradients[:block]
        critic_gradients = [
            gradients[(j + 1) * block : (j + 2) * block] for j in range(2)
        ]
        metrics = dict(policy_loss=0.0, critic_loss=0.0, entropy=0.0)
        for step in batch:
            target = step.reward
            if not step.terminated:
                p, logs = policy_distribution(
                    self.network.actor.predict(step.next_observation)
                )
                q = [
                    critic.predict(step.next_observation)
                    for critic in self.target_critics
                ]
                target += c["discount_factor"] * soft_value(p, logs, *q, alpha)
            values = []
            for j, critic in enumerate(self.network.critics):
                outputs, tape = critic.forward(step.observation)
                values.append(outputs)
                error = outputs[self.action_indices[step.action]] - target
                derivative = [0.0] * len(self.actions)
                derivative[self.action_indices[step.action]] = error
                critic.accumulate_gradient(tape, derivative, critic_gradients[j])
                metrics["critic_loss"] += 0.5 * error * error
            logits, tape = self.network.actor.forward(step.observation)
            loss, derivative, entropy = actor_objective(logits, *values, alpha)
            self.network.actor.accumulate_gradient(tape, derivative, actor_gradients)
            metrics["policy_loss"] += loss
            metrics["entropy"] += entropy
        self._adam_step(gradients, len(batch))
        tau = c["target_tau"]
        retained = 1 - tau
        for target, critic in zip(self.target_critics, self.network.critics):
            for left, right in zip(target.layers, critic.layers):
                for key, values in left.items():
                    for i, value in enumerate(values):
                        values[i] = retained * value + tau * right[key][i]
        self.rollout_updates += 1
        self.last_metrics = {k: v / len(batch) for k, v in metrics.items()}

    def diagnostics(self, observation, include_activations=False):
        info = super().diagnostics(observation, include_activations)
        q = [critic.predict(observation) for critic in self.network.critics]
        p = list(info["action_probabilities"].values())
        info.update(
            buffer_kind="replay",
            buffer_size=len(self.replay),
            rollout_size=0,
            state_value=sum(v * min(a, b) for v, a, b in zip(p, *q))
            + self.config["temperature"] * info["entropy"],
            temperature=self.config["temperature"],
            twin_q_values=[dict(zip(self.actions, values)) for values in q],
            critic_separate=True,
        )
        return info

    def checkpoint(self):
        state = super().checkpoint()
        state.update(
            replay=[s._asdict() for s in self.replay],
            cursor=self.cursor,
            target_critics=[n.checkpoint() for n in self.target_critics],
        )
        return state

    @classmethod
    def restore(cls, state):
        brain = super().restore(state)
        brain.replay = [
            Experience(
                tuple(s["observation"]),
                s["action"],
                s["reward"],
                None if s["next_observation"] is None else tuple(s["next_observation"]),
                s["terminated"],
                s.get("truncated", False),
            )
            for s in state["replay"]
        ]
        brain.cursor = state["cursor"]
        for target, layers in zip(brain.target_critics, state["target_critics"]):
            target.layers = packed(layers, precision_of(brain))
        return brain
