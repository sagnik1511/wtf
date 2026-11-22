"""Categorical TRPO with an exact Fisher-vector product and separate value fit.

Conjugate gradients solve the damped natural-gradient system; backtracking tests
actual stored candidate weights against the empirical forward KL constraint.
"""

import math
from wtf.learning.brain import observation_inputs
from wtf.learning.ppo import PPOBrain, advantages_and_returns, policy_distribution
from wtf.learning.policy_gradients import normalized
from wtf.learning.policy_networks import PolicyBundle


def flatten(layers):
    return [float(v) for layer in layers for values in layer.values() for v in values]


def assign(layers, vector):
    cursor = 0
    for layer in layers:
        for values in layer.values():
            for i in range(len(values)):
                values[i] = vector[cursor]
                cursor += 1


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def conjugate_gradient(product, target, iterations, tolerance):
    x = [0.0] * len(target)
    residual = list(target)
    direction = list(residual)
    rr = dot(residual, residual)
    for _ in range(iterations):
        if rr <= tolerance:
            break
        applied = product(direction)
        denominator = dot(direction, applied)
        if denominator <= 0 or not math.isfinite(denominator):
            break
        scale = rr / denominator
        x = [a + scale * b for a, b in zip(x, direction)]
        residual = [a - scale * b for a, b in zip(residual, applied)]
        following = dot(residual, residual)
        direction = [a + (following / rr) * b for a, b in zip(residual, direction)]
        rr = following
    return x


def split_direction(network, vector):
    """One FVP-local partition, shared by every observation in that product."""
    cursor = 0
    result = []
    for layer in network.layers:
        weights = len(layer["weights"])
        bias = len(layer["bias"])
        result.append(
            (
                vector[cursor : cursor + weights],
                vector[cursor + weights : cursor + weights + bias],
            )
        )
        cursor += weights + bias
    return result


def logit_tangent(network, tape, vector, directions=None):
    """J v, without perturbing/rounding live FP32 policy parameters."""
    if directions is None:
        directions = split_direction(network, vector)
    tangent = [0.0] * len(tape[0])
    for layer_index, (layer, (dw, db)) in enumerate(zip(network.layers, directions)):
        x = tape[layer_index]
        width = len(x)
        out = []
        weights = layer["weights"]
        n = len(layer["bias"])
        for j in range(n):
            value = db[j] + sum(
                dw[j * width + i] * x[i] + weights[j * width + i] * tangent[i]
                for i in range(width)
            )
            out.append(
                value * (1 - tape[layer_index + 1][j] ** 2)
                if layer_index < len(network.layers) - 1
                else value
            )
        tangent = out
    return tangent


def fisher_product(network, tapes, probabilities, vector, damping):
    gradients = network.zeros()
    directions = split_direction(network, vector)
    for tape, p in zip(tapes, probabilities):
        u = logit_tangent(network, tape, vector, directions)
        mean = dot(p, u)
        network.accumulate_gradient(
            tape, [v * (t - mean) for v, t in zip(p, u)], gradients
        )
    return [g / len(tapes) + damping * v for g, v in zip(flatten(gradients), vector)]


def surrogate(
    logits,
    action,
    old_logp,
    advantage,
    entropy_coefficient,
    distribution=None,
    with_gradient=True,
):
    p, logs = policy_distribution(logits) if distribution is None else distribution
    ratio = math.exp(max(-60.0, min(60.0, logs[action] - old_logp)))
    entropy = -dot(p, logs)
    gradient = (
        [
            advantage * ratio * ((1.0 if i == action else 0.0) - v)
            - entropy_coefficient * v * (logs[i] + entropy)
            for i, v in enumerate(p)
        ]
        if with_gradient
        else None
    )
    return ratio * advantage + entropy_coefficient * entropy, gradient


class TRPOBrain(PPOBrain):
    algorithm = "trpo"

    def __init__(self, config, actions, base_seed, individual_seed):
        super().__init__(config, actions, base_seed, individual_seed)
        self.policy_updates = 0

    def _new_network(self, config, actions, seed):
        return PolicyBundle(config, actions, seed, observation_inputs(config))

    def _update(self):
        c = self.config
        actor = self.network.actor
        critic = self.network.critics[0]
        block = self.network.block
        advantages, targets = advantages_and_returns(
            self.rollout, c["discount_factor"], c["gae_lambda"]
        )
        if c["normalize_advantages"]:
            advantages = normalized(advantages, inplace=True)
        tapes = []
        probabilities = []
        old_logs = []
        gradient = actor.zeros()
        before_objective = 0.0
        for step, adv in zip(self.rollout, advantages):
            logits, tape = actor.forward(step.observation)
            p, logs = policy_distribution(logits)
            tapes.append(tape)
            probabilities.append(p)
            old_logs.append(logs)
            objective, derivative = surrogate(
                logits,
                step.action,
                step.log_probability,
                adv,
                c["entropy_coefficient"],
                distribution=(p, logs),
            )
            actor.accumulate_gradient(tape, derivative, gradient)
            before_objective += objective
        count = len(self.rollout)
        before_objective /= count
        g = [v / count for v in flatten(gradient)]
        product = lambda vector: fisher_product(
            actor, tapes, probabilities, vector, c["cg_damping"]
        )
        direction = conjugate_gradient(
            product, g, c["cg_iterations"], c["cg_tolerance"]
        )
        curvature = dot(direction, product(direction))
        base = flatten(actor.layers)
        accepted = False
        kl = 0.0
        improvement = 0.0
        if curvature > 0 and math.isfinite(curvature):
            scale = math.sqrt(2 * c["max_kl"] / curvature)
            full = [scale * v for v in direction]
            expected = dot(g, full)
            for n in range(c["backtrack_steps"]):
                fraction = c["backtrack_coefficient"] ** n
                assign(actor.layers, [a + fraction * b for a, b in zip(base, full)])
                objective = 0.0
                kl = 0.0
                for step, adv, p, logs in zip(
                    self.rollout, advantages, probabilities, old_logs
                ):
                    outputs = actor.predict(step.observation)
                    candidate_distribution = policy_distribution(outputs)
                    following = candidate_distribution[1]
                    objective += surrogate(
                        outputs,
                        step.action,
                        step.log_probability,
                        adv,
                        c["entropy_coefficient"],
                        distribution=candidate_distribution,
                        with_gradient=False,
                    )[0]
                    kl += sum(
                        prob * (lp - new) for prob, lp, new in zip(p, logs, following)
                    )
                kl = max(0.0, kl / count)
                improvement = objective / count - before_objective
                if (
                    math.isfinite(improvement)
                    and math.isfinite(kl)
                    and kl <= c["max_kl"]
                    and improvement > 0
                    and improvement >= c["accept_ratio"] * fraction * expected
                ):
                    accepted = True
                    self.policy_updates += 1
                    break
            if not accepted:
                assign(actor.layers, base)
                kl = 0.0
                improvement = 0.0
        loss = 0.0
        indices = list(range(count))
        for _ in range(c["value_epochs"]):
            self.training_rng.shuffle(indices)
            loss = 0.0
            for offset in range(0, count, c["batch_size"]):
                batch = indices[offset : offset + c["batch_size"]]
                gradients = self.network.zeros()
                for i in batch:
                    step = self.rollout[i]
                    target = targets[i]
                    outputs, tape = critic.forward(step.observation)
                    error = outputs[0] - target
                    critic.accumulate_gradient(
                        tape, [c["value_coefficient"] * error], gradients[block:]
                    )
                    loss += 0.5 * error * error
                self._adam_step(gradients, len(batch))
        self.last_metrics = dict(
            empirical_kl=kl,
            policy_accepted=accepted,
            surrogate_improvement=improvement,
            value_loss=loss / count,
        )
        self.rollout_updates += 1
        self.rollout.clear()

    def diagnostics(self, observation, include_activations=False):
        info = super().diagnostics(observation, include_activations)
        info.update(critic_separate=True, policy_updates=self.policy_updates)
        return info

    def checkpoint(self):
        return dict(super().checkpoint(), policy_updates=self.policy_updates)

    @classmethod
    def restore(cls, state):
        brain = super().restore(state)
        brain.policy_updates = state["policy_updates"]
        return brain
