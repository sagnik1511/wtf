"""Matched GRPO trials: sequential world clones, read-only other policies."""
import random
from .policy_gradients import PolicyStep
from .ppo import policy_distribution
from .world import World
from .brains import offspring_seed


def sample_index(probabilities, rng):
    draw, cumulative = rng.random(), 0.
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def other_action(brain, observation, rng):
    if brain.config['type'] == 'q_network':
        if rng.random() < brain.epsilon:
            return rng.choice(brain.actions)
        values = brain.forward(observation)[1]
        return brain.actions[max(range(len(values)), key=values.__getitem__)]
    return brain.actions[sample_index(brain.action_probabilities(observation), rng)]


def collect_group(world, brains, focal_id):
    """Never call choose()/learn() on real controllers inside trial worlds.

    Environment and other-agent RNG streams restart identically per trial.
    Only the focal agent's sampled actions differ. No replay/model copies and
    no more than one cloned world are retained at once.
    """
    focal = brains[focal_id]
    if focal_id not in world.animals:
        raise ValueError('GRPO focal agent must be alive')
    initial_world = world.checkpoint()
    states = {i: brains[i].rng.getstate() for i in world.animals}
    horizon = min(focal.config['rollout_horizon'], world.config['world']['max_steps'] - world.tick)
    if horizon < 1:
        raise ValueError('No simulation steps remain for a GRPO trial')
    group = []
    for _ in range(focal.config['group_size']):
        trial = World.restore(initial_world)
        generators = {}
        for i, state in states.items():
            generators[i] = random.Random(0)
            generators[i].setstate(state)
        generators[focal_id] = random.Random(focal.group_rng.getrandbits(64))
        policies = {i: brains[i] for i in trial.animals}
        observations = trial.observe_all()
        trajectory = []
        for step in range(horizon):
            observation = observations[focal_id]
            probabilities, logs = policy_distribution(focal.network.forward(observation)[0])
            index = sample_index(probabilities, generators[focal_id])
            actions = {i: focal.actions[index] if i == focal_id else other_action(policies[i], o, generators[i]) for i, o in observations.items()}
            result = trial.step(actions)
            for event in result.events:
                if event.kind == 'born':
                    # Trials never learn: descendants can read the same frozen
                    # parent policy with their own RNG, avoiding model copies.
                    policies[event.target] = policies[event.actor]
                    generators[event.target] = random.Random(offspring_seed(trial, event.target))
            for animal_id in result.dead_ids:
                policies.pop(animal_id, None)
                generators.pop(animal_id, None)
            died = focal_id in result.dead_ids
            trajectory.append(PolicyStep(observation, index, result.rewards[focal_id], logs[index], died, not died and step + 1 == horizon))
            if died:
                break
            observations = result.observations
        group.append(trajectory)
        del trial, observations, result, generators, policies
    return group
