"""Matched GRPO trials: sequential world clones, read-only other policies."""

import random
from .policy_gradients import PolicyStep
from .ppo import policy_distribution
from .world import World
from .brains import offspring_seed, mix_parent_brains


def sample_index(probabilities, rng):
    draw, cumulative = rng.random(), 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def other_action(brain, observation, rng, world=None, animal_id=None, memories=None, occupied=None):
    kind = brain.config['type']
    if kind == 'recurrent_ppo':
        from .ppo import policy_distribution
        hidden = memories[animal_id]
        outputs, memories[animal_id] = brain.network.step(observation, hidden)
        return brain.actions[sample_index(policy_distribution(outputs[:-1])[0], rng)]
    if kind == 'maskable_ppo':
        from .contextual_ppo import action_mask
        mask = action_mask(world, world.animals[animal_id], brain.actions, occupied)
        outputs = brain.network.predict(observation)
        probabilities = brain.distribution(outputs, tuple(observation) + mask)[0]
        return brain.actions[sample_index(probabilities, rng)]
    if kind in ('q_network', 'qrdqn'):
        if rng.random() < brain.epsilon:
            return rng.choice(brain.actions)
        values = brain.forward(observation)[1] if kind == "q_network" else brain.values(observation)
        return brain.actions[max(range(len(values)), key=values.__getitem__)]
    return brain.actions[sample_index(brain.action_probabilities(observation), rng)]


def collect_group(world, brains, focal_id):
    """Never call choose()/learn() on real controllers inside trial worlds.

    Environment and other-agent RNG streams restart identically per trial.
    Only the focal agent's sampled actions differ. No replay buffers are copied;
    policies copy only for rare mutated newborns. One cloned world is retained.
    """
    focal = brains[focal_id]
    if focal_id not in world.animals:
        raise ValueError("GRPO focal agent must be alive")
    initial_world = world.checkpoint()
    # Historic deaths do not affect future physics or focal rewards. Preserve
    # next_animal_id, which is the authoritative identity counter in trials.
    initial_world['lifespans'] = []
    states = {i: brains[i].rng.getstate() for i in world.animals}
    horizon = min(
        focal.config["rollout_horizon"], world.config["world"]["max_steps"] - world.tick
    )
    if horizon < 1:
        raise ValueError("No simulation steps remain for a GRPO trial")
    group = []
    has_masks = any(brains[i].config['type']=='maskable_ppo' for i in world.animals)
    for _ in range(focal.config["group_size"]):
        trial = World.restore(initial_world)
        generators = {}
        for i, state in states.items():
            generators[i] = random.Random(0)
            generators[i].setstate(state)
        generators[focal_id] = random.Random(focal.group_rng.getrandbits(64))
        policies = {i: brains[i] for i in trial.animals}
        memories = {i: b.hidden[:] for i, b in policies.items() if b.config['type']=='recurrent_ppo'}
        observations = trial.observe_all()
        trajectory = []
        for step in range(horizon):
            observation = observations[focal_id]
            probabilities, logs = policy_distribution(
                focal.network.predict(observation)
            )
            index = sample_index(probabilities, generators[focal_id])
            occupied = {a.position: a for a in trial.animals.values()} if has_masks else None
            actions = {
                i: (
                    focal.actions[index]
                    if i == focal_id
                    else other_action(policies[i], o, generators[i], trial, i, memories, occupied)
                )
                for i, o in observations.items()
            }
            result = trial.step(actions)
            mutation_events = []
            for event in result.events:
                if event.kind == "born":
                    # Crossover creates an independent frozen trial policy.
                    child_policy = mix_parent_brains(policies[event.actor], policies[event.mate],
                                                     offspring_seed(trial, event.target))
                    from .mutation import mutate_birth
                    policies[event.target], mutation_event = mutate_birth(
                        trial,event,policies[event.actor],child_policy)
                    if mutation_event is not None:mutation_events.append(mutation_event)
                    if policies[event.target].config["type"] == "recurrent_ppo":
                        memories[event.target] = [0.0] * policies[event.target].config["recurrent_size"]
                    generators[event.target] = random.Random(
                        offspring_seed(trial, event.target)
                    )
            result.events.extend(mutation_events)
            for animal_id in result.dead_ids:
                policies.pop(animal_id, None)
                generators.pop(animal_id, None)
                memories.pop(animal_id, None)
            died = focal_id in result.dead_ids
            trajectory.append(
                PolicyStep(
                    observation,
                    index,
                    result.rewards[focal_id],
                    logs[index],
                    died,
                    not died and step + 1 == horizon,
                )
            )
            if died:
                break
            observations = result.observations
        group.append(trajectory)
        del trial, observations, result, generators, policies, memories
    return group
