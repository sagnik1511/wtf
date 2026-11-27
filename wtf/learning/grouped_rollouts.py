"""Matched GRPO trials: sequential world clones, read-only other policies."""

import random
from wtf.learning.policy_gradients import PolicyStep
from wtf.learning.ppo import policy_distribution
from wtf.simulation.world import World
from wtf.learning.brains import offspring_seed, mix_parent_brains


def sample_index(probabilities, rng):
    """Sample a categorical index using the supplied private random generator."""
    draw, cumulative = rng.random(), 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def other_action(
    brain,
    observation,
    rng,
    world=None,
    animal_id=None,
    memories=None,
    occupied=None,
    populations=None,
    mating_context=None,
    outputs=None,
):
    """Choose a nonfocal animal's action for an isolated trial rollout."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-positional-arguments
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    kind = brain.config["type"]
    if kind in ("recurrent_ppo", "lstm_ppo"):
        hidden = memories[animal_id]
        outputs, memories[animal_id] = brain.network.step(observation, hidden)
        return brain.actions[sample_index(policy_distribution(outputs[:-1])[0], rng)]
    if kind == "maskable_ppo":
        from wtf.learning.contextual_ppo import action_mask

        mask = action_mask(
            world,
            world.animals[animal_id],
            brain.actions,
            occupied,
            populations,
            mating_context,
        )
        outputs = brain.network.predict(observation) if outputs is None else outputs
        probabilities = brain.distribution(outputs, tuple(observation) + mask)[0]
        return brain.actions[sample_index(probabilities, rng)]
    if kind in ("q_network", "qrdqn", "dqn", "ddqn"):
        if rng.random() < brain.epsilon:
            return rng.choice(brain.actions)
        values = (
            brain.forward(observation)[1]
            if kind == "q_network"
            else brain.values(observation)
        )
        return brain.actions[max(range(len(values)), key=values.__getitem__)]
    if outputs is None:
        probabilities = brain.action_probabilities(observation)
    elif kind in ("ppo", "a2c"):
        probabilities = brain.distribution(outputs, observation)[0]
    else:
        probabilities = policy_distribution(outputs)[0]
    return brain.actions[sample_index(probabilities, rng)]


def collect_group(world, brains, focal_id, *, reuse_world=True):
    """Never call choose()/learn() on real controllers inside trial worlds.

    Environment and other-agent RNG streams restart identically per trial.
    Only the focal agent's sampled actions differ. No replay buffers are copied;
    policies copy only for rare mutated newborns. One cloned world is retained.
    """

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel
    # This internal collaborator intentionally inspects the owning component's private
    # state.
    # pylint: disable=protected-access

    focal = brains[focal_id]
    if focal_id not in world.animals:
        raise ValueError("GRPO focal agent must be alive")
    initial_world = world.checkpoint()
    # Historic deaths do not affect future physics or focal rewards. Preserve
    # next_animal_id, which is the authoritative identity counter in trials.
    initial_world["lifespans"] = []
    states = {i: brains[i].rng.getstate() for i in world.animals}
    horizon = min(
        focal.config["rollout_horizon"], world.config["world"]["max_steps"] - world.tick
    )
    if horizon < 1:
        raise ValueError("No simulation steps remain for a GRPO trial")
    from wtf.learning.batched import policy_plan, predict_policies

    initial_prediction_plan = policy_plan({i: brains[i] for i in world.animals})
    group = []
    has_masks = any(brains[i].config["type"] == "maskable_ppo" for i in world.animals)
    reusable_generators = {i: random.Random(0) for i in states if i != focal_id}
    trial = None
    initial_observations = None
    for _ in range(focal.config["group_size"]):
        if trial is not None and reuse_world:
            trial._reset_trial(initial_world)
        else:
            trial = (
                World.restore(initial_world, static_geography=world.geography)
                if world.geography
                else World.restore(initial_world)
            )
        generators = {}
        for i, generator in reusable_generators.items():
            generator.setstate(states[i])
            generators[i] = generator
        generators[focal_id] = random.Random(focal.group_rng.getrandbits(64))
        policies = {i: brains[i] for i in trial.animals}
        prediction_plan = initial_prediction_plan.copy()
        memories = {
            i: b.hidden[:]
            for i, b in policies.items()
            if b.config["type"] in ("recurrent_ppo", "lstm_ppo")
        }
        if initial_observations is None or not reuse_world:
            observations = trial.observe_all()
            if reuse_world:
                initial_observations = observations
                if trial._tracking is not None and trial._tracking.history_enabled:
                    # Seeing the initial frame updates personal visual memory.
                    # Cached observations must restart with that same memory,
                    # especially when the saved frame predates the trial tick.
                    initial_world["sensory_state"][
                        "visual"
                    ] = trial._tracking.visual.checkpoint()
        else:
            observations = initial_observations
        trajectory = []
        for step in range(horizon):
            observation = observations[focal_id]
            predictions = predict_policies(policies, observations, prediction_plan)
            outputs = predictions.get(focal_id)
            probabilities, logs = policy_distribution(
                focal.network.predict(observation) if outputs is None else outputs
            )
            index = sample_index(probabilities, generators[focal_id])
            mating_context = (
                getattr(trial, "_mate_context", None) if has_masks else None
            )
            if has_masks and mating_context is None:
                mating_context = trial._prepare_mating_context()
            occupied, populations = mating_context[:2] if has_masks else (None, None)
            actions = {
                i: (
                    focal.actions[index]
                    if i == focal_id
                    else other_action(
                        policies[i],
                        o,
                        generators[i],
                        trial,
                        i,
                        memories,
                        occupied,
                        populations,
                        mating_context,
                        predictions.get(i),
                    )
                )
                for i, o in observations.items()
            }
            del predictions, outputs
            result = trial.step(actions)
            mutation_events = []
            for event in result.events:
                if event.kind == "born":
                    # Crossover creates an independent frozen trial policy.
                    child_policy = mix_parent_brains(
                        policies[event.actor],
                        policies[event.mate],
                        offspring_seed(trial, event.target),
                    )
                    from wtf.learning.mutation import mutate_birth

                    policies[event.target], mutation_event = mutate_birth(
                        trial, event, policies[event.actor], child_policy
                    )
                    prediction_plan.update(
                        policy_plan({event.target: policies[event.target]})
                    )
                    if mutation_event is not None:
                        mutation_events.append(mutation_event)
                    if policies[event.target].config["type"] in (
                        "recurrent_ppo",
                        "lstm_ppo",
                    ):
                        # The policy owns the complete reset state: LSTM includes
                        # both hidden and cell vectors, unlike GRU's single vector.
                        memories[event.target] = policies[event.target].hidden[:]
                    generators[event.target] = random.Random(
                        offspring_seed(trial, event.target)
                    )
            result.events.extend(mutation_events)
            for animal_id in result.dead_ids:
                policies.pop(animal_id, None)
                prediction_plan.pop(animal_id, None)
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
        del observations, result, generators, policies, memories, prediction_plan
        if not reuse_world:
            trial = None
    return group
