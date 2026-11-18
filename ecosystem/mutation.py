"""Rare sparse newborn mutations, independent of physical/action RNG streams."""

import bisect
import math
import random
from .numeric import Float32

from .brains import inherit_brain, offspring_seed, reset_birth_training
from .world import Event

MUTATION_SEED_SALT = 0x4D55544154494F4E


def mutation_trial(settings, seed):
    """A dedicated draw per birth; disabled/zero settings consume no randomness."""
    if (
        not settings
        or not settings["enabled"]
        or settings["probability_per_birth"] <= 0
        or settings["noise_std"] <= 0
    ):
        return None
    rng = random.Random(seed ^ MUTATION_SEED_SALT)
    return rng if rng.random() < settings["probability_per_birth"] else None


def mutate_weights(brain, settings, rng):
    """Perturb 1..max distinct matrix weights, excluding biases; return changes.

    Select flat indices without building a list of every parameter. Work outside
    rare successes is limited to a probability draw, not a model traversal.
    """
    if brain.config["type"] == "q_network":
        arrays = [brain.weights[k] for k in ("w1", "w2")]
    else:
        arrays = [
            values
            for layer in brain.network.layers
            for key, values in layer.items()
            if key in ("weights", "input_weights", "recurrent_weights")
        ]
    ends = []
    total = 0
    for values in arrays:
        total += len(values)
        ends.append(total)
    if not total:
        return 0
    count = rng.randint(1, min(settings["max_changed_weights"], total))
    changes = []
    for index in rng.sample(range(total), count):
        layer = bisect.bisect_right(ends, index)
        offset = index - (ends[layer - 1] if layer else 0)
        before = arrays[layer][offset]
        after = before + rng.gauss(0.0, settings["noise_std"])
        if isinstance(arrays[layer], Float32):
            after = Float32([after])[0]
        if not math.isfinite(after):
            raise ValueError("Mutation produced a non-finite weight")
        if after != before:
            changes.append((arrays[layer], offset, after))
    if not changes:
        return 0
    for values, index, value in changes:
        values[index] = value
    reset_birth_training(brain)
    return len(changes)


def mutate_birth(world, birth, parent_brain, child_brain=None):
    """Return a newborn policy/event; callers provide the crossed child policy."""
    child = world.animals[birth.target]
    settings = world.config["species"][child.species].get("mutation")
    seed = offspring_seed(world, birth.target)
    rng = mutation_trial(settings, seed)
    policy = parent_brain if child_brain is None else child_brain
    if rng is None:
        return policy, None
    if child_brain is None:
        policy = inherit_brain(parent_brain, seed)
    changed = mutate_weights(policy, settings, rng)
    if not changed:
        return policy, None
    child.mutated = True
    child.mutation_weights = changed
    child.mutation_count += 1
    world.mutations[child.species] += 1
    return policy, Event("mutated", birth.actor, birth.target, changed, "weights")
