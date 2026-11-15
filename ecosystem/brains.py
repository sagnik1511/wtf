"""Algorithm dispatch; neural architecture belongs to each brain template."""

from .brain import Brain


def create_brain(config, actions, base_seed, individual_seed):
    if config["type"] == "q_network":
        return Brain(config, actions, base_seed, individual_seed)
    if config["type"] == "ppo":
        from .ppo import PPOBrain

        return PPOBrain(config, actions, base_seed, individual_seed)
    if config["type"] in ("a2c", "reinforce", "grpo"):
        from .policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain

        return {"a2c": A2CBrain, "reinforce": ReinforceBrain, "grpo": GRPOBrain}[
            config["type"]
        ](config, actions, base_seed, individual_seed)
    raise ValueError(f'Unsupported brain type: {config["type"]}')


def restore_brain(state):
    kind = state["config"]["type"]
    if kind == "q_network":
        return Brain.restore(state)
    if kind == "ppo":
        from .ppo import PPOBrain

        return PPOBrain.restore(state)
    if kind in ("a2c", "reinforce", "grpo"):
        from .policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain

        return {"a2c": A2CBrain, "reinforce": ReinforceBrain, "grpo": GRPOBrain}[
            kind
        ].restore(state)
    raise ValueError(f"Unsupported checkpoint brain type: {kind}")


def inherit_brain(parent, seed):
    """Copy learned state without copying a parent's lifetime experiences.

    Retain policy/value/target networks, optimizer moments and their counters,
    exploration schedule and learned baselines. Newborns have independent RNGs
    and empty experience buffers. Filter before copying to keep birth memory
    proportional to model size, not replay/episode capacity.
    """
    import copy
    import random

    buffers = ("replay", "rollout", "trajectory")
    state = {
        k: v
        for k, v in vars(parent).items()
        if k not in buffers + ("pending", "rng", "training_rng", "group_rng")
    }
    child = type(parent).__new__(type(parent))
    child.__dict__.update(copy.deepcopy(state))
    for key in buffers:
        if hasattr(parent, key):
            setattr(child, key, [])
    if hasattr(parent, "cursor"):
        child.cursor = 0
    if hasattr(parent, "pending"):
        child.pending = None
    child.rng = random.Random(seed)
    child.training_rng = random.Random(seed ^ 0x5F3759DF)
    if hasattr(parent, "group_rng"):
        child.group_rng = random.Random(seed ^ 0x0BADC0DE)
        child.last_group_step = child.steps
    return child


def offspring_seed(world, child_id):
    # Stable across checkpoints and matched trials; never consume parent RNGs.
    return (world.seed << 32) ^ (child_id * 7919 + 13)
