"""Algorithm dispatch; neural architecture belongs to each brain template."""

from wtf.native.numeric import Float32, packed, plain, precision_of


from wtf.learning.brain import Brain


def create_brain(config, actions, base_seed, individual_seed):
    if config["type"] == "q_network":
        return Brain(config, actions, base_seed, individual_seed)
    if config["type"] in ("recurrent_ppo", "qrdqn", "maskable_ppo", "mappo"):
        return advanced_brain(config["type"])(
            config, actions, base_seed, individual_seed
        )
    if config["type"] == "ppo":
        from wtf.learning.ppo import PPOBrain

        return PPOBrain(config, actions, base_seed, individual_seed)
    if config["type"] in ("a2c", "reinforce", "grpo"):
        from wtf.learning.policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain

        return {"a2c": A2CBrain, "reinforce": ReinforceBrain, "grpo": GRPOBrain}[
            config["type"]
        ](config, actions, base_seed, individual_seed)
    raise ValueError(f'Unsupported brain type: {config["type"]}')


def advanced_brain(kind):
    from wtf.learning.contextual_ppo import MaskablePPOBrain, MAPPOBrain
    from wtf.learning.qrdqn import QRDQNBrain
    from wtf.learning.recurrent_ppo import RecurrentPPOBrain

    return {
        "recurrent_ppo": RecurrentPPOBrain,
        "qrdqn": QRDQNBrain,
        "maskable_ppo": MaskablePPOBrain,
        "mappo": MAPPOBrain,
    }[kind]


def restore_brain(state):
    kind = state["config"]["type"]
    if kind == "q_network":
        return Brain.restore(state)
    if kind in ("recurrent_ppo", "qrdqn", "maskable_ppo", "mappo"):
        return advanced_brain(kind).restore(state)
    if kind == "ppo":
        from wtf.learning.ppo import PPOBrain

        return PPOBrain.restore(state)
    if kind in ("a2c", "reinforce", "grpo"):
        from wtf.learning.policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain

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
    if hasattr(child, "reset_memory"):
        child.reset_memory()
    return child


def offspring_seed(world, child_id):
    # Stable across checkpoints and matched trials; never consume parent RNGs.
    return (world.seed << 32) ^ (child_id * 7919 + 13)


def model_signature(config, actions):
    """The exact input/action/architecture contract required for warm starts."""
    from wtf.learning.brain import observation_inputs

    kind = config["type"]
    signature = dict(
        observation_version=config.get("observation_version", 1),
        inputs=observation_inputs(config),
        algorithm=kind,
        hidden_layers=list(
            config["hidden_layers"]
            if kind == "q_network"
            else config["network"]["hidden_layers"]
        ),
        activation="tanh",
        actions=list(actions),
        value_head=kind in ("ppo", "a2c", "recurrent_ppo", "maskable_ppo", "mappo"),
    )
    if config.get("observation_top_k", 1) != 1:
        signature["observation_top_k"] = config["observation_top_k"]
    if kind == "recurrent_ppo":
        signature.update(recurrent_size=config["recurrent_size"], recurrent_cell="gru")
    elif kind == "qrdqn":
        signature["num_quantiles"] = config["num_quantiles"]
    elif kind == "mappo":
        signature.update(
            critic_hidden_layers=list(config["critic_hidden_layers"]),
            centralized_context_version=1,
            centralized_inputs=16,
        )
    elif kind == "maskable_ppo":
        signature["action_mask_version"] = 1
    return signature


def model_snapshot(brain):
    if brain.config["type"] == "q_network":
        parameters = plain(brain.weights)
    else:
        parameters = brain.network.checkpoint()
    return dict(
        signature=model_signature(brain.config, brain.actions),
        precision=precision_of(brain),
        parameters=parameters,
    )


def load_model_weights(brain, model):
    """Load compatible finite weights into a freshly constructed controller.

    Optimizers, exploration settings, lifetime counters and experience storage
    belong to the new run. Q targets and GRPO references start at loaded weights.
    """
    import math

    if model.get("signature") != model_signature(brain.config, brain.actions):
        raise ValueError(
            "Saved brain algorithm, architecture, observation version or action order is incompatible"
        )
    parameters = model.get("parameters")
    expected = (
        brain.weights if brain.config["type"] == "q_network" else brain.network.layers
    )

    def valid(actual, template):
        if isinstance(template, dict):
            return (
                isinstance(actual, dict)
                and actual.keys() == template.keys()
                and all(valid(actual[k], v) for k, v in template.items())
            )
        if isinstance(template, (list, Float32)):
            return (
                isinstance(actual, list)
                and len(actual) == len(template)
                and all(valid(a, b) for a, b in zip(actual, template))
            )
        return type(actual) in (int, float) and math.isfinite(actual)

    if not valid(parameters, expected):
        raise ValueError(
            "Saved brain has invalid parameter dimensions or non-finite weights"
        )
    if brain.config["type"] == "q_network":
        brain.weights = packed(parameters, precision_of(brain))
        brain.target = packed(parameters, precision_of(brain))
    else:
        brain.network.layers = packed(parameters, precision_of(brain))
        if hasattr(brain, "target_network"):
            brain.target_network.layers = packed(parameters, precision_of(brain))
        if hasattr(brain, "reference_network"):
            brain.reference_network.layers = packed(parameters, precision_of(brain))


def reset_birth_training(brain):
    """New inherited parameters cannot reuse either parent's temporal state."""
    import copy

    for key in ("replay", "rollout", "trajectory"):
        if hasattr(brain, key):
            getattr(brain, key).clear()
    for key in ("steps", "updates", "rollout_updates", "cursor", "last_group_step"):
        if hasattr(brain, key):
            setattr(brain, key, 0)
    if hasattr(brain, "pending"):
        brain.pending = None
    if hasattr(brain, "last_metrics"):
        brain.last_metrics = {}
    if hasattr(brain, "baseline"):
        brain.baseline = 0.0
    if hasattr(brain, "moments"):
        brain.moments = packed(brain.network.zeros(), precision_of(brain))
        brain.variances = packed(brain.network.zeros(), precision_of(brain))
    if hasattr(brain, "reset_memory"):
        brain.reset_memory()
    if hasattr(brain, "target"):
        brain.target = copy.deepcopy(brain.weights)
    if hasattr(brain, "target_network"):
        brain.target_network.layers = copy.deepcopy(brain.network.layers)
    if hasattr(brain, "reference_network"):
        brain.reference_network.layers = copy.deepcopy(brain.network.layers)


def mix_parent_brains(mother, father, seed):
    """Uniform parameter crossover; independent child, RNGs and learning state.

    Copies both matrix weights and biases, choosing each scalar from one parent.
    Neurons in independently learned networks need not align; crossover is an
    experimental inheritance rule, not a guarantee of preserving performance.
    """
    import random

    if model_signature(mother.config, mother.actions) != model_signature(
        father.config, father.actions
    ):
        raise ValueError("Parents must have compatible brain architectures and actions")
    child = inherit_brain(mother, seed)
    left = (
        child.weights if child.config["type"] == "q_network" else child.network.layers
    )
    right = (
        father.weights
        if father.config["type"] == "q_network"
        else father.network.layers
    )
    rng = random.Random(seed ^ 0x43524F53534F564552)

    def cross(a, b):
        if isinstance(a, dict):
            for key in a:
                cross(a[key], b[key])
        elif a and isinstance(a[0], dict):
            for x, y in zip(a, b):
                cross(x, y)
        else:
            for index in range(len(a)):
                if rng.getrandbits(1):
                    a[index] = b[index]

    cross(left, right)
    reset_birth_training(child)
    return child
