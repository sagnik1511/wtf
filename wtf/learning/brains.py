"""Algorithm dispatch; neural architecture belongs to each brain template."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


from wtf.native.numeric import Float32, packed, plain, precision_of


from wtf.learning.brain import Brain


def create_brain(config, actions, base_seed, individual_seed):
    """Construct a controller matching its algorithm and observation contract."""

    # Keep independent numeric/compatibility or benchmark paths explicit to preserve
    # their exact contracts.
    # pylint: disable=duplicate-code

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    if config["type"] == "q_network":
        return Brain(config, actions, base_seed, individual_seed)
    if config["type"] in (
        "recurrent_ppo",
        "qrdqn",
        "dqn",
        "ddqn",
        "maskable_ppo",
        "mappo",
        "discrete_sac",
        "trpo",
        "lstm_ppo",
    ):
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
    """Construct one of the supported extended policy implementations."""

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    from wtf.learning.contextual_ppo import MaskablePPOBrain, MAPPOBrain
    from wtf.learning.qrdqn import QRDQNBrain
    from wtf.learning.dqn import DQNBrain, DDQNBrain
    from wtf.learning.recurrent_ppo import RecurrentPPOBrain

    from wtf.learning.discrete_sac import DiscreteSACBrain
    from wtf.learning.trpo import TRPOBrain
    from wtf.learning.lstm_ppo import LSTMPPOBrain

    return {
        "discrete_sac": DiscreteSACBrain,
        "trpo": TRPOBrain,
        "lstm_ppo": LSTMPPOBrain,
        "recurrent_ppo": RecurrentPPOBrain,
        "qrdqn": QRDQNBrain,
        "dqn": DQNBrain,
        "ddqn": DDQNBrain,
        "maskable_ppo": MaskablePPOBrain,
        "mappo": MAPPOBrain,
    }[kind]


def restore_brain(state):
    """Dispatch a persisted controller to its matching restore implementation."""
    # Keep dispatch identifiers local rather than importing the archived-XAI whitelist.
    # pylint: disable=duplicate-code

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    kind = state["config"]["type"]
    if kind == "q_network":
        return Brain.restore(state)
    if kind in (
        "recurrent_ppo",
        "qrdqn",
        "dqn",
        "ddqn",
        "maskable_ppo",
        "mappo",
        "discrete_sac",
        "trpo",
        "lstm_ppo",
    ):
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

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

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
    """Derive a reproducible private random seed for an offspring controller."""
    return (world.seed << 32) ^ (child_id * 7919 + 13)


def model_signature(config, actions):
    """The exact input/action/architecture contract required for warm starts."""
    # Explicit branches keep each public algorithm/signature distinction visible.
    # pylint: disable=too-many-branches

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    from wtf.learning.brain import observation_inputs

    kind = config["type"]
    signature = {
        "observation_version": config.get("observation_version", 1),
        "inputs": observation_inputs(config),
        "algorithm": kind,
        "hidden_layers": list(
            config["hidden_layers"]
            if kind == "q_network"
            else config["network"]["hidden_layers"]
        ),
        "activation": "tanh",
        "actions": list(actions),
        "value_head": kind
        in ("ppo", "a2c", "recurrent_ppo", "maskable_ppo", "mappo", "trpo", "lstm_ppo"),
    }
    if config.get("observation_version", 1) == 7:
        from wtf.simulation.compact_observations import compact_layout

        signature["observation_actions"] = list(compact_layout(config).actions)
        signature["compact_layout_version"] = 1
        signature["movement_summary_version"] = 1
        signature["scent_summary_version"] = 1
    if "late_consequence" in config:
        from wtf.simulation.compact_observations import compact_layout

        signature["late_consequence_version"] = 1
        signature["late_consequence"] = {
            "max_adjustment": config["late_consequence"].get("max_adjustment", 2.0)
        }
        signature["policy_inputs"] = compact_layout(config).physical_width
    if config.get("observation_version", 1) == 6:
        from wtf.simulation.soft_observations import soft_contract

        signature["soft_categories"] = {
            key: list(value) for key, value in soft_contract(config).items()
        }
    if config.get("observation_top_k", 1) != 1:
        signature["observation_top_k"] = config["observation_top_k"]
    if config.get("movement_history_steps", 0):
        signature["movement_history_steps"] = config["movement_history_steps"]
    if "scent_observation" in config:
        from wtf.simulation.tracking import scent_contract

        signature["scent_observation"] = scent_contract(config)
    from wtf.simulation.social import social_contract

    if config.get("social_observation") is not None:
        signature["social_observation"] = social_contract(config)
    if "auxiliary_consequence" in config:
        signature["auxiliary_consequence_version"] = 1
    if config.get("consequence_observation", False):
        signature["consequence_observation"] = True
        if config.get("consequence_scope", "context") != "context":
            signature["consequence_scope"] = config["consequence_scope"]
    if kind in ("recurrent_ppo", "lstm_ppo"):
        signature.update(
            recurrent_size=config["recurrent_size"],
            recurrent_cell="lstm" if kind == "lstm_ppo" else "gru",
        )
    elif kind == "discrete_sac":
        signature.update(critic_count=2, critic_outputs=len(actions))
    elif kind == "trpo":
        signature.update(critic_separate=True)
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
    """Extract the model parameters used for inheritance or warm starts."""
    if brain.config["type"] == "q_network":
        parameters = plain(brain.weights)
    else:
        parameters = brain.network.checkpoint()
    model = {
        "signature": model_signature(brain.config, brain.actions),
        "precision": precision_of(brain),
        "parameters": parameters,
    }
    if hasattr(brain, "auxiliary"):
        model["auxiliary_parameters"] = brain.auxiliary.network.checkpoint()
    if hasattr(brain, "late"):
        model["late_parameters"] = brain.late.network.checkpoint()
    return model


# Validate and prepare every parameter block before committing any live weights.
# pylint: disable-next=too-many-branches
def load_model_weights(brain, model):
    """Load compatible finite weights into a freshly constructed controller.

    Optimizers, exploration settings, lifetime counters and experience storage
    belong to the new run. Q targets and GRPO references start at loaded weights.
    """

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

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
    if hasattr(brain, "auxiliary"):
        auxiliary_parameters = model.get("auxiliary_parameters")
        if not valid(auxiliary_parameters, brain.auxiliary.network.layers):
            raise ValueError("Saved brain has invalid auxiliary head parameters")
        auxiliary_parameters = packed(auxiliary_parameters, precision_of(brain))
    if hasattr(brain, "late"):
        if not valid(model.get("late_parameters"), brain.late.network.layers) or any(
            model["late_parameters"][0]["bias"]
        ):
            raise ValueError("Saved brain has invalid late-consequence parameters")
        late_parameters = packed(model["late_parameters"], precision_of(brain))
    # Convert all components before committing live parameters: finite JSON
    # values can still overflow FP32 storage. Failed imports leave the brain intact.
    prepared_parameters = packed(parameters, precision_of(brain))
    if brain.config["type"] == "q_network":
        brain.weights = prepared_parameters
        brain.target = packed(parameters, precision_of(brain))
    else:
        brain.network.layers = prepared_parameters
        if hasattr(brain, "target_network"):
            brain.target_network.layers = packed(parameters, precision_of(brain))
        if hasattr(brain, "reference_network"):
            brain.reference_network.layers = packed(parameters, precision_of(brain))
    if hasattr(brain, "auxiliary"):
        brain.auxiliary.network.layers = auxiliary_parameters
    if hasattr(brain, "late"):
        brain.late.network.layers = late_parameters

    if hasattr(brain, "synchronize_targets"):
        brain.synchronize_targets()


def reset_birth_training(brain):
    """New inherited parameters cannot reuse either parent's temporal state."""

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel
    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-branches

    import copy

    for key in ("replay", "rollout", "trajectory"):
        if hasattr(brain, key):
            getattr(brain, key).clear()
    for key in (
        "steps",
        "updates",
        "rollout_updates",
        "cursor",
        "last_group_step",
        "policy_updates",
    ):
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
    if hasattr(brain, "auxiliary"):
        auxiliary = brain.auxiliary
        auxiliary.moments = packed(auxiliary.network.zeros(), precision_of(brain))
        auxiliary.variances = packed(auxiliary.network.zeros(), precision_of(brain))
        auxiliary.updates = 0
    if hasattr(brain, "late"):
        brain.late.moments = packed(brain.late.network.zeros(), precision_of(brain))
        brain.late.variances = packed(brain.late.network.zeros(), precision_of(brain))
        brain.late.updates = 0
    if hasattr(brain, "reset_memory"):
        brain.reset_memory()
    if hasattr(brain, "target"):
        brain.target = copy.deepcopy(brain.weights)
    if hasattr(brain, "target_network"):
        brain.target_network.layers = copy.deepcopy(brain.network.layers)
    if hasattr(brain, "reference_network"):
        brain.reference_network.layers = copy.deepcopy(brain.network.layers)

    if hasattr(brain, "synchronize_targets"):
        brain.synchronize_targets()


def mix_parent_brains(mother, father, seed):
    """Uniform parameter crossover; independent child, RNGs and learning state.

    Copies both matrix weights and biases, choosing each scalar from one parent.
    Neurons in independently learned networks need not align; crossover is an
    experimental inheritance rule, not a guarantee of preserving performance.
    """

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

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

        # Explicit indices align the related packed arrays without introducing another
        # captured value.
        # pylint: disable=consider-using-enumerate

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
    if hasattr(child, "auxiliary"):
        cross(child.auxiliary.network.layers, father.auxiliary.network.layers)
    if hasattr(child, "late"):
        cross(child.late.network.layers, father.late.network.layers)
    reset_birth_training(child)
    return child
