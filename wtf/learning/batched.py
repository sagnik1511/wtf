"""Inference-only batching; sampling and pending transitions remain per animal."""

from types import FunctionType
from wtf.native import kernels
from wtf.native.numeric import Float32
from wtf.learning.ppo import MLP, PPOBrain
from wtf.learning.policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain
from wtf.learning.contextual_ppo import (
    LocalMLP,
    MaskablePPOBrain,
    MAPPOBrain,
    CentralNetwork,
)

KINDS = (PPOBrain, A2CBrain, ReinforceBrain, GRPOBrain, MaskablePPOBrain, MAPPOBrain)


def choose_actions(
    brains, observations, explore=True, random_baseline=False, allow_batch=True
):
    if not allow_batch or random_baseline or not kernels.available("mlp_predict_many"):
        return {
            i: brains[i].choose(o, explore=explore, random_baseline=random_baseline)
            for i, o in observations.items()
        }
    prepared, jobs = {}, []
    for identity, observation in observations.items():
        brain = brains[identity]
        if (
            type(brain) not in KINDS
            or "choose" in vars(brain)
            or not isinstance(type(brain).choose, FunctionType)
            or type(brain).choose.__module__
            not in (
                "wtf.learning.ppo",
                "wtf.learning.policy_gradients",
                "wtf.learning.contextual_ppo",
            )
        ):
            continue
        network = brain.network
        if (
            type(network) not in (MLP, LocalMLP, CentralNetwork)
            or "predict" in vars(network)
            or not isinstance(type(network).predict, FunctionType)
            or type(network).predict.__module__
            not in ("wtf.learning.ppo", "wtf.learning.contextual_ppo")
        ):
            continue
        if not isinstance(network.layers[0]["weights"], Float32):
            continue
        kind = type(brain)
        if kind is MAPPOBrain:
            if (
                type(network.actor) is not MLP
                or type(network.critic) is not MLP
                or "predict" in vars(network.actor)
                or "predict" in vars(network.critic)
                or (not explore and "action_probabilities" in vars(brain))
            ):
                continue
            packed = tuple(observation) + brain.context if explore else observation
            index = len(jobs)
            jobs.append((network.actor.layers, tuple(observation)))
            if explore:
                jobs.append((network.critic.layers, packed))
            prepared[identity] = packed, index, 2 if explore else 0
        else:
            packed = (
                brain._pack(observation) if kind is MaskablePPOBrain else observation
            )
            prepared[identity] = packed, len(jobs), 1
            jobs.append((network.layers, tuple(packed[: network.sizes[0]])))
    if len(prepared) < 2:
        return {
            i: brains[i].choose(o, explore=explore, random_baseline=False)
            for i, o in observations.items()
        }
    outputs = kernels.native.mlp_predict_many(jobs)
    actions = {}
    for identity, observation in observations.items():
        brain = brains[identity]
        if identity not in prepared:
            actions[identity] = brain.choose(
                observation, explore=explore, random_baseline=False
            )
            continue
        packed, index, parts = prepared[identity]
        values = outputs[index]
        if parts == 2:
            values = values + outputs[index + 1]
        elif parts == 0:
            values = values + [0.0]  # Evaluation's centralized critic is unused.
        actions[identity] = brain._choose_outputs(packed, values, explore)
    return actions


def policy_plan(brains):
    """Read-only model qualification for one frozen GRPO policy set."""
    if not kernels.available("mlp_predict_many"):
        return {}
    plan = {}
    for identity, brain in brains.items():
        if type(brain) not in KINDS or "action_probabilities" in vars(brain):
            continue
        network = brain.network.actor if type(brain) is MAPPOBrain else brain.network
        if (
            type(network) not in (MLP, LocalMLP)
            or "predict" in vars(network)
            or not isinstance(type(network).predict, FunctionType)
            or type(network).predict.__module__
            not in ("wtf.learning.ppo", "wtf.learning.contextual_ppo")
        ):
            continue
        if isinstance(network.layers[0]["weights"], Float32):
            plan[identity] = network
    return plan


def predict_policies(brains, observations, plan=None):
    """Frozen outputs only; the group owns qualification and birth/death updates."""
    if plan is None:
        plan = policy_plan(brains)
    if len(plan) < 2:
        return {}
    identities = list(plan)
    jobs = [
        (network.layers, tuple(observations[i][: network.sizes[0]]))
        for i, network in plan.items()
    ]
    outputs = kernels.native.mlp_predict_many(jobs)
    return dict(zip(identities, outputs))
