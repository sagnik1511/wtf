"""Small explicit fixtures for newly added generic policy implementations."""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-function-docstring


from tests.test_ppo import ppo_settings
from tests.test_simulation import config

KINDS = ("discrete_sac", "trpo", "lstm_ppo")


def settings(kind):
    c = ppo_settings()
    c["type"] = kind
    if kind == "lstm_ppo":
        c.update(recurrent_size=2, sequence_length=2)
    elif kind == "discrete_sac":
        c = {
            key: c[key]
            for key in (
                "type",
                "network",
                "learning_rate",
                "discount_factor",
                "max_grad_norm",
            )
        }
        c.update(
            batch_size=2,
            replay_capacity=8,
            learning_starts_after=2,
            train_every_steps=1,
            temperature=0.1,
            target_tau=0.05,
        )
    elif kind == "trpo":
        for key in ("clip_ratio", "epochs", "target_kl"):
            c.pop(key)
        c.update(
            value_epochs=2,
            max_kl=0.01,
            cg_iterations=4,
            cg_damping=0.1,
            cg_tolerance=1e-10,
            backtrack_steps=6,
            backtrack_coefficient=0.5,
            accept_ratio=0.1,
        )
    return c


def environment(kind):
    c = config()
    c["brains"] = {"new_policy": settings(kind)}
    for s in c["species"].values():
        s["brain"] = "new_policy"
    c["world"]["max_steps"] = 9
    c["experiment"]["runs"] = 2
    return c
