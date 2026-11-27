"""Shared optional-engine loading and exact native-state comparison encoding."""

import json

__all__ = ["_engine", "encoded", "NATIVE_KINDS"]

try:
    from wtf.native import _engine
except ImportError:
    _engine = None


def encoded(value):
    """Retain integer/float types and signed zeros when comparing full states."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


# An independent explicit expectation: new Python policies do not silently expand
# the native test matrix without a completed native implementation.
NATIVE_KINDS = (
    "q_network",
    "dqn",
    "ddqn",
    "qrdqn",
    "discrete_sac",
    "ppo",
    "a2c",
    "reinforce",
    "grpo",
    "trpo",
    "recurrent_ppo",
    "lstm_ppo",
    "maskable_ppo",
    "mappo",
)
