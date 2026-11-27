"""Verify installed native PPO execution and Python querying outside a checkout."""

import json

import wtf
from wtf import load_config
from wtf.experiment.runner import Experiment
from wtf.experiment.native import NativeExperiment


def main():
    """Compare a small matched installed run without creating result artifacts."""
    for name, expected in (
        ("Experiment", Experiment),
        ("NativeExperiment", NativeExperiment),
    ):
        assert getattr(wtf, name) is expected
    config = load_config("ppo")
    config["seed"] = 100
    config["world"]["max_steps"] = 16
    config["experiment"].update(runs=1, world_seed_policy="fixed")
    reference = Experiment(config)
    native = NativeExperiment.from_experiment(reference)
    native.run()
    while not reference.done:
        reference.step()
    expected = json.dumps(reference.checkpoint(), sort_keys=True)
    if json.dumps(native.checkpoint(), sort_keys=True) != expected:
        raise AssertionError("Installed C++ episode diverged from the Python reference")
    if json.dumps(native.query().checkpoint(), sort_keys=True) != expected:
        raise AssertionError(
            "Installed native checkpoint did not restore through Python"
        )
    print("Installed C++ PPO loop and detached Python queries match exactly.")


if __name__ == "__main__":
    main()
