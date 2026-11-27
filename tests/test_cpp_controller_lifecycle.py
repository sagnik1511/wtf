"""Complete native-controller lifecycles preserve weights and private state."""

# Fixtures intentionally inspect the optional native ownership boundary.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# Compiled parity probes are optional and resolved at runtime.
# pylint: disable=c-extension-no-member

import unittest

from tests.native_core_support import _engine, encoded, NATIVE_KINDS
from tests.test_cpp_actor_policies import settings as actor_settings
from tests.test_cpp_value_policies import settings as value_settings
from tests.test_advanced_algorithms import settings as contextual_settings
from tests.test_ppo import ppo_settings
from tests.test_simulation import config as world_config
from wtf.experiment.runner import Experiment
from wtf.experiment.native import NativeExperiment
from wtf.learning.brains import inherit_brain, mix_parent_brains, model_signature


def configuration(kind):
    """Isolate controller integration from terrain and contextual-head fixtures."""
    config = world_config()
    if kind == "ppo":
        template = ppo_settings()
    elif kind in ("maskable_ppo", "mappo"):
        template = contextual_settings(kind)
    elif kind in ("q_network", "dqn", "ddqn", "qrdqn", "discrete_sac"):
        template = value_settings(kind, [4, 3])
    else:
        template = actor_settings(kind, [4, 3])
    for key in {
        "a2c": ("batch_size", "epochs", "target_kl"),
        "reinforce": ("rollout_steps", "batch_size", "epochs", "target_kl"),
        "grpo": ("rollout_steps", "normalize_advantages"),
        "trpo": ("epochs", "target_kl"),
    }.get(kind, ()):
        template.pop(key, None)
    config["brains"] = {"learner": template}
    config.pop("scent_trails", None)
    for species in config["species"].values():
        species["brain"] = "learner"
    config["world"]["max_steps"] = 16
    config["experiment"].update(runs=1, world_seed_policy="fixed")
    return config


def payload(experiment):
    """Export explicit per-controller contracts alongside captured observations."""
    return {
        "checkpoint": experiment.checkpoint(),
        "observations": {str(i): list(o) for i, o in experiment.observations.items()},
        "signatures": {
            str(i): model_signature(b.config, b.actions)
            for i, b in experiment.brains.items()
        },
    }


@unittest.skipUnless(_engine is not None, "Build the optional native episode engine")
class NativeControllerLifecycleTests(unittest.TestCase):
    def test_independent_policy_families_complete_the_same_episode(self):
        for kind in NATIVE_KINDS:
            with self.subTest(kind=kind):
                reference = Experiment(configuration(kind))
                native = NativeExperiment.from_experiment(reference)
                handle = native._handle
                for step in range(16):
                    reference.step()
                    _engine.run(handle, 1, "", {})
                    self.assertEqual(
                        encoded(reference.checkpoint()),
                        encoded(_engine.checkpoint(handle)),
                        f"{kind} step {step + 1}",
                    )
                actual = _engine.snapshot(handle, True, True)
                for animal_id, brain in reference.brains.items():
                    if animal_id in reference.observations:
                        self.assertEqual(
                            encoded(
                                brain.diagnostics(
                                    reference.observations[animal_id],
                                    include_activations=True,
                                )
                            ),
                            encoded(actual["diagnostics"][str(animal_id)]),
                        )

    def test_birth_state_resets_and_frozen_inheritance_match_all_families(self):
        seed = (2**64 - 1) ^ (19 * 7919 + 13)
        words = [(seed >> offset) & (2**32 - 1) for offset in (0, 32, 64)]
        for kind in NATIVE_KINDS:
            with self.subTest(kind=kind):
                reference = Experiment(configuration(kind))
                # Several updates populate private moments, replay and baselines.
                for _ in range(6):
                    reference.step()
                parents = list(reference.brains.values())[:2]
                for crossover in (False, True):
                    expected = (mix_parent_brains if crossover else inherit_brain)(
                        *parents if crossover else parents[:1], seed
                    )
                    actual = _engine.policy_inherit(
                        parents[0].checkpoint(),
                        parents[1].checkpoint(),
                        words,
                        crossover,
                    )
                    self.assertEqual(encoded(expected.checkpoint()), encoded(actual))
