"""Native integer boundaries reject malformed counters without narrowing seeds."""

# Tests exercise the optional compiled probes and controlled lifecycle state.
# pylint: disable=c-extension-no-member,protected-access
# pylint: disable=missing-class-docstring,missing-function-docstring


import copy
import unittest

from tests.native_core_support import _engine, encoded
from tests.reproduction_support import force_reproduction
from tests.test_cpp_ppo import observation, settings
from tests.test_cpp_session import configuration
from wtf.experiment.native import NativeExperiment
from wtf.experiment.runner import Experiment
from wtf.learning.ppo import PPOBrain


@unittest.skipUnless(_engine is not None, "Build the optional native PPO engine")
class NativeNumericSafetyTests(unittest.TestCase):
    def test_counters_reject_nonintegral_nonfinite_negative_and_unsigned_overflow(self):
        original = PPOBrain(settings([3]), ["rest", "eat"], 17, 901).checkpoint()
        invalid = (
            -1,
            1.25,
            float("nan"),
            float("inf"),
            float("-inf"),
            2**63,
            2**64 - 1,
        )
        for key in ("steps", "updates", "rollout_updates"):
            for value in invalid:
                candidate = copy.deepcopy(original)
                candidate[key] = value
                with (
                    self.subTest(counter=key, value=value),
                    self.assertRaises(ValueError),
                ):
                    _engine.ppo_create(candidate)

    def test_next_increment_refuses_signed_counter_overflow(self):
        original = PPOBrain(settings([3]), ["rest", "eat"], 17, 901).checkpoint()
        for key in ("steps", "updates", "rollout_updates"):
            candidate = copy.deepcopy(original)
            candidate[key] = 2**63 - 1
            handle = _engine.ppo_create(candidate)
            with (
                self.subTest(counter=key),
                self.assertRaisesRegex(ValueError, "capacity"),
            ):
                _engine.ppo_learn(
                    handle, observation(0), "rest", 1.0, None, True, False
                )
            self.assertEqual(_engine.ppo_checkpoint(handle)[key], 2**63 - 1)

    def test_fractional_action_index_is_rejected_instead_of_truncated(self):
        brain = PPOBrain(settings([3]), ["rest", "eat"], 17, 901)
        brain.choose(observation(0))
        candidate = brain.checkpoint()
        candidate["pending"] = list(candidate["pending"])
        candidate["pending"][1] = 0.5
        with self.assertRaises(ValueError):
            _engine.ppo_create(candidate)

    def test_large_inherited_counts_stay_exact_and_total_overflow_rejects(self):
        experiment = Experiment(configuration(2))
        animal_id = next(iter(experiment.brains))
        experiment.brains[animal_id].updates = 2**53 + 2
        experiment.inherited_updates[animal_id] = 2**53 + 1
        native = NativeExperiment.from_experiment(experiment)
        self.assertEqual(native.snapshot()["total_updates"], 1)
        experiment.brains[animal_id].updates = 2**63 - 1
        experiment.inherited_updates[animal_id] = 0
        experiment.retired_updates = 1
        with self.assertRaisesRegex(ValueError, "capacity"):
            NativeExperiment.from_experiment(experiment)

    def test_unsigned_world_seed_keeps_full_offspring_seed_words(self):
        config = configuration(3)
        config["obstacles"].update(generation="explicit", explicit_cells=[])
        for species in config["species"].values():
            species["reproduction"].update(
                min_age_steps=1,
                cooldown_steps=1,
                health_threshold=0,
                min_energy=1,
                energy_cost=1,
                max_population=6,
            )
        reference = Experiment(config)
        reference.world.seed = 2**63 + 12345
        for index, animal in enumerate(reference.world.animals.values()):
            animal.x, animal.y = ((5, 5), (6, 5), (20, 20), (21, 20))[index]
            animal.sex = "female" if index % 2 == 0 else "male"
        reference.observations = reference.world.observe_all()
        force_reproduction(reference)
        native = NativeExperiment.from_experiment(reference)
        for _ in range(3):
            reference.step()
            native.run(1)
            self.assertEqual(
                encoded(reference.checkpoint()), encoded(native.checkpoint())
            )
        self.assertGreater(sum(reference.world.births.values()), 0)


if __name__ == "__main__":
    unittest.main()
