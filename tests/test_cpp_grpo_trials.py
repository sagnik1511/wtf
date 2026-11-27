"""Exact native isolated GRPO worlds and frozen peer-policy continuation."""

# Regression fixtures inspect private state to verify exact owner isolation.
# pylint: disable=protected-access,c-extension-no-member
# Scenario names and parameterized cases document the expected contract.
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from tests.native_core_support import _engine, encoded, NATIVE_KINDS as KINDS
from tests.test_policy_algorithms import GRPOTests
from tests.test_reproduction import breeding_experiment
from tests.test_mutation import environment, force_reproduction
from tests.reproduction_support import force_action
from wtf.experiment.runner import Experiment
from wtf.learning.brains import create_brain
from wtf.learning.grouped_rollouts import collect_group


@unittest.skipUnless(
    _engine is not None and hasattr(_engine, "group_collect"),
    "Build the optional native GRPO collector",
)
class NativeGRPOTrialTests(unittest.TestCase):
    def assert_group(self, world, brains, focal):
        before = world.checkpoint()
        states = {str(i): b.checkpoint() for i, b in brains.items()}
        result = _engine.group_collect(before, states, focal)
        expected = collect_group(world, brains, focal)
        self.assertEqual(
            encoded([[s._asdict() for s in row] for row in expected]),
            encoded(result["group"]),
        )
        self.assertEqual(encoded(before), encoded(result["world"]))
        self.assertEqual(encoded(before), encoded(world.checkpoint()))
        for animal_id, brain in brains.items():
            self.assertEqual(
                encoded(brain.checkpoint()), encoded(result["brains"][str(animal_id)])
            )

    def test_encounter_historical_deaths_births_and_mutation(self):
        world, brains = GRPOTests().encounter()
        self.assert_group(world, brains, 1)
        world, brains = GRPOTests().encounter()
        brains[1].config["rollout_horizon"] = 4
        world.lifespans = [
            {"id": 500, "species": "prey", "age": 10, "cause": "starvation"}
        ]
        world.next_animal_id = 10000
        self.assert_group(world, brains, 1)
        experiment = breeding_experiment("grpo")
        experiment.brains[0].config["rollout_horizon"] = 4
        self.assert_group(experiment.world, experiment.brains, 0)
        experiment = force_reproduction(Experiment(environment("grpo")))
        focal = next(
            i
            for i, animal in experiment.world.animals.items()
            if animal.species == "predator"
        )
        self.assert_group(experiment.world, experiment.brains, focal)

    def test_all_fourteen_frozen_policy_families(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                world, brains = GRPOTests().encounter()
                brains[1].config.update(rollout_horizon=5, group_size=3)
                config = next(
                    v for v in environment(kind)["brains"].values() if v["type"] == kind
                )
                brains[0] = create_brain(
                    config, world.config["species"]["prey"]["allowed_actions"], 11, 17
                )
                # Seed recurrent hidden/cell memory and pending state; frozen
                # queries must preserve both exactly, along with private RNGs.
                for _ in range(2):
                    brains[0].choose(world.observe(world.animals[0]))
                self.assert_group(world, brains, 1)

    def test_recurrent_and_masked_newborns_use_private_reset_memory(self):
        for kind in ("recurrent_ppo", "lstm_ppo", "maskable_ppo"):
            with self.subTest(kind=kind):
                experiment = breeding_experiment("grpo")
                config = next(
                    v for v in environment(kind)["brains"].values() if v["type"] == kind
                )
                for animal_id, animal in experiment.world.animals.items():
                    if animal.species == "prey":
                        experiment.brains[animal_id] = create_brain(
                            config,
                            experiment.brains[animal_id].actions,
                            19,
                            51 + animal_id,
                        )
                        force_action(experiment.brains[animal_id], "reproduce")
                focal = next(
                    i
                    for i, animal in experiment.world.animals.items()
                    if animal.species == "predator"
                )
                experiment.brains[focal].config.update(rollout_horizon=6, group_size=3)
                self.assert_group(experiment.world, experiment.brains, focal)


if __name__ == "__main__":
    unittest.main()
