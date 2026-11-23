"""Physical feeding/breeding contracts for the ordinary-PPO starter preset.

These checks exercise chosen actions; they do not assume a learned policy will
choose them or establish long-term ecological stability.
"""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import unittest

from wtf.config import load
from wtf.experiment.runner import Experiment
from wtf.learning.ppo import PPOBrain
from wtf.simulation.world import Animal, World


class PPOStarterPresetTests(unittest.TestCase):
    def world(self):
        w = World(load("ppo"), 100)
        w.obstacles.clear()
        w.resources.clear()
        w.animals = {}
        w.next_animal_id = 2
        return w

    def test_controllers_remain_ordinary_ppo(self):
        e = Experiment(load("ppo"))
        for brain in e.brains.values():
            self.assertIs(type(brain), PPOBrain)

    def test_successful_hunt_replenishes_energy_with_one_attack(self):
        w = self.world()
        body = w.config["species"]["predator"]["body"]
        predator = Animal(0, "predator", 5, 5, 0, 20.0, body["max_health"])
        x, y = w.front(predator)
        prey = Animal(
            1, "prey", x, y, 0, 60.0, w.config["species"]["prey"]["body"]["max_health"]
        )
        w.animals = {0: predator, 1: prey}
        result = w.step({0: "attack", 1: "rest"})
        self.assertIn(1, result.dead_ids)
        self.assertTrue(any(e.kind == "kill" and e.actor == 0 for e in result.events))
        expected = min(body["max_energy"], 20.0 + body["energy_per_kill"]) - (
            body["living_cost"] + body["attack_cost"]
        )
        self.assertAlmostEqual(predator.energy, expected)
        self.assertGreater(predator.energy, 20.0)

    def test_healthy_mature_pair_can_breed_without_spending_all_energy(self):
        w = self.world()
        spec = w.config["species"]["predator"]
        body = spec["body"]
        parents = [
            Animal(
                0, "predator", 5, 5, 0, 31.0, body["max_health"], age=200, sex="female"
            ),
            Animal(
                1, "predator", 6, 5, 0, 31.0, body["max_health"], age=200, sex="male"
            ),
        ]
        w.animals = {p.id: p for p in parents}
        result = w.step({0: "reproduce", 1: "reproduce"})
        self.assertEqual(sum(e.kind == "born" for e in result.events), 1)
        for parent in parents:
            self.assertAlmostEqual(
                parent.energy,
                31.0 - body["living_cost"] - spec["reproduction"]["energy_cost"],
            )
            self.assertGreater(parent.energy, 0.0)
            self.assertEqual(parent.health, body["max_health"])
