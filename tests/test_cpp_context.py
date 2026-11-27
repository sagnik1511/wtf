"""Exact leaf parity for native decision masks and centralized critic inputs."""

# Fixtures deliberately inspect optional native probes and private decision indexes.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# pylint: disable=c-extension-no-member

import copy
import json
import random
import unittest

from wtf.config import validate
from wtf.learning.contextual_ppo import action_mask, central_contexts
from wtf.simulation.world import World, Resource
from tests.test_geography import animal, configuration
from tests.native_core_support import _engine


class Native:
    """Resolve optional compiled probes only inside non-skipped fixtures."""

    @staticmethod
    def contexts(state):
        return _engine.world_contexts(state)

    @staticmethod
    def masks(state, actions):
        return _engine.world_masks(state, actions)


native = Native()


@unittest.skipUnless(_engine is not None, "Build the optional native decision context")
class ContextParityTests(unittest.TestCase):
    def same(self, w):
        before = json.dumps(w.checkpoint(), sort_keys=True)
        expected = {name: list(value) for name, value in central_contexts(w).items()}
        self.assertEqual(expected, native.contexts(w.checkpoint()))
        context = w._prepare_mating_context()
        expected = {
            str(a.id): list(
                action_mask(
                    w,
                    a,
                    w.config["species"][a.species]["allowed_actions"],
                    mating_context=context,
                )
            )
            for a in w.animals.values()
        }
        self.assertEqual(expected, native.masks(w.checkpoint(), None))
        # Both observers are read-only, including RNG and due-age scheduling.
        self.assertEqual(before, json.dumps(w.checkpoint(), sort_keys=True))

    def make(self, terrain=False):
        c = configuration("maskable_ppo", rows=["..........."] * 7)
        c["world"]["max_steps"] = 200
        if not terrain:
            c.pop("terrain", None)
            for spec in c["species"].values():
                spec["allowed_actions"].remove("swim")
        for spec in c["species"].values():
            spec["reproduction"].update(
                enabled=True,
                min_age_steps=10,
                cooldown_steps=15,
                health_threshold=20,
                min_energy=15,
                energy_cost=10,
                max_population=30,
                mate_radius=3,
            )
            spec["competition"] = {
                "enabled": True,
                "aggression": True,
                "food": True,
                "blocked_path": True,
            }
            if "challenge" not in spec["allowed_actions"]:
                spec["allowed_actions"].append("challenge")
        w = World(validate(c), 100)
        w.animals = {
            0: animal(0, 3, 2, sex="female"),
            1: animal(1, 4, 2, sex="male"),
            2: animal(2, 5, 2, species="predator", sex="female"),
            3: animal(3, 6, 2, species="predator", sex="male"),
        }
        w.next_animal_id = 4
        w.resources = {(3, 2): Resource(3, 2, 2.5), (4, 2): Resource(4, 2, 0.0)}
        return w

    def test_randomized_state_and_all_action_families(self):
        rng = random.Random(21)
        for terrain in (False, True):
            w = self.make(terrain)
            for _ in range(120):
                for a in w.animals.values():
                    a.energy = rng.choice([0, 10, 15, 30, 99, 100])
                    a.health = rng.choice([0, 20, 21, 40, 100])
                    a.age = rng.randrange(25)
                    a.heading = rng.randrange(4)
                    a.last_reproduced_age = rng.choice([-1, 3])
                    a.reproduction_due_age = rng.choice([-1, 10, 21])
                self.same(w)

    def test_birth_space_cliffs_water_and_population_caps(self):
        for token in (".", "#", "~", "3"):
            w = self.make(True)
            for row in w.config["terrain"]["rows"]:
                self.assertEqual(len(row), 11)
            rows = ["###########"] * 7
            rows[2] = "###.." + token + "#####"
            rows[3] = "###" + token + token + "######"
            c = copy.deepcopy(w.config)
            c["terrain"]["rows"] = rows
            w = World(validate(c), 100)
            w.animals = {
                0: animal(0, 3, 2, sex="female"),
                1: animal(1, 4, 2, sex="male"),
            }
            w.next_animal_id = 2
            w.resources = {}
            for a in w.animals.values():
                a.energy = 100
                a.health = 100
                a.age = 15
            self.same(w)
            w.animals[0].reproduction_due_age = 16
            self.same(w)
            w.config["species"]["prey"]["reproduction"]["max_population"] = 2
            self.same(w)

    def test_multispecies_empty_team_and_compensated_sum(self):
        w = self.make()
        w.config["species"]["other"] = copy.deepcopy(w.config["species"]["prey"])
        w.animals[3].species = "other"
        for index, a in enumerate(w.animals.values()):
            a.energy = 0.1 + index / 7
            a.health = 0.2 + index / 9
            a.age = 1 + index
        w.resources = {
            (0, 0): Resource(0, 0, 1e16),
            (1, 0): Resource(1, 0, 1.0),
            (2, 0): Resource(2, 0, 1.0),
        }
        self.same(w)
        w.animals = {}
        self.same(w)

    def test_no_available_action_rejected(self):
        w = self.make()
        w.animals[0].health = 0
        # None can eat at max energy; error contract is common across the phase.
        for a in w.animals.values():
            a.energy = w.body(a)["max_energy"]
        with self.assertRaises(ValueError):
            native.masks(w.checkpoint(), ["eat"])
        with self.assertRaises(ValueError):
            action_mask(w, w.animals[0], ["eat"])


if __name__ == "__main__":
    unittest.main()
