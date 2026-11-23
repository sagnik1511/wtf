"""Adjacent founder placement, opt-out equivalence and restoration contracts."""

import copy
from collections import Counter
from unittest.mock import patch
import unittest

from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from wtf.simulation.placement import place_adjacent_pairs, zone
from wtf.simulation.world import World
from wtf.learning.brains import model_snapshot

# Fixtures inspect stored RNG state and placement gates without exposing new APIs.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring


def configuration():
    c = load("environments/baseline.json")
    c["species"]["predator"]["initial_population"] = 6
    c["species"]["predator"]["spawn"] = {"mode": "adjacent_pairs"}
    return validate(c)


class AdjacentSpawnTests(unittest.TestCase):
    def assert_pairs(self, world, species):
        animals = [a for a in world.animals.values() if a.species == species]
        females = [a for a in animals if a.sex == "female"]
        males = [a for a in animals if a.sex == "male"]
        self.assertEqual(len(females), len(males))
        for female, male in zip(females, males):
            self.assertEqual(abs(female.x - male.x) + abs(female.y - male.y), 1)
            self.assertTrue(world._mating_land_route(female, male, 3))
            self.assertEqual(female.age, 0)
            self.assertEqual(male.mating_intent_until_tick, -1)
        self.assertEqual(
            len(world.animals), len({a.position for a in world.animals.values()})
        )

    def test_opt_out_and_explicit_random_are_exact(self):
        c = configuration()
        del c["species"]["predator"]["spawn"]
        explicit = copy.deepcopy(c)
        explicit["species"]["predator"]["spawn"] = {"mode": "random"}
        for seed in (0, 42, 100):
            left, right = World(c, seed), World(explicit, seed)
            self.assertEqual(left.snapshot(), right.snapshot())
            for _ in range(5):
                actions = {i: "rest" for i in left.animals}
                self.assertEqual(left.step(actions), right.step(actions))
            self.assertEqual(left.rng.getstate(), right.rng.getstate())

    def test_only_positions_change_and_seed_repeats_on_terrain(self):
        c = load("environments/seaside.json")
        c["species"]["predator"]["initial_population"] = 6
        paired = copy.deepcopy(c)
        paired["species"]["predator"]["spawn"] = {"mode": "adjacent_pairs"}
        for seed in range(30):
            normal, actual = World(c, seed), World(paired, seed)
            self.assert_pairs(actual, "predator")
            self.assertEqual(normal.resources, actual.resources)
            self.assertEqual(normal.rng.getstate(), actual.rng.getstate())
            for i, animal in actual.animals.items():
                old = vars(normal.animals[i]).copy()
                new = vars(animal).copy()
                if animal.species == "predator":
                    for key in ("x", "y"):
                        del old[key]
                        del new[key]
                self.assertEqual(old, new)
                self.assertFalse(actual.geography.depth(animal.position))
            self.assertEqual(actual.snapshot(), World(paired, seed).snapshot())

    def test_enclosure_and_multiple_species(self):
        c = load("environments/enclosed_valleys.json")
        for spec in c["species"].values():
            spec["spawn"] = {"mode": "adjacent_pairs"}
        for seed in (0, 42, 100):
            world = World(validate(c), seed)
            for species in c["species"]:
                self.assert_pairs(world, species)
            for animal in world.animals.values():
                self.assertEqual(
                    zone(animal.position, c["placement"]["enclosure_bounds"]),
                    c["placement"]["species"][animal.species],
                )
        c = configuration()
        c["species"]["lynx"] = copy.deepcopy(c["species"]["predator"])
        c["species"]["lynx"]["initial_population"] = 2
        world = World(validate(c), 10)
        self.assert_pairs(world, "predator")
        self.assert_pairs(world, "lynx")

    def test_even_population_validation_and_empty_species(self):
        for mode in ("unknown", False, None):
            c = configuration()
            c["species"]["predator"]["spawn"]["mode"] = mode
            with self.assertRaises(ValueError):
                validate(c)
        c = configuration()
        c["species"]["predator"]["initial_population"] = 3
        with self.assertRaisesRegex(ValueError, "even"):
            validate(c)
        c["species"]["predator"]["initial_population"] = 0
        self.assertNotIn(
            "predator",
            Counter(a.species for a in World(validate(c), 0).animals.values()),
        )

    def test_fallback_finds_disjoint_pairs_in_fully_occupied_corridor(self):
        c = configuration()
        c["species"]["prey"]["initial_population"] = 0
        c["resources"]["patch_count"] = 0
        c["obstacles"].update(
            generation="explicit",
            explicit_cells=[
                [x, y]
                for y in range(c["world"]["height"])
                for x in range(c["world"]["width"])
                if y != 2 or not 1 <= x <= 6
            ],
        )
        normal = copy.deepcopy(c)
        del normal["species"]["predator"]["spawn"]
        world = World(validate(normal), 7)
        females = [a for a in world.animals.values() if a.sex == "female"]
        males = [a for a in world.animals.values() if a.sex == "male"]
        for x, animal in enumerate(females + males, 1):
            animal.x, animal.y = x, 2
        world.config["species"]["predator"]["spawn"] = {"mode": "adjacent_pairs"}
        before = world.rng.getstate()
        place_adjacent_pairs(world)
        self.assert_pairs(world, "predator")
        self.assertEqual(before, world.rng.getstate())
        c["obstacles"]["explicit_cells"] = [
            [x, y]
            for y in range(c["world"]["height"])
            for x in range(c["world"]["width"])
            if y != 2 or x not in (1, 3, 5, 7, 9, 11)
        ]
        with self.assertRaisesRegex(ValueError, "insufficient adjacent"):
            World(validate(c), 7)

    def test_initial_brain_weights_and_sexes_are_unchanged(self):
        c = configuration()
        c["experiment"]["world_seed_policy"] = "fixed"
        random_config = copy.deepcopy(c)
        del random_config["species"]["predator"]["spawn"]
        normal = Experiment(random_config, mode="evaluate")
        paired = Experiment(c, mode="evaluate")
        for i, brain in paired.brains.items():
            self.assertEqual(normal.world.animals[i].sex, paired.world.animals[i].sex)
            self.assertEqual(model_snapshot(normal.brains[i]), model_snapshot(brain))

    def test_checkpoint_does_not_reapply_spawning(self):
        e = Experiment(configuration(), mode="evaluate")
        for _ in range(4):
            e.step()
        with patch(
            "wtf.simulation.world.place_adjacent_pairs",
            side_effect=AssertionError("restore must keep positions"),
        ):
            restored = Experiment.restore(e.checkpoint())
        self.assertEqual(e.snapshot(), restored.snapshot())
        for _ in range(4):
            e.step()
            restored.step()
            self.assertEqual(e.snapshot(), restored.snapshot())
        for i, brain in e.brains.items():
            self.assertEqual(model_snapshot(brain), model_snapshot(restored.brains[i]))
