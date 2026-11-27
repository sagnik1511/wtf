"""Terrain, circadian and social inputs retain every observation version."""

# Fixtures inspect the optional compiled observation boundary.
# pylint: disable=missing-class-docstring,missing-function-docstring,c-extension-no-member
import copy
import math
import random
import unittest
from tests.native_core_support import _engine, encoded
from wtf.config import load
from wtf.simulation.world import World


@unittest.skipUnless(_engine is not None, "Build the optional native episode engine")
class NativeSensingFeatureTests(unittest.TestCase):
    def assert_sensing(self, state):
        original = encoded(state)
        world = World.restore(state)
        expected = world.observe_all()
        actual = _engine.observe(state)
        observations = actual["observations"]
        self.assertEqual(set(expected), {int(key) for key in observations})
        for animal_id, values in expected.items():
            observed = observations[str(animal_id)]
            self.assertEqual(tuple(values), tuple(observed), animal_id)
            for index, (left, right) in enumerate(zip(values, observed)):
                self.assertEqual(
                    math.copysign(1.0, left),
                    math.copysign(1.0, right),
                    (animal_id, index),
                )
        self.assertEqual(encoded(world.checkpoint()), encoded(actual["world"]))
        self.assertEqual(original, encoded(state))
        return world

    def test_geography_circadian_social_tracking_and_all_versions(self):
        names = (
            "ppo",
            "river_crossing",
            "highlands",
            "enclosed_valleys",
            "day_night",
            "tracking",
            "land_bridge_collective",
            "competition",
        )
        for name in names:
            for version in range(1, 8):
                with self.subTest(name=name, version=version):
                    config = copy.deepcopy(load(name))
                    for template in config["brains"].values():
                        template["observation_version"] = version
                    world = World(config, 100)
                    rng = random.Random(3)
                    for _ in range(4):
                        world = self.assert_sensing(world.checkpoint())
                        world.step(
                            {
                                animal.id: rng.choice(
                                    config["species"][animal.species]["allowed_actions"]
                                )
                                for animal in world.animals.values()
                            }
                        )

    def test_circadian_tick_boundaries_and_nonempty_consequence_memory(self):
        for name in ("day_night", "land_bridge_collective"):
            for version in (5, 6, 7):
                with self.subTest(name=name, version=version):
                    config = copy.deepcopy(load(name))
                    for template in config["brains"].values():
                        template["observation_version"] = version
                    world = World(config, 101)
                    rng = random.Random(19)
                    for step in range(48):
                        world.step(
                            {
                                animal.id: rng.choice(
                                    config["species"][animal.species]["allowed_actions"]
                                )
                                for animal in world.animals.values()
                            }
                        )
                        if step % 8 == 0:
                            world = self.assert_sensing(world.checkpoint())
                    if name == "day_night":
                        for tick in (0, 100, 200, 300, 399):
                            world.tick = tick
                            self.assert_sensing(world.checkpoint())
