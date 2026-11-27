"""Exact standalone native sensing and mixed numeric food metrics."""

# Test scenarios deliberately edit private sensory continuation state.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring

# The optional compiled API is exercised by the native differential tests.
# pylint: disable=c-extension-no-member


import copy
import math
from pathlib import Path
import random
import unittest

from tests.native_core_support import _engine
from wtf.config import load
from wtf.simulation.compact_observations import compact_layout
from wtf.simulation.world import World


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return load(ROOT / "environments" / "ppo.json")


@unittest.skipUnless(_engine is not None, "Optional native engine is unavailable")
class NativeSensingTests(unittest.TestCase):
    def assert_observations(self, state):
        original = copy.deepcopy(state)
        reference = World.restore(state)
        expected = reference.observe_all()
        actual = _engine.observe(state)
        self.assertEqual(
            expected,
            {int(key): tuple(value) for key, value in actual["observations"].items()},
        )
        for animal_id, features in expected.items():
            for index, (left, right) in enumerate(
                zip(features, actual["observations"][str(animal_id)])
            ):
                self.assertEqual(
                    math.copysign(1.0, left),
                    math.copysign(1.0, right),
                    (animal_id, index),
                )
        self.assertEqual(
            reference.checkpoint().get("sensory_state"),
            actual["world"].get("sensory_state"),
        )
        self.assertEqual(original, state)
        return actual["world"], expected

    def test_visual_fields_history_scent_and_checkpoint_continuation(self):
        for degrees, history, count in ((360, 0, 1), (150, 3, 3), (180, 3, 10)):
            with self.subTest(degrees=degrees, history=history, top_k=count):
                config = configuration()
                for species in config["species"].values():
                    species["body"]["field_of_view_degrees"] = degrees
                for template in config["brains"].values():
                    template["movement_history_steps"] = history
                    template["observation_top_k"] = count
                world = World(config, 14)
                action_rng = random.Random(42)
                for _ in range(5):
                    state, _ = self.assert_observations(world.checkpoint())
                    world = World.restore(state)
                    world.step(
                        {
                            animal.id: action_rng.choice(
                                config["species"][animal.species]["allowed_actions"]
                            )
                            for animal in world.animals.values()
                        }
                    )

    def test_history_compensated_path_missing_sighting_and_same_tick(self):
        world = World(configuration(), 19)
        observer = next(
            animal
            for animal in world.animals.values()
            if world.diet(animal) == "resource"
        )
        target = next(
            animal for animal in world.animals.values() if world.diet(animal) == "prey"
        )
        world.animals = {animal.id: animal for animal in (observer, target)}
        world.obstacles.clear()
        observer.x, observer.y, observer.heading = 15, 15, 1
        target.x, target.y = 11, 15
        world.tick = 3
        state = world.checkpoint()
        state["sensory_state"]["visual"] = [
            [
                observer.id,
                [
                    [0, [[target.id, 1, 8, 12]]],
                    [1, [[target.id, 1, 9, 13]]],
                    [2, [[target.id, 1, 10, 14]]],
                ],
            ],
            [target.id, []],
        ]
        state, observations = self.assert_observations(state)
        template = world.config["brains"][
            world.config["species"][observer.species]["brain"]
        ]
        index = compact_layout(template).names.index(
            "Predator track 1 motion reliability"
        )
        # Compensated path length equals the straight line exactly. A plain
        # floating sum produced 0.9999999999999999 for this history.
        self.assertEqual(1.0, observations[observer.id][index])
        self.assert_observations(state)
        reference = World.restore(state)
        del reference.animals[target.id]
        reference.tick += 1
        state, _ = self.assert_observations(reference.checkpoint())
        self.assertNotIn(
            target.id, [animal_id for animal_id, _ in state["sensory_state"]["visual"]]
        )
        reference = World.restore(state)
        reference.tick += 4
        self.assert_observations(reference.checkpoint())

    def test_rotated_padding_preserves_signed_zero(self):
        world = World(configuration(), 19)
        observer = next(iter(world.animals.values()))
        world.animals = {observer.id: observer}
        world.resources.clear()
        template = world.config["brains"][
            world.config["species"][observer.species]["brain"]
        ]
        layout = compact_layout(template)
        for heading in range(4):
            observer.heading = heading
            with self.subTest(heading=heading):
                _, observations = self.assert_observations(world.checkpoint())
                features = observations[observer.id]
                if heading == 3:
                    for name in ("Food 1 right", "Flow right", "Mate right"):
                        self.assertEqual(
                            -1.0, math.copysign(1.0, features[layout.names.index(name)])
                        )

    def test_food_metrics_preserve_mixed_integer_float_sum_and_type(self):
        quantities = (
            [],
            [1, 2, 3],
            [1, 2, 0.1, 20, 0.2],
            [3.7, 0.3, 1, 0.3, 0.3, 3.7, 19.9, 20, 2, 0.1],
        )
        for amounts in quantities:
            with self.subTest(quantities=amounts):
                config = configuration()
                config["resources"]["growth_per_step"] = 0
                world = World(config, 42)
                world.animals.clear()
                resources = list(world.resources.values())[: len(amounts)]
                for resource, amount in zip(resources, amounts):
                    resource.quantity = amount
                world.resources = {
                    (resource.x, resource.y): resource for resource in resources
                }
                state = world.checkpoint()
                world.step({})
                expected = world.metrics()["food"]
                actual = _engine.world_step(state, {})["world"]["snapshot"]["metrics"][
                    "food"
                ]
                self.assertEqual(sum(amounts), expected)
                self.assertEqual(expected, actual)
                self.assertIs(type(expected), type(actual))


if __name__ == "__main__":
    unittest.main()
