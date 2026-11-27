"""Front visual fields keep ranges, private histories and physical rules separate."""

# Descriptive test names document assertions; fixtures inspect ephemeral sight caches.
# pylint: disable=missing-function-docstring,protected-access

import copy
import json
import unittest
from unittest.mock import patch
from tests.test_social_learning import config as social_config
from tests.test_water_vision import setup as river_scene
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.native import kernels
from wtf.simulation.actions import DIRECTIONS
from wtf.simulation.world import Animal, Resource, World
from wtf.simulation.vision import in_field


def configuration():
    """Small dry map with wide sight and multiple slots."""
    c = social_config()
    for spec in c["species"].values():
        spec["body"].update(vision_radius=4, field_of_view_degrees=180)
    for brain in c["brains"].values():
        brain.update(observation_top_k=3, observation_version=4)
    return validate(c)


def scene():
    """Place front, rear and side-boundary targets at symmetric distances."""
    w = World(configuration(), 100)
    w.animals = {
        0: Animal(0, "prey", 4, 3, 1, 60, 30, age=500, sex="male"),
        1: Animal(1, "predator", 6, 3, 0, 80, 55),
        2: Animal(2, "predator", 2, 3, 0, 80, 55),
        3: Animal(3, "predator", 4, 5, 0, 80, 55),
    }
    w.resources = {p: Resource(*p, 10) for p in ((4, 3), (5, 3), (3, 3), (4, 2))}
    return w


class ForwardVisionTests(unittest.TestCase):
    """Boundary geometry and propagation into every visual consumer."""

    def test_cardinal_half_planes_include_sides_and_origin(self):
        for heading, (hx, hy) in enumerate(DIRECTIONS):
            self.assertTrue(in_field((0, 0), (hx, hy), heading, 180))
            self.assertFalse(in_field((0, 0), (-hx, -hy), heading, 180))
            self.assertTrue(in_field((0, 0), (hy, -hx), heading, 180))
            self.assertTrue(in_field((0, 0), (0, 0), heading, 180))
            self.assertTrue(in_field((0, 0), (-hx, -hy), heading, 360))

    def test_narrow_field_and_angle_cache_invalidation(self):
        for heading, (hx, hy) in enumerate(DIRECTIONS):
            self.assertTrue(in_field((0, 0), (hx + 3 * hy, hy - 3 * hx), heading, 150))
            self.assertFalse(in_field((0, 0), (hx + 4 * hy, hy - 4 * hx), heading, 150))
            self.assertFalse(in_field((0, 0), (hy, -hx), heading, 150))
            self.assertTrue(in_field((0, 0), (0, 0), heading, 150))
        for k in (1, 3):
            w = scene()
            del w.animals[1]  # Side target must be the nearest available predator.
            for b in w.config["brains"].values():
                b["observation_top_k"] = k
            wide = w.observe_all()[0]
            for spec in w.config["species"].values():
                spec["body"]["field_of_view_degrees"] = 150
            narrow = w.observe_all()[0]
            self.assertNotEqual(wide, narrow)
            available = kernels.available
            with patch.object(
                kernels,
                "available",
                side_effect=lambda name, dispatch=available: name
                not in ("nearest", "nearest_many")
                and dispatch(name),
            ):
                for index in w._perception:
                    index.results.clear()
                self.assertEqual(w.observe_all()[0], narrow)

    def test_top_k_and_stationary_turn_cache_include_heading(self):
        w = scene()
        east = w.observe_all()[0]
        predator_slots = east[32:44]  # 8 + two categories * 4 * k.
        self.assertEqual([predator_slots[i] for i in range(3, 12, 4)], [1, 1, 0])
        self.assertTrue(all(predator_slots[i] >= 0 for i in range(0, 12, 4)))
        w.animals[0].heading = 3
        west = w.observe_all()[0]
        self.assertLess(west[32], 0)
        self.assertNotEqual(west[32:44], predator_slots)

    def test_native_selectors_and_reference_match_without_disabling_native_work(self):
        w = scene()
        expected = w.observe_all()
        available = kernels.available
        with patch.object(
            kernels,
            "available",
            side_effect=lambda name: name not in ("nearest", "nearest_many")
            and available(name),
        ):
            for index in w._perception:
                index.results.clear()
            actual = w.observe_all()
        self.assertEqual(actual, expected)
        if kernels.available("nearest_many"):
            for index in w._perception:
                index.results.clear()
            # Native method is built dynamically by the optional CPython extension.
            with patch.object(
                kernels.native,
                "nearest_many",
                wraps=kernels.native.nearest_many,  # pylint: disable=c-extension-no-member
            ) as native:
                self.assertEqual(w.observe_all(), expected)
                self.assertGreater(native.call_count, 0)

    def test_water_extension_stays_front_only(self):
        w = river_scene()
        w.config["species"]["prey"]["body"]["field_of_view_degrees"] = 180
        self.assertEqual(w.observe_all()[0][19], 1)
        w.animals[0].heading = 3
        self.assertEqual(w.observe_all()[0][19], 0)

    def test_hidden_mate_does_not_change_physical_reproduction_eligibility(self):
        w = scene()
        a = w.animals[0]
        w.config["species"]["prey"]["reproduction"].update(enabled=True, mate_radius=2)
        w.animals = {0: a, 4: Animal(4, "prey", 3, 3, 1, 60, 30, age=500, sex="female")}
        self.assertTrue(w.reproduction_available(a))
        hidden = w.observe_all()[0]
        self.assertEqual(hidden[63], 0)  # k=3 mate-presence slot.
        self.assertEqual(hidden[65], 0)
        a.heading = 3
        visible = w.observe_all()[0]
        self.assertEqual(visible[63], 1)
        self.assertEqual(visible[65], 1)

    def test_hidden_actions_are_not_witnessed(self):
        totals = []
        for heading in (1, 3):
            w = scene()
            w.animals = {0: w.animals[0], 1: Animal(1, "prey", 5, 3, 0, 30, 30)}
            w.animals[0].heading = heading
            w.resources = {(5, 3): Resource(5, 3, 10)}
            w.step({0: "rest", 1: "eat"})
            template = w.config["brains"][w.config["species"]["prey"]["brain"]]
            features = w._social.observe(w, w.animals[0], template)
            totals.append(sum(features[:24]))
        self.assertGreater(totals[0], 0)
        self.assertEqual(totals[1], 0)

    def test_visual_history_does_not_refresh_behind_observer_but_scent_remains(self):
        c = configuration()
        c["scent_trails"] = {"enabled": True, "lifetime_steps": 20}
        for b in c["brains"].values():
            b.update(
                movement_history_steps=3,
                scent_observation={"mode": "blended", "radius": 2},
            )
        w = World(validate(c), 100)
        w.animals = {
            0: Animal(0, "prey", 4, 3, 1, 60, 30),
            1: Animal(1, "predator", 5, 3, 1, 80, 55),
        }
        w.observe_all()
        a = w.animals[0]
        a.heading = 3
        w.animals[1].x = 6
        w.tick += 1
        w.observe_all()
        frames = w._tracking.visual.frames[0]
        self.assertNotIn(1, frames[-1][1])
        self.assertEqual(frames[0][1][1][1:], (5, 3))
        w.step({0: "rest", 1: "rest"})
        template = c["brains"][c["species"]["prey"]["brain"]]
        rear = w._tracking.scent.sample(w, a, template, w._scent_context)
        a.heading = 1
        front = w._tracking.scent.sample(w, a, template, w._scent_context)
        self.assertEqual(rear, front)
        self.assertTrue(any(front))

    def test_default_360_and_explicit_360_have_identical_policy_and_physics(self):
        c = configuration()
        for spec in c["species"].values():
            spec["body"].pop("field_of_view_degrees")
        explicit = copy.deepcopy(c)
        for spec in explicit["species"].values():
            spec["body"]["field_of_view_degrees"] = 360
        left, right = Experiment(c), Experiment(explicit)
        for _ in range(5):
            left.step()
            right.step()
            self.assertEqual(left.observations, right.observations)
            self.assertEqual(left.world.snapshot(), right.world.snapshot())
            self.assertEqual(
                {i: b.checkpoint() for i, b in left.brains.items()},
                {i: b.checkpoint() for i, b in right.brains.items()},
            )

    def test_restore_and_validation(self):
        e = Experiment(configuration())
        e.step()
        restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
        for _ in range(3):
            e.step()
            restored.step()
        self.assertEqual(e.checkpoint(), restored.checkpoint())
        for bad in (0, 90, 151, 181, 361, True, 180.0, "180"):
            c = configuration()
            c["species"]["prey"]["body"]["field_of_view_degrees"] = bad
            with self.assertRaises(ValueError):
                validate(c)
