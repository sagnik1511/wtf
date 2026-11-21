import copy
import unittest
from pathlib import Path

from wtf.config import load, validate
from wtf.simulation.day_night import sunlight, effective_body
from wtf.simulation.world import World, Resource

ROOT = Path(__file__).resolve().parents[1]


class DayNightTests(unittest.TestCase):
    def config(self):
        c = load(ROOT / "environments/baseline.json")
        for spec in c["species"].values():
            spec["initial_population"] = 1
            spec["reproduction"]["enabled"] = False
        c["species"]["predator"]["initial_population"] = 0
        c["obstacles"].update(generation="explicit", explicit_cells=[])
        c["resources"]["patch_count"] = 1
        c["day_night"] = dict(enabled=True, cycle_steps=100, start_phase=0)
        c["species"]["prey"]["body"]["vision_radius"] = 6
        c["species"]["prey"]["circadian"] = dict(
            night_vision_radius_multiplier=0.5,
            night_living_cost_multiplier=0.5,
            night_movement_cost_multiplier=2,
        )
        return c

    def test_sunlight_curve_and_periodicity(self):
        cycle = self.config()["day_night"]
        self.assertEqual(
            [sunlight(cycle, t) for t in (0, 25, 50, 75, 100)], [0, 0, 1, 0, 0]
        )
        self.assertAlmostEqual(sunlight(cycle, 35), sunlight(cycle, 65))
        self.assertLess(sunlight(cycle, 30), sunlight(cycle, 40))
        self.assertEqual(sunlight(None, 50), 1)

    def test_effective_body_is_shared_per_species_tick_without_config_mutation(self):
        w = World(self.config(), 100)
        a = next(iter(w.animals.values()))
        original = copy.deepcopy(w.config)
        self.assertEqual(w.body(a)["vision_radius"], 3)
        self.assertIs(w.body(a), w.body(a))
        w.tick = 50
        self.assertEqual(w.body(a)["vision_radius"], 6)
        self.assertEqual(w.config, original)
        spec = w.config["species"][a.species]
        self.assertEqual(effective_body(spec, 1), spec["body"])

    def test_night_observation_range_and_metabolism(self):
        w = World(self.config(), 100)
        a = next(iter(w.animals.values()))
        a.x, a.y = 10, 10
        w.resources = {(14, 10): Resource(14, 10, 10)}
        self.assertEqual(w.observe(a)[11], 0)
        start = a.energy
        cost = w.config["species"][a.species]["body"]["living_cost"]
        w.step({a.id: "rest"})
        self.assertAlmostEqual(a.energy, start - cost * 0.5)
        w.tick = 50
        self.assertEqual(w.observe(a)[11], 1)
        start = a.energy
        w.step({a.id: "rest"})
        self.assertAlmostEqual(a.energy, start - cost)

    def test_night_movement_cost(self):
        w = World(self.config(), 100)
        a = next(iter(w.animals.values()))
        a.x, a.y, a.heading = 10, 10, 0
        body = w.body(a)
        cost = body["living_cost"] + body["movement_cost"]
        start = a.energy
        w.step({a.id: "move_forward"})
        self.assertAlmostEqual(a.energy, start - cost)

    def test_checkpoint_continues_across_sunrise(self):
        w = World(self.config(), 100)
        w.tick = 24
        w.body(next(iter(w.animals.values())))
        restored = World.restore(w.checkpoint())
        for _ in range(31):
            actions = {i: "rest" for i in w.animals}
            self.assertEqual(w.observe_all(), restored.observe_all())
            w.step(actions)
            restored.step(actions)
            self.assertEqual(w.checkpoint(), restored.checkpoint())

    def test_disabled_cycle_preserves_existing_simulation(self):
        c = self.config()
        c.pop("day_night")
        original = World(c, 100)
        c["day_night"] = dict(enabled=False, cycle_steps=100)
        disabled = World(c, 100)
        for _ in range(30):
            self.assertEqual(original.observe_all(), disabled.observe_all())
            actions = {i: "rest" for i in original.animals}
            original.step(actions)
            disabled.step(actions)
            self.assertEqual(original.snapshot(), disabled.snapshot())

    def test_validation_and_preset(self):
        load(ROOT / "environments/day_night.json")
        validate(self.config())
        for path, value in [("cycle_steps", 0), ("start_phase", 2), ("enabled", 1)]:
            c = self.config()
            c["day_night"][path] = value
            with self.assertRaises(ValueError):
                validate(c)
        c = self.config()
        c["species"]["prey"]["circadian"]["night_vision_radius_multiplier"] = 0
        with self.assertRaises(ValueError):
            validate(c)

    def test_cycle_cache_is_once_per_tick_and_private_after_restore(self):
        from unittest.mock import patch

        w = World(self.config(), 100)
        a = next(iter(w.animals.values()))
        with patch("wtf.simulation.world.sunlight", wraps=sunlight) as light:
            body = w.body(a)
            specs = w._metabolism_species()
            self.assertIs(specs[a.species]["body"], body)
            self.assertIs(specs, w._metabolism_species())
            self.assertEqual(light.call_count, 1)
            restored = World.restore(w.checkpoint())
            self.assertIsNot(restored._metabolism_species(), specs)
            self.assertEqual(restored.body(restored.animals[a.id]), body)
            self.assertIsNot(restored.body(restored.animals[a.id]), body)
            w.tick += 1
            self.assertIsNot(w._metabolism_species(), specs)
            self.assertEqual(light.call_count, 3)

    def test_native_cycle_metabolism_matches_previous_python_path_exactly(self):
        from unittest.mock import patch
        from wtf.native import kernels
        import random

        if not kernels.available("flat_metabolism"):
            self.skipTest("Native metabolism unavailable")
        c = self.config()
        c["species"]["prey"]["initial_population"] = 6
        c["species"]["predator"]["initial_population"] = 2
        c["species"]["predator"]["circadian"] = dict(
            night_living_cost_multiplier=1.15, night_movement_cost_multiplier=0.9
        )
        native = World(c, 100)
        first = next(iter(native.animals.values()))
        first.energy = 0.01
        first.health = 0.25
        reference = World.restore(native.checkpoint())
        available = kernels.available
        rng = random.Random(31)
        with patch.object(
            kernels.native, "flat_metabolism", wraps=kernels.native.flat_metabolism
        ) as call:
            for _ in range(120):
                actions = {
                    i: rng.choice(c["species"][a.species]["allowed_actions"])
                    for i, a in native.animals.items()
                }
                actual = native.step(actions)
                with patch(
                    "wtf.native.kernels.available",
                    side_effect=lambda name: name != "flat_metabolism"
                    and available(name),
                ):
                    expected = reference.step(actions)
                self.assertEqual(actual, expected)
                self.assertEqual(native.checkpoint(), reference.checkpoint())
            self.assertEqual(call.call_count, 120)
