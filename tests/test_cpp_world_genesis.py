"""Native founder creation preserves seeded layout and independent RNG streams."""

# These fixtures deliberately inspect the optional compiled construction boundary.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# The optional engine exposes its construction probe only when built.
# pylint: disable=c-extension-no-member

import copy
import unittest
from unittest.mock import patch

from tests.native_core_support import _engine, encoded
from wtf.config import load, validate
from wtf.simulation import placement
from wtf.simulation.world import World


def configuration(name="ppo"):
    """Isolate genesis from the separate initial social-state caller seam."""
    config = copy.deepcopy(load(name))
    config.pop("consequence_learning", None)
    for template in config["brains"].values():
        for key in tuple(template):
            if key.startswith(("social_", "consequence_", "auxiliary_", "late_")):
                template.pop(key)
    return validate(config)


@unittest.skipUnless(_engine is not None, "Build the optional native episode engine")
class NativeWorldGenesisTests(unittest.TestCase):
    def assert_genesis(self, config, seed):
        expected = World(config, seed).checkpoint()
        actual = _engine.world_create(config, seed)
        self.assertEqual(encoded(expected), encoded(actual))
        return actual

    def test_presets_terrain_tracking_and_unsigned_world_seeds(self):
        names = (
            "baseline",
            "ppo",
            "qrdqn",
            "meandering_river",
            "land_bridge",
            "day_night",
            "competition",
            "enclosed_valleys",
            "rolling_hills",
            "seaside",
        )
        for name in names:
            config = configuration(name)
            for seed in (0, 100, 2**63 + 12345):
                with self.subTest(preset=name, seed=seed):
                    self.assert_genesis(config, seed)

    def test_resource_pool_sampling_preserves_integer_float_and_signed_zero_values(
        self,
    ):
        for amount in (1, 1.25, -0.0):
            with self.subTest(quantity=amount):
                config = configuration()
                config["world"].update(width=7, height=7)
                config["obstacles"].update(generation="explicit", explicit_cells=[])
                config["resources"].update(
                    patch_count=45, initial_quantity_per_patch=amount
                )
                for species in config["species"].values():
                    species["initial_population"] = 2
                self.assert_genesis(validate(config), 71)

    def test_adjacent_matching_fallback_is_exact_and_rejects_disconnected_cells(self):
        config = configuration("baseline")
        config["species"]["prey"]["initial_population"] = 0
        config["species"]["predator"].update(
            initial_population=6, spawn={"mode": "adjacent_pairs"}
        )
        config["resources"]["patch_count"] = 0
        config["obstacles"].update(
            generation="explicit",
            explicit_cells=[
                [x, y]
                for y in range(config["world"]["height"])
                for x in range(config["world"]["width"])
                if y != 2 or not 1 <= x <= 6
            ],
        )
        config = validate(config)
        with patch.object(
            placement, "_pair_cells", wraps=placement._pair_cells
        ) as fallback:
            for seed in range(12):
                self.assert_genesis(config, seed)
            self.assertTrue(fallback.called)
        config["obstacles"]["explicit_cells"] = [
            [x, y]
            for y in range(config["world"]["height"])
            for x in range(config["world"]["width"])
            if y != 2 or x not in (1, 3, 5, 7, 9, 11)
        ]
        with self.assertRaises(ValueError):
            World(config, 151)
        with self.assertRaises(ValueError):
            _engine.world_create(config, 151)

    def test_enclosed_zone_pairing_with_arbitrary_species_names(self):
        config = configuration("enclosed_valleys")
        for spec in config["species"].values():
            spec["spawn"] = {"mode": "adjacent_pairs"}
        config["species"]["grazer"] = config["species"].pop("prey")
        config["placement"]["species"]["grazer"] = config["placement"]["species"].pop(
            "prey"
        )
        config = validate(config)
        for seed in (0, 42, 100):
            self.assert_genesis(config, seed)

    def test_native_genesis_never_constructs_or_observes_a_python_world(self):
        config = configuration()
        expected = encoded(World(config, 81).checkpoint())
        with (
            patch.object(
                World, "__init__", side_effect=AssertionError("Python construction")
            ),
            patch.object(
                World, "observe_all", side_effect=AssertionError("Python observation")
            ),
        ):
            actual = _engine.world_create(config, 81)
        self.assertEqual(expected, encoded(actual))


if __name__ == "__main__":
    unittest.main()
