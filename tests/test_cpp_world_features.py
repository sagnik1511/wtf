"""Native physical features preserve world values, event order and private RNGs."""

# Fixtures inspect exact optional-native state and deliberately place animals.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# The private leaf probe is exposed only by the compiled extension.
# pylint: disable=c-extension-no-member

# The explicit terrain coverage matrix is independent of the production preset inventory.
# pylint: disable=duplicate-code

import copy
import random
import unittest

from tests.native_core_support import _engine, encoded
from wtf.config import load
from wtf.simulation.world import Resource, World


def physical_config(name="ppo"):
    """Isolate physical rules from separately tested social/tracking observers."""
    config = copy.deepcopy(load(name))
    config.pop("scent_trails", None)
    config.pop("consequence_learning", None)
    for template in config["brains"].values():
        for key in tuple(template):
            if key.startswith(
                ("social_", "consequence_", "scent_", "auxiliary_", "late_")
            ):
                template.pop(key)
        template["movement_history_steps"] = 0
    for species in config["species"].values():
        species["initial_population"] = min(4, species["initial_population"])
    return config


def small_map(flow=(0, 0)):
    config = physical_config("river_crossing")
    config.pop("placement", None)
    config["world"].update(width=9, height=7)
    config["obstacles"].update(generation="explicit", explicit_cells=[])
    config["resources"].update(patch_count=0)
    rows = ["........."] + ["..~~~~~.."] * 5 + ["........."]
    config["terrain"] = {
        "map_version": 1,
        "name": "Physical feature regression",
        "width": 9,
        "height": 7,
        "rows": rows,
        "legend": {
            ".": {"elevation": 0},
            "~": {"water_depth": 1.0, "flow": list(flow)},
            "h": {"elevation": 2},
        },
    }
    for species in config["species"].values():
        species["initial_population"] = 2
        if "swim" not in species["allowed_actions"]:
            species["allowed_actions"].append("swim")
    return config


@unittest.skipUnless(_engine is not None, "Build the optional native episode engine")
class NativeWorldFeatureTests(unittest.TestCase):
    def step_equal(self, world, actions):
        before = copy.deepcopy(world.checkpoint())
        result = world.step(actions)
        # The optional third flag isolates physical computation; observation
        # generation has its own value/signed-zero and sensory-memory suites.
        actual = _engine.world_step(before, actions, False)
        self.assertEqual(encoded(world.checkpoint()), encoded(actual["world"]))
        self.assertEqual(
            encoded({str(key): value for key, value in result.rewards.items()}),
            encoded(actual["rewards"]),
        )
        self.assertEqual(
            encoded([vars(event) for event in result.events]), encoded(actual["events"])
        )
        self.assertEqual(sorted(result.dead_ids), actual["dead"])
        return result

    def test_random_actions_across_geography_and_day_night_presets(self):
        names = (
            "river_crossing",
            "highlands",
            "enclosed_valleys",
            "grasslands",
            "seaside",
            "rolling_hills",
            "wetlands",
            "canyon",
            "oasis",
            "alpine_pass",
            "meandering_river",
            "day_night",
            "competition",
        )
        for name in names:
            with self.subTest(preset=name):
                config = physical_config(name)
                if name == "day_night":
                    config["day_night"].update(cycle_steps=8, start_phase=0.0)
                world = World(config, 151)
                generator = random.Random(771)
                for _ in range(12):
                    self.step_equal(
                        world,
                        {
                            key: generator.choice(
                                config["species"][animal.species]["allowed_actions"]
                            )
                            for key, animal in world.animals.items()
                        },
                    )

    def test_fractional_diagonal_currents_swimming_and_drowning(self):
        seen = set()
        for flow in ((0, 0), (0.75, 0.25), (2, -1), (-0.5, 1.25), (0, -3)):
            with self.subTest(flow=flow):
                config = small_map(flow)
                world = World(config, 171)
                for index, animal in enumerate(world.animals.values()):
                    animal.x, animal.y = ((1, 2), (4, 4), (4, 3), (7, 5))[index]
                    animal.heading = 1
                world.animals[1].energy = 0.2
                world.animals[1].health = 0.5
                for tick in range(5):
                    actions = {key: "rest" for key in world.animals}
                    if 0 in actions:
                        actions[0] = "swim" if tick < 2 else "rest"
                    result = self.step_equal(world, actions)
                    seen.update((event.kind, event.cause) for event in result.events)
        self.assertIn(("drifted", "current"), seen)
        self.assertIn(("damage_taken", "drowning"), seen)
        self.assertIn(("died", "drowning"), seen)
        self.assertIn(("energy_spent", "swim"), seen)

    def test_challenge_food_priority_displacement_aggression_and_death(self):
        for kind in ("food", "blocked_path", "aggression", "lethal_tie"):
            with self.subTest(kind=kind):
                config = physical_config("competition")
                config["obstacles"].update(generation="explicit", explicit_cells=[])
                config["species"]["prey"]["initial_population"] = 2
                config["species"]["predator"]["initial_population"] = 0
                rules = config["species"]["prey"]["competition"]
                rules.update(
                    enabled=True,
                    damage=100 if kind == "lethal_tie" else 7.5,
                    energy_cost=1.0,
                    food=True,
                    blocked_path=True,
                    aggression=True,
                )
                world = World(config, 151)
                left, right = world.animals.values()
                left.x, left.y, left.heading = 5, 5, 0 if kind == "aggression" else 1
                right.x, right.y, right.heading = 6, 5, 3
                left.energy, right.energy = (
                    (40, 40) if kind == "lethal_tie" else (60, 20)
                )
                world.resources = {(6, 5): Resource(6, 5, 10)} if kind == "food" else {}
                if kind == "lethal_tie":
                    left.health = right.health = 1
                actions = {
                    left.id: "challenge",
                    right.id: (
                        "eat"
                        if kind == "food"
                        else "challenge" if kind == "lethal_tie" else "rest"
                    ),
                }
                result = self.step_equal(world, actions)
                labels = {(event.kind, event.cause) for event in result.events}
                self.assertIn(
                    ("contest_won", "blocked_path" if kind == "lethal_tie" else kind),
                    labels,
                )
                if kind == "food":
                    self.assertIn(("energy_gained", "competition_food"), labels)
                elif kind == "blocked_path":
                    self.assertIn(("contest_moved", "competition"), labels)
                    self.assertIn(("displaced", "competition"), labels)
                elif kind == "lethal_tie":
                    self.assertIn(("died", "competition"), labels)

    def test_mixed_attacker_damage_uses_exact_python_compensated_sum(self):
        config = physical_config("ppo")
        config["obstacles"].update(generation="explicit", explicit_cells=[])
        config["species"]["prey"]["initial_population"] = 1
        base = copy.deepcopy(config["species"]["predator"])
        config["species"].pop("predator")
        for index, damage in enumerate((1.0, 0.1, 0.1, 0.1)):
            spec = copy.deepcopy(base)
            spec["initial_population"] = 1
            spec["body"]["attack_damage"] = damage
            config["species"][f"hunter_{index}"] = spec
        world = World(config, 151)
        for animal, (x, y, heading) in zip(
            world.animals.values(),
            ((5, 5, 0), (5, 4, 2), (6, 5, 3), (5, 6, 0), (4, 5, 1)),
        ):
            animal.x, animal.y, animal.heading = x, y, heading
        result = self.step_equal(
            world, {key: "rest" if key == 0 else "attack" for key in world.animals}
        )
        taken = next(
            event
            for event in result.events
            if event.kind == "damage_taken" and event.cause == "attack"
        )
        self.assertEqual(taken.amount, 1.3)

    def test_height_and_water_gate_mating_and_dry_birth_cells(self):
        for mode in ("wet_parent", "cliff", "dry"):
            with self.subTest(mode=mode):
                config = small_map()
                config["species"]["predator"]["initial_population"] = 0
                config["species"]["prey"]["reproduction"].update(
                    enabled=True,
                    min_age_steps=1,
                    cooldown_steps=1,
                    age_deviation_pct=0,
                    cooldown_deviation_pct=0,
                    health_threshold=0,
                    min_energy=1,
                    energy_cost=1,
                    max_population=6,
                    mate_radius=3,
                )
                rows = ["........."] * 7
                if mode == "wet_parent":
                    rows[3] = "...~....."
                elif mode == "cliff":
                    rows = ["....h...."] * 7
                config["terrain"]["rows"] = rows
                world = World(config, 151)
                mother, father = world.animals.values()
                mother.x, mother.y, mother.sex = 3, 3, "female"
                father.x, father.y, father.sex = 4, 3, "male"
                result = self.step_equal(
                    world, {mother.id: "reproduce", father.id: "reproduce"}
                )
                births = [event for event in result.events if event.kind == "born"]
                self.assertEqual(bool(births), mode == "dry")

    def test_malformed_terrain_rejects_before_native_grid_access(self):
        world = World(small_map(), 151)
        original = world.checkpoint()
        changes = (
            ("rows", ["."]),
            ("legend", {".": {"elevation": float("inf")}, "~": {"water_depth": 1}}),
            (
                "legend",
                {".": {"elevation": 0, "flow": [1, 0]}, "~": {"water_depth": 1}},
            ),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                state = copy.deepcopy(original)
                state["config"]["terrain"][key] = value
                with self.assertRaises(ValueError):
                    _engine.world_step(
                        state, {key: "rest" for key in world.animals}, False
                    )


if __name__ == "__main__":
    unittest.main()
