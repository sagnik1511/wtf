"""Phase-local mating reuse preserves fresh checks, RNGs and birth ordering."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from wtf.config import load
from wtf.learning.contextual_ppo import action_mask
from wtf.experiment.runner import Experiment
from wtf.simulation.world import Animal, DIRECTIONS, World
from tests.test_geography import configuration as terrain_config
from tests.test_mating_intent import world as breeding_world
from tests.test_advanced_algorithms import environment as algorithm_config
from tests.reproduction_support import force_action


def legacy_birth_cells(world, mother, father, occupied):
    """Original ordered/deduplicated enumeration, independent of the fast path."""
    spec = world.config["species"][mother.species]
    return list(
        dict.fromkeys(
            (parent.x + dx, parent.y + dy)
            for parent in (mother, father)
            for dx, dy in DIRECTIONS
            if not world.blocked((parent.x + dx, parent.y + dy))
            and not (
                world.geography
                and (
                    world.geography.depth((parent.x + dx, parent.y + dy))
                    or abs(
                        world.geography.elevation((parent.x + dx, parent.y + dy))
                        - world.geography.elevation(parent.position)
                    )
                    > spec.get("locomotion", {}).get("max_climb", 1.0)
                )
            )
            and (parent.x + dx, parent.y + dy) not in occupied
        )
    )


class BirthSpaceChecks(unittest.TestCase):
    def test_boolean_and_ordered_cells_match_reference_on_flat_water_cliffs_and_occupancy(
        self,
    ):
        for terrain in (False, True):
            for father_x in (2, 3, 4):
                c = (
                    terrain_config(rows=["....~....", "..3.~...."] + ["....~...."] * 5)
                    if terrain
                    else load(Path("environments/maskable_ppo.json"))
                )
                w = World(c, 100)
                w.obstacles.clear()
                mother = Animal(0, "prey", 2, 2, 0, 60, 30, sex="female")
                father = Animal(1, "prey", father_x, 2, 0, 60, 30, sex="male")
                for crowded in (False, True):
                    occupied = {mother.position, father.position}
                    if crowded:
                        occupied.update(
                            (x, y) for x in range(w.width) for y in range(w.height)
                        )
                    before = copy.deepcopy(w.checkpoint())
                    expected = legacy_birth_cells(w, mother, father, occupied)
                    self.assertEqual(w._birth_cells(mother, father, occupied), expected)
                    self.assertEqual(
                        w._has_birth_space(mother, father, occupied), bool(expected)
                    )
                    self.assertEqual(w.checkpoint(), before)

    def test_availability_stops_early_without_materializing_birth_choices(self):
        w = breeding_world()
        mother, father = w.animals.values()
        occupied = {a.position for a in w.animals.values()}
        with (
            patch.object(w, "blocked", wraps=w.blocked) as blocked,
            patch.object(
                w,
                "_birth_cells",
                side_effect=AssertionError("Boolean query built a list"),
            ),
        ):
            self.assertTrue(w._has_birth_space(mother, father, occupied))
            self.assertEqual(blocked.call_count, 1)
            self.assertTrue(action_mask(w, mother, ["rest", "reproduce"])[1])


class SharedMatingContextChecks(unittest.TestCase):
    def test_each_masked_phase_prepares_readiness_once_with_or_without_mating_observations(
        self,
    ):
        for version in (1, 3, 4):
            c = algorithm_config("maskable_ppo")
            for b in c["brains"].values():
                b["observation_version"] = version
            for spec in c["species"].values():
                spec["reproduction"]["enabled"] = True
            e = Experiment(c)
            w = e.world
            for a in w.animals.values():
                a.age = 1000
                force_action(e.brains[a.id], "rest")
            e.observations = w.observe_all()
            e._refresh_context(mating_context=w._mate_context)
            with (
                patch.object(
                    w, "_prepare_mating_context", wraps=w._prepare_mating_context
                ) as prepare,
                patch.object(
                    w, "_reproduction_eligible", wraps=w._reproduction_eligible
                ) as eligible,
            ):
                e.step()
                self.assertEqual(prepare.call_count, 1, version)
                active = sum(
                    bool(
                        w.config["species"][a.species]
                        .get("reproduction", {})
                        .get("enabled")
                    )
                    and "reproduce" in w.config["species"][a.species]["allowed_actions"]
                    for a in w.animals.values()
                )
                self.assertGreater(active, 0)
                self.assertEqual(eligible.call_count, active, version)

    def test_direct_checks_and_standalone_refresh_see_same_tick_edits(self):
        w = breeding_world()
        a, mate = w.animals.values()
        w._prepare_mating_context()
        self.assertTrue(action_mask(w, a, ["rest", "reproduce"])[1])
        mate.health = 0
        self.assertFalse(action_mask(w, a, ["rest", "reproduce"])[1])
        mate.health = 30
        rules = w.config["species"][a.species]["reproduction"]
        rules["enabled"] = False
        self.assertFalse(action_mask(w, a, ["rest", "reproduce"])[1])
        rules["enabled"] = True
        w.animals.pop(mate.id)
        self.assertFalse(action_mask(w, a, ["rest", "reproduce"])[1])
        e = Experiment(algorithm_config("maskable_ppo"))
        with patch.object(
            e.world, "_prepare_mating_context", wraps=e.world._prepare_mating_context
        ) as prepare:
            e._refresh_context()
            self.assertEqual(prepare.call_count, 1)

    def test_shared_and_fresh_masks_match_without_rng_or_body_changes(self):
        w = breeding_world()
        for energy, health in ((60, 30), (0, 30), (60, 24)):
            for a in w.animals.values():
                a.energy, a.health = energy, health
            before = copy.deepcopy(w.checkpoint())
            context = w._prepare_mating_context()
            for a in w.animals.values():
                self.assertEqual(
                    action_mask(w, a, ["rest", "reproduce"], mating_context=context),
                    action_mask(w, a, ["rest", "reproduce"]),
                )
            self.assertEqual(w.checkpoint(), before)

    def test_action_mask_does_not_reuse_the_vision_limited_pair_flag(self):
        c = load(Path("environments/maskable_ppo.json"))
        w = World(c, 100)
        w.obstacles.clear()
        w.config["species"]["prey"]["body"]["vision_radius"] = 1
        w.config["species"]["prey"]["reproduction"]["mate_radius"] = 3
        w.animals = {
            0: Animal(
                0,
                "prey",
                2,
                2,
                0,
                60,
                40,
                age=1000,
                sex="female",
                reproduction_due_age=300,
            ),
            1: Animal(
                1,
                "prey",
                4,
                2,
                0,
                60,
                40,
                age=1000,
                sex="male",
                reproduction_due_age=300,
            ),
        }
        observations = w.observe_all()
        self.assertEqual(observations[0][-4], 0)
        self.assertTrue(
            action_mask(
                w, w.animals[0], ["rest", "reproduce"], mating_context=w._mate_context
            )[1]
        )

    def test_birth_phase_rechecks_energy_after_decision_costs(self):
        w = breeding_world()
        spec = w.config["species"]["prey"]
        spec["body"]["living_cost"] = 2
        spec["reproduction"]["min_energy"] = 10
        for a in w.animals.values():
            a.energy = 10
        context = w._prepare_mating_context()
        self.assertTrue(
            action_mask(w, w.animals[0], ["rest", "reproduce"], mating_context=context)[
                1
            ]
        )
        result = w.step({i: "reproduce" for i in w.animals})
        self.assertFalse(any(e.kind == "born" for e in result.events))
        self.assertTrue(
            any(
                e.kind == "mating_failed" and e.cause == "low_energy"
                for e in result.events
            )
        )

    def test_checkpoint_restore_has_independent_transient_context(self):
        e = Experiment(algorithm_config("maskable_ppo"))
        state = e.checkpoint()
        restored = Experiment.restore(state)
        self.assertIsNot(e.world._mate_context[0], restored.world._mate_context[0])
        self.assertIsNot(e.world._mate_context[3], restored.world._mate_context[3])
        self.assertEqual(e.checkpoint(), restored.checkpoint())
        e.step()
        restored.step()
        self.assertEqual(e.checkpoint(), restored.checkpoint())


if __name__ == "__main__":
    unittest.main()
