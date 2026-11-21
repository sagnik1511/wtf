"""Survival gates and lineage behavior under the legacy mating contract."""

import copy
from pathlib import Path
import unittest

from wtf.learning.brain import observation_inputs
from wtf.learning.brains import create_brain, load_model_weights, model_snapshot
from wtf.config import load
from wtf.storage.library import selection_criterion
from wtf.experiment.lineage import LineageBook
from wtf.experiment.runner import Experiment
from wtf.simulation.world import Animal, World


PRESET = Path("tests/fixtures/mating_v3.json")


def encounter(config):
    w = World(config, 100)
    w.obstacles.clear()
    w.resources.clear()
    w.animals = {
        0: Animal(
            0, "prey", 3, 3, 0, 60, config["species"]["prey"]["body"]["max_health"]
        ),
        1: Animal(
            1,
            "predator",
            3,
            4,
            0,
            80,
            config["species"]["predator"]["body"]["max_health"],
        ),
    }
    w.next_animal_id = 2
    return w


class PreySurvivalPresetTests(unittest.TestCase):
    def test_updated_networks_train_and_chain_matching_lineage_banks(self):
        c = load(PRESET)
        c["world"]["max_steps"] = 80
        c["experiment"].update(runs=2, world_seed_policy="increment")
        e = Experiment(c)
        while not e.finished_run:
            e.step()
        self.assertGreater(e.total_updates, 0)
        self.assertTrue(e.next_run())
        for species, groups in e.run_initial_models.items():
            for sex, entry in groups.items():
                self.assertIn(sex, ("female", "male"))
                self.assertEqual(entry["provenance"]["criterion"], "lineage")
                self.assertEqual(entry["provenance"]["sex"], sex)
                expected = 66 if species == "prey" else 42
                self.assertEqual(entry["model"]["signature"]["inputs"], expected)

    def test_single_hit_survives_and_next_step_escape_is_possible(self):
        current = load(PRESET)
        w = encounter(current)
        w.step({0: "rest", 1: "attack"})
        self.assertEqual(w.animals[0].health, 10)
        w.step({0: "move_forward", 1: "attack"})
        self.assertEqual(w.animals[0].position, (3, 2))
        self.assertEqual(w.animals[0].health, 10)
        self.assertEqual(w.kills.get("predator", 0), 0)
        legacy = copy.deepcopy(current)
        legacy["species"]["prey"]["body"]["max_health"] = 30
        legacy["species"]["prey"]["reproduction"]["health_threshold"] = 24
        baseline = encounter(legacy)
        baseline.step({0: "rest", 1: "attack"})
        self.assertNotIn(0, baseline.animals)

    def test_lingering_or_multiple_attackers_can_still_kill_prey(self):
        w = encounter(load(PRESET))
        w.step({0: "rest", 1: "attack"})
        w.step({0: "rest", 1: "attack"})
        self.assertNotIn(0, w.animals)
        w = encounter(load(PRESET))
        w.animals[2] = Animal(2, "predator", 2, 3, 1, 80, 55)
        w.next_animal_id = 3
        w.step({0: "rest", 1: "attack", 2: "attack"})
        self.assertNotIn(0, w.animals)
        self.assertEqual(w.kills["predator"], 1)

    def test_three_visible_threats_are_encoded_and_moving_target_refreshes(self):
        c = load(PRESET)
        w = encounter(c)
        w.animals[0].x, w.animals[0].y = 2, 5
        for i, x in enumerate((9, 10, 11), 1):
            w.animals[i] = Animal(i, "predator", x, 5, 0, 80, 55)
        w.next_animal_id = 4
        observation = w.observe_all()[0]
        self.assertEqual(len(observation), 66)
        # 8 base values + 3 food slots + 3 resource-eater slots, four values each.
        start = 8 + 3 * 4 * 2
        self.assertEqual([observation[start + 4 * i + 3] for i in range(3)], [1, 1, 1])
        w.animals[3].x = 20
        refreshed = w.observe_all()[0]
        self.assertEqual([refreshed[start + 4 * i + 3] for i in range(3)], [1, 1, 0])

    def test_lineage_qualifies_mature_fed_or_hunting_descendants(self):
        c = load(PRESET)
        for species in c["species"]:
            for sex in ("female", "male"):
                self.assertEqual(selection_criterion(c, species, sex=sex), "lineage")
            min_age = c["species"][species]["champion_selection"][
                "lineage_min_age_steps"
            ]
            parents = [
                Animal(
                    i,
                    species,
                    i + 1,
                    1,
                    0,
                    60,
                    40,
                    age=1500,
                    kills=20,
                    energy_gained=1000,
                )
                for i in range(2)
            ]
            child = Animal(
                2,
                species,
                3,
                1,
                0,
                60,
                40,
                age=min_age - 1,
                kills=1,
                energy_gained=10,
                parent_id=0,
                second_parent_id=1,
            )
            book = LineageBook(c)
            book.observe(parents + [child])
            self.assertFalse(book.rank(child)[1]["qualified"])
            self.assertLess(book.rank(child)[0], book.rank(parents[0])[0])
            child.age += 1
            book.observe(parents + [child])
            self.assertTrue(book.rank(child)[1]["qualified"])
            self.assertGreater(book.rank(child)[0], book.rank(parents[0])[0])

    def test_maturity_radius_and_health_fraction_are_preserved(self):
        c = load(PRESET)
        baseline = load(Path("environments/maskable_ppo.json"))
        for species, spec in c["species"].items():
            original = baseline["species"][species]
            self.assertEqual(spec["reproduction"]["mate_radius"], 2)
            self.assertEqual(
                spec["reproduction"]["min_age_steps"],
                original["reproduction"]["min_age_steps"],
            )
            self.assertAlmostEqual(
                spec["reproduction"]["health_threshold"] / spec["body"]["max_health"],
                0.8,
            )
        self.assertLess(
            c["species"]["prey"]["body"]["max_health"],
            c["species"]["predator"]["body"]["max_health"],
        )
        self.assertTrue(c["experiment"]["guard_lineage_chaining"])

    def test_old_prey_weights_are_rejected_and_predator_contract_is_preserved(self):
        current = load(PRESET)
        for species in current["species"]:
            spec = current["species"][species]
            template = current["brains"][spec["brain"]]
            self.assertEqual(
                observation_inputs(template), 66 if species == "prey" else 42
            )
            previous = copy.deepcopy(template)
            previous["observation_top_k"] = 1
            actions = spec["allowed_actions"]
            old_brain = create_brain(previous, actions, 1, 2)
            new_brain = create_brain(template, actions, 1, 2)
            if species == "prey":
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_model_weights(new_brain, model_snapshot(old_brain))
            else:
                load_model_weights(new_brain, model_snapshot(old_brain))


if __name__ == "__main__":
    unittest.main()
