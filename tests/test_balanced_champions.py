"""Combined survival/hunting selection, old archives, and chaining contracts."""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
from pathlib import Path
import tempfile
import unittest

from wtf.learning.brains import model_snapshot
from wtf.config import load, validate
from wtf.storage.library import (
    ChampionBook,
    balanced_policy,
    balanced_rank,
    export_champions,
    apply_brain_overrides,
)
from wtf.experiment.runner import Experiment
from tests.test_library import settings
from tests.test_simulation import config


def balanced_config():
    c = config()
    c["species"]["predator"]["champion_selection"] = {
        "criterion": "balanced",
        "survival_steps_per_kill": 300,
        "max_survival_bonus": 2,
    }
    return c


class BalancedChampionsTests(unittest.TestCase):
    def test_score_rewards_both_caps_bonus_and_does_not_reward_passive_survival(self):

        # Keep the small synchronous numeric callback adjacent to the expression it
        # parameterizes.
        # pylint: disable=unnecessary-lambda-assignment

        policy = balanced_policy(balanced_config(), "predator")
        rank = lambda age, kills: balanced_rank(age, kills, 0, 55, policy)
        self.assertEqual(rank(600, 8)[0], 10)
        self.assertGreater(rank(600, 8), rank(100, 9))
        self.assertGreater(rank(100, 10), rank(600, 8))
        self.assertEqual(rank(10000, 0)[0], 0)
        self.assertGreater(rank(1, 1), rank(10000, 0))
        self.assertEqual(rank(10000, 1)[0], 3)
        self.assertGreater(
            rank(300, 3), rank(600, 2)
        )  # Equal scores prefer more kills.

    def test_champion_snapshots_store_scoring_policy_and_remain_independent(self):
        e = Experiment(balanced_config())
        book = ChampionBook(e.config)
        a = copy.deepcopy(e.world.animals[3])
        b = copy.deepcopy(a)
        a.id = 3
        b.id = 99
        a.kills, a.age = 8, 600
        b.kills, b.age = 9, 100
        brains = {3: e.brains[3], 99: e.brains[3]}
        book.consider([a, b], brains, {"tick": 600})
        self.assertEqual(
            book.entries["predator"]["female"]["balanced"]["source"]["animal_id"], 3
        )
        self.assertEqual(
            book.entries["predator"]["female"]["kills"]["source"]["animal_id"], 99
        )
        candidate = book.entries["predator"]["female"]["balanced"]
        self.assertEqual(candidate["score"], 10)
        self.assertEqual(candidate["selection_policy"]["max_survival_bonus"], 2)
        saved = copy.deepcopy(candidate)
        e.brains[3].weights["b2"][0] += 10
        self.assertEqual(candidate, saved)
        self.assertNotIn("balanced", e.champions.entries["prey"])

    def test_chaining_prefers_configured_balanced_criterion_all_algorithms(self):
        for kind in ("q_network", "ppo", "a2c", "reinforce", "grpo"):
            with self.subTest(kind=kind):
                c = settings(kind)
                c["species"]["predator"]["champion_selection"] = {
                    "criterion": "balanced"
                }
                e = Experiment(c)
                while not e.finished_run:
                    e.step()
                expected = copy.deepcopy(
                    e.champions.entries["predator"]["female"]["balanced"]["model"]
                )
                e.next_run()
                for i, a in e.world.animals.items():
                    if a.species == "predator":
                        self.assertEqual(model_snapshot(e.brains[i]), expected)
                self.assertEqual(
                    e.run_initial_models["predator"]["female"]["provenance"][
                        "criterion"
                    ],
                    "balanced",
                )

    def test_old_archives_rescore_available_models_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e = Experiment(config())
            while not e.finished_run:
                e.step()
            # Simulate an archive created before combined scoring existed.
            e.champions.entries["predator"]["female"].pop("balanced")
            export_champions(e, root / "source/champions/run_0001.json")
            files = {
                p: p.read_bytes() for p in (root / "source").rglob("*") if p.is_file()
            }
            c = balanced_config()
            apply_brain_overrides(c, source_experiment=root / "source")
            a = Experiment(c)
            self.assertEqual(
                a.initial_models["predator"]["female"]["provenance"][
                    "chaining_criterion"
                ],
                "balanced",
            )
            while not a.finished_run:
                a.step()
            expected = copy.deepcopy(
                a.champions.entries["predator"]["female"]["balanced"]["model"]
            )
            a.next_run()
            for i, animal in a.world.animals.items():
                if animal.species == "predator":
                    self.assertEqual(model_snapshot(a.brains[i]), expected)
            for p, data in files.items():
                self.assertEqual(p.read_bytes(), data)

    def test_new_archives_can_be_loaded_as_balanced_and_explicit_kills_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e = Experiment(balanced_config())
            export_champions(e, root / "source/champions/run_0001.json")
            artifact = json.loads(
                (
                    root / "source/champions/run_0001/predator/female/balanced.json"
                ).read_text()
            )
            self.assertEqual(
                artifact["selection_policy"], balanced_policy(e.config, "predator")
            )
            for metric in ("balanced", "kills"):
                c = balanced_config()
                apply_brain_overrides(c, [f"predator={metric}"], root / "source")
                a = Experiment(c)
                self.assertEqual(
                    a.initial_models["predator"]["female"]["provenance"]["criterion"],
                    metric,
                )
                self.assertEqual(
                    c["species"]["predator"]["champion_selection_by_sex"]["female"],
                    metric,
                )

    def test_balanced_library_selection_requires_compatible_scoring_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e = Experiment(balanced_config())
            export_champions(e, root / "index.json", root=root / "library")
            c = balanced_config()
            c["species"]["predator"]["initial_brain"] = {
                "selection": "best",
                "metric": "balanced",
            }
            Experiment(c, library_root=root / "library")
            c["species"]["predator"]["champion_selection"]["max_survival_bonus"] = 3
            with self.assertRaisesRegex(ValueError, "No compatible balanced"):
                Experiment(c, library_root=root / "library")

    def test_invalid_policies_and_non_predator_combined_selection_rejected(self):
        for field, value in [
            ("criterion", "other"),
            ("survival_steps_per_kill", 0),
            ("survival_steps_per_kill", True),
            ("max_survival_bonus", -1),
            ("max_survival_bonus", float("nan")),
        ]:
            c = balanced_config()
            c["species"]["predator"]["champion_selection"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate(c)
        c = balanced_config()
        c["species"]["prey"]["champion_selection"] = {"criterion": "balanced"}
        with self.assertRaisesRegex(ValueError, "predator"):
            validate(c)
        validate(load(Path(__file__).resolve().parents[1] / "environments/grpo.json"))

    def test_checkpoint_continuation_preserves_balanced_selection(self):

        # This self-contained regression intentionally repeats reference/setup data.
        # pylint: disable=duplicate-code

        a = Experiment(balanced_config())
        a.step()
        b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
        while not a.done:
            a.step()
            b.step()
        self.assertEqual(a.checkpoint(), b.checkpoint())
