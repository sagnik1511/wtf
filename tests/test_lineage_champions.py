"""Pedigree deduplication, proven generations, independent pools and persistence."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ecosystem.brains import model_snapshot
from ecosystem.config import validate
from ecosystem.library import (
    ChampionBook,
    apply_brain_overrides,
    apply_champion_overrides,
    export_champions,
    load_initial_models,
)
from ecosystem.lineage import LineageBook
from ecosystem.runner import Experiment
from ecosystem.world import Animal
from tests.test_library import settings
from tests.test_simulation import config


def configured(kind="q_network"):
    c = settings(kind)
    for spec in c["species"].values():
        spec["champion_selection"] = {
            "criterion": "lineage",
            "lineage_min_age_steps": 3,
        }
    return c


def body(i, species="predator", age=3, kills=1, food=2, parents=(-1, -1), sex="female"):
    return Animal(
        i,
        species,
        1,
        1,
        0,
        40,
        30,
        age=age,
        kills=kills,
        energy_gained=food,
        parent_id=parents[0],
        second_parent_id=parents[1],
        sex=sex,
    )


class LineageTests(unittest.TestCase):
    def test_rank_only_preserves_family_ties_and_detached_metadata(self):
        for species in ("prey", "predator"):
            with self.subTest(species=species):
                book = LineageBook(configured())
                parents = [body(0, species), body(1, species)]
                child = body(2, species, parents=(0, 1))
                book.observe(parents + [child])
                rank, metadata = book.rank(child)
                self.assertEqual(metadata["founder_id"], 0)
                self.assertEqual(book.rank(child, include_metadata=False), (rank, None))
                saved = copy.deepcopy(metadata)
                parents[1].age += 10
                parents[1].kills += 5
                parents[1].energy_gained += 10
                book.observe(parents + [child])
                updated, family = book.rank(child)
                self.assertEqual(family["founder_id"], 1)
                self.assertEqual(
                    book.rank(child, include_metadata=False), (updated, None)
                )
                self.assertEqual(metadata, saved)

    def test_champions_materialize_lineage_metadata_only_for_winners(self):
        from unittest.mock import patch

        e = Experiment(configured())
        animals = tuple(e.world.animals.values())
        book = ChampionBook(e.config)
        with patch.object(book.lineage, "rank", wraps=book.lineage.rank) as rank:
            book.consider(animals, e.brains, {"tick": 0})
            full_calls = sum(
                call.kwargs.get("include_metadata", True)
                for call in rank.call_args_list
            )
            pools = sum(len(groups) for groups in book.entries.values())
            self.assertEqual(full_calls, pools)
            self.assertEqual(rank.call_count, len(animals) + pools)
            rank.reset_mock()
            book.consider(animals, e.brains, {"tick": 1})
            self.assertEqual(rank.call_count, len(animals))
            self.assertTrue(
                all(
                    call.kwargs == {"include_metadata": False}
                    for call in rank.call_args_list
                )
            )

    def test_fractional_food_accumulation_continues_exactly(self):
        import random

        rng = random.Random(19)
        animals = [body(0, "prey"), body(1, "prey"), body(2, "prey", parents=(0, 1))]
        original = LineageBook(configured())
        for _ in range(101):
            for a in animals:
                a.energy_gained += rng.random() / 7
                a.age += 1
            original.observe(animals)
        restored = LineageBook(
            configured(), json.loads(json.dumps(original.checkpoint()))
        )
        for a in animals:
            a.age += 1
            a.energy_gained += 0.1
        original.observe(animals)
        restored.observe(animals)
        self.assertEqual(original.checkpoint(), restored.checkpoint())
        self.assertEqual(original.rank(animals[-1]), restored.rank(animals[-1]))

    def test_shared_ancestors_count_once_and_dead_relatives_remain(self):
        c = configured()
        book = LineageBook(c)
        animals = [
            body(0, kills=2),
            body(1, kills=3),
            body(2, kills=5, parents=(0, 1)),
            body(3, kills=7, parents=(0, 1)),
            body(4, kills=11, parents=(2, 3)),
        ]
        book.observe(reversed(animals))
        rank, family = book.rank(animals[-1])
        self.assertEqual(rank[0], 2)
        self.assertEqual(book.nodes["4"]["roots"], [0, 1])
        self.assertEqual(family["founder_id"], 1)
        self.assertEqual(family["members"], 4)
        self.assertEqual(family["kills"], 26)  # 3 + 5 + 7 + 11, once each.
        self.assertEqual(family["viable_descendants"], 3)
        book.observe(animals)  # Re-observing cannot double counts.
        self.assertEqual(book.rank(animals[-1])[1], family)
        animals[-1].age += 1
        animals[-1].kills += 1
        book.observe([animals[-1]])  # All its relatives may already be dead.
        self.assertEqual(book.rank(animals[-1])[1]["kills"], 27)
        restored = LineageBook(c, json.loads(json.dumps(book.checkpoint())))
        self.assertEqual(restored.rank(animals[-1]), book.rank(animals[-1]))

    def test_newborn_and_passive_predator_cannot_displace_proven_hunter(self):
        book = LineageBook(configured())
        a, b = body(0, kills=10), body(1, kills=2)
        child = body(2, age=0, kills=0, parents=(0, 1))
        book.observe([a, b, child])
        self.assertLess(book.rank(child)[0], book.rank(a)[0])
        self.assertEqual(book.families[0]["viable_descendants"], 0)
        child.age = 3
        book.observe([child])
        self.assertFalse(book.rank(child)[1]["qualified"])
        self.assertEqual(book.families[0]["viable_descendants"], 1)
        child.kills = 1
        book.observe([child])
        self.assertGreater(book.rank(child)[0], book.rank(a)[0])

    def test_family_kills_break_generation_ties_and_prey_use_survival_and_food(self):
        book = LineageBook(configured())
        animals = [
            body(0, kills=20),
            body(1, kills=1),
            body(2, kills=2),
            body(3, kills=2),
            body(4, kills=1, parents=(0, 1)),
            body(5, kills=2, parents=(2, 3)),
        ]
        book.observe(animals)
        self.assertGreater(book.rank(animals[4])[0], book.rank(animals[5])[0])
        deer = [
            body(10, "prey", age=20, kills=0),
            body(11, "prey", age=3, kills=0),
            body(12, "prey", kills=0, food=0, parents=(10, 11)),
        ]
        book.observe(deer)
        self.assertFalse(book.rank(deer[-1])[1]["qualified"])
        deer[-1].energy_gained = 2
        book.observe([deer[-1]])
        self.assertTrue(book.rank(deer[-1])[1]["qualified"])
        self.assertEqual(book.rank(deer[-1])[1]["animal_steps"], 23)
        self.assertGreater(book.rank(deer[-1])[0], book.rank(deer[0])[0])

    def test_generic_species_sex_pools_old_modes_and_snapshot_isolation(self):
        c = configured()
        c["species"]["deer"] = c["species"].pop("prey")
        c["species"]["wolf"] = c["species"].pop("predator")
        e = Experiment(c)
        for species in c["species"]:
            for sex in ("female", "male"):
                animals = [body(i, species, sex=sex) for i in range(2)]
                book = ChampionBook(c)
                brain = next(iter(e.brains.values()))
                book.consider(animals, {0: brain, 1: brain}, {"tick": 3})
                pool = book.entries[species][sex]
                self.assertTrue(
                    {"health", "kills", "food", "lifespan", "lineage"} <= pool.keys()
                )
                self.assertEqual("balanced" in pool, species == "wolf")
                saved = copy.deepcopy(pool)
                book.lineage.families[0]["kills"] += 100
                self.assertEqual(pool, saved)

    def test_chaining_all_existing_algorithms_and_birth_generation_reset(self):
        for kind in (
            "q_network",
            "ppo",
            "a2c",
            "reinforce",
            "grpo",
            "recurrent_ppo",
            "qrdqn",
            "maskable_ppo",
            "mappo",
        ):
            # Advanced fixtures are supplied by their own configuration helper.
            if kind in ("recurrent_ppo", "qrdqn", "maskable_ppo", "mappo"):
                from tests.test_mutation import environment

                c = environment(kind, 0)
                for spec in c["species"].values():
                    spec["champion_selection"] = {
                        "criterion": "lineage",
                        "lineage_min_age_steps": 3,
                    }
            else:
                c = configured(kind)
            c["experiment"]["runs"] = 2
            with self.subTest(kind=kind):
                e = Experiment(c)
                while not e.finished_run:
                    e.step()
                expected = copy.deepcopy(e.champions.entries)
                e.next_run()
                for i, a in e.world.animals.items():
                    self.assertEqual(
                        model_snapshot(e.brains[i]),
                        expected[a.species][a.sex]["lineage"]["model"],
                    )
                    self.assertEqual(e.champions.lineage.nodes[str(i)]["generation"], 0)
                    self.assertEqual(e.champions.lineage.nodes[str(i)]["kills"], 0)

    def test_checkpoint_exact_resume_and_legacy_boundary(self):
        e = Experiment(configured("ppo"))
        for _ in range(5):
            e.step()
        state = json.loads(json.dumps(e.checkpoint()))
        restored = Experiment.restore(state)
        e.step()
        restored.step()
        self.assertEqual(e.checkpoint(), restored.checkpoint())
        state.pop("lineage")
        for groups in state["champions"].values():
            for pool in groups.values():
                pool.pop("lineage", None)
        legacy = Experiment.restore(state)
        legacy.step()
        self.assertTrue(legacy.champions.lineage.partial)
        for pool in legacy.champions.entries.values():
            for champions in pool.values():
                self.assertTrue(champions["lineage"]["lineage"]["partial_history"])

    def test_archive_loading_policy_matching_and_no_invention_from_old_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = configured()
            e = Experiment(c)
            e.step()
            export_champions(
                e, root / "source/champions/run_0001.json", root=root / "library"
            )
            selected = copy.deepcopy(c)
            apply_brain_overrides(
                selected, ["prey=lineage", "predator=lineage"], root / "source"
            )
            loaded = Experiment(selected)
            for s, groups in loaded.initial_models.items():
                for sex, item in groups.items():
                    self.assertEqual(
                        item["model"], e.champions.entries[s][sex]["lineage"]["model"]
                    )
            selected = copy.deepcopy(c)
            for spec in selected["species"].values():
                spec["initial_brain"] = {"selection": "best", "metric": "lineage"}
            self.assertTrue(load_initial_models(selected, root / "library"))
            selected["species"]["prey"]["champion_selection"][
                "lineage_min_age_steps"
            ] = 5
            with self.assertRaisesRegex(ValueError, "No compatible"):
                load_initial_models(selected, root / "library")
            for groups in e.champions.entries.values():
                for pool in groups.values():
                    pool.pop("lineage")
            export_champions(e, root / "old/champions/run_0001.json")
            with self.assertRaisesRegex(ValueError, "no prey lineage"):
                apply_brain_overrides(copy.deepcopy(c), ["prey=lineage"], root / "old")

    def test_selection_only_cli_overrides_and_validation(self):
        c = config()
        apply_champion_overrides(
            c, ["prey:male=food", "prey=lineage", "predator=lineage"]
        )
        self.assertEqual(
            c["species"]["prey"]["champion_selection_by_sex"],
            {"female": "lineage", "male": "food"},
        )
        self.assertNotIn("initial_brain", c["species"]["prey"])
        validate(c)
        for request in (
            "missing=lineage",
            "prey:unknown=lineage",
            "prey=balanced",
            "prey=nope",
        ):
            with self.assertRaises(ValueError):
                apply_champion_overrides(config(), [request])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "environment.json"
            path.write_text(json.dumps(config()))
            p = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ecosystem.cli",
                    str(path),
                    "--runs",
                    "2",
                    "--steps",
                    "3",
                    "--champion",
                    "prey=lineage",
                    "--champion",
                    "predator=lineage",
                    "--no-progress",
                    "--no-record",
                    "--output",
                    str(root / "out"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            summary = json.loads(
                next((root / "out").glob("*/summary.json")).read_text()
            )
            self.assertEqual(
                summary[1]["initialized_from"]["prey"]["female"]["criterion"],
                "lineage",
            )


if __name__ == "__main__":
    unittest.main()
