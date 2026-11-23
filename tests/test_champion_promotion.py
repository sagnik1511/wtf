"""Strict best-so-far promotion and portable, deduplicated model storage."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
import lzma
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from wtf.learning.brains import model_snapshot
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.config import validate
from wtf.storage.library import apply_brain_overrides, export_champions, read_artifact
from wtf.experiment.runner import Experiment
from tests.test_mutation import KINDS, environment
from tests.test_simulation import config


def configuration():
    c = config()
    c["experiment"].update(runs=4)
    for spec in c["species"].values():
        spec["champion_selection"] = {"criterion": "lifespan"}
    return c


def establish_records(e):
    for animal in e.world.animals.values():
        animal.age = 100
        animal.kills = 5
        animal.energy_gained = 10000
    e.champions.consider(e.world.animals.values(), e.brains, e._champion_source())
    e.finished_run = True


class ChampionPromotionTests(unittest.TestCase):
    def test_default_strict_retains_weights_and_generation_across_worse_runs(self):
        e = Experiment(configuration())
        self.assertTrue(e.improvement_only)
        establish_records(e)
        original = copy.deepcopy(e.champions.entries)
        e.next_run()
        provenance = copy.deepcopy(e.run_initial_models)
        while not e.finished_run:
            e.step()
        for species, groups in original.items():
            for sex, entries in groups.items():
                self.assertEqual(
                    e.champions.entries[species][sex]["lifespan"], entries["lifespan"]
                )
        e.next_run()
        for i, animal in e.world.animals.items():
            self.assertEqual(
                model_snapshot(e.brains[i]),
                original[animal.species][animal.sex]["lifespan"]["model"],
            )
            self.assertEqual(e.brains[i].updates, 0)
        self.assertEqual(e.run_initial_models, provenance)

    def test_independent_groups_promote_only_strictly_better_ranks(self):
        e = Experiment(configuration())
        establish_records(e)
        e.next_run()
        old = copy.deepcopy(e.champions.entries)
        animal = e.world.animals[0]
        animal.age = 100
        animal.kills = 5
        animal.energy_gained = 10000
        e.brains[animal.id].weights["b2"][0] += 0.25
        e.champions.consider([animal], e.brains, e._champion_source())
        # An exactly tied rank must not swap in differently trained weights.
        self.assertEqual(e.champions.entries, old)
        animal.age = 101
        e.champions.consider([animal], e.brains, e._champion_source())
        group = e.champions.entries[animal.species][animal.sex]
        self.assertEqual(
            group["lifespan"]["model"], model_snapshot(e.brains[animal.id])
        )
        for species, groups in old.items():
            for sex, entries in groups.items():
                if (species, sex) != (animal.species, animal.sex):
                    self.assertEqual(e.champions.entries[species][sex], entries)

    def test_archives_retain_identity_and_write_no_new_weights_for_ties(self):

        # Keep setup, action and assertions together as one regression scenario.
        # pylint: disable=too-many-locals

        e = Experiment(configuration())
        establish_records(e)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = export_champions(e, root / "champions/run_0001.json")
            blobs = {
                p.name: p.read_bytes() for p in (root / "champions/models").iterdir()
            }
            modified_times = {
                p.name: p.stat().st_mtime_ns
                for p in (root / "champions/models").iterdir()
            }
            ids = copy.deepcopy(e.champion_artifacts)
            e.next_run()
            e.finished_run = True
            second = export_champions(e, root / "champions/run_0002.json")
            self.assertEqual(
                blobs,
                {p.name: p.read_bytes() for p in (root / "champions/models").iterdir()},
            )
            self.assertEqual(
                modified_times,
                {
                    p.name: p.stat().st_mtime_ns
                    for p in (root / "champions/models").iterdir()
                },
            )
            self.assertFalse((second / "models").exists())
            manifest = json.loads((second / "manifest.json").read_text())
            self.assertTrue(all(not entry["improved"] for entry in manifest["entries"]))
            for species, groups in ids.items():
                for sex, entries in groups.items():
                    for criterion, prior in entries.items():
                        current = e.champion_artifacts[species][sex][criterion]
                        self.assertEqual(prior["artifact_id"], current["artifact_id"])
                        self.assertEqual(
                            prior["model_generation"], current["model_generation"]
                        )
                        a = read_artifact(first / species / sex / (criterion + ".json"))
                        b = read_artifact(
                            second / species / sex / (criterion + ".json")
                        )
                        self.assertEqual(a["model"], b["model"])

    def test_warm_start_imports_rank_and_survives_source_removal(self):
        e = Experiment(configuration())
        establish_records(e)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export_champions(e, root / "source/champions/run_0001.json")
            c = configuration()
            apply_brain_overrides(c, source_experiment=root / "source")
            new = Experiment(c)
            imported = copy.deepcopy(new.run_initial_models)
            shutil.rmtree(root / "source")
            while not new.finished_run:
                new.step()
            archive = export_champions(new, root / "next/champions/run_0001.json")
            retained = read_artifact(archive / "prey/female/lifespan.json")
            self.assertEqual(retained["training_config"], e.config)
            new.next_run()
            for species, groups in imported.items():
                for sex, initial in groups.items():
                    self.assertEqual(
                        new.run_initial_models[species][sex]["model"], initial["model"]
                    )
                    self.assertEqual(
                        new.run_initial_models[species][sex]["provenance"][
                            "model_generation"
                        ],
                        initial["provenance"]["model_generation"],
                    )

    def test_no_chain_still_restarts_initial_weights_and_keeps_archive_records(self):
        c = configuration()
        c["experiment"]["chain_best_brains"] = False
        e = Experiment(c)
        initial = model_snapshot(e.brains[0])
        establish_records(e)
        best = copy.deepcopy(e.champions.entries)
        e.next_run()
        self.assertEqual(model_snapshot(e.brains[0]), initial)
        self.assertEqual(e.champions.entries, best)

    def test_binary_checkpoint_retains_best_records_for_all_algorithms(self):
        for kind in KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                c = environment(kind, 0)
                e = Experiment(c)
                while not e.finished_run:
                    e.step()
                e.next_run()
                path = Path(folder) / "state.bin.xz"
                e.write_checkpoint(path)
                restored = Experiment.restore(read_checkpoint(path))
                self.assertTrue(restored.improvement_only)
                while not e.done:
                    e.step()
                    restored.step()
                self.assertEqual(e.checkpoint(), restored.checkpoint())

    def test_legacy_resume_and_explicit_opt_out(self):
        e = Experiment(configuration())
        saved = e.checkpoint()
        saved.pop("improvement_only")
        self.assertFalse(Experiment.restore(saved).improvement_only)
        c = configuration()
        c["experiment"]["promote_only_improved"] = False
        e = Experiment(c)
        establish_records(e)
        e.next_run()
        self.assertTrue(
            all(
                x["source"]["run"] == 2
                for groups in e.champions.entries.values()
                for entries in groups.values()
                for x in entries.values()
            )
        )
        c["experiment"]["promote_only_improved"] = "false"
        with self.assertRaisesRegex(ValueError, "promote_only_improved"):
            validate(c)

    def test_float64_signed_zero_legacy_json_and_corruption(self):
        e = Experiment(configuration())
        model = e.champions.entries["prey"]["female"]["lifespan"]["model"]
        model["precision"] = "float64"
        model["parameters"]["w1"][:2] = [-0.0, 1e-300]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = export_champions(e, root / "champions/run_0001.json")
            path = archive / "prey/female/lifespan.json"
            artifact = read_artifact(path)
            self.assertEqual(artifact["model"], model)
            self.assertEqual(
                math.copysign(1, artifact["model"]["parameters"]["w1"][0]), -1
            )
            self.assertEqual(artifact["model"]["parameters"]["w1"][1], 1e-300)
            legacy = copy.deepcopy(artifact)
            legacy["brain_artifact_version"] = 2
            legacy_path = root / "legacy.json"
            legacy_path.write_text(json.dumps(legacy))
            self.assertEqual(read_artifact(legacy_path)["model"], model)
            metadata = json.loads(path.read_text())
            reference = copy.deepcopy(metadata)
            reference["model"]["weights_file"] = "../../outside.bin.xz"
            path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "reference"):
                read_artifact(path)
            path.write_text(json.dumps(metadata))
            blob = root / "champions/models" / (metadata["model"]["sha256"] + ".bin.xz")
            with lzma.open(blob, "wb") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_artifact(path)

    def test_changing_one_criterion_records_only_that_improvement(self):
        e = Experiment(configuration())
        establish_records(e)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export_champions(e, root / "champions/run_0001.json")
            entry = e.champions.entries["prey"]["female"]["lifespan"]
            entry["rank"][0] += 1
            entry["score"] += 1
            # Same weights, better score: metadata changes without another blob.
            before = set((root / "champions/models").iterdir())
            archive = export_champions(e, root / "champions/run_0002.json")
            self.assertEqual(before, set((root / "champions/models").iterdir()))
            improved = [
                x
                for x in json.loads((archive / "manifest.json").read_text())["entries"]
                if x["improved"]
            ]
            self.assertEqual(
                [(x["species"], x["sex"], x["criterion"]) for x in improved],
                [("prey", "female", "lifespan")],
            )
