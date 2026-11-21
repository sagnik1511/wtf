"""Species/sex champion identity, warm starts, chaining and legacy boundaries."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from wtf.learning.brains import model_snapshot
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.config import validate
from wtf.storage.library import (
    ChampionBook,
    apply_brain_overrides,
    catalog,
    export_champions,
    read_artifact,
)
from wtf.experiment.runner import Experiment
from tests.test_mutation import KINDS, environment


def configuration(kind="q_network"):
    c = environment(kind, 0)
    c["species"]["predator"]["initial_population"] = 2
    c["experiment"].update(runs=2, chain_best_brains=True)
    for spec in c["species"].values():
        spec["reproduction"]["enabled"] = False
    return c


def source(folder, kind="q_network"):
    e = Experiment(configuration(kind))
    for i, a in e.world.animals.items():
        a.age = 10 + i
        a.kills = i
        a.energy_gained = i * 3
    e.champions.consider(e.world.animals.values(), e.brains, e._champion_source())
    archive = export_champions(e, folder / "champions/run_0001.json")
    return e, archive


class SexChampionTests(unittest.TestCase):
    def test_four_groups_rank_independently_and_export_only_sex_paths(self):
        with tempfile.TemporaryDirectory() as d:
            e, archive = source(Path(d))
            manifest = json.loads((archive / "manifest.json").read_text())
            self.assertEqual(manifest["library_version"], 2)
            self.assertEqual(len(manifest["entries"]), 22)
            self.assertEqual(
                {(x["species"], x["sex"]) for x in manifest["entries"]},
                {
                    ("prey", "female"),
                    ("prey", "male"),
                    ("predator", "female"),
                    ("predator", "male"),
                },
            )
            for entry in manifest["entries"]:
                artifact = read_artifact(archive / entry["file"])
                self.assertEqual(artifact["source"]["sex"], entry["sex"])
                self.assertEqual(
                    Path(entry["file"]).parts[:2], (entry["species"], entry["sex"])
                )
                self.assertEqual(artifact["brain_artifact_version"], 3)
            self.assertFalse((archive / "prey/lifespan.json").exists())

    def test_automatic_source_loading_and_chain_match_sex_for_every_algorithm(self):
        for kind in KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                old, _ = source(root, kind)
                c = configuration(kind)
                apply_brain_overrides(c, source_experiment=root)
                e = Experiment(c)
                for i, a in e.world.animals.items():
                    metric = c["species"][a.species]["champion_selection_by_sex"][a.sex]
                    self.assertEqual(
                        model_snapshot(e.brains[i]),
                        old.champions.entries[a.species][a.sex][metric]["model"],
                    )
                while not e.finished_run:
                    e.step()
                expected = {
                    (s, sex): copy.deepcopy(
                        groups[c["species"][s]["champion_selection_by_sex"][sex]][
                            "model"
                        ]
                    )
                    for s, bank in e.champions.entries.items()
                    for sex, groups in bank.items()
                }
                export_champions(e, root / "next/champions/run_0001.json")
                e.next_run()
                for i, a in e.world.animals.items():
                    self.assertEqual(
                        model_snapshot(e.brains[i]), expected[a.species, a.sex]
                    )
                    self.assertEqual(
                        e.run_initial_models[a.species][a.sex]["provenance"]["sex"],
                        a.sex,
                    )

    def test_narrow_cli_selectors_override_broad_selectors_in_any_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            e, _ = source(root)
            for requests in (
                ["prey:male=kills", "prey=food"],
                ["prey=food", "prey:male=kills"],
            ):
                c = configuration()
                apply_brain_overrides(c, requests, root)
                a = Experiment(c)
                self.assertEqual(
                    c["species"]["prey"]["champion_selection_by_sex"],
                    {"female": "food", "male": "kills"},
                )
                for i, body in a.world.animals.items():
                    if body.species == "prey":
                        metric = "kills" if body.sex == "male" else "food"
                        self.assertEqual(
                            model_snapshot(a.brains[i]),
                            e.champions.entries["prey"][body.sex][metric]["model"],
                        )

    def test_sex_specific_fresh_preserves_other_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source(root)
            c = configuration()
            apply_brain_overrides(c, ["prey:male=fresh"], root)
            e = Experiment(c)
            self.assertIn("female", e.initial_models["prey"])
            self.assertNotIn("male", e.initial_models["prey"])
            female = e.initial_models["prey"]["female"]["model"]
            self.assertNotEqual(
                model_snapshot(
                    next(
                        e.brains[i]
                        for i, a in e.world.animals.items()
                        if a.species == "prey" and a.sex == "male"
                    )
                ),
                female,
            )

    def test_mismatched_sex_path_is_rejected_and_broad_path_only_loads_matching_sex(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, archive = source(root)
            c = configuration()
            apply_brain_overrides(c, [f"prey:male={archive}/prey/female/lifespan.json"])
            with self.assertRaisesRegex(ValueError, "sex does not match"):
                Experiment(c)
            c = configuration()
            apply_brain_overrides(c, [f"prey={archive}/prey/female/lifespan.json"])
            self.assertEqual(set(Experiment(c).initial_models["prey"]), {"female"})

    def test_missing_sex_uses_fresh_automatically_but_explicit_request_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = configuration()
            c["species"]["predator"]["initial_population"] = 1
            old = Experiment(c)
            export_champions(old, root / "champions/run_0001.json")
            c = configuration()
            apply_brain_overrides(c, source_experiment=root)
            self.assertIsNone(c["species"]["predator"]["initial_brains"]["male"])
            self.assertEqual(set(Experiment(c).initial_models["predator"]), {"female"})
            with self.assertRaisesRegex(ValueError, "predator:male"):
                apply_brain_overrides(configuration(), ["predator:male=lifespan"], root)

    def test_library_best_and_latest_filter_by_sex(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            e = Experiment(configuration())
            for i, a in e.world.animals.items():
                a.age = 100 if a.sex == "male" else 20
            e.champions.consider(
                e.world.animals.values(), e.brains, e._champion_source()
            )
            export_champions(e, root / "index.json", root=root / "library")
            e = Experiment(configuration())
            export_champions(e, root / "index2.json", root=root / "library")
            c = configuration()
            apply_brain_overrides(
                c, ["prey:male=best:lifespan", "prey:female=latest:lifespan"]
            )
            loaded = Experiment(c, library_root=root / "library")
            self.assertEqual(
                loaded.initial_models["prey"]["male"]["provenance"]["score"], 100
            )
            self.assertEqual(
                loaded.initial_models["prey"]["female"]["provenance"]["score"], 0
            )
            self.assertTrue(
                all(x["sex"] in ("female", "male") for x in catalog(root / "library"))
            )

    def test_checkpoints_keep_group_provenance_after_source_deleted(self):
        import shutil

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source(root / "source")
            c = configuration("ppo")
            # Build a compatible PPO source.
            source(root / "ppo", "ppo")
            apply_brain_overrides(c, source_experiment=root / "ppo")
            a = Experiment(c)
            a.step()
            path = root / "checkpoint.bin.xz"
            a.write_checkpoint(path)
            b = Experiment.restore(read_checkpoint(path))
            shutil.rmtree(root / "ppo")
            while not a.done:
                a.step()
                b.step()
            self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_legacy_untyped_archive_can_seed_both_but_known_sex_never_crosses(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, archive = source(root)
            typed = read_artifact(archive / "prey/female/lifespan.json")
            c = configuration()
            c["species"]["prey"]["initial_brain"] = {
                "path": str(archive / "prey/female/lifespan.json")
            }
            self.assertEqual(set(Experiment(c).initial_models["prey"]), {"female"})
            legacy = dict(typed, brain_artifact_version=1)
            legacy["source"] = dict(typed["source"])
            legacy["source"].pop("sex")
            path = root / "legacy.json"
            path.write_text(json.dumps(legacy))
            c["species"]["prey"]["initial_brain"] = {"path": str(path)}
            loaded = Experiment(c)
            self.assertEqual(set(loaded.initial_models["prey"]), {"female", "male"})

    def test_legacy_champion_book_keeps_only_recorded_sex_winners(self):
        e = Experiment(configuration())
        legacy = {
            s: {k: copy.deepcopy(v) for k, v in groups["female"].items()}
            for s, groups in e.champions.entries.items()
        }
        book = ChampionBook(e.config, legacy)
        self.assertEqual(set(book.entries["prey"]), {"female"})
        for groups in legacy.values():
            for entry in groups.values():
                entry["source"].pop("sex")
        self.assertEqual(ChampionBook(e.config, legacy).entries, {})

    def test_real_version_one_index_resolves_untyped_legacy_models(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, archive = source(root / "new")
            old = root / "legacy/champions/run_0001"
            old.mkdir(parents=True)
            manifest = json.loads((archive / "manifest.json").read_text())
            entries = []
            for entry in manifest["entries"]:
                if entry["sex"] != "female":
                    continue
                item = copy.deepcopy(entry)
                item.pop("sex")
                item["source"].pop("sex")
                item["file"] = item["species"] + "/" + item["criterion"] + ".json"
                artifact = read_artifact(archive / entry["file"])
                artifact["brain_artifact_version"] = 1
                artifact["source"].pop("sex")
                target = old / item["file"]
                target.parent.mkdir(exist_ok=True)
                target.write_text(json.dumps(artifact))
                entries.append(item)
            index = dict(
                manifest, library_version=1, entries=entries, model_directory="run_0001"
            )
            (old.parent / "run_0001.json").write_text(json.dumps(index))
            c = configuration()
            apply_brain_overrides(c, source_experiment=root / "legacy")
            loaded = Experiment(c)
            for species in ("prey", "predator"):
                self.assertEqual(
                    set(loaded.initial_models[species]), {"female", "male"}
                )
                self.assertEqual(
                    loaded.initial_models[species]["male"]["model"],
                    loaded.initial_models[species]["female"]["model"],
                )

    def test_saved_artifact_ancestry_follows_each_group(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, archive = source(root / "source")
            old = json.loads((archive / "manifest.json").read_text())
            ids = {
                (x["species"], x["sex"], x["criterion"]): x["artifact_id"]
                for x in old["entries"]
            }
            c = configuration()
            # Exercise explicit per-episode publication ancestry; strict mode
            # retains imported identities when no improvement has occurred.
            c["experiment"]["promote_only_improved"] = False
            apply_brain_overrides(c, source_experiment=root / "source")
            e = Experiment(c)
            target = export_champions(e, root / "next/champions/run_0001.json")
            new = json.loads((target / "manifest.json").read_text())
            for entry in new["entries"]:
                metric = c["species"][entry["species"]]["champion_selection_by_sex"][
                    entry["sex"]
                ]
                artifact = json.loads((target / entry["file"]).read_text())
                self.assertEqual(
                    artifact["parent_artifact"]["artifact_id"],
                    ids[entry["species"], entry["sex"], metric],
                )
                self.assertEqual(artifact["parent_artifact"]["sex"], entry["sex"])

    def test_config_rejects_unknown_sexes_and_invalid_per_sex_requests(self):
        for field, value in [
            ("initial_brains", {"other": None}),
            ("initial_brains", {"male": {"path": ""}}),
            ("champion_selection_by_sex", {"male": "unknown"}),
            ("champion_selection_by_sex", {"other": "kills"}),
            ("champion_selection_by_sex", {"male": "balanced"}),
        ]:
            c = configuration()
            c["species"]["prey"][field] = value
            with self.assertRaises(ValueError):
                validate(c)
        for request in ("prey:other=food", "prey:male=", "missing:male=fresh"):
            with self.assertRaises(ValueError):
                apply_brain_overrides(configuration(), [request])

    def test_cli_records_each_sex_source_and_can_select_only_male(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source(root / "source")
            c = configuration()
            path = root / "config.json"
            path.write_text(json.dumps(c))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wtf.cli",
                    str(path),
                    "--runs",
                    "1",
                    "--steps",
                    "2",
                    "--brains-from",
                    str(root / "source"),
                    "--brain",
                    "prey:male=kills",
                    "--no-progress",
                    "--output",
                    str(root / "results"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            folder = next((root / "results").iterdir())
            metadata = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(
                metadata["initialized_from"]["prey"]["male"]["criterion"], "kills"
            )
            self.assertEqual(
                metadata["initialized_from"]["prey"]["female"]["criterion"], "lifespan"
            )
            self.assertTrue(
                (folder / "champions/run_0001/prey/male/kills.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
