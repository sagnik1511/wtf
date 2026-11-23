"""Species-scoped plateau patience, finalization, and exact continuation."""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from wtf.storage.checkpoint_io import read_checkpoint
from wtf.config import validate
from wtf.experiment.early_stopping import EarlyStopping
from wtf.experiment.runner import Experiment
from tests.test_mutation import KINDS, environment
from tests.test_simulation import config


def settings(patience=2, species=None):
    c = config()
    c["experiment"].update(runs=10, chain_best_brains=False, world_seed_policy="fixed")
    c["world"]["max_steps"] = 3
    for spec in c["species"].values():
        spec["champion_selection"] = {"criterion": "lifespan"}
    c["experiment"]["early_stopping"] = {"patience_runs": patience}
    if species is not None:
        c["experiment"]["early_stopping"]["species"] = species
    return c


def records(prey=1, predator=1):
    return {
        "prey": {"female": {"lifespan": {"rank": [prey]}}},
        "predator": {"male": {"lifespan": {"rank": [predator]}}},
    }


class EarlyStoppingTests(unittest.TestCase):
    def test_baseline_exact_patience_ties_regression_and_duplicate_finish(self):
        stop = EarlyStopping(settings(), "train")
        stop.finish_run(1, records(5, 5))
        self.assertEqual(stop.stagnant_runs, 0)
        stop.finish_run(2, records(5, 5))
        stop.finish_run(2, records(5, 5))
        self.assertEqual(stop.stagnant_runs, 1)
        self.assertFalse(stop.stopped)
        stop.finish_run(3, records(3, 4))
        self.assertTrue(stop.stopped)
        self.assertEqual(stop.reason["run"], 3)
        self.assertEqual(stop.best["prey"]["female"]["rank"], [5])

    def test_any_monitored_species_or_sex_improvement_resets_patience(self):
        stop = EarlyStopping(settings(), "train")
        stop.finish_run(1, records())
        stop.finish_run(2, records())
        stop.finish_run(3, records(predator=2))
        self.assertEqual(stop.stagnant_runs, 0)
        current = records(predator=2)
        current["prey"]["male"] = {"lifespan": {"rank": [1]}}
        stop.finish_run(4, current)
        self.assertEqual(stop.stagnant_runs, 0)
        stop.finish_run(5, current)
        current["prey"]["male"]["lifespan"]["rank"] = [2]
        stop.finish_run(6, current)
        self.assertFalse(stop.stopped)
        self.assertEqual(stop.stagnant_runs, 0)

    def test_species_filter_ignores_other_species_and_unused_criteria(self):
        for species in ("prey", "predator"):
            with self.subTest(species=species):
                stop = EarlyStopping(settings(1, [species]), "train")
                stop.finish_run(1, records())
                other = "predator" if species == "prey" else "prey"
                current = records(**{other: 99})
                sex = next(iter(current[species]))
                current[species][sex]["health"] = {"rank": [1000]}
                stop.finish_run(2, current)
                self.assertTrue(stop.stopped)

    def test_custom_species_names_and_sex_specific_selected_criterion(self):
        c = settings(1)
        c["species"]["deer"] = c["species"].pop("prey")
        c["species"]["wolf"] = c["species"].pop("predator")
        c["species"]["wolf"]["champion_selection_by_sex"] = {"female": "kills"}
        c["experiment"]["early_stopping"]["species"] = ["wolf"]
        stop = EarlyStopping(c, "train")
        first = {
            "wolf": {"female": {"kills": {"rank": [3]}, "lifespan": {"rank": [5]}}}
        }
        stop.finish_run(1, first)
        first["wolf"]["female"]["lifespan"]["rank"] = [100]
        stop.finish_run(2, first)
        self.assertTrue(stop.stopped)

    def test_warm_start_seed_and_criterion_fallback(self):
        c = settings(1, ["prey"])
        c["species"]["prey"].pop("champion_selection")
        initial = {"prey": {"female": {"provenance": {"criterion": "food"}}}}
        stop = EarlyStopping(c, "train", initial)
        entries = {"prey": {"female": {"food": {"rank": [100]}}}}
        stop.seed(entries)
        self.assertEqual(stop.last_run, 0)
        stop.finish_run(1, entries)
        self.assertTrue(stop.stopped)

    def test_disabled_and_non_training_modes_do_not_stop(self):
        for mode, patience in (("train", 0), ("evaluate", 1), ("random", 1)):
            stop = EarlyStopping(settings(patience), mode)
            for run in range(1, 5):
                stop.finish_run(run, records())
            self.assertFalse(stop.stopped)
            self.assertEqual(stop.stagnant_runs, 0)

    def test_config_rejects_invalid_patience_and_species(self):
        for bad in (-1, True, 1.5, "2"):
            c = settings(bad)
            with self.assertRaisesRegex(ValueError, "patience_runs"):
                validate(c)
        for bad in ([], ["missing"], ["prey", "prey"], "prey", [True]):
            c = settings(species=bad)
            with self.assertRaisesRegex(ValueError, "early_stopping.species"):
                validate(c)

    def test_experiment_stops_at_run_boundary_and_stays_stopped_on_resume(self):
        e = Experiment(settings())
        while not e.done:
            e.step()
        self.assertEqual(e.run_index + 1, 3)
        self.assertEqual(len(e.summaries), 3)
        self.assertEqual(len(e.run_seeds), 3)
        self.assertEqual(e.world.tick, 3)
        self.assertEqual(e.summaries[-1]["early_stopping"]["stagnant_runs"], 2)
        before = e.checkpoint()
        self.assertFalse(e.next_run())
        e.step()
        self.assertEqual(e.checkpoint(), before)
        restored = Experiment.restore(json.loads(json.dumps(before)))
        self.assertTrue(restored.done)
        self.assertFalse(restored.next_run())
        self.assertEqual(restored.checkpoint(), before)

    def test_checkpoint_continues_patience_for_all_algorithms(self):
        for kind in KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                c = environment(kind, 0)
                c["experiment"].update(
                    runs=8,
                    chain_best_brains=False,
                    world_seed_policy="fixed",
                    early_stopping={"patience_runs": 2},
                )
                e = Experiment(c)
                while len(e.summaries) < 2:
                    e.step()
                self.assertEqual(e.early_stopping.stagnant_runs, 1)
                e.next_run()
                e.step()
                checkpoint = Path(folder) / "state.bin.xz"
                e.write_checkpoint(checkpoint)
                restored = Experiment.restore(read_checkpoint(checkpoint))
                while not e.done:
                    e.step()
                    restored.step()
                self.assertEqual(len(e.summaries), 3)
                self.assertEqual(e.checkpoint(), restored.checkpoint())

    def test_monitor_does_not_change_world_or_brain_choices_before_stopping(self):
        c = settings()
        reference = copy.deepcopy(c)
        reference["experiment"].pop("early_stopping")
        e, other = Experiment(c), Experiment(reference)
        while not e.done:
            e.step()
            other.step()
            actual_world = copy.deepcopy(e.world.checkpoint())
            expected_world = copy.deepcopy(other.world.checkpoint())
            actual_world["config"]["experiment"].pop("early_stopping", None)
            expected_world["config"]["experiment"].pop("early_stopping", None)
            self.assertEqual(actual_world, expected_world)
            self.assertEqual(set(e.brains), set(other.brains))
            for animal_id in e.brains:
                self.assertEqual(
                    e.brains[animal_id].checkpoint(),
                    other.brains[animal_id].checkpoint(),
                )
        self.assertFalse(other.done)

    def test_max_runs_and_old_checkpoints_remain_valid(self):
        c = settings(10)
        c["experiment"]["runs"] = 2
        e = Experiment(c)
        while not e.done:
            e.step()
        self.assertFalse(e.early_stopping.stopped)
        old = Experiment(config()).checkpoint()
        old.pop("early_stopping")
        restored = Experiment.restore(old)
        self.assertFalse(restored.early_stopping.active)

    def test_cli_finishes_replay_archive_summary_and_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "environment.json"
            path.write_text(json.dumps(settings(0)))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wtf.cli",
                    str(path),
                    "--early-stop-runs",
                    "2",
                    "--early-stop-species",
                    "prey",
                    "--no-progress",
                    "--output",
                    str(root / "output"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Early stopping after run 3", result.stdout)
            output = next((root / "output").iterdir())
            summary = json.loads((output / "summary.json").read_text())
            stop = json.loads((output / "early_stopping.json").read_text())
            self.assertEqual(len(summary), 3)
            self.assertTrue(stop["stopped"])
            self.assertEqual(stop["species"], ["prey"])
            replay = json.loads((output / "playback/recording.json").read_text())
            self.assertEqual(len(replay["runs"]), 3)
            self.assertTrue(all(run["complete"] for run in replay["runs"]))
            self.assertTrue((output / "champions/run_0003.json").is_file())
            final = Experiment.restore(
                read_checkpoint(output / "checkpoints/final.bin.xz")
            )
            self.assertTrue(final.done)
            # A completed-run periodic checkpoint starts recording at the next
            # actual episode, preserving the already-used patience.
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wtf.cli",
                    "--resume",
                    str(output / "checkpoints/run_0002.bin.xz"),
                    "--no-progress",
                    "--output",
                    str(root / "periodic"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            periodic = next((root / "periodic").iterdir())
            replay = json.loads((periodic / "playback/recording.json").read_text())
            self.assertEqual([run["run"] for run in replay["runs"]], [3])
            self.assertTrue(replay["runs"][0]["complete"])
            # Resuming a completed early stop must not create a phantom replay run.
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wtf.cli",
                    "--resume",
                    str(output / "checkpoints/final.bin.xz"),
                    "--no-progress",
                    "--output",
                    str(root / "resumed"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            resumed = next((root / "resumed").iterdir())
            replay = json.loads((resumed / "playback/recording.json").read_text())
            self.assertEqual(replay["runs"], [])

    def test_cli_requires_patience_and_rejects_resume_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "environment.json"
            path.write_text(json.dumps(config()))
            for args, message in (
                (["--early-stop-species", "prey"], "needs --early-stop-runs"),
                (["--resume", "missing", "--early-stop-runs", "2"], "overrides"),
            ):
                result = subprocess.run(
                    [sys.executable, "-m", "wtf.cli", str(path), *args],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
