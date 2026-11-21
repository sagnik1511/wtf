"""Founder state isolation and explicitly scoped report reuse."""

import copy
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wtf.cli import run_experiment
from wtf.learning.brains import create_brain, load_model_weights, model_snapshot
from wtf.native.numeric import precision_scope
from wtf.progress import RunProgress
from wtf.storage.recording import Recorder
from wtf.experiment.runner import Experiment
from tests.test_mutation import KINDS, environment


class FounderInitializationTests(unittest.TestCase):
    def test_one_constructor_per_species_sex_with_original_seeded_state(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = environment(kind, 0)
                c["world"].update(width=12, height=12)
                for spec in c["species"].values():
                    spec["initial_population"] = 8
                with patch(
                    "wtf.experiment.runner.create_brain", wraps=create_brain
                ) as construct:
                    e = Experiment(c)
                groups = {(a.species, a.sex) for a in e.world.animals.values()}
                self.assertEqual(construct.call_count, len(groups))
                for index, (species, spec) in enumerate(e.config["species"].items()):
                    config = copy.deepcopy(e.config["brains"][spec["brain"]])
                    config["learning_rate"] *= spec.get("learning_rate_multiplier", 1)
                    for i, a in e.world.animals.items():
                        if a.species != species:
                            continue
                        base = e.config["seed"] + 1000003 * (index + 1)
                        with precision_scope(e.brain_precision):
                            expected = create_brain(
                                config,
                                spec["allowed_actions"],
                                base ^ (0x4D414C45 if a.sex == "male" else 0),
                                e.config["seed"] + i * 7919 + 13,
                            )
                        load_model_weights(expected, e.initial_weights[species][a.sex])
                        actual = e.brains[i]
                        # World-derived mask/central context is assigned after initialization.
                        for name in ("mask", "context", "next_context"):
                            if hasattr(actual, name):
                                setattr(
                                    expected, name, copy.deepcopy(getattr(actual, name))
                                )
                        self.assertEqual(actual.checkpoint(), expected.checkpoint())
                before = {i: copy.deepcopy(b.checkpoint()) for i, b in e.brains.items()}
                first = next(iter(e.brains))
                b = e.brains[first]
                b.rng.random()
                b.training_rng.random()
                b.config["learning_rate"] += 1
                if hasattr(b, "network"):
                    b.network.layers[0][next(iter(b.network.layers[0]))][0] += 1
                else:
                    b.weights["w1"][0] += 1
                for i, other in e.brains.items():
                    if i != first:
                        self.assertEqual(other.checkpoint(), before[i])

    def test_warm_start_loads_once_per_group_and_keeps_independent_state(self):
        e = Experiment(environment("ppo", 0))
        while not e.finished_run:
            e.step()
        with (
            patch(
                "wtf.experiment.runner.load_model_weights", wraps=load_model_weights
            ) as load,
            patch(
                "wtf.experiment.runner.create_brain", wraps=create_brain
            ) as construct,
        ):
            e.next_run()
        groups = {(a.species, a.sex) for a in e.world.animals.values()}
        self.assertEqual(construct.call_count, len(groups))
        self.assertEqual(load.call_count, len(groups))
        for i, a in e.world.animals.items():
            b = e.brains[i]
            source = e.run_initial_models.get(a.species, {}).get(a.sex)
            expected = (
                source["model"] if source else e.initial_weights[a.species][a.sex]
            )
            self.assertEqual(model_snapshot(b), expected)
            self.assertEqual(b.steps, 0)
            self.assertEqual(b.rollout, [])


class SharedReportingTests(unittest.TestCase):
    def test_terminal_report_scans_once_and_summary_owns_maps(self):
        c = environment("ppo", 0)
        c["world"]["max_steps"] = 1
        e = Experiment(c)
        with patch.object(e.world, "metrics", wraps=e.world.metrics) as metrics:
            e.step()
        self.assertEqual(metrics.call_count, 1)
        self.assertTrue(e.finished_run)
        e.last_record["populations"].clear()
        self.assertTrue(e.summaries[-1]["populations"])

    def test_reporting_consumers_reuse_explicit_metrics_and_detach_snapshots(self):
        e = Experiment(environment("ppo", 0))
        report = e.world.metrics()
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder), e.config, "train")
            progress = RunProgress(enabled=True, stream=io.StringIO())
            with patch.object(e.world, "metrics", wraps=e.world.metrics) as metrics:
                recorder.begin_run(e, metrics=report)
                progress.start(e, metrics=report)
                snapshot = e.world.snapshot(metrics=report)
                self.assertEqual(metrics.call_count, 0)
            snapshot["metrics"]["populations"].clear()
            self.assertTrue(report["populations"])
            progress.close()
            recorder.close()

    def test_standalone_consumers_read_same_tick_changes(self):
        e = Experiment(environment("ppo", 0))
        old = e.world.metrics()
        e.world.animals.clear()
        fresh = e.world.snapshot()["metrics"]
        self.assertNotEqual(old["populations"], fresh["populations"])
        self.assertEqual(sum(fresh["populations"].values()), 0)
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder), e.config, "train")
            progress = RunProgress(enabled=True, stream=io.StringIO())
            with patch.object(e.world, "metrics", wraps=e.world.metrics) as metrics:
                recorder.begin_run(e)
                progress.start(e)
                self.assertEqual(metrics.call_count, 2)
            progress.close()
            recorder.close()

    def test_cli_shares_report_between_recording_progress_csv_and_terminal(self):
        c = environment("ppo", 0)
        c["world"]["max_steps"] = 3
        c["experiment"].update(runs=1, record_metrics_every_steps=1)
        c["recording"]["frame_every_steps"] = 1
        e = Experiment(c)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            recorder = Recorder(folder, e.config, "train")
            progress = RunProgress(enabled=True, stream=io.StringIO())
            with (
                patch.object(e.world, "metrics", wraps=e.world.metrics) as metrics,
                patch.object(e, "write_checkpoint"),
                redirect_stdout(io.StringIO()),
            ):
                run_experiment(
                    e,
                    folder,
                    lambda *_: None,
                    recorder,
                    progress=progress,
                    save_brains=False,
                )
                self.assertEqual(metrics.call_count, 1 + e.world.tick)
            recorder.close()
            progress.close()
            self.assertIn(str(e.world.tick), (folder / "metrics.csv").read_text())

    def test_unsampled_capture_does_not_compute_metrics(self):
        e = Experiment(environment("ppo", 0))
        e.config["recording"]["frame_every_steps"] = 3
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder), e.config, "train")
            recorder.begin_run(e)
            e.step()
            with patch.object(e.world, "metrics", wraps=e.world.metrics) as metrics:
                self.assertFalse(recorder.capture(e))
                self.assertEqual(metrics.call_count, 0)
            recorder.close()
