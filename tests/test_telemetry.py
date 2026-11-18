"""Per-step outcomes, sampled predictions, gaps and independently seekable pages."""

import json
from pathlib import Path
import tempfile
import unittest

from ecosystem.recording import Recorder, RecordingStore
from ecosystem.runner import Experiment
from ecosystem.telemetry import summarize, groups
from tests.test_analysis import fixture
from tests.test_simulation import config


class TelemetryTests(unittest.TestCase):
    def test_per_step_population_hunt_birth_death_and_health(self):
        c = config()
        a, b, d = fixture()
        initial = summarize(c, a)
        self.assertIsNone(initial["species"]["predator"]["kills"])
        first = summarize(c, b, a)["species"]
        self.assertEqual(first["prey"]["population"], 2)
        self.assertEqual(first["prey"]["births"], 1)
        self.assertEqual(first["prey"]["attempts"], 1)
        self.assertEqual(first["prey"]["success"], 1)
        self.assertEqual(first["predator"]["attempts"], 1)
        self.assertEqual(first["predator"]["success"], 0)
        last = summarize(c, d, b)["species"]
        self.assertEqual(last["prey"]["deaths"], 1)
        self.assertEqual(last["prey"]["predation_deaths"], 1)
        self.assertEqual(last["predator"]["kills"], 1)
        self.assertEqual(last["predator"]["success"], 1)
        self.assertEqual(last["prey"]["births"], 0)  # Per-step, not cumulative.
        self.assertEqual(
            last["predator"]["health"],
            30 / c["species"]["predator"]["body"]["max_health"],
        )
        self.assertEqual(last["prey"]["action_coverage"], 0.5)

    def test_gaps_and_unsampled_predictions_remain_missing(self):
        c = config()
        a, b, d = fixture()
        a["diagnostics"] = {
            "0": {"action_values": {"rest": -2, "eat": 3}, "epsilon": 0.2},
            "1": {
                "action_probabilities": {"rest": 0.25, "attack": 0.75},
                "entropy": 0.5,
                "state_value": -1,
            },
        }
        point = summarize(c, a)
        self.assertEqual(point["species"]["prey"]["q_values"]["eat"], 3)
        self.assertEqual(point["species"]["predator"]["predictions"]["attack"], 0.75)
        self.assertEqual(point["species"]["predator"]["state_value"], -1)
        later = summarize(c, d, a)
        self.assertFalse(later["consecutive"])
        self.assertIsNone(later["species"]["predator"]["kills"])
        self.assertEqual(later["species"]["predator"]["predictions"], {})
        self.assertIsNone(later["species"]["predator"]["entropy"])
        json.dumps(later, allow_nan=False)

    def test_spatial_components_transitive_links_solitary_and_empty(self):
        animals = [dict(x=0, y=0), dict(x=3, y=0), dict(x=6, y=0), dict(x=20, y=20)]
        self.assertEqual(groups(animals), dict(clusters=1, solitary=1, largest=3))
        self.assertEqual(groups([]), dict(clusters=0, solitary=0, largest=0))

    def test_pages_cross_chunks_exactly_and_reject_missing_or_invalid_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = config()
            c["world"]["max_steps"] = 135
            c["experiment"]["runs"] = 1
            c["recording"]["frames_per_chunk"] = 13
            e = Experiment(c)
            recorder = Recorder(root / "test", c, "train")
            recorder.begin_run(e)
            while not e.done:
                e.step()
                recorder.capture(e, force=e.finished_run)
            recorder.end_run(e)
            recorder.close()
            store = RecordingStore(root)
            first = store.telemetry("test", 1, 0)
            last = store.telemetry("test", 1, 1)
            self.assertEqual(len(first["points"]), 128)
            self.assertEqual([p["tick"] for p in last["points"]], list(range(128, 136)))
            self.assertTrue(last["points"][0]["consecutive"])
            for page in (2, 3):
                with self.assertRaises(FileNotFoundError):
                    store.telemetry("test", 1, page)
            with self.assertRaises(ValueError):
                store.telemetry("test", 1, -1)
            with self.assertRaises(FileNotFoundError):
                store.telemetry("test", 99, 0)
