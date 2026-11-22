"""Scheduled-tail playback selection without altering training or retained reports."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.test_simulation import config
from wtf.cli import run_experiment
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.storage.recording import Recorder, RecordingStore, settings
from wtf.storage.replay_io import load_chunk

ROOT = Path(__file__).resolve().parents[1]


class LastEpisodeRecordingTests(unittest.TestCase):
    def setup_config(self, total=3, last=-1):
        c = config()
        c["world"]["max_steps"] = 3
        c["experiment"]["runs"] = total
        c["recording"].update(
            record_last_episodes=last, frames_per_chunk=2, diagnostics_every_steps=1
        )
        return c

    def invoke(self, *args, check=True):
        return subprocess.run(
            [sys.executable, "-m", "wtf", "run", *map(str, args), "--no-progress"],
            cwd=ROOT,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_validation_defaults_and_disabling_precedence(self):
        self.assertEqual(settings(config())["record_last_episodes"], -1)
        for value in (-2, True, 1.5, "2", None):
            c = self.setup_config(last=value)
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "record_last_episodes"),
            ):
                validate(c)
        for k in (-1, 0, 1, 99):
            validate(self.setup_config(last=k))
        for mode in ("train", "evaluate", "random"):
            self.assertFalse(settings(self.setup_config(last=0), mode)["enabled"])
        c = self.setup_config(last=2)
        c["recording"]["profile"] = "training-only"
        self.assertFalse(settings(c)["enabled"])
        self.assertTrue(settings(c, "evaluate")["enabled"])
        c["recording"]["enabled"] = False
        self.assertFalse(settings(c, "evaluate")["enabled"])

    def test_tail_clamping_and_exact_frames_training_state(self):
        saved = {}
        with tempfile.TemporaryDirectory() as directory:
            for k in (-1, 0, 1, 2, 3, 99):
                with self.subTest(last=k):
                    c = self.setup_config(last=k)
                    e, control = Experiment(c), Experiment(c)
                    folder = Path(directory) / f"case_{k}"
                    recorder = (
                        Recorder(folder, c, "train") if settings(c)["enabled"] else None
                    )
                    if recorder:
                        recorder.begin_run(e)
                    while not e.done:
                        if e.finished_run:
                            e.next_run()
                            control.next_run()
                            if recorder:
                                recorder.begin_run(e)
                        e.step()
                        control.step()
                        if recorder:
                            recorder.capture(e, force=e.finished_run)
                            if e.finished_run:
                                recorder.end_run(e)
                    if recorder:
                        recorder.close()
                        expected = list(range(1 if k == -1 else max(1, 4 - k), 4))
                        self.assertEqual(
                            [r["run"] for r in recorder.manifest["runs"]], expected
                        )
                        saved[k] = {}
                        store = RecordingStore(directory)
                        for entry in recorder.manifest["runs"]:
                            frames = [
                                f
                                for index in range(entry["chunk_count"])
                                for f in load_chunk(
                                    store.chunk(folder.name, entry["run"], index)
                                )
                            ]
                            self.assertEqual([f["tick"] for f in frames], [0, 1, 2, 3])
                            self.assertTrue(frames[-1]["finished_run"])
                            self.assertIn("diagnostics", frames[0])
                            saved[k][entry["run"]] = frames
                        for skipped in set(range(1, 4)) - set(expected):
                            self.assertFalse(
                                list(
                                    (folder / "playback").glob(
                                        f"run_{skipped:04d}_chunk*"
                                    )
                                )
                            )
                    else:
                        self.assertFalse(folder.exists())
                    self.assertEqual(e.checkpoint(), control.checkpoint())
            for k, episodes in saved.items():
                for episode, frames in episodes.items():
                    self.assertEqual(frames, saved[-1][episode])

    def test_skipped_episode_does_not_request_snapshot_or_open_stream(self):
        c = self.setup_config(total=10, last=2)
        e = Experiment(c)
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / "tail", c, "train")
            with patch.object(
                e.world, "snapshot", side_effect=AssertionError("snapshot")
            ):
                self.assertFalse(recorder.begin_run(e))
                self.assertFalse(recorder.capture(e, force=True))
            self.assertIsNone(recorder.stream)
            self.assertEqual(recorder.manifest["runs"], [])
            recorder.close()

    def test_cli_keeps_all_reports_brains_and_original_episode_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.setup_config(total=4)
            path = root / "environment.json"
            path.write_text(json.dumps(c))
            output = root / "results"
            self.invoke(path, "--record-last-episodes", "2", "--output", output)
            folder = next(output.iterdir())
            store = RecordingStore(output)
            metadata = store.metadata(folder.name)
            self.assertEqual([r["run"] for r in metadata["runs"]], [3, 4])
            self.assertEqual(metadata["recording"]["record_last_episodes"], 2)
            self.assertEqual(len(json.loads((folder / "summary.json").read_text())), 4)
            with (folder / "metrics.csv").open() as handle:
                self.assertEqual(
                    {int(r["run"]) for r in csv.DictReader(handle)}, {1, 2, 3, 4}
                )
            for n in range(1, 5):
                self.assertTrue((folder / f"champions/run_{n:04d}.json").is_file())
            self.assertTrue((folder / "checkpoints/final.bin.xz").is_file())
            self.assertEqual(
                json.loads((folder / "environment.json").read_text())["recording"][
                    "record_last_episodes"
                ],
                2,
            )
            self.assertEqual(
                json.loads((folder / "manifest.json").read_text())[
                    "recording_settings"
                ]["record_last_episodes"],
                2,
            )
            for extra in (
                ("--record-last-episodes", "0"),
                ("--record-last-episodes", "2", "--no-record"),
                ("--record-last-episodes", "2", "--recording-profile", "training-only"),
            ):
                disabled = root / ("disabled_" + str(len(list(root.iterdir()))))
                self.invoke(path, *extra, "--output", disabled)
                result = next(disabled.iterdir())
                self.assertFalse((result / "playback").exists())
                self.assertTrue((result / "metrics.csv").exists())
            invalid = root / "invalid"
            result = self.invoke(
                path, "--record-last-episodes", "-2", "--output", invalid, check=False
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse(invalid.exists())

    def test_resume_before_and_inside_tail_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for episode in (2, 3):
                c = self.setup_config(total=4, last=2)
                e = Experiment(c)
                while e.run_index + 1 < episode:
                    e.step()
                self.assertEqual(e.world.tick, 1)
                checkpoint = root / f"episode_{episode}.bin.xz"
                e.write_checkpoint(checkpoint)
                for override in (False, True):
                    output = root / f"resume_{episode}_{override}"
                    args = ["--resume", checkpoint, "--output", output, "--no-library"]
                    if override:
                        args += ["--record-last-episodes", "-1"]
                    self.invoke(*args)
                    folder = next(output.iterdir())
                    metadata = RecordingStore(output).metadata(folder.name)
                    expected = list(range(episode if override else max(episode, 3), 5))
                    self.assertEqual([r["run"] for r in metadata["runs"]], expected)
                    self.assertEqual(
                        metadata["runs"][0]["first_tick"],
                        1 if expected[0] == episode else 0,
                    )

    def test_early_stop_does_not_backfill_skipped_scheduled_episodes(self):
        c = self.setup_config(total=10, last=2)
        e = Experiment(c)
        original = e.step

        def step():
            result = original()
            if e.finished_run:
                e.early_stopping.stopped = True
            return result

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(e, "step", side_effect=step),
        ):
            folder = Path(directory) / "early"
            recorder = Recorder(folder, c, "train")

            def save(name, data):
                (folder / name).write_text(json.dumps(data))

            run_experiment(
                e, folder, save, recorder, save_brains=False, checkpoint_format="binary"
            )
            recorder.close()
            self.assertEqual(len(e.summaries), 1)
            self.assertEqual(recorder.manifest["runs"], [])
            self.assertEqual(RecordingStore(directory).catalog(), [])
            self.assertFalse(list((folder / "playback").glob("*chunk*")))
