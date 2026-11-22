"""Training capture policies and explicit frozen-policy recorded evaluation."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.learning.brains import model_snapshot
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.recording import Recorder, RecordingStore, settings
from wtf.storage.replay_io import load_chunk
from tests.test_simulation import config
from tests.test_reproduction import breeding_experiment
from tests.reproduction_support import force_reproduction
from tests.test_ppo import ppo_environment
from tests.test_policy_algorithms import environment as policy_environment
from tests.test_advanced_algorithms import environment as advanced_environment

ROOT = Path(__file__).resolve().parents[1]
KINDS = (
    "q_network",
    "ppo",
    "a2c",
    "reinforce",
    "grpo",
    "recurrent_ppo",
    "qrdqn",
    "maskable_ppo",
    "mappo",
)


class RecordingProfileTests(unittest.TestCase):
    def test_resolution_is_mode_aware_and_keeps_config_unchanged(self):
        c = config()
        original = copy.deepcopy(c)
        self.assertEqual(settings(c)["profile"], "full")
        for profile in ("full", "compact", "training-only"):
            c["recording"]["profile"] = profile
            before = copy.deepcopy(c)
            for mode in ("train", "evaluate", "random"):
                opts = settings(c, mode)
                self.assertEqual(
                    opts["profile"], profile if mode == "train" else "full"
                )
                self.assertEqual(
                    opts["enabled"],
                    not (mode == "train" and profile == "training-only"),
                )
                self.assertEqual(
                    opts["frame_every_steps"], c["recording"]["frame_every_steps"]
                )
                self.assertEqual(
                    opts["record_activations"],
                    (
                        False
                        if mode == "train" and profile == "compact"
                        else c["recording"].get("record_activations", True)
                    ),
                )
            self.assertEqual(c, before)
        original["recording"]["profile"] = "compact"
        for interval, expected in ((0, 0), (10, 50), (100, 100)):
            original["recording"]["diagnostics_every_steps"] = interval
            self.assertEqual(settings(original)["diagnostics_every_steps"], expected)
        original["recording"]["enabled"] = False
        self.assertFalse(settings(original, "evaluate")["enabled"])
        c["recording"]["profile"] = "typo"
        with self.assertRaisesRegex(ValueError, "recording.profile"):
            validate(c)

    def test_recording_does_not_change_learning_and_compact_keeps_world_frames(self):
        saved = {}
        with tempfile.TemporaryDirectory() as directory:
            for profile in ("full", "compact", "training-only"):
                c = config()
                c["experiment"]["runs"] = 1
                c["recording"].update(
                    profile=profile, frames_per_chunk=64, diagnostics_every_steps=5
                )
                e, control = Experiment(c), Experiment(c)
                root = Path(directory) / profile
                recorder = (
                    Recorder(root, c, "train") if settings(c)["enabled"] else None
                )
                if recorder:
                    recorder.begin_run(e)
                while not e.done:
                    e.step()
                    control.step()
                    if recorder:
                        recorder.capture(e, force=e.finished_run)
                if recorder:
                    recorder.end_run(e)
                    recorder.close()
                    store = RecordingStore(directory)
                    saved[profile] = load_chunk(store.chunk(profile, 1, 0))
                else:
                    self.assertFalse(root.exists())
                    with self.assertRaisesRegex(ValueError, "disabled"):
                        Recorder(root, c, "train")
                    self.assertFalse(root.exists())
                self.assertEqual(e.checkpoint(), control.checkpoint())
            world = lambda frames: [
                {k: v for k, v in f.items() if k != "diagnostics"} for f in frames
            ]
            self.assertEqual(world(saved["full"]), world(saved["compact"]))
            self.assertLess(
                sum("diagnostics" in f for f in saved["compact"]),
                sum("diagnostics" in f for f in saved["full"]),
            )
            for frame in saved["compact"]:
                for diagnostic in frame.get("diagnostics", {}).values():
                    self.assertNotIn("activations", diagnostic)

    def invoke(self, *arguments, check=True):
        return subprocess.run(
            [sys.executable, "-m", "wtf", "run", *map(str, arguments), "--no-progress"],
            cwd=ROOT,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_training_only_to_frozen_recorded_evaluation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = config()
            c["species"]["prey"]["initial_population"] = 4
            c["species"]["predator"]["initial_population"] = 2
            c["experiment"]["runs"] = 1
            c["world"]["max_steps"] = 4
            path = root / "env.json"
            path.write_text(json.dumps(c))
            output = root / "train"
            self.invoke(
                path, "--recording-profile", "training-only", "--output", output
            )
            trained = next(output.iterdir())
            self.assertFalse((trained / "playback").exists())
            for name in (
                "metrics.csv",
                "summary.json",
                "champions/run_0001.json",
                "checkpoints/final.bin.xz",
            ):
                self.assertTrue((trained / name).is_file(), name)
            train_state = read_checkpoint(trained / "checkpoints/final.bin.xz")
            self.assertEqual(
                train_state["config"]["recording"]["profile"], "training-only"
            )
            manifest = json.loads((trained / "manifest.json").read_text())
            self.assertFalse(manifest["playback_recording"])
            self.assertGreater(
                len((trained / "metrics.csv").read_text().splitlines()), 1
            )
            output = root / "evaluation"
            self.invoke(
                trained / "environment.json",
                "--mode",
                "evaluate",
                "--freeze-brains",
                "--brains-from",
                trained,
                "--output",
                output,
            )
            evaluated = next(output.iterdir())
            manifest = json.loads((evaluated / "manifest.json").read_text())
            self.assertTrue(manifest["playback_recording"])
            self.assertEqual(manifest["recording_profile"], "full")
            self.assertTrue(manifest["freeze_evaluation_brains"])
            state = read_checkpoint(evaluated / "checkpoints/final.bin.xz")
            experiment = Experiment.restore(state)
            self.assertEqual(experiment.total_updates, 0)
            for identity, brain in experiment.brains.items():
                a = experiment.world.animals[identity]
                self.assertEqual(
                    model_snapshot(brain),
                    experiment.initial_models[a.species][a.sex]["model"],
                )
            store = RecordingStore(output)
            entry = store.metadata(evaluated.name)["runs"][0]
            frames = [
                f
                for i in range(entry["chunk_count"])
                for f in load_chunk(store.chunk(evaluated.name, 1, i))
            ]
            self.assertEqual([f["tick"] for f in frames], list(range(5)))
            self.assertTrue(any("diagnostics" in f for f in frames))
            resumed = root / "resumed"
            self.invoke(
                "--resume", trained / "checkpoints/final.bin.xz", "--output", resumed
            )
            self.assertFalse((next(resumed.iterdir()) / "playback").exists())
            skipped = root / "skipped"
            self.invoke(
                path,
                "--mode",
                "evaluate",
                "--recording-profile",
                "full",
                "--no-record",
                "--output",
                skipped,
            )
            self.assertFalse((next(skipped.iterdir()) / "playback").exists())

    def test_training_profiles_and_freeze_flag_reject_wrong_modes_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for args in (
                ("--mode", "evaluate", "--recording-profile", "compact"),
                ("--mode", "random", "--recording-profile", "training-only"),
                ("--freeze-brains",),
            ):
                result = self.invoke(
                    "baseline", "--steps", "1", *args, "--output", output, check=False
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(output.exists())


class FrozenEvaluationTests(unittest.TestCase):
    def test_parameters_never_learn_in_evaluation_all_algorithms(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = (
                    config()
                    if kind == "q_network"
                    else (
                        ppo_environment()
                        if kind == "ppo"
                        else (
                            policy_environment(kind)
                            if kind in ("a2c", "reinforce", "grpo")
                            else advanced_environment(kind)
                        )
                    )
                )
                c["experiment"]["runs"] = 1
                c["world"]["max_steps"] = 4
                e = Experiment(c, mode="evaluate", freeze_evaluation_brains=True)
                original = {i: model_snapshot(b) for i, b in e.brains.items()}
                for _ in range(2):
                    e.step()
                restored = Experiment.restore(e.checkpoint())
                self.assertTrue(restored.freeze_evaluation_brains)
                while not e.done:
                    e.step()
                    restored.step()
                self.assertEqual(e.checkpoint(), restored.checkpoint())
                self.assertEqual(e.total_updates, 0)
                for i, b in e.brains.items():
                    self.assertEqual(model_snapshot(b), original[i])
                legacy = e.checkpoint()
                legacy.pop("freeze_evaluation_brains")
                self.assertFalse(Experiment.restore(legacy).freeze_evaluation_brains)

    def test_newborn_copies_same_sex_parent_without_crossover_or_mutation(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                if kind in ("recurrent_ppo", "qrdqn", "maskable_ppo", "mappo"):
                    c = copy.deepcopy(breeding_experiment("ppo").config)
                    c["brains"] = advanced_environment(kind)["brains"]
                    for spec in c["species"].values():
                        spec["brain"] = "test"
                    e = Experiment(c, mode="evaluate", freeze_evaluation_brains=True)
                    for i, a in e.world.animals.items():
                        a.x, a.y = i + 1, 2
                        if a.species == "prey":
                            a.sex = "female" if i == 0 else "male"
                    e.observations = e.world.observe_all()
                    force_reproduction(e)
                else:
                    e = breeding_experiment(kind)
                    e.mode = "evaluate"
                    e.freeze_evaluation_brains = True
                before = {i: model_snapshot(b) for i, b in e.brains.items()}
                with (
                    patch(
                        "wtf.experiment.runner.mix_parent_brains",
                        side_effect=AssertionError("crossover"),
                    ),
                    patch(
                        "wtf.learning.mutation.mutate_birth",
                        side_effect=AssertionError("mutation"),
                    ),
                ):
                    e.step()
                children = [a for a in e.world.animals.values() if a.id not in before]
                self.assertTrue(children)
                for child in children:
                    parent = (
                        child.parent_id
                        if child.sex == "female"
                        else child.second_parent_id
                    )
                    self.assertEqual(model_snapshot(e.brains[child.id]), before[parent])
                    self.assertIsNot(e.brains[child.id], e.brains[parent])
