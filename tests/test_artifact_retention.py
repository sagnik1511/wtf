"""Exact state, publication ordering and bounded ownership for artifact retention."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_mutation import environment
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.library import apply_brain_overrides
from wtf.storage.retention import ArtifactRetention


class ArtifactRetentionTests(unittest.TestCase):
    def test_shared_metadata_roundtrip_and_rolling_never_prune_foreign_files(self):
        c = environment("ppo", 0)
        c["experiment"]["runs"] = 2
        e = Experiment(c)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoints").mkdir()
            foreign = root / "checkpoints/old.bin.xz"
            foreign.write_bytes(b"existing artifact")
            policy = ArtifactRetention(
                root, {"checkpoint_keep_last": 2, "checkpoint_shared_metadata": True}
            )
            for tick in range(4):
                e.step()
                path = policy.checkpoint(e, f"step_{tick}")
                state = read_checkpoint(path)
                self.assertEqual(Experiment.restore(state).checkpoint(), e.checkpoint())
                restored = Experiment.restore(state)
                e.step()
                restored.step()
                self.assertEqual(e.checkpoint(), restored.checkpoint())
            self.assertEqual(len(list((root / "checkpoints").glob("step*"))), 2)
            self.assertEqual(foreign.read_bytes(), b"existing artifact")
            self.assertTrue(list((root / "checkpoints/_metadata").glob("*.json.xz")))
            while not e.done:
                e.step()
                restored.step()
                self.assertEqual(e.checkpoint(), restored.checkpoint())
            # Checksum failures are explicit; never silently substitute metadata.
            metadata = next((root / "checkpoints/_metadata").glob("*.json.xz"))
            metadata.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "metadata"):
                read_checkpoint(path)

    def test_failed_checkpoint_publication_does_not_prune_last_recovery(self):
        e = Experiment(environment("ppo", 0))
        with tempfile.TemporaryDirectory() as folder:
            policy = ArtifactRetention(folder, {"checkpoint_keep_last": 1})
            first = policy.checkpoint(e, "first")
            with self.assertRaises(FileExistsError):
                policy.checkpoint(e, "first")
            with patch.object(e, "write_checkpoint", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    policy.checkpoint(e, "second")
            self.assertEqual(
                Experiment.restore(read_checkpoint(first)).checkpoint(), e.checkpoint()
            )

    def test_periodic_brains_skip_unchanged_weights_and_preserve_chaining(self):
        e = Experiment(environment("ppo", 0))
        with tempfile.TemporaryDirectory() as folder:
            policy = ArtifactRetention(
                folder, {"brain_save_every_steps": 1, "brain_keep_last": 1}
            )
            before = e.checkpoint()
            self.assertTrue(policy.step_brains(e))
            self.assertEqual(before, e.checkpoint())
            self.assertFalse(policy.step_brains(e))
            e.champions.entries = copy.deepcopy(e.champions.entries)
            for groups in e.champions.entries.values():
                for criteria in groups.values():
                    for entry in criteria.values():
                        entry["rank"][0] += 1
            # Better score alone does not write identical weights again.
            self.assertFalse(policy.step_brains(e))
            e.champions.entries = copy.deepcopy(e.champions.entries)
            for groups in e.champions.entries.values():
                for criteria in groups.values():
                    for entry in criteria.values():
                        entry["model"]["parameters"][0]["weights"][0] += 0.01
            policy.options["brain_save_min_score_delta"] = 2
            self.assertFalse(policy.step_brains(e))
            policy.options["brain_save_min_score_delta"] = 0
            e.world.tick += 1
            before = e.checkpoint()
            self.assertTrue(policy.step_brains(e))
            self.assertEqual(before, e.checkpoint())
            root = Path(folder) / "champions/steps"
            self.assertEqual(len(list(root.glob("step_*.json"))), 1)
            index = json.loads((root / "latest.json").read_text())
            self.assertTrue(index["entries"])
            c = environment("ppo", 0)
            apply_brain_overrides(c, source_experiment=root / "latest.json")
            loaded = Experiment(c)
            self.assertTrue(loaded.initial_models)
            # Retained periodic models only; old numbered run archives are elsewhere.
            self.assertEqual(
                {p.name for p in (root / "models").glob("*.bin.xz")},
                {digest.hex() + ".bin.xz" for digest in policy.digests.values()},
            )

    def test_threshold_and_policy_validation(self):
        for name, value in [
            ("checkpoint_keep_last", -1),
            ("brain_save_every_steps", True),
            ("checkpoint_every_steps", -1),
            ("brain_save_min_score_delta", -1),
            ("checkpoint_shared_metadata", 1),
        ]:
            c = environment("ppo", 0)
            c["experiment"][name] = value
            with self.assertRaises(ValueError):
                validate(c)
        e = Experiment(environment("ppo", 0))
        with tempfile.TemporaryDirectory() as folder:
            policy = ArtifactRetention(folder, {"brain_save_every_steps": 0})
            self.assertFalse(policy.step_brains(e))
            self.assertFalse((Path(folder) / "champions").exists())

    def test_recent_best_milestones_and_all_replays_are_preserved(self):
        import contextlib
        import io
        from wtf.cli import run_experiment
        from wtf.storage.recording import Recorder
        from wtf.storage.replay_io import load_chunk
        from tests.test_ppo import ppo_environment

        c = ppo_environment()
        c["recording"] = dict(
            enabled=True,
            encoding="binary",
            compression="xz",
            frames_per_chunk=3,
            frame_every_steps=1,
            diagnostics_every_steps=2,
        )
        outcomes = []
        with tempfile.TemporaryDirectory() as folder:
            for name, options in [
                (
                    "before",
                    dict(
                        checkpoint_keep_last=0,
                        checkpoint_every_steps=0,
                        brain_save_every_steps=0,
                        checkpoint_shared_metadata=False,
                    ),
                ),
                (
                    "after",
                    dict(
                        checkpoint_keep_last=2,
                        checkpoint_every_steps=3,
                        brain_save_every_steps=2,
                        brain_keep_last=2,
                    ),
                ),
            ]:
                root = Path(folder) / name
                root.mkdir()
                e = Experiment(copy.deepcopy(c))
                recorder = Recorder(root, c, "train")
                with contextlib.redirect_stdout(io.StringIO()):
                    run_experiment(
                        e,
                        root,
                        lambda *_: None,
                        recorder,
                        checkpoint_format="binary",
                        storage_options=options,
                    )
                recorder.close()
                replay = [
                    load_chunk(path)
                    for path in sorted((root / "playback").glob("*.bin.xz"))
                ]
                outcomes.append((e, replay))
                if name == "after":
                    self.assertLessEqual(
                        len(list((root / "checkpoints").glob("*.bin.xz"))), 2
                    )
                    self.assertLessEqual(
                        len(list((root / "champions/steps").glob("step_*.json"))), 2
                    )
                    self.assertEqual(
                        len(list((root / "champions").glob("run_*.json"))), 3
                    )
                    restored = Experiment.restore(
                        read_checkpoint(root / "checkpoints/final.bin.xz")
                    )
                    self.assertEqual(restored.checkpoint(), e.checkpoint())
            self.assertEqual(outcomes[0][1], outcomes[1][1])
            self.assertEqual(
                outcomes[0][0].world.checkpoint(), outcomes[1][0].world.checkpoint()
            )
            self.assertEqual(
                {i: b.checkpoint() for i, b in outcomes[0][0].brains.items()},
                {i: b.checkpoint() for i, b in outcomes[1][0].brains.items()},
            )
