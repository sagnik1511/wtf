"""Exact compact state, old-reader compatibility and verified atomic conversion."""

import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wtf.storage.checkpoint_io import Encoder, Decoder, read_checkpoint
from wtf.storage.optimize_checkpoints import optimize_experiment, state_digest
from wtf.experiment.runner import Experiment
from tests.test_simulation import config
from tests.test_ppo import ppo_environment
from tests.test_policy_algorithms import environment


class CheckpointStorageTests(unittest.TestCase):
    def test_exact_continuation_for_every_algorithm_and_mixed_world(self):
        scenarios = [config(), ppo_environment(), ppo_environment(True)] + [
            environment(k) for k in ("a2c", "reinforce", "grpo")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json.gz"
            for c in scenarios:
                with self.subTest(algorithms=[b["type"] for b in c["brains"].values()]):
                    a = Experiment(c)
                    for _ in range(3):
                        a.step()
                    before = state_digest(a.checkpoint())
                    a.write_checkpoint(path)
                    with gzip.open(path, "rt") as f:
                        envelope = json.load(f)
                    self.assertEqual(envelope["checkpoint_version"], 2)
                    self.assertGreater(len(envelope["shared"]), 0)
                    saved = read_checkpoint(path)
                    self.assertEqual(state_digest(saved), before)
                    b = Experiment.restore(saved)
                    for _ in range(6):
                        a.step()
                        b.step()
                    self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_number_types_signed_zero_reserved_keys_and_reference_isolation(self):
        encoder = Encoder()
        value = {
            "float": [0.0, -0.0] + [0.125] * 20,
            "uint": [0, 2**32 - 1] * 20,
            "mixed": [True, 1, 1.0, -1, 2**35] * 20,
            "nested": {"$ref": 4, "$literal": "ordinary data"},
            "a": {"config": {"learning_rate": 0.1}},
            "b": {"config": {"learning_rate": 0.1}},
        }
        encoded = encoder.encode(value)
        decoded = Decoder(encoder.shared).decode(json.loads(json.dumps(encoded)))
        self.assertEqual(state_digest(value), state_digest(decoded))
        self.assertEqual(math.copysign(1, decoded["float"][1]), -1)
        decoded["a"]["config"]["learning_rate"] = 2
        self.assertEqual(decoded["b"]["config"]["learning_rate"], 0.1)

    def test_legacy_plain_compressed_magic_and_old_path_alias(self):
        e = Experiment(config())
        state = json.loads(json.dumps(e.checkpoint()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = root / "legacy.json"
            e.write_checkpoint(plain)
            self.assertEqual(json.loads(plain.read_text()), state)
            compressed = root / "legacy-compressed.json"
            compressed.write_bytes(gzip.compress(plain.read_bytes()))
            self.assertEqual(read_checkpoint(compressed), state)
            packed = root / "new.json.gz"
            e.write_checkpoint(packed)
            self.assertEqual(read_checkpoint(root / "new.json"), state)

    def test_failed_write_preserves_previous_file_and_cleans_temporary(self):
        e = Experiment(config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json.gz"
            e.write_checkpoint(path)
            original = path.read_bytes()
            e.brains[0].weights["w1"][0] = float("nan")
            with self.assertRaises(ValueError):
                e.write_checkpoint(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_bad_references_and_array_lengths_fail(self):
        for value, shared in [
            ({"$ref": 0}, []),
            ({"$ref": True}, [{}]),
            ({"$ref": 0}, [{"$ref": 0}]),
            ({"$f64": "AA=="}, []),
            ({"$u32": "not-base64"}, []),
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Decoder(shared).decode(value)

    def fixture(self, root):
        (root / "checkpoints").mkdir()
        (root / "playback").mkdir()
        (root / "playback/recording.json").write_text(
            json.dumps({"status": "complete"})
        )
        e = Experiment(config())
        e.step()
        e.write_checkpoint(root / "checkpoints/final.json")
        return e

    def test_verified_conversion_is_smaller_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e = self.fixture(root)
            (root / "champions").mkdir()
            sentinel = root / "champions/untouched.json"
            sentinel.write_text("unchanged")
            result = optimize_experiment(root)
            self.assertEqual(result["checkpoints_converted"], 1)
            self.assertGreater(result["saved_bytes"], 0)
            self.assertFalse((root / "checkpoints/final.json").exists())
            self.assertEqual(
                state_digest(read_checkpoint(root / "checkpoints/final.json")),
                state_digest(e.checkpoint()),
            )
            self.assertEqual(optimize_experiment(root)["saved_bytes"], 0)
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_active_unconfirmed_or_conflicting_experiments_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e = self.fixture(root)
            manifest = root / "playback/recording.json"
            manifest.write_text('{"status":"writing"}')
            raw = root / "checkpoints/final.json"
            original = raw.read_bytes()
            with self.assertRaisesRegex(ValueError, "still writing"):
                optimize_experiment(root)
            self.assertEqual(raw.read_bytes(), original)
            manifest.write_text('{"status":"complete"}')
            e.retired_updates += 1
            e.write_checkpoint(raw.with_suffix(".json.gz"))
            with self.assertRaisesRegex(ValueError, "copies differ"):
                optimize_experiment(root)
            self.assertEqual(raw.read_bytes(), original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            with self.assertRaisesRegex(ValueError, "may still be writing"):
                optimize_experiment(root)

    def test_failed_conversion_verification_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            raw = root / "checkpoints/final.json"
            original = raw.read_bytes()
            with patch(
                "wtf.storage.optimize_checkpoints.state_digest",
                side_effect=[b"a", b"b"],
            ):
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    optimize_experiment(root)
            self.assertEqual(raw.read_bytes(), original)
            self.assertEqual(list((root / "checkpoints").glob("*.gz")), [])
