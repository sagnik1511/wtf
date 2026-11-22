"""Per-save reuse, bounded payloads and unchanged public checkpoint contracts."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_mutation import environment
from wtf.experiment.runner import Experiment
from wtf.storage import model_io
from wtf.storage.checkpoint_io import write_checkpoint, read_checkpoint
from wtf.storage.model_io import ModelEncodings, encode_model
from wtf.storage.retention import ArtifactRetention


class EncodingEfficiencyTests(unittest.TestCase):
    def test_periodic_gate_and_export_encode_each_snapshot_once(self):
        e = Experiment(environment("ppo", 0))
        unique = {
            id(entry["model"])
            for groups in e.champions.entries.values()
            for entries in groups.values()
            for entry in entries.values()
        }
        with tempfile.TemporaryDirectory() as folder:
            retention = ArtifactRetention(folder, {"brain_save_every_steps": 1})
            before = e.checkpoint()
            with patch.object(model_io, "encode_model", wraps=encode_model) as encode:
                self.assertTrue(retention.step_brains(e))
                self.assertEqual(encode.call_count, len(unique))
            self.assertEqual(e.checkpoint(), before)
            e.champions.entries = copy.deepcopy(e.champions.entries)
            for groups in e.champions.entries.values():
                for entries in groups.values():
                    for entry in entries.values():
                        entry["rank"][0] += 1
            with patch.object(model_io, "encode_model", wraps=encode_model) as encode:
                self.assertFalse(retention.step_brains(e))
                self.assertEqual(encode.call_count, len(unique))

    def test_cache_eviction_and_new_transaction_observe_weight_changes(self):
        e = Experiment(environment("ppo", 0))
        models = [
            entry["model"]
            for groups in e.champions.entries.values()
            for entries in groups.values()
            for entry in entries.values()
        ]
        first = models[0]
        second = next(m for m in models if m is not first)
        expected = encode_model(first)
        cache = ModelEncodings(
            max_bytes=max(len(expected[0]), len(encode_model(second)[0]))
        )
        self.assertEqual(cache.encoded(first), expected)
        cache.encoded(second)
        self.assertNotIn(id(first), cache.payloads)
        self.assertLessEqual(cache.bytes, cache.max_bytes)
        self.assertEqual(cache.encoded(first), expected)
        tiny = ModelEncodings(max_bytes=0)
        self.assertEqual(tiny.encoded(first), expected)
        self.assertEqual(tiny.bytes, 0)
        changed = copy.deepcopy(first)
        old = ModelEncodings().digest(changed)
        changed["parameters"][0]["weights"][0] += 0.1
        self.assertNotEqual(ModelEncodings().digest(changed), old)
        # A bounded cache may re-encode; changed input must not be published under an old hash.
        tiny.encoded(second)
        second = copy.deepcopy(second)
        tiny.digest(second)
        second["parameters"][0]["weights"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "changed"):
            tiny.encoded(second)

    def test_streamed_and_detached_metadata_write_identical_checkpoints(self):
        e = Experiment(environment("ppo", 0))
        e.step()
        before = e.checkpoint()
        with tempfile.TemporaryDirectory() as folder:
            for suffix in (".json", ".json.gz", ".bin.xz"):
                with self.subTest(suffix=suffix):
                    root = Path(folder)
                    reference = root / ("reference" + suffix)
                    direct = root / ("direct" + suffix)
                    write_checkpoint(
                        reference,
                        e.checkpoint(include_brains=False),
                        ((i, b.checkpoint()) for i, b in e.brains.items()),
                    )
                    e.write_checkpoint(direct)
                    self.assertEqual(
                        read_checkpoint(reference), read_checkpoint(direct)
                    )
                    if suffix != ".json.gz":
                        self.assertEqual(reference.read_bytes(), direct.read_bytes())
                    restored = Experiment.restore(read_checkpoint(direct))
                    self.assertEqual(e.checkpoint(), restored.checkpoint())
            self.assertEqual(e.checkpoint(), before)
        # Public snapshot mutations must never reach live controller/model metadata.
        snapshot = e.checkpoint()
        snapshot["initial_weights"].clear()
        snapshot["champions"].clear()
        snapshot["config"]["name"] = "changed"
        snapshot["history"].append({"not": "live"})
        self.assertEqual(e.checkpoint(), before)

    def test_checkpoint_extension_hook_remains_visible_to_file_writer(self):
        class TaggedExperiment(Experiment):
            def checkpoint(self, include_brains=True):
                state = super().checkpoint(include_brains)
                state["custom_metadata"] = {"tag": "preserved"}
                return state

        e = TaggedExperiment(environment("ppo", 0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tagged.bin.xz"
            e.write_checkpoint(path)
            self.assertEqual(
                read_checkpoint(path)["custom_metadata"], {"tag": "preserved"}
            )
