"""Bounded metadata identity reuse and update-local PPO sample preparation."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from wtf.storage import checkpoint_io
from wtf.storage.checkpoint_io import (
    Encoder,
    Decoder,
    write_checkpoint,
    read_checkpoint,
)
from wtf.learning import ppo
from wtf.native import kernels
from wtf.learning.brain import Experience
from tests.test_ppo import ppo_settings


class CheckpointMetadataReuseTests(unittest.TestCase):
    def test_shared_values_hash_once_but_distinct_equal_values_still_deduplicate(self):
        model = {"parameters": [0.25] * 20}
        metadata = {"entries": [{"model": model} for _ in range(8)]}
        reference = Encoder(binary=True)
        with patch.object(
            checkpoint_io.json, "dumps", wraps=checkpoint_io.json.dumps
        ) as dumps:
            expected = reference.encode(metadata)
            self.assertEqual(dumps.call_count, 8)
        actual = Encoder(binary=True)
        with patch.object(
            checkpoint_io.json, "dumps", wraps=checkpoint_io.json.dumps
        ) as dumps:
            encoded = actual._encode_metadata(metadata)
            self.assertEqual(dumps.call_count, 1)
        self.assertEqual(encoded, expected)
        self.assertEqual(actual.shared, reference.shared)
        self.assertIsNone(actual._identity_refs)
        # A separate equal object uses content deduplication, not identity reuse.
        self.assertEqual(actual.encode(copy.deepcopy(model), "model"), {"$ref": 0})
        self.assertEqual(len(actual.shared), 1)
        decoded = Decoder(actual.shared).decode(encoded)
        decoded["entries"][0]["model"]["parameters"][0] = 2.0
        self.assertEqual(decoded["entries"][1]["model"]["parameters"][0], 0.25)

    def test_cache_is_bounded_cleared_on_failure_and_observes_later_mutation(self):
        encoder = Encoder()
        largest = 0
        contents = encoder.contents

        def watch(value):
            nonlocal largest
            if encoder._identity_refs is not None:
                largest = max(largest, len(encoder._identity_refs))
            return contents(value)

        value = {"entries": [{"model": {"v": i}} for i in range(200)]}
        with patch.object(encoder, "contents", side_effect=watch):
            encoded = encoder._encode_metadata(value)
        self.assertEqual(largest, 128)
        self.assertEqual(Decoder(encoder.shared).decode(encoded), value)
        value["entries"][0]["model"]["v"] = -1
        changed = encoder._encode_metadata(value)
        self.assertEqual(Decoder(encoder.shared).decode(changed), value)
        with self.assertRaises(ValueError):
            encoder._encode_metadata({"model": {"v": float("nan")}})
        self.assertIsNone(encoder._identity_refs)

    def test_public_encoder_and_streamed_reused_containers_remain_mutation_sensitive(
        self,
    ):
        encoder = Encoder()
        value = {"v": 1}
        a = encoder.encode(value, "model")
        value["v"] = 2
        b = encoder.encode(value, "model")
        self.assertNotEqual(a, b)
        with tempfile.TemporaryDirectory() as folder:
            for suffix in (".json.gz", ".bin.xz"):
                state = {"checkpoint_version": 1, "model": value}

                def brains():
                    yield 0, {"config": value}
                    value["v"] += 1
                    yield 1, {"config": value}

                before = value["v"]
                path = Path(folder) / ("reuse" + suffix)
                write_checkpoint(path, state, brains())
                restored = read_checkpoint(path)
                self.assertEqual(restored["model"]["v"], before)
                self.assertEqual(restored["brains"]["0"]["config"]["v"], before)
                self.assertEqual(restored["brains"]["1"]["config"]["v"], before + 1)


class PPOSampleReuseTests(unittest.TestCase):
    def test_prepared_rows_match_legacy_packing_including_mask(self):
        rollout = [
            ppo.RolloutStep(tuple(range(7)), 1, 0.4, -0.5, 0.2, 0.3, False, True)
        ]
        for inputs in (None, 4):
            step = rollout[0]
            expected = (
                step.observation if inputs is None else step.observation[:inputs],
                1,
                -0.5,
                0.7,
                0.9,
            )
            if inputs is not None:
                expected += (tuple(step.observation[inputs:]),)
            self.assertEqual(
                ppo._prepare_native_samples(rollout, [0.7], [0.9], inputs), [expected]
            )

    @unittest.skipUnless(kernels.available("ppo_batch"), "Native PPO batch required")
    def test_native_rows_prepared_once_and_reused_across_epochs_and_resume(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        c = ppo_settings()
        c.update(epochs=3, rollout_steps=4, batch_size=2, target_kl=0)
        brain = ppo.PPOBrain(c, ["rest", "attack"], 3, 5)
        native = kernels.native.ppo_batch
        seen = []

        def capture(*args):
            seen.extend(args[2])
            return native(*args)

        with (
            patch.object(
                ppo, "_prepare_native_samples", wraps=ppo._prepare_native_samples
            ) as prepare,
            patch.object(kernels.native, "ppo_batch", side_effect=capture),
        ):
            for tick in range(8):
                obs = (tick / 10.0,) * 20
                action = brain.choose(obs)
                brain.learn(Experience(obs, action, 0.2, (0.0,) * 20, False, tick == 7))
                if tick == 3:
                    brain = ppo.PPOBrain.restore(brain.checkpoint())
            self.assertEqual(prepare.call_count, 2)
        self.assertEqual(len(seen), 24)
        self.assertEqual(len({id(row) for row in seen}), 8)
        self.assertNotIn("prepared_samples", brain.checkpoint())

    def test_fallback_does_not_prepare_native_samples(self):
        brain = ppo.PPOBrain(ppo_settings(), ["rest", "attack"], 3, 5)
        with (
            patch.object(kernels, "available", return_value=False),
            patch.object(ppo, "_prepare_native_samples") as prepare,
        ):
            for _tick in range(4):
                obs = (0.0,) * 20
                action = brain.choose(obs)
                brain.learn(Experience(obs, action, 0.2, obs, False, False))
            prepare.assert_not_called()
