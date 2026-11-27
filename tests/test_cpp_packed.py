"""Exact packed model snapshots and explicit Float32 buffer imports."""

# Compiled probes verify boundary ownership rather than public learner APIs.
# pylint: disable=c-extension-no-member,missing-class-docstring,missing-function-docstring

from array import array

import copy
import struct
import unittest

from tests.native_core_support import _engine, encoded
from tests.test_cpp_ppo import settings
from wtf.learning.ppo import PPOBrain


def state():
    return PPOBrain(settings([3]), ["rest", "eat"], 17, 901).checkpoint()


@unittest.skipUnless(_engine is not None, "Build the optional native PPO engine")
class NativePackedTests(unittest.TestCase):
    def test_explicit_float32_buffers_roundtrip_and_detach(self):
        original = state()
        packed = copy.deepcopy(original)
        for key in ("network", "moments", "variances"):
            for layer in packed[key]:
                for field in ("weights", "bias"):
                    layer[field] = array("f", layer[field])
        handle = _engine.ppo_create(packed)
        before = encoded(original)
        self.assertEqual(before, encoded(_engine.ppo_checkpoint(handle)))
        # Resizing succeeds only after the boundary releases its Py_buffer.
        packed["network"][0]["weights"].append(0.0)
        packed["network"][0]["weights"][0] = 123
        exported = _engine.ppo_checkpoint(handle)
        self.assertIsInstance(exported["network"][0]["weights"], list)
        exported["network"][0]["weights"][0] = 456
        self.assertEqual(before, encoded(_engine.ppo_checkpoint(handle)))

    def test_bit_patterns_preserve_float32_without_extra_quantization(self):
        checkpoint = state()
        bits = (0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF)
        values = [struct.unpack("=f", struct.pack("=I", word))[0] for word in bits]
        checkpoint["network"][0]["weights"][: len(values)] = values
        handle = _engine.ppo_create(checkpoint)
        self.assertEqual(encoded(checkpoint), encoded(_engine.ppo_checkpoint(handle)))
        packed = copy.deepcopy(checkpoint)
        packed["network"][0]["weights"] = memoryview(
            array("f", packed["network"][0]["weights"])
        ).toreadonly()
        handle = _engine.ppo_create(packed)
        self.assertEqual(encoded(checkpoint), encoded(_engine.ppo_checkpoint(handle)))

    def test_unsupported_float32_buffer_shapes_reject(self):
        checkpoint = state()
        values = array("f", checkpoint["network"][0]["weights"])
        noncontiguous = memoryview(values)[::2]
        multidimensional = memoryview(values).cast("B").cast("f", [1, len(values)])
        for buffer in (noncontiguous, multidimensional):
            candidate = copy.deepcopy(checkpoint)
            candidate["network"][0]["weights"] = buffer
            with (
                self.subTest(shape=buffer.shape),
                self.assertRaisesRegex(ValueError, "contiguous and one-dimensional"),
            ):
                _engine.ppo_create(candidate)
        # A rejected import must also release borrowed memoryview exports.
        noncontiguous.release()
        multidimensional.release()
        values.append(0.0)

    def test_nonmodel_float64_buffers_keep_exact_values(self):
        checkpoint = state()
        value = 0.12345678901234567
        checkpoint["config"]["buffer"] = array("d", [value, -0.0])
        checkpoint["config"]["empty_fp32_buffer"] = array("f")
        handle = _engine.ppo_create(checkpoint)
        exported = _engine.ppo_checkpoint(handle)
        self.assertEqual(exported["config"]["buffer"], [value, -0.0])
        self.assertEqual(
            struct.pack("=d", exported["config"]["buffer"][0]),
            struct.pack("=d", value),
        )
        self.assertEqual(exported["config"]["empty_fp32_buffer"], [])


if __name__ == "__main__":
    unittest.main()
