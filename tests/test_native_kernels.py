"""Native/reference arithmetic, backend switching and malformed-input checks."""

from array import array
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ecosystem import kernels
from ecosystem.numeric import Float32, precision_scope
from ecosystem.ppo import MLP
from ecosystem.runner import Experiment
from tests.test_numeric_precision import KINDS, settings
from tests.test_simulation import config

try:
    from ecosystem import _kernels
except ImportError:
    _kernels = None


@unittest.skipIf(
    _kernels is None or not (3, 12) <= sys.version_info[:2] < (3, 14),
    "Build supported native kernels first",
)
class NativeKernelTests(unittest.TestCase):
    def test_float_reductions_match_compensated_python_sum(self):
        # The old left-to-right native reduction returned zero for this input.
        cases = (
            [1e16, 1.0, -1e16],
            [1e16, 1.0, 1.0, -1e16],
            [0.1] * 257,
            [-0.0],
            [float("inf"), 1.0],
            [1e308, 1e308],
            [float("inf"), -float("inf")],
        )
        for values in cases:
            with self.subTest(values=values[:4]):
                got = _kernels.affine(
                    values, Float32([1.0] * len(values)), Float32([0.0]), False, 0, 1
                )[0]
                expected = 0.0 + sum(values)
                if math.isnan(expected):
                    self.assertTrue(math.isnan(got))
                else:
                    self.assertEqual(got, expected)
                    self.assertEqual(
                        math.copysign(1.0, got), math.copysign(1.0, expected)
                    )
        # Fused inference has a separate dense loop from affine().
        network = MLP([], 1, 7, inputs=3)
        network.layers[0]["weights"][:] = Float32([1.0] * 6)
        network.layers[0]["bias"][:] = Float32([0.0, 0.0])
        with patch.object(kernels, "native", _kernels):
            self.assertEqual(network.predict((1e16, 1.0, -1e16)), [1.0, 1.0])

    def test_dense_forward_ranges_and_gradients_are_bit_exact(self):
        rng = random.Random(123)
        for width in (20, 32):
            for hidden in ([], [4], [5, 3]):
                a = MLP(hidden, 3, 7, inputs=width)
                b = copy.deepcopy(a)
                for _ in range(6):
                    observation = tuple(rng.uniform(-1, 1) for _ in range(width))
                    derivative = [rng.uniform(-1, 1) for _ in range(4)]
                    ga, gb = a.zeros(), b.zeros()
                    for propagate in (False, True):
                        with patch.object(kernels, "native", _kernels):
                            outputs, tape = a.forward(observation)
                            got = a.accumulate_gradient(tape, derivative, ga, propagate)
                            inferred = a.predict(observation)
                            value = a.value(observation)
                            selected = a.predict_range(observation, 1, 3)
                        with patch.object(kernels, "native", None):
                            expected, tape = b.forward(observation)
                            reference = b.accumulate_gradient(
                                tape, derivative, gb, propagate
                            )
                        self.assertEqual(outputs, expected)
                        self.assertEqual(inferred, expected)
                        self.assertEqual(value, expected[-1])
                        self.assertEqual(selected, expected[1:3])
                        self.assertEqual(got, reference)
                        self.assertEqual(ga, gb)

    def test_all_algorithms_and_backend_switches_continue_exactly(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = config()
                c["world"]["max_steps"] = 12
                c["experiment"].update(runs=2, world_seed_policy="increment")
                c["brains"] = {"test": settings(kind)}
                for species in c["species"].values():
                    species["brain"] = "test"
                if kind == "q_network":
                    c["brains"]["test"].update(
                        batch_size=2, learning_starts_after=2, train_every_steps=1
                    )
                elif kind == "grpo":
                    c["brains"]["test"]["train_every_steps"] = 2
                with patch.object(kernels, "native", _kernels):
                    a = Experiment(c)
                with patch.object(kernels, "native", None):
                    b = Experiment(c)
                for tick in range(24):
                    with patch.object(
                        kernels, "native", _kernels if tick % 4 < 2 else None
                    ):
                        a.step()
                    with patch.object(kernels, "native", None):
                        b.step()
                    self.assertEqual(a.checkpoint(), b.checkpoint())
                    if tick == 6:
                        a = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
                        b = Experiment.restore(json.loads(json.dumps(b.checkpoint())))

    def test_adam_and_sgd_match_reference_with_nonzero_momentum_and_clipping(self):
        from ecosystem.brains import create_brain

        rng = random.Random(39)
        for magnitude in (0.001, 50.0):
            a = create_brain(settings("ppo"), ["rest", "eat"], 7, 11)
            b = copy.deepcopy(a)
            for step in range(12):
                ga = a.network.zeros()
                for layer in ga:
                    for values in layer.values():
                        for i in range(len(values)):
                            values[i] = (
                                rng.uniform(-magnitude, magnitude) if step % 3 else 0.0
                            )
                with patch.object(kernels, "native", _kernels):
                    a._adam_step(ga, 3)
                with patch.object(kernels, "native", None):
                    b._adam_step(ga, 3)
                self.assertEqual(a.checkpoint(), b.checkpoint())
        p = Float32([0.1, -0.2, 0.3])
        g = array("d", [0.1, 1e-8, -10.0])
        expected = p.copy()
        for i, v in enumerate(g):
            expected[i] -= 0.003 * v
        _kernels.sgd(p, g, 0.003)
        self.assertEqual(p, expected)

    def test_invalid_dimensions_and_buffer_types_fail_before_writes(self):
        f = Float32([1.0, 2.0, 3.0, 4.0])
        d = array("d", [0.0] * 4)
        for call in (
            lambda: _kernels.affine(
                [1.0, 2.0, 3.0], f, Float32([0.0, 0.0]), False, 0, 2
            ),
            lambda: _kernels.affine([1.0, 2.0], f, Float32([0.0, 0.0]), False, -1, 1),
            lambda: _kernels.affine(
                [1.0, 2.0], array("d", f), Float32([0.0, 0.0]), False, 0, 2
            ),
            lambda: _kernels.backward(
                [1.0, 2.0], [1.0, 2.0], f, d, array("d", [0.0]), True
            ),
            lambda: _kernels.sgd(f, array("d", [1.0]), 0.1),
            lambda: _kernels.adam([(f, f, f, array("d", [1.0]))], 1, 0.5, 0.1, 1),
            lambda: _kernels.q_backward(
                [1.0, 2.0],
                [1.0, 2.0],
                f,
                2,
                d,
                array("d", [0.0, 0.0]),
                d,
                array("d", [0.0, 0.0]),
                1.0,
            ),
        ):
            with self.assertRaises((ValueError, TypeError)):
                call()
            self.assertEqual(f, Float32([1.0, 2.0, 3.0, 4.0]))
            self.assertEqual(d, array("d", [0.0] * 4))

    def test_recorded_binary_frames_match_through_chunk_boundaries(self):
        from ecosystem.recording import Recorder, RecordingStore
        from ecosystem.replay_io import load_chunk

        c = config()
        c["world"]["max_steps"] = 9
        c["experiment"].update(runs=1, world_seed_policy="increment")
        c["recording"].update(
            encoding="binary",
            compression="xz",
            frames_per_chunk=4,
            diagnostics_every_steps=1,
        )
        decoded = []
        with tempfile.TemporaryDirectory() as directory:
            for name, backend in (("native", _kernels), ("python", None)):
                with patch.object(kernels, "native", backend):
                    e = Experiment(c)
                    writer = Recorder(Path(directory) / name, c, "train")
                    writer.begin_run(e)
                    while not e.done:
                        e.step()
                        writer.capture(e, force=e.finished_run)
                    writer.end_run(e)
                    writer.close()
                store = RecordingStore(directory)
                run = store.metadata(name)["runs"][0]
                self.assertGreater(run["chunk_count"], 1)
                decoded.append(
                    [
                        frame
                        for j in range(run["chunk_count"])
                        for frame in load_chunk(store.chunk(name, 1, j))
                    ]
                )
        self.assertEqual(decoded[0], decoded[1])

    def test_legacy_list_weights_never_enter_native_kernels(self):
        with precision_scope("float64"):
            a = MLP([3], 2, 7)
        with patch.object(kernels, "native") as native:
            out, tape = a.forward((0.1,) * 20)
            a.accumulate_gradient(tape, [0.1] * 3, a.zeros())
            a.predict((0.1,) * 20)
            self.assertEqual(native.mock_calls, [])


class BackendSelectionTests(unittest.TestCase):
    def test_future_interpreter_falls_back_until_native_is_validated(self):
        code = """import sys
sys.version_info = (3, 14)
from ecosystem.kernels import status
assert status()['backend'] == 'python'
assert '3.12–3.13' in status()['fallback_reason']
"""
        for mode in ("auto", "native"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                env=dict(os.environ, WTF_KERNELS=mode),
                capture_output=True,
                text=True,
            )
            if mode == "auto":
                self.assertEqual(result.returncode, 0, result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("3.12–3.13", result.stderr)

    def test_explicit_python_backend_never_imports_extension(self):
        code = 'import sys;from ecosystem.kernels import status;assert status()["backend"]=="python";assert "ecosystem._kernels" not in sys.modules'
        p = subprocess.run(
            [sys.executable, "-c", code],
            env=dict(os.environ, WTF_KERNELS="python"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_python_scratch_stays_list_based_and_native_uses_doubles(self):
        values = Float32([1.0, 2.0])
        with patch.object(kernels, "native", None):
            self.assertIs(type(kernels.gradient_zeros(values)), list)
            # Existing buffers must remain assignable if a test changes backends.
            existing = array("d", [3.0, 4.0])
            existing[:] = kernels.gradient_zeros(existing)
            self.assertEqual(existing, array("d", [0.0, 0.0]))
        if _kernels is not None:
            with patch.object(kernels, "native", _kernels):
                self.assertEqual(kernels.gradient_zeros(values).typecode, "d")

    def test_auto_falls_back_and_required_native_explains_missing_build(self):
        code = """import sys
from importlib.abc import MetaPathFinder
class Missing(MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname=='ecosystem._kernels':raise ImportError('hidden for fallback test')
sys.meta_path.insert(0,Missing())
from ecosystem.kernels import status
assert status()['backend']=='python'
"""
        p = subprocess.run(
            [sys.executable, "-c", code],
            env=dict(os.environ, WTF_KERNELS="auto"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        p = subprocess.run(
            [sys.executable, "-c", code],
            env=dict(os.environ, WTF_KERNELS="native"),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("work.build_native", p.stderr)
