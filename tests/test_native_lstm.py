"""Exact LSTM cell kernels, controller continuation and recorded-playback parity."""

from array import array
from contextlib import contextmanager
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from tests.new_policy_support import settings, environment
from wtf.learning.brain import Experience
from wtf.learning.brains import create_brain, restore_brain
from wtf.learning.lstm_ppo import LSTMNetwork
from wtf.native import kernels
from wtf.native.numeric import precision_scope
from wtf.experiment.runner import Experiment
from wtf.storage.recording import Recorder
from wtf.storage.replay_io import load_chunk, chunk_path
from wtf.storage.checkpoint_io import read_checkpoint


@contextmanager
def reference_cell():
    available = kernels.available
    with patch.object(
        kernels,
        "available",
        side_effect=lambda name: (
            False if name in ("lstm_step", "lstm_backward") else available(name)
        ),
    ):
        yield


NATIVE = kernels.native
SUPPORTED = kernels.available("lstm_step") and kernels.available("lstm_backward")


@unittest.skipUnless(SUPPORTED, "Build the optional LSTM kernels first")
class NativeLSTMTests(unittest.TestCase):
    def test_forward_tapes_backward_and_nonzero_gradient_accumulation_are_exact(self):
        for h in (1, 2, 8, 16, 128):
            c = settings("lstm_ppo")
            c.update(recurrent_size=h, observation_version=4, observation_top_k=3)
            network = LSTMNetwork(c, 3, 7)
            rng = random.Random(h)
            x = tuple(rng.uniform(-2, 2) for _ in range(network.inputs))
            old = [rng.uniform(-3, 3) for _ in range(2 * h)]
            before = network.checkpoint()
            saved_old = old[:]
            for tape in (False, True):
                actual = network.step(x, old, tape)
                with reference_cell():
                    expected = network.step(x, old, tape)
                self.assertEqual(actual, expected, h)
            with reference_cell():
                expected = network.step(x, old, value_only=True)
            self.assertEqual(network.step(x, old, value_only=True), expected)
            _, _, tape = network.step(x, old, True)
            ga = network.zeros()
            for layer in ga:
                for values in layer.values():
                    for i in range(len(values)):
                        values[i] = 0.1 + (i % 7) / 10
            gb = copy.deepcopy(ga)
            future = [rng.uniform(-0.5, 0.5) for _ in old]
            derivative = [0.2, -0.3, 0.4, -0.1]
            actual = network.backward(tape, derivative, future, ga)
            with reference_cell():
                expected = network.backward(tape, derivative, future, gb)
            self.assertEqual(actual, expected, h)
            self.assertEqual(ga, gb, h)
            self.assertEqual(network.checkpoint(), before)
            self.assertEqual(old, saved_old)

    def test_native_memory_gradients_match_finite_differences(self):
        network = LSTMNetwork(settings("lstm_ppo"), 2, 7)
        x = (0.2,) * 20
        memory = [0.1, -0.2, 0.3, -0.4]
        _, _, tape = network.step(x, memory, True)
        gradient = network.backward(tape, [1.0, 0.0, 0.0], [0.0] * 4, network.zeros())
        for i in range(4):
            plus = memory[:]
            minus = memory[:]
            plus[i] += 1e-5
            minus[i] -= 1e-5
            actual = (network.step(x, plus)[0][0] - network.step(x, minus)[0][0]) / 2e-5
            self.assertAlmostEqual(gradient[i], actual, places=8)

    def test_saturated_gates_and_signed_values_match_reference(self):
        network = LSTMNetwork(settings("lstm_ppo"), 2, 7)
        for i in range(len(network.cell["bias"])):
            network.cell["bias"][i] = 1000.0 if i % 2 else -1000.0
        memory = [-0.0, 0.1, -100.0, 100.0]
        x = tuple(-0.0 if i % 2 else 0.5 for i in range(20))
        actual = network.step(x, memory, True)
        with reference_cell():
            expected = network.step(x, memory, True)
        self.assertEqual(actual, expected)

    def test_pending_memory_rng_and_optimizer_continue_identically(self):
        c = settings("lstm_ppo")
        c.update(rollout_steps=4, sequence_length=2, epochs=3, batch_size=4)
        a = create_brain(c, ["rest", "attack"], 3, 5)
        b = create_brain(c, ["rest", "attack"], 3, 5)
        for i in range(19):
            x = tuple(((i + j) % 9 - 4) / 10 for j in range(20))
            action = a.choose(x)
            with reference_cell():
                self.assertEqual(b.choose(x), action)
            if i == 2:
                a = restore_brain(a.checkpoint())
                b = restore_brain(b.checkpoint())
            feedback = Experience(
                x, action, (-1.0) ** i, None if i == 8 else x, i == 8, i == 14
            )
            a.learn(feedback)
            with reference_cell():
                b.learn(feedback)
            self.assertEqual(a.checkpoint(), b.checkpoint())
            with reference_cell():
                expected = b.diagnostics(x, True)
            self.assertEqual(a.diagnostics(x, True), expected)

    def test_both_individual_entry_points_can_be_missing_and_fp64_falls_back(self):
        c = settings("lstm_ppo")
        n = LSTMNetwork(c, 2, 7)
        x = (0.1,) * 20
        memory = [0.2] * 4
        available = kernels.available
        expected = n.step(x, memory, True)
        for missing in ("lstm_step", "lstm_backward"):
            with patch.object(
                kernels,
                "available",
                side_effect=lambda name: False if name == missing else available(name),
            ):
                got = n.step(x, memory, True)
                grad = n.zeros()
                previous = n.backward(got[2], [1.0, 0.0, 0.0], [0.0] * 4, grad)
            native_grad = n.zeros()
            native_previous = n.backward(
                expected[2], [1.0, 0.0, 0.0], [0.0] * 4, native_grad
            )
            self.assertEqual(got, expected)
            self.assertEqual(previous, native_previous)
            self.assertEqual(grad, native_grad)
        with precision_scope("float64"):
            legacy = LSTMNetwork(c, 2, 7)
        with (
            patch.object(
                NATIVE, "lstm_step", side_effect=AssertionError("legacy dispatch")
            ),
            patch.object(
                NATIVE, "lstm_backward", side_effect=AssertionError("legacy dispatch")
            ),
        ):
            _, _, tape = legacy.step(x, memory, True)
            legacy.backward(tape, [1.0, 0.0, 0.0], [0.0] * 4, legacy.zeros())

    def test_invalid_native_calls_do_not_mutate_gradients(self):
        n = LSTMNetwork(settings("lstm_ppo"), 2, 7)
        x = (0.1,) * 20
        memory = [0.2] * 4
        _, _, tape = n.step(x, memory, True)
        parts = (*tape[:-1], (0.1, 0.2), (0.0,) * 4)
        for invalid in (
            parts[:-1],
            parts[:-1] + ((0.0,),),
            parts[:7] + (("not numeric", 0.2), parts[-1]),
        ):
            gradients = n.zeros()[0]
            before = copy.deepcopy(gradients)
            with self.assertRaises((TypeError, ValueError)):
                NATIVE.lstm_backward(invalid, n.cell["recurrent_weights"], gradients)
            self.assertEqual(gradients, before)
        for key in ("input_weights", "recurrent_weights", "bias"):
            for value in (
                array("d", [0.0]),
                array("f", [0.0]),
                memoryview(array("d", [0.0])).toreadonly(),
            ):
                gradients = n.zeros()[0]
                gradients[key] = value
                before = {k: list(v) for k, v in gradients.items()}
                with self.assertRaises((TypeError, ValueError, BufferError)):
                    NATIVE.lstm_backward(parts, n.cell["recurrent_weights"], gradients)
                self.assertEqual({k: list(v) for k, v in gradients.items()}, before)
        for bad_memory in ([], [0.0], [0.0] * 3, [0.0] * 258):
            with self.assertRaises(ValueError):
                NATIVE.lstm_step(
                    x,
                    bad_memory,
                    n.cell["input_weights"],
                    n.cell["recurrent_weights"],
                    n.cell["bias"],
                    False,
                )
        with self.assertRaises(ValueError):
            NATIVE.lstm_step(
                [],
                memory,
                n.cell["input_weights"],
                n.cell["recurrent_weights"],
                n.cell["bias"],
                False,
            )
        with self.assertRaises(TypeError):
            NATIVE.lstm_step(
                x, memory, [0.0], n.cell["recurrent_weights"], n.cell["bias"], False
            )

    def test_world_replay_chunks_and_restoration_match_reference(self):
        c = environment("lstm_ppo")
        c["recording"] = dict(
            enabled=True,
            frame_every_steps=1,
            frames_per_chunk=3,
            diagnostics_every_steps=2,
        )
        a = Experiment(c)
        b = Experiment(c)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ra = Recorder(root / "native", c, "train")
            rb = Recorder(root / "reference", c, "train")
            for run in (1, 2):
                ra.begin_run(a)
                with reference_cell():
                    rb.begin_run(b)
                while not a.finished_run:
                    a.step()
                    with reference_cell():
                        b.step()
                    ra.capture(a, force=a.finished_run, metrics=a.last_record)
                    with reference_cell():
                        rb.capture(b, force=b.finished_run, metrics=b.last_record)
                    self.assertEqual(a.checkpoint(), b.checkpoint())
                ra.end_run(a)
                with reference_cell():
                    rb.end_run(b)
                for i in range(ra.manifest["runs"][-1]["chunk_count"]):
                    left = chunk_path(root / "native/playback", run, i)
                    right = chunk_path(root / "reference/playback", run, i)
                    self.assertEqual(load_chunk(left), load_chunk(right))
                    self.assertEqual(left.read_bytes(), right.read_bytes())
                path = root / f"state{run}.bin.xz"
                a.write_checkpoint(path)
                a = Experiment.restore(read_checkpoint(path))
                self.assertEqual(a.checkpoint(), b.checkpoint())
                if run == 1:
                    a.next_run()
                    b.next_run()
            ra.close()
            rb.close()
