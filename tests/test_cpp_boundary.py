"""Safety and ownership at the optional Python/C++ engine boundary."""

# These fixtures inspect handles to exercise the compiled boundary directly.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring
# The optional extension exposes its test probes only in the compiled module.
# pylint: disable=c-extension-no-member


import copy
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from tests.native_core_support import _engine, encoded
from tests.test_cpp_ppo import observation, settings
from tests.test_cpp_session import configuration
from wtf.experiment.native import NativeExperiment
from wtf.learning.brain import Experience
from wtf.learning.ppo import PPOBrain


@unittest.skipUnless(_engine is not None, "Build the optional native PPO engine")
class NativeBoundaryTests(unittest.TestCase):
    def make_brain(self):
        return PPOBrain(settings([3]), ["rest", "eat"], 17, 901)

    def test_invalid_pending_samples_reject_before_unchecked_access(self):
        brain = self.make_brain()
        brain.choose(observation(0))
        state = brain.checkpoint()
        for label, pending in (
            ("missing fields", [observation(0)]),
            ("wrong width", [[0.0], 0, -0.7, 0.0]),
            ("negative action", [observation(0), -1, -0.7, 0.0]),
            ("action beyond output", [observation(0), 2, -0.7, 0.0]),
            ("nonfinite action", [observation(0), float("inf"), -0.7, 0.0]),
        ):
            candidate = copy.deepcopy(state)
            candidate["pending"] = pending
            with self.subTest(label=label), self.assertRaises(ValueError):
                _engine.ppo_create(candidate)
        # A rejected import does not poison a later, valid handle.
        handle = _engine.ppo_create(state)
        self.assertEqual(encoded(state), encoded(_engine.ppo_checkpoint(handle)))

    def test_invalid_rollout_samples_reject_before_unchecked_access(self):
        brain = self.make_brain()
        current, following = observation(0), observation(1)
        action = brain.choose(current)
        brain.learn(Experience(current, action, 0.25, following, False, False))
        state = brain.checkpoint()
        self.assertEqual(len(state["rollout"]), 1)
        for key, value in (
            ("observation", [0.0]),
            ("action", -1),
            ("action", 2),
            ("action", 2**63),
        ):
            candidate = copy.deepcopy(state)
            candidate["rollout"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                _engine.ppo_create(candidate)
        handle = _engine.ppo_create(state)
        self.assertEqual(encoded(state), encoded(_engine.ppo_checkpoint(handle)))

    def test_cycles_reject_and_shared_inputs_remain_detached(self):
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises((RecursionError, ValueError)):
            _engine.ppo_create(cyclic)
        state = self.make_brain().checkpoint()
        shared = {"text": "embedded\x00text", "values": [1, 2.0, -0.0]}
        state["config"].update(first=shared, second=shared)
        expected = encoded(state)
        left = _engine.ppo_create(state)
        right = _engine.ppo_create(state)
        state["network"][0]["weights"][0] = 123.0
        shared["values"][0] = 456
        exported = _engine.ppo_checkpoint(left)
        self.assertEqual(expected, encoded(exported))
        exported["network"][0]["weights"][0] = 789.0
        exported["config"]["first"]["values"][0] = 987
        self.assertEqual(expected, encoded(_engine.ppo_checkpoint(left)))
        current, following = observation(0), observation(1)
        action = _engine.ppo_choose(left, current, True)
        _engine.ppo_learn(left, current, action, 1.0, following, True, False)
        self.assertNotEqual(expected, encoded(_engine.ppo_checkpoint(left)))
        self.assertEqual(expected, encoded(_engine.ppo_checkpoint(right)))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO synchronization requires POSIX")
    def test_running_session_releases_gil_and_rejects_concurrent_queries(self):
        # A FIFO blocks capture opening inside native run, after it owns the
        # session lock and releases the GIL. This avoids relying on CPU speed or
        # a long simulation to create the concurrency window.
        session = NativeExperiment(configuration(2))
        entered = threading.Event()
        errors = []
        reports = []
        with tempfile.TemporaryDirectory(prefix="wtf-native-boundary-") as temporary:
            path = Path(temporary) / "capture.fifo"
            os.mkfifo(path)

            def run():
                entered.set()
                try:
                    reports.append(session.run(1, capture_path=path))
                # Transfer any worker failure to assertions on the test thread.
                # pylint: disable-next=broad-exception-caught
                except Exception as error:
                    errors.append(error)

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            self.assertTrue(entered.wait(2))
            deadline = time.monotonic() + 5
            busy = False
            try:
                while worker.is_alive() and time.monotonic() < deadline:
                    try:
                        session.snapshot()
                    except ValueError as error:
                        self.assertIn("busy", str(error))
                        busy = True
                        break
                    time.sleep(0.001)
            finally:
                # Nonblocking opening/draining always releases a pending writer,
                # including when an assertion failed before observing the lock.
                descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                try:
                    while worker.is_alive() and time.monotonic() < deadline:
                        try:
                            os.read(descriptor, 65536)
                        except BlockingIOError:
                            pass
                        worker.join(0.001)
                finally:
                    os.close(descriptor)
                worker.join(1)
            self.assertFalse(
                worker.is_alive(), "Native run did not finish after draining"
            )
            self.assertTrue(
                busy, "Concurrent Python query never observed native ownership"
            )
            self.assertEqual(errors, [])
            self.assertEqual(reports[0]["tick"], 1)
            # A rejected concurrent read leaves the session usable afterward.
            self.assertEqual(session.snapshot()["tick"], 1)


if __name__ == "__main__":
    unittest.main()
