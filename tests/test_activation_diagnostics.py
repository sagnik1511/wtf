"""Compact actual-animal activation samples preserve controller and world state."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ecosystem.config import validate
from ecosystem.recording import Recorder
from ecosystem.replay_io import load_chunk
from ecosystem.runner import Experiment
from tests.test_mutation import KINDS, environment


class ActivationDiagnosticsTests(unittest.TestCase):
    def test_all_algorithms_sample_actual_layers_without_changing_any_state(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                e = Experiment(environment(kind, 0))
                i = next(iter(e.brains))
                b = e.brains[i]
                o = e.observations[i]
                if kind == "recurrent_ppo":
                    b.hidden = [(j + 1) / 100 for j in range(len(b.hidden))]
                before = copy.deepcopy(b.checkpoint())
                old = b.diagnostics(o)
                result = b.diagnostics(o, include_activations=True)
                sample = result.pop("activations")
                self.assertEqual(result, old)
                self.assertEqual(b.checkpoint(), before)
                self.assertEqual(sample["version"], 1)
                widths = (
                    [b.hidden]
                    if kind == "q_network"
                    else b.config["network"]["hidden_layers"]
                )
                if kind == "recurrent_ppo":
                    widths = [b.config["recurrent_size"]] + widths
                    for actual, expected in zip(
                        sample["recurrent_prior"], before["hidden"]
                    ):
                        self.assertAlmostEqual(actual, expected, places=7)
                    _, current = b.network.step(o, b.hidden)
                    for actual, expected in zip(sample["layers"][0], current):
                        self.assertAlmostEqual(actual, expected, places=7)
                self.assertEqual([len(layer) for layer in sample["layers"]], widths)
                self.assertNotIn("weights", sample)
                sample["layers"][0][0] += 1
                self.assertEqual(b.checkpoint(), before)

    def test_one_inference_pass_and_no_unused_critic_tape(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                e = Experiment(environment(kind, 0))
                i = next(iter(e.brains))
                b = e.brains[i]
                o = e.observations[i]
                if kind == "q_network":
                    owner, name = b, "forward"
                elif kind == "recurrent_ppo":
                    owner, name = b.network, "step"
                elif kind == "mappo":
                    owner, name = b.network.actor, "forward"
                else:
                    owner, name = b.network, "forward"
                with patch.object(owner, name, wraps=getattr(owner, name)) as forward:
                    if kind == "mappo":
                        with patch.object(
                            b.network.critic,
                            "forward",
                            side_effect=AssertionError("unused critic tape"),
                        ):
                            b.diagnostics(o, include_activations=True)
                    else:
                        b.diagnostics(o, include_activations=True)
                    self.assertEqual(forward.call_count, 1)

    def test_recording_roundtrips_all_animals_and_keeps_simulation_identical(self):
        for kind in KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                c = environment(kind, 0)
                c["world"]["max_steps"] = 3
                c["recording"].update(
                    encoding="binary",
                    compression="xz",
                    frames_per_chunk=2,
                    diagnostics_every_steps=1,
                )
                a, b = Experiment(c), Experiment(c)
                off = copy.deepcopy(a.config)
                off["recording"]["record_activations"] = False
                one = Recorder(Path(folder) / "on", a.config, "train")
                two = Recorder(Path(folder) / "off", off, "train")
                one.begin_run(a)
                two.begin_run(b)
                while not a.finished_run:
                    a.step()
                    b.step()
                    one.capture(a, force=a.finished_run)
                    two.capture(b, force=b.finished_run)
                    self.assertEqual(a.checkpoint(), b.checkpoint())
                one.end_run(a)
                two.end_run(b)
                one.close()
                two.close()
                for left, right in zip(
                    sorted((Path(folder) / "on/playback").glob("*.bin.xz")),
                    sorted((Path(folder) / "off/playback").glob("*.bin.xz")),
                ):
                    frames, expected = load_chunk(left), load_chunk(right)
                    for frame, reference in zip(frames, expected):
                        if "diagnostics" in frame:
                            self.assertEqual(
                                set(frame["diagnostics"]),
                                {str(animal["id"]) for animal in frame["animals"]},
                            )
                            for diagnostic in frame["diagnostics"].values():
                                self.assertIn("activations", diagnostic)
                                diagnostic.pop("activations")
                        self.assertEqual(frame, reference)

    def test_only_visualization_values_are_rounded_to_fp32(self):
        from array import array
        from ecosystem.activation_diagnostics import activation_sample

        value = 0.123456789012345
        original = [value]
        info = {"features": [value], "action_values": {"rest": value}}
        result = activation_sample(info, [original], original)
        expected = list(array("f", original))
        self.assertEqual(result["activations"]["layers"], [expected])
        self.assertEqual(result["activations"]["recurrent_prior"], expected)
        self.assertNotEqual(expected, original)
        self.assertEqual(result["features"], original)
        self.assertEqual(result["action_values"]["rest"], value)
        self.assertEqual(original, [value])

    def test_capture_follows_diagnostic_schedule_and_disable_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            c = environment("ppo", 0)
            c["recording"].update(diagnostics_every_steps=2, frames_per_chunk=64)
            e = Experiment(c)
            r = Recorder(Path(folder), e.config, "train")
            r.begin_run(e)
            with patch.object(
                next(iter(e.brains.values())),
                "diagnostics",
                side_effect=AssertionError("unsampled computation"),
            ):
                e.step()
                r.capture(e)
            r.close()
        c = environment("ppo", 0)
        c["recording"]["record_activations"] = "yes"
        with self.assertRaises(ValueError):
            validate(c)
