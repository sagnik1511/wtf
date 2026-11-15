import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecosystem.config import validate
from ecosystem.runner import Experiment
from tests.test_simulation import config


class RandomizationTests(unittest.TestCase):
    def random_config(self):
        c = config()
        c["experiment"].update(world_seed_policy="random", runs=4)
        c["world"]["max_steps"] = 2
        return c

    def test_same_config_new_invocations_have_different_worlds_and_brains(self):
        c = self.random_config()
        original = copy.deepcopy(c)
        with patch("ecosystem.runner.secrets.randbits", side_effect=[12345, 67890]):
            a, b = Experiment(c), Experiment(c)
        self.assertEqual(c, original)
        self.assertEqual(a.config, b.config)
        self.assertNotEqual(a.world.seed, b.world.seed)
        self.assertNotEqual(a.world.snapshot(), b.world.snapshot())
        self.assertNotEqual(a.brains[0].weights, b.brains[0].weights)
        self.assertEqual(a.brains[0].weights, a.brains[1].weights)

    def test_each_run_has_unique_seed_and_retains_learning(self):
        e = Experiment(self.random_config())
        brain = e.brains[0]
        initial = []
        while not e.done:
            if e.finished_run:
                e.next_run()
            if e.world.tick == 0:
                initial.append(e.world.snapshot())
            self.assertIs(e.brains[0], brain)
            e.step()
        self.assertEqual(len(e.run_seeds), 4)
        self.assertEqual(len(set(e.run_seeds)), 4)
        self.assertEqual(len(initial), 4)
        self.assertEqual(brain.steps, 8)

    def test_reset_brains_are_fresh_in_randomized_runs(self):
        c = self.random_config()
        c["experiment"]["retain_brains_between_runs"] = False
        e = Experiment(c)
        weights = copy.deepcopy(e.brains[0].weights)
        e.step()
        e.step()
        e.next_run()
        self.assertNotEqual(e.brains[0].weights, weights)
        self.assertEqual(e.brains[0].steps, 0)
        self.assertEqual(e.brains[0].weights, e.brains[1].weights)

    def test_random_checkpoint_continues_future_seed_sequence_exactly(self):
        for retain in (True, False):
            with self.subTest(retain=retain):
                c = self.random_config()
                c["experiment"]["retain_brains_between_runs"] = retain
                a = Experiment(c)
                a.step()
                saved = json.loads(json.dumps(a.checkpoint()))
                with patch(
                    "ecosystem.runner.secrets.randbits",
                    side_effect=AssertionError("Resume must not choose fresh entropy"),
                ):
                    b = Experiment.restore(saved)
                while not a.done:
                    a.step()
                    b.step()
                self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_fixed_increment_and_legacy_checkpoint_still_reproduce(self):
        for policy in ("fixed", "increment"):
            c = config()
            c["experiment"]["world_seed_policy"] = policy
            a, b = Experiment(c), Experiment(c)
            a.step()
            b.step()
            self.assertEqual(a.checkpoint(), b.checkpoint())
            state = a.checkpoint()
            del state["randomization_seed"]
            del state["seed_rng_state"]
            restored = Experiment.restore(json.loads(json.dumps(state)))
            while not a.done:
                a.step()
                restored.step()
            self.assertEqual(a.checkpoint(), restored.checkpoint())
            self.assertEqual(a.run_seeds, [42, 42 if policy == "fixed" else 43])

    def test_invalid_seed_policy_is_rejected(self):
        c = config()
        c["experiment"]["world_seed_policy"] = "typo"
        with self.assertRaisesRegex(ValueError, "world_seed_policy"):
            validate(c)

    def test_cli_explicit_seed_is_reproducible_with_random_default(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for policy, expected in ((None, [7, 8]), ("fixed", [7, 7])):
                output = Path(directory) / (policy or "increment")
                args = [
                    sys.executable,
                    "-m",
                    "ecosystem.cli",
                        "--no-library",
                    "environments/baseline.json",
                    "--runs",
                    "2",
                    "--steps",
                    "1",
                    "--seed",
                    "7",
                    "--output",
                    str(output),
                    "--no-record",
                ]
                if policy:
                    args += ["--seed-policy", policy]
                subprocess.run(args, cwd=root, check=True, capture_output=True)
                folder = next(output.iterdir())
                self.assertEqual(
                    json.loads((folder / "run_seeds.json").read_text()), expected
                )
                c = json.loads((folder / "environment.json").read_text())
                self.assertEqual(
                    c["experiment"]["world_seed_policy"], policy or "increment"
                )
