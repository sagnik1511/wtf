import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from ecosystem.brain import Brain, Experience
from ecosystem.brains import create_brain, restore_brain
from ecosystem.config import load, validate
from ecosystem.ppo import (
    MLP,
    PPOBrain,
    RolloutStep,
    advantages_and_returns,
    loss_and_gradient,
    policy_distribution,
)
from ecosystem.recording import Recorder
from ecosystem.runner import Experiment
from tests.test_simulation import config

ROOT = Path(__file__).resolve().parents[1]


def ppo_settings():
    p = copy.deepcopy(load(ROOT / "environments/ppo.json")["brains"]["small_ppo"])
    p["network"]["hidden_layers"] = [4, 3]
    p.update(rollout_steps=4, batch_size=2, epochs=2, target_kl=0)
    return p


def ppo_environment(mixed=False):
    c = config()
    c["brains"]["small_ppo"] = ppo_settings()
    c["species"]["predator"]["brain"] = "small_ppo"
    if not mixed:
        c["species"]["prey"]["brain"] = "small_ppo"
    c["world"]["max_steps"] = 7
    c["experiment"]["runs"] = 3
    return c


class PPOMathTests(unittest.TestCase):
    def test_softmax_is_normalized_and_stable(self):
        for logits in ([1000.0, -1000.0, 999.0], [-1000.0, -1001.0], [0.0, 0.0]):
            p, logs = policy_distribution(logits)
            self.assertAlmostEqual(sum(p), 1.0)
            self.assertTrue(all(math.isfinite(x) for x in logs))
            self.assertTrue(all(0 <= x <= 1 for x in p))

    def test_gae_bootstraps_time_limits_and_breaks_episode_traces(self):
        o = (0.0,) * 20
        rollout = [
            RolloutStep(o, 0, 1.0, 0.0, 0.5, 0.6, False, False),
            RolloutStep(o, 0, 2.0, 0.0, 0.6, 0.7, False, True),
            RolloutStep(o, 0, 100.0, 0.0, 0.7, 999.0, True, False),
        ]
        advantages, returns = advantages_and_returns(rollout, 0.9, 0.8)
        self.assertAlmostEqual(advantages[1], 2.03)
        self.assertAlmostEqual(advantages[0], 1.04 + 0.72 * 2.03)
        self.assertAlmostEqual(advantages[2], 99.3)
        self.assertAlmostEqual(returns[0], advantages[0] + 0.5)

    def test_clipped_policy_derivatives_match_finite_differences(self):
        c = ppo_settings()
        c.update(entropy_coefficient=0.03, value_coefficient=0.6)
        outputs = [0.2, -0.1, 0.3]
        old = policy_distribution(outputs[:-1])[1][0] + 0.05
        _, gradient, _ = loss_and_gradient(outputs, 0, old, 0.7, 0.8, c)
        for i in range(len(outputs)):
            plus, minus = outputs[:], outputs[:]
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numeric = (
                loss_and_gradient(plus, 0, old, 0.7, 0.8, c)[0]
                - loss_and_gradient(minus, 0, old, 0.7, 0.8, c)[0]
            ) / 2e-6
            self.assertAlmostEqual(gradient[i], numeric, places=6)

    def test_clipping_stops_only_the_overoptimistic_branch(self):
        c = ppo_settings()
        c.update(entropy_coefficient=0.0, value_coefficient=0.0)
        outputs = [math.log(0.4), math.log(0.6), 0.0]
        for advantage, ratio, clipped in (
            (1.0, 1.6, True),
            (-1.0, 0.5, True),
            (1.0, 0.5, False),
            (-1.0, 1.6, False),
        ):
            _, gradient, _ = loss_and_gradient(
                outputs, 0, math.log(0.4 / ratio), advantage, 0.0, c
            )
            if clipped:
                self.assertEqual(gradient, [0.0, 0.0, 0.0])
            else:
                self.assertGreater(sum(abs(g) for g in gradient), 0.0)

    def test_multilayer_backprop_matches_finite_differences(self):
        network = MLP([3, 2], 2, 14)
        observation = tuple((i - 10) / 20 for i in range(20))
        outputs, activations = network.forward(observation)
        c = ppo_settings()
        old = policy_distribution(outputs[:-1])[1][1]
        gradients = network.zeros()
        _, derivative, _ = loss_and_gradient(outputs, 1, old, -0.6, 0.8, c)
        network.accumulate_gradient(activations, derivative, gradients)
        for layer_index, layer in enumerate(network.layers):
            for key, values in layer.items():
                for index in range(len(values)):
                    original = values[index]
                    values[index] = original + 1e-6
                    plus = loss_and_gradient(
                        network.forward(observation)[0], 1, old, -0.6, 0.8, c
                    )[0]
                    values[index] = original - 1e-6
                    minus = loss_and_gradient(
                        network.forward(observation)[0], 1, old, -0.6, 0.8, c
                    )[0]
                    values[index] = original
                    self.assertAlmostEqual(
                        gradients[layer_index][key][index],
                        (plus - minus) / 2e-6,
                        places=5,
                    )


class PPOBrainTests(unittest.TestCase):
    def test_rewarded_action_is_learned(self):
        c = ppo_settings()
        c["network"]["hidden_layers"] = [4]
        c.update(learning_rate=0.01, epochs=4)
        b = PPOBrain(c, ["rest", "eat"], 7, 9)
        o = (0.0,) * 20
        for _ in range(100):
            action = b.choose(o)
            b.learn(Experience(o, action, 1.0 if action == "eat" else -1.0, None, True))
        self.assertGreater(b.diagnostics(o)["action_probabilities"]["eat"], 0.9)

    def test_individuals_do_not_share_learning_or_optimizer_state(self):
        c = ppo_settings()
        a, b = PPOBrain(c, ["rest", "eat"], 1, 2), PPOBrain(c, ["rest", "eat"], 1, 3)
        self.assertEqual(a.network.layers, b.network.layers)
        saved = b.checkpoint()
        o = (0.0,) * 20
        action = a.choose(o)
        a.learn(Experience(o, action, 1.0, None, True))
        self.assertNotEqual(a.network.layers, b.network.layers)
        self.assertEqual(b.checkpoint(), saved)

    def test_rollouts_are_bounded_consumed_and_flushed_at_episode_end(self):
        b = PPOBrain(ppo_settings(), ["rest", "eat"], 1, 2)
        o = (0.0,) * 20
        for _ in range(8):
            action = b.choose(o)
            b.learn(Experience(o, action, 1.0, o, False))
            self.assertLess(len(b.rollout), 4)
        self.assertEqual(b.rollout_updates, 2)
        for terminal in (False, True):
            action = b.choose(o)
            b.learn(
                Experience(
                    o, action, 0.5, None if terminal else o, terminal, not terminal
                )
            )
            self.assertEqual(b.rollout, [])
        self.assertEqual(b.rollout_updates, 4)

    def test_evaluation_and_diagnostics_do_not_change_learning_state(self):
        b = PPOBrain(ppo_settings(), ["rest", "eat"], 1, 2)
        o = (0.0,) * 20
        saved = b.checkpoint()
        for _ in range(4):
            b.choose(o, explore=False)
            b.diagnostics(o)
        self.assertEqual(saved, b.checkpoint())

    def test_pending_action_and_optimizer_restore_exactly(self):
        a = PPOBrain(ppo_settings(), ["rest", "eat"], 1, 2)
        o = (0.0,) * 20
        action = a.choose(o)
        b = restore_brain(json.loads(json.dumps(a.checkpoint())))
        experience = Experience(o, action, 1.0, o, False)
        a.learn(experience)
        b.learn(experience)
        for _ in range(10):
            aa, ab = a.choose(o), b.choose(o)
            self.assertEqual(aa, ab)
            a.learn(Experience(o, aa, 0.5, o, False))
            b.learn(Experience(o, ab, 0.5, o, False))
        self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_kl_cap_stops_remaining_policy_updates(self):
        c = ppo_settings()
        c.update(
            learning_rate=0.03, epochs=10, target_kl=1e-12, normalize_advantages=False
        )
        b = PPOBrain(c, ["rest", "eat"], 1, 2)
        o = (0.1,) * 20
        for _ in range(4):
            action = b.choose(o)
            b.learn(Experience(o, action, 1.0 if action == "eat" else -1.0, o, False))
        self.assertTrue(b.last_metrics["early_stopped"])
        self.assertLess(b.updates, 20)
        self.assertGreater(b.updates, 0)
        self.assertFalse(b.rollout)

    def test_gradient_clipping_bounds_first_adam_moment(self):
        c = ppo_settings()
        b = PPOBrain(c, ["rest", "eat"], 1, 2)
        gradients = [
            {key: [10.0] * len(v) for key, v in layer.items()}
            for layer in b.network.layers
        ]
        b._adam_step(gradients, 1)
        moment_norm = math.sqrt(
            sum(
                v * v
                for layer in b.moments
                for values in layer.values()
                for v in values
            )
        )
        self.assertAlmostEqual(moment_norm, 0.1 * c["max_grad_norm"])

    def test_old_q_checkpoints_still_dispatch_correctly(self):
        c = config()
        brain = create_brain(c["brains"]["small_q"], ["rest", "eat"], 1, 2)
        self.assertIsInstance(brain, Brain)
        restored = restore_brain(json.loads(json.dumps(brain.checkpoint())))
        self.assertEqual(brain.checkpoint(), restored.checkpoint())


class PPOIntegrationTests(unittest.TestCase):
    def test_config_validation(self):
        c = ppo_environment()
        validate(c)
        cases = [
            ("batch_size", 5),
            ("rollout_steps", 0),
            ("normalize_advantages", 1),
            ("clip_ratio", 0),
            ("network", {"type": "mlp", "hidden_layers": []}),
            ("network", {"type": "mlp", "hidden_layers": [4], "activation": "relu"}),
        ]
        for key, value in cases:
            bad = copy.deepcopy(c)
            bad["brains"]["small_ppo"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(bad)

    def test_ppo_and_mixed_checkpoint_continuation_across_runs(self):
        for mixed in (False, True):
            for retain in (False, True):
                with self.subTest(mixed=mixed, retain=retain):
                    c = ppo_environment(mixed)
                    c["experiment"]["retain_brains_between_runs"] = retain
                    a = Experiment(c)
                    a.step()
                    a.step()  # Partial on-policy buffers in the checkpoint.
                    b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
                    while not a.done:
                        a.step()
                        b.step()
                    self.assertEqual(a.checkpoint(), b.checkpoint())
                    for brain in a.brains.values():
                        if isinstance(brain, PPOBrain):
                            self.assertFalse(brain.rollout)
                            self.assertGreater(brain.updates, 0)

    def test_evaluation_runs_do_not_learn(self):
        e = Experiment(ppo_environment(True), mode="evaluate")
        saved = {i: b.checkpoint() for i, b in e.brains.items()}
        for _ in range(4):
            e.step()
        self.assertEqual(saved, {i: b.checkpoint() for i, b in e.brains.items()})

    def test_recording_is_transparent_and_saves_algorithm_specific_diagnostics(self):
        c = ppo_environment(True)
        a, control = Experiment(c), Experiment(c)
        with tempfile.TemporaryDirectory() as folder:
            writer = Recorder(folder, c, "train")
            writer.begin_run(a)
            for _ in range(5):
                a.step()
                control.step()
                writer.capture(a)
            writer.close(completed=False)
            self.assertEqual(a.checkpoint(), control.checkpoint())
            frames = json.loads(
                next((Path(folder) / "playback").glob("*chunk*.json")).read_text()
            )
            diagnostics = frames[0]["diagnostics"]
            self.assertEqual(diagnostics["0"]["algorithm"], "q_learning")
            self.assertIn("action_values", diagnostics["0"])
            self.assertEqual(diagnostics["3"]["algorithm"], "ppo")
            self.assertAlmostEqual(
                sum(diagnostics["3"]["action_probabilities"].values()), 1.0
            )
            self.assertNotIn("network", diagnostics["3"])
            self.assertNotIn("rollout", diagnostics["3"])
            self.assertEqual(a.snapshot(3)["selected"]["algorithm"], "ppo")

    def test_random_policy_baseline_does_not_collect_or_train(self):
        e = Experiment(ppo_environment(), mode="random")
        for _ in range(5):
            e.step()
        self.assertTrue(
            all(
                not b.rollout and b.steps == 0 and b.updates == 0
                for b in e.brains.values()
            )
        )

    def test_streamed_mixed_checkpoint_file_is_equivalent(self):
        e = Experiment(ppo_environment(True))
        for _ in range(5):
            e.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.json"
            e.write_checkpoint(path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved, json.loads(json.dumps(e.checkpoint())))
            restored = Experiment.restore(saved)
            e.step()
            restored.step()
            self.assertEqual(e.checkpoint(), restored.checkpoint())
