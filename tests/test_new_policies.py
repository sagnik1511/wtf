"""Numerical objectives, LSTM BPTT, trust bounds and complete policy lifecycles."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.new_policy_support import KINDS, settings, environment
from wtf.config import validate
from wtf.learning.brain import Experience
from wtf.learning.brains import (
    create_brain,
    restore_brain,
    model_snapshot,
    load_model_weights,
    mix_parent_brains,
)
from wtf.learning import trpo
from wtf.learning.ppo import RolloutStep, policy_distribution
from wtf.learning.discrete_sac import actor_objective, soft_value
from wtf.learning.lstm_ppo import LSTMNetwork
from wtf.native.numeric import precision_scope
from wtf.experiment.runner import Experiment
from wtf.storage.checkpoint_io import read_checkpoint


class NewPolicyMathTests(unittest.TestCase):
    def test_sac_exact_categorical_expectation_and_actor_gradient(self):
        logits = [0.2, -0.4, 0.6]
        first = [1.0, -0.5, 0.7]
        second = [0.4, 0.6, 0.8]
        alpha = 0.2
        loss, gradient, _entropy = actor_objective(logits, first, second, alpha)
        p, logs = policy_distribution(logits)
        self.assertAlmostEqual(loss, -soft_value(p, logs, first, second, alpha))
        for i in range(3):
            plus = logits[:]
            minus = logits[:]
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numerical = (
                actor_objective(plus, first, second, alpha)[0]
                - actor_objective(minus, first, second, alpha)[0]
            ) / 2e-6
            self.assertAlmostEqual(gradient[i], numerical, places=7)

    def test_sac_bootstraps_truncation_but_not_death(self):

        # Explicit indices align the related packed arrays without introducing another
        # captured value.
        # pylint: disable=consider-using-enumerate

        for terminated in (True, False):
            c = settings("discrete_sac")
            c.update(batch_size=1, learning_starts_after=1)
            b = create_brain(c, ["rest", "attack"], 3, 5)
            for layer in b.network.layers:
                for values in layer.values():
                    for i in range(len(values)):
                        values[i] = 0.0
            for i, target in enumerate(b.target_critics):
                for layer in target.layers:
                    for values in layer.values():
                        for j in range(len(values)):
                            values[j] = 0.0
                for j in range(2):
                    target.layers[-1]["bias"][j] = i + 1.0
            with patch.object(b, "_adam_step"):
                b.learn(
                    Experience(
                        (0.0,) * 20,
                        "rest",
                        3.0,
                        None if terminated else (0.0,) * 20,
                        terminated,
                        not terminated,
                    )
                )
            expected = (
                3.0
                if terminated
                else 3.0 + c["discount_factor"] * (1.0 + c["temperature"] * math.log(2))
            )
            self.assertAlmostEqual(b.last_metrics["critic_loss"], expected**2)

    def test_trpo_surrogate_gradient_and_conjugate_gradient(self):
        logits = [0.2, -0.4, 0.6]
        _value, gradient = trpo.surrogate(logits, 1, -1.2, 0.7, 0.03)
        for i in range(3):
            plus = logits[:]
            minus = logits[:]
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numerical = (
                trpo.surrogate(plus, 1, -1.2, 0.7, 0.03)[0]
                - trpo.surrogate(minus, 1, -1.2, 0.7, 0.03)[0]
            ) / 2e-6
            self.assertAlmostEqual(gradient[i], numerical, places=7)
        solved = trpo.conjugate_gradient(
            lambda v: [4 * v[0] + v[1], v[0] + 3 * v[1]], [1.0, 2.0], 10, 1e-20
        )
        self.assertAlmostEqual(solved[0], 1 / 11, places=10)
        self.assertAlmostEqual(solved[1], 7 / 11, places=10)

    def test_exact_fisher_vector_matches_kl_gradient_difference(self):

        # Keep setup, action and assertions together as one regression scenario.
        # pylint: disable=too-many-locals

        with precision_scope("float64"):
            b = create_brain(settings("trpo"), ["rest", "attack"], 3, 5)
        network = b.network.actor
        x = (0.2,) * 20
        logits, tape = network.forward(x)
        p, _logs = policy_distribution(logits)
        base = trpo.flatten(network.layers)
        direction = [((i % 7) - 3) * 0.02 for i in range(len(base))]
        actual = trpo.fisher_product(network, [tape], [p], direction, 0.0)

        def kl_gradient(scale):
            trpo.assign(
                network.layers, [a + scale * v for a, v in zip(base, direction)]
            )
            outputs, following = network.forward(x)
            q = policy_distribution(outputs)[0]
            gradient = network.zeros()
            network.accumulate_gradient(
                following, [a - v for a, v in zip(q, p)], gradient
            )
            return trpo.flatten(gradient)

        hi = kl_gradient(1e-5)
        lo = kl_gradient(-1e-5)
        for a, h, l in zip(actual, hi, lo):
            self.assertAlmostEqual(a, (h - l) / 2e-5, places=8)
        self.assertGreaterEqual(trpo.dot(direction, actual), 0.0)

    def test_lstm_two_step_bptt_matches_finite_differences_with_nonzero_memory(self):

        # Keep setup, action and assertions together as one regression scenario.
        # pylint: disable=too-many-locals

        c = settings("lstm_ppo")
        with precision_scope("float64"):
            network = LSTMNetwork(c, 2, 7)
        initial = [0.1, -0.2, 0.3, -0.1]
        first = (0.1,) * 20
        second = (-0.2,) * 20
        _, hidden, t1 = network.step(first, initial, tape=True)
        _outputs, _, t2 = network.step(second, hidden, tape=True)
        gradients = network.zeros()
        future = network.backward(t2, [1.0, 0.0, 0.0], [0.0] * 4, gradients)
        network.backward(t1, [0.0, 0.0, 0.0], future, gradients)

        def loss():
            _, state = network.step(first, initial)
            return network.step(second, state)[0][0]

        for layer_index, keys in (
            (0, ("input_weights", "recurrent_weights", "bias")),
            (1, ("weights", "bias")),
        ):
            for key in keys:
                values = network.layers[layer_index][key]
                for index in (0, len(values) // 2, len(values) - 1):
                    original = values[index]
                    values[index] = original + 1e-6
                    hi = loss()
                    values[index] = original - 1e-6
                    lo = loss()
                    values[index] = original
                    self.assertAlmostEqual(
                        gradients[layer_index][key][index], (hi - lo) / 2e-6, places=7
                    )

    def test_trpo_acceptance_obeys_kl_and_failed_search_restores_actor(self):
        for reject in (False, True):
            b = create_brain(settings("trpo"), ["rest", "attack"], 3, 5)
            for i in range(4):
                obs = (i / 10.0,) * 20
                outputs = b.network.predict(obs)
                logs = policy_distribution(outputs[:-1])[1]
                b.rollout.append(
                    RolloutStep(
                        obs,
                        i % 2,
                        float(i % 2),
                        logs[i % 2],
                        outputs[-1],
                        0.0,
                        False,
                        False,
                    )
                )
            base = trpo.flatten(b.network.actor.layers)
            objective = trpo.surrogate
            calls = 0

            def evaluate(*args, **kwargs):

                # The callback is consumed synchronously within this iteration; it
                # never escapes the loop.
                # pylint: disable=cell-var-from-loop

                nonlocal calls
                calls += 1
                v, g = objective(*args, **kwargs)
                return (v - 1e6 if reject and calls > 4 else v), g

            with patch.object(trpo, "surrogate", side_effect=evaluate):
                b._update()
            self.assertLessEqual(b.last_metrics["empirical_kl"], b.config["max_kl"])
            if reject:
                self.assertFalse(b.last_metrics["policy_accepted"])
                self.assertEqual(trpo.flatten(b.network.actor.layers), base)
            else:
                self.assertTrue(b.last_metrics["policy_accepted"])


class NewPolicyLifecycleTests(unittest.TestCase):
    def test_all_kinds_resume_pending_and_buffered_state_exactly(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                b = create_brain(settings(kind), ["rest", "attack"], 3, 5)
                for i in range(2):
                    obs = (i / 10.0,) * 20
                    action = b.choose(obs)
                    b.learn(Experience(obs, action, 0.3, obs, False, False))
                pending = (0.2,) * 20
                action = b.choose(pending)
                resumed = restore_brain(json.loads(json.dumps(b.checkpoint())))
                feedback = Experience(pending, action, 0.4, pending, False, False)
                b.learn(feedback)
                resumed.learn(feedback)
                for i in range(12):
                    obs = (i / 20.0,) * 20
                    action = b.choose(obs)
                    self.assertEqual(resumed.choose(obs), action)
                    feedback = Experience(
                        obs,
                        action,
                        (-1.0) ** i,
                        None if i == 5 else obs,
                        i == 5,
                        i == 9,
                    )
                    b.learn(feedback)
                    resumed.learn(feedback)
                    self.assertEqual(b.checkpoint(), resumed.checkpoint())
                self.assertGreater(b.updates, 0)
                saved = copy.deepcopy(b.checkpoint())
                b.diagnostics(obs, True)
                self.assertEqual(b.checkpoint(), saved)

    def test_warm_start_birth_and_sac_targets_are_independent(self):
        for kind in KINDS:
            b = create_brain(settings(kind), ["rest", "attack"], 3, 5)
            clone = create_brain(settings(kind), ["rest", "attack"], 9, 11)
            load_model_weights(clone, model_snapshot(b))
            self.assertEqual(clone.network.layers, b.network.layers)
            child = mix_parent_brains(b, clone, 13)
            self.assertEqual(child.updates, 0)
            if kind == "lstm_ppo":
                self.assertEqual(child.hidden, [0.0] * 4)
            if kind == "discrete_sac":
                for target, critic in zip(child.target_critics, child.network.critics):
                    self.assertEqual(target.layers, critic.layers)
            child.network.layers[0][
                "input_weights" if kind == "lstm_ppo" else "weights"
            ][0] += 0.1
            self.assertNotEqual(child.network.layers, b.network.layers)

    def test_world_checkpoints_continue_and_diagnostics_do_not_change_learning(self):
        for kind in KINDS:
            c = environment(kind)
            validate(c)
            a = Experiment(c)
            b = Experiment(c)
            for _ in range(6):
                a.step()
                b.step()
                for i, brain in b.brains.items():
                    brain.diagnostics(b.observations[i], True)
                self.assertEqual(a.checkpoint(), b.checkpoint())
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "saved.bin.xz"
                a.write_checkpoint(path)
                r = Experiment.restore(read_checkpoint(path))
                for _ in range(5):
                    a.step()
                    r.step()
                self.assertEqual(a.checkpoint(), r.checkpoint())

    def test_new_fields_reject_bad_values_and_legacy_presets_are_unchanged(self):
        for kind, key, bad in (
            ("discrete_sac", "temperature", 0),
            ("discrete_sac", "target_tau", 2),
            ("trpo", "max_kl", 0),
            ("trpo", "cg_iterations", 0),
            ("trpo", "backtrack_coefficient", 1),
            ("lstm_ppo", "sequence_length", 5),
        ):
            c = environment(kind)
            c["brains"]["new_policy"][key] = bad
            with self.assertRaisesRegex(ValueError, key):
                validate(c)

    def test_grpo_trials_do_not_advance_new_recurrent_policy_memory(self):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel

        from tests.test_policy_algorithms import settings as policy_settings
        from wtf.learning.grouped_rollouts import collect_group

        c = environment("lstm_ppo")
        c["brains"]["grouped"] = policy_settings("grpo")
        c["brains"]["grouped"].update(group_size=2, rollout_horizon=2)
        c["species"]["prey"]["brain"] = "grouped"
        e = Experiment(c)
        focal = next(i for i, a in e.world.animals.items() if a.species == "prey")
        before = e.checkpoint()
        trajectories = collect_group(e.world, e.brains, focal)
        self.assertEqual(len(trajectories), 2)
        # The focal rollout RNG deliberately advances; other policies/world do not.
        after = e.checkpoint()
        before["brains"][str(focal)]["group_rng_state"] = after["brains"][str(focal)][
            "group_rng_state"
        ]
        self.assertEqual(before, after)

    def test_fisher_direction_is_split_once_for_all_samples(self):
        brain = create_brain(settings("trpo"), ["rest", "attack"], 3, 5)
        actor = brain.network.actor
        rows = [actor.forward((i / 10.0,) * 20) for i in range(4)]
        tapes = [t for _, t in rows]
        probabilities = [policy_distribution(z)[0] for z, _ in rows]
        direction = [0.1] * len(trpo.flatten(actor.layers))
        with patch.object(trpo, "split_direction", wraps=trpo.split_direction) as split:
            trpo.fisher_product(actor, tapes, probabilities, direction, 0.1)
            self.assertEqual(split.call_count, 1)
