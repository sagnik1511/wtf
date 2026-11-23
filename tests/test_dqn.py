"""Scalar DQN math, target semantics and deep-network persistence/XAI."""

import copy
import json
import math
import unittest
from unittest.mock import patch

from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.inspection.explainability import prepare_model, explain_model
from wtf.learning.brain import Experience, INPUTS
from wtf.learning.brains import (
    create_brain,
    restore_brain,
    model_snapshot,
    mix_parent_brains,
    load_model_weights,
)
from wtf.learning.dqn import huber_td
from tests.test_advanced_algorithms import settings, environment

# Numerical fixtures inspect private controller arrays to verify target ownership.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access


class DQNTests(unittest.TestCase):
    def test_huber_derivative_and_saturation(self):
        for value in [-3.0, 0.2, 1.4, 4.0]:
            loss, derivative = huber_td(value, 1.0, 1.0)
            epsilon = 1e-6
            numeric = (
                huber_td(value + epsilon, 1.0, 1.0)[0]
                - huber_td(value - epsilon, 1.0, 1.0)[0]
            ) / (2 * epsilon)
            self.assertAlmostEqual(derivative, numeric, places=6)
            self.assertGreaterEqual(loss, 0)
            self.assertLessEqual(abs(derivative), 1)

    def test_target_network_max_death_and_time_limit(self):
        for terminal, truncated, expected in [
            (False, False, 4.6),
            (False, True, 4.6),
            (True, False, 1.0),
        ]:
            c = settings("dqn")
            c.update(
                batch_size=1,
                learning_starts_after=1,
                discount_factor=0.9,
                target_update_every_training_steps=50,
            )
            b = create_brain(c, ["rest", "eat"], 1, 2)
            o = (0.2,) * INPUTS
            with (
                patch.object(
                    b.network,
                    "predict",
                    side_effect=AssertionError(
                        "DQN must not select online bootstrap actions"
                    ),
                ),
                patch.object(b.target_network, "predict", return_value=[2.0, 4.0]),
                patch("wtf.learning.dqn.huber_td", wraps=huber_td) as loss,
            ):
                b.learn(
                    Experience(
                        o, "rest", 1.0, None if terminal else o, terminal, truncated
                    )
                )
            self.assertEqual(loss.call_count, 1)
            self.assertAlmostEqual(loss.call_args.args[1], expected)
            self.assertEqual(b.updates, 1)

    def test_three_hidden_layers_restore_birth_and_archived_explanation(self):
        for kind in ("dqn", "ddqn"):
            c = settings(kind)
            c["network"]["hidden_layers"] = [5, 4, 3]
            b = create_brain(c, ["rest", "eat"], 1, 2)
            o = (0.2,) * INPUTS
            for _ in range(3):
                b.learn(Experience(o, b.choose(o), 1, o, False))
            restored = restore_brain(json.loads(json.dumps(b.checkpoint())))
            for _ in range(3):
                a = b.choose(o)
                self.assertEqual(a, restored.choose(o))
                item = Experience(o, a, 0.3, o, False)
                b.learn(item)
                restored.learn(item)
            self.assertEqual(b.checkpoint(), restored.checkpoint())
            diag = b.diagnostics(o, include_activations=True)
            self.assertEqual([len(v) for v in diag["activations"]["layers"]], [5, 4, 3])
            self.assertNotIn("return_quantiles", diag)
            model = model_snapshot(b)
            explained = explain_model(prepare_model(model), list(o))
            self.assertEqual(explained["algorithm"], kind)
            other = create_brain(c, ["rest", "eat"], 3, 4)
            load_model_weights(other, model)
            self.assertEqual(b.values(o), other.values(o))
            child = mix_parent_brains(b, other, 99)
            self.assertEqual(child.replay, [])
            self.assertEqual(child.updates, 0)
            self.assertEqual(child.network.layers, child.target_network.layers)
            self.assertIsNot(
                child.network.layers[0]["weights"], b.network.layers[0]["weights"]
            )
            self.assertTrue(all(math.isfinite(v) for v in b.values(o)))

    def test_ddqn_selects_online_action_and_evaluates_only_target_head(self):
        c = settings("ddqn")
        b = create_brain(c, ["rest", "eat"], 1, 2)
        o = (0.2,) * INPUTS
        with (
            patch.object(b.network, "predict", return_value=[10.0, 0.0]),
            patch.object(
                b.target_network, "predict_range", return_value=[2.0]
            ) as target,
        ):
            self.assertEqual(b.bootstrap(o), 2.0)
            target.assert_called_once_with(o, 0, 1)
        self.assertEqual(b.diagnostics(o)["algorithm"], "ddqn")
        self.assertNotIn("return_quantiles", b.diagnostics(o))

    def test_depth_bounds_and_evaluation_no_learning(self):
        c = environment("dqn")
        c["brains"]["test"]["network"]["hidden_layers"] = [5, 4, 3]
        e = Experiment(validate(c), mode="evaluate")
        for _ in range(4):
            e.step()
        self.assertEqual(e.total_updates, 0)
        for bad in [[], [2] * 5, [0], ["wide"]]:
            invalid = copy.deepcopy(c)
            invalid["brains"]["test"]["network"]["hidden_layers"] = bad
            with self.assertRaises(ValueError):
                validate(invalid)
