"""Species rate isolation, real optimizer steps, and lifecycle persistence."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
import unittest

from wtf.learning.brain import Experience
from wtf.learning.brains import (
    create_brain,
    inherit_brain,
    model_snapshot,
    load_model_weights,
)
from wtf.config import validate
from wtf.experiment.runner import Experiment
from tests.test_simulation import config
from tests.test_ppo import ppo_environment
from tests.test_policy_algorithms import environment


def scenario(kind):
    c = (
        config()
        if kind == "q_network"
        else ppo_environment() if kind == "ppo" else environment(kind)
    )
    c["species"]["prey"]["learning_rate_multiplier"] = 1.0
    c["species"]["predator"]["learning_rate_multiplier"] = 1.25
    return c


class SpeciesLearningTests(unittest.TestCase):
    def test_effective_rates_all_algorithms_without_mutating_templates(self):
        for kind in ("q_network", "ppo", "a2c", "reinforce", "grpo"):
            with self.subTest(kind=kind):
                c = scenario(kind)
                original = copy.deepcopy(c)
                e = Experiment(c)
                for i, animal in e.world.animals.items():
                    spec = c["species"][animal.species]
                    template = c["brains"][spec["brain"]]
                    self.assertEqual(
                        e.brains[i].config["learning_rate"],
                        template["learning_rate"] * spec["learning_rate_multiplier"],
                    )
                    child = inherit_brain(e.brains[i], 102)
                    self.assertEqual(
                        child.config["learning_rate"],
                        e.brains[i].config["learning_rate"],
                    )
                    child.config["learning_rate"] *= 2
                    self.assertNotEqual(
                        child.config["learning_rate"],
                        e.brains[i].config["learning_rate"],
                    )
                self.assertEqual(c, original)
                self.assertEqual(e.config, original)

    def test_missing_multiplier_preserves_legacy_behavior(self):
        c = scenario("a2c")
        for spec in c["species"].values():
            spec.pop("learning_rate_multiplier")
        e = Experiment(c)
        for i, animal in e.world.animals.items():
            template = c["brains"][c["species"][animal.species]["brain"]]
            self.assertEqual(e.brains[i].config, template)

    def test_invalid_multipliers_and_effective_rates_rejected(self):
        for value in (True, 0, -1, 11, float("inf"), float("nan"), "1.25"):
            c = scenario("ppo")
            c["species"]["predator"]["learning_rate_multiplier"] = value
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "learning_rate_multiplier"),
            ):
                validate(c)
        for kind, rate, multiplier in [
            ("ppo", 0.1, 1.25),
            ("q_network", 1, 1.25),
            ("a2c", 0.000001, 0.5),
        ]:
            c = scenario(kind)
            template = c["species"]["predator"]["brain"]
            c["brains"][template]["learning_rate"] = rate
            c["species"]["predator"]["learning_rate_multiplier"] = multiplier
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "effective_learning_rate"),
            ):
                validate(c)

    def test_identical_feedback_produces_scaled_q_update(self):
        e = Experiment(scenario("q_network"))
        base = next(iter(e.brains.values())).config
        c = dict(base, batch_size=1, learning_starts_after=1, train_every_steps=1)
        a = create_brain(c, ["rest"], 42, 24)
        b = create_brain(
            dict(c, learning_rate=c["learning_rate"] * 1.25), ["rest"], 42, 24
        )
        original = copy.deepcopy(a.weights)
        feedback = Experience((0.1,) * 20, "rest", 2.0, None, True)
        a.learn(feedback)
        b.learn(feedback)
        self.assertEqual(a.updates, b.updates)
        self.assertGreater(a.updates, 0)
        for key in original:
            for before, low, high in zip(original[key], a.weights[key], b.weights[key]):
                self.assertAlmostEqual(
                    before - high,
                    1.25 * (before - low),
                    delta=3e-7 * max(abs(before), abs(low), abs(high), 1e-6),
                )

    def test_policy_optimizer_uses_scaled_rate_without_more_updates(self):
        for kind in ("ppo", "a2c", "reinforce", "grpo"):
            with self.subTest(kind=kind):
                e = Experiment(scenario(kind))
                c = next(iter(e.brains.values())).config
                a = create_brain(c, ["rest", "turn_left"], 42, 24)
                b = create_brain(
                    dict(c, learning_rate=c["learning_rate"] * 1.25),
                    ["rest", "turn_left"],
                    42,
                    24,
                )
                original = copy.deepcopy(a.network.layers)
                gradients = [
                    {k: [0.2] * len(v) for k, v in layer.items()} for layer in original
                ]
                a._adam_step(gradients, 1)
                b._adam_step(gradients, 1)
                self.assertEqual(a.updates, b.updates)
                for before, low, high in zip(
                    original, a.network.layers, b.network.layers
                ):
                    for key in before:
                        for x, y, z in zip(before[key], low[key], high[key]):
                            self.assertAlmostEqual(
                                x - z,
                                1.25 * (x - y),
                                delta=3e-7 * max(abs(x), abs(y), abs(z), 1e-6),
                            )

    def test_exact_resume_preserves_effective_rates_and_future_learning(self):
        for kind in ("q_network", "ppo", "a2c", "reinforce", "grpo"):
            with self.subTest(kind=kind):
                a = Experiment(scenario(kind))
                a.step()
                b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
                a.step()
                b.step()
                self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_chaining_and_reset_apply_multiplier_once_on_each_run(self):
        for chain in (False, True):
            c = scenario("a2c")
            c["experiment"]["chain_best_brains"] = chain
            e = Experiment(c)
            rates = {
                animal.species: e.brains[i].config["learning_rate"]
                for i, animal in e.world.animals.items()
            }
            while not e.finished_run:
                e.step()
            e.next_run()
            for i, animal in e.world.animals.items():
                self.assertEqual(
                    e.brains[i].config["learning_rate"], rates[animal.species]
                )

    def test_loading_weights_keeps_current_species_rate(self):
        e = Experiment(scenario("ppo"))
        brain = next(iter(e.brains.values()))
        model = model_snapshot(brain)
        fresh = create_brain(
            dict(brain.config, learning_rate=0.002), brain.actions, 42, 24
        )
        load_model_weights(fresh, model)
        self.assertEqual(fresh.config["learning_rate"], 0.002)
        self.assertEqual(fresh.updates, 0)
