"""Soft semantic observations preserve older contracts and independent PPO heads."""

# Fixtures inspect private state to assert exact continuation and target isolation.
# pylint: disable=missing-function-docstring,protected-access
import copy
import json
import math
import unittest
from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from wtf.learning.brains import create_brain, model_snapshot, load_model_weights
from wtf.inspection.explainability import feature_names, prepare_model, explain_model
from wtf.simulation.soft_observations import (
    memberships,
    direction,
    bearing,
    soft_layout,
    encode_soft,
    decode_soft,
)
from tests.test_mutation import KINDS, environment
from tests.test_auxiliary_consequence import auxiliary_config
from work.compare_observation_versions import version_config


class ObservationV6Tests(unittest.TestCase):
    """Membership continuity, semantic layout, physics and saved-state contracts."""

    def test_soft_hats_and_wrapped_bearings_are_continuous(self):
        for value in (0, 0.1, 0.3, 0.5, 0.999, 1):
            weights = memberships(value, (0, 0.3, 1))
            self.assertAlmostEqual(sum(weights), 1)
            self.assertAlmostEqual(
                sum(w * k for w, k in zip(weights, (0, 0.3, 1))), value
            )
        for i in range(-100, 100):
            angle = i * math.pi / 99
            weights = direction(math.cos(angle), math.sin(angle))
            decoded = bearing(weights)
            self.assertAlmostEqual(math.sin(angle), math.sin(decoded), places=12)
            self.assertAlmostEqual(math.cos(angle), math.cos(decoded), places=12)
        left = memberships(0.3 - 1e-8, (0, 0.3, 1))
        right = memberships(0.3 + 1e-8, (0, 0.3, 1))
        self.assertLess(sum(abs(a - b) for a, b in zip(left, right)), 1e-6)

    def test_all_learners_train_restore_and_reject_older_weights(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = environment(kind, 0)
                for b in c["brains"].values():
                    b.update(observation_version=6, observation_top_k=2)
                e = Experiment(c)
                e.step()
                restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
                for a in e.world.animals.values():
                    b = e.brains[a.id]
                    model = model_snapshot(b)
                    names = feature_names(model["signature"])
                    self.assertEqual(len(names), len(e.observations[a.id]))
                    explain_model(prepare_model(model), e.observations[a.id])
                for _ in range(2):
                    e.step()
                    restored.step()
                self.assertEqual(e.checkpoint(), restored.checkpoint())

    def test_v5_and_v6_observation_calls_preserve_fixed_action_physics_and_rng(self):
        for name in ("seaside_150_v5", "river_crossing_strong_predator", "day_night"):
            c5 = load("environments/" + name + ".json")
            c5["seed"] = 100
            c5["world"]["max_steps"] = 5
            c5["experiment"].update(runs=1, world_seed_policy="fixed")
            c6 = copy.deepcopy(c5)
            for b in c5["brains"].values():
                b.pop("observation_actions", None)
                b["observation_version"] = 5
            for b in c6["brains"].values():
                b.pop("observation_actions", None)
                b["observation_version"] = 6
            left, right = Experiment(c5), Experiment(c6)
            for _ in range(3):
                a = left.world.step({i: "rest" for i in left.world.animals})
                b = right.world.step({i: "rest" for i in right.world.animals})
                self.assertEqual(left.world.snapshot(), right.world.snapshot())
                self.assertEqual(a.events, b.events)
                self.assertEqual(a.rewards, b.rewards)
                self.assertEqual(left.world.rng.getstate(), right.world.rng.getstate())

    def test_auxiliary_targets_are_decoded_and_every_target_channel_is_zeroed(self):
        # Repeat exact continuation here with expanded target-bearing inputs.
        # pylint: disable=duplicate-code
        c = auxiliary_config()
        for b in c["brains"].values():
            b.update(observation_version=6, rollout_steps=2)
        e = Experiment(c)
        b = e.brains[0]
        layout = soft_layout(b.config)
        raw = [0.0] * len(layout.raw_names)
        raw[-96:] = [0.7, 0.002, 0.3, 0.0004] * 24
        encoded = encode_soft(raw, b.config)
        physical, targets = b.auxiliary.targets(encoded, len(encoded))
        for got, expected in zip(targets[0], (0.7, 0.002, 0.3, 0.0004)):
            self.assertAlmostEqual(got, expected, places=14)
        start = layout.offset(len(raw) - 96)
        self.assertTrue(all(v == 0 for v in physical[start:]))
        self.assertEqual(physical[:start], tuple(encoded[:start]))
        for _ in range(5):
            e.step()
        restored = Experiment.restore(e.checkpoint())
        e.step()
        restored.step()
        self.assertEqual(e.checkpoint(), restored.checkpoint())

    def test_configuration_signatures_widths_and_rare_outcomes(self):
        c = load("environments/seaside_150_v6.json")
        e = Experiment(c)
        self.assertEqual(e.brains[0].algorithm, "ppo")
        self.assertFalse(hasattr(e.brains[0], "mask"))
        older = copy.deepcopy(c)
        for b in older["brains"].values():
            b["observation_version"] = 5
        prior = Experiment(older)
        with self.assertRaises(ValueError):
            load_model_weights(e.brains[0], model_snapshot(prior.brains[0]))
        explicit = copy.deepcopy(c)
        explicit["brains"]["prey_ppo"]["soft_categories"] = {"distance": [0, 0.2, 1]}
        changed = Experiment(explicit)
        with self.assertRaises(ValueError):
            load_model_weights(e.brains[0], model_snapshot(changed.brains[0]))
        for bad in ([0, 0.3, 0.2], [0, True, 1], [0, 0.3, float("nan")], [0, 0.3]):
            invalid = copy.deepcopy(c)
            invalid["brains"]["prey_ppo"]["soft_categories"] = {"distance": bad}
            with self.assertRaises(ValueError):
                validate(invalid)
        self.assertNotEqual(
            memberships(0.0001, (0, 0.001, 0.01, 0.1, 1)),
            memberships(0.001, (0, 0.001, 0.01, 0.1, 1)),
        )
        for b in older["brains"].values():
            b["soft_categories"] = {}
        with self.assertRaises(ValueError):
            validate(older)

    def test_private_auxiliary_layout_does_not_alias_caller_thresholds(self):
        c = auxiliary_config()
        template = c["brains"][c["species"]["prey"]["brain"]]
        template.update(
            observation_version=6, soft_categories={"outcome": [0, 0.001, 0.01, 0.1, 1]}
        )
        brain = create_brain(template, c["species"]["prey"]["allowed_actions"], 1, 2)
        layout = soft_layout(template)
        raw = [0.0] * len(layout.raw_names)
        raw[-96:] = [0.7, 0.01, 0.3, 0.001] * 24
        values = encode_soft(raw, template)
        before = brain.auxiliary.targets(values, len(values))
        template["soft_categories"]["outcome"][2] = 0.02
        self.assertEqual(brain.auxiliary.targets(values, len(values)), before)
        self.assertEqual(brain.config["soft_categories"]["outcome"][2], 0.01)

    def test_comparison_treatments_accept_custom_v6_knots_without_mutating_source(self):
        c = load("environments/seaside_150_v6.json")
        c["brains"]["prey_ppo"]["soft_categories"] = {"distance": [0, 0.2, 1]}
        original = copy.deepcopy(c)
        old, new = version_config(c, 5), version_config(c, 6)
        self.assertNotIn("soft_categories", validate(old)["brains"]["prey_ppo"])
        self.assertEqual(
            validate(new)["brains"]["prey_ppo"]["soft_categories"]["distance"],
            [0, 0.2, 1],
        )
        self.assertEqual(c, original)

    def test_optional_memory_scent_and_social_blocks_match_canonical_values(self):
        c = load("environments/seaside_150_v6.json")
        c["scent_trails"] = {"enabled": True, "lifetime_steps": 20}
        for b in c["brains"].values():
            b.update(
                movement_history_steps=2,
                scent_observation={"mode": "blended", "radius": 2},
            )
        e = Experiment(c)
        for animal in e.world.animals.values():
            template = e.brains[animal.id].config
            layout = soft_layout(template)
            decoded = decode_soft(e.observations[animal.id], template)
            self.assertEqual(len(decoded), len(layout.raw_names))
            self.assertEqual(len(e.observations[animal.id]), layout.width)
            self.assertTrue(all(math.isfinite(v) for v in decoded))
            self.assertTrue(any("Scent blended front" in n for n in layout.names))
