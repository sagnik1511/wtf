"""Exact module-boundary staging, native auxiliary batches and hook fallbacks."""

# Regression probes inspect private ownership and optimizer state deliberately.
# pylint: disable=missing-function-docstring,protected-access
import copy
import random
import struct
import unittest
from unittest.mock import patch

from tests import test_late_batch_efficiency as late_tests
from tests.test_observation_v7 import config
from wtf.experiment.runner import Experiment
from wtf.learning import auxiliary_consequence as auxiliary
from wtf.learning.ppo import PPOBrain
from wtf.native import kernels
from wtf.simulation.actions import ACTIONS
from wtf.simulation.compact_observations import compact_layout, encode_compact
from wtf.simulation.social import SocialState


def bits(values):
    """Retain signed zero in exact observation/gradient comparisons."""
    return struct.pack(f"<{len(values)}d", *values)


def counter_provider(history, name):
    """Own each custom provider's call history without shared default state."""

    def provider(*args):
        history.append(args)
        row = (len(history) / 100, 0.0, 0.0, 0.0)
        return (row,) * len(ACTIONS) if name == "action_rates" else row

    return provider


class FlowEfficiencyTests(unittest.TestCase):
    """Prepared evidence cannot change private learning or recorded diagnostics."""

    def test_direct_compact_matches_full_v5_projection(self):
        for scope in ("both", "context", "species_action"):
            c = config()
            for template in c["brains"].values():
                template["consequence_scope"] = scope
            experiment = Experiment(c)
            world = experiment.world
            for animal in world.animals.values():
                template = world.config["brains"][
                    world.config["species"][animal.species]["brain"]
                ]
                for heading in range(4):
                    animal.heading = heading
                    template["observation_version"] = 5
                    raw = world.observe(animal)
                    template["observation_version"] = 7
                    scent = world._tracking.scent.sample_compact(
                        world, animal, template
                    )
                    expected = encode_compact(raw, template, scent)
                    actual = world.observe(animal)
                    self.assertEqual(bits(actual), bits(expected))
                    self.assertEqual(
                        [type(v) for v in actual], [type(v) for v in expected]
                    )

    def test_custom_social_provider_keeps_three_argument_contract(self):
        experiment = Experiment(config())
        world = experiment.world
        original = SocialState.observe

        def provider(state, current_world, animal, template):
            return original(state, current_world, animal, template)

        expected = world.observe_all()
        with patch.object(SocialState, "observe", provider):
            self.assertEqual(world.observe_all(), expected)

    def test_diagnostics_reuses_residual_and_preserves_custom_distribution(self):
        fixture = late_tests.LateBatchEfficiencyTests()
        for kind in ("ppo", "maskable_ppo"):
            brain, _ = fixture.brains(kind)
            observation = brain.rollout[0].observation[
                : compact_layout(brain.config).width
            ]
            # Installing a custom distribution forces the original reference path.
            original = brain.distribution
            brain.distribution = lambda *args, method=original: method(*args)
            expected = brain.diagnostics(observation, include_activations=True)
            del brain.distribution
            with patch.object(brain.late, "native_compatible", return_value=True):
                with patch.object(
                    brain.late, "feature_rows", wraps=brain.late.feature_rows
                ) as calls:
                    actual = brain.diagnostics(observation, include_activations=True)
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(actual, expected)

    # Keep both reference projections and provider-call histories in one probe.
    # pylint: disable-next=too-many-locals
    def test_stateful_consequence_providers_keep_canonical_call_order(self):
        experiment = Experiment(config())
        world = experiment.world
        state = world._social
        animal = next(iter(world.animals.values()))
        template = world.config["brains"][
            world.config["species"][animal.species]["brain"]
        ]
        actions = compact_layout(template).actions
        for name in ("predict", "predict_action", "action_rates"):
            calls = []

            with patch.object(state.knowledge, name, counter_provider(calls, name)):
                raw = state.observe(world, animal, template)
                expected = raw[:24]
                for start in range(24, len(raw), 48):
                    for action in actions:
                        offset = start + 4 * ACTIONS.index(action)
                        expected.extend(raw[offset : offset + 4])
                expected_calls = calls[:]
                calls.clear()
                actual = state.observe_compact(world, animal, template, actions)
                self.assertEqual(actual, expected)
                self.assertEqual(calls, expected_calls)

    def test_auxiliary_preparation_once_per_rollout_and_exact_update(self):
        fixture = late_tests.LateBatchEfficiencyTests()
        for kind in ("ppo", "maskable_ppo"):
            for scope in ("both", "context", "species_action"):
                with self.subTest(kind=kind, scope=scope):
                    fast, reference = fixture.brains(kind, scope)
                    fast.config["epochs"] = reference.config["epochs"] = 3
                    # Instance hooks must disable both staging and native fusion.
                    original = reference.auxiliary.targets
                    reference.auxiliary.targets = lambda *args, method=original: method(
                        *args
                    )
                    reference._update()
                    with patch.object(
                        auxiliary, "consequence_rows", wraps=auxiliary.consequence_rows
                    ) as calls:
                        fast._update()
                    self.assertEqual(calls.call_count, 4)
                    self.assertEqual(fast.checkpoint(), reference.checkpoint())
                    self.assertFalse(hasattr(fast.auxiliary, "prepared"))

    @unittest.skipUnless(
        kernels.available("auxiliary_batch"), "optional native auxiliary kernel"
    )
    # Keep the exact-gradient reference and private buffers together in this probe.
    # pylint: disable-next=too-many-locals
    def test_native_auxiliary_gradients_are_exact_with_nonzero_accumulators(self):
        fixture = late_tests.LateBatchEfficiencyTests()
        brain, _ = fixture.brains("ppo")
        observations = [step.observation for step in brain.rollout]
        rng = random.Random(77)
        gradients = brain.network.zeros()
        for layer in gradients:
            for values in layer.values():
                for index, _ in enumerate(values):
                    values[index] = rng.uniform(-0.1, 0.1)
        reference = copy.deepcopy(gradients)
        available = kernels.available
        with patch.object(
            kernels,
            "available",
            side_effect=lambda name: name != "auxiliary_batch" and available(name),
        ):
            expected, expected_head = brain.auxiliary.accumulate(
                brain.network, observations, reference
            )
        actual, actual_head = brain.auxiliary.accumulate(
            brain.network, observations, gradients
        )
        self.assertEqual(actual, expected)
        for left, right in zip(gradients + actual_head, reference + expected_head):
            for key in left:
                self.assertEqual(bits(left[key]), bits(right[key]))

    @unittest.skipUnless(
        kernels.available("auxiliary_batch"), "optional native auxiliary kernel"
    )
    def test_malformed_later_native_sample_is_atomic_and_buffers_release(self):
        # The optional API is gated above; malformed input must never write gradients.
        # pylint: disable=c-extension-no-member
        fixture = late_tests.LateBatchEfficiencyTests()
        brain, _ = fixture.brains("ppo")
        view = brain.network.layers[:-1] + brain.auxiliary.network.layers
        gradients = brain.network.zeros()[:-1] + brain.auxiliary.network.zeros()
        before = copy.deepcopy(gradients)
        samples = brain.auxiliary.prepare(
            [step.observation for step in brain.rollout], brain.network.sizes[0]
        )
        for invalid in (
            ((), ()),
            ("bad", samples[0][1]),
            (samples[0][0], ((1, 0, 0, 0),)),
        ):
            with self.assertRaises((TypeError, ValueError)):
                kernels.native.auxiliary_batch(
                    view, gradients, [samples[0], invalid], 0.1
                )
            self.assertEqual(gradients, before)
        # A successful retry proves temporary shape validation left no corruption.
        kernels.native.auxiliary_batch(view, gradients, samples, 0.1)
        self.assertNotEqual(gradients, before)
        # Exported buffers must be released, including the failure path.
        brain.auxiliary.network.layers[0]["weights"].append(0)

    def test_custom_actor_optimizer_disables_auxiliary_epoch_staging(self):
        brain, _ = late_tests.LateBatchEfficiencyTests().brains("ppo")
        original = PPOBrain._adam_step

        def optimizer(instance, gradients, count):
            return original(instance, gradients, count)

        with patch.object(PPOBrain, "_adam_step", optimizer):
            with patch.object(
                auxiliary, "consequence_rows", wraps=auxiliary.consequence_rows
            ) as calls:
                brain._update()
        self.assertEqual(calls.call_count, 8)
