"""Exact native/reference late PPO math, fallback hooks and malformed buffers."""

# Probes inspect private state to verify unchanged numerical and ownership contracts.
# pylint: disable=missing-function-docstring,protected-access
import copy
from array import array
from itertools import chain
from functools import wraps
import math
import random
import unittest
from unittest.mock import patch
from wtf.learning.brains import create_brain
from wtf.learning.ppo import RolloutStep, PPOBrain
from wtf.learning.late_consequence import LateConsequence
from wtf.native import kernels
from tests.test_observation_v7 import config


@unittest.skipUnless(
    kernels.available("late_ppo_batch"), "optional native late PPO kernel"
)
class LateBatchEfficiencyTests(unittest.TestCase):
    """Independent buffers, evidence snapshots and ordered reductions stay exact."""

    def brains(self, kind, scope="both", local=True):
        c = config(kind)
        t = c["brains"]["prey_ppo"]
        actions = c["species"]["prey"]["allowed_actions"]
        t.update(
            consequence_scope=scope,
            target_kl=0,
            epochs=2,
            rollout_steps=4,
            batch_size=2,
        )
        if not local:
            t.pop("social_observation")
        b = create_brain(t, actions, 101, 202)
        rng = random.Random(44)
        for i, _ in enumerate(b.late.network.layers[0]["weights"]):
            b.late.network.layers[0]["weights"][i] = rng.uniform(-0.3, 0.3)
        for i in range(4):
            obs = tuple(rng.random() for _ in b.late.layout.names)
            if kind == "maskable_ppo":
                b.mask = tuple(j % 3 != i % 3 or j == 0 for j in range(len(actions)))
                obs = b._pack(obs)
            out = b.network.predict(obs)
            _, logs = b.distribution(out, obs)
            b.rollout.append(
                RolloutStep(
                    obs, 0, i / 3, logs[0], out[-1], out[-1] + 0.1, False, i == 3
                )
            )
        return b, type(b).restore(copy.deepcopy(b.checkpoint()))

    def test_native_and_reference_complete_updates_match_all_evidence_scopes(self):
        available = kernels.available
        for kind in ("ppo", "maskable_ppo"):
            for scope in ("both", "context", "species_action"):
                for local in (False, True):
                    with self.subTest(kind=kind, scope=scope, local=local):
                        fast, reference = self.brains(kind, scope, local)
                        with patch.object(
                            kernels,
                            "available",
                            side_effect=lambda name: (
                                False if name == "late_ppo_batch" else available(name)
                            ),
                        ):
                            reference._update()
                        fast._update()
                        self.assertEqual(fast.checkpoint(), reference.checkpoint())

    # Keep independent reference/native buffers and malformed-call probes together.
    # The extension API is availability-gated and exercised at runtime here.
    # pylint: disable-next=too-many-locals,c-extension-no-member
    def test_kernel_actor_and_head_gradients_match_reference_exactly(self):
        # The optional extension is availability-gated and checked by these calls.
        # pylint: disable=c-extension-no-member
        b, _ = self.brains("maskable_ppo")
        actor = b.network.zeros()
        head = b.late.network.zeros()
        samples = []
        metrics = {}
        for step in b.rollout:
            outputs, tape = b.network.forward(step.observation)
            _, derivative, info = b.objective(outputs, step, 0.7, 0.2)
            b.network.accumulate_gradient(tape, derivative, actor)
            b.late.accumulate(step.observation, derivative, head)
            for key, value in info.items():
                metrics[key] = metrics.get(key, 0.0) + value
            samples.append(
                (
                    step.observation[: b.network.sizes[0]],
                    step.action,
                    step.log_probability,
                    0.7,
                    0.2,
                    tuple(step.observation[len(b.late.layout.names) :]),
                    array(
                        "d", chain.from_iterable(b.late.feature_rows(step.observation))
                    ),
                )
            )
        other_actor = b.network.zeros()
        other_head = b.late.network.zeros()
        result = kernels.native.late_ppo_batch(
            b.network.layers,
            other_actor,
            b.late.network.layers[0]["weights"],
            other_head[0]["weights"],
            samples,
            b.config["clip_ratio"],
            b.config["entropy_coefficient"],
            b.config["value_coefficient"],
            b.late.bound,
        )
        self.assertEqual(actor, other_actor)
        self.assertEqual(head, other_head)
        self.assertEqual(metrics, result)
        corrupted = copy.deepcopy(samples)
        row = list(corrupted[0])
        row[6] = array("d", [math.nan] * (b.late.features * len(b.actions)))
        corrupted[0] = tuple(row)
        actor = b.network.zeros()
        head = b.late.network.zeros()
        before = copy.deepcopy((actor, head))
        with self.assertRaises(ValueError):
            kernels.native.late_ppo_batch(
                b.network.layers,
                actor,
                b.late.network.layers[0]["weights"],
                head[0]["weights"],
                corrupted,
                0.2,
                0.03,
                0.5,
                2,
            )
        self.assertEqual((actor, head), before)

    def test_wrapped_class_objectives_bypass_native_loss(self):
        for kind in ("ppo", "maskable_ppo"):
            brain, _ = self.brains(kind)
            owner = type(brain)
            original = owner.objective

            @wraps(original)
            def wrapped(instance, *args, original_objective=original):
                return original_objective(instance, *args)

            with patch.object(owner, "objective", wrapped):
                with patch.object(
                    kernels.native,
                    "late_ppo_batch",
                    side_effect=AssertionError("wrapped objective fused"),
                ):
                    brain._update()
            self.assertGreater(brain.updates, 0)

    def test_class_optimizer_hooks_bypass_prepared_evidence(self):
        for owner in (PPOBrain, LateConsequence):
            with self.subTest(owner=owner.__name__):
                brain, _ = self.brains("ppo")
                original = owner._adam_step

                def custom_optimizer(
                    instance, gradients, count, original_step=original
                ):
                    original_step(instance, gradients, count)

                with patch.object(owner, "_adam_step", custom_optimizer):
                    with patch.object(
                        kernels.native,
                        "late_ppo_batch",
                        side_effect=AssertionError("optimizer hook fused"),
                    ):
                        brain._update()
                self.assertGreater(brain.updates, 0)

    def test_custom_instance_and_class_evidence_hooks_bypass_fusion(self):
        b, _ = self.brains("ppo")
        bound_original = b.late.feature_rows

        def bound_rows(obs):
            return bound_original(obs)

        b.late.feature_rows = bound_rows
        self.assertFalse(b.late.native_compatible())
        with patch.object(
            kernels.native,
            "late_ppo_batch",
            side_effect=AssertionError("custom hook fused"),
        ):
            b._update()
        b, _ = self.brains("ppo")
        class_original = LateConsequence.feature_rows

        def custom_rows(head, obs):
            return class_original(head, obs)

        with patch.object(LateConsequence, "feature_rows", custom_rows):
            self.assertFalse(b.late.native_compatible())
            with patch.object(
                kernels.native,
                "late_ppo_batch",
                side_effect=AssertionError("class hook fused"),
            ):
                b._update()


if __name__ == "__main__":
    unittest.main()
