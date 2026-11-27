"""Private supervised heads cannot import other animals' PPO trajectories."""

# Fixture names describe assertions; checkpoint tests intentionally inspect ownership.
# pylint: disable=missing-function-docstring,protected-access

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests.test_social_learning import config
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.learning.brains import (
    create_brain,
    model_snapshot,
    load_model_weights,
    mix_parent_brains,
)
from wtf.learning.ppo import RolloutStep
from wtf.native.numeric import precision_scope
from wtf.learning.cuda_training import CudaTraining
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.native import kernels


def auxiliary_config(kind="ppo", scope="both"):
    c = config(kind)
    used = {spec["brain"] for spec in c["species"].values()}
    c["brains"] = {name: brain for name, brain in c["brains"].items() if name in used}
    for brain in c["brains"].values():
        brain["consequence_scope"] = scope
        brain["auxiliary_consequence"] = {"coefficient": 0.1}
    c["consequence_learning"].update(
        persistent_action_totals=True, update_schedule="episode"
    )
    return validate(c)


def controller(kind="ppo", scope="both", seed=9):
    c = auxiliary_config(kind, scope)
    spec = c["species"]["prey"]
    return create_brain(c["brains"][spec["brain"]], spec["allowed_actions"], 7, seed)


def observation(brain, confidence=1.0):
    values = [0.1] * brain.network.sizes[0]
    width = 96 if brain.config["consequence_scope"] == "both" else 48
    values[-width:] = [confidence, 1, 0, 0] * (width // 4)
    return tuple(values)


class AuxiliaryConsequenceTests(unittest.TestCase):
    """Check gradients, isolation, unknown evidence and lifecycle boundaries."""

    def test_no_direct_target_copy_and_context_fallback(self):
        for scope in ("both", "species_action", "context"):
            brain = controller(scope=scope)
            values = observation(brain)
            inputs, targets = brain.auxiliary.targets(values, len(values))
            width = 96 if scope == "both" else 48
            self.assertEqual(inputs[-width:], (0.0,) * width)
            self.assertEqual(targets[0], (1, 1, 0, 0))
        brain = controller()
        values = list(observation(brain))
        values[-48:] = [0, 0, 0, 0] * 12
        self.assertEqual(
            brain.auxiliary.targets(values, len(values))[1][0], (1, 1, 0, 0)
        )

    def test_joint_gradient_matches_finite_difference_and_leaves_actor_head_zero(self):
        with precision_scope("float64"):
            brain = controller()
        values = observation(brain)
        gradients = brain.network.zeros()
        metrics, head_gradient = brain.auxiliary.accumulate(
            brain.network, [values], gradients
        )
        self.assertGreater(metrics["auxiliary_loss"], 0)
        self.assertTrue(
            all(v == 0 for values in gradients[-1].values() for v in values)
        )
        parameter = brain.network.layers[0]["weights"]
        index = 0
        original = parameter[index]
        epsilon = 1e-5
        losses = []
        for sign in (-1, 1):
            parameter[index] = original + sign * epsilon
            result, _ = brain.auxiliary.accumulate(
                brain.network, [values], brain.network.zeros()
            )
            losses.append(result["auxiliary_loss"])
        parameter[index] = original
        self.assertAlmostEqual(
            gradients[0]["weights"][index],
            (losses[1] - losses[0]) / (2 * epsilon),
            places=8,
        )
        self.assertTrue(any(v != 0 for v in head_gradient[0]["weights"]))

    def test_unknown_evidence_has_no_gradient_or_head_update(self):
        brain = controller()
        values = observation(brain, confidence=0)
        gradients = brain.network.zeros()
        metrics, proposal = brain.auxiliary.accumulate(
            brain.network, [values], gradients
        )
        self.assertEqual(metrics, {"auxiliary_loss": 0, "auxiliary_coverage": 0})
        self.assertTrue(
            all(
                v == 0
                for layer in gradients + proposal
                for values in layer.values()
                for v in values
            )
        )

    def test_auxiliary_does_not_change_other_brain_or_policy_rng(self):
        left, right = controller(), controller(seed=10)
        before = right.checkpoint()
        rng = left.rng.getstate(), left.training_rng.getstate()
        values = observation(left)
        _, proposal = left.auxiliary.accumulate(
            left.network, [values], left.network.zeros()
        )
        left.auxiliary.apply(proposal, 1)
        self.assertEqual(right.checkpoint(), before)
        self.assertEqual((left.rng.getstate(), left.training_rng.getstate()), rng)
        self.assertFalse(CudaTraining.accepts_ppo(left))

    def test_kl_rejection_skips_auxiliary_and_rollout_is_cleared(self):
        brain = controller()
        brain.config["target_kl"] = 0.03
        values = observation(brain)
        brain.rollout = [RolloutStep(values, 0, 0, 0, 0, 0, False, True)]
        before = brain.auxiliary.checkpoint()
        with patch.object(
            brain,
            "objective",
            return_value=(0, [0] * (len(brain.actions) + 1), {"approximate_kl": 1}),
        ):
            brain._update()
        self.assertEqual(brain.auxiliary.checkpoint(), before)
        self.assertEqual(brain.rollout, [])

    def test_model_checkpoint_birth_and_masked_contracts(self):
        for kind in ("ppo", "maskable_ppo"):
            left, right = controller(kind), controller(kind, seed=10)
            self.assertEqual(left.auxiliary.updates, 0)
            _, proposal = left.auxiliary.accumulate(
                left.network, [observation(left)], left.network.zeros()
            )
            left.auxiliary.apply(proposal, 1)
            self.assertGreater(left.auxiliary.updates, 0)
            model = model_snapshot(left)
            load_model_weights(right, model)
            self.assertEqual(model_snapshot(right), model)
            child = mix_parent_brains(left, right, 20)
            self.assertEqual(child.auxiliary.updates, 0)
            self.assertIsNot(
                child.auxiliary.network.layers[0]["weights"],
                left.auxiliary.network.layers[0]["weights"],
            )
            state = json.loads(json.dumps(left.checkpoint()))
            restored = type(left).restore(state)
            self.assertEqual(restored.checkpoint(), left.checkpoint())

    def test_exact_experiment_resume_with_pending_rollouts_and_staged_totals(self):
        c = auxiliary_config()
        c["world"]["max_steps"] = 12
        e = Experiment(c)
        for _ in range(3):
            e.step()
        restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
        for _ in range(12):
            if e.done:
                break
            e.step()
            restored.step()
            self.assertEqual(restored.snapshot(), e.snapshot())
        self.assertEqual(restored.checkpoint(), e.checkpoint())

    def test_native_and_reference_auxiliary_gradients_agree(self):
        brain = controller()
        values = observation(brain)
        native_gradients = brain.network.zeros()
        native_metrics, native_head = brain.auxiliary.accumulate(
            brain.network, [values], native_gradients
        )
        with patch.object(kernels, "native", None):
            reference_gradients = brain.network.zeros()
            reference_metrics, reference_head = brain.auxiliary.accumulate(
                brain.network, [values], reference_gradients
            )
        self.assertAlmostEqual(
            native_metrics["auxiliary_loss"],
            reference_metrics["auxiliary_loss"],
            places=13,
        )
        for left, right in zip(
            native_gradients + native_head, reference_gradients + reference_head
        ):
            for key in left:
                for a, b in zip(left[key], right[key]):
                    self.assertAlmostEqual(a, b, places=13)

    def test_binary_checkpoint_preserves_auxiliary_optimizer_and_tables(self):
        c = auxiliary_config()
        c["world"]["max_steps"] = 12
        for brain in c["brains"].values():
            brain["rollout_steps"] = 2
        e = Experiment(c)
        while e.run_index == 0 or not any(
            b.auxiliary.updates for b in e.brains.values()
        ):
            e.step()
        e.step()  # Retain a partial on-policy rollout after a real auxiliary update.
        self.assertTrue(any(b.auxiliary.updates for b in e.brains.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.bin.xz"
            e.write_checkpoint(path, shared_metadata=True)
            restored = Experiment.restore(read_checkpoint(path))
            self.assertEqual(restored.checkpoint(), e.checkpoint())
            while not e.done:
                e.step()
                restored.step()
            self.assertEqual(restored.checkpoint(), e.checkpoint())

    def test_failed_compact_weight_load_is_atomic_for_policy_and_head(self):
        brain = controller()
        before = brain.checkpoint()
        for field in ("parameters", "auxiliary_parameters"):
            model = copy.deepcopy(model_snapshot(brain))
            model["auxiliary_parameters"][0]["weights"][0] = 0.25
            model[field][0]["weights"][0] = 1e100
            with self.assertRaises(ValueError):
                load_model_weights(brain, model)
            self.assertEqual(brain.checkpoint(), before)

    def test_configuration_rejects_unsupported_algorithms_and_disabled_evidence(self):
        for kind in ("dqn", "a2c", "recurrent_ppo"):
            with self.assertRaises(ValueError):
                auxiliary_config(kind)
        c = auxiliary_config()
        for brain in c["brains"].values():
            brain["consequence_observation"] = False
        with self.assertRaises(ValueError):
            validate(c)
