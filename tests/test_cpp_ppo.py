"""Exact native ordinary-PPO continuation against the Python implementation."""

# Regression fixtures inspect private state to verify exact continuation.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring

# The optional compiled API is exercised by the native differential tests.
# pylint: disable=c-extension-no-member


import copy
import math
import unittest
from unittest.mock import patch

from tests.native_core_support import _engine
from wtf.learning.brain import Experience
from wtf.learning.ppo import PPOBrain
from wtf.native import kernels


def settings(hidden, target_kl=0):
    return {
        "type": "ppo",
        "observation_version": 1,
        "network": {"hidden_layers": hidden},
        "rollout_steps": 7,
        "batch_size": 3,
        "epochs": 2,
        "discount_factor": 0.97,
        "gae_lambda": 0.93,
        "normalize_advantages": True,
        "clip_ratio": 0.2,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
        "target_kl": target_kl,
        "learning_rate": 0.003,
        "max_grad_norm": 0.5,
    }


def observation(step):
    return tuple(math.sin(step * 0.17 + i * 0.13) * 0.8 for i in range(20))


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(item) for item in value]
    return value


@unittest.skipUnless(_engine is not None, "Optional native engine is unavailable")
class NativePPOTests(unittest.TestCase):
    def assert_tree_equal(self, expected, actual, path="checkpoint"):
        expected, actual = normalized(expected), normalized(actual)
        if isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys(), path)
            for key in expected:
                self.assert_tree_equal(expected[key], actual[key], f"{path}.{key}")
        elif isinstance(expected, list):
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_tree_equal(left, right, f"{path}[{index}]")
        else:
            self.assertEqual(expected, actual, path)

    def assert_state(self, brain, handle, obs):
        self.assert_tree_equal(brain.checkpoint(), _engine.ppo_checkpoint(handle))
        self.assert_tree_equal(
            brain.diagnostics(obs),
            _engine.ppo_diagnostics(handle, obs, False),
            "diagnostics",
        )
        self.assert_tree_equal(
            brain.diagnostics(obs, include_activations=True),
            _engine.ppo_diagnostics(handle, obs, True),
            "activation_diagnostics",
        )

    def exercise(self, hidden, target_kl):
        brain = PPOBrain(settings(hidden, target_kl), ["rest", "eat", "move"], 17, 901)
        handle = _engine.ppo_create(brain.checkpoint())
        self.assert_state(brain, handle, observation(0))
        # Seven samples produce three non-divisible minibatches. Death and time
        # limits flush partial buffers independently; later samples resume fresh.
        for step in range(26):
            obs, nxt = observation(step), observation(step + 1)
            self.assertEqual(
                brain.choose(obs, explore=False), _engine.ppo_choose(handle, obs, False)
            )
            self.assert_state(brain, handle, obs)
            action = brain.choose(obs)
            self.assertEqual(action, _engine.ppo_choose(handle, obs, True))
            self.assert_state(brain, handle, obs)
            if step in (5, 12):
                # Pending logp/value and sampled RNG continuation survive restore.
                brain = PPOBrain.restore(brain.checkpoint())
                handle = _engine.ppo_create(_engine.ppo_checkpoint(handle))
                self.assert_state(brain, handle, obs)
            if step % 6 == 2:
                # An external action must recompute the old-policy logp/value.
                action = brain.actions[(brain.action_indices[action] + 1) % 3]
            if step % 8 == 3:
                # A changed observation also invalidates the pending sample.
                obs = tuple(x + 0.001 for x in obs)
            terminated, truncated = step in (9, 25), step in (14,)
            next_obs = None if terminated else nxt
            reward = math.cos(step * 0.21) * 2.0 - 0.1
            brain.learn(
                Experience(obs, action, reward, next_obs, terminated, truncated)
            )
            _engine.ppo_learn(
                handle, obs, action, reward, next_obs, terminated, truncated
            )
            self.assert_state(brain, handle, nxt)
            if step in (2, 7, 14, 19):
                # Round-trip pending/partial-rollout, moments, and both RNGs.
                brain = PPOBrain.restore(brain.checkpoint())
                handle = _engine.ppo_create(_engine.ppo_checkpoint(handle))
                self.assert_state(brain, handle, nxt)

    def test_exact_continuation_native_kernel_reference(self):
        for hidden in ([3], [7, 5]):
            for target_kl in (0, 0.03, 1e-12):
                with self.subTest(hidden=hidden, target_kl=target_kl):
                    self.exercise(hidden, target_kl)

    def test_exact_continuation_pure_python_reference(self):
        with patch.object(kernels, "native", None):
            for hidden in ([3], [7, 5]):
                with self.subTest(hidden=hidden):
                    self.exercise(hidden, 0)

    def test_rejects_unsupported_precision_and_heads(self):
        brain = PPOBrain(settings([3]), ["rest", "eat"], 17, 901)
        state = brain.checkpoint()
        for field, value in (("precision", "float64"), ("late", {}), ("auxiliary", {})):
            candidate = copy.deepcopy(state)
            candidate[field] = value
            with (
                self.subTest(field=field),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                _engine.ppo_create(candidate)

    def test_singleton_terminal_and_truncated_rollouts(self):
        for terminated, truncated in ((True, False), (False, True), (True, True)):
            with self.subTest(terminated=terminated, truncated=truncated):
                brain = PPOBrain(settings([3]), ["rest", "eat"], 17, 901)
                handle = _engine.ppo_create(brain.checkpoint())
                obs, nxt = observation(0), observation(1)
                action = brain.choose(obs)
                self.assertEqual(action, _engine.ppo_choose(handle, obs, True))
                next_obs = None if terminated else nxt
                brain.learn(
                    Experience(obs, action, 0.25, next_obs, terminated, truncated)
                )
                _engine.ppo_learn(
                    handle, obs, action, 0.25, next_obs, terminated, truncated
                )
                self.assert_state(brain, handle, nxt)


if __name__ == "__main__":
    unittest.main()
