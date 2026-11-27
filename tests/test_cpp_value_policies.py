"""Independent native off-policy updates, diagnostics and exact recovery."""

# Numerical fixtures intentionally inspect exact optional-extension state.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import copy
import json
import math
import random
import unittest
from types import SimpleNamespace

from tests.native_core_support import _engine, encoded
from tests.test_advanced_algorithms import settings as advanced_settings
from tests.new_policy_support import settings as new_settings
from tests.test_simulation import config as world_config
from wtf.learning.brains import create_brain, restore_brain
from wtf.learning.brain import Experience, observation_inputs
from wtf.learning.grouped_rollouts import other_action

_METHODS = ("create", "choose", "learn", "checkpoint", "diagnostics", "trial")
native = (
    SimpleNamespace(**{name: getattr(_engine, "policy_" + name) for name in _METHODS})
    if _engine is not None
    and all(hasattr(_engine, "policy_" + name) for name in _METHODS)
    else None
)

KINDS = ("q_network", "dqn", "ddqn", "qrdqn", "discrete_sac")


def settings(kind, hidden=None):
    config = (
        world_config()["brains"]["small_q"]
        if kind == "q_network"
        else new_settings(kind) if kind == "discrete_sac" else advanced_settings(kind)
    )
    config = copy.deepcopy(config)
    for key in (
        "observation_actions",
        "movement_history_steps",
        "scent_observation",
        "social_observation",
        "consequence_observation",
    ):
        config.pop(key, None)
    config.update(
        observation_version=1,
        observation_top_k=1,
        replay_capacity=11,
        batch_size=3,
        learning_starts_after=3,
        train_every_steps=2,
    )
    if kind != "discrete_sac":
        config["target_update_every_training_steps"] = 3
    if kind == "q_network":
        config["hidden_layers"] = [4]
    elif hidden is not None:
        config["network"]["hidden_layers"] = hidden
    return config


def obs(step, width=20):
    return tuple(
        math.sin(step * 0.37 + i * 0.13) if i % 4 else (-0.0 if step % 2 else 0.0)
        for i in range(width)
    )


# Each mismatch branch returns its shortest useful diagnostic path.
# pylint: disable-next=too-many-return-statements
def mismatch(a, b, path=""):
    if type(a) != type(b) and not isinstance(a, (list, tuple)):
        return path, f"types {type(a)} {type(b)}", repr(a)[:150], repr(b)[:150]
    if isinstance(a, dict):
        if set(a) != set(b):
            return path, "keys", set(a) - set(b), set(b) - set(a)
        for key in a:
            diff = mismatch(a[key], b[key], path + "." + key)
            if diff:
                return diff
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return path, "length", len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            diff = mismatch(x, y, f"{path}[{i}]")
            if diff:
                return diff
    elif a != b or isinstance(a, float) and math.copysign(1, a) != math.copysign(1, b):
        return path, "value", repr(a), repr(b)
    return None


@unittest.skipUnless(native is not None, "Build the optional all-policy native probe")
class ValuePolicyTests(unittest.TestCase):
    def compare(self, brain, handle, label):
        expected = brain.checkpoint()
        actual = native.checkpoint(handle)
        self.assertEqual(
            encoded(expected), encoded(actual), f"{label}: {mismatch(expected,actual)}"
        )

    # Keep the complete transition and all parity assertions together.
    # pylint: disable-next=too-many-locals
    def test_exact_all_policies_updates_restore_and_interleaved_diagnostics(self):
        for kind in KINDS:
            for hidden in ([3], [5, 4, 3]):
                if kind == "q_network" and len(hidden) > 1:
                    continue
                with self.subTest(kind=kind, hidden=hidden):
                    config = settings(kind, hidden)
                    brain = create_brain(config, ["rest", "eat", "move"], 17, 901)
                    handle = native.create(brain.checkpoint())
                    self.compare(brain, handle, kind + " initial")
                    width = observation_inputs(config)
                    for step in range(96):
                        current, following = obs(step, width), obs(step + 1, width)
                        for activation in (False, True):
                            expected = brain.diagnostics(
                                current, include_activations=activation
                            )
                            actual = native.diagnostics(handle, current, activation)
                            self.assertEqual(
                                encoded(expected),
                                encoded(actual),
                                f"{kind} diagnostics {step}: {mismatch(expected,actual)}",
                            )
                        if step % 9 == 0:
                            self.assertEqual(
                                brain.choose(current, explore=False),
                                native.choose(handle, current, False),
                            )
                            self.assertEqual(
                                brain.choose(current, random_baseline=True),
                                native.choose(handle, current, True, True),
                            )
                        action = brain.choose(current)
                        self.assertEqual(
                            action,
                            native.choose(handle, current, True),
                            f"{kind} action {step}",
                        )
                        self.compare(brain, handle, f"{kind} afterchoose {step}")
                        terminal = step in (13, 38, 95)
                        truncated = step in (9, 29, 67)
                        reward = math.sin(step * 0.19) + 0.1
                        brain.learn(
                            Experience(
                                current,
                                action,
                                reward,
                                None if terminal else following,
                                terminal,
                                truncated,
                            )
                        )
                        native.learn(
                            handle,
                            current,
                            action,
                            reward,
                            None if terminal else following,
                            terminal,
                            truncated,
                        )
                        self.compare(brain, handle, f"{kind} afterlearn {step}")
                        if step in (17, 41):
                            state = json.loads(encoded(brain.checkpoint()))
                            brain = restore_brain(state)
                            handle = native.create(state)
                    self.assertGreater(brain.updates, 0)
                    print("PASS", kind, hidden, "updates", brain.updates, flush=True)

    def test_both_replay_sampling_paths_and_sac_target_extremes(self):
        cases = [
            ("dqn", 3, 64, None),
            ("ddqn", 8, 128, None),
            ("discrete_sac", 3, 64, 0.0),
            ("discrete_sac", 3, 64, 1.0),
        ]
        for kind, batch, capacity, tau in cases:
            config = settings(kind, [4, 3])
            config.update(
                batch_size=batch,
                replay_capacity=capacity,
                learning_starts_after=batch,
                train_every_steps=1,
            )
            if tau is not None:
                config["target_tau"] = tau
            brain = create_brain(config, ["rest", "eat", "move"], 17, 901)
            handle = native.create(brain.checkpoint())
            for step in range(190):
                current, following = obs(step), obs(step + 1)
                action = brain.choose(current)
                self.assertEqual(action, native.choose(handle, current, True))
                terminal = step % 31 == 30
                reward = math.sin(step * 0.3)
                brain.learn(
                    Experience(
                        current,
                        action,
                        reward,
                        None if terminal else following,
                        terminal,
                        False,
                    )
                )
                native.learn(
                    handle,
                    current,
                    action,
                    reward,
                    None if terminal else following,
                    terminal,
                    False,
                )
                self.compare(brain, handle, f"{kind} samplepath {step}")

    def test_restored_sparse_scalar_types_are_exact_and_models_stay_independent(self):
        for kind in KINDS:
            brain = create_brain(settings(kind, [4]), ["rest", "eat", "move"], 21, 109)
            current = list(obs(0))
            current[0] = 2**60 + 1
            current[1] = True
            current[2] = -1
            following = list(obs(1))
            following[2] = 1
            following[3] = False
            # Import exact record scalar types without retaining whole dictionary trees.
            state = brain.checkpoint()
            state["replay"] = [
                {
                    "observation": current,
                    "action": "rest",
                    "reward": 1,
                    "next_observation": following,
                    "terminated": False,
                    "truncated": False,
                }
            ]
            state["cursor"] = 1
            left = native.create(state)
            right = native.create(state)
            self.assertEqual(encoded(state), encoded(native.checkpoint(left)))
            expected = encoded(state)
            for step in range(4):
                native.learn(
                    left, obs(step), "rest", float(step), obs(step + 1), False, False
                )
            self.assertEqual(expected, encoded(native.checkpoint(right)))
            self.assertNotEqual(expected, encoded(native.checkpoint(left)))

    def test_frozen_trial_actions_use_only_external_random_state(self):
        for kind in KINDS:
            brain = create_brain(settings(kind, [4]), ["rest", "eat", "move"], 91, 104)
            handle = native.create(brain.checkpoint())
            before = encoded(brain.checkpoint())
            generator = random.Random(513)
            state = generator.getstate()
            for step in range(40):
                current = obs(step)
                expected = other_action(brain, current, generator)
                actual = native.trial(handle, current, state)
                self.assertEqual(expected, actual["action"])
                self.assertEqual(encoded(generator.getstate()), encoded(actual["rng"]))
                state = actual["rng"]
            self.assertEqual(before, encoded(brain.checkpoint()))
            self.assertEqual(before, encoded(native.checkpoint(handle)))


if __name__ == "__main__":
    unittest.main()
