"""Exact native actor, trust-region and recurrent policy continuation contracts."""

# Optional compiled probes intentionally expose private continuation state.
# pylint: disable=c-extension-no-member,protected-access
# Test names and parameterized scenarios describe their expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

# Paired transition assertions stay explicit per algorithm family for independent parity diagnosis.
# pylint: disable=duplicate-code

import copy
import math
import random
import unittest

from tests.native_core_support import _engine, encoded
from tests.new_policy_support import settings as new_policy_settings
from wtf.config import load
from wtf.learning.brain import Experience, observation_inputs
from wtf.learning.brains import create_brain, restore_brain
from wtf.learning.grouped_rollouts import other_action
from wtf.learning.policy_gradients import PolicyStep
from wtf.learning.ppo import policy_distribution

KINDS = ("a2c", "reinforce", "grpo", "trpo", "recurrent_ppo", "lstm_ppo")


def settings(kind, hidden):
    """Use current learners with a small explicit legacy observation fixture."""
    config = (
        new_policy_settings(kind)
        if kind == "trpo"
        else next(v for v in load(kind)["brains"].values() if v["type"] == kind)
    )
    config = copy.deepcopy(config)
    config.update(observation_version=1, observation_top_k=1, movement_history_steps=0)
    for key in (
        "scent_observation",
        "observation_actions",
        "social_observation",
        "consequence_observation",
        "auxiliary_consequence",
        "late_consequence",
        "soft_observation",
    ):
        config.pop(key, None)
    config["network"]["hidden_layers"] = hidden
    config.update(
        rollout_steps=7,
        batch_size=3,
        epochs=2,
        normalize_advantages=True,
        target_kl=0.03,
    )
    if kind == "reinforce":
        config.update(max_episode_steps=100, baseline_decay=0.7)
    elif kind == "grpo":
        config.update(
            group_size=3, rollout_horizon=3, train_every_steps=2, kl_coefficient=0.01
        )
    elif kind == "trpo":
        config.update(cg_iterations=3, value_epochs=2, backtrack_steps=4)
    elif kind in ("recurrent_ppo", "lstm_ppo"):
        config.update(recurrent_size=3, sequence_length=3)
    return config


def observation(step, count):
    """Exercise nonzero continuous inputs without environmental RNG changes."""
    return tuple(math.sin(step * 0.17 + i * 0.23) * 0.7 for i in range(count))


def groups(brain, step, count, zero=False, unlikely=False):
    """Use matched policies but different outcome groups, as GRPO expects."""
    result = []
    for group in range(3):
        trajectory = []
        for index in range(group + 1):
            values = observation(step + index + group, count)
            _, logs = policy_distribution(brain.network.predict(values))
            action = (index + group) % len(brain.actions)
            trajectory.append(
                PolicyStep(
                    values,
                    action,
                    0.0 if zero else math.sin(index * 0.3 + group),
                    -50.0 if unlikely else logs[action],
                    index == group,
                    False,
                )
            )
        result.append(trajectory)
    return result


@unittest.skipUnless(
    _engine is not None and hasattr(_engine, "policy_create"),
    "Build the optional native policy factory",
)
class NativeActorPolicyTests(unittest.TestCase):
    def assert_state(self, brain, handle):
        self.assertEqual(
            encoded(brain.checkpoint()), encoded(_engine.policy_checkpoint(handle))
        )

    # Keep reference/native phase values together so comparison cadence is explicit.
    # pylint: disable-next=too-many-locals
    def test_all_six_families_match_every_parameter_diagnostic_and_restore(self):
        for kind in KINDS:
            for hidden in ([], [4, 3]):
                with self.subTest(kind=kind, hidden=hidden):
                    config = settings(kind, hidden)
                    brain = create_brain(config, ["rest", "eat", "move"], 17, 901)
                    handle = _engine.policy_create(brain.checkpoint())
                    width = observation_inputs(config)
                    self.assert_state(brain, handle)
                    for step in range(36):
                        current = observation(step, width)
                        following = observation(step + 1, width)
                        action = brain.choose(current)
                        self.assertEqual(
                            action, _engine.policy_choose(handle, current, True)
                        )
                        self.assert_state(brain, handle)
                        self.assertEqual(
                            encoded(
                                brain.diagnostics(current, include_activations=True)
                            ),
                            encoded(_engine.policy_diagnostics(handle, current, True)),
                        )
                        terminal, truncated = step in (11, 35), step == 23
                        reward = math.sin(step * 0.41) - 0.1
                        next_input = None if terminal else following
                        brain.learn(
                            Experience(
                                current, action, reward, next_input, terminal, truncated
                            )
                        )
                        _engine.policy_learn(
                            handle,
                            current,
                            action,
                            reward,
                            next_input,
                            terminal,
                            truncated,
                        )
                        self.assert_state(brain, handle)
                        if kind == "grpo" and step % 3 == 2:
                            trajectories = groups(brain, step, width)
                            brain.update_group(trajectories)
                            _engine.policy_group(
                                handle,
                                [
                                    [sample._asdict() for sample in row]
                                    for row in trajectories
                                ],
                            )
                            self.assert_state(brain, handle)
                        if step == 16:
                            state = brain.checkpoint()
                            brain = restore_brain(state)
                            handle = _engine.policy_create(state)

    # Keep independent live, trial and saved owners visible in each paired fixture.
    # pylint: disable-next=too-many-locals
    def test_world_versions_random_evaluation_and_external_trials_stay_independent(
        self,
    ):
        for kind in KINDS:
            for version in (1, 6, 7):
                with self.subTest(kind=kind, version=version):
                    config = settings(kind, [3, 2, 3])
                    config["observation_version"] = version
                    brain = create_brain(config, ["rest", "eat", "move"], 43, 789)
                    handle = _engine.policy_create(brain.checkpoint())
                    _engine.policy_world(handle)
                    width = observation_inputs(config)
                    heading = 7 if version == 6 else 2
                    for step in range(10):
                        values = list(observation(step, width))
                        values[heading : heading + 2] = (
                            (0, -1),
                            (1, 0),
                            (0, 1),
                            (-1, 0),
                        )[step % 4]
                        current = tuple(values)
                        for training, random_baseline in (
                            (False, False),
                            (False, True),
                            (True, False),
                        ):
                            action = brain.choose(
                                current,
                                explore=training,
                                random_baseline=random_baseline,
                            )
                            self.assertEqual(
                                action,
                                _engine.policy_choose(
                                    handle, current, training, random_baseline
                                ),
                            )
                            self.assert_state(brain, handle)
                        self.assertEqual(
                            encoded(
                                brain.diagnostics(current, include_activations=True)
                            ),
                            encoded(_engine.policy_diagnostics(handle, current, True)),
                        )
                        following = list(observation(step + 1, width))
                        following[heading : heading + 2] = (0, -1)
                        terminal = step == 9
                        next_input = None if terminal else tuple(following)
                        reward = math.cos(step * 0.3)
                        brain.learn(
                            Experience(
                                current, action, reward, next_input, terminal, False
                            )
                        )
                        _engine.policy_learn(
                            handle, current, action, reward, next_input, terminal, False
                        )
                        self.assert_state(brain, handle)
                        before = encoded(_engine.policy_checkpoint(handle))
                        generator = random.Random(step + 23)
                        initial_rng = generator.getstate()
                        memories = (
                            {1: brain.hidden[:]} if hasattr(brain, "hidden") else {}
                        )
                        memory = brain.hidden[:] if hasattr(brain, "hidden") else []
                        expected = other_action(
                            brain, current, generator, animal_id=1, memories=memories
                        )
                        result = _engine.policy_trial(
                            handle, current, initial_rng, memory
                        )
                        self.assertEqual(expected, result["action"])
                        # RNG states may cross the boundary as lists or tuples.
                        self.assertEqual(
                            encoded(generator.getstate()), encoded(result["rng"])
                        )
                        self.assertEqual(
                            encoded(memories.get(1, memory)), encoded(result["memory"])
                        )
                        self.assertEqual(
                            before, encoded(_engine.policy_checkpoint(handle))
                        )

    def test_grpo_zero_variance_and_kl_rejection_preserve_reference_and_rng(self):
        config = settings("grpo", [4, 3])
        for zero, unlikely in ((True, False), (False, True)):
            brain = create_brain(config, ["rest", "eat", "move"], 17, 901)
            handle = _engine.policy_create(brain.checkpoint())
            trajectories = groups(brain, 0, observation_inputs(config), zero, unlikely)
            brain.update_group(trajectories)
            _engine.policy_group(
                handle, [[sample._asdict() for sample in row] for row in trajectories]
            )
            self.assert_state(brain, handle)
            self.assertEqual(brain.updates, 0)

    def test_invalid_counters_reject_before_integer_narrowing(self):
        for kind in KINDS:
            brain = create_brain(settings(kind, [3]), ["rest", "eat", "move"], 17, 901)
            original = brain.checkpoint()
            for counter in ("steps", "updates", "rollout_updates"):
                for invalid in (-1, 1.25, float("nan"), float("inf"), 2**63):
                    state = copy.deepcopy(original)
                    state[counter] = invalid
                    with self.subTest(kind=kind, counter=counter, invalid=invalid):
                        with self.assertRaises(ValueError):
                            _engine.policy_create(state)

    def test_recurrent_invalid_layouts_reject_before_indexing(self):
        for kind in ("recurrent_ppo", "lstm_ppo"):
            brain = create_brain(settings(kind, [3]), ["rest", "eat", "move"], 17, 901)
            original = brain.checkpoint()
            for empty in ([], original["network"][:1]):
                state = copy.deepcopy(original)
                state["network"] = empty
                with self.subTest(kind=kind, network_layers=len(empty)):
                    with self.assertRaises(ValueError):
                        _engine.policy_create(state)
            for size in (0, -1):
                state = copy.deepcopy(original)
                state["config"]["recurrent_size"] = size
                with self.subTest(kind=kind, size=size):
                    with self.assertRaises(ValueError):
                        _engine.policy_create(state)
            state = copy.deepcopy(original)
            state["network"][0]["input_weights"] = []
            with self.assertRaises(ValueError):
                _engine.policy_create(state)
            state = copy.deepcopy(original)
            state["network"][1]["weights"] += [0.0] * len(state["network"][1]["bias"])
            with self.assertRaises(ValueError):
                _engine.policy_create(state)


if __name__ == "__main__":
    unittest.main()
