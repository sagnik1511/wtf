"""Exact masked, centralized and consequence-augmented PPO continuation."""

# Native probes intentionally expose private continuation state.
# pylint: disable=c-extension-no-member,protected-access
# Paired fixtures keep each Python/native phase visible for exact comparison.
# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
# Parameterized cases describe the assertion contract.
# pylint: disable=missing-function-docstring,missing-class-docstring
# Paired reference/native transitions remain explicit to expose head and mask state boundaries.
# pylint: disable=duplicate-code

import math
import random
import unittest
from tests.native_core_support import _engine, encoded
from tests.test_cpp_ppo import settings
from wtf.learning.brains import create_brain, restore_brain
from wtf.learning.brain import Experience, observation_inputs
from wtf.simulation.soft_observations import soft_layout, encode_soft
from wtf.learning.ppo import RolloutStep
from wtf.learning.grouped_rollouts import sample_index


# Defer optional extension attribute lookup until a non-skipped test runs.
# pylint: disable=unnecessary-lambda,too-few-public-methods
class Native:
    create = staticmethod(lambda *a: _engine.policy_create(*a))
    checkpoint = staticmethod(lambda *a: _engine.policy_checkpoint(*a))
    choose = staticmethod(lambda *a: _engine.policy_choose(*a))
    learn = staticmethod(lambda *a: _engine.policy_learn(*a))
    diagnostics = staticmethod(lambda *a: _engine.policy_diagnostics(*a))
    context = staticmethod(lambda *a: _engine.policy_context(*a))

    @staticmethod
    def trial(*args):
        result = _engine.policy_trial(*args)
        return result["action"], result["rng"], result["memory"]


native = Native()


def compare(expected, actual, label):
    if encoded(expected) != encoded(actual):
        raise AssertionError(f"Native continuation mismatch at {label}")


ACTIONS = ["move_forward", "rest", "eat", "attack", "reproduce"]


def config(
    kind, version, aux=False, late=False, scope="both", local=True, hidden=None, kl=0
):
    c = settings([4, 3] if hidden is None else hidden, kl)
    c.update(type=kind, observation_version=version, observation_top_k=1)
    if kind == "mappo":
        c["critic_hidden_layers"] = [3, 2]
    if local:
        c["social_observation"] = {"history_steps": 4, "max_neighbors": 4}
    if aux or late:
        c.update(consequence_observation=True, consequence_scope=scope)
    if version == 7:
        c["observation_actions"] = ACTIONS[:]
    if aux:
        c["auxiliary_consequence"] = {"coefficient": 0.13}
    if late:
        c["late_consequence"] = {"max_adjustment": 1.7}
    return c


def observation(c, step, zero=False):
    if c["observation_version"] == 6:
        count = len(soft_layout(c).raw_names)
    else:
        count = observation_inputs(c)
    values = [0.45 + 0.4 * math.sin(step * 0.17 + i * 0.23) for i in range(count)]
    if c.get("consequence_observation"):
        width = (
            (len(ACTIONS) if c["observation_version"] == 7 else 12)
            * 4
            * (2 if c.get("consequence_scope") == "both" else 1)
        )
        for i in range(count - width, count, 4):
            values[i] = 0.0 if zero or (i // 4 + step) % 4 == 0 else values[i]
    if c["observation_version"] == 6:
        return tuple(encode_soft(tuple(values), c))
    return tuple(values)


def run_case(
    kind, version, aux=False, late=False, scope="both", local=True, hidden=None, kl=0
):
    c = config(kind, version, aux, late, scope, local, hidden, kl)
    brain = create_brain(c, ACTIONS, 17, 901)
    if late:
        generator = random.Random(34)
        for i in range(len(brain.late.network.layers[0]["weights"])):
            brain.late.network.layers[0]["weights"][i] = generator.uniform(-0.2, 0.2)
    handle = native.create(brain.checkpoint())
    label = (kind, version, aux, late, scope, local, hidden, kl)
    compare(brain.checkpoint(), native.checkpoint(handle), (label, "initial"))
    for step in range(24):
        current, following = observation(c, step, step % 7 == 3), observation(
            c, step + 1, step % 7 == 2
        )
        ctx = {}
        if kind == "maskable_ppo":
            brain.mask = tuple(
                i == 1 or (i + step) % 3 != 0 for i in range(len(ACTIONS))
            )
            ctx["mask"] = brain.mask
        if kind == "mappo":
            brain.context = tuple(math.sin(step * 0.31 + i * 0.17) for i in range(16))
            brain.next_context = tuple(
                math.cos(step * 0.21 + i * 0.19) for i in range(16)
            )
            ctx.update(context=brain.context, next_context=brain.next_context)
        native.context(handle, ctx)
        compare(brain.checkpoint(), native.checkpoint(handle), (label, "context", step))
        action = brain.choose(current)
        assert action == native.choose(handle, current, True), (label, step, "action")
        compare(brain.checkpoint(), native.checkpoint(handle), (label, "choose", step))
        compare(
            brain.diagnostics(current, include_activations=True),
            native.diagnostics(handle, current, True),
            (label, "diagnostics", step),
        )
        terminal = step in (11, 23)
        truncated = step == 17
        reward = math.sin(step * 0.41) - 0.1
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
        compare(brain.checkpoint(), native.checkpoint(handle), (label, "learn", step))
        if step == 9:
            state = brain.checkpoint()
            brain = restore_brain(state)
            handle = native.create(state)


@unittest.skipUnless(
    _engine is not None and hasattr(_engine, "policy_context"),
    "Build the optional native policy factory",
)
class NativeAugmentedPPOTests(unittest.TestCase):
    def test_observation_versions_scopes_and_head_combinations(self):

        cases = []
        for kind in ("maskable_ppo", "mappo"):
            for v in (1, 4, 5, 6, 7):
                cases.append((kind, v))
        for kind in ("ppo", "maskable_ppo"):
            for v in (4, 5, 6, 7):
                for scope in ("both", "context", "species_action"):
                    cases.append((kind, v, True, False, scope))
            for aux in (False, True):
                for scope in ("both", "context", "species_action"):
                    for local in (False, True):
                        cases.append((kind, 7, aux, True, scope, local))
        for case in cases:
            run_case(*case)

    # Keep all paired edge phases together so ownership checks stay explicit.
    # pylint: disable-next=too-many-branches,too-many-statements
    def test_depth_kl_gates_modes_and_detached_trials(self):
        for kind in ("ppo", "maskable_ppo", "mappo"):
            for hidden in ([], [3, 2, 3]):
                for kl in (0, 1e-7):
                    run_case(
                        kind,
                        7,
                        kind != "mappo",
                        kind != "mappo",
                        "both",
                        True,
                        hidden,
                        kl,
                    )

        for kind in ("ppo", "maskable_ppo"):
            c = config(kind, 7, True, True, "both", True, [], 1e-9)
            brain = create_brain(c, ACTIONS, 17, 901)
            if kind == "maskable_ppo":
                brain.mask = tuple(i != 2 for i in range(len(ACTIONS)))
            for i in range(6):
                raw = observation(c, i)
                packed = brain._pack(raw) if kind == "maskable_ppo" else raw
                out = brain.network.predict(packed)
                _, logs = brain.distribution(out, packed)
                brain.rollout.append(
                    RolloutStep(
                        packed,
                        1,
                        0.3 + i / 10,
                        logs[1] - 0.3,
                        out[-1],
                        out[-1] + 0.1,
                        False,
                        False,
                    )
                )
            state = brain.checkpoint()
            handle = native.create(state)
            current, following = observation(c, 6), observation(c, 7)
            action = brain.choose(current)
            assert action == native.choose(handle, current, True)
            brain.learn(Experience(current, action, 0.7, following, False, False))
            native.learn(handle, current, action, 0.7, following, False, False)
            compare(
                brain.checkpoint(), native.checkpoint(handle), (kind, "KL rejection")
            )
            assert brain.updates == brain.late.updates == brain.auxiliary.updates == 0
            assert brain.last_metrics == {"early_stopped": True}

        for kind in ("ppo", "maskable_ppo", "mappo"):
            c = config(kind, 7, kind != "mappo", kind != "mappo", "both", True, [4, 3])
            brain = create_brain(c, ACTIONS, 17, 901)
            handle = native.create(brain.checkpoint())
            for step in range(10):
                current, following = observation(c, step), observation(c, step + 1)
                ctx = {}
                if kind == "maskable_ppo":
                    brain.mask = tuple(
                        i == 1 or (i + step) % 2 == 0 for i in range(len(ACTIONS))
                    )
                    ctx["mask"] = brain.mask
                if kind == "mappo":
                    brain.context = tuple(math.sin(step + i * 0.1) for i in range(16))
                    brain.next_context = tuple(
                        math.cos(step + i * 0.1) for i in range(16)
                    )
                    ctx.update(context=brain.context, next_context=brain.next_context)
                native.context(handle, ctx)
                for train, random_baseline in (
                    (False, False),
                    (False, True),
                    (True, False),
                ):
                    action = brain.choose(
                        current, explore=train, random_baseline=random_baseline
                    )
                    assert action == native.choose(
                        handle, current, train, random_baseline
                    ), (kind, "mode")
                    compare(
                        brain.checkpoint(), native.checkpoint(handle), (kind, "mode")
                    )
                before = encoded(native.checkpoint(handle))
                generator = random.Random(71 + step)
                trial_ctx = {}
                if kind == "maskable_ppo":
                    m = tuple(
                        i == 1 or (i + step) % 3 == 0 for i in range(len(ACTIONS))
                    )
                    trial_ctx["mask"] = m
                    packed = tuple(current) + m
                    p, _ = brain.distribution(brain.network.predict(packed), packed)
                else:
                    p = brain.action_probabilities(current)
                initial = generator.getstate()
                expected = ACTIONS[sample_index(p, generator)]
                result = native.trial(handle, current, initial, None, trial_ctx)
                assert result[0] == expected
                assert encoded(result[1]) == encoded(generator.getstate())
                assert encoded(native.checkpoint(handle)) == before
                brain.learn(Experience(current, action, 0.3, following, False, False))
                native.learn(handle, current, action, 0.3, following, False, False)
                compare(
                    brain.checkpoint(),
                    native.checkpoint(handle),
                    (kind, "frozen trial"),
                )


if __name__ == "__main__":
    unittest.main()
