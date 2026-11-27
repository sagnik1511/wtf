"""Exact native continuation for reordered valid checkpoint parameter fields."""

# Private packed states are inspected to establish ownership and numeric equality.
# pylint: disable=protected-access,c-extension-no-member
# Each paired phase stays together so parameter/order comparisons remain explicit.
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# Parameterized scenarios document the test contracts.
# pylint: disable=missing-function-docstring,missing-class-docstring
import json
import math
import unittest
from tests.native_core_support import _engine, encoded, NATIVE_KINDS
from tests.test_cpp_actor_policies import settings as actor_settings
from tests.test_cpp_value_policies import settings as value_settings
from tests.test_cpp_augmented_ppo import config, observation, ACTIONS, native
from wtf.learning.brains import create_brain, restore_brain
from wtf.learning.brain import Experience, observation_inputs
from wtf.learning.policy_gradients import PolicyStep
from wtf.learning.ppo import policy_distribution


def equal(a, b, label):
    if encoded(a) != encoded(b):

        def find(x, y, path=""):
            if isinstance(x, dict):
                for k in x:
                    if k not in y:
                        return (path, k, "missing")
                    r = find(x[k], y[k], path + "/" + k)
                    if r:
                        return r
            elif isinstance(x, (tuple, list)):
                if len(x) != len(y):
                    return (path, len(x), len(y))
                for i, (v, w) in enumerate(zip(x, y)):
                    r = find(v, w, path + "/" + str(i))
                    if r:
                        return r
            elif encoded(x) != encoded(y):
                return (path, x, y)
            return None

        raise AssertionError((label, find(a, b)))


def checkorder(a, b):
    for key in (
        "network",
        "moments",
        "variances",
        "reference_network",
        "target_network",
    ):
        if key in a:
            assert [list(r) for r in a[key]] == [list(r) for r in b[key]], (
                "order",
                key,
            )
    for key in ("weights", "target"):
        if key in a:
            assert list(a[key]) == list(b[key]), ("order", key)
    for key in ("auxiliary", "late"):
        if key in a:
            checkorder(a[key], b[key])


def run(kind, extra=False, sorted_keys=True):
    if kind in ("q_network", "dqn", "ddqn", "qrdqn", "discrete_sac"):
        c = value_settings(kind, [4, 3])
    elif kind in ("ppo", "maskable_ppo", "mappo"):
        c = config(kind, 7, extra, extra) if kind != "mappo" else config(kind, 7)
    else:
        c = actor_settings(kind, [4, 3])
    # Lifecycle A2C must not depend on unrelated PPO-only fixture fields.
    if kind == "a2c":
        c.pop("batch_size", None)
    brain = create_brain(c, ACTIONS, 17, 901)
    state = json.loads(json.dumps(brain.checkpoint(), sort_keys=bool(sorted_keys)))
    if sorted_keys == "mixed":
        for key in (
            "network",
            "moments",
            "variances",
            "reference_network",
            "target_network",
        ):
            if key not in state:
                continue
            for index, row in enumerate(state[key]):
                names = list(row)
                shift = (index + (key == "moments") + 1) % len(names)
                names = names[shift:] + names[:shift]
                state[key][index] = {name: row[name] for name in names}
        for key in ("weights", "target"):
            if key in state:
                state[key] = dict(reversed(list(state[key].items())))
    brain = restore_brain(state)
    handle = native.create(state)
    equal(brain.checkpoint(), native.checkpoint(handle), (kind, "initial"))
    width = observation_inputs(c)
    for step in range(35):
        values = (
            observation(c, step)
            if kind in ("ppo", "maskable_ppo", "mappo")
            else tuple(math.sin(step * 0.17 + i * 0.23) * 0.7 for i in range(width))
        )
        following = (
            observation(c, step + 1)
            if kind in ("ppo", "maskable_ppo", "mappo")
            else tuple(
                math.sin((step + 1) * 0.17 + i * 0.23) * 0.7 for i in range(width)
            )
        )
        ctx = {}
        if kind == "maskable_ppo":
            brain.mask = tuple(
                i == 1 or (i + step) % 3 != 0 for i in range(len(ACTIONS))
            )
            ctx["mask"] = brain.mask
        if kind == "mappo":
            brain.context = tuple(math.sin(step + i * 0.1) for i in range(16))
            brain.next_context = tuple(math.cos(step + i * 0.1) for i in range(16))
            ctx.update(context=brain.context, next_context=brain.next_context)
        native.context(handle, ctx)
        action = brain.choose(values)
        assert action == native.choose(handle, values, True), (kind, step, "action")
        term = step in (11, 34)
        trunc = step == 23
        reward = math.sin(step * 0.41) - 0.1
        brain.learn(
            Experience(values, action, reward, None if term else following, term, trunc)
        )
        native.learn(
            handle, values, action, reward, None if term else following, term, trunc
        )
        equal(
            brain.checkpoint(),
            native.checkpoint(handle),
            (kind, extra, sorted_keys, step),
        )
        checkorder(brain.checkpoint(), native.checkpoint(handle))
        if kind == "grpo" and step % 4 == 3:
            groups = []
            for g in range(3):
                row = []
                for j in range(g + 1):
                    _, logs = policy_distribution(brain.network.predict(values))
                    a = (g + j) % len(ACTIONS)
                    row.append(
                        PolicyStep(
                            values, a, math.sin(j * 0.3 + g), logs[a], False, j == g
                        )
                    )
                groups.append(row)
            brain.update_group(groups)
            _engine.policy_group(handle, [[s._asdict() for s in row] for row in groups])
            equal(brain.checkpoint(), native.checkpoint(handle), (kind, "group", step))


@unittest.skipUnless(
    _engine is not None and hasattr(_engine, "policy_context"),
    "Build the optional native policy factory",
)
class NativeCheckpointOrderTests(unittest.TestCase):
    def test_all_families_preserve_values_and_field_orders(self):
        for order in (False, True, "mixed"):
            for kind in NATIVE_KINDS:
                with self.subTest(kind=kind, order=order):
                    run(kind, sorted_keys=order)
            for kind in ("ppo", "maskable_ppo"):
                with self.subTest(kind=kind, order=order, heads=True):
                    run(kind, True, order)


if __name__ == "__main__":
    unittest.main()
