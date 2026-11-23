"""Exact v2 math/spatial/data behavior; no timing thresholds."""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

from array import array
import copy
import json
import math
import random

import unittest
from unittest.mock import patch

from wtf.native import kernels
from wtf.learning.ppo import MLP, policy_distribution, loss_and_gradient
from wtf.learning.qrdqn import quantile_huber
from wtf.learning.recurrent_ppo import GRUNetwork
from wtf.storage.binary_io import Encoder, Decoder, varint
from wtf.storage.replay_codec import clone, same
from wtf.inspection.telemetry import groups
from wtf.simulation.world import World, Animal, Resource, Event
from tests.test_ppo import ppo_settings
from tests.test_advanced_algorithms import settings as advanced_settings
from tests.test_simulation import config

NATIVE = kernels.native
SUPPORTED = NATIVE is not None and getattr(NATIVE, "api_version", 1) == 2


@unittest.skipUnless(SUPPORTED, "Build v2 native kernels first")
class NativeExtensionTests(unittest.TestCase):
    def test_masked_ppo_batch_matches_scalar_training_exactly(self):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        from tests.test_mutation import environment
        from wtf.experiment.runner import Experiment

        c = environment("maskable_ppo", 0)
        c["world"]["max_steps"] = 24
        for template in c["brains"].values():
            template.update(rollout_steps=4, batch_size=2, epochs=2, target_kl=0)
        batched, scalar = Experiment(c), Experiment(c)
        available = kernels.available
        for _ in range(24):
            batched.step()
            with patch.object(
                kernels,
                "available",
                side_effect=lambda name: (
                    False if name == "masked_ppo_batch" else available(name)
                ),
            ):
                scalar.step()
            self.assertEqual(batched.checkpoint(), scalar.checkpoint())

        brain = next(iter(batched.brains.values()))
        gradients = brain.network.zeros()
        baseline = copy.deepcopy(gradients)
        with self.assertRaises(ValueError):
            NATIVE.masked_ppo_batch(
                brain.network.layers,
                gradients,
                [
                    (
                        tuple([0.0] * brain.network.sizes[0]),
                        0,
                        0.0,
                        1.0,
                        0.0,
                        tuple(False for _ in brain.actions),
                    )
                ],
                0.2,
                0.01,
                0.5,
            )
        self.assertEqual(gradients, baseline)

    def test_policy_math_and_fused_batch_gradients_are_exact(self):

        # Keep setup, action and assertions together as one regression scenario.
        # pylint: disable=too-many-locals
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        rng = random.Random(987)
        for hidden in ([], [3], [4, 3]):
            network = MLP(hidden, 3, 42)
            observations = [
                tuple(rng.uniform(-1, 1) for _ in range(20)) for _ in range(5)
            ]
            samples = []
            c = ppo_settings()
            for i, obs in enumerate(observations):
                outputs = network.predict(obs)
                a = i % 3
                old = policy_distribution(outputs[:-1])[1][a]
                samples.append(
                    (obs, a, old + (0.3 if i % 2 else -0.4), (-1) ** i * 0.7, 0.6)
                )
            for entropy in (0.0, 0.01):
                c["entropy_coefficient"] = entropy
                ga, gb = network.zeros(), network.zeros()
                totals = {}
                got = NATIVE.ppo_batch(
                    network.layers,
                    ga,
                    samples,
                    c["clip_ratio"],
                    entropy,
                    c["value_coefficient"],
                )
                with patch.object(kernels, "native", None):
                    for obs, a, old, adv, target in samples:
                        outputs, tape = network.forward(obs)
                        loss, gradient, metrics = loss_and_gradient(
                            outputs, a, old, adv, target, c
                        )
                        network.accumulate_gradient(tape, gradient, gb)
                        for key, value in metrics.items():
                            totals[key] = totals.get(key, 0.0) + value
                        actual = NATIVE.ppo_loss(
                            outputs,
                            a,
                            old,
                            adv,
                            target,
                            c["clip_ratio"],
                            entropy,
                            c["value_coefficient"],
                        )
                        self.assertEqual(actual, (loss, gradient, metrics))
                self.assertEqual(got, totals)
                self.assertEqual(ga, gb)

    def test_gru_gates_and_backward_match_python_exactly(self):
        c = advanced_settings("recurrent_ppo")
        network = GRUNetwork(c, 3, 77)
        obs = tuple((i - 7) / 13 for i in range(20))
        hidden = [0.13, -0.21]
        with patch.object(kernels, "native", NATIVE):
            outputs, h, tape = network.step(obs, hidden, tape=True)
            ga = network.zeros()
            previous = network.backward(tape, [0.4, -0.3, 0.7, 0.2], [0.1, -0.04], ga)
        with patch.object(kernels, "native", None):
            other, oh, ot = network.step(obs, hidden, tape=True)
            gb = network.zeros()
            reference = network.backward(ot, [0.4, -0.3, 0.7, 0.2], [0.1, -0.04], gb)
        self.assertEqual(outputs, other)
        self.assertEqual(h, oh)
        self.assertEqual(tape, ot)
        self.assertEqual(previous, reference)
        self.assertEqual(
            [[list(v) for v in l.values()] for l in ga],
            [[list(v) for v in l.values()] for l in gb],
        )

    def test_quantile_loss_keeps_pairwise_accumulation_order(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        rng = random.Random(6)
        for n in (1, 3, 16):
            p = [rng.uniform(-5, 5) for _ in range(n)]
            t = [rng.uniform(-5, 5) for _ in range(n + 2)]
            for k in (0.4, 1.0, 2.0):
                with patch.object(kernels, "native", None):
                    expected = quantile_huber(p, t, k)
                self.assertEqual(NATIVE.quantile(p, t, k), expected)

    def test_group_metrics_match_sparse_python_for_dense_and_negative_coordinates(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        rng = random.Random(42)
        for n in (0, 1, 60, 500):
            animals = [
                {"x": rng.randrange(-20, 40), "y": rng.randrange(-10, 28)}
                for _ in range(n)
            ]
            got = NATIVE.groups(animals, 3)
            with patch.object(kernels, "native", None):
                expected = groups(animals)
            self.assertEqual(got, expected)

    def test_codec_keeps_bytes_types_signed_zero_aliases_and_detachment(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        shared = [{"x": -0.0, "y": 1.0}]
        data = {
            "first": shared,
            "second": shared,
            "arrays": [[0, 1, -1, 2**32, -(2**53 - 1)]],
        }
        a = clone(data)
        b = copy.deepcopy(data)
        self.assertEqual(a, b)
        self.assertIs(a["first"], a["second"])
        self.assertIsNot(a["first"], shared)
        self.assertEqual(math.copysign(1, a["first"][0]["x"]), -1.0)
        a["first"][0]["y"] = 9.0
        self.assertEqual(shared[0]["y"], 1.0)
        for pair in (
            (0, 0.0),
            (-0.0, 0.0),
            (data, copy.deepcopy(data)),
            ([True], [1]),
            ((1, [2]), (1, [2])),
        ):
            with patch.object(kernels, "native", None):
                expected = same(*pair)
            self.assertEqual(same(*pair), expected)
        for i in (0, 127, 128, 2**32 - 1, 2**64 - 1):
            with patch.object(kernels, "native", None):
                expected = varint(i)
            self.assertEqual(NATIVE.varint(i), expected)
        with patch.object(kernels, "native", None):
            encoded = Encoder().encode(data)
        self.assertEqual(Encoder().encode(data), encoded)
        self.assertEqual(Decoder().decode(encoded), data)

    def test_body_and_growth_phases_preserve_numeric_types_and_death_events(self):

        # Keep setup, action and assertions together as one regression scenario.
        # pylint: disable=too-many-locals
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member
        # Energy depletion and positive health are separate physical gates kept
        # explicit for auditability.
        # pylint: disable=chained-comparison

        c = config()
        w = World(c, 12)
        for i, a in w.animals.items():
            a.energy = 0.03 if i % 2 else 0
            a.health = 2 if i % 2 else 0.5
        x = copy.deepcopy(w.animals)
        y = copy.deepcopy(w.animals)
        actions = {i: "move_forward" if i % 2 else "rest" for i in x}
        r = {i: 0.0 for i in x}
        er = {i: 0.0 for i in y}
        events = []
        expected = []
        NATIVE.flat_metabolism(
            x.values(), actions, c["species"], r, events, Event, Animal
        )
        for i, a in y.items():
            b = c["species"][a.species]["body"]
            cost = b["living_cost"]
            if actions[i] == "move_forward":
                cost += b["movement_cost"]
            a.energy = max(0.0, a.energy - cost)
            er[i] += cost * c["species"][a.species]["rewards"].get("per_energy_cost", 0)
            a.age += 1
            if a.energy <= 0 and a.health > 0:
                damage = min(a.health, b["starvation_damage"])
                a.health -= damage
                expected.append(
                    Event("damage_taken", i, amount=damage, cause="starvation")
                )
        self.assertEqual(
            json.dumps([vars(a) for a in x.values()]),
            json.dumps([vars(a) for a in y.values()]),
        )
        self.assertEqual(events, expected)
        self.assertEqual(r, er)
        patches = [Resource(i, 0, q) for i, q in enumerate((0, 0.0, -0.0, 19.9, 20))]
        reference = copy.deepcopy(patches)
        NATIVE.resource_growth(patches, Resource, 20, 0.1)
        for p in reference:
            p.quantity = min(20, p.quantity + 0.1)
        self.assertEqual(
            json.dumps([vars(p) for p in patches]),
            json.dumps([vars(p) for p in reference]),
        )

    def test_native_views_are_released_and_bad_inputs_are_rejected(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        f = array("f", [1.0, 2.0, 3.0, 4.0])
        d = array("d", [0.0] * 4)
        for call in (
            lambda: NATIVE.gru_step([1.0], [0.0], f, f, f, False),
            lambda: NATIVE.nearest([], 0, 0, 1, 3, b"", b"", 4, 4, 3, 0),
            lambda: NATIVE.mlp_predict([], [], 0, 0),
            lambda: NATIVE.ppo_batch([], [], [], 0.2, 0.01, 0.5),
            lambda: NATIVE.decode_integers(b"\x80", 0, 1),
        ):
            with self.assertRaises((ValueError, TypeError)):
                call()
        f.append(0.0)
        f.pop()
        d.append(0.0)
        d.pop()
        with self.assertRaises(ValueError):
            NATIVE.quantile([], [], 1.0)

    def test_stale_extension_does_not_activate_v2_functions(self):
        class Old:

            # This record/owner deliberately keeps its related fields and lifecycle
            # operations together without artificial wrappers.
            # pylint: disable=too-few-public-methods

            api_version = 1

        with patch.object(kernels, "native", Old()):
            self.assertFalse(kernels.available("nearest"))
