"""Exact V6 coefficient reuse and zero-evidence auxiliary fast paths."""

# Fixtures inspect bounded structural caches and private heads to enforce ownership.
# pylint: disable=missing-function-docstring,protected-access
import copy
import math
import random
import struct
import unittest
from unittest.mock import patch
from wtf.config import load
from wtf.learning.brains import create_brain
from wtf.simulation import soft_observations as soft


def reference(values, template):
    """Straightforward allocating encoder, independent of the optimized assembly."""
    result = []
    for field in soft.soft_layout(template).fields:
        index = field.raw
        if field.kind == "raw":
            result.append(values[index])
        elif field.kind in ("position", "vector"):
            if field.count == 3 and not values[index + 3]:
                result.extend([0.0] * 11)
                continue
            forward, right = values[index : index + 2]
            distance = (
                values[index + 2] if field.count == 3 else math.hypot(forward, right)
            )
            result.extend(soft.memberships(distance, field.knots))
            result.extend(soft.direction(forward, right))
        else:
            result.extend(soft.memberships(values[index], field.knots))
    return result


def bits(values):
    """Check signed zeros as well as ordinary numerical equality."""
    return struct.pack("<" + "d" * len(values), *values)


class V6EfficiencyTests(unittest.TestCase):
    """Sharing immutable numeric coefficients cannot change observed/trained values."""

    def test_reference_matches_bits_types_and_mutation_boundaries(self):
        c = load("environments/seaside_150_v6.json")
        rng = random.Random(11)
        for source in c["brains"].values():
            template = copy.deepcopy(source)
            template.update(
                movement_history_steps=2,
                scent_observation={"mode": "blended", "radius": 2},
            )
            template["soft_categories"] = {"climb": [-1, -0.3, 0, 0.2, 1]}
            layout = soft.soft_layout(template)
            valuesets = [[0.0] * len(layout.raw_names), [-0.0] * len(layout.raw_names)]
            valuesets += [
                [
                    rng.choice([-1, -0.0, 0, 0.0001, 0.1, 0.5, 1])
                    for _ in layout.raw_names
                ]
                for _ in range(24)
            ]
            for values in valuesets:
                expected = reference(values, template)
                actual = soft.encode_soft(values, template)
                self.assertEqual(bits(actual), bits(expected))
                self.assertEqual([type(x) for x in actual], [type(x) for x in expected])
                saved = actual.copy()
                values[0] = 0.456
                soft._basis.cache_clear()
                soft._direction_entries.cache_clear()
                self.assertEqual(bits(actual), bits(saved))
            self.assertEqual(copy.deepcopy(layout), layout)

    def test_value_caches_are_bounded_and_keys_include_knots(self):
        for i in range(5000):
            soft._basis(i / 5001, (0, 0.3, 1))
        # Pylint mistakes the LRU wrapper method for the wrapped function.
        # pylint: disable-next=no-value-for-parameter
        self.assertLessEqual(soft._basis.cache_info().currsize, 4096)
        self.assertNotEqual(
            soft._basis(0.1, (0, 0.3, 1)), soft._basis(0.1, (0, 0.2, 1))
        )
        for i in range(2200):
            soft._direction_entries(i / 2201, 0.25)
        # Pylint mistakes the LRU wrapper method for the wrapped function.
        # pylint: disable-next=no-value-for-parameter
        self.assertLessEqual(soft._direction_entries.cache_info().currsize, 2048)

    def test_partial_decoding_matches_allocating_reference_and_cache_is_bounded(self):
        c = load("environments/seaside_150_v6.json")
        template = c["brains"]["predator_ppo"]
        layout = soft.soft_layout(template)
        rng = random.Random(9)
        for _ in range(12):
            raw = [rng.choice([-0.0, 0, 0.001, 0.2, 0.7, 1]) for _ in layout.raw_names]
            values = tuple(reference(raw, template))
            for start in (0, layout.core, len(raw) - 96, len(raw) - 48):
                expected = []
                for field in layout.fields:
                    if field.raw < start:
                        continue
                    block = values[field.offset : field.offset + len(field.names)]
                    if field.kind == "raw":
                        expected.append(block[0])
                    elif field.kind in ("position", "vector"):
                        n = len(field.knots)
                        d = sum(a * b for a, b in zip(block[:n], field.knots))
                        angle = soft.bearing(block[n:])
                        expected.extend((d * math.cos(angle), d * math.sin(angle)))
                        if field.count == 3:
                            expected.append(d)
                    else:
                        expected.append(sum(a * b for a, b in zip(block, field.knots)))
                self.assertEqual(
                    bits(soft.decode_layout(values, layout, start)), bits(expected)
                )
        for i in range(4300):
            soft._unbasis((i / 4301, 1 - i / 4301, 0), (0, 0.3, 1))
        # Pylint mistakes the LRU wrapper method for the wrapped function.
        # pylint: disable-next=no-value-for-parameter
        self.assertLessEqual(soft._unbasis.cache_info().currsize, 4096)

    def test_no_evidence_skips_decode_and_preserves_custom_target_hooks(self):
        c = load("environments/seaside_150_v6.json")
        spec = c["species"]["prey"]
        config = c["brains"][spec["brain"]]
        brain = create_brain(config, spec["allowed_actions"], 1, 2)
        layout = soft.soft_layout(config)
        observation = tuple(soft.encode_soft([0.0] * len(layout.raw_names), config))
        gradients = brain.network.zeros()
        with patch(
            "wtf.learning.auxiliary_consequence.decode_layout",
            side_effect=AssertionError("unused decode"),
        ):
            metrics, head = brain.auxiliary.accumulate(
                brain.network, [observation], gradients
            )
        self.assertEqual(metrics, {"auxiliary_loss": 0.0, "auxiliary_coverage": 0})
        self.assertTrue(
            all(not v for layer in head for values in layer.values() for v in values)
        )
        brain.auxiliary.targets = lambda obs, inputs: (
            tuple(obs[:inputs]),
            [(1, 0, 0, 0)] * 12,
        )
        metrics, _ = brain.auxiliary.accumulate(
            brain.network, [observation], brain.network.zeros()
        )
        self.assertEqual(metrics["auxiliary_coverage"], 1)
        del brain.auxiliary.targets

        def custom(_self, obs, inputs):
            return tuple(obs[:inputs]), [(1, 0, 0, 0)] * 12

        with patch.object(type(brain.auxiliary), "targets", new=custom):
            metrics, _ = brain.auxiliary.accumulate(
                brain.network, [observation], brain.network.zeros()
            )
        self.assertEqual(metrics["auxiliary_coverage"], 1)
