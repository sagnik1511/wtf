"""Check optimizations against reference algorithms, not timing thresholds."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from ecosystem.brain import Brain
from ecosystem.runner import Experiment
from ecosystem.world import DIRECTIONS, World
from tests.test_simulation import config


def reference_observation(world, a):
    body = world.body(a)
    radius = body["vision_radius"]
    occupied = {other.position for other in world.animals.values() if other.id != a.id}
    features = [
        a.energy / body["max_energy"],
        a.health / body["max_health"],
        *DIRECTIONS[a.heading],
    ]
    for offset in range(4):
        dx, dy = DIRECTIONS[(a.heading + offset) % 4]
        p = a.x + dx, a.y + dy
        features.append(float(world.blocked(p) or p in occupied))

    def nearest(points):
        best, distance = None, float("inf")
        for p in points:
            d = math.hypot(p[0] - a.x, p[1] - a.y)
            if d <= radius and d < distance and world.visible(a.position, p):
                best, distance = p, d
        return (
            [0.0, 0.0, 0.0, 0.0]
            if best is None
            else [
                (best[0] - a.x) / radius,
                (best[1] - a.y) / radius,
                distance / radius,
                1.0,
            ]
        )

    features += nearest(p for p, r in world.resources.items() if r.quantity > 0.001)
    features += nearest(
        b.position
        for b in world.animals.values()
        if b.id != a.id and world.diet(b) == "resource"
    )
    features += nearest(
        b.position
        for b in world.animals.values()
        if b.id != a.id and world.diet(b) == "prey"
    )
    return tuple(features)


class EfficiencyTests(unittest.TestCase):
    def test_shared_perception_matches_reference_and_never_goes_stale(self):
        c = config()
        c["world"].update(width=40, height=28)
        c["resources"]["patch_count"] = 80
        c["species"]["prey"]["initial_population"] = 48
        c["species"]["predator"]["initial_population"] = 12
        for seed in (1, 42, 99):
            w = World(c, seed)
            for _ in range(4):
                self.assertEqual(
                    w.observe_all(),
                    {i: reference_observation(w, a) for i, a in w.animals.items()},
                )
                w.step({i: "move_forward" for i in w.animals})
            # No persistent perception cache may survive manual world changes.
            for patch in w.resources.values():
                patch.quantity = 0
            w.obstacles.clear()
            self.assertEqual(
                w.observe_all(),
                {i: reference_observation(w, a) for i, a in w.animals.items()},
            )

    def test_native_forward_has_identical_arithmetic(self):
        c = config()
        b = Brain(c["brains"]["small_q"], ["rest", "eat"], 14, 15)
        for observation in (
            (0.0,) * 20,
            tuple(i / 20 for i in range(20)),
            tuple((-1) ** i / (i + 1) for i in range(20)),
        ):
            for weights in (b.weights, b.target):
                h = [
                    math.tanh(
                        weights["b1"][j]
                        + sum(
                            x * weights["w1"][j * 20 + i]
                            for i, x in enumerate(observation)
                        )
                    )
                    for j in range(b.hidden)
                ]
                q = [
                    weights["b2"][a]
                    + sum(x * weights["w2"][a * b.hidden + j] for j, x in enumerate(h))
                    for a in range(len(b.actions))
                ]
                self.assertEqual(b.forward(observation, weights), (h, q))

    def test_world_snapshots_are_detached(self):
        e = Experiment(config())
        snapshot = e.world.snapshot()
        snapshot["animals"][0]["health"] = -1
        snapshot["resources"][0]["quantity"] = -1
        self.assertGreater(e.world.animals[0].health, 0)
        self.assertGreaterEqual(next(iter(e.world.resources.values())).quantity, 0)

    def test_streamed_checkpoint_equals_snapshot_and_resumes_exactly(self):
        a = Experiment(config())
        for _ in range(7):
            a.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            a.write_checkpoint(path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved, json.loads(json.dumps(a.checkpoint())))
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            b = Experiment.restore(saved)
        for _ in range(20):
            a.step()
            b.step()
        self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_failed_checkpoint_preserves_previous_file(self):
        a = Experiment(config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            a.write_checkpoint(path)
            before = path.read_bytes()
            a.brains[0].weights["w1"][0] = float("nan")
            with self.assertRaises(ValueError):
                a.write_checkpoint(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_checkpoint_does_not_expose_mutable_model_arrays(self):
        e = Experiment(config())
        state = e.checkpoint()
        state["brains"]["0"]["weights"]["w1"][0] += 10
        state["brains"]["0"]["target"]["w2"][0] += 10
        self.assertNotEqual(state["brains"]["0"]["weights"], e.brains[0].weights)
        self.assertNotEqual(state["brains"]["0"]["target"], e.brains[0].target)
