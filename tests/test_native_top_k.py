"""Top-k selector parity, dispatch safety and exact training/replay continuation."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from wtf.config import load
from wtf.experiment.runner import Experiment
from wtf.native import kernels
from wtf.simulation.world import World
from wtf.storage.recording import Recorder
from tests.test_mutation import KINDS, environment

NATIVE = kernels.native
SUPPORTED = kernels.available("nearest_many")


def reference_backend():
    available = kernels.available
    return patch.object(
        kernels,
        "available",
        new=lambda name: name != "nearest_many" and available(name),
    )


@unittest.skipUnless(SUPPORTED, "Build the optional top-k kernel")
class NativeTopKTests(unittest.TestCase):
    def test_randomized_candidate_order_ties_occlusion_water_and_padding(self):

        # Keep setup, action and assertions together as one regression scenario.
        # pylint: disable=too-many-locals
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        rng = random.Random(894)
        width, height = 31, 23
        for trial in range(100):
            blocked = bytes(rng.randrange(13) == 0 for _ in range(width * height))
            water = bytes(rng.randrange(4) == 0 for _ in range(width * height))
            x, y = rng.randrange(width), rng.randrange(height)
            base, bonus = rng.randrange(1, 9), rng.randrange(5)
            radius, k = base + bonus, rng.randrange(1, 17)
            points = [
                (i, i, rng.randrange(width), rng.randrange(height)) for i in range(180)
            ]
            rng.shuffle(points)
            expected = []
            candidates = sorted(
                ((tx - x) ** 2 + (ty - y) ** 2, order, tx, ty)
                for order, identity, tx, ty in points
                if identity != 7 and (tx - x) ** 2 + (ty - y) ** 2 <= radius**2
            )
            for squared, _, tx, ty in candidates:
                visible, wet = NATIVE.ray(x, y, tx, ty, blocked, water, width, height)
                if visible and (not bonus or squared <= (base + min(bonus, wet)) ** 2):
                    expected.append((tx - x, ty - y))
                    if len(expected) == k:
                        break
            self.assertEqual(
                NATIVE.nearest_many(
                    iter(points),
                    x,
                    y,
                    7,
                    radius,
                    k,
                    blocked,
                    water,
                    width,
                    height,
                    base,
                    bonus,
                ),
                expected,
                trial,
            )
        points = [(3, None, 4, 3), (1, None, 2, 3), (2, None, 3, 4), (0, None, 3, 3)]
        self.assertEqual(
            NATIVE.nearest_many(
                points, 3, 3, 99, 4, 16, bytes(49), bytes(49), 7, 7, 4, 0
            ),
            [(0, 0), (-1, 0), (0, 1), (1, 0)],
        )

    def test_invalid_arguments_and_iterator_exceptions_propagate(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        valid = [[], 2, 2, 0, 4, 3, bytes(25), bytes(25), 5, 5, 4, 0]
        for index, value in (
            (0, [(0, 1, 9, 0)]),
            (0, [(0, 1, 2)]),
            (0, [(0, 1, 1.5, 2)]),
            (1, -1),
            (4, 0),
            (5, 0),
            (5, 17),
            (6, b""),
            (8, 4097),
            (10, -1),
            (11, 4097),
        ):
            args = valid.copy()
            args[index] = value
            with (
                self.subTest(index=index, value=value),
                self.assertRaises((TypeError, ValueError, OverflowError)),
            ):
                NATIVE.nearest_many(*args)

        def broken():
            yield (0, 1, 2, 2)
            raise RuntimeError("iterator stopped")

        args = valid.copy()
        args[0] = broken()
        with self.assertRaisesRegex(RuntimeError, "iterator stopped"):
            NATIVE.nearest_many(*args)
        self.assertEqual(NATIVE.nearest_many(*valid), [])

    def test_native_dispatch_custom_callback_fallback_and_missing_symbol(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        c = load("environments/competition.json")
        w = World(c, 100)
        w.observe_all()
        a = next(iter(w.animals.values()))
        index = w._perception[0]
        index.results.clear()
        with patch.object(
            NATIVE, "nearest_many", wraps=NATIVE.nearest_many
        ) as selected:
            native = index.nearest_many(a, 8, w._perception_visible, 3)
            self.assertEqual(selected.call_count, 1)
        index.results.clear()
        with (
            patch.object(
                NATIVE, "nearest_many", side_effect=AssertionError("k=1 changed")
            ),
            patch.object(NATIVE, "nearest", wraps=NATIVE.nearest) as nearest,
        ):
            self.assertEqual(
                index.nearest_many(a, 8, w._perception_visible, 1), native[:4]
            )
            self.assertEqual(nearest.call_count, 1)
        index.results.clear()
        calls = []

        def custom(start, end):
            calls.append(end)
            return w.visible(start, end)

        with patch.object(
            NATIVE,
            "nearest_many",
            side_effect=AssertionError("custom callback dispatched"),
        ):
            self.assertEqual(index.nearest_many(a, 8, custom, 3), native)
        self.assertTrue(calls)
        index.results.clear()

        class OldExtension:

            # This record/owner deliberately keeps its related fields and lifecycle
            # operations together without artificial wrappers.
            # pylint: disable=too-few-public-methods

            api_version = 2

        with patch.object(kernels, "native", OldExtension()):
            self.assertEqual(index.nearest_many(a, 8, w._perception_visible, 3), native)

    def test_dynamic_worlds_day_night_terrain_and_restore_match_reference(self):
        for preset in (
            "competition",
            "day_night",
            "highlands",
            "river_crossing_strong_predator_deep",
        ):
            c = load(f"environments/{preset}.json")
            for b in c["brains"].values():
                b["observation_top_k"] = 4
            w = World(c, 100)
            reference = World.restore(w.checkpoint())
            for tick in (0, 80, 200, 300, 400):
                w.tick = reference.tick = tick
                for world in (w, reference):
                    if tick == 80:
                        world.obstacles.add((1, 1))
                    if tick == 200:
                        world.obstacles.discard((1, 1))
                        if world.resources:
                            next(iter(world.resources.values())).quantity = 0
                    if tick == 300 and len(world.animals) > 1:
                        world.animals.pop(next(iter(world.animals)))
                actual = w.observe_all()
                with reference_backend():
                    expected = reference.observe_all()
                self.assertEqual(actual, expected, (preset, tick))

    def test_all_controllers_and_recorded_replay_match_reference(self):

        # This self-contained regression intentionally repeats reference/setup data.
        # pylint: disable=duplicate-code

        for kind in KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                c = environment(kind, 0)
                for i, (name, species) in enumerate(c["species"].items()):
                    b = copy.deepcopy(c["brains"][species["brain"]])
                    b["observation_top_k"] = 3 + i
                    c["brains"][name + "_top_k"] = b
                    species["brain"] = name + "_top_k"
                c["recording"] = {
                    "enabled": True,
                    "frames_per_chunk": 2,
                    "frame_every_steps": 1,
                    "diagnostics_every_steps": 1,
                    "record_activations": True,
                    "compression": "xz",
                    "compression_level": 1,
                    "encoding": "binary",
                }
                e = Experiment(c)
                with reference_backend():
                    r = Experiment(c)
                recorders = []
                for name, experiment in (("native", e), ("reference", r)):
                    recorder = Recorder(
                        Path(folder) / name, experiment.config, experiment.mode
                    )
                    recorder.begin_run(experiment)
                    recorder.capture(experiment)
                    recorders.append(recorder)
                for _ in range(7):
                    e.step()
                    with reference_backend():
                        r.step()
                    self.assertEqual(e.checkpoint(), r.checkpoint())
                    for recorder, experiment in zip(recorders, (e, r)):
                        recorder.capture(experiment, force=experiment.finished_run)
                for recorder in recorders:
                    recorder.close()
                left = sorted((Path(folder) / "native").rglob("*.bin.xz"))
                right = sorted((Path(folder) / "reference").rglob("*.bin.xz"))
                self.assertTrue(left)
                self.assertEqual(
                    [p.read_bytes() for p in left], [p.read_bytes() for p in right]
                )
