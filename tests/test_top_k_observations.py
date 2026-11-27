"""Visible multi-target slots, dynamic invalidation and all learner contracts."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from wtf.learning.brain import observation_inputs
from wtf.learning.brains import load_model_weights, model_signature, model_snapshot
from wtf.config import load, validate
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.recording import Recorder
from wtf.storage.replay_io import load_chunk
from wtf.experiment.runner import Experiment
from wtf.simulation.world import Animal, Resource, World
from tests.test_mutation import KINDS, environment
from tests.reproduction_support import force_reproduction
from tests.test_simulation import config
from tests.test_water_vision import setup


def reference_targets(world, animal, k):
    """Full uncached scan: filter sight before selecting distance-ordered slots."""

    # Keep setup, action and assertions together as one regression scenario.
    # pylint: disable=too-many-locals

    body = world.body(animal)
    base = body["vision_radius"]
    bonus = (
        body.get("water_vision_bonus", 0)
        if world.geography and world.geography.water
        else 0
    )
    radius = base + bonus
    groups = [
        [p for p, r in world.resources.items() if r.quantity > 0.001],
        [
            a.position
            for a in world.animals.values()
            if a.id != animal.id and world.diet(a) == "resource"
        ],
        [
            a.position
            for a in world.animals.values()
            if a.id != animal.id and world.diet(a) == "prey"
        ],
    ]
    values = []
    for points in groups:
        visible = []
        for order, p in enumerate(points):
            dx, dy = p[0] - animal.x, p[1] - animal.y
            distance = math.hypot(dx, dy)
            allowed = (
                base + min(bonus, world._water_sight_cells(animal.position, p))
                if bonus
                else base
            )
            if distance <= allowed and world.visible(animal.position, p):
                visible.append((distance, order, dx, dy))
        visible.sort()
        for distance, order, dx, dy in visible[:k]:
            values.extend((dx / radius, dy / radius, distance / radius, 1.0))
        values.extend([0.0] * (4 * (k - min(k, len(visible)))))
    return tuple(values)


class TopKObservationTests(unittest.TestCase):
    def test_dimensions_defaults_validation_and_model_signatures(self):
        for version, base in ((1, 20), (2, 32), (3, 42), (4, 45), (5, 48)):
            for k in (1, 4, 16):
                c = config()
                for b in c["brains"].values():
                    b.update(observation_version=version, observation_top_k=k)
                e = Experiment(c)
                for a in e.world.animals.values():
                    b = c["brains"][c["species"][a.species]["brain"]]
                    self.assertEqual(observation_inputs(b), base + 12 * (k - 1))
                    self.assertEqual(len(e.observations[a.id]), observation_inputs(b))
        c = config()
        name = next(iter(c["brains"]))
        original = copy.deepcopy(c["brains"][name])
        actions = c["species"]["prey"]["allowed_actions"]
        c["brains"][name]["observation_top_k"] = 1
        self.assertEqual(
            model_signature(original, actions),
            model_signature(c["brains"][name], actions),
        )
        for bad in (0, 17, 1.5, True, "4"):
            c["brains"][name]["observation_top_k"] = bad
            with self.assertRaisesRegex(ValueError, "observation_top_k"):
                validate(c)

    def test_sorted_slots_ties_self_exclusion_presence_and_padding(self):
        c = config()
        for b in c["brains"].values():
            b["observation_top_k"] = 4
        w = World(c, 42)
        a = Animal(0, "prey", 3, 3, 0, 40, 30)
        w.animals = {
            0: a,
            2: Animal(2, "prey", 4, 3, 0, 40, 30),
            1: Animal(1, "prey", 2, 3, 0, 40, 30),
        }
        w.resources = {(3, 3): Resource(3, 3, 1), (4, 4): Resource(4, 4, 0)}
        w.obstacles.clear()
        o = w.observe_all()[0]
        self.assertEqual(o[8:56], reference_targets(w, a, 4))
        self.assertEqual(o[8:12], (0, 0, 0, 1))  # Food at current cell is present.
        self.assertGreater(o[24], 0)  # Equal-distance tie retains insertion order.
        self.assertLess(o[28], 0)
        self.assertEqual(o[32:56], (0.0,) * 24)

    def test_stationary_observer_sees_moves_births_deaths_food_and_walls(self):
        c = config()
        c["world"].update(width=60, height=40)
        for b in c["brains"].values():
            b["observation_top_k"] = 4
        w = World(c, 42)
        w.obstacles.clear()
        w.animals = {
            i: Animal(
                i,
                "prey" if i % 2 == 0 else "predator",
                (i % 10) * 5,
                (i // 10) * 5,
                0,
                40,
                30,
            )
            for i in range(80)
        }
        w.resources = {
            (x, y): Resource(x, y, 1) for x in range(1, 60, 5) for y in range(1, 40, 5)
        }
        observer = w.animals[0]
        observer.x, observer.y = 10, 10
        rng = random.Random(25)
        for tick in range(25):
            target = w.animals[1]
            target.x, target.y = rng.randrange(8, 15), rng.randrange(8, 15)
            w.animals[100 + tick] = Animal(100 + tick, "prey", 11, 10, 0, 40, 30)
            if tick:
                del w.animals[99 + tick]
            w.resources[(10, 11)] = Resource(10, 11, tick % 2)
            w.obstacles.symmetric_difference_update({(11, 10)})
            o = w.observe_all()[0]
            self.assertEqual(o[8:56], reference_targets(w, observer, 4))
            # The unchanged query reuses one entry; derived cache never grows by time.
            with patch.object(
                w, "_perception_visible", side_effect=AssertionError("cache miss")
            ):
                self.assertEqual(w.observe(observer), o)
            self.assertLessEqual(len(w._perception[0].results), len(w.animals))
        self.assertNotIn("_perception", w.checkpoint())

    def test_water_direction_walls_and_k_range_key_changes(self):
        w = setup()
        for b in w.config["brains"].values():
            b["observation_top_k"] = 4
        a = w.animals[0]
        w.animals[2] = Animal(2, "predator", 1, 6, 0, 40, 30)
        for obstacles in (set(), {(3, 3)}, set()):
            w.obstacles = obstacles
            self.assertEqual(w.observe_all()[0][8:56], reference_targets(w, a, 4))
        template = w.config["brains"][w.config["species"][a.species]["brain"]]
        template["observation_top_k"] = 2
        self.assertEqual(w.observe_all()[0][8:32], reference_targets(w, a, 2))
        w.body(a).update(vision_radius=4, water_vision_bonus=1)
        self.assertEqual(w.observe_all()[0][8:32], reference_targets(w, a, 2))
        template["observation_top_k"] = 1
        old = w.observe_all()[0]
        template["observation_top_k"] = 4
        new = w.observe_all()[0]
        for group in range(3):
            self.assertEqual(
                old[8 + 4 * group : 12 + 4 * group],
                new[8 + 16 * group : 12 + 16 * group],
            )

    def test_top_k_model_cannot_load_nearest_only_weights(self):
        c = environment("ppo", 0)
        old = Experiment(c)
        for b in c["brains"].values():
            b["observation_top_k"] = 4
        new = Experiment(c)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_model_weights(new.brains[0], model_snapshot(old.brains[0]))

    def test_all_algorithms_train_and_resume_with_mixed_k_and_versions(self):

        # This self-contained regression intentionally repeats reference/setup data.
        # pylint: disable=duplicate-code

        for kind in KINDS:
            for version in (1, 2, 3, 4, 5):
                with self.subTest(kind=kind, version=version):
                    c = environment(kind, 0)
                    # Species can have different k even when they formerly shared a template.
                    for i, (species, spec) in enumerate(c["species"].items()):
                        b = copy.deepcopy(c["brains"][spec["brain"]])
                        b.update(observation_top_k=2 + i, observation_version=version)
                        c["brains"][species + "_top_k"] = b
                        spec["brain"] = species + "_top_k"
                    e = Experiment(c)
                    for _ in range(4):
                        e.step()
                    r = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
                    while not e.finished_run:
                        e.step()
                        r.step()
                    self.assertEqual(e.checkpoint(), r.checkpoint())
                    e.next_run()
                    r.next_run()
                    e.step()
                    r.step()
                    self.assertEqual(e.checkpoint(), r.checkpoint())

    def test_all_algorithms_mix_top_k_parent_weights_at_birth(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = environment(kind, 0)
                for b in c["brains"].values():
                    b.update(observation_top_k=4, observation_version=3)
                e = force_reproduction(Experiment(c))
                e.step()
                children = [a for a in e.world.animals.values() if a.parent_id >= 0]
                self.assertTrue(children)
                self.assertTrue(all(len(e.observations[a.id]) == 78 for a in children))
                for a in children:
                    self.assertEqual(
                        model_snapshot(e.brains[a.id])["signature"]["inputs"], 78
                    )
                with tempfile.TemporaryDirectory() as folder:
                    checkpoint = Path(folder) / "checkpoint.bin.xz"
                    e.write_checkpoint(checkpoint)
                    r = Experiment.restore(read_checkpoint(checkpoint))
                e.step()
                r.step()
                self.assertEqual(e.checkpoint(), r.checkpoint())

    def test_new_river_preset_records_binary_replay_transparently(self):
        c = load("environments/river_crossing_strong_predator.json")
        c["seed"] = 100
        c["experiment"].update(runs=1, world_seed_policy="increment")
        c["world"]["max_steps"] = 4
        c["recording"]["frames_per_chunk"] = 2
        expected, recorded = Experiment(c), Experiment(c)
        self.assertTrue(all(len(o) == 130 for o in recorded.observations.values()))
        with tempfile.TemporaryDirectory() as folder:
            writer = Recorder(folder, c, "train")
            writer.begin_run(recorded)
            while not recorded.done:
                expected.step()
                recorded.step()
                writer.capture(recorded, force=recorded.finished_run)
            writer.end_run(recorded)
            writer.close()
            paths = sorted(Path(folder).rglob("run_*_chunk_*.bin.xz"))
            self.assertGreater(len(paths), 1)
            for path in paths:
                self.assertTrue(load_chunk(path))
        self.assertEqual(expected.checkpoint(), recorded.checkpoint())


if __name__ == "__main__":
    unittest.main()
