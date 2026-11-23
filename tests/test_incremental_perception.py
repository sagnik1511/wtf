"""Incremental observations must match a fresh scan of the current world."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import random
import unittest
from unittest.mock import patch

from wtf.simulation import perception
from wtf.simulation.perception import SpatialIndex
from wtf.simulation.world import Animal, Resource, World
from tests.test_efficiency import reference_observation
from tests.test_simulation import config


def world():
    c = config()
    c["world"].update(width=64, height=48)
    c["resources"]["patch_count"] = 120
    c["species"]["prey"]["initial_population"] = 48
    c["species"]["predator"]["initial_population"] = 12
    return World(c, 42)


class IncrementalPerceptionTests(unittest.TestCase):
    def test_shifting_window_visits_only_entering_and_leaving_strips(self):
        index = SpatialIndex()
        points = [
            (i, (x, y))
            for i, (x, y) in enumerate(
                (x, y) for x in range(0, 64, 8) for y in range(0, 64, 8)
            )
        ]
        observer = Animal(999, "prey", 1, 16, 0, 60, 30)
        index.sync(points, {observer.id})
        index.nearest(observer, 6, lambda a, b: True)
        _old_bounds, old_cells = index.windows[observer.id]
        references = old_cells.copy()
        observer.x += 1
        visited = []
        original = perception._cells

        def track(bounds):
            for cell in original(bounds):
                visited.append(cell)
                yield cell

        with patch.object(perception, "_cells", track):
            index.nearest(observer, 6, lambda a, b: True)
        _bounds, cells = index.windows[observer.id]
        self.assertIs(cells, old_cells)
        self.assertEqual(len(cells), 6)
        self.assertEqual(len(visited), 2)  # Four interior buckets were reused.
        for cell, bucket in references.items():
            self.assertIs(cells[cell], bucket)

    def test_stationary_observer_reuses_only_unchanged_geometry(self):
        w = world()
        a = w.animals[0]
        a.x, a.y = 7, 7
        w.animals = {a.id: a}
        w.obstacles.clear()
        w.resources[(9, 7)] = Resource(9, 7, 10)
        w.config["resources"]["growth_per_step"] = 0
        with patch.object(w, "visible", wraps=w.visible) as visible:
            first = w.observe_all()[a.id]
            self.assertGreater(visible.call_count, 0)
            visible.reset_mock()
            w.step({a.id: "rest"})
            second = w.observe_all()[a.id]
            self.assertEqual(visible.call_count, 0)
            self.assertNotEqual(first[0], second[0])  # Energy is always current.
            self.assertEqual(first[8:], second[8:])
            # Removing the nearest food must refresh target selection; cached
            # obstacle-only rays may still satisfy the new search without tracing.
            previous_key = w._perception[0].results[a.id][0]
            w.resources[(9, 7)].quantity = 0
            current = w.observe_all()[a.id]
            self.assertNotEqual(w._perception[0].results[a.id][0], previous_key)
            self.assertNotEqual(second[8:12], current[8:12])
        self.assertEqual(current, reference_observation(w, a))

    def test_overlap_updates_birth_death_food_obstacles_and_vision(self):

        # Local fixture/geometry names describe a separate scope and do not consume
        # the outer binding.
        # pylint: disable=redefined-outer-name
        # Keep setup, action and assertions together as one regression scenario.
        # pylint: disable=too-many-branches

        w = world()
        rng = random.Random(72)
        for spec in w.config["species"].values():
            spec["reproduction"].update(
                enabled=True,
                min_age_steps=3,
                cooldown_steps=4,
                min_energy=10,
                energy_cost=5,
                max_population=80,
            )
        for tick in range(30):
            if tick == 6:
                for patch in w.resources.values():
                    patch.quantity = 0
                w.config["resources"]["growth_per_step"] = 0
            if tick == 9:
                for patch in w.resources.values():
                    patch.quantity = 10
            if tick == 12:
                w.obstacles.clear()
            if tick == 15:
                w = World.restore(w.checkpoint())  # Caches are derived, not saved.
            if tick == 18:
                for spec in w.config["species"].values():
                    spec["body"]["vision_radius"] = 1
            if tick == 21:
                for spec in w.config["species"].values():
                    spec["body"]["vision_radius"] = 64
            if tick == 24:
                for spec in w.config["species"].values():
                    spec["body"]["vision_radius"] = 6
            observed = w.observe_all()
            self.assertEqual(
                observed, {i: reference_observation(w, a) for i, a in w.animals.items()}
            )
            actions = {
                i: rng.choice(w.config["species"][a.species]["allowed_actions"])
                for i, a in w.animals.items()
            }
            w.step(actions)
        self.assertGreater(sum(w.births.values()), 0)

    def test_bucket_changes_in_overlap_invalidate_stationary_result(self):
        index = SpatialIndex()
        points = [
            (i, (x, y))
            for i, (x, y) in enumerate(
                (x, y) for x in range(0, 64, 8) for y in range(0, 64, 8)
            )
        ]
        observer = Animal(999, "prey", 7, 7, 0, 60, 30)
        points += [(1000, (8, 7))]
        index.sync(points, {observer.id})
        first = index.nearest(observer, 6, lambda a, b: True)
        self.assertGreater(first[0], 0)
        points[-1] = (1000, (6, 7))
        index.sync(points, {observer.id})
        changed = index.nearest(observer, 6, lambda a, b: True)
        self.assertLess(changed[0], 0)
        # Reordering equidistant targets must also invalidate cached ties.
        points += [(1001, (8, 7))]
        index.sync(points, {observer.id})
        self.assertLess(index.nearest(observer, 6, lambda a, b: True)[0], 0)
        points[-2:] = reversed(points[-2:])
        index.sync(points, {observer.id})
        self.assertGreater(index.nearest(observer, 6, lambda a, b: True)[0], 0)

    def test_distant_changes_preserve_stationary_nearest_result(self):
        index = SpatialIndex()
        points = [
            (i, (x, y))
            for i, (x, y) in enumerate(
                (x, y) for x in range(0, 64, 8) for y in range(0, 64, 8)
            )
        ]
        observer = Animal(999, "prey", 7, 7, 0, 60, 30)
        index.sync(points, {observer.id})
        calls = []

        def visible(start, end):

            # The callback keeps its caller's keyword signature; this branch does not
            # consume every input.
            # pylint: disable=unused-argument

            calls.append(end)
            return True

        first = index.nearest(observer, 6, visible)
        self.assertGreater(len(calls), 0)
        calls.clear()
        points[-1] = (63, (55, 56))
        index.sync(points, {observer.id})
        self.assertEqual(index.nearest(observer, 6, visible), first)
        self.assertEqual(calls, [])

    def test_dead_observer_caches_are_released(self):
        w = world()
        w.observe_all()
        animal_id = next(iter(w.animals))
        del w.animals[animal_id]
        w.observe_all()
        for index in w._perception:
            self.assertNotIn(animal_id, index.windows)
            self.assertNotIn(animal_id, index.results)
        self.assertNotIn("_perception", w.checkpoint())


if __name__ == "__main__":
    unittest.main()
