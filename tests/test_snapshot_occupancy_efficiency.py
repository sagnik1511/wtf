"""Exact snapshot reuse and phase-local competition occupancy ownership."""

import copy
import gc
import math
import random
import unittest
from unittest.mock import patch
import weakref

from tests.test_mutation import KINDS, environment
from tests.test_competition import arena
from tests.test_simulation import animal
from wtf.experiment.runner import Experiment
from wtf.learning.brains import model_snapshot, create_brain
from wtf.native.numeric import Float32, precision_scope
from wtf.simulation.competition import resolve
from wtf.storage.library import ChampionBook


def first_buffer(brain):
    layer = (
        brain.weights
        if brain.config["type"] == "q_network"
        else brain.network.layers[0]
    )
    return next(iter(layer.values()))


class SnapshotReuseTests(unittest.TestCase):
    def test_all_controllers_reuse_exact_weights_but_refresh_metadata_and_detect_edits(
        self,
    ):
        for kind in KINDS:
            with self.subTest(kind=kind):
                e = Experiment(environment(kind, 0))
                a = next(iter(e.world.animals.values()))
                b = e.brains[a.id]
                book = ChampionBook(e.config)
                with patch(
                    "wtf.storage.library.model_snapshot", wraps=model_snapshot
                ) as capture:
                    book.consider([a], e.brains, {"tick": 0})
                    old = book.entries[a.species][a.sex]["lifespan"]["model"]
                    a.age += 1
                    book.consider([a], e.brains, {"tick": 1})
                    entry = book.entries[a.species][a.sex]["lifespan"]
                    self.assertIs(entry["model"], old)
                    self.assertEqual(entry["source"]["tick"], 1)
                    self.assertEqual(entry["rank"][0], a.age)
                    self.assertEqual(capture.call_count, 1)
                    first_buffer(b)[0] += 0.125  # No optimizer-counter change.
                    a.age += 1
                    book.consider([a], e.brains, {"tick": 2})
                    self.assertEqual(capture.call_count, 2)
                    self.assertEqual(
                        book.entries[a.species][a.sex]["lifespan"]["model"],
                        model_snapshot(b),
                    )
                    self.assertNotEqual(old, model_snapshot(b))
                    self.assertEqual(len(book._snapshot_cache), 1)

    def test_optimizer_changes_skip_comparison_buffers(self):
        e = Experiment(environment("ppo", 0))
        a = next(iter(e.world.animals.values()))
        b = e.brains[a.id]
        book = ChampionBook(e.config)
        book.consider([a], e.brains, {"tick": 0})
        first_buffer(b)[0] += 0.125
        b.updates += 1
        a.age += 1
        with patch.object(
            Float32, "tobytes", side_effect=AssertionError("unneeded comparison")
        ):
            book.consider([a], e.brains, {"tick": 1})
        self.assertIsNone(book._snapshot_cache[id(b)][2])
        self.assertEqual(
            book.entries[a.species][a.sex]["lifespan"]["model"], model_snapshot(b)
        )

    def test_signed_zero_and_signature_changes_invalidate(self):
        e = Experiment(environment("ppo", 0))
        a = next(iter(e.world.animals.values()))
        b = e.brains[a.id]
        book = ChampionBook(e.config)
        first_buffer(b)[0] = 0.0
        book.consider([a], e.brains, {"tick": 0})
        old = book.entries[a.species][a.sex]["lifespan"]["model"]
        first_buffer(b)[0] = -0.0
        a.age += 1
        book.consider([a], e.brains, {"tick": 1})
        new = book.entries[a.species][a.sex]["lifespan"]["model"]
        self.assertIsNot(old, new)
        self.assertEqual(
            math.copysign(1, new["parameters"][0][next(iter(b.network.layers[0]))][0]),
            -1,
        )
        b.actions = list(reversed(b.actions))
        a.age += 1
        book.consider([a], e.brains, {"tick": 2})
        self.assertEqual(
            book.entries[a.species][a.sex]["lifespan"]["model"]["signature"]["actions"],
            b.actions,
        )

    def test_cache_does_not_retain_controllers_and_prunes_non_winners(self):
        e = Experiment(environment("ppo", 0))
        a = next(iter(e.world.animals.values()))
        book = ChampionBook(e.config)
        book.consider([a], e.brains, {"tick": 0})
        controller = weakref.ref(e.brains[a.id])
        del e.brains[a.id]
        gc.collect()
        self.assertIsNone(controller())
        book.consider([], {}, {"tick": 1})
        self.assertEqual(book._snapshot_cache, {})

    def test_legacy_models_use_uncached_path(self):
        e = Experiment(environment("ppo", 0))
        a = next(iter(e.world.animals.values()))
        original = e.brains[a.id]
        with precision_scope("float64"):
            e.brains[a.id] = create_brain(original.config, original.actions, 0, 0)
        book = ChampionBook(e.config)
        with patch(
            "wtf.storage.library.model_snapshot", wraps=model_snapshot
        ) as capture:
            book.consider([a], e.brains, {"tick": 0})
            a.age += 1
            book.consider([a], e.brains, {"tick": 1})
            self.assertEqual(capture.call_count, 2)
            self.assertEqual(book._snapshot_cache, {})

    def test_cached_and_uncached_experiments_have_identical_checkpoints(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                e = Experiment(environment(kind, 0))
                reference = Experiment.restore(e.checkpoint())
                for _ in range(7):
                    e.step()
                    with patch.object(
                        ChampionBook,
                        "_snapshot",
                        lambda self, brain, retained: model_snapshot(brain),
                    ):
                        reference.step()
                    self.assertEqual(e.checkpoint(), reference.checkpoint())


class OccupancyReuseTests(unittest.TestCase):
    def test_many_displacements_build_one_shared_live_occupancy(self):
        class CountedAnimals(dict):
            scans = 0

            def values(self):
                self.scans += 1
                return super().values()

        w = arena()
        w.width = 20
        w.obstacles.clear()
        w.resources.clear()
        w.animals = CountedAnimals(
            {
                i: animal(
                    i,
                    "prey",
                    2 + i // 2 * 5 + i % 2,
                    2,
                    heading=1 if i % 2 == 0 else 3,
                    energy=50 if i % 2 == 0 else 15,
                    health=30 if i % 2 == 0 else 20,
                )
                for i in range(6)
            }
        )
        events = []
        resolve(
            w, {i: "challenge" if i % 2 == 0 else "rest" for i in w.animals}, events
        )
        self.assertEqual(sum(e.kind == "contest_moved" for e in events), 3)
        self.assertEqual(w.animals.scans, 3)  # Targets, challengers, one live set.
        self.assertEqual(len({a.position for a in w.animals.values()}), 6)

    def test_later_retreat_can_use_a_cell_vacated_by_an_earlier_contest(self):
        w = arena()
        w.width = w.height = 9
        w.animals.update(
            {
                2: animal(2, "prey", 2, 4, heading=0, energy=50, health=30),
                3: animal(3, "prey", 2, 3, heading=2, energy=15, health=20),
            }
        )
        w.step({0: "challenge", 1: "rest", 2: "challenge", 3: "rest"})
        self.assertEqual(w.animals[3].position, (2, 2))
        self.assertEqual(w.animals[2].position, (2, 3))

    def test_later_retreat_can_use_a_cell_freed_by_a_contest_death(self):
        w = arena()
        w.width = w.height = 9
        w.animals.update(
            {
                2: animal(2, "prey", 6, 3, heading=0, energy=10, health=3),
                3: animal(3, "prey", 6, 2, heading=2, energy=60, health=30),
                4: animal(4, "prey", 6, 5, heading=0, energy=50, health=30),
                5: animal(5, "prey", 6, 4, heading=2, energy=15, health=20),
            }
        )
        w.step({i: "challenge" if i % 2 == 0 else "rest" for i in w.animals})
        self.assertNotIn(2, w.animals)
        self.assertEqual(w.animals[5].position, (6, 3))
        self.assertEqual(w.animals[4].position, (6, 4))

    def test_phase_occupancy_matches_rebuilt_reference_after_moves_and_deaths(self):
        # Rebuild before each advance, reproducing the original helper's ownership.
        from wtf.simulation.competition import _advance

        def rebuilt(world, winner, loser, events, Event, occupied):
            return _advance(
                world,
                winner,
                loser,
                events,
                Event,
                {a.position for a in world.animals.values() if a.health > 0},
            )

        for seed in range(40):
            w = arena()
            w.width = w.height = 9
            rng = random.Random(seed)
            cells = rng.sample([(x, y) for x in range(1, 8) for y in range(1, 8)], 20)
            w.obstacles = set(cells[14:])
            w.animals = {
                i: animal(
                    i,
                    "prey",
                    *p,
                    heading=rng.randrange(4),
                    energy=rng.choice((15, 50)),
                    health=rng.choice((3, 20, 30)),
                )
                for i, p in enumerate(cells[:14])
            }
            reference = copy.deepcopy(w)
            for _ in range(8):
                actions = {
                    i: rng.choice(("challenge", "rest", "move_forward"))
                    for i in w.animals
                }
                w.step(actions)
                with patch("wtf.simulation.competition._advance", side_effect=rebuilt):
                    reference.step(actions)
                self.assertEqual(w.checkpoint(), reference.checkpoint())
