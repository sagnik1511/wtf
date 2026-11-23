"""Exact DQN batches and bounded sensory work reuse, including external edits."""

# Numerical/cache fixtures inspect private state to verify independent ownership.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring

from array import array
import copy
import random
import unittest
from unittest.mock import patch

from tests.test_advanced_algorithms import settings
from tests.test_tracking import scene
from tests.test_social_learning import config, controlled
from tests.test_mutation import environment
from wtf.experiment.runner import Experiment
from wtf.learning.brain import Experience, INPUTS
from wtf.learning.brains import create_brain, restore_brain
from wtf.native import kernels
from wtf.native.numeric import precision_scope
from wtf.simulation.social import SocialState


def without(*names):
    available = kernels.available
    return patch.object(
        kernels, "available", lambda name: name not in names and available(name)
    )


class SharedSensoryTests(unittest.TestCase):
    def test_scent_matches_uncached_after_edits_decay_and_deposits(self):
        world = scene(scent=True)
        field = world._tracking.scent
        animal = world.animals[0]
        for mode in ("separated", "blended"):
            template = {
                "scent_observation": {
                    "species": ["prey", "predator"],
                    "radius": 3,
                    "mode": mode,
                }
            }
            for tick in range(5):
                world.tick = tick
                world.animals[1].x = 3 + tick % 3
                world.obstacles = {(3, 1)} if tick % 2 else set()
                field.advance(world)
                phase = field.prepare(world)
                expected = field.sample(world, animal, template)
                self.assertEqual(field.sample(world, animal, template, phase), expected)
                self.assertEqual(field.sample(world, animal, template, phase), expected)
                self.assertTrue(phase)
                self.assertLessEqual(len(field._reachable), 128)
        old = next(iter(field._reachable.values()))
        world.obstacles.add((2, 1))
        phase = field.prepare(world)
        self.assertFalse(field._reachable)
        self.assertEqual(
            field.sample(world, animal, template, phase),
            field.sample(world, animal, template),
        )
        self.assertIsNot(next(iter(field._reachable.values())), old)
        with patch.object(world, "blocked", wraps=world.blocked):
            self.assertIsNone(field.prepare(world))
        self.assertNotIn("_reachable", world.checkpoint()["sensory_state"])

    def test_water_edits_invalidate_reachability_and_restore_is_independent(self):
        world = scene(scent=True)
        field = world._tracking.scent
        world.geography = controlled(config()).geography
        template = {"scent_observation": {"mode": "blended", "radius": 3}}
        animal = world.animals[0]
        field.sample(world, animal, template, field.prepare(world))
        self.assertTrue(field._reachable)
        world.geography.water = {(3, 2): (1, 0, 0)}
        phase = field.prepare(world)
        self.assertFalse(field._reachable)
        self.assertEqual(
            field.sample(world, animal, template, phase),
            field.sample(world, animal, template),
        )
        fresh_world = scene(scent=True)
        self.assertFalse(fresh_world._tracking.scent._reachable)
        self.assertIsNot(fresh_world._tracking.scent._reachable, field._reachable)

    def test_geometry_lru_and_blending_mode_are_separate(self):
        world = scene(scent=True)
        field = world._tracking.scent
        template = {
            "scent_observation": {
                "species": ["prey"],
                "radius": 1,
                "separate_sexes": False,
            }
        }
        phase = field.prepare(world)
        for i in range(150):
            field._cells(world, (i, 0), 1, True)
        self.assertEqual(len(field._reachable), 128)
        # A one-species world has identical group plans for blended and merged-sex
        # observations, but max and capped-sum reductions must remain distinct.
        field.groups = {("prey", "female"): 0, ("prey", "male"): 1}
        field.channels = 2
        field._plans.clear()
        field._reachable.clear()
        animal = world.animals[0]
        cell = (animal.y * field.width + animal.x) * 2
        field.cells = {cell: world.tick - 2, cell + 1: world.tick - 2}
        separated = field.sample(world, animal, template, phase)
        template["scent_observation"]["mode"] = "blended"
        blended = field.sample(world, animal, template, phase)
        self.assertNotEqual(separated, blended)
        self.assertEqual(blended, field.sample(world, animal, template))

    def test_social_bucket_identity_order_replacement_and_within_cell_motion(self):
        world = controlled(config())
        state = world._social
        state.prepare(world)
        buckets = state.buckets
        state.prepare(world)
        self.assertIs(state.buckets, buckets)
        first = next(iter(world.animals.values()))
        first.x += 1
        state.prepare(world)
        if first.x // 8 == (first.x - 1) // 8:
            self.assertIs(state.buckets, buckets)
        world.animals[first.id] = copy.deepcopy(first)
        state.prepare(world)
        self.assertIsNot(state.buckets, buckets)
        self.assertIs(
            next(a for row in state.buckets.values() for a in row if a.id == first.id),
            world.animals[first.id],
        )
        old = state.buckets
        world.animals = dict(reversed(list(world.animals.items())))
        state.prepare(world)
        self.assertIsNot(state.buckets, old)
        for action in ("cross", "birth", "death"):
            old = state.buckets
            if action == "cross":
                world.animals[first.id].x += 8
            elif action == "birth":
                newborn = copy.deepcopy(first)
                newborn.id = max(world.animals) + 1
                world.animals[newborn.id] = newborn
            else:
                world.animals.pop(max(world.animals))
            state.prepare(world)
            self.assertIsNot(state.buckets, old)
        self.assertNotIn("_bucket_members", state.checkpoint())

    def test_social_reuse_matches_rebuild_for_full_transitions(self):
        actual = Experiment(config())
        expected = Experiment(config())
        prepare = SocialState.prepare

        def fresh(state, world):
            state._bucket_members = ()
            prepare(state, world)

        for _ in range(12):
            actual.step()
            with patch.object(SocialState, "prepare", fresh):
                expected.step()
            self.assertEqual(actual.checkpoint(), expected.checkpoint())


@unittest.skipUnless(kernels.available("dqn_batch"), "Rebuild optional native kernels")
class NativeDQNBatchTests(unittest.TestCase):
    def test_batch_matches_scalar_gradients_updates_rng_wrap_and_restore(self):
        rng = random.Random(102)
        for kind in ("dqn", "ddqn"):
            c = settings(kind)
            c["network"]["hidden_layers"] = [7, 5, 3]
            c.update(
                batch_size=4,
                replay_capacity=8,
                learning_starts_after=4,
                target_update_every_training_steps=2,
            )
            batched = create_brain(c, ["rest", "eat", "turn_left"], 1, 2)
            scalar = restore_brain(batched.checkpoint())
            for step in range(22):
                observation = tuple(rng.uniform(-1, 1) for _ in range(INPUTS))
                following = tuple(rng.uniform(-1, 1) for _ in range(INPUTS))
                terminal = step % 5 == 0
                item = Experience(
                    observation,
                    batched.actions[step % 3],
                    rng.uniform(-3, 3),
                    None if terminal else following,
                    terminal,
                    not terminal and step % 7 == 0,
                )
                batched.learn(item)
                with without("dqn_batch"):
                    scalar.learn(item)
                self.assertEqual(batched.checkpoint(), scalar.checkpoint())
                if step == 12:
                    batched = restore_brain(batched.checkpoint())
                    scalar = restore_brain(scalar.checkpoint())

    def test_invalid_late_sample_does_not_mutate_gradients(self):
        # Optional CPython symbols are runtime feature-detected by the test guard.
        # pylint: disable=c-extension-no-member
        brain = create_brain(settings("ddqn"), ["rest", "eat"], 1, 2)
        obs = (0.2,) * INPUTS
        gradients = brain.network.zeros()
        for layer in gradients:
            for values in layer.values():
                values[:] = array("d", [0.3]) * len(values)
        before = copy.deepcopy(gradients)
        for invalid in (
            (obs, 10, 1, obs, False),
            (obs, 0, 1, None, False),
            ((object(),) * INPUTS, 0, 1, obs, False),
        ):
            with self.assertRaises((ValueError, TypeError)):
                kernels.native.dqn_batch(
                    brain.network.layers,
                    brain.target_network.layers,
                    gradients,
                    [(obs, 0, 1, obs, False), invalid],
                    0.9,
                    1,
                    True,
                )
            self.assertEqual(gradients, before)

    def test_legacy_and_custom_hooks_use_reference(self):
        for precision in ("float32", "float64"):
            with precision_scope(precision):
                brain = create_brain(settings("dqn"), ["rest", "eat"], 1, 2)
            self.assertEqual(brain._native_batchable(), precision == "float32")
            with patch.object(brain, "bootstrap", return_value=42):
                self.assertFalse(brain._native_batchable())

    def test_world_checkpoints_match_without_new_native_batches(self):
        for kind in ("dqn", "ddqn"):
            c = environment(kind, 0)
            actual, expected = Experiment(c), Experiment(c)
            for _ in range(16):
                actual.step()
                with without("dqn_batch", "mlp_predict_many"):
                    expected.step()
                self.assertEqual(actual.checkpoint(), expected.checkpoint())
