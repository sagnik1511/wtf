"""Species/action knowledge remains visible across masks and physical states."""

# Regression fixtures inspect private state to verify publication and cache ownership.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_social_learning import config
from tests.test_mutation import KINDS
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.learning.brains import create_brain, model_snapshot, load_model_weights
from wtf.learning.consequences import Consequences
from wtf.learning.contextual_ppo import action_mask
from wtf.inspection.explainability import prepare_model, feature_names
from wtf.simulation.actions import ACTIONS
from wtf.simulation.social import _context, _local_context
from wtf.simulation.world import Animal, World
from wtf.storage.checkpoint_io import read_checkpoint


def action_config(kind="maskable_ppo", schedule="continuous"):
    c = config(kind, horizon=1)
    c["consequence_learning"]["update_schedule"] = schedule
    for brain in c["brains"].values():
        brain["consequence_scope"] = "species_action"
    return validate(c)


def key(action="attack", energy=0):
    return (ACTIONS.index(action), energy, 3, 0, 0, 0, 0, 1, 0, 0, 1, 0)


class ActionConsequenceTests(unittest.TestCase):
    def test_action_estimates_pool_retained_context_counts_and_cache_detached_results(
        self,
    ):
        c = action_config()
        m = Consequences(c)
        attack = ACTIONS.index("attack")
        m.learn("predator", key(), [1, 0, 0])
        m.learn("predator", key(energy=3), [0, 1, 0])
        m.learn("predator", key(energy=3), [0, 1, 0])
        expected = [3 / 7, 1 / 3, 2 / 3, 0]
        before = m.checkpoint()
        self.assertEqual(m.predict_action("predator", attack), expected)
        cached = m._action_cache["predator"]
        self.assertEqual(m.predict_action("prey", attack), [0] * 4)
        result = m.predict_action("predator", attack)
        result[0] = 99
        self.assertIs(result[1], m.predict_action("predator", attack)[1])
        self.assertEqual(m.predict_action("predator", attack), expected)
        self.assertIs(m._action_cache["predator"], cached)
        self.assertEqual(m.checkpoint(), before)
        m.learn("prey", key(), [0, 1, 0])
        self.assertIs(m._action_cache["predator"], cached)
        m.learn("predator", key(energy=3), [0, 0, 1])
        self.assertNotIn("predator", m._action_cache)
        self.assertEqual(m.predict_action("predator", attack), [0.5, 0.25, 0.5, 0.25])

    def test_actual_hunt_reaches_an_observer_with_attack_masked_and_different_context(
        self,
    ):
        c = action_config()
        c["species"]["predator"]["body"].update(attack_damage=2, energy_per_kill=10)
        w = World(c, 42)
        w.animals = {
            0: Animal(0, "predator", 1, 1, 1, 10, 40, sex="female"),
            1: Animal(1, "prey", 2, 1, 0, 10, 1, sex="female"),
            2: Animal(2, "predator", 7, 5, 1, 30, 30, sex="male"),
        }
        w.next_animal_id = 3
        w.resources.clear()
        result = w.step({0: "attack", 1: "rest", 2: "rest"})
        self.assertIn(1, result.dead_ids)
        observer = w.animals[2]
        actions = c["species"]["predator"]["allowed_actions"]
        mask = action_mask(w, observer, actions)
        self.assertFalse(mask[actions.index("attack")])
        model = w._social.knowledge
        state = model.checkpoint()
        rng = w.rng.getstate()
        features = w.observe(observer)[-48:]
        offset = ACTIONS.index("attack") * 4
        self.assertGreater(features[offset], 0)
        self.assertEqual(features[offset + 2], 1)
        # Exact-context lookup at the observer remains unknown; general knowledge
        # comes from the other sex's real hunt and exposes no hidden prey position.
        context = _context(
            w, observer, "attack", _local_context(w, observer, w._social.buckets)
        )
        self.assertEqual(model.predict("predator", context), [0] * 4)
        self.assertEqual(model.checkpoint(), state)
        self.assertEqual(w.rng.getstate(), rng)
        # Knowledge remains identical when attack becomes feasible. Neural inputs
        # still expose changing body/target state; no hand-written mode gates it.
        w.animals[3] = Animal(3, "prey", 8, 5, 0, 10, 10, sex="female")
        w.next_animal_id = 4
        observer.health = 20
        self.assertTrue(action_mask(w, observer, actions)[actions.index("attack")])
        self.assertEqual(w.observe(observer)[-48:], features)
        model.learn("predator", key("reproduce"), [0, 0, 1])
        self.assertFalse(action_mask(w, observer, actions)[actions.index("reproduce")])
        self.assertEqual(
            w.observe(observer)[-48 + ACTIONS.index("reproduce") * 4 + 3], 1
        )

    def test_episode_cache_does_not_reveal_staged_evidence(self):
        c = action_config(schedule="episode")
        model = Consequences(c)
        attack = ACTIONS.index("attack")
        self.assertEqual(model.predict_action("predator", attack), [0] * 4)
        old_cache = model._action_cache["predator"]
        model.learn("predator", key(), [0, 1, 0])
        self.assertEqual(model.predict_action("predator", attack), [0] * 4)
        self.assertIs(model._action_cache["predator"], old_cache)
        restored = Consequences.restore(c, json.loads(json.dumps(model.checkpoint())))
        self.assertFalse(restored._action_cache)
        self.assertEqual(restored.predict_action("predator", attack), [0] * 4)
        model.publish()
        restored.publish()
        self.assertEqual(model.predict_action("predator", attack), [0.2, 0, 1, 0])
        self.assertEqual(model.checkpoint(), restored.checkpoint())
        model.learn("predator", key(energy=2), [1, 0, 0])
        self.assertEqual(model.predict_action("predator", attack), [0.2, 0, 1, 0])
        model.publish()
        self.assertEqual(model.predict_action("predator", attack), [2 / 6, 0.5, 0.5, 0])

    def test_context_eviction_rebuilds_cache_and_old_artifacts_keep_their_format(self):
        c = action_config()
        c["consequence_learning"]["max_contexts"] = 2
        model = Consequences(c)
        model.learn("predator", key(), [0, 1, 0])
        self.assertGreater(
            model.predict_action("predator", ACTIONS.index("attack"))[0], 0
        )
        model.learn("predator", key("rest"), [0, 0, 0])
        model.learn("predator", key("eat"), [0, 1, 0])
        self.assertEqual(
            model.predict_action("predator", ACTIONS.index("attack")), [0] * 4
        )
        state = model.checkpoint()
        self.assertEqual(state["version"], 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "consequences.json.xz"
            model.save(path)
            self.assertEqual(Consequences.load(c, path).checkpoint(), state)
        legacy = config(horizon=1)
        legacy["consequence_learning"]["max_contexts"] = 2
        self.assertEqual(Consequences.restore(legacy, state).checkpoint(), state)

    def test_all_algorithms_resume_and_new_scope_has_explicit_model_compatibility(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = action_config(kind)
                e = Experiment(c)
                for _ in range(2):
                    e.step()
                restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
                self.assertEqual(restored.checkpoint(), e.checkpoint())
                e.step()
                restored.step()
                self.assertEqual(restored.checkpoint(), e.checkpoint())
                brain = next(iter(e.brains.values()))
                model = model_snapshot(brain)
                prepared = prepare_model(model)
                self.assertEqual(
                    prepared["signature"]["consequence_scope"], "species_action"
                )
                self.assertTrue(
                    any(
                        name.startswith("Species action ")
                        for name in feature_names(brain.config)
                    )
                )
                legacy_config = copy.deepcopy(brain.config)
                legacy_config.pop("consequence_scope")
                legacy_brain = create_brain(legacy_config, brain.actions, 1, 2)
                with self.assertRaises(ValueError):
                    load_model_weights(legacy_brain, model)
                legacy_model = model_snapshot(legacy_brain)
                legacy_config["consequence_scope"] = "context"
                same = create_brain(legacy_config, brain.actions, 1, 2)
                load_model_weights(same, legacy_model)

    def test_binary_restore_and_evaluation_never_update_knowledge(self):
        c = action_config("dqn", schedule="episode")
        e = Experiment(c)
        e.step()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "state.bin.xz"
            e.write_checkpoint(checkpoint, shared_metadata=True)
            resumed = Experiment.restore(read_checkpoint(checkpoint), mode="evaluate")
        before = resumed.world._social.knowledge.checkpoint()
        for _ in range(4):
            resumed.step()
        self.assertEqual(resumed.world._social.knowledge.checkpoint(), before)

    def test_invalid_scopes_are_rejected(self):
        for value in (None, False, [], "all", ""):
            c = action_config()
            next(iter(c["brains"].values()))["consequence_scope"] = value
            with self.assertRaises(ValueError):
                validate(c)
        c = action_config()
        next(iter(c["brains"].values()))["consequence_observation"] = False
        with self.assertRaises(ValueError):
            validate(c)
