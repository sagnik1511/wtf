"""Both consequence sources reach independent models without blending or gating."""

# Fixtures inspect internal state to verify contracts, private ownership and restore.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring

import copy
import json
import tempfile
from pathlib import Path
import unittest

from tests.test_action_consequences import action_config
from tests.test_mutation import KINDS
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.inspection.explainability import prepare_model, feature_names
from wtf.learning.contextual_ppo import action_mask
from wtf.learning.brain import Experience, observation_inputs
from wtf.learning.brains import (
    create_brain,
    model_snapshot,
    load_model_weights,
    mix_parent_brains,
)
from wtf.simulation.actions import ACTIONS
from wtf.simulation.social import _context, _local_context
from wtf.simulation.world import Animal, World
from wtf.storage.checkpoint_io import read_checkpoint


def combined_config(kind="dqn", schedule="continuous"):
    c = action_config(kind, schedule)
    for template in c["brains"].values():
        template["consequence_scope"] = "both"
    return validate(c)


class CombinedConsequencesTests(unittest.TestCase):
    def test_independent_general_and_context_blocks_remain_visible_while_masked(self):
        c = combined_config("maskable_ppo")
        w = World(c, 42)
        w.animals = {
            0: Animal(0, "predator", 7, 5, 1, 20, 30, sex="male"),
            1: Animal(1, "prey", 1, 1, 0, 20, 20, sex="female"),
        }
        w.observe_all()
        observer = w.animals[0]
        actions = c["species"]["predator"]["allowed_actions"]
        self.assertFalse(action_mask(w, observer, actions)[actions.index("attack")])
        model = w._social.knowledge
        context = _context(
            w, observer, "attack", _local_context(w, observer, w._social.buckets)
        )
        other = (context[0], (context[1] + 1) % 4, *context[2:])
        model.learn("predator", context, [1, 0, 0])
        model.learn("predator", other, [0, 1, 0])
        before = model.checkpoint()
        features = w.observe(observer)
        offset = ACTIONS.index("attack") * 4
        self.assertEqual(
            features[-96 + offset : -96 + offset + 4], (2 / 6, 0.5, 0.5, 0)
        )
        self.assertEqual(features[-48 + offset : -48 + offset + 4], (0.2, 1, 0, 0))
        observer.health = 1
        after = w.observe(observer)
        self.assertEqual(after[-96:-48], features[-96:-48])
        self.assertEqual(after[-48 + offset : -48 + offset + 4], (0, 0, 0, 0))
        self.assertEqual(model.checkpoint(), before)
        self.assertEqual(model.updates, 2)

    def test_all_learners_restore_inherit_and_explain_combined_layout(self):
        # Keep the independent restore sequence explicit for this wider layout.
        # pylint: disable=duplicate-code
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = combined_config(kind)
                e = Experiment(c)
                for _ in range(2):
                    e.step()
                restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
                self.assertEqual(restored.checkpoint(), e.checkpoint())
                e.step()
                restored.step()
                self.assertEqual(restored.checkpoint(), e.checkpoint())
                brain = next(iter(e.brains.values()))
                width = observation_inputs(brain.config)
                legacy = copy.deepcopy(brain.config)
                legacy["consequence_scope"] = "species_action"
                self.assertEqual(width, observation_inputs(legacy) + 48)
                names = feature_names(brain.config)
                self.assertEqual(len(names), width)
                self.assertTrue(names[-96].startswith("Species action "))
                self.assertTrue(names[-48].startswith("Consequence "))
                model = model_snapshot(brain)
                self.assertEqual(
                    prepare_model(model)["signature"]["consequence_scope"], "both"
                )
                parent = create_brain(brain.config, brain.actions, 3, 4)
                child = mix_parent_brains(brain, parent, 99)
                self.assertEqual(model_snapshot(child)["signature"], model["signature"])
                self.assertIsNot(child.config, brain.config)
                other = create_brain(legacy, brain.actions, 3, 4)
                with self.assertRaises(ValueError):
                    load_model_weights(other, model)

    def test_both_sources_reach_trainable_first_layer_weights(self):
        c = combined_config()
        spec = c["species"]["predator"]
        template = copy.deepcopy(c["brains"][spec["brain"]])
        template.update(
            batch_size=1,
            learning_starts_after=1,
            train_every_steps=1,
            discount_factor=0,
        )
        brain = create_brain(template, spec["allowed_actions"], 1, 2)
        width = observation_inputs(template)
        observation = (0.1,) * width
        before = tuple(brain.network.layers[0]["weights"])
        brain.learn(Experience(observation, brain.actions[0], 1, None, True))
        after = brain.network.layers[0]["weights"]
        self.assertEqual(brain.updates, 1)
        self.assertNotEqual(before[width - 96], after[width - 96])
        self.assertNotEqual(before[width - 48], after[width - 48])

    def test_episode_publication_and_binary_restore_do_not_duplicate_evidence(self):
        e = Experiment(combined_config(schedule="episode"))
        e.step()
        model = e.world._social.knowledge
        self.assertTrue(
            all(all(v == 0 for v in o[-96:]) for o in e.observations.values())
        )
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "checkpoint.bin.xz"
            e.write_checkpoint(checkpoint, shared_metadata=True)
            clone = Experiment.restore(read_checkpoint(checkpoint))
        while not e.finished_run:
            e.step()
            clone.step()
        updates = model.updates
        self.assertGreater(updates, 0)
        e.next_run()
        clone.next_run()
        self.assertEqual(model.updates, updates)
        self.assertEqual(model.published_updates, updates)
        self.assertEqual(e.checkpoint(), clone.checkpoint())
