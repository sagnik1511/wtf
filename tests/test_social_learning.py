"""Optional collective sensing, bounded consequence memory and independent policies."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.test_geography import configuration, animal
from tests.test_mutation import environment, KINDS
from wtf.config import validate, load
from wtf.experiment.runner import Experiment
from wtf.simulation.world import World, Resource
from wtf.learning.consequences import Consequences
from wtf.simulation.social import SocialState, social_width, social_names
from wtf.learning.brain import observation_inputs
from wtf.learning.brains import model_signature, model_snapshot, load_model_weights
from wtf.inspection.explainability import feature_names, prepare_model
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.recording import RecordingStore
from wtf.storage.replay_io import load_chunk


def config(kind="ppo", horizon=2):
    c = configuration(kind, rows=["........."] * 7, legend={".": {}})
    c["world"]["max_steps"] = 6
    c["experiment"].update(runs=2, world_seed_policy="increment")
    c["consequence_learning"] = dict(
        enabled=True, max_contexts=16, horizon_steps=horizon
    )
    for b in c["brains"].values():
        b["social_observation"] = dict(history_steps=2, max_neighbors=2)
        b["consequence_observation"] = True
    return validate(c)


def controlled(c=None):
    w = World(c or config(), 42)
    w.animals = {0: animal(0, 2, 2), 1: animal(1, 3, 2)}
    w.next_animal_id = 2
    return w


class SocialLearningTests(unittest.TestCase):
    def test_disabled_contracts_and_world_state_remain_legacy(self):
        c = validate(configuration("ppo"))
        w = World(c, 42)
        self.assertIsNone(w._social)
        self.assertNotIn("social_state", w.checkpoint())
        for b in c["brains"].values():
            original = model_signature(b, ["rest"])
            disabled = {**b, "consequence_observation": False}
            self.assertEqual(social_width(disabled), 0)
            self.assertEqual(model_signature(disabled, ["rest"]), original)
        self.assertEqual(World.restore(w.checkpoint()).checkpoint(), w.checkpoint())

    def test_direct_water_world_can_step_before_initial_observation(self):
        c = load("environments/seaside_social.json")
        w = World(c, 100)
        result = w.step({i: "rest" for i in w.animals})
        self.assertEqual(w.tick, 1)
        self.assertEqual(len(result.observations), len(w.animals))

    def test_config_rejects_invalid_layouts_and_missing_enablement(self):
        for field, value in [
            ("max_neighbors", 0),
            ("max_neighbors", True),
            ("history_steps", 17),
            ("history_steps", 1.5),
            ("surprise", 1),
        ]:
            c = config()
            next(iter(c["brains"].values()))["social_observation"][field] = value
            with self.assertRaises(ValueError):
                validate(c)
        for value in (1, None, "yes"):
            c = config()
            next(iter(c["brains"].values()))["consequence_observation"] = value
            with self.assertRaises(ValueError):
                validate(c)
        for field, value in [
            ("enabled", 1),
            ("max_contexts", True),
            ("max_contexts", 15),
            ("horizon_steps", 65),
            ("horizon_steps", 0),
        ]:
            c = config()
            c["consequence_learning"][field] = value
            with self.assertRaises(ValueError):
                validate(c)
        c = config()
        c.pop("consequence_learning")
        with self.assertRaises(ValueError):
            validate(c)

    def test_blocks_are_independently_optional_and_layouts_match(self):
        for local, persistent in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            c = config()
            for b in c["brains"].values():
                if not local:
                    b.pop("social_observation")
                b["consequence_observation"] = persistent
            c["consequence_learning"]["enabled"] = persistent
            w = World(validate(c), 42)
            observations = w.observe_all()
            expected = 32 + 24 * local + 48 * persistent
            self.assertTrue(all(len(o) == expected for o in observations.values()))
            w.step({i: "rest" for i in w.animals})
            self.assertEqual(w.checkpoint(), World.restore(w.checkpoint()).checkpoint())
            if w._social is not None:
                self.assertEqual(w._social.knowledge is not None, persistent)
        c = config()
        next(iter(c["brains"].values()))["social_observation"] = None
        with self.assertRaises(ValueError):
            validate(c)

    def test_pending_outcomes_are_not_future_labels_and_censoring_is_not_safety(self):
        c = config(horizon=2)
        w = controlled(c)
        w.config["species"]["prey"]["body"]["living_cost"] = 1
        w.animals[1].energy = 0
        w.animals[1].health = 10
        w.step({0: "rest", 1: "rest"})
        self.assertEqual(w._social.knowledge.updates, 0)
        key = w._social.pending[1][2][0][1]
        w.step({0: "rest", 1: "rest"})
        self.assertEqual(w._social.knowledge.predict("prey", key)[1], 1)
        c = config(horizon=4)
        c["world"]["max_steps"] = 1
        e = Experiment(c)
        e.step()
        self.assertEqual(e.world._social.knowledge.updates, 0)
        e.next_run()
        self.assertFalse(e.world._social.pending)
        self.assertEqual(e.world._social.knowledge.updates, 0)

    def test_visible_death_is_pooled_without_identity_or_exact_reward(self):
        a = controlled()
        b = controlled()
        for w in (a, b):
            w.animals[1].health = 1
            w.animals[1].energy = 0
            w.config["species"]["prey"]["body"]["living_cost"] = 1
        b.config = copy.deepcopy(b.config)
        b.config["species"]["prey"]["rewards"].update(death=-999, per_damage_taken=-20)
        first = a.step({0: "rest", 1: "rest"})
        second = b.step({0: "rest", 1: "rest"})
        self.assertNotEqual(first.rewards[1], second.rewards[1])
        self.assertEqual(a._social.checkpoint(), b._social.checkpoint())
        self.assertIn(1, first.dead_ids)
        features = a.observe(a.animals[0])
        tail = features[-72:-48]
        self.assertGreater(tail[20], 0)  # idle confidence
        self.assertEqual(tail[21], 1)  # visible injury/death
        self.assertEqual(len(tail), 24)
        self.assertNotIn("reward", json.dumps(a._social.checkpoint()))
        self.assertGreater(a._social.knowledge.updates, 0)

    def test_visible_feeding_and_births_are_real_outcomes_and_newborns_have_no_sighting_history(
        self,
    ):
        w = controlled()
        w.resources[(3, 2)] = Resource(3, 2, 10)
        w.step({0: "rest", 1: "eat"})
        feed = w.observe(w.animals[0])[-72:-48]
        self.assertGreater(feed[8], 0)
        self.assertEqual(feed[10], 1)
        c = config(horizon=1)
        c["species"]["prey"]["reproduction"].update(
            enabled=True,
            min_age_steps=1,
            cooldown_steps=1,
            min_energy=1,
            energy_cost=1,
            health_threshold=1,
            max_population=8,
            mate_radius=2,
        )
        w = controlled(validate(c))
        w.animals[1].sex = "male"
        w.animals[2] = animal(2, 2, 3)
        w.next_animal_id = 3
        result = w.step({0: "reproduce", 1: "reproduce", 2: "rest"})
        birth = next(e for e in result.events if e.kind == "born")
        self.assertEqual(w.observe(w.animals[2])[-72:-48][19], 1)
        self.assertEqual(w.observe(w.animals[birth.target])[-72:-48], (0.0,) * 24)

    def test_hidden_death_and_species_mismatch_do_not_enter_collective_sightings(self):
        for hidden in ("wall", "radius", "species"):
            w = controlled()
            if hidden == "wall":
                w.obstacles.add((3, 2))
                w.animals[1].x = 4
            elif hidden == "radius":
                w.config["species"]["prey"]["body"]["vision_radius"] = 1
                w.animals[1].x = 5
            else:
                w.animals[1].species = "predator"
            w.animals[1].energy = 0
            w.animals[1].health = 1
            w.config["species"][w.animals[1].species]["body"]["living_cost"] = 1
            w.step({0: "rest", 1: "rest"})
            self.assertEqual(w.observe(w.animals[0])[-72:-48], (0.0,) * 24)

    def test_disappearance_beyond_sight_does_not_mean_observed_death(self):
        w = controlled()
        w.config["species"]["prey"]["body"].update(vision_radius=1, living_cost=1)
        w.animals[1].energy = 0
        w.animals[1].health = 1
        w.step({0: "rest", 1: "move_forward"})
        self.assertNotIn(1, w.animals)
        self.assertEqual(w.observe(w.animals[0])[-72:-48], (0.0,) * 24)

    def test_collective_history_is_bounded_and_observation_is_read_only(self):
        w = controlled()
        for _ in range(5):
            w.step({i: "rest" for i in w.animals})
        self.assertEqual(len(w._social.sightings[0]), 2)
        before = copy.deepcopy(w.checkpoint())
        one = w.observe_all()
        two = w.observe_all()
        self.assertEqual(one, two)
        self.assertEqual(w.checkpoint(), before)
        self.assertTrue(
            all(len(queue) < 2 for _, _, queue in w._social.pending.values())
        )
        self.assertTrue(
            all(
                sum(counts[::4]) <= 2
                for history in w._social.sightings.values()
                for _, counts in history
            )
        )

    def test_outcome_memory_is_coordinate_free_and_counter_context_bounded(self):
        c = config(horizon=1)
        model = Consequences(c)
        for i in range(80):
            key = (i % 12, (i // 12) % 4, (i // 48) % 4, 0, 0, 0, 0, 1, 0, 0, 0, 0)
            model.learn("prey", key, [1, 0, 1])
        self.assertEqual(len(model.tables["prey"]), 16)
        key = next(reversed(model.tables["prey"]))
        old = model.checkpoint()
        prediction = model.predict("prey", key)
        self.assertEqual(prediction, [0.2, 1, 0, 1])
        self.assertEqual(model.checkpoint(), old)
        clone = Consequences.restore(c, json.loads(json.dumps(old)))
        self.assertEqual(clone.checkpoint(), old)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consequences.json.xz"
            model.save(path)
            self.assertEqual(Consequences.load(c, path).checkpoint(), old)
            self.assertLess(path.stat().st_size, len(json.dumps(old)))
            changed = copy.deepcopy(c)
            changed["species"]["prey"]["body"]["max_health"] += 1
            with self.assertRaises(ValueError):
                Consequences.load(changed, path)
            changed = copy.deepcopy(c)
            changed["species"]["prey"]["rewards"]["death"] = -99
            self.assertEqual(Consequences.load(changed, path).checkpoint(), old)

    def test_exact_species_pooling_and_sex_independent_knowledge(self):
        c = config(horizon=1)
        c["species"]["other_prey"] = copy.deepcopy(c["species"]["prey"])
        c["species"]["other_prey"]["initial_population"] = 0
        w = controlled(validate(c))
        w.animals[1].species = "other_prey"
        w.step({0: "rest", 1: "rest"})
        self.assertEqual(w.observe(w.animals[0])[-72:-48], (0.0,) * 24)
        self.assertTrue(w._social.knowledge.tables["prey"])
        self.assertTrue(w._social.knowledge.tables["other_prey"])
        self.assertIsNot(
            w._social.knowledge.tables["prey"], w._social.knowledge.tables["other_prey"]
        )
        w = controlled(config(horizon=1))
        w.animals[0].sex = "male"
        w.animals[1].sex = "female"
        w.step({0: "rest", 1: "rest"})
        self.assertEqual(w._social.knowledge.updates, 2)
        self.assertEqual(len(w._social.knowledge.tables["prey"]), 1)

    def test_restored_histories_cannot_exceed_memory_bounds(self):
        w = controlled()
        w.step({0: "rest", 1: "rest"})
        state = w._social.checkpoint()
        state["sightings"]["0"] *= 20
        with self.assertRaises(ValueError):
            SocialState.restore(w.config, state)
        state = w._social.checkpoint()
        state["pending"]["0"][2] *= 100
        with self.assertRaises(ValueError):
            SocialState.restore(w.config, state)

    def test_checkpoint_exact_continuation_and_evaluation_freeze(self):
        e = Experiment(config())
        for _ in range(3):
            e.step()
        f = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
        self.assertEqual(e.checkpoint(), f.checkpoint())
        for _ in range(2):
            e.step()
            f.step()
            self.assertEqual(e.checkpoint(), f.checkpoint())
        evaluator = Experiment.restore(e.checkpoint(), mode="evaluate")
        before = evaluator.world._social.knowledge.checkpoint()
        self.assertFalse(evaluator.world._social.pending)
        for _ in range(3):
            evaluator.step()
        self.assertEqual(evaluator.world._social.knowledge.checkpoint(), before)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.bin.xz"
            e.write_checkpoint(path, shared_metadata=True)
            restored = Experiment.restore(read_checkpoint(path))
            self.assertEqual(restored.checkpoint(), e.checkpoint())

    def test_episode_reset_keeps_population_knowledge_and_independent_brains(self):
        e = Experiment(config(horizon=1))
        while not e.finished_run:
            e.step()
        knowledge = e.world._social.knowledge
        before = knowledge.checkpoint()
        self.assertTrue(e.next_run())
        self.assertIs(e.world._social.knowledge, knowledge)
        self.assertEqual(knowledge.checkpoint(), before)
        self.assertFalse(e.world._social.sightings)
        self.assertFalse(e.world._social.pending)
        brains = list(e.brains.values())
        self.assertIsNot(brains[0], brains[1])
        self.assertIsNot(brains[0].network, brains[1].network)

    def test_all_controllers_support_inputs_and_restore_without_policy_sharing(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = config(kind)
                e = Experiment(c)
                for b in c["brains"].values():
                    self.assertEqual(observation_inputs(b), 32 + 72)
                    sig = model_signature(b, c["species"]["prey"]["allowed_actions"])
                    self.assertEqual(len(feature_names(sig)), observation_inputs(b))
                    self.assertEqual(len(social_names(b)), 72)
                e.step()
                clone = Experiment.restore(e.checkpoint())
                e.step()
                clone.step()
                self.assertEqual(e.checkpoint(), clone.checkpoint())
                brain = next(iter(e.brains.values()))
                model = model_snapshot(brain)
                prepare_model(model)
                other = copy.deepcopy(brain)
                other.config.pop("social_observation")
                with self.assertRaises(ValueError):
                    load_model_weights(other, model)

    def test_grpo_trials_and_reused_worlds_cannot_update_real_population_knowledge(
        self,
    ):
        from wtf.learning.grouped_rollouts import collect_group

        e = Experiment(config("grpo", horizon=1))
        before = copy.deepcopy(e.world.checkpoint())
        focal = next(iter(e.world.animals))
        collect_group(e.world, e.brains, focal)
        self.assertEqual(e.world.checkpoint(), before)
        trial = World.restore(before)
        trial.step({i: "rest" for i in trial.animals})
        self.assertGreater(
            trial._social.knowledge.updates, e.world._social.knowledge.updates
        )
        self.assertEqual(e.world.checkpoint(), before)
        trial._reset_trial(before)
        self.assertEqual(trial.checkpoint(), before)

    def test_cli_export_import_and_recorded_features(self):
        c = config(horizon=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "config.json"
            path.write_text(json.dumps(c))
            previous = None
            for name in ("first", "second"):
                out = root / name
                args = [
                    sys.executable,
                    "-m",
                    "wtf.cli",
                    str(path),
                    "--runs",
                    "1",
                    "--steps",
                    "3",
                    "--no-progress",
                    "--no-library",
                    "--output",
                    str(out),
                ]
                if previous:
                    args += ["--consequences-from", str(previous)]
                result = subprocess.run(args, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                folder = next(out.iterdir())
                model = Consequences.load(c, folder)
                self.assertGreater(model.updates, 0)
                store = RecordingStore(out)
                meta = store.metadata(folder.name)
                chunk = load_chunk(store.chunk(folder.name, 1, 0))
                self.assertEqual(
                    meta["config"]["consequence_learning"], c["consequence_learning"]
                )
                self.assertTrue(any(frame.get("diagnostics") for frame in chunk))
                previous = folder
