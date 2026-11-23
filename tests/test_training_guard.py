"""Qualified promotion and versioned, local, RNG-free mating observations."""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
import unittest

from wtf.learning.brain import observation_inputs
from wtf.learning.brains import model_snapshot, load_model_weights
from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from tests.test_reproduction import breeding_world
from tests.test_mutation import KINDS, environment
from tests.test_lineage_champions import configured
from tests.reproduction_support import force_reproduction


class MatingObservationTests(unittest.TestCase):
    def world(self):
        w = breeding_world()
        w.config["brains"][w.config["species"]["prey"]["brain"]][
            "observation_version"
        ] = 3
        w.config["species"]["prey"]["reproduction"].update(
            mate_radius=2, min_age_steps=5
        )
        for a in w.animals.values():
            a.age = 5
        return w

    def test_local_features_and_no_rng_or_clock_mutation(self):
        w = self.world()
        before = w.checkpoint()
        a = w.observe_all()[0]
        self.assertEqual(len(a), 42)
        self.assertEqual(a[32:36], (-1.0, 1.0, 0.0, 1.0))
        self.assertEqual(a[39:], (1.0, 1.0, 1.0))
        self.assertGreater(a[36], 0)
        self.assertEqual(w.observe_all()[0], a)
        self.assertEqual(w.checkpoint(), before)
        w.animals[1].sex = "female"
        self.assertEqual(w.observe_all()[0][36:], (0.0,) * 6)

    def test_readiness_refreshes_without_movement_and_cooldown_is_observed(self):
        w = self.world()
        w.animals[1].age = 0
        self.assertEqual(w.observe_all()[0][40:], (0.0, 0.0))
        w.animals[1].age = 5
        self.assertEqual(w.observe_all()[0][40:], (1.0, 1.0))
        w.animals[0].last_reproduced_age = 5
        w.animals[0].reproduction_due_age = 8
        o = w.observe_all()[0]
        self.assertAlmostEqual(o[34], 2 / 3)
        self.assertEqual(o[35], 0)
        self.assertEqual(o[41], 0)
        w.animals[0].age = 8
        self.assertEqual(w.observe_all()[0][34:36], (0.0, 1.0))

    def test_visibility_species_capacity_and_birth_space_are_respected(self):
        w = self.world()
        w.obstacles.add((3, 2))
        self.assertEqual(w.observe_all()[0][39:], (0.0, 0.0, 0.0))
        w.obstacles.clear()
        w.animals[1].x = 8
        w.config["species"]["prey"]["body"]["vision_radius"] = 2
        self.assertEqual(w.observe_all()[0][39:], (0.0, 0.0, 0.0))
        w.animals[1].x = 3
        rules = w.config["species"]["prey"]["reproduction"]
        rules["max_population"] = 2
        self.assertEqual(w.observe_all()[0][40:], (1.0, 0.0))
        rules["max_population"] = 8
        w.config["species"]["other"] = copy.deepcopy(w.config["species"]["prey"])
        w.animals[1].species = "other"
        self.assertEqual(w.observe_all()[0][39:], (0.0, 0.0, 0.0))

    def test_legacy_vectors_and_incompatible_warm_starts(self):
        w = self.world()
        template = w.config["brains"][w.config["species"]["prey"]["brain"]]
        v3 = w.observe_all()[0]
        for version, width in ((1, 20), (2, 32)):
            template["observation_version"] = version
            self.assertEqual(w.observe_all()[0], v3[:width])
        e = Experiment(environment("ppo", 0))
        model = model_snapshot(e.brains[0])
        c = environment("ppo", 0)
        for b in c["brains"].values():
            b["observation_version"] = 3
        new = Experiment(c)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_model_weights(new.brains[0], model)

    def test_all_algorithms_v3_birth_and_resume(self):
        for kind in KINDS:
            for wet in (False, True):
                with self.subTest(kind=kind, wet=wet):
                    c = environment(kind, 0)
                    if wet:
                        c["terrain"] = {
                            "map_version": 1,
                            "name": "test_river",
                            "width": 5,
                            "height": 5,
                            "rows": ["..~.."] * 5,
                            "legend": {
                                ".": {},
                                "~": {"water_depth": 1, "flow": [0, 0.2]},
                            },
                        }
                        for spec in c["species"].values():
                            spec["allowed_actions"].append("swim")
                    for b in c["brains"].values():
                        b["observation_version"] = 3
                    a = force_reproduction(Experiment(c))
                    self.assertTrue(all(len(o) == 42 for o in a.observations.values()))
                    a.step()
                    if not wet:
                        self.assertTrue(
                            any(event.kind == "born" for event in a.world.events)
                        )
                    b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
                    a.step()
                    b.step()
                    self.assertEqual(a.checkpoint(), b.checkpoint())
                    self.assertTrue(
                        all(observation_inputs(x) == 42 for x in c["brains"].values())
                    )

    def test_meandering_configuration(self):
        c = load("environments/meandering_river.json")
        self.assertEqual(c["brains"]["small_grpo"]["rollout_horizon"], 16)
        self.assertEqual(c["brains"]["small_grpo"]["train_every_steps"], 32)
        self.assertTrue(
            all(b["observation_version"] == 3 for b in c["brains"].values())
        )
        c["experiment"]["guard_lineage_chaining"] = "true"
        with self.assertRaisesRegex(ValueError, "guard_lineage_chaining"):
            validate(c)

    def test_mate_sight_obeys_directional_water_bonus_and_walls(self):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel

        from wtf.simulation.geography import Geography

        w = self.world()
        definition = {
            "map_version": 1,
            "name": "sight",
            "width": 10,
            "height": 8,
            "rows": [".........."] * 2 + ["...~~~...."] + [".........."] * 5,
            "legend": {".": {}, "~": {"water_depth": 1}},
        }
        w.geography = Geography(definition, 10, 8)
        w.config["terrain"] = definition
        w.config["species"]["prey"]["body"].update(
            vision_radius=2, water_vision_bonus=3
        )
        w.animals[1].x = 6
        self.assertEqual(w.observe_all()[0][39], 1)
        w.obstacles.add((4, 2))
        self.assertEqual(w.observe_all()[0][39], 0)
        w.obstacles.clear()
        w.config["species"]["prey"]["body"]["water_vision_bonus"] = 0
        self.assertEqual(w.observe_all()[0][39], 0)

    def test_longer_grpo_trials_keep_v3_inputs_and_isolate_real_world(self):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel

        from wtf.learning.grouped_rollouts import collect_group

        c = load("environments/meandering_river.json")
        c["experiment"].update(runs=1, world_seed_policy="increment")
        for spec in c["species"].values():
            spec["initial_population"] = 2
        e = Experiment(c)
        focal = next(i for i, b in e.brains.items() if b.config["type"] == "grpo")
        before = e.world.checkpoint()
        others = {i: b.checkpoint() for i, b in e.brains.items() if i != focal}
        group = collect_group(e.world, e.brains, focal)
        self.assertEqual([len(t) for t in group], [16] * 4)
        self.assertTrue(all(len(s.observation) == 42 for t in group for s in t))
        self.assertEqual(e.world.checkpoint(), before)
        self.assertEqual(
            {i: b.checkpoint() for i, b in e.brains.items() if i != focal}, others
        )


class PromotionGuardTests(unittest.TestCase):
    def test_loaded_qualified_model_is_protected_without_cross_sex_fallback(self):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel

        import tempfile
        from pathlib import Path
        from wtf.storage.library import export_champions, apply_brain_overrides

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = configured()
            e = Experiment(c)
            e.champions.entries["prey"]["female"]["lineage"]["lineage"][
                "qualified"
            ] = True
            e.champions.entries["prey"]["male"]["lineage"]["lineage"][
                "qualified"
            ] = False
            export_champions(e, root / "source/champions/run_0001.json")
            apply_brain_overrides(c, ["prey=lineage"], root / "source")
            a = Experiment(c)
            previous = copy.deepcopy(a.initial_models["prey"]["female"])
            self.assertTrue(previous["provenance"]["lineage_qualified"])
            while not a.finished_run:
                a.step()
            for sex in ("female", "male"):
                candidate = a.champions.entries["prey"][sex]["lineage"]
                candidate["lineage"]["qualified"] = False
                candidate["model"]["parameters"]["b2"][0] += 2
            incoming_male = copy.deepcopy(
                a.champions.entries["prey"]["male"]["lineage"]["model"]
            )
            a.next_run()
            self.assertEqual(
                a.run_initial_models["prey"]["female"]["model"], previous["model"]
            )
            self.assertEqual(
                a.run_initial_models["prey"]["male"]["model"], incoming_male
            )

    def test_proven_brain_survives_multiple_bad_runs_and_checkpoint(self):
        for guard in (True, False):
            with self.subTest(guard=guard):
                c = configured()
                c["experiment"].update(
                    runs=4, guard_lineage_chaining=guard, promote_only_improved=False
                )
                a = Experiment(c)
                while not a.finished_run:
                    a.step()
                first = a.champions.entries["prey"]["female"]["lineage"]
                first["lineage"]["qualified"] = True
                a.next_run()
                expected = copy.deepcopy(a.run_initial_models["prey"]["female"])
                for run in (2, 3):
                    while not a.finished_run:
                        a.step()
                    candidate = a.champions.entries["prey"]["female"]["lineage"]
                    candidate["lineage"]["qualified"] = False
                    candidate["model"]["parameters"]["b2"][0] += 2
                    incoming = copy.deepcopy(candidate["model"])
                    restored = Experiment.restore(
                        json.loads(json.dumps(a.checkpoint()))
                    )
                    a.next_run()
                    restored.next_run()
                    self.assertEqual(a.checkpoint(), restored.checkpoint())
                    chosen = a.run_initial_models["prey"]["female"]
                    if guard:
                        self.assertEqual(chosen["model"], expected["model"])
                        self.assertEqual(
                            chosen["provenance"]["model_generation"],
                            expected["provenance"]["model_generation"],
                        )
                        self.assertEqual(
                            chosen["provenance"]["retained_after_run"], run
                        )
                    else:
                        self.assertEqual(chosen["model"], incoming)

    def test_old_checkpoint_keeps_original_chaining_behavior(self):
        e = Experiment(configured())
        state = e.checkpoint()
        state.pop("guard_lineage_chaining")
        state["config"]["experiment"].pop("guard_lineage_chaining", None)
        restored = Experiment.restore(state)
        self.assertFalse(restored.guard_lineage_chaining)


if __name__ == "__main__":
    unittest.main()
