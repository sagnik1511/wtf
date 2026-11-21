"""Mutual reproduction choices, physical failure gates and RL reward feedback."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from wtf.inspection.analyze import analyze_run, analyze_recording
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.config import validate, load
from wtf.learning.contextual_ppo import action_mask
from wtf.storage.recording import Recorder, RecordingStore
from wtf.storage.replay_io import load_chunk
from wtf.experiment.runner import Experiment
from wtf.inspection.telemetry import summarize
from wtf.simulation.world import World
from tests.test_reproduction import breeding_world
from tests.test_mutation import KINDS, environment
from tests.test_geography import configuration as terrain_configuration, animal
from tests.reproduction_support import force_reproduction, force_action


def feedback(w):
    for spec in w.config["species"].values():
        spec["rewards"].update(
            per_energy_cost=-0.01, failed_action=-0.05, per_offspring=1.0
        )
    return w


def choose(w, actions=None):
    values = {i: "rest" for i in w.animals}
    values.update(actions or {})
    return w.step(values)


class ReproductionDecisionTests(unittest.TestCase):
    def test_nearby_animals_do_not_reproduce_without_both_choices(self):
        for actions in ({}, {0: "reproduce"}, {1: "reproduce"}):
            w = feedback(breeding_world())
            r = choose(w, actions)
            self.assertFalse(any(e.kind == "born" for e in r.events))
            self.assertEqual(len(w.animals), 2)
            for i, a in w.animals.items():
                self.assertEqual(a.energy, 60)
                self.assertEqual(a.last_reproduced_age, -1)
                self.assertEqual(r.rewards[i], -0.05 if i in actions else 0)

    def test_both_choose_but_outside_radius_fails_without_cost_or_cooldown(self):
        w = feedback(breeding_world())
        w.config["species"]["prey"]["reproduction"]["mate_radius"] = 2
        w.animals[1].x, w.animals[1].y = 4, 3  # sqrt(5), just outside radius 2.
        r = choose(w, {0: "reproduce", 1: "reproduce"})
        self.assertEqual(len(w.animals), 2)
        self.assertEqual(r.rewards, {0: -0.05, 1: -0.05})
        self.assertTrue(
            all(
                a.energy == 60 and a.last_reproduced_age == -1
                for a in w.animals.values()
            )
        )

    def test_boundary_distance_success_keeps_existing_child_and_parent_rules(self):
        w = feedback(breeding_world())
        w.config["species"]["prey"]["reproduction"]["mate_radius"] = 2
        w.animals[1].x, w.animals[1].y = 4, 2
        r = choose(w, {0: "reproduce", 1: "reproduce"})
        birth = next(e for e in r.events if e.kind == "born")
        child = w.animals[birth.target]
        self.assertEqual((child.parent_id, child.second_parent_id), (0, 1))
        self.assertEqual((child.age, child.health, child.energy), (0, 30, 60))
        self.assertEqual(child.last_reward, 0)
        for i in (0, 1):
            self.assertEqual(w.animals[i].energy, 55)
            self.assertEqual(w.animals[i].last_reproduced_age, 1)
            self.assertAlmostEqual(r.rewards[i], 0.95)
        r = choose(w, {0: "reproduce", 1: "reproduce"})
        self.assertEqual(w.births["prey"], 1)
        self.assertEqual(r.rewards[0], -0.05)
        self.assertEqual(w.animals[0].energy, 55)

    def test_health_capacity_obstacles_and_disabled_rules_still_block_consensual_pairs(
        self,
    ):
        for reason in ("health", "capacity", "obstacle", "disabled"):
            w = feedback(breeding_world())
            rules = w.config["species"]["prey"]["reproduction"]
            if reason == "health":
                w.animals[1].health = rules["health_threshold"]
            elif reason == "capacity":
                rules["max_population"] = 2
            elif reason == "disabled":
                rules["enabled"] = False
            else:
                w.animals[1].x = 4
                w.obstacles.add((3, 2))
            r = choose(w, {0: "reproduce", 1: "reproduce"})
            self.assertFalse(any(e.kind == "born" for e in r.events), reason)
            self.assertEqual(r.rewards, {0: -0.05, 1: -0.05})
            self.assertEqual([w.animals[i].energy for i in (0, 1)], [60, 60])

    def test_mask_checks_distance_and_eligibility_without_rng_clock_or_consent_changes(
        self,
    ):
        w = feedback(breeding_world())
        a = w.animals[0]
        original = copy.deepcopy(w.checkpoint())
        self.assertEqual(action_mask(w, a, ["rest", "reproduce"]), (True, True))
        self.assertEqual(w.checkpoint(), original)
        w.config["species"]["prey"]["reproduction"]["mate_radius"] = 1
        w.animals[1].x = 5
        self.assertEqual(action_mask(w, a, ["rest", "reproduce"]), (True, False))
        w.animals[1].x = 3
        w.animals[1].health = 24
        self.assertEqual(action_mask(w, a, ["rest", "reproduce"]), (True, False))
        w.animals[1].health = 30
        # The mask cannot know whether the partner will choose rest next.
        self.assertEqual(action_mask(w, a, ["rest", "reproduce"]), (True, True))
        r = choose(w, {0: "reproduce"})
        self.assertEqual(r.rewards[0], -0.05)

    def test_old_action_sets_cannot_trigger_automatic_births(self):
        w = breeding_world()
        for s in w.config["species"].values():
            s["allowed_actions"].remove("reproduce")
        choose(w)
        self.assertEqual(len(w.animals), 2)
        with self.assertRaisesRegex(ValueError, "unsupported action"):
            choose(w, {0: "reproduce"})


class ActionFeedbackTests(unittest.TestCase):
    def test_walking_effort_and_blocked_action_penalty_are_separate(self):
        w = feedback(breeding_world())
        spec = w.config["species"]["prey"]
        spec["body"].update(living_cost=1, movement_cost=2)
        spec["rewards"]["per_energy_cost"] = -0.1
        r = choose(w, {0: "move_forward"})
        self.assertAlmostEqual(r.rewards[0], -0.3)
        self.assertEqual(w.animals[0].energy, 57)
        w.obstacles.add(w.front(w.animals[0]))
        r = choose(w, {0: "move_forward"})
        self.assertAlmostEqual(r.rewards[0], -0.35)
        self.assertEqual(w.animals[0].energy, 54)

    def test_water_and_uphill_effort_produce_immediate_reward_cost(self):
        c = terrain_configuration()
        w = feedback(World(validate(c), 42))
        w.animals = {0: animal(0, 3, 2)}
        w.config["species"]["prey"]["rewards"]["per_energy_cost"] = -0.1
        w.config["species"]["prey"]["locomotion"]["water_energy_cost"] = 0.5
        r = choose(w, {0: "swim"})
        self.assertEqual(w.animals[0].energy, 37.5)
        self.assertAlmostEqual(r.rewards[0], -0.25)
        rows = ["........."] * 7
        rows[2] = "..1......"
        c = terrain_configuration(rows=rows)
        w = feedback(World(validate(c), 42))
        w.animals = {0: animal(0, 1, 2)}
        w.config["species"]["prey"]["rewards"]["per_energy_cost"] = -0.1
        r = choose(w, {0: "move_forward"})
        self.assertEqual(w.animals[0].energy, 37)
        self.assertAlmostEqual(r.rewards[0], -0.3)

    def test_failed_feeding_attack_and_swim_get_one_penalty_but_rest_and_turn_do_not(
        self,
    ):
        w = feedback(breeding_world())
        r = choose(w, {0: "eat", 1: "rest"})
        self.assertEqual(r.rewards, {0: -0.05, 1: 0})
        w.animals[1].species = "predator"
        r = choose(w, {0: "turn_right", 1: "attack"})
        self.assertEqual(r.rewards[0], 0)
        self.assertLessEqual(r.rewards[1], -0.05)
        w = feedback(World(validate(terrain_configuration()), 42))
        w.animals = {0: animal(0, 1, 2)}
        r = choose(w, {0: "swim"})
        self.assertEqual(r.rewards[0], -0.05)

    def test_terminal_failed_actions_and_analysis_reconstruct_actual_reward(self):
        w = feedback(breeding_world())
        w.animals[0].energy = 0
        w.animals[0].health = 0.1
        first = w.snapshot()
        r = choose(w, {0: "reproduce"})
        last = w.snapshot()
        self.assertIn(0, r.dead_ids)
        self.assertTrue(
            any(e.kind == "action_failed" and e.cause == "reproduce" for e in r.events)
        )
        result = analyze_run(w.config, [first, last])["species"]["prey"]
        self.assertAlmostEqual(
            result["reward_per_agent_step"], sum(r.rewards.values()) / 2
        )
        self.assertEqual(result["failed_actions"], 1)
        self.assertEqual(result["reproduction_attempts"], 1)
        self.assertEqual(
            summarize(w.config, last, first)["species"]["prey"]["failed_actions"], 1
        )

    def test_birth_feedback_analysis_and_dashboard_match_both_parent_rewards(self):
        w = feedback(breeding_world())
        w.config["species"]["prey"]["body"]["living_cost"] = 1
        first = w.snapshot()
        r = choose(w, {0: "reproduce", 1: "reproduce"})
        last = w.snapshot()
        self.assertAlmostEqual(r.rewards[0], 0.94)
        self.assertAlmostEqual(r.rewards[1], 0.94)
        stats = analyze_run(w.config, [first, last])["species"]["prey"]
        self.assertAlmostEqual(stats["reward_per_agent_step"], 0.94)
        self.assertEqual(stats["reproduction_attempts"], 2)
        self.assertEqual(stats["reproduction_success"], 1)
        telemetry = summarize(w.config, last, first)["species"]["prey"]
        self.assertEqual(telemetry["failed_actions"], 0)
        self.assertEqual(telemetry["reproduction_attempts"], 2)
        self.assertEqual(telemetry["reproduction_success"], 1)

    def test_omitted_feedback_fields_preserve_old_reward_formula(self):
        w = breeding_world()
        r = choose(w, {0: "reproduce", 1: "reproduce"})
        self.assertEqual(r.rewards, {0: 0, 1: 0})
        self.assertEqual(w.animals[0].energy, 55)

    def test_reward_coefficients_and_all_presets_validate(self):
        c = environment("ppo", 0)
        for key, value in (
            ("per_energy_cost", 0.1),
            ("failed_action", 0.1),
            ("per_offspring", -1),
        ):
            invalid = copy.deepcopy(c)
            invalid["species"]["prey"]["rewards"][key] = value
            with self.assertRaisesRegex(ValueError, key):
                validate(invalid)
        for path in (Path(__file__).resolve().parents[1] / "environments").glob(
            "*.json"
        ):
            for spec in load(path)["species"].values():
                self.assertIn("reproduce", spec["allowed_actions"])
                self.assertEqual(spec["rewards"]["failed_action"], -0.05)

    def test_every_algorithm_receives_success_and_failed_consent_rewards(self):
        for kind in KINDS:
            for consent in (True, False):
                c = environment(kind, 0)
                for s in c["species"].values():
                    s["rewards"].update(
                        per_energy_cost=-0.01, failed_action=-0.05, per_offspring=1
                    )
                e = force_reproduction(Experiment(c))
                original_ids = list(e.world.animals)
                if not consent:
                    for i, a in e.world.animals.items():
                        if a.sex == "male":
                            force_action(e.brains[i], "rest")
                with patch.object(
                    e.brains[0], "learn", wraps=e.brains[0].learn
                ) as learned:
                    e.step()
                exp = learned.call_args[0][0]
                self.assertEqual(exp.action, "reproduce", kind)
                parents = {
                    p
                    for event in e.world.events
                    if event.kind == "born"
                    for p in (event.actor, event.mate)
                }
                raw = 0.95 if 0 in parents else -0.05
                self.assertAlmostEqual(e.world.animals[0].last_reward, raw)
                expected = (
                    sum(
                        e.world.animals[i].last_reward
                        for i in original_ids
                        if e.world.animals[i].species == "prey"
                    )
                    / 3
                    if kind == "mappo"
                    else raw
                )
                self.assertAlmostEqual(exp.reward, expected, places=10, msg=kind)
                if not consent:
                    self.assertFalse(parents)

    def test_positive_birth_and_failed_choice_checkpoint_resume_remain_exact(self):
        c = environment("ppo", 1)
        for s in c["species"].values():
            s["rewards"].update(
                per_energy_cost=-0.01, failed_action=-0.05, per_offspring=1
            )
        a = force_reproduction(Experiment(c))
        a.step()
        b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.bin.xz"
            a.write_checkpoint(path)
            z = Experiment.restore(read_checkpoint(path))
        while not a.done:
            a.step()
            b.step()
            z.step()
            self.assertEqual(a.checkpoint(), b.checkpoint())
            self.assertEqual(a.checkpoint(), z.checkpoint())


if __name__ == "__main__":
    unittest.main()
