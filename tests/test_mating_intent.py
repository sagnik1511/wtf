"""Consent persistence, cancellation, physical gates and stored playback state."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wtf.config import load, validate
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.replay_codec import DeltaEncoder, decode_chunk
from wtf.inspection.telemetry import summarize
from wtf.simulation.world import World
from wtf.experiment.runner import Experiment
from tests.test_reproduction import breeding_world
from tests.test_reproduction_decisions import choose, feedback
from tests.reproduction_support import force_action


def world(steps=5):
    w = feedback(breeding_world())
    w.config["species"]["prey"]["reproduction"].update(
        intent_steps=steps, mate_radius=2
    )
    return w


def failures(result):
    return {e.actor: e.cause for e in result.events if e.kind == "mating_failed"}


class MatingIntentTests(unittest.TestCase):
    def test_masked_experiment_and_binary_resume_keep_delayed_consent(self):
        c = load(Path("tests/fixtures/mating_v3.json"))
        c["world"]["max_steps"] = 20
        c["experiment"].update(runs=1, world_seed_policy="fixed")
        c["obstacles"].update(
            generation="explicit", explicit_cells=[], occupied_fraction=0
        )
        c["species"]["prey"]["initial_population"] = 2
        c["species"]["predator"]["initial_population"] = 0
        c["species"]["prey"]["body"]["living_cost"] = 0
        c["species"]["prey"]["reproduction"].update(
            min_age_steps=1, age_deviation_pct=0
        )
        e = Experiment(c, mode="evaluate")
        for i, sex in ((0, "female"), (1, "male")):
            a = e.world.animals[i]
            a.x, a.y, a.sex = 2 + i, 2, sex
            force_action(e.brains[i], "reproduce" if i == 0 else "rest")
        e.observations = e.world.observe_all()
        e._refresh_context()
        e.step()
        self.assertEqual(e.world.births.get("prey", 0), 0)
        self.assertEqual(e.world.animals[0].last_mating_failure, "awaiting_consent")
        restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "pending.bin.xz"
            e.write_checkpoint(path)
            binary = Experiment.restore(read_checkpoint(path))
        for experiment in (e, restored, binary):
            force_action(experiment.brains[0], "rest")
            force_action(experiment.brains[1], "reproduce")
            experiment.step()
            self.assertEqual(experiment.world.births["prey"], 1)
            self.assertEqual(experiment.world.animals[2].generation, 1)
        self.assertEqual(e.checkpoint(), restored.checkpoint())
        self.assertEqual(e.checkpoint(), binary.checkpoint())

    def test_waiting_cannot_bypass_death_and_reports_terminal_reason(self):
        w = world()
        choose(w, {0: "reproduce"})
        w.animals[0].energy = 0
        w.animals[0].health = 1
        w.config["species"]["prey"]["body"]["living_cost"] = 1
        result = choose(w, {0: "reproduce", 1: "reproduce"})
        self.assertNotIn(0, w.animals)
        self.assertEqual(w.births.get("prey", 0), 0)
        self.assertEqual(failures(result)[0], "parent_died")

    def test_cooldown_failure_is_distinct_and_clears_consumed_consent(self):
        w = world()
        choose(w, {0: "reproduce", 1: "reproduce"})
        result = choose(w, {0: "reproduce", 1: "reproduce"})
        self.assertEqual(failures(result)[0], "cooldown")
        self.assertEqual(w.animals[0].mating_intent_until_tick, -1)

    def test_staggered_choices_create_one_child_and_consume_both_intents(self):
        w = world()
        choose(w, {0: "reproduce"})
        self.assertEqual(w.animals[0].mating_intent_until_tick, 4)
        before = w.snapshot()
        result = choose(w, {1: "reproduce"})
        self.assertEqual(w.births["prey"], 1)
        self.assertEqual((w.animals[0].energy, w.animals[1].energy), (55, 55))
        self.assertTrue(
            all(w.animals[i].mating_intent_until_tick == -1 for i in (0, 1))
        )
        self.assertEqual(failures(result), {})
        # Delayed consent does not invent a same-step success ratio >100%.
        metric = summarize(w.config, w.snapshot(), before)["species"]["prey"]
        self.assertEqual(metric["reproduction_attempts"], 1)
        self.assertIsNone(metric["reproduction_success"])
        choose(w)
        self.assertEqual(w.births["prey"], 1)

    def test_expiration_inclusive_boundary_and_legacy_same_step(self):
        for steps, delay, expected in ((5, 4, 1), (5, 5, 0), (1, 1, 0)):
            w = world(steps)
            choose(w, {0: "reproduce"})
            for _ in range(delay - 1):
                choose(w)
            choose(w, {1: "reproduce"})
            self.assertEqual(w.births.get("prey", 0), expected)

    def test_non_rest_choices_cancel_consent(self):
        for action in ("turn_left", "turn_right", "move_forward", "eat"):
            w = world()
            choose(w, {0: "reproduce"})
            choose(w, {0: action, 1: "reproduce"})
            self.assertEqual(w.births.get("prey", 0), 0, action)
            self.assertEqual(w.animals[0].mating_intent_until_tick, -1)

    def test_low_health_energy_and_distance_invalidate_pending_pair(self):
        for reason in ("low_health", "low_energy", "no_nearby_mate"):
            w = world()
            choose(w, {0: "reproduce"})
            if reason == "low_health":
                w.animals[0].health = 24
            elif reason == "low_energy":
                w.animals[0].energy = 5
            else:
                w.animals[0].x = 6
            result = choose(w, {1: "reproduce"})
            self.assertEqual(w.births.get("prey", 0), 0)
            self.assertEqual(
                failures(result)[1],
                "no_nearby_mate" if reason == "no_nearby_mate" else "mate_not_ready",
            )

    def test_attempt_reports_personal_and_pair_reasons_without_rng_mutation(self):
        for reason in (
            "disabled",
            "immature",
            "low_health",
            "low_energy",
            "population_cap",
            "awaiting_consent",
            "no_nearby_mate",
            "blocked_route",
            "no_birth_space",
        ):
            w = world()
            r = w.config["species"]["prey"]["reproduction"]
            if reason == "disabled":
                r["enabled"] = False
            elif reason == "immature":
                r["min_age_steps"] = 100
            elif reason == "low_health":
                w.animals[0].health = 24
            elif reason == "low_energy":
                w.animals[0].energy = 5
            elif reason == "population_cap":
                r["max_population"] = 2
            elif reason == "no_nearby_mate":
                w.animals[1].x = 6
            elif reason == "blocked_route":
                w.animals[1].x = 4
                w.obstacles.add((3, 2))
            elif reason == "no_birth_space":
                w.obstacles.update(
                    (x, y)
                    for x in range(w.width)
                    for y in range(w.height)
                    if (x, y) not in {(2, 2), (3, 2)}
                )
            actions = {0: "reproduce"}
            if reason == "no_birth_space":
                actions[1] = "reproduce"
            result = choose(w, actions)
            self.assertEqual(failures(result)[0], reason)
            self.assertEqual(w.animals[0].last_mating_failure, reason)
            metrics = summarize(
                w.config, w.snapshot(), dict(w.snapshot(), tick=w.tick - 1)
            )
            self.assertGreaterEqual(
                metrics["species"]["prey"]["mating_failures"][reason], 1
            )
            snapshot = copy.deepcopy(w.checkpoint())
            w.observe_all()
            self.assertEqual(w.checkpoint(), snapshot)

    def test_restore_and_delta_preserve_pending_intent_and_failure(self):
        w = world()
        choose(w, {0: "reproduce"})
        snapshot = w.snapshot()
        restored = World.restore(json.loads(json.dumps(w.checkpoint())))
        _a = choose(w, {1: "reproduce"})
        _b = choose(restored, {1: "reproduce"})
        self.assertEqual(w.checkpoint(), restored.checkpoint())
        encoder = DeltaEncoder()
        payload = dict(
            encoder.header(),
            frames=[encoder.encode(snapshot), encoder.encode(w.snapshot())],
        )
        self.assertEqual(decode_chunk(payload), [snapshot, w.snapshot()])

    def test_preset_is_masked_v3_with_proximal_radius(self):
        c = load(Path("tests/fixtures/mating_v3.json"))
        for spec in c["species"].values():
            self.assertEqual(spec["reproduction"]["mate_radius"], 2)
            self.assertEqual(spec["reproduction"]["intent_steps"], 5)
            brain = c["brains"][spec["brain"]]
            self.assertEqual(
                (brain["type"], brain["observation_version"]), ("maskable_ppo", 3)
            )
        for invalid in (0, 129, 2.5, True):
            changed = copy.deepcopy(c)
            changed["species"]["prey"]["reproduction"]["intent_steps"] = invalid
            with self.assertRaises(ValueError):
                validate(changed)


if __name__ == "__main__":
    unittest.main()
