"""Local escape physics, observable consent and calibrated preset lifecycle."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ecosystem import kernels
from ecosystem.analyze import analyze_run
from ecosystem.brain import observation_inputs
from ecosystem.brains import (
    model_signature,
    model_snapshot,
    load_model_weights,
    create_brain,
)
from ecosystem.checkpoint_io import read_checkpoint
from ecosystem.config import load, validate
from ecosystem.contextual_ppo import action_mask
from ecosystem.recording import Recorder
from ecosystem.replay_io import chunk_path, load_chunk
from ecosystem.runner import Experiment
from ecosystem.world import Animal, World
from tests.test_mating_intent import world as breeding_world, choose
from tests.test_mutation import environment, KINDS
from tests.test_geography import configuration as terrain_config
from tests.reproduction_support import force_action

ESCAPES = ("move_backward", "strafe_left", "strafe_right")


def world():
    c = load(Path("tests/fixtures/consent_v4_nearest.json"))
    w = World(c, 100)
    w.obstacles.clear()
    w.resources.clear()
    w.animals = {0: Animal(0, "prey", 3, 3, 0, 60, 40)}
    w.next_animal_id = 1
    return w


class EscapeTests(unittest.TestCase):
    def test_moves_preserve_heading_and_charge_walking_effort(self):
        offsets = ((0, 1), (-1, 0), (1, 0))
        for heading in range(4):
            for action, (dx, dy) in zip(ESCAPES, offsets):
                w = world()
                a = w.animals[0]
                a.heading = heading
                for _ in range(heading):
                    dx, dy = -dy, dx
                w.step({0: action})
                self.assertEqual(a.position, (3 + dx, 3 + dy))
                self.assertEqual(a.heading, heading)
                self.assertAlmostEqual(a.energy, 59.8)
                self.assertAlmostEqual(a.last_reward, -0.002)
                self.assertEqual(
                    next(e.cause for e in w.events if e.kind == "moved"), action
                )

    def test_escape_can_evade_an_attack_without_turning(self):
        for action in ESCAPES:
            w = world()
            w.animals[1] = Animal(1, "predator", 3, 2, 2, 80, 55)
            w.next_animal_id = 2
            w.step({0: action, 1: "attack"})
            self.assertEqual(w.animals[0].health, 40)
            self.assertEqual(w.animals[0].heading, 0)

    def test_blocked_escape_costs_energy_and_receives_failure_feedback(self):
        w = world()
        w.obstacles.add((2, 3))
        before = w.snapshot()
        w.step({0: "strafe_left"})
        self.assertEqual(w.animals[0].position, (3, 3))
        self.assertAlmostEqual(w.animals[0].energy, 59.8)
        self.assertAlmostEqual(w.animals[0].last_reward, -0.052)
        self.assertEqual(w.events[0].cause, "strafe_left:obstacle")
        report = analyze_run(w.config, iter((before, w.snapshot())))["species"]["prey"]
        self.assertEqual(report["actions"]["strafe_left"], 1)
        self.assertEqual(report["movement_success"], 0)
        self.assertAlmostEqual(report["energy_demand"], 0.2)

    def test_initial_occupancy_prevents_swaps_and_contention_still_has_one_winner(self):
        w = world()
        w.animals[1] = Animal(1, "prey", 4, 3, 0, 60, 40)
        w.step({0: "strafe_right", 1: "strafe_left"})
        self.assertEqual([a.position for a in w.animals.values()], [(3, 3), (4, 3)])
        w.animals[1].x = 5
        w.step({0: "strafe_right", 1: "strafe_left"})
        self.assertEqual(sum(a.position == (4, 3) for a in w.animals.values()), 1)
        self.assertTrue(any(e.cause.endswith("competing_claim") for e in w.events))

    def test_side_and_backward_moves_cannot_bypass_water_or_cliffs(self):
        c = terrain_config(rows=["........."] * 7)
        for s in c["species"].values():
            s["allowed_actions"].extend(ESCAPES)
        w = World(validate(c), 42)
        w.animals = {0: Animal(0, "prey", 3, 3, 0, 60, 30)}
        for action, symbol, reason in (
            ("strafe_left", "~", "requires_swim"),
            ("move_backward", "3", "slope"),
        ):
            rows = ["........."] * 7
            target = w.movement_destination(w.animals[0], action)
            row = list(rows[target[1]])
            row[target[0]] = symbol
            rows[target[1]] = "".join(row)
            wc = terrain_config(rows=rows)
            for s in wc["species"].values():
                s["allowed_actions"].extend(ESCAPES)
            test = World(validate(wc), 42)
            test.animals = copy.deepcopy(w.animals)
            self.assertFalse(action_mask(test, test.animals[0], [action, "rest"])[0])
            test.step({0: action})
            self.assertEqual(test.animals[0].position, (3, 3))
            self.assertEqual(test.events[0].cause, action + ":" + reason)

    def test_python_and_native_effort_match_and_stale_native_falls_back(self):
        for action in ESCAPES:
            original = world()
            reference = World.restore(original.checkpoint())
            original.step({0: action})
            with patch.object(kernels, "native", None):
                reference.step({0: action})
            self.assertEqual(original.checkpoint(), reference.checkpoint())
            if kernels.native is not None:
                stale = world()
                with patch.object(kernels.native, "escape_movement", 0, create=True):
                    stale.step({0: action})
                self.assertEqual(stale.checkpoint(), reference.checkpoint())


class VisibleConsentTests(unittest.TestCase):
    def test_v4_binary_checkpoint_and_recorded_observations_preserve_consent(self):
        c = load(Path("tests/fixtures/consent_v4_nearest.json"))
        c["species"]["prey"]["initial_population"] = 2
        c["species"]["predator"]["initial_population"] = 0
        c["species"]["prey"]["reproduction"].update(
            min_age_steps=1, age_deviation_pct=0
        )
        c["experiment"].update(runs=1, world_seed_policy="fixed")
        c["world"]["max_steps"] = 4
        c["recording"]["diagnostics_every_steps"] = 1
        e = Experiment(c, mode="evaluate")
        e.world.obstacles.clear()
        for i in (0, 1):
            a = e.world.animals[i]
            a.x, a.y, a.heading = 3 + i, 3, 0
            a.sex = "female" if i == 0 else "male"
            force_action(e.brains[i], "reproduce" if i == 0 else "rest")
        e.observations = e.world.observe_all()
        e._refresh_context()
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(folder, e.config, e.mode)
            recorder.begin_run(e)
            e.step()
            recorder.capture(e)
            path = Path(folder) / "state.bin.xz"
            e.write_checkpoint(path)
            resumed = Experiment.restore(read_checkpoint(path))
            for x in (e, resumed):
                force_action(x.brains[0], "strafe_left")
                force_action(x.brains[1], "reproduce")
                x.step()
            self.assertEqual(e.checkpoint(), resumed.checkpoint())
            recorder.capture(e)
            recorder.close(completed=True)
            frames = load_chunk(chunk_path(Path(folder) / "playback", 1, 0))
            self.assertEqual(frames[-1]["animals"], e.world.snapshot()["animals"])
            self.assertEqual(len(frames[1]["diagnostics"]["1"]["features"]), 69)
            self.assertEqual(frames[1]["diagnostics"]["1"]["features"][-2:], [0.8, 1.0])

    def test_version_three_weights_cannot_silently_load_into_four(self):
        c = load(Path("tests/fixtures/consent_v4_nearest.json"))
        s = c["species"]["predator"]
        new = c["brains"][s["brain"]]
        old = copy.deepcopy(new)
        old["observation_version"] = 3
        prior = create_brain(old, s["allowed_actions"], 1, 2)
        current = create_brain(new, s["allowed_actions"], 1, 2)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_model_weights(current, model_snapshot(prior))

    def test_local_intent_expires_and_observation_reads_do_not_advance_state(self):
        w = breeding_world()
        for b in w.config["brains"].values():
            b["observation_version"] = 4
        self.assertEqual(w.observe_all()[1][-3:], (0, 0, 0))
        choose(w, {0: "reproduce"})
        state = copy.deepcopy(w.checkpoint())
        values = w.observe_all()
        self.assertEqual(values[1][-3:], (0, 0.8, 1))
        self.assertEqual(values[0][-3:], (0.8, 0, 0))
        self.assertEqual(w.checkpoint(), state)
        for _ in range(4):
            choose(w)
        self.assertEqual(w.observe_all()[1][-3:], (0, 0, 0))

    def test_invisible_or_distant_intent_is_not_leaked(self):
        w = breeding_world()
        for b in w.config["brains"].values():
            b["observation_version"] = 4
        choose(w, {0: "reproduce"})
        w.animals[1].x = 4
        w.obstacles.add((3, 2))
        self.assertEqual(w.observe_all()[1][-2:], (0, 0))
        w.obstacles.clear()
        w.animals[1].x, w.animals[1].y = 9, 7
        self.assertEqual(w.observe_all()[1][-2:], (0, 0))

    def test_ready_consenting_neighbor_is_detected_even_when_not_nearest(self):
        w = breeding_world()
        for b in w.config["brains"].values():
            b["observation_version"] = 4
        w.animals[2] = Animal(
            2, "prey", 4, 2, 0, 60, 30, age=10, sex="male", mating_intent_until_tick=4
        )
        w.next_animal_id = 3
        # The nearer male at (3,2) has not consented; the farther one has.
        self.assertEqual(w.observe_all()[0][-3:], (0, 0, 1))

    def test_escape_cancels_intent(self):
        for action in ESCAPES:
            w = breeding_world()
            w.config["species"]["prey"]["allowed_actions"].extend(ESCAPES)
            choose(w, {0: "reproduce"})
            choose(w, {0: action, 1: "reproduce"})
            self.assertEqual(w.births.get("prey", 0), 0)
            self.assertEqual(w.animals[0].mating_intent_until_tick, -1)

    def test_all_algorithms_accept_v4_escape_actions_and_exact_resume(self):
        for kind in KINDS:
            c = environment(kind, 0)
            for b in c["brains"].values():
                b["observation_version"] = 4
            for s in c["species"].values():
                s["allowed_actions"].extend(ESCAPES)
                s["reproduction"]["intent_steps"] = 5
            e = Experiment(validate(c))
            e.step()
            restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
            while not e.done:
                e.step()
                restored.step()
                self.assertEqual(e.checkpoint(), restored.checkpoint(), kind)

    def test_common_preset_trains_and_resumes_with_current_observation_contract(self):
        c = load(Path("environments/maskable_ppo.json"))
        c["world"]["max_steps"] = 6
        c["experiment"].update(runs=1, world_seed_policy="fixed")
        e = Experiment(c)
        for i, animal in e.world.animals.items():
            expected = 69 if animal.species == "prey" else 153
            self.assertEqual(len(e.observations[i]), expected)
            self.assertEqual(observation_inputs(e.brains[i].config), expected)
        e.step()
        e.step()
        restored = Experiment.restore(e.checkpoint())
        while not e.done:
            e.step()
            restored.step()
        self.assertGreater(e.total_updates, 0)
        self.assertEqual(e.checkpoint(), restored.checkpoint())


if __name__ == "__main__":
    unittest.main()
