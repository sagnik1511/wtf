"""Opt-in contest mechanics, physical priority, rewards and exact continuation."""

import copy
import json
from pathlib import Path
import unittest
from wtf.config import validate
from wtf.learning.contextual_ppo import action_mask
from wtf.simulation.world import World, Resource
from wtf.inspection.telemetry import summarize
from tests.test_simulation import config, empty_world, animal


def arena(**rules):
    w = empty_world()
    for spec in w.config["species"].values():
        spec["allowed_actions"].append("challenge")
        spec["competition"] = dict(
            enabled=True, damage=5, energy_cost=1, food=True, blocked_path=True, **rules
        )
        spec["body"]["living_cost"] = 0
        spec["body"]["movement_cost"] = 0.2
        spec["rewards"].update(per_energy_cost=-0.1, failed_action=-0.5)
    w.animals = {
        0: animal(0, "prey", 2, 2, heading=1, energy=50, health=30),
        1: animal(1, "prey", 3, 2, heading=3, energy=15, health=20),
    }
    w.next_animal_id = 2
    return w


class CompetitionTests(unittest.TestCase):
    def test_aggression_needs_no_objective_or_facing_and_grants_no_priority(self):
        for position in ((2, 1), (2, 3), (1, 2)):
            with self.subTest(position=position):
                w = arena()
                w.config["species"]["prey"]["competition"].update(
                    food=False, blocked_path=False
                )
                w.animals[1].x, w.animals[1].y = position
                before = [a.position for a in w.animals.values()]
                mask = action_mask(
                    w, w.animals[0], w.config["species"]["prey"]["allowed_actions"]
                )
                self.assertTrue(mask[-1])
                w.step({0: "challenge", 1: "rest"})
                win = next(e for e in w.events if e.kind == "contest_won")
                self.assertEqual(win.cause, "aggression")
                self.assertEqual([a.position for a in w.animals.values()], before)
                self.assertEqual(w.animals[1].health, 15)
                self.assertFalse(
                    any(
                        e.kind in ("contest_moved", "displaced", "food_eaten")
                        for e in w.events
                    )
                )

    def test_food_winner_gets_first_normal_bite_with_configured_damage_and_cost(self):
        w = arena()
        w.resources = {(3, 2): Resource(3, 2, 1)}
        before = [a.energy for a in w.animals.values()]
        result = w.step({0: "challenge", 1: "eat"})
        win = next(e for e in result.events if e.kind == "contest_won")
        self.assertEqual(
            (win.actor, win.target, win.cause, win.amount), (0, 1, "food", 5)
        )
        self.assertEqual(w.animals[1].health, 15)
        self.assertGreater(w.animals[0].energy, before[0] - 1)
        self.assertEqual(w.animals[1].energy, before[1])
        self.assertEqual(w.animals[0].position, (2, 2))
        self.assertFalse(any(e.kind == "kill" for e in result.events))

    def test_path_winner_displaces_only_into_free_cell_and_pays_movement(self):
        w = arena()
        w.step({0: "challenge", 1: "rest"})
        self.assertEqual(w.animals[0].position, (3, 2))
        self.assertEqual(w.animals[1].position, (4, 2))
        self.assertEqual(len({a.position for a in w.animals.values()}), 2)
        self.assertAlmostEqual(w.animals[0].energy, 48.8)
        self.assertAlmostEqual(w.animals[1].energy, 14.8)
        self.assertTrue(any(e.kind == "displaced" for e in w.events))

    def test_enclosed_blocker_takes_damage_but_is_not_swapped_or_teleported(self):
        w = arena()
        w.obstacles.update({(4, 2), (3, 1), (3, 3)})
        w.step({0: "challenge", 1: "rest"})
        self.assertEqual([a.position for a in w.animals.values()], [(2, 2), (3, 2)])
        self.assertEqual(w.animals[1].health, 15)

    def test_defender_can_win_and_keep_its_cell(self):
        w = arena()
        w.animals[0].health = 5
        w.step({0: "challenge", 1: "rest"})
        self.assertNotIn(0, w.animals)
        self.assertEqual(w.animals[1].position, (3, 2))
        self.assertEqual(w.deaths["competition"], 1)
        self.assertEqual(w.animals[1].kills, 0)
        self.assertFalse(any(e.kind == "kill" for e in w.events))
        death = next(e for e in w.events if e.kind == "died")
        self.assertEqual((death.cause, death.position), ("competition", [2, 2]))

    def test_reciprocal_challenges_resolve_one_contest_not_two(self):
        w = arena()
        w.step({0: "challenge", 1: "challenge"})
        self.assertEqual(sum(e.kind == "contest_won" for e in w.events), 1)
        self.assertEqual(sum(e.kind == "challenge_resolved" for e in w.events), 2)
        self.assertEqual(w.animals[1].health, 15)

    def test_another_species_is_not_a_legal_target(self):
        w = arena()
        w.animals[1].species = "predator"
        self.assertEqual(
            action_mask(w, w.animals[0], ["rest", "challenge"]), (True, False)
        )
        hp = w.animals[1].health
        w.step({0: "challenge", 1: "rest"})
        self.assertEqual(w.animals[1].health, hp)
        self.assertAlmostEqual(w.animals[0].last_reward, -0.6)

    def test_food_only_does_not_read_future_consent_in_mask(self):
        w = arena()
        w.config["species"]["prey"]["competition"]["aggression"] = False
        w.config["species"]["prey"]["competition"]["blocked_path"] = False
        w.resources = {(3, 2): Resource(3, 2, 1)}
        checkpoint = copy.deepcopy(w.checkpoint())
        self.assertEqual(
            action_mask(w, w.animals[0], ["rest", "challenge"]), (True, True)
        )
        self.assertEqual(w.checkpoint(), checkpoint)
        w.step({0: "challenge", 1: "rest"})
        self.assertFalse(any(e.kind == "contest_won" for e in w.events))
        self.assertAlmostEqual(w.animals[0].last_reward, -0.6)

    def test_flag_disabled_preserves_physics_and_costs_of_rest(self):
        w = arena()
        w.config["species"]["prey"]["competition"]["enabled"] = False
        before = w.animals[0].health, w.animals[1].health
        w.step({0: "challenge", 1: "rest"})
        self.assertEqual(before, (w.animals[0].health, w.animals[1].health))
        self.assertEqual(w.animals[0].energy, 50)
        self.assertFalse(any(e.kind == "contest_won" for e in w.events))

    def test_checkpoint_continuation_and_recorded_telemetry(self):
        w = arena()
        previous = w.snapshot()
        checkpoint = json.loads(json.dumps(w.checkpoint()))
        other = World.restore(checkpoint)
        actions = {0: "challenge", 1: "eat"}
        w.step(actions)
        other.step(actions)
        self.assertEqual(w.checkpoint(), other.checkpoint())
        frame = w.snapshot()
        metrics = summarize(w.config, frame, previous)["species"]["prey"]
        self.assertEqual(metrics["contest_wins"], 1)
        self.assertEqual(metrics["challenge_attempts"], 1)
        self.assertEqual(metrics["kills"], 0)
        self.assertEqual(metrics["failed_actions"], 1)  # losing eat finds no food

    def test_third_party_challenge_is_not_marked_successful_by_defending(self):
        w = arena()
        w.animals[1].heading = 1
        w.animals[2] = animal(2, "prey", 4, 2, energy=40, health=30)
        w.step({0: "challenge", 1: "challenge", 2: "rest"})
        self.assertEqual(
            [e.actor for e in w.events if e.kind == "challenge_resolved"], [0]
        )
        self.assertEqual(
            summarize(w.config, w.snapshot(), None)["species"]["prey"]["contest_wins"],
            None,
        )

    def test_displacement_cannot_enter_water_or_exceed_climb_limit(self):
        from wtf.simulation.geography import Geography

        w = arena()
        rows = [list("." * w.width) for _ in range(w.height)]
        for x, y, symbol in ((4, 2, "~"), (3, 1, "3"), (3, 3, "#")):
            rows[y][x] = symbol
        definition = dict(
            map_version=1,
            name="contest_limits",
            width=w.width,
            height=w.height,
            rows=["".join(row) for row in rows],
            legend={
                ".": {},
                "~": {"water_depth": 1, "flow": [0, 0]},
                "3": {"elevation": 3},
                "#": {"blocked": True},
            },
        )
        w.config["terrain"] = definition
        w.geography = Geography(definition, w.width, w.height)
        w.obstacles |= w.geography.obstacles
        w.step({0: "challenge", 1: "rest"})
        self.assertEqual([a.position for a in w.animals.values()], [(2, 2), (3, 2)])

    def test_replay_codec_retains_contest_events_without_new_fields(self):
        from wtf.storage.replay_codec import encode_chunk, decode_chunk

        w = arena()
        frames = [w.snapshot()]
        w.step({0: "challenge", 1: "rest"})
        frames.append(w.snapshot())
        self.assertEqual(decode_chunk(encode_chunk(frames)), frames)

    def test_every_controller_can_choose_learn_and_resume_challenge(self):
        from tests.test_mutation import KINDS, environment
        from tests.reproduction_support import force_action
        from wtf.experiment.runner import Experiment

        for kind in KINDS:
            with self.subTest(kind=kind):
                c = environment(kind, 0)
                for spec in c["species"].values():
                    spec["allowed_actions"].append("challenge")
                    spec["competition"] = dict(enabled=True, damage=2, energy_cost=0.2)
                    spec["reproduction"]["enabled"] = False
                e = Experiment(validate(c))
                w = e.world
                w.animals = {
                    0: animal(0, "prey", 1, 1, heading=1, energy=50, health=30),
                    1: animal(1, "prey", 2, 1, heading=3, energy=15, health=20),
                }
                w.resources.clear()
                e.brains = {i: e.brains[i] for i in w.animals}
                force_action(e.brains[0], "challenge")
                force_action(e.brains[1], "rest")
                e.observations = w.observe_all()
                e._refresh_context()
                e.step()
                self.assertTrue(any(x.kind == "contest_won" for x in w.events))
                restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
                e.step()
                restored.step()
                self.assertEqual(e.checkpoint(), restored.checkpoint())

    def test_later_predation_records_actual_displaced_death_position(self):
        w = arena()
        w.animals[2] = animal(2, "predator", 4, 1, heading=2, energy=50, health=50)
        w.config["species"]["predator"]["body"]["attack_damage"] = 100
        w.step({0: "challenge", 1: "rest", 2: "attack"})
        kill = next(e for e in w.events if e.kind == "kill")
        death = next(e for e in w.events if e.kind == "died")
        self.assertEqual((kill.actor, kill.target, kill.position), (2, 1, [4, 2]))
        self.assertEqual((death.cause, death.position), ("predation", [4, 2]))

    def test_schema_configuration_validation_and_opt_in(self):
        c = config()
        c["species"]["prey"]["competition"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "requires challenge"):
            validate(c)
        c["species"]["prey"]["allowed_actions"].append("challenge")
        validate(c)
        c["species"]["prey"]["competition"]["damage"] = -1
        with self.assertRaises(ValueError):
            validate(c)


if __name__ == "__main__":
    unittest.main()
