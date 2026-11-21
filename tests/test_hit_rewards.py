import json
import unittest
from pathlib import Path

from wtf.config import validate
from wtf.experiment.runner import Experiment
from tests.test_simulation import animal, config, empty_world


class HitRewardTests(unittest.TestCase):
    def make_encounter(self, health=30):
        w = empty_world()
        w.animals = {
            0: animal(0, "prey", 2, 2, health=health),
            1: animal(1, "predator", 1, 2, 1),
        }
        return w

    def test_hit_rewards_predator_and_penalizes_prey(self):
        w = self.make_encounter()
        result = w.step({0: "rest", 1: "attack"})
        self.assertAlmostEqual(result.rewards[1], 0.5)
        self.assertAlmostEqual(result.rewards[0], -2.0)
        self.assertEqual(w.animals[0].health, 20)
        self.assertAlmostEqual(w.animals[1].energy, 18.85)  # No food from a hit.
        self.assertEqual(dict(w.kills), {})
        dealt = [e for e in result.events if e.kind == "damage_dealt"]
        self.assertEqual([(e.actor, e.target, e.amount) for e in dealt], [(1, 0, 10)])

    def test_missed_attack_has_no_positive_feedback(self):
        w = self.make_encounter()
        w.animals[1].heading = 3
        result = w.step({0: "rest", 1: "attack"})
        self.assertEqual(result.rewards[1], 0)
        self.assertFalse(any(e.kind == "damage_dealt" for e in result.events))

    def test_lethal_hit_capped_at_remaining_health_and_keeps_food_reward(self):
        w = self.make_encounter(health=3)
        result = w.step({0: "rest", 1: "attack"})
        self.assertAlmostEqual(result.rewards[1], 0.15 + 4.0)
        self.assertEqual(result.dead_ids, {0})
        self.assertEqual(w.kills["predator"], 1)
        self.assertEqual(
            sum(e.amount for e in result.events if e.kind == "damage_dealt"), 3
        )
        self.assertEqual(
            sum(e.amount for e in result.events if e.kind == "energy_gained"), 40
        )

    def test_simultaneous_overkill_splits_credit_without_duplicate_food(self):
        w = self.make_encounter(health=3)
        w.animals[2] = animal(2, "predator", 3, 2, 3)
        result = w.step({0: "rest", 1: "attack", 2: "attack"})
        dealt = [e for e in result.events if e.kind == "damage_dealt"]
        self.assertEqual([e.amount for e in dealt], [1.5, 1.5])
        self.assertAlmostEqual(result.rewards[1] + result.rewards[2], 0.15 + 4.0)
        self.assertEqual(sum(e.kind == "kill" for e in result.events), 1)
        self.assertEqual(w.kills["predator"], 1)

    def test_different_strengths_get_proportional_credit(self):
        w = self.make_encounter(health=15)
        w.config["species"]["predator"]["body"]["attack_damage"] = 5
        stronger = json.loads(json.dumps(w.config["species"]["predator"]))
        stronger["body"]["attack_damage"] = 20
        w.config["species"]["stronger"] = stronger
        w.animals[2] = animal(2, "stronger", 3, 2, 3)
        result = w.step({0: "rest", 1: "attack", 2: "attack"})
        credit = {e.actor: e.amount for e in result.events if e.kind == "damage_dealt"}
        self.assertEqual(credit, {1: 3.0, 2: 12.0})

    def test_reward_is_configurable_and_old_configs_remain_compatible(self):
        for coefficient in (None, 0, 0.1):
            w = self.make_encounter()
            r = w.config["species"]["predator"]["rewards"]
            if coefficient is None:
                del r["per_damage_dealt"]
            else:
                r["per_damage_dealt"] = coefficient
            validate(w.config)
            result = w.step({0: "rest", 1: "attack"})
            self.assertAlmostEqual(result.rewards[1], 10 * (coefficient or 0))
        c = config()
        c["species"]["predator"]["rewards"]["per_damage_dealt"] = -1
        with self.assertRaisesRegex(ValueError, "per_damage_dealt"):
            validate(c)

    def test_runner_delivers_hit_reward_to_only_attacking_individual(self):
        e = Experiment(config())
        e.world = self.make_encounter()
        e.world.animals[3] = e.world.animals.pop(1)
        e.world.animals[3].id = 3  # Controller 3 is the actual predator brain.
        e.observations = e.world.observe_all()
        # Fix actions to isolate reward delivery from policy exploration.
        e.brains[0].choose = lambda *args, **kwargs: "rest"
        e.brains[3].choose = lambda *args, **kwargs: "attack"
        e.step()
        e.step()  # Cross the learning threshold and train on attack experiences.
        self.assertAlmostEqual(e.brains[3].replay[-1].reward, 0.5)
        self.assertAlmostEqual(e.brains[0].replay[-1].reward, -2.0)
        self.assertEqual(e.brains[2].replay, [])
        self.assertEqual(e.brains[3].updates, 1)
        self.assertEqual(e.brains[1].replay, [])
        self.assertFalse(e.brains[3].replay[-1].terminated)


if __name__ == "__main__":
    unittest.main()
