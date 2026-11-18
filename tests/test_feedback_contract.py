"""The same environment feedback reaches every algorithm on every preset."""

import copy
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from ecosystem.config import load
from ecosystem.runner import Experiment
from tests.reproduction_support import force_action
from tests.test_mutation import KINDS, environment

ROOT = Path(__file__).resolve().parents[1]


class FeedbackContractTests(unittest.TestCase):
    def test_all_presets_and_algorithms_deliver_individual_or_species_mean_feedback(
        self,
    ):
        for path in sorted((ROOT / "environments").glob("*.json")):
            preset = load(path)
            for kind in KINDS:
                with self.subTest(environment=path.stem, algorithm=kind):
                    c = copy.deepcopy(preset)
                    reference = environment(kind, 0)
                    c["brains"] = {}
                    renamed = {}
                    names = {}
                    for old, spec in c["species"].items():
                        # Names deliberately differ from the prey/predator defaults.
                        name = f"animal_{len(renamed)}"
                        names[old] = name
                        template_species = next(
                            s
                            for s in reference["species"].values()
                            if s["diet"] == spec["diet"]
                        )
                        template = copy.deepcopy(
                            reference["brains"][template_species["brain"]]
                        )
                        template["observation_version"] = 2 if "terrain" in c else 1
                        if kind == "grpo":
                            template["train_every_steps"] = 1000
                        if kind == "reinforce":
                            template["max_episode_steps"] = c["world"]["max_steps"]
                        c["brains"][name] = template
                        spec["brain"] = name
                        spec["initial_population"] = 2
                        spec.pop("initial_brain", None)
                        spec.pop("initial_brains", None)
                        renamed[name] = spec
                    c["species"] = renamed
                    if "placement" in c:
                        c["placement"]["species"] = {
                            names[s]: zone
                            for s, zone in c["placement"]["species"].items()
                        }
                    c["experiment"].update(runs=1, world_seed_policy="increment")
                    c["seed"] = 100
                    e = Experiment(c)
                    ids = list(e.world.animals)
                    original_species = {i: e.world.animals[i].species for i in ids}
                    for brain in e.brains.values():
                        force_action(
                            brain, "rest"
                        )  # Available even under action masks.
                    victim = next(
                        i
                        for i in ids
                        if c["species"][original_species[i]]["diet"] == "resource"
                    )
                    for terminal in (False, True):
                        expected = {}
                        if terminal:
                            e.world.animals[victim].energy = 0
                            e.world.animals[victim].health = 0.25
                        e.observations = e.world.observe_all()
                        e._refresh_context()
                        learned = {}
                        with ExitStack() as stack:
                            for i in ids:
                                learned[i] = stack.enter_context(
                                    patch.object(
                                        e.brains[i], "learn", wraps=e.brains[i].learn
                                    )
                                )
                                spec = c["species"][original_species[i]]
                                expected[i] = spec["body"]["living_cost"] * spec[
                                    "rewards"
                                ].get("per_energy_cost", 0)
                                if terminal and i == victim:
                                    damage = min(
                                        0.25, spec["body"]["starvation_damage"]
                                    )
                                    expected[i] += (
                                        damage * spec["rewards"]["per_damage_taken"]
                                    )
                                    if damage == 0.25:
                                        expected[i] += spec["rewards"]["death"]
                            e.step()
                        for i in ids:
                            transition = learned[i].call_args[0][0]
                            self.assertEqual(transition.action, "rest")
                            reward = expected[i]
                            if kind == "mappo":
                                team = [
                                    j
                                    for j in ids
                                    if original_species[j] == original_species[i]
                                ]
                                reward = sum(expected[j] for j in team) / len(team)
                            self.assertAlmostEqual(transition.reward, reward)
                            self.assertEqual(
                                transition.terminated,
                                terminal and i == victim and damage == 0.25,
                            )
                            self.assertFalse(transition.truncated)
                            if transition.terminated:
                                self.assertIsNone(transition.next_observation)
                            else:
                                self.assertAlmostEqual(
                                    e.world.animals[i].last_reward, expected[i]
                                )


if __name__ == "__main__":
    unittest.main()
