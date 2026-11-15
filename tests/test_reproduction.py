"""Birth mechanics, learned-state inheritance and exact continuation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ecosystem.brain import Experience
from ecosystem.brains import inherit_brain
from ecosystem.config import validate
from ecosystem.grouped_rollouts import collect_group
from ecosystem.recording import Recorder, RecordingStore
from ecosystem.runner import Experiment
from ecosystem.world import World
from tests.test_simulation import config, empty_world, animal
from tests.test_ppo import ppo_environment
from tests.test_policy_algorithms import environment


def reproduction(**overrides):
    return dict(
        dict(
            enabled=True,
            health_threshold=24,
            min_age_steps=1,
            cooldown_steps=3,
            min_energy=10,
            energy_cost=5,
            max_population=8,
        ),
        **overrides,
    )


def breeding_world():
    w = empty_world()
    w.config["species"]["prey"]["reproduction"] = reproduction()
    w.config["species"]["prey"]["body"]["living_cost"] = 0
    w.animals = {0: animal(0, "prey", 2, 2, energy=60)}
    return w


def breeding_experiment(kind="q_network"):
    c = (
        config()
        if kind == "q_network"
        else ppo_environment() if kind == "ppo" else environment(kind)
    )
    c["obstacles"].update(generation="explicit", explicit_cells=[])
    c["experiment"]["record_metrics_every_steps"] = 1
    for spec in c["species"].values():
        spec["reproduction"] = reproduction()
        spec["body"].update(living_cost=0, movement_cost=0)
        if spec["diet"] == "prey":
            spec["body"].update(attack_damage=0, attack_cost=0)
    return Experiment(c)


class BirthMechanicsTests(unittest.TestCase):
    def test_birth_state_lineage_placement_cost_and_transition(self):
        w = breeding_world()
        result = w.step({0: "rest"})
        born = [e for e in result.events if e.kind == "born"]
        self.assertEqual(len(born), 1)
        child = w.animals[born[0].target]
        self.assertEqual(born[0].actor, 0)
        self.assertEqual(
            (child.species, child.parent_id, child.generation, child.age),
            ("prey", 0, 1, 0),
        )
        self.assertEqual((child.energy, child.health), (60, 30))
        self.assertEqual(abs(child.x - 2) + abs(child.y - 2), 1)
        self.assertEqual(w.animals[0].energy, 55)
        self.assertEqual(w.animals[0].last_reproduced_age, 1)
        self.assertEqual(w.metrics()["births"], {"prey": 1})
        self.assertEqual(set(result.rewards), {0})
        self.assertEqual(set(result.observations), {0, child.id})
        self.assertEqual(len(w.animals), 2)  # No newborn cascade in birth step.

    def test_strict_health_threshold(self):
        for health, count in ((24, 1), (23, 1), (24.001, 2)):
            w = breeding_world()
            w.animals[0].health = health
            w.step({0: "rest"})
            with self.subTest(health=health):
                self.assertEqual(len(w.animals), count)

    def test_age_energy_and_enabled_gates(self):
        for change in ({"min_age_steps": 2}, {"min_energy": 61}, {"enabled": False}):
            w = breeding_world()
            w.config["species"]["prey"]["reproduction"].update(change)
            w.step({0: "rest"})
            with self.subTest(change=change):
                self.assertEqual(len(w.animals), 1)
        w = breeding_world()
        w.animals[0].energy = 5
        w.config["species"]["prey"]["reproduction"]["min_energy"] = 5
        w.step({0: "rest"})
        self.assertEqual(len(w.animals), 1)  # Cost cannot leave a parent starving.

    def test_cooldown_and_child_maturity(self):
        w = breeding_world()
        w.config["species"]["prey"]["reproduction"]["min_age_steps"] = 3
        w.animals[0].age = 2
        w.step({0: "rest"})
        for _ in range(2):
            w.step({i: "rest" for i in w.animals})
            self.assertEqual(len(w.animals), 2)
        w.step({i: "rest" for i in w.animals})
        self.assertEqual(len(w.animals), 4)
        self.assertEqual(sorted(a.generation for a in w.animals.values()), [0, 1, 1, 2])

    def test_capacity_and_blocking_do_not_charge_or_start_cooldown(self):
        for blocked in ("cap", "obstacles", "animals"):
            w = breeding_world()
            adjacent = [(2, 1), (3, 2), (2, 3), (1, 2)]
            if blocked == "cap":
                w.config["species"]["prey"]["reproduction"]["max_population"] = 1
            elif blocked == "obstacles":
                w.obstacles = set(adjacent)
            else:
                for i, (x, y) in enumerate(adjacent, 1):
                    w.animals[i] = animal(i, "predator", x, y)
            w.step({i: "rest" for i in w.animals})
            with self.subTest(blocked=blocked):
                self.assertEqual(w.animals[0].energy, 60)
                self.assertEqual(w.animals[0].last_reproduced_age, -1)
                self.assertEqual(dict(w.births), {})

    def test_contention_respects_cap_and_no_overlaps(self):
        w = breeding_world()
        w.animals[1] = animal(1, "prey", 3, 2, energy=60)
        w.config["species"]["prey"]["reproduction"]["max_population"] = 3
        w.step({0: "rest", 1: "rest"})
        self.assertEqual(len(w.animals), 3)
        self.assertEqual(len({a.position for a in w.animals.values()}), 3)
        self.assertEqual(w.births["prey"], 1)

    def test_post_combat_and_starvation_health_decides(self):
        for health, energy in ((0.1, 0), (30, 60)):
            w = breeding_world()
            w.animals[0].health = health
            w.animals[0].energy = energy
            actions = {0: "rest"}
            if energy:
                w.animals[1] = animal(1, "predator", 1, 2, heading=1)
                actions[1] = "attack"  # Ten damage drops prey below threshold.
            result = w.step(actions)
            self.assertFalse(any(e.kind == "born" for e in result.events))

    def test_restore_old_world_and_absent_config(self):
        w = breeding_world()
        del w.config["species"]["prey"]["reproduction"]
        state = json.loads(json.dumps(w.checkpoint()))
        del state["next_animal_id"]
        del state["snapshot"]["metrics"]["births"]
        for a in state["snapshot"]["animals"]:
            for key in ("parent_id", "generation", "last_reproduced_age"):
                del a[key]
        restored = World.restore(state)
        restored.step({0: "rest"})
        self.assertEqual(len(restored.animals), 1)
        self.assertEqual(restored.animals[0].generation, 0)


class InheritanceTests(unittest.TestCase):
    def test_all_algorithms_copy_learned_state_independently(self):
        for kind in ("q_network", "ppo", "a2c", "reinforce", "grpo"):
            with self.subTest(kind=kind):
                e = breeding_experiment(kind)
                parent = e.brains[0]
                observation = e.observations[0]
                # Populate optimizer state and an unfinished lifetime buffer.
                for _ in range(4):
                    action = parent.choose(observation)
                    parent.learn(Experience(observation, action, 1, observation, False))
                before = parent.checkpoint()
                child = inherit_brain(parent, 123456)
                self.assertEqual(parent.checkpoint(), before)
                self.assertEqual(child.steps, parent.steps)
                self.assertEqual(child.updates, parent.updates)
                self.assertNotEqual(child.rng.getstate(), parent.rng.getstate())
                self.assertNotEqual(
                    child.training_rng.getstate(), parent.training_rng.getstate()
                )
                for name in ("replay", "rollout", "trajectory"):
                    if hasattr(child, name):
                        self.assertEqual(getattr(child, name), [])
                self.assertIsNone(getattr(child, "pending", None))
                for key in (
                    "weights",
                    "target",
                    "network",
                    "moments",
                    "variances",
                    "epsilon",
                    "baseline",
                    "reference_network",
                ):
                    if key in before:
                        self.assertEqual(child.checkpoint()[key], before[key])
                if kind == "q_network":
                    child.weights["b2"][0] += 1
                else:
                    child.network.layers[0]["bias"][0] += 1
                self.assertEqual(parent.checkpoint(), before)

    def test_runner_inherits_after_feedback_and_child_learns_next_step(self):
        e = breeding_experiment()
        for brain in e.brains.values():
            brain.config.update(learning_starts_after=1, batch_size=1)
        e.step()
        child = next(a for a in e.world.animals.values() if a.parent_id == 0)
        self.assertEqual(e.brains[child.id].weights, e.brains[0].weights)
        self.assertEqual(e.brains[child.id].replay, [])
        self.assertEqual(child.age, 0)
        e.step()
        self.assertEqual(child.age, 1)
        self.assertEqual(len(e.brains[child.id].replay), 1)

    def test_births_do_not_inflate_total_training_updates(self):
        e = breeding_experiment()
        for brain in e.brains.values():
            brain.updates = 7
            brain.config["learning_starts_after"] = 512
        before = e.total_updates
        e.step()  # No optimizer update on the first step of these Q brains.
        self.assertEqual(e.total_updates, before)
        self.assertGreater(len(e.brains), 4)
        restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
        self.assertEqual(restored.total_updates, before)
        descendant = next(i for i in e.brains if i >= 4)
        e.brains[descendant].updates += 2
        e.world.animals[descendant].energy = 0
        e.world.animals[descendant].health = 0.1
        e.observations = e.world.observe_all()
        e.step()
        self.assertEqual(e.total_updates, before + 2)
        self.assertNotIn(descendant, e.inherited_updates)

    def test_exact_checkpoint_with_births_all_algorithms(self):
        for kind in ("q_network", "ppo", "a2c", "reinforce", "grpo"):
            with self.subTest(kind=kind):
                a = breeding_experiment(kind)
                a.step()
                a.step()
                b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
                for _ in range(7):
                    a.step()
                    b.step()
                self.assertEqual(a.checkpoint(), b.checkpoint())
                ids = list(a.world.animals) + [x["id"] for x in a.world.lifespans]
                self.assertGreaterEqual(a.world.next_animal_id, max(ids) + 1)

    def test_dead_child_released_ids_not_reused_next_run_is_fresh(self):
        e = breeding_experiment()
        e.step()
        child_id = next(i for i, a in e.world.animals.items() if a.parent_id == 0)
        next_id = e.world.next_animal_id
        e.world.animals[child_id].energy = 0
        e.world.animals[child_id].health = 0.1
        e.observations = e.world.observe_all()
        e.step()
        self.assertNotIn(child_id, e.brains)
        self.assertGreaterEqual(e.world.next_animal_id, next_id)
        founding_brain = e.brains[0]
        while not e.finished_run:
            e.step()
        e.next_run()
        self.assertEqual(set(e.brains), {0, 1, 2, 3})
        self.assertIsNot(e.brains[0], founding_brain)
        self.assertTrue(all(a.parent_id == -1 for a in e.world.animals.values()))

    def test_grpo_trials_handle_multigeneration_births_without_mutation(self):
        e = breeding_experiment("grpo")
        focal = e.brains[0]
        focal.config["rollout_horizon"] = 4
        before = e.world.checkpoint()
        others = {i: b.checkpoint() for i, b in e.brains.items() if i != 0}
        births = []
        original = World._reproduce

        def track(world, events):
            original(world, events)
            births.extend(event for event in events if event.kind == "born")

        with patch.object(World, "_reproduce", track):
            group = collect_group(e.world, e.brains, 0)
        self.assertGreater(len(births), 4)
        self.assertEqual(len(group[0]), 4)
        self.assertEqual(e.world.checkpoint(), before)
        self.assertEqual(
            {i: b.checkpoint() for i, b in e.brains.items() if i != 0}, others
        )

    def test_recording_has_children_birth_events_and_diagnostics(self):
        e = breeding_experiment()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "births"
            recorder = Recorder(folder, e.config, e.mode)
            recorder.begin_run(e)
            e.step()
            recorder.capture(e, force=True)
            recorder.close(completed=False)
            store = RecordingStore(directory)
            frames = json.loads(store.chunk("births", 1, 0).read_text())
            newborn = next(a for a in frames[-1]["animals"] if a["parent_id"] == 0)
            self.assertIn(str(newborn["id"]), frames[-1]["diagnostics"])
            self.assertGreater(frames[-1]["metrics"]["births"]["prey"], 0)
            self.assertTrue(
                any(
                    event["target"] == newborn["id"]
                    for event in frames[-1]["events"]
                    if event["kind"] == "born"
                )
            )


class ReproductionConfigTests(unittest.TestCase):
    def test_reject_invalid_reproduction(self):
        for key, value in (
            ("enabled", 1),
            ("health_threshold", 31),
            ("health_threshold", -1),
            ("min_age_steps", 0),
            ("cooldown_steps", 0),
            ("energy_cost", 0),
            ("min_energy", 4),
            ("max_population", 1.5),
            ("typo", 1),
        ):
            c = config()
            c["species"]["prey"]["reproduction"] = reproduction()
            c["species"]["prey"]["reproduction"][key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(ValueError, key),
            ):
                validate(c)


if __name__ == "__main__":
    unittest.main()
