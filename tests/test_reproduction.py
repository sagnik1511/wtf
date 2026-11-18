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
from ecosystem.replay_io import load_chunk
from ecosystem.runner import Experiment
from ecosystem.world import World
from tests.test_simulation import config, empty_world, animal
from tests.reproduction_support import force_reproduction
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
            mate_radius=10,
        ),
        **overrides,
    )


def breeding_world():
    w = empty_world()
    w.config["species"]["prey"]["reproduction"] = reproduction()
    w.config["species"]["prey"]["body"]["living_cost"] = 0
    w.animals = {
        0: animal(0, "prey", 2, 2, energy=60),
        1: animal(1, "prey", 3, 2, energy=60),
    }
    w.animals[0].sex = "female"
    w.animals[1].sex = "male"
    w.next_animal_id = 2
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
    e = Experiment(c)
    for i, a in e.world.animals.items():
        a.x, a.y = i + 1, 2
        if a.species == "prey":
            a.sex = "female" if i == 0 else "male"
    e.observations = e.world.observe_all()
    return force_reproduction(e)


class BirthMechanicsTests(unittest.TestCase):
    def step(self, w):
        return w.step({i: "reproduce" for i in w.animals})

    def test_birth_state_two_parents_cost_and_transition(self):
        w = breeding_world()
        r = self.step(w)
        born = next(e for e in r.events if e.kind == "born")
        child = w.animals[born.target]
        self.assertEqual((born.actor, born.mate), (0, 1))
        self.assertEqual(
            (child.parent_id, child.second_parent_id, child.age, child.generation),
            (0, 1, 0, 1),
        )
        self.assertIn(child.sex, ("female", "male"))
        self.assertEqual((child.energy, child.health), (60, 30))
        self.assertEqual([w.animals[i].energy for i in (0, 1)], [55, 55])
        self.assertEqual([w.animals[i].last_reproduced_age for i in (0, 1)], [1, 1])
        self.assertEqual(len(w.animals), 3)
        self.assertEqual(set(r.rewards), {0, 1})
        self.assertEqual(set(r.observations), {0, 1, child.id})

    def test_no_asexual_same_sex_or_cross_species_births(self):
        for mode in ("single", "same_sex", "species", "distant", "occluded"):
            w = breeding_world()
            if mode == "single":
                del w.animals[1]
            elif mode == "same_sex":
                w.animals[1].sex = "female"
            elif mode == "species":
                w.animals[1].species = "predator"
            elif mode == "distant":
                w.config["species"]["prey"]["reproduction"]["mate_radius"] = 1
                w.animals[1].x = 5
            else:
                w.animals[1].x = 4
                w.obstacles = {(3, 2)}
            self.assertFalse(any(e.kind == "born" for e in self.step(w).events), mode)
            self.assertEqual(w.animals[0].energy, 60)
            self.assertEqual(w.animals[0].last_reproduced_age, -1)

    def test_both_parents_must_pass_health_age_energy_cooldown(self):
        for index in (0, 1):
            for field, value in (
                ("health", 24),
                ("energy", 5),
                ("age", -1),
                ("last_reproduced_age", 0),
            ):
                w = breeding_world()
                setattr(w.animals[index], field, value)
                self.assertFalse(
                    any(e.kind == "born" for e in self.step(w).events), (index, field)
                )
        w = breeding_world()
        w.animals[1].health = 24.001
        self.assertTrue(any(e.kind == "born" for e in self.step(w).events))

    def test_capacity_and_no_free_space_leave_parents_uncharged(self):
        for mode in ("capacity", "space"):
            w = breeding_world()
            if mode == "capacity":
                w.config["species"]["prey"]["reproduction"]["max_population"] = 2
            else:
                w.obstacles = {
                    (a.x + dx, a.y + dy)
                    for a in w.animals.values()
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                } - {a.position for a in w.animals.values()}
            self.assertFalse(any(e.kind == "born" for e in self.step(w).events))
            for a in w.animals.values():
                self.assertEqual(a.energy, 60)
                self.assertEqual(a.last_reproduced_age, -1)

    def test_one_male_can_only_parent_once_per_step_and_no_newborn_cascade(self):
        w = breeding_world()
        w.animals[2] = animal(2, "prey", 4, 2, energy=60)
        w.next_animal_id = 3
        self.step(w)
        self.assertEqual(w.births["prey"], 1)
        self.assertEqual(len({a.position for a in w.animals.values()}), 4)
        self.assertEqual(sum(a.last_reproduced_age == 1 for a in w.animals.values()), 2)

    def test_post_combat_death_cannot_parent(self):
        w = breeding_world()
        w.animals[1].health = 0.1
        w.animals[1].energy = 0
        r = self.step(w)
        self.assertIn(1, r.dead_ids)
        self.assertFalse(any(e.kind == "born" for e in r.events))

    def test_old_world_restores_sexes_but_cannot_clone(self):
        w = breeding_world()
        del w.animals[1]
        state = json.loads(json.dumps(w.checkpoint()))
        for a in state["snapshot"]["animals"]:
            a.pop("sex")
            a.pop("second_parent_id")
        restored = World.restore(state)
        self.step(restored)
        self.assertEqual(len(restored.animals), 1)
        self.assertEqual(restored.animals[0].sex, "female")


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
        for key, values in e.brains[child.id].weights.items():
            self.assertTrue(
                all(
                    v
                    in (
                        e.brains[child.parent_id].weights[key][j],
                        e.brains[child.second_parent_id].weights[key][j],
                    )
                    for j, v in enumerate(values)
                )
            )
        self.assertEqual(e.brains[child.id].updates, 0)
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
        self.assertGreater(len(births), 0)
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
            frames = load_chunk(store.chunk("births", 1, 0))
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
            ("mate_radius", 0),
            ("mate_radius", 11),
            ("mate_radius", True),
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
