"""Numerical objectives, information boundaries and full algorithm lifecycles."""

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecosystem.brain import Experience, INPUTS
from ecosystem.brains import (
    create_brain,
    inherit_brain,
    restore_brain,
    model_snapshot,
    load_model_weights,
)
from ecosystem.checkpoint_io import read_checkpoint
from ecosystem.config import load, validate
from ecosystem.contextual_ppo import action_mask, central_contexts
from ecosystem.ppo import policy_distribution
from ecosystem.qrdqn import quantile_huber
from ecosystem.recording import Recorder
from ecosystem.replay_io import load_chunk, chunk_path
from ecosystem.runner import Experiment
from tests.test_simulation import config, empty_world, animal

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("recurrent_ppo", "qrdqn", "maskable_ppo", "mappo")


def settings(kind):
    c = copy.deepcopy(
        load(ROOT / f"environments/{kind}.json")["brains"]["prey_" + kind]
    )
    # Numerical unit fixtures keep their fixed 20-input contract as presets evolve.
    c.update(observation_version=1, observation_top_k=1)
    c["network"]["hidden_layers"] = [3]
    if kind == "qrdqn":
        c.update(
            num_quantiles=3,
            replay_capacity=8,
            batch_size=2,
            learning_starts_after=2,
            train_every_steps=1,
            target_update_every_training_steps=2,
        )
    else:
        c.update(rollout_steps=4, epochs=2, batch_size=4, target_kl=0)
        if kind == "recurrent_ppo":
            c.update(recurrent_size=2, sequence_length=2)
        elif kind == "mappo":
            c["critic_hidden_layers"] = [3]
    return c


def environment(kind):
    c = config()
    c["brains"] = {"test": settings(kind)}
    for s in c["species"].values():
        s["brain"] = "test"
        s["champion_selection"] = {"criterion": "lifespan"}
    c["world"]["max_steps"] = 9
    c["experiment"]["runs"] = 2
    return c


def roundtrip(value):
    return json.loads(json.dumps(value, allow_nan=False))


class AdvancedMathTests(unittest.TestCase):
    def test_quantile_huber_gradient(self):
        values, targets = [-0.3, 0.2, 1.7], [-0.6, 0.8, 2.15]
        for kappa in (0.4, 1.0):
            _, gradient = quantile_huber(values, targets, kappa)
            for i, analytic in enumerate(gradient):
                plus, minus = values[:], values[:]
                plus[i] += 1e-6
                minus[i] -= 1e-6
                numeric = (
                    quantile_huber(plus, targets, kappa)[0]
                    - quantile_huber(minus, targets, kappa)[0]
                ) / 2e-6
                self.assertAlmostEqual(analytic, numeric, places=7)

    def test_gru_two_step_backprop_matches_numerical_gradient(self):
        b = create_brain(settings("recurrent_ppo"), ["rest", "eat"], 7, 4)
        network = b.network
        x1 = tuple((i - 10) / 20 for i in range(INPUTS))
        x2 = tuple((7 - i) / 21 for i in range(INPUTS))
        initial = [0.13, -0.2]
        gradient = [0.4, -0.7, 0.2]
        _, hidden, tape1 = network.step(x1, initial, tape=True)
        _, _, tape2 = network.step(x2, hidden, tape=True)
        gradients = network.zeros()
        future = network.backward(tape2, gradient, [0.0, 0.0], gradients)
        network.backward(tape1, [0.0] * 3, future, gradients)

        def objective():
            _, h = network.step(x1, initial)
            outputs, _ = network.step(x2, h)
            return sum(a * b for a, b in zip(outputs, gradient))

        for li, layer in enumerate(network.layers):
            for key, values in layer.items():
                for i in range(len(values)):
                    old = values[i]
                    values[i] = old + 1e-6
                    upper = values[i]
                    plus = objective()
                    values[i] = old - 1e-6
                    lower = values[i]
                    minus = objective()
                    values[i] = old
                    self.assertAlmostEqual(
                        gradients[li][key][i],
                        (plus - minus) / (upper - lower),
                        places=7,
                        msg=f"{li}.{key}.{i}",
                    )

    def test_masked_objective_and_entropy_gradient(self):
        b = create_brain(
            settings("maskable_ppo"), ["rest", "eat", "move_forward"], 1, 2
        )
        b.mask = (True, False, True)
        observation = b._pack((0.0,) * INPUTS)
        from ecosystem.ppo import RolloutStep

        outputs = [0.3, 100.0, -0.2, 0.5]
        old = b.distribution(outputs, observation)[1][0]
        step = RolloutStep(observation, 0, 1, old, 0.5, 0.0, True, False)
        _, gradients, _ = b.objective(outputs, step, 0.7, 1.2)
        self.assertEqual(gradients[1], 0.0)
        for i, analytic in enumerate(gradients):
            plus, minus = outputs[:], outputs[:]
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numeric = (
                b.objective(plus, step, 0.7, 1.2)[0]
                - b.objective(minus, step, 0.7, 1.2)[0]
            ) / 2e-6
            self.assertAlmostEqual(analytic, numeric, places=6)

    def test_centralized_critic_does_not_change_actor(self):
        b = create_brain(settings("mappo"), ["rest", "eat"], 1, 2)
        o = (0.2,) * INPUTS
        b.context = (0.0,) * 16
        p1 = b.action_probabilities(o)
        v1 = b.network.value(o + b.context)
        b.context = (0.8,) * 16
        self.assertEqual(p1, b.action_probabilities(o))
        self.assertNotEqual(v1, b.network.value(o + b.context))
        before_actor = copy.deepcopy(b.network.actor.layers)
        gradient = b.network.zeros()
        _, tape = b.network.forward(o + b.context)
        b.network.accumulate_gradient(tape, [0.0, 0.0, 1.0], gradient)
        self.assertTrue(
            all(
                g == 0
                for layer in gradient[: len(b.network.actor.layers)]
                for values in layer.values()
                for g in values
            )
        )
        b._adam_step(gradient, 1)
        self.assertEqual(before_actor, b.network.actor.layers)


class MaskAndMemoryTests(unittest.TestCase):
    def test_masks_use_immediate_physics_and_preserve_turns(self):
        w = empty_world()
        prey = animal(0, "prey", 4, 3)
        pred = animal(1, "predator", 4, 4)
        w.animals = {0: prey, 1: pred}
        actions = ["move_forward", "turn_left", "turn_right", "rest", "attack"]
        self.assertEqual(action_mask(w, pred, actions), (False, True, True, True, True))
        prey.x = 6
        self.assertEqual(action_mask(w, pred, actions), (True, True, True, True, False))
        from ecosystem.world import Resource

        w.resources = {(4, 3): Resource(4, 3, 3)}
        self.assertTrue(action_mask(w, pred, ["rest", "move_forward"])[1])
        prey.x, prey.y = 4, 3
        self.assertTrue(action_mask(w, prey, ["rest", "eat"])[1])
        prey.energy = w.body(prey)["max_energy"]
        self.assertFalse(action_mask(w, prey, ["rest", "eat"])[1])

    def test_masked_actions_never_sampled_even_with_large_logits(self):
        b = create_brain(settings("maskable_ppo"), ["rest", "eat"], 1, 2)
        b.mask = (True, False)
        b.network.layers[-1]["bias"][1] = 1e9
        for _ in range(50):
            self.assertEqual(b.choose((0.0,) * INPUTS), "rest")
        self.assertEqual(b.action_probabilities((0.0,) * INPUTS), [1.0, 0.0])
        self.assertEqual(b.choose((0.0,) * INPUTS, random_baseline=True), "rest")

    def test_mask_changes_do_not_rewrite_collected_masks(self):
        b = create_brain(settings("maskable_ppo"), ["rest", "eat"], 1, 2)
        o = (0.2,) * INPUTS
        b.mask = (True, False)
        a = b.choose(o)
        b.learn(Experience(o, a, 1, o, False))
        b.mask = (False, True)
        a = b.choose(o)
        b.learn(Experience(o, a, 1, o, False))
        self.assertEqual(b.rollout[0].observation[INPUTS:], (True, False))
        self.assertEqual(b.rollout[1].observation[INPUTS:], (False, True))
        self.assertEqual(b.rollout[0].log_probability, 0.0)

    def test_memory_advances_in_evaluation_but_diagnostics_are_read_only(self):
        b = create_brain(settings("recurrent_ppo"), ["rest", "eat"], 1, 2)
        o = (0.2,) * INPUTS
        before = copy.deepcopy(b.network.layers)
        initial = b.checkpoint()
        b.diagnostics(o)
        b.action_probabilities(o)
        self.assertEqual(initial, b.checkpoint())
        b.choose(o, explore=False)
        self.assertNotEqual(b.hidden, [0.0, 0.0])
        self.assertEqual(before, b.network.layers)
        self.assertEqual(b.updates, 0)

    def test_newborn_inherits_weights_but_not_memory_or_buffers(self):
        for kind in KINDS:
            b = create_brain(settings(kind), ["rest", "eat"], 1, 2)
            o = (0.2,) * INPUTS
            a = b.choose(o)
            b.learn(Experience(o, a, 1, o, False))
            child = inherit_brain(b, 99)
            self.assertEqual(model_snapshot(child), model_snapshot(b))
            self.assertIsNot(child.network, b.network)
            if kind == "recurrent_ppo":
                self.assertEqual(child.hidden, [0.0, 0.0])
            self.assertEqual(
                getattr(child, "rollout", getattr(child, "replay", None)), []
            )
            child.network.layers[0][next(iter(child.network.layers[0]))][0] += 1
            self.assertNotEqual(model_snapshot(child), model_snapshot(b))

    def test_terminal_and_truncated_recurrent_bootstrap(self):
        for terminal in (True, False):
            b = create_brain(settings("recurrent_ppo"), ["rest", "eat"], 1, 2)
            o = (0.2,) * INPUTS
            a = b.choose(o)
            expected = b.network.step(o, b.hidden)[0][-1]
            with patch.object(b, "_update"):
                b.learn(
                    Experience(o, a, 1, None if terminal else o, terminal, not terminal)
                )
            self.assertAlmostEqual(
                b.rollout[-1].next_value, 0.0 if terminal else expected
            )
            self.assertEqual(b.hidden, [0.0, 0.0])

    def test_qrdqn_double_target_and_time_limit_bootstrap(self):
        c = settings("qrdqn")
        c.update(batch_size=1, learning_starts_after=1)
        for terminal in (True, False):
            b = create_brain(c, ["rest", "eat"], 1, 2)
            captured = []

            def capture(predictions, targets, kappa):
                captured.append(targets)
                return 0.0, [0.0] * len(predictions)

            with (
                patch.object(b, "values", return_value=[2, 1]),
                patch.object(b.target_network, "predict_range", return_value=[3, 4, 5]),
                patch("ecosystem.qrdqn.quantile_huber", side_effect=capture),
            ):
                b.learn(
                    Experience(
                        (0.0,) * 20,
                        "rest",
                        2,
                        None if terminal else (0.1,) * 20,
                        terminal,
                        not terminal,
                    )
                )
            self.assertEqual(
                captured[0],
                (
                    [2] * 3
                    if terminal
                    else [2 + c["discount_factor"] * v for v in (3, 4, 5)]
                ),
            )


class AdvancedLifecycleTests(unittest.TestCase):
    def test_json_and_compact_checkpoints_resume_exactly(self):
        for kind in KINDS:
            for retain in (False, True):
                a = Experiment(environment(kind))
                a.config["experiment"]["retain_brains_between_runs"] = retain
                for _ in range(3):
                    a.step()
                b = Experiment.restore(roundtrip(a.checkpoint()))
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "state.json.gz"
                    a.write_checkpoint(path)
                    d = Experiment.restore(read_checkpoint(path))
                while not a.done:
                    a.step()
                    b.step()
                    d.step()
                self.assertEqual(a.checkpoint(), b.checkpoint(), (kind, retain))
                self.assertEqual(
                    a.checkpoint(), d.checkpoint(), (kind, retain, "compact")
                )
                self.assertGreater(a.total_updates, 0, kind)

    def test_pending_controller_roundtrip_and_warm_start(self):
        for kind in KINDS:
            b = create_brain(settings(kind), ["rest", "eat"], 1, 2)
            o = (0.2,) * INPUTS
            if kind == "maskable_ppo":
                b.mask = (False, True)
            if kind == "mappo":
                b.context = b.next_context = (0.3,) * 16
            a = b.choose(o)
            r = restore_brain(roundtrip(b.checkpoint()))
            e = Experience(o, a, 2, o, False)
            b.learn(e)
            r.learn(e)
            self.assertEqual(b.checkpoint(), r.checkpoint(), kind)
            fresh = create_brain(settings(kind), ["rest", "eat"], 3, 4)
            load_model_weights(fresh, model_snapshot(b))
            self.assertEqual(model_snapshot(fresh), model_snapshot(b))
            self.assertEqual(fresh.updates, 0)
            if kind == "recurrent_ppo":
                self.assertEqual(fresh.hidden, [0.0, 0.0])
            if kind == "qrdqn":
                self.assertEqual(fresh.target_network.layers, fresh.network.layers)

    def test_signature_rejects_architecture_changes(self):
        for kind, key, value in [
            ("recurrent_ppo", "recurrent_size", 3),
            ("qrdqn", "num_quantiles", 4),
            ("mappo", "critic_hidden_layers", [4]),
        ]:
            c = settings(kind)
            a = create_brain(c, ["rest", "eat"], 1, 2)
            c[key] = value
            b = create_brain(c, ["rest", "eat"], 1, 2)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_model_weights(b, model_snapshot(a))

    def test_mappo_collects_team_rewards_and_current_next_context(self):
        e = Experiment(environment("mappo"))
        before = central_contexts(e.world)
        captured = {}

        def record(brain, experience):
            captured[id(brain)] = experience

        from ecosystem.contextual_ppo import MAPPOBrain

        with patch.object(MAPPOBrain, "learn", record):
            e.step()
        after = central_contexts(e.world)
        prey = [b for i, b in e.brains.items() if e.world.animals[i].species == "prey"]
        self.assertEqual(len({captured[id(b)].reward for b in prey}), 1)
        self.assertNotEqual(before["prey"][0], after["prey"][0])
        self.assertEqual(prey[0].context, after["prey"])

    def test_mappo_stores_separate_current_and_next_context(self):
        b = create_brain(settings("mappo"), ["rest", "eat"], 1, 2)
        b.context = (0.1,) * 16
        b.next_context = (0.9,) * 16
        o = (0.2,) * 20
        a = b.choose(o)
        expected = b.network.value(o + b.next_context)
        b.learn(Experience(o, a, 2, o, False))
        self.assertEqual(b.rollout[0].observation[20:], b.context)
        self.assertAlmostEqual(b.rollout[0].next_value, expected)

    def test_recording_and_snapshot_diagnostics_do_not_change_training(self):
        for kind in KINDS:
            a = Experiment(environment(kind))
            b = Experiment(environment(kind))
            with tempfile.TemporaryDirectory() as folder:
                rec = Recorder(folder, a.config, a.mode)
                rec.begin_run(a)
                for _ in range(5):
                    a.snapshot(0)
                    rec.capture(a)
                    a.step()
                    b.step()
                rec.capture(a)
                rec.close(False)
                chunk = load_chunk(chunk_path(Path(folder) / "playback", 1, 0))
                self.assertTrue(any(f.get("diagnostics") for f in chunk))
            self.assertEqual(a.checkpoint(), b.checkpoint(), kind)

    def test_evaluation_weights_and_counters_are_frozen(self):
        for kind in KINDS:
            e = Experiment(environment(kind), mode="evaluate")
            before = {i: model_snapshot(b) for i, b in e.brains.items()}
            for _ in range(5):
                e.step()
            self.assertEqual(
                before, {i: model_snapshot(b) for i, b in e.brains.items()}, kind
            )
            self.assertEqual(e.total_updates, 0)

    def test_reproduction_death_and_checkpoint_release(self):
        from tests.test_reproduction import reproduction

        for kind in KINDS:
            c = environment(kind)
            c["obstacles"].update(generation="explicit", explicit_cells=[])
            for spec in c["species"].values():
                spec["reproduction"] = reproduction()
                spec["body"].update(living_cost=0, movement_cost=0)
                if spec["diet"] == "prey":
                    spec["body"].update(attack_damage=0, attack_cost=0)
            from tests.reproduction_support import force_reproduction

            a = force_reproduction(Experiment(c))
            a.step()
            children = [x for x in a.world.animals.values() if x.parent_id >= 0]
            self.assertTrue(children, kind)
            if kind == "recurrent_ppo":
                for child in children:
                    self.assertEqual(a.brains[child.id].hidden, [0.0, 0.0])
            child = children[0]
            child.energy = 0
            child.health = 0.1
            # Any possible action must leave this animal starving on its next step.
            a.world.config["species"][child.species]["body"]["living_cost"] = 1
            if child.species == "prey":
                a.world.config["species"][child.species]["body"][
                    "energy_per_food_unit"
                ] = 0
            a.observations = a.world.observe_all()
            a._refresh_context()
            b = Experiment.restore(roundtrip(a.checkpoint()))
            a.step()
            b.step()
            self.assertEqual(a.checkpoint(), b.checkpoint(), kind)
            self.assertNotIn(child.id, a.brains)

    def test_mixed_species_algorithms_resume(self):
        for left, right in zip(KINDS, KINDS[1:] + KINDS[:1]):
            c = environment(left)
            c["brains"]["other"] = settings(right)
            c["species"]["predator"]["brain"] = "other"
            a = Experiment(c)
            for _ in range(3):
                a.step()
            b = Experiment.restore(roundtrip(a.checkpoint()))
            while not a.done:
                a.step()
                b.step()
            self.assertEqual(a.checkpoint(), b.checkpoint(), (left, right))

    def test_cli_progress_archive_warm_start_and_replay(self):
        for kind in KINDS:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config_path = root / "environment.json"
                c = environment(kind)
                config_path.write_text(json.dumps(c))
                command = [
                    sys.executable,
                    "-m",
                    "ecosystem.cli",
                    str(config_path),
                    "--output",
                    str(root / "results"),
                    "--runs",
                    "2",
                    "--steps",
                    "4",
                    "--seed",
                    "100",
                    "--progress",
                ]
                result = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("100%", result.stderr)
                source = next((root / "results").iterdir())
                manifest = json.loads((source / "playback/recording.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(len(manifest["runs"]), 2)
                self.assertEqual(
                    manifest["runs"][1]["initialized_from"]["predator"]["female"][
                        "criterion"
                    ],
                    "lifespan",
                )
                command = [
                    sys.executable,
                    "-m",
                    "ecosystem.cli",
                    str(config_path),
                    "--output",
                    str(root / "warm"),
                    "--runs",
                    "1",
                    "--steps",
                    "3",
                    "--brains-from",
                    str(source),
                    "--brain",
                    "predator=lifespan",
                    "--no-progress",
                ]
                result = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                destination = next((root / "warm").iterdir())
                loaded = json.loads((destination / "manifest.json").read_text())
                self.assertEqual(set(loaded["initialized_from"]), {"prey", "predator"})

    def test_invalid_config_bounds_and_mask_fallback(self):
        cases = [
            ("recurrent_ppo", "recurrent_size", 0),
            ("recurrent_ppo", "sequence_length", 5),
            ("qrdqn", "num_quantiles", 1),
            ("qrdqn", "huber_threshold", 0),
            ("qrdqn", "replay_capacity", 1),
            ("mappo", "critic_hidden_layers", []),
        ]
        for kind, key, value in cases:
            c = environment(kind)
            c["brains"]["test"][key] = value
            with self.assertRaises(ValueError, msg=(kind, key)):
                validate(c)
        c = environment("maskable_ppo")
        c["species"]["prey"]["allowed_actions"] = ["eat"]
        with self.assertRaisesRegex(ValueError, "fallback"):
            validate(c)

    def test_grpo_trials_keep_other_algorithms_unchanged(self):
        from ecosystem.grouped_rollouts import collect_group
        from tests.test_policy_algorithms import GRPOTests

        for kind in KINDS:
            w, brains = GRPOTests().encounter()
            brains[0] = create_brain(settings(kind), ["rest"], 3, 4)
            before = brains[0].checkpoint()
            world = w.checkpoint()
            collect_group(w, brains, 1)
            self.assertEqual(before, brains[0].checkpoint(), kind)
            self.assertEqual(world, w.checkpoint())


if __name__ == "__main__":
    unittest.main()
