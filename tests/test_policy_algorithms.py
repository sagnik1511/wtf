import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wtf.learning.brain import Brain, Experience
from wtf.learning.brains import create_brain
from wtf.config import load, validate
from wtf.learning.grouped_rollouts import collect_group
from wtf.learning.policy_gradients import GRPOBrain, PolicyStep, actor_loss_gradient
from wtf.experiment.runner import Experiment
from tests.test_simulation import config, empty_world, animal

ROOT = Path(__file__).resolve().parents[1]


def settings(kind):
    c = copy.deepcopy(
        load(ROOT / f"environments/{kind}.json")["brains"][f"small_{kind}"]
    )
    c["network"]["hidden_layers"] = [4]
    if kind == "a2c":
        c["rollout_steps"] = 4
    elif kind == "reinforce":
        c["max_episode_steps"] = 20
    elif kind == "grpo":
        c.update(
            group_size=2,
            rollout_horizon=2,
            train_every_steps=2,
            epochs=1,
            batch_size=2,
            target_kl=0,
        )
    return c


def environment(kind):
    c = config()
    c["brains"]["test"] = settings(kind)
    for species in c["species"].values():
        species["brain"] = "test"
    c["world"]["max_steps"] = 7
    c["experiment"]["runs"] = 2
    return c


class PolicyGradientMathTests(unittest.TestCase):
    def test_policy_and_exact_reference_kl_gradients(self):
        for clipped in (False, True):
            outputs = [0.3, -0.4, 0.1]
            _, analytic, _ = actor_loss_gradient(
                outputs,
                1,
                -0.7,
                0.02,
                old_logp=-1.4 if clipped else None,
                reference_logits=[-0.1, 0.2, 0.4],
                kl_coefficient=0.08,
            )
            for i in range(3):
                plus, minus = outputs[:], outputs[:]
                plus[i] += 1e-6
                minus[i] -= 1e-6
                lp = actor_loss_gradient(
                    plus,
                    1,
                    -0.7,
                    0.02,
                    old_logp=-1.4 if clipped else None,
                    reference_logits=[-0.1, 0.2, 0.4],
                    kl_coefficient=0.08,
                )[0]
                lm = actor_loss_gradient(
                    minus,
                    1,
                    -0.7,
                    0.02,
                    old_logp=-1.4 if clipped else None,
                    reference_logits=[-0.1, 0.2, 0.4],
                    kl_coefficient=0.08,
                )[0]
                self.assertAlmostEqual(analytic[i], (lp - lm) / 2e-6, places=6)

    def test_a2c_and_reinforce_learn_rewarded_action(self):
        for kind in ("a2c", "reinforce"):
            c = settings(kind)
            c["learning_rate"] = 0.01
            brain = create_brain(c, ["rest", "eat"], 7, 9)
            observation = (0.0,) * 20
            for _ in range(100):
                action = brain.choose(observation)
                brain.learn(
                    Experience(
                        observation,
                        action,
                        1.0 if action == "eat" else -1.0,
                        None,
                        True,
                    )
                )
            with self.subTest(kind=kind):
                self.assertGreater(
                    brain.diagnostics(observation)["action_probabilities"]["eat"], 0.9
                )

    def test_reinforce_waits_for_complete_episode(self):
        c = settings("reinforce")
        brain = create_brain(c, ["rest", "eat"], 1, 2)
        observation = (0.0,) * 20
        for _ in range(3):
            action = brain.choose(observation)
            brain.learn(Experience(observation, action, 1.0, observation, False))
        self.assertEqual(brain.updates, 0)
        self.assertEqual(len(brain.trajectory), 3)
        action = brain.choose(observation)
        brain.learn(Experience(observation, action, 1.0, observation, False, True))
        self.assertEqual(brain.updates, 1)
        self.assertEqual(brain.trajectory, [])
        self.assertEqual(brain.network.sizes[-1], len(brain.actions))


class GRPOTests(unittest.TestCase):
    def encounter(self):
        c = settings("grpo")
        c.update(
            group_size=8, rollout_horizon=1, learning_rate=0.01, batch_size=8, epochs=2
        )
        focal = GRPOBrain(c, ["attack", "rest"], 1, 2)
        w = empty_world()
        w.config["species"]["predator"]["body"]["attack_damage"] = 30
        w.animals = {0: animal(0, "prey", 2, 2), 1: animal(1, "predator", 1, 2, 1)}
        other = Brain(config()["brains"]["small_q"], ["rest"], 3, 4)
        return w, {0: other, 1: focal}

    def test_matched_trials_do_not_change_real_world_or_other_agent(self):
        w, brains = self.encounter()
        world_before, other_before = w.checkpoint(), brains[0].checkpoint()
        actual_policy_rng = brains[1].rng.getstate()
        group = collect_group(w, brains, 1)
        self.assertEqual(w.checkpoint(), world_before)
        self.assertEqual(brains[0].checkpoint(), other_before)
        self.assertEqual(brains[1].rng.getstate(), actual_policy_rng)
        self.assertEqual(len(group), 8)
        self.assertEqual(len({t[0].observation for t in group}), 1)
        self.assertEqual(set(sum(s.reward for s in t) for t in group), {0.0, 5.5})
        self.assertEqual(dict(w.kills), {})
        before = brains[1].diagnostics(w.observe(w.animals[1]))["action_probabilities"][
            "attack"
        ]
        brains[1].update_group(group)
        after = brains[1].diagnostics(w.observe(w.animals[1]))["action_probabilities"][
            "attack"
        ]
        self.assertGreater(after, before)
        self.assertGreater(brains[1].updates, 0)
        self.assertEqual(w.checkpoint(), world_before)

    def test_reference_policy_stays_frozen_and_restores(self):
        w, brains = self.encounter()
        brain = brains[1]
        reference = copy.deepcopy(brain.reference_network.layers)
        brain.update_group(collect_group(w, brains, 1))
        self.assertEqual(reference, brain.reference_network.layers)
        self.assertNotEqual(brain.network.layers, reference)
        state = json.loads(json.dumps(brain.checkpoint()))
        restored = GRPOBrain.restore(state)
        group_a = collect_group(w, brains, 1)
        group_b = collect_group(w, {0: brains[0], 1: restored}, 1)
        self.assertEqual(group_a, group_b)
        brain.update_group(group_a)
        restored.update_group(group_b)
        self.assertEqual(brain.checkpoint(), restored.checkpoint())

    def test_equal_rewards_skip_uninformative_updates(self):
        w, brains = self.encounter()
        brain = brains[1]
        o = w.observe(w.animals[1])
        trajectory = [PolicyStep(o, 1, 0.0, math.log(0.5), False, True)]
        before = copy.deepcopy(brain.network.layers)
        brain.update_group([trajectory, trajectory])
        self.assertEqual(before, brain.network.layers)
        self.assertEqual(brain.updates, 0)
        self.assertTrue(brain.last_metrics["zero_variance"])

    def test_group_sampling_does_not_mutate_other_policy_algorithms(self):
        for kind in ("ppo", "a2c", "reinforce", "grpo"):
            w, brains = self.encounter()
            if kind == "ppo":
                c = load(ROOT / "environments/ppo.json")["brains"]["prey_ppo"]
                c.update(observation_version=1, observation_top_k=1)
            else:
                c = settings(kind)
            brains[0] = create_brain(c, ["rest"], 3, 4)
            before = brains[0].checkpoint()
            collect_group(w, brains, 1)
            self.assertEqual(before, brains[0].checkpoint(), kind)

    def test_runner_limits_groups_per_real_step(self):
        e = Experiment(environment("grpo"))
        for brain in e.brains.values():
            brain.steps = 2
        calls = []

        def fake_group(world, brains, focal_id):
            calls.append(focal_id)
            o = world.observe(world.animals[focal_id])
            return [[PolicyStep(o, 0, 0.0, 0.0, False, True)]] * 2

        with patch(
            "wtf.learning.grouped_rollouts.collect_group", side_effect=fake_group
        ):
            for _ in range(4):
                before_tick = e.world.tick
                e.step()
                self.assertEqual(e.world.tick, before_tick + 1)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(set(calls)), 4)


class ExtraAlgorithmIntegrationTests(unittest.TestCase):
    def test_resuming_models_and_partial_episodes_is_exact(self):
        for kind in ("a2c", "reinforce", "grpo"):
            for retain in (False, True):
                with self.subTest(kind=kind, retain=retain):
                    c = environment(kind)
                    c["experiment"]["retain_brains_between_runs"] = retain
                    a = Experiment(c)
                    a.step()
                    a.step()
                    a.step()
                    b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
                    while not a.done:
                        a.step()
                        b.step()
                    self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_evaluation_is_read_only_for_each_new_algorithm(self):
        for kind in ("a2c", "reinforce", "grpo"):
            e = Experiment(environment(kind), mode="evaluate")
            before = {i: b.checkpoint() for i, b in e.brains.items()}
            for _ in range(4):
                e.step()
            self.assertEqual(
                before, {i: b.checkpoint() for i, b in e.brains.items()}, kind
            )

    def test_validation_of_runtime_and_memory_bounds(self):
        c = environment("reinforce")
        c["brains"]["test"]["max_episode_steps"] = 2
        with self.assertRaisesRegex(ValueError, "max_episode_steps"):
            validate(c)
        for key, value in [
            ("group_size", 1),
            ("group_size", 9),
            ("rollout_horizon", 17),
            ("batch_size", 1000),
        ]:
            c = environment("grpo")
            c["brains"]["test"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate(c)
        c = environment("grpo")
        c["experiment"]["grpo_updates_per_step"] = 0
        with self.assertRaisesRegex(ValueError, "grpo_updates_per_step"):
            validate(c)

    def test_streamed_checkpoints_and_recordings_dispatch(self):
        for kind in ("a2c", "reinforce", "grpo"):
            e = Experiment(environment(kind))
            e.step()
            e.step()
            e.step()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "checkpoint.json"
                e.write_checkpoint(path)
                b = Experiment.restore(json.loads(path.read_text()))
                e.step()
                b.step()
                self.assertEqual(e.checkpoint(), b.checkpoint(), kind)
            d = e.snapshot(0)["selected"]
            self.assertEqual(d["algorithm"], kind)
            self.assertEqual("state_value" in d, kind == "a2c")
            self.assertAlmostEqual(sum(d["action_probabilities"].values()), 1.0)
