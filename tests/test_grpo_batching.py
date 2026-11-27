"""Independent actor batches and reset GRPO worlds preserve exact state."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from wtf.experiment.runner import Experiment
from wtf.learning import grouped_rollouts
from wtf.learning.batched import choose_actions
from wtf.learning.brains import create_brain
from wtf.learning.ppo import MLP
from wtf.native import kernels
from wtf.simulation.world import World
from wtf.storage.recording import Recorder
from wtf.config import load
from tests.test_mutation import KINDS, environment
from tests import test_policy_algorithms as policy_fixtures
from tests.test_advanced_algorithms import settings
from tests.test_simulation import animal
from tests.reproduction_support import force_action


def scalar_backend():
    available = kernels.available
    return patch.object(
        kernels,
        "available",
        new=lambda name: name != "mlp_predict_many" and available(name),
    )


class TrialReuseTests(unittest.TestCase):
    def test_world_reset_restores_dynamic_state_and_discards_all_caches(self):
        e = Experiment(environment("grpo", 0))
        state = copy.deepcopy(e.world.checkpoint())
        trial = World.restore(state)
        config = trial.config
        rng = trial.rng
        trial.observe_all()
        trial.external_marker = object()
        old = next(iter(trial.animals.values()))
        trial.obstacles.add((0, 0))
        old.energy = 1
        trial._reset_trial(state)
        self.assertEqual(trial.checkpoint(), state)
        self.assertIs(trial.config, config)
        self.assertIs(trial.rng, rng)
        self.assertIsNot(next(iter(trial.animals.values())), old)
        self.assertNotIn("_perception", vars(trial))
        self.assertNotIn("external_marker", vars(trial))
        self.assertIsNot(trial.config, e.world.config)
        trial.config["species"]["prey"]["body"]["living_cost"] = 99
        self.assertNotEqual(trial.config, e.world.config)

    def test_group_matches_fresh_worlds_with_masks_recurrent_and_centralized_policies(
        self,
    ):
        for kind in ("maskable_ppo", "recurrent_ppo", "mappo"):
            with self.subTest(kind=kind):
                world, brains = policy_fixtures.GRPOTests().encounter()
                world.animals[2] = animal(2, "prey", 5, 5)
                for i in (0, 2):
                    brains[i] = create_brain(settings(kind), ["rest"], 3, 4 + i)
                if kind == "mappo":
                    for i in (0, 2):
                        brains[i].context = (0.0,) * 16
                focal = brains[1]
                focal.config["rollout_horizon"] = 4
                before = copy.deepcopy(world.checkpoint())
                brain_states = {i: b.checkpoint() for i, b in brains.items()}
                group_state = focal.group_rng.getstate()
                with patch.object(World, "restore", wraps=World.restore) as restored:
                    actual = grouped_rollouts.collect_group(world, brains, 1)
                    self.assertEqual(restored.call_count, 1)
                focal.group_rng.setstate(group_state)
                with scalar_backend():
                    expected = grouped_rollouts.collect_group(
                        world, brains, 1, reuse_world=False
                    )
                self.assertEqual(actual, expected)
                self.assertEqual(world.checkpoint(), before)
                focal.group_rng.setstate(group_state)
                self.assertEqual(
                    {i: b.checkpoint() for i, b in brains.items()}, brain_states
                )

    def test_lstm_newborn_in_matched_trial_receives_hidden_and_cell_memory(self):
        config = environment("lstm_ppo", 0)
        config["species"]["prey"]["initial_population"] = 2
        config["species"]["predator"]["initial_population"] = 1
        config["world"]["max_steps"] = 4
        experiment = Experiment(config)
        parents = [a for a in experiment.world.animals.values() if a.species == "prey"]
        for index, parent in enumerate(parents):
            parent.x, parent.y = 1 + index, 1
            parent.sex = "female" if index == 0 else "male"
            force_action(experiment.brains[parent.id], "reproduce")
        focal = next(
            a for a in experiment.world.animals.values() if a.species == "predator"
        )
        focal.x = focal.y = 4
        template = policy_fixtures.settings("grpo")
        template.update(group_size=2, rollout_horizon=3)
        experiment.brains[focal.id] = create_brain(template, ["rest"], 3, 4)
        before = copy.deepcopy(experiment.world.checkpoint())
        brains = {i: brain.checkpoint() for i, brain in experiment.brains.items()}
        group_rng = experiment.brains[focal.id].group_rng.getstate()
        with patch.object(
            grouped_rollouts,
            "mix_parent_brains",
            wraps=grouped_rollouts.mix_parent_brains,
        ) as inherited:
            trajectories = grouped_rollouts.collect_group(
                experiment.world, experiment.brains, focal.id
            )
        self.assertGreater(inherited.call_count, 0)
        self.assertEqual([len(trajectory) for trajectory in trajectories], [3, 3])
        self.assertEqual(before, experiment.world.checkpoint())
        experiment.brains[focal.id].group_rng.setstate(group_rng)
        self.assertEqual(
            brains, {i: brain.checkpoint() for i, brain in experiment.brains.items()}
        )

    def test_terrain_and_day_night_groups_match_fresh_worlds(self):
        for preset in ("meandering_river", "day_night"):
            c = load("environments/" + preset + ".json")
            if preset == "day_night":
                template = load("environments/grpo.json")["brains"]["small_grpo"]
                predator = next(
                    spec for spec in c["species"].values() if spec["diet"] == "prey"
                )
                template["observation_actions"] = list(predator["allowed_actions"])
                c["brains"]["trial_grpo"] = template
                predator["brain"] = "trial_grpo"
            c["seed"] = 100
            c["experiment"].update(runs=1, world_seed_policy="fixed")
            e = Experiment(c)
            focal_id = next(
                i for i in e.world.animals if e.brains[i].config["type"] == "grpo"
            )
            focal = e.brains[focal_id]
            focal.config.update(group_size=2, rollout_horizon=2)
            for tick in (0, 200, 300):
                e.world.tick = tick
                before = copy.deepcopy(e.world.checkpoint())
                rng = focal.group_rng.getstate()
                actual = grouped_rollouts.collect_group(e.world, e.brains, focal_id)
                focal.group_rng.setstate(rng)
                with scalar_backend():
                    expected = grouped_rollouts.collect_group(
                        e.world, e.brains, focal_id, reuse_world=False
                    )
                self.assertEqual(actual, expected, (preset, tick))
                self.assertEqual(e.world.checkpoint(), before)

    def test_cached_initial_visual_memory_also_matches_legacy_v4_trials(self):
        c = load("environments/meandering_river.json")
        for template in c["brains"].values():
            template["observation_version"] = 4
            template.pop("observation_actions", None)
        e = Experiment(c)
        focal_id = next(i for i, b in e.brains.items() if b.config["type"] == "grpo")
        e.world.tick = 200  # The previous personally seen frame is now stale.
        focal = e.brains[focal_id]
        focal.config.update(group_size=2, rollout_horizon=2)
        before = copy.deepcopy(e.world.checkpoint())
        rng = focal.group_rng.getstate()
        actual = grouped_rollouts.collect_group(e.world, e.brains, focal_id)
        focal.group_rng.setstate(rng)
        expected = grouped_rollouts.collect_group(
            e.world, e.brains, focal_id, reuse_world=False
        )
        self.assertEqual(actual, expected)
        self.assertEqual(e.world.checkpoint(), before)

    def test_frozen_model_qualification_is_once_per_group(self):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel

        from wtf.learning.batched import policy_plan

        world, brains = policy_fixtures.GRPOTests().encounter()
        world.animals[2] = animal(2, "prey", 5, 5)
        for i in (0, 2):
            brains[i] = create_brain(settings("maskable_ppo"), ["rest"], 3, 4 + i)
        brains[1].config["rollout_horizon"] = 4
        with patch("wtf.learning.batched.policy_plan", wraps=policy_plan) as qualified:
            grouped_rollouts.collect_group(world, brains, 1)
            self.assertEqual(qualified.call_count, 1)

    def test_trials_with_birth_mutation_match_and_do_not_mutate_real_parents(self):

        # Keep imports lazy so optional tools and training dependencies load only on
        # this path.
        # pylint: disable=import-outside-toplevel

        from wtf.learning.mutation import mutate_birth

        e = Experiment(environment("grpo", 1))
        w = e.world
        w.animals = {i: w.animals[i] for i in (0, 1, 3)}
        w.config["species"]["prey"]["reproduction"]["mate_radius"] = 1
        for i in (0, 1):
            a = w.animals[i]
            a.x, a.y = 1 + i, 1
            a.sex = "female" if i == 0 else "male"
            a.age = 3
            a.health = 30
            a.energy = 90
            force_action(e.brains[i], "reproduce")
        e.brains[3] = create_brain(e.brains[3].config, ["rest"], 1, 2)
        e.brains[3].config["rollout_horizon"] = 4
        w.animals[3].x, w.animals[3].y = 4, 4
        before = copy.deepcopy(w.checkpoint())
        parents = {i: e.brains[i].checkpoint() for i in (0, 1)}
        rng = e.brains[3].group_rng.getstate()
        with patch("wtf.learning.mutation.mutate_birth", wraps=mutate_birth) as mutate:
            actual = grouped_rollouts.collect_group(w, e.brains, 3)
            self.assertGreater(mutate.call_count, 0)
        e.brains[3].group_rng.setstate(rng)
        with scalar_backend():
            expected = grouped_rollouts.collect_group(w, e.brains, 3, reuse_world=False)
        self.assertEqual(actual, expected)
        self.assertEqual(w.checkpoint(), before)
        self.assertEqual({i: e.brains[i].checkpoint() for i in (0, 1)}, parents)


@unittest.skipUnless(
    kernels.available("mlp_predict_many"), "Build batched native inference"
)
class IndependentBatchTests(unittest.TestCase):
    def test_native_models_shapes_edits_signed_zero_and_error_cleanup(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        rng = random.Random(4)
        nets = [
            MLP(hidden, 3, i, inputs=n)
            for i, (hidden, n) in enumerate((([], 2), ([3], 4), ([5, 2], 6)))
        ]
        observations = [
            tuple(rng.random() for _ in range(net.sizes[0])) for net in nets
        ]
        for _turn in range(3):
            jobs = [(net.layers, o) for net, o in zip(nets, observations)]
            expected = [net.predict(o) for net, o in zip(nets, observations)]
            self.assertEqual(
                json.dumps(kernels.native.mlp_predict_many(jobs)), json.dumps(expected)
            )
            nets[0].layers[0]["weights"][0] += 0.125
        self.assertEqual(kernels.native.mlp_predict_many([]), [])
        with self.assertRaises((ValueError, TypeError)):
            kernels.native.mlp_predict_many(jobs + [(nets[0].layers, ())])
        for net in nets:
            net.layers[0]["weights"].append(0)
            net.layers[0]["weights"].pop()
        with self.assertRaises((ValueError, TypeError)):
            kernels.native.mlp_predict_many([None])

    def test_sampling_pending_rng_memory_and_masks_match_all_controllers_and_modes(
        self,
    ):
        for kind in KINDS:
            for mode in ("train", "evaluate", "random"):
                with self.subTest(kind=kind, mode=mode):
                    e = Experiment(environment(kind, 0), mode=mode)
                    reference = Experiment.restore(e.checkpoint())
                    for _ in range(4):
                        actual = choose_actions(
                            e.brains,
                            e.observations,
                            explore=mode == "train",
                            random_baseline=mode == "random",
                        )
                        expected = {
                            i: reference.brains[i].choose(
                                o,
                                explore=mode == "train",
                                random_baseline=mode == "random",
                            )
                            for i, o in reference.observations.items()
                        }
                        self.assertEqual(actual, expected)
                        self.assertEqual(
                            {i: b.checkpoint() for i, b in e.brains.items()},
                            {i: b.checkpoint() for i, b in reference.brains.items()},
                        )

    def test_exact_training_chaining_resume_and_binary_replays(self):
        collect = grouped_rollouts.collect_group
        for kind in KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                c = environment(kind, 1)
                c["recording"] = {
                    "enabled": True,
                    "frames_per_chunk": 2,
                    "frame_every_steps": 1,
                    "diagnostics_every_steps": 1,
                    "record_activations": True,
                    "compression": "xz",
                    "compression_level": 1,
                    "encoding": "binary",
                }
                e = Experiment(c)
                reference = Experiment.restore(e.checkpoint())
                recorders = [
                    Recorder(Path(folder) / name, c, e.mode)
                    for name in ("batch", "scalar")
                ]
                for rec, x in zip(recorders, (e, reference)):
                    rec.begin_run(x)
                    rec.capture(x)
                while not e.done:
                    if e.finished_run:
                        for rec, x in zip(recorders, (e, reference)):
                            rec.end_run(x)
                        e.next_run()
                        reference.next_run()
                        for rec, x in zip(recorders, (e, reference)):
                            rec.begin_run(x)
                            rec.capture(x)
                    e.step()
                    with (
                        scalar_backend(),
                        patch.object(
                            grouped_rollouts,
                            "collect_group",
                            new=lambda w, b, i: collect(w, b, i, reuse_world=False),
                        ),
                    ):
                        reference.step()
                    self.assertEqual(e.checkpoint(), reference.checkpoint())
                    if e.world.tick == 3:
                        e = Experiment.restore(e.checkpoint())
                        reference = Experiment.restore(reference.checkpoint())
                    for rec, x in zip(recorders, (e, reference)):
                        rec.capture(x, force=x.finished_run)
                for rec in recorders:
                    rec.close()
                left = sorted((Path(folder) / "batch").rglob("*.bin.xz"))
                right = sorted((Path(folder) / "scalar").rglob("*.bin.xz"))
                self.assertEqual(
                    [p.read_bytes() for p in left], [p.read_bytes() for p in right]
                )

    def test_one_native_entry_for_multiple_independent_actors(self):

        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        e = Experiment(environment("ppo", 0))
        with (
            patch.object(
                kernels.native,
                "mlp_predict_many",
                wraps=kernels.native.mlp_predict_many,
            ) as batched,
            patch.object(
                kernels.native, "mlp_predict", wraps=kernels.native.mlp_predict
            ) as scalar,
        ):
            choose_actions(e.brains, e.observations)
            self.assertEqual(batched.call_count, 1)
            self.assertEqual(scalar.call_count, 0)
            self.assertEqual(len(batched.call_args.args[0]), len(e.brains))

    def test_custom_predict_and_random_mode_are_not_bypassed(self):
        e = Experiment(environment("ppo", 0))
        brain = e.brains[0]
        with patch.object(
            brain.network, "predict", wraps=brain.network.predict
        ) as predict:
            choose_actions(e.brains, e.observations)
            self.assertEqual(predict.call_count, 1)
        with patch.object(
            kernels.native,
            "mlp_predict_many",
            side_effect=AssertionError("random baseline inference"),
        ):
            choose_actions(e.brains, e.observations, random_baseline=True)
