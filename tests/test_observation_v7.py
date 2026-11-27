"""Compact sensory contracts, independent late policies and exact recovery."""

# Numerical probes deliberately inspect private gradients, buffers and head state.
# pylint: disable=missing-function-docstring,protected-access
import copy
import json
import tempfile
import unittest
from pathlib import Path
from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from wtf.learning.brain import observation_inputs
from wtf.learning.brains import (
    create_brain,
    model_snapshot,
    load_model_weights,
    mix_parent_brains,
)
from wtf.learning.ppo import loss_and_gradient, RolloutStep
from wtf.inspection.explainability import prepare_model, explain_model
from wtf.simulation.compact_observations import compact_layout, movement_summary
from wtf.storage.recording import Recorder
from wtf.storage.replay_io import chunk_path, load_chunk
from tests.test_mutation import KINDS, environment


def config(kind="ppo"):
    c = load("environments/seaside_150_v7.json")
    c["world"]["max_steps"] = 6
    c["experiment"]["runs"] = 1
    c["recording"].update(record_last_episodes=-1, diagnostics_every_steps=1)
    for spec in c["species"].values():
        spec["initial_population"] = 2
    for brain in c["brains"].values():
        brain.update(type=kind, rollout_steps=4, batch_size=2, epochs=1)
        brain["network"]["hidden_layers"] = [4]
    return validate(c)


class ObservationV7Tests(unittest.TestCase):
    """Exercise sensory semantics, numerical derivatives and persistence boundaries."""

    def test_compact_inputs_work_with_all_existing_learners(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = environment(kind, 0)
                c["scent_trails"] = {"enabled": True, "lifetime_steps": 32}
                c["consequence_learning"] = {
                    "enabled": True,
                    "horizon_steps": 2,
                    "max_contexts": 512,
                    "persistent_action_totals": True,
                }
                for name, spec in c["species"].items():
                    key = spec["brain"] + "_" + name
                    c["brains"][key] = copy.deepcopy(c["brains"][spec["brain"]])
                    spec["brain"] = key
                    c["brains"][key].update(
                        observation_version=7,
                        observation_actions=list(spec["allowed_actions"]),
                        movement_history_steps=3,
                        scent_observation={"mode": "blended", "radius": 2},
                        social_observation={"history_steps": 4, "max_neighbors": 4},
                        consequence_observation=True,
                        consequence_scope="both",
                    )
                e = Experiment(c)
                e.step()
                r = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
                e.step()
                r.step()
                self.assertEqual(e.checkpoint(), r.checkpoint())
                for i, brain in e.brains.items():
                    model = model_snapshot(brain)
                    self.assertEqual(
                        len(prepare_model(model)["signature"]["observation_actions"]),
                        len(brain.actions),
                    )
                    explain_model(prepare_model(model), e.observations[i])

    def test_invalid_species_collection_reports_a_configuration_error(self):
        for species in (None, [], "prey", {}):
            c = config()
            c["species"] = species
            with self.assertRaisesRegex(ValueError, "species"):
                validate(c)

    def test_compact_widths_and_optional_tails(self):
        c = config()
        for name, expected in (("prey", 211), ("predator", 348)):
            t = c["brains"][c["species"][name]["brain"]]
            l = compact_layout(t)
            self.assertEqual(l.width, expected)
            self.assertFalse(any(n.endswith(" distance") for n in l.names))
            self.assertEqual(len([n for n in l.names if n.startswith("Scent ")]), 4)
            t = copy.deepcopy(t)
            t.pop("scent_observation")
            t["movement_history_steps"] = 0
            self.assertEqual(observation_inputs(t), 165 if name == "prey" else 204)
        with self.assertRaises(ValueError):
            bad = copy.deepcopy(c)
            bad["brains"]["prey_ppo"]["observation_actions"] = ["rest"]
            validate(bad)

    def test_motion_summarizes_seen_points_without_extrapolation(self):
        # Target moves forward one normalized unit per step; no hidden intermediate point.
        block = [3, 0, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1]
        result = movement_summary(block, 3)
        self.assertEqual(result[:4], [3, 0, 1, 0])
        self.assertAlmostEqual(result[-1], 2 / 3)
        missing = block[:]
        missing[4:7] = [1 / 3, 0.75, 0]
        self.assertEqual(movement_summary(missing, 3)[:2], [3, 0])
        self.assertEqual(movement_summary([0.0] * 16, 3), [0.0] * 7)
        turning = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
        self.assertEqual(movement_summary(turning, 2)[-1], 0)

    def test_scent_is_off_cell_merged_and_has_independent_freshness(self):
        e = Experiment(config())
        a = next(iter(e.world.animals.values()))
        field = e.world._tracking.scent
        a.x, a.y, a.heading = 10, 10, 0
        e.world.tick = 20
        field.cells.clear()
        # Set a dry local neighborhood without leaking any animal locations.
        e.world.geography.water = {}
        e.world.obstacles.clear()
        own = (a.y * e.world.width + a.x) * field.channels
        field.cells[own] = 20
        self.assertEqual(
            field.sample_compact(e.world, a, e.config["brains"]["prey_ppo"]), [0.0] * 4
        )
        cell = ((a.y - 1) * e.world.width + a.x) * field.channels
        field.cells[cell] = 4
        field.cells[cell + 1] = 4
        value = field.sample_compact(e.world, a, e.config["brains"]["prey_ppo"])
        self.assertEqual(value, [1, 0, 1, 0.5])
        a.heading = 1
        self.assertEqual(
            field.sample_compact(e.world, a, e.config["brains"]["prey_ppo"])[:2],
            [0, -1],
        )
        e.world.geography.water[(10, 9)] = (1, 0, 0)
        self.assertEqual(
            field.sample_compact(e.world, a, e.config["brains"]["prey_ppo"]), [0.0] * 4
        )

    def test_ppo_and_masked_ppo_restore_birth_and_archived_outputs(self):
        for kind in ("ppo", "maskable_ppo"):
            with self.subTest(kind=kind):
                e = Experiment(config(kind))
                e.step()
                r = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
                while not e.done:
                    e.step()
                    r.step()
                self.assertEqual(e.checkpoint(), r.checkpoint())
                brain = next(iter(e.brains.values()))
                obs = next(iter(e.observations.values()))
                model = model_snapshot(brain)
                other = create_brain(brain.config, brain.actions, 123, 456)
                load_model_weights(other, model)
                if kind == "maskable_ppo":
                    other.mask = brain.mask
                self.assertEqual(
                    brain.action_probabilities(obs), other.action_probabilities(obs)
                )
                explained = explain_model(
                    prepare_model(model),
                    obs,
                    mask=dict(
                        zip(
                            brain.actions,
                            getattr(brain, "mask", [True] * len(brain.actions)),
                        )
                    ),
                )
                for row in explained["outputs"]:
                    self.assertAlmostEqual(
                        row["score"],
                        brain.action_probabilities(obs)[
                            brain.actions.index(row["action"])
                        ],
                        places=6,
                    )
                child = mix_parent_brains(brain, other, 7)
                self.assertIsNot(
                    child.late.network.layers[0]["weights"],
                    brain.late.network.layers[0]["weights"],
                )
                self.assertEqual(child.late.updates, 0)
                self.assertEqual(
                    child.late.network.layers[0]["weights"],
                    brain.late.network.layers[0]["weights"],
                )

    # The independent finite-difference probe needs both analytic and perturbed state.
    # pylint: disable-next=too-many-locals
    def test_late_gradient_and_final_probability_match_finite_difference(self):
        e = Experiment(config())
        brain = next(iter(e.brains.values()))
        obs = list(next(iter(e.observations.values())))
        l = compact_layout(brain.config)
        obs[l.consequence_start : l.consequence_start + 4] = [0.8, 0.3, 0.7, 0.2]
        obs[0] = 0.4
        obs[1] = 0.6
        weights = brain.late.network.layers[0]["weights"]
        for i, _ in enumerate(weights):
            weights[i] = 0.1
        outputs = brain.network.predict(obs)
        adjusted = brain.late.adjusted(outputs, obs)
        probabilities, logs = brain.distribution(outputs, obs)
        self.assertNotEqual(adjusted, outputs)
        _, derivative, _ = loss_and_gradient(adjusted, 0, logs[0], 1, 0.1, brain.config)
        gradients = brain.late.network.zeros()
        brain.late.accumulate(obs, derivative, gradients)
        i = next(i for i, v in enumerate(gradients[0]["weights"]) if abs(v) > 1e-6)
        original = weights[i]
        eps = 1e-3

        def loss():
            return loss_and_gradient(
                brain.late.adjusted(outputs, obs), 0, logs[0], 1, 0.1, brain.config
            )[0]

        weights[i] = original + eps
        plus = loss()
        weights[i] = original - eps
        minus = loss()
        weights[i] = original
        self.assertAlmostEqual(
            (plus - minus) / (2 * eps), gradients[0]["weights"][i], places=4
        )
        model = prepare_model(model_snapshot(brain))
        explanation = explain_model(model, obs, brain.actions[0])
        f = l.consequence_start + 1
        up = obs[:]
        down = obs[:]
        up[f] += eps
        down[f] -= eps
        approx = (
            brain.action_probabilities(up)[0] - brain.action_probabilities(down)[0]
        ) / (2 * eps)
        self.assertAlmostEqual(explanation["features"][f]["gradient"], approx, places=5)
        empty = obs[:]
        empty[l.physical_width :] = [0.0] * (l.width - l.physical_width)
        self.assertEqual(brain.late.residuals(empty), [0.0] * len(brain.actions))
        self.assertAlmostEqual(sum(probabilities), 1)

    def test_accepted_update_restores_nonzero_private_adapter_state(self):
        e = config()
        b = next(iter(Experiment(e).brains.values()))
        b.config.update(target_kl=0, normalize_advantages=False)
        obs = [0.0] * observation_inputs(b.config)
        obs[0] = 0.4
        obs[1] = 0.6
        start = compact_layout(b.config).consequence_start
        obs[start : start + 4] = [1, 0.2, 0.8, 0.1]

        def update(brain):
            outputs = brain.network.predict(obs)
            _, logs = brain.distribution(outputs, obs)
            brain.rollout = [
                RolloutStep(tuple(obs), 0, 1, logs[0], outputs[-1], 0, True, False)
            ]
            brain._update()

        update(b)
        self.assertGreater(b.late.updates, 0)
        self.assertTrue(any(b.late.network.layers[0]["weights"]))
        restored = type(b).restore(json.loads(json.dumps(b.checkpoint())))
        self.assertEqual(b.checkpoint(), restored.checkpoint())
        update(b)
        update(restored)
        self.assertEqual(b.checkpoint(), restored.checkpoint())

    def test_kl_rejection_does_not_update_adapter(self):
        e = Experiment(config())
        b = next(iter(e.brains.values()))
        obs = next(iter(e.observations.values()))
        b.config.update(target_kl=1e-12, rollout_steps=1)
        b.pending = None
        outputs = b.network.predict(obs)
        prob, logs = b.distribution(outputs, obs)
        b.rollout = [RolloutStep(obs, 0, 1, logs[0] - 5, outputs[-1], 0, True, False)]
        before = b.late.checkpoint()
        b._update()
        self.assertTrue(b.last_metrics["early_stopped"])
        self.assertEqual(before, b.late.checkpoint())
        self.assertAlmostEqual(sum(prob), 1)

    def test_v7_replay_keeps_features_and_late_diagnostics(self):
        e = Experiment(config())
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(folder, e.config, "train")
            # Exercise the public recording lifecycle exactly as a caller would.
            # pylint: disable=duplicate-code
            recorder.begin_run(e)
            while not e.done:
                e.step()
                recorder.capture(e, force=e.finished_run)
            recorder.end_run(e)
            recorder.close()
            meta = json.loads((Path(folder) / "playback/recording.json").read_text())
            frames = [
                f
                for i in range(meta["runs"][0]["chunk_count"])
                for f in load_chunk(chunk_path(Path(folder) / "playback", 1, i))
            ]
            self.assertTrue(frames)
            self.assertEqual(
                meta["config"]["brains"]["prey_ppo"]["observation_version"], 7
            )


if __name__ == "__main__":
    unittest.main()
