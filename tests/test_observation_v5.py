"""Body-relative encoding, sensed affordances and versioned lifecycle contracts."""

# Regression fixture names describe their assertions; internals verify RNG isolation.
# pylint: disable=missing-function-docstring,protected-access
import copy
import json
import random
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from wtf.inspection.explainability import feature_names, prepare_model, explain_model
from wtf.learning.brain import observation_inputs
from wtf.learning.brains import model_snapshot, load_model_weights
from wtf.learning.contextual_ppo import action_mask
from wtf.native import kernels
from wtf.simulation.actions import DIRECTIONS
from wtf.simulation.observations import relative, encode_v5
from wtf.simulation.tracking import SECTORS, tracking_names
from wtf.storage.recording import Recorder
from wtf.storage.replay_io import load_chunk, chunk_path
from tests.test_auxiliary_consequence import auxiliary_config
from tests.test_forward_vision import scene
from tests.test_mutation import KINDS, environment
from work.compare_observation_versions import frozen_sampler


class ObservationV5Tests(unittest.TestCase):
    """Older vectors stay unchanged and new vectors have explicit semantics."""

    def test_world_physics_rng_and_target_encoding_across_headings(self):
        for heading, (hx, hy) in enumerate(DIRECTIONS):
            w = scene()
            w.animals[0].heading = heading
            before = w.observe_all()[0]
            rng = w.rng.getstate()
            for b in w.config["brains"].values():
                b["observation_version"] = 5
            after = w.observe_all()[0]
            self.assertEqual(w.rng.getstate(), rng)
            self.assertEqual(len(after), len(before) + 3)
            for i in range(8, 44, 4):
                dx, dy = before[i : i + 2]
                self.assertEqual(
                    after[i : i + 2], (dx * hx + dy * hy, -dx * hy + dy * hx)
                )
                self.assertEqual(after[i + 2 : i + 4], before[i + 2 : i + 4])
            for b in w.config["brains"].values():
                b["observation_version"] = 4
            self.assertEqual(w.observe_all()[0], before)

    def test_tracking_flow_mates_and_scent_rotation_all_headings(self):
        template = {
            "observation_top_k": 2,
            "movement_history_steps": 2,
            "scent_observation": {"mode": "blended", "radius": 2},
        }
        core = 57
        width = core + 4 * 13 + 9
        for heading in range(4):
            v = [0.0] * width
            v[core : core + 4] = [0.2, -0.3, 0.1, 0.4]
            v[core + 7 : core + 9] = [0.6, -0.7]
            v[33:35] = [0.25, -0.5]  # Flow after k=2 target blocks.
            v[48:50] = [0.3, 0.2]  # Mate offsets.
            v[-9:] = list(range(9))
            old = v.copy()
            encoded = encode_v5(v, template, heading, (1, 0, 1))
            self.assertEqual(encoded[core : core + 3], [1, 0, 1])
            self.assertEqual(
                tuple(encoded[core + 3 : core + 5]), relative(0.2, -0.3, heading)
            )
            self.assertEqual(
                tuple(encoded[core + 5 : core + 7]), relative(0.1, 0.4, heading)
            )
            self.assertEqual(tuple(encoded[33:35]), relative(0.25, -0.5, heading))
            self.assertEqual(tuple(encoded[48:50]), relative(0.3, 0.2, heading))
            hx, hy = DIRECTIONS[heading]
            for i, (right, down) in enumerate(SECTORS):
                world = (-down * hx - right * hy, -down * hy + right * hx)
                self.assertEqual(encoded[-9 + i], old[-9 + SECTORS.index(world)])

    def test_affordances_are_inputs_and_do_not_mask_ordinary_ppo(self):
        c = load("environments/seaside_150_v5.json")
        e = Experiment(c)
        for a in e.world.animals.values():
            b = e.brains[a.id]
            core = 45 + 12 * (b.config["observation_top_k"] - 1)
            allowed = tuple(
                name for name in ("attack", "eat", "swim", "rest") if name in b.actions
            )
            mask = dict(zip(allowed, action_mask(e.world, a, allowed)))
            expected = tuple(
                mask.get(name, False) for name in ("attack", "eat", "swim")
            )
            self.assertEqual(
                e.observations[a.id][core : core + 3],
                tuple(
                    float(v and name in b.actions)
                    for v, name in zip(expected, ("attack", "eat", "swim"))
                ),
            )
            self.assertEqual(b.algorithm, "ppo")
            self.assertFalse(hasattr(b, "mask"))
            self.assertTrue(
                all(v > 0 for v in b.action_probabilities(e.observations[a.id]))
            )

    def test_all_learners_widths_explanations_and_exact_restore(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = environment(kind, 0)
                for b in c["brains"].values():
                    b.update(observation_version=5, observation_top_k=2)
                e = Experiment(validate(c))
                e.step()
                for a in e.world.animals.values():
                    brain = e.brains[a.id]
                    self.assertEqual(
                        len(e.observations[a.id]), observation_inputs(brain.config)
                    )
                    model = model_snapshot(brain)
                    names = feature_names(model["signature"])
                    self.assertEqual(len(names), len(e.observations[a.id]))
                    explain_model(prepare_model(model), e.observations[a.id])
                restored = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
                for _ in range(2):
                    e.step()
                    restored.step()
                self.assertEqual(e.checkpoint(), restored.checkpoint())

    def test_v4_weights_rejected_and_auxiliary_v5_restores(self):
        # Keep this independent versioned restore scenario explicit.
        # pylint: disable=duplicate-code
        c = load("environments/seaside_150_v5.json")
        c["species"]["prey"]["initial_population"] = 2
        c["species"]["predator"]["initial_population"] = 2
        e = Experiment(c)
        before = e.checkpoint()
        e.step()
        clone = Experiment.restore(e.checkpoint())
        e.step()
        clone.step()
        self.assertEqual(e.checkpoint(), clone.checkpoint())
        old = copy.deepcopy(c)
        for b in old["brains"].values():
            b["observation_version"] = 4
        legacy = Experiment(old)
        with self.assertRaises(ValueError):
            load_model_weights(e.brains[0], model_snapshot(legacy.brains[0]))
        self.assertEqual(
            before["config"]["brains"]["prey_ppo"]["observation_version"], 5
        )

    def test_v5_auxiliary_updates_and_replay_keep_exact_vectors(self):
        c = auxiliary_config()
        for brain in c["brains"].values():
            brain.update(observation_version=5, rollout_steps=2)
        c["world"]["max_steps"] = 12
        c.setdefault("recording", {}).update(
            enabled=True, diagnostics_every_steps=1, record_last_episodes=-1
        )
        e = Experiment(c)
        while e.run_index == 0 or not any(
            b.auxiliary.updates for b in e.brains.values()
        ):
            e.step()
        restored = Experiment.restore(e.checkpoint())
        e.step()
        restored.step()
        self.assertEqual(e.checkpoint(), restored.checkpoint())
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory), e.config, mode="train")
            recorder.begin_run(e)
            recorder.capture(e, force=True)
            expected = {str(i): list(o) for i, o in e.observations.items()}
            recorder.end_run(e)
            recorder.close()
            frames = load_chunk(
                chunk_path(Path(directory) / "playback", e.run_index + 1, 0)
            )
            for identity, d in frames[-1]["diagnostics"].items():
                self.assertEqual(d["features"], expected[str(identity)])

    def test_arbitrary_species_names_are_not_rewritten_as_coordinates(self):
        names = tracking_names(
            {
                "observation_version": 5,
                "scent_observation": {"species": ["red dx"], "radius": 2},
            }
        )
        self.assertIn("Scent red dx female front", names)
        self.assertIn("Scent red dx male right", names)

    def test_frozen_sampler_allocates_one_rng_per_animal_and_preserves_choices(self):
        sampler = frozen_sampler(19)
        brain = SimpleNamespace(
            actions=["rest", "eat"], action_probabilities=lambda _o: [0.25, 0.75]
        )
        original = random.Random
        with patch(
            "work.compare_observation_versions.random.Random", wraps=original
        ) as constructor:
            actual = [sampler({7: brain}, {7: (0.0,)})[7] for _ in range(10)]
            self.assertEqual(constructor.call_count, 1)
        expected = original((19 << 32) ^ 7)
        self.assertEqual(
            actual,
            [
                expected.choices(brain.actions, weights=[0.25, 0.75])[0]
                for _ in range(10)
            ],
        )

    def test_native_and_python_perception_agree_for_v5(self):
        # Independently exercise native dispatch with the V5 projection enabled.
        # pylint: disable=duplicate-code
        w = scene()
        for b in w.config["brains"].values():
            b["observation_version"] = 5
        expected = w.observe_all()
        available = kernels.available
        with patch.object(
            kernels,
            "available",
            side_effect=lambda name: name not in ("nearest", "nearest_many")
            and available(name),
        ):
            for index in w._perception:
                index.results.clear()
            self.assertEqual(w.observe_all(), expected)
