"""Personal information boundaries, anonymous scent, exact resume and compact replays."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from wtf.inspection.explainability import feature_names, prepare_model
from wtf.learning.brain import observation_inputs
from wtf.learning.brains import model_snapshot, load_model_weights
from wtf.simulation.tracking import Tracking, tracking_names, tracking_width
from wtf.simulation.world import Animal, World
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.recording import Recorder
from wtf.storage.replay_io import load_chunk
from tests.test_mutation import KINDS, environment
from tests.test_simulation import config
from tests.reproduction_support import force_reproduction


def tracked(c, steps=3, scent=True):
    if scent:
        c["scent_trails"] = dict(enabled=True, lifetime_steps=8)
    for b in c["brains"].values():
        b["movement_history_steps"] = steps
        if scent:
            b["scent_observation"] = dict(species=list(c["species"]), radius=2)
    return c


def scene(steps=3, scent=False):
    c = tracked(config(), steps, scent)
    c["obstacles"].update(generation="explicit", explicit_cells=[])
    for spec in c["species"].values():
        spec["body"]["vision_radius"] = 4
    w = World(c, 100)
    w.animals = {
        0: Animal(0, "predator", 2, 2, 0, 40, 30, sex="male"),
        1: Animal(1, "prey", 4, 2, 0, 40, 30, sex="female"),
    }
    w.resources = {}
    if scent:
        w._tracking.scent.cells.clear()
        w._tracking.scent.advance(w)
    return w


class MovementTests(unittest.TestCase):
    def test_personal_sightings_velocity_and_observer_motion(self):
        w = scene()
        a, target = w.animals.values()
        first = w.observe(a)[20:]
        self.assertEqual(first[:7], (0.5, 0, 0, 0, 0, 1, 1))
        self.assertEqual(first[7:16], (0,) * 9)
        w.tick = 1
        target.x = 5
        second = w.observe(a)[20:]
        self.assertEqual(second[:4], (0.75, 0, 0.25, 0))
        self.assertEqual(second[7:10], (0.5, 0, 1))
        # Own movement changes relative historical positions, not target velocity.
        w.tick = 2
        a.x = 3
        third = w.observe(a)[20:]
        self.assertEqual(third[:4], (0.5, 0, 0, 0))
        self.assertEqual(third[7:10], (0.5, 0, 1))
        self.assertEqual(w.observe(a), w.observe(a))
        self.assertEqual(len(w._tracking.visual.frames[0]), 3)

    def test_hidden_movement_death_and_unsighted_arrivals_do_not_leak(self):
        w = scene()
        a, target = w.animals.values()
        w.observe(a)
        w.tick = 1
        target.x = 9
        hidden = w.observe(a)[20:]
        self.assertEqual(hidden[:2], (0.5, 0))
        self.assertEqual(hidden[4:7], (1 / 3, 0.75, 0))
        target.y = 7
        self.assertEqual(w.observe(a)[20:], hidden)
        del w.animals[1]
        self.assertEqual(w.observe(a)[20:], hidden)
        w.tick = 4
        self.assertEqual(w.observe(a)[20:], (0,) * 32)
        # Seeing a new animal never recovers movements made before detection.
        w.animals[2] = Animal(2, "prey", 3, 2, 0, 40, 30)
        fresh = w.observe(a)[20:]
        self.assertEqual(fresh[7:16], (0,) * 9)

    def test_reordering_does_not_splice_target_histories_and_caches_stay_bounded(self):
        c = tracked(config(), 2, False)
        c["obstacles"].update(generation="explicit", explicit_cells=[])
        for b in c["brains"].values():
            b["observation_top_k"] = 2
        w = World(c, 10)
        w.animals = {
            0: Animal(0, "predator", 2, 2, 0, 40, 30),
            1: Animal(1, "prey", 3, 2, 0, 40, 30),
            2: Animal(2, "prey", 5, 2, 0, 40, 30),
        }
        w.observe(w.animals[0])
        w.tick = 1
        w.animals[1].x = 6
        result = w.observe(w.animals[0])[32:]
        self.assertEqual(
            result[7:10], (3 / w.body(w.animals[0])["vision_radius"], 0, 1)
        )
        self.assertEqual(
            result[20:23], (1 / w.body(w.animals[0])["vision_radius"], 0, 1)
        )
        for tick in range(2, 50):
            w.tick = tick
            w.observe_all()
        self.assertTrue(all(len(v) <= 3 for v in w._tracking.visual.frames.values()))
        del w.animals[1]
        w.observe_all()
        self.assertNotIn(1, w._tracking.visual.frames)


class ScentTests(unittest.TestCase):
    def test_anonymous_trails_merge_and_preserve_multiple_directions_and_sexes(self):
        w = scene(0, True)
        w.animals[1].x = 1
        w.animals[2] = Animal(2, "prey", 3, 2, 0, 40, 30, sex="female")
        w.animals[3] = Animal(3, "prey", 2, 1, 0, 40, 30, sex="male")
        scent = w._tracking.scent
        scent.cells.clear()
        scent.advance(w)
        template = dict(scent_observation=dict(species=["prey"], radius=2))
        values = scent.sample(w, w.animals[0], template)
        self.assertEqual(values[3], 1)  # east female
        self.assertEqual(values[7], 1)  # west female; not a single selected prey
        self.assertEqual(values[9 + 1], 1)  # north male
        del w.animals[1]
        self.assertEqual(scent.sample(w, w.animals[0], template), values)
        w.animals[4] = Animal(4, "prey", 3, 2, 0, 40, 30, sex="female")
        scent.advance(w)
        self.assertEqual(len(scent.cells), 4)  # overlapping same-group traces merge
        for tick in range(1, 50):
            w.tick = tick
            scent.advance(w)
        self.assertLessEqual(len(scent.cells), 4)
        self.assertTrue(
            all(
                isinstance(k, int) and isinstance(t, int)
                for k, t in scent.cells.items()
            )
        )

    def test_fading_expiration_walls_and_water(self):
        w = scene(0, True)
        scent = w._tracking.scent
        template = dict(scent_observation=dict(species=["prey"], radius=2))
        del w.animals[1]
        w.tick = 4
        scent.advance(w)
        self.assertEqual(scent.sample(w, w.animals[0], template)[3], 0.5)
        w.obstacles = {(3, y) for y in range(w.height)}
        self.assertEqual(scent.sample(w, w.animals[0], template), [0] * 18)
        w.obstacles.clear()
        w.geography = SimpleNamespace(depth=lambda p: int(p[0] == 3))
        self.assertEqual(scent.sample(w, w.animals[0], template), [0] * 18)
        w.geography = None
        w.tick = 8
        scent.advance(w)
        self.assertEqual(scent.sample(w, w.animals[0], template), [0] * 18)
        self.assertEqual(len(scent.cells), 1)  # only observer's refreshed own scent


class TrackingIntegrationTests(unittest.TestCase):
    def test_disabled_layout_signatures_and_state_are_legacy_compatible(self):
        c = config()
        e = Experiment(c)
        self.assertIsNone(e.world._tracking)
        self.assertNotIn("sensory_state", e.world.checkpoint())
        for b in c["brains"].values():
            b["movement_history_steps"] = 0
        new = Experiment(c)
        for _ in range(3):
            e.step()
            new.step()
            self.assertEqual(e.world.snapshot(), new.world.snapshot())
            self.assertEqual(e.observations, new.observations)
        self.assertEqual(model_snapshot(new.brains[0]), model_snapshot(e.brains[0]))
        self.assertEqual(e.observations, new.observations)

    def test_all_algorithms_learning_births_and_exact_binary_resume(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = tracked(environment(kind, 0), 2)
                for b in c["brains"].values():
                    b.update(observation_top_k=2, observation_version=3)
                e = force_reproduction(Experiment(c))
                e.step()
                self.assertTrue(any(a.generation > 0 for a in e.world.animals.values()))
                for i, a in e.world.animals.items():
                    b = e.brains[i]
                    self.assertEqual(
                        len(e.observations[i]), observation_inputs(b.config)
                    )
                    self.assertEqual(
                        len(feature_names(model_snapshot(b)["signature"])),
                        len(e.observations[i]),
                    )
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "checkpoint.bin.xz"
                    e.write_checkpoint(path)
                    r = Experiment.restore(read_checkpoint(path))
                self.assertEqual(e.checkpoint(), r.checkpoint())
                while not e.finished_run:
                    e.step()
                    r.step()
                    self.assertEqual(e.checkpoint(), r.checkpoint())
                e.next_run()
                self.assertTrue(
                    all(len(v) == 1 for v in e.world._tracking.visual.frames.values())
                )

    def test_models_and_explanations_reject_different_tracking_contracts(self):
        c = tracked(environment("ppo", 0), 2)
        e = Experiment(c)
        b = e.brains[0]
        model = model_snapshot(b)
        prepare_model(model)
        self.assertEqual(len(tracking_names(b.config)), tracking_width(b.config))
        c["brains"][c["species"][e.world.animals[0].species]["brain"]][
            "scent_observation"
        ]["species"].reverse()
        other = Experiment(c)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_model_weights(other.brains[0], model)
        for steps in (-1, 17, True):
            invalid = copy.deepcopy(model)
            invalid["signature"]["movement_history_steps"] = steps
            with self.assertRaises(ValueError):
                prepare_model(invalid)

    def test_grpo_reset_owns_its_sensory_state(self):
        w = scene(3, True)
        w.observe_all()
        state = w.checkpoint()
        trial = World.restore(state)
        trial.step({i: "rest" for i in trial.animals})
        self.assertEqual(w.checkpoint(), state)
        trial._reset_trial(state)
        self.assertEqual(trial.checkpoint(), state)
        self.assertEqual(w.observe_all(), trial.observe_all())
        self.assertIsNot(trial._tracking.scent.cells, w._tracking.scent.cells)

    def test_binary_replay_reuses_sampled_features_without_world_scent_or_histories(
        self,
    ):
        c = tracked(environment("ppo", 0), 2)
        c["recording"].update(
            encoding="binary",
            compression="xz",
            frames_per_chunk=2,
            diagnostics_every_steps=2,
        )
        expected, e = Experiment(c), Experiment(c)
        with tempfile.TemporaryDirectory() as folder:
            writer = Recorder(folder, c, "train")
            writer.begin_run(e)
            while not e.finished_run:
                expected.step()
                e.step()
                writer.capture(e, force=e.finished_run)
            writer.end_run(e)
            writer.close()
            paths = sorted(Path(folder).rglob("*.bin.xz"))
            self.assertGreater(len(paths), 1)
            for path in paths:
                for frame in load_chunk(path):
                    self.assertNotIn("sensory_state", frame)
                    self.assertNotIn("scent", frame)
                    for d in frame.get("diagnostics", {}).values():
                        self.assertTrue(d["features"])
        self.assertEqual(e.checkpoint(), expected.checkpoint())

    def test_json_key_reordering_and_trial_reset_preserve_scent_channel_identity(self):
        w = scene(3, True)
        w.observe_all()
        state = json.loads(json.dumps(w.checkpoint(), sort_keys=True))
        r = World.restore(state)
        self.assertEqual(w.observe_all(), r.observe_all())
        r._reset_trial(w.checkpoint())
        self.assertEqual(w.observe_all(), r.observe_all())
        del state["sensory_state"]
        with self.assertRaisesRegex(ValueError, "sensory_state"):
            World.restore(state)

    def test_native_and_reference_visibility_feed_identical_personal_histories(self):
        w = scene(3, True)
        r = World.restore(w.checkpoint())
        for tick in range(5):
            w.tick = r.tick = tick
            w.animals[1].x = r.animals[1].x = 4 + tick
            expected = w.observe_all()
            # An unrecognized callback deliberately takes the established Python
            # selector, without changing visibility itself or other kernels.
            with patch.object(
                r, "_perception_visible", side_effect=r._perception_visible
            ):
                actual = r.observe_all()
            self.assertEqual(expected, actual)

    def test_validation_and_shipped_preset(self):
        validate(load("tracking"))
        for key, value in (
            ("movement_history_steps", -1),
            ("movement_history_steps", 17),
            ("movement_history_steps", True),
            ("scent_observation", dict(species=["missing"])),
            ("scent_observation", dict(species=["prey", "prey"])),
            ("scent_observation", dict(species=["prey"], radius=9)),
            ("scent_observation", dict(species=["prey"], separate_sexes=1)),
        ):
            c = tracked(config())
            next(iter(c["brains"].values()))[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate(c)
        c = tracked(config())
        c["scent_trails"]["enabled"] = False
        with self.assertRaises(ValueError):
            validate(c)
