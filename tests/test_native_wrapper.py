"""Boundary contract checks; staged independently of a compiled Session API."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

from array import array
import copy
import unittest
from unittest.mock import Mock, patch
from tests.test_cpp_controller_lifecycle import configuration
from tests.test_cpp_feature_integration import configuration as rich_config
from wtf import load_config
from wtf.experiment.runner import Experiment

from wtf.experiment import native


class NativeWrapperTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("environments/ppo.json")
        for species in self.config["species"].values():
            species["initial_population"] = 2
        self.config["experiment"]["runs"] = 2
        self.experiment = Experiment(self.config)
        self.engine = Mock()
        self.engine.create.return_value = object()
        self.engine.run.return_value = {
            "tick": 3,
            "finished_run": False,
            "total_updates": 7,
        }
        self.engine.checkpoint.side_effect = lambda handle: copy.deepcopy(
            self.experiment.checkpoint()
        )

    def test_one_native_run_and_no_python_restoration(self):
        with patch.object(native, "_engine_module", return_value=self.engine):
            session = native.NativeExperiment.from_experiment(self.experiment)
        self.engine.create.assert_called_once()
        payload = self.engine.create.call_args.args[0]
        self.assertEqual(set(payload), {"checkpoint", "observations", "signatures"})
        with patch.object(
            Experiment, "restore", side_effect=AssertionError("unexpected restore")
        ):
            report = session.run(3)
            self.assertEqual(report["tick"], 3)
            self.assertEqual(session.total_updates, 7)
            self.assertFalse(session.finished_run)
            self.assertFalse(session.done)
            self.assertEqual(session.run_index, 0)
            session.snapshot()
        self.engine.run.assert_called_once_with(session._handle, 3, "", {})
        self.engine.checkpoint.assert_not_called()
        copy_config = session.config
        copy_config["name"] = "changed"
        self.assertNotEqual(copy_config, session.config)

    def test_model_import_uses_exact_typed_buffers_without_rounding_other_data(self):
        state = self.experiment.checkpoint()
        source = copy.deepcopy(state)
        legacy = {
            "precision": "float64",
            "parameters": [{"weights": [0.123456789012345], "bias": [-0.0]}],
        }
        state["initial_weights"]["legacy"] = legacy
        state["history"] = [{"food": 0.123456789012345}]
        result = native._pack_import_models(state)
        for key, brain in result["brains"].items():
            for field in ("network", "moments", "variances"):
                for before, after in zip(source["brains"][key][field], brain[field]):
                    for name in ("weights", "bias"):
                        self.assertIsInstance(after[name], array)
                        self.assertEqual(after[name].itemsize, 4)
                        self.assertEqual(list(after[name]), before[name])
        self.assertIs(result["initial_weights"]["legacy"], legacy)
        self.assertEqual(legacy["parameters"][0]["weights"], [0.123456789012345])
        self.assertEqual(result["history"][0]["food"], 0.123456789012345)
        self.assertIsInstance(
            next(iter(self.experiment.brains.values())).checkpoint()["network"][0][
                "weights"
            ],
            list,
        )

    def test_noncanonical_world_samples_reject_before_lossy_heading_conversion(self):
        animal_id = next(iter(self.experiment.observations))
        values = list(self.experiment.observations[animal_id])
        values[2] = 0.25
        self.experiment.observations[animal_id] = values
        with patch.object(native, "_engine_module", return_value=self.engine):
            with self.assertRaisesRegex(ValueError, "canonical integer world headings"):
                native.NativeExperiment.from_experiment(self.experiment)

    def test_queries_are_independent(self):
        with patch.object(native, "_engine_module", return_value=self.engine):
            session = native.NativeExperiment.from_experiment(self.experiment)
        queried = session.query()
        queried.world.tick = 99
        self.assertEqual(self.experiment.world.tick, 0)
        self.assertEqual(session.checkpoint()["world"]["snapshot"]["tick"], 0)
        self.engine.observations.return_value = {"5": [1.0, 2.0]}
        self.assertEqual(session.observations, {5: [1.0, 2.0]})

    def test_custom_target_network_hooks_reject_before_native_import(self):
        experiment = Experiment(configuration("dqn"))
        brain = next(iter(experiment.brains.values()))
        brain.target_network.predict = lambda observation: [0.0] * len(brain.actions)
        with patch.object(native, "_engine_module", return_value=self.engine):
            with self.assertRaisesRegex(ValueError, "Python network overrides"):
                native.NativeExperiment.from_experiment(experiment)
        self.engine.create.assert_not_called()

    def test_instance_execution_hooks_reject_before_native_import(self):
        cases = (
            ("ppo", "experiment", "step"),
            ("grpo", "brain", "update_group"),
            ("ppo", "world", "_river_drift"),
            ("ppo", "network", "_predict"),
        )
        for kind, owner, method in cases:
            experiment = Experiment(configuration(kind))
            brain = next(iter(experiment.brains.values()))
            target = {
                "experiment": experiment,
                "world": experiment.world,
                "brain": brain,
                "network": brain.network,
            }[owner]
            # Replace an actual method; adding arbitrary metadata is unrelated.
            self.assertTrue(callable(getattr(type(target), method)))
            setattr(target, method, lambda *_args, **_kwargs: None)
            with (
                self.subTest(owner=owner, method=method),
                patch.object(native, "_engine_module", return_value=self.engine),
                self.assertRaisesRegex(
                    ValueError, "ordinary Experiment|Python .* overrides"
                ),
            ):
                native.NativeExperiment.from_experiment(experiment)
            self.engine.create.assert_not_called()

    def test_optional_head_execution_hooks_reject_before_native_import(self):
        for field, method in (("late", "adjusted"), ("auxiliary", "targets")):
            experiment = Experiment(rich_config("ppo", 2))
            head = getattr(next(iter(experiment.brains.values())), field)
            setattr(head, method, lambda *_args, **_kwargs: None)
            with (
                self.subTest(head=field),
                patch.object(native, "_engine_module", return_value=self.engine),
                self.assertRaisesRegex(ValueError, "Python head overrides"),
            ):
                native.NativeExperiment.from_experiment(experiment)
            self.engine.create.assert_not_called()

    def test_shared_helpers_and_private_rng_hooks_reject_before_import(self):
        cases = (
            ("social", "finish"),
            ("knowledge", "learn"),
            ("geography", "depth"),
            ("visual", "observe"),
            ("scent", "advance"),
            ("champions", "consider"),
            ("lineage", "rank"),
            ("stopping", "finish_run"),
            ("world_rng", "random"),
            ("brain_rng", "random"),
            ("training_rng", "shuffle"),
            ("seed_rng", "getrandbits"),
        )
        for field, method in cases:
            config = rich_config("ppo", 2)
            config["experiment"]["world_seed_policy"] = "random"
            experiment = Experiment(config)
            brain = next(iter(experiment.brains.values()))
            owners = {
                "social": experiment.world._social,
                "knowledge": experiment.world._social.knowledge,
                "geography": experiment.world.geography,
                "visual": experiment.world._tracking.visual,
                "scent": experiment.world._tracking.scent,
                "champions": experiment.champions,
                "lineage": experiment.champions.lineage,
                "stopping": experiment.early_stopping,
                "world_rng": experiment.world.rng,
                "brain_rng": brain.rng,
                "training_rng": brain.training_rng,
                "seed_rng": experiment.seed_rng,
            }
            setattr(owners[field], method, lambda *_args, **_kwargs: None)
            with (
                self.subTest(owner=field, method=method),
                patch.object(native, "_engine_module", return_value=self.engine),
                self.assertRaisesRegex(ValueError, "Python .* overrides"),
            ):
                native.NativeExperiment.from_experiment(experiment)
            self.engine.create.assert_not_called()

    def test_rejects_unknown_execution_mode_before_bootstrap(self):
        with patch.object(native, "Experiment") as constructor:
            with self.assertRaisesRegex(ValueError, "train, evaluate or random"):
                native.NativeExperiment(self.config, mode="custom")
            constructor.assert_not_called()

    def test_no_boundary_restore_until_finished(self):
        with patch.object(native, "_engine_module", return_value=self.engine):
            session = native.NativeExperiment.from_experiment(self.experiment)
        self.assertFalse(session.next_run())
        self.engine.checkpoint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
