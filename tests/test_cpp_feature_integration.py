"""Native experiment integration preserves features, model state and sampled XAI."""

# Fixtures inspect native ownership and force policy outputs without physics hooks.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# pylint: disable=c-extension-no-member
import copy
import unittest
from unittest.mock import patch
from tests.native_core_support import _engine, encoded, NATIVE_KINDS as KINDS
from tests.test_cpp_controller_lifecycle import configuration as learner_config
from tests.reproduction_support import force_reproduction
from tests.test_cpp_value_policies import mismatch
from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from wtf.experiment.native import NativeExperiment
from wtf.simulation.world import World


def configuration(kind, steps=12):
    config = copy.deepcopy(load("land_bridge_collective"))
    template = next(iter(learner_config(kind)["brains"].values()))
    config["world"]["max_steps"] = steps
    config["experiment"].update(
        runs=1, world_seed_policy="fixed", record_metrics_every_steps=1
    )
    config["consequence_learning"].update(
        horizon_steps=2, update_schedule="continuous", max_contexts=32
    )
    config["species"]["grazer"] = copy.deepcopy(config["species"]["prey"])
    config["brains"] = {}
    for name, species in config["species"].items():
        species["initial_population"] = 2
        species["brain"] = name + "_learner"
        species["reproduction"]["max_population"] = 8
        brain = copy.deepcopy(template)
        brain.update(
            observation_version=7,
            observation_top_k=1,
            movement_history_steps=2,
            scent_observation={"mode": "blended", "radius": 2},
            social_observation={"history_steps": 2, "max_neighbors": 4},
            consequence_observation=True,
            consequence_scope="both",
        )
        brain.pop("observation_actions", None)
        if kind in ("ppo", "maskable_ppo"):
            brain["auxiliary_consequence"] = {"coefficient": 0.1}
            brain["late_consequence"] = {"max_adjustment": 1}
        config["brains"][species["brain"]] = brain
    return validate(config)


@unittest.skipUnless(_engine is not None, "Build the optional native episode engine")
class NativeFeatureIntegrationTests(unittest.TestCase):
    def assert_state(self, reference, native, label):
        self.assertIsNone(mismatch(reference.checkpoint(), native.checkpoint()), label)
        actual = _engine.snapshot(native._handle, True, True)["diagnostics"]
        for animal_id, observation in reference.observations.items():
            expected = reference.brains[animal_id].diagnostics(
                observation, include_activations=True
            )
            self.assertEqual(
                encoded(expected),
                encoded(actual[str(animal_id)]),
                (label, animal_id, "XAI"),
            )

    def run_pair(self, config, mode="train", frozen=False, setup=None):
        reference = Experiment(config, mode, freeze_evaluation_brains=frozen)
        if setup is not None:
            setup(reference)
        native = (
            NativeExperiment.from_experiment(reference)
            if setup is not None or frozen
            else NativeExperiment(config, mode)
        )
        self.assert_state(reference, native, "initial")
        for step in range(config["world"]["max_steps"]):
            reference.step()
            native.run(1)
            self.assert_state(reference, native, (config["name"], mode, frozen, step))
            if step == 5:
                reference = Experiment.restore(reference.checkpoint())
                native = NativeExperiment.restore(native.checkpoint())
                self.assert_state(reference, native, "restored")
        return reference

    def test_all_learners_all_features_training_and_restore(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.run_pair(configuration(kind))

    def test_native_bootstrap_and_steps_do_not_construct_or_step_python_worlds(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                config = configuration(kind)
                reference = Experiment(config)
                reference.step()
                with (
                    patch.object(
                        Experiment,
                        "__init__",
                        side_effect=AssertionError("Python experiment bootstrap"),
                    ),
                    patch.object(
                        World,
                        "__init__",
                        side_effect=AssertionError("Python world bootstrap"),
                    ),
                    patch.object(
                        World, "step", side_effect=AssertionError("Python physics")
                    ),
                ):
                    native = NativeExperiment(config)
                    native.run(1)
                self.assert_state(
                    reference, native, (kind, "native-only bootstrap/step")
                )

    def test_nontraining_modes_keep_private_state_and_xai(self):
        for kind in KINDS:
            for mode, frozen in (
                ("evaluate", False),
                ("evaluate", True),
                ("random", False),
            ):
                with self.subTest(kind=kind, mode=mode, frozen=frozen):
                    self.run_pair(configuration(kind), mode, frozen)

    def test_proximal_birth_mutation_death_and_restored_continuation(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                config = configuration(kind, 24)
                for species in config["species"].values():
                    species["reproduction"].update(
                        min_age_steps=1,
                        cooldown_steps=3,
                        health_threshold=0,
                        min_energy=1,
                        energy_cost=1,
                        max_population=6,
                    )
                    species["mutation"].update(
                        probability_per_birth=1, max_changed_weights=1
                    )
                # Use the same rich sensing/social templates on an unobstructed
                # board to isolate paired inheritance from terrain reachability.
                config["terrain"] = {
                    "map_version": 1,
                    "name": "flat lifecycle fixture",
                    "width": config["world"]["width"],
                    "height": config["world"]["height"],
                    "rows": ["." * config["world"]["width"]]
                    * config["world"]["height"],
                    "legend": {".": {"elevation": 0}},
                }
                config["obstacles"].update(generation="explicit", explicit_cells=[])
                for species in config["species"].values():
                    species.pop("spawn", None)

                def setup(reference):
                    for index, animal in enumerate(reference.world.animals.values()):
                        animal.x, animal.y = 3 + index % 2, 3 + 4 * (index // 2)
                        animal.sex = "female" if index % 2 == 0 else "male"
                        if index == 4:
                            animal.energy = 0.0
                            animal.health = 0.1
                    reference.observations = reference.world.observe_all()
                    force_reproduction(reference)

                reference = self.run_pair(validate(config), setup=setup)
                self.assertGreater(sum(reference.world.births.values()), 0)
                self.assertGreater(sum(reference.world.mutations.values()), 0)
                self.assertGreater(sum(reference.world.deaths.values()), 0)
