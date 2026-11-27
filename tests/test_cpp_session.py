"""Native episodes preserve lifecycle, querying and exact playback artifacts."""

# Controlled fixtures inspect ownership and force choices without bypassing physics.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# Exercise the same public workflow as the benchmark; no extra wrapper is needed.
# pylint: disable=duplicate-code


import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.native_core_support import _engine, encoded
from tests.reproduction_support import force_reproduction
from wtf.cli import run_experiment
from wtf.config import load
from wtf.experiment.native import NativeExperiment
from wtf.experiment.runner import Experiment
from wtf.learning.ppo import PPOBrain
from wtf.simulation.world import World
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.recording import Recorder


def configuration(steps=48):
    config = load("ppo")
    config["seed"] = 100
    config["world"]["max_steps"] = steps
    config["experiment"].update(runs=1, world_seed_policy="fixed")
    for species in config["species"].values():
        species["initial_population"] = 2
    for template in config["brains"].values():
        template.update(rollout_steps=7, batch_size=3, epochs=2)
    return config


@unittest.skipUnless(_engine is not None, "Build the optional native PPO engine")
class NativeSessionTests(unittest.TestCase):
    def test_native_loop_does_not_call_python_physics_or_learning(self):
        reference = Experiment(configuration())
        native = NativeExperiment.from_experiment(reference)
        while not reference.done:
            reference.step()
        with patch.object(World, "step", side_effect=AssertionError("Python physics")):
            with patch.object(
                PPOBrain, "learn", side_effect=AssertionError("Python learning")
            ):
                report = native.run()
        self.assertTrue(report["finished_run"])
        self.assertEqual(encoded(reference.checkpoint()), encoded(native.checkpoint()))
        self.assertEqual(
            encoded(reference.checkpoint()), encoded(native.query().checkpoint())
        )
        query = native.query()
        next(iter(query.world.animals.values())).energy = -1
        self.assertEqual(encoded(reference.checkpoint()), encoded(native.checkpoint()))

    def test_native_phase_profiling_is_transient_and_does_not_change_results(self):
        reference = NativeExperiment(configuration(12))
        measured = NativeExperiment(configuration(12))
        measured_report = measured.run(6, options={"profile_native": True})
        reference.run(6)
        self.assertEqual(
            encoded(reference.checkpoint()), encoded(measured.checkpoint())
        )
        timings = measured_report["profile_native"]
        self.assertEqual(
            set(timings),
            {
                "choose",
                "physics",
                "observe",
                "learn",
                "champions",
                "lifecycle",
                "metrics",
                "capture",
            },
        )
        self.assertTrue(all(value >= 0 for value in timings.values()))
        self.assertNotIn("profile_native", measured.run())
        reference.run()
        self.assertEqual(
            encoded(reference.checkpoint()), encoded(measured.checkpoint())
        )

    def test_birth_mutation_restore_and_native_siblings_remain_independent(self):
        config = configuration(24)
        config["obstacles"].update(generation="explicit", explicit_cells=[])
        for species in config["species"].values():
            species["reproduction"].update(
                min_age_steps=1,
                cooldown_steps=3,
                health_threshold=0,
                min_energy=1,
                energy_cost=1,
                max_population=8,
            )
            species["mutation"].update(probability_per_birth=1, max_changed_weights=1)
        reference = Experiment(config)
        positions = [(5, 5), (6, 5), (20, 20), (21, 20)]
        for index, animal in enumerate(reference.world.animals.values()):
            animal.x, animal.y = positions[index]
            animal.sex = "female" if index % 2 == 0 else "male"
        reference.observations = reference.world.observe_all()
        force_reproduction(reference)
        native = NativeExperiment.from_experiment(reference)
        sibling = NativeExperiment.from_experiment(reference)
        untouched = encoded(sibling.checkpoint())
        for tick in range(24):
            reference.step()
            native.run(1)
            self.assertEqual(
                encoded(reference.checkpoint()),
                encoded(native.checkpoint()),
                f"tick {tick+1}",
            )
            if tick == 7:
                native = NativeExperiment.restore(native.checkpoint())
        self.assertGreater(sum(reference.world.births.values()), 0)
        self.assertGreater(sum(reference.world.mutations.values()), 0)
        self.assertEqual(untouched, encoded(sibling.checkpoint()))
        sibling.run()
        self.assertEqual(encoded(native.checkpoint()), encoded(sibling.checkpoint()))

    def test_cli_batches_preserve_all_selected_replays_and_metrics(self):
        config = configuration(24)
        config["experiment"]["runs"] = 2
        config["recording"].update(
            record_last_episodes=1,
            frames_per_chunk=8,
            diagnostics_every_steps=3,
            record_activations=True,
        )
        outputs = []
        with tempfile.TemporaryDirectory() as temporary:
            for backend in ("python", "cpp"):
                folder = Path(temporary) / backend
                folder.mkdir()
                (folder / "checkpoints").mkdir()
                experiment = Experiment(copy.deepcopy(config))
                recorder = Recorder(folder, experiment.config, experiment.mode)

                def save(name, data, destination=folder):
                    (destination / name).write_text(json.dumps(data), encoding="utf-8")

                run_experiment(
                    experiment,
                    folder,
                    save,
                    recorder,
                    save_brains=False,
                    checkpoint_format="binary",
                    engine=backend,
                )
                recorder.close()
                chunks = {
                    p.name: p.read_bytes()
                    for p in (folder / "playback").glob("*.bin.xz")
                }
                manifest = json.loads((folder / "playback/recording.json").read_text())
                self.assertEqual([r["run"] for r in manifest["runs"]], [2])
                outputs.append(
                    (
                        encoded(read_checkpoint(folder / "checkpoints/final.bin.xz")),
                        (folder / "metrics.csv").read_bytes(),
                        chunks,
                    )
                )
                self.assertFalse((folder / ".native-frames.jsonl").exists())
        self.assertEqual(outputs[0], outputs[1])
