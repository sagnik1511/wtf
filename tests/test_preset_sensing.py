"""Maintained presets opt into compact sensing without changing their learners."""

# Test fixtures inspect stored dimensions and restoration ownership deliberately.
# pylint: disable=missing-function-docstring,protected-access
import json
from pathlib import Path
import unittest

from wtf.config import load, validate
from wtf.experiment.runner import Experiment
from wtf.learning.brain import observation_inputs
from wtf.learning.brains import load_model_weights, model_snapshot

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {
    "baseline",
    "champion_lineage",
    "seaside_150_v5",
    "seaside_150_v6",
    "seaside_150_v7",
}


def promoted_presets():
    """Only shipped presets participate; ignored local experiments stay untouched."""
    return [
        ROOT / name
        for name in json.loads((ROOT / "packaging_assets.json").read_text())
        if name.startswith("environments/") and Path(name).stem not in CONTROLS
    ]


class PresetSensingTests(unittest.TestCase):
    """Exercise real policy dispatch, terrain, sensing and saved-state boundaries."""

    def test_promoted_presets_have_explicit_compatible_sensory_contracts(self):
        self.assertTrue(promoted_presets())
        for path in promoted_presets():
            with self.subTest(preset=path.stem):
                c = load(path)
                self.assertTrue(c["scent_trails"]["enabled"])
                for spec in c["species"].values():
                    template = c["brains"][spec["brain"]]
                    self.assertEqual(template["observation_version"], 7)
                    self.assertEqual(
                        template["observation_actions"], spec["allowed_actions"]
                    )
                    self.assertEqual(template["movement_history_steps"], 3)
                    self.assertEqual(
                        template["scent_observation"], {"mode": "blended", "radius": 2}
                    )
                # Representation defaults must not silently enable a different objective.
                for template in c["brains"].values():
                    self.assertNotIn("late_consequence", template)

    def test_each_promoted_preset_predicts_and_restores_without_replay_changes(self):
        for path in promoted_presets():
            with self.subTest(preset=path.stem):
                c = load(path)
                c["world"]["max_steps"] = 2
                c["experiment"]["runs"] = 1
                for spec in c["species"].values():
                    spec["initial_population"] = 2
                experiment = Experiment(validate(c), mode="evaluate")
                experiment.step()
                restored = Experiment.restore(experiment.checkpoint(), mode="evaluate")
                for animal_id, observation in experiment.observations.items():
                    animal = experiment.world.animals[animal_id]
                    template = c["brains"][c["species"][animal.species]["brain"]]
                    self.assertEqual(len(observation), observation_inputs(template))
                    diagnostic = experiment.brains[animal_id].diagnostics(observation)
                    self.assertEqual(tuple(diagnostic["features"]), observation)
                experiment.step()
                restored.step()
                self.assertEqual(experiment.checkpoint(), restored.checkpoint())

    def test_explicit_controls_keep_their_observation_versions(self):
        for preset, expected in (
            ("baseline", 1),
            ("champion_lineage", 1),
            ("seaside_150_v5", 5),
            ("seaside_150_v6", 6),
            ("seaside_150_v7", 7),
        ):
            with self.subTest(preset=preset):
                c = load(preset)
                self.assertTrue(
                    all(
                        b.get("observation_version", 1) == expected
                        for b in c["brains"].values()
                    )
                )

    def test_legacy_brains_are_rejected_but_exact_old_checkpoints_restore(self):
        legacy = Experiment(load("baseline"), mode="evaluate")
        upgraded = Experiment(load("dqn"), mode="evaluate")
        old = next(
            b
            for i, b in legacy.brains.items()
            if legacy.world.animals[i].species == "prey"
        )
        new = next(
            b
            for i, b in upgraded.brains.items()
            if upgraded.world.animals[i].species == "prey"
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_model_weights(new, model_snapshot(old))
        state = legacy.checkpoint()
        restored = Experiment.restore(state, mode="evaluate")
        self.assertEqual(restored.checkpoint(), state)
        self.assertTrue(all(len(o) == 20 for o in restored.observations.values()))
