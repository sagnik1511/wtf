"""Archive-local reuse without stale configurations or changed saved values."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wtf.storage.library import export_champions, read_artifact
from wtf.experiment.runner import Experiment
from tests.test_simulation import config


class ExportEfficiencyTests(unittest.TestCase):
    def test_configurations_serialize_once_per_object_and_refresh_between_exports(self):
        experiment = Experiment(config())
        entries = [
            entry
            for groups in experiment.champions.entries.values()
            for champions in groups.values()
            for entry in champions.values()
        ]
        self.assertGreater(len(entries), 3)
        historical = copy.deepcopy(experiment.config)
        historical["seed"] += 17
        for entry in entries[:2]:
            entry["training_config"] = historical
        dumps = json.dumps
        counts = {}

        def counted(value, *args, **kwargs):
            if value is historical or value is experiment.config:
                counts[id(value)] = counts.get(id(value), 0) + 1
            return dumps(value, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            for run in (1, 2):
                with patch("wtf.storage.library.json.dumps", side_effect=counted):
                    archive = export_champions(
                        experiment, Path(folder) / f"champions/run_{run:04d}.json"
                    )
                self.assertEqual(
                    counts, {id(historical): run, id(experiment.config): run}
                )
                expected_bytes = {
                    dumps(c, separators=(",", ":"), allow_nan=False) + "\n"
                    for c in (historical, experiment.config)
                }
                self.assertEqual(
                    {p.read_text() for p in (archive / "configs").iterdir()},
                    expected_bytes,
                )
                for species, groups in experiment.champions.entries.items():
                    for sex, champions in groups.items():
                        for criterion, entry in champions.items():
                            artifact = read_artifact(
                                archive / species / sex / (criterion + ".json")
                            )
                            self.assertEqual(artifact["model"], entry["model"])
                            self.assertEqual(
                                artifact["training_config"],
                                entry.get("training_config", experiment.config),
                            )
                # No identity-based cache may survive this export call.
                experiment.config["seed"] += 1
