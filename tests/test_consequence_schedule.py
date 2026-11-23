"""Publication timing changes knowledge visibility, not individual learner ownership."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import json
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest

from tests.test_social_learning import config
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.learning.consequences import Consequences
from wtf.storage.checkpoint_io import read_checkpoint

KEY = (6, 3, 3, 0, 0, 0, 0, 1, 0, 0, 1, 0)


def episode_config():
    c = config(horizon=1)
    c["consequence_learning"]["update_schedule"] = "episode"
    c["world"]["max_steps"] = 3
    return validate(c)


class ConsequenceScheduleTests(unittest.TestCase):
    def test_continuous_default_retains_legacy_checkpoint_and_update_behavior(self):
        c = config()
        m = Consequences(c)
        m.learn("prey", KEY, [1, 0, 0])
        self.assertEqual(m.predict("prey", KEY), [0.2, 1, 0, 0])
        state = m.checkpoint()
        self.assertEqual(state["version"], 1)
        self.assertNotIn("staged_tables", state)
        self.assertEqual(Consequences.restore(c, state).checkpoint(), state)
        explicit = copy.deepcopy(c)
        explicit["consequence_learning"]["update_schedule"] = "continuous"
        self.assertEqual(Consequences.restore(explicit, state).checkpoint(), state)

    def test_episode_stage_never_mutates_published_rows_until_publication(self):
        m = Consequences(episode_config())
        m.learn("prey", KEY, [1, 0, 0])
        self.assertEqual(m.predict("prey", KEY), [0.0] * 4)
        self.assertEqual(m.published_updates, 0)
        m.publish()
        self.assertEqual(m.predict("prey", KEY), [0.2, 1, 0, 0])
        published = copy.deepcopy(m.tables)
        m.learn("prey", KEY, [0, 1, 0])
        self.assertEqual(m.tables, published)
        self.assertIsNot(m.tables["prey"][KEY], m.staged_tables["prey"][KEY])
        self.assertEqual(m.predict("prey", KEY), [0.2, 1, 0, 0])
        self.assertEqual(m.updates, 2)
        self.assertEqual(m.published_updates, 1)
        m.publish()
        self.assertEqual(m.predict("prey", KEY), [2 / 6, 0.5, 0.5, 0])
        self.assertFalse(m.staged_tables)

    def test_publication_happens_before_next_episode_first_observation(self):
        e = Experiment(episode_config())
        model = e.world._social.knowledge
        for _ in range(3):
            e.step()
            self.assertTrue(all(not rows for rows in model.tables.values()))
            self.assertTrue(
                all(all(v == 0 for v in obs[-48:]) for obs in e.observations.values())
            )
        collected = model.updates
        self.assertGreater(collected, 0)
        staged = copy.deepcopy(model.staged_tables)
        self.assertTrue(e.finished_run)
        self.assertTrue(e.next_run())
        self.assertIs(e.world._social.knowledge, model)
        self.assertEqual(model.published_updates, collected)
        for name, rows in staged.items():
            self.assertEqual(model.tables[name], rows)
        self.assertFalse(model.staged_tables)
        self.assertEqual(e.world.tick, 0)
        self.assertEqual(e.observations, e.world.observe_all())
        self.assertFalse(e.world._social.sightings)

    def test_mid_episode_checkpoint_resume_keeps_both_views_exact(self):
        e = Experiment(episode_config())
        e.step()
        f = Experiment.restore(json.loads(json.dumps(e.checkpoint())))
        self.assertEqual(e.checkpoint(), f.checkpoint())
        for _ in range(4):
            e.step()
            f.step()
            self.assertEqual(e.checkpoint(), f.checkpoint())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.bin.xz"
            e.write_checkpoint(path, shared_metadata=True)
            self.assertEqual(
                Experiment.restore(read_checkpoint(path)).checkpoint(), e.checkpoint()
            )

    def test_nontraining_episode_boundaries_preserve_published_and_staged_knowledge(
        self,
    ):
        c = episode_config()
        c["experiment"]["runs"] = 4
        e = Experiment(c)
        while not e.finished_run:
            e.step()
        self.assertTrue(e.next_run())
        e.step()
        state = e.checkpoint()
        model = e.world._social.knowledge
        self.assertGreater(model.published_updates, 0)
        self.assertGreater(model.updates, model.published_updates)
        expected = copy.deepcopy(model.checkpoint())
        for mode in ("evaluate", "random"):
            with self.subTest(mode=mode):
                frozen = Experiment.restore(copy.deepcopy(state), mode=mode)
                while not frozen.finished_run:
                    frozen.step()
                self.assertTrue(frozen.next_run())
                self.assertEqual(frozen.world._social.knowledge.checkpoint(), expected)
                self.assertEqual(frozen.observations, frozen.world.observe_all())

                resumed = Experiment.restore(frozen.checkpoint(), mode="train")
                self.assertEqual(resumed.world._social.knowledge.checkpoint(), expected)
                while not resumed.finished_run:
                    resumed.step()
                staged = copy.deepcopy(resumed.world._social.knowledge.staged_tables)
                self.assertTrue(resumed.next_run())
                published = resumed.world._social.knowledge
                self.assertFalse(published.staged_tables)
                self.assertEqual(published.published_updates, published.updates)
                for species, table in staged.items():
                    self.assertEqual(published.tables[species], table)

    def test_final_export_includes_staged_evidence_without_rewriting_replay_view(self):
        c = episode_config()
        m = Consequences(c)
        m.learn("prey", KEY, [1, 0, 0])
        before = copy.deepcopy(m.checkpoint())
        self.assertEqual(before["version"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consequences.json.xz"
            m.save(path)
            imported = Consequences.load(c, path)
            self.assertEqual(imported.predict("prey", KEY), [0.2, 1, 0, 0])
            self.assertEqual(imported.published_updates, 1)
            self.assertFalse(imported.staged_tables)
            continuous = config(horizon=1)
            self.assertEqual(
                Consequences.load(continuous, path).predict("prey", KEY), [0.2, 1, 0, 0]
            )
        self.assertEqual(m.checkpoint(), before)
        self.assertEqual(m.predict("prey", KEY), [0.0] * 4)

    def test_episode_state_validation_and_stage_capacity(self):
        c = episode_config()
        m = Consequences(c)
        for i in range(80):
            m.learn("prey", (i % 12, (i // 12) % 4, (i // 48) % 4, *KEY[3:]), [1, 0, 0])
        self.assertEqual(len(m.staged_tables["prey"]), 16)
        self.assertFalse(m.tables["prey"])
        state = m.checkpoint()
        clone = Consequences.restore(c, json.loads(json.dumps(state)))
        self.assertEqual(clone.checkpoint(), state)
        for invalid in ("immediate", "end", False, None):
            other = copy.deepcopy(c)
            other["consequence_learning"]["update_schedule"] = invalid
            with self.assertRaises(ValueError):
                validate(other)
        with self.assertRaises(ValueError):
            Consequences.restore(config(), state)
        bad = copy.deepcopy(state)
        bad["published_updates"] = m.updates + 1
        with self.assertRaises(ValueError):
            Consequences.restore(c, bad)

    def test_cli_override_is_saved_and_exact_resume_cannot_change_it(self):

        # This self-contained regression intentionally repeats reference/setup data.
        # pylint: disable=duplicate-code

        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "config.json"
            path.write_text(json.dumps(c))
            command = [
                sys.executable,
                "-m",
                "wtf.cli",
                str(path),
                "--runs",
                "2",
                "--steps",
                "3",
                "--no-progress",
                "--no-library",
                "--no-record",
                "--output",
                str(root / "results"),
                "--consequence-update-schedule",
                "episode",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            folder = next((root / "results").iterdir())
            saved = json.loads((folder / "environment.json").read_text())
            self.assertEqual(
                saved["consequence_learning"]["update_schedule"], "episode"
            )
            model = Consequences.load(saved, folder)
            self.assertGreater(model.updates, 0)
            checkpoint = next((folder / "checkpoints").glob("final*"))
            rejected = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wtf.cli",
                    "--resume",
                    str(checkpoint),
                    "--consequence-update-schedule",
                    "continuous",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("cannot override exact resume", rejected.stderr)
