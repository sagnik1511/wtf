import io
import unittest
from wtf.progress import RunProgress
from wtf.experiment.runner import Experiment
from tests.test_simulation import config


class ProgressTests(unittest.TestCase):
    def test_progress_has_no_effect_on_results(self):
        a, b = Experiment(config()), Experiment(config())
        stream = io.StringIO()
        progress = RunProgress(enabled=True, stream=stream)
        progress.start(a)
        while not a.done:
            if a.finished_run:
                a.next_run()
                b.next_run()
                progress.start(a)
            a.step()
            b.step()
            progress.tick(a)
        progress.close()
        self.assertEqual(a.checkpoint(), b.checkpoint())
        output = stream.getvalue()
        self.assertIn("Train run 1/2", output)
        self.assertIn("Train run 2/2", output)
        self.assertIn("20/20", output)
        self.assertIn("kills=", output)

    def test_resume_progress_starts_at_saved_step(self):
        e = Experiment(config())
        e.step()
        e.step()
        progress = RunProgress(enabled=True, stream=io.StringIO())
        progress.start(e)
        self.assertEqual(progress.bar.n, 2)
        e.step()
        progress.tick(e)
        self.assertEqual(progress.bar.n, 3)
        progress.close()

    def test_default_non_terminal_output_is_quiet(self):
        progress = RunProgress(stream=io.StringIO())
        progress.start(Experiment(config()))
        self.assertFalse(progress.enabled)
        self.assertIsNone(progress.bar)

    def test_extinction_finishes_bar_at_actual_step(self):
        e = Experiment(config())
        for animal in e.world.animals.values():
            animal.energy = 0
            animal.health = 0.01
        e.observations = e.world.observe_all()
        stream = io.StringIO()
        progress = RunProgress(enabled=True, stream=stream)
        progress.start(e)
        e.step()
        progress.tick(e)
        self.assertIsNone(progress.bar)
        self.assertIn("extinct", stream.getvalue())
        self.assertIn("1/1", stream.getvalue())


class ProgressCLITests(unittest.TestCase):
    def test_cli_progress_can_be_forced_or_disabled(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "work") as folder:
            for flag, visible in (("--progress", True), ("--no-progress", False)):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "wtf.cli",
                        "--no-library",
                        "environments/baseline.json",
                        "--runs",
                        "1",
                        "--steps",
                        "2",
                        "--seed",
                        "7",
                        "--no-record",
                        "--output",
                        folder,
                        flag,
                    ],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual("Train run 1/1" in result.stderr, visible)
                self.assertIn("Results:", result.stdout)
