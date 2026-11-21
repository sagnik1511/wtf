import json
from pathlib import Path
import tempfile
import unittest

from wtf.storage.checkpoint_io import read_checkpoint
from wtf.storage.recording import RecordingStore
from wtf.storage.replay_io import chunk_path, load_chunk
from wtf.experiment.runner import Experiment
from wtf.server import argument_parser
from examples.complete_experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]


class DistributionExamplesTests(unittest.TestCase):
    def test_complete_example_records_exports_and_restores(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = run_experiment(output=Path(temporary), steps=5, seed=100)
            for name in (
                "environment.json",
                "summary.json",
                "metrics.json",
                "analysis.json",
                "champions/run_0001.json",
            ):
                self.assertTrue((folder / name).is_file(), name)
            restored = Experiment.restore(
                read_checkpoint(folder / "checkpoints/final.bin.xz")
            )
            self.assertTrue(restored.done)
            self.assertEqual(restored.world.tick, 5)
            metadata = RecordingStore(temporary).metadata(folder.name)
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["runs"][0]["last_tick"], 5)
            frames = load_chunk(chunk_path(folder / "playback", 1, 0))
            self.assertEqual([frame["tick"] for frame in frames], list(range(6)))
            self.assertEqual(frames[-1]["metrics"], restored.world.metrics())

    def test_tutorial_contains_the_complete_runnable_example(self):
        source = (ROOT / "examples/complete_experiment.py").read_text().strip()
        tutorial = (ROOT / "docs/tutorials/02-run-record-analyze.md").read_text()
        excerpt = tutorial.split("<!-- complete-example:start -->")[1].split(
            "<!-- complete-example:end -->"
        )[0]
        code = excerpt.split("```python\n", 1)[1].rsplit("```", 1)[0].strip()
        self.assertEqual(code, source)
        compile(code, "<tutorial-example>", "exec")

    def test_server_binding_is_opt_in(self):
        self.assertEqual(argument_parser().parse_args([]).host, "127.0.0.1")
        args = argument_parser().parse_args(
            ["--host", "0.0.0.0", "--results", "/data/results"]
        )
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.results, Path("/data/results"))
