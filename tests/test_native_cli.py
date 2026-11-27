"""Native execution boundaries preserve Python queries and artifact cadence."""

# Fixtures inspect private boundaries without changing the public runtime.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from wtf.experiment import native_cli as helper


def experiment(tick=0, mode="train"):
    return SimpleNamespace(
        world=SimpleNamespace(tick=tick),
        config={"world": {"max_steps": 1500}},
        mode=mode,
    )


class NativeCliTests(unittest.TestCase):
    def test_budget_boundaries(self):
        cases = [
            (0, {}, True, "train", 250),
            (249, {}, True, "train", 1),
            (499, {"brain_save_every_steps": 0}, False, "train", 1),
            (
                123,
                {"checkpoint_every_steps": 0, "brain_save_every_steps": 0},
                True,
                "train",
                250,
            ),
            (1499, {}, False, "train", 1),
            (
                99,
                {"checkpoint_every_steps": 128, "brain_save_every_steps": 100},
                True,
                "train",
                1,
            ),
            (
                99,
                {"checkpoint_every_steps": 128, "brain_save_every_steps": 100},
                True,
                "evaluate",
                29,
            ),
        ]
        for tick, options, save, mode, expected in cases:
            with self.subTest(tick=tick, options=options, mode=mode):
                self.assertEqual(
                    helper._step_budget(experiment(tick, mode), options, save), expected
                )

    def test_frames_projection_and_newmetrics(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "batch.jsonl"
            native = Mock()

            def run(*_args, **kwargs):
                Path(kwargs["capture_path"]).write_text(
                    "\n".join(json.dumps({"tick": i}) for i in [11, 12]) + "\n",
                    encoding="utf-8",
                )
                return {"last_record": {"tick": 12}}

            native.run.side_effect = run
            native.to_python.return_value = SimpleNamespace(
                history=[{"tick": 10}, {"tick": 11}, {"tick": 12}]
            )
            recorder = SimpleNamespace(
                run={}, options={"enabled": True}, capture_frame=Mock()
            )
            projection, returned, metrics = helper.advance(
                experiment(10), native, recorder, {}, True, path
            )
            native.run.assert_called_once_with(
                240, capture_path=path, options=recorder.options
            )
            native.to_python.assert_called_once()
            self.assertIs(native, returned)
            self.assertEqual(metrics, [{"tick": 11}, {"tick": 12}])
            self.assertEqual(projection.last_record, {"tick": 12})
            self.assertEqual(
                [call.args[0] for call in recorder.capture_frame.call_args_list],
                [{"tick": 11}, {"tick": 12}],
            )
            self.assertFalse(path.exists())

    def test_stale_spool_is_preserved_and_batch_not_run(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "batch.jsonl"
            path.write_text("prior data")
            native = Mock()
            with self.assertRaises(FileExistsError):
                helper.advance(experiment(), native, None, {}, False, path)
            native.run.assert_not_called()
            self.assertEqual(path.read_text(), "prior data")

    def test_failure_exposes_native_and_cleans_spool(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "batch.jsonl"
            native = Mock()

            def run(*_args, **_kwargs):
                path.write_text("partial")
                raise ValueError("native failure")

            native.run.side_effect = run
            with self.assertRaises(helper.NativeBatchError) as caught:
                helper.advance(experiment(), native, None, {}, False, path)
            self.assertIs(caught.exception.native, native)
            self.assertIsInstance(caught.exception.__cause__, ValueError)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
