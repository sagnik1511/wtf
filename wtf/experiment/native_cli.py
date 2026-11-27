"""Bounded native CLI batches with ordinary Python artifact handling.

The native session remains authoritative between batches. Restored projections
serve existing progress, retention and export code; they never step the world.
"""

import json
from pathlib import Path

from wtf.experiment.native import NativeExperiment


class NativeBatchError(RuntimeError):
    """Retain the authoritative session when a batch or artifact transfer fails."""

    def __init__(self, native, error):
        super().__init__(f"Native batch failed: {error}")
        self.native = native


def _step_budget(experiment, storage_options, save_brains):
    """End a batch at the next requested checkpoint or champion-save boundary."""
    tick = experiment.world.tick
    remaining = experiment.config["world"]["max_steps"] - tick
    limits = [250, remaining]
    intervals = [storage_options.get("checkpoint_every_steps", 500)]
    if save_brains and experiment.mode == "train":
        intervals.append(storage_options.get("brain_save_every_steps", 250))
    for interval in intervals:
        if interval:
            limits.append(interval - tick % interval)
    return max(1, min(limits))


# The CLI boundary supplies the existing artifact owners and its private spool.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def advance(experiment, native, recorder, storage_options, save_brains, spool_path):
    """Execute one C++ batch and return its Python projection and new metrics.

    ``spool_path`` must be a fresh path in the caller's private temporary folder.
    Prepared native frames are streamed into the recorder and then removed. On
    failure, ``NativeBatchError.native`` permits checkpointing the live session;
    partially transferred recorder output must not be retried as a fresh batch.
    """
    old_tick = experiment.world.tick
    if native is None:
        native = NativeExperiment.from_experiment(experiment)
    recording = recorder is not None and recorder.run is not None
    path = Path(spool_path)
    if path.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing native frame spool: {path}"
        )
    try:
        report = native.run(
            _step_budget(experiment, storage_options, save_brains),
            capture_path=path if recording else None,
            options=recorder.options if recording else None,
        )
        if recording and path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        recorder.capture_frame(json.loads(line))
        projection = native.to_python()
        projection.last_record = report.get("last_record")
        metrics = [row for row in projection.history if row["tick"] > old_tick]
        return projection, native, metrics
    except Exception as error:
        raise NativeBatchError(native, error) from error
    finally:
        path.unlink(missing_ok=True)
