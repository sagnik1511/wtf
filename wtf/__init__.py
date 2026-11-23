"""Wildlife Trial Framework: independent learning animals and offline replay.

Heavy simulation modules are loaded only when requested, so playback imports
remain independent of training.
"""

import sys

if sys.version_info < (3, 12):
    raise RuntimeError(
        "Wildlife Trial Framework requires Python 3.12+; recreate .venv with python3.12"
    )

# Experiment is resolved by __getattr__; eager binding would import training here.
# pylint: disable-next=undefined-all-variable
__all__ = ["Experiment", "load_config"]


def load_config(name_or_path="baseline"):
    """Load a bundled preset name or a local environment JSON and validate it."""

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    from wtf.config import load

    return load(name_or_path)


def __getattr__(name):

    # Keep imports lazy so optional tools and training dependencies load only on this
    # path.
    # pylint: disable=import-outside-toplevel

    if name == "Experiment":
        from wtf.experiment.runner import Experiment

        return Experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
