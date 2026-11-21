"""Wildlife Trial Framework: independent learning animals and offline replay.

Heavy simulation modules are loaded only when requested, so playback imports
remain independent of training.
"""

__all__ = ["Experiment", "load_config"]


def load_config(name_or_path="baseline"):
    """Load a bundled preset name or a local environment JSON and validate it."""
    from .config import load

    return load(name_or_path)


def __getattr__(name):
    if name == "Experiment":
        from .runner import Experiment

        return Experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
