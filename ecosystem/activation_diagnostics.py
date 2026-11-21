"""Compact, detached activation samples. No weights, gradients or training state."""

from array import array


def activation_sample(info, layers, prior=None):
    """Attach storage-only FP32 hidden values; inputs and outputs already exist in info."""
    info["activations"] = dict(
        version=1, layers=[list(array("f", values)) for values in layers]
    )
    if prior is not None:
        info["activations"]["recurrent_prior"] = list(array("f", prior))
    return info
