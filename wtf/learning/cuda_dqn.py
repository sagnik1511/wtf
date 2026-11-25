"""Publish coherent CPU snapshots after device-resident DQN/DDQN updates."""

from array import array
import math
import time


def update(engine, handle, jobs, buffers, discount, threshold, double_q, started):
    """Keep gradients on GPU; publish private FP32 parameters and Adam state in bulk."""
    # Packing, validation and publication are one atomic host-visible update phase.
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    moments = engine.pack_models(handle, "dqn_moments", [b.moments for b, _, _ in jobs])
    variances = engine.pack_models(
        handle, "dqn_variances", [b.variances for b, _, _ in jobs]
    )
    metadata = array("i")
    for brain, _, _ in jobs:
        step = brain.updates + 1
        if not 1 <= step <= 2147483647:
            raise ValueError("CUDA Adam step must fit a positive int32")
        metadata.extend(
            (step, int(step % brain.config["target_update_every_training_steps"] == 0))
        )
    config = jobs[0][0].config
    options = array(
        "d",
        (
            discount,
            threshold,
            config["max_grad_norm"],
            config["learning_rate"],
            double_q,
        ),
    )
    loss = engine.buffer(handle, "dqn_losses", engine.typecode, len(jobs))
    if engine.profile:
        engine.packing_seconds += time.perf_counter() - started
    engine.library.dqn_update(
        handle, buffers, moments, variances, metadata, options, loss
    )
    if not getattr(engine.library, "validates_output", False) and (
        any(
            not math.isfinite(v)
            for values in (buffers[0], moments, variances, loss)
            for v in values
        )
        or any(v < 0 for v in variances)
    ):
        raise RuntimeError("Non-finite CUDA DQN model/optimizer state or loss")
    started = time.perf_counter() if engine.profile else 0.0
    offset = 0
    for index, (brain, samples, _) in enumerate(jobs):
        for layer, m, v in zip(brain.network.layers, brain.moments, brain.variances):
            for key in ("weights", "bias"):
                count = len(layer[key])
                layer[key][:] = buffers[0][offset : offset + count]
                m[key][:] = moments[offset : offset + count]
                v[key][:] = variances[offset : offset + count]
                offset += count
        brain.updates = metadata[index * 2]
        if metadata[index * 2 + 1]:
            for target, online in zip(
                brain.target_network.layers, brain.network.layers
            ):
                for key in ("weights", "bias"):
                    target[key][:] = online[key]
        brain.last_metrics = {"td_loss": loss[index] / len(samples)}
    if engine.profile:
        engine.optimizer_seconds += time.perf_counter() - started
    engine.batches += 1
    engine.models += len(jobs)
