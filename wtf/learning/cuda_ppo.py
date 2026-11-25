"""Host schedules for independent PPO updates with device-resident minibatch state."""

from array import array
import math
import time

METRICS = ("policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction")
OPTIONS = (
    "clip_ratio",
    "entropy_coefficient",
    "value_coefficient",
    "max_grad_norm",
    "learning_rate",
    "target_kl",
)


def groups(jobs):
    """Batch equal layouts/objectives; preserve each animal's rollout and RNG."""
    result = {}
    for brain, advantages, targets in jobs:
        key = (
            tuple(brain.network.sizes),
            brain.config["batch_size"],
            tuple(
                brain.config.get(key, 0.03 if key == "target_kl" else None)
                for key in OPTIONS
            ),
        )
        result.setdefault(key, []).append((brain, advantages, targets))
    return result


def train(engine, jobs, sizes, batch, options):
    """Keep GPU parameters/Adam state across rounds, publishing once on success."""
    # Explicit state mirrors the original epoch/minibatch/early-stop boundaries.
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,protected-access
    started = time.perf_counter() if engine.profile else 0.0
    handle = engine._workspace(sizes, len(jobs), batch, 2)
    weights = engine.pack_models(
        handle, "ppo_weights", [b.network.layers for b, _, _ in jobs]
    )
    moments = engine.pack_models(handle, "ppo_moments", [b.moments for b, _, _ in jobs])
    variances = engine.pack_models(
        handle, "ppo_variances", [b.variances for b, _, _ in jobs]
    )
    if engine.profile:
        engine.packing_seconds += time.perf_counter() - started
    engine.library.ppo_begin(handle, weights, moments, variances)
    states = []
    for brain, advantages, targets in jobs:
        states.append(
            {
                "brain": brain,
                "advantages": advantages,
                "targets": targets,
                "indices": list(range(len(brain.rollout))),
                "epoch": 0,
                "offset": 0,
                "updates": brain.updates,
                "samples": 0,
                "totals": [0.0] * 5,
                "stopped": False,
            }
        )
    samples = len(jobs) * batch
    obs = engine.buffer(handle, "ppo_obs", engine.typecode, samples * sizes[0])
    data = engine.buffer(handle, "ppo_data", engine.typecode, samples * 3)
    actions = engine.buffer(handle, "ppo_actions", "i", samples)
    valid = engine.buffer(handle, "ppo_valid", "i", samples)
    masks = engine.buffer(handle, "ppo_masks", "B", samples * (sizes[-1] - 1))
    metadata = engine.buffer(handle, "ppo_metadata", "i", len(jobs) * 2)
    summary = engine.buffer(handle, "ppo_summary", engine.typecode, len(jobs) * 6)
    flags = array("d", options)
    while True:
        started = time.perf_counter() if engine.profile else 0.0
        active = []
        valid[:] = array("i", [0]) * samples
        for model, state in enumerate(states):
            brain = state["brain"]
            if state["stopped"] or state["epoch"] >= brain.config["epochs"]:
                metadata[model * 2] = 0
                continue
            if state["offset"] == 0:
                brain.training_rng.shuffle(state["indices"])
            selected = state["indices"][state["offset"] : state["offset"] + batch]
            active.append((model, selected))
            metadata[model * 2 : model * 2 + 2] = array(
                "i", (len(selected), state["updates"] + 1)
            )
            for row, index in enumerate(selected):
                sample = model * batch + row
                item = brain.rollout[index]
                if (
                    len(item.observation) < sizes[0]
                    or not 0 <= item.action < sizes[-1] - 1
                ):
                    raise ValueError(
                        "Invalid CUDA PPO observation width or action index"
                    )
                obs[sample * sizes[0] : (sample + 1) * sizes[0]] = array(
                    engine.typecode, item.observation[: sizes[0]]
                )
                data[sample * 3 : sample * 3 + 3] = array(
                    engine.typecode,
                    (
                        item.log_probability,
                        state["advantages"][index],
                        state["targets"][index],
                    ),
                )
                actions[sample] = item.action
                valid[sample] = 1
                mask = item.observation[sizes[0] :]
                if not mask:
                    mask = (1,) * (sizes[-1] - 1)
                if len(mask) != sizes[-1] - 1 or not any(mask) or not mask[item.action]:
                    raise ValueError("Invalid CUDA PPO action mask")
                masks[sample * (sizes[-1] - 1) : (sample + 1) * (sizes[-1] - 1)] = (
                    array("B", map(bool, mask))
                )
        if not active:
            break
        if any(not math.isfinite(v) for values in (obs, data) for v in values):
            raise ValueError("Non-finite CUDA PPO sample")
        if engine.profile:
            engine.packing_seconds += time.perf_counter() - started
        engine.library.ppo(
            handle, (obs, data, actions, valid, masks, metadata), flags, summary
        )
        if not all(math.isfinite(v) for v in summary):
            raise RuntimeError("Non-finite CUDA PPO metrics")
        for model, selected in active:
            state = states[model]
            if not summary[model * 6 + 5]:
                state["stopped"] = True
                continue
            state["updates"] += 1
            state["samples"] += len(selected)
            for metric in range(5):
                state["totals"][metric] += summary[model * 6 + metric]
            state["offset"] += len(selected)
            if state["offset"] >= len(state["indices"]):
                state["epoch"] += 1
                state["offset"] = 0
        engine.ppo_batches += 1
    engine.library.ppo_end(handle, weights, moments, variances)
    if any(
        not math.isfinite(v) for values in (weights, moments, variances) for v in values
    ):
        raise RuntimeError("Non-finite CUDA PPO model/optimizer state")
    offset = 0
    for state in states:
        brain = state["brain"]
        for layer, m, v in zip(brain.network.layers, brain.moments, brain.variances):
            for key in ("weights", "bias"):
                count = len(layer[key])
                layer[key][:] = weights[offset : offset + count]
                m[key][:] = moments[offset : offset + count]
                v[key][:] = variances[offset : offset + count]
                offset += count
        brain.updates = state["updates"]
        brain.rollout_updates += 1
        brain.last_metrics = (
            {
                key: value / state["samples"]
                for key, value in zip(METRICS, state["totals"])
            }
            if state["samples"]
            else {}
        )
        brain.last_metrics["early_stopped"] = state["stopped"]
        brain.rollout.clear()
    engine.ppo_models += len(jobs)
