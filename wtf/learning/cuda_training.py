"""Opt-in CUDA dense prediction and DQN/DDQN gradient batches.

CPU snapshots remain authoritative; PPO minibatch state is transiently GPU-resident.

Device buffers are transient, bounded, and absent from checkpoints. No third-party
Python dependency or compiler execution occurs during imports.
"""

from array import array
from collections import OrderedDict
from contextlib import contextmanager
import ctypes
import math
import os
from pathlib import Path
import sys
import time

from wtf.learning.dqn import DQNBrain, DDQNBrain
from wtf.learning.qrdqn import QRDQNBrain
from wtf.learning.ppo import PPOBrain, PPO_ACCELERATOR
from wtf.learning.policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain
from wtf.learning.contextual_ppo import (
    MaskablePPOBrain,
    MAPPOBrain,
    LocalMLP,
    CentralNetwork,
)
from wtf.learning.policy_networks import PolicyBundle
from wtf.learning.discrete_sac import DiscreteSACBrain
from wtf.learning.trpo import TRPOBrain
from wtf.learning.ppo import MLP
from wtf.native.numeric import Float32


class CudaLibrary:
    """Small, versioned C ABI around a separately NVCC-compiled shared library."""

    validates_output = True

    def __init__(self):
        if sys.platform != "linux":
            raise RuntimeError(
                "CUDA training requires Linux/NVIDIA. "
                "Build and test on Colab; use --device cpu here."
            )
        path = (
            Path(
                os.environ.get(
                    "WTF_CUDA_LIBRARY",
                    Path(__file__).parents[1] / "native/_cuda_dqn.so",
                )
            )
            .expanduser()
            .resolve()
        )
        try:
            self.library = ctypes.CDLL(str(path))
        except OSError as error:
            raise RuntimeError(
                "CUDA kernels unavailable. Run python -m wtf.native.cuda_build "
                "on Linux/NVIDIA; or set WTF_CUDA_LIBRARY."
            ) from error
        try:
            self._bind()
        except AttributeError as error:
            raise RuntimeError(
                "Incomplete CUDA kernel ABI; rebuild the library."
            ) from error

    def _bind(self):
        """Bind the complete ABI before probing or allocating device memory."""
        lib = self.library
        lib.wtf_cuda_api_version.argtypes = []
        lib.wtf_cuda_api_version.restype = ctypes.c_int
        if lib.wtf_cuda_api_version() != 2:
            raise RuntimeError("Unsupported CUDA kernel API; rebuild the library.")
        lib.wtf_cuda_error.argtypes = []
        lib.wtf_cuda_error.restype = ctypes.c_char_p
        lib.wtf_cuda_probe.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.wtf_cuda_probe.restype = ctypes.c_int
        lib.wtf_cuda_create.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 6
        lib.wtf_cuda_create.restype = ctypes.c_void_p
        lib.wtf_cuda_destroy.argtypes = [ctypes.c_void_p]
        lib.wtf_cuda_destroy.restype = None
        lib.wtf_cuda_run.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.wtf_cuda_run.restype = ctypes.c_int
        lib.wtf_cuda_predict.argtypes = [ctypes.c_void_p] * 4
        lib.wtf_cuda_predict.restype = ctypes.c_int
        for name, count in (("stats", 2), ("ppo_begin", 4), ("ppo", 9), ("ppo_end", 4)):
            function = getattr(lib, "wtf_cuda_" + name)
            function.argtypes = [ctypes.c_void_p] * count
            function.restype = ctypes.c_int
        name = ctypes.create_string_buffer(256)
        self.check(lib.wtf_cuda_probe(name, len(name)))
        self.device = name.value.decode("utf-8", "replace")

    def check(self, result):
        """Surface errors instead of silently substituting another backend."""
        if result:
            detail = self.library.wtf_cuda_error().decode("utf-8", "replace")
            raise RuntimeError(f"CUDA training failed: {detail}")

    # Architecture counts and opt-in execution controls mirror the native ABI.
    # pylint: disable-next=too-many-arguments
    def create(
        self, sizes, models, batch, training=True, *, precision="fp64", profile=False
    ):
        """Allocate one architecture/count workspace after Python size validation."""
        layout = array("i", sizes)
        if layout.itemsize != 4:
            raise RuntimeError("CUDA ABI requires 32-bit C int")
        handle = self.library.wtf_cuda_create(
            layout.buffer_info()[0],
            len(sizes) - 1,
            models,
            batch,
            int(training),
            int(precision == "fp32"),
            int(profile),
        )
        self.check(not handle)
        return handle

    def destroy(self, handle):
        """Release one workspace on eviction or shutdown."""
        self.library.wtf_cuda_destroy(handle)

    # Arguments mirror the explicit versioned native ABI.
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def run(self, handle, buffers, discount, threshold, double_q, output):
        """Synchronous transfer/compute/readback; no updates are left in flight."""
        self.check(
            self.library.wtf_cuda_run(
                handle,
                *(b.buffer_info()[0] for b in buffers),
                discount,
                threshold,
                int(double_q),
                output.buffer_info()[0],
            )
        )

    def predict(self, handle, weights, observations, output):
        """Evaluate independent dense networks in one synchronous batch."""
        self.check(
            self.library.wtf_cuda_predict(
                handle,
                weights.buffer_info()[0],
                observations.buffer_info()[0],
                output.buffer_info()[0],
            )
        )

    def stats(self, handle):
        """Read counters without synchronizing an unprofiled device."""
        result = array("d", [0.0]) * 8
        self.check(self.library.wtf_cuda_stats(handle, result.buffer_info()[0]))
        return list(result)

    def ppo_begin(self, handle, weights, moments, variances):
        """Upload private PPO parameter/optimizer state once per rollout update."""
        self.check(
            self.library.wtf_cuda_ppo_begin(
                handle, *(b.buffer_info()[0] for b in (weights, moments, variances))
            )
        )

    def ppo(self, handle, buffers, options, summary):
        """One synchronized multi-animal PPO minibatch with device Adam and KL gate."""
        self.check(
            self.library.wtf_cuda_ppo(
                handle,
                *(b.buffer_info()[0] for b in buffers),
                options.buffer_info()[0],
                summary.buffer_info()[0],
            )
        )

    def ppo_end(self, handle, weights, moments, variances):
        """Publish coherent model and Adam buffers at the rollout boundary."""
        self.check(
            self.library.wtf_cuda_ppo_end(
                handle, *(b.buffer_info()[0] for b in (weights, moments, variances))
            )
        )


def workspace_bytes(sizes, models, batch, training=True, precision="fp64"):
    """Count device buffers; host packing/cache and driver allocations are separate."""
    parameters = sum((ni + 1) * no for ni, no in zip(sizes, sizes[1:]))
    real = 4 if precision == "fp32" else 8
    samples = models * batch
    base = 4 * (3 * len(sizes) + 1) + 4 * models * parameters
    base += real * (samples * sizes[0] + samples * 2 * sum(sizes))
    if not training:
        return base + real * models * sizes[-1]
    if training == 2:
        return (
            base
            + 8 * models * parameters
            + real * models * (parameters + 5)
            + samples * (8 + 8 * real + sizes[-1] - 1)
            + 8 * models
            + real * models * 6
        )
    return (
        base
        + 4 * models * parameters
        + real * models * (parameters + 1)
        + samples * (8 + 2 * real + real * sizes[0])
    )


# Workspace ownership, limits and execution counters have separate lifetimes.
# pylint: disable-next=too-many-instance-attributes
class CudaTraining:
    """Batch independent stock brains, retaining bounded reusable workspaces."""

    # One experiment owns cache limits, precision, profiling and action strategy.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        max_agents=64,
        workspace_mb=256,
        library=None,
        *,
        precision="fp64",
        profile=False,
        inference="auto",
    ):
        # Exact integers reject bool and accidental implicit coercion.
        # pylint: disable=unidiomatic-typecheck
        if type(max_agents) is not int or not 1 <= max_agents <= 4096:
            raise ValueError("CUDA batch agents must be 1..4096")
        if type(workspace_mb) is not int or not 1 <= workspace_mb <= 16384:
            raise ValueError("CUDA workspace MB must be 1..16384")
        if precision not in ("fp64", "fp32") or type(profile) is not bool:
            raise ValueError(
                "CUDA precision must be fp64/fp32 and profile must be bool"
            )
        if inference not in ("auto", "cpu", "cuda"):
            raise ValueError("CUDA inference must be auto/cpu/cuda")
        self.inference = inference
        self.precision, self.profile = precision, profile
        self.typecode = "f" if precision == "fp32" else "d"
        self.buffers = {}
        self.layouts = {}
        self.packing_seconds = self.optimizer_seconds = 0.0
        self.evicted_stats = [0.0] * 8
        self.ppo_pending = []
        self.ppo_batches = self.ppo_models = 0
        self.library = library if library is not None else CudaLibrary()
        self.max_agents = max_agents
        self.budget = workspace_mb * 1024 * 1024
        self.workspaces = OrderedDict()
        self.allocated = 0
        self.pending = []
        self.active = False
        self.batches = self.models = 0
        self.prediction_batches = self.predictions = 0

    def buffer(self, handle, label, typecode, count):
        """Reuse one fixed-size staging array per live workspace and purpose."""
        key = handle, label
        result = self.buffers.get(key)
        if result is None or result.typecode != typecode or len(result) != count:
            result = array(typecode, [0]) * count
            self.buffers[key] = result
        return result

    def pack_models(self, handle, label, models):
        """Bulk-copy contiguous FP32 storage; native cache compares exact contents."""
        sizes, models_count = self.layouts[handle]
        if len(models) != models_count:
            raise ValueError("CUDA model count mismatch")
        for layers in models:
            if len(layers) != len(sizes) - 1 or any(
                len(layer["weights"]) != ni * no or len(layer["bias"]) != no
                for layer, ni, no in zip(layers, sizes, sizes[1:])
            ):
                raise ValueError("CUDA parameter buffer dimensions mismatch")
        count = sum(
            len(layer[key])
            for layers in models
            for layer in layers
            for key in ("weights", "bias")
        )
        result = self.buffer(handle, label, "f", count)
        offset = 0
        for layers in models:
            for layer in layers:
                for key in ("weights", "bias"):
                    values = layer[key]
                    result[offset : offset + len(values)] = values
                    offset += len(values)
        return result

    def _destroy(self, handle):
        for i, value in enumerate(self.library.stats(handle)):
            self.evicted_stats[i] += value
        self.library.destroy(handle)
        self.layouts.pop(handle, None)
        for key in tuple(self.buffers):
            if key[0] == handle:
                del self.buffers[key]

    @classmethod
    def accepts_ppo(cls, brain):
        """Only stock PPO/masked-PPO objectives and packed private Adam state."""
        # Exact types keep custom objectives and recurrent training on CPU.
        # pylint: disable=unidiomatic-typecheck
        if type(brain) not in (PPOBrain, MaskablePPOBrain) or not cls.supports(brain):
            return False
        names = ("learn", "_update", "objective", "_adam_step")
        if any(
            name in vars(brain)
            or getattr(getattr(type(brain), name), "__module__", "")
            not in (
                "wtf.learning.ppo",
                "wtf.learning.contextual_ppo",
                "wtf.learning.optimization",
            )
            for name in names
        ):
            return False
        if any(
            name in vars(brain.network)
            for name in ("accumulate_gradient", "clear_gradients", "zeros")
        ):
            return False
        return all(
            isinstance(v, Float32)
            for tree in (brain.moments, brain.variances)
            for layer in tree
            for v in layer.values()
        )

    def submit_ppo(self, brain, advantages, targets):
        """Queue ready rollout updates; all private state publishes before inheritance."""
        if not self.active:
            raise RuntimeError("Submit CUDA updates inside batch()")
        self.ppo_pending.append((brain, advantages, targets))
        if len(self.ppo_pending) >= self.max_agents:
            self.flush()

    def _flush_ppo(self):
        # Optional module has no effect on normal CPU imports/training.
        # pylint: disable=import-outside-toplevel
        from wtf.learning.cuda_ppo import groups, train

        jobs, self.ppo_pending = self.ppo_pending, []
        for (sizes, batch, options), group in groups(jobs).items():
            per_model = workspace_bytes(sizes, 1, batch, 2, self.precision) - 4 * (
                3 * len(sizes) + 1
            )
            capacity = min(
                self.max_agents, (self.budget - 4 * (3 * len(sizes) + 1)) // per_model
            )
            if capacity < 1:
                raise RuntimeError("One PPO model exceeds --cuda-workspace-mb")
            for start in range(0, len(group), capacity):
                train(self, group[start : start + capacity], sizes, batch, options)

    @staticmethod
    def accepts(brain):
        """Preserve legacy FP64 storage, custom learner/network hooks and other algorithms."""
        # Exact stock types are intentional: custom hooks stay on their CPU path.
        # pylint: disable=unidiomatic-typecheck
        if type(brain) not in (DQNBrain, DDQNBrain) or any(
            name in vars(brain)
            for name in (
                "learn",
                "_batch_loss",
                "_adam_step",
                "_finish_update",
                "bootstrap",
                "values",
                "_mean_values",
            )
        ):
            return False
        if any(
            getattr(getattr(type(brain), name), "__module__", "")
            not in (
                "wtf.learning.dqn",
                "wtf.learning.qrdqn",
                "wtf.learning.optimization",
            )
            for name in (
                "learn",
                "_batch_loss",
                "_adam_step",
                "_finish_update",
                "bootstrap",
                "values",
                "_mean_values",
            )
        ):
            return False
        return all(
            type(net) is MLP
            and not any(
                name in vars(net)
                for name in (
                    "forward",
                    "predict",
                    "predict_range",
                    "_predict",
                    "accumulate_gradient",
                )
            )
            and all(
                getattr(getattr(type(net), name), "__module__", "")
                == "wtf.learning.ppo"
                for name in (
                    "forward",
                    "predict",
                    "predict_range",
                    "_predict",
                    "accumulate_gradient",
                )
            )
            and all(
                isinstance(values, Float32)
                for layer in net.layers
                for values in layer.values()
            )
            for net in (brain.network, brain.target_network)
        )

    @contextmanager
    def batch(self):
        """Flush before champion selection/inheritance; discard queued work on errors."""
        if self.active:
            raise RuntimeError("Nested CUDA training batches are unsupported")
        self.active = True
        token = PPO_ACCELERATOR.set(self)
        try:
            yield
            self.flush()
        except BaseException:
            self.pending.clear()
            self.ppo_pending.clear()
            raise
        finally:
            PPO_ACCELERATOR.reset(token)
            self.active = False

    def submit(self, brain, samples, gradients):
        """Retain at most max_agents jobs before synchronous bounded processing."""
        if not self.active:
            raise RuntimeError("Submit CUDA updates inside batch()")
        if not 1 <= len(samples) <= 4096:
            raise ValueError("CUDA sample batch must be 1..4096")
        self.pending.append((brain, samples, gradients))
        if len(self.pending) >= self.max_agents:
            self.flush()

    def _workspace(self, sizes, count, batch, training=True):
        key = sizes, count, batch, training
        if key in self.workspaces:
            self.workspaces.move_to_end(key)
            return self.workspaces[key][0]
        required = workspace_bytes(sizes, count, batch, training, self.precision)
        while self.workspaces and self.allocated + required > self.budget:
            _, (handle, size) = self.workspaces.popitem(last=False)
            self._destroy(handle)
            self.allocated -= size
        if required > self.budget:
            raise RuntimeError(
                "One CUDA job exceeds the workspace budget; increase --cuda-workspace-mb"
            )
        handle = self.library.create(
            sizes,
            count,
            batch,
            training,
            precision=self.precision,
            profile=self.profile,
        )
        self.workspaces[key] = handle, required
        self.layouts[handle] = sizes, count
        self.allocated += required
        return handle

    # Group dimensions and budget arithmetic belong to one scheduling operation.
    # pylint: disable-next=too-many-locals
    def flush(self):
        """Group compatible layouts without mixing weights, samples or gradient sums."""
        jobs, self.pending = self.pending, []
        groups = {}
        for job in jobs:
            brain, samples, _ = job
            key = (
                tuple(brain.network.sizes),
                len(samples),
                brain.algorithm,
                brain.config["discount_factor"],
                brain.config["huber_threshold"],
            )
            groups.setdefault(key, []).append(job)
        for (sizes, batch, algorithm, discount, threshold), group in groups.items():
            if not 2 <= len(sizes) <= 17 or any(not 1 <= n <= 4096 for n in sizes):
                raise RuntimeError(
                    "CUDA supports 1..16 dense layers, each 1..4096 wide"
                )
            per_model = workspace_bytes(sizes, 1, batch, True, self.precision) - 4 * (
                3 * len(sizes) + 1
            )
            capacity = min(
                self.max_agents,
                max(0, (self.budget - 4 * (3 * len(sizes) + 1)) // per_model),
            )
            if not capacity:
                raise RuntimeError("One CUDA job exceeds --cuda-workspace-mb")
            for start in range(0, len(group), capacity):
                part = group[start : start + capacity]
                self._run(part, sizes, batch, discount, threshold, algorithm == "ddqn")

        self._flush_ppo()

    @staticmethod
    # Early exits keep unsupported contracts out of native buffer handling.
    # pylint: disable-next=too-many-return-statements
    def supports(brain):
        """Recognize stock dense action-selection contracts without importing a device."""
        # Exact stock types and method origins keep caller overrides on CPU.
        # pylint: disable=unidiomatic-typecheck
        policy = (
            PPOBrain,
            A2CBrain,
            ReinforceBrain,
            GRPOBrain,
            MaskablePPOBrain,
            MAPPOBrain,
            TRPOBrain,
            DiscreteSACBrain,
        )
        if type(brain) not in policy + (DQNBrain, DDQNBrain, QRDQNBrain):
            return False
        hooks = (
            "choose",
            "_choose_outputs",
            "distribution",
            "values",
            "_mean_values",
            "action_probabilities",
        )
        if any(key in vars(brain) for key in hooks):
            return False
        if getattr(type(brain).choose, "__module__", "") not in (
            "wtf.learning.ppo",
            "wtf.learning.policy_gradients",
            "wtf.learning.contextual_ppo",
            "wtf.learning.discrete_sac",
            "wtf.learning.qrdqn",
        ):
            return False
        network = brain.network
        if type(brain) is MAPPOBrain:
            if type(network) is not CentralNetwork:
                return False
            nets = (network.actor, network.critic)
        elif type(brain) in (TRPOBrain, DiscreteSACBrain):
            if type(network) is not PolicyBundle:
                return False
            nets = (network.actor, *network.critics)
        else:
            nets = (network,)
        if any(key in vars(network) for key in ("predict", "value")):
            return False
        return all(
            type(net) in (MLP, LocalMLP)
            and not any(
                key in vars(net)
                for key in ("predict", "forward", "predict_range", "_predict")
            )
            and all(
                getattr(getattr(type(net), hook), "__module__", "")
                in ("wtf.learning.ppo", "wtf.learning.contextual_ppo")
                for hook in ("predict", "forward", "predict_range", "_predict")
            )
            and all(
                isinstance(value, Float32)
                for layer in net.layers
                for value in layer.values()
            )
            for net in nets
        )

    # Retain the CPU policy sampling contract locally without changing its baseline.
    # pylint: disable=duplicate-code
    def choose_actions(self, brains, observations, explore=True):
        """Batch dense stock policies/value nets; sampling stays on each animal's CPU RNG."""
        # Deliberate exact-type qualification preserves recurrent/custom controller hooks.
        # pylint: disable=unidiomatic-typecheck,protected-access,too-many-locals,too-many-branches,too-many-statements
        # Deferred import avoids the batched-inference collaborator cycle.
        # pylint: disable=import-outside-toplevel
        from wtf.learning.batched import choose_actions as cpu_actions

        cpu_observations = {
            i: o
            for i, o in observations.items()
            if self.inference == "cpu"
            or (
                self.inference == "auto"
                and (self.accepts(brains[i]) or self.accepts_ppo(brains[i]))
            )
        }
        if cpu_observations:
            result = cpu_actions(brains, cpu_observations, explore=explore)
            remaining = {
                i: o for i, o in observations.items() if i not in cpu_observations
            }
            if remaining:
                result.update(self.choose_actions(brains, remaining, explore))
            return {i: result[i] for i in observations}
        value_kinds = (DQNBrain, DDQNBrain, QRDQNBrain)
        jobs, prepared, actions = [], {}, {}
        for identity, observation in observations.items():
            brain = brains[identity]
            kind = type(brain)
            if not self.supports(brain):
                continue
            network = brain.network
            bundle = kind in (MAPPOBrain, TRPOBrain, DiscreteSACBrain)
            if kind in value_kinds and explore and brain.rng.random() < brain.epsilon:
                actions[identity] = brain.rng.choice(brain.actions)
                continue
            packed = (
                brain._pack(observation)
                if kind is MaskablePPOBrain
                else (
                    tuple(observation) + brain.context
                    if kind is MAPPOBrain and explore
                    else observation
                )
            )
            index = len(jobs)
            if bundle:
                jobs.append((network.actor.layers, tuple(observation)))
                if kind is not DiscreteSACBrain and explore:
                    critic = (
                        network.critic if kind is MAPPOBrain else network.critics[0]
                    )
                    jobs.append((critic.layers, tuple(packed)))
                    parts = 2
                else:
                    parts = 0
            else:
                jobs.append((network.layers, tuple(packed[: network.sizes[0]])))
                parts = -1 if kind in value_kinds else 1
            prepared[identity] = packed, index, parts
        outputs = self.predict_many(jobs)
        for identity, observation in observations.items():
            if identity in actions:
                continue
            brain = brains[identity]
            if identity not in prepared:
                actions[identity] = brain.choose(
                    observation, explore=explore, random_baseline=False
                )
                continue
            packed, index, parts = prepared[identity]
            values = outputs[index]
            if parts == -1:
                values = brain._mean_values(values)
                actions[identity] = brain.actions[
                    max(range(len(values)), key=values.__getitem__)
                ]
            else:
                if parts == 2:
                    values = values + outputs[index + 1]
                elif parts == 0:
                    values = values + [0.0]
                actions[identity] = brain._choose_outputs(packed, values, explore)
        return actions

    # pylint: enable=duplicate-code

    # Keep ABI packing and its capacity checks adjacent for auditability.
    # pylint: disable-next=too-many-locals
    def predict_many(self, jobs):
        """Group architecture-compatible private policy/value forwards, preserving order."""
        outputs = [None] * len(jobs)
        groups = {}
        for index, (layers, observation) in enumerate(jobs):
            sizes = (len(observation), *(len(layer["bias"]) for layer in layers))
            if not 2 <= len(sizes) <= 17 or any(
                not 1 <= size <= 4096 for size in sizes
            ):
                raise ValueError("Unsupported CUDA prediction layout")
            if any(
                len(layer["weights"]) != ni * no
                for layer, ni, no in zip(layers, sizes, sizes[1:])
            ):
                raise ValueError("CUDA prediction parameter dimensions mismatch")
            groups.setdefault(sizes, []).append((index, layers, observation))
        for sizes, group in groups.items():
            fixed = 4 * (3 * len(sizes) + 1)
            per_model = workspace_bytes(sizes, 1, 1, False, self.precision) - fixed
            capacity = min(self.max_agents, max(0, (self.budget - fixed) // per_model))
            if not capacity:
                raise RuntimeError("One prediction model exceeds --cuda-workspace-mb")
            for start in range(0, len(group), capacity):
                part = group[start : start + capacity]
                started = time.perf_counter() if self.profile else 0.0
                handle = self._workspace(sizes, len(part), 1, False)
                weights = self.pack_models(
                    handle, "predict_weights", [layers for _, layers, _ in part]
                )
                observations = self.buffer(
                    handle, "predict_obs", self.typecode, len(part) * sizes[0]
                )
                for row, (_, _, observation) in enumerate(part):
                    observations[row * sizes[0] : (row + 1) * sizes[0]] = array(
                        self.typecode, observation
                    )
                if any(not math.isfinite(value) for value in observations):
                    raise ValueError("Non-finite CUDA prediction input")
                result = self.buffer(
                    handle, "predict_result", self.typecode, len(part) * sizes[-1]
                )
                if self.profile:
                    self.packing_seconds += time.perf_counter() - started
                self.library.predict(handle, weights, observations, result)
                if not all(math.isfinite(value) for value in result):
                    raise RuntimeError("Non-finite CUDA prediction output")
                for row, (index, _, _) in enumerate(part):
                    outputs[index] = list(
                        result[row * sizes[-1] : (row + 1) * sizes[-1]]
                    )
                self.prediction_batches += 1
                self.predictions += len(part)
        return outputs

    # Explicit batch/layout and TD-objective arguments prevent implicit ABI state.
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-branches
    def _run(self, jobs, sizes, batch, discount, threshold, double_q):
        # Buffer ordering is the explicit C ABI; keep it visible beside the pack/readback loop.
        # pylint: disable=too-many-locals,protected-access
        started = time.perf_counter() if self.profile else 0.0
        handle = self._workspace(sizes, len(jobs), batch)
        weights = self.pack_models(
            handle, "dqn_weights", [b.network.layers for b, _, _ in jobs]
        )
        targets = self.pack_models(
            handle, "dqn_targets", [b.target_network.layers for b, _, _ in jobs]
        )
        observations, next_observations, rewards = (
            array(self.typecode),
            array(self.typecode),
            array(self.typecode),
        )
        actions, terminal = array("i"), array("i")
        parameters = sum((ni + 1) * no for ni, no in zip(sizes, sizes[1:]))
        for brain, samples, _ in jobs:
            for net in (brain.network, brain.target_network):
                if tuple(net.sizes) != sizes or len(net.layers) != len(sizes) - 1:
                    raise ValueError("CUDA online/target layout mismatch")
            for item in samples:
                if len(item.observation) != sizes[0] or (
                    not item.terminated
                    and (
                        item.next_observation is None
                        or len(item.next_observation) != sizes[0]
                    )
                ):
                    raise ValueError(
                        "CUDA DQN observation width mismatch or missing bootstrap input"
                    )
                observations.extend(item.observation)
                next_observations.extend(
                    [0.0] * sizes[0] if item.terminated else item.next_observation
                )
                rewards.append(item.reward)
                action = brain.action_indices[item.action]
                if not 0 <= action < sizes[-1]:
                    raise ValueError("CUDA action index is outside the output layer")
                actions.append(action)
                terminal.append(int(item.terminated))
        buffers = (
            weights,
            targets,
            observations,
            next_observations,
            rewards,
            actions,
            terminal,
        )
        if (weights.itemsize, observations.itemsize, actions.itemsize) != (
            4,
            4 if self.precision == "fp32" else 8,
            4,
        ):
            raise RuntimeError("CUDA ABI requires float32/float64/int32 host buffers")
        if any(not math.isfinite(value) for buffer in buffers[2:5] for value in buffer):
            raise ValueError("Non-finite CUDA input")
        output = self.buffer(
            handle, "dqn_result", self.typecode, len(jobs) * (parameters + 1)
        )
        if self.profile:
            self.packing_seconds += time.perf_counter() - started
        self.library.run(handle, buffers, discount, threshold, double_q, output)
        if not getattr(self.library, "validates_output", False) and not all(
            math.isfinite(value) for value in output
        ):
            raise RuntimeError("CUDA returned non-finite gradients/loss")
        started = time.perf_counter() if self.profile else 0.0
        for index, (brain, samples, gradients) in enumerate(jobs):
            offset = index * (parameters + 1)
            for layer in gradients:
                for key in ("weights", "bias"):
                    count = len(layer[key])
                    layer[key][:] = (
                        array(layer[key].typecode, output[offset : offset + count])
                        if isinstance(layer[key], array)
                        else output[offset : offset + count]
                    )
                    offset += count
            brain._finish_update(gradients, output[offset], len(samples))
        if self.profile:
            self.optimizer_seconds += time.perf_counter() - started
        self.batches += 1
        self.models += len(jobs)

    def status(self):
        """Report execution/workspace facts, never claim an unmeasured speedup."""
        stats = list(self.evicted_stats)
        for handle, _ in self.workspaces.values():
            for i, value in enumerate(self.library.stats(handle)):
                stats[i] += value
        return {
            "precision": self.precision,
            "profile": self.profile,
            "inference": self.inference,
            "packing_seconds": self.packing_seconds,
            "cpu_optimizer_seconds": self.optimizer_seconds,
            "cuda_upload_seconds": stats[0],
            "cuda_compute_seconds": stats[1],
            "cuda_download_seconds": stats[2],
            "host_to_device_bytes": int(stats[3]),
            "device_to_host_bytes": int(stats[4]),
            "weight_uploads": int(stats[5]),
            "weight_cache_hits": int(stats[6]),
            "ppo_batches": self.ppo_batches,
            "ppo_model_updates": self.ppo_models,
            "backend": "cuda-experimental",
            "device": self.library.device,
            "workspace_bytes": self.allocated,
            "workspace_budget_bytes": self.budget,
            "cached_workspaces": len(self.workspaces),
            "batches": self.batches,
            "model_updates": self.models,
            "prediction_batches": self.prediction_batches,
            "predictions": self.predictions,
        }

    def close(self):
        """Release only this experiment's workspaces."""
        for handle, _ in self.workspaces.values():
            self._destroy(handle)
        self.workspaces.clear()
        self.allocated = 0
        self.pending.clear()
        self.ppo_pending.clear()
        self.buffers.clear()
        self.layouts.clear()

    def __del__(self):
        # Partial initialization/interpreter shutdown must not leak a Python exception.
        try:
            self.close()
        except (AttributeError, TypeError, OSError):
            pass
