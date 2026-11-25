"""CUDA orchestration without hardware, plus explicitly requested NVIDIA tests.

The reference double validates packing/scheduling, NOT CUDA execution or speed.
Set WTF_TEST_CUDA=1 on Linux/NVIDIA to require the real library/device tests.
"""

# Regression fixtures inspect private state to verify independent updates and buffers.
# The independent ABI oracle deliberately repeats its buffer layout/algorithm inventory.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring,duplicate-code

from array import array
import copy
import os
import random
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from tests.test_advanced_algorithms import settings
from tests.test_mutation import environment
from tests.reproduction_support import force_reproduction
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.learning.brain import Experience
from wtf.learning.brains import create_brain
from wtf.learning.cuda_training import CudaLibrary, CudaTraining, workspace_bytes
from wtf.learning.dqn import huber_td
from wtf.learning.ppo import MLP, PPOBrain, RolloutStep, loss_and_gradient
from wtf.native.cuda_build import main as build_cuda
from work.benchmark_cuda import configuration, KINDS, worker


class ReferenceLibrary:
    """CPU oracle at the device ABI boundary, without emulating CUDA availability."""

    device = "CPU test double (not a GPU)"

    def __init__(self):
        self.handles = {}
        self.next_handle = 0
        self.freed = []
        self.fail = False
        self.ppo_states = {}

    def create(self, sizes, models, batch, training=True, **_options):
        self.next_handle += 1
        self.handles[self.next_handle] = sizes, models, batch, training
        return self.next_handle

    def stats(self, _handle):
        return [0.0] * 8

    def destroy(self, handle):
        self.freed.append(handle)
        del self.handles[handle]
        self.ppo_states.pop(handle, None)

    @staticmethod
    def network(sizes, weights):
        network = MLP(
            list(sizes[1:-1]), sizes[-1], 0, value_head=False, inputs=sizes[0]
        )
        offset = 0
        for layer in network.layers:
            for key in ("weights", "bias"):
                length = len(layer[key])
                layer[key][:] = array("f", weights[offset : offset + length])
                offset += length
        return network

    def predict(self, handle, weights, observations, output):
        sizes, models, _, training = self.handles[handle]
        assert not training
        parameters = sum((ni + 1) * no for ni, no in zip(sizes, sizes[1:]))
        for model in range(models):
            network = self.network(
                sizes, weights[model * parameters : (model + 1) * parameters]
            )
            values = network.predict(
                observations[model * sizes[0] : (model + 1) * sizes[0]]
            )
            output[model * sizes[-1] : (model + 1) * sizes[-1]] = array(
                output.typecode, values
            )

    # Keep the complete independent scalar oracle beside the matching ABI signature.
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
    def run(self, handle, buffers, discount, threshold, double_q, output):
        if self.fail:
            raise RuntimeError("injected GPU failure")
        sizes, models, batch, training = self.handles[handle]
        assert training
        (
            weights,
            targets,
            observations,
            next_observations,
            rewards,
            actions,
            terminal,
        ) = buffers
        parameters = sum((ni + 1) * no for ni, no in zip(sizes, sizes[1:]))
        for model in range(models):
            online = self.network(
                sizes, weights[model * parameters : (model + 1) * parameters]
            )
            target_net = self.network(
                sizes, targets[model * parameters : (model + 1) * parameters]
            )
            gradients, total = online.zeros(), 0.0
            for sample in range(model * batch, (model + 1) * batch):
                obs = observations[sample * sizes[0] : (sample + 1) * sizes[0]]
                values, tape = online.forward(obs)
                target = rewards[sample]
                if not terminal[sample]:
                    next_obs = next_observations[
                        sample * sizes[0] : (sample + 1) * sizes[0]
                    ]
                    q = target_net.predict(next_obs)
                    if double_q:
                        choice = online.predict(next_obs)
                        target += (
                            discount
                            * q[max(range(len(choice)), key=choice.__getitem__)]
                        )
                    else:
                        target += discount * max(q)
                loss, derivative = huber_td(values[actions[sample]], target, threshold)
                total += loss
                delta = [0.0] * sizes[-1]
                delta[actions[sample]] = derivative
                online.accumulate_gradient(tape, delta, gradients)
            packed = [
                value
                for layer in gradients
                for key in ("weights", "bias")
                for value in layer[key]
            ] + [total]
            output[model * (parameters + 1) : (model + 1) * (parameters + 1)] = array(
                output.typecode, packed
            )

    def ppo_begin(self, handle, weights, moments, variances):
        sizes, models, _, mode = self.handles[handle]
        assert mode == 2
        width = sum((ni + 1) * no for ni, no in zip(sizes, sizes[1:]))
        brains = []
        for model in range(models):
            brain = object.__new__(PPOBrain)
            part = slice(model * width, (model + 1) * width)
            brain.network = self.network(sizes, weights[part])
            brain.moments = self.network(sizes, moments[part]).layers
            brain.variances = self.network(sizes, variances[part]).layers
            brains.append(brain)
        self.ppo_states[handle] = brains

    # The independent oracle keeps all scalar objective/optimizer state visible.
    # pylint: disable-next=too-many-locals
    def ppo(self, handle, buffers, options, summary):
        if self.fail:
            raise RuntimeError("injected GPU failure")
        sizes, _models, batch, _ = self.handles[handle]
        observations, data, actions, valid, masks, meta = buffers
        config = dict(
            zip(
                (
                    "clip_ratio",
                    "entropy_coefficient",
                    "value_coefficient",
                    "max_grad_norm",
                    "learning_rate",
                    "target_kl",
                ),
                options,
            )
        )
        keys = (
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl",
            "clip_fraction",
        )
        for model, brain in enumerate(self.ppo_states[handle]):
            brain.config = config
            brain.updates = meta[model * 2 + 1] - 1
            count = meta[model * 2]
            gradient = brain.network.zeros()
            metrics = [0.0] * 5
            for sample in range(model * batch, (model + 1) * batch):
                if not valid[sample]:
                    continue
                obs = observations[sample * sizes[0] : (sample + 1) * sizes[0]]
                outputs, tape = brain.network.forward(obs)
                mask = masks[sample * (sizes[-1] - 1) : (sample + 1) * (sizes[-1] - 1)]
                masked = [
                    v if enabled else -1e30 for v, enabled in zip(outputs[:-1], mask)
                ] + [outputs[-1]]
                _, derivative, info = loss_and_gradient(
                    masked, actions[sample], *data[sample * 3 : sample * 3 + 3], config
                )
                for j, enabled in enumerate(mask):
                    if not enabled:
                        derivative[j] = 0.0
                brain.network.accumulate_gradient(tape, derivative, gradient)
                for j, key in enumerate(keys):
                    metrics[j] += info[key]
            accepted = count > 0 and (
                not options[5] or metrics[3] / count <= options[5]
            )
            if accepted:
                brain._adam_step(gradient, count)
            summary[model * 6 : (model + 1) * 6] = array(
                summary.typecode, metrics + [float(accepted)]
            )

    def ppo_end(self, handle, weights, moments, variances):
        for output, name in (
            (weights, "network"),
            (moments, "moments"),
            (variances, "variances"),
        ):
            flat = []
            for brain in self.ppo_states[handle]:
                layers = (
                    brain.network.layers if name == "network" else getattr(brain, name)
                )
                flat.extend(
                    v
                    for layer in layers
                    for key in ("weights", "bias")
                    for v in layer[key]
                )
            output[:] = array("f", flat)


class CapturingLibrary:
    """Capture raw native gradients before Adam can obscure a scaling defect."""

    def __init__(self, library):
        self.library = library
        self.outputs = []

    def __getattr__(self, name):
        return getattr(self.library, name)

    def run(self, *args):
        self.library.run(*args)
        self.outputs.append(list(args[-1]))


def accelerated(config, *, max_agents=64, workspace_mb=8):
    experiment = Experiment(config)
    experiment._cuda_training = CudaTraining(
        max_agents, workspace_mb, ReferenceLibrary()
    )
    return experiment


class CudaSchedulingTests(unittest.TestCase):
    def test_benchmark_worlds_use_the_requested_seed(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                config = configuration(kind, 4, 100, [3])
                left, right = Experiment(config), Experiment(copy.deepcopy(config))
                self.assertEqual(left.world.seed, 100)
                self.assertEqual(left.checkpoint(), right.checkpoint())
                self.assertEqual(
                    Experiment(configuration(kind, 4, 101, [3])).world.seed, 101
                )

    def test_private_dqn_and_ddqn_updates_match_cpu_and_checkpoint_restore(self):
        for kind in ("dqn", "ddqn"):
            with self.subTest(kind=kind):
                config = environment(kind)
                config["world"]["max_steps"] = 12
                cpu, device = Experiment(config), accelerated(config, max_agents=2)
                for _ in range(12):
                    cpu.step()
                    device.step()
                    self.assertEqual(cpu.checkpoint(), device.checkpoint())
                self.assertGreater(device._cuda_training.models, 0)
                restored = Experiment.restore(device.checkpoint())
                self.assertIsNone(restored._cuda_training)
                self.assertEqual(restored.checkpoint(), device.checkpoint())
                handles = device._cuda_training.library.handles
                device.set_training_device("cpu")
                self.assertFalse(handles)

    def test_dense_policy_families_match_cpu_with_reference_forward(self):
        for kind in (
            "ppo",
            "maskable_ppo",
            "a2c",
            "reinforce",
            "grpo",
            "mappo",
            "trpo",
            "discrete_sac",
            "qrdqn",
        ):
            with self.subTest(kind=kind):
                config = environment(kind)
                config["world"]["max_steps"] = 8
                if kind == "qrdqn":
                    for cfg in config["brains"].values():
                        if cfg["type"] == kind:
                            cfg["exploration"].update(initial=0, minimum=0)
                cpu, device = Experiment(config), accelerated(config)
                for _ in range(8):
                    cpu.step()
                    device.step()
                    self.assertEqual(cpu.checkpoint(), device.checkpoint())
                self.assertGreater(
                    device._cuda_training.predictions
                    + device._cuda_training.ppo_models,
                    0,
                    kind,
                )

    def test_three_species_and_mixed_unsupported_recurrent_brains_remain_private(self):
        config = environment("dqn")
        config["brains"]["recurrent"] = settings("recurrent_ppo")
        config["species"]["lynx"] = copy.deepcopy(config["species"]["predator"])
        config["species"]["lynx"]["brain"] = "recurrent"
        config["species"]["predator"]["brain"] = "test"
        config["species"]["lynx"]["initial_population"] = 2
        validate(config)
        cpu, device = Experiment(config), accelerated(config, max_agents=1)
        for _ in range(10):
            cpu.step()
            device.step()
            self.assertEqual(cpu.checkpoint(), device.checkpoint())
        self.assertEqual(
            len({id(b.network) for b in device.brains.values()}), len(device.brains)
        )

    def test_custom_hooks_and_legacy_storage_use_cpu(self):
        brain = create_brain(settings("dqn"), ["rest", "eat"], 2, 3)
        self.assertTrue(CudaTraining.accepts(brain))
        self.assertTrue(CudaTraining.supports(brain))
        brain.choose = lambda *args, **kwargs: "rest"
        self.assertFalse(CudaTraining.supports(brain))
        del brain.choose
        brain.network._predict = lambda *args, **kwargs: [-1.0, 1.0]
        self.assertFalse(CudaTraining.supports(brain))
        self.assertFalse(CudaTraining.accepts(brain))
        del brain.network._predict
        with patch.object(MLP, "_predict", lambda *args, **kwargs: [-1.0, 1.0]):
            self.assertFalse(CudaTraining.supports(brain))
            self.assertFalse(CudaTraining.accepts(brain))
        brain.network.layers[0]["weights"] = list(brain.network.layers[0]["weights"])
        self.assertFalse(CudaTraining.accepts(brain))
        self.assertFalse(CudaTraining.supports(brain))

    def test_budget_lru_close_and_failed_batch_do_not_publish_weights(self):
        library = ReferenceLibrary()
        engine = CudaTraining(4, 1, library)
        for count in (1, 2, 3, 4):
            engine._workspace((20, 512, 2), count, 2)
            self.assertLessEqual(engine.allocated, engine.budget)
        self.assertTrue(library.freed)
        with engine.batch():
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                with engine.batch():
                    pass
        with self.assertRaisesRegex(RuntimeError, "inside batch"):
            engine.submit(None, [], [])
        with self.assertRaises(RuntimeError):
            engine._workspace((4096, 4096), 1, 32)
        config = environment("dqn")
        experiment = Experiment(config)
        experiment._cuda_training = engine
        library.fail = True
        first = [copy.deepcopy(b.network.layers) for b in experiment.brains.values()]
        experiment.step()  # warmup before batch-size two
        with self.assertRaisesRegex(RuntimeError, "injected GPU failure"):
            experiment.step()
        self.assertFalse(engine.pending)
        self.assertEqual(first, [b.network.layers for b in experiment.brains.values()])
        engine.close()
        self.assertEqual(engine.allocated, 0)
        self.assertFalse(library.handles)

    def test_layout_modes_and_exact_device_buffer_accounting(self):
        sizes = (20, 3, 2)
        parameters = 20 * 3 + 3 + 3 * 2 + 2
        self.assertEqual(
            workspace_bytes(sizes, 1, 2),
            4 * 10
            + 8 * parameters
            + 8 * (2 * 20 * 2 + 2 * 2 * 25 + parameters + 1 + 2 * 2)
            + 8 * 2,
        )
        self.assertLess(
            workspace_bytes(sizes, 1, 2, True, "fp32"), workspace_bytes(sizes, 1, 2)
        )
        self.assertLess(
            workspace_bytes(sizes, 1, 1, False), workspace_bytes(sizes, 1, 1, True)
        )
        for maximum, memory in ((0, 1), (1, 0), (True, 1)):
            with self.assertRaises(ValueError):
                CudaTraining(maximum, memory, ReferenceLibrary())

    def test_staging_detects_raw_buffer_changes_and_is_evicted(self):
        engine = CudaTraining(2, 1, ReferenceLibrary())
        brain = create_brain(settings("dqn"), ["rest", "eat"], 2, 3)
        handle = engine._workspace(tuple(brain.network.sizes), 1, 1, False)
        first = engine.pack_models(handle, "test", [brain.network.layers])
        same = engine.pack_models(handle, "test", [brain.network.layers])
        self.assertIs(first, same)
        view = memoryview(brain.network.layers[0]["weights"])
        view[0] = 0.75
        again = engine.pack_models(handle, "test", [brain.network.layers])
        self.assertEqual(again[0], 0.75)
        engine.close()
        self.assertFalse(engine.buffers)

    def test_ppo_rollout_boundaries_kl_and_custom_hooks(self):
        for kind in ("ppo", "maskable_ppo"):
            for cap in (0, 1e-12):
                config = environment(kind)
                for brain in config["brains"].values():
                    if brain["type"] == kind:
                        brain["target_kl"] = cap
                cpu, device = Experiment(config), accelerated(config, max_agents=2)
                for _ in range(9):
                    cpu.step()
                    device.step()
                    self.assertEqual(cpu.checkpoint(), device.checkpoint())
                brain = next(iter(device.brains.values()))
                self.assertTrue(CudaTraining.accepts_ppo(brain))
                brain.objective = lambda *args: None
                self.assertFalse(CudaTraining.accepts_ppo(brain))

    def test_ppo_publishes_before_inheritance_and_checkpoint_restore(self):
        config = environment("ppo", probability=0)
        for brain in config["brains"].values():
            if brain["type"] == "ppo":
                brain["rollout_steps"] = 2
        cpu, device = Experiment(config), accelerated(config)
        for experiment in (cpu, device):
            for species in config["species"]:
                animals = [
                    a for a in experiment.world.animals.values() if a.species == species
                ]
                for i, animal in enumerate(animals):
                    animal.sex = "female" if i % 2 == 0 else "male"
            force_reproduction(experiment)
        for _ in range(6):
            cpu.step()
            device.step()
            self.assertEqual(cpu.checkpoint(), device.checkpoint())
        self.assertGreater(sum(device.world.births.values()), 0)
        self.assertGreater(device._cuda_training.ppo_models, 0)
        restored = Experiment.restore(device.checkpoint())
        self.assertIsNone(restored._cuda_training)
        self.assertEqual(restored.checkpoint(), device.checkpoint())

    def test_invalid_ppo_buffers_are_rejected_before_native_reads(self):
        config = settings("ppo")
        for action, observation in ((-1, (0.0,) * 20), (0, (0.0,) * 19)):
            brain = create_brain(config, ["rest", "eat"], 2, 3)
            brain.rollout = [
                RolloutStep(observation, action, 1.0, -0.6, 0.1, 0.0, False, True)
            ]
            engine = CudaTraining(library=ReferenceLibrary())
            with self.assertRaisesRegex(ValueError, "observation width or action"):
                with engine.batch():
                    brain._update()
            engine.close()
        brain = create_brain(config, ["rest", "eat"], 2, 3)
        brain.network.layers[0]["weights"].pop()
        engine = CudaTraining(library=ReferenceLibrary())
        handle = engine._workspace(tuple(brain.network.sizes), 1, 1, 2)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            engine.pack_models(handle, "bad", [brain.network.layers])
        engine.close()

    def test_worker_propagates_precision_profile_and_inference(self):

        original = Experiment.set_training_device

        def fake_device(experiment, device="cpu", **options):
            if device == "cuda":
                experiment._cuda_training = CudaTraining(
                    library=ReferenceLibrary(), **options
                )
            else:
                original(experiment, device)

        args = SimpleNamespace(
            algorithm="ppo",
            steps=4,
            warmup=1,
            seed=100,
            hidden=[3],
            worker="cuda",
            batch_agents=64,
            workspace_mb=8,
            cuda_precision="fp32",
            cuda_profile=True,
            cuda_inference="cuda",
        )
        with patch.object(Experiment, "set_training_device", fake_device):
            report = worker(args)
        self.assertEqual(report["accelerator"]["precision"], "fp32")
        self.assertEqual(report["accelerator"]["inference"], "cuda")
        self.assertTrue(report["accelerator"]["profile"])
        self.assertGreater(report["accelerator"]["packing_seconds"], 0)

    def test_missing_gpu_is_explicit_and_build_is_not_implicit(self):
        if sys.platform != "linux":
            with self.assertRaisesRegex(RuntimeError, "Linux/NVIDIA"):
                Experiment(environment("dqn")).set_training_device("cuda")
        with (
            patch("wtf.native.cuda_build.sys.platform", "linux"),
            patch("wtf.native.cuda_build.shutil.which", return_value=None),
        ):
            with self.assertRaises(SystemExit):
                build_cuda([])


@unittest.skipUnless(
    os.environ.get("WTF_TEST_CUDA") == "1",
    "requires explicit Linux/NVIDIA CUDA test run",
)
class CudaHardwareTests(unittest.TestCase):
    def test_real_resident_ppo_and_fp32(self):
        # Full fixed-rollout lifecycle covers private Adam, masks, padding and epochs.
        # pylint: disable=too-many-locals,too-many-nested-blocks
        rng = random.Random(99)
        for precision in ("fp64", "fp32"):
            for kind in ("ppo", "maskable_ppo"):
                config = settings(kind)
                config.update(epochs=2, batch_size=4, target_kl=0)
                config["network"]["hidden_layers"] = [7, 5]
                brains = []
                for model, length in enumerate((3, 5, 8)):
                    brain = create_brain(
                        config, ["rest", "eat"], 10 + model, 70 + model
                    )
                    for step in range(length):
                        obs = tuple(rng.uniform(-1, 1) for _ in range(20))
                        mask = (1, int(step % 2 == 0))
                        outputs = brain.network.predict(obs)
                        packed = obs + mask if kind == "maskable_ppo" else obs
                        logp = brain.distribution(outputs, packed)[1][0]
                        brain.rollout.append(
                            RolloutStep(
                                packed,
                                0,
                                0.3,
                                logp,
                                outputs[-1],
                                0.1,
                                False,
                                step == length - 1,
                            )
                        )
                    brains.append(brain)
                oracle, actual = copy.deepcopy(brains), copy.deepcopy(brains)
                reference = CudaTraining(library=ReferenceLibrary())
                device = CudaTraining(precision=precision, profile=True)
                for engine, population in ((reference, oracle), (device, actual)):
                    with engine.batch():
                        for brain in population:
                            brain._update()
                for a, b in zip(oracle, actual):
                    self.assertEqual(a.updates, b.updates)
                    self.assertEqual(
                        a.training_rng.getstate(), b.training_rng.getstate()
                    )
                    tolerance = 3e-4 if precision == "fp32" else 3e-7
                    for left, right in zip(
                        (a.network.layers, a.moments, a.variances),
                        (b.network.layers, b.moments, b.variances),
                    ):
                        for x, y in zip(left, right):
                            for key in ("weights", "bias"):
                                for u, v in zip(x[key], y[key]):
                                    self.assertAlmostEqual(u, v, delta=tolerance)
                    for key in a.last_metrics:
                        self.assertAlmostEqual(
                            a.last_metrics[key], b.last_metrics[key], delta=tolerance
                        )
                    self.assertFalse(b.rollout)
                self.assertGreater(device.status()["cuda_compute_seconds"], 0)
                reference.close()
                device.close()

    # Raw gradients plus repeated-call cache and numerical checks share one GPU lifecycle.
    # pylint: disable-next=too-many-locals
    def test_real_fp32_forward_gradient_cache_and_timing(self):
        rng = random.Random(401)
        config = settings("ddqn")
        config["network"]["hidden_layers"] = [7, 5]
        brain = create_brain(config, ["rest", "eat"], 2, 3)
        samples = [
            Experience(
                tuple(rng.uniform(-1, 1) for _ in range(20)),
                "eat",
                0.2,
                tuple(rng.uniform(-1, 1) for _ in range(20)),
                False,
                False,
            )
            for _ in range(3)
        ]
        left, right = copy.deepcopy(brain), copy.deepcopy(brain)
        ref_library = CapturingLibrary(ReferenceLibrary())
        gpu_library = CapturingLibrary(CudaLibrary())
        reference = CudaTraining(library=ref_library)
        device = CudaTraining(
            library=gpu_library, precision="fp32", profile=True, inference="cuda"
        )
        with reference.batch(), device.batch():
            reference.submit(left, samples, left.network.zeros())
            device.submit(right, samples, right.network.zeros())
        for expected, actual in zip(ref_library.outputs, gpu_library.outputs):
            for u, v in zip(expected, actual):
                self.assertAlmostEqual(u, v, delta=2e-4)
        for x, y in zip(left.network.layers, right.network.layers):
            for key in x:
                for u, v in zip(x[key], y[key]):
                    self.assertAlmostEqual(u, v, delta=2e-4)
        jobs = [(right.network.layers, samples[0].observation)]
        first = device.predict_many(jobs)
        before = device.status()
        self.assertEqual(device.predict_many(jobs), first)
        after = device.status()
        self.assertGreater(after["weight_cache_hits"], before["weight_cache_hits"])
        self.assertGreater(after["cuda_compute_seconds"], 0)
        self.assertGreater(after["host_to_device_bytes"], 0)
        right.network.layers[0]["weights"][0] += 0.1
        device.predict_many(jobs)
        self.assertGreater(device.status()["weight_uploads"], after["weight_uploads"])
        reference.close()
        device.close()

    # One explicit hardware lifecycle exercises allocation, forwards, gradients and cleanup.
    # pylint: disable-next=too-many-locals,too-many-branches
    def test_real_gradients_and_dense_forwards_agree_with_reference(self):
        rng = random.Random(731)
        for kind in ("dqn", "ddqn"):
            config = settings(kind)
            config["network"]["hidden_layers"] = [7, 5]
            brains = [
                create_brain(config, ["rest", "eat"], 40 + i, 90 + i) for i in range(3)
            ]
            for brain in brains:
                for layer in brain.target_network.layers:
                    for key in layer:
                        layer[key][:] = array(
                            "f", (rng.uniform(-0.5, 0.5) for _ in layer[key])
                        )
            samples = []
            for _ in brains:
                samples.append(
                    [
                        Experience(
                            tuple(rng.uniform(-1, 1) for _ in range(20)),
                            "eat",
                            0.3,
                            (
                                None
                                if terminal
                                else tuple(rng.uniform(-1, 1) for _ in range(20))
                            ),
                            terminal,
                            truncated,
                        )
                        for terminal, truncated in (
                            (False, False),
                            (True, False),
                            (False, True),
                        )
                    ]
                )
            oracle, device = copy.deepcopy(brains), copy.deepcopy(brains)
            reference_library = CapturingLibrary(ReferenceLibrary())
            real_library = CapturingLibrary(CudaLibrary())
            expected = CudaTraining(library=reference_library)
            actual = CudaTraining(library=real_library)
            with expected.batch(), actual.batch():
                for i in range(3):
                    expected.submit(oracle[i], samples[i], oracle[i].network.zeros())
                    actual.submit(device[i], samples[i], device[i].network.zeros())
            self.assertEqual(len(reference_library.outputs), len(real_library.outputs))
            for expected_output, actual_output in zip(
                reference_library.outputs, real_library.outputs
            ):
                self.assertEqual(len(expected_output), len(actual_output))
                for expected_gradient, actual_gradient in zip(
                    expected_output, actual_output
                ):
                    self.assertAlmostEqual(
                        expected_gradient, actual_gradient, delta=1e-8
                    )
            for a, b in zip(oracle, device):
                for x, y in zip(a.network.layers, b.network.layers):
                    for key in x:
                        for left, right in zip(x[key], y[key]):
                            self.assertAlmostEqual(left, right, delta=2e-6)
                self.assertAlmostEqual(
                    a.last_metrics["td_loss"], b.last_metrics["td_loss"], delta=1e-8
                )
            for policy in (
                "ppo",
                "maskable_ppo",
                "a2c",
                "reinforce",
                "grpo",
                "mappo",
                "trpo",
                "discrete_sac",
                "qrdqn",
            ):
                config = environment(policy)
                probe = Experiment(config)
                jobs = []
                for brain in probe.brains.values():
                    if CudaTraining.supports(brain):
                        net = brain.network
                        if hasattr(net, "actor"):
                            net = net.actor
                        obs = tuple(rng.uniform(-1, 1) for _ in range(net.sizes[0]))
                        jobs.append((net.layers, obs))
                self.assertTrue(jobs, policy)
                outputs = actual.predict_many(jobs)
                for (layers, obs), values in zip(jobs, outputs):
                    net = ReferenceLibrary.network(
                        (len(obs), *(len(l["bias"]) for l in layers)),
                        [
                            v
                            for l in layers
                            for key in ("weights", "bias")
                            for v in l[key]
                        ],
                    )
                    for left, right in zip(net.predict(obs), values):
                        self.assertAlmostEqual(left, right, delta=1e-8)
            expected.close()
            actual.close()
