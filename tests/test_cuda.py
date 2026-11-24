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

from tests.test_advanced_algorithms import settings
from tests.test_mutation import environment
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.learning.brain import Experience
from wtf.learning.brains import create_brain
from wtf.learning.cuda_training import CudaLibrary, CudaTraining, workspace_bytes
from wtf.learning.dqn import huber_td
from wtf.learning.ppo import MLP
from wtf.native.cuda_build import main as build_cuda
from work.benchmark_cuda import configuration, KINDS


class ReferenceLibrary:
    """CPU oracle at the device ABI boundary, without emulating CUDA availability."""

    device = "CPU test double (not a GPU)"

    def __init__(self):
        self.handles = {}
        self.next_handle = 0
        self.freed = []
        self.fail = False

    def create(self, sizes, models, batch, training=True):
        self.next_handle += 1
        self.handles[self.next_handle] = sizes, models, batch, training
        return self.next_handle

    def destroy(self, handle):
        self.freed.append(handle)
        del self.handles[handle]

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
            output[model * sizes[-1] : (model + 1) * sizes[-1]] = array("d", values)

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
                "d", packed
            )


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
                self.assertGreater(device._cuda_training.predictions, 0, kind)

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
            engine._workspace((20, 160, 2), count, 2)
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
            + 2 * (16 * 20 + 24 + 8 * (2 * 25 + 2 * 20 + parameters))
            + 8 * (parameters + 1),
        )
        self.assertLess(
            workspace_bytes(sizes, 1, 1, False), workspace_bytes(sizes, 1, 1, True)
        )
        for maximum, memory in ((0, 1), (1, 0), (True, 1)):
            with self.assertRaises(ValueError):
                CudaTraining(maximum, memory, ReferenceLibrary())

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
