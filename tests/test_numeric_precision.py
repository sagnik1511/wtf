"""Storage precision, exact continuation, inheritance and lossless narrow encodings."""
import copy
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from ecosystem.binary_io import Encoder, Decoder
from ecosystem.brain import Experience
from ecosystem.brains import (create_brain, restore_brain, model_snapshot,
                              load_model_weights, mix_parent_brains)
from ecosystem.checkpoint_io import read_checkpoint
from ecosystem.numeric import Float32, packed, plain, precision_scope, precision_of
from ecosystem.runner import Experiment
from tests.test_advanced_algorithms import settings as advanced_settings
from tests.test_policy_algorithms import settings as policy_settings
from tests.test_ppo import ppo_settings
from tests.test_simulation import config

KINDS = ('q_network', 'ppo', 'a2c', 'reinforce', 'grpo', 'recurrent_ppo',
         'qrdqn', 'maskable_ppo', 'mappo')


def settings(kind):
    if kind == 'q_network':return next(iter(config()['brains'].values()))
    if kind == 'ppo':return ppo_settings()
    if kind in ('a2c', 'reinforce', 'grpo'):return policy_settings(kind)
    return advanced_settings(kind)


def trees(brain):
    if hasattr(brain, 'weights'):
        yield brain.weights
        yield brain.target
    else:
        yield brain.network.layers
        yield brain.moments
        yield brain.variances
        for name in ('reference_network', 'target_network'):
            if hasattr(brain, name):yield getattr(brain, name).layers


def leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():yield from leaves(value)
    elif tree and isinstance(tree[0], dict):
        for value in tree:yield from leaves(value)
    else:yield tree


class NumericPrecisionTests(unittest.TestCase):
    def test_all_algorithms_use_independent_four_byte_buffers_through_resume_and_birth(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                a = create_brain(settings(kind), ['rest', 'eat'], 7, 11)
                b = create_brain(settings(kind), ['rest', 'eat'], 13, 17)
                for tree in trees(a):
                    for values in leaves(tree):
                        self.assertIsInstance(values, Float32)
                        self.assertEqual(values.itemsize, 4)
                state = json.loads(json.dumps(a.checkpoint(), allow_nan=False))
                restored = restore_brain(state)
                # JSON changes RNG tuples to lists, so compare canonical JSON.
                self.assertEqual(json.dumps(restored.checkpoint()), json.dumps(a.checkpoint()))
                child = mix_parent_brains(a, b, 19)
                for original, inherited in zip(leaves(next(trees(a))), leaves(next(trees(child)))):
                    self.assertIsInstance(inherited, Float32)
                    self.assertIsNot(original, inherited)
                loaded = create_brain(settings(kind), ['rest', 'eat'], 21, 23)
                load_model_weights(loaded, model_snapshot(a))
                self.assertEqual(model_snapshot(a), model_snapshot(loaded))
                for values in leaves(next(trees(loaded))):self.assertIsInstance(values, Float32)

    def test_legacy_brains_restore_without_rounding_and_keep_float64_optimizer_state(self):
        for kind in KINDS:
            with self.subTest(kind=kind), precision_scope('float64'):
                original = create_brain(settings(kind), ['rest', 'eat'], 7, 11)
                # Include a value that float32 cannot preserve.
                next(leaves(next(trees(original))))[0] = 0.123456789012345
                state = original.checkpoint()
                state.pop('precision')
                restored = restore_brain(json.loads(json.dumps(state)))
                self.assertEqual(precision_of(restored), 'float64')
                self.assertEqual(model_snapshot(restored), model_snapshot(original))
                for tree in trees(restored):
                    for values in leaves(tree):self.assertIs(type(values), list)
                if hasattr(original, '_adam_step'):
                    gradients = original.network.zeros()
                    for layer in gradients:
                        for values in layer.values():
                            for i in range(len(values)):values[i] = .17
                    original._adam_step(gradients, 1)
                    restored._adam_step(gradients, 1)
                else:
                    feedback = Experience((.1,)*20, 'rest', 2., None, True)
                    original.learn(feedback);restored.learn(feedback)
                self.assertEqual(json.dumps(original.checkpoint()), json.dumps(restored.checkpoint()))

    def test_legacy_experiment_keeps_precision_in_later_fresh_runs(self):
        c = config();c['experiment'].update(runs=2, retain_brains_between_runs=False)
        c['world']['max_steps'] = 3
        a = Experiment(c)
        a.brain_precision = 'float64'
        # Reinitialize with legacy arithmetic before generating legacy state.
        a.brains = {};a.initial_weights = {};a.run_seeds = [];a._start_run()
        state = json.loads(json.dumps(a.checkpoint()))
        state.pop('brain_precision')
        for value in state['brains'].values():value.pop('precision')
        b = Experiment.restore(state)
        while not a.done:
            a.step();b.step()
            self.assertEqual(json.dumps(a.checkpoint(),sort_keys=True), json.dumps(b.checkpoint(),sort_keys=True))
        self.assertEqual(b.run_index, 1)
        for brain in b.brains.values():self.assertEqual(precision_of(brain), 'float64')

    def test_float32_checkpoint_formats_resume_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.json', '.json.gz', '.bin', '.bin.gz', '.bin.xz'):
                a = Experiment(config());a.step()
                path = Path(directory)/('state'+suffix)
                a.write_checkpoint(path)
                b = Experiment.restore(read_checkpoint(path))
                for _ in range(5):a.step();b.step()
                self.assertEqual(a.checkpoint(), b.checkpoint())
                for brain in b.brains.values():self.assertEqual(precision_of(brain), 'float32')

    def test_invalid_precision_and_unrepresentable_warm_start_are_rejected(self):
        with self.assertRaises(ValueError):packed([1.], 'float16')
        with self.assertRaises(ValueError):Float32([1e100])
        a = create_brain(settings('ppo'), ['rest', 'eat'], 7, 11)
        model = model_snapshot(a)
        model['parameters'][0]['weights'][0] = 1e100
        before = a.checkpoint()
        with self.assertRaises(ValueError):load_model_weights(a, model)
        self.assertEqual(a.checkpoint(), before)

    def test_payload_memory_is_four_bytes_per_weight(self):
        values = Float32(i/7 for i in range(1024))
        self.assertEqual(len(values)*values.itemsize, 4096)
        self.assertLess(sys.getsizeof(values), 8192)
        clone = copy.deepcopy(values)
        self.assertIsInstance(clone, Float32)
        clone[0] = .123456789
        self.assertEqual(values[0], 0.)
        self.assertEqual(clone[0], struct.unpack('<f', struct.pack('<f', .123456789))[0])
        self.assertIs(type(plain(values)), list)


class NarrowBinaryTests(unittest.TestCase):
    def test_exact_float32_arrays_and_f64_fallback(self):
        for values, tag in (([.5, -0., 1., -2.], 15), ([.1, .2, .3, .4], 10),
                            ([1e100]*4, 10), ([1e-100]*4, 10)):
            with self.subTest(values=values):
                encoded = Encoder().encode(values)
                self.assertEqual(encoded[0], tag)
                restored = Decoder().decode(encoded)
                self.assertEqual(restored, values)
                self.assertEqual([math.copysign(1,v) for v in restored],
                                 [math.copysign(1,v) for v in values])

    def test_integer_widths_and_boundaries_are_lossless(self):
        for values, tag in (([64, 127, 128, 255], 16), ([256, 32767, 32768, 65535], 17),
                            ([2**32-1]*4, 12), ([0, 1, 2, 3], 9),
                            ([-1, -128, 255, 65536], 9), ([2**32]*4, 9)):
            with self.subTest(values=values):
                encoded = Encoder().encode(values)
                self.assertEqual(encoded[0], tag)
                self.assertEqual(Decoder().decode(encoded), values)
        for values in ([2**64, -2**100], [True, False, 1, 2]):
            self.assertEqual(Decoder().decode(Encoder().encode(values)), values)

    def test_truncated_and_nonfinite_narrow_arrays_are_rejected(self):
        for raw in (b'\x0f\x04'+b'\0'*15, b'\x10\x04\0', b'\x11\x04\0',
                    b'\x0f\x04'+struct.pack('<4f', 1., 2., float('inf'), 4.)):
            with self.assertRaises(ValueError):Decoder().decode(raw)
