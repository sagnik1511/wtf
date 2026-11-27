"""Native observation plans and V6 projections preserve archived contracts."""

# The optional extension exposes private numerical probes for differential checks.
# pylint: disable=missing-class-docstring,missing-function-docstring,c-extension-no-member

import copy
from dataclasses import asdict
import random
import unittest

from tests.native_core_support import _engine, encoded
from wtf.simulation.compact_observations import compact_layout
from wtf.simulation.observations import raw_feature_names
from wtf.simulation.soft_observations import encode_soft, decode_soft, soft_layout


@unittest.skipUnless(
    _engine is not None and hasattr(_engine, "layout"), "Build native layout probes"
)
class NativeLayoutTests(unittest.TestCase):
    def templates(self):
        for case in range(24):
            template = {
                "observation_top_k": (1, 3)[case % 2],
                "movement_history_steps": (0, 3)[case % 2],
            }
            if case % 3:
                template["scent_observation"] = (
                    {"mode": "blended", "radius": 2}
                    if case % 3 == 1
                    else {
                        "species": ["grazer", "hunter"],
                        "separate_sexes": bool(case % 2),
                        "radius": 3,
                    }
                )
            if case % 2:
                template["social_observation"] = {
                    "history_steps": 4,
                    "max_neighbors": 8,
                }
            if case % 4:
                template.update(
                    consequence_observation=True,
                    consequence_scope=("context", "species_action", "both")[
                        case % 4 - 1
                    ],
                )
            if case % 5 == 0:
                template["soft_categories"] = {
                    "distance": [0.0, 0.2, 1.0],
                    "climb": [-1.0, -0.2, 0.0, 0.2, 1.0],
                }
            yield template

    def test_layout_names_widths_and_offsets_across_all_generations(self):
        for template in self.templates():
            for version in range(1, 8):
                with self.subTest(template=template, version=version):
                    config = dict(template, observation_version=version)
                    if version == 7:
                        config["observation_actions"] = [
                            "move_forward",
                            "turn_left",
                            "rest",
                            "eat",
                            "reproduce",
                        ]
                    actual = _engine.layout(config)
                    if version == 6:
                        expected = soft_layout(config)
                        self.assertEqual(actual["names"], list(expected.names))
                        self.assertEqual(
                            encoded(actual["fields"]),
                            encoded([asdict(field) for field in expected.fields]),
                        )
                        self.assertEqual(
                            encoded(actual["zeroes"]), encoded(expected.zeroes)
                        )
                        self.assertEqual(
                            actual["confidence_offsets"],
                            list(expected.confidence_offsets),
                        )
                    elif version == 7:
                        expected = compact_layout(config)
                        self.assertEqual(actual["names"], list(expected.names))
                        self.assertEqual(
                            actual["core_indices"], list(expected.core_indices)
                        )
                        self.assertEqual(
                            actual["physical_width"], expected.physical_width
                        )
                        self.assertEqual(
                            actual["consequence_start"], expected.consequence_start
                        )
                    else:
                        self.assertEqual(actual["names"], raw_feature_names(config))
                    self.assertEqual(actual["width"], len(actual["names"]))

    def test_soft_encoding_decoding_and_raw_types_are_exact(self):
        generator = random.Random(581)
        for template in self.templates():
            config = dict(template, observation_version=6)
            layout = soft_layout(config)
            for sample in range(8):
                raw = [generator.uniform(-1, 1) for _ in layout.raw_names]
                raw[2:4] = [-1, 0]
                for i, name in enumerate(layout.raw_names):
                    if name.endswith(" present"):
                        raw[i] = float(generator.randrange(2))
                    elif name.endswith(" climb"):
                        raw[i] = -0.0 if sample % 2 else 0.0
                expected = encode_soft(raw, config)
                actual = _engine.soft_encode(config, raw)
                self.assertEqual(encoded(actual), encoded(expected))
                for start in (0, layout.core + layout.tracking, len(layout.raw_names)):
                    self.assertEqual(
                        encoded(_engine.soft_decode(config, actual, start)),
                        encoded(decode_soft(expected, config, start)),
                    )
                copy_input = copy.deepcopy(actual)
                _engine.soft_decode(config, actual)
                self.assertEqual(encoded(actual), encoded(copy_input))


if __name__ == "__main__":
    unittest.main()
