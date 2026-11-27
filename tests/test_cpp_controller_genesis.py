"""Native policy genesis matches every initialized weight, bias and RNG stream."""

# Fixtures deliberately inspect the optional compiled boundary and raw checkpoint.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# pylint: disable=c-extension-no-member
# Constructor matrices keep all initialization contracts in one comparable fixture.
# pylint: disable=too-many-locals,too-many-nested-blocks
import copy
import unittest
from tests.native_core_support import _engine, encoded, NATIVE_KINDS as KINDS
from wtf.config import load
from wtf.learning.brains import create_brain


def words(seed):
    return [
        (seed >> (32 * i)) & 0xFFFFFFFF
        for i in range(max(1, (seed.bit_length() + 31) // 32))
    ]


@unittest.skipUnless(_engine is not None, "Build the optional native episode engine")
class NativeControllerGenesisTests(unittest.TestCase):
    def test_all_algorithms_depths_observation_versions_and_wide_seeds(self):
        for kind in KINDS:
            environment = load("baseline" if kind == "q_network" else "ppo")
            template = copy.deepcopy(next(iter(environment["brains"].values())))
            template["type"] = kind
            for depth in ([3], [3, 4, 2], []):
                if kind == "q_network" and not depth:
                    continue
                for version in range(1, 8):
                    for seed in (0, 2**96 + 123456789):
                        with self.subTest(
                            kind=kind, depth=depth, version=version, seed=seed
                        ):
                            config = copy.deepcopy(template)
                            (config if kind == "q_network" else config["network"])[
                                "hidden_layers"
                            ] = depth
                            config["observation_version"] = version
                            config.setdefault("recurrent_size", 3)
                            config.setdefault("critic_hidden_layers", [2])
                            config.setdefault("num_quantiles", 3)
                            config.setdefault("exploration", {"initial": 1})
                            config.setdefault("group_size", 2)
                            config.setdefault("rollout_horizon", 2)
                            if kind in ("ppo", "maskable_ppo") and version == 7:
                                config["social_observation"] = {}
                                config["consequence_scope"] = "both"
                                config["late_consequence"] = {"bound": 1}
                                config["auxiliary_consequence"] = {"coefficient": 0.1}
                            actions = (
                                config.get("observation_actions")
                                or next(iter(environment["species"].values()))[
                                    "allowed_actions"
                                ]
                            )
                            expected = create_brain(
                                config, actions, seed, seed + 456
                            ).checkpoint()
                            actual = _engine.controller_create_state(
                                config, actions, words(seed), words(seed + 456)
                            )
                            self.assertEqual(encoded(expected), encoded(actual))
