"""Exact native episode genesis, model contracts and cross-run inheritance."""

# Lifecycle fixtures intentionally inspect persisted internal ownership and state.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring
import copy
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from tests.native_core_support import _engine
from tests.test_cpp_value_policies import mismatch as difference
from tests.test_mutation import environment, KINDS
from tests.test_social_learning import config as social_config
from wtf.config import validate
from wtf.experiment.runner import Experiment
from wtf.learning.brains import model_signature, model_snapshot

native = (
    SimpleNamespace(
        start=getattr(_engine, "start_native_run"),
        signature=getattr(_engine, "model_signature"),
    )
    if _engine is not None
    and all(hasattr(_engine, name) for name in ("start_native_run", "model_signature"))
    else None
)


def encoded(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)


@unittest.skipUnless(native is not None, "Build the optional native genesis probes")
class RunGenesisTests(unittest.TestCase):
    def same(self, e, payload):
        expected = e.checkpoint()
        actual = payload["checkpoint"]
        self.assertTrue(
            encoded(expected) == encoded(actual), str(difference(expected, actual))
        )
        for animal_id, brain in e.brains.items():
            self.assertEqual(
                encoded(model_signature(brain.config, brain.actions)),
                encoded(payload["signatures"][str(animal_id)]),
            )
        self.assertEqual(
            encoded({str(i): o for i, o in e.observations.items()}),
            encoded(payload["observations"]),
        )

    def test_initial_all_algorithms_and_modes(self):
        for kind in KINDS:
            for mode in ("train", "evaluate", "random"):
                with self.subTest(kind=kind, mode=mode):
                    c = validate(environment(kind))
                    c["experiment"]["world_seed_policy"] = "fixed"
                    e = Experiment(c, mode)
                    self.same(
                        e,
                        native.start({"config": c, "mode": mode, "initial_models": {}}),
                    )

    def test_next_run_modern_chaining_and_retention(self):
        for kind in (
            "q_network",
            "ppo",
            "dqn",
            "ddqn",
            "discrete_sac",
            "grpo",
            "recurrent_ppo",
            "lstm_ppo",
            "maskable_ppo",
            "mappo",
        ):
            for chain in (False, True):
                for improve in (False, True):
                    with self.subTest(kind=kind, chain=chain, improve=improve):
                        c = validate(environment(kind))
                        c["experiment"].update(
                            runs=3,
                            chain_best_brains=chain,
                            promote_only_improved=improve,
                            world_seed_policy="increment",
                        )
                        e = Experiment(c)
                        for _ in range(2):
                            while not e.finished_run:
                                e.step()
                            checkpoint = e.checkpoint()
                            checkpoint["run_index"] += 1
                            e.next_run()
                            self.same(e, native.start(checkpoint))

    def test_random_seeds_and_early_stopping(self):
        for mode in ("train", "evaluate"):
            c = validate(environment("ppo"))
            c["experiment"].update(
                runs=3, world_seed_policy="random", early_stopping={"patience_runs": 2}
            )
            c = validate(c)
            with patch(
                "wtf.experiment.runner.secrets.randbits", return_value=0xF1234567ABCDEF
            ):
                e = Experiment(c, mode)
            self.same(
                e,
                native.start(
                    {
                        "config": c,
                        "mode": mode,
                        "initial_models": {},
                        "randomization_seed": e.randomization_seed,
                    }
                ),
            )
            while not e.finished_run:
                e.step()
            state = e.checkpoint()
            state["run_index"] += 1
            e.next_run()
            self.same(e, native.start(state))

    def test_shared_knowledge_crosses_episode(self):
        for schedule in ("continuous", "episode"):
            c = social_config()
            c.pop("terrain", None)
            for spec in c["species"].values():
                if "swim" in spec["allowed_actions"]:
                    spec["allowed_actions"].remove("swim")
            c["experiment"]["runs"] = 3
            for brain in c["brains"].values():
                brain["observation_version"] = 1
            c["consequence_learning"].update(
                update_schedule=schedule, persistent_action_totals=True
            )
            c = validate(c)
            e = Experiment(c)
            self.same(
                e, native.start({"config": c, "mode": "train", "initial_models": {}})
            )
            while not e.finished_run:
                e.step()
            state = e.checkpoint()
            state["run_index"] += 1
            e.next_run()
            self.same(e, native.start(state))

    def test_model_signatures_all_versions_and_optional_contracts(self):
        for kind in KINDS:
            c = validate(environment(kind))
            spec = next(iter(c["species"].values()))
            base = c["brains"][spec["brain"]]
            for version in range(1, 8):
                b = copy.deepcopy(base)
                b.update(
                    observation_version=version,
                    observation_top_k=3,
                    movement_history_steps=2,
                    scent_observation={"mode": "blended", "radius": 3},
                    social_observation={"history_steps": 3, "max_neighbors": 5},
                    consequence_observation=True,
                    consequence_scope="both",
                )
                if version == 7:
                    b["observation_actions"] = list(spec["allowed_actions"])
                if version == 6:
                    b["soft_categories"] = {"level": [0, 0.6, 1]}
                with self.subTest(kind=kind, version=version):
                    self.assertEqual(
                        encoded(model_signature(b, spec["allowed_actions"])),
                        encoded(native.signature(b, spec["allowed_actions"])),
                    )
        for kind in ("ppo", "maskable_ppo"):
            c = validate(environment(kind))
            spec = next(iter(c["species"].values()))
            b = c["brains"][spec["brain"]]
            b.update(
                observation_version=7,
                observation_actions=list(spec["allowed_actions"]),
                social_observation={"history_steps": 2, "max_neighbors": 4},
                consequence_observation=True,
                consequence_scope="both",
                late_consequence={"max_adjustment": 1.5},
                auxiliary_consequence={},
            )
            self.assertEqual(
                encoded(model_signature(b, spec["allowed_actions"])),
                encoded(native.signature(b, spec["allowed_actions"])),
            )

    def test_warm_starts_reset_optimizer_and_synchronize_targets(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = validate(environment(kind))
                source = Experiment(c)
                models = {}
                for a in source.world.animals.values():
                    model = model_snapshot(source.brains[a.id])
                    parameters = model["parameters"]
                    values = (
                        parameters["w1"]
                        if kind == "q_network"
                        else parameters[0].get(
                            "weights", parameters[0].get("input_weights")
                        )
                    )
                    values[0] += 0.0125
                    models.setdefault(a.species, {})[a.sex] = {
                        "model": model,
                        "provenance": {
                            "sex": a.sex,
                            "criterion": "lifespan",
                            "rank": [1, 2, 3, 4],
                            "model_generation": 2,
                        },
                    }
                with patch(
                    "wtf.experiment.runner.load_initial_models",
                    return_value=copy.deepcopy(models),
                ):
                    e = Experiment(c)
                self.same(
                    e,
                    native.start(
                        {"config": c, "mode": "train", "initial_models": models}
                    ),
                )

    def test_legacy_live_founder_retention(self):
        for retained in (True, False):
            c = validate(environment("ppo"))
            c["experiment"].update(runs=3, retain_brains_between_runs=retained)
            e = Experiment(c)
            e.legacy_brain_retention = True
            while not e.finished_run:
                e.step()
            state = e.checkpoint()
            state["run_index"] += 1
            e.next_run()
            self.same(e, native.start(state))

    def test_empty_species_single_sex_and_maximum_seed(self):
        for counts in ((0, 1), (1, 0), (1, 2), (2, 1)):
            for policy in ("fixed", "increment", "random"):
                with self.subTest(counts=counts, policy=policy):
                    c = validate(environment("ppo"))
                    c["seed"] = 2**32 - 1
                    c["experiment"].update(runs=3, world_seed_policy=policy)
                    for spec, n in zip(c["species"].values(), counts):
                        spec["initial_population"] = n
                    c = validate(c)
                    with patch(
                        "wtf.experiment.runner.secrets.randbits", return_value=2**64 - 1
                    ):
                        e = Experiment(c)
                    self.same(
                        e,
                        native.start(
                            {
                                "config": c,
                                "mode": "train",
                                "initial_models": {},
                                "randomization_seed": 2**64 - 1,
                            }
                        ),
                    )
                    state = e.checkpoint()
                    state["run_index"] = 1
                    e.run_index = 1
                    e._start_run()
                    self.same(e, native.start(state))

    def test_legacy_metadata_defaults_and_sex_migrations(self):
        c = validate(environment("ppo"))
        c["experiment"]["runs"] = 3
        e = Experiment(c)
        while not e.finished_run:
            e.step()
        state = e.checkpoint()
        for key in (
            "initial_models",
            "run_initial_models",
            "initial_weights",
            "champion_artifacts",
            "guard_lineage_chaining",
            "improvement_only",
            "legacy_brain_retention",
            "freeze_evaluation_brains",
            "retired_updates",
            "inherited_updates",
            "randomization_seed",
            "seed_rng_state",
            "early_stopping",
        ):
            state.pop(key, None)
        source = Experiment.restore(state)
        source.run_index += 1
        source._start_run()
        state["run_index"] += 1
        self.same(source, native.start(state))
        state = e.checkpoint()
        for name, groups in list(state["initial_weights"].items()):
            state["initial_weights"][name] = next(iter(groups.values()))
        source = Experiment.restore(state)
        source.run_index += 1
        source._start_run()
        state["run_index"] += 1
        self.same(source, native.start(state))

    def test_promotion_lineage_guard_and_artifact_provenance(self):
        for lineage in (False, True):
            for artifact_matches in (False, True):
                c = validate(environment("ppo"))
                c["experiment"].update(
                    runs=4, chain_best_brains=True, promote_only_improved=True
                )
                metric = "lineage" if lineage else "lifespan"
                for spec in c["species"].values():
                    spec["champion_selection"] = {"criterion": metric}
                c = validate(c)
                e = Experiment(c)
                while not e.finished_run:
                    e.step()
                e.next_run()
                for name, groups in e.run_initial_models.items():
                    for sex, previous in groups.items():
                        candidate = e.champions.entries[name][sex][metric]
                        changed = list(candidate["rank"])
                        changed[0] += 1
                        candidate["rank"] = tuple(changed)
                        previous["provenance"]["lineage_qualified"] = lineage
                        if lineage:
                            candidate["lineage"]["qualified"] = False
                        e.champion_artifacts.setdefault(name, {}).setdefault(sex, {})[
                            metric
                        ] = {
                            "artifact_id": "known",
                            "path": "brains/test.model",
                            "criterion": metric,
                            "rank": changed if artifact_matches else [99],
                            "sex": sex,
                            "parent_artifact": "discard",
                        }
                state = e.checkpoint()
                state["run_index"] += 1
                e.run_index += 1
                e._start_run()
                self.same(e, native.start(state))

    def test_invalid_warm_model_and_random_checkpoint_rejected(self):
        for kind in ("q_network", "ppo", "discrete_sac"):
            c = validate(environment(kind))
            e = Experiment(c)
            a = next(iter(e.world.animals.values()))
            base = model_snapshot(e.brains[a.id])
            for case in ("signature", "dimensions", "nan", "overflow", "boolean"):
                model = copy.deepcopy(base)
                if case == "signature":
                    model["signature"]["algorithm"] = "invalid"
                else:
                    values = (
                        model["parameters"]["w1"]
                        if kind == "q_network"
                        else model["parameters"][0]["weights"]
                    )
                    if case == "dimensions":
                        values.pop()
                    else:
                        values[0] = {
                            "nan": float("nan"),
                            "overflow": 1e300,
                            "boolean": True,
                        }[case]
                models = {
                    a.species: {
                        a.sex: {
                            "model": model,
                            "provenance": {"criterion": "lifespan", "sex": a.sex},
                        }
                    }
                }
                with patch(
                    "wtf.experiment.runner.load_initial_models",
                    return_value=copy.deepcopy(models),
                ):
                    with self.assertRaises((ValueError, OverflowError)):
                        Experiment(c)
                with self.assertRaises(ValueError):
                    native.start(
                        {"config": c, "mode": "train", "initial_models": models}
                    )
        c = validate(environment("ppo"))
        c["experiment"]["world_seed_policy"] = "random"
        with patch("wtf.experiment.runner.secrets.randbits", return_value=7):
            e = Experiment(c)
        state = e.checkpoint()
        state["seed_rng_state"] = None
        with self.assertRaises(ValueError):
            Experiment.restore(state)
        with self.assertRaises(ValueError):
            native.start(state)


if __name__ == "__main__":
    unittest.main()
