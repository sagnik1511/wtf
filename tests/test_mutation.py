"""Rare independent birth mutations, optimizer resets and lifecycle persistence."""

import copy
import base64
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wtf.learning.brains import create_brain, inherit_brain, model_snapshot
from wtf.storage.checkpoint_io import read_checkpoint
from wtf.config import validate, load
from wtf.learning.mutation import mutation_trial, mutate_weights, mutate_birth
from wtf.storage.recording import Recorder
from wtf.storage.replay_io import chunk_path, load_chunk, read_chunk_bytes
from wtf.experiment.runner import Experiment
from tests.reproduction_support import force_reproduction
from wtf.inspection.telemetry import summarize
from wtf.inspection.analyze import analyze_recording
from wtf.storage.library import export_champions
from tests.test_simulation import config
from tests.test_ppo import ppo_environment
from tests.test_policy_algorithms import environment as policy_environment
from tests.test_advanced_algorithms import environment as advanced_environment
from tests.test_reproduction import reproduction
from tests.new_policy_support import environment as new_policy_environment

KINDS = (
    "q_network",
    "ppo",
    "a2c",
    "reinforce",
    "grpo",
    "recurrent_ppo",
    "qrdqn",
    "maskable_ppo",
    "mappo",
    "discrete_sac",
    "trpo",
    "lstm_ppo",
)
SETTINGS = dict(
    enabled=True, probability_per_birth=1.0, max_changed_weights=3, noise_std=0.005
)


def environment(kind, probability=1.0):
    c = (
        config()
        if kind == "q_network"
        else (
            ppo_environment()
            if kind == "ppo"
            else (
                policy_environment(kind)
                if kind in ("a2c", "reinforce", "grpo")
                else (
                    new_policy_environment(kind)
                    if kind in ("discrete_sac", "trpo", "lstm_ppo")
                    else advanced_environment(kind)
                )
            )
        )
    )
    c["world"].update(width=5, height=5, max_steps=7)
    c["experiment"]["runs"] = 2
    c["obstacles"].update(generation="explicit", explicit_cells=[])
    for spec in c["species"].values():
        spec["reproduction"] = reproduction(health_threshold=0, max_population=7)
        spec["body"].update(living_cost=0, movement_cost=0)
        if spec["diet"] == "prey":
            spec["body"].update(attack_damage=0, attack_cost=0)
        spec["mutation"] = dict(SETTINGS, probability_per_birth=probability)
    return c


def matrices(brain):
    if brain.config["type"] == "q_network":
        return [brain.weights[k] for k in ("w1", "w2")]
    return [
        values
        for layer in brain.network.layers
        for key, values in layer.items()
        if key in ("weights", "input_weights", "recurrent_weights")
    ]


def biases(brain):
    if brain.config["type"] == "q_network":
        return [brain.weights[k] for k in ("b1", "b2")]
    return [
        values
        for layer in brain.network.layers
        for key, values in layer.items()
        if key == "bias"
    ]


class MutationMathTests(unittest.TestCase):
    def test_sparse_distinct_matrix_changes_and_parent_independence_for_all_algorithms(
        self,
    ):
        for kind in KINDS:
            e = force_reproduction(Experiment(environment(kind)))
            parent = e.brains[0]
            parent.updates = 7
            if hasattr(parent, "moments"):
                for layer in parent.moments:
                    for values in layer.values():
                        values[:] = type(values)([0.5] * len(values))
            before = copy.deepcopy(parent.checkpoint())
            child = inherit_brain(parent, 11)
            old = copy.deepcopy(matrices(child))
            old_bias = copy.deepcopy(biases(child))
            count = mutate_weights(child, SETTINGS, mutation_trial(SETTINGS, 11))
            changed = sum(
                a != b
                for left, right in zip(old, matrices(child))
                for a, b in zip(left, right)
            )
            self.assertEqual(changed, count, kind)
            self.assertTrue(1 <= count <= 3)
            self.assertEqual(old_bias, biases(child), kind)
            self.assertEqual(before, parent.checkpoint(), kind)
            self.assertEqual(child.updates, 0)
            self.assertEqual(child.steps, 0)
            if hasattr(child, "moments"):
                self.assertTrue(
                    all(
                        v == 0
                        for layer in child.moments
                        for values in layer.values()
                        for v in values
                    )
                )
                self.assertTrue(
                    all(
                        v == 0
                        for layer in child.variances
                        for values in layer.values()
                        for v in values
                    )
                )
            if hasattr(child, "target"):
                self.assertEqual(child.target, child.weights)
            if hasattr(child, "target_network"):
                self.assertEqual(child.target_network.layers, child.network.layers)
            if hasattr(child, "reference_network"):
                self.assertEqual(child.reference_network.layers, child.network.layers)
            if kind == "recurrent_ppo":
                self.assertEqual(child.hidden, [0.0] * child.config["recurrent_size"])
            if hasattr(child, "last_group_step"):
                self.assertEqual(child.last_group_step, 0)

    def test_disabled_zero_probability_and_zero_noise_skip_draws(self):
        for change in (
            {"enabled": False},
            {"probability_per_birth": 0},
            {"noise_std": 0},
        ):
            with patch(
                "wtf.learning.mutation.random.Random",
                side_effect=AssertionError("unexpected RNG"),
            ):
                self.assertIsNone(mutation_trial(dict(SETTINGS, **change), 2))
        self.assertIsNone(mutation_trial(None, 2))

    def test_deterministic_seed_and_independent_streams(self):
        e = force_reproduction(Experiment(environment("ppo")))
        parent = e.brains[0]
        original = parent.checkpoint()
        world_rng = e.world.rng.getstate()
        a, b = inherit_brain(parent, 4), inherit_brain(parent, 4)
        mutate_weights(a, SETTINGS, mutation_trial(SETTINGS, 4))
        mutate_weights(b, SETTINGS, mutation_trial(SETTINGS, 4))
        self.assertEqual(a.checkpoint(), b.checkpoint())
        self.assertEqual(parent.checkpoint(), original)
        self.assertEqual(e.world.rng.getstate(), world_rng)
        other = inherit_brain(parent, 5)
        mutate_weights(other, SETTINGS, mutation_trial(SETTINGS, 5))
        self.assertNotEqual(model_snapshot(a), model_snapshot(other))

    def test_zero_effect_draw_does_not_reset_optimizer(self):
        e = force_reproduction(Experiment(environment("ppo")))
        b = e.brains[0]
        b.updates = 7
        rng = random.Random(1)
        with patch.object(rng, "gauss", return_value=0):
            self.assertEqual(mutate_weights(b, SETTINGS, rng), 0)
        self.assertEqual(b.updates, 7)

    def test_nontrigger_does_not_traverse_weights(self):
        settings = dict(SETTINGS, probability_per_birth=0.0001)
        with patch(
            "wtf.learning.mutation.mutate_weights",
            side_effect=AssertionError("model traversal"),
        ):
            e = force_reproduction(Experiment(environment("ppo", probability=0)))
            e.step()
            self.assertEqual(dict(e.world.mutations), {})


class MutationLifecycleTests(unittest.TestCase):
    def test_birth_metadata_and_update_accounting(self):
        for kind in KINDS:
            e = force_reproduction(Experiment(environment(kind)))
            for b in e.brains.values():
                b.updates = 7
            e.step()
            children = [a for a in e.world.animals.values() if a.parent_id >= 0]
            self.assertTrue(children, kind)
            self.assertEqual(sum(e.world.mutations.values()), len(children), kind)
            self.assertEqual(
                sum(x.kind == "mutated" for x in e.world.events), len(children)
            )
            for child in children:
                self.assertTrue(child.mutated)
                self.assertEqual(child.mutation_count, 1)
                self.assertTrue(1 <= child.mutation_weights <= 3)
                self.assertEqual(e.inherited_updates[child.id], 0)
            self.assertEqual(
                e.total_updates,
                sum(
                    b.updates
                    for i, b in e.brains.items()
                    if i not in e.inherited_updates
                ),
            )
            previous = e.total_updates
            for _ in range(5):
                e.step()
                self.assertGreaterEqual(e.total_updates, previous, kind)
                previous = e.total_updates

    def test_mutation_lineage_persists_for_unmutated_grandchild(self):
        e = force_reproduction(Experiment(environment("ppo")))
        e.step()
        parent = next(a for a in e.world.animals.values() if a.parent_id >= 0)
        e.config["species"][parent.species]["mutation"]["probability_per_birth"] = 0
        e.world.config["species"][parent.species]["mutation"][
            "probability_per_birth"
        ] = 0
        parent.sex = "female"
        parent.age = 10
        parent.last_reproduced_age = -1
        parent.x, parent.y = 1, 1
        mate = e.world.animals[parent.second_parent_id]
        mate.sex = "male"
        mate.age = 10
        mate.last_reproduced_age = -1
        mate.x, mate.y = 2, 1
        for other in e.world.animals.values():
            if other.id not in (parent.id, mate.id):
                other.last_reproduced_age = other.age
        e.observations = e.world.observe_all()
        with patch.object(
            type(e.brains[parent.id]), "choose", return_value="reproduce"
        ):
            e.step()
        descendants = [a for a in e.world.animals.values() if a.parent_id == parent.id]
        self.assertTrue(descendants)
        self.assertTrue(
            all(
                not a.mutated and a.mutation_count == 1 and a.mutation_weights == 0
                for a in descendants
            )
        )

    def test_evaluation_and_random_baselines_keep_newborn_weights_frozen(self):
        for mode in ("evaluate", "random"):
            e = force_reproduction(Experiment(environment("ppo"), mode=mode))
            with patch.object(type(e.brains[0]), "choose", return_value="reproduce"):
                e.step()
            self.assertTrue(any(a.parent_id >= 0 for a in e.world.animals.values()))
            self.assertEqual(dict(e.world.mutations), {})
            for child in e.world.animals.values():
                if child.parent_id >= 0:
                    self.assertFalse(child.mutated)
                    from wtf.learning.brains import mix_parent_brains, offspring_seed

                    expected = mix_parent_brains(
                        e.brains[child.parent_id],
                        e.brains[child.second_parent_id],
                        offspring_seed(e.world, child.id),
                    )
                    self.assertEqual(
                        model_snapshot(e.brains[child.id]), model_snapshot(expected)
                    )

    def test_json_and_binary_resume_preserve_future_mutations_for_every_algorithm(self):
        for kind in KINDS:
            a = force_reproduction(Experiment(environment(kind)))
            a.step()
            a.step()
            b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "checkpoint.bin.xz"
                a.write_checkpoint(p)
                c = Experiment.restore(read_checkpoint(p))
            while not a.done:
                a.step()
                b.step()
                c.step()
            self.assertEqual(a.checkpoint(), b.checkpoint(), kind)
            self.assertEqual(a.checkpoint(), c.checkpoint(), kind)

    def test_zero_probability_matches_missing_config_and_physical_rng(self):
        a = force_reproduction(Experiment(environment("ppo", 0)))
        cfg = environment("ppo", 0)
        for s in cfg["species"].values():
            s.pop("mutation")
        b = force_reproduction(Experiment(cfg))
        while not a.done:
            a.step()
            b.step()
            self.assertEqual(a.world.snapshot(), b.world.snapshot())
            self.assertEqual(a.world.rng.getstate(), b.world.rng.getstate())
            self.assertEqual(
                {i: x.checkpoint() for i, x in a.brains.items()},
                {i: x.checkpoint() for i, x in b.brains.items()},
            )

    def test_trial_birth_mutations_do_not_change_real_world_or_other_policies(self):
        from wtf.learning.grouped_rollouts import collect_group

        e = force_reproduction(Experiment(environment("grpo")))
        focal = next(i for i, a in e.world.animals.items() if a.species == "predator")
        before = e.world.checkpoint()
        other = {i: b.checkpoint() for i, b in e.brains.items() if i != focal}
        observed = []

        def capture(world, birth, parent, child_brain=None):
            policy, event = mutate_birth(world, birth, parent, child_brain)
            if event:
                observed.append((world.animals[birth.target].mutated, event.amount))
            return policy, event

        with patch("wtf.learning.mutation.mutate_birth", side_effect=capture):
            collect_group(e.world, e.brains, focal)
        self.assertTrue(observed)
        self.assertTrue(all(flag and 1 <= n <= 3 for flag, n in observed))
        self.assertEqual(before, e.world.checkpoint())
        self.assertEqual(
            other, {i: b.checkpoint() for i, b in e.brains.items() if i != focal}
        )

    def test_binary_replay_analysis_telemetry_and_champion_provenance(self):
        e = force_reproduction(Experiment(environment("ppo")))
        e.config["recording"].update(
            encoding="binary", compression="xz", frames_per_chunk=4
        )
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(d, e.config, "train")
            rec.begin_run(e)
            e.step()
            rec.capture(e)
            rec.end_run(e)
            rec.close(False)
            f = load_chunk(chunk_path(Path(d) / "playback", 1, 0))
            self.assertTrue(any(a.get("mutated") for a in f[-1]["animals"]))
            script = """import assert from 'node:assert/strict';
            import {decodeBinaryReplay} from './viewer/binary-codec.mjs';
            import {decodeChunk} from './viewer/replay-codec.mjs';
            let input='';for await(const data of process.stdin)input+=data;
            const d=JSON.parse(input),frames=decodeChunk(decodeBinaryReplay(Buffer.from(d.bytes,'base64')));
            assert.deepStrictEqual(frames,d.expected);
            assert(frames.at(-1).animals.some(a=>a.mutated&&a.mutation_count===1&&a.mutation_weights>0));
            assert(frames.at(-1).events.some(e=>e.kind==='mutated'));
            """
            data = dict(
                bytes=base64.b64encode(
                    read_chunk_bytes(chunk_path(Path(d) / "playback", 1, 0))
                ).decode(),
                expected=f,
            )
            result = subprocess.run(
                ["node", "--input-type=module", "-e", script],
                input=json.dumps(data),
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                sum(
                    s["mutations"]
                    for s in summarize(e.config, f[-1], f[-2])["species"].values()
                ),
                sum(e.world.mutations.values()),
            )
            report = analyze_recording(d)["runs"][0]["species"]
            self.assertEqual(
                sum(s["mutations"] for s in report.values()),
                sum(e.world.mutations.values()),
            )
            child = next(a for a in e.world.animals.values() if a.mutated)
            child.age = 100
            e.champions.consider([child], e.brains, e._champion_source())
            archive = export_champions(e, Path(d) / "champions/run_0001.json")
            saved = json.loads(
                (archive / child.species / child.sex / "lifespan.json").read_text()
            )
            self.assertTrue(saved["source"]["mutated"])
            self.assertEqual(saved["source"]["mutation_count"], 1)

    def test_cli_stores_mutation_counts_in_readable_metrics_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "environment.json"
            path.write_text(json.dumps(environment("ppo")))
            seed = force_reproduction(Experiment(environment("ppo")))
            export_champions(seed, root / "source/champions/run_0001.json")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wtf.cli",
                    str(path),
                    "--runs",
                    "1",
                    "--brains-from",
                    str(root / "source"),
                    "--no-progress",
                    "--output",
                    str(root / "results"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = next((root / "results").iterdir())
            with (output / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertIn("prey_mutations", rows[0])
            self.assertIn("predator_mutations", rows[0])
            total = sum(int(rows[-1][s + "_mutations"]) for s in ("prey", "predator"))
            self.assertGreater(total, 0)
            self.assertEqual(
                sum(
                    json.loads((output / "summary.json").read_text())[0][
                        "mutations"
                    ].values()
                ),
                total,
            )

    def test_legacy_checkpoint_missing_mutation_fields_restores(self):
        e = force_reproduction(Experiment(environment("ppo", 0)))
        state = json.loads(json.dumps(e.checkpoint()))
        for spec in state["config"]["species"].values():
            spec.pop("mutation", None)
        for spec in state["world"]["config"]["species"].values():
            spec.pop("mutation", None)
        state["world"]["snapshot"]["metrics"].pop("mutations")
        for a in state["world"]["snapshot"]["animals"]:
            for k in ("mutated", "mutation_count", "mutation_weights"):
                a.pop(k)
        restored = Experiment.restore(state)
        restored.step()
        self.assertEqual(dict(restored.world.mutations), {})

    def test_config_validation_and_presets(self):
        for key, value in [
            ("enabled", 1),
            ("probability_per_birth", -0.1),
            ("probability_per_birth", 1.1),
            ("max_changed_weights", 0),
            ("max_changed_weights", 1.5),
            ("noise_std", float("nan")),
            ("noise_std", -0.1),
        ]:
            c = environment("ppo")
            c["species"]["prey"]["mutation"][key] = value
            with self.assertRaisesRegex(ValueError, key):
                validate(c)
        for p in (Path(__file__).resolve().parents[1] / "environments").glob("*.json"):
            c = load(p)
            self.assertTrue(
                all(
                    s["mutation"]["probability_per_birth"] == 0.0001
                    for s in c["species"].values()
                )
            )
