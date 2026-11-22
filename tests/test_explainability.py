"""Archived inference parity, local derivative checks and bounded playback access."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wtf.learning.brains import create_brain, model_snapshot
from wtf.config import validate
from wtf.inspection.explainability import prepare_model, explain_model, feature_names
from wtf.inspection.explanations import ExplanationStore
from wtf.storage.library import export_champions
from wtf.storage.recording import Recorder, RecordingStore
from wtf.experiment.runner import Experiment
from tests.test_mutation import KINDS, environment


class InferenceExplanationTests(unittest.TestCase):
    def test_all_algorithms_match_actual_predictions_without_mutation(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = validate(environment(kind, 0))
                spec = next(iter(c["species"].values()))
                b = create_brain(
                    c["brains"][spec["brain"]], spec["allowed_actions"], 42, 123
                )
                x = [
                    ((i % 7) - 3) / 7
                    for i in range(
                        b.inputs
                        if kind == "q_network"
                        else (
                            b.network.inputs
                            if hasattr(b.network, "inputs")
                            else b.network.sizes[0]
                        )
                    )
                ]
                before = copy.deepcopy(b.checkpoint())
                expected = b.diagnostics(tuple(x))
                actual = explain_model(
                    prepare_model(model_snapshot(b)),
                    x,
                    mask=expected.get("action_mask"),
                )
                scores = expected.get(
                    "action_probabilities", expected.get("action_values")
                )
                for row in actual["outputs"]:
                    self.assertAlmostEqual(
                        row["score"], scores[row["action"]], places=11
                    )
                self.assertEqual(before, b.checkpoint())
                self.assertLessEqual(
                    len(actual["edges"]), 3 * 32 * len(actual["layers"])
                )
                self.assertTrue(
                    all(len(layer["nodes"]) <= 32 for layer in actual["layers"])
                )
                if kind == "qrdqn":
                    self.assertTrue(
                        all(" q" in n["label"] for n in actual["layers"][-1]["nodes"])
                    )
                    self.assertTrue(
                        any(n["target"] for n in actual["layers"][-1]["nodes"])
                    )
                self.assertEqual(
                    actual["recurrent_reset"], kind in ("recurrent_ppo", "lstm_ppo")
                )
                self.assertEqual(
                    actual["critic_separate"], kind in ("mappo", "trpo", "discrete_sac")
                )

    def test_gradients_match_finite_differences_for_all_algorithms(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                e = Experiment(environment(kind, 0))
                i = next(iter(e.brains))
                b = e.brains[i]
                model = prepare_model(model_snapshot(b))
                x = list(e.observations[i])
                r = explain_model(
                    model,
                    x,
                    mask=getattr(b, "mask", None) and dict(zip(b.actions, b.mask)),
                )
                action = r["target"]
                for feature in (0, 1, 7, len(x) - 1):
                    lo = x[:]
                    hi = x[:]
                    lo[feature] -= 1e-5
                    hi[feature] += 1e-5
                    # Mask is held fixed for the derivative, including on masked PPO.
                    mask = dict(zip(b.actions, b.mask)) if hasattr(b, "mask") else None
                    a = explain_model(model, lo, action, mask)
                    z = explain_model(model, hi, action, mask)
                    score = lambda v: next(
                        row["score"] for row in v["outputs"] if row["action"] == action
                    )
                    self.assertAlmostEqual(
                        (score(z) - score(a)) / 2e-5,
                        r["features"][feature]["gradient"],
                        places=7,
                    )

    def test_feature_names_and_masked_action_have_explicit_semantics(self):
        for version, width in ((1, 20), (2, 32), (3, 42), (4, 45)):
            self.assertEqual(
                len(
                    feature_names(
                        dict(observation_version=version, observation_top_k=3)
                    )
                ),
                width + 24,
            )
        e = Experiment(environment("maskable_ppo", 0))
        b = next(iter(e.brains.values()))
        model = prepare_model(model_snapshot(b))
        x = list(next(iter(e.observations.values())))
        mask = {a: True for a in b.actions}
        mask[b.actions[0]] = False
        result = explain_model(model, x, b.actions[0], mask)
        self.assertEqual(result["outputs"][0]["score"], 0)
        self.assertTrue(all(row["gradient"] == 0 for row in result["features"]))
        x[0] = float("nan")
        with self.assertRaises(ValueError):
            explain_model(model, x)


class ExplanationStoreTests(unittest.TestCase):
    def fixture(self, root):
        c = environment("ppo", 0)
        c["world"]["max_steps"] = 3
        c["recording"].update(
            encoding="binary",
            compression="xz",
            frames_per_chunk=2,
            diagnostics_every_steps=2,
        )
        e = Experiment(c)
        folder = root / "saved"
        r = Recorder(folder, e.config, "train")
        r.begin_run(e)
        animal = next(iter(e.world.animals))
        while not e.finished_run:
            e.step()
            r.capture(e, force=e.finished_run)
        r.end_run(e)
        r.close()
        export_champions(e, folder / "champions/run_0001.json")
        return e, animal, folder

    def test_bounded_context_and_shared_weight_cache_across_criteria_and_what_if(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            e, animal, _ = self.fixture(root)
            files_before = {
                p.relative_to(root): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            store = ExplanationStore(RecordingStore(root))
            from wtf.inspection.explanations import load_chunk, read_model

            with (
                patch(
                    "wtf.inspection.explanations.load_chunk", wraps=load_chunk
                ) as chunks,
                patch(
                    "wtf.inspection.explanations.read_model", wraps=read_model
                ) as models,
            ):
                first = store.explain("saved", 1, 0, animal, criterion="health")
                again = store.explain(
                    "saved", 1, 0, animal, criterion="health", feature=0, value=0.25
                )
                self.assertEqual(chunks.call_count, 1)
                self.assertEqual(models.call_count, 1)
                self.assertEqual(
                    first["features"][0]["value"],
                    e.config["species"][first["species"]]["body"]["initial_energy"]
                    / e.config["species"][first["species"]]["body"]["max_energy"],
                )
                self.assertEqual(again["features"][0]["value"], 0.25)
                self.assertIsNotNone(again["intervention"])
                for frame in (1, 2, 3):
                    store.explain("saved", 1, frame, animal)
            self.assertLessEqual(len(store.contexts), 2)
            self.assertLessEqual(len(store.models), 2)
            self.assertLessEqual(
                sum(m["parameters"] for m in store.models.values()), 1_000_000
            )
            self.assertNotIn("parameters", first["source"])
            json.dumps(first, allow_nan=False)
            self.assertEqual(
                files_before,
                {
                    p.relative_to(root): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                },
            )

    def test_missing_archives_invalid_queries_paths_and_absent_animals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, animal, run = self.fixture(root)
            store = ExplanationStore(RecordingStore(root))
            for kwargs in (
                dict(frame=-1),
                dict(animal=99999),
                dict(criterion="unknown"),
                dict(feature=0, value=float("nan")),
                dict(action="fly"),
            ):
                args = dict(recording_id="saved", run=1, frame=0, animal=animal)
                args.update(kwargs)
                with self.assertRaises(ValueError):
                    store.explain(**args)
            index = run / "champions/run_0001.json"
            data = json.loads(index.read_text())
            data["model_directory"] = "../../../outside"
            index.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                store.explain("saved", 1, 0, animal)
            index.unlink()
            with self.assertRaisesRegex(ValueError, "No saved champions"):
                store.explain("saved", 1, 0, animal)


class ExplanationAPITests(unittest.TestCase):
    def test_http_endpoint_and_read_only_boundary(self):
        import threading
        from http.server import ThreadingHTTPServer
        from urllib.error import HTTPError
        from urllib.request import urlopen
        from wtf.server import State, make_handler

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, animal, _ = ExplanationStoreTests().fixture(root)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(State(root)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}/api/recordings/saved/runs/1/explain/{animal}/0"
            try:
                with urlopen(base) as response:
                    result = json.load(response)
                self.assertEqual(result["animal"], animal)
                self.assertEqual(result["diagnostic_tick"], 0)
                with urlopen(base + "?feature=0&value=0.25") as response:
                    self.assertEqual(json.load(response)["features"][0]["value"], 0.25)
                for query in (
                    "?feature=0&value=nan",
                    "?criterion=../../x",
                    "?action=eat&action=rest",
                ):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base + query)
                    self.assertEqual(error.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_server_analysis_does_not_import_live_brains_or_world(self):
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import wtf.server, sys; assert not {'wtf.experiment.runner','wtf.learning.brains','wtf.learning.brain','wtf.simulation.world','wtf.learning.ppo'} & set(sys.modules)",
            ],
            check=True,
        )


class ExplanationCacheContractTests(unittest.TestCase):
    def test_blob_dedup_freshness_and_legacy_inline_weights(self):
        import os
        from wtf.storage.model_io import read_model, model_path

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, animal, run = ExplanationStoreTests().fixture(root)
            store = ExplanationStore(RecordingStore(root))
            first = store.explain("saved", 1, 0, animal, criterion="health")
            directory = run / "champions/run_0001" / first["species"] / first["sex"]
            health = directory / "health.json"
            food = directory / "food.json"
            artifact = json.loads(health.read_text())
            other = json.loads(food.read_text())
            other["model"] = artifact["model"]
            food.write_text(json.dumps(other))
            with patch(
                "wtf.inspection.explanations.read_model", wraps=read_model
            ) as read:
                store.explain("saved", 1, 0, animal, criterion="food")
                self.assertEqual(read.call_count, 0)
                blob = model_path(health, artifact["model"])
                stat = blob.stat()
                os.utime(blob, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
                store.explain("saved", 1, 0, animal, criterion="health")
                self.assertEqual(read.call_count, 1)
            inline = read_model(health, artifact["model"])
            artifact.update(brain_artifact_version=2, model=inline)
            health.write_text(json.dumps(artifact))
            result = store.explain("saved", 1, 1, animal, criterion="health")
            self.assertEqual(result["diagnostic_tick"], 0)
            self.assertEqual(result["outputs"], first["outputs"])
            self.assertEqual(store.explain("saved", 1, 2, animal)["diagnostic_tick"], 2)

    def test_deep_top_k_version_four_archives_match_all_nine_algorithms(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                c = environment(kind, 0)
                for b in c["brains"].values():
                    b.update(observation_version=4, observation_top_k=3)
                    if b["type"] != "q_network":
                        b.setdefault("network", {})["hidden_layers"] = [8, 4]
                e = Experiment(c)
                i = next(iter(e.brains))
                b = e.brains[i]
                x = tuple(e.observations[i])
                d = b.diagnostics(x)
                result = explain_model(
                    prepare_model(model_snapshot(b)), list(x), mask=d.get("action_mask")
                )
                scores = d.get("action_probabilities", d.get("action_values"))
                for row in result["outputs"]:
                    self.assertAlmostEqual(
                        row["score"], scores[row["action"]], places=11
                    )
