"""Lossless layout changes, bounded chunks, and cross-language reconstruction."""

import copy
import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from wtf.storage.replay_codec import encode_chunk, decode_chunk
from wtf.storage.replay_io import load_chunk, load_chunk_payload
from wtf.storage.compress_recordings import compress_recording, frame_digest
from wtf.storage.recording import Recorder
from wtf.experiment.runner import Experiment
from tests.test_simulation import config


def frames():
    first = dict(
        step=0,
        animals=[dict(id=1, species="prey", x=1, y=2, health=5.0, age=0)],
        resources=[dict(x=0, y=0, quantity=1), dict(x=2, y=3, quantity=3.0)],
        metrics=dict(food=4, deaths=0),
        diagnostics={"1": {"observation": [1, 2]}},
    )
    second = copy.deepcopy(first)
    second.update(step=1)
    second.pop("diagnostics")
    second["animals"][0].update(x=2, age=1, health=-0.0)
    second["animals"].append(dict(id=2, species="prey", x=3, y=4, health=5, age=0))
    second["resources"][0]["quantity"] = 1.0
    second["resources"].reverse()
    second["metrics"].update(food=4.0, births=1)
    third = copy.deepcopy(second)
    third.update(step=2, diagnostics={})
    third["animals"].reverse()
    third["animals"][1]["extension"] = {"nested": [True, 0.0]}
    third["metrics"].pop("deaths")
    third["resources"].pop()
    fourth = copy.deepcopy(third)
    fourth.update(step=3)
    fourth["animals"].pop()
    fourth["resources"].append(dict(x=5, y=6, quantity=2))
    fifth = copy.deepcopy(fourth)
    fifth.pop("animals")
    fifth.pop("metrics")
    sixth = copy.deepcopy(first)
    sixth["step"] = 5
    return [first, second, third, fourth, fifth, sixth]


class CodecTests(unittest.TestCase):
    def test_exact_roundtrip_birth_death_reordering_extensions_sparse_diagnostics(self):
        original = frames()
        encoded = encode_chunk(original)
        decoded = decode_chunk(json.loads(json.dumps(encoded)))
        self.assertEqual(frame_digest(decoded), frame_digest(original))
        self.assertIn("diagnostics", encoded["frames"][1]["drop"])
        self.assertIn("update", encoded["frames"][1]["animals"])
        self.assertIn("order", encoded["frames"][2]["animals"])
        decoded[0]["animals"][0]["health"] = 999
        self.assertEqual(decoded[1]["animals"][0]["health"], -0.0)
        self.assertEqual(original[0]["animals"][0]["health"], 5.0)

    def test_legacy_arrays_and_independent_chunk_baselines(self):
        original = frames()
        self.assertEqual(decode_chunk(original), original)
        for i in range(0, len(original), 2):
            part = original[i : i + 2]
            encoded = encode_chunk(part)
            self.assertEqual(encoded["frames"][0], part[0])
            self.assertEqual(frame_digest(decode_chunk(encoded)), frame_digest(part))

    def test_bad_codec_masks_and_order_rejected(self):
        with self.assertRaises(ValueError):
            decode_chunk({"codec": "unknown"})
        for mutation in [
            lambda p: p["frames"][1]["animals"]["update"][0].__setitem__(1, -1),
            lambda p: p["frames"][1]["animals"].__setitem__("order", [1, 1]),
            lambda p: p.__setitem__("animal_fields", ["x", "x"]),
        ]:
            payload = encode_chunk(frames())
            mutation(payload)
            with self.assertRaises((ValueError, KeyError)):
                decode_chunk(payload)

    def test_javascript_decodes_python_packets(self):
        payload = encode_chunk(frames())
        script = """import {decodeChunk} from './viewer/replay-codec.mjs';
        import assert from 'node:assert/strict';
        let input='';for await(const data of process.stdin)input+=data;
        const {payload,expected}=JSON.parse(input),decoded=decodeChunk(payload);
        assert.deepStrictEqual(decoded,expected);
        decoded[0].animals[0].health=999;
        assert(Object.is(decoded[1].animals[0].health,-0));
        """
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            input=json.dumps(dict(payload=payload, expected=frames())),
            text=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_recorder_snapshot_and_delta_are_exact_and_training_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for encoding in ("snapshot", "delta"):
                c = config()
                c["world"]["max_steps"] = 9
                c["recording"].update(encoding=encoding, frames_per_chunk=3)
                e = Experiment(c)
                r = Recorder(Path(directory) / encoding, c, "train")
                r.begin_run(e)
                while not e.done:
                    e.step()
                    r.capture(e, force=e.finished_run)
                r.end_run(e)
                r.close()
                chunks = sorted(
                    (Path(directory) / encoding / "playback").glob("*chunk*.json.gz")
                )
                results.append(
                    ([frame for p in chunks for frame in load_chunk(p)], e.checkpoint())
                )
                for path in chunks:
                    packet = load_chunk_payload(path)
                    self.assertIsInstance(packet, dict if encoding == "delta" else list)
            self.assertEqual(frame_digest(results[0][0]), frame_digest(results[1][0]))
            for _, checkpoint in results:
                checkpoint["config"]["recording"].pop("encoding")
                checkpoint["world"]["config"]["recording"].pop("encoding")
            self.assertEqual(results[0][1], results[1][1])

    def test_upgrade_gzip_snapshots_verifies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            playback = root / "playback"
            playback.mkdir()
            import random

            rng = random.Random(42)
            original = []
            animals = [
                dict(
                    id=i,
                    species="prey",
                    x=i,
                    y=2,
                    health=5.0,
                    age=0,
                    energy=rng.random(),
                )
                for i in range(80)
            ]
            for step in range(64):
                for animal in animals:
                    animal.update(age=step, health=rng.random(), x=rng.random())
                original.append(
                    dict(
                        step=step,
                        animals=copy.deepcopy(animals),
                        resources=[],
                        metrics={},
                    )
                )
            path = playback / "run_0001_chunk_000000.json.gz"
            path.write_bytes(gzip.compress(json.dumps(original).encode()))
            (playback / "recording.json").write_text(
                json.dumps(
                    dict(
                        recording_version=1,
                        status="complete",
                        runs=[dict(run=1, chunk_count=1)],
                    )
                )
            )
            result = compress_recording(root, delta=True)
            self.assertGreater(result["saved_bytes"], 0)
            self.assertEqual(frame_digest(load_chunk(path)), frame_digest(original))
            self.assertEqual(load_chunk_payload(path)["codec"], "wtf-delta-v1")
            self.assertEqual(compress_recording(root, delta=True)["saved_bytes"], 0)

    def test_upgrade_failed_verification_leaves_original_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            playback = root / "playback"
            playback.mkdir()
            path = playback / "run_0001_chunk_000000.json"
            path.write_text(json.dumps(frames()))
            original = path.read_bytes()
            (playback / "recording.json").write_text(
                json.dumps(
                    dict(
                        recording_version=1,
                        status="complete",
                        runs=[dict(run=1, chunk_count=1)],
                    )
                )
            )
            with patch(
                "wtf.storage.compress_recordings.frame_digest", side_effect=[b"a", b"b"]
            ):
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    compress_recording(root, delta=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(playback.glob("*.tmp")))
