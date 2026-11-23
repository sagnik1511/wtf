"""Streamed replay-prefix counts, information boundaries and Python/JS parity."""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen
import unittest
from unittest.mock import patch

from tests.test_analysis import animal, frame
from tests.test_simulation import config
from wtf.inspection.analyze import analyze_run
from wtf.inspection.movement import MovementCounts
from wtf.storage.recording import RecordingStore
from wtf.storage.replay_io import load_chunk
from wtf.server import State, make_handler

ROOT = Path(__file__).resolve().parents[1]


def moving_frames():
    a = animal(0, "prey")
    b = animal(1, "predator")
    return [
        frame(0, [a, b]),
        frame(1, [dict(a, x=3), b]),
        frame(2, [dict(a, x=4), b, animal(2, "prey", parent_id=0)]),
        frame(
            3,
            [b, animal(2, "prey", parent_id=0)],
            [{"kind": "died", "actor": 0, "cause": "starvation", "position": [5, 2]}],
        ),
        frame(5, [dict(b, x=4), animal(2, "prey", parent_id=0)]),
        frame(6, [dict(b, x=5), animal(2, "prey", parent_id=0)]),
    ]


class MovementPrefixTests(unittest.TestCase):
    def save(self, root, frames):
        folder = root / "example" / "playback"
        folder.mkdir(parents=True)
        manifest = {
            "recording_version": 1,
            "name": "example",
            "mode": "train",
            "config": config(),
            "recording": {"frames_per_chunk": 2},
            "runs": [
                {
                    "run": 1,
                    "frame_count": len(frames),
                    "chunk_count": 3,
                    "seed": 100,
                    "first_tick": frames[0]["tick"],
                    "last_tick": frames[-1]["tick"],
                    "complete": True,
                }
            ],
        }
        (folder / "recording.json").write_text(json.dumps(manifest))
        for i in range(0, len(frames), 2):
            (folder / f"run_0001_chunk_{i//2:06d}.json").write_text(
                json.dumps(frames[i : i + 2])
            )
        return RecordingStore(root)

    def test_every_prefix_matches_analysis_and_does_not_write_or_read_later_chunks(
        self,
    ):
        frames = moving_frames()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self.save(root, frames)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            for i, f in enumerate(frames):
                with patch(
                    "wtf.storage.recording.load_chunk", wraps=load_chunk
                ) as reader:
                    result = store.movement("example", 1, i)
                    self.assertEqual(reader.call_count, i // 2 + 1)
                expected = analyze_run(config(), frames[: i + 1])["movement_heatmap"]
                self.assertEqual(result, dict(expected, frame_index=i, tick=f["tick"]))
            self.assertEqual(
                before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            )
            for run, index, error in (
                (1, -1, ValueError),
                (1, 6, FileNotFoundError),
                (2, 0, FileNotFoundError),
                (True, 0, ValueError),
                (1, True, ValueError),
            ):
                with self.subTest(run=run, index=index), self.assertRaises(error):
                    store.movement("example", run, index)

    def test_species_prefixes_partition_all_movements_including_terminal_drift(self):
        frames = moving_frames()
        # A third species of the same diet is independent of the first one.
        for i, f in enumerate(frames):
            f["animals"].append(dict(animal(3, "field_mouse"), x=i % 4))
        with tempfile.TemporaryDirectory() as temporary:
            store = self.save(Path(temporary), frames)
            manifest_path = Path(temporary) / "example/playback/recording.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["config"]["species"]["field_mouse"] = manifest["config"][
                "species"
            ]["prey"]
            manifest_path.write_text(json.dumps(manifest))
            for i in range(len(frames)):
                with patch(
                    "wtf.storage.recording.load_chunk", wraps=load_chunk
                ) as reader:
                    grouped = store.movement("example", 1, i, by_species=True)
                self.assertEqual(reader.call_count, i // 2 + 1)
                self.assertEqual(
                    grouped["counts"], store.movement("example", 1, i)["counts"]
                )
                groups = [
                    store.movement("example", 1, i, name)
                    for name in ("prey", "predator", "field_mouse")
                ]
                self.assertEqual(
                    grouped["counts"],
                    [sum(cells) for cells in zip(*(g["counts"] for g in groups))],
                )
                for name, group in zip(("prey", "predator", "field_mouse"), groups):
                    self.assertEqual(group["species"], name)
                    self.assertEqual(
                        grouped["by_species"][name]["counts"], group["counts"]
                    )
            for invalid in ("unknown", "", 1, []):
                with self.assertRaises(ValueError):
                    store.movement("example", 1, 2, invalid)
            prey = store.movement("example", 1, 3, "prey")
            self.assertEqual(prey["counts"][2 * 10 + 5], 1)
            self.assertEqual(prey["arrivals"], 3)

    def test_http_species_query_and_invalid_filter_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.save(root, moving_frames())
            with ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(State(root))
            ) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = (
                    f"http://127.0.0.1:{server.server_port}"
                    "/api/recordings/example/runs/1/movement/3"
                )
                try:
                    with urlopen(base + "?species=prey", timeout=5) as response:
                        result = json.load(response)
                    self.assertEqual(result["species"], "prey")
                    self.assertEqual(result["arrivals"], 3)
                    with urlopen(base + "?by_species=1", timeout=5) as response:
                        grouped = json.load(response)
                    self.assertEqual(grouped["by_species"]["prey"]["arrivals"], 3)
                    for query in (
                        "?species=unknown",
                        "?species=",
                        "?species=prey&species=predator",
                        "?other=prey",
                        "?by_species=bad",
                        "?by_species=1&species=prey",
                    ):
                        with self.assertRaises(HTTPError) as error:
                            with urlopen(base + query, timeout=5) as response:
                                self.fail(f"Unexpected status {response.status}")
                        self.assertEqual(error.exception.code, 400)
                finally:
                    server.shutdown()
                    thread.join(timeout=5)

    def test_viewer_behavior_report_omits_unused_full_run_grid(self):
        frames = moving_frames()
        full = analyze_run(config(), frames)
        current = analyze_run(config(), frames, include_movement=False)
        full.pop("movement_heatmap")
        self.assertEqual(full, current)
        with tempfile.TemporaryDirectory() as temporary:
            store = self.save(Path(temporary), frames)
            self.assertNotIn(
                "movement_heatmap", store.analysis("example", 1)["runs"][0]
            )

    def test_counter_keeps_only_grid_and_prior_positions(self):
        counter = MovementCounts(10, 8)
        for f in moving_frames():
            counter.consume(f)
        self.assertEqual(counter.counts.itemsize, 4)
        self.assertEqual(len(counter.counts), 80)
        self.assertTrue(all(set(a) == {"x", "y"} for a in counter.previous.values()))
        self.assertEqual(counter.summary()["arrivals"], 4)

    @unittest.skipUnless(
        shutil.which("node"), "Node is optional for Python-only development"
    )
    def test_js_forward_rewind_and_large_jumps_match_python_prefixes(self):
        frames = moving_frames()
        expected = [
            dict(
                analyze_run(config(), frames[: i + 1])["movement_heatmap"],
                frame_index=i,
                tick=f["tick"],
            )
            for i, f in enumerate(frames)
        ]
        script = """import fs from 'node:fs';
import assert from 'node:assert/strict';
import {MovementHistory} from './viewer/movement-history.mjs';
const {frames,expected,species,species_names=[]}=JSON.parse(fs.readFileSync(0,'utf8'));
const history=new MovementHistory();history.reset(10,8,species,species_names);const buffer=history.map.counts.buffer;
let requests=0;
for(const i of [0,1,0,1,2,3,2,5,4,5,0,5]) {
 const start=Math.floor(i/2)*2;
 await history.move(frames.slice(start,start+2),start,i,async index=>{requests++;return expected[index];});
 assert.deepEqual([...history.map.counts],expected[i].counts);
 assert.equal(history.map.arrivals,expected[i].arrivals);
 assert.equal(history.map.maximum,expected[i].maximum);
 assert.equal(history.map.tick,frames[i].tick);
 assert.equal(history.map.counts.buffer,buffer);
 for(const name of species_names) assert.deepEqual([...history.map.by_species[name].counts], expected[i].by_species[name].counts);
}
assert(requests>0&&requests<6);
console.log('prefix parity passed');"""
        for name in (None, "prey", "predator"):
            if name is not None:
                counter = MovementCounts(10, 8, name)
                expected = []
                for i, f in enumerate(frames):
                    counter.consume(f)
                    expected.append(
                        dict(counter.summary(), frame_index=i, tick=f["tick"])
                    )
            self.assert_js_parity(script, frames, expected, name)

        counter = MovementCounts(10, 8, species_names=("prey", "predator", "absent"))
        expected = []
        for i, f in enumerate(frames):
            counter.consume(f)
            expected.append(dict(counter.summary(), frame_index=i, tick=f["tick"]))
        self.assert_js_parity(
            script, frames, expected, None, ("prey", "predator", "absent")
        )

    def assert_js_parity(self, script, frames, expected, species, species_names=()):
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            input=json.dumps(
                {
                    "frames": frames,
                    "expected": expected,
                    "species": species,
                    "species_names": species_names,
                }
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
