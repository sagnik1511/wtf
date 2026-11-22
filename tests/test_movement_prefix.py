"""Streamed replay-prefix counts, information boundaries and Python/JS parity."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.test_analysis import animal, frame
from tests.test_simulation import config
from wtf.inspection.analyze import analyze_run
from wtf.inspection.movement import MovementCounts
from wtf.storage.recording import RecordingStore
from wtf.storage.replay_io import load_chunk

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
            [dict(kind="died", actor=0, cause="starvation", position=[5, 2])],
        ),
        frame(5, [dict(b, x=4), animal(2, "prey", parent_id=0)]),
        frame(6, [dict(b, x=5), animal(2, "prey", parent_id=0)]),
    ]


class MovementPrefixTests(unittest.TestCase):
    def save(self, root, frames):
        folder = root / "example" / "playback"
        folder.mkdir(parents=True)
        manifest = dict(
            recording_version=1,
            name="example",
            mode="train",
            config=config(),
            recording=dict(frames_per_chunk=2),
            runs=[
                dict(
                    run=1,
                    frame_count=len(frames),
                    chunk_count=3,
                    seed=100,
                    first_tick=frames[0]["tick"],
                    last_tick=frames[-1]["tick"],
                    complete=True,
                )
            ],
        )
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
const {frames,expected}=JSON.parse(fs.readFileSync(0,'utf8'));
const history=new MovementHistory();history.reset(10,8);const buffer=history.map.counts.buffer;
let requests=0;
for(const i of [0,1,0,1,2,3,2,5,4,5,0,5]) {
 const start=Math.floor(i/2)*2;
 await history.move(frames.slice(start,start+2),start,i,async index=>{requests++;return expected[index];});
 assert.deepEqual([...history.map.counts],expected[i].counts);
 assert.equal(history.map.arrivals,expected[i].arrivals);
 assert.equal(history.map.maximum,expected[i].maximum);
 assert.equal(history.map.tick,frames[i].tick);
 assert.equal(history.map.counts.buffer,buffer);
}
assert(requests>0&&requests<6);
console.log('prefix parity passed');"""
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            input=json.dumps(dict(frames=frames, expected=expected)),
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
