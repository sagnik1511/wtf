import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ecosystem.recording import Recorder, RecordingStore
from ecosystem.runner import Experiment
from tests.test_simulation import config


class RecordingTests(unittest.TestCase):
    def record(self, folder, experiment):
        recorder = Recorder(folder, experiment.config, experiment.mode)
        recorder.begin_run(experiment)
        while not experiment.done:
            if experiment.finished_run:
                experiment.next_run()
                recorder.begin_run(experiment)
            experiment.step()
            recorder.capture(experiment, force=experiment.finished_run)
            if experiment.finished_run:
                recorder.end_run(experiment)
        recorder.close()
        return recorder

    def test_multiple_runs_and_exact_frame_content(self):
        c = config()
        c['recording'].update(frames_per_chunk=3, diagnostics_every_steps=2)
        a, control = Experiment(c), Experiment(c)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'example'
            recorder = self.record(folder, a)
            while not control.done:
                control.step()
            self.assertEqual(a.checkpoint(), control.checkpoint(), 'Recording must not change random or learning state')
            store = RecordingStore(directory)
            self.assertEqual(len(store.catalog()), 1)
            manifest = store.metadata('example')
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual([r['frame_count'] for r in manifest['runs']], [21, 21])
            for entry in manifest['runs']:
                frames = []
                for i in range(entry['chunk_count']):
                    chunk = json.loads(store.chunk('example', entry['run'], i).read_text())
                    self.assertLessEqual(len(chunk), 3)
                    frames.extend(chunk)
                self.assertEqual([f['tick'] for f in frames], list(range(21)))
                self.assertTrue(frames[-1]['finished_run'])
                self.assertTrue(entry['complete'])
                self.assertNotIn('obstacles', frames[0])
                self.assertIn('diagnostics', frames[0])
                self.assertNotIn('weights', json.dumps(frames))
                self.assertNotIn('replay', frames[0]['diagnostics']['0'])
            final_frame = json.loads(store.chunk('example', 2, 6).read_text())[-1]
            actual = a.world.snapshot()
            self.assertEqual(final_frame['animals'], actual['animals'])
            self.assertEqual(final_frame['resources'], actual['resources'])
            self.assertEqual(final_frame['metrics'], actual['metrics'])

    def test_sampled_frames_always_include_initial_and_terminal(self):
        c = config()
        c['world']['max_steps'] = 11
        c['experiment']['runs'] = 1
        c['recording'].update(frame_every_steps=4, frames_per_chunk=2, diagnostics_every_steps=0)
        with tempfile.TemporaryDirectory() as directory:
            self.record(Path(directory) / 'sample', Experiment(c))
            store = RecordingStore(directory)
            frames = sum([json.loads(store.chunk('sample', 1, i).read_text()) for i in range(2)], [])
            self.assertEqual([f['tick'] for f in frames], [0, 4, 8, 11])
            self.assertTrue(all('diagnostics' not in f for f in frames))

    def test_only_committed_chunks_are_visible(self):
        c = config()
        c['recording']['frames_per_chunk'] = 2
        e = Experiment(c)
        with tempfile.TemporaryDirectory() as directory:
            r = Recorder(Path(directory) / 'partial', c, 'train')
            r.begin_run(e)
            store = RecordingStore(directory)
            self.assertEqual(store.catalog(), [])
            with self.assertRaises(FileNotFoundError):
                store.chunk('partial', 1, 0)
            e.step()
            r.capture(e)
            self.assertEqual(store.metadata('partial')['runs'][0]['frame_count'], 2)
            e.step()
            r.capture(e)
            # This next chunk is still being written and is not exposed.
            self.assertEqual(store.metadata('partial')['runs'][0]['chunk_count'], 1)
            r.close(completed=False)
            manifest = store.metadata('partial')
            self.assertEqual(manifest['status'], 'interrupted')
            self.assertEqual(manifest['runs'][0]['frame_count'], 3)
            self.assertFalse(manifest['runs'][0]['complete'])

    def test_path_traversal_is_rejected(self):
        store = RecordingStore('/tmp')
        for name in ('..', '.', '../outside', '/absolute'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                store.metadata(name)

    def test_recording_options_validate(self):
        from ecosystem.config import validate
        for key, value in [('frames_per_chunk', 0), ('frame_every_steps', 0), ('diagnostics_every_steps', -1), ('enabled', 1)]:
            c = config()
            c['recording'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate(c)
        c = config()
        del c['recording']
        validate(c)  # Previous environment/checkpoint files stay compatible.

    def test_cli_recording_and_opt_out(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for skip in (False, True):
                output = Path(directory) / ('skip' if skip else 'record')
                args = [sys.executable, '-m', 'ecosystem.cli', 'environments/baseline.json', '--runs', '2', '--steps', '3', '--output', str(output)]
                if skip:
                    args.append('--no-record')
                subprocess.run(args, cwd=root, check=True, capture_output=True, text=True)
                folder = next(output.iterdir())
                self.assertTrue((folder / 'metrics.csv').is_file())
                self.assertEqual((folder / 'playback').exists(), not skip)
                if not skip:
                    store = RecordingStore(output)
                    manifest = store.metadata(folder.name)
                    self.assertEqual([r['frame_count'] for r in manifest['runs']], [4, 4])
                    summary = json.loads((folder / 'summary.json').read_text())
                    frame = json.loads(store.chunk(folder.name, 2, 0).read_text())[-1]
                    for key, value in frame['metrics'].items():
                        self.assertEqual(summary[-1][key], value)
                    # A completed learning checkpoint can still be recorded as
                    # one final frame, without inventing more simulation steps.
                    resumed = Path(directory) / 'resumed'
                    subprocess.run([sys.executable, '-m', 'ecosystem.cli', '--resume', str(folder / 'checkpoints/final.json'), '--output', str(resumed)], cwd=root, check=True, capture_output=True)
                    m = RecordingStore(resumed).metadata(next(resumed.iterdir()).name)
                    self.assertEqual(m['runs'][0]['frame_count'], 1)
                    self.assertTrue(m['runs'][0]['complete'])

    def test_viewer_imports_no_simulation(self):
        subprocess.run([sys.executable, '-c', "import sys; from ecosystem.server import State; s=State(); assert not hasattr(s, 'experiment'); assert 'ecosystem.brain' not in sys.modules; assert 'ecosystem.world' not in sys.modules; assert 'ecosystem.runner' not in sys.modules"], check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    unittest.main()
