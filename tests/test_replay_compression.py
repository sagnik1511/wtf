"""Compressed replays preserve exact data and stay readable during conversion."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ecosystem.analyze import analyze_recording
from ecosystem.compress_recordings import compress_recording
from ecosystem.config import validate
from ecosystem.recording import Recorder, RecordingStore
from ecosystem.replay_io import chunk_path, load_chunk, read_chunk_bytes
from ecosystem.runner import Experiment
from tests.test_simulation import config


def record(folder,compression='gzip',steps=8):
    c=config();c['world']['max_steps']=steps;c['experiment']['runs']=2
    c['recording'].update(compression=compression,compression_level=3,frames_per_chunk=3)
    e=Experiment(c);r=Recorder(folder,c,'train');r.begin_run(e)
    while not e.done:
        if e.finished_run:e.next_run();r.begin_run(e)
        e.step();r.capture(e,force=e.finished_run)
        if e.finished_run:r.end_run(e)
    r.close()
    return e


class ReplayCompressionTests(unittest.TestCase):
    def test_default_streams_gzip_and_preserves_exact_frames_and_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plain=record(root/'plain','none');packed=record(root/'packed')
            self.assertEqual(plain.world.snapshot(),packed.world.snapshot())
            self.assertEqual({i:b.checkpoint() for i,b in plain.brains.items()},
                             {i:b.checkpoint() for i,b in packed.brains.items()})
            store=RecordingStore(root)
            for run in (1,2):
                metadata=store.metadata('plain')['runs'][run-1]
                for index in range(metadata['chunk_count']):
                    a,b=store.chunk('plain',run,index),store.chunk('packed',run,index)
                    self.assertEqual(a.suffix,'.json');self.assertEqual(b.suffix,'.gz')
                    self.assertEqual(a.read_bytes(),read_chunk_bytes(b))
                    self.assertEqual(load_chunk(a),load_chunk(b))
            self.assertFalse(list((root/'packed/playback').glob('*.tmp')))
            self.assertEqual(store.metrics('plain',1),store.metrics('packed',1))
            self.assertEqual(analyze_recording(root/'plain'),analyze_recording(root/'packed'))

    def test_existing_recordings_convert_losslessly_and_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'example';record(folder,'none')
            playback=folder/'playback'
            original={p.name:p.read_bytes() for p in playback.glob('*chunk*.json')}
            report_before=analyze_recording(folder)
            checkpoint=folder/'checkpoint.json';checkpoint.write_text('untouched')
            result=compress_recording(folder)
            self.assertGreater(result['saved_bytes'],0)
            self.assertEqual(result['chunks_converted'],len(original))
            self.assertEqual(list(playback.glob('*chunk*.json')),[])
            for name,data in original.items():
                self.assertEqual(read_chunk_bytes(playback/(name+'.gz')),data)
            self.assertEqual(analyze_recording(folder),report_before)
            self.assertEqual(checkpoint.read_text(),'untouched')
            self.assertEqual(compress_recording(folder)['chunks_converted'],0)
            self.assertEqual(compress_recording(folder)['saved_bytes'],0)

    def test_failed_verification_keeps_the_plain_original(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'example';record(folder,'none',steps=2)
            originals={p:p.read_bytes() for p in (folder/'playback').glob('*chunk*.json')}
            with patch('ecosystem.compress_recordings.digest',return_value=b'invalid'):
                with self.assertRaisesRegex(ValueError,'verification failed'):
                    compress_recording(folder)
            for path,data in originals.items():self.assertEqual(path.read_bytes(),data)
            self.assertFalse(list((folder/'playback').glob('*.tmp')))

    def test_partially_converted_recording_still_plays_and_resumes_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'example';record(folder,'none')
            original=next((folder/'playback').glob('*chunk*.json'))
            packed=original.with_suffix('.json.gz');packed.write_bytes(gzip.compress(original.read_bytes()))
            # A crash after publication can leave both formats. Verify before removal.
            run=int(original.name.split('_')[1]);index=int(original.name.split('_')[3].split('.')[0])
            self.assertEqual(chunk_path(folder/'playback',run,index),packed.resolve())
            compress_recording(folder)
            self.assertFalse(original.exists())
            self.assertTrue(packed.exists())
            self.assertEqual(analyze_recording(folder)['runs'][0]['coverage']['step_fraction'],1)

    def test_refuse_active_recording_and_mismatched_existing_gzip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'example';record(folder,'none')
            path=folder/'playback/recording.json';metadata=json.loads(path.read_text())
            metadata['status']='writing';path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'still being written'):compress_recording(folder)
            metadata['status']='complete';path.write_text(json.dumps(metadata))
            original=next((folder/'playback').glob('*chunk*.json'))
            original.with_suffix('.json.gz').write_bytes(gzip.compress(b'[]'))
            with self.assertRaisesRegex(ValueError,'differs'):compress_recording(folder)
            self.assertTrue(original.exists())

    def test_invalid_compression_config_and_symlink_traversal(self):
        for key,value in [('encoding','unknown'),('compression','zip'),('compression_level',0),('compression_level',10),('compression_level',True)]:
            c=config();c['recording'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate(c)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'playback';folder.mkdir()
            target=root/'private.json.gz';target.write_bytes(gzip.compress(b'[]'))
            (folder/'run_0001_chunk_000000.json.gz').symlink_to(target)
            with self.assertRaises(ValueError):chunk_path(folder,1,0)
