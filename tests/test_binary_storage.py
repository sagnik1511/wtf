"""Binary Python/JavaScript parity, exact resume, transport and verified migration."""
import base64
import copy
import gzip
import io
import json
import lzma
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from unittest.mock import patch

from ecosystem.binary_io import Encoder,Decoder,RecordWriter,records,replay_payload,MAGIC
from ecosystem.checkpoint_io import read_checkpoint
from ecosystem.compress_recordings import frame_digest,compress_recording
from ecosystem.optimize_storage import optimize_experiment
from ecosystem.recording import Recorder,RecordingStore
from ecosystem.replay_codec import encode_chunk,decode_chunk
from ecosystem.replay_io import chunk_path,load_chunk,read_chunk_bytes
from ecosystem.runner import Experiment
from ecosystem.server import make_handler,State
from tests.test_advanced_algorithms import environment
from tests.test_simulation import config
from tests.test_replay_compression import record
from tests.test_replay_codec import frames

ROOT=Path(__file__).resolve().parents[1]


def binary(frames):
    payload=encode_chunk(frames);out=io.BytesIO();writer=RecordWriter(out)
    writer.write(dict(kind='replay',delta={k:v for k,v in payload.items() if k!='frames'}))
    for packet in payload['frames']:writer.write(packet)
    writer.write(dict(kind='replay_end',frames=len(frames)))
    return out.getvalue()


def recording(folder,encoding='binary',compression='xz',steps=12):
    c=config();c['recording'].update(encoding=encoding,compression=compression,frames_per_chunk=4)
    c['world']['max_steps']=steps;c['experiment']['runs']=1
    e=Experiment(c);r=Recorder(folder,c,'train');r.begin_run(e)
    while not e.done:e.step();r.capture(e)
    r.end_run(e);r.close()
    return e


class BinaryCodecTests(unittest.TestCase):
    def test_typed_columns_arrays_symbols_integer_types_and_signed_zero(self):
        values=[None,False,True,0,-1,1.0,-0.0,'prey','prey',2**53-1,-(2**53-1),2**64-1,-2**100,
                [1,2,3,4],[1.0,2.0,-0.0,4.0],[2**32-1]*20,
                [dict(id=i,species='prey',health=10.0+i) for i in range(4)],
                {'$ref':3,'__proto__':'safe','odd':[[1.0,0,False]]}]
        encoder,decoder=Encoder(),Decoder()
        result=decoder.decode(encoder.encode(values))
        self.assertEqual(result,values)
        self.assertIs(type(result[5]),float)
        self.assertEqual(math.copysign(1,result[6]),-1)
        self.assertEqual(math.copysign(1,result[14][2]),-1)
        self.assertEqual(decoder.decode(encoder.encode('prey')),'prey')

    def test_full_replay_roundtrip_is_lossless(self):
        original=frames();data=binary(original)
        self.assertTrue(data.startswith(MAGIC))
        self.assertEqual(frame_digest(decode_chunk(replay_payload(data))),frame_digest(original))

    def test_javascript_decodes_python_columns_and_exact_safe_integers(self):
        original=frames()
        original[0]['extension']={'__proto__':'safe','big':2**53-1,'negative':-(2**53-1),'array':[2**32-1]*20}
        script="""import assert from 'node:assert/strict';
        import {decodeBinaryReplay} from './viewer/binary-codec.mjs';
        import {decodeChunk} from './viewer/replay-codec.mjs';
        let data='';for await(const chunk of process.stdin)data+=chunk;
        const {bytes,expected}=JSON.parse(data);
        const result=decodeChunk(decodeBinaryReplay(Buffer.from(bytes,'base64')));
        assert.deepStrictEqual(result,expected);
        assert(Object.is(result[1].animals[0].health,-0));
        """
        payload=dict(bytes=base64.b64encode(binary(original)).decode(),expected=original)
        p=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(payload),text=True,capture_output=True,cwd=ROOT)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_corruption_unknown_tags_and_incomplete_files_are_rejected(self):
        for value in (float('nan'),float('inf'),[0.0]*4+[float('inf')]):
            with self.assertRaises(ValueError):Encoder().encode(value)
        for data in (b'\xff',b'\x06\x00',b'\x04',b'\x00\x00'):
            with self.assertRaises(ValueError):Decoder().decode(data)
        data=binary(frames())
        for n in (0,1,4,6,len(data)-1):
            with self.assertRaises((ValueError,EOFError)):replay_payload(data[:n])
        out=io.BytesIO();w=RecordWriter(out);w.write(dict(kind='replay',delta=encode_chunk([])))
        with self.assertRaisesRegex(ValueError,'footer'):replay_payload(out.getvalue())


class BinaryLifecycleTests(unittest.TestCase):
    def test_all_compressions_render_identically_and_do_not_change_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);outputs=[];states=[]
            for encoding,compression in (('snapshot','none'),('delta','gzip'),('binary','none'),('binary','gzip'),('binary','xz')):
                folder=root/(encoding+'_'+compression);e=recording(folder,encoding,compression)
                state=e.checkpoint()
                state['config'].pop('recording');state['world']['config'].pop('recording')
                states.append(state);store=RecordingStore(root);name=folder.name
                m=store.metadata(name);out=[]
                for i in range(m['runs'][0]['chunk_count']):out.extend(load_chunk(store.chunk(name,1,i)))
                outputs.append(out)
                self.assertTrue((store.chunk(name,1,0)).is_file())
                self.assertFalse(list((folder/'playback').glob('*.tmp')))
            self.assertTrue(all(s==states[0] for s in states))
            self.assertTrue(all(frame_digest(f)==frame_digest(outputs[0]) for f in outputs))

    def test_javascript_playback_seeks_binary_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';recording(p);store=RecordingStore(d)
            metadata=store.metadata('run');chunks={};expected=[]
            for i in range(metadata['runs'][0]['chunk_count']):
                path=store.chunk('run',1,i)
                chunks[str(i)]=base64.b64encode(read_chunk_bytes(path)).decode()
                expected.extend(load_chunk(path))
            payload=dict(metadata=metadata,metrics=store.metrics('run',1),chunks=chunks,expected=expected)
            script="""import assert from 'node:assert/strict';
            import {Playback} from './viewer/playback.mjs';
            import {decodeBinaryReplay} from './viewer/binary-codec.mjs';
            let input='';for await(const data of process.stdin)input+=data;
            const p=JSON.parse(input);
            const player=new Playback(async url=>{
              if(url.endsWith('/metrics'))return p.metrics;
              if(url.includes('/chunks/'))return decodeBinaryReplay(Buffer.from(p.chunks[url.split('/').at(-1)],'base64'));
              return p.metadata;
            });
            await player.open('run');
            for(const i of [0,3,4,8,12,2]){
              const state=await player.seek(i);
              assert.equal(state.tick,p.expected[i].tick);
              assert.deepStrictEqual(state.animals,p.expected[i].animals);
              assert.deepStrictEqual(state.resources,p.expected[i].resources);
            }
            """
            result=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(payload),text=True,capture_output=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_all_algorithms_resume_binary_checkpoints_exactly(self):
        from tests.test_policy_algorithms import environment as policy_environment
        from tests.test_ppo import ppo_environment
        cases=[config(),ppo_environment()]+[policy_environment(k) for k in ('a2c','reinforce','grpo')]+[environment(k) for k in ('recurrent_ppo','qrdqn','maskable_ppo','mappo')]
        with tempfile.TemporaryDirectory() as d:
            for c in cases:
                a=Experiment(c)
                for _ in range(3):a.step()
                for suffix in ('.bin.gz','.bin.xz'):
                    p=Path(d)/('state'+suffix);a.write_checkpoint(p)
                    b=Experiment.restore(read_checkpoint(p));control=Experiment.restore(json.loads(json.dumps(a.checkpoint())))
                    for _ in range(4):b.step();control.step()
                    self.assertEqual(b.checkpoint(),control.checkpoint())

    def test_randomized_64_bit_seed_checkpoint_and_old_path_alias(self):
        c=config();c['experiment']['world_seed_policy']='random'
        with patch('secrets.randbits',return_value=2**64-1):a=Experiment(c)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'final.bin.xz';a.write_checkpoint(p)
            state=read_checkpoint(Path(d)/'final.json.gz')
            self.assertEqual(state['randomization_seed'],2**64-1)
            b=Experiment.restore(state)
            a.step();b.step();self.assertEqual(a.checkpoint(),b.checkpoint())

    def test_replay_and_checkpoint_conversion_preserves_sources_until_verified(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';a=record(p,'gzip',steps=12)
            (p/'checkpoints').mkdir();a.write_checkpoint(p/'checkpoints/final.json.gz')
            (p/'metrics.csv').write_text('readable metrics')
            state=read_checkpoint(p/'checkpoints/final.json.gz')
            from ecosystem.analyze import analyze_recording
            report=analyze_recording(p)
            result=optimize_experiment(p)
            self.assertGreater(result['saved_bytes'],0)
            self.assertEqual(state,read_checkpoint(p/'checkpoints/final.json.gz'))
            self.assertEqual(report,analyze_recording(p))
            self.assertEqual((p/'metrics.csv').read_text(),'readable metrics')
            self.assertEqual(optimize_experiment(p)['saved_bytes'],0)
            self.assertEqual(compress_recording(p)['saved_bytes'],0)
            self.assertFalse(list(p.rglob('*.tmp')))

    def test_failed_verification_or_active_run_keeps_originals(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';record(p,'gzip',steps=2)
            original={f:f.read_bytes() for f in (p/'playback').glob('*chunk*')}
            normal=load_chunk
            def corrupt(path):return [{'invalid':True}] if Path(path).name.startswith('.') else normal(path)
            with patch('ecosystem.optimize_storage.load_chunk',side_effect=corrupt):
                with self.assertRaisesRegex(ValueError,'verification failed'):optimize_experiment(p)
            self.assertTrue(all(f.read_bytes()==v for f,v in original.items()))
            self.assertFalse(list((p/'playback').glob('.*.bin.xz')))
            m=p/'playback/recording.json';v=json.loads(m.read_text());v['status']='writing';m.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'still writing'):optimize_experiment(p)

    def test_telemetry_cache_is_bounded_detached_and_file_validated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';recording(p,steps=520);store=RecordingStore(d)
            from ecosystem.recording import load_chunk as reader
            with patch('ecosystem.recording.load_chunk',wraps=reader) as read:
                first=store.telemetry('run',1,0);count=read.call_count
                first['points'].clear()
                self.assertTrue(store.telemetry('run',1,0)['points']);self.assertEqual(read.call_count,count)
                source=store.chunk('run',1,0);stat=source.stat();os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
                store.telemetry('run',1,0);self.assertGreater(read.call_count,count)
            for page in range(5):store.telemetry('run',1,page)
            self.assertLessEqual(len(store.telemetry_cache),4)

    def test_conversion_keeps_a_smaller_original_and_its_layout(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);playback=p/'playback';playback.mkdir()
            source=playback/'run_0001_chunk_000000.json';source.write_text('[{}]')
            metadata=dict(recording_version=1,status='complete',runs=[dict(run=1,chunk_count=1)])
            (playback/'recording.json').write_text(json.dumps(metadata))
            result=optimize_experiment(p,checkpoints=False)
            self.assertEqual(result['saved_bytes'],0)
            self.assertEqual(source.read_text(),'[{}]')
            self.assertEqual(json.loads((playback/'recording.json').read_text())['storage_format'],'snapshot')

    def test_javascript_rejects_truncated_or_unknown_binary_records(self):
        payload=base64.b64encode(binary(frames())).decode()
        script="""import assert from 'node:assert/strict';
        import {decodeBinaryReplay} from './viewer/binary-codec.mjs';
        let input='';for await(const data of process.stdin)input+=data;
        const bytes=Buffer.from(input,'base64');
        for(const n of [0,4,bytes.length-1])assert.throws(()=>decodeBinaryReplay(bytes.subarray(0,n)));
        const bad=Buffer.from(bytes);bad[9]=255;
        assert.throws(()=>decodeBinaryReplay(bad));
        """
        p=subprocess.run(['node','--input-type=module','-e',script],input=payload,text=True,capture_output=True,cwd=ROOT)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_native_binary_transport_and_legacy_json_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';recording(p)
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(State(d)))
            t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
            url=f'http://127.0.0.1:{server.server_port}/api/recordings/run/runs/1/chunks/0'
            try:
                with urlopen(Request(url,headers={'X-WTF-Replay':'binary-v1'})) as r:
                    self.assertIn('application/vnd.wtf.replay-binary',r.headers['Content-Type'])
                    self.assertEqual(r.headers.get('X-WTF-Replay-Protocol'),'binary-v1')
                    data=r.read();decoded=decode_chunk(replay_payload(data))
                for headers in ({},{'X-WTF-Replay':'delta-v1'}):
                    with urlopen(Request(url,headers=headers)) as r:self.assertEqual(json.loads(r.read()),decoded)
            finally:server.shutdown();server.server_close();t.join()
