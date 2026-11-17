"""Verify and shrink completed recordings/checkpoints using lossless binary data."""
import argparse
import gzip
import hashlib
import json
import lzma
from pathlib import Path
import uuid

from .binary_io import RecordWriter
from .checkpoint_io import read_checkpoint, write_checkpoint
from .compress_recordings import frame_digest
from .optimize_checkpoints import state_digest
from .recording import write_json
from .replay_codec import encode_chunk, decode_chunk
from .replay_io import chunk_path, load_chunk, load_chunk_payload

SUFFIXES=('.bin.xz','.bin.gz','.bin','.json.xz','.json.gz','.json')


def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.digest()


def copies(folder,stem):
    result=[]
    for suffix in SUFFIXES:
        p=(folder/(stem+suffix)).resolve()
        if p.parent!=folder:raise ValueError('Storage path outside experiment')
        if p.is_file():result.append(p)
    if not result:raise FileNotFoundError(stem)
    return result


def publish_candidate(originals, hashes, temporary, target):
    if any(file_digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during storage optimization')
    smallest=min(originals,key=lambda p:p.stat().st_size)
    if temporary.stat().st_size<smallest.stat().st_size:
        temporary.replace(target);kept=target
    else:kept=smallest
    for p in originals:
        if p!=kept:p.unlink()
    return kept


def optimize_experiment(folder, replays=True, checkpoints=True, compression='xz', progress=None):
    if compression not in ('xz','gzip'):raise ValueError('Binary storage compression must be xz or gzip')
    folder=Path(folder).resolve();playback=(folder/'playback').resolve()
    if playback.parent!=folder:raise ValueError('Playback path outside experiment')
    manifest_path=playback/'recording.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest and manifest.get('status')=='writing':raise ValueError('Experiment is still writing')
    if manifest and manifest.get('recording_version')!=1:raise ValueError('Unsupported recording version')
    cp=(folder/'checkpoints').resolve()
    if cp.parent!=folder:raise ValueError('Checkpoint path outside experiment')
    if manifest is None and not any((cp/('final'+s)).is_file() for s in SUFFIXES):
        raise ValueError('No completion marker; experiment may still be writing')
    suffix='.bin.xz' if compression=='xz' else '.bin.gz'
    before=after=converted=0;replay_before=replay_after=checkpoint_before=checkpoint_after=0
    if replays and manifest:
        formats=set();compressions=set()
        for run in manifest['runs']:
            for index in range(run['chunk_count']):
                stem=f"run_{run['run']:04d}_chunk_{index:06d}"
                originals=copies(playback,stem);old=sum(p.stat().st_size for p in originals)
                source=chunk_path(playback,run['run'],index);hashes={p:file_digest(p) for p in originals}
                # Already binary in the requested compression needs no re-encoding.
                payload=load_chunk_payload(source)
                source_layout='binary' if '.bin' in source.suffixes else 'delta' if isinstance(payload,dict) else 'snapshot'
                frames=decode_chunk(payload);expected=frame_digest(frames)
                for p in originals:
                    if p!=source and frame_digest(load_chunk(p))!=expected:raise ValueError('Replay copies differ: '+stem)
                target=playback/(stem+suffix);temp=playback/('.'+uuid.uuid4().hex+suffix)
                try:
                    if source==target:
                        kept=min(originals,key=lambda p:p.stat().st_size)
                        if any(file_digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during storage optimization')
                        for p in originals:
                            if p!=kept:p.unlink()
                    else:
                        if isinstance(payload,list):payload=encode_chunk(frames)
                        opener=lzma.open if compression=='xz' else gzip.open
                        options=dict(filters=[dict(id=lzma.FILTER_LZMA2,preset=6,dict_size=1<<20)]) if compression=='xz' else dict(compresslevel=6)
                        with opener(temp,'wb',**options) as stream:
                            writer=RecordWriter(stream);writer.write(dict(kind='replay',delta={k:v for k,v in payload.items() if k!='frames'}))
                            for packet in payload['frames']:writer.write(packet)
                            writer.write(dict(kind='replay_end',frames=len(payload['frames'])))
                        if frame_digest(load_chunk(temp))!=expected:raise ValueError('Binary replay verification failed: '+stem)
                        kept=publish_candidate(originals,hashes,temp,target)
                    size=kept.stat().st_size;before+=old;after+=size;replay_before+=old;replay_after+=size
                    converted+=int(size<old)
                    formats.add(source_layout if kept==source else 'binary' if '.bin' in kept.suffixes else 'delta' if isinstance(load_chunk_payload(kept),dict) else 'snapshot')
                    compressions.add('xz' if kept.suffix=='.xz' else 'gzip' if kept.suffix=='.gz' else 'none')
                    if progress and index==run['chunk_count']-1:progress(f"Verified replay run {run['run']}")
                finally:
                    if temp.exists():temp.unlink()
                del frames
        manifest['storage_format']=next(iter(formats),'binary') if len(formats)<=1 else 'mixed'
        manifest['storage_compression']=next(iter(compressions),'none') if len(compressions)<=1 else 'mixed'
        write_json(manifest_path,manifest)
    if checkpoints and cp.is_dir():
        stems=set()
        for p in cp.iterdir():
            if p.name.startswith('.'):continue
            for s in SUFFIXES:
                if p.name.endswith(s):stems.add(p.name[:-len(s)]);break
        for stem in sorted(stems):
            originals=copies(cp,stem);old=sum(p.stat().st_size for p in originals)
            target=cp/(stem+suffix);source=target if target in originals else originals[0]
            hashes={p:file_digest(p) for p in originals}
            state=read_checkpoint(source);expected=state_digest(state)
            for p in originals:
                if p!=source and state_digest(read_checkpoint(p))!=expected:raise ValueError('Checkpoint copies differ: '+stem)
            temp=cp/('.'+uuid.uuid4().hex+suffix)
            try:
                if source==target:
                    kept=min(originals,key=lambda p:p.stat().st_size)
                    if any(file_digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during storage optimization')
                    for p in originals:
                        if p!=kept:p.unlink()
                else:
                    write_checkpoint(temp,{k:v for k,v in state.items() if k!='brains'},state['brains'].items())
                    if state_digest(read_checkpoint(temp))!=expected:raise ValueError('Binary checkpoint verification failed: '+stem)
                    kept=publish_candidate(originals,hashes,temp,target)
                size=kept.stat().st_size;before+=old;after+=size;checkpoint_before+=old;checkpoint_after+=size
                converted+=int(size<old)
                if progress:progress('Verified checkpoint '+stem)
            finally:
                if temp.exists():temp.unlink()
            del state
    return dict(folder=str(folder),before_bytes=before,after_bytes=after,saved_bytes=before-after,
                files_reduced=converted,replay_before_bytes=replay_before,replay_after_bytes=replay_after,
                checkpoint_before_bytes=checkpoint_before,checkpoint_after_bytes=checkpoint_after)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--all',action='store_true',help='Optimize completed experiments under a results root')
    parser.add_argument('--only',choices=('replays','checkpoints','both'),default='both')
    parser.add_argument('--compression',choices=('xz','gzip'),default='xz')
    args=parser.parse_args()
    folders=sorted({p.parent.parent for p in args.path.glob('*/playback/recording.json')} | {p.parent for p in args.path.glob('*/checkpoints') if p.is_dir()}) if args.all else [args.path]
    if not folders:parser.error('No experiment found')
    for folder in folders:
        try:
            result=optimize_experiment(folder,args.only!='checkpoints',args.only!='replays',args.compression,
                                      lambda msg:print(msg,flush=True))
        except ValueError as error:
            if args.all and ('still writing' in str(error) or 'may still be writing' in str(error)):
                print('Skipped active experiment: '+str(folder),flush=True);continue
            raise
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
