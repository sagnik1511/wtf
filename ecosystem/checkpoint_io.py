"""Lossless portable numeric arrays and shared metadata for resume checkpoints.

Public loaders always return the original version-1 state. New compressed files
stream one brain at a time; plain JSON output stays compatible with old readers.
"""
from array import array
import base64
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ENCODING = 'wtf-checkpoint-v2'
TAGS = {'$ref', '$literal', '$f64', '$u32'}
SHARED = {'config', 'actions', 'model', 'signature'}


def dumps(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


class Encoder:
    def __init__(self):
        self.shared = []
        self.indices = {}

    def encode(self, value, key=None):
        if isinstance(value, (dict, list, tuple)) and (
            key in SHARED or isinstance(value, dict) and set(value)=={'signature', 'parameters'}
        ):
            fingerprint = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).digest()
            if fingerprint not in self.indices:
                # Model-shaped dictionaries would match again; bypass just this
                # node while recursively encoding its contents.
                encoded = self.contents(value)
                self.indices[fingerprint] = len(self.shared)
                self.shared.append(encoded)
            return {'$ref': self.indices[fingerprint]}
        return self.contents(value)

    def contents(self, value):
        if isinstance(value, dict):
            if TAGS.intersection(value):
                return {'$literal': [[k,self.encode(v,k)] for k,v in value.items()]}
            return {k:self.encode(v,k) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            code = None
            if len(value)>=16 and all(type(v) is float for v in value):
                if not all(math.isfinite(v) for v in value):raise ValueError('Non-finite checkpoint number')
                code, tag = 'd', '$f64'
            elif len(value)>=16 and all(type(v) is int and 0<=v<2**32 for v in value):
                code, tag = 'I', '$u32'
            if code:
                packed = array(code,value)
                if packed.itemsize != (8 if code=='d' else 4):
                    raise ValueError('Unsupported native numeric array width')
                if sys.byteorder!='little':packed.byteswap()
                return {tag:base64.b64encode(packed.tobytes()).decode('ascii')}
            return [self.encode(v) for v in value]
        return value


class Decoder:
    def __init__(self, shared):
        if not isinstance(shared,list):raise ValueError('Invalid checkpoint shared table')
        self.shared, self.decoded, self.resolving = shared, {}, set()

    def decode(self, value):
        if isinstance(value,list):return [self.decode(v) for v in value]
        if not isinstance(value,dict):return value
        tags = TAGS.intersection(value)
        if not tags:return {k:self.decode(v) for k,v in value.items()}
        if len(value)!=1:raise ValueError('Invalid checkpoint tag')
        tag = next(iter(tags));data = value[tag]
        if tag=='$ref':
            if type(data) is not int or not 0<=data<len(self.shared):raise ValueError('Invalid checkpoint reference')
            if data in self.resolving:raise ValueError('Cyclic checkpoint reference')
            if data not in self.decoded:
                self.resolving.add(data)
                try:self.decoded[data] = self.decode(self.shared[data])
                finally:self.resolving.remove(data)
            return copy.deepcopy(self.decoded[data])
        if tag=='$literal':return {k:self.decode(v) for k,v in data}
        if not isinstance(data,str):raise ValueError('Invalid checkpoint numeric data')
        raw = base64.b64decode(data,validate=True)
        code, width = ('d',8) if tag=='$f64' else ('I',4)
        if len(raw)%width:raise ValueError('Invalid checkpoint array length')
        packed = array(code)
        if packed.itemsize!=width:raise ValueError('Unsupported native numeric array width')
        packed.frombytes(raw)
        if sys.byteorder!='little':packed.byteswap()
        if code=='d' and not all(math.isfinite(v) for v in packed):raise ValueError('Non-finite checkpoint number')
        return packed.tolist()


def decode_checkpoint(payload):
    if payload.get('checkpoint_version')==1:return payload
    if payload.get('checkpoint_version')!=2 or payload.get('checkpoint_encoding')!=ENCODING:
        raise ValueError('Unsupported checkpoint encoding')
    decoder = Decoder(payload['shared'])
    state = decoder.decode(payload['metadata'])
    state['brains'] = {key:decoder.decode(value) for key,value in payload['brains'].items()}
    if state.get('checkpoint_version')!=1:raise ValueError('Invalid checkpoint metadata')
    return state


def read_checkpoint(path):
    path = Path(path)
    # Inspect the magic bytes, allowing older scripts to keep using .json paths
    # after conversion and supporting compressed legacy version-1 checkpoints.
    if not path.is_file() and path.suffix=='.json' and path.with_suffix('.json.gz').is_file():
        path = path.with_suffix('.json.gz')
    with path.open('rb') as source:
        compressed = source.read(2)==b'\x1f\x8b'
    opener = gzip.open if compressed else open
    with opener(path,'rt',encoding='utf-8') as handle:payload = json.load(handle)
    return decode_checkpoint(payload)


def write_checkpoint(path, metadata, brains):
    """Write metadata and an iterable of (id, detached brain state) atomically."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    compact = path.suffix=='.gz'
    encoder = Encoder() if compact else None
    try:
        opener = gzip.open if compact else open
        with opener(temporary,'wt',encoding='utf-8',**({'compresslevel':6} if compact else {})) as handle:
            if compact:
                handle.write('{"checkpoint_version":2,"checkpoint_encoding":'+dumps(ENCODING)+',"metadata":')
                handle.write(dumps(encoder.encode(metadata)))
                handle.write(',"brains":{')
            else:handle.write(dumps(metadata)[:-1]+',"brains":{')
            for index,(animal_id,state) in enumerate(brains):
                if index:handle.write(',')
                handle.write(dumps(str(animal_id))+':')
                handle.write(dumps(encoder.encode(state) if compact else state))
                del state  # Release this detached brain before requesting the next one.
            handle.write('}')
            if compact:handle.write(',"shared":'+dumps(encoder.shared))
            handle.write('}\n')
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()
