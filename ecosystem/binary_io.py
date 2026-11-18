"""Versioned, lossless binary records: typed columns, numeric arrays and symbols.

No pickle, JSON payloads, optional packages, or whole-file buffering. One record
(frame or controller) is buffered at a time. A symbol table resets per file.
"""
from . import kernels
from array import array
import io
import math
import struct

MAGIC = b'WTFB\x01'
MAX_RECORD = 256 * 1024 * 1024
MAX_SAFE_INTEGER = 2**53 - 1


def _python_varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return result


def varint(value):
    if kernels.available("varint") and 0 <= value <= 2**64-1:
        return kernels.native.varint(value)
    return _python_varint(value)


class Encoder:
    def __init__(self):
        self.symbols = {}
        self._varint = kernels.native.varint if kernels.available('varint') else _python_varint

    def encode(self, value):
        output = bytearray()
        encode_varint = self._varint
        def put(v, depth=0):
            if depth > 128:raise ValueError('Binary nesting too deep')
            if v is None:output.append(0)
            elif v is False:output.append(1)
            elif v is True:output.append(2)
            elif type(v) is int:
                if abs(v)>MAX_SAFE_INTEGER:
                    n=(v.bit_length()+8)//8
                    if n>128:raise ValueError('Binary integer too large')
                    output.append(14);output.extend(encode_varint(n));output.extend(v.to_bytes(n,'little',signed=True))
                else:
                    output.append(3);output.extend(encode_varint(v*2 if v>=0 else -v*2-1))
            elif type(v) is float:
                if not math.isfinite(v):raise ValueError('Non-finite binary number')
                output.append(4);output.extend(struct.pack('<d',v))
            elif isinstance(v,str):
                if v in self.symbols:
                    output.append(6);output.extend(encode_varint(self.symbols[v]))
                else:
                    self.symbols[v]=len(self.symbols)
                    data=v.encode('utf-8');output.append(5);output.extend(encode_varint(len(data)));output.extend(data)
            elif isinstance(v,dict):
                output.append(8);output.extend(encode_varint(len(v)))
                for key,item in v.items():
                    if not isinstance(key,str):raise ValueError('Binary map keys must be strings')
                    put(key,depth+1);put(item,depth+1)
            elif isinstance(v,(list,tuple)):
                n=len(v)
                if n>=4 and all(type(item) is float for item in v):
                    if not all(math.isfinite(item) for item in v):raise ValueError('Non-finite binary array')
                    # Narrow only when every float is exactly representable;
                    # no replay/legacy-checkpoint quantization is allowed.
                    try:
                        narrowed = array('f', v)
                        exact = narrowed.itemsize == 4 and all(a == b for a, b in zip(narrowed, v))
                    except OverflowError:
                        exact = False
                    code, tag = ('f', 15) if exact else ('d', 10)
                    output.append(tag);output.extend(encode_varint(n));output.extend(struct.pack('<'+str(n)+code,*v))
                elif n>=4 and all(type(item) is int and abs(item)<=MAX_SAFE_INTEGER for item in v):
                    varbytes = sum(max(1, ((item*2 if item>=0 else -item*2-1).bit_length()+6)//7) for item in v)
                    minimum, maximum = min(v), max(v)
                    width = 1 if maximum < 256 else 2 if maximum < 65536 else 4
                    if minimum >= 0 and maximum < 2**32 and width*n < varbytes:
                        tag, code = {1:(16,'B'), 2:(17,'H'), 4:(12,'I')}[width]
                        output.append(tag);output.extend(encode_varint(n));output.extend(struct.pack('<'+str(n)+code,*v))
                    else:
                        output.append(9);output.extend(encode_varint(n))
                        if kernels.available('integer_array'):
                            output.extend(kernels.native.integer_array(v))
                        else:
                            for item in v:
                                output.extend(encode_varint(item*2 if item>=0 else -item*2-1))
                elif n>=2 and isinstance(v[0],dict) and v[0] and all(isinstance(row,dict) and list(row)==list(v[0]) for row in v):
                    # Known-shaped entity/event records become columns, with one
                    # field table instead of a dictionary per entity.
                    output.append(11);output.extend(encode_varint(n));put(list(v[0]),depth+1)
                    for key in v[0]:put([row[key] for row in v],depth+1)
                else:
                    output.append(7);output.extend(encode_varint(n))
                    for item in v:put(item,depth+1)
            else:raise ValueError('Unsupported binary value: '+type(v).__name__)
        put(value)
        if len(output)>MAX_RECORD:raise ValueError('Binary record too large')
        return output


class Decoder:
    def __init__(self):
        self.symbols=[]

    def decode(self, data):
        view=memoryview(data);position=0
        def take(n):
            nonlocal position
            if n<0 or position+n>len(view):raise ValueError('Truncated binary value')
            result=view[position:position+n];position+=n;return result
        def uint():
            value=0
            for shift in range(0,56,7):
                byte=take(1)[0];value+=(byte & 127)<<shift
                if not byte & 128:
                    if value>2*MAX_SAFE_INTEGER+1:raise ValueError('Invalid binary integer')
                    return value
            raise ValueError('Invalid binary varint')
        def length():
            n=uint()
            if n>MAX_RECORD:raise ValueError('Binary collection too large')
            return n
        def get(depth=0):
            nonlocal position
            if depth>128:raise ValueError('Binary nesting too deep')
            tag=take(1)[0]
            if tag==0:return None
            if tag==1:return False
            if tag==2:return True
            if tag==3:
                n=uint();return n//2 if n%2==0 else -(n//2)-1
            if tag==4:
                v=struct.unpack('<d',take(8))[0]
                if not math.isfinite(v):raise ValueError('Non-finite binary number')
                return v
            if tag==5:
                value=bytes(take(length())).decode('utf-8');self.symbols.append(value);return value
            if tag==6:
                index=uint()
                if index>=len(self.symbols):raise ValueError('Invalid binary symbol')
                return self.symbols[index]
            if tag==7:return [get(depth+1) for _ in range(length())]
            if tag==8:
                result={}
                for _ in range(length()):
                    key=get(depth+1)
                    if not isinstance(key,str) or key in result:raise ValueError('Invalid binary map key')
                    result[key]=get(depth+1)
                return result
            if tag==9:
                n=length()
                if kernels.available('decode_integers'):
                    values,position=kernels.native.decode_integers(view,position,n)
                    return values
                result=[]
                for _ in range(n):
                    n=uint();result.append(n//2 if n%2==0 else -(n//2)-1)
                return result
            if tag in (10,15):
                width,code = (8,'d') if tag==10 else (4,'f')
                n=length();values=list(struct.unpack('<'+str(n)+code,take(n*width)))
                if not all(math.isfinite(v) for v in values):raise ValueError('Non-finite binary array')
                return values
            if tag==14:
                n=length()
                if not 0<n<=128:raise ValueError('Binary integer too large')
                return int.from_bytes(take(n),'little',signed=True)
            if tag in (12,16,17):
                width,code = {12:(4,'I'),16:(1,'B'),17:(2,'H')}[tag]
                n=length();return list(struct.unpack('<'+str(n)+code,take(n*width)))
            if tag==11:
                n=length();fields=get(depth+1)
                if not isinstance(fields,list) or not fields or any(not isinstance(k,str) for k in fields) or len(set(fields))!=len(fields):raise ValueError('Invalid binary column fields')
                # A table cannot legitimately have more rows than encoded bytes.
                if n>len(view):raise ValueError('Invalid binary row count')
                columns=[get(depth+1) for _ in fields]
                if any(not isinstance(c,list) or len(c)!=n for c in columns):raise ValueError('Invalid binary column length')
                return [dict(zip(fields,values)) for values in zip(*columns)]
            raise ValueError('Unknown binary tag')
        result=get()
        if position!=len(view):raise ValueError('Trailing binary record data')
        return result


class RecordWriter:
    def __init__(self, stream):
        self.stream=stream;self.encoder=Encoder();stream.write(MAGIC)

    def write(self, value):
        data=self.encoder.encode(value)
        self.stream.write(struct.pack('<I',len(data)));self.stream.write(data)


def records(stream):
    if stream.read(len(MAGIC))!=MAGIC:raise ValueError('Unsupported binary file version')
    decoder=Decoder()
    while True:
        size=stream.read(4)
        if not size:return
        if len(size)!=4:raise ValueError('Truncated binary record size')
        n=struct.unpack('<I',size)[0]
        if not 0<n<=MAX_RECORD:raise ValueError('Invalid binary record size')
        data=stream.read(n)
        if len(data)!=n:raise ValueError('Truncated binary record')
        yield decoder.decode(data)


def replay_payload(data):
    values=records(io.BytesIO(data));header=next(values,None)
    if not isinstance(header,dict) or header.get('kind')!='replay':raise ValueError('Invalid binary replay header')
    frames=list(values)
    footer=frames.pop() if frames else None
    if not isinstance(footer,dict) or footer.get('kind')!='replay_end' or footer.get('frames')!=len(frames):raise ValueError('Missing or invalid binary replay footer')
    payload=header['delta'];payload['frames']=frames
    return payload
