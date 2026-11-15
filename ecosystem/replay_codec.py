"""Chunk-local keyframes and lossless sparse field updates; no simulation logic."""
import copy
import math

CODEC = 'wtf-delta-v1'
ANIMAL_FIELDS = ['id','species','x','y','heading','energy','health','age','last_action',
                 'last_reward','parent_id','generation','last_reproduced_age','kills',
                 'energy_gained','statistics_since_age']
RESOURCE_FIELDS = ['x','y','quantity']


def same(a, b):
    # JSON numbers distinguish integer/float representations and signed zero.
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a == b and (a != 0 or math.copysign(1, a) == math.copysign(1, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(v, b[k]) for k,v in a.items())
    if isinstance(a, (list,tuple)):
        return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def mapping_delta(old, new):
    patch = {}
    changed = {k:v for k,v in new.items() if k not in old or not same(old[k],v)}
    removed = [k for k in old if k not in new]
    if changed:patch['set'] = changed
    if removed:patch['drop'] = removed
    return patch


def entity_key(record, kind):
    return record['id'] if kind == 'animals' else (record['x'], record['y'])


def normalize_key(key, kind):
    return key if kind == 'animals' else tuple(key)


def entity_delta(old, new, fields, kind):
    previous = {entity_key(r,kind):r for r in old}
    current = {entity_key(r,kind):r for r in new}
    if len(previous) != len(old) or len(current) != len(new):
        raise ValueError('Duplicate replay entity keys')
    added, replaced, changed = [], [], []
    for key,record in current.items():
        prior = previous.get(key)
        if prior is None:
            added.append(record)
        elif not same(prior,record):
            if prior.keys() != record.keys() or any(k not in fields for k in record):
                replaced.append(record)
            else:
                mask,values = 0,[]
                for i,field in enumerate(fields):
                    if field in record and not same(prior[field],record[field]):
                        mask |= 1 << i
                        values.append(record[field])
                changed.append([key,mask,values])
    removed = [key for key in previous if key not in current]
    result = {}
    for name,value in [('add',added),('replace',replaced),('update',changed),('drop',removed)]:
        if value:result[name] = value
    expected = [key for key in previous if key in current] + [key for key in current if key not in previous]
    if list(current) != expected:result['order'] = list(current)
    return result


class DeltaEncoder:
    """Retain one previous frame, resetting the baseline at every chunk."""
    def __init__(self):
        self.previous = None

    @staticmethod
    def header():
        return dict(codec=CODEC,animal_fields=ANIMAL_FIELDS,resource_fields=RESOURCE_FIELDS)

    def encode(self, frame):
        previous,self.previous = self.previous,frame
        if previous is None:
            return frame
        excluded = {kind for kind in ('animals','resources')
                    if isinstance(previous.get(kind),list) and isinstance(frame.get(kind),list)}
        if isinstance(previous.get('metrics'),dict) and isinstance(frame.get('metrics'),dict):
            excluded.add('metrics')
        result = mapping_delta({k:v for k,v in previous.items() if k not in excluded},
                               {k:v for k,v in frame.items() if k not in excluded})
        if 'metrics' in excluded:
            metrics = mapping_delta(previous['metrics'],frame['metrics'])
            if metrics:result['metrics'] = metrics
        for kind,fields in [('animals',ANIMAL_FIELDS),('resources',RESOURCE_FIELDS)]:
            if kind not in excluded:continue
            delta = entity_delta(previous[kind],frame[kind],fields,kind)
            if delta:result[kind] = delta
        return result


def apply_mapping(target, patch):
    for key in patch.get('drop',[]):target.pop(key,None)
    target.update(copy.deepcopy(patch.get('set',{})))


def apply_entities(records, patch, fields, kind):
    entities = {entity_key(r,kind):r for r in records}
    for key in patch.get('drop',[]):
        entities.pop(normalize_key(key,kind))
    for name in ('replace','add'):
        for record in patch.get(name,[]):
            entities[entity_key(record,kind)] = copy.deepcopy(record)
    for key,mask,values in patch.get('update',[]):
        if type(mask) is not int or mask < 0 or mask >= 2**len(fields) or bin(mask).count('1') != len(values):
            raise ValueError('Invalid replay field mask')
        record = entities[normalize_key(key,kind)]
        value_index = 0
        for i,field in enumerate(fields):
            if mask & (1 << i):
                record[field] = copy.deepcopy(values[value_index]);value_index += 1
    if 'order' in patch:
        order = [normalize_key(k,kind) for k in patch['order']]
        if len(order) != len(entities) or set(order) != entities.keys():
            raise ValueError('Invalid replay entity order')
        return [entities[key] for key in order]
    return list(entities.values())


def iter_decode(payload):
    if isinstance(payload,list):
        yield from payload
        return
    if not isinstance(payload,dict) or payload.get('codec') != CODEC:
        raise ValueError('Unsupported replay chunk encoding')
    fields = {'animals':payload['animal_fields'],'resources':payload['resource_fields']}
    for table in fields.values():
        if not isinstance(table,list) or len(table)>52 or any(not isinstance(k,str) for k in table) or len(set(table))!=len(table):
            raise ValueError('Invalid replay field table')
    if not isinstance(payload.get('frames'),list):
        raise ValueError('Invalid replay packets')
    previous = None
    for packet in payload['frames']:
        if not isinstance(packet,dict):raise ValueError('Invalid replay frame')
        if previous is None:
            current = copy.deepcopy(packet)
        else:
            overwritten = set(packet.get('drop',[])) | packet.get('set',{}).keys()
            current = {k:copy.deepcopy(v) for k,v in previous.items() if k not in overwritten}
            apply_mapping(current,packet)
            if 'metrics' in packet:apply_mapping(current.setdefault('metrics',{}),packet['metrics'])
            for kind in fields:
                if kind in packet:current[kind] = apply_entities(current.get(kind,[]),packet[kind],fields[kind],kind)
        yield current
        previous = current


def decode_chunk(payload):
    return list(iter_decode(payload))


def encode_chunk(frames):
    encoder = DeltaEncoder()
    return dict(encoder.header(),frames=[encoder.encode(frame) for frame in frames])
