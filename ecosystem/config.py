"""Strict environment validation. No third-party packages required."""
import json
import math
import re
from pathlib import Path

ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'rest', 'eat', 'attack')


def fail(path, message):
    raise ValueError(f'{path}: {message}')


def obj(value, path, fields, optional=()):
    if not isinstance(value, dict):
        fail(path, 'must be an object')
    for key in value:
        if key not in fields and key not in optional:
            fail(f'{path}.{key}', 'unknown field')
    for key in fields:
        if key not in value:
            fail(f'{path}.{key}', 'required field')


def num(value, path, low=0, high=float('inf'), integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high or (integer and not isinstance(value, int)):
        fail(path, f'must be {"an integer" if integer else "a number"} between {low} and {high}')


def validate(c):
    obj(c, 'config', ('schema_version', 'name', 'description', 'seed', 'world', 'obstacles', 'resources', 'species', 'brains', 'experiment'), optional=('recording',))
    if type(c['schema_version']) is not int or c['schema_version'] != 1:
        fail('schema_version', 'only version 1 is supported')
    if not isinstance(c['name'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', c['name']):
        fail('name', 'use letters, digits, underscores or hyphens')
    if not isinstance(c['description'], str):
        fail('description', 'must be a string')
    num(c['seed'], 'seed', 0, 2**32 - 1, True)
    w = c['world']
    obj(w, 'world', ('width', 'height', 'boundary', 'max_steps'))
    for k in ('width', 'height'):
        num(w[k], f'world.{k}', 3, 256, True)
    num(w['max_steps'], 'world.max_steps', 1, 10000000, True)
    if w['boundary'] != 'closed':
        fail('world.boundary', 'must be closed')
    o = c['obstacles']
    obj(o, 'obstacles', ('generation', 'occupied_fraction', 'explicit_cells'))
    if o['generation'] not in ('scattered', 'explicit'):
        fail('obstacles.generation', 'must be scattered or explicit')
    num(o['occupied_fraction'], 'obstacles.occupied_fraction', 0, .8)
    if not isinstance(o['explicit_cells'], list):
        fail('obstacles.explicit_cells', 'must be an array')
    cells = set()
    for i, p in enumerate(o['explicit_cells']):
        if not isinstance(p, list) or len(p) != 2:
            fail(f'obstacles.explicit_cells.{i}', 'must be [x, y]')
        num(p[0], f'obstacles.explicit_cells.{i}.x', 0, w['width'] - 1, True)
        num(p[1], f'obstacles.explicit_cells.{i}.y', 0, w['height'] - 1, True)
        if tuple(p) in cells:
            fail('obstacles.explicit_cells', 'duplicate cell')
        cells.add(tuple(p))
    if o['generation'] == 'scattered' and cells:
        fail('obstacles.explicit_cells', 'must be empty for scattered generation')
    r = c['resources']
    obj(r, 'resources', ('patch_count', 'capacity_per_patch', 'initial_quantity_per_patch', 'growth_per_step'))
    num(r['patch_count'], 'resources.patch_count', 0, w['width'] * w['height'], True)
    num(r['capacity_per_patch'], 'resources.capacity_per_patch', .001)
    num(r['initial_quantity_per_patch'], 'resources.initial_quantity_per_patch', 0, r['capacity_per_patch'])
    num(r['growth_per_step'], 'resources.growth_per_step')
    if not isinstance(c['brains'], dict) or not c['brains']:
        fail('brains', 'at least one template required')
    for name, b in c['brains'].items():
        p = f'brains.{name}'
        obj(b, p, ('type', 'hidden_layers', 'learning_rate', 'discount_factor', 'batch_size', 'replay_capacity', 'learning_starts_after', 'train_every_steps', 'target_update_every_training_steps', 'exploration'))
        if b['type'] != 'q_network':
            fail(p + '.type', 'must be q_network')
        if not isinstance(b['hidden_layers'], list) or len(b['hidden_layers']) != 1:
            fail(p + '.hidden_layers', 'exactly one hidden layer required')
        num(b['hidden_layers'][0], p + '.hidden_layers.0', 1, 256, True)
        num(b['learning_rate'], p + '.learning_rate', .000001, 1)
        num(b['discount_factor'], p + '.discount_factor', 0, 1)
        for k in ('batch_size', 'replay_capacity', 'learning_starts_after', 'train_every_steps', 'target_update_every_training_steps'):
            num(b[k], f'{p}.{k}', 1, 1000000, True)
        if max(b['batch_size'], b['learning_starts_after']) > b['replay_capacity']:
            fail(p, 'batch size and learning start must not exceed replay capacity')
        e = b['exploration']
        obj(e, p + '.exploration', ('initial', 'minimum', 'decay_per_decision'))
        for k in e:
            num(e[k], f'{p}.exploration.{k}', 0, 1)
        if e['minimum'] > e['initial']:
            fail(p + '.exploration.minimum', 'must not exceed initial')
    if not isinstance(c['species'], dict) or not c['species']:
        fail('species', 'at least one species required')
    population = 0
    for name, s in c['species'].items():
        p = f'species.{name}'
        obj(s, p, ('initial_population', 'diet', 'allowed_actions', 'body', 'brain', 'rewards'))
        num(s['initial_population'], p + '.initial_population', 0, 10000, True)
        population += s['initial_population']
        if s['diet'] not in ('resource', 'prey'):
            fail(p + '.diet', 'must be resource or prey')
        actions = s['allowed_actions']
        if not isinstance(actions, list) or not actions or any(not isinstance(a, str) or a not in ACTIONS for a in actions) or len(set(actions)) != len(actions):
            fail(p + '.allowed_actions', 'must contain unique supported actions')
        required, forbidden = ('eat', 'attack') if s['diet'] == 'resource' else ('attack', 'eat')
        if required not in actions or forbidden in actions:
            fail(p + '.allowed_actions', 'feeding action incompatible with diet')
        if not isinstance(s['brain'], str) or s['brain'] not in c['brains']:
            fail(p + '.brain', 'unknown brain template')
        fields = ['initial_energy', 'max_energy', 'max_health', 'vision_radius', 'living_cost', 'movement_cost', 'starvation_damage']
        fields += ['bite_size', 'energy_per_food_unit'] if s['diet'] == 'resource' else ['attack_damage', 'attack_cost', 'energy_per_kill']
        obj(s['body'], p + '.body', fields)
        for k in fields:
            num(s['body'][k], f'{p}.body.{k}')
        for k in ('max_energy', 'max_health'):
            num(s['body'][k], f'{p}.body.{k}', .001)
        num(s['body']['vision_radius'], p + '.body.vision_radius', 1, 256, True)
        if s['body']['initial_energy'] > s['body']['max_energy']:
            fail(p + '.body.initial_energy', 'exceeds maximum')
        obj(s['rewards'], p + '.rewards', ('per_energy_gained', 'per_damage_taken', 'death'))
        for k, value in s['rewards'].items():
            num(value, f'{p}.rewards.{k}', -100000, 100000)
    if not population:
        fail('species', 'total population must be positive')
    blocked = len(cells) if o['generation'] == 'explicit' else int(w['width'] * w['height'] * o['occupied_fraction'])
    free = w['width'] * w['height'] - blocked
    if max(population, r['patch_count']) > free:
        fail('world', 'insufficient free cells for animals or resources')
    e = c['experiment']
    obj(e, 'experiment', ('runs', 'retain_brains_between_runs', 'world_seed_policy', 'record_metrics_every_steps', 'checkpoint_every_runs'))
    for k in ('runs', 'record_metrics_every_steps', 'checkpoint_every_runs'):
        num(e[k], f'experiment.{k}', 1, 1000000, True)
    if type(e['retain_brains_between_runs']) is not bool:
        fail('experiment.retain_brains_between_runs', 'must be boolean')
    if e['world_seed_policy'] not in ('fixed', 'increment'):
        fail('experiment.world_seed_policy', 'must be fixed or increment')
    if 'recording' in c:
        r = c['recording']
        obj(r, 'recording', ('enabled', 'frame_every_steps', 'frames_per_chunk', 'diagnostics_every_steps'))
        if type(r['enabled']) is not bool:
            fail('recording.enabled', 'must be boolean')
        num(r['frame_every_steps'], 'recording.frame_every_steps', 1, 1000000, True)
        num(r['frames_per_chunk'], 'recording.frames_per_chunk', 1, 256, True)
        num(r['diagnostics_every_steps'], 'recording.diagnostics_every_steps', 0, 1000000, True)
    return c


def load(path):
    return validate(json.loads(Path(path).read_text()))
