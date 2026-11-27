import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { observationView } from '../viewer/inspector-stats.mjs';
import { inputLabels, semanticLayout } from '../viewer/input-layout.mjs';
import { decodeSoft } from '../viewer/soft-observations.mjs';
import { trackingSample } from '../viewer/tracking.mjs';

function close(a, b, path = '') {
  if (typeof a === 'number') {
    assert(Math.abs(a - b) < 1e-11, `${path}: ${a} vs ${b}`);
    return;
  }
  if (a && typeof a === 'object') {
    assert.deepEqual(Object.keys(a), Object.keys(b), path);
    for (const k of Object.keys(a)) close(a[k], b[k], path + '.' + k);
    return;
  }
  assert.equal(a, b, path);
}

test('V6 labels and representative views match Python across directions and optional blocks', () => {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const script = `
import copy,json
from wtf.config import load
from wtf.simulation.world import World
from wtf.simulation.soft_observations import decode_soft
from wtf.inspection.explainability import feature_names
from wtf.learning.brains import model_signature
c=load('environments/seaside_150_v6.json');c['scent_trails']={'enabled':True,'lifetime_steps':20}
for b in c['brains'].values():b.update(movement_history_steps=2,scent_observation={'mode':'blended','radius':2})
w=World(c,100);rows=[]
for heading in range(4):
 for a in w.animals.values():a.heading=heading
 w.step({i:'rest' for i in w.animals})
 values=w.observe_all()
 for species in c['species']:
  a=next(a for a in w.animals.values() if a.species==species);template=c['brains'][c['species'][species]['brain']]
  rows.append(dict(features=values[a.id],raw=decode_soft(values[a.id],template),template=template,names=feature_names(model_signature(template,c['species'][species]['allowed_actions'])),animal=dict(id=a.id,x=a.x,y=a.y),tick=w.tick))
print(json.dumps(rows))`;
  const result = spawnSync(root + '.venv/bin/python', ['-c', script], {
    cwd: root,
    encoding: 'utf8',
    maxBuffer: 24 * 1024 * 1024,
  });
  assert.equal(result.status, 0, result.stderr);
  for (const row of JSON.parse(result.stdout)) {
    const layout = semanticLayout(row.template);
    assert.deepEqual(inputLabels(row.template, row.features.length), row.names);
    close(decodeSoft(row.features, layout), row.raw);
    const view = observationView(row.features, row.template),
      raw = observationView(row.raw, {
        ...row.template,
        observation_version: 5,
      });
    assert(view);
    assert.equal(view.version, 6);
    assert(view.semantic.length > 0);
    for (const key of [
      'targets',
      'terrain',
      'tracking',
      'social',
      'mating',
      'opportunities',
    ])
      close(view[key], raw[key], key);
    const sample = trackingSample(
      [{ tick: row.tick, animals: [row.animal] }],
      0,
      row.animal.id,
      { tick: row.tick, features: row.features },
      row.template,
      11,
    );
    close(sample.tracks, raw.tracking.tracks);
  }
});

test('V6 reader rejects invalid knot metadata and incompatible vectors', () => {
  assert.equal(observationView([], { observation_version: 6 }), null);
  assert.equal(
    observationView(Array(130).fill(0), {
      observation_version: 6,
      soft_categories: { distance: [0, 0, 1] },
    }),
    null,
  );
});

test('V6 cached layout owns immutable knots independently of caller metadata', () => {
  const template = {
    observation_version: 6,
    soft_categories: { distance: [0, 0.3, 1] },
  };
  const first = semanticLayout(template);
  template.soft_categories.distance[1] = 0.2;
  const same = semanticLayout({
    observation_version: 6,
    soft_categories: { distance: [0, 0.3, 1] },
  });
  const position = same.fields.find((f) => f.kind === 'position');
  assert.equal(position.knots[1], 0.3);
  assert.equal(first, same);
  assert.throws(() => {
    position.knots[1] = 0.4;
  }, TypeError);
});
