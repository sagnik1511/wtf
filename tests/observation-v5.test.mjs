import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { observationView } from '../viewer/inspector-stats.mjs';
import { inputLabels } from '../viewer/network-structure.mjs';
import { trackingSample, trackingLabels } from '../viewer/tracking.mjs';
import { worldVector, worldScent } from '../viewer/observation-v5.mjs';

test('V5 Python vectors project back to V4 world diagrams, histories, flow and scent', () => {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const script = `
import copy,json
from wtf.config import load,validate
from wtf.simulation.world import World
from wtf.inspection.explainability import feature_names
from wtf.learning.brains import model_signature
c=load('environments/seaside_150_v5.json');c['scent_trails']={'enabled':True,'lifetime_steps':20}
for b in c['brains'].values():b.update(movement_history_steps=2,scent_observation={'mode':'blended','radius':2})
old=copy.deepcopy(c)
for b in old['brains'].values():b['observation_version']=4
left,right=World(validate(old),100),World(validate(c),100)
rows=[]
for heading in range(4):
 for w in (left,right):
  for a in w.animals.values():a.heading=heading
  w.step({i:'rest' for i in w.animals})
 a4,a5=left.observe_all(),right.observe_all()
 for species in c['species']:
  animal=next(a for a in right.animals.values() if a.species==species)
  b=c['brains'][c['species'][species]['brain']];b4=old['brains'][old['species'][species]['brain']]
  rows.append(dict(old=a4[animal.id],new=a5[animal.id],template=b,oldTemplate=b4,names=feature_names(model_signature(b,c['species'][species]['allowed_actions'])),animal=dict(id=animal.id,x=animal.x,y=animal.y),tick=right.tick))
print(json.dumps(rows))`;
  const result = spawnSync(root + '.venv/bin/python', ['-c', script], {
    cwd: root,
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
  });
  assert.equal(result.status, 0, result.stderr);
  for (const row of JSON.parse(result.stdout)) {
    const before = observationView(row.old, row.oldTemplate),
      after = observationView(row.new, row.template);
    assert(after);
    for (const key of ['targets', 'terrain', 'mating', 'tracking', 'social'])
      assert.deepEqual(
        JSON.parse(JSON.stringify(after[key])),
        JSON.parse(JSON.stringify(before[key])),
        key,
      );
    assert.deepEqual(inputLabels(row.template, row.new.length), row.names);
    assert.equal(Object.keys(after.opportunities).length, 3);
    const sample = trackingSample(
      [{ tick: row.tick, animals: [row.animal] }],
      0,
      row.animal.id,
      { tick: row.tick, features: row.new },
      row.template,
      11,
    );
    assert.deepEqual(sample.tracks, after.tracking.tracks);
  }
});

test('V5 cardinal projections preserve scent pattern and do not infer a new observation', () => {
  assert.deepEqual(worldVector(1, -1, [1, 0]), [1, -1]);
  assert.deepEqual(
    worldScent([0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 0]),
    [0, 7, 8, 1, 2, 3, 4, 5, 6],
  );
  assert.equal(
    observationView(Array(45).fill(0), { observation_version: 5 }),
    null,
  );
});

test('V5 scent labels preserve arbitrary species names', () => {
  const labels = trackingLabels({
    observation_version: 5,
    scent_observation: { species: ['red dx'], radius: 2 },
  });
  assert(labels.includes('Scent red dx female front'));
  assert(labels.includes('Scent red dx male right'));
});
