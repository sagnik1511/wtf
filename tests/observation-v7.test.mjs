import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { observationView } from '../viewer/inspector-stats.mjs';
import { inputLabels, compactLayout } from '../viewer/input-layout.mjs';
import { trackingSample } from '../viewer/tracking.mjs';

test('V7 Python labels, side channels and replay views agree across headings and exact species', () => {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const result = spawnSync(
    root + '.venv/bin/python',
    [
      '-c',
      `
import json
from wtf.config import load
from wtf.experiment.runner import Experiment
from wtf.inspection.explainability import feature_names
from wtf.learning.brains import model_signature
c=load('environments/seaside_150_v7.json')
for s in c['species'].values():s['initial_population']=3
c['experiment']['runs']=1
x=Experiment(c);rows=[]
for heading in range(4):
 for a in x.world.animals.values():a.heading=heading
 x.step()
 for a in x.world.animals.values():
  t=c['brains'][c['species'][a.species]['brain']]
  o=x.observations[a.id]
  rows.append(dict(features=o,names=feature_names(model_signature(t,c['species'][a.species]['allowed_actions'])),template=t,animal=dict(id=a.id,x=a.x,y=a.y),tick=x.world.tick))
print(json.dumps(rows))
`,
    ],
    { cwd: root, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
  for (const row of JSON.parse(result.stdout)) {
    assert.deepEqual(inputLabels(row.template, row.features.length), row.names);
    const view = observationView(row.features, row.template);
    assert.equal(view.version, 7);
    assert.equal(view.inputs, row.features.length);
    assert.equal(view.compactScents.length, 1);
    assert.equal(view.tracking.scent.length, 0);
    assert.equal(
      view.compactScents[0].strength,
      row.features[compactLayout(row.template).scentStart + 2],
    );
    const sample = trackingSample(
      [{ tick: row.tick, animals: [row.animal] }],
      0,
      row.animal.id,
      { tick: row.tick, features: row.features },
      row.template,
      10,
    );
    assert.ok(sample);
    assert.equal(sample.scent.length, 0);
    assert.ok(sample.tracks.every((t) => t.history.every((p) => !p.seen)));
    assert.equal(observationView(row.features.slice(1), row.template), null);
  }
});
