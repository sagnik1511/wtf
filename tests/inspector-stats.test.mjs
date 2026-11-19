import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  observationView,
  predictionView,
  predictionSamples,
  visibilityCells,
  ObservationOverlay,
  PredictionHistory,
} from '../viewer/inspector-stats.mjs';
import { terrainView } from '../viewer/terrain.mjs';

function features(version, k) {
  const result = [0.6, 0.9, 0, -1, 0, 1, 0, 0];
  for (let category = 0; category < 3; category++)
    for (let slot = 0; slot < k; slot++)
      result.push(
        slot / 10,
        category / 10,
        Math.hypot(slot, category) / 10,
        slot === 0 ? 1 : 0,
      );
  if (version >= 2)
    result.push(
      0.2,
      0.125,
      -0.25,
      0.03,
      0.1,
      0.01,
      0.2,
      -0.01,
      0,
      0,
      0.4,
      0.05,
    );
  if (version === 3) result.push(1, 0.9, 0.2, 1, 0.25, -0.25, 0.35, 1, 0, 1);
  return result;
}

test('Observation slots decode all versions and k without confusing terrain or mating offsets', () => {
  for (const version of [1, 2, 3])
    for (const k of [1, 4, 16]) {
      const values = features(version, k),
        copy = JSON.stringify(values),
        view = observationView(values, {
          observation_version: version,
          observation_top_k: k,
        });
      assert.equal(view.k, k);
      assert.equal(
        view.inputs,
        { 1: 20, 2: 32, 3: 42 }[version] + 12 * (k - 1),
      );
      assert.deepEqual(view.heading, [0, -1]);
      assert.deepEqual(view.blocked, [0, 1, 0, 0]);
      assert.equal(view.targets.filter((t) => t.category === 'food').length, 1);
      assert.equal(view.targets[0].distance, 0); // Presence, not nonzero distance, selects slots.
      if (version >= 2) {
        assert.equal(view.terrain.depth, 2);
        assert.deepEqual(view.terrain.flow, [0.5, -1]);
        assert.equal(view.terrain.elevation, 3);
        assert.equal(view.terrain.neighbors[1].climb, -1);
      }
      if (version === 3) {
        assert.equal(view.mating.sex, 'male');
        assert.equal(view.mating.pair, 1);
        assert.equal(view.mating.mateReady, 0);
        assert.equal(view.targets.at(-1).category, 'mate');
      }
      assert.equal(JSON.stringify(values), copy);
    }
});

test('Legacy widths work while unknown, mismatched and non-finite layouts stay unlabelled', () => {
  for (const version of [1, 2, 3])
    assert.equal(observationView(features(version, 1)).version, version);
  assert.equal(observationView(features(2, 4)), null);
  assert.equal(
    observationView(features(2, 1), { observation_version: 1 }),
    null,
  );
  assert.equal(observationView(undefined), null);
  assert.equal(observationView([NaN]), null);
  assert.equal(observationView(features(1, 1), { observation_top_k: 0 }), null);
});

test('Policy bars retain real probabilities, masks, top predictions and zero widths', () => {
  const view = predictionView({
    action_probabilities: { eat: 0.75, rest: 0.25, swim: 0 },
    action_mask: { swim: false },
  });
  assert.equal(view.policy, true);
  assert.equal(view.actions[0].width, 75);
  assert.equal(view.actions[0].best, true);
  assert.equal(view.actions[2].width, 0);
  assert.equal(view.actions[2].masked, true);
  assert.equal(view.zero, 0);
  assert.equal(view.max, 1);
});

test('Q graphs preserve signed returns and use a real zero axis rather than fake probabilities', () => {
  let view = predictionView({
    action_values: { attack: -2, rest: 2, swim: 0 },
  });
  assert.equal(view.policy, false);
  assert.equal(view.zero, 50);
  assert.equal(view.actions[0].left, 0);
  assert.equal(view.actions[0].width, 50);
  assert.equal(view.actions[1].left, 50);
  assert.equal(view.actions[1].value, 2);
  view = predictionView({ action_values: { rest: -1, attack: -5 } });
  assert.equal(view.zero, 100);
  assert.equal(view.actions[0].best, true);
  assert.equal(predictionView({}), null);
  assert.equal(predictionView({ action_values: { rest: NaN } }), null);
});

test('Prediction history is bounded, seek-safe, agent-specific and excludes carried/future samples', () => {
  const frames = Array.from({ length: 200 }, (_, tick) => ({
    tick,
    diagnostics: { 3: { tick, action_values: { rest: tick } } },
  }));
  const current = predictionSamples(frames, 80, 3);
  assert.equal(current.length, 64);
  assert.equal(current[0].tick, 17);
  assert.equal(current.at(-1).tick, 80);
  assert.deepEqual(
    predictionSamples(frames, 1, 3).map((p) => p.tick),
    [0, 1],
  );
  assert.deepEqual(predictionSamples(frames, 80, 4), []);
  frames[1].diagnostics['3'].tick = 0;
  assert.deepEqual(
    predictionSamples(frames, 1, 3).map((p) => p.tick),
    [0],
  );
});

test('Sight overlay matches Python geometry across maps, species, river banks and occluders', () => {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const script = `
import json
from ecosystem.config import load
from ecosystem.world import World
cases=[]
for name in ('baseline','river_crossing_strong_predator','highlands','enclosed_valleys','meandering_river','river_delta','alpine_pass','lake_islands'):
    c=load('environments/'+name+'.json')
    w=World(c,100)
    w._observation_context()
    for species,spec in c['species'].items():
        animal=next(a for a in w.animals.values() if a.species==species)
        for x,y in ((1,1),(w.width//2-2,w.height//2),(w.width-2,w.height-2)):
            animal.x,animal.y=x,y
            base=spec['body']['vision_radius']
            bonus=spec['body'].get('water_vision_bonus',0) if w.geography and w.geography.water else 0
            expected=[]
            for yy in range(w.height):
                for xx in range(w.width):
                    squared=(xx-x)**2+(yy-y)**2
                    if squared>(base+bonus)**2 or not w.visible((x,y),(xx,yy)):
                        continue
                    allowed=base+min(bonus,w._water_sight_cells((x,y),(xx,yy))) if bonus else base
                    if squared<=allowed**2:expected.append([xx,yy])
            cases.append(dict(name=name,animal=dict(id=animal.id,x=x,y=y,species=species),state=dict(width=w.width,height=w.height,species=c['species'],obstacles=list(w.obstacles)),terrain=c.get('terrain'),expected=expected))
print(json.dumps(cases))
`;
  const result = spawnSync(root + '.venv/bin/python', ['-c', script], {
    cwd: root,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  assert.equal(result.status, 0, result.stderr);
  for (const c of JSON.parse(result.stdout))
    assert.deepEqual(
      visibilityCells(c.state, c.animal, terrainView(c.terrain)),
      c.expected,
      `${c.name} ${c.animal.species} (${c.animal.x},${c.animal.y})`,
    );
});

test('Overlay cache reuses static sight, changes with position/geometry, and ignores moving animals', () => {
  const state = {
      width: 8,
      height: 8,
      obstacles: [],
      species: { deer: { body: { vision_radius: 2 } } },
    },
    animal = { id: 1, species: 'deer', x: 3, y: 3 };
  const overlay = new ObservationOverlay(),
    ctx = { fillRect() {} };
  overlay.draw(ctx, state, animal, null, 10, 10);
  const first = overlay.cells;
  state.animals = [{ id: 2, x: 4, y: 4 }];
  overlay.draw(ctx, state, animal, null, 10, 10);
  assert.equal(overlay.cells, first);
  animal.x++;
  overlay.draw(ctx, state, animal, null, 10, 10);
  assert.notEqual(overlay.cells, first);
  const moved = overlay.cells;
  state.obstacles = [[5, 3]];
  overlay.draw(ctx, state, animal, null, 10, 10);
  assert.notEqual(overlay.cells, moved);
  overlay.clear();
  assert.equal(overlay.cells, null);
});

test('Selected-agent history crosses chunks, drops future samples on seek, and resets by run/agent', () => {
  const history = new PredictionHistory();
  const sample = (tick) => ({
    tick,
    diagnostic: { action_probabilities: { rest: 0.5 } },
  });
  history.update('recording:1:3', [sample(0), sample(10)], 10);
  assert.deepEqual(
    history.update('recording:1:3', [sample(64)], 64).map((p) => p.tick),
    [0, 10, 64],
  );
  assert.deepEqual(
    history.update('recording:1:3', [sample(0)], 5).map((p) => p.tick),
    [0],
  );
  assert.deepEqual(
    history.update('recording:2:3', [sample(0)], 0).map((p) => p.tick),
    [0],
  );
  assert.deepEqual(history.update('recording:2:4', [], 0), []);
});

test('History retains at most 64 detached score vectors, not observation arrays or decoded frames', () => {
  const history = new PredictionHistory();
  const samples = Array.from({ length: 100 }, (_, tick) => ({
    tick,
    diagnostic: { features: Array(78).fill(0), action_values: { rest: tick } },
  }));
  const points = history.update('a', samples, 99);
  assert.equal(points.length, 64);
  assert.equal(points[0].tick, 36);
  assert.equal(points[0].diagnostic.features, undefined);
  samples[36].diagnostic.action_values.rest = 999;
  assert.equal(points[0].diagnostic.action_values.rest, 36);
});

test('Sample collection validates finite scores without constructing bar or mask data', () => {
  const diagnostic = {
    tick: 0,
    action_values: { rest: -2, attack: NaN },
    get action_mask() {
      throw new Error('Sample collection must not build bars');
    },
  };
  const frames = [
    { tick: 0, diagnostics: { 3: diagnostic } },
    { tick: 1, diagnostics: { 3: { tick: 1, action_values: { rest: NaN } } } },
    {
      tick: 2,
      diagnostics: {
        3: { tick: 2, action_probabilities: {}, action_values: { rest: 1 } },
      },
    },
    {
      tick: 3,
      diagnostics: { 3: { tick: 3, action_probabilities: { rest: 0 } } },
    },
  ];
  assert.deepEqual(
    predictionSamples(frames, 3, 3).map((p) => p.tick),
    [0, 3],
  );
});
