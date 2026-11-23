import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CollectiveTotals,
  CollectiveHistory,
} from '../viewer/collective-history.mjs';
import { Playback } from '../viewer/playback.mjs';

const template = {
  observation_version: 1,
  social_observation: { history_steps: 4, max_neighbors: 8 },
  consequence_observation: true,
};
const config = {
  brains: { test: template },
  species: {
    prey: { brain: 'test', allowed_actions: ['eat', 'rest'] },
    predator: { brain: 'test', allowed_actions: ['attack', 'rest'] },
  },
  world: { width: 8, height: 8 },
  resources: { capacity_per_patch: 20 },
  consequence_learning: { update_schedule: 'episode' },
};
const diagnostic = (tick, harm, confidence = 1) => ({
  tick,
  features: [
    ...Array(20).fill(0),
    ...Array.from({ length: 18 }, () => [confidence, harm, 0, 0]).flat(),
  ],
});
const animal = (id, species = 'prey') => ({ id, species, x: id, y: 1 });
const frame = (tick, animals = [animal(0)], diagnostics) => ({
  tick,
  animals,
  diagnostics,
  resources: [],
  events: [],
  metrics: { food: 0, populations: { prey: animals.length } },
});
const harm = (scope) => scope.collective[0].harm;

test('Time averages weight elapsed steps, hold only past samples, and leave raw vectors intact', () => {
  const totals = new CollectiveTotals(config),
    first = diagnostic(0, 0),
    next = diagnostic(4, 1);
  const raw = JSON.stringify([first, next]);
  totals.consume(frame(0, undefined, { 0: first }));
  totals.consume(frame(4, undefined, { 0: next }));
  assert.equal(harm(totals.view(0).animal), 1 / 5);
  assert.equal(totals.view(0).animal.steps, 5);
  totals.consume(frame(5));
  assert.equal(harm(totals.view(0).animal), 2 / 6);
  assert.equal(harm(totals.view(0).by_species.prey), 2 / 6);
  assert.equal(totals.view(0).schedule, 'episode');
  assert.equal(JSON.stringify([first, next]), raw);
  const detached = totals.view(0);
  detached.animal.collective[0].harm = 99;
  assert.equal(harm(totals.view(0).animal), 2 / 6);
});

test('Species means are averaged per step; individual lifetimes and extinctions are independent', () => {
  const totals = new CollectiveTotals(config);
  totals.consume(
    frame(0, [animal(0), animal(1)], {
      0: diagnostic(0, 0),
      1: diagnostic(0, 1),
    }),
  );
  totals.consume(frame(1, [animal(1)]));
  assert.equal(harm(totals.view(1).animal), 1);
  assert.equal(harm(totals.view(1).by_species.prey), 0.75);
  assert.equal(totals.view(0).animal, null);
  totals.consume(frame(2, []));
  totals.consume(frame(3, []));
  assert.equal(harm(totals.view(null).by_species.prey), 0.75);
  assert.equal(totals.view(null).by_species.prey.steps, 2);
  totals.consume(frame(4, [animal(2)], { 2: diagnostic(4, 0.2) }));
  assert.equal(totals.view(2).animal.steps, 1);
  assert.equal(harm(totals.view(2).animal), 0.2);
  assert.equal(totals.agents.size, 1);
});

test('Unknown evidence and absent samples are not treated as zero outcome rates', () => {
  const totals = new CollectiveTotals(config);
  totals.consume(frame(0));
  assert.equal(totals.view(0).animal, null);
  totals.consume(frame(1, undefined, { 0: diagnostic(1, 0, 0) }));
  totals.consume(frame(2, undefined, { 0: diagnostic(2, 1) }));
  const result = totals.view(0);
  assert.equal(harm(result.animal), 1);
  assert.equal(result.animal.collective[0].confidence, 0.5);
  assert.equal(result.animal.steps, 2);
  assert.equal(harm(result.by_species.prey), 1);
});

test('Future diagnostics are ignored and live memory does not retain dead agents', () => {
  const totals = new CollectiveTotals(config);
  totals.consume(frame(0, undefined, { 0: diagnostic(10, 1) }));
  assert.equal(totals.view(0).animal, null);
  for (let tick = 1; tick < 100; tick++) {
    totals.consume(
      frame(tick, [animal(tick)], { [tick]: diagnostic(tick, 0.5) }),
    );
    assert.equal(totals.agents.size, 1);
    assert.equal(totals.species.size, 1);
  }
  assert.throws(() => totals.consume(frame(99)), /increasing/);
});

function playback() {
  const frames = [
    frame(0, [animal(0), animal(1, 'predator')], {
      0: diagnostic(0, 0),
      1: diagnostic(0, 1),
    }),
    frame(1, [animal(0), animal(1, 'predator')]),
    frame(2, [animal(0), animal(1, 'predator')], {
      0: diagnostic(2, 1),
      1: diagnostic(2, 0),
    }),
    frame(3, [animal(0), animal(1, 'predator')]),
    frame(4, [animal(0), animal(1, 'predator')], {
      0: diagnostic(4, 0.5),
      1: diagnostic(4, 0.5),
    }),
  ];
  const meta = {
    recording_version: 1,
    name: 'test',
    mode: 'train',
    status: 'complete',
    config,
    recording: { frames_per_chunk: 2 },
    runs: [
      { run: 1, frame_count: frames.length, obstacles: [] },
      { run: 2, frame_count: 1, obstacles: [] },
    ],
  };
  const calls = [];
  const p = new Playback(
    async (path) => {
      calls.push(path);
      if (path.endsWith('/metrics')) return [];
      if (!path.includes('/chunks/')) return meta;
      const [run, chunk] = path
        .match(/runs\/(\d+)\/chunks\/(\d+)/)
        .slice(1)
        .map(Number);
      return run === 2
        ? [frame(0, undefined, { 0: diagnostic(0, 0.75) })]
        : frames.slice(chunk * 2, chunk * 2 + 2);
    },
    undefined,
    { collective: true },
  );
  return { p, calls };
}

test('Playback averages work across chunks, direct seeks, local rewinds and selection changes', async () => {
  const { p, calls } = playback();
  await p.open('test');
  let current = await p.seek(4, 0);
  assert.equal(harm(current.collective.animal), 0.5);
  assert.equal(harm(current.collective.by_species.predator), 0.5);
  current = await p.seek(3, 0);
  assert.equal(harm(current.collective.animal), 0.5);
  const before = calls.length;
  current = await p.seek(2, 0);
  assert.equal(harm(current.collective.animal), 1 / 3);
  assert.equal(calls.length, before);
  assert.equal(harm(p.snapshot(1).collective.animal), 2 / 3);
  assert.equal(calls.length, before);
  current = await p.seek(0, 0);
  assert.equal(harm(current.collective.animal), 0);
  await p.advance(0);
  current = await p.advance(0);
  assert.equal(harm(current.collective.animal), 1 / 3);
  current = await p.changeRun(2, 0);
  assert.equal(harm(current.collective.animal), 0.75);
  assert.equal(current.collective.animal.steps, 1);
});

test('A reset cancels an outstanding prefix without applying the previous episode state', async () => {
  const history = new CollectiveHistory();
  history.reset(config, 2);
  let resolve;
  const prefix = new Promise((done) => {
    resolve = done;
  });
  const pending = history.move([frame(4)], 4, 4, () => prefix);
  history.reset(config, 2);
  resolve([frame(0), frame(1)]);
  await pending;
  assert.equal(history.index, -1);
  assert.equal(history.totals.tick, null);
  assert.equal(history.totals.agents.size, 0);
});

test('Combined-source cumulative averages keep confidence, weighting and rewind state separate', async () => {
  const bothConfig = structuredClone(config);
  bothConfig.brains.test.consequence_scope = 'both';
  const diag = (tick, general, context) => ({
    tick,
    features: [
      ...Array(20).fill(0),
      ...Array(24).fill(0),
      ...Array.from({ length: 12 }, () => [1, general, 0, 0]).flat(),
      ...Array.from({ length: 12 }, () =>
        context === null ? [0, 0, 0, 0] : [0.5, context, 0, 0],
      ).flat(),
    ],
  });
  const frames = [
    frame(0, undefined, { 0: diag(0, 0.8, null) }),
    frame(1),
    frame(2, undefined, { 0: diag(2, 0.2, 0.9) }),
    frame(3),
  ];
  const history = new CollectiveHistory();
  history.reset(bothConfig, 2);
  await history.move(frames.slice(0, 2), 0, 1, () => {
    throw new Error('No fetch needed');
  });
  await history.move(frames.slice(2), 2, 3, () => {
    throw new Error('Sequential chunks reuse totals');
  });
  let result = history.totals.view(0);
  assert.equal(result.animal.general_consequences[0].harm, 0.5);
  assert.equal(result.animal.consequences[0].harm, 0.9);
  assert.equal(result.by_species.prey.general_consequences[0].harm, 0.5);
  assert.equal(result.by_species.prey.consequences[0].harm, 0.9);
  assert.equal(result.animal.consequences[0].confidence, 0.25);
  await history.move(frames.slice(0, 2), 0, 1, () => {
    throw new Error('Prefix starts at zero');
  });
  result = history.totals.view(0);
  assert.equal(result.animal.general_consequences[0].harm, 0.8);
  assert.equal(result.animal.consequences[0].confidence, 0);
});

test('Species mean reuse matches fresh reduction across samples, lifecycle changes and copies', () => {
  const c = structuredClone(config);
  c.brains.test.consequence_scope = 'both';
  const cached = new CollectiveTotals(c),
    fresh = new CollectiveTotals(c);
  const diag = (tick, harm) => ({
    tick,
    features: [
      ...Array(20).fill(0),
      ...Array.from({ length: 30 }, () => [0.75, harm, 0.2, 0.1]).flat(),
    ],
  });
  const animals = [animal(0), animal(1), animal(2, 'predator')];
  let reused = 0;
  for (let tick = 0; tick < 15; tick++) {
    let living = tick < 5 ? animals : [animals[1], animals[2]];
    if (tick >= 10) living = [];
    if (tick >= 7 && tick < 10) living = [...living, animal(3)];
    if (tick === 4) living = [...living].reverse();
    const diagnostics =
      tick % 3 === 0 || tick === 7
        ? Object.fromEntries(
            living.map((a) => [a.id, diag(tick, (a.id + tick) / 20)]),
          )
        : undefined;
    const input = frame(tick, living, diagnostics);
    const prior = cached.species.get('prey')?.rows;
    fresh.members.clear();
    cached.consume(input);
    fresh.consume(input);
    assert.deepEqual(cached.view(1), fresh.view(1));
    if (tick === 1 || tick === 2) {
      assert.equal(cached.species.get('prey').rows, prior);
      reused++;
    }
    assert.ok(
      [...cached.members.values()]
        .flat()
        .every(([member]) => [...cached.agents.values()].includes(member)),
    );
  }
  assert.equal(reused, 2);
  assert.equal(cached.members.size, 0);
  const source = new CollectiveTotals(c);
  source.consume(
    frame(
      0,
      animals,
      Object.fromEntries(animals.map((a) => [a.id, diag(0, 0.4)])),
    ),
  );
  const copied = source.copy();
  assert.equal(copied.members.size, 0);
  copied.consume(frame(1, animals));
  source.consume(frame(1, animals));
  assert.deepEqual(copied.view(0), source.view(0));
  assert.notEqual(
    copied.members.get('prey')[0][0],
    source.members.get('prey')[0][0],
  );
});
