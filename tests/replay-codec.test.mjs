import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeChunk } from '../viewer/replay-codec.mjs';
import { Playback } from '../viewer/playback.mjs';

const packet = (frames, fields = ['id', 'x', 'health']) => ({
  codec: 'wtf-delta-v1',
  animal_fields: fields,
  resource_fields: ['x', 'y', 'quantity'],
  frames,
});
test('Compact chunks preserve removals, additions, sparse diagnostics, ordering and independent frames', () => {
  const payload = packet([
    {
      tick: 0,
      animals: [{ id: 1, x: 0, health: 5 }],
      resources: [{ x: 0, y: 0, quantity: 1 }],
      metrics: { food: 1 },
      diagnostics: { sample: 0 },
    },
    {
      set: { tick: 1 },
      drop: ['diagnostics'],
      animals: {
        update: [[1, 6, [1, -0]]],
        add: [{ id: 2, x: 2, health: 4 }],
        order: [2, 1],
      },
      resources: { update: [[[0, 0], 4, [2]]] },
      metrics: { set: { food: 2 } },
    },
    { set: { tick: 2 }, animals: { drop: [1] }, resources: { drop: [[0, 0]] } },
  ]);
  const frames = decodeChunk(payload);
  assert.deepEqual(
    frames[1].animals.map((a) => a.id),
    [2, 1],
  );
  assert(Object.is(frames[1].animals[1].health, -0));
  assert(!('diagnostics' in frames[1]));
  assert.deepEqual(frames[2].resources, []);
  frames[1].animals[0].health = 999;
  assert.equal(frames[2].animals[0].health, 4);
});
test('Invalid encodings, field masks and ordering are rejected; legacy arrays remain supported', () => {
  const initial = {
    tick: 0,
    animals: [{ id: 1, x: 0, health: 5 }],
    resources: [],
  };
  for (const delta of [
    { animals: { update: [[1, -1, []]] } },
    { animals: { update: [[1, 4, []]] } },
    { animals: { order: [1, 1] } },
  ]) {
    assert.throws(() => decodeChunk(packet([initial, delta])));
  }
  assert.throws(() => decodeChunk({ codec: 'unknown' }));
  const legacy = [initial];
  assert.equal(decodeChunk(legacy), legacy);
});
test('Compact replay loads and seeks through independent chunks and runs', async () => {
  const metadata = {
    recording_version: 1,
    recording: { frames_per_chunk: 2 },
    config: { world: { width: 8, height: 8 }, species: {}, resources: {} },
    runs: [
      { run: 1, frame_count: 4 },
      { run: 2, frame_count: 2 },
    ],
  };
  const p = new Playback(async (path) => {
    if (path === '/api/recordings/compact') return metadata;
    if (path.endsWith('/metrics')) return [];
    const chunk = Number(path.match(/chunks\/(\d+)/)[1]);
    return packet([
      {
        tick: chunk * 2,
        animals: [],
        resources: [],
        metrics: { tick: chunk * 2, populations: {}, food: 3 },
      },
      {
        set: { tick: chunk * 2 + 1 },
        metrics: { set: { tick: chunk * 2 + 1 } },
      },
    ]);
  });
  await p.open('compact');
  assert.equal(p.snapshot().tick, 0);
  await p.seek(3);
  assert.equal(p.snapshot().tick, 3);
  assert.equal(p.frames.length, 2);
  await p.seek(1);
  assert.equal(p.snapshot().tick, 1);
  await p.seek(3);
  await p.advance();
  assert.equal(p.run.run, 2);
  assert.equal(p.snapshot().tick, 0);
});
