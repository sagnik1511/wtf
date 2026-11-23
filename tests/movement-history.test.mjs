import test from 'node:test';
import assert from 'node:assert/strict';
import { MovementHistory } from '../viewer/movement-history.mjs';

test('Movement reset rejects a stale in-flight prefix and retains one bounded grid', async () => {
  const h = new MovementHistory();
  h.reset(256, 256);
  assert.equal(h.map.counts.byteLength, 256 * 256 * 4);
  let release;
  const waiting = h.move(
    [{ tick: 10, animals: [] }],
    10,
    10,
    () =>
      new Promise((resolve) => {
        release = resolve;
      }),
  );
  h.reset(2, 2);
  release({
    width: 256,
    height: 256,
    frame_index: 10,
    tick: 10,
    counts: Array(256 * 256).fill(1),
  });
  await waiting;
  assert.equal(h.map.frame_index, -1);
  assert.equal(h.map.arrivals, 0);
  assert.equal(h.map.counts.length, 4);
});

test('Movement prefixes validate requested time and reject impossible count data', async () => {
  const h = new MovementHistory();
  h.reset(2, 2);
  const f = [{ tick: 10, animals: [] }];
  for (const bad of [
    { width: 2, height: 2, frame_index: 11, tick: 10, counts: [0, 0, 0, 0] },
    { width: 2, height: 2, frame_index: 10, tick: 11, counts: [0, 0, 0, 0] },
    { width: 2, height: 2, frame_index: 10, tick: 10, counts: [-1, 0, 0, 0] },
  ])
    await assert.rejects(
      h.move(f, 10, 10, async () => bad),
      /Invalid movement/,
    );
  assert.equal(h.map.arrivals, 0);
});

test('Grouped prefixes validate all species before mutating any grid', async () => {
  const h = new MovementHistory();
  h.reset(2, 2, null, ['hare', 'field mouse', 'fox']);
  const group = (species) => ({
    width: 2,
    height: 2,
    species,
    counts: [0, 1, 0, 0],
  });
  const data = {
    ...group(null),
    frame_index: 10,
    tick: 10,
    by_species: Object.fromEntries(
      ['hare', 'field mouse', 'fox'].map((name) => [name, group(name)]),
    ),
  };
  data.by_species.fox.counts[0] = -1;
  await assert.rejects(
    h.move([{ tick: 10, animals: [] }], 10, 10, async () => data),
    /Invalid movement/,
  );
  for (const map of h.maps)
    assert.equal(
      map.counts.reduce((a, b) => a + b),
      0,
    );
  data.by_species.fox.counts[0] = 0;
  await h.move([{ tick: 10, animals: [] }], 10, 10, async () => data);
  assert.equal(h.maps.length, 4);
  for (const map of h.maps) {
    assert.equal(map.counts.byteLength, 16);
    assert.equal(map.frame_index, 10);
  }
});
