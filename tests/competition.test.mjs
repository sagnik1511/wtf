import test from 'node:test';
import assert from 'node:assert/strict';
import { recentContests, contestLabel } from '../viewer/competition.mjs';
const frame = {
  tick: 7,
  animals: [],
  events: [
    {
      kind: 'contest_won',
      actor: 1,
      target: 2,
      position: [3, 2],
      amount: 5,
      cause: 'aggression',
    },
    {
      kind: 'contest_lost',
      actor: 2,
      target: 1,
      position: [3, 3],
      amount: 5,
      cause: 'aggression',
    },
  ],
};
test('contest markers retain recorded fight positions even after displacement or death', () => {
  const [marker] = recentContests([frame], 0, 0);
  assert.deepEqual(marker.position, [3, 2]);
  assert.deepEqual(marker.loser_position, [3, 3]);
  assert.equal(contestLabel(marker.cause), 'Aggression');
});
test('markers carry across chunks, deduplicate and expire without future leakage', () => {
  const markers = recentContests([frame], 0, 0);
  assert.equal(recentContests([frame], 0, 0, markers).length, 1);
  assert.equal(
    recentContests([{ tick: 8, events: [] }], 1, 0, markers)[0].age,
    1,
  );
  assert.deepEqual(recentContests([{ events: [] }], 12, 0, markers), []);
  assert.deepEqual(
    recentContests([{ events: [] }], 0, 0, [{ ...markers[0], frame_index: 1 }]),
    [],
  );
});
test('crowded fights have bounded transient memory and legacy frames remain usable', () => {
  assert.deepEqual(recentContests([{ animals: [] }], 0, 0), []);
  const crowded = {
    ...frame,
    events: Array.from({ length: 100 }, (_, i) => ({
      ...frame.events[0],
      actor: i,
    })),
  };
  assert.equal(recentContests([crowded], 0, 0).length, 32);
});
