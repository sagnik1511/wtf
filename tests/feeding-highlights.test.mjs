import test from 'node:test';
import assert from 'node:assert/strict';
import { recentFeedings, drawFeedings } from '../viewer/feeding-highlights.mjs';
import { Playback } from '../viewer/playback.mjs';
const frame = (tick, events = []) => ({
  tick,
  animals: [{ id: 7, x: 1, y: 2 }],
  events,
  metrics: { food: 1, populations: { deer: 1 } },
});
const gain = { kind: 'energy_gained', actor: 7, cause: 'resource', amount: 4 };
test('Only positive confirmed food gains produce one cue per eater; no guessed missing positions', () => {
  const frames = [
    frame(1, [
      gain,
      gain,
      { ...gain, actor: 8 },
      { ...gain, cause: 'prey' },
      { ...gain, amount: 0 },
    ]),
  ];
  const before = JSON.stringify(frames);
  assert.deepEqual(recentFeedings(frames, 0, 0), [
    { actor: 7, position: [1, 2], frame_index: 0, tick: 1, age: 0 },
  ]);
  assert.equal(JSON.stringify(frames), before);
  assert.equal(
    recentFeedings([frame(1, [{ ...gain, cause: 'competition_food' }])], 0, 0)
      .length,
    1,
  );
  assert.equal(
    recentFeedings(
      [
        frame(1, [
          { ...gain, amount: -1 },
          { ...gain, cause: undefined },
        ]),
      ],
      0,
      0,
    ).length,
    0,
  );
});
test('Feeding carry expires, excludes future cues and remains bounded', () => {
  const cue = recentFeedings([frame(0, [gain])], 0, 0);
  assert.equal(recentFeedings([frame(3)], 3, 0, cue)[0].age, 3);
  assert.deepEqual(recentFeedings([frame(4)], 4, 0, cue), []);
  assert.deepEqual(recentFeedings([frame(0)], -1, 0, cue), []);
  const crowded = frame(
    0,
    Array.from({ length: 500 }, (_, actor) => ({ ...gain, actor })),
  );
  crowded.animals = Array.from({ length: 500 }, (_, id) => ({
    id,
    x: 0,
    y: 0,
  }));
  assert.equal(recentFeedings([crowded], 0, 0).length, 64);
});
test('Sequential playback, random seek and tiny chunk boundaries show identical cues', async () => {
  const frames = Array.from({ length: 7 }, (_, i) =>
    frame(i, i === 0 ? [gain] : []),
  );
  const metadata = {
    recording_version: 1,
    name: 'test',
    mode: 'train',
    recording: { frames_per_chunk: 1 },
    config: {
      world: { width: 4, height: 4 },
      resources: { capacity_per_patch: 20 },
      species: { deer: {} },
      experiment: { runs: 2 },
    },
    runs: [
      { run: 1, frame_count: 7, obstacles: [] },
      { run: 2, frame_count: 7, obstacles: [] },
    ],
  };
  const load = async (url) =>
    url.endsWith('/metrics')
      ? []
      : url.includes('/chunks/')
        ? [frames[Number(url.split('/').at(-1))]]
        : metadata;
  const player = new Playback(load);
  await player.open('test');
  for (let i = 1; i < 5; i++) {
    const sequential = await player.advance();
    await player.seek(6);
    assert.deepEqual(
      (await player.seek(i)).feeding_highlights,
      sequential.feeding_highlights,
    );
  }
  await player.changeRun(2);
  assert.equal(player.feedingCarry.length, 0);
});
test('Feeding renderer freezes decorative phases under reduced motion and restores context', () => {
  const draw = (age, light = false, reducedMotion = false) => {
    const calls = [];
    const ctx = {
      save() {
        calls.push('save');
      },
      restore() {
        calls.push('restore');
      },
      beginPath() {},
      arc(...args) {
        calls.push([this.fillStyle, this.globalAlpha, ...args]);
      },
      fill() {},
      stroke() {},
    };
    drawFeedings(ctx, [{ position: [1, 2], age }], 20, 20, {
      light,
      reducedMotion,
    });
    return calls;
  };
  assert.deepEqual(draw(0, false, true), draw(3, false, true));
  assert.notDeepEqual(draw(0), draw(3));
  assert.notDeepEqual(draw(0), draw(0, true));
  assert.equal(draw(0).at(-1), 'restore');
});
