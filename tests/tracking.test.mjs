import test from 'node:test';
import assert from 'node:assert/strict';
import {
  trackingLayout,
  trackingLabels,
  trackingView,
} from '../viewer/tracking.mjs';
import { observationView } from '../viewer/inspector-stats.mjs';
import { inputLabels } from '../viewer/network-structure.mjs';

const template = {
  observation_version: 1,
  movement_history_steps: 2,
  scent_observation: { species: ['grazer', 'hunter'], radius: 3 },
};
const history = [0.4, 0.2, 0.1, 0, 0.5, 2 / 3, 0, 0.4, 0.2, 1, 0, 0, 0];
const features = [
  ...Array(20).fill(0),
  ...history,
  ...Array(13).fill(0),
  ...Array.from({ length: 36 }, (_, i) => i / 36),
];
test('Optional tracking layouts retain old input counts and validate limits', () => {
  assert.equal(trackingLayout({}).width, 0);
  assert.equal(observationView(Array(20).fill(0), {}).inputs, 20);
  assert.equal(trackingLayout(template).width, 62);
  for (const invalid of [
    { movement_history_steps: 17 },
    { movement_history_steps: true },
    { scent_observation: { species: ['x', 'x'] } },
    { scent_observation: { species: ['x'], radius: 0 } },
    { scent_observation: { species: ['x'], separate_sexes: 1 } },
  ])
    assert.equal(trackingLayout(invalid), null);
  assert.equal(observationView(features, {}), null);
  assert.equal(observationView(features.slice(1), template), null);
});
test('Tracking parser preserves gaps, velocity estimates, and ordered anonymous scent channels', () => {
  const result = observationView(features, template).tracking;
  assert.equal(result.tracks.length, 1);
  assert.equal(result.tracks[0].age, 1);
  assert.equal(result.tracks[0].visible, false);
  assert.equal(result.tracks[0].history[1].seen, false);
  assert.equal(result.tracks[0].vx, 0.1);
  assert.deepEqual(
    result.scent.map((c) => [c.species, c.sex]),
    [
      ['grazer', 'female'],
      ['grazer', 'male'],
      ['hunter', 'female'],
      ['hunter', 'male'],
    ],
  );
  assert.deepEqual(result.scent[1].strengths, features.slice(55, 64));
  assert(!JSON.stringify(result).includes('animal_id'));
  const names = inputLabels(template, features.length);
  assert.deepEqual(names.slice(20), trackingLabels(template));
  assert.equal(names[20], 'Prey track 1 last dx');
  assert.equal(names.at(-1), 'Scent hunter male northwest');
});
test('Merged-sex channels and scent-only layouts need no visual memory', () => {
  const t = { scent_observation: { species: ['deer'], separate_sexes: false } };
  assert.equal(trackingLayout(t).width, 9);
  const result = trackingView(Array(9).fill(1), t, 0);
  assert.equal(result.tracks.length, 0);
  assert.equal(result.scent[0].sex, 'any');
  assert.deepEqual(result.scent[0].strengths, Array(9).fill(1));
});

test('Board tracks use the sampled observer position and reject unavailable or future origins', async () => {
  const { trackingSample, drawVisualTracks } = await import(
    '../viewer/tracking.mjs'
  );
  const diagnostic = { tick: 4, features };
  const frames = [
    { tick: 4, animals: [{ id: 7, x: 2, y: 3 }] },
    { tick: 5, animals: [{ id: 7, x: 9, y: 8 }] },
  ];
  const sample = trackingSample(frames, 1, 7, diagnostic, template, 10);
  assert.deepEqual(sample.origin, { x: 2, y: 3 });
  assert.equal(
    trackingSample(frames, 0, 7, { ...diagnostic, tick: 5 }, template, 10),
    null,
  );
  assert.equal(trackingSample(frames, 1, 8, diagnostic, template, 10), null);
  const calls = [],
    ctx = {
      save() {},
      restore() {},
      setLineDash(v) {
        calls.push(['dash', v]);
      },
      beginPath() {},
      moveTo(...v) {
        calls.push(['move', ...v]);
      },
      lineTo(...v) {
        calls.push(['line', ...v]);
      },
      arc(...v) {
        calls.push(['arc', ...v]);
      },
      fill() {},
      stroke() {},
    };
  drawVisualTracks(ctx, sample, 10, 10, false);
  assert(calls.some((c) => c[0] === 'arc'));
  assert(
    !calls.some((c) => c[0] === 'dash' && c[1].length),
    'Board trails contain recorded sightings only, without velocity arrows',
  );
  const withoutScent = { ...sample, scent: [] };
  const before = JSON.stringify(calls);
  calls.length = 0;
  drawVisualTracks(ctx, withoutScent, 10, 10, false);
  assert.equal(
    JSON.stringify(calls),
    before,
    'Board visual tracks never draw scent hotspots',
  );
  assert.equal(sample.origin.x, 2);
});

test('Blended scent exposes one anonymous nine-sector field and rejects ambiguous modes', () => {
  const t = { scent_observation: { mode: 'blended', radius: 2 } };
  assert.equal(trackingLayout(t).width, 9);
  const view = trackingView(Array(9).fill(0.4), t, 0);
  assert.deepEqual(view.scent, [
    { blended: true, strengths: Array(9).fill(0.4) },
  ]);
  assert.equal(trackingLabels(t)[0], 'Scent blended here');
  for (const scent_observation of [
    { mode: 'invalid' },
    { mode: 'blended', species: ['prey'] },
    { mode: 'blended', separate_sexes: false },
  ])
    assert.equal(trackingLayout({ scent_observation }), null);
});
