import test from 'node:test';
import assert from 'node:assert/strict';
import { daylight, circadianBody, drawDaylight } from '../viewer/day-night.mjs';
import {
  visibilityCells,
  ObservationOverlay,
} from '../viewer/inspector-stats.mjs';
const cycle = { enabled: true, cycle_steps: 100, start_phase: 0 };
const species = {
  body: { vision_radius: 6, living_cost: 0.1, movement_cost: 0.2 },
  circadian: {
    night_vision_radius_multiplier: 0.5,
    night_living_cost_multiplier: 0.8,
    night_movement_cost_multiplier: 1.1,
  },
};
test('sunlight rises from dawn, peaks at midday and falls to dusk; seeking is stateless', () => {
  assert.deepEqual(
    [0, 25, 50, 75, 100].map((t) => daylight(cycle, t).light),
    [0, 0, 1, 0, 0],
  );
  assert.equal(daylight(cycle, 50).label, 'Midday');
  assert.equal(daylight(cycle, 0).label, 'Night');
  assert.ok(daylight(cycle, 30).light < daylight(cycle, 40).light);
  assert.ok(
    Math.abs(daylight(cycle, 35).light - daylight(cycle, 65).light) < 1e-12,
  );
  assert.deepEqual(daylight(cycle, 10), daylight(cycle, 110));
  assert.equal(daylight(undefined, 0).light, 1);
});
test('species modifiers leave metadata untouched and match Python rounding', () => {
  const before = JSON.stringify(species);
  assert.equal(circadianBody(species, cycle, 0).vision_radius, 3);
  assert.equal(circadianBody(species, cycle, 50).vision_radius, 6);
  assert.equal(
    circadianBody(species, cycle, 0).living_cost,
    0.08000000000000002,
  );
  assert.equal(JSON.stringify(species), before);
  assert.equal(circadianBody(species, undefined, 0), species.body);
});
test('board sight and overlay cache update when the same animal experiences day/night', () => {
  const animal = { id: 1, species: 'grazer', x: 8, y: 8 };
  const state = {
    width: 20,
    height: 20,
    species: { grazer: species },
    obstacles: [],
    tick: 0,
    day_night: cycle,
  };
  assert.equal(
    visibilityCells(state, animal, null).some(([x, y]) => x === 12 && y === 8),
    false,
  );
  const overlay = new ObservationOverlay(),
    ctx = { fillRect() {} };
  overlay.draw(ctx, state, animal, null, 10, 10);
  const night = overlay.cells.length;
  state.tick = 50;
  overlay.draw(ctx, state, animal, null, 10, 10);
  assert.ok(overlay.cells.length > night);
  state.tick = 0;
  overlay.draw(ctx, state, animal, null, 10, 10);
  assert.equal(overlay.cells.length, night);
});
test('lighting is opt-in and keeps night objects legible', () => {
  const draws = [],
    ctx = {
      fillRect(...args) {
        draws.push([this.fillStyle, args]);
      },
    };
  drawDaylight(ctx, undefined, 0, 100, 100);
  assert.equal(draws.length, 0);
  drawDaylight(ctx, cycle, 0, 100, 100);
  assert.match(draws[0][0], /0.48/);
  assert.deepEqual(draws[0][1], [0, 0, 100, 100]);
});

test('clock orbits continuously through sunrise/noon/sunset/night and reverses on seek', async () => {
  const { clockState } = await import('../viewer/day-night.mjs');
  const midnight = clockState(cycle, 0),
    rise = clockState(cycle, 25),
    noon = clockState(cycle, 50),
    set = clockState(cycle, 75);
  assert.equal(midnight.y, 58);
  assert.equal(noon.y, 10);
  assert.equal(rise.x, 10);
  assert.equal(set.x, 58);
  assert.equal(noon.sun, true);
  assert.equal(midnight.sun, false);
  assert.deepEqual(clockState(cycle, 25), rise);
  assert.deepEqual(clockState(cycle, 125), rise);
});

test('clock is opt-in, reuses its SVG nodes, freezes at a paused tick and clears on disabled recordings', async () => {
  const { DayNightClock } = await import('../viewer/day-night.mjs');
  const node = () => ({
    attributes: {},
    setAttribute(k, v) {
      this.attributes[k] = v;
    },
  });
  const nodes = Object.fromEntries(
    ['[data-marker]', '[data-body]', '[data-crescent]'].map((k) => [k, node()]),
  );
  const disc = node();
  nodes['[data-marker]'].querySelector = () => disc;
  const root = {
    ...node(),
    builds: 0,
    set innerHTML(value) {
      this.markup = value;
      this.builds++;
    },
    querySelector(k) {
      return nodes[k];
    },
  };
  const clock = new DayNightClock(root);
  clock.update(undefined, 0);
  assert.equal(root.hidden, true);
  assert.equal(root.builds, 0);
  clock.update(cycle, 50);
  assert.equal(root.hidden, false);
  assert.equal(root.builds, 1);
  assert.equal(nodes['[data-body]'].textContent, 'Sun');
  const position = nodes['[data-marker]'].attributes.transform;
  clock.update(cycle, 50);
  assert.equal(nodes['[data-marker]'].attributes.transform, position);
  clock.update(cycle, 0);
  assert.notEqual(nodes['[data-marker]'].attributes.transform, position);
  assert.equal(nodes['[data-body]'].textContent, 'Moon');
  clock.update(cycle, 50);
  assert.equal(nodes['[data-marker]'].attributes.transform, position);
  assert.equal(root.builds, 1);
  clock.update({ enabled: false }, 50);
  assert.equal(root.hidden, true);
});

test('daylight illuminates the ground most at midday and leaves disabled/night worlds untouched', async () => {
  const { drawSunlitGround } = await import('../viewer/day-night.mjs');
  const fills = [],
    ctx = {
      fillRect() {
        fills.push(this.fillStyle);
      },
    };
  drawSunlitGround(ctx, undefined, 50, 100, 100);
  drawSunlitGround(ctx, cycle, 0, 100, 100);
  assert.equal(fills.length, 0);
  drawSunlitGround(ctx, cycle, 30, 100, 100);
  const dawn = Number(fills[0].match(/,([\d.]+)\)/)[1]);
  drawSunlitGround(ctx, cycle, 50, 100, 100);
  const noon = Number(fills[1].match(/,([\d.]+)\)/)[1]);
  assert.ok(noon > dawn);
  assert.equal(noon, 0.28);
});

test('render helpers share one sunlight value and skip orbit work on a paused frame', async () => {
  const { drawSunlitGround, clockState, DayNightClock } = await import(
    '../viewer/day-night.mjs'
  );
  const illumination = daylight(cycle, 50);
  const forbidden = new Proxy(
    {},
    {
      get() {
        throw Error('Repeated sunlight lookup');
      },
    },
  );
  const ctx = { fillRect() {} };
  drawSunlitGround(ctx, forbidden, 50, 100, 100, illumination);
  drawDaylight(ctx, forbidden, 50, 100, 100, illumination);
  assert.equal(
    circadianBody(species, forbidden, 50, illumination).vision_radius,
    6,
  );
  assert.equal(clockState(forbidden, 50, illumination).phase, 0.5);
  const root = { hidden: false };
  const clock = new DayNightClock(root);
  clock.phase = 0.5;
  // A paused update needs neither DOM nodes nor any orbit trigonometry.
  clock.update(forbidden, 50, illumination);
  const state = {
    species: { grazer: species },
    day_night: forbidden,
    width: 20,
    height: 20,
    obstacles: [],
    tick: 50,
  };
  const body = circadianBody(species, cycle, 50);
  assert.ok(
    visibilityCells(state, { species: 'grazer', x: 8, y: 8 }, null, body)
      .length > 0,
  );
});
