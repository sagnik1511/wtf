import test from 'node:test';
import assert from 'node:assert/strict';
import { daylight, shadowState } from '../viewer/day-night.mjs';
import { drawEnvironmentShadows } from '../viewer/environment-objects.mjs';
const cycle = { enabled: true, cycle_steps: 100, start_phase: 0 };
const world = {
  obstacles: [[0, 0]],
  resources: [{ x: 1, y: 1, quantity: 20 }],
  resource_capacity: 20,
  animals: [{ id: 1, x: 2, y: 2 }],
};
function render(
  tick,
  {
    light = false,
    reducedMotion = false,
    state = world,
    cw = 20,
    ch = 12,
  } = {},
) {
  const calls = [];
  const ctx = {
    globalAlpha: 0.7,
    fillStyle: 'original',
    save() {
      this.saved = [this.globalAlpha, this.fillStyle];
    },
    restore() {
      [this.globalAlpha, this.fillStyle] = this.saved;
    },
    beginPath() {},
    closePath() {},
    moveTo(...args) {
      calls.push(['move', ...args]);
    },
    lineTo(...args) {
      calls.push(['line', ...args]);
    },
    fill() {
      calls.push(['fill', this.globalAlpha, this.fillStyle]);
    },
  };
  drawEnvironmentShadows(ctx, state, cw, ch, daylight(cycle, tick), {
    light,
    reducedMotion,
  });
  assert.equal(ctx.globalAlpha, 0.7);
  assert.equal(ctx.fillStyle, 'original');
  return calls;
}
test('Shadows oppose sun direction, shorten at noon, fade at horizon and repeat on seek', () => {
  const morning = shadowState(daylight(cycle, 30)),
    noon = shadowState(daylight(cycle, 50)),
    evening = shadowState(daylight(cycle, 70));
  assert.ok(morning.dx > 0);
  assert.ok(evening.dx < 0);
  assert.ok(Math.abs(noon.dx) < 1e-12);
  assert.ok(Math.hypot(noon.dx, noon.dy) < Math.hypot(morning.dx, morning.dy));
  assert.ok(shadowState(daylight(cycle, 25.001)).alpha < 0.001);
  assert.deepEqual(render(30), render(130));
  assert.notDeepEqual(render(30), render(70));
});
test('Disabled cycles and night draw no directional shadows', () => {
  for (const tick of [0, 25, 75, 99]) assert.deepEqual(render(tick), []);
  assert.equal(shadowState(daylight(null, 50)), null);
  assert.equal(shadowState(daylight({ ...cycle, enabled: false }, 50)), null);
});
test('Reduced motion freezes geometry; daily illumination can still fade opacity', () => {
  const morning = shadowState(daylight(cycle, 30), true),
    noon = shadowState(daylight(cycle, 50), true);
  assert.deepEqual([morning.dx, morning.dy], [noon.dx, noon.dy]);
  assert.notEqual(morning.alpha, noon.alpha);
  const geometry = (t) =>
    render(t, { reducedMotion: true }).filter((c) => c[0] !== 'fill');
  assert.deepEqual(geometry(30), geometry(50));
  assert.deepEqual(geometry(30), geometry(70));
});
test('One batched path covers rocks, growing plants and animals with theme-specific ink', () => {
  const before = JSON.stringify(world),
    dark = render(30),
    light = render(30, { light: true });
  assert.equal(dark.filter((c) => c[0] === 'move').length, 3);
  assert.equal(dark.filter((c) => c[0] === 'fill').length, 1);
  assert.deepEqual(dark.slice(0, -1), light.slice(0, -1));
  assert.notDeepEqual(dark.at(-1), light.at(-1));
  assert.equal(JSON.stringify(world), before);
  const depleted = { ...world, resources: [{ x: 1, y: 1, quantity: 0 }] };
  assert.equal(
    render(30, { state: depleted }).filter((c) => c[0] === 'move').length,
    2,
  );
  assert.deepEqual(render(30, { cw: 0 }), []);
  for (const row of render(30, { cw: 1, ch: 0.7 }))
    for (const value of row.slice(1))
      if (typeof value === 'number') assert.ok(Number.isFinite(value));
});
