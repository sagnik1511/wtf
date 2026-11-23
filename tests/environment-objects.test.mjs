import test from 'node:test';
import assert from 'node:assert/strict';
import {
  drawEnvironmentObjects,
  objectMotion,
} from '../viewer/environment-objects.mjs';

class Context {
  constructor() {
    this.calls = [];
    this.fillStyle = 'initial';
    this.strokeStyle = 'initial';
    this.globalAlpha = 1;
    this.lineWidth = 1;
    this.stack = [];
  }
  save() {
    this.stack.push([
      this.fillStyle,
      this.strokeStyle,
      this.globalAlpha,
      this.lineWidth,
    ]);
  }
  restore() {
    [this.fillStyle, this.strokeStyle, this.globalAlpha, this.lineWidth] =
      this.stack.pop();
  }
}
for (const method of [
  'fillRect',
  'beginPath',
  'moveTo',
  'lineTo',
  'closePath',
  'fill',
  'stroke',
  'quadraticCurveTo',
  'arc',
]) {
  Context.prototype[method] = function (...args) {
    this.calls.push({
      method,
      args,
      fill: this.fillStyle,
      stroke: this.strokeStyle,
      alpha: this.globalAlpha,
    });
  };
}
function scene(options = {}, quantity = 20, obstacles = [[1, 0]]) {
  const ctx = new Context();
  drawEnvironmentObjects(
    ctx,
    [{ x: 0, y: 0, quantity }],
    obstacles,
    20,
    32,
    32,
    options,
  );
  assert.equal(ctx.fillStyle, 'initial');
  assert.equal(ctx.globalAlpha, 1);
  assert.equal(ctx.stack.length, 0);
  return ctx.calls;
}

test('Object motion is deterministic at a replay tick and follows seeking', () => {
  assert.deepEqual(scene({ tick: 14 }), scene({ tick: 14 }));
  assert.notDeepEqual(scene({ tick: 14 }), scene({ tick: 28 }));
  assert.deepEqual(scene({ tick: 14 }), scene({ tick: 14 }));
  assert.notDeepEqual(objectMotion(14, 1, 0), objectMotion(14, 0, 1));
});
test('Reduced motion disables sway and shimmer changes while retaining quantity', () => {
  assert.deepEqual(
    scene({ tick: 0, reducedMotion: true }),
    scene({ tick: 300, reducedMotion: true }),
  );
  assert.notDeepEqual(
    scene({ tick: 0, reducedMotion: true }, 1),
    scene({ tick: 0, reducedMotion: true }, 20),
  );
});
test('Obstacle geometry stays fixed; only surface opacity changes', () => {
  const geometry = (tick) => {
    const ctx = new Context();
    drawEnvironmentObjects(ctx, [], [[1, 1]], 20, 32, 32, { tick });
    return ctx.calls.map(({ method, args }) => ({ method, args }));
  };
  assert.deepEqual(geometry(0), geometry(28));
});
test('Empty patches have no leaves, and quantity/capacity is safely clamped', () => {
  const leaves = (calls) =>
    calls.filter(
      (c) => c.method === 'fill' && ['#63ab6f', '#a1c8b0'].includes(c.fill),
    );
  assert.equal(leaves(scene({}, 0, [])).length, 0);
  assert.equal(leaves(scene({}, 20, [])).length, 2);
  assert.deepEqual(scene({}, -1, []), scene({}, 0, []));
  assert.deepEqual(scene({}, 200, []), scene({}, 20, []));
  for (const capacity of [0, -1, NaN, Infinity]) {
    const ctx = new Context();
    drawEnvironmentObjects(
      ctx,
      [{ x: 0, y: 0, quantity: 5 }],
      [],
      capacity,
      16,
      16,
    );
    for (const call of ctx.calls) assert(call.args.every(Number.isFinite));
  }
});
test('Both themes use distinct palettes without mutating recorded inputs or RNG', () => {
  assert.notDeepEqual(scene({ light: true }), scene({ light: false }));
  const resources = Object.freeze([
    Object.freeze({ x: 0, y: 0, quantity: 10 }),
  ]);
  const obstacles = Object.freeze([Object.freeze([1, 1])]);
  const random = Math.random;
  Math.random = () => {
    throw Error('Renderer must not sample RNG');
  };
  try {
    drawEnvironmentObjects(new Context(), resources, obstacles, 20, 32, 32, {
      tick: 18,
    });
  } finally {
    Math.random = random;
  }
  assert.equal(resources[0].quantity, 10);
});
test('Very small cells stay finite and invalid dimensions skip drawing', () => {
  const ctx = new Context();
  drawEnvironmentObjects(
    ctx,
    [{ x: 0, y: 0, quantity: 20 }],
    [[1, 1]],
    20,
    2,
    2,
    { tick: 10 },
  );
  for (const call of ctx.calls) assert(call.args.every(Number.isFinite));
  const empty = new Context();
  drawEnvironmentObjects(empty, [], [[1, 1]], 20, 0, 0);
  assert.equal(empty.calls.length, 0);
});
