import test from 'node:test';
import assert from 'node:assert/strict';
import { behaviorRows } from '../viewer/behavior-stats.mjs';

function stats(diet) {
  return {
    diet,
    mean_energy_fraction: 0.5,
    energy_coverage: 0.4,
    starving_fraction: 0,
    movement_success: 0.75,
    reward_per_agent_step: -0.01,
    action_coverage: 1,
    feeding_success: diet === 'resource' ? 0.1 : null,
    hunt_success: diet === 'prey' ? 0.02 : null,
    learning: {
      algorithm: diet === 'resource' ? 'q_network' : 'grpo',
      normalized_entropy: diet === 'prey' ? 1 : null,
    },
  };
}
test('Behavior summary compares early and late windows with species-specific metrics', () => {
  const early = stats('resource'),
    late = { ...stats('resource'), feeding_success: 0.34 };
  const rows = behaviorRows(
    { windows: [{ species: { prey: early } }, { species: { prey: late } }] },
    'prey',
  );
  assert.deepEqual(
    rows.find((r) => r[0] === 'Productive feeding'),
    ['Productive feeding', '10.0%', '34.0%'],
  );
  assert.equal(
    rows.some((r) => r[0] === 'Kills / attack'),
    false,
  );
  assert.deepEqual(
    rows.find((r) => r[0] === 'Reward / animal-step'),
    ['Reward / animal-step', '-0.010', '-0.010'],
  );
});
test('Predator summary labels entropy and avoids fabricated values for missing observations', () => {
  const s = { ...stats('prey'), hunt_success: null, action_coverage: null };
  const rows = behaviorRows(
    { windows: [{ species: { predator: s } }] },
    'predator',
  );
  assert.deepEqual(
    rows.find((r) => r[0] === 'Kills / attack'),
    ['Kills / attack', '—', '—'],
  );
  assert.deepEqual(
    rows.find((r) => r[0] === 'Policy entropy'),
    ['Policy entropy', '100.0%', '100.0%'],
  );
  assert.equal(
    rows.some((r) => r[0] === 'Productive feeding'),
    false,
  );
  assert.deepEqual(behaviorRows({ windows: [] }, 'prey'), []);
});

test('Movement heatmap draws only positive arrival counts with theme-aware colors', async () => {
  const { drawMovementHeatmap } = await import('../viewer/behavior-stats.mjs');
  const paints = [],
    ctx = {
      fillStyle: '',
      fillRect(...rect) {
        paints.push({ color: this.fillStyle, rect });
      },
    };
  const map = {
    width: 2,
    height: 2,
    counts: [0, 1, 4, 0],
    maximum: 4,
    arrivals: 5,
  };
  assert.equal(drawMovementHeatmap(ctx, map, 'light', [[1, 1]]), true);
  assert.equal(paints.length, 4);
  assert.deepEqual(paints[1].rect, [1, 0, 1, 1]);
  const first = paints[0].color;
  paints.length = 0;
  drawMovementHeatmap(ctx, map, 'dark');
  assert.notEqual(paints[0].color, first);
  assert.equal(
    drawMovementHeatmap(ctx, { ...map, counts: [1] }, 'light'),
    false,
  );
});

test('Dark heatmap has increasing brightness across the log scale', async () => {
  const { movementHeatColor } = await import('../viewer/behavior-stats.mjs');
  const luminance = (color) => {
    const rgb = color.startsWith('#')
      ? [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16))
      : color.match(/\d+/g).map(Number);
    return rgb
      .map((v) => {
        const c = v / 255;
        return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
      })
      .reduce((s, v, i) => s + v * [0.2126, 0.7152, 0.0722][i], 0);
  };
  const levels = [0, 1, 3, 10, 30, 78].map((c) =>
    luminance(movementHeatColor(c, 78, 'dark')),
  );
  for (let i = 1; i < levels.length; i++) assert(levels[i] > levels[i - 1]);
  assert(
    levels.at(-1) > levels[1] * 5,
    'Busy cells should be clearly brighter than lightly visited cells',
  );
});

test('Shared species heatmap scale gives equal counts equal colors despite different local maxima', async () => {
  const { drawMovementHeatmap } = await import('../viewer/behavior-stats.mjs');
  for (const theme of ['light', 'dark']) {
    const colors = [];
    for (const maximum of [1, 8]) {
      const ctx = {
        fillStyle: '',
        fillRect(x) {
          if (x === 1) colors.push(this.fillStyle);
        },
      };
      drawMovementHeatmap(
        ctx,
        { width: 2, height: 1, counts: [0, 1], maximum },
        theme,
        [],
        8,
      );
    }
    assert.equal(colors[0], colors[1]);
  }
});
