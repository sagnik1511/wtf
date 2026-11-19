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
