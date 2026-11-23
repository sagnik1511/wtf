import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('All literal app control IDs exist in the shipped viewer HTML', async () => {
  const [app, html] = await Promise.all(
    ['../viewer/app.js', '../viewer/index.html'].map((p) =>
      readFile(new URL(p, import.meta.url), 'utf8'),
    ),
  );
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
  for (const [, id] of app.matchAll(/\$\('([^']+)'\)/g))
    assert(
      ids.has(id),
      `Missing #${id} prevents viewer startup or interaction`,
    );
});

test('Population and food overview belongs to the metrics dialog and opens visibly', async () => {
  const html = await readFile(
    new URL('../viewer/index.html', import.meta.url),
    'utf8',
  );
  const dialog = html.slice(
    html.indexOf('<dialog id="metrics-dialog"'),
    html.indexOf('</dialog>', html.indexOf('<dialog id="metrics-dialog"')),
  );
  assert.equal([...html.matchAll(/id="chart"/g)].length, 1);
  assert(dialog.includes('id="chart"'));
  assert(dialog.includes('<details class="chart-panel" open>'));
  assert(dialog.includes('food uses right axis'));
  assert(
    dialog
      .replace(/\s+/g, ' ')
      .includes('independently of the Species and History filters'),
  );
});

test('Population and food overview supports large multi-species histories without argument overflow', async () => {
  const vm = await import('node:vm');
  const source = await readFile(
    new URL('../viewer/app.js', import.meta.url),
    'utf8',
  );
  const start = source.indexOf('function drawChart()');
  const end = source.indexOf('new ResizeObserver', start);
  const names = Array.from({ length: 150 }, (_, i) => `species_${i}`);
  const context2d = new Proxy(
    { measureText: () => ({ width: 20 }) },
    { get: (target, key) => target[key] ?? (() => {}) },
  );
  const canvas = {
    getBoundingClientRect: () => ({ width: 500, height: 170 }),
    getContext: () => context2d,
  };
  const style = {
    fontFamily: 'system-ui',
    fontSize: '12px',
    fontWeight: '400',
    getPropertyValue: () => '#758595',
  };
  const context = {
    $: () => canvas,
    window: { devicePixelRatio: 1 },
    document: { documentElement: {} },
    getComputedStyle: () => style,
    speciesColor: () => '#758595',
    state: {
      tick: 1023,
      species: Object.fromEntries(names.map((name) => [name, {}])),
      history: Array.from({ length: 1024 }, (_, tick) => ({
        tick,
        food: tick + 1,
        populations: Object.fromEntries(
          names.map((name, i) => [name, i + tick]),
        ),
      })),
    },
  };
  assert.doesNotThrow(() =>
    vm.runInNewContext(source.slice(start, end) + '\ndrawChart();', context),
  );
});

test('Passive viewer redraws keep committed state during an incomplete cross-chunk seek', async () => {
  const vm = await import('node:vm');
  const source = await readFile(
    new URL('../viewer/app.js', import.meta.url),
    'utf8',
  );
  const start = source.indexOf('function render(snapshot = null)');
  const end = source.indexOf('async function openRecording()', start);
  const state = { run: 1, frame_index: 0, tick: 0 };
  const context = {
    state,
    player: {
      frameIndex: 0,
      chunkIndex: 2,
      metadata: { recording: { frames_per_chunk: 3 } },
      snapshot() {
        throw Error('incomplete snapshot must not be requested');
      },
    },
  };
  assert.doesNotThrow(() =>
    vm.runInNewContext(source.slice(start, end) + '\nrender();', context),
  );
  assert.equal(context.state, state);
});
