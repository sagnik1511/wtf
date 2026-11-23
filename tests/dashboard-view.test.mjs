import test from 'node:test';
import assert from 'node:assert/strict';
import { Dashboard } from '../viewer/dashboard.mjs';

// Small DOM/canvas harness to verify asynchronous view behavior without packages.
class Node {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.style = {};
    this.attributes = {};
    this.open = false;
    this.value = '';
    this.textContent = '';
  }
  append(...children) {
    this.children.push(...children);
  }
  replaceChildren(...children) {
    this.children = [...children];
  }
  get firstChild() {
    return this.children[0];
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
  getBoundingClientRect() {
    return { width: 400, height: 150 };
  }
  getContext() {
    return new Proxy({}, { get: () => () => {} });
  }
  showModal() {
    this.open = true;
  }
}
const nodes = new Map();
globalThis.document = {
  getElementById: (id) => {
    if (!nodes.has(id)) nodes.set(id, new Node());
    return nodes.get(id);
  },
  createElement: (tag) => new Node(tag),
  createElementNS: (_, tag) => new Node(tag),
};
globalThis.window = { devicePixelRatio: 1 };
globalThis.getComputedStyle = () => ({
  fontFamily: 'system-ui',
  getPropertyValue: (key) =>
    ({ '--muted': '#4b6073', '--border': '#c0cdd8' })[key] ?? '',
});
globalThis.ResizeObserver = class {
  observe() {}
};
const $ = (id) => document.getElementById(id);
const delay = () => new Promise((resolve) => setTimeout(resolve, 0));
const stat = (population) => ({
  population,
  health: 0.7,
  energy: 0.5,
  births: 0,
  deaths: 0,
  kills: 0,
  attempts: 0,
  mean_age: 10,
  mean_generation: 1,
  groups: { clusters: 1, solitary: 0 },
  predictions: { rest: 0.2 },
  q_values: {},
});

test('Dashboard follows step changes and backward seeks, clears on run changes, and hides future points', async () => {
  $('dashboard-window').value = '128';
  const calls = [];
  const dashboard = new Dashboard(
    async (path) => {
      calls.push(path);
      const page = Number(path.split('/').at(-1));
      return {
        page,
        page_size: 128,
        start_frame: page * 128,
        points: Array.from({ length: 128 }, (_, i) => ({
          tick: page * 128 + i,
          consecutive: true,
          species: { prey: stat(page * 128 + i), predator: stat(2) },
        })),
      };
    },
    (name) => (name === 'prey' ? '#7de2bd' : '#ff9a77'),
  );
  const context = (index) => ({
    id: 'test',
    species: {
      prey: { allowed_actions: ['rest', 'eat'] },
      predator: { allowed_actions: ['rest', 'attack'] },
    },
    state: { run: 1, tick: index, frame_index: index, frame_count: 300 },
    playing: false,
  });
  await dashboard.update(context(0));
  assert.equal(calls.length, 0);
  $('metrics-open').onclick();
  await delay();
  assert.equal(dashboard.points.at(-1).tick, 0);
  assert.equal(dashboard.charts.length, 6);
  await dashboard.update(context(140));
  assert.equal(dashboard.points.length, 128);
  assert.equal(dashboard.points.at(-1).tick, 140);
  assert($('dashboard-status').textContent.includes('No future frames shown'));
  await dashboard.update(context(4));
  assert.deepEqual(
    dashboard.points.map((p) => p.tick),
    [0, 1, 2, 3, 4],
  );
  assert($('dashboard-position').textContent.includes('Step 4'));
  $('dashboard-species').value = 'predator';
  $('dashboard-species').onchange();
  assert.equal($('species-now').children.length, 1);
  const next = context(0);
  next.state.run = 2;
  await dashboard.update(next);
  assert.equal(dashboard.points.length, 1);
  assert(calls.at(-1).includes('/runs/2/'));
  $('metrics-dialog').open = false;
  const count = calls.length;
  await dashboard.update(context(299));
  assert.equal(calls.length, count);
});

test('Slower previous seek responses cannot overwrite the latest dashboard position', async () => {
  let release;
  const dashboard = new Dashboard(
    (path) =>
      new Promise((resolve) => {
        const page = Number(path.split('/').at(-1));
        const result = {
          page,
          page_size: 128,
          start_frame: page * 128,
          points: Array.from({ length: 128 }, (_, i) => ({
            tick: page * 128 + i,
            consecutive: true,
            species: { prey: stat(i) },
          })),
        };
        if (page === 1) release = () => resolve(result);
        else resolve(result);
      }),
    () => '#7de2bd',
  );
  $('metrics-dialog').open = true;
  $('dashboard-window').value = '128';
  const context = (index) => ({
    id: 'race',
    species: { prey: { allowed_actions: ['rest'] } },
    state: { run: 1, tick: index, frame_index: index, frame_count: 300 },
    playing: false,
  });
  const old = dashboard.update(context(140));
  await delay();
  await dashboard.update(context(3));
  release();
  await old;
  assert.equal(dashboard.points.at(-1).tick, 3);
});

test('Species cumulative charts belong to the dashboard, update in place and follow its filter', () => {
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
  );
  $('metrics-dialog').open = true;
  $('dashboard-species').value = '';
  const social = (value) => ({
    collective: [
      { action: 'walk', confidence: 1, harm: value, feeding: 0, birth: 0 },
    ],
    consequences: [
      { action: 'rest', confidence: 1, harm: value, feeding: 0, birth: 0 },
    ],
  });
  dashboard.context = {
    id: 'averages',
    species: {
      prey: { allowed_actions: ['rest'] },
      predator: { allowed_actions: ['rest'] },
    },
    state: {
      run: 1,
      tick: 20,
      collective: {
        tick: 20,
        schedule: 'episode',
        by_species: { prey: social(0.2), predator: social(0.6) },
      },
    },
  };
  dashboard.drawCollective();
  assert.equal(dashboard.collectivePanels.size, 2);
  assert.equal($('collective-section').hidden, false);
  const panel = dashboard.collectivePanels.get('prey');
  const walk = (node) => [node, ...node.children.flatMap(walk)];
  assert.ok(
    walk(panel).some((n) => n.textContent.includes('frozen at episode start')),
  );
  const nodes = walk(panel);
  dashboard.context.state.collective.tick = 21;
  dashboard.context.state.collective.by_species.prey = social(0.4);
  $('dashboard-species').value = 'prey';
  dashboard.drawCollective();
  assert.equal(dashboard.collectivePanels.get('prey'), panel);
  assert.deepEqual(walk(panel), nodes);
  assert.equal(panel.hidden, false);
  assert.equal(dashboard.collectivePanels.get('predator').hidden, true);
  assert.ok(
    walk(panel).some((n) =>
      n.textContent.includes('Running averages through step 21'),
    ),
  );
  assert.ok(walk(panel).some((n) => n.textContent.includes('walk harm: 40%')));
  // Rewind before predator's first diagnostic: its later chart must disappear.
  $('dashboard-species').value = '';
  const predator = dashboard.collectivePanels.get('predator');
  dashboard.context.state.collective = {
    tick: 0,
    by_species: { prey: social(0) },
  };
  dashboard.drawCollective();
  assert.equal(predator.hidden, true);
  assert.equal(panel.hidden, false);
  dashboard.context.state.collective.by_species.predator = social(0.1);
  dashboard.drawCollective();
  assert.equal(dashboard.collectivePanels.get('predator'), predator);
  assert.equal(predator.hidden, false);
  dashboard.context.state.collective.by_species = {};
  dashboard.drawCollective();
  assert.equal(panel.hidden, true);
  assert.equal(predator.hidden, true);
  assert.equal($('collective-section').hidden, true);
  dashboard.context.state.run = 2;
  dashboard.context.state.collective = null;
  dashboard.drawCollective();
  assert.equal(dashboard.collectivePanels.size, 0);
  assert.equal($('collective-section').hidden, true);
  $('metrics-dialog').open = false;
});

test('Dashboard heatmap filters arbitrary species without showing a stale grid', async () => {
  const selected = [];
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
    async (name) => {
      selected.push(name);
      dashboard.context.state.movement_heatmap = {
        ...dashboard.context.state.movement_heatmap,
        species: name,
        counts: new Uint32Array([0, 2, 0, 0]),
      };
    },
  );
  $('metrics-dialog').open = true;
  $('dashboard-species').value = '';
  dashboard.context = {
    id: 'three-species',
    species: Object.fromEntries(
      ['hare', 'field mouse', 'fox'].map((name) => [
        name,
        { allowed_actions: ['rest'] },
      ]),
    ),
    state: {
      run: 1,
      tick: 12,
      frame_index: 12,
      movement_heatmap: {
        width: 2,
        height: 2,
        counts: new Uint32Array([1, 1, 0, 0]),
        arrivals: 2,
        maximum: 1,
        tick: 12,
        frame_index: 12,
        species: null,
      },
    },
  };
  dashboard.drawMovement();
  const all = dashboard.heatmap;
  assert.equal(all, null);
  assert.equal(
    $('dashboard-movement').hidden,
    true,
    'combined overview belongs to the sidebar',
  );
  $('dashboard-species').value = 'field mouse';
  // Wrong-group data is hidden immediately, before the asynchronous rebuild.
  dashboard.drawMovement();
  assert.equal(dashboard.heatmap, null);
  assert($('dashboard-movement').firstChild.textContent.startsWith('Loading'));
  await $('dashboard-species').onchange();
  assert.deepEqual(selected, ['field mouse']);
  assert.notEqual(dashboard.heatmap, all);
  assert(
    dashboard.heatmap.element.children[0].textContent.includes('field mouse'),
  );
  const widget = dashboard.heatmap;
  dashboard.drawMovement();
  assert.equal(dashboard.heatmap, widget);
  document.documentElement = { dataset: { theme: 'light' } };
  dashboard.drawMovement();
  document.documentElement.dataset.theme = 'dark';
  dashboard.drawMovement();
  dashboard.context.state.movement_heatmap = null;
  dashboard.drawMovement();
  assert.equal($('dashboard-movement').hidden, true);
  assert.equal(dashboard.heatmap, null);
  $('metrics-dialog').open = false;
});

test('Local/general/context charts have matching intrinsic column and row geometry', () => {
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
  );
  $('metrics-dialog').open = true;
  $('dashboard-species').value = '';
  const row = (action) => ({
    action,
    confidence: 1,
    harm: 0.1,
    feeding: 0.2,
    birth: 0.3,
  });
  const social = {
    scope: 'both',
    collective: [row('walk'), row('rest')],
    general_consequences: [row('rest')],
    consequences: [row('rest')],
  };
  dashboard.context = {
    id: 'alignment',
    species: { deer: { allowed_actions: ['rest'] } },
    state: {
      run: 1,
      tick: 1,
      collective: { tick: 1, by_species: { deer: social } },
    },
  };
  dashboard.drawCollective();
  const walk = (node) => [node, ...node.children.flatMap(walk)];
  const panel = dashboard.collectivePanels.get('deer');
  const charts = walk(panel).filter((node) => node.tagName === 'svg');
  assert.equal(charts.length, 3);
  assert.deepEqual(
    charts.map((chart) => chart.attributes.width),
    ['340', '340', '340'],
  );
  assert.deepEqual(
    charts.map((chart) => chart.attributes.height),
    ['80', '54', '54'],
  );
  for (const chart of charts) {
    assert.equal(chart.attributes.preserveAspectRatio, 'xMinYMin meet');
    assert.equal(
      chart.children.find((n) => n.textContent === 'Action').attributes.x,
      '2',
    );
    assert.equal(
      chart.children.find((n) => n.textContent === 'harm').attributes.x,
      '172',
    );
    assert.equal(
      chart.children.find((n) => n.textContent === 'harm').attributes[
        'text-anchor'
      ],
      'middle',
    );
  }
  $('metrics-dialog').open = false;
});

test('All-species dashboard places synchronized heatmaps beside metrics for every configured species', async () => {
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
  );
  $('metrics-dialog').open = true;
  $('dashboard-species').value = '';
  const names = ['hare', 'field mouse', 'fox'];
  const group = (species, maximum) => ({
    species,
    width: 2,
    height: 2,
    counts: new Uint32Array([0, maximum, 0, 0]),
    maximum,
    arrivals: maximum,
    frame_index: 12,
    tick: 12,
  });
  dashboard.context = {
    id: 'simultaneous',
    species: Object.fromEntries(
      names.map((name) => [name, { allowed_actions: ['rest'] }]),
    ),
    state: {
      run: 1,
      tick: 12,
      frame_index: 12,
      movement_heatmap: group(null, 13),
      movement_by_species: Object.fromEntries(
        names.map((name, i) => [name, group(name, [1, 4, 8][i])]),
      ),
    },
  };
  // Even an extinct species or a recording lacking telemetry gets its movement map.
  dashboard.points = [{ tick: 12, species: { hare: stat(2), fox: stat(1) } }];
  dashboard.draw();
  assert.equal($('species-now').children.length, 3);
  assert.equal(dashboard.speciesHeatmaps.size, 3);
  for (const name of names) {
    const widget = dashboard.speciesHeatmaps.get(name);
    assert(
      $('species-now').children.some((card) =>
        card.children.includes(widget.element),
      ),
    );
    assert(
      widget.element.children[0].textContent.includes(
        `${name} · through step 12`,
      ),
    );
    assert.equal(widget.element.children[3].hidden, true);
    const values = widget.element.children[2].children.map(
      (item) => item.children[1].textContent,
    );
    assert.equal(
      values.at(-1),
      '8',
      'species maps share the largest species count scale',
    );
  }
  const widgets = [...dashboard.speciesHeatmaps.values()];
  document.documentElement.dataset.theme = 'light';
  dashboard.draw();
  assert.deepEqual([...dashboard.speciesHeatmaps.values()], widgets);
  $('dashboard-species').value = 'fox';
  dashboard.context.state.movement_heatmap =
    dashboard.context.state.movement_by_species.fox;
  dashboard.draw();
  assert.equal($('species-now').children.length, 1);
  assert.equal(
    $('species-now').firstChild.children.includes(widgets[2].element),
    false,
  );
  assert(dashboard.heatmap.element.children[0].textContent.includes('fox'));
  dashboard.context.state.run = 2;
  dashboard.context.state.movement_by_species = {};
  dashboard.context.state.movement_heatmap = null;
  dashboard.draw();
  assert.equal(dashboard.speciesHeatmaps.size, 0);
  $('metrics-dialog').open = false;
});

test('Sidebar heatmap always uses the combined grid even when the dashboard selects a species', async () => {
  const { readFile } = await import('node:fs/promises');
  const vm = await import('node:vm');
  const source = await readFile(
    new URL('../viewer/app.js', import.meta.url),
    'utf8',
  );
  const start = source.indexOf('function drawAllMovement()');
  const end = source.indexOf('function selectedTracking()', start);
  const map = { species: null, frame_index: 1, tick: 1 },
    chosen = { species: 'fox' },
    calls = [];
  const context = {
    state: {
      run: 1,
      tick: 1,
      frame_index: 1,
      movement_heatmap: chosen,
      obstacles: [],
    },
    player: { id: 'test', run: { run: 1 }, movement: { map } },
    heatmapView: null,
    heatmapKey: null,
    document: { documentElement: { dataset: { theme: 'light' } } },
    $: () => ({
      replaceChildren: (...children) => calls.push(['replace', children]),
    }),
    movementHeatmap: (value) => {
      calls.push(['create', value]);
      return {
        element: {},
        update: (next, theme) => calls.push(['update', next, theme]),
      };
    },
  };
  vm.runInNewContext(
    source.slice(start, end) + '\ndrawAllMovement();drawAllMovement();',
    context,
  );
  assert.equal(calls.filter(([kind]) => kind === 'create').length, 1);
  assert.equal(calls[0][1], map);
  assert.equal(calls.at(-1)[1], map);
  assert.equal(calls.at(-1)[2], 'light');
  context.state.run = 2;
  context.player.run.run = 2;
  vm.runInNewContext('drawAllMovement();', context);
  assert.equal(calls.filter(([kind]) => kind === 'create').length, 2);
});

test('Dashboard reuses species cards and legends while updating values, themes and filters', () => {
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
  );
  $('metrics-dialog').open = true;
  $('dashboard-species').value = '';
  const names = ['rabbits', 'deer', 'lynx'];
  dashboard.context = {
    id: 'reuse',
    species: Object.fromEntries(
      names.map((name) => [name, { allowed_actions: ['rest'] }]),
    ),
    state: { run: 1, tick: 1, frame_index: 1 },
  };
  dashboard.points = [
    {
      tick: 1,
      species: Object.fromEntries(names.map((name) => [name, stat(3)])),
    },
  ];
  dashboard.draw();
  const cards = [...$('species-now').children],
    legend = [...$('dashboard-legend').children];
  dashboard.points[0].species.deer = stat(7);
  document.documentElement.dataset.theme = 'light';
  dashboard.draw();
  assert.deepEqual($('species-now').children, cards);
  assert.deepEqual($('dashboard-legend').children, legend);
  assert.equal(cards[1].children[1].children[1].textContent, '7');
  $('dashboard-species').value = 'deer';
  dashboard.draw();
  assert.deepEqual($('species-now').children, [cards[1]]);
  $('dashboard-species').value = '';
  dashboard.draw();
  assert.deepEqual($('species-now').children, cards);
  assert.equal(dashboard.speciesCards.size, 3);
  dashboard.context.state.run = 2;
  dashboard.draw();
  assert.notEqual($('species-now').firstChild, cards[0]);
  assert.equal(dashboard.speciesCards.size, 3);
  $('metrics-dialog').open = false;
});

test('A backward seek immediately removes future telemetry while the older page is pending', async () => {
  let release;
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
  );
  $('metrics-dialog').open = true;
  $('dashboard-species').value = '';
  $('dashboard-window').value = '128';
  dashboard.key = 'censor:1';
  dashboard.points = [{ tick: 150, species: { deer: stat(99) } }];
  dashboard.context = {
    id: 'censor',
    species: { deer: { allowed_actions: ['rest'] } },
    state: { run: 1, tick: 150, frame_index: 150 },
  };
  dashboard.draw();
  assert.equal(
    $('species-now').firstChild.children[1].children[1].textContent,
    '99',
  );
  dashboard.cache.history = () =>
    new Promise((resolve) => {
      release = resolve;
    });
  const pending = dashboard.update({
    ...dashboard.context,
    state: { run: 1, tick: 2, frame_index: 2, frame_count: 300 },
  });
  assert.equal(dashboard.points.length, 0);
  assert.equal($('species-now').children.length, 0);
  dashboard.draw(); // Theme/resize repaint while the request is still pending.
  assert.equal(dashboard.points.length, 0);
  release([{ tick: 2, species: { deer: stat(4) } }]);
  await pending;
  assert.equal(
    $('species-now').firstChild.children[1].children[1].textContent,
    '4',
  );
  $('metrics-dialog').open = false;
});

test('Repeated chart choices reuse one prepared series within the same draw only', () => {
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
  );
  $('metrics-dialog').open = true;
  $('dashboard-species').value = '';
  dashboard.context = {
    id: 'series',
    species: { deer: { allowed_actions: ['rest'] } },
    state: { run: 1, tick: 1 },
  };
  dashboard.points = [{ tick: 1, species: { deer: stat(3) } }];
  dashboard.chartTypes.fill('population');
  const prepared = [];
  dashboard.plot = (_, lines) => prepared.push(lines);
  dashboard.draw();
  assert.equal(new Set(prepared).size, 1);
  assert.equal(prepared[0][0].points[0].value, 3);
  prepared.length = 0;
  dashboard.points[0].species.deer.population = 6;
  dashboard.draw();
  assert.equal(new Set(prepared).size, 1);
  assert.equal(prepared[0][0].points[0].value, 6);
  $('metrics-dialog').open = false;
});

test('Chart bounds support many species without spread-argument overflow', () => {
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
  );
  dashboard.points = Array.from({ length: 1024 }, (_, tick) => ({ tick }));
  const lines = Array.from({ length: 150 }, (_, species) => ({
    name: `species_${species}`,
    points: dashboard.points.map(({ tick }) => ({
      tick,
      value: species + tick,
      consecutive: true,
    })),
  }));
  assert.doesNotThrow(() =>
    dashboard.plot(new Node('canvas'), lines, { percent: false }),
  );
});

test('Selected-species playback avoids repainting hidden compact heatmaps and refreshes on return', () => {
  const dashboard = new Dashboard(
    async () => ({}),
    () => '#7de2bd',
  );
  $('metrics-dialog').open = true;
  $('dashboard-species').value = '';
  const map = (species) => ({
    species,
    width: 2,
    height: 2,
    counts: new Uint32Array([0, 1, 0, 0]),
    maximum: 1,
    arrivals: 1,
    frame_index: 1,
    tick: 1,
  });
  dashboard.context = {
    id: 'hidden',
    species: {
      deer: { allowed_actions: ['rest'] },
      fox: { allowed_actions: ['rest'] },
    },
    state: {
      run: 1,
      tick: 1,
      frame_index: 1,
      movement_heatmap: map(null),
      movement_by_species: { deer: map('deer'), fox: map('fox') },
    },
  };
  dashboard.draw();
  let paints = 0;
  for (const widget of dashboard.speciesHeatmaps.values()) {
    const update = widget.update;
    widget.update = (...args) => {
      paints++;
      return update(...args);
    };
  }
  $('dashboard-species').value = 'fox';
  dashboard.context.state.movement_heatmap =
    dashboard.context.state.movement_by_species.fox;
  dashboard.context.state.frame_index = 2;
  for (const group of Object.values(
    dashboard.context.state.movement_by_species,
  ))
    group.frame_index = 2;
  dashboard.draw();
  assert.equal(paints, 0);
  $('dashboard-species').value = '';
  dashboard.draw();
  assert(paints >= 2);
  $('metrics-dialog').open = false;
});

test('Sidebar hides movement from a pending frame or another run until committed state catches up', async () => {
  const { readFile } = await import('node:fs/promises');
  const vm = await import('node:vm');
  const source = await readFile(
    new URL('../viewer/app.js', import.meta.url),
    'utf8',
  );
  const start = source.indexOf('function drawAllMovement()');
  const end = source.indexOf('function selectedTracking()', start);
  let paints = 0;
  const box = { hidden: false, replaceChildren() {} };
  const map = { frame_index: 0, tick: 0, by_species: {} };
  const context = {
    state: {
      run: 1,
      frame_index: 0,
      tick: 0,
      obstacles: [],
      movement_by_species: map.by_species,
    },
    player: { id: 'test', run: { run: 1 }, movement: { map } },
    heatmapView: null,
    heatmapKey: null,
    document: { documentElement: { dataset: { theme: 'dark' } } },
    $: () => box,
    movementHeatmap: () => ({
      element: {},
      update() {
        paints++;
      },
    }),
  };
  vm.runInNewContext(source.slice(start, end), context);
  vm.runInNewContext('drawAllMovement();', context);
  assert.equal(paints, 1);
  // Movement has advanced, but collective history is still loading for that seek.
  map.frame_index = 7;
  map.tick = 7;
  context.document.documentElement.dataset.theme = 'light';
  vm.runInNewContext('drawAllMovement();', context);
  assert.equal(box.hidden, true);
  assert.equal(paints, 1);
  context.state.frame_index = 7;
  context.state.tick = 7;
  vm.runInNewContext('drawAllMovement();', context);
  assert.equal(box.hidden, false);
  assert.equal(paints, 2);
  map.by_species = {};
  vm.runInNewContext('drawAllMovement();', context);
  assert.equal(
    box.hidden,
    true,
    'a different recording at the same step is not committed',
  );
  assert.equal(paints, 2);
  context.state.movement_by_species = map.by_species;
  context.player.run.run = 2;
  vm.runInNewContext('drawAllMovement();', context);
  assert.equal(box.hidden, true);
  assert.equal(paints, 2);
});
