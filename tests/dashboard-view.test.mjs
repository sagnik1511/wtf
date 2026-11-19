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
};
globalThis.window = { devicePixelRatio: 1 };
globalThis.getComputedStyle = () => ({ fontFamily: 'system-ui' });
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
