import test from 'node:test';
import assert from 'node:assert/strict';
import { inspectorGraphs } from '../viewer/inspector.mjs';
class Node {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.className = '';
    this.textContent = '';
  }
  append(...children) {
    this.children.push(...children);
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
}
globalThis.document = {
  createElement: (tag) => new Node(tag),
  createElementNS: (_, tag) => new Node(tag),
};
const walk = (n) => [n, ...n.children.flatMap(walk)];
const text = (n) =>
  walk(n)
    .map((x) => x.textContent)
    .join(' ');
const body = { vision_radius: 6, water_vision_bonus: 3 };
const features = [0.5, 1, 0, -1, 0, 0, 1, 0, ...Array(12).fill(0)];

test('Inspector renders radar, prediction bars, actual history dots and keeps unknown layouts readable', () => {
  const d = {
    tick: 10,
    features,
    action_probabilities: { rest: 0.75, eat: 0.25 },
    action_mask: { eat: false },
  };
  const root = inspectorGraphs({
    diagnostic: d,
    template: { observation_version: 1 },
    spec: { body },
    hasWater: false,
    samples: [{ tick: 10, diagnostic: d }],
    sections: { history: true },
  });
  assert.equal(walk(root).filter((n) => n.tagName === 'svg').length, 2);
  assert(text(root).includes('k=1 · 20 inputs'));
  assert(text(root).includes('75.0%'));
  assert(text(root).includes('eat · masked'));
  assert(text(root).includes('no future frames'));
  const unknown = inspectorGraphs({
    diagnostic: { tick: 4, action_values: { rest: -2 } },
    spec: { body },
    hasWater: false,
  });
  assert(text(unknown).includes('Raw inputs remain available'));
  assert(text(unknown).includes('Q values, not probabilities'));
  assert(text(unknown).includes('-2.000'));
});

test('Top-k mating/terrain inputs render independently from current agent body state', () => {
  const f = [
    ...features.slice(0, 8),
    ...Array(48).fill(0),
    0.1,
    0.1,
    0,
    0.02,
    ...Array(8).fill(0),
    -1,
    0.8,
    0.1,
    1,
    0.2,
    0,
    0.2,
    1,
    0,
    1,
  ];
  const root = inspectorGraphs({
    diagnostic: {
      tick: 5,
      features: f,
      action_probabilities: { reproduce: 1 },
    },
    template: { observation_version: 3, observation_top_k: 4 },
    spec: { body },
    hasWater: true,
  });
  assert(text(root).includes('Step 5 · v3 · k=4 · 78 inputs'));
  assert(text(root).includes('Maturity 80%'));
  assert(text(root).includes('Pair opportunity yes'));
  assert(text(root).includes('female'));
  assert(text(root).includes('Elevation 2.0'));
});

test('Expanded prediction history stays open on the next diagnostic sample', () => {
  const sections = {},
    diagnostic = { tick: 0, features, action_values: { rest: 1 } };
  let root = inspectorGraphs({
    diagnostic,
    spec: { body },
    hasWater: false,
    sections,
  });
  let history = walk(root).find((n) => n.className === 'prediction-history');
  history.open = true;
  history.ontoggle();
  root = inspectorGraphs({
    diagnostic: { ...diagnostic, tick: 10 },
    spec: { body },
    hasWater: false,
    sections,
  });
  history = walk(root).find((n) => n.className === 'prediction-history');
  assert.equal(history.open, true);
});

test('Zero-valued Q histories keep a finite, non-inverted axis', () => {
  const diagnostic = { tick: 0, features, action_values: { rest: 0, eat: 0 } };
  const root = inspectorGraphs({
    diagnostic,
    spec: { body },
    hasWater: false,
    samples: [{ tick: 0, diagnostic }],
    sections: { history: true },
  });
  const trend = walk(root).find((n) => n.className === 'prediction-history');
  assert(text(trend).includes('1.0'));
  assert(text(trend).includes('0.0'));
  assert(!text(trend).includes('-1.0'));
});

test('Closed history defers score processing and SVG creation; opening reads each sample once', () => {
  let reads = 0;
  const scores = {
    get rest() {
      reads++;
      return -0.5;
    },
    attack: 2,
  };
  const samples = [
    { tick: 5, diagnostic: { action_values: scores } },
    { tick: 10, diagnostic: { action_probabilities: { rest: 1 } } },
  ];
  const sections = {},
    root = inspectorGraphs({
      diagnostic: { tick: 5, features, action_values: { rest: 1 } },
      spec: { body },
      hasWater: false,
      samples,
      sections,
    });
  const history = walk(root).find((n) => n.className === 'prediction-history');
  assert.equal(reads, 0);
  assert.equal(walk(history).filter((n) => n.tagName === 'svg').length, 0);
  history.open = true;
  history.ontoggle();
  assert.equal(reads, 1);
  assert.equal(sections.history, true);
  const chart = walk(history).find((n) => n.tagName === 'svg');
  assert.equal(walk(chart).filter((n) => n.tagName === 'circle').length, 2); // Q samples only.
  assert(text(chart).includes('rest · step 5: -0.500'));
  assert(text(chart).includes('attack · step 5: 2.000'));
  history.open = false;
  history.ontoggle();
  history.open = true;
  history.ontoggle();
  assert.equal(reads, 1);
  assert.equal(walk(history).filter((n) => n.tagName === 'svg').length, 1);
  assert.equal(
    walk(history).find((n) => n.tagName === 'svg'),
    chart,
  );
});
