import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BrainExplanation,
  explanationCharts,
} from '../viewer/explainability.mjs';
class Node {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.value = '';
    this.textContent = '';
    this.open = false;
  }
  append(...children) {
    this.children.push(...children);
  }
  replaceChildren(...children) {
    this.children = [...children];
    this.value = '';
  }
  setAttribute(k, v) {
    this.attributes[k] = v;
  }
  showModal() {
    this.open = true;
  }
  close() {
    this.open = false;
    this.onclose?.();
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
const $ = (id) => document.getElementById(id);
const walk = (n) => [n, ...n.children.flatMap(walk)];
const texts = (n) =>
  walk(n)
    .map((n) => n.textContent)
    .join(' ');
const report = (animal = 2) => ({
  species: 'prey',
  sex: 'female',
  animal,
  run: 1,
  tick: 10,
  diagnostic_tick: 9,
  criterion: 'lifespan',
  available_criteria: ['health', 'lifespan'],
  source: { animal_id: 8, run: 1, tick: 15 },
  model_generation: 2,
  parameters: 128,
  target: 'rest',
  policy: true,
  recorded_policy: true,
  recorded_predictions: { rest: 0.2, eat: 0.8 },
  outputs: [
    { action: 'rest', score: 0.7, masked: false },
    { action: 'eat', score: 0.3, masked: false },
  ],
  features: [
    { index: 0, label: 'Energy', value: 0.6, gradient: 0.2, attribution: 0.12 },
  ],
  top_features: [
    { index: 0, label: 'Energy', value: 0.6, gradient: 0.2, attribution: 0.12 },
  ],
  layers: [
    {
      label: 'Observation',
      size: 1,
      activation: 'input',
      nodes: [{ index: 0, label: 'Energy', value: 0.6, sensitivity: 0.2 }],
    },
    {
      label: 'Output logits',
      size: 2,
      activation: 'linear',
      values: [0.4, -0.1],
      nodes: [
        { index: 0, label: 'rest', value: 0.4, sensitivity: 0.2 },
        { index: 1, label: 'eat', value: -0.1, sensitivity: -0.2 },
      ],
    },
  ],
  edges: [
    { layer: 0, source: 0, target: 0, weight: 0.3, signal: 0.18, local: false },
  ],
  method: 'input × local gradient of action probability',
});

test('Explanation charts expose signed routes, historical comparison and non-causal attribution', () => {
  const root = explanationCharts(report(), { sections: { advanced: true } });
  assert(walk(root).filter((n) => n.tagName === 'svg').length >= 4);
  assert(texts(root).includes('not causal importance'));
  assert(texts(root).includes('different weights/time'));
  assert(texts(root).includes('Weight +0.300'));
  assert(texts(root).includes('Energy'));
});

test('No request until opened; controls issue selected-only requests and closing releases state', async () => {
  const calls = [];
  const panel = new BrainExplanation(async (path) => {
    calls.push(path);
    return report();
  });
  assert.equal(calls.length, 0);
  await panel.open({ id: 'a run', run: 1, frame: 10, animal: 2 });
  assert(calls[0].includes('a%20run/runs/1/explain/2/10'));
  assert($('explain-status').textContent.includes('champion #8'));
  $('explain-feature').value = '0';
  $('explain-value').value = '0.4';
  $('explain-apply').onclick();
  await new Promise((r) => setTimeout(r, 0));
  assert(calls.at(-1).includes('feature=0&value=0.4'));
  $('explain-action').value = 'eat';
  await panel.load();
  assert(calls.at(-1).includes('action=eat'));
  const count = calls.length;
  await panel.load();
  assert.equal(calls.length, count);
  $('explain-close').onclick();
  assert.equal(panel.result, null);
  assert.equal($('explain-content').children.length, 0);
});

test('Stale async responses and missing models never overwrite another animal view', async () => {
  const pending = [];
  const panel = new BrainExplanation(
    () => new Promise((resolve, reject) => pending.push({ resolve, reject })),
  );
  const first = panel.open({ id: 'r', run: 1, frame: 0, animal: 2 });
  const second = panel.open({ id: 'r', run: 1, frame: 1, animal: 3 });
  assert.equal(pending.length, 1);
  pending[0].resolve(report(2));
  await first;
  assert.equal(pending.length, 2);
  pending[1].resolve(report(3));
  await second;
  assert.equal(panel.result.animal, 3);
  $('explain-action').value = 'eat';
  const request = panel.load();
  pending[2].reject(new Error('No saved champions'));
  await request;
  assert($('explain-status').textContent.includes('No saved champions'));
  assert.equal($('explain-content').children.length, 0);
  assert.equal($('explain-apply').disabled, true);
  $('explain-close').onclick();
});

const replayContext = (frame, sample = frame) => ({
  id: 'follow',
  run: 1,
  frame,
  animal: 2,
  tick: frame,
  diagnosticTick: sample,
  playing: true,
  done: false,
});
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test('Saved explanation follows new diagnostic samples but reuses charts between samples', async () => {
  const calls = [];
  const panel = new BrainExplanation(async (path) => {
    calls.push(path);
    const frame = Number(path.match(/explain\/2\/(\d+)/)[1]);
    return { ...report(), tick: frame, diagnostic_tick: frame };
  });
  await panel.open(replayContext(0));
  const chart = $('explain-content').children[0];
  panel.update(replayContext(3, 0));
  assert.equal(calls.length, 1);
  assert.equal($('explain-content').children[0], chart);
  assert(
    $('explain-status').textContent.includes(
      'body step 3, explained observation step 0',
    ),
  );
  panel.update(replayContext(10));
  await settle();
  assert.equal(calls.length, 2);
  assert.notEqual($('explain-content').children[0], chart);
  assert.equal(panel.result.diagnostic_tick, 10);
  panel.update(replayContext(0));
  await settle();
  assert.equal(panel.result.diagnostic_tick, 0);
  $('explain-follow').checked = false;
  $('explain-follow').onchange();
  panel.update(replayContext(20));
  assert.equal(calls.length, 3);
  $('explain-follow').checked = true;
  $('explain-follow').onchange();
  await settle();
  assert.equal(panel.result.diagnostic_tick, 20);
  $('explain-follow').checked = false;
  $('explain-follow').onchange();
  panel.update({ ...replayContext(21), animal: null, diagnosticTick: null });
  const count = calls.length;
  $('explain-follow').checked = true;
  $('explain-follow').onchange();
  await settle();
  assert.equal(calls.length, count);
  assert.equal(panel.result, null);
  $('explain-close').onclick();
});

test('One in-flight request coalesces rapid samples to the newest pending observation', async () => {
  const calls = [],
    pending = [];
  let active = 0,
    maxActive = 0;
  const panel = new BrainExplanation((path) => {
    calls.push(path);
    active++;
    maxActive = Math.max(maxActive, active);
    return new Promise((resolve) =>
      pending.push((r) => {
        active--;
        resolve(r);
      }),
    );
  });
  const first = panel.open(replayContext(0));
  panel.update(replayContext(10));
  panel.update(replayContext(20));
  panel.update(replayContext(30));
  assert.equal(calls.length, 1);
  pending[0]({ ...report(), diagnostic_tick: 0 });
  await first;
  assert.equal(calls.length, 2);
  assert(calls[1].includes('/2/30?'));
  pending[1]({ ...report(), diagnostic_tick: 30 });
  await settle();
  assert.equal(maxActive, 1);
  assert.equal(panel.result.diagnostic_tick, 30);
  panel.update(replayContext(40));
  panel.update(replayContext(30));
  pending[2]({ ...report(), diagnostic_tick: 40 });
  await settle();
  assert.equal(panel.result.diagnostic_tick, 30);
  panel.update({ ...replayContext(31), animal: null, diagnosticTick: null });
  assert.equal(panel.result, null);
  assert.equal($('explain-content').children.length, 0);
  assert(
    $('explain-status').textContent.includes(
      'previous explanation has been cleared',
    ),
  );
  $('explain-close').onclick();
});

test('What-if pins the displayed observation, errors stop auto retry and closing drops queued work', async () => {
  const calls = [];
  const panel = new BrainExplanation(async (path) => {
    calls.push(path);
    return report();
  });
  let pins = 0;
  panel.onPin = () => pins++;
  await panel.open(replayContext(0));
  $('explain-feature').value = '0';
  $('explain-value').value = '0.2';
  $('explain-apply').onclick();
  await settle();
  assert.equal(pins, 1);
  assert.equal(panel.following, false);
  assert.equal($('explain-follow').checked, false);
  panel.update(replayContext(20));
  assert.equal(calls.length, 2);
  $('explain-close').onclick();
  let errors = 0;
  const failed = new BrainExplanation(async () => {
    errors++;
    throw new Error('No saved champions');
  });
  await failed.open(replayContext(0));
  failed.update(replayContext(10));
  failed.update(replayContext(20));
  assert.equal(errors, 1);
  $('explain-retry').onclick();
  await settle();
  assert.equal(errors, 2);
  $('explain-close').onclick();
  const delayed = [];
  const queued = new BrainExplanation(
    () => new Promise((resolve) => delayed.push(resolve)),
  );
  const opening = queued.open(replayContext(0));
  queued.update(replayContext(10));
  $('explain-close').onclick();
  delayed[0](report());
  await opening;
  await settle();
  assert.equal(delayed.length, 1);
  assert.equal(queued.result, null);
  const initial = [];
  const startup = new BrainExplanation(
    () => new Promise((resolve) => initial.push(resolve)),
  );
  const start = startup.open(replayContext(0));
  $('explain-follow').checked = false;
  $('explain-follow').onchange();
  initial[0](report());
  await start;
  initial[1](report());
  await settle();
  assert.equal(startup.following, false);
  assert(startup.result);
  $('explain-close').onclick();
});

test('Inputs and outputs lead; advanced neuron maps stay lazy and k changes never request model data', async () => {
  const root = explanationCharts(report());
  assert(texts(root.children[0]).includes('How inputs affect'));
  assert(texts(root.children[1]).includes('Predicted action probabilities'));
  assert(!texts(root).includes('Weight +0.300'));
  const advanced = root.children.at(-1);
  advanced.open = true;
  advanced.ontoggle();
  assert(texts(root).includes('Weight +0.300'));
  const calls = [];
  const view = new BrainExplanation(async (p) => {
    calls.push(p);
    return report();
  });
  await view.open({ id: 'r', run: 1, frame: 0, animal: 2 });
  const before = $('explain-content').children[0];
  view.setRankLimit(3);
  assert.notEqual($('explain-content').children[0], before);
  assert.equal(calls.length, 1);
  $('explain-close').onclick();
});
