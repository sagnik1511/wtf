import test from 'node:test';
import assert from 'node:assert/strict';
import {
  activationPanel,
  AnimalBrainView,
} from '../viewer/brain-activations.mjs';
import { Playback } from '../viewer/playback.mjs';
class Node {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.textContent = '';
    this.open = false;
  }
  append(...n) {
    this.children.push(...n);
  }
  replaceChildren(...n) {
    this.children = [...n];
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
  createElement: (t) => new Node(t),
  createElementNS: (_, t) => new Node(t),
  getElementById: (id) => {
    if (!nodes.has(id)) nodes.set(id, new Node());
    return nodes.get(id);
  },
};
const $ = (id) => document.getElementById(id),
  walk = (n) => [n, ...n.children.flatMap(walk)],
  text = (n) =>
    walk(n)
      .map((n) => n.textContent)
      .join(' ');
const diagnostic = (tick, value = 0.2) => ({
  tick,
  algorithm: 'recurrent_ppo',
  updates: tick,
  features: Array(20).fill(0),
  action_probabilities: { rest: 0.75, eat: 0.25 },
  activations: {
    version: 1,
    layers: [[value, -value], [value / 2]],
    recurrent_prior: [0.1, -0.1],
  },
});
const context = (d) => ({
  id: 'recording',
  run: 1,
  tick: d.tick,
  animal: { id: 7, species: 'prey', sex: 'female' },
  diagnostic: d,
  template: { type: 'recurrent_ppo' },
  playing: true,
  done: false,
});

test('Actual signed activations, recurrent memory and output scores render without invented weights', () => {
  const panel = activationPanel(
    diagnostic(4),
    { type: 'recurrent_ppo' },
    { compact: false, sections: { activations: true } },
  );
  assert(
    text(panel).includes('this animal’s own sampled next-decision activations'),
  );
  assert(text(panel).includes('Actual prior recurrent memory'));
  assert(text(panel).includes('75.0%'));
  assert(
    text(panel).includes(
      'Connections, weights and causal attributions are not stored',
    ),
  );
  const many = diagnostic(5);
  many.activations.layers = [Array(100).fill(0.2)];
  assert(
    text(
      activationPanel(many, null, { sections: { activations: true } }),
    ).includes('64/100 units'),
  );
  assert(
    text(
      activationPanel(many, null, {
        compact: false,
        sections: { activations: true },
      }),
    ).includes('100/100 units'),
  );
});

test('Open brain follows replay samples, reuses the same chart between samples and clears on death', () => {
  const view = new AnimalBrainView(),
    first = diagnostic(0);
  view.open(context(first));
  const graph = $('brain-content').children[0];
  view.update({ ...context(first), tick: 3 });
  assert.equal($('brain-content').children[0], graph);
  assert(
    $('brain-status').textContent.includes('body step 3 · brain sample 0'),
  );
  view.update(context(diagnostic(10, 0.9)));
  assert.notEqual($('brain-content').children[0], graph);
  assert(text($('brain-content')).includes('Step 10'));
  view.update({ ...context(first), tick: 2 });
  assert(text($('brain-content')).includes('Step 0'));
  view.update({
    ...context(first),
    animal: undefined,
    diagnostic: undefined,
    tick: 12,
  });
  assert($('brain-status').textContent.includes('No selected animal'));
  assert.equal(view.sample, undefined);
  assert(!text($('brain-content')).includes('Step 0'));
  $('brain-close').onclick();
  assert.equal(view.sample, null);
  assert.equal($('brain-content').children.length, 0);
});

test('Older recordings retain real prediction bars and explain the absent activation samples', () => {
  const old = diagnostic(9);
  delete old.activations;
  const panel = activationPanel(old, null, { compact: false });
  assert(text(panel).includes('Internal activations were not recorded'));
  assert(text(panel).includes('75.0%'));
  const broken = diagnostic(10);
  broken.activations.layers = [[NaN]];
  assert(
    text(activationPanel(broken, null)).includes(
      'Unsupported activation sample',
    ),
  );
});

test('Replay selection keeps actual samples at or before the playhead across chunks, backward seeks and runs', async () => {
  const metadata = {
    recording_version: 1,
    name: 'test',
    mode: 'train',
    recording: { frames_per_chunk: 2 },
    config: {
      world: { width: 8, height: 8 },
      resources: { capacity_per_patch: 20 },
      species: { prey: { diet: 'resource' } },
    },
    runs: [
      { run: 1, obstacles: [], frame_count: 4, complete: true },
      { run: 2, obstacles: [], frame_count: 2, complete: true },
    ],
  };
  const calls = [];
  const player = new Playback(async (path) => {
    calls.push(path);
    if (path === '/api/recordings/test') return metadata;
    if (path.endsWith('/metrics')) return [];
    const match = path.match(/runs\/(\d+)\/chunks\/(\d+)$/);
    assert(match);
    const run = +match[1],
      chunk = +match[2];
    return [0, 1].map((i) => ({
      tick: chunk * 2 + i,
      animals: [{ id: 7, species: 'prey', x: 1, y: 1 }],
      resources: [],
      events: [],
      metrics: { tick: chunk * 2 + i, populations: { prey: 1 }, food: 0 },
      ...(i === 0
        ? { diagnostics: { 7: diagnostic(chunk * 2, run / 10) } }
        : {}),
    }));
  });
  await player.open('test');
  assert.equal((await player.seek(1, 7)).selected.tick, 0);
  assert.equal((await player.advance(7)).selected.tick, 2);
  assert.equal(
    (await player.seek(0, 7)).selected.activations.layers[0][0],
    0.1,
  );
  await player.changeRun(2, 7);
  assert.equal(player.snapshot(7).selected.activations.layers[0][0], 0.2);
  assert.equal(player.frames.length, 2);
  assert(
    calls.every((p) => !p.includes('/explain/') && !p.includes('/api/step')),
  );
});

test('Recorded view keeps input/output visible, summarizes deeper brains and defers neuron maps', () => {
  const d = diagnostic(5);
  d.activations.layers = [[0.1], [0.2], [0.3]];
  const panel = activationPanel(
    d,
    { type: 'recurrent_ppo' },
    { compact: false },
  );
  assert(text(panel).includes('What the inputs describe'));
  assert(text(panel).includes('75.0%'));
  assert(text(panel).includes('rank 3 > k 2'));
  assert.equal(walk(panel).filter((n) => n.tagName === 'svg').length, 0);
  const details = walk(panel).find((n) => n.tagName === 'details');
  assert.equal(details.children.length, 1);
  details.open = true;
  details.ontoggle();
  assert(walk(details).some((n) => n.tagName === 'svg'));
  const view = new AnimalBrainView();
  view.open({ ...context(d), rankLimit: 2 });
  const before = $('brain-content').children[0];
  view.update({ ...context(d), rankLimit: 3 });
  assert.notEqual($('brain-content').children[0], before);
  assert(text($('brain-content')).includes('rank 3 ≤ k 3'));
  $('brain-close').onclick();
});
