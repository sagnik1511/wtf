import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { socialWidth, socialLabels, socialView } from '../viewer/social.mjs';
import { observationView } from '../viewer/inspector-stats.mjs';
import { inputLabels } from '../viewer/network-structure.mjs';
import { inspectorGraphs } from '../viewer/inspector.mjs';
const template = {
  observation_version: 4,
  observation_top_k: 3,
  movement_history_steps: 2,
  scent_observation: { mode: 'blended', radius: 2 },
  social_observation: { history_steps: 4, max_neighbors: 8 },
  consequence_observation: true,
};
const base = 45 + 24,
  tracking = 2 * 3 * (7 + 6) + 9;
const features = [...Array(base + tracking).fill(0), ...Array(72).fill(0)];
features[base + tracking + 4] = 0.5;
features[base + tracking + 5] = 0.75;
features[base + tracking + 24 + 9 * 4] = 0.8;
features[base + tracking + 24 + 9 * 4 + 1] = 0.6;
test('Optional social layouts preserve tracking offsets and distinguish unknown evidence', () => {
  assert.equal(socialWidth({}), 0);
  assert.equal(socialWidth(template), 72);
  const before = JSON.stringify(features),
    v = observationView(features, template);
  assert.ok(v);
  assert.ok(v.tracking);
  assert.equal(v.social.collective.length, 6);
  assert.deepEqual(v.social.collective[1], {
    action: 'swim',
    confidence: 0.5,
    harm: 0.75,
    feeding: 0,
    birth: 0,
  });
  assert.equal(
    v.social.consequences.find((r) => r.action === 'swim').harm,
    0.6,
  );
  assert.equal(v.social.consequences[0].confidence, 0);
  assert.equal(JSON.stringify(features), before);
  assert.equal(observationView(features.slice(0, -1), template), null);
});
test('Invalid social layouts are not silently mislabelled', () => {
  for (const social_observation of [
    null,
    [],
    { history_steps: 0 },
    { max_neighbors: 17 },
    { max_neighbors: true },
    { unexpected: 1 },
  ])
    assert.equal(socialWidth({ social_observation }), null);
  assert.equal(socialWidth({ consequence_observation: 1 }), null);
  assert.deepEqual(socialLabels({}), []);
});
test('Python, network labels and viewer agree for every optional feature combination', () => {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const code = `import json\nfrom wtf.learning.brain import observation_inputs\nfrom wtf.inspection.explainability import feature_names\nfrom wtf.simulation.social import social_names\nb=${JSON.stringify(template).replaceAll('true', 'True')}\nprint(json.dumps(dict(width=observation_inputs(b),names=feature_names(b),social=social_names(b))))`;
  const r = spawnSync(root + '.venv/bin/python', ['-c', code], {
    cwd: root,
    encoding: 'utf8',
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
  });
  assert.equal(r.status, 0, r.stderr);
  const p = JSON.parse(r.stdout);
  assert.equal(p.width, features.length);
  assert.deepEqual(p.social, socialLabels(template));
  assert.deepEqual(inputLabels(template, p.width), p.names);
});
class Node {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.textContent = '';
  }
  append(...n) {
    this.children.push(...n);
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
  }
}
const walk = (n) => [n, ...n.children.flatMap(walk)];
test('Inspector social charts use recorded samples, show unknown states and adapt to themes', () => {
  const original = JSON.stringify(features);
  for (const theme of ['light', 'dark']) {
    globalThis.document = {
      documentElement: { dataset: { theme } },
      createElement: (t) => new Node(t),
      createElementNS: (_, t) => new Node(t),
    };
    const root = inspectorGraphs({
      diagnostic: {
        tick: 27,
        features,
        action_probabilities: { swim: 0.2, rest: 0.8 },
      },
      template,
      spec: { body: { vision_radius: 5 }, allowed_actions: ['swim', 'rest'] },
      hasWater: true,
    });
    const nodes = walk(root),
      text = nodes.map((n) => n.textContent).join(' ');
    assert.ok(text.includes('Collective learning'));
    assert.ok(text.includes('Recorded sample at step 27'));
    assert.ok(text.includes('Local witnesses'));
    assert.ok(text.includes('Species consequence estimates'));
    assert.ok(text.includes('unknown'));
    assert.ok(nodes.filter((n) => n.tagName === 'svg').length >= 4);
    const serialized = JSON.stringify(root);
    assert.ok(!serialized.includes('NaN'));
    assert.ok(!serialized.includes('Infinity'));
  }
  assert.equal(JSON.stringify(features), original);
});

test('Playback shows saved collective samples at their own time and reverses on seek', async () => {
  const { Playback } = await import('../viewer/playback.mjs');
  const cold = [...features.slice(0, base + tracking), ...Array(72).fill(0)];
  const frames = [0, 1, 2].map((tick) => ({
    tick,
    animals: [{ id: 7, x: 1, y: 1 }],
    events: [],
    resources: [],
    metrics: { food: 0, populations: { prey: 1 } },
    ...(tick === 1
      ? {}
      : {
          diagnostics: { 7: { tick, features: tick === 0 ? cold : features } },
        }),
  }));
  const meta = {
    recording_version: 1,
    name: 'social',
    mode: 'train',
    status: 'complete',
    recording: { frames_per_chunk: 3 },
    config: {
      world: { width: 3, height: 3 },
      resources: { capacity_per_patch: 20 },
      species: { prey: { brain: 'brain' } },
      brains: { brain: template },
      experiment: { runs: 1 },
    },
    runs: [{ run: 1, frame_count: 3, obstacles: [] }],
  };
  const p = new Playback(async (url) =>
    url.endsWith('/metrics') ? [] : url.includes('/chunks/') ? frames : meta,
  );
  await p.open('social');
  const read = async (i) =>
    observationView((await p.seek(i, 7)).selected.features, template).social
      .collective[1].harm;
  assert.equal(await read(0), 0);
  assert.equal(await read(1), 0);
  assert.equal(await read(2), 0.75);
  assert.equal(await read(0), 0);
});

test('Inspector explains episode-frozen knowledge separately from changing context', () => {
  globalThis.document = {
    documentElement: { dataset: { theme: 'dark' } },
    createElement: (t) => new Node(t),
    createElementNS: (_, t) => new Node(t),
  };
  const root = inspectorGraphs({
    consequenceSchedule: 'episode',
    diagnostic: {
      tick: 27,
      features,
      action_probabilities: { swim: 0.2, rest: 0.8 },
    },
    template,
    spec: { body: { vision_radius: 5 }, allowed_actions: ['swim', 'rest'] },
    hasWater: true,
  });
  const text = walk(root)
    .map((n) => n.textContent)
    .join(' ');
  assert.ok(text.includes('frozen at episode start'));
  assert.ok(text.includes('New evidence applies next episode'));
  assert.ok(text.includes('sampled context'));
});
