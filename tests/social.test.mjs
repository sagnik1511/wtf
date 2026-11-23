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
    assert.ok(text.includes('Consequence estimates'));
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

test('Inspector updates cumulative animal charts in place without adding species dashboard charts', () => {
  globalThis.document = {
    documentElement: { dataset: { theme: 'light' } },
    createElement: (t) => new Node(t),
    createElementNS: (_, t) => new Node(t),
  };
  const social = observationView(features, template).social;
  const before = { tick: 27, animal: social, by_species: { prey: social } };
  const root = inspectorGraphs({
    diagnostic: { tick: 27, features, action_values: { swim: 1, rest: 0 } },
    collectiveAverages: before,
    template,
    spec: { body: { vision_radius: 5 }, allowed_actions: ['swim', 'rest'] },
    hasWater: true,
  });
  const originalNodes = walk(root),
    count = originalNodes.length;
  const after = structuredClone(before);
  after.tick = 28;
  after.animal.collective[1].harm = 0.25;
  root.collective.update(after);
  const updated = walk(root),
    text = updated.map((n) => n.textContent).join(' ');
  assert.equal(updated.length, count);
  assert.deepEqual(updated, originalNodes);
  assert.ok(text.includes('Running averages through step 28'));
  assert.ok(text.includes('swim harm: 25%'));
  assert.ok(!text.includes('Species average ·'));
  assert.equal(features[base + tracking + 5], 0.75);
  assert.ok(JSON.stringify(root).includes('var(--viz-'));
});

test('State-independent consequence scope keeps width but labels knowledge explicitly', () => {
  const scoped = { ...template, consequence_scope: 'species_action' };
  assert.equal(socialWidth(scoped), socialWidth(template));
  const social = observationView(features, scoped).social;
  assert.equal(social.scope, 'species_action');
  assert.ok(
    socialLabels(scoped).some((name) =>
      name.startsWith('Species action attack'),
    ),
  );
  for (const invalid of [null, false, [], 'all'])
    assert.equal(
      socialWidth({ ...template, consequence_scope: invalid }),
      null,
    );
  assert.equal(socialWidth({ consequence_scope: 'species_action' }), null);
  const root = fileURLToPath(new URL('../', import.meta.url));
  const code =
    'import json,sys\nfrom wtf.simulation.social import social_names\nprint(json.dumps(social_names(json.loads(sys.argv[1]))))';
  const process = spawnSync(
    root + '.venv/bin/python',
    ['-c', code, JSON.stringify(scoped)],
    { cwd: root, encoding: 'utf8' },
  );
  assert.equal(process.status, 0, process.stderr);
  assert.deepEqual(socialLabels(scoped), JSON.parse(process.stdout));
  globalThis.document = {
    documentElement: { dataset: { theme: 'dark' } },
    createElement: (t) => new Node(t),
    createElementNS: (_, t) => new Node(t),
  };
  const graph = inspectorGraphs({
    diagnostic: { tick: 1, features, action_values: { swim: 1, rest: 0 } },
    template: scoped,
    spec: { body: { vision_radius: 4 }, allowed_actions: ['swim', 'rest'] },
    hasWater: true,
  });
  assert.ok(
    walk(graph).some((n) =>
      n.textContent.includes('regardless of current state or action mask'),
    ),
  );
});

test('Combined scope decodes separate general/context blocks and renders both without blending', () => {
  const scoped = { ...template, consequence_scope: 'both' };
  const combined = [
    ...features.slice(0, base + tracking + 24),
    ...Array.from({ length: 12 }, () => [1, 0.8, 0.3, 0.2]).flat(),
    ...Array.from({ length: 12 }, () => [0.5, 0.1, 0.6, 0]).flat(),
  ];
  assert.equal(socialWidth(scoped), socialWidth(template) + 48);
  const social = observationView(combined, scoped).social;
  assert.equal(social.general_consequences.length, 12);
  assert.equal(social.consequences.length, 12);
  assert.equal(social.general_consequences[8].harm, 0.8);
  assert.equal(social.consequences[8].harm, 0.1);
  assert.equal(socialLabels(scoped).length, 120);
  assert.ok(socialLabels(scoped)[24].startsWith('Species action '));
  assert.ok(socialLabels(scoped)[72].startsWith('Consequence '));
  const root = fileURLToPath(new URL('../', import.meta.url));
  const code =
    'import json,sys\nfrom wtf.simulation.social import social_names\nprint(json.dumps(social_names(json.loads(sys.argv[1]))))';
  const result = spawnSync(
    root + '.venv/bin/python',
    ['-c', code, JSON.stringify(scoped)],
    { cwd: root, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(socialLabels(scoped), JSON.parse(result.stdout));
  globalThis.document = {
    documentElement: { dataset: { theme: 'light' } },
    createElement: (t) => new Node(t),
    createElementNS: (_, t) => new Node(t),
  };
  const graph = inspectorGraphs({
    diagnostic: { tick: 2, features: combined, action_values: { attack: 1 } },
    collectiveAverages: { tick: 2, animal: social },
    template: scoped,
    spec: { body: { vision_radius: 4 }, allowed_actions: ['attack'] },
    hasWater: true,
  });
  const text = walk(graph)
    .map((n) => n.textContent)
    .join(' ');
  assert.ok(text.includes('General consequence estimates'));
  assert.ok(text.includes('Context consequence estimates'));
  assert.ok(text.includes('attack harm: 80%'));
  assert.ok(text.includes('attack harm: 10%'));
  const count = walk(graph).length;
  const update = structuredClone(social);
  update.general_consequences[8].harm = 0.6;
  update.consequences[8].harm = 0.2;
  graph.collective.update({ tick: 3, animal: update });
  assert.equal(walk(graph).length, count);
  const changed = walk(graph)
    .map((n) => n.textContent)
    .join(' ');
  assert.ok(changed.includes('attack harm: 60%'));
  assert.ok(changed.includes('attack harm: 20%'));
  assert.equal(social.general_consequences[8].harm, 0.8);
});
