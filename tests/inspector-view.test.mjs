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

test('Version four renders recorded consent gauges and escape prediction bars', () => {
  const f = [
    ...features,
    ...Array(12).fill(0),
    -1,
    1,
    0,
    1,
    0.2,
    0,
    0.2,
    1,
    1,
    1,
    0.8,
    0.4,
    1,
  ];
  const root = inspectorGraphs({
    diagnostic: {
      tick: 1,
      features: f,
      action_probabilities: { strafe_left: 0.7, rest: 0.3 },
    },
    template: { observation_version: 4 },
    spec: { body },
    hasWater: false,
    samples: [],
    sections: { mating: true },
  });
  const labels = text(root);
  assert(labels.includes('45 inputs'));
  assert(labels.includes('Own consent remaining'));
  assert(labels.includes('Visible mate consent remaining'));
  assert(labels.includes('Nearby consenting pair opportunity yes'));
  assert(labels.includes('strafe left'));
});

test('Movement and scent inspector plots use sampled history and anonymous species/sex groups', () => {
  const template = {
    observation_version: 1,
    movement_history_steps: 2,
    scent_observation: { species: ['prey'], radius: 2 },
  };
  const track = [0.5, 0, 0.25, 0, 0.5, 2 / 3, 0, 0.5, 0, 1, 0, 0, 0];
  const root = inspectorGraphs({
    diagnostic: {
      tick: 8,
      features: [
        ...features,
        ...track,
        ...Array(13).fill(0),
        ...Array(18).fill(0.5),
      ],
      action_probabilities: { rest: 1 },
    },
    template,
    spec: { body },
    hasWater: false,
  });
  const labels = text(root);
  assert(labels.includes('Movement memory & scent'));
  assert(labels.includes('1 steps since seen'));
  assert(labels.includes('67% confidence'));
  assert.equal(
    walk(root).filter(
      (n) => n.tagName === 'p' && /track \d/.test(n.textContent),
    ).length,
    0,
  );
  assert(labels.includes('prey · female'));
  assert(labels.includes('prey · male'));
  assert(labels.includes('Local scent · radius 2'));
  assert(!labels.includes('not one animal'));
  assert.equal(
    walk(root).filter((n) => n.attributes.class === 'scent-grid').length,
    2,
  );
  assert.equal(
    walk(root).filter((n) => n.attributes.class === 'movement-history').length,
    1,
  );
  assert(walk(root).some((n) => n.attributes['stroke-dasharray'] === '3 2'));
  assert(
    walk(root).some((n) => n.textContent.includes('own scent may be included')),
  );
});

test('Blended scent renders exactly one anonymous heatmap', () => {
  const root = inspectorGraphs({
    diagnostic: {
      tick: 1,
      features: [...features, ...Array(9).fill(0.4)],
      action_probabilities: { rest: 1 },
    },
    template: { scent_observation: { mode: 'blended' } },
    spec: { body },
    hasWater: false,
  });
  assert.equal(
    walk(root).filter((n) => n.attributes.class === 'scent-grid').length,
    1,
  );
  assert(text(root).includes('Blended scent'));
  assert(!text(root).includes('all sexes'));
  assert(text(root).includes('40% blended strength'));
});

test('Sampled radar marks the front half using recorded heading; old full circles remain', () => {
  const root = inspectorGraphs({
    diagnostic: { tick: 10, features, action_probabilities: { rest: 1 } },
    template: { observation_version: 1 },
    spec: { body: { ...body, field_of_view_degrees: 180 } },
    hasWater: false,
  });
  const sector = walk(root).find(
    (n) => n.attributes['data-visual-field'] === '180',
  );
  assert(sector);
  assert(sector.attributes.d.includes('A 76 76'));
  assert(text(root).includes('front 180°'));
  const legacy = inspectorGraphs({
    diagnostic: { tick: 10, features, action_probabilities: { rest: 1 } },
    template: { observation_version: 1 },
    spec: { body },
    hasWater: false,
  });
  assert(
    !walk(legacy).some((n) => n.attributes['data-visual-field'] === '180'),
  );
});

test('Narrow sampled radar uses a 150-degree sector and caption', () => {
  const root = inspectorGraphs({
    diagnostic: { tick: 10, features, action_probabilities: { rest: 1 } },
    template: { observation_version: 1 },
    spec: { body: { ...body, field_of_view_degrees: 150 } },
    hasWater: false,
  });
  assert(walk(root).some((n) => n.attributes['data-visual-field'] === '150'));
  assert(text(root).includes('front 150°'));
});

test('V5 inspector identifies its input frame and shows unmasked immediate opportunities', () => {
  const values = Array(48).fill(0);
  values.splice(0, 4, 0.2, 1, 1, 0);
  values.splice(45, 3, 1, 0, 1);
  const root = inspectorGraphs({
    diagnostic: {
      tick: 9,
      features: values,
      action_probabilities: { attack: 0.2, rest: 0.8 },
    },
    template: { observation_version: 5 },
    spec: { body },
    hasWater: true,
  });
  assert(text(root).includes('v5'));
  assert(text(root).includes('forward/right inputs'));
  assert(text(root).includes('attack: possible'));
  assert(text(root).includes('eat: unavailable'));
  assert(text(root).includes('swim: possible'));
  assert(!text(root).includes('attack · masked'));
});

test('V6 inspector shows semantic memberships without masking ordinary predictions', () => {
  const template = { observation_version: 6 };
  // Mixed memberships must be visible while the raw flags remain unmasked.
  const values = Array(130).fill(0);
  values.splice(0, 4, 0, 0.7, 0.3, 0);
  values[7] = 1;
  const root = inspectorGraphs({
    diagnostic: {
      tick: 9,
      features: values,
      action_probabilities: { attack: 0.3, rest: 0.7 },
    },
    template,
    spec: { body },
    hasWater: true,
    sections: { semantic: true },
  });
  assert(text(root).includes('V6 soft categories'));
  assert(text(root).includes('Energy hungry: 70%'));
  assert(text(root).includes('Energy comfortable: 30%'));
  assert(text(root).includes('category-derived positions'));
  assert(!text(root).includes('attack · masked'));
});

test('V7 inspector separates compact scent from regional heatmaps and shows final-policy adjustments', async () => {
  const { compactLayout } = await import('../viewer/input-layout.mjs');
  const template = {
    observation_version: 7,
    observation_top_k: 1,
    observation_actions: ['rest', 'eat'],
    network: { hidden_layers: [4] },
    scent_observation: { mode: 'blended', radius: 2 },
    consequence_observation: true,
    consequence_scope: 'both',
    late_consequence: { max_adjustment: 2 },
  };
  const layout = compactLayout(template),
    values = Array(layout.width).fill(0);
  values[2] = 1;
  values.splice(layout.scentStart, 4, 1, 0, 0.9, 0.4);
  const d = {
    tick: 2,
    algorithm: 'ppo',
    features: values,
    action_probabilities: { rest: 0.2, eat: 0.8 },
    late_consequence: {
      base_logits: [0, 0],
      adjustments: [-0.5, 0.5],
      final_logits: [-0.5, 0.5],
    },
  };
  const root = inspectorGraphs({
    diagnostic: d,
    template,
    spec: { body },
    hasWater: false,
  });
  assert(
    walk(root).some(
      (n) =>
        n.attributes?.['aria-label'] === 'Scent bearing relative to the animal',
    ),
  );
  assert(!text(root).includes('Body-relative bearing:'));
  assert(!text(root).includes('Movement shows a trend'));
  assert(text(root).includes('Before consequences'));
  assert(text(root).includes('Strength'));
  assert(text(root).includes('Freshness'));
  assert(text(root).includes('80.0%'));
  assert(
    walk(root).some(
      (n) =>
        n.attributes?.['aria-label'] ===
        'Signed consequence adjustments to action logits',
    ),
  );
  assert(!walk(root).some((n) => n.attributes?.class === 'scent-grid'));
});

test('Masked late-policy comparison uses the same mask before and after consequences', async () => {
  const { compactLayout } = await import('../viewer/input-layout.mjs');
  const template = {
    observation_version: 7,
    observation_actions: ['rest', 'eat'],
    consequence_observation: true,
    late_consequence: { max_adjustment: 2 },
    network: { hidden_layers: [4] },
  };
  const layout = compactLayout(template),
    features = Array(layout.width).fill(0);
  features[2] = 1;
  const root = inspectorGraphs({
    diagnostic: {
      tick: 1,
      algorithm: 'maskable_ppo',
      features,
      action_probabilities: { rest: 1, eat: 0 },
      action_mask: { rest: true, eat: false },
      late_consequence: {
        base_logits: [0, 0],
        adjustments: [0, 0],
        final_logits: [0, 0],
      },
    },
    template,
    spec: { body },
    hasWater: false,
  });
  const before = walk(root).find(
    (n) => n.children?.[0]?.textContent === 'Before consequences',
  );
  assert.ok(before);
  assert(text(before).includes('100.0%'));
  assert(!text(before).includes('50.0%'));
});
