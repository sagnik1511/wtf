import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_NETWORK_RANK,
  rankLimit,
  inputLabels,
  animalStructure,
  archivedStructure,
  structurePlan,
  networkStructure,
  inputInterpretation,
} from '../viewer/network-structure.mjs';
class Node {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.className = '';
    this.textContent = '';
    this.style = {};
  }
  append(...n) {
    this.children.push(...n);
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
  }
}
globalThis.document = {
  createElement: (t) => new Node(t),
  createElementNS: (_, t) => new Node(t),
};
const walk = (n) => [n, ...n.children.flatMap(walk)],
  text = (n) =>
    walk(n)
      .map((n) => n.textContent)
      .join(' ');
const descriptor = (hidden) => ({
  rank: hidden.length,
  available: true,
  stages: [
    { label: 'Input', values: Array(153).fill(0.2) },
    ...hidden.map((n, i) => ({
      label: `Hidden ${i}`,
      values: Array(n).fill(-0.4),
    })),
    { label: 'Outputs', values: Array(7).fill(0.1) },
  ],
});

test('Default k=2 means hidden-block depth, includes the boundary and counts every unit', () => {
  assert.equal(DEFAULT_NETWORK_RANK, 2);
  assert.equal(rankLimit(undefined), 2);
  assert.equal(rankLimit(''), 2);
  assert.equal(rankLimit('3'), 3);
  assert.equal(rankLimit(-1), 2);
  const plan = structurePlan(descriptor([64, 32]));
  assert.equal(plan.full, true);
  assert.equal(plan.rank, 2);
  assert.equal(plan.nodes, 256);
  assert.equal(plan.connections, 12064);
  const graph = networkStructure(descriptor([64, 32]));
  assert.equal(walk(graph).filter((n) => n.tagName === 'rect').length, 256);
  assert(text(graph).includes('bundles'));
  assert(text(graph).includes('topology, not weight strength'));
});

test('Deep networks default to input/output summary; changing k reveals all units', () => {
  const d = descriptor([16, 8, 4]);
  assert.equal(structurePlan(d).full, false);
  assert.equal(
    walk(networkStructure(d)).filter((n) => n.tagName === 'rect').length,
    0,
  );
  assert(text(networkStructure(d)).includes('rank 3 > k 2'));
  assert.equal(
    walk(networkStructure(d, 3)).filter((n) => n.tagName === 'rect').length,
    188,
  );
});

test('Recorded structure counts GRU as one block and uses actual action scores and memory activations', () => {
  const d = {
    algorithm: 'recurrent_ppo',
    features: Array(20).fill(0),
    action_probabilities: { rest: 0.7, eat: 0.3 },
    state_value: 0.1,
    activations: { version: 1, layers: [[0.2, -0.2], [0.1]] },
  };
  const plan = animalStructure(d, { type: 'recurrent_ppo' });
  assert.equal(plan.rank, 2);
  assert(plan.stages[1].recurrent);
  assert.equal(plan.stages.at(-1).values.length, 3);
  assert.deepEqual(plan.stages[1].values, [0.2, -0.2]);
  delete d.activations;
  const declared = animalStructure(d, {
    type: 'recurrent_ppo',
    recurrent_size: 2,
    network: { hidden_layers: [1] },
  });
  assert.equal(declared.available, false);
  assert(
    text(networkStructure(declared)).includes(
      'gray units show declared architecture only',
    ),
  );
});

test('Input labels match version/top-k layout and summaries describe observations without claiming influence', () => {
  for (const [version, count] of [
    [1, 20],
    [2, 32],
    [3, 42],
    [4, 45],
  ])
    assert.equal(
      inputLabels(
        { observation_version: version, observation_top_k: 3 },
        count + 24,
      ).length,
      count + 24,
    );
  assert.equal(
    inputLabels({ observation_version: 4, observation_top_k: 3 }, 69)[68],
    'Consenting pair',
  );
  const x = Array(20).fill(0);
  x[0] = 0.6;
  x[1] = 0.9;
  const panel = inputInterpretation(
    { features: x },
    { observation_version: 1 },
  );
  assert(text(panel).includes('60%'));
  assert(text(panel).includes('90%'));
  assert(text(panel).includes('not causal importance'));
});

test('Archived topology uses full activation arrays rather than the pruned weighted-route nodes', () => {
  const r = {
    features: Array.from({ length: 20 }, (_, i) => ({
      label: `Input ${i}`,
      value: 0,
    })),
    layers: [
      { label: 'Input', size: 20, values: [], nodes: [] },
      {
        label: 'Hidden',
        size: 40,
        values: Array(40).fill(0.2),
        nodes: [{ index: 0, label: '0' }],
      },
      {
        label: 'Output',
        size: 2,
        values: [0.1, 0.9],
        nodes: [
          { index: 0, label: 'rest' },
          { index: 1, label: 'eat' },
        ],
      },
    ],
  };
  const d = archivedStructure(r);
  assert.equal(d.rank, 1);
  assert.equal(structurePlan(d).nodes, 62);
  assert.equal(
    walk(networkStructure(d)).filter((n) => n.tagName === 'rect').length,
    62,
  );
});

test('LSTM and separate actor critics use honest network routes', () => {
  const diagnostic = {
    algorithm: 'lstm_ppo',
    features: Array(20).fill(0),
    action_probabilities: { rest: 0.4, attack: 0.6 },
    state_value: 0.2,
    activations: {
      version: 1,
      layers: [
        [0.1, 0.2],
        [0.3, 0.4, 0.5],
      ],
    },
  };
  const template = {
    type: 'lstm_ppo',
    recurrent_size: 2,
    network: { hidden_layers: [3] },
  };
  const recurrent = animalStructure(diagnostic, template);
  assert.equal(recurrent.stages[1].label, 'LSTM memory block');
  for (const algorithm of ['trpo', 'discrete_sac']) {
    const actor = animalStructure(
      {
        ...diagnostic,
        algorithm,
        activations: { version: 1, layers: [[0.1, 0.2, 0.3]] },
      },
      { ...template, type: algorithm },
    );
    assert.equal(actor.stages.at(-1).values.length, 2);
  }
});
