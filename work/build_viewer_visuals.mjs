#!/usr/bin/env node
import {
  observationLayout,
  observationComparison,
  compactPolicyVisual,
} from './build_observation_layout.mjs';
import {
  semanticLayout,
  rawInputLabels,
  compactLayout,
} from '../viewer/input-layout.mjs';
import { decodeSoft } from '../viewer/soft-observations.mjs';
import { buildDocAnimations } from './build_doc_animations.mjs';
import { drawFeedings } from '../viewer/feeding-highlights.mjs';
import {
  drawEnvironmentObjects,
  drawEnvironmentShadows,
} from '../viewer/environment-objects.mjs';
import { drawVisualTracks, trackingView } from '../viewer/tracking.mjs';
import { drawMovementHeatmap } from '../viewer/behavior-stats.mjs';
import { drawContests } from '../viewer/competition.mjs';
import {
  daylight,
  circadianBody,
  drawDaylight,
  drawSunlitGround,
  clockMarkup,
  clockState,
} from '../viewer/day-night.mjs';
// Controlled documentation specimens, not experiment replays or learned outputs.
import { boardPalette } from '../viewer/theme.mjs';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { speciesStyles } from '../viewer/species-style.mjs';
import { drawTerrain, terrainView } from '../viewer/terrain.mjs';
import { ObservationOverlay } from '../viewer/inspector-stats.mjs';
import { inspectorGraphs } from '../viewer/inspector.mjs';
import { CollectiveTotals } from '../viewer/collective-history.mjs';
import { explanationCharts } from '../viewer/explainability.mjs';
import {
  animalStructure,
  networkStructure,
} from '../viewer/network-structure.mjs';
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url))),
  outdir = path.join(root, 'docs/assets/viewer');
const esc = (s) =>
  String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
const num = (n) => Number(Number(n).toFixed(4));
const text = (
  x,
  y,
  value,
  size = 16,
  fill = '#e6edf3',
  anchor = 'start',
  bold = false,
) =>
  `<text x="${x}" y="${y}" font-family="system-ui,sans-serif" font-size="${size}" fill="${fill}" text-anchor="${anchor}"${bold ? ' font-weight="600"' : ''}>${esc(value)}</text>`;
const box = (x, y, w, h) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#111c27" stroke="#415569"/>`;
const lines = (x, y, values, size = 14) =>
  values.map((v, i) => text(x, y + i * 20, v, size, '#a9bacc')).join('');
const sheet = (title, description, height, body) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="${height}" viewBox="0 0 1200 ${height}" role="img" aria-labelledby="title desc"><title id="title">${esc(title)}</title><desc id="desc">${esc(description)}</desc><style>.viewer-component text{font:12px system-ui,sans-serif;fill:#a9bacc}</style><rect class="sheet-background" width="1200" height="${height}" rx="14" fill="#0b1118"/><g class="sheet-heading">${text(28, 38, title, 25, '#e6edf3', 'start', true)}${text(28, 67, description, 14, '#a9bacc')}</g>${body}</svg>\n`;

class SVGCanvas {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.style = { width: `${width}px` };
    this.output = [];
    this.stack = [];
    this.transform = [];
    this.fillStyle = '#000';
    this.strokeStyle = '#000';
    this.lineWidth = 1;
    this.globalAlpha = 1;
    this.font = '14px system-ui';
    this.textAlign = 'left';
    this.lineDash = [];
    this.textBaseline = 'alphabetic';
  }
  getContext() {
    return this;
  }
  save() {
    this.stack.push(
      Object.fromEntries(
        [
          'fillStyle',
          'strokeStyle',
          'lineWidth',
          'globalAlpha',
          'font',
          'textAlign',
          'lineDash',
          'textBaseline',
          'shadowBlur',
          'shadowColor',
        ].map((k) => [k, this[k]]),
      ),
    );
    this.stack.at(-1).transform = [...this.transform];
  }
  restore() {
    Object.assign(this, this.stack.pop());
  }
  translate(x, y) {
    this.transform.push(`translate(${num(x)} ${num(y)})`);
  }
  rotate(v) {
    this.transform.push(`rotate(${num((v * 180) / Math.PI)})`);
  }
  attrs() {
    return `opacity="${num(this.globalAlpha)}" transform="${this.transform.join(' ')}"`;
  }
  fillRect(x, y, w, h) {
    this.output.push(
      `<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" fill="${esc(this.fillStyle)}" ${this.attrs()}/>`,
    );
  }
  beginPath() {
    this.d = '';
  }
  moveTo(x, y) {
    this.d += `M${num(x)} ${num(y)}`;
  }
  lineTo(x, y) {
    this.d += `L${num(x)} ${num(y)}`;
  }
  quadraticCurveTo(cx, cy, x, y) {
    this.d += `Q${num(cx)} ${num(cy)} ${num(x)} ${num(y)}`;
  }
  closePath() {
    this.d += 'Z';
  }
  arc(x, y, r, start, end) {
    assert(
      Math.abs(end - start - Math.PI * 2) < 1e-7,
      'Only full-circle board arcs are currently supported',
    );
    this.d += `M${num(x + r)} ${num(y)}A${num(r)} ${num(r)} 0 1 0 ${num(x - r)} ${num(y)}A${num(r)} ${num(r)} 0 1 0 ${num(x + r)} ${num(y)}Z`;
  }
  fill() {
    this.output.push(
      `<path d="${this.d}" fill="${esc(this.fillStyle)}" ${this.attrs()}/>`,
    );
  }
  setLineDash(values) {
    this.lineDash = [...values];
  }
  stroke() {
    this.output.push(
      `<path d="${this.d}" fill="none" stroke="${esc(this.strokeStyle)}" stroke-width="${this.lineWidth}"${this.lineDash.length ? ` stroke-dasharray="${this.lineDash.join(' ')}"` : ''} ${this.attrs()}/>`,
    );
  }
  strokeText(value, x, y) {
    const start = this.output.length;
    this.fillText(value, x, y);
    this.output[start] = this.output[start].replace(
      `fill="${esc(this.fillStyle)}"`,
      `fill="none" stroke="${esc(this.strokeStyle)}" stroke-width="${this.lineWidth}" stroke-linejoin="round"`,
    );
  }
  fillText(value, x, y) {
    const size = this.font.match(/([\d.]+)px/)?.[1] ?? 14;
    this.output.push(
      `<text x="${num(x)}" y="${num(y)}" font-family="system-ui,sans-serif" font-size="${size}" ${this.textBaseline === 'bottom' ? 'dominant-baseline="text-after-edge" ' : ''}font-weight="${this.font.includes('bold') ? '700' : '400'}" text-anchor="${this.textAlign === 'center' ? 'middle' : this.textAlign === 'right' ? 'end' : 'start'}" fill="${esc(this.fillStyle)}" ${this.attrs()}>${esc(value)}</text>`,
    );
  }
}
const source = fs.readFileSync(path.join(root, 'viewer/app.js'), 'utf8');
const sourceFunction = (name) => {
  const start = source.indexOf(`function ${name}() {`);
  assert(start >= 0, `Missing ${name}`);
  const end = source.indexOf('\nfunction ', start + 1);
  assert(end > start, `No function boundary after ${name}`);
  return source.slice(start, end);
};
const baseSpecies = {
  prey: { diet: 'resource', body: { vision_radius: 2, water_vision_bonus: 2 } },
  predator: { diet: 'prey', body: { vision_radius: 2 } },
  grazer: { diet: 'resource', body: { vision_radius: 2 } },
};
const styles = speciesStyles(baseSpecies);
assert.equal(styles.get('prey').female, '#7de2bd');
assert.equal(styles.get('predator').male, '#f6d365');
function board({
  width = 3,
  height = 3,
  animals = [],
  resources = [],
  obstacles = [],
  selected = null,
  definition = null,
  sight = false,
  births = [],
  feedings = [],
  kills = [],
  deaths = [],
  contests = [],
  theme = 'dark',
  day_night = null,
  tick = 0,
  tracking = null,
  reduced = false,
  fieldOfView = 360,
} = {}) {
  const canvas = new SVGCanvas(width * 48, height * 48),
    terrain = terrainView(definition),
    state = {
      width,
      height,
      day_night,
      tick,
      animals,
      resources,
      obstacles,
      species:
        fieldOfView === 360
          ? baseSpecies
          : Object.fromEntries(
              Object.entries(baseSpecies).map(([name, spec]) => [
                name,
                {
                  ...spec,
                  body: { ...spec.body, field_of_view_degrees: fieldOfView },
                },
              ]),
            ),
      resource_capacity: 20,
      contest_highlights: contests,
      birth_highlights: births,
      feeding_highlights: feedings,
      kill_highlights: kills,
      natural_death_highlights: deaths,
    };
  const controls = {
    world: canvas,
    'show-observation': { checked: sight },
    'observation-status': {},
  };
  const context = vm.createContext({
    document: { documentElement: { dataset: { theme } } },
    boardPalette,
    dayNightClock: { update() {} },
    daylight,
    circadianBody,
    drawDaylight,
    drawSunlitGround,
    drawContests,
    state,
    selected,
    terrain,
    drawTerrain,
    drawEnvironmentObjects,
    drawEnvironmentShadows,
    drawFeedings,
    reducedMotion: { matches: reduced },
    observationOverlay: new ObservationOverlay(),
    $: (id) => controls[id],
    fitWorld: () => {},
    animalColor: (a) =>
      styles.get(a.species)?.[a.sex] ?? styles.get(a.species)?.color,
  });
  vm.runInContext(
    [
      'drawWorld',
      'drawFeedingHighlights',
      'drawBirthHighlights',
      'drawDeathHighlights',
      'drawContestHighlights',
    ]
      .map(sourceFunction)
      .join('\n') +
      '\ndrawWorld();drawFeedingHighlights();drawBirthHighlights();drawDeathHighlights();drawContestHighlights();',
    context,
    { timeout: 1000 },
  );
  if (tracking) drawVisualTracks(canvas, tracking, 48, 48, theme === 'light');
  return `<svg width="${canvas.width}" height="${canvas.height}" viewBox="0 0 ${canvas.width} ${canvas.height}">${canvas.output.join('')}</svg>`;
}
const animal = (
  species = 'prey',
  sex = 'female',
  heading = 0,
  id = 0,
  x = 1,
  y = 1,
) => ({ id, species, sex, heading, x, y });
function tile(x, y, title, subtitle, scene, w = 268, h = 220) {
  return (
    box(x, y, w, h) +
    text(x + 14, y + 27, title, 17, '#e6edf3', 'start', true) +
    `<g transform="translate(${x + (w - 144) / 2} ${y + 43})">${scene}</g>` +
    lines(x + 14, y + h - 27, subtitle, 13)
  );
}
const objectCards = [
  ['Prey · female', ['Mint circle; heading dot'], { animals: [animal()] }],
  [
    'Prey · male',
    ['Blue circle; light outline'],
    { animals: [animal('prey', 'male')] },
  ],
  [
    'Predator · female',
    ['Coral triangle; heading dot'],
    { animals: [animal('predator')] },
  ],
  [
    'Predator · male',
    ['Gold triangle; light outline'],
    { animals: [animal('predator', 'male')] },
  ],
  [
    'Resource · empty',
    ['Bare rooted patch (0/20)'],
    { resources: [{ x: 1, y: 1, quantity: 0 }] },
  ],
  [
    'Resource · half full',
    ['Growing leafy patch (10/20)'],
    { resources: [{ x: 1, y: 1, quantity: 10 }] },
  ],
  [
    'Resource · full',
    ['Full leafy patch (20/20)'],
    { resources: [{ x: 1, y: 1, quantity: 20 }] },
  ],
  ['Obstacle', ['Faceted rock; blocks whole cell'], { obstacles: [[1, 1]] }],
  [
    'Selected animal',
    ['White ring; opens inspector'],
    { animals: [animal()], selected: 0 },
  ],
  [
    'Birth',
    ['Cyan ring, plus and BORN'],
    { animals: [animal()], births: [{ target: 0, position: [1, 1], age: 0 }] },
  ],
  [
    'Kill',
    ['Red cross; hunter gets gold ring'],
    {
      animals: [animal('predator', 'female', 1, 1, 0, 1)],
      kills: [{ actor: 1, position: [1, 1], age: 0 }],
    },
  ],
  [
    'Natural death · prey',
    ['Green ring / cross and DIED'],
    { deaths: [{ species: 'prey', position: [1, 1], age: 0 }] },
  ],
  [
    'Natural death · predator',
    ['Amber ring / cross and DIED'],
    { deaths: [{ species: 'predator', position: [1, 1], age: 0 }] },
  ],
  [
    'Facing east',
    ['Heading dot rotates with body'],
    { animals: [animal('prey', 'female', 1)] },
  ],
  [
    'Legacy · sex not recorded',
    ['Base color; never infer sex'],
    { animals: [{ ...animal(), sex: undefined }] },
  ],
  [
    'Additional species',
    ['Automatic palette or JSON colors'],
    { animals: [animal('grazer')] },
  ],
  ...['prey', 'predator'].map((species) => [
    `Challenge · ${species}`,
    ['Paired rings; winner / damage label'],
    {
      width: 4,
      animals: [
        animal(species, 'female', 1, 0, 1, 1),
        animal(species, 'male', 3, 1, 2, 1),
      ],
      contests: [
        {
          actor: 0,
          target: 1,
          position: [1, 1],
          loser_position: [2, 1],
          amount: 5,
          cause: 'aggression',
          age: 0,
        },
      ],
    },
  ]),
  [
    'Challenge · shared food',
    ['Same marker; banner names context'],
    {
      width: 4,
      resources: [{ x: 2, y: 1, quantity: 20 }],
      animals: [
        animal('prey', 'female', 1, 0, 1, 1),
        animal('prey', 'male', 3, 1, 2, 1),
      ],
      contests: [
        {
          actor: 0,
          target: 1,
          position: [1, 1],
          loser_position: [2, 1],
          amount: 5,
          cause: 'food',
          age: 0,
        },
      ],
    },
  ],
  [
    'Competition death',
    ['DIED marker; cause in event banner'],
    {
      deaths: [
        { species: 'prey', position: [1, 1], cause: 'competition', age: 0 },
      ],
    },
  ],
];
let body = objectCards
  .map(([title, caption, options], i) =>
    tile(
      28 + (i % 4) * 288,
      95 + Math.floor(i / 4) * 236,
      title,
      caption,
      board(options).replace('width="192"', 'width="144"'),
    ),
  )
  .join('');
const objects = sheet(
  'Board objects and event markers',
  'Actual viewer drawing code · enlarged controlled specimens · event age 0 · canvas glow omitted',
  95 + Math.ceil(objectCards.length / 4) * 236 + 21,
  body,
);
function geography(rows, legend) {
  return { name: 'Controlled example', width: rows[0].length, rows, legend };
}
const land = {
  '.': { elevation: 0 },
  '~': { water_depth: 1, flow: [0, 1] },
  D: { water_depth: 3, flow: [1, 0] },
  1: { elevation: 10 },
  2: { elevation: 20 },
};
const river = geography(['..~..', '..~..', '..~..', '..~..', '..~..'], land);
const terrainCards = [
  [
    'Flat land',
    ['Board ground and faint grid.', 'No terrain-specific shading.'],
    { width: 5, height: 5, animals: [animal('prey', 'female', 0, 0, 2, 2)] },
  ],
  [
    'River and flow',
    ['Blue tiles; arrows show current.', 'Depth changes the blue shade.'],
    {
      width: 5,
      height: 5,
      definition: geography(
        ['..~..', '..~D.', '..~D.', '..~..', '..~..'],
        land,
      ),
    },
  ],
  [
    'Elevation',
    [
      'Terrain colors scale with map height.',
      'Sparse numerals mark elevation.',
    ],
    {
      width: 5,
      height: 5,
      definition: geography(
        ['....1', '...11', '..112', '.1122', '11222'],
        land,
      ),
    },
  ],
  [
    'Wall-limited sight',
    [
      'Blue wash = selected animal’s sight.',
      'Walls occlude rays, not the board.',
    ],
    {
      width: 5,
      height: 5,
      selected: 0,
      sight: true,
      animals: [animal('prey', 'female', 1, 0, 1, 2)],
      obstacles: [
        [2, 1],
        [2, 2],
        [2, 3],
      ],
    },
  ],
  [
    'Water-directed sight',
    ['Water can extend permitted rays.', 'Same selected sight overlay.'],
    {
      width: 5,
      height: 5,
      definition: river,
      selected: 0,
      sight: true,
      animals: [animal('prey', 'female', 1, 0, 1, 2)],
    },
  ],
  [
    'Enclosure and gate',
    ['Walls form the enclosure.', 'The gap remains traversable.'],
    {
      width: 5,
      height: 5,
      animals: [
        animal('prey', 'female', 0, 0, 2, 2),
        animal('predator', 'female', 2, 1, 2, 0),
      ],
      obstacles: [
        [1, 1],
        [3, 1],
        [1, 2],
        [3, 2],
        [1, 3],
        [2, 3],
        [3, 3],
      ],
    },
  ],
];
body = terrainCards
  .map(([title, caption, options], i) => {
    const x = 28 + (i % 3) * 390,
      y = 95 + Math.floor(i / 3) * 340;
    return (
      box(x, y, 364, 320) +
      text(x + 16, y + 29, title, 18, '#e6edf3', 'start', true) +
      `<g transform="translate(${x + 62} ${y + 45}) scale(.85)">${board(options)}</g>` +
      lines(x + 16, y + 270, caption, 14)
    );
  })
  .join('');
const terrains = sheet(
  'Terrain, geography and current sight',
  'Actual terrain renderer and sight overlay · constructed 5 × 5 scenes, not environment-run results',
  790,
  body,
);

// Tiny DOM adapter: execute pure SVG UI components, never app bootstrap or training.
class Node {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.className = '';
    this.textContent = '';
    this.open = false;
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
const walk = (n) => [n, ...n.children.flatMap(walk)];
function serialize(n) {
  const attrs = { ...n.attributes };
  if (n.className) attrs.class = n.className;
  return `<${n.tagName}${Object.entries(attrs)
    .map(([k, v]) => ` ${k}="${esc(v)}"`)
    .join(
      '',
    )}>${esc(n.textContent)}${n.children.map(serialize).join('')}</${n.tagName}>`;
}
function component(n, x, y, w, h) {
  const vb = n.attributes.viewBox.split(' ').map(Number),
    scale = Math.min(w / vb[2], h / vb[3]);
  return `<g class="viewer-component" transform="translate(${x} ${y}) scale(${num(scale)})">${n.children.map(serialize).join('')}</g>`;
}
const legacyFeatures = [
  0.6,
  0.85,
  0,
  -1,
  0,
  1,
  0,
  0,
  0.25,
  0.5,
  Math.hypot(0.25, 0.5),
  1,
  0,
  0,
  0,
  0,
  0.6,
  0,
  0.6,
  1,
];
const template = {
  type: 'ppo',
  observation_version: 7,
  observation_top_k: 3,
  observation_actions: ['rest', 'move_backward', 'eat'],
  movement_history_steps: 3,
  scent_observation: { mode: 'blended', radius: 2 },
  network: { hidden_layers: [4, 3] },
};
const features = Array(compactLayout(template).names.length).fill(0);
features.splice(0, 8, 0.6, 0.85, 0, -1, 0, 1, 0, 0);
features.splice(8, 3, 0.25, 0.5, 1);
const diagnostic = {
  tick: 20,
  algorithm: 'ppo',
  features,
  updates: 4,
  action_probabilities: { rest: 0.65, move_backward: 0.35, eat: 0 },
  action_mask: { rest: true, move_backward: true, eat: false },
  state_value: 0.12,
  activations: {
    version: 1,
    layers: [
      [0.2, -0.4, 0.7, -0.1],
      [0.35, -0.2, 0.1],
    ],
  },
};
const inspector = inspectorGraphs({
  diagnostic,
  template,
  spec: { body: { vision_radius: 4 } },
  hasWater: false,
});
const radar = walk(inspector).find((n) => n.tagName === 'svg');
assert(radar, 'Inspector radar missing');
const r = {
  algorithm: 'ppo',
  target: 'rest',
  policy: true,
  features: features.map((value, index) => ({
    index,
    label: `Input ${index}`,
    value,
    gradient: index % 2 ? 0.03 : -0.02,
    attribution: value * (index % 2 ? 0.03 : -0.02),
  })),
  outputs: Object.entries(diagnostic.action_probabilities).map(
    ([action, score]) => ({ action, score, masked: action === 'eat' }),
  ),
  recorded_policy: true,
  recorded_predictions: { rest: 0.45, move_backward: 0.55, eat: 0 },
  method: 'input × local gradient of action probability',
  layers: [
    {
      label: 'Observation',
      activation: 'input',
      size: features.length,
      nodes: [],
      values: [],
    },
    {
      label: 'Hidden 1',
      activation: 'tanh',
      size: 4,
      nodes: [],
      values: [0.2, -0.4, 0.7, -0.1],
    },
    {
      label: 'Hidden 2',
      activation: 'tanh',
      size: 3,
      nodes: [],
      values: [0.35, -0.2, 0.1],
    },
    {
      label: 'Output logits',
      activation: 'linear',
      size: 4,
      nodes: [],
      values: [0.6, 0, -1, 0.12],
    },
  ],
  edges: [],
};
r.top_features = r.features.slice(0, 8);
const charts = explanationCharts(r),
  svgs = walk(charts).filter((n) => n.tagName === 'svg');
assert(
  svgs.length === 3,
  'Default archive view should show influence, output and topology only',
);
const structure = walk(
  networkStructure(animalStructure(diagnostic, template)),
).find((n) => n.tagName === 'svg');
body =
  box(28, 95, 560, 350) +
  text(
    46,
    125,
    'Sampled observations in the inspector',
    19,
    '#e6edf3',
    'start',
    true,
  ) +
  component(radar, 120, 146, 350, 240) +
  lines(46, 411, ['Circle = sampled target range; dots = input slots.'], 14);
body +=
  box(612, 95, 560, 350) +
  text(
    630,
    125,
    'Input → predicted action scores',
    19,
    '#e6edf3',
    'start',
    true,
  ) +
  component(svgs[1], 630, 158, 520, 235) +
  lines(630, 411, ['Colored = archived inference; gray = recorded score.'], 14);
body +=
  box(28, 465, 560, 400) +
  text(
    46,
    495,
    'Whole structure · rank 2 ≤ k 2',
    19,
    '#e6edf3',
    'start',
    true,
  ) +
  component(structure, 46, 519, 520, 280) +
  lines(46, 830, ['All units; lines show topology, not causal influence.'], 14);
body +=
  box(612, 465, 560, 400) +
  text(630, 495, 'Deeper brain · rank 3 > k 2', 19, '#e6edf3', 'start', true) +
  lines(
    630,
    548,
    [
      `${features.length} inputs → 4 → 3 → 2 → action scores`,
      'Default: input/output summary.',
      'Advanced: neuron maps and weighted routes.',
      'Rank threshold changes display only.',
      'Follow replay: update on new observation samples.',
      'What-if: pin the observation and pause.',
    ],
    16,
  ) +
  lines(630, 830, ['Here k is hidden-stage depth, not observation top-k.'], 14);
const brains = sheet(
  'Inspector and brain-view appearances',
  'Source-rendered SVG components · illustrative inputs, activations and scores; no learned result is implied',
  900,
  body,
);

// Actual inspector components with controlled, anonymous sensory specimens.
const sensoryTemplate = {
  type: 'ppo',
  observation_version: 1,
  movement_history_steps: 3,
  scent_observation: { mode: 'blended', radius: 2 },
};
const sensoryFeatures = [
  ...legacyFeatures,
  0.6,
  0,
  0.2,
  0,
  0,
  1,
  1,
  0.4,
  0,
  1,
  0.2,
  0,
  1,
  0,
  0,
  0,
  -0.4,
  0.2,
  -0.1,
  0,
  1 / 3,
  0.75,
  0,
  -0.4,
  0.2,
  1,
  0,
  0,
  0,
  -0.2,
  0.2,
  1,
  0.5,
  0.9,
  0.6,
  1,
  0.7,
  0.5,
  0.2,
  0.4,
  0.8,
];
const sensoryInspector = inspectorGraphs({
  diagnostic: {
    tick: 20,
    features: sensoryFeatures,
    action_probabilities: { rest: 0.5, move_forward: 0.5 },
  },
  template: sensoryTemplate,
  spec: { body: { vision_radius: 4 } },
  hasWater: false,
});
const sensoryPlots = walk(sensoryInspector).filter((n) =>
  ['movement-history', 'scent-grid'].includes(n.attributes?.class),
);
assert.equal(sensoryPlots.length, 2, 'Sensory inspector specimens missing');
let sensoryBody =
  box(28, 95, 500, 475) +
  text(46, 126, 'Personal visual memory', 21, '#e6edf3', 'start', true) +
  component(sensoryPlots[0], 45, 154, 460, 345) +
  lines(
    46,
    520,
    [
      'Solid dots: observed positions only.',
      'Hollow marker: fading last-seen memory.',
    ],
    15,
  );
sensoryBody +=
  box(552, 95, 606, 475) +
  text(575, 126, 'Blended scent · all animals', 21, '#e6edf3', 'start', true) +
  component(sensoryPlots[1], 690, 155, 300, 330);
sensoryBody += lines(
  40,
  610,
  [
    'One nine-sector field: blended strength, no species, sex or individual identity.',
    'Ground scent cannot cross a sealed wall or wet cells. Gaps in sight remain unknown.',
    'These panels update at recorded diagnostic samples, using existing replay input vectors.',
  ],
  16,
);
const sampledBoard = board({
  width: 10,
  height: 8,
  selected: 0,
  animals: [
    animal('prey', 'female', 0, 0, 4, 3),
    animal('prey', 'male', 0, 1, 7, 3),
    animal('predator', 'female', 2, 2, 0, 7),
  ],
  tracking: {
    origin: { x: 4, y: 3 },
    radius: 5,
    tick: 20,
    ...trackingView(sensoryFeatures, sensoryTemplate, 20),
  },
});
sensoryBody +=
  box(28, 700, 560, 430) +
  text(
    46,
    730,
    'Board: selected observer’s visual tracks',
    19,
    '#e6edf3',
    'start',
    true,
  ) +
  `<g transform="translate(92,752) scale(.76)">${sampledBoard}</g>` +
  lines(46, 1085, ['Recorded sightings only; no world scent overlay.'], 14);
const heatCounts = Array.from({ length: 80 }, (_, i) =>
  i % 10 === 5 ? 12 : i % 10 === 4 ? 5 : i % 7 === 0 ? 1 : 0,
);
let heatBoards = '';
for (const theme of ['dark', 'light']) {
  for (const [i, name] of ['hare', 'field mouse', 'fox'].entries()) {
    const heatCanvas = new SVGCanvas(10, 8);
    drawMovementHeatmap(
      heatCanvas,
      {
        width: 10,
        height: 8,
        counts: heatCounts.map((count) => Math.round((count * (i + 1)) / 3)),
        maximum: 12,
      },
      theme,
      [
        [0, 0],
        [1, 0],
      ],
      12,
    );
    const x = 640 + i * 170;
    heatBoards += `<g class="movement-heatmap-${theme}" transform="translate(${x},810) scale(15)">${heatCanvas.output.join('')}</g>`;
    if (theme === 'dark')
      heatBoards += text(x + 75, 790, name, 14, '#e6edf3', 'middle');
  }
}

sensoryBody +=
  box(612, 700, 560, 430) +
  text(
    630,
    730,
    'Metrics dashboard: movement beside species metrics',
    19,
    '#e6edf3',
    'start',
    true,
  ) +
  heatBoards +
  lines(
    630,
    1085,
    ['Shared count scale; same replay step; select one to enlarge.'],
    14,
  ) +
  `<style>.movement-heatmap-light{display:none}@media(prefers-color-scheme:light){.movement-heatmap-light{display:inline}.movement-heatmap-dark{display:none}}</style>`;
const trackingSpecimen = sheet(
  'Inspector: movement memory and ground scent',
  'Current inspector renderers · controlled illustrative observations, not an experiment outcome',
  1160,
  sensoryBody,
);

const demo = board(terrainCards[4][2]);
body =
  box(28, 95, 1144, 74) +
  text(
    46,
    122,
    '1 · Recording, run and playback controls',
    18,
    '#e6edf3',
    'start',
    true,
  ) +
  text(
    46,
    150,
    'Saved experiment ▾   Run ▾    Metrics dashboard    Back   Next frame   Play recording',
    16,
    '#a9bacc',
  );
body +=
  box(28, 185, 740, 61) +
  text(
    46,
    221,
    '2 · Step / run / populations / food / kills / births / updates',
    17,
  );
body +=
  box(28, 262, 740, 345) +
  text(
    46,
    292,
    '3 · World board and species legend',
    18,
    '#e6edf3',
    'start',
    true,
  ) +
  `<g transform="translate(268 311) scale(1.05)">${demo}</g>` +
  text(
    46,
    586,
    '● prey ♀ / ♂     ▲ predator ♀ / ♂     ♧ resource    ◆ obstacle',
    15,
    '#a9bacc',
  );
body +=
  box(790, 185, 382, 422) +
  text(808, 217, '4 · Agent inspector', 18, '#e6edf3', 'start', true) +
  lines(
    808,
    251,
    [
      'Selected animal / sex / parents',
      'Current sight toggle + Animal ID',
      'Whole structure when rank ≤ k: 2',
      'Watch animal brain',
      'Explain saved brain',
      'Sampled observation radar',
      'Input meanings + prediction bars',
      'Rank-limited structure / Advanced',
    ],
    16,
  ) +
  component(radar, 845, 415, 255, 170);
body +=
  box(28, 623, 740, 130) +
  text(46, 653, '5 · Speed and replay timeline', 18, '#e6edf3', 'start', true) +
  `<path d="M80 710H720" fill="none" stroke="#a9bacc" stroke-width="3"/><circle cx="360" cy="710" r="7" fill="#7de2bd"/>` +
  text(
    46,
    777,
    '6 · Metrics dashboard: population / food + species charts.',
    16,
    '#a9bacc',
  );
body +=
  box(790, 623, 382, 202) +
  text(808, 654, '7 · Supporting panels', 18, '#e6edf3', 'start', true) +
  lines(
    808,
    688,
    [
      'Recorded outcomes: counts',
      'Behavior & learning: whole run',
      'Recorded environment: JSON',
      'Create another recording: commands',
    ],
    15,
  );
const workspace = sheet(
  'Viewer workspace: where each visual belongs',
  'Schematic layout map with real board/radar glyphs · illustrative chart line; not a literal screenshot',
  850,
  body,
);
const demoCycle = { enabled: true, cycle_steps: 400, start_phase: 0 };
const dayCards = [
  ['Dawn', 110],
  ['Midday', 200],
  ['Dusk', 290],
  ['Night', 340],
];
body = dayCards
  .map(([name, tick], i) => {
    const x = 28 + i * 288;
    const scene = ['dark', 'light']
      .map(
        (theme) =>
          `<g class="board-palette-${theme}">${board({
            theme,
            day_night: demoCycle,
            tick,
            animals: [animal(), animal('predator', 'male', 1, 1, 2, 1)],
            resources: [{ x: 1, y: 2, quantity: 20 }],
            obstacles: [[0, 0]],
          })}</g>`,
      )
      .join('');
    return tile(
      x,
      95,
      name,
      [`Sunlight ${Math.round(daylight(demoCycle, tick).light * 100)}%`],
      scene,
    );
  })
  .join('');
const curve = Array.from(
  { length: 401 },
  (_, tick) =>
    `${tick ? 'L' : 'M'}${80 + tick * 2.6} ${535 - daylight(demoCycle, tick).light * 150}`,
).join(' ');
body +=
  box(28, 345, 1144, 240) +
  text(50, 377, 'Sunlight over one cycle', 19, '#e6edf3', 'start', true) +
  `<path d="${curve}" fill="none" stroke="#f6d365" stroke-width="3"/>` +
  text(80, 562, 'Midnight', 14) +
  text(340, 562, 'Sunrise', 14) +
  text(600, 562, 'Midday', 14) +
  text(860, 562, 'Sunset', 14) +
  text(1070, 562, 'Midnight', 14);
body += text(
  50,
  623,
  'Frame-synchronized sun/moon indicator',
  19,
  '#e6edf3',
  'start',
  true,
);
body += dayCards
  .map(
    ([name, tick], i) =>
      `<g class="clock-specimen" transform="translate(${80 + i * 288} 643)">${clockMarkup(clockState(demoCycle, tick))}</g>` +
      text(80 + i * 288, 738, name, 14),
  )
  .join('');
const dayNight = sheet(
  'Day–night lighting in playback',
  'Same objects at different saved ticks · actual lighting renderer · controlled 400-step cycle',
  770,
  body,
);
const themeCSS = fs.readFileSync(path.join(root, 'viewer/style.css'), 'utf8');
const themeTokens = (selector) =>
  Object.fromEntries(
    [
      ...themeCSS
        .slice(themeCSS.indexOf(selector))
        .split('}')[0]
        .matchAll(/--([a-z-]+):\s*([^;]+);/g),
    ].map((m) => [m[1], m[2].trim()]),
  );
function interfaceComparison(currentTheme) {
  const order = [currentTheme, currentTheme === 'dark' ? 'light' : 'dark'];
  let comparison = order
    .map((theme, i) => {
      const c = themeTokens(
          theme === 'dark' ? ':root {' : ":root[data-theme='light']",
        ),
        x = 28 + i * 582;
      return (
        '<g class="fixed-comparison">' +
        `<rect x="${x}" y="95" width="560" height="380" rx="12" fill="${c.bg}" stroke="${c.border}"/>` +
        text(
          x + 20,
          128,
          `${theme === 'dark' ? 'Dark' : 'Light'} interface · ${i === 0 ? 'selected' : 'alternate'}`,
          20,
          c.text,
          'start',
          true,
        ) +
        `<rect x="${x + 20}" y="149" width="520" height="140" rx="8" fill="${c.panel}" stroke="${c.border}"/>` +
        text(x + 38, 180, 'Agent inspector', 18, c.text, 'start', true) +
        text(x + 38, 213, 'Observed inputs · action predictions', 15, c.muted) +
        `<rect x="${x + 38}" y="235" width="235" height="34" rx="5" fill="${c.surface}" stroke="${c.border}"/>` +
        text(x + 48, 257, 'Animal ID', 14, c['code-text']) +
        text(
          x + 20,
          324,
          'Saved locally · applies to all recordings',
          15,
          c.muted,
        ) +
        text(
          x + 20,
          354,
          'Independent of world sunlight and time',
          15,
          c.muted,
        ) +
        `<rect x="${x + 20}" y="380" width="176" height="42" rx="6" fill="${c.panel}" stroke="${c.border}"/>` +
        text(
          x + 34,
          406,
          theme === 'dark' ? 'Light mode' : 'Dark mode',
          16,
          c.text,
        ) +
        '</g>'
      );
    })
    .join('');
  comparison += order
    .flatMap((theme, i) =>
      [50, 0].map((tick, j) => {
        const x = 28 + (i * 2 + j) * 288;
        return tile(
          x,
          510,
          `${theme} · ${tick === 50 ? 'midday' : 'night'}`,
          ['Theme palette + saved sunlight'],
          board({
            theme,
            day_night: { enabled: true, cycle_steps: 100, start_phase: 0 },
            tick,
            animals: [animal(), animal('predator', 'male', 1, 1, 2, 1)],
            resources: [{ x: 1, y: 2, quantity: 20 }],
            obstacles: [[0, 0]],
          }),
        );
      }),
    )
    .join('');
  return `<g class="comparison-current-${currentTheme}">${comparison}</g>`;
}
body = interfaceComparison('dark') + interfaceComparison('light');
const themes = sheet(
  'Viewer light and dark themes',
  'Schematic interface specimens using current CSS tokens · world lighting remains tied to the recording',
  755,
  body,
);
const environmentScene = {
  resources: [
    { x: 0, y: 2, quantity: 2 },
    { x: 1, y: 2, quantity: 10 },
    { x: 2, y: 2, quantity: 20 },
  ],
  obstacles: [
    [0, 0],
    [1, 0],
    [2, 0],
  ],
};
const environmentFrames = sheet(
  'Resources and obstacles during playback',
  'Actual renderer · saved-tick phases · fixed rock footprints · reduced motion freezes decorative movement',
  630,
  [0, 14, 28]
    .map((tick, i) =>
      tile(
        28 + i * 384,
        96,
        `Replay step ${tick}`,
        ['Quantity stays 2 / 10 / 20'],
        board({ ...environmentScene, tick }),
        360,
        240,
      ),
    )
    .join('') +
    [
      [
        'Reduced motion · step 0',
        { ...environmentScene, tick: 0, reduced: true },
      ],
      [
        'Reduced motion · step 28',
        { ...environmentScene, tick: 28, reduced: true },
      ],
      [
        'Depleted patches',
        {
          ...environmentScene,
          resources: environmentScene.resources.map((r) => ({
            ...r,
            quantity: 0,
          })),
          tick: 28,
        },
      ],
    ]
      .map(([title, scene], i) =>
        tile(
          28 + i * 384,
          360,
          title,
          ['Same cells; no invented growth'],
          board(scene),
          360,
          240,
        ),
      )
      .join(''),
);
const riverScene = {
  definition: {
    width: 3,
    rows: ['~~~', '~~~', '~~~'],
    legend: { '~': { water_depth: 1, flow: [1, 0] } },
  },
  animals: [
    {
      id: 7,
      species: 'prey',
      sex: 'female',
      x: 1,
      y: 1,
      heading: 0,
      health: 80,
    },
  ],
};
const riverFeeding = sheet(
  'River flow and feeding cues',
  'Actual renderer · rightward current · small green food-gain dot beside the eater · no plant position inferred',
  370,
  [0, 1, 3]
    .map((age, i) =>
      tile(
        28 + i * 384,
        96,
        `Replay step ${age}`,
        ['Flow follows saved map vectors; cue fades'],
        board({
          ...riverScene,
          tick: age,
          feedings: [{ actor: 7, position: [1, 1], age }],
        }),
        360,
        240,
      ),
    )
    .join(''),
);
const shadowCycle = { enabled: true, cycle_steps: 100, start_phase: 0 };
const shadowScene = {
  ...environmentScene,
  obstacles: [[1, 0]],
  animals: [{ id: 8, species: 'prey', sex: 'female', x: 1, y: 1, heading: 0 }],
  day_night: shadowCycle,
};
const shadows = sheet(
  'Sunlight and object shadows',
  'Actual renderer · long opposite-sun shadows morning/evening · short at midday · absent at night',
  630,
  [30, 50, 70]
    .map((tick, i) =>
      tile(
        28 + i * 384,
        96,
        [
          'Dawn · shadows right',
          'Midday · short shadows',
          'Dusk · shadows left',
        ][i],
        ['Saved sunlight controls direction and opacity'],
        board({ ...shadowScene, tick }),
        360,
        240,
      ),
    )
    .join('') +
    [
      [0, false, 'Night · no sun shadows'],
      [30, true, 'Reduced motion · dawn'],
      [70, true, 'Reduced motion · dusk'],
    ]
      .map(([tick, reduced, title], i) =>
        tile(
          28 + i * 384,
          360,
          title,
          ['Reduced motion fixes footprint; sunlight still fades'],
          board({ ...shadowScene, tick, reduced }),
          360,
          240,
        ),
      )
      .join(''),
);
// Controlled saved-input specimens; these values are not experimental results.
const collectiveTemplate = {
  consequence_scope: 'both',
  observation_version: 4,
  observation_top_k: 3,
  social_observation: { history_steps: 4, max_neighbors: 8 },
  consequence_observation: true,
};
const collectiveFeatures = [...Array(69).fill(0), ...Array(120).fill(0)];
collectiveFeatures[0] = 0.6;
collectiveFeatures[1] = 1;
collectiveFeatures.splice(69 + 4, 4, 0.75, 0.8, 0, 0);
collectiveFeatures.splice(69 + 8, 4, 0.8, 0, 0.9, 0);
collectiveFeatures.splice(69 + 16, 4, 0.5, 0, 0, 0.5);
collectiveFeatures.splice(69 + 24 + 9 * 4, 4, 0.9, 0.7, 0, 0);
collectiveFeatures.splice(69 + 24 + 7 * 4, 4, 0.8, 0, 0.9, 0);
collectiveFeatures.splice(69 + 24 + 48 + 9 * 4, 4, 0.7, 0.9, 0, 0);
collectiveFeatures.splice(69 + 24 + 48 + 7 * 4, 4, 0.6, 0, 0.5, 0);
const collectiveCharts = (features, tick, collectiveAverages) => {
  const tree = inspectorGraphs({
    collectiveAverages,
    diagnostic: {
      tick,
      features,
      action_probabilities: { rest: 0.4, eat: 0.4, swim: 0.2 },
    },
    template: collectiveTemplate,
    spec: {
      body: { vision_radius: 6 },
      allowed_actions: ['move_forward', 'rest', 'eat', 'swim', 'reproduce'],
    },
    hasWater: true,
  });
  return walk(tree).filter(
    (n) =>
      n.tagName === 'svg' &&
      [
        'Local witnesses',
        'General consequence estimates',
        'Context consequence estimates',
      ].includes(n.attributes['aria-label']),
  );
};
const illustratedTotals = new CollectiveTotals({
  brains: { test: collectiveTemplate },
  species: { prey: { brain: 'test' } },
});
const speciesFeatures = collectiveFeatures.map((v, i) =>
  i >= 69 && (i < 93 || i >= 141) && (i - 69) % 4 === 1 ? v / 2 : v,
);
const illustratedFrame = (tick) => ({
  tick,
  animals: [
    { id: 0, species: 'prey' },
    { id: 1, species: 'prey' },
  ],
  diagnostics: {
    0: { tick, features: collectiveFeatures },
    1: { tick, features: speciesFeatures },
  },
});
illustratedTotals.consume(illustratedFrame(0));
illustratedTotals.consume(illustratedFrame(40));
const illustratedAverage = illustratedTotals.view(0);
const animalCollective = collectiveCharts(
    collectiveFeatures,
    40,
    illustratedAverage,
  ),
  speciesCollective = collectiveCharts(collectiveFeatures, 40, {
    tick: 40,
    animal: illustratedAverage.by_species.prey,
  });
assert.equal(animalCollective.length, 3);
assert.equal(speciesCollective.length, 3);
const socialLearning = sheet(
  'Cumulative collective outcomes',
  'Combined mode · actual charts · illustrative inputs · animal inspector above · species metrics below',
  640,
  [animalCollective, speciesCollective]
    .map((charts, row) =>
      charts
        .map(
          (chart, col) =>
            box(28 + col * 384, 96 + row * 264, 360, 242) +
            text(
              46 + col * 384,
              124 + row * 264,
              `${row ? 'Species' : 'Inspector'} · ${['witnesses', 'general', 'context'][col]}`,
              18,
              '#e6edf3',
              'start',
              true,
            ) +
            component(chart, 46 + col * 384, 142 + row * 264, 324, 176),
        )
        .join(''),
    )
    .join(''),
);
const forwardBoards = ['North', 'East', 'South', 'West']
  .map((name, heading) => {
    const x = 28 + heading * 288;
    return (
      box(x, 95, 268, 320) +
      text(x + 16, 124, `Facing ${name}`, 18) +
      `<g transform="translate(${x + 24} 144) scale(.65)">${board({ width: 7, height: 5, fieldOfView: 180, selected: 0, sight: true, animals: [animal('prey', 'female', heading, 0, 3, 2), animal('predator', 'male', 3, 1, 5, 2), animal('prey', 'male', 1, 2, 1, 2)], resources: [{ x: 4, y: 1, quantity: 20 }] })}</g>` +
      lines(
        x + 16,
        350,
        [
          'Blue cells = selected visual field.',
          'Overview still shows all bodies.',
        ],
        13,
      )
    );
  })
  .join('');
const forwardRadars = [360, 180, 150]
  .map((degrees, index) => {
    const vector = [
      0.6,
      0.85,
      1,
      0,
      0,
      0,
      0,
      0,
      ...(degrees === 360 ? [-0.5, 0, 0.5, 1] : [0, 0, 0, 0]),
      0.25,
      0,
      0.25,
      1,
      ...(degrees === 360
        ? [-0.5, -0.25, Math.hypot(0.5, 0.25), 1]
        : [0, 0, 0, 0]),
    ];
    const tree = inspectorGraphs({
      diagnostic: {
        tick: 10,
        features: vector,
        action_probabilities: { rest: 1 },
      },
      template: { observation_version: 1 },
      spec: { body: { vision_radius: 4, field_of_view_degrees: degrees } },
      hasWater: false,
    });
    const plot = walk(tree).find((node) => node.tagName === 'svg');
    const x = 28 + index * 390;
    return (
      box(x, 435, 368, 310) +
      text(x + 16, 465, `${degrees}° radar · facing east`, 17) +
      component(plot, x + 16, 488, 336, 240)
    );
  })
  .join('');
const forwardVision = sheet(
  'Forward visual fields',
  'Current board and inspector renderers · controlled states · scent and remembered sightings remain separate',
  770,
  forwardBoards +
    forwardRadars +
    `<style>@media(prefers-color-scheme:light){[fill='#8ac8ff22']{fill:#2d78b82e}}</style>`,
);
const v5Example = Array(48).fill(0);
v5Example.splice(0, 4, 0.15, 1, 1, 0);
v5Example.splice(12, 4, 1 / 11, -1 / 11, Math.SQRT2 / 11, 1);
v5Example.splice(45, 3, 0, 0, 0);
const v5Tree = inspectorGraphs({
  diagnostic: {
    tick: 10,
    features: v5Example,
    action_probabilities: {
      move_forward: 0.2,
      turn_left: 0.4,
      turn_right: 0.1,
      rest: 0.05,
      attack: 0.15,
      reproduce: 0.05,
      swim: 0.05,
    },
  },
  template: { observation_version: 5 },
  spec: {
    body: {
      vision_radius: 8,
      water_vision_bonus: 3,
      field_of_view_degrees: 150,
    },
  },
  hasWater: true,
});
const v5Radar = walk(v5Tree).find((node) => node.tagName === 'svg');
const v5Structure = sheet(
  'Observation V5: body-relative inputs, world-relative diagrams',
  'Controlled input/renderer demonstration; example probabilities are authored, not experimental results',
  660,
  box(28, 95, 545, 225) +
    text(48, 125, 'V4 · world-axis target vector', 22) +
    lines(
      48,
      164,
      [
        'Heading: east',
        'Prey: dx = 1/11, dy = -1/11',
        'The network combines heading and position.',
        'Base width: 45 + 12 × (k − 1)',
      ],
      17,
    ) +
    box(28, 344, 545, 285) +
    text(48, 376, 'V5 · forward/right target vector', 22) +
    lines(
      48,
      417,
      [
        'Same prey: forward = 1/11, right = -1/11',
        'Negative right means left of the animal.',
        'Attack / eat / swim opportunities: 0 / 0 / 0',
        'Base width: 48 + 12 × (k − 1)',
        'Optional memory and scent rotate consistently.',
        'Brains remain independent; PPO is unmasked.',
      ],
      17,
    ) +
    box(601, 95, 571, 534) +
    text(625, 132, 'Agent inspector · facing east', 22) +
    component(v5Radar, 650, 170, 470, 326) +
    lines(
      625,
      550,
      [
        'Raw features use forward/right labels.',
        'Radar, history and scent diagrams face north.',
        'The selected board overlay keeps world geometry.',
      ],
      16,
    ),
);
const v6Template = { observation_version: 6 };
const v6Layout = semanticLayout(v6Template),
  v6Values = Array(v6Layout.width).fill(0);
for (const field of v6Layout.fields) {
  const values =
    field.label === 'Energy'
      ? [0, 0.7, 0.3, 0]
      : field.label === 'Health'
        ? [0, 0.2, 0.8]
        : field.label === 'Heading x'
          ? [1]
          : field.label === 'Prey 1'
            ? [0.7, 0.3, 0, 0.2, 0, 0, 0, 0, 0, 0, 0.8]
            : field.label === 'Prey 1 present'
              ? [1]
              : null;
  if (values) values.forEach((v, i) => (v6Values[field.offset + i] = v));
}
const v6Tree = inspectorGraphs({
  diagnostic: {
    tick: 10,
    features: v6Values,
    action_probabilities: { rest: 0.5, attack: 0.5 },
  },
  template: v6Template,
  spec: {
    body: {
      vision_radius: 8,
      water_vision_bonus: 3,
      field_of_view_degrees: 150,
    },
  },
  hasWater: true,
});
const v6Radar = walk(v6Tree).find((node) => node.tagName === 'svg');
const softRows = [
  ['Energy', ['critical', 'hungry', 'comfortable', 'full'], [0, 0.7, 0.3, 0]],
  ['Prey 1 range', ['near', 'medium', 'far'], [0.7, 0.3, 0]],
  ['Prey 1 bearing', ['front', 'front-left'], [0.2, 0.8]],
  ['Health', ['critical', 'weak', 'healthy'], [0, 0.2, 0.8]],
];
const v6Diagram = sheet(
  'Observation V6: overlapping semantic memberships',
  'Authored category example using the current V6 decoder and inspector; no experimental predictions',
  660,
  box(28, 95, 545, 534) +
    text(48, 130, 'What the independent brain receives', 22) +
    softRows
      .map(([name, labels, values], i) => {
        const y = 174 + i * 95;
        return (
          text(48, y, name, 18) +
          labels
            .map((label, j) => {
              const x = 48 + j * 125;
              return (
                `<rect x="${x}" y="${y + 12}" width="110" height="16" rx="3" fill="#263341"/><rect x="${x}" y="${y + 12}" width="${values[j] * 110}" height="16" rx="3" fill="#7de2bd"/>` +
                text(x, y + 48, `${label} ${Math.round(values[j] * 100)}%`, 12)
              );
            })
            .join('')
        );
      })
      .join('') +
    lines(
      48,
      585,
      [
        'Opportunity and consent flags remain exact.',
        'Unknown evidence stays distinct from low outcome rates.',
      ],
      14,
    ) +
    box(601, 95, 571, 534) +
    text(625, 132, 'Inspector: category-derived position', 22) +
    component(v6Radar, 650, 170, 470, 326) +
    lines(
      625,
      550,
      [
        'Soft ranges and bearings reconstruct the radar.',
        'Saturated values show the edge of their category range.',
        'The board still uses exact saved world positions.',
      ],
      16,
    ),
);
const outputs = {
  'observation-v7.svg': compactPolicyVisual({
    sheet,
    box,
    text,
    lines,
    component,
    walk,
    inspectorGraphs,
    compactLayout,
    assert,
    config: JSON.parse(
      fs.readFileSync(
        path.join(root, 'environments/seaside_150_v7.json'),
        'utf8',
      ),
    ),
  }),
  'observation-v5-v6.svg': observationComparison({
    sheet,
    box,
    text,
    lines,
    semanticLayout,
    rawInputLabels,
    decodeSoft,
    assert,
    config: JSON.parse(
      fs.readFileSync(
        path.join(root, 'environments/seaside_150_v6.json'),
        'utf8',
      ),
    ),
  }),
  'observation-v6-layout.svg': observationLayout({
    sheet,
    box,
    text,
    lines,
    semanticLayout,
    assert,
    config: JSON.parse(
      fs.readFileSync(
        path.join(root, 'environments/seaside_150_v6.json'),
        'utf8',
      ),
    ),
  }),
  'observation-v6.svg': v6Diagram,
  'observation-v5.svg': v5Structure,
  'forward-vision.svg': forwardVision,
  ...buildDocAnimations({
    board,
    animal,
    sheet,
    box,
    text,
    lines,
    geography,
    land,
    inspectorGraphs,
    walk,
    component,
    diagnostic,
    template,
    networkStructure,
    animalStructure,
    explanationCharts,
    report: r,
  }),
  'social-learning.svg': socialLearning,
  'object-shadows.svg': shadows,
  'river-feeding.svg': riverFeeding,
  'environment-objects.svg': environmentFrames,
  'tracking.svg': trackingSpecimen,
  'themes.svg': themes,
  'day-night.svg': dayNight,
  'board-objects.svg': objects,
  'terrain-and-sight.svg': terrains,
  'brain-views.svg': brains,
  'workspace-map.svg': workspace,
};
const check = process.argv.includes('--check');
if (!check) fs.mkdirSync(outdir, { recursive: true });
for (const [name, original] of Object.entries(outputs)) {
  // Terrain colors are continuous HSL values, so map each specimen's actual
  // dark fill using the renderer's light palette instead of a finite hex list.
  const terrainRules = new Set();
  for (const match of original.matchAll(
    /fill="(hsl\(([-\d.]+), (\d+)%, ([-\d.]+)%\))"/g,
  )) {
    const [, color, hue, saturation, luminance] = match;
    const value = Number(luminance);
    const lightness =
      Number(saturation) === 65 && Number(hue) === 207
        ? 87 - value
        : Number(saturation) === 25
          ? 84 - (20 * (value - 19)) / 14
          : null;
    if (lightness !== null)
      terrainRules.add(
        `[fill='${color}'] { fill: hsl(${hue}, ${saturation}%, ${lightness}%); }`,
      );
  }
  const terrainCSS = `@media (prefers-color-scheme: light) {${[...terrainRules].join('\n')}}`;

  const contents = ['themes.svg', 'day-night.svg'].includes(name)
    ? original.replace(
        /<\/svg>\s*$/,
        `<style>${fs.readFileSync(path.join(outdir, 'comparison-palette.css'), 'utf8')}</style></svg>\n`,
      )
    : original.replace(
        /<\/svg>\s*$/,
        `<style>${fs.readFileSync(path.join(outdir, 'theme-palette.css'), 'utf8')}${terrainCSS}</style></svg>\n`,
      );
  const target = path.join(outdir, name);
  if (check)
    assert.equal(fs.readFileSync(target, 'utf8'), contents, `Outdated ${name}`);
  else fs.writeFileSync(target, contents);
}
console.log(
  `${check ? 'Verified' : 'Built'} ${Object.keys(outputs).length} controlled viewer references from current renderers.`,
);
