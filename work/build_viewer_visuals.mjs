#!/usr/bin/env node
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
  stroke() {
    this.output.push(
      `<path d="${this.d}" fill="none" stroke="${esc(this.strokeStyle)}" stroke-width="${this.lineWidth}" ${this.attrs()}/>`,
    );
  }
  fillText(value, x, y) {
    const size = this.font.match(/([\d.]+)px/)?.[1] ?? 14;
    this.output.push(
      `<text x="${num(x)}" y="${num(y)}" font-family="system-ui,sans-serif" font-size="${size}" font-weight="${this.font.includes('bold') ? '700' : '400'}" text-anchor="${this.textAlign === 'center' ? 'middle' : this.textAlign === 'right' ? 'end' : 'start'}" fill="${esc(this.fillStyle)}" ${this.attrs()}>${esc(value)}</text>`,
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
  kills = [],
  deaths = [],
  theme = 'dark',
  day_night = null,
  tick = 0,
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
      species: baseSpecies,
      resource_capacity: 20,
      birth_highlights: births,
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
    state,
    selected,
    terrain,
    drawTerrain,
    observationOverlay: new ObservationOverlay(),
    $: (id) => controls[id],
    fitWorld: () => {},
    animalColor: (a) =>
      styles.get(a.species)?.[a.sex] ?? styles.get(a.species)?.color,
  });
  vm.runInContext(
    ['drawWorld', 'drawBirthHighlights', 'drawDeathHighlights']
      .map(sourceFunction)
      .join('\n') +
      '\ndrawWorld();drawBirthHighlights();drawDeathHighlights();',
    context,
    { timeout: 1000 },
  );
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
    ['Small, faint square (0/20)'],
    { resources: [{ x: 1, y: 1, quantity: 0 }] },
  ],
  [
    'Resource · half full',
    ['Size and opacity grow (10/20)'],
    { resources: [{ x: 1, y: 1, quantity: 10 }] },
  ],
  [
    'Resource · full',
    ['Largest, strongest fill (20/20)'],
    { resources: [{ x: 1, y: 1, quantity: 20 }] },
  ],
  ['Obstacle', ['Slate square; blocks walking'], { obstacles: [[1, 1]] }],
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
];
let body = objectCards
  .map(([title, caption, options], i) =>
    tile(
      28 + (i % 4) * 288,
      95 + Math.floor(i / 4) * 236,
      title,
      caption,
      board(options),
    ),
  )
  .join('');
const objects = sheet(
  'Board objects and event markers',
  'Actual viewer drawing code · enlarged controlled specimens · event age 0 · canvas glow omitted',
  1060,
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
const features = [
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
const template = {
  type: 'ppo',
  observation_version: 1,
  observation_top_k: 1,
  network: { hidden_layers: [4, 3] },
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
      size: 20,
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
      '20 inputs → 4 → 3 → 2 → action scores',
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
    '● prey ♀ / ♂     ▲ predator ♀ / ♂     ■ resource    ■ obstacle',
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
  text(
    46,
    653,
    '5 · Speed, timeline and population / food history',
    18,
    '#e6edf3',
    'start',
    true,
  ) +
  `<path d="M50 713H745M62 732V681M68 709L190 698L310 704L420 677L530 687L728 671" fill="none" stroke="#7de2bd" stroke-width="2"/>` +
  text(
    46,
    777,
    '6 · Metrics dashboard opens step-following species charts.',
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
const outputs = {
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
  const contents = ['themes.svg', 'day-night.svg'].includes(name)
    ? original.replace(
        /<\/svg>\s*$/,
        `<style>${fs.readFileSync(path.join(outdir, 'comparison-palette.css'), 'utf8')}</style></svg>\n`,
      )
    : original.replace(
        /<\/svg>\s*$/,
        `<style>${fs.readFileSync(path.join(outdir, 'theme-palette.css'), 'utf8')}</style></svg>\n`,
      );
  const target = path.join(outdir, name);
  if (check)
    assert.equal(fs.readFileSync(target, 'utf8'), contents, `Outdated ${name}`);
  else fs.writeFileSync(target, contents);
}
console.log(
  `${check ? 'Verified' : 'Built'} ${Object.keys(outputs).length} controlled viewer references from current renderers.`,
);
