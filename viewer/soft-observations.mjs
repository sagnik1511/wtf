// V6 display decoding mirrors the Python semantic layout; parity is regression-tested.
export const DEFAULT_CATEGORIES = {
  distance: [0.0, 0.3, 1.0],
  energy: [0.0, 0.25, 0.75, 1.0],
  health: [0.0, 0.5, 1.0],
  level: [0.0, 0.5, 1.0],
  depth: [0.0, 0.1, 1.0],
  elevation: [0.0, 0.1, 1.0],
  climb: [-1.0, -0.1, 0.0, 0.1, 1.0],
  speed: [0.0, 0.25, 1.4142135623730951],
  outcome: [0.0, 0.001, 0.01, 0.1, 1.0],
};
for (const values of Object.values(DEFAULT_CATEGORIES)) Object.freeze(values);
Object.freeze(DEFAULT_CATEGORIES);
const CATEGORY_LABELS = {
  distance: ['near', 'medium', 'far'],
  energy: ['critical', 'hungry', 'comfortable', 'full'],
  health: ['critical', 'weak', 'healthy'],
  level: ['none', 'partial', 'strong'],
  depth: ['dry', 'shallow', 'deep'],
  elevation: ['low', 'middle', 'high'],
  climb: ['steep-down', 'gentle-down', 'level', 'gentle-up', 'steep-up'],
  speed: ['still', 'gentle', 'strong'],
  outcome: ['zero', 'very-low', 'low', 'moderate', 'high'],
};
const BEARINGS = [
  'front',
  'front-right',
  'right',
  'back-right',
  'back',
  'back-left',
  'left',
  'front-left',
];

function scalarKind(name) {
  if (name === 'Energy') return 'energy';
  if (name === 'Health') return 'health';
  if (
    ['Maturity', 'Cooldown', 'Self consent', 'Mate consent'].includes(name) ||
    name.endsWith(' age') ||
    name.endsWith(' confidence') ||
    name.startsWith('Scent ')
  )
    return 'level';
  if (name === 'Water depth' || name.endsWith(' water')) return 'depth';
  if (name === 'Elevation') return 'elevation';
  if (name.endsWith(' climb')) return 'climb';
  if (
    ['Collective ', 'Species action ', 'Consequence '].some((p) =>
      name.startsWith(p),
    ) &&
    [' harm', ' feeding', ' birth'].some((p) => name.endsWith(p))
  )
    return 'outcome';
  return 'raw';
}
const layouts = new Map();
export function buildSoftLayout(raw, template) {
  const knots = { ...DEFAULT_CATEGORIES, ...template.soft_categories };
  if (Object.keys(knots).some((k) => !Object.hasOwn(DEFAULT_CATEGORIES, k)))
    throw new Error('Unknown soft category');
  for (const [key, values] of Object.entries(knots)) {
    const base = DEFAULT_CATEGORIES[key];
    if (
      !Array.isArray(values) ||
      values.length !== base.length ||
      !values.every(Number.isFinite) ||
      values.some((v, i) => i && v <= values[i - 1]) ||
      values[0] !== base[0] ||
      values.at(-1) !== base.at(-1)
    )
      throw new Error('Invalid soft category knots');
    knots[key] = Object.freeze([...values]);
  }
  const key = JSON.stringify([raw, knots]);
  if (layouts.has(key)) return layouts.get(key);
  const fields = [],
    names = [];
  for (let i = 0; i < raw.length; ) {
    const name = raw[i];
    let count = 1,
      kind = scalarKind(name),
      label = name,
      centers = [],
      labels;
    if (name === 'Flow forward' || name.endsWith('velocity forward')) {
      count = 2;
      kind = 'vector';
      label = name.slice(0, name.lastIndexOf(' '));
    } else if (
      name.endsWith(' forward') &&
      ['Food ', 'Prey ', 'Predator ', 'Mate '].some((p) => name.startsWith(p))
    ) {
      count = raw[i + 2]?.endsWith(' distance') ? 3 : 2;
      kind = 'position';
      label = name.slice(0, name.lastIndexOf(' '));
    }
    if (kind === 'position' || kind === 'vector') {
      const group = kind === 'position' ? 'distance' : 'speed';
      centers = knots[group];
      labels = [
        ...CATEGORY_LABELS[group].map(
          (n) => `${label} ${kind === 'position' ? 'range' : 'speed'} ${n}`,
        ),
        ...BEARINGS.map((n) => `${label} direction ${n}`),
      ];
    } else if (kind === 'raw') labels = [name];
    else {
      centers = knots[kind];
      labels = CATEGORY_LABELS[kind].map((n) => `${name} ${n}`);
    }
    fields.push(
      Object.freeze({
        raw: i,
        count,
        offset: names.length,
        label,
        kind,
        knots: centers,
        names: Object.freeze(labels),
      }),
    );
    names.push(...labels);
    i += count;
  }
  const result = Object.freeze({
    fields: Object.freeze(fields),
    names: Object.freeze(names),
    width: names.length,
    rawNames: Object.freeze([...raw]),
  });
  layouts.set(key, result);
  if (layouts.size > 128) layouts.delete(layouts.keys().next().value);
  return result;
}
export function decodeSoft(values, layout) {
  const output = [];
  for (const f of layout.fields) {
    const block = values.slice(f.offset, f.offset + f.names.length);
    if (f.kind === 'raw') {
      output.push(block[0]);
      continue;
    }
    const distance = f.knots.reduce((sum, x, i) => sum + x * block[i], 0);
    if (f.kind === 'position' || f.kind === 'vector') {
      const weights = block.slice(f.knots.length);
      let angle = 0;
      if (weights.some(Boolean)) {
        const anchor = weights.indexOf(Math.max(...weights));
        angle =
          (weights.reduce(
            (sum, w, i) => sum + (anchor + (((i - anchor + 12) % 8) - 4)) * w,
            0,
          ) *
            Math.PI) /
          4;
      }
      output.push(distance * Math.cos(angle), distance * Math.sin(angle));
      if (f.count === 3) output.push(distance);
    } else output.push(distance);
  }
  return output;
}
export function semanticPreview(values, layout, limit = 48) {
  const rows = [];
  for (const f of layout.fields) {
    if (f.kind === 'raw') continue;
    const members = f.names
      .map((name, i) => ({ name, value: values[f.offset + i] }))
      .filter((r) => r.value > 0);
    if (members.length) rows.push({ label: f.label, members });
    if (rows.length >= limit) break;
  }
  return rows;
}
