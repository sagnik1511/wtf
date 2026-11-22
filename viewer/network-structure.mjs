import { trackingLabels } from './tracking.mjs';
import { visualColor } from './theme.mjs';
// Display-only topology from existing observations and layer arrays. No weight requests.
import { observationView, predictionView } from './inspector-stats.mjs';
const NS = 'http://www.w3.org/2000/svg';
export const DEFAULT_NETWORK_RANK = 2;
export const rankLimit = (value) => {
  if (value == null || String(value).trim() === '') return DEFAULT_NETWORK_RANK;
  const n = Number(value);
  return Number.isInteger(n) && n >= 0 && n <= 12 ? n : DEFAULT_NETWORK_RANK;
};
const el = (tag, text, cls) => {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
};
const svg = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (text !== undefined) n.textContent = text;
  return n;
};
export function inputLabels(template = {}, length = 20) {
  template ??= {};
  const names = [
    'Energy',
    'Health',
    'Heading x',
    'Heading y',
    'Blocked front',
    'Blocked right',
    'Blocked back',
    'Blocked left',
  ];
  for (const category of ['Food', 'Prey', 'Predator'])
    for (let i = 1; i <= (template.observation_top_k ?? 1); i++)
      for (const part of ['dx', 'dy', 'distance', 'present'])
        names.push(`${category} ${i} ${part}`);
  if ((template.observation_version ?? 1) >= 2) {
    names.push('Water depth', 'Flow x', 'Flow y', 'Elevation');
    for (const side of ['Front', 'Right', 'Back', 'Left'])
      names.push(`${side} water`, `${side} climb`);
  }
  if ((template.observation_version ?? 1) >= 3)
    names.push(
      'Sex',
      'Maturity',
      'Cooldown',
      'Self ready',
      'Mate dx',
      'Mate dy',
      'Mate distance',
      'Mate present',
      'Mate ready',
      'Pair opportunity',
    );
  if ((template.observation_version ?? 1) >= 4)
    names.push('Self consent', 'Mate consent', 'Consenting pair');
  names.push(...trackingLabels(template));
  return Array.from({ length }, (_, i) => names[i] ?? `Input ${i}`);
}
export function inputInterpretation(diagnostic, template) {
  const root = el('section', undefined, 'brain-input-meaning'),
    o = observationView(diagnostic?.features, template);
  root.append(el('h4', 'What the inputs describe'));
  if (!o) {
    root.append(
      el(
        'p',
        'See the recorded input vector and observation radar for this layout.',
        'muted',
      ),
    );
    return root;
  }
  const rows = [
    ['Energy', `${(o.energy * 100).toFixed(0)}%`],
    ['Health', `${(o.health * 100).toFixed(0)}%`],
    ['Blocked walking directions', `${o.blocked.filter(Boolean).length}/4`],
  ];
  for (const category of ['food', 'prey', 'predator'])
    rows.push([
      `Visible ${category} slots`,
      `${o.targets.filter((t) => t.category === category).length}/${o.k}`,
    ]);
  if (o.terrain)
    rows.push(
      ['Local water depth', o.terrain.depth.toFixed(2)],
      ['Elevation', o.terrain.elevation.toFixed(2)],
    );
  if (o.mating)
    rows.push(
      ['Mating readiness', o.mating.ready ? 'ready' : 'not ready'],
      ['Visible mate', o.mating.present ? 'present' : 'absent'],
    );
  const grid = el('div', undefined, 'brain-input-grid');
  for (const [name, value] of rows) {
    const item = el('div');
    item.append(el('span', name), el('strong', value));
    grid.append(item);
  }
  root.append(grid);
  root.append(
    el(
      'p',
      'These are observed values, not causal importance scores. Outputs estimate action probabilities or returns; exploration can choose another action.',
      'muted',
    ),
  );
  return root;
}
export function animalStructure(
  diagnostic,
  template = {},
  layers = diagnostic?.activations?.layers,
) {
  template ??= {};
  if (
    !Array.isArray(diagnostic?.features) ||
    !diagnostic.features.every(Number.isFinite)
  )
    return null;
  if (
    diagnostic.activations?.version !== 1 ||
    !Array.isArray(layers) ||
    !layers.every((a) => Array.isArray(a) && a.every(Number.isFinite))
  )
    layers = undefined;
  const recurrent = (diagnostic.algorithm ?? template.type) === 'recurrent_ppo';
  const widths = layers?.map((a) => a.length) ?? [
    ...(recurrent ? [template.recurrent_size] : []),
    ...(template.type === 'q_network'
      ? (template.hidden_layers ?? [])
      : (template.network?.hidden_layers ?? [])),
  ];
  if (widths.some((w) => !Number.isInteger(w) || w < 1) || !widths.length)
    return null;
  const labels = inputLabels(template, diagnostic.features.length),
    view = predictionView(diagnostic);
  const output = diagnostic.return_quantiles
    ? Object.entries(diagnostic.return_quantiles).flatMap(([action, values]) =>
        values.map((value, i) => ({ label: `${action} q${i + 1}`, value })),
      )
    : (view?.actions ?? []).map((a) => ({ label: a.action, value: a.value }));
  if (!output.length) return null;
  if (
    !diagnostic.return_quantiles &&
    Number.isFinite(diagnostic.state_value) &&
    diagnostic.algorithm !== 'mappo'
  )
    output.push({ label: 'Value', value: diagnostic.state_value });
  return {
    rank: widths.length,
    stages: [
      { label: 'Observation', values: diagnostic.features, labels },
      ...widths.map((size, i) => ({
        label: recurrent && i === 0 ? 'GRU memory block' : `Hidden ${i + 1}`,
        values: layers?.[i] ?? Array(size).fill(null),
        recurrent: recurrent && i === 0,
      })),
      {
        label: diagnostic.return_quantiles
          ? 'Return quantiles'
          : view?.policy
            ? 'Policy scores'
            : 'Action Q',
        values: output.map((a) => a.value),
        labels: output.map((a) => a.label),
      },
    ],
    available: Boolean(layers),
  };
}
export function archivedStructure(report) {
  return {
    rank: Math.max(0, report.layers.length - 2),
    available: true,
    stages: report.layers.map((layer, i) => ({
      label: layer.label,
      values: i ? layer.values : report.features.map((f) => f.value),
      labels: i
        ? Array.from(
            { length: layer.size },
            (_, j) => layer.nodes.find((n) => n.index === j)?.label ?? `${j}`,
          )
        : report.features.map((f) => f.label),
      recurrent: layer.activation === 'gru',
    })),
  };
}
export function structurePlan(description, k = DEFAULT_NETWORK_RANK) {
  const full = description.rank <= rankLimit(k),
    stages = description.stages;
  return {
    full,
    rank: description.rank,
    k: rankLimit(k),
    nodes: stages.reduce((sum, s) => sum + s.values.length, 0),
    connections: stages
      .slice(1)
      .reduce(
        (sum, s, i) => sum + s.values.length * stages[i].values.length,
        0,
      ),
  };
}
export function networkStructure(description, k = DEFAULT_NETWORK_RANK) {
  const root = el('section', undefined, 'network-structure');
  if (!description) return root;
  const plan = structurePlan(description, k);
  root.append(
    el(
      'h3',
      plan.full
        ? `Whole structure · rank ${plan.rank} ≤ k ${plan.k}`
        : `Input → output summary · rank ${plan.rank} > k ${plan.k}`,
    ),
  );
  if (!plan.full) {
    root.append(
      el(
        'p',
        description.stages
          .map((s) => `${s.label} (${s.values.length})`)
          .join(' → '),
        'brain-flow',
      ),
    );
    root.append(
      el(
        'p',
        'Detailed hidden-unit views stay under Advanced. Increase k to show the whole structure.',
        'muted',
      ),
    );
    return root;
  }
  const rows = 32,
    step = 10,
    gap = 100,
    margin = 30,
    stages = description.stages,
    maxRows = Math.max(
      1,
      ...stages.map((s) => Math.min(rows, s.values.length)),
    );
  let cursor = margin;
  const positions = stages.map((s) => {
    const cols = Math.ceil(s.values.length / rows),
      width = Math.max(120, cols * step + 24),
      cx = cursor + width / 2,
      start = cx - (cols * step) / 2;
    cursor += width + gap;
    return {
      cx,
      width,
      points: s.values.map((value, i) => ({
        x: start + Math.floor(i / rows) * step,
        y:
          85 +
          ((maxRows - Math.min(rows, s.values.length)) * step) / 2 +
          (i % rows) * step,
        value,
        index: i,
      })),
    };
  });
  const width = Math.max(700, cursor - gap + margin),
    height = maxRows * step + 150,
    chart = svg('svg', {
      viewBox: `0 0 ${width} ${height}`,
      role: 'img',
      'aria-label': `Whole network structure: ${plan.nodes} units; rank ${plan.rank}; topology connections, not weight values`,
    });
  chart.append(
    svg(
      'title',
      {},
      `All ${plan.nodes} units; ${plan.connections} dense inter-stage connections. Internal GRU gates are collapsed into a memory block.`,
    ),
  );
  const bundled = plan.connections > 1024;
  for (let i = 0; i < positions.length - 1; i++) {
    const a = positions[i].points,
      b = positions[i + 1].points;
    if (!a.length || !b.length) continue;
    if (bundled) {
      const right = Math.max(...a.map((p) => p.x)) + 4,
        left = Math.min(...b.map((p) => p.x)) - 1,
        lowA = Math.min(...a.map((p) => p.y)),
        highA = Math.max(...a.map((p) => p.y)) + 7,
        lowB = Math.min(...b.map((p) => p.y)),
        highB = Math.max(...b.map((p) => p.y)) + 7;
      chart.append(
        svg('polygon', {
          points: `${right},${lowA} ${left},${lowB} ${left},${highB} ${right},${highA}`,
          fill: visualColor('#8ac8ff'),
          opacity: 0.08,
          stroke: visualColor('#8ac8ff'),
        }),
      );
    } else {
      let path = '';
      for (const p of a)
        for (const q of b)
          path += `M${p.x + 3} ${p.y + 3}L${q.x + 3} ${q.y + 3}`;
      chart.append(
        svg('path', {
          d: path,
          fill: 'none',
          stroke: visualColor('#8ac8ff'),
          'stroke-width': 0.5,
          opacity: 0.16,
        }),
      );
    }
    chart.append(
      svg(
        'text',
        {
          x: (positions[i].cx + positions[i + 1].cx) / 2,
          y: height - 34,
          'text-anchor': 'middle',
        },
        `${a.length * b.length} links${bundled ? ' · bundled' : ''}`,
      ),
    );
  }
  stages.forEach((s, i) => {
    const p = positions[i];
    chart.append(
      svg('text', { x: p.cx, y: 24, 'text-anchor': 'middle' }, s.label),
      svg(
        'text',
        { x: p.cx, y: 46, 'text-anchor': 'middle' },
        `${s.values.length} units${s.recurrent ? ' · ↻ memory' : ''}`,
      ),
    );
    for (const unit of p.points) {
      const v = unit.value,
        known = Number.isFinite(v),
        box = svg('rect', {
          x: unit.x,
          y: unit.y,
          width: 7,
          height: 7,
          rx: 2,
          fill: known
            ? v >= 0
              ? visualColor('#7de2bd')
              : visualColor('#ff9aab')
            : visualColor('#61798f'),
          opacity: known ? 0.25 + 0.75 * Math.min(1, Math.abs(v)) : 0.4,
        });
      box.append(
        svg(
          'title',
          {},
          `${s.labels?.[unit.index] ?? `Unit ${unit.index}`}: ${known ? v.toPrecision(6) : 'activation not recorded'}`,
        ),
      );
      chart.append(box);
    }
  });
  chart.append(
    svg(
      'text',
      { x: width / 2, y: height - 8, 'text-anchor': 'middle' },
      'All units shown · connections show topology, not weight strength · hover for values',
    ),
  );
  const scroll = el('div', undefined, 'network-structure-scroll');
  scroll.append(chart);
  root.append(scroll);
  root.append(
    el(
      'p',
      `${plan.nodes} units and complete dense inter-stage connectivity${bundled ? ' represented with bundles' : ''}. ${description.available ? 'Colors show available activations/scores.' : 'Hidden values were not recorded; gray units show declared architecture only.'} GRU gates are collapsed into a memory block. This is structure, not a causal explanation.`,
      'muted',
    ),
  );
  return root;
}
