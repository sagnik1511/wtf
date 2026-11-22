import { SCENT_OFFSETS, SCENT_SECTORS } from './tracking.mjs';
import { visualColor } from './theme.mjs';
import {
  observationView,
  predictionView,
  TARGET_COLORS,
} from './inspector-stats.mjs';
import { DEFAULT_NETWORK_RANK } from './network-structure.mjs';
import { activationPanel } from './brain-activations.mjs';
const NS = 'http://www.w3.org/2000/svg';
const label = (action) => action.replaceAll('_', ' ');
const el = (tag, value, cls) => {
  const n = document.createElement(tag);
  if (value !== undefined) n.textContent = value;
  if (cls) n.className = cls;
  return n;
};
const svg = (tag, attrs = {}, value) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (value !== undefined) n.textContent = value;
  return n;
};
const format = (value, policy) =>
  policy ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
function section(title) {
  const n = el('section', undefined, 'inspector-visual');
  n.append(el('h3', title));
  return n;
}
function gauge(name, value) {
  const n = el('div', undefined, 'input-gauge'),
    track = el('div', undefined, 'input-gauge-track'),
    fill = el('span');
  fill.style.width = `${100 * Math.max(0, Math.min(1, value))}%`;
  track.append(fill);
  n.append(el('span', `${name} ${(value * 100).toFixed(0)}%`), track);
  return n;
}
function radar(observation, baseFraction) {
  const chart = svg('svg', {
    viewBox: '0 0 280 194',
    role: 'img',
    'aria-label':
      'Sampled observation radar. Targets are relative to the animal; north is up.',
    class: 'observation-radar',
  });
  const cx = 140,
    cy = 94,
    r = 76;
  chart.append(
    svg('circle', {
      cx,
      cy,
      r,
      fill: visualColor('#0b131c'),
      stroke: visualColor('#415569'),
    }),
  );
  for (const [x1, y1, x2, y2] of [
    [cx - r, cy, cx + r, cy],
    [cx, cy - r, cx, cy + r],
  ])
    chart.append(
      svg('line', { x1, y1, x2, y2, stroke: visualColor('#263341') }),
    );
  if (baseFraction < 1)
    chart.append(
      svg('circle', {
        cx,
        cy,
        r: r * baseFraction,
        fill: 'none',
        stroke: visualColor('#8ac8ff88'),
        'stroke-dasharray': '3 4',
      }),
    );
  chart.append(
    svg('text', { x: cx, y: 12, 'text-anchor': 'middle' }, 'N'),
    svg('text', { x: cx + r + 9, y: cy + 4 }, 'E'),
  );
  for (const target of observation.targets) {
    const x = cx + target.dx * r,
      y = cy + target.dy * r,
      color = TARGET_COLORS[target.category];
    const mark =
      target.category === 'food'
        ? svg('rect', {
            x: x - 3.5,
            y: y - 3.5,
            width: 7,
            height: 7,
            fill: color,
          })
        : target.category === 'predator'
          ? svg('path', {
              d: `M ${x} ${y - 5} L ${x + 4.5} ${y + 4} L ${x - 4.5} ${y + 4} Z`,
              fill: color,
            })
          : svg('circle', {
              cx: x,
              cy: y,
              r: 4,
              fill: target.category === 'mate' ? 'none' : color,
              stroke: color,
            });
    mark.append(
      svg(
        'title',
        {},
        `${target.category} slot ${target.slot}: relative (${target.dx.toFixed(3)}, ${target.dy.toFixed(3)}), distance ${target.distance.toFixed(3)}`,
      ),
    );
    chart.append(mark);
  }
  const [hx, hy] = observation.heading;
  chart.append(
    svg('line', {
      x1: cx,
      y1: cy,
      x2: cx + hx * 12,
      y2: cy + hy * 12,
      stroke: visualColor('#fff'),
      'stroke-width': 2,
    }),
    svg('circle', { cx, cy, r: 2.5, fill: visualColor('#fff') }),
  );
  chart.append(
    svg(
      'text',
      { x: cx, y: 190, 'text-anchor': 'middle' },
      'Heading shown · circle = maximum sight',
    ),
  );
  return chart;
}
function trackingPanel(tracking) {
  const panel = section('Movement memory & scent');
  panel.className += ' tracking-panel';
  if (tracking.steps) {
    panel.append(
      el(
        'p',
        `${tracking.steps} previous steps · personal sightings only · hollow markers are last-seen memories`,
        'muted',
      ),
    );
    const chart = svg('svg', {
      viewBox: '0 0 280 210',
      role: 'img',
      'aria-label':
        'Personal movement history. Gaps are unobserved, dashed lines estimate last observed velocity; no hidden movement is shown.',
      class: 'movement-history',
    });
    const extent = Math.max(
      1,
      ...tracking.tracks.flatMap((t) => [
        Math.abs(t.dx) + Math.abs(t.vx),
        Math.abs(t.dy) + Math.abs(t.vy),
        ...t.history
          .filter((p) => p.seen)
          .flatMap((p) => [Math.abs(p.dx), Math.abs(p.dy)]),
      ]),
    );
    const r = 76 / extent,
      x = (v) => 140 + v * r,
      y = (v) => 96 + v * r;
    chart.append(
      svg('circle', {
        cx: 140,
        cy: 96,
        r: 76,
        fill: visualColor('#0b131c'),
        stroke: visualColor('#415569'),
      }),
      svg('text', { x: 140, y: 12, 'text-anchor': 'middle' }, 'N'),
      svg('circle', { cx: 140, cy: 96, r: 3, fill: visualColor('#fff') }),
    );
    for (const t of tracking.tracks) {
      const color = TARGET_COLORS[t.category],
        group = svg('g', { opacity: Math.max(0.25, t.confidence) });
      const points = [
        ...t.history.filter((p) => p.seen).reverse(),
        { lag: t.age, dx: t.dx, dy: t.dy },
      ].sort((a, b) => b.lag - a.lag);
      for (let i = 0; i < points.length; i++) {
        const p = points[i],
          prior = points[i - 1];
        if (prior && prior.lag - p.lag === 1)
          group.append(
            svg('line', {
              x1: x(prior.dx),
              y1: y(prior.dy),
              x2: x(p.dx),
              y2: y(p.dy),
              stroke: color,
              'stroke-width': 2,
            }),
          );
        group.append(
          svg('circle', { cx: x(p.dx), cy: y(p.dy), r: 2, fill: color }),
        );
      }
      const current = svg('circle', {
        cx: x(t.dx),
        cy: y(t.dy),
        r: 5,
        stroke: color,
        'stroke-width': 2,
        fill: t.visible ? color : 'none',
      });
      current.append(
        svg(
          'title',
          {},
          `${t.category} track ${t.slot}: ${t.visible ? 'visible' : `${t.age} steps since seen`} · ${(t.confidence * 100).toFixed(0)}% confidence`,
        ),
      );
      group.append(current);
      if (t.vx || t.vy) {
        const ex = x(t.dx + t.vx),
          ey = y(t.dy + t.vy);
        group.append(
          svg('line', {
            x1: x(t.dx),
            y1: y(t.dy),
            x2: ex,
            y2: ey,
            stroke: color,
            'stroke-dasharray': '3 2',
            'stroke-width': 2,
          }),
        );
      }
      chart.append(group);
    }
    chart.append(
      svg(
        'text',
        { x: 140, y: 198, 'text-anchor': 'middle' },
        'Dots = sightings · dashed line = velocity estimate',
      ),
    );
    panel.append(chart);
    if (!tracking.tracks.length)
      panel.append(
        el('p', 'No recent animal sightings in this sample.', 'muted'),
      );
    for (const t of tracking.tracks)
      panel.append(
        el(
          'p',
          `${t.category} track ${t.slot} · ${t.visible ? 'visible' : `last seen ${t.age} steps ago`} · ${(t.confidence * 100).toFixed(0)}% confidence`,
          'muted',
        ),
      );
  }
  if (tracking.scent.length) {
    panel.append(
      el(
        'p',
        `Anonymous ground scent · radius ${tracking.radius} dry-cell steps · each grid is a species/sex group, not one animal`,
        'muted',
      ),
    );
    const grids = el('div', undefined, 'scent-grids');
    for (const channel of tracking.scent) {
      const tile = el('div', undefined, 'scent-channel');
      tile.append(
        el(
          'h4',
          `${channel.species} · ${channel.sex === 'any' ? 'all sexes' : channel.sex}`,
        ),
      );
      const chart = svg('svg', {
        viewBox: '0 0 126 138',
        role: 'img',
        'aria-label': `${channel.species} ${channel.sex} local scent pattern; north is up.`,
        class: 'scent-grid',
      });
      chart.append(svg('text', { x: 63, y: 11, 'text-anchor': 'middle' }, 'N'));
      channel.strengths.forEach((strength, i) => {
        const [dx, dy] = SCENT_OFFSETS[i],
          rx = 3 + (dx + 1) * 40,
          ry = 16 + (dy + 1) * 40,
          color = visualColor(channel.sex === 'male' ? '#c6a5ff' : '#7de2bd');
        chart.append(
          svg('rect', {
            x: rx,
            y: ry,
            width: 38,
            height: 38,
            rx: 3,
            fill: color,
            'fill-opacity': 0.05 + 0.5 * strength,
            stroke: visualColor('#415569'),
          }),
        );
        const number = svg(
          'text',
          {
            x: rx + 19,
            y: ry + 23,
            'text-anchor': 'middle',
            style: `fill:${visualColor('#fff')}`,
          },
          String(Math.round(strength * 100)),
        );
        number.append(
          svg(
            'title',
            {},
            `${SCENT_SECTORS[i]} · ${(strength * 100).toFixed(0)}% freshness${i === 0 ? ' · own cell; own scent may be included' : ''}`,
          ),
        );
        chart.append(number);
      });
      tile.append(chart);
      grids.append(tile);
    }
    panel.append(grids);
  }
  return panel;
}

function savedDisclosure(node, key, sections) {
  node.open = Boolean(sections[key]);
  node.ontoggle = () => {
    sections[key] = node.open;
  };
  return node;
}
function inputDetails(observation, sections) {
  const fragment = el('div');
  const vitals = el('div', undefined, 'input-gauges');
  vitals.append(
    gauge('Energy', observation.energy),
    gauge('Health', observation.health),
  );
  fragment.append(vitals);
  const neighbors = el('div', undefined, 'blocked-inputs');
  observation.blocked.forEach((v, i) =>
    neighbors.append(
      el(
        'span',
        `${['Front', 'Right', 'Back', 'Left'][i]} ${v ? 'blocked' : 'open'}`,
        v ? 'blocked' : 'open',
      ),
    ),
  );
  fragment.append(neighbors);
  if (observation.terrain) {
    const t = observation.terrain,
      d = el('details', undefined, 'feature-details');
    d.append(el('summary', 'Sampled terrain inputs'));
    savedDisclosure(d, 'terrain', sections);
    d.append(
      el(
        'p',
        `Depth ${t.depth.toFixed(1)} · Elevation ${t.elevation.toFixed(1)} · Flow (${t.flow.map((v) => v.toFixed(2)).join(', ')})`,
        'muted',
      ),
    );
    d.append(
      el(
        'p',
        t.neighbors
          .map(
            (v, i) =>
              `${['Front', 'Right', 'Back', 'Left'][i]}: depth ${v.depth.toFixed(1)}, climb ${v.climb.toFixed(1)}`,
          )
          .join(' · '),
        'muted',
      ),
    );
    fragment.append(d);
  }
  if (observation.mating) {
    const m = observation.mating,
      d = el('details', undefined, 'feature-details');
    d.append(
      el(
        'summary',
        `Mating inputs · ${m.ready ? 'self ready' : 'self not ready'}`,
      ),
    );
    savedDisclosure(d, 'mating', sections);
    if (Number.isFinite(m.selfIntent)) {
      d.append(
        gauge('Own consent remaining', m.selfIntent),
        gauge('Visible mate consent remaining', m.mateIntent),
        el(
          'p',
          `Nearby consenting pair opportunity ${m.consentingPair ? 'yes' : 'no'}`,
        ),
      );
    }
    d.append(
      gauge('Maturity', m.maturity),
      gauge('Cooldown remaining', m.cooldown),
    );
    d.append(
      el(
        'p',
        `${m.sex} · Nearest mate ${m.present ? (m.mateReady ? 'ready' : 'not ready') : 'not seen'} · Pair opportunity ${m.pair ? 'yes' : 'no'}`,
        'muted',
      ),
    );
    fragment.append(d);
  }
  return fragment;
}
function predictions(diagnostic) {
  const values = predictionView(diagnostic),
    root = section(
      diagnostic.action_probabilities
        ? 'Action probabilities'
        : 'Estimated action returns',
    );
  if (!values) {
    root.append(el('p', 'No recorded action predictions.', 'muted'));
    return root;
  }
  root.append(
    el(
      'p',
      values.policy
        ? 'Sampled next-decision probabilities; outlined = top prediction.'
        : 'Q values, not probabilities. Exploration can choose another action.',
      'muted inspector-note',
    ),
  );
  for (const action of values.actions) {
    const row = el(
        'div',
        undefined,
        `prediction-row${action.best ? ' best' : ''}${action.masked ? ' masked' : ''}`,
      ),
      track = el('div', undefined, 'prediction-track'),
      fill = el('span', undefined, 'prediction-fill'),
      zero = el('i', undefined, 'prediction-zero');
    fill.style.left = `${action.left}%`;
    fill.style.width = `${action.width}%`;
    fill.style.backgroundColor = action.color;
    zero.style.left = `${values.zero}%`;
    track.append(fill, zero);
    row.append(
      el('span', label(action.action) + (action.masked ? ' · masked' : '')),
      track,
      el('strong', format(action.value, values.policy)),
    );
    root.append(row);
  }
  root.append(
    el(
      'p',
      values.policy
        ? 'Scale 0–100%'
        : `Scale ${values.min.toFixed(2)} to ${values.max.toFixed(2)} · vertical line = zero`,
      'muted inspector-scale',
    ),
  );
  return root;
}
function historyGraph(samples, policy, sections) {
  const d = el('details', undefined, 'prediction-history');
  d.append(el('summary', 'Recent prediction samples'));
  d.open = Boolean(sections.history);
  d.append(
    el(
      'p',
      'Up to 64 recorded samples · no future frames. Dots show recorded values; gaps are not filled.',
      'muted inspector-note',
    ),
  );
  let rendered = false;
  function render() {
    if (rendered) return;
    rendered = true;
    const points = [];
    for (const sample of samples) {
      const view = predictionView(sample.diagnostic);
      if (view?.policy === policy) points.push({ tick: sample.tick, view });
    }
    if (!points.length) {
      d.append(el('p', 'No samples in the loaded replay window.', 'muted'));
      return;
    }
    const chart = svg('svg', {
      viewBox: '0 0 280 146',
      role: 'img',
      'aria-label': policy
        ? 'Recorded action probabilities over time'
        : 'Recorded Q values over time',
      class: 'prediction-trend',
    });
    const first = points[0].tick,
      last = points.at(-1).tick,
      min = policy ? 0 : Math.min(...points.map((p) => p.view.min)),
      maximum = policy ? 1 : Math.max(...points.map((p) => p.view.max));
    const max = maximum === min ? min + 1 : maximum,
      span = max - min;
    for (const fraction of [0, 0.5, 1]) {
      const y = 10 + fraction * 105;
      chart.append(
        svg('line', {
          x1: 38,
          y1: y,
          x2: 272,
          y2: y,
          stroke: visualColor('#263341'),
        }),
        svg(
          'text',
          { x: 34, y: y + 4, 'text-anchor': 'end' },
          policy
            ? `${(1 - fraction) * 100}%`
            : (max - fraction * span).toFixed(1),
        ),
      );
    }
    for (const p of points)
      for (const a of p.view.actions) {
        const dot = svg('circle', {
          cx:
            first === last
              ? 155
              : 38 + ((p.tick - first) / (last - first)) * 234,
          cy: 10 + ((max - a.value) / span) * 105,
          r: 2.5,
          fill: a.color,
        });
        dot.append(
          svg(
            'title',
            {},
            `${label(a.action)} · step ${p.tick}: ${format(a.value, policy)}`,
          ),
        );
        chart.append(dot);
      }
    chart.append(
      svg('text', { x: 38, y: 137 }, String(first)),
      svg('text', { x: 272, y: 137, 'text-anchor': 'end' }, String(last)),
    );
    d.append(chart);
  }
  d.ontoggle = () => {
    sections.history = d.open;
    if (d.open) render();
  };
  if (d.open) render();
  return d;
}

export function inspectorGraphs({
  diagnostic,
  template,
  spec,
  hasWater,
  samples = [],
  sections = {},
  rank = DEFAULT_NETWORK_RANK,
}) {
  const root = el('div', undefined, 'inspector-graphs'),
    observation = observationView(diagnostic.features, template);
  const inputs = section('Sampled observation');
  if (observation) {
    const base = spec.body?.vision_radius ?? 1,
      bonus = hasWater ? (spec.body?.water_vision_bonus ?? 0) : 0;
    inputs.append(
      el(
        'p',
        `Step ${diagnostic.tick} · v${observation.version} · k=${observation.k} · ${observation.inputs} inputs`,
        'muted inspector-note',
      ),
    );
    inputs.append(radar(observation, base / (base + bonus)));
    const legend = el('div', undefined, 'observation-legend');
    for (const [category, title] of [
      ['food', '■ Food'],
      ['prey', '● Prey'],
      ['predator', '▲ Predators'],
      ...(observation.mating ? [['mate', '○ Mate']] : []),
    ]) {
      const count = observation.targets.filter(
          (t) => t.category === category,
        ).length,
        n = el(
          'span',
          `${title} ${count}/${category === 'mate' ? 1 : observation.k}`,
        );
      n.style.color = TARGET_COLORS[category];
      legend.append(n);
    }
    inputs.append(legend);
    if (bonus)
      inputs.append(
        el(
          'p',
          'Dashed ring = base range; board shading includes water and wall limits.',
          'muted inspector-note',
        ),
      );
    inputs.append(inputDetails(observation, sections));
  } else
    inputs.append(
      el(
        'p',
        'Observation layout is missing or unsupported. Raw inputs remain available below.',
        'muted',
      ),
    );
  root.append(
    inputs,
    ...(observation?.tracking ? [trackingPanel(observation.tracking)] : []),
    predictions(diagnostic),
    activationPanel(diagnostic, template, { rank, sections }),
    historyGraph(samples, Boolean(diagnostic.action_probabilities), sections),
  );
  return root;
}
