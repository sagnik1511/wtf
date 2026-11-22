import { visualColor } from './theme.mjs';
import {
  DEFAULT_NETWORK_RANK,
  rankLimit,
  archivedStructure,
  networkStructure,
} from './network-structure.mjs';
// Bounded visualization of one server-computed archived-model explanation.
// No weights, training state or replay chunks are retained here.
const NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);
const label = (s) => String(s).replaceAll('_', ' ');
const node = (tag, text, cls) => {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
};
const svg = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs))
    n.setAttribute(key, String(value));
  if (text !== undefined) n.textContent = text;
  return n;
};
const signed = (v) => `${v >= 0 ? '+' : ''}${v.toPrecision(3)}`;
const color = (v) => (v >= 0 ? visualColor('#7de2bd') : visualColor('#ff9aab'));
function chart(title, width, height) {
  const n = svg('svg', {
    viewBox: `0 0 ${width} ${height}`,
    role: 'img',
    'aria-label': title,
  });
  n.append(svg('title', {}, title));
  return n;
}
function card(title, description) {
  const c = node('section', undefined, 'explain-card');
  c.append(node('h3', title), node('p', description, 'muted'));
  return c;
}
export function routeGraph(report) {
  const width = Math.max(850, report.layers.length * 180),
    height = 490;
  const graph = chart(
    'Archived brain activation routes and strongest visible connections',
    width,
    height,
  );
  const positions = new Map();
  report.layers.forEach((layer, i) => {
    const x = 112 + i * ((width - 224) / (report.layers.length - 1));
    graph.append(
      svg(
        'text',
        { x, y: 23, 'text-anchor': 'middle', class: 'explain-layer' },
        layer.label,
      ),
    );
    graph.append(
      svg(
        'text',
        { x, y: 43, 'text-anchor': 'middle' },
        `${layer.nodes.length}/${layer.size} · ${layer.activation}`,
      ),
    );
    layer.nodes.forEach((n, j) =>
      positions.set(`${i}:${n.index}`, {
        x,
        y: 82 + j * (350 / Math.max(1, layer.nodes.length - 1)),
      }),
    );
  });
  const max = Math.max(1e-12, ...report.edges.map((e) => Math.abs(e.signal)));
  for (const e of report.edges) {
    const a = positions.get(`${e.layer}:${e.source}`),
      b = positions.get(`${e.layer + 1}:${e.target}`);
    if (!a || !b) continue;
    const line = svg('line', {
      x1: a.x + 9,
      y1: a.y,
      x2: b.x - 9,
      y2: b.y,
      stroke: color(e.signal),
      'stroke-width': 0.5 + 3 * Math.sqrt(Math.abs(e.signal) / max),
      opacity: 0.18 + 0.6 * Math.sqrt(Math.abs(e.signal) / max),
    });
    line.append(
      svg(
        'title',
        {},
        `${e.local ? 'Recurrent local Jacobian' : 'Weight'} ${signed(e.weight)} · ${e.local ? 'local signal' : 'weighted activation'} ${signed(e.signal)}`,
      ),
    );
    graph.append(line);
  }
  report.layers.forEach((layer, i) => {
    for (const n of layer.nodes) {
      const p = positions.get(`${i}:${n.index}`);
      const active =
        i === report.layers.length - 1 &&
        (n.target || n.label === report.target);
      const dot = svg('circle', {
        cx: p.x,
        cy: p.y,
        r: active ? 10 : 7,
        fill: color(n.value),
        opacity: 0.4 + 0.6 * Math.min(1, Math.abs(n.value)),
        stroke: active ? visualColor('#fff') : visualColor('#415569'),
        'stroke-width': active ? 2 : 1,
      });
      dot.append(
        svg(
          'title',
          {},
          `${n.label}: activation ${signed(n.value)}${Number.isFinite(n.bias) ? ` · bias ${signed(n.bias)}` : ''} · output sensitivity ${signed(n.sensitivity)}`,
        ),
      );
      graph.append(
        dot,
        svg(
          'text',
          {
            x: p.x + (i === 0 ? -14 : 14),
            y: p.y + 4,
            'text-anchor': i === 0 ? 'end' : 'start',
          },
          label(n.label).slice(0, 24),
        ),
      );
    }
  });
  graph.append(
    svg(
      'text',
      { x: width / 2, y: 475, 'text-anchor': 'middle' },
      'Green = positive signal · pink = negative · hover for values · white ring = target action',
    ),
  );
  return graph;
}
function attributionChart(report) {
  const rows = report.top_features,
    height = rows.length * 26 + 40;
  const graph = chart(
    `Local input sensitivity for ${label(report.target)}`,
    650,
    height,
  );
  const max = Math.max(1e-12, ...rows.map((r) => Math.abs(r.attribution))),
    zero = 425;
  graph.append(
    svg('line', {
      x1: zero,
      y1: 6,
      x2: zero,
      y2: height - 24,
      stroke: visualColor('#61798f'),
    }),
  );
  rows.forEach((r, i) => {
    const y = 10 + i * 26,
      w = (Math.abs(r.attribution) / max) * 150;
    graph.append(
      svg('text', { x: 240, y: y + 12, 'text-anchor': 'end' }, r.label),
    );
    const bar = svg('rect', {
      x: r.attribution < 0 ? zero - w : zero,
      y,
      width: w,
      height: 15,
      fill: color(r.attribution),
      rx: 2,
    });
    bar.append(
      svg(
        'title',
        {},
        `Input ${r.value.toPrecision(4)} × gradient ${signed(r.gradient)} = ${signed(r.attribution)}`,
      ),
    );
    graph.append(
      bar,
      svg(
        'text',
        { x: 640, y: y + 12, 'text-anchor': 'end' },
        signed(r.attribution),
      ),
    );
  });
  graph.append(
    svg(
      'text',
      { x: zero, y: height - 5, 'text-anchor': 'middle' },
      '← negative local influence · zero · positive →',
    ),
  );
  return graph;
}
function outputChart(report) {
  const rows = report.outputs,
    height = rows.length * 36 + 35;
  const graph = chart(
    'Archived model predictions compared with recorded animal predictions',
    650,
    height,
  );
  const recorded =
    report.recorded_policy === report.policy ? report.recorded_predictions : {};
  const all = rows.flatMap((r) => [r.score, recorded[r.action] ?? 0]);
  const min = report.policy ? 0 : Math.min(0, ...all),
    max = report.policy ? 1 : Math.max(0, ...all),
    span = max - min || 1;
  const x = (v) => 205 + ((v - min) / span) * 345;
  for (const [i, r] of rows.entries()) {
    const y = 8 + i * 36;
    graph.append(
      svg(
        'text',
        { x: 195, y: y + 14, 'text-anchor': 'end' },
        label(r.action) + (r.masked ? ' · masked' : ''),
      ),
    );
    const rect = svg('rect', {
      x: Math.min(x(0), x(r.score)),
      y,
      width: Math.abs(x(r.score) - x(0)),
      height: 11,
      fill:
        r.action === report.target
          ? visualColor('#7de2bd')
          : visualColor('#8ac8ff'),
      rx: 2,
    });
    rect.append(
      svg('title', {}, `Archived inference ${r.score.toPrecision(5)}`),
    );
    graph.append(rect);
    if (Number.isFinite(recorded[r.action]))
      graph.append(
        svg('rect', {
          x: Math.min(x(0), x(recorded[r.action])),
          y: y + 13,
          width: Math.abs(x(recorded[r.action]) - x(0)),
          height: 6,
          fill: visualColor('#8b9cad'),
          rx: 2,
        }),
      );
    graph.append(
      svg(
        'text',
        { x: 640, y: y + 14, 'text-anchor': 'end' },
        report.policy ? `${(r.score * 100).toFixed(1)}%` : r.score.toFixed(3),
      ),
    );
  }
  graph.append(
    svg(
      'text',
      { x: 205, y: height - 7 },
      'Color = archived inference · gray = recorded prediction (different weights/time)',
    ),
  );
  return graph;
}
function heatmap(title, values) {
  const columns = 32,
    graph = chart(title, 650, Math.ceil(values.length / columns) * 18 + 35);
  values.forEach((v, i) => {
    const box = svg('rect', {
      x: 8 + (i % columns) * 19,
      y: 6 + Math.floor(i / columns) * 18,
      width: 16,
      height: 14,
      fill: color(v),
      opacity: 0.18 + 0.82 * Math.min(1, Math.abs(v)),
      rx: 2,
    });
    box.append(svg('title', {}, `Unit ${i}: ${signed(v)}`));
    graph.append(box);
  });
  graph.append(
    svg(
      'text',
      { x: 8, y: Math.ceil(values.length / columns) * 18 + 28 },
      `${values.length} units · sign by color · intensity clipped at |activation| = 1`,
    ),
  );
  return graph;
}
export function explanationCharts(
  report,
  { rank = DEFAULT_NETWORK_RANK, sections = {} } = {},
) {
  const root = node('div', undefined, 'explain-charts');
  const influence = card(
    `How inputs affect “${label(report.target)}”`,
    `${report.method}. Local sensitivity, not causal importance or an additive decomposition. Zero-valued inputs have zero input×gradient even if their derivative is nonzero.`,
  );
  influence.append(attributionChart(report));
  const outputs = card(
    report.policy
      ? 'Predicted action probabilities'
      : 'Predicted action returns',
    'Color = archived model on this observation. Gray = the animal’s recorded prediction. Exploration may choose a different action.',
  );
  outputs.append(outputChart(report));
  const structure = networkStructure(archivedStructure(report), rank);
  structure.className += ' explain-card explain-wide';
  const advanced = node('details', undefined, 'explain-card explain-wide');
  advanced.append(
    node(
      'summary',
      'Advanced · weighted routes, individual activations and recurrent gates',
    ),
  );
  advanced.open = Boolean(sections.advanced);
  let rendered = false;
  const render = () => {
    if (rendered) return;
    rendered = true;
    const routes = card(
      'Selected weighted routes',
      'Up to 12 influential units per hidden layer and 3 incoming visible links per unit. Omitted routes still participate; hidden units have no assigned biological meanings.',
    );
    const scroll = node('div', undefined, 'explain-routes');
    scroll.append(routeGraph(report));
    routes.append(scroll);
    advanced.append(routes);
    const activation = card(
      'Detailed activation maps',
      'All non-input units; hover for signed activation.',
    );
    report.layers
      .slice(1)
      .forEach((layer, i) =>
        activation.append(
          node('h4', `${layer.label} · ${i + 1}`),
          heatmap(layer.label, layer.values),
        ),
      );
    if (report.gates)
      for (const [name, values] of Object.entries(report.gates))
        activation.append(
          node(
            'h4',
            `${report.algorithm === 'lstm_ppo' ? 'LSTM' : 'GRU'} ${name}`,
          ),
          heatmap(
            `${report.algorithm === 'lstm_ppo' ? 'LSTM' : 'GRU'} ${name}`,
            values,
          ),
        );
    advanced.append(activation);
  };
  advanced.ontoggle = () => {
    sections.advanced = advanced.open;
    if (advanced.open) render();
  };
  if (advanced.open) render();
  root.append(influence, outputs, structure, advanced);
  return root;
}
export class BrainExplanation {
  constructor(loadJSON) {
    this.loadJSON = loadJSON;
    this.rank = DEFAULT_NETWORK_RANK;
    this.sections = {};
    this.request = 0;
    this.context = null;
    this.override = null;
    this.inFlight = false;
    this.pending = null;
    this.following = true;
    $('explain-close').onclick = () => $('explain-dialog').close();
    $('explain-dialog').onclose = () => {
      this.invalidate();
      this.context = null;
      this.latest = null;
      this.renderedContext = null;
      this.result = null;
      this.key = null;
      $('explain-content').replaceChildren();
    };
    $('explain-follow').checked = true;
    $('explain-follow').onchange = () => {
      this.following = $('explain-follow').checked;
      if (this.following) {
        this.override = null;
        this.failure = null;
        this.sampleKey = null;
        if (this.latest) this.update(this.latest);
      } else {
        this.pin();
        if (this.result) this.status(this.result);
        else this.load();
      }
    };
    for (const id of ['explain-criterion', 'explain-action'])
      $(id).onchange = () => this.load();
    $('explain-retry').onclick = () => this.load();
    $('explain-feature').onchange = () => this.syncFeature();
    $('explain-apply').onclick = () => {
      const index = Number($('explain-feature').value),
        value = Number($('explain-value').value);
      if (
        !Number.isInteger(index) ||
        !Number.isFinite(value) ||
        value < -1 ||
        value > 1 ||
        !$('explain-value').value.trim()
      ) {
        $('explain-status').textContent =
          'Choose an input value between −1 and 1.';
        return;
      }
      this.following = false;
      $('explain-follow').checked = false;
      this.pin();
      this.onPin?.();
      this.override = { index, value };
      this.load();
    };
    $('explain-reset').onclick = () => {
      this.override = null;
      this.load();
    };
  }
  setRankLimit(value) {
    this.rank = rankLimit(value);
    if (this.result && $('explain-dialog').open) this.render();
  }
  render() {
    $('explain-content').replaceChildren(
      explanationCharts(this.result, {
        rank: this.rank,
        sections: this.sections,
      }),
    );
  }
  pin() {
    this.invalidate();
    if (this.renderedContext) this.context = { ...this.renderedContext };
  }
  invalidate() {
    ++this.request;
    if (this.pending) this.pending.resolve();
    this.pending = null;
  }
  group(context) {
    return `${context.id}:${context.run}:${context.animal}`;
  }
  sample(context) {
    return `${this.group(context)}:${context.diagnosticTick ?? context.frame}`;
  }
  open(context) {
    this.invalidate();
    this.context = { ...context };
    this.latest = { ...context };
    this.override = null;
    this.result = null;
    this.renderedContext = null;
    this.key = null;
    this.failure = null;
    this.following = true;
    $('explain-follow').checked = true;
    this.sampleKey = this.sample(context);
    this.clearOptions();
    $('explain-dialog').showModal();
    this.controls(context);
    return this.load();
  }
  clearOptions() {
    for (const id of ['explain-criterion', 'explain-action', 'explain-feature'])
      $(id).replaceChildren();
    $('explain-apply').disabled = true;
  }
  controls(context) {
    $('explain-play').textContent = context.playing
      ? 'Pause replay'
      : 'Play replay';
    $('explain-play').disabled = !context.playing && Boolean(context.done);
  }
  update(context) {
    if (!$('explain-dialog').open) return;
    this.latest = { ...context };
    this.controls(context);
    if (!this.following) return;
    if (
      context.animal === null ||
      context.animal === undefined ||
      context.diagnosticTick === null
    ) {
      this.invalidate();
      this.context = null;
      this.result = null;
      this.key = null;
      this.sampleKey = null;
      $('explain-content').replaceChildren();
      $('explain-apply').disabled = true;
      $('explain-status').textContent =
        'Select a living animal with recorded observations at this frame. The previous explanation has been cleared.';
      return;
    }
    const changedGroup =
      !this.context || this.group(context) !== this.group(this.context);
    if (changedGroup) {
      this.clearOptions();
      this.override = null;
      this.failure = null;
      this.result = null;
      this.key = null;
      $('explain-content').replaceChildren();
    }
    const changedSample = this.sample(context) !== this.sampleKey;
    this.context = { ...context };
    this.sampleKey = this.sample(context);
    if (this.result) this.status(this.result);
    if ((changedGroup || changedSample) && !this.failure) this.load();
  }
  syncFeature() {
    const feature = this.result?.features.find(
      (f) => f.index === Number($('explain-feature').value),
    );
    if (feature) $('explain-value').value = feature.value;
  }
  options(id, values, first = null) {
    const select = $(id),
      previous = select.value;
    select.replaceChildren();
    if (first) {
      const o = node('option', first);
      o.value = '';
      select.append(o);
    }
    for (const [value, title] of values) {
      const o = node('option', title);
      o.value = String(value);
      select.append(o);
    }
    if (
      values.some(([v]) => String(v) === previous) ||
      (first && previous === '')
    )
      select.value = previous;
  }
  status(r) {
    const source = r.source ?? {},
      body = this.following ? (this.latest?.tick ?? r.tick) : r.tick;
    $('explain-status').textContent =
      `${this.following ? 'Following replay' : 'Pinned observation'} · ${r.species} ${r.sex} #${r.animal} · run ${r.run}, body step ${body}, explained observation step ${r.diagnostic_tick}. Archived ${r.criterion} champion #${source.animal_id ?? '?'} from run ${source.run ?? '?'}, step ${source.tick ?? '?'}, model generation ${r.model_generation ?? '?'}. ${r.parameters.toLocaleString()} parameters. Target: ${label(r.target)}.${r.recurrent_reset ? ' Archived recurrent explanations use zero prior memory; replay activations remain actual recorded samples.' : ''}${r.critic_separate ? ' Only the actor is shown; separate critics are not action-output routes.' : ''}${r.intervention ? ' What-if input override active; geometry, other inputs and recorded masks are held fixed. This is not a feasible-world simulation.' : ''}`;
  }
  job() {
    const context = { ...this.context },
      query = new URLSearchParams();
    const criterion =
        $('explain-criterion').value || this.result?.criterion || '',
      action = $('explain-action').value || '';
    if (criterion) query.set('criterion', criterion);
    if (action) query.set('action', action);
    const override = this.override ? { ...this.override } : null;
    if (override) {
      query.set('feature', override.index);
      query.set('value', override.value);
    }
    const identity = (resolved) =>
      `${this.sample(context)}:${resolved}:${action}:${override ? `${override.index}=${override.value}` : ''}`;
    return {
      context,
      key: identity(criterion),
      identity,
      path: `/api/recordings/${encodeURIComponent(context.id)}/runs/${context.run}/explain/${context.animal}/${context.frame}?${query}`,
    };
  }
  load() {
    if (!this.context || !$('explain-dialog').open) return Promise.resolve();
    const job = this.job();
    if (job.key === this.key && this.result) {
      // A backward seek can return to the displayed sample while future work is pending.
      this.invalidate();
      this.status(this.result);
      return Promise.resolve();
    }
    if (this.pending?.key === job.key) return this.pending.promise;
    if (this.current?.key === job.key && this.current.token === this.request)
      return this.current.promise;
    this.failure = null;
    job.token = ++this.request;
    job.promise = new Promise((resolve) => {
      job.resolve = resolve;
    });
    if (this.pending) this.pending.resolve();
    this.pending = job;
    if (!this.result)
      $('explain-status').textContent =
        'Loading one archived brain and the selected observation…';
    else
      $('explain-status').textContent +=
        ' Updating for the latest recorded sample…';
    if (!this.inFlight) this.drain();
    return job.promise;
  }
  async drain() {
    this.inFlight = true;
    try {
      while (this.pending) {
        const job = this.pending;
        this.pending = null;
        this.current = job;
        try {
          const r = await this.loadJSON(job.path);
          if (job.token !== this.request || !$('explain-dialog').open) continue;
          this.result = r;
          this.renderedContext = { ...job.context };
          this.key = job.identity(r.criterion);
          this.options(
            'explain-criterion',
            r.available_criteria.map((c) => [c, label(c)]),
          );
          $('explain-criterion').value = r.criterion;
          this.options(
            'explain-action',
            r.outputs.map((a) => [a.action, label(a.action)]),
            'Highest predicted action',
          );
          this.options(
            'explain-feature',
            r.features.map((f) => [f.index, `${f.index} · ${f.label}`]),
          );
          this.syncFeature();
          $('explain-apply').disabled = false;
          this.status(r);
          this.render();
        } catch (error) {
          if (job.token !== this.request || !$('explain-dialog').open) continue;
          this.result = null;
          this.key = null;
          this.failure = true;
          $('explain-content').replaceChildren();
          $('explain-apply').disabled = true;
          $('explain-status').textContent =
            `Explanation unavailable: ${error.message} Use Retry after correcting the issue.`;
        } finally {
          job.resolve();
        }
      }
    } finally {
      this.current = null;
      this.inFlight = false;
    }
  }
}
