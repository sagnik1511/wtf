import { movementHeatmap } from './behavior-stats.mjs';
import { collectivePanel } from './collective-view.mjs';
import { TelemetryCache, metrics, series } from './dashboard-stats.mjs';
const $ = (id) => document.getElementById(id);
const number = (value) =>
  Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
    : '—';
const percent = (value) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const element = (tag, value, cls) => {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = value;
  if (cls) node.className = cls;
  return node;
};

const replaceIfChanged = (parent, children) => {
  if (
    parent.children.length !== children.length ||
    children.some((node, i) => parent.children[i] !== node)
  )
    parent.replaceChildren(...children);
};

export class Dashboard {
  constructor(loadJSON, color, selectMovementSpecies = null) {
    this.cache = new TelemetryCache(loadJSON);
    this.color = color;
    this.selectMovementSpecies = selectMovementSpecies;
    this.speciesHeatmaps = new Map();
    this.heatmap = null;
    this.request = 0;
    this.points = [];
    this.collectivePanels = new Map();
    this.speciesCards = new Map();
    this.legendItems = new Map();
    this.chartTypes = [
      'population',
      'energy',
      'births',
      'kills',
      'probability',
      'clusters',
    ];
    this.charts = this.chartTypes.map((type, i) => {
      const panel = element('section', undefined, 'trend-card'),
        label = element('label', `Chart ${i + 1}`, 'visually-hidden');
      const select = element('select');
      select.id = `trend-${i}`;
      label.htmlFor = select.id;
      for (const [key, value] of Object.entries(metrics)) {
        const option = element('option', value.label);
        option.value = key;
        select.append(option);
      }
      select.value = type;
      select.onchange = () => {
        this.chartTypes[i] = select.value;
        this.draw();
      };
      const canvas = element('canvas');
      canvas.setAttribute('role', 'img');
      const summary = element('p', '', 'trend-current muted');
      panel.append(label, select, canvas, summary);
      $('trend-grid').append(panel);
      return { canvas, summary, select };
    });
    $('dashboard-species').onchange = async () => {
      this.draw();
      if (!this.selectMovementSpecies) return;
      try {
        await this.selectMovementSpecies($('dashboard-species').value || null);
        this.drawMovement();
      } catch (error) {
        $('dashboard-movement').replaceChildren(
          element(
            'p',
            `Movement heatmap unavailable: ${error.message}`,
            'muted',
          ),
        );
      }
    };
    $('dashboard-action').onchange = () => this.draw();
    $('dashboard-window').onchange = () => this.update(this.context);
    this.resize = new ResizeObserver(() => this.draw());
    this.resize.observe($('trend-grid'));
    $('metrics-open').onclick = () => {
      if (!this.context) return;
      $('metrics-dialog').showModal();
      this.update(this.context);
    };
  }
  speciesMovement(name) {
    const { state } = this.context;
    const map = state.movement_by_species?.[name];
    if (!map || map.frame_index !== state.frame_index) return null;
    let widget = this.speciesHeatmaps.get(name);
    if (!widget) {
      widget = movementHeatmap(map, state.obstacles ?? [], { compact: true });
      this.speciesHeatmaps.set(name, widget);
    }
    widget.update(
      map,
      document.documentElement?.dataset.theme ?? 'dark',
      this.speciesMaximum,
    );
    return widget;
  }
  drawMovement() {
    if (!this.context || !$('metrics-dialog').open) return;
    const { state, id } = this.context;
    const runKey = `${id}:${state.run}`;
    if (this.movementKey !== runKey) {
      this.movementKey = runKey;
      this.speciesHeatmaps.clear();
    }
    this.speciesMaximum = Math.max(
      0,
      ...Object.values(state.movement_by_species ?? {}).map(
        (group) => group.maximum,
      ),
    );
    if (!$('dashboard-species').value)
      for (const name of this.speciesHeatmaps.keys())
        this.speciesMovement(name);
    const map = state.movement_heatmap;
    const box = $('dashboard-movement');
    const chosen = $('dashboard-species').value || null;
    box.hidden = !map || !chosen;
    if (!map || !chosen) {
      this.heatmap = null;
      if (box.firstChild) box.replaceChildren();
      return;
    }
    const key = `${id}:${state.run}:${chosen}`;
    if (
      (map.species ?? null) !== chosen ||
      map.frame_index !== state.frame_index
    ) {
      this.heatmap = null;
      box.replaceChildren(
        element(
          'p',
          'Loading selected species movement through the current step…',
          'muted',
        ),
      );
      return;
    }
    const theme = document.documentElement?.dataset.theme ?? 'dark';
    if (!this.heatmap || this.heatmapKey !== key) {
      this.heatmapKey = key;
      this.heatmap = movementHeatmap(map, state.obstacles ?? []);
      box.replaceChildren(this.heatmap.element);
    }
    this.heatmap.update(map, theme);
  }
  drawCollective() {
    if (!this.context || !$('metrics-dialog').open) return;
    const { state, id, species } = this.context;
    const key = `${id}:${state.run}`;
    if (this.collectiveKey !== key) {
      this.collectiveKey = key;
      this.collectivePanels.clear();
      $('collective-grid').replaceChildren();
    }
    const chosen = $('dashboard-species').value;
    // A backward seek may precede a species' first saved evidence.
    for (const panel of this.collectivePanels.values()) panel.hidden = true;
    let visible = false;
    for (const [name, social] of Object.entries(
      state.collective?.by_species ?? {},
    )) {
      if (!social) continue;
      let panel = this.collectivePanels.get(name);
      const averages = { animal: social, tick: state.collective.tick };
      if (!panel) {
        panel = collectivePanel(
          social,
          species[name],
          state.tick,
          state.collective.schedule ?? 'continuous',
          averages,
          name,
        );
        this.collectivePanels.set(name, panel);
        $('collective-grid').append(panel);
      }
      panel.hidden = Boolean(chosen && chosen !== name);
      if (!panel.hidden) {
        panel.update(averages);
        visible = true;
      }
    }
    $('collective-section').hidden = !visible;
  }
  async update(context) {
    this.context = context;
    if (!context || !$('metrics-dialog').open) return;
    const { state, id, species } = context,
      request = ++this.request,
      key = `${id}:${state.run}`;
    if (this.key !== key) {
      this.key = key;
      this.points = [];
      this.draw();
      $('dashboard-species').replaceChildren(element('option', 'All species'));
      $('dashboard-species').firstChild.value = '';
      for (const name of Object.keys(species)) {
        const option = element('option', name);
        option.value = name;
        $('dashboard-species').append(option);
      }
      const actions = [
        ...new Set(
          Object.values(species).flatMap((spec) => spec.allowed_actions),
        ),
      ];
      $('dashboard-action').replaceChildren(
        ...actions.map((action) => {
          const option = element('option', action.replaceAll('_', ' '));
          option.value = action;
          return option;
        }),
      );
      $('dashboard-action').value = actions.includes('rest')
        ? 'rest'
        : actions[0];
    }
    $('dashboard-position').textContent =
      `Run ${state.run} · Step ${state.tick.toLocaleString()} · Frame ${state.frame_index + 1}`;
    $('dashboard-timeline').max = state.frame_count - 1;
    $('dashboard-timeline').value = state.frame_index;
    $('dashboard-play').textContent = context.playing
      ? 'Pause'
      : 'Play recording';
    // While an older prefix loads, resize/theme/filter redraws must not reuse
    // telemetry from after the newly selected playback step.
    if (this.points.at(-1)?.tick > state.tick) {
      this.points = this.points.filter((point) => point.tick <= state.tick);
      this.draw();
    }
    this.drawMovement();
    this.drawCollective();
    $('dashboard-status').textContent = 'Loading per-step history…';
    try {
      const points = await this.cache.history(
        id,
        state.run,
        state.frame_index,
        state.frame_count,
        Number($('dashboard-window').value),
      );
      if (request !== this.request || !$('metrics-dialog').open) return;
      this.points = points;
      this.draw();
      $('dashboard-status').textContent =
        `${points.length} recorded frames · Steps ${points[0].tick}–${points.at(-1).tick} · No future frames shown`;
    } catch (error) {
      if (request === this.request) {
        this.points = [];
        this.draw();
        $('dashboard-status').textContent =
          `Metrics unavailable: ${error.message}`;
      }
    }
  }
  draw() {
    this.drawMovement();
    this.drawCollective();
    if (!$('metrics-dialog').open || !this.context) return;
    const selected = $('dashboard-species').value,
      names = Object.keys(this.context.species).filter(
        (name) => !selected || name === selected,
      );
    const viewKey = `${this.context.id}:${this.context.state.run}`;
    if (this.cardsKey !== viewKey) {
      this.cardsKey = viewKey;
      this.speciesCards.clear();
      this.legendItems.clear();
    }
    const legend = names.map((name) => {
      let entry = this.legendItems.get(name);
      if (!entry) {
        const item = element('span', '', 'legend-key'),
          marker = element('span', '●', 'legend-symbol');
        item.append(marker, element('span', name));
        entry = { item, marker };
        this.legendItems.set(name, entry);
      }
      entry.marker.style.color = this.color(name);
      return entry.item;
    });
    replaceIfChanged($('dashboard-legend'), legend);
    const point = this.points.at(-1),
      cards = $('species-now');
    const visibleCards = [];
    for (const name of names) {
      const s = point?.species[name] ?? { groups: {} };
      if (
        !point?.species[name] &&
        !Object.hasOwn(this.context.state.movement_by_species ?? {}, name)
      )
        continue;
      const rows = [
        ['Population', number(s.population)],
        ['Health / energy', `${percent(s.health)} / ${percent(s.energy)}`],
        [
          'Births / deaths this step',
          `${number(s.births)} / ${number(s.deaths)}`,
        ],
        [
          'Kills / attempts this step',
          `${number(s.kills)} / ${number(s.attempts)}`,
        ],
        [
          'Groups / solitary',
          `${number(s.groups.clusters)} / ${number(s.groups.solitary)}`,
        ],
        [
          'Mean age / generation',
          `${number(s.mean_age)} / ${number(s.mean_generation)}`,
        ],
      ];
      let card = this.speciesCards.get(name);
      if (!card) {
        const node = element('article', undefined, 'species-card'),
          heading = element('strong', name, 'series-heading'),
          values = [],
          base = [heading];
        for (const [label] of rows) {
          const row = element('div'),
            value = element('b', '');
          values.push(value);
          row.append(element('span', label), value);
          base.push(row);
        }
        node.append(...base);
        card = { node, heading, values, base, movement: null };
        this.speciesCards.set(name, card);
      }
      card.heading.style.borderLeftColor = this.color(name);
      rows.forEach(([, value], i) => {
        if (card.values[i].textContent !== value)
          card.values[i].textContent = value;
      });
      const movement = !selected
        ? (this.speciesMovement(name)?.element ?? null)
        : null;
      if (card.movement !== movement) {
        card.movement = movement;
        card.node.replaceChildren(
          ...card.base,
          ...(movement ? [movement] : []),
        );
      }
      visibleCards.push(card.node);
    }
    replaceIfChanged(cards, visibleCards);
    const lineCache = new Map();
    this.charts.forEach((chart, i) => {
      const type = this.chartTypes[i],
        definition = metrics[type],
        action = $('dashboard-action').value;
      if (!lineCache.has(type))
        lineCache.set(type, series(this.points, names, type, action));
      const lines = lineCache.get(type);
      const suffix = definition.action
        ? ` · ${action.replaceAll('_', ' ')}`
        : '';
      chart.canvas.setAttribute(
        'aria-label',
        `${definition.label}${suffix} over recorded simulation steps`,
      );
      this.plot(chart.canvas, lines, definition);
      const current = lines
        .map((line) => {
          const last = line.points.at(-1);
          return `${line.name}: ${definition.percent ? percent(last?.value) : number(last?.value)}`;
        })
        .join(' · ');
      chart.summary.textContent =
        (current || 'No recorded values') +
        suffix +
        (definition.sampled ? ' · dots = recorded diagnostic samples' : '');
    });
  }
  plot(canvas, lines, definition) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const width = rect.width,
      height = rect.height,
      dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr))
      canvas.width = Math.round(width * dpr);
    if (canvas.height !== Math.round(height * dpr))
      canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const style = getComputedStyle(canvas);
    ctx.font = `12px ${style.fontFamily}`;
    ctx.textBaseline = 'middle';
    const left = 53,
      right = 12,
      top = 14,
      bottom = 24,
      w = Math.max(1, width - left - right),
      h = Math.max(1, height - top - bottom);
    let min = 0,
      max = definition.percent ? 1 : 0,
      hasValues = false;
    for (const line of lines)
      for (const point of line.points)
        if (Number.isFinite(point.value)) {
          hasValues = true;
          min = Math.min(min, point.value);
          max = Math.max(max, point.value);
        }
    if (max === min) max = min + 1;
    const first = this.points[0]?.tick ?? 0,
      last = this.points.at(-1)?.tick ?? 1,
      span = Math.max(1, last - first);
    const x = (tick) => left + (w * (tick - first)) / span,
      y = (value) => top + h * (1 - (value - min) / (max - min));
    ctx.fillStyle = style.getPropertyValue('--muted').trim() || '#8b9cad';
    ctx.textAlign = 'right';
    const label = (value) =>
      definition.percent ? `${Math.round(value * 100)}%` : number(value);
    ctx.fillText(label(max), left - 6, top);
    ctx.fillText(label(min), left - 6, top + h);
    ctx.textAlign = 'left';
    ctx.fillText(String(first), left, height - 8);
    ctx.textAlign = 'right';
    ctx.fillText(String(last), width - right, height - 8);
    for (let i = 0; i < 3; i++) {
      ctx.strokeStyle = style.getPropertyValue('--border').trim() || '#263341';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, top + (h * i) / 2);
      ctx.lineTo(width - right, top + (h * i) / 2);
      ctx.stroke();
    }
    for (const line of lines) {
      ctx.strokeStyle = this.color(line.name);
      ctx.fillStyle = this.color(line.name);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let open = false;
      for (const p of line.points) {
        if (!Number.isFinite(p.value)) {
          open = false;
          continue;
        }
        if (!definition.sampled) {
          if (!open || !p.consecutive) ctx.moveTo(x(p.tick), y(p.value));
          else ctx.lineTo(x(p.tick), y(p.value));
          open = true;
        }
      }
      ctx.stroke();
      // Dots show actual samples, including isolated points; do not interpolate diagnostics.
      for (const p of line.points)
        if (
          Number.isFinite(p.value) &&
          (definition.sampled || line.points.length < 3)
        ) {
          ctx.beginPath();
          ctx.arc(x(p.tick), y(p.value), 2.3, 0, 2 * Math.PI);
          ctx.fill();
        }
    }
    if (!hasValues) {
      ctx.fillStyle = style.getPropertyValue('--muted').trim() || '#8b9cad';
      ctx.textAlign = 'center';
      ctx.fillText('No samples in this time window', left + w / 2, top + h / 2);
    }
  }
}
