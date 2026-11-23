import { drawFeedings } from './feeding-highlights.mjs';
import {
  drawEnvironmentObjects,
  drawEnvironmentShadows,
} from './environment-objects.mjs';
import { trackingSample, drawVisualTracks } from './tracking.mjs';
import { drawContests, contestLabel } from './competition.mjs';
import { installTheme, boardPalette } from './theme.mjs';
import {
  daylight,
  circadianBody,
  drawDaylight,
  drawSunlitGround,
  DayNightClock,
} from './day-night.mjs';
import { DEFAULT_NETWORK_RANK, rankLimit } from './network-structure.mjs';
import { api } from './api.mjs';
import { Dashboard } from './dashboard.mjs';
import { Playback } from './playback.mjs';
import { behaviorRows, movementHeatmap } from './behavior-stats.mjs';
import { speciesStyles } from './species-style.mjs';
import { terrainView, drawTerrain } from './terrain.mjs';
import { inspectorGraphs } from './inspector.mjs';
import { BrainExplanation } from './explainability.mjs';
import { AnimalBrainView } from './brain-activations.mjs';
import {
  ObservationOverlay,
  PredictionHistory,
  predictionSamples,
} from './inspector-stats.mjs';
const $ = (id) => document.getElementById(id);
let state,
  selected = null,
  playing = false,
  busy = false;
let styledSpecies = null,
  styles = new Map(),
  terrainDefinition = null,
  terrain = null;
let errorTimer,
  analysisKey = '',
  analysisRequest = 0,
  runMenuKey = '';
function showError(error) {
  $('error').textContent = error.message;
  $('error').hidden = false;
  clearTimeout(errorTimer);
  errorTimer = setTimeout(() => ($('error').hidden = true), 12000);
}

let networkRankLimit = DEFAULT_NETWORK_RANK;
const player = new Playback(api, undefined, { movement: true });
const reducedMotion = window.matchMedia?.(
  '(prefers-reduced-motion: reduce)',
) ?? { matches: false };
reducedMotion.addEventListener?.('change', () => {
  if (state) render();
});
const dayNightClock = new DayNightClock($('day-night-clock'));
const observationOverlay = new ObservationOverlay();
const brainExplanation = new BrainExplanation(api);
const animalBrain = new AnimalBrainView();
for (const id of ['network-rank', 'brain-rank', 'explain-rank']) {
  const update = () => {
    const next = rankLimit($(id).value);
    for (const key of ['network-rank', 'brain-rank', 'explain-rank'])
      $(key).value = String(next);
    if (next === networkRankLimit) return;
    networkRankLimit = next;
    brainExplanation.setRankLimit(next);
    inspectorVisuals = null;
    if (state) render();
  };
  $(id).onchange = update;
  $(id).oninput = () => {
    const value = $(id).value;
    if (
      value.trim() !== '' &&
      Number.isInteger(Number(value)) &&
      Number(value) >= 0 &&
      Number(value) <= 12
    )
      update();
  };
}
$('brain-open').onclick = () => {
  if (!busy && state && brainContext().animal) animalBrain.open(brainContext());
};
$('brain-play').onclick = () => $('play').onclick();
$('brain-step').onclick = () => $('step').onclick();
$('brain-back').onclick = () => $('back').onclick();
$('explain-play').onclick = () => $('play').onclick();
$('explain-step').onclick = () => $('step').onclick();
$('explain-back').onclick = () => $('back').onclick();
brainExplanation.onPin = () => pause();
function brainContext() {
  const animal = state?.animals.find((a) => a.id === selected);
  return {
    id: player.id,
    run: state?.run,
    tick: state?.tick,
    animal,
    diagnostic: state?.selected,
    rankLimit: networkRankLimit,
    template: animal
      ? player.metadata.config.brains?.[state.species[animal.species].brain]
      : null,
    playing,
    done: state?.done,
  };
}
$('inspect-animal').onclick = () => {
  if (busy) return;
  const value = $('inspect-animal-id').value.trim(),
    id = Number(value);
  if (
    !value ||
    !Number.isInteger(id) ||
    !state?.animals.some((a) => a.id === id)
  ) {
    showError(
      new Error('Enter the ID of an animal alive at this replay frame.'),
    );
    return;
  }
  selected = id;
  render();
};
let sensoryCache = null;
let heatmapView = null;
function selectedTracking() {
  const animal = state?.animals.find((a) => a.id === selected);
  const diagnostic = state?.selected;
  if (!animal || !diagnostic) return null;
  const template =
    player.metadata.config.brains?.[state.species[animal.species].brain];
  if (
    sensoryCache?.diagnostic === diagnostic &&
    sensoryCache.animal === selected &&
    sensoryCache.frames === player.frames &&
    sensoryCache.template === template
  )
    return sensoryCache.sample;
  const body = circadianBody(
    state.species[animal.species],
    state.day_night,
    diagnostic.tick,
  );
  const radius =
    body.vision_radius +
    (terrain?.hasWater ? (body.water_vision_bonus ?? 0) : 0);
  const sample = trackingSample(
    player.frames,
    state.frame_index % player.metadata.recording.frames_per_chunk,
    selected,
    diagnostic,
    template,
    radius,
  );
  sensoryCache = {
    diagnostic,
    animal: selected,
    frames: player.frames,
    template,
    sample,
  };
  return sample;
}
function drawSelectedTracking() {
  if (!$('show-tracking').checked) return;
  const sample = selectedTracking();
  if (!sample) return;
  const canvas = $('world');
  drawVisualTracks(
    canvas.getContext('2d'),
    sample,
    canvas.width / state.width,
    canvas.height / state.height,
    document.documentElement.dataset.theme === 'light',
  );
}
$('tracking-open').onclick = () => {
  if (!busy)
    $('inspector')
      .querySelector('.tracking-panel')
      ?.scrollIntoView({ block: 'start', behavior: 'auto' });
};
$('movement-open').onclick = () => {
  if (!busy)
    heatmapView?.element.scrollIntoView({ block: 'start', behavior: 'auto' });
};
$('show-tracking').onchange = () => {
  if (state) render();
};
$('explain-open').onclick = () => {
  if (busy) return;
  if (!state || selected === null || !state.selected?.features) return;
  brainExplanation.open({
    id: player.id,
    run: state.run,
    frame: state.frame_index,
    animal: selected,
    diagnosticTick: state.selected.tick,
    tick: state.tick,
    playing,
    done: state.done,
  });
};
const inspectorSections = {};
const inspectorHistory = new PredictionHistory();
let inspectorVisuals = null,
  inspectorVisualKey = '',
  inspectorDiagnostic = null;
function pause() {
  playing = false;
  $('play').textContent = 'Play recording';
  $('brain-play').textContent = 'Play replay';
  $('explain-play').textContent = 'Play replay';
  $('dashboard-play').textContent = 'Play recording';
}
function setRecordingLoading(active) {
  for (const theme of ['light', 'dark'])
    $(`brand-source-${theme}`).setAttribute(
      'srcset',
      `/branding/wtf-symbol-${active ? 'motion-' : ''}${theme}.svg`,
    );
  $('world').setAttribute('aria-busy', String(active));
}
async function operation(fn, { recording = false } = {}) {
  if (busy) return;
  busy = true;
  const previousConnection = $('connection').textContent;
  if (recording) {
    setRecordingLoading(true);
    $('connection').textContent = 'Loading recording…';
  }
  for (const id of [
    'step',
    'back',
    'load',
    'refresh',
    'recorded-run',
    'timeline',
    'previous-run',
    'next-run',
    'dashboard-back',
    'dashboard-next',
    'dashboard-timeline',
    'inspect-animal',
    'inspect-animal-id',
    'brain-open',
    'explain-open',
    'tracking-open',
    'show-tracking',
  ])
    $(id).disabled = true;
  try {
    await fn();
  } catch (e) {
    pause();
    showError(e);
  } finally {
    if (recording) {
      setRecordingLoading(false);
      if ($('connection').textContent === 'Loading recording…')
        $('connection').textContent = previousConnection;
    }
    busy = false;
    for (const id of [
      'step',
      'back',
      'load',
      'refresh',
      'recorded-run',
      'timeline',
      'previous-run',
      'next-run',
      'dashboard-back',
      'dashboard-next',
      'dashboard-timeline',
      'inspect-animal',
      'inspect-animal-id',
      'brain-open',
      'explain-open',
      'tracking-open',
      'show-tracking',
    ])
      $(id).disabled = false;
    const alive = state?.animals.some((a) => a.id === selected);
    $('inspect-animal').disabled = !state?.animals.length;
    $('brain-open').disabled = !alive;
    $('explain-open').disabled = !alive || !state?.selected?.features;
    const senses = selectedTracking();
    $('tracking-open').disabled = !alive || !senses;
    $('show-tracking').disabled = !alive || !senses?.steps;
    updateRunButtons();
  }
}
function speciesColor(name) {
  return styles.get(name)?.color ?? '#c6d3df';
}
function animalColor(animal) {
  const style = styles.get(animal.species);
  return style?.[animal.sex] ?? style?.color ?? '#c6d3df';
}
function syncSpeciesLegend() {
  if (styledSpecies === state.species) return;
  styledSpecies = state.species;
  styles = speciesStyles(state.species);
  const nodes = [];
  for (const [name, style] of styles) {
    for (const [sex, symbol] of [
      ['female', '♀'],
      ['male', '♂'],
    ]) {
      const item = text('span', '', 'legend-key'),
        marker = text('span', style.symbol, 'legend-symbol');
      marker.style.color = style[sex];
      item.append(marker, text('span', `${name} ${symbol}`));
      item.title = `${name} · ${sex}${sex === 'male' ? ' · light outline' : ''}`;
      item.setAttribute('aria-label', `${name} ${sex}`);
      nodes.push(item);
    }
  }
  $('species-legend').replaceChildren(...nodes);
}
function syncTerrain() {
  if (terrainDefinition === (state.terrain ?? null)) return;
  terrainDefinition = state.terrain ?? null;
  terrain = terrainView(terrainDefinition);
  const legend = $('terrain-legend');
  legend.hidden = !terrain;
  legend.textContent = terrain
    ? `${terrain.name}${terrain.hasWater ? ' · ≈ River → current' : ''}${terrain.hasElevation ? ' · ▧ Elevation' : ''}`
    : '';
}
const dashboard = new Dashboard(api, speciesColor);
function fitWorld() {
  const canvas = $('world'),
    holder = canvas.parentElement;
  if (!holder.clientWidth || !holder.clientHeight) return;
  const ratio = canvas.width / canvas.height;
  const width = Math.min(holder.clientWidth, holder.clientHeight * ratio);
  // Match the element's actual bounds to the drawn board so click-to-inspect
  // stays accurate when a wide window leaves space beside the smaller board.
  const cssWidth = `${width}px`,
    cssHeight = `${width / ratio}px`;
  if (canvas.style.width !== cssWidth) canvas.style.width = cssWidth;
  if (canvas.style.height !== cssHeight) canvas.style.height = cssHeight;
}
new ResizeObserver(fitWorld).observe($('world').parentElement);
function drawWorld() {
  const canvas = $('world'),
    ctx = canvas.getContext('2d');
  const height = Math.round((canvas.width * state.height) / state.width);
  if (canvas.height !== height) {
    canvas.height = height;
    fitWorld();
  } else if (!canvas.style.width) fitWorld();
  const cw = canvas.width / state.width,
    ch = canvas.height / state.height;
  const palette = boardPalette(document.documentElement.dataset.theme);
  ctx.fillStyle = palette.background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawTerrain(ctx, terrain, cw, ch, palette.light, {
    tick: state.tick,
    reducedMotion: reducedMotion.matches,
  });
  const sun = daylight(state.day_night, state.tick);
  drawSunlitGround(
    ctx,
    state.day_night,
    state.tick,
    canvas.width,
    canvas.height,
    sun,
  );
  drawEnvironmentShadows(ctx, state, cw, ch, sun, {
    light: palette.light,
    reducedMotion: reducedMotion.matches,
  });
  dayNightClock.update(state.day_night, state.tick, sun);
  const cycleStatus = $('day-night-status');
  if (cycleStatus) {
    if (cycleStatus.hidden !== !sun.enabled) cycleStatus.hidden = !sun.enabled;
    if (sun.enabled) {
      const label = `${sun.label} · sunlight ${Math.round(sun.light * 100)}% · cycle ${Math.floor(state.tick / state.day_night.cycle_steps) + 1}`;
      if (cycleStatus.textContent !== label) cycleStatus.textContent = label;
    }
  }
  const observed = state.animals.find((a) => a.id === selected),
    sight = $('observation-status');
  if (observed && $('show-observation').checked) {
    const body = circadianBody(
        state.species[observed.species],
        state.day_night,
        state.tick,
        sun,
      ),
      bonus = terrain?.hasWater ? (body?.water_vision_bonus ?? 0) : 0;
    observationOverlay.draw(
      ctx,
      state,
      observed,
      terrain,
      cw,
      ch,
      palette.light,
      body,
    );
    sight.hidden = false;
    sight.textContent = `Current sight #${observed.id} · radius ${body?.vision_radius ?? '?'}${bonus ? ` + up to ${bonus} across water` : ''}`;
  } else {
    observationOverlay.clear();
    sight.hidden = true;
  }
  ctx.strokeStyle = palette.grid;
  ctx.lineWidth = 0.6;
  for (let x = 0; x <= state.width; x++) {
    ctx.beginPath();
    ctx.moveTo(x * cw, 0);
    ctx.lineTo(x * cw, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= state.height; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y * ch);
    ctx.lineTo(canvas.width, y * ch);
    ctx.stroke();
  }
  drawEnvironmentObjects(
    ctx,
    state.resources,
    state.obstacles,
    state.resource_capacity,
    cw,
    ch,
    {
      tick: state.tick,
      light: palette.light,
      reducedMotion: reducedMotion.matches,
    },
  );
  for (const a of state.animals) {
    const x = (a.x + 0.5) * cw,
      y = (a.y + 0.5) * ch,
      r = cw * 0.31;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate((a.heading * Math.PI) / 2);
    ctx.fillStyle = animalColor(a);
    ctx.beginPath();
    if (state.species[a.species].diet === 'prey') {
      ctx.moveTo(0, -r * 1.2);
      ctx.lineTo(r, r);
      ctx.lineTo(-r, r);
      ctx.closePath();
    } else ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    if (palette.light) {
      ctx.strokeStyle = '#304c52';
      ctx.lineWidth = a.sex === 'male' ? 2.8 : 1;
      ctx.stroke();
    }
    if (a.sex === 'male') {
      ctx.strokeStyle = '#edf5ff';
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
    ctx.fillStyle = '#13212a';
    ctx.beginPath();
    ctx.arc(0, -r * 0.55, 1.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    if (a.id === selected) {
      ctx.strokeStyle =
        palette.light && sun.light > 0.25 ? palette.selection : '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, r + 5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  drawDaylight(
    ctx,
    state.day_night,
    state.tick,
    canvas.width,
    canvas.height,
    sun,
  );
}
function drawFeedingHighlights() {
  const canvas = $('world');
  drawFeedings(
    canvas.getContext('2d'),
    state.feeding_highlights,
    canvas.width / state.width,
    canvas.height / state.height,
    {
      light: document.documentElement.dataset.theme === 'light',
      reducedMotion: reducedMotion.matches,
    },
  );
}
function drawContestHighlights() {
  const canvas = $('world');
  drawContests(
    canvas.getContext('2d'),
    state.contest_highlights,
    canvas.width / state.width,
    canvas.height / state.height,
    document.documentElement.dataset.theme === 'dark',
  );
}
function drawDeathHighlights() {
  const canvas = $('world'),
    ctx = canvas.getContext('2d');
  const cw = canvas.width / state.width,
    ch = canvas.height / state.height;
  const light = document.documentElement.dataset.theme === 'light';
  const markers = [
    ...state.natural_death_highlights.map((e) => ({ ...e, natural: true })),
    ...state.kill_highlights,
  ];
  for (const kill of markers) {
    if (!kill.position) continue;
    const x = (kill.position[0] + 0.5) * cw,
      y = (kill.position[1] + 0.5) * ch;
    const radius = cw * (0.65 + kill.age * 0.035);
    ctx.save();
    ctx.globalAlpha = Math.max(0.25, 1 - kill.age / 14);
    const preyDeath =
      kill.natural && state.species[kill.species]?.diet === 'resource';
    const color = kill.natural
      ? preyDeath
        ? light
          ? '#19714f'
          : '#7de2bd'
        : light
          ? '#846118'
          : '#f1c36b'
      : light
        ? '#b52b42'
        : '#ff5268';
    ctx.fillStyle = kill.natural
      ? preyDeath
        ? '#7de2bd25'
        : '#f1c36b25'
      : '#ff4f6425';
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = light
      ? color
      : kill.natural
        ? preyDeath
          ? '#d4fff0'
          : '#fff0cb'
        : '#ffe3df';
    ctx.lineWidth = 3;
    const cross = cw * 0.22;
    ctx.beginPath();
    ctx.moveTo(x - cross, y - cross);
    ctx.lineTo(x + cross, y + cross);
    ctx.moveTo(x + cross, y - cross);
    ctx.lineTo(x - cross, y + cross);
    ctx.stroke();
    ctx.font = 'bold 15px system-ui';
    ctx.textAlign = 'center';
    const labelX = Math.max(30, Math.min(canvas.width - 30, x)),
      labelY = y < 50 ? y + radius + 19 : y - radius - 8;
    ctx.fillStyle = light
      ? kill.natural
        ? preyDeath
          ? '#e8f5ee'
          : '#fff3d6'
        : '#fff0f2'
      : '#1c1019';
    ctx.fillRect(labelX - 25, labelY - 16, 50, 21);
    ctx.fillStyle = light
      ? color
      : kill.natural
        ? preyDeath
          ? '#a4f2d6'
          : '#f7d99e'
        : '#ffadad';
    ctx.fillText(kill.natural ? 'DIED' : 'KILL', labelX, labelY);
    const hunter = kill.natural
      ? null
      : state.animals.find((a) => a.id === kill.actor);
    if (hunter) {
      ctx.strokeStyle = light ? '#846118' : '#ffd18a';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(
        (hunter.x + 0.5) * cw,
        (hunter.y + 0.5) * ch,
        cw * 0.48,
        0,
        Math.PI * 2,
      );
      ctx.stroke();
    }
    ctx.restore();
  }
}
function drawBirthHighlights() {
  const canvas = $('world'),
    ctx = canvas.getContext('2d');
  const cw = canvas.width / state.width,
    ch = canvas.height / state.height;
  const light = document.documentElement.dataset.theme === 'light';
  for (const birth of state.birth_highlights) {
    if (!birth.position) continue;
    const child = state.animals.find((a) => a.id === birth.target);
    const x = ((child?.x ?? birth.position[0]) + 0.5) * cw,
      y = ((child?.y ?? birth.position[1]) + 0.5) * ch;
    const radius = cw * (0.52 + birth.age * 0.035);
    ctx.save();
    ctx.globalAlpha = Math.max(0.25, 1 - birth.age / 14);
    ctx.strokeStyle = light ? '#176b95' : '#6edbff';
    ctx.fillStyle = light ? '#176b9520' : '#6edbff20';
    ctx.lineWidth = 3;
    ctx.shadowColor = light ? '#176b95' : '#6edbff';
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    // A plus sign and a word distinguish births from crossed death markers.
    const plusX = x + cw * 0.38,
      plusY = y - cw * 0.38,
      size = cw * 0.13;
    ctx.strokeStyle = light ? '#145574' : '#d9f6ff';
    ctx.beginPath();
    ctx.moveTo(plusX - size, plusY);
    ctx.lineTo(plusX + size, plusY);
    ctx.moveTo(plusX, plusY - size);
    ctx.lineTo(plusX, plusY + size);
    ctx.stroke();
    const labelX = Math.max(38, Math.min(canvas.width - 38, x));
    const labelY = y < 50 ? y + radius + 19 : y - radius - 8;
    ctx.font = 'bold 15px system-ui';
    ctx.textAlign = 'center';
    ctx.fillStyle = light ? '#e8f4fa' : '#102632';
    ctx.fillRect(labelX - 35, labelY - 16, 70, 21);
    ctx.fillStyle = light ? '#145574' : '#b6eeff';
    ctx.fillText('BORN', labelX, labelY);
    ctx.restore();
  }
}
function drawChart() {
  const c = $('chart'),
    rect = c.getBoundingClientRect();
  if (!state || !rect.width || !rect.height) return;
  const width = rect.width,
    height = rect.height,
    dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * dpr),
    pixelHeight = Math.round(height * dpr);
  if (c.width !== pixelWidth) c.width = pixelWidth;
  if (c.height !== pixelHeight) c.height = pixelHeight;
  const ctx = c.getContext('2d'),
    style = getComputedStyle(c),
    history = state.history;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  // Draw in CSS pixels so labels match surrounding text, without stretching
  // the old fixed-size bitmap into the compact chart's aspect ratio.
  ctx.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  ctx.fillStyle = getComputedStyle(document.documentElement)
    .getPropertyValue('--muted')
    .trim();
  ctx.textBaseline = 'middle';
  const popMax = Math.max(
    1,
    ...history.flatMap((m) => Object.values(m.populations)),
  );
  const foodMax = Math.max(1, ...history.map((m) => m.food)),
    timeMax = Math.max(1, state.tick);
  const popLabel = popMax.toLocaleString(),
    foodLabel = Math.round(foodMax).toLocaleString();
  const left = Math.max(30, ctx.measureText(popLabel).width + 10);
  const right = Math.max(36, ctx.measureText(foodLabel).width + 10),
    top = 12,
    bottom = 23;
  const w = Math.max(1, width - left - right),
    h = Math.max(1, height - top - bottom);
  ctx.textAlign = 'right';
  ctx.fillText(popLabel, left - 8, top);
  ctx.fillText('0', left - 8, top + h);
  ctx.textAlign = 'left';
  ctx.fillText(foodLabel, left + w + 8, top);
  ctx.fillText('0', left, height - 7);
  ctx.textAlign = 'right';
  ctx.fillText(state.tick.toLocaleString(), left + w, height - 7);
  for (let n = 0; n <= 3; n++) {
    const y = top + (h * n) / 3;
    ctx.strokeStyle = style.getPropertyValue('--border').trim() || '#263341';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + w, y);
    ctx.stroke();
  }
  const series = [
    ...Object.keys(state.species).map((s) => ({
      color: speciesColor(s),
      max: popMax,
      get: (m) => m.populations[s],
    })),
    { color: '#548c63', max: foodMax, get: (m) => m.food },
  ];
  for (const seriesItem of series) {
    ctx.strokeStyle = seriesItem.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    history.forEach((m, i) => {
      const x = left + (w * m.tick) / timeMax,
        y = top + h * (1 - seriesItem.get(m) / seriesItem.max);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}
new ResizeObserver(() => {
  if (state) drawChart();
}).observe($('chart'));
function text(tag, value, cls) {
  const el = document.createElement(tag);
  el.textContent = value;
  if (cls) el.className = cls;
  return el;
}
function drawInspector() {
  const box = $('inspector');
  box.replaceChildren();
  const a = state.animals.find((a) => a.id === selected),
    b = state.selected;
  animalBrain.update(brainContext());
  brainExplanation.update({
    id: player.id,
    run: state.run,
    frame: state.frame_index,
    animal: a?.id ?? null,
    diagnosticTick: b?.tick ?? null,
    tick: state.tick,
    playing,
    done: state.done,
  });
  $('brain-open').disabled = !a;
  $('inspect-animal').disabled = !state.animals.length;
  $('explain-open').disabled = !a || !b?.features;
  $('agent-label').textContent = a ? `${a.species} #${a.id}` : 'No selection';
  $('inspect-hint').hidden = Boolean(a);
  $('show-observation').disabled = !a;
  const sensory = selectedTracking();
  $('tracking-open').disabled = busy || !sensory;
  $('show-tracking').disabled = busy || !sensory?.steps;
  $('tracking-status').textContent = sensory
    ? `Sample ${sensory.tick} · body ${state.tick}. Board: selected animal's observed tracks only. Scent stays in the charts. Updates at recorded samples.`
    : a
      ? 'This recording has no supported sampled tracking inputs for this animal.'
      : 'Select an animal to inspect its recorded senses.';
  if (!a) {
    inspectorVisuals = null;
    inspectorDiagnostic = null;
    inspectorHistory.clear();
    return;
  }
  box.append(
    text(
      'p',
      `${a.sex ?? 'Sex not recorded'} · Generation ${a.generation ?? 0} · ${a.parent_id >= 0 ? `Parents #${a.parent_id}${a.second_parent_id >= 0 ? ` + #${a.second_parent_id}` : ''}` : 'Founding animal'}`,
      'muted',
    ),
  );
  if (a.mutation_count > 0)
    box.append(
      text(
        'p',
        `${a.mutated ? `Mutated at birth: ${a.mutation_weights} weights` : 'Inherited mutated lineage'} · Lineage mutations ${a.mutation_count}`,
        'muted',
      ),
    );
  if (terrain) {
    const t = terrain.cells.get(a.y * terrain.width + a.x);
    box.append(
      text(
        'p',
        `Elevation ${t?.elevation ?? 0}${t?.depth ? ` · Water depth ${t.depth} · Flow (${t.flow.join(', ')}) cells/step` : ''}`,
        'muted',
      ),
    );
  }
  if (!b) {
    inspectorVisuals = null;
    inspectorDiagnostic = null;
    inspectorHistory.clear();
    box.append(
      text(
        'p',
        `Energy ${a.energy.toFixed(1)} · Health ${a.health.toFixed(1)} · Age ${a.age} · Action ${a.last_action} · Reward ${a.last_reward.toFixed(3)}`,
        'muted',
      ),
    );
    box.append(
      text('p', 'Brain diagnostics were not recorded for this frame.', 'muted'),
    );
    return;
  }
  box.append(
    text(
      'p',
      `Brain diagnostics recorded at step ${b.tick}; body state is current.`,
      'muted',
    ),
  );
  const policy = Boolean(b.action_probabilities);
  const algorithm =
    {
      ppo: 'PPO',
      a2c: 'A2C',
      reinforce: 'REINFORCE',
      grpo: 'GRPO (experimental)',
      recurrent_ppo: 'Recurrent PPO (GRU)',
      lstm_ppo: 'Recurrent PPO (LSTM)',
      discrete_sac: 'Discrete SAC',
      trpo: 'TRPO',
      qrdqn: 'QR-DQN',
      dqn: 'DQN',
      ddqn: 'Double DQN',
      maskable_ppo: 'Maskable PPO',
      mappo: 'MAPPO (experimental)',
    }[b.algorithm] || 'Q-learning';
  const details = text('div', '', 'details');
  const fields = [
    ['Brain', algorithm],
    ['Energy', a.energy.toFixed(1)],
    ['Health', a.health.toFixed(1)],
    ['Age', a.age],
    ['Last reward', a.last_reward.toFixed(3)],
    ['Training updates', b.updates],
    ['Last action', a.last_action],
  ];
  if (state.day_night?.enabled) {
    const body = circadianBody(
      state.species[a.species],
      state.day_night,
      state.tick,
    );
    fields.push(
      ['Current sight radius', body.vision_radius],
      ['Current living cost', body.living_cost.toFixed(3)],
      ['Current movement cost', body.movement_cost.toFixed(3)],
    );
  }
  if (Number.isFinite(a.mating_intent_until_tick)) {
    const remaining = Math.max(0, a.mating_intent_until_tick - state.tick + 1);
    fields.push([
      'Mating intent',
      remaining ? `${remaining} steps remaining` : 'Inactive',
    ]);
  }
  if (a.last_mating_failure)
    fields.push([
      'Last mating failure',
      a.last_mating_failure.replaceAll('_', ' '),
    ]);
  if (Number.isFinite(a.kills))
    fields.push(
      ['Lifetime kills', a.kills],
      ['Food energy gained', a.energy_gained.toFixed(1)],
    );
  if (policy) {
    fields.push(['Policy entropy', b.entropy?.toFixed?.(3) ?? '—']);
    if (Number.isFinite(b.state_value))
      fields.push(['State value', b.state_value.toFixed(3)]);
    if (Number.isFinite(b.return_baseline))
      fields.push(['Return baseline', b.return_baseline.toFixed(3)]);
    if (b.buffer_kind === 'group') {
      fields.push(
        ['Trial groups', b.rollout_updates],
        ['Trials per group', b.group_size],
        ['Trial horizon', b.rollout_horizon],
      );
      if (Number.isFinite(b.training_metrics?.return_std))
        fields.push([
          'Group reward spread',
          b.training_metrics.return_std.toFixed(3),
        ]);
    } else
      fields.push([
        b.buffer_kind === 'episode'
          ? 'Episode buffer'
          : b.buffer_kind === 'sequence'
            ? 'Sequence buffer'
            : 'Rollout buffer',
        b.buffer_size ?? b.rollout_size,
      ]);
  } else
    fields.push(
      ['Exploration', b.epsilon?.toFixed?.(3) ?? '—'],
      ['Own experiences', b.replay_size],
    );
  if (Number.isFinite(a.reproduction_due_age) && a.reproduction_due_age >= 0)
    fields.push(['Reproduction eligible at age', a.reproduction_due_age]);
  if (Number.isFinite(b.memory_norm))
    fields.push(['Memory norm', b.memory_norm.toFixed(3)]);
  if (b.reward_scope)
    fields.push(['Reward scope', b.reward_scope.replaceAll('_', ' ')]);
  for (const [label, value] of fields) {
    const cell = text('div', '');
    cell.append(text('span', label), text('strong', String(value)));
    details.append(cell);
  }
  const visualKey = `${player.id}:${state.run}:${selected}:${player.chunkIndex}:${b.tick}`;
  if (
    visualKey !== inspectorVisualKey ||
    b !== inspectorDiagnostic ||
    !inspectorVisuals
  ) {
    inspectorVisualKey = visualKey;
    inspectorDiagnostic = b;
    inspectorVisuals = inspectorGraphs({
      consequenceSchedule:
        player.metadata.config.consequence_learning?.update_schedule ??
        'continuous',
      diagnostic: b,
      template: player.metadata.config.brains?.[state.species[a.species].brain],
      spec: {
        ...state.species[a.species],
        body: circadianBody(state.species[a.species], state.day_night, b.tick),
      },
      hasWater: terrain?.hasWater ?? false,
      sections: inspectorSections,
      rank: networkRankLimit,
      samples: inspectorHistory.update(
        `${player.id}:${state.run}:${selected}`,
        predictionSamples(
          player.frames,
          state.frame_index % player.metadata.recording.frames_per_chunk,
          selected,
        ),
        state.tick,
      ),
    });
  }
  box.append(inspectorVisuals);
  const bodyDetails = text('details', '', 'feature-details');
  bodyDetails.open = Boolean(inspectorSections.body);
  bodyDetails.ontoggle = () => {
    inspectorSections.body = bodyDetails.open;
  };
  bodyDetails.append(
    text('summary', 'Current body & training details'),
    details,
  );
  box.append(bodyDetails);
  const features = text('details', '', 'feature-details');
  features.open = Boolean(inspectorSections.raw);
  features.ontoggle = () => {
    inspectorSections.raw = features.open;
  };
  features.append(
    text('summary', `Raw observation inputs (${b.features?.length ?? 0})`),
  );
  features.append(
    text(
      'pre',
      Array.isArray(b.features)
        ? JSON.stringify(
            b.features.map((v) =>
              Number.isFinite(v) ? Number(v.toFixed(3)) : v,
            ),
            null,
            2,
          )
        : 'Not recorded',
    ),
  );
  box.append(features);
}
async function loadBehavior() {
  const key = `${player.id}:${state.run}:${player.run.complete}`;
  if (key === analysisKey) return;
  analysisKey = key;
  const request = ++analysisRequest,
    box = $('behavior');
  box.replaceChildren(text('p', 'Analyzing saved frames…'));
  heatmapView = movementHeatmap(player.movement.map, player.run.obstacles);
  $('movement-open').disabled = !heatmapView;
  if (heatmapView) {
    heatmapView.draw(document.documentElement.dataset.theme);
    box.append(heatmapView.element);
  }
  try {
    const report = await api(
      `/api/recordings/${encodeURIComponent(player.id)}/runs/${state.run}/analysis`,
    );
    if (request !== analysisRequest) return;
    const run = report.runs[0];
    box.replaceChildren();
    const coverage = run.coverage.step_fraction;
    box.append(
      text(
        'p',
        `Steps ${run.first_tick}–${run.last_tick} · ${run.complete ? 'Complete' : 'Partial'} run · Consecutive-step coverage ${coverage === null ? '—' : `${(coverage * 100).toFixed(1)}%`}`,
      ),
    );
    $('movement-open').disabled = !heatmapView;
    if (heatmapView) {
      heatmapView.draw(document.documentElement.dataset.theme);
      box.append(heatmapView.element);
    } else
      box.append(
        text(
          'p',
          'Movement heatmap unavailable from this server. Restart the playback server after updating Python.',
          'muted',
        ),
      );
    for (const species of Object.keys(run.species)) {
      const section = text('div', '', 'behavior-species');
      section.append(
        text(
          'strong',
          `${species} · ${run.species[species].learning.algorithm}`,
        ),
      );
      const table = document.createElement('table');
      table.className = 'behavior-table';
      const heading = document.createElement('tr');
      for (const label of ['Metric', 'Early', 'Late'])
        heading.append(text('th', label));
      const head = document.createElement('thead');
      head.append(heading);
      table.append(head);
      const body = document.createElement('tbody');
      for (const row of behaviorRows(run, species)) {
        const tr = document.createElement('tr');
        for (const value of row) tr.append(text('td', value));
        body.append(tr);
      }
      table.append(body);
      section.append(table);
      const stats = run.species[species],
        learning = stats.learning;
      section.append(
        text(
          'p',
          `Observed optimizer updates: ≥${learning.observed_optimizer_updates} · Mean age at death: ${stats.mean_completed_lifespan === null ? '—' : stats.mean_completed_lifespan.toFixed(0)} steps · ${stats.survivors} survivors (excluded from lifespan mean).`,
        ),
      );
      const budget = stats.energy_budget;
      section.append(
        text(
          'p',
          `Food supplied ${budget.food_input_share === null ? '—' : `${(budget.food_input_share * 100).toFixed(1)}%`} of recorded energy inputs. Birth grants added ${budget.net_reproduction_energy.toFixed(0)} net energy after parent costs.`,
        ),
      );
      if (learning.algorithm === 'grpo') {
        section.append(
          text(
            'p',
            `New trial groups observed: ${learning.observed_grpo_groups} · Equal-return groups: ${learning.grpo_zero_variance_fraction === null ? '—' : `${(learning.grpo_zero_variance_fraction * 100).toFixed(1)}%`} of ${learning.classified_grpo_groups} classified. Equal returns produce no GRPO update.`,
          ),
        );
      }
      box.append(section);
    }
    box.append(
      text(
        'p',
        `Early: ${run.windows[0].start_tick}–${run.windows[0].end_tick}; late: ${run.windows.at(-1).start_tick}–${run.windows.at(-1).end_tick}. Policy entropy of 100% means uniform probabilities. Updates exclude inherited counter baselines. Successful actions are reconstructed where possible on death steps. Compare frozen policies on held-out seeds to establish learning.`,
      ),
    );
  } catch (error) {
    if (request === analysisRequest)
      box.replaceChildren(text('p', `Analysis unavailable: ${error.message}`));
  }
}
function updateRunButtons() {
  const index = state
    ? player.runs.findIndex((run) => run.run === state.run)
    : -1;
  $('previous-run').disabled = busy || index <= 0;
  $('next-run').disabled = busy || index < 0 || index >= player.runs.length - 1;
}
function syncRunMenu() {
  const key = `${player.id}:${player.runs.map((run) => `${run.run}:${run.frame_count}:${run.complete}`).join(',')}`;
  if (key !== runMenuKey) {
    runMenuKey = key;
    $('recorded-run').replaceChildren(
      ...player.runs.map((run) => {
        const option = text(
          'option',
          `Run ${run.run} / ${player.metadata.config.experiment?.runs ?? player.runs.length}${run.complete === false ? ' · partial' : ''}`,
        );
        option.value = run.run;
        option.title = `Seed ${run.seed} · ${run.frame_count} frames`;
        return option;
      }),
    );
  }
  $('recorded-run').value = state.run;
  updateRunButtons();
}
async function changeRunBy(offset) {
  pause();
  await operation(
    async () => {
      const index = player.runs.findIndex((run) => run.run === state?.run),
        target = player.runs[index + offset];
      if (!target) return;
      selected = null;
      render(await player.changeRun(target.run));
    },
    { recording: true },
  );
}
function render(snapshot = null) {
  const previousRun = state?.run;
  state = snapshot ?? player.snapshot(selected);
  if (previousRun !== undefined && previousRun !== state.run) {
    selected = null;
    state = player.snapshot();
  }
  syncRunMenu();
  syncSpeciesLegend();
  syncTerrain();
  dashboard.update({ state, id: player.id, species: state.species, playing });
  $('tick').textContent = state.tick.toLocaleString();
  $('run').textContent = `${state.run} / ${state.total_runs}`;
  let prey = 0,
    predators = 0;
  for (const [name, s] of Object.entries(state.species)) {
    if (s.diet === 'resource') prey += state.metrics.populations[name];
    else predators += state.metrics.populations[name];
  }
  $('prey').textContent = prey;
  $('predator').textContent = predators;
  for (const [id, diet] of [
    ['prey', 'resource'],
    ['predator', 'prey'],
  ]) {
    $(id).title = Object.entries(state.species)
      .filter(([, spec]) => spec.diet === diet)
      .map(([name]) => `${name}: ${state.metrics.populations[name] ?? 0}`)
      .join(' · ');
  }
  $('food').textContent = Math.round(state.metrics.food).toLocaleString();
  $('updates').textContent = state.total_updates.toLocaleString();
  const killCount = Object.values(state.metrics.kills).reduce(
    (a, b) => a + b,
    0,
  );
  $('kills').textContent = killCount.toLocaleString();
  $('births').textContent = Object.values(state.metrics.births ?? {})
    .reduce((a, b) => a + b, 0)
    .toLocaleString();
  $('births').parentElement.classList.toggle(
    'birth-now',
    state.birth_highlights.some((b) => b.frame_index === state.frame_index),
  );
  const latestBirth = state.birth_highlights.at(-1);
  $('birth-banner').hidden = !latestBirth;
  if (latestBirth) {
    const count = state.birth_highlights.filter(
      (b) => b.frame_index === latestBirth.frame_index,
    ).length;
    const species = latestBirth.species ?? 'animal';
    const message = `BORN · ${species} #${latestBirth.target} · Parents #${latestBirth.actor}${latestBirth.mate >= 0 ? ` + #${latestBirth.mate}` : ''} · Step ${latestBirth.tick}${count > 1 ? ` · ${count} births shown` : ''}`;
    if ($('birth-banner').textContent !== message)
      $('birth-banner').textContent = message;
  }
  $('kills').parentElement.classList.toggle(
    'kill-now',
    state.kill_highlights.some((k) => k.frame_index === state.frame_index),
  );
  const contest = state.contest_highlights.at(-1);
  $('contest-banner').hidden = !contest;
  if (contest)
    $('contest-banner').textContent =
      `CONTEST · #${contest.actor} beat #${contest.target} · ${contestLabel(contest.cause)} · ${contest.amount} damage · Step ${contest.tick}`;
  const latestEvent = [
    ...state.natural_death_highlights.map((e) => ({ ...e, natural: true })),
    ...state.kill_highlights,
  ]
    .sort((a, b) => a.frame_index - b.frame_index)
    .at(-1);
  $('kill-banner').hidden = !latestEvent;
  $('kill-banner').classList.toggle(
    'natural-death',
    Boolean(latestEvent?.natural),
  );
  $('kill-banner').classList.toggle(
    'prey-natural-death',
    Boolean(
      latestEvent?.natural &&
        state.species[latestEvent.species]?.diet === 'resource',
    ),
  );
  const species = latestEvent?.species
    ? `${latestEvent.species[0].toUpperCase()}${latestEvent.species.slice(1)}`
    : 'Animal';
  const banner = !latestEvent
    ? ''
    : latestEvent.natural
      ? `${latestEvent.cause === 'competition' ? 'COMPETITION DEATH' : 'NATURAL DEATH'} · ${species} #${latestEvent.actor} · ${latestEvent.cause} · Step ${latestEvent.tick}`
      : `KILL · ${latestEvent.attacker_species ?? 'Predator'} #${latestEvent.actor} → prey #${latestEvent.target} · Step ${latestEvent.tick}`;
  if ($('kill-banner').textContent !== banner)
    $('kill-banner').textContent = banner;
  $('connection').textContent = state.waiting
    ? 'Waiting for more recorded frames or runs'
    : `Recorded playback · ${state.mode}`;
  $('empty').hidden = !state.done;
  $('empty').textContent =
    'End of recording. Seek backward or load another saved experiment.';
  $('outcomes').textContent =
    `Births: ${Object.values(state.metrics.births ?? {}).reduce((a, b) => a + b, 0)} · Starvation: ${state.metrics.deaths.starvation || 0} · Predation: ${state.metrics.deaths.predation || 0}${state.terrain ? ` · Drowning: ${state.metrics.deaths.drowning ?? 0}` : ''} · Hunts: ${Object.values(state.metrics.kills).reduce((a, b) => a + b, 0)}`;
  $('recorded-run').value = state.run;
  $('timeline').max = state.frame_count - 1;
  $('timeline').value = state.frame_index;
  $('frame-label').textContent =
    `Frame ${state.frame_index + 1} / ${state.frame_count}`;
  $('summaries').textContent =
    `${player.frames.length} frames loaded · one chunk in memory`;
  loadBehavior();
  heatmapView?.update(
    player.movement.map,
    document.documentElement.dataset.theme,
  );
  drawWorld();
  drawSelectedTracking();
  drawFeedingHighlights();
  drawBirthHighlights();
  drawDeathHighlights();
  drawContestHighlights();
  drawChart();
  if (state.done) pause();
  drawInspector();
}
async function openRecording() {
  const id = $('recording').value;
  if (!id) return;
  selected = null;
  analysisKey = '';
  await player.open(id);
  runMenuKey = '';
  $('config').value = JSON.stringify(player.metadata.config, null, 2);
  $('play').disabled = false;
  $('metrics-open').disabled = false;
  render();
}
async function refreshCatalog() {
  const records = await api('/api/recordings'),
    previous = $('recording').value;
  $('recording').replaceChildren(
    ...records.map((record) => {
      const option = text(
        'option',
        `${record.name} · ${record.id} · ${record.status}`,
      );
      option.value = record.id;
      return option;
    }),
  );
  if (records.some((r) => r.id === previous)) $('recording').value = previous;
  if (!records.length) {
    $('connection').textContent = 'No recordings yet';
    $('empty').hidden = false;
    $('empty').textContent =
      'Run Python from the terminal to create a recording, then refresh this list.';
    return;
  }
  await openRecording();
}
async function loop() {
  if (!playing) return;
  await operation(async () => {
    render(await player.advance(selected));
  });
  if (playing) setTimeout(loop, 1000 / Number($('speed').value));
}
$('play').onclick = () => {
  if (playing) {
    pause();
    return;
  }
  if (busy || !state || state.done) return;
  playing = true;
  $('play').textContent = 'Pause';
  loop();
};
$('step').onclick = () => {
  pause();
  if (state)
    operation(async () => {
      render(await player.advance(selected));
    });
};
$('back').onclick = () => {
  pause();
  if (state)
    operation(async () => {
      render(await player.seek(Math.max(0, player.frameIndex - 1), selected));
    });
};
$('load').onclick = () => {
  pause();
  operation(openRecording, { recording: true });
};
$('refresh').onclick = () => {
  pause();
  operation(refreshCatalog, { recording: true });
};
$('previous-run').onclick = () => {
  if (state) changeRunBy(-1);
};
$('next-run').onclick = () => {
  if (state) changeRunBy(1);
};
$('recorded-run').onchange = () => {
  pause();
  if (state)
    operation(
      async () => {
        selected = null;
        render(await player.changeRun($('recorded-run').value, selected));
      },
      { recording: true },
    );
};
$('timeline').onchange = () => {
  pause();
  if (state)
    operation(async () => {
      render(await player.seek(Number($('timeline').value), selected));
    });
};
$('speed').oninput = () => {
  $('speed-value').textContent = $('speed').value;
};
$('world').onclick = (event) => {
  if (busy || !state) return;
  const rect = $('world').getBoundingClientRect(),
    x = Math.floor(((event.clientX - rect.left) / rect.width) * state.width),
    y = Math.floor(((event.clientY - rect.top) / rect.height) * state.height);
  selected = state.animals.find((a) => a.x === x && a.y === y)?.id ?? null;
  render();
};
$('dashboard-play').onclick = () => {
  $('play').click();
  $('dashboard-play').textContent = playing ? 'Pause' : 'Play recording';
};
$('dashboard-next').onclick = () => $('step').click();
$('dashboard-back').onclick = () => $('back').click();
$('dashboard-timeline').onchange = () => {
  pause();
  if (state)
    operation(async () =>
      render(
        await player.seek(Number($('dashboard-timeline').value), selected),
      ),
    );
};
let themeStorage;
try {
  themeStorage = window.localStorage;
} catch {
  /* Embedded browsers can deny storage access. */
}
installTheme({
  root: document.documentElement,
  button: $('theme-toggle'),
  storage: themeStorage,
  onChange() {
    if (state) {
      drawWorld();
      drawSelectedTracking();
      drawFeedingHighlights();
      drawBirthHighlights();
      drawDeathHighlights();
      drawContestHighlights();
      drawChart();
    }
    heatmapView?.draw(document.documentElement.dataset.theme);
    if ($('metrics-dialog').open) dashboard.draw();
  },
});
operation(refreshCatalog, { recording: true });

$('show-observation').onchange = () => {
  if (state) render();
};
