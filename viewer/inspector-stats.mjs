import { visualColor } from './theme.mjs';
import { circadianBody } from './day-night.mjs';
// Read recorded inputs only. No inference, model weights or simulation updates.
const WIDTHS = { 1: 20, 2: 32, 3: 42, 4: 45 };
export const TARGET_COLORS = {
  food: visualColor('#a2c76b'),
  prey: visualColor('#7de2bd'),
  predator: visualColor('#ff9a77'),
  mate: visualColor('#c6a5ff'),
};
export const ACTION_COLORS = [
  visualColor('#8ac8ff'),
  visualColor('#c6a5ff'),
  visualColor('#ffbd75'),
  visualColor('#a1c8b0'),
  visualColor('#7de2bd'),
  visualColor('#ff9aab'),
  visualColor('#e1d17c'),
  visualColor('#79d6df'),
];

export function observationView(features, template) {
  if (!Array.isArray(features) || !features.every(Number.isFinite)) return null;
  const version = template
    ? (template.observation_version ?? 1)
    : { 20: 1, 32: 2, 42: 3, 45: 4 }[features.length];
  const k = template?.observation_top_k ?? 1;
  if (
    !WIDTHS[version] ||
    !Number.isInteger(k) ||
    k < 1 ||
    k > 16 ||
    features.length !== WIDTHS[version] + 12 * (k - 1)
  )
    return null;
  let offset = 8;
  const targets = [];
  for (const category of ['food', 'prey', 'predator'])
    for (let slot = 1; slot <= k; slot++) {
      const [dx, dy, distance, present] = features.slice(offset, offset + 4);
      offset += 4;
      if (present > 0) targets.push({ category, slot, dx, dy, distance });
    }
  const result = {
    version,
    k,
    inputs: features.length,
    energy: features[0],
    health: features[1],
    heading: features.slice(2, 4),
    blocked: features.slice(4, 8),
    targets,
  };
  if (version >= 2) {
    result.terrain = {
      depth: features[offset] * 10,
      flow: features.slice(offset + 1, offset + 3).map((v) => v * 4),
      elevation: features[offset + 3] * 100,
      neighbors: Array.from({ length: 4 }, (_, i) => ({
        depth: features[offset + 4 + 2 * i] * 10,
        climb: features[offset + 5 + 2 * i] * 100,
      })),
    };
    offset += 12;
  }
  if (version >= 3) {
    const [
      sex,
      maturity,
      cooldown,
      ready,
      dx,
      dy,
      distance,
      present,
      mateReady,
      pair,
    ] = features.slice(offset);
    result.mating = {
      sex: sex === 1 ? 'male' : sex === -1 ? 'female' : 'unknown',
      maturity,
      cooldown,
      ready,
      mateReady,
      pair,
      present,
    };
    if (present > 0)
      targets.push({ category: 'mate', slot: 1, dx, dy, distance });
    if (version >= 4) {
      const [selfIntent, mateIntent, consentingPair] = features.slice(
        offset + 10,
      );
      Object.assign(result.mating, { selfIntent, mateIntent, consentingPair });
    }
  }
  return result;
}

export function predictionView(diagnostic) {
  const policy = Boolean(diagnostic?.action_probabilities);
  const raw = policy
    ? diagnostic.action_probabilities
    : diagnostic?.action_values;
  const actions = Object.entries(raw ?? {})
    .filter(([, v]) => Number.isFinite(v))
    .map(([action, value], i) => ({
      action,
      value,
      color: ACTION_COLORS[i % ACTION_COLORS.length],
      masked: diagnostic.action_mask?.[action] === false,
    }));
  if (!actions.length) return null;
  const min = policy ? 0 : Math.min(0, ...actions.map((a) => a.value));
  const max = policy ? 1 : Math.max(0, ...actions.map((a) => a.value));
  const span = max - min || 1,
    zero = (100 * (0 - min)) / span,
    best = Math.max(...actions.map((a) => a.value));
  return {
    policy,
    min,
    max,
    zero,
    actions: actions.map((a) => ({
      ...a,
      best: a.value === best,
      left: (100 * (Math.min(0, a.value) - min)) / span,
      width: (100 * Math.abs(a.value)) / span,
    })),
  };
}

function hasPredictions(diagnostic) {
  const scores = diagnostic.action_probabilities || diagnostic.action_values;
  return Object.values(scores ?? {}).some(Number.isFinite);
}

export function predictionSamples(frames, local, animalId, limit = 64) {
  const samples = [];
  for (
    let i = Math.min(local, frames.length - 1);
    i >= 0 && samples.length < limit;
    i--
  ) {
    const frame = frames[i],
      diagnostic = frame.diagnostics?.[String(animalId)];
    if (
      diagnostic &&
      diagnostic.tick === frame.tick &&
      hasPredictions(diagnostic)
    )
      samples.push({ tick: frame.tick, diagnostic });
  }
  return samples.reverse();
}

function ray(start, end, blocked, water) {
  let [x, y] = start;
  const [tx, ty] = end,
    dx = Math.abs(tx - x),
    dy = -Math.abs(ty - y),
    sx = x < tx ? 1 : -1,
    sy = y < ty ? 1 : -1;
  let error = dx + dy,
    wet = 0;
  while (x !== tx || y !== ty) {
    const e2 = 2 * error;
    if (e2 >= dy) {
      error += dy;
      x += sx;
    }
    if (e2 <= dx) {
      error += dx;
      y += sy;
    }
    if (blocked.has(`${x},${y}`)) return { visible: false, wet };
    if ((x !== tx || y !== ty) && water(x, y)) wet++;
  }
  return { visible: true, wet };
}

export function visibilityCells(
  state,
  animal,
  terrain,
  body = circadianBody(
    state.species[animal.species],
    state.day_night,
    state.tick,
  ),
) {
  const base = body?.vision_radius;
  if (!Number.isFinite(base) || base < 1) return [];
  const bonus = terrain?.hasWater ? (body.water_vision_bonus ?? 0) : 0,
    max = base + bonus;
  const blocked = new Set((state.obstacles ?? []).map(([x, y]) => `${x},${y}`));
  const wet = (x, y) =>
    Boolean(terrain?.cells.get(y * terrain.width + x)?.depth);
  const cells = [];
  for (
    let y = Math.max(0, animal.y - max);
    y <= Math.min(state.height - 1, animal.y + max);
    y++
  )
    for (
      let x = Math.max(0, animal.x - max);
      x <= Math.min(state.width - 1, animal.x + max);
      x++
    ) {
      const squared = (x - animal.x) ** 2 + (y - animal.y) ** 2;
      if (squared > max * max) continue;
      const sight = ray([animal.x, animal.y], [x, y], blocked, wet),
        allowed = base + Math.min(bonus, sight.wet);
      if (sight.visible && squared <= allowed * allowed) cells.push([x, y]);
    }
  return cells;
}

export class ObservationOverlay {
  draw(
    ctx,
    state,
    animal,
    terrain,
    cw,
    ch,
    light = false,
    body = animal
      ? circadianBody(
          state.species[animal.species],
          state.day_night,
          state.tick,
        )
      : null,
  ) {
    if (!animal) return;
    const key = [
      animal.id,
      animal.x,
      animal.y,
      body?.vision_radius,
      body?.water_vision_bonus,
      state.width,
      state.height,
      state.obstacles,
      terrain,
    ];
    if (!this.key || key.some((v, i) => v !== this.key[i])) {
      this.cells = visibilityCells(state, animal, terrain, body);
      this.key = key;
    }
    ctx.fillStyle = light ? '#2d78b82e' : '#8ac8ff22';
    for (const [x, y] of this.cells)
      ctx.fillRect(
        x * cw + 1,
        y * ch + 1,
        Math.max(1, cw - 2),
        Math.max(1, ch - 2),
      );
  }
  clear() {
    this.key = null;
    this.cells = null;
  }
}

// Retain only one selected agent's tiny score vectors, never frames/features.
export class PredictionHistory {
  update(key, samples, tick) {
    if (key !== this.key) {
      this.clear();
      this.key = key;
    }
    const points = new Map(
      (this.samples ?? [])
        .filter((p) => p.tick <= tick)
        .map((p) => [p.tick, p]),
    );
    for (const sample of samples) {
      if (sample.tick > tick) continue;
      const d = sample.diagnostic,
        policy = Boolean(d.action_probabilities);
      const diagnostic = policy
        ? { action_probabilities: { ...d.action_probabilities } }
        : { action_values: { ...d.action_values } };
      points.set(sample.tick, { tick: sample.tick, diagnostic });
    }
    this.samples = [...points.values()]
      .sort((a, b) => a.tick - b.tick)
      .slice(-64);
    return this.samples;
  }
  clear() {
    this.samples = [];
    this.key = null;
  }
}
