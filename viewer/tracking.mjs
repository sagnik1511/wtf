import { decodeCompact } from './compact-observations.mjs';
import { semanticLayout, compactLayout } from './input-layout.mjs';
import { decodeSoft } from './soft-observations.mjs';
import {
  BASE_WIDTHS,
  BODY_SECTORS,
  worldVector,
  worldScent,
} from './observation-v5.mjs';
import { socialWidth } from './social.mjs';
// Decode only the selected agent's sampled inputs; never reconstruct hidden paths.
export const SCENT_SECTORS = [
  'here',
  'north',
  'northeast',
  'east',
  'southeast',
  'south',
  'southwest',
  'west',
  'northwest',
];
export const SCENT_OFFSETS = [
  [0, 0],
  [0, -1],
  [1, -1],
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
];

export function trackingLayout(template = {}) {
  const steps = template.movement_history_steps ?? 0,
    k = template.observation_top_k ?? 1,
    spec = template.scent_observation;
  if (
    !Number.isInteger(steps) ||
    steps < 0 ||
    steps > 16 ||
    !Number.isInteger(k) ||
    k < 1 ||
    k > 16
  )
    return null;
  const channels = [];
  if (spec !== undefined) {
    if (
      !spec ||
      !['separated', 'blended'].includes(spec.mode ?? 'separated') ||
      !Number.isInteger(spec.radius ?? 2) ||
      (spec.radius ?? 2) < 1 ||
      (spec.radius ?? 2) > 8
    )
      return null;
    if (spec.mode === 'blended') {
      if ('species' in spec || 'separate_sexes' in spec) return null;
      channels.push({ blended: true });
    } else {
      if (
        !Array.isArray(spec.species) ||
        spec.species.length < 1 ||
        spec.species.length > 8 ||
        new Set(spec.species).size !== spec.species.length ||
        !spec.species.every((n) => typeof n === 'string' && n.length) ||
        typeof (spec.separate_sexes ?? true) !== 'boolean'
      )
        return null;
      for (const species of spec.species)
        for (const sex of (spec.separate_sexes ?? true)
          ? ['female', 'male']
          : ['any'])
          channels.push({ species, sex });
    }
  }
  return {
    steps,
    k,
    channels,
    radius: spec?.radius ?? 2,
    width: (steps ? 2 * k * (7 + 3 * steps) : 0) + 9 * channels.length,
  };
}

export function trackingLabels(template) {
  const layout = trackingLayout(template);
  if (!layout) return [];
  const names = [];
  if (layout.steps)
    for (const category of ['Prey', 'Predator'])
      for (let slot = 1; slot <= layout.k; slot++) {
        const prefix = `${category} track ${slot}`;
        for (const part of [
          'last dx',
          'last dy',
          'velocity x',
          'velocity y',
          'age',
          'confidence',
          'visible',
        ])
          names.push(`${prefix} ${part}`);
        for (let lag = 1; lag <= layout.steps; lag++)
          for (const part of ['dx', 'dy', 'seen'])
            names.push(`${prefix} lag ${lag} ${part}`);
      }
  for (const { species, sex, blended } of layout.channels)
    for (const side of template.observation_version === 5
      ? BODY_SECTORS
      : SCENT_SECTORS)
      names.push(
        blended ? `Scent blended ${side}` : `Scent ${species} ${sex} ${side}`,
      );
  if (template.observation_version === 5)
    return names.map((n) =>
      n.startsWith('Prey track ') || n.startsWith('Predator track ')
        ? n
            .replace('last dx', 'last forward')
            .replace('last dy', 'last right')
            .replace('velocity x', 'velocity forward')
            .replace('velocity y', 'velocity right')
            .replace(' dx', ' forward')
            .replace(' dy', ' right')
        : n,
    );
  return names;
}

export function trackingView(features, template, offset) {
  if (template?.observation_version === 7) {
    try {
      const d = decodeCompact(features, compactLayout(template), template);
      const view = trackingView(
        d.features,
        { ...template, observation_version: 5 },
        48 + 12 * ((template.observation_top_k ?? 1) - 1),
      );
      if (view) {
        view.scent = [];
        view.summary = true;
      }
      return view;
    } catch {
      return null;
    }
  }

  const layout = trackingLayout(template);
  if (!layout) return null;
  const tracks = [];
  if (layout.steps)
    for (const category of ['prey', 'predator'])
      for (let slot = 1; slot <= layout.k; slot++) {
        let [dx, dy, vx, vy, age, confidence, visible] = features.slice(
          offset,
          offset + 7,
        );
        offset += 7;
        const history = [];
        for (let lag = 1; lag <= layout.steps; lag++) {
          let [x, y, seen] = features.slice(offset, offset + 3);
          offset += 3;
          if (template.observation_version === 5)
            [x, y] = worldVector(x, y, features.slice(2, 4));
          history.push({ lag, dx: x, dy: y, seen: seen > 0 });
        }
        if (template.observation_version === 5) {
          [dx, dy] = worldVector(dx, dy, features.slice(2, 4));
          [vx, vy] = worldVector(vx, vy, features.slice(2, 4));
        }
        if (confidence > 0)
          tracks.push({
            category,
            slot,
            dx,
            dy,
            vx,
            vy,
            age: Math.round(age * layout.steps),
            confidence,
            visible: visible > 0,
            history,
          });
      }
  const scent = layout.channels.map((channel) => {
    let strengths = features.slice(offset, offset + 9);
    if (template.observation_version === 5)
      strengths = worldScent(strengths, features.slice(2, 4));
    offset += 9;
    return { ...channel, strengths };
  });
  return { steps: layout.steps, tracks, scent, radius: layout.radius };
}

// Locate the observer at the diagnostic's time, never anchor old inputs to its
// current body or reconstruct targets from omniscient replay positions.
export function trackingSample(
  frames,
  localIndex,
  animalId,
  diagnostic,
  template,
  radius,
) {
  if (template?.observation_version === 7) {
    try {
      const d = decodeCompact(
        diagnostic.features,
        compactLayout(template),
        template,
      );
      const value = trackingSample(
        frames,
        localIndex,
        animalId,
        { ...diagnostic, features: d.features },
        { ...template, observation_version: 5 },
        radius,
      );
      if (value) {
        value.scent = [];
        value.summary = true;
      }
      return value;
    } catch {
      return null;
    }
  }
  if (template?.observation_version === 6) {
    try {
      const layout = semanticLayout(template);
      if (
        !Array.isArray(diagnostic?.features) ||
        diagnostic.features.length !== layout.width
      )
        return null;
      return trackingSample(
        frames,
        localIndex,
        animalId,
        { ...diagnostic, features: decodeSoft(diagnostic.features, layout) },
        { ...template, observation_version: 5 },
        radius,
      );
    } catch {
      return null;
    }
  }
  const layout = trackingLayout(template ?? {});
  const base =
    BASE_WIDTHS[template?.observation_version ?? 1] +
    12 * ((template?.observation_top_k ?? 1) - 1);
  if (
    !layout?.width ||
    !Array.isArray(diagnostic?.features) ||
    diagnostic.features.length !==
      base + layout.width + (socialWidth(template ?? {}) ?? 0) ||
    !diagnostic.features.every(Number.isFinite) ||
    !(radius > 0)
  )
    return null;
  for (let i = Math.min(localIndex, frames.length - 1); i >= 0; i--) {
    const frame = frames[i];
    if (frame.tick !== diagnostic.tick) continue;
    const origin = frame.animals.find((a) => a.id === animalId);
    return origin
      ? {
          tick: frame.tick,
          origin: { x: origin.x, y: origin.y },
          radius,
          ...trackingView(diagnostic.features, template, base),
        }
      : null;
  }
  return null;
}

export function drawVisualTracks(ctx, sample, cw, ch, light) {
  if (!sample || !sample.steps) return;
  const ox = (sample.origin.x + 0.5) * cw,
    oy = (sample.origin.y + 0.5) * ch;
  ctx.save();
  for (const t of sample.tracks) {
    const color =
      t.category === 'prey'
        ? light
          ? '#116448'
          : '#7de2bd'
        : light
          ? '#a33a25'
          : '#ff9a77';
    const x = (v) => ox + v * sample.radius * cw,
      y = (v) => oy + v * sample.radius * ch;
    const points = [
      ...t.history.filter((p) => p.seen).reverse(),
      { lag: t.age, dx: t.dx, dy: t.dy },
    ].sort((a, b) => b.lag - a.lag);
    ctx.globalAlpha = Math.max(0.4, t.confidence);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = Math.max(3, cw * 0.12);
    ctx.setLineDash([]);
    for (let i = 0; i < points.length; i++) {
      const p = points[i],
        prior = points[i - 1];
      if (prior && prior.lag - p.lag === 1) {
        ctx.beginPath();
        ctx.moveTo(x(prior.dx), y(prior.dy));
        ctx.lineTo(x(p.dx), y(p.dy));
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(x(p.dx), y(p.dy), Math.max(3, cw * 0.13), 0, Math.PI * 2);
      ctx.fill();
    }
    if (!t.visible) {
      ctx.beginPath();
      ctx.arc(x(t.dx), y(t.dy), Math.max(6, cw * 0.27), 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  ctx.strokeStyle = light ? '#172c3c' : '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(ox - 5, oy);
  ctx.lineTo(ox + 5, oy);
  ctx.moveTo(ox, oy - 5);
  ctx.lineTo(ox, oy + 5);
  ctx.stroke();
  ctx.restore();
}
