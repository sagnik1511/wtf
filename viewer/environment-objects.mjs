import { shadowState } from './day-night.mjs';
// Cosmetic only: recorded quantity drives plant size; saved ticks drive motion.
// No timers, mutable history, random draws, images or extra replay fields.
const palettes = {
  dark: {
    soil: '#263341',
    stem: '#a2c76b',
    leaf: '#63ab6f',
    tip: '#a1c8b0',
    rock: '#3a4b5c',
    face: '#61798f',
    edge: '#8b9cad',
  },
  light: {
    soil: '#b7c8d7',
    stem: '#5c741a',
    leaf: '#34784a',
    tip: '#416d55',
    rock: '#6a7d89',
    face: '#617385',
    edge: '#667788',
  },
};

export function objectMotion(tick, x, y, reducedMotion = false) {
  if (reducedMotion) return { sway: 0, shimmer: 0.16 };
  const phase = (Number.isFinite(tick) ? tick : 0) * 0.11 + x * 1.73 + y * 2.41;
  return {
    sway: Math.sin(phase) * 0.035,
    shimmer: 0.16 + 0.08 * Math.sin(phase * 0.55),
  };
}

function polygon(ctx, points, fill) {
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.moveTo(...points[0]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(...points[i]);
  ctx.closePath();
  ctx.fill();
}

function leaf(ctx, x, y, direction, size) {
  ctx.moveTo(x, y);
  ctx.quadraticCurveTo(
    x + direction * size * 0.2,
    y - size,
    x + direction * size,
    y - size * 0.6,
  );
  ctx.quadraticCurveTo(x + direction * size * 0.8, y + size * 0.3, x, y);
}

export function drawEnvironmentObjects(
  ctx,
  resources,
  obstacles,
  capacity,
  cw,
  ch,
  { tick = 0, light = false, reducedMotion = false } = {},
) {
  if (!(cw > 0 && ch > 0)) return;
  const colors = light ? palettes.light : palettes.dark;
  ctx.save();
  // Rocks keep their footprint: only the surface highlight changes with time.
  for (const [x, y] of obstacles) {
    const left = x * cw,
      top = y * ch;
    const point = (a, b) => [left + a * cw, top + b * ch];
    ctx.fillStyle = colors.soil;
    ctx.fillRect(left + cw * 0.04, top + ch * 0.04, cw * 0.92, ch * 0.92);
    polygon(
      ctx,
      [
        point(0.12, 0.3),
        point(0.32, 0.12),
        point(0.76, 0.14),
        point(0.91, 0.4),
        point(0.85, 0.85),
        point(0.26, 0.9),
        point(0.1, 0.66),
      ],
      colors.rock,
    );
    polygon(
      ctx,
      [
        point(0.12, 0.3),
        point(0.32, 0.12),
        point(0.76, 0.14),
        point(0.64, 0.46),
        point(0.33, 0.56),
      ],
      colors.face,
    );
    ctx.globalAlpha = objectMotion(tick, x, y, reducedMotion).shimmer;
    polygon(
      ctx,
      [
        point(0.33, 0.56),
        point(0.64, 0.46),
        point(0.76, 0.14),
        point(0.91, 0.4),
        point(0.85, 0.85),
      ],
      colors.edge,
    );
    ctx.globalAlpha = 1;
    ctx.strokeStyle = colors.edge;
    ctx.lineWidth = Math.max(0.55, Math.min(cw, ch) * 0.025);
    ctx.beginPath();
    ctx.moveTo(...point(0.32, 0.13));
    ctx.lineTo(...point(0.64, 0.46));
    ctx.lineTo(...point(0.55, 0.72));
    ctx.stroke();
  }
  for (const resource of resources) {
    const fraction =
      Number.isFinite(capacity) &&
      capacity > 0 &&
      Number.isFinite(resource.quantity)
        ? Math.max(0, Math.min(1, resource.quantity / capacity))
        : 0;
    const x = (resource.x + 0.5) * cw,
      base = (resource.y + 0.84) * ch;
    const growth = Math.sqrt(fraction),
      height = ch * (0.12 + 0.57 * growth);
    const sway =
      fraction > 0
        ? objectMotion(tick, resource.x, resource.y, reducedMotion).sway *
          cw *
          growth
        : 0;
    ctx.fillStyle = colors.soil;
    ctx.beginPath();
    ctx.arc(x, base, Math.min(cw, ch) * (0.08 + 0.1 * growth), 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = fraction > 0 ? colors.stem : colors.face;
    ctx.lineWidth = Math.max(0.7, Math.min(cw, ch) * 0.045);
    ctx.beginPath();
    ctx.moveTo(x, base);
    ctx.quadraticCurveTo(
      x + sway * 0.3,
      base - height * 0.5,
      x + sway,
      base - height,
    );
    ctx.stroke();
    if (!fraction) continue;
    const pairs = 1 + Math.floor(fraction * 2),
      size = cw * (0.09 + 0.16 * growth);
    for (const direction of [-1, 1]) {
      ctx.fillStyle = direction < 0 ? colors.leaf : colors.tip;
      ctx.beginPath();
      for (let i = 0; i < pairs; i++) {
        const t = (i + 1) / (pairs + 1);
        leaf(
          ctx,
          x + sway * t,
          base - height * t,
          direction,
          size * (1 - 0.2 * i),
        );
      }
      ctx.fill();
    }
  }
  ctx.restore();
}

// One filled path for all silhouettes avoids per-object filters and compositing.
// Draw below grid, observation overlays and glyphs; shadows never imply occlusion.
export function drawEnvironmentShadows(
  ctx,
  state,
  cw,
  ch,
  illumination,
  { light = false, reducedMotion = false } = {},
) {
  const shadow = shadowState(illumination, reducedMotion);
  if (!shadow || !(cw > 0 && ch > 0)) return;
  ctx.save();
  ctx.fillStyle = light ? '#304c52' : '#03091c';
  ctx.globalAlpha = shadow.alpha;
  ctx.beginPath();
  const cast = (x, y, width, height) => {
    const dx = shadow.dx * height * cw,
      dy = shadow.dy * height * ch;
    const px = x * cw,
      py = y * ch,
      r = width * cw;
    ctx.moveTo(px - r, py);
    ctx.lineTo(px + r, py);
    ctx.lineTo(px + dx + r * 0.8, py + dy);
    ctx.lineTo(px + dx + r * 0.45, py + dy + ch * 0.08);
    ctx.lineTo(px + dx - r * 0.6, py + dy + ch * 0.08);
    ctx.lineTo(px + dx - r, py + dy);
    ctx.closePath();
  };
  for (const [x, y] of state.obstacles) cast(x + 0.5, y + 0.7, 0.32, 1);
  const capacity = state.resource_capacity;
  for (const resource of state.resources) {
    if (
      !(capacity > 0) ||
      !Number.isFinite(resource.quantity) ||
      !(resource.quantity > 0)
    )
      continue;
    const growth = Math.sqrt(Math.min(1, resource.quantity / capacity));
    cast(
      resource.x + 0.5,
      resource.y + 0.84,
      0.07 + growth * 0.12,
      growth * 0.8,
    );
  }
  for (const animal of state.animals)
    cast(animal.x + 0.5, animal.y + 0.65, 0.2, 0.65);
  ctx.fill();
  ctx.restore();
}
