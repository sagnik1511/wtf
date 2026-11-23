// Derive lighting from saved configuration/tick; no per-frame payload or timer.
export function daylight(cycle, tick = 0) {
  if (!cycle?.enabled)
    return { enabled: false, light: 1, phase: 0.5, label: 'Day' };
  const phase =
    ((cycle.start_phase ?? 0.25) +
      (tick % cycle.cycle_steps) / cycle.cycle_steps) %
    1;
  const light =
    phase > 0.25 && phase < 0.75 ? Math.cos(2 * Math.PI * (phase - 0.5)) : 0;
  const label =
    phase < 0.25 || phase >= 0.75
      ? 'Night'
      : phase < 1 / 3
        ? 'Dawn'
        : phase < 0.45
          ? 'Morning'
          : phase <= 0.55
            ? 'Midday'
            : phase < 2 / 3
              ? 'Afternoon'
              : 'Dusk';
  return { enabled: true, light, phase, label };
}
// Decorative projection opposite the dial's sun; computed once per board draw.
// Reduced motion keeps a short, fixed footprint, while sunlight still fades it.
export function shadowState(illumination, reducedMotion = false) {
  if (!illumination?.enabled || !(illumination.light > 0)) return null;
  const strength = Math.max(0, Math.min(1, illumination.light));
  const reach = 0.12 + 0.62 * (1 - strength);
  return {
    dx: reducedMotion
      ? 0.12
      : Math.cos(2 * Math.PI * (illumination.phase - 0.25)) * reach,
    dy: reducedMotion ? 0.1 : 0.1 + strength * 0.06,
    alpha: (0.09 + 0.16 * strength) * Math.min(1, strength * 5),
  };
}

export function circadianBody(
  species,
  cycle,
  tick = 0,
  light = daylight(cycle, tick),
) {
  if (!light.enabled || !species?.circadian) return species?.body;
  const body = { ...species.body };
  for (const field of ['vision_radius', 'living_cost', 'movement_cost']) {
    const night = species.circadian[`night_${field}_multiplier`] ?? 1;
    const value = body[field] * (night + (1 - night) * light.light);
    body[field] =
      field === 'vision_radius'
        ? Math.max(1, Math.min(256, Math.floor(value + 0.5)))
        : value;
  }
  return body;
}
// Illuminate the ground before glyphs, so animals and event colors stay distinct.
export function drawSunlitGround(
  ctx,
  cycle,
  tick,
  width,
  height,
  illumination = daylight(cycle, tick),
) {
  const { enabled, light } = illumination;
  if (!enabled || light <= 0) return;
  ctx.fillStyle = `rgba(255,240,185,${0.28 * light})`;
  ctx.fillRect(0, 0, width, height);
}
export function drawDaylight(
  ctx,
  cycle,
  tick,
  width,
  height,
  illumination = daylight(cycle, tick),
) {
  const { enabled, light } = illumination;
  if (!enabled) return;
  ctx.fillStyle = `rgba(3,9,28,${0.48 * (1 - light)})`;
  ctx.fillRect(0, 0, width, height);
  if (light > 0 && light < 0.6) {
    ctx.fillStyle = `rgba(255,164,82,${0.09 * Math.sin((Math.PI * light) / 0.6)})`;
    ctx.fillRect(0, 0, width, height);
  }
}

// The marker follows one full orbit per simulated day. No browser animation timer.
export function clockState(cycle, tick, state = daylight(cycle, tick)) {
  const angle = 2 * Math.PI * state.phase;
  return {
    ...state,
    x: 34 - 24 * Math.sin(angle),
    y: 34 + 24 * Math.cos(angle),
    sun: state.phase >= 0.25 && state.phase < 0.75,
  };
}
export function clockMarkup(value = clockState(null, 0)) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 112 68" width="112" height="68" aria-hidden="true">
    <path d="M10 34 A24 24 0 0 1 58 34" fill="none" stroke="#e9c469" stroke-width="2"/>
    <path d="M58 34 A24 24 0 0 1 10 34" fill="none" stroke="#718cc4" stroke-width="2"/>
    <path d="M5 34 H63" stroke="#61798f" stroke-dasharray="2 3"/>
    <g data-marker="" transform="translate(${value.x} ${value.y})"><circle r="6" fill="${value.sun ? '#f6d365' : '#b9d4ff'}"/><circle data-crescent="" visibility="${value.sun ? 'hidden' : 'visible'}" cx="3" cy="-2" r="5" fill="#101a22"/></g>
    <text data-body="" x="70" y="38" fill="#e6edf3" font-family="system-ui,sans-serif" font-size="12">${value.sun ? 'Sun' : 'Moon'}</text>
  </svg>`;
}
export class DayNightClock {
  constructor(root) {
    this.root = root;
  }
  update(cycle, tick, illumination = daylight(cycle, tick)) {
    if (this.root.hidden !== !illumination.enabled)
      this.root.hidden = !illumination.enabled;
    if (!illumination.enabled || this.phase === illumination.phase) return;
    const value = clockState(cycle, tick, illumination);
    if (!this.marker) {
      this.root.innerHTML = clockMarkup(value);
      this.marker = this.root.querySelector('[data-marker]');
      this.body = this.root.querySelector('[data-body]');
      this.crescent = this.root.querySelector('[data-crescent]');
      this.disc = this.marker.querySelector('circle');
    }
    this.phase = value.phase;
    this.marker.setAttribute('transform', `translate(${value.x} ${value.y})`);
    this.disc.setAttribute('fill', value.sun ? '#f6d365' : '#b9d4ff');
    this.crescent.setAttribute('visibility', value.sun ? 'hidden' : 'visible');
    this.body.textContent = value.sun ? 'Sun' : 'Moon';
    this.root.setAttribute(
      'aria-label',
      `${value.label}, sunlight ${Math.round(value.light * 100)} percent`,
    );
  }
}
