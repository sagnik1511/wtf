const percent = (value) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const number = (value) => (Number.isFinite(value) ? value.toFixed(3) : '—');

export function behaviorRows(run, species) {
  const early = run.windows[0]?.species[species],
    late = run.windows.at(-1)?.species[species];
  if (!early || !late) return [];
  const rows = [
    ['Mean energy', (s) => percent(s.mean_energy_fraction)],
    ['Energy coverage', (s) => percent(s.energy_coverage)],
    ['Starving time', (s) => percent(s.starving_fraction)],
    ['Movement success', (s) => percent(s.movement_success)],
    ['Reward / animal-step', (s) => number(s.reward_per_agent_step)],
    ['Action coverage', (s) => percent(s.action_coverage)],
  ];
  if (early.diet === 'resource') {
    rows.push(['Productive feeding', (s) => percent(s.feeding_success)]);
  }
  if (early.diet === 'prey') {
    rows.push(['Kills / attack', (s) => percent(s.hunt_success)]);
  }
  if (
    early.learning.normalized_entropy !== null ||
    late.learning.normalized_entropy !== null
  ) {
    rows.push([
      'Policy entropy',
      (s) => percent(s.learning.normalized_entropy),
    ]);
  }
  return rows.map(([label, get]) => [label, get(early), get(late)]);
}

const DARK_HEAT_COLORS = [
  [41, 65, 111],
  [41, 97, 147],
  [33, 155, 153],
  [157, 211, 140],
  [255, 244, 175],
];

// Increasing luminance keeps a busy dark-mode run from collapsing into one hue.
export function movementHeatColor(count, maximum, theme) {
  const light = theme === 'light';
  if (!(count > 0)) return light ? '#eaf0f5' : '#0b1118';
  const strength = Math.min(
    1,
    Math.log1p(count) / Math.log1p(Math.max(1, maximum)),
  );
  if (light) return `hsl(${48 - 42 * strength}, 85%, ${83 - 42 * strength}%)`;
  const colors = DARK_HEAT_COLORS;
  const point = strength * (colors.length - 1),
    index = Math.min(colors.length - 2, Math.floor(point)),
    fraction = point - index;
  return `rgb(${colors[index].map((v, i) => Math.round(v + (colors[index + 1][i] - v) * fraction)).join(', ')})`;
}

export function drawMovementHeatmap(
  ctx,
  map,
  theme,
  obstacles = [],
  scaleMaximum = map?.maximum,
) {
  if (
    !map ||
    !Number.isInteger(map.width) ||
    !Number.isInteger(map.height) ||
    map.width < 1 ||
    map.height < 1 ||
    map.width > 256 ||
    map.height > 256 ||
    (!Array.isArray(map.counts) && !(map.counts instanceof Uint32Array)) ||
    map.counts.length !== map.width * map.height
  )
    return false;
  const light = theme === 'light',
    maximum = Math.max(1, scaleMaximum ?? 0);
  ctx.fillStyle = movementHeatColor(0, maximum, theme);
  ctx.fillRect(0, 0, map.width, map.height);
  const colorCache = new Map();
  map.counts.forEach((count, i) => {
    if (!Number.isFinite(count) || count <= 0) return;
    if (!colorCache.has(count))
      colorCache.set(count, movementHeatColor(count, maximum, theme));
    ctx.fillStyle = colorCache.get(count);
    ctx.fillRect(i % map.width, Math.floor(i / map.width), 1, 1);
  });
  ctx.fillStyle = light ? '#7b8a94' : '#71808c';
  for (const [x, y] of obstacles)
    if (x >= 0 && x < map.width && y >= 0 && y < map.height)
      ctx.fillRect(x, y, 1, 1);
  return true;
}

export function movementHeatmap(map, obstacles = [], { compact = false } = {}) {
  if (!map?.counts) return null;
  const element = document.createElement('section'),
    title = document.createElement('h3'),
    canvas = document.createElement('canvas'),
    note = document.createElement('p'),
    legend = document.createElement('div');
  element.className = 'movement-heatmap';
  canvas.width = map.width;
  canvas.height = map.height;
  canvas.setAttribute('role', 'img');
  note.className = 'muted';
  note.hidden = compact;
  legend.className = 'movement-heatmap-legend';
  legend.setAttribute(
    'aria-label',
    'Heatmap arrival count color scale; gray cells are obstacles',
  );
  element.append(title, canvas, legend, note);
  let lastMaximum = null,
    swatches = [],
    paintedFrame = null,
    paintedTheme = null,
    paintedMap = null,
    scaleMaximum = map.maximum,
    paintedMaximum = null;
  const view = {
    element,
    update(next, theme, maximum = next.maximum) {
      map = next;
      scaleMaximum = maximum;
      return view.draw(theme);
    },
    draw(theme) {
      if (
        map.frame_index !== undefined &&
        paintedMap === map &&
        paintedFrame === map.frame_index &&
        paintedTheme === theme &&
        paintedMaximum === scaleMaximum
      )
        return true;
      const group =
        map.species === null || map.species === undefined
          ? 'all species'
          : map.species;
      title.textContent = `Movement heatmap · ${group} · through step ${map.tick ?? 0}`;
      canvas.setAttribute(
        'aria-label',
        `Movement arrivals through replay step ${map.tick ?? 0}, ${group}: ${map.arrivals} arrivals, maximum ${map.maximum} per cell. North is up.`,
      );
      note.textContent = `${map.arrivals.toLocaleString()} changed-position arrivals · busiest cell ${map.maximum.toLocaleString()} · ${group}, both sexes · logarithmic color scale. Through replay step ${map.tick ?? 0} only; future frames excluded. Stationary time, births, frame gaps and unknown terminal routes are excluded. Gray = obstacles.`;
      if (scaleMaximum !== lastMaximum) {
        lastMaximum = scaleMaximum;
        legend.replaceChildren();
        const values =
          scaleMaximum > 0
            ? [
                ...new Set([
                  0,
                  ...Array.from({ length: 5 }, (_, i) =>
                    Math.round(
                      Math.expm1(
                        Math.log1p(scaleMaximum) * (0.15 + (0.85 * i) / 4),
                      ),
                    ),
                  ),
                ]),
              ]
            : [0];
        swatches = values.map((value) => {
          const item = document.createElement('span'),
            swatch = document.createElement('i'),
            label = document.createElement('span');
          label.textContent = String(value);
          item.append(swatch, label);
          legend.append(item);
          return [swatch, value];
        });
      }
      for (const [swatch, value] of swatches)
        swatch.style.backgroundColor = movementHeatColor(
          value,
          scaleMaximum,
          theme,
        );
      paintedMap = map;
      paintedMaximum = scaleMaximum;
      paintedFrame = map.frame_index;
      paintedTheme = theme;
      return drawMovementHeatmap(
        canvas.getContext('2d'),
        map,
        theme,
        obstacles,
        scaleMaximum,
      );
    },
  };
  return view;
}
