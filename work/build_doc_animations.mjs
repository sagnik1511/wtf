import { daylight, clockMarkup, clockState } from '../viewer/day-night.mjs';
import { predictionView } from '../viewer/inspector-stats.mjs';
// Controlled vector demonstrations using current viewer drawing components.
// Values and paths are illustrative: these are not saved experiments or UI captures.
export function buildDocAnimations({
  board,
  animal,
  sheet,
  box,
  text,
  lines,
  geography,
  land,
  inspectorGraphs,
  walk,
  component,
  diagnostic,
  template,
  networkStructure,
  animalStructure,
  explanationCharts,
  report,
}) {
  const count = 8;
  const cycle = { enabled: true, cycle_steps: 400, start_phase: 0 };
  const cycleTicks = [0, 110, 150, 190, 220, 270, 300, 350];
  const animationStyle = `<style>
    .demo-frame:first-of-type{visibility:visible}
    .demo-frame{visibility:hidden;animation:demo-frame 4s steps(1,end) infinite}
    @keyframes demo-frame{0%,12.499%{visibility:visible}12.5%,100%{visibility:hidden}}
    .demo-light{display:none}
    @media(prefers-color-scheme:light){.demo-light{display:inline}.demo-dark{display:none}}
    @media(prefers-reduced-motion:reduce){.demo-frame{animation:none!important;visibility:hidden}.demo-frame:first-of-type{visibility:visible}}
  </style>`;
  const frame = (index, body) =>
    `<g class="demo-frame" style="animation-delay:-${4 - index / 2}s">${body}</g>`;
  const boards = (options) =>
    ['dark', 'light']
      .map(
        (theme) =>
          `<g class="demo-${theme}">${board({ ...options, theme })}</g>`,
      )
      .join('');
  const wrap = (title, frames) =>
    sheet(
      title,
      'Animated vector demonstration · illustrative states, not a recorded experiment · 4-second loop',
      480,
      `${animationStyle}<g class="demo-sequence">${frames.map((body, i) => frame(i, body)).join('')}</g>`,
    );
  const makeBoard = (kind, index) => {
    let definition = null,
      obstacles = [],
      x = 1 + (index % 4),
      y = 3;
    if (kind === 'river-crossing') {
      definition = geography(
        [
          '...~....',
          '...~....',
          '...~....',
          '...~....',
          '...~....',
          '...~....',
        ],
        land,
      );
    } else if (kind === 'highlands') {
      definition = geography(
        [
          '....1112',
          '...11122',
          '..111222',
          '.1112222',
          '11122222',
          '11222222',
        ],
        land,
      );
      y = 4 - Math.floor(index / 3);
    } else if (kind === 'enclosed-valleys') {
      obstacles = [
        [2, 1],
        [3, 1],
        [4, 1],
        [5, 1],
        [2, 2],
        [5, 2],
        [2, 3],
        [2, 4],
        [3, 4],
        [4, 4],
        [5, 4],
      ];
      x = [3, 4, 4, 5, 6, 6, 5, 4][index];
      y = 3;
    }
    const options = {
      width: 8,
      height: 6,
      definition,
      obstacles,
      animals: [
        animal('prey', 'female', 1, 0, x, y),
        animal('predator', 'male', 1, 1, 1 + (index % 3), 1),
      ],
      resources: [{ x: 6, y: 4, quantity: 8 + index }],
      tick: kind === 'day-night' ? cycleTicks[index] : index * 50,
      selected: 0,
      sight: true,
    };
    if (kind === 'day-night')
      options.day_night = { enabled: true, cycle_steps: 400, start_phase: 0 };
    return `<g transform="translate(40 112)">${boards(options)}</g>`;
  };
  const outputs = {};
  for (const [name, title, caption] of [
    [
      'simulation-playback',
      'World playback and selected-animal sight',
      'Select an animal to inspect its local view.',
    ],
    [
      'river-crossing',
      'River crossing',
      'Water entry requires swim; current can cause drift.',
    ],
    ['highlands', 'Highlands', 'Elevation affects travel effort and feedback.'],
    [
      'enclosed-valleys',
      'Enclosed valleys',
      'Walls constrain routes; the gate permits passage.',
    ],
    [
      'day-night',
      'Day–night cycle',
      'Saved world time drives sunlight and visibility.',
    ],
  ]) {
    outputs[`animation-${name}.svg`] = wrap(
      title,
      Array.from(
        { length: count },
        (_, i) =>
          makeBoard(name, i) +
          box(460, 112, 700, 288) +
          text(482, 148, caption, 20) +
          lines(
            482,
            190,
            [
              name === 'day-night'
                ? `${daylight(cycle, cycleTicks[i]).label} · sunlight ${Math.round(daylight(cycle, cycleTicks[i]).light * 100)}%`
                : `Illustrative state ${i + 1} / ${count}`,
              'Mint / blue circles: prey; coral / gold triangles: predators.',
              'Blue overlay: selected animal’s observation area.',
              'Movement and values are authored examples, not policy results.',
            ],
            16,
          ) +
          (name === 'day-night'
            ? `<g transform="translate(490 292) scale(1.3)">${clockMarkup(clockState(cycle, cycleTicks[i]))}</g>`
            : ''),
      ),
    );
  }
  outputs['animation-species-metrics.svg'] = wrap(
    'Per-species metrics follow the playhead',
    Array.from(
      { length: count },
      (_, i) =>
        box(28, 100, 1144, 330) +
        text(
          52,
          135,
          'Illustrative cumulative history through the current step',
          20,
        ) +
        ['prey', 'predator', 'grazer']
          .map((species, row) => {
            const values = Array.from(
              { length: i + 1 },
              (_, j) => 5 + row * 2 + ((j * (row + 1)) % 5),
            );
            return (
              text(52, 190 + row * 90, species, 18) +
              values
                .map(
                  (value, j) =>
                    `<rect x="${220 + j * 106}" y="${198 + row * 90 - value * 4}" width="64" height="${value * 4}" rx="3" fill="${['#7de2bd', '#ff9aab', '#8ac8ff'][row]}"/>` +
                    text(
                      252 + j * 106,
                      218 + row * 90,
                      String(j + 1),
                      12,
                      '#a9bacc',
                      'middle',
                    ),
                )
                .join('')
            );
          })
          .join(''),
    ),
  );
  outputs['animation-animal-brain.svg'] = wrap(
    'Selected-animal observations and brain activity',
    Array.from({ length: count }, (_, i) => {
      const sample = {
        ...diagnostic,
        features: diagnostic.features.map((v, j) => (j === 8 ? i / 10 : v)),
        action_probabilities: {
          rest: 0.8 - i * 0.08,
          move_backward: 0.2 + i * 0.08,
          eat: 0,
        },
        activations: {
          version: 1,
          layers: [
            [i / 10, -0.4, 0.7 - i / 10, -0.1],
            [0.35, -i / 10, 0.1],
          ],
        },
      };
      const plots = walk(
        inspectorGraphs({
          diagnostic: sample,
          template,
          spec: { body: { vision_radius: 4 } },
          hasWater: false,
        }),
      ).filter((n) => n.tagName === 'svg');
      const scores = predictionView(sample)
        .actions.map(
          (action, j) =>
            text(310, 190 + j * 58, action.action.replaceAll('_', ' '), 13) +
            `<rect x="310" y="${200 + j * 58}" width="${(220 * action.width) / 100}" height="12" rx="3" fill="#7de2bd"/>` +
            text(
              540,
              211 + j * 58,
              `${Math.round(action.value * 100)}%`,
              12,
              '#a9bacc',
              'end',
            ),
        )
        .join('');
      const network = walk(
        networkStructure(animalStructure(sample, template)),
      ).find((n) => n.tagName === 'svg');
      return (
        box(28, 100, 550, 330) +
        box(598, 100, 574, 330) +
        text(50, 132, 'Illustrative sampled observations / predictions', 18) +
        component(plots[0], 50, 155, 240, 240) +
        scores +
        text(620, 132, 'Small-network activity · current renderer', 18) +
        component(network, 620, 155, 525, 250)
      );
    }),
  );
  outputs['animation-archived-brain.svg'] = wrap(
    'Archived-model explanation follows input samples',
    Array.from({ length: count }, (_, i) => {
      const sample = {
        ...report,
        outputs: report.outputs.map((out, j) => ({
          ...out,
          score: j === 0 ? 0.8 - i * 0.08 : j === 1 ? 0.2 + i * 0.08 : 0,
        })),
        top_features: report.top_features.map((f) => ({
          ...f,
          value: f.value * (1 + i / 10),
          attribution: f.attribution * (1 + i / 10),
        })),
      };
      const plots = walk(explanationCharts(sample)).filter(
        (n) => n.tagName === 'svg',
      );
      return (
        box(28, 100, 1144, 330) +
        text(
          50,
          132,
          'Fixed archived model · illustrative changing inputs and output comparison',
          20,
        ) +
        plots
          .slice(0, 2)
          .map((plot, j) => component(plot, 50 + j * 560, 155, 530, 245))
          .join('')
      );
    }),
  );
  return outputs;
}
