import { visualColor } from './theme.mjs';
import {
  DEFAULT_NETWORK_RANK,
  inputInterpretation,
  animalStructure,
  networkStructure,
  rankLimit,
} from './network-structure.mjs';
// Actual per-animal diagnostic samples, read directly from the loaded replay chunk.
// No archive requests, weight inference or interpolation between samples.
import { predictionView } from './inspector-stats.mjs';
const NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);
const el = (tag, value, cls) => {
  const n = document.createElement(tag);
  if (value !== undefined) n.textContent = value;
  if (cls) n.className = cls;
  return n;
};
const svg = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (text !== undefined) n.textContent = text;
  return n;
};
const validLayers = (sample) =>
  sample?.version === 1 &&
  Array.isArray(sample.layers) &&
  sample.layers.length <= 12 &&
  sample.layers.every(
    (layer) =>
      Array.isArray(layer) &&
      layer.length <= 16384 &&
      layer.every(Number.isFinite),
  );
function activationMap(values, title, compact) {
  const limit = compact ? 64 : values.length,
    count = Math.min(limit, values.length),
    columns = compact ? 16 : 32,
    cell = compact ? 17 : 20,
    width = compact ? 300 : 680,
    height = 24 + Math.ceil(count / columns) * cell;
  const chart = svg('svg', {
    viewBox: `0 0 ${width} ${height}`,
    role: 'img',
    'aria-label': title,
  });
  chart.append(svg('title', {}, title));
  for (let i = 0; i < count; i++) {
    const v = values[i],
      r = svg('rect', {
        x: 6 + (i % columns) * cell,
        y: 4 + Math.floor(i / columns) * cell,
        width: cell - 3,
        height: cell - 4,
        rx: 2,
        fill: v >= 0 ? visualColor('#7de2bd') : visualColor('#ff9aab'),
        opacity: 0.18 + 0.82 * Math.min(1, Math.abs(v)),
      });
    r.append(svg('title', {}, `Unit ${i}: ${v.toPrecision(6)}`));
    chart.append(r);
  }
  chart.append(
    svg(
      'text',
      { x: 6, y: height - 3 },
      `${count}/${values.length} units · color = sign · intensity clipped at |value| = 1`,
    ),
  );
  return chart;
}
function predictionBars(diagnostic) {
  const root = el('section', undefined, 'brain-layer'),
    view = predictionView(diagnostic);
  root.append(
    el(
      'h4',
      view?.policy
        ? 'Actual sampled action probabilities'
        : 'Actual sampled Q estimates',
    ),
  );
  if (!view) {
    root.append(
      el('p', 'No action scores were recorded for this sample.', 'muted'),
    );
    return root;
  }
  for (const action of view.actions) {
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
    zero.style.left = `${view.zero}%`;
    track.append(fill, zero);
    row.append(
      el(
        'span',
        action.action.replaceAll('_', ' ') + (action.masked ? ' · masked' : ''),
      ),
      track,
      el(
        'strong',
        view.policy
          ? `${(action.value * 100).toFixed(1)}%`
          : action.value.toFixed(3),
      ),
    );
    root.append(row);
  }
  return root;
}
export function activationPanel(
  diagnostic,
  template,
  { compact = true, rank = DEFAULT_NETWORK_RANK, sections = {} } = {},
) {
  const root = el('section', undefined, 'animal-brain-activity'),
    sample = diagnostic?.activations;
  root.append(el('h3', 'Recorded animal brain activity'));
  if (validLayers(sample)) {
    root.append(
      el(
        'p',
        `Step ${diagnostic.tick} · ${diagnostic.updates ?? 0} optimizer updates · ${diagnostic.algorithm ?? template?.type ?? 'brain'}. Visualization values are stored as FP32. These are this animal’s own sampled next-decision activations, not a champion’s weights or the action already taken.`,
        'muted',
      ),
    );
  }
  if (!compact && diagnostic) {
    root.append(
      inputInterpretation(diagnostic, template),
      predictionBars(diagnostic),
    );
  }
  const structure = animalStructure(diagnostic, template);
  if (structure) root.append(networkStructure(structure, rank));
  if (!validLayers(sample)) {
    root.append(
      el(
        'p',
        sample
          ? 'Unsupported activation sample. Recorded predictions remain available.'
          : 'Internal activations were not recorded in this sample. New recordings include them by default; older recordings cannot recover hidden activations from predictions alone.',
        'muted',
      ),
    );
    return root;
  }
  const details = el('details', undefined, 'brain-activation-details');
  details.append(el('summary', 'Advanced · individual neuron activation maps'));
  details.open = Boolean(sections.activations);
  let rendered = false;
  const render = () => {
    if (rendered) return;
    rendered = true;
    sample.layers.forEach((values, index) => {
      const layer = el('section', undefined, 'brain-layer'),
        recurrent = ['recurrent_ppo', 'lstm_ppo'].includes(
          diagnostic.algorithm,
        );
      layer.append(
        el(
          'h4',
          recurrent && index === 0
            ? `${diagnostic.algorithm === 'lstm_ppo' ? 'LSTM' : 'GRU'} current activation`
            : `Hidden layer ${index + (recurrent ? 0 : 1)} · ${values.length} units`,
        ),
        activationMap(
          values,
          `Actual sampled hidden activations, layer ${index + 1}`,
          compact,
        ),
      );
      details.append(layer);
    });
    if (
      !compact &&
      Array.isArray(sample.recurrent_prior) &&
      sample.recurrent_prior.length <= 16384 &&
      sample.recurrent_prior.every(Number.isFinite)
    ) {
      const prior = el('section', undefined, 'brain-layer');
      prior.append(
        el(
          'h4',
          diagnostic.algorithm === 'lstm_ppo'
            ? 'Actual prior LSTM memory · hidden then cell'
            : 'Actual prior recurrent memory',
        ),
        el(
          'p',
          'The sampled forward pass uses this animal’s existing memory without advancing or resetting it.',
          'muted',
        ),
        activationMap(
          sample.recurrent_prior,
          'Actual sampled prior recurrent memory',
          false,
        ),
      );
      details.append(prior);
    }
  };
  details.ontoggle = () => {
    sections.activations = details.open;
    if (details.open) render();
  };
  if (details.open) render();
  root.append(details);
  root.append(
    el(
      'p',
      compact
        ? 'Green = positive · pink = negative. Open Watch animal brain for all units and playback controls.'
        : 'Activity updates only at recorded diagnostic samples. Connections, weights and causal attributions are not stored in this view; there is no interpolation or new inference during playback.',
      'muted brain-note',
    ),
  );
  return root;
}
export class AnimalBrainView {
  constructor() {
    this.sample = null;
    this.key = '';
    this.sections = {};
    $('brain-close').onclick = () => $('brain-dialog').close();
    $('brain-dialog').onclose = () => {
      this.sample = null;
      this.key = '';
      $('brain-content').replaceChildren();
    };
  }
  open(context) {
    $('brain-dialog').showModal();
    this.update(context, true);
  }
  update(context, force = false) {
    if (!$('brain-dialog').open) return;
    const { id, run, tick, animal, diagnostic, template, playing, done } =
      context;
    const rank = rankLimit(context.rankLimit);
    $('brain-play').textContent = playing ? 'Pause replay' : 'Play replay';
    $('brain-play').disabled = !playing && done;
    $('brain-status').textContent = animal
      ? `${animal.species} ${animal.sex ?? ''} #${animal.id} · run ${run} · body step ${tick} · brain sample ${diagnostic?.tick ?? 'unavailable'}. Following the selected animal as replay plays.`
      : `No selected animal is alive in this frame (run ${run}, step ${tick}). Select an animal on the board to inspect its recorded brain.`;
    const sample = animal ? diagnostic : undefined;
    const key = `${id}:${run}:${animal?.id ?? ''}:${sample?.tick ?? ''}:${rank}`;
    if (!force && key === this.key && this.sample === sample) return;
    this.key = key;
    this.sample = sample;
    $('brain-content').replaceChildren(
      animal
        ? activationPanel(diagnostic, template, {
            compact: false,
            rank,
            sections: this.sections,
          })
        : el(
            'p',
            'The previous animal’s activation display has been cleared.',
            'muted',
          ),
    );
  }
}
