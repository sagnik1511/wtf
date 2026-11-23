import { visualColor } from './theme.mjs';
const el = (tag, value, cls) => {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = value;
  if (cls) node.className = cls;
  return node;
};
const svg = (tag, attrs, value) => {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, val] of Object.entries(attrs))
    node.setAttribute(key, String(val));
  if (value !== undefined) node.textContent = value;
  return node;
};
const label = (action) => action.replaceAll('_', ' ');

export function collectivePanel(
  social,
  spec,
  tick,
  consequenceSchedule,
  averages,
  title = 'Collective learning',
) {
  const panel = el('section', undefined, 'inspector-visual');
  panel.append(el('h3', title));
  const caption = el('p', '', 'muted');
  panel.append(caption);
  panel.append(
    el(
      'p',
      'Visible group outcomes and saved species estimates. Gray/unknown means no evidence; rates are associations.',
      'muted',
    ),
  );
  if ((averages?.animal ?? social).scope === 'both')
    panel.append(
      el(
        'p',
        'General and context estimates are separate inputs; the network learns their relevance.',
        'muted',
      ),
    );
  if ((averages?.animal ?? social).scope === 'species_action')
    panel.append(
      el(
        'p',
        'Species/action knowledge stays available regardless of current state or action mask.',
        'muted',
      ),
    );
  if (social.consequences.length)
    panel.append(
      el(
        'p',
        consequenceSchedule === 'episode'
          ? 'Consequence model frozen at episode start. New evidence applies next episode; averages can still change with the sampled context.'
          : 'Consequence model updates continuously during training; playback reads saved samples only.',
        'muted',
      ),
    );

  function charts(container, data) {
    const updates = [];
    const groups = el('div', undefined, 'collective-groups');
    container.append(groups);
    for (const [title, key] of [
      ['Local witnesses', 'collective'],
      ...(data.general_consequences?.length
        ? [
            ['General consequence estimates', 'general_consequences'],
            ['Context consequence estimates', 'consequences'],
          ]
        : [['Consequence estimates', 'consequences']]),
    ]) {
      const rows = data[key].filter(
        (row) =>
          key === 'collective' || spec?.allowed_actions?.includes(row.action),
      );
      if (!rows.length) continue;
      const group = el('section', undefined, 'collective-group');
      groups.append(group);
      group.append(el('h4', title));
      const chart = svg('svg', {
        viewBox: `0 0 340 ${28 + rows.length * 26}`,
        width: 340,
        height: 28 + rows.length * 26,
        preserveAspectRatio: 'xMinYMin meet',
        class: 'collective-chart',
        'font-family': 'system-ui, sans-serif',
        role: 'img',
        'aria-label': title,
      });
      chart.append(
        svg(
          'text',
          { x: 2, y: 13, fill: visualColor('#a9bacc'), 'font-size': 10 },
          'Action',
        ),
      );
      const columns = [
        ['harm', '#ff9aab'],
        ['feeding', '#7de2bd'],
        ['birth', '#8ac8ff'],
      ];
      columns.forEach(([name, color], i) =>
        chart.append(
          svg(
            'text',
            {
              x: 172 + i * 62,
              'text-anchor': 'middle',
              y: 13,
              fill: visualColor(color),
              'font-size': 10,
            },
            name,
          ),
        ),
      );
      rows.forEach((row, i) => {
        const y = 34 + i * 26;
        chart.append(
          svg(
            'text',
            { x: 2, y: y + 5, fill: visualColor('#a9bacc'), 'font-size': 11 },
            label(row.action),
          ),
        );
        const unknown = svg(
          'text',
          { x: 148, y: y + 5, fill: visualColor('#a9bacc'), 'font-size': 10 },
          'unknown',
        );
        chart.append(unknown);
        const bars = columns.map(([name, color], j) => {
          const track = svg('rect', {
            x: 148 + j * 62,
            y: y - 4,
            width: 48,
            height: 9,
            rx: 2,
            fill: visualColor('#263341'),
          });
          const bar = svg('rect', {
            x: 148 + j * 62,
            y: y - 4,
            width: 0,
            height: 9,
            rx: 2,
            fill: visualColor(color),
          });
          const hint = svg('title', {}, '');
          bar.append(hint);
          chart.append(track, bar);
          return { name, track, bar, hint };
        });
        updates.push((next) => {
          const value = next?.[key]?.find((r) => r.action === row.action) ?? {
            confidence: 0,
          };
          const known = value.confidence > 0;
          unknown.setAttribute('visibility', known ? 'hidden' : 'visible');
          for (const { name, track, bar, hint } of bars) {
            track.setAttribute('visibility', known ? 'visible' : 'hidden');
            bar.setAttribute('visibility', known ? 'visible' : 'hidden');
            bar.setAttribute(
              'width',
              known ? Math.max(0, Math.min(1, value[name])) * 48 : 0,
            );
            bar.setAttribute('opacity', 0.35 + 0.65 * (value.confidence ?? 0));
            hint.textContent = `${label(row.action)} ${name}: ${Math.round((value[name] ?? 0) * 100)}%; evidence ${Math.round((value.confidence ?? 0) * 100)}%`;
          }
        });
      });
      group.append(chart);
    }
    const update = (next) => updates.forEach((f) => f(next));
    update(data);
    return update;
  }
  const updateAnimal = charts(panel, averages?.animal ?? social);
  panel.update = (next) => {
    if (!next) {
      caption.textContent = `Recorded sample at step ${tick}. The display follows saved samples.`;
      return;
    }
    caption.textContent = `Running averages through step ${next.tick}. Saved values are held between samples; unknown evidence is excluded from rates.`;
    updateAnimal(next.animal);
  };
  panel.update(averages);
  return panel;
}
