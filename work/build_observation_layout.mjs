// Documentation-only vector layout, verified against the current semantic decoder.
export function observationLayout({
  sheet,
  box,
  text,
  lines,
  semanticLayout,
  config,
  assert,
}) {
  const template = config.brains[config.species.prey.brain];
  const predator = config.brains[config.species.predator.brain];
  const layout = semanticLayout(template);
  const k = template.observation_top_k;
  const groups = [
    ['Own state', 13, 'Energy 4 · health 3 · heading 2 · blocked directions 4'],
    [
      'Food slots',
      12 * k,
      `${k} slots × 12 channels; each slot keeps its own presence`,
    ],
    [
      'Prey slots',
      12 * k,
      `${k} slots × 12 channels; no pooling of visible animals`,
    ],
    [
      'Predator slots',
      12 * k,
      `${k} slots × 12 channels; body-relative bearings`,
    ],
    ['Terrain', 49, 'Water 3 · flow 11 · height 3 · four adjacent cells × 8'],
    [
      'Mating and consent',
      29,
      'Sex, readiness, maturity, cooldown, mate and consent',
    ],
    [
      'Immediate opportunities',
      3,
      'Attack · eat · swim; exact flags, ordinary PPO unmasked',
    ],
    [
      'Local visible outcomes',
      108,
      '6 action groups × (3 confidence + 3 outcomes × 5)',
    ],
    [
      'General consequences',
      216,
      '12 × (5 harm + 5 feeding + 5 birth + 3 confidence)',
    ],
    [
      'Context consequences',
      216,
      'Same groups, conditioned on the current context',
    ],
  ];
  let offset = 0;
  const rows = groups
    .map(([name, width, detail], i) => {
      const start = offset;
      offset += width;
      const y = 305 + i * 65;
      return (
        box(48, y, 520, 57) +
        text(63, y + 22, name, 17) +
        text(
          552,
          y + 22,
          `[${start}, ${offset}) · ${width}`,
          14,
          '#7de2bd',
          'end',
        ) +
        text(63, y + 44, detail, 12)
      );
    })
    .join('');
  assert.equal(offset, layout.names.length);
  assert.equal(k, 3);
  assert.equal(layout.names.length, 742);
  assert.equal(semanticLayout(predator).names.length, 994);
  // Check displayed boundaries against actual labels, not just the total width.
  for (const [index, prefix] of [
    [0, 'Energy'],
    [13, 'Food 1'],
    [49, 'Prey 1'],
    [85, 'Predator 1'],
    [121, 'Water depth'],
    [170, 'Sex'],
    [199, 'Attack possible now'],
    [202, 'Collective '],
    [310, 'Species action '],
    [526, 'Consequence '],
  ]) {
    assert.ok(
      layout.names[index].startsWith(prefix),
      `${index}: ${layout.names[index]}`,
    );
  }
  const cells = (x, y, labels, values, width) =>
    labels
      .map(
        (label, i) =>
          box(x + i * width, y, width - 5, 61) +
          text(
            x + i * width + (width - 5) / 2,
            y + 24,
            label,
            12,
            '#a9bacc',
            'middle',
          ) +
          text(
            x + i * width + (width - 5) / 2,
            y + 47,
            values[i],
            16,
            '#7de2bd',
            'middle',
          ),
      )
      .join('');
  const arrow = (x1, y1, x2, y2) =>
    `<path d="M${x1} ${y1}L${x2} ${y2}" stroke="#7de2bd" stroke-width="2" fill="none"/><path d="M${x2 - 5} ${y2 - 7}L${x2} ${y2}L${x2 + 5} ${y2 - 7}" stroke="#7de2bd" stroke-width="2" fill="none"/>`;
  return sheet(
    'Observation V6: the complete input vector',
    'Ordered scalar channels, optional blocks and one target-slot expansion · schematic, blocks not to scale',
    1150,
    box(28, 96, 352, 103) +
      text(48, 127, '1. Sense the current world', 20) +
      lines(
        48,
        153,
        [
          'V5 sensed cues, visible / remembered only',
          'No hidden positions or future outcomes',
        ],
        14,
      ) +
      box(398, 96, 378, 103) +
      text(418, 127, '2. Encode soft categories', 20) +
      lines(
        418,
        153,
        [
          'Overlapping memberships + exact raw flags',
          'Same field order; independent top-k slots',
        ],
        14,
      ) +
      box(794, 96, 378, 103) +
      text(814, 127, '3. Feed one private brain', 20) +
      lines(
        814,
        153,
        [
          'One flat vector → private network → actions',
          'Private weights, optimizer and RNG',
        ],
        14,
      ) +
      box(28, 224, 560, 896) +
      text(48, 259, 'Ordered vector · seaside_150_v6 prey', 22) +
      text(
        48,
        285,
        'k=3 · history/scent off · both consequence scopes',
        14,
        '#a9bacc',
      ) +
      rows +
      lines(
        48,
        990,
        [
          'Indices are zero-based [start, end); the end is excluded.',
          'The first 202 channels are the core: 94 + 36k.',
          'Then append enabled blocks in the order shown.',
          'Total prey inputs: 742 · predator k=10: 994.',
          'Widths are computed/checked against the current layout.',
        ],
        14,
      ) +
      box(610, 224, 562, 465) +
      text(630, 259, 'Zoom: one target slot becomes 12 inputs', 21) +
      text(630, 289, 'V5: forward · right · distance · present  →  V6:', 16) +
      text(630, 324, '3 range memberships', 17) +
      cells(630, 339, ['near', 'medium', 'far'], ['0.7', '0.3', '0'], 171) +
      text(630, 433, '8 bearing memberships · relative to body heading', 17) +
      cells(
        630,
        448,
        ['F', 'FR', 'R', 'BR', 'B', 'BL', 'L', 'FL'],
        ['0.2', '0', '0', '0', '0', '0', '0', '0.8'],
        65,
      ) +
      text(630, 544, '1 exact presence flag: 1', 17) +
      arrow(900, 555, 900, 584) +
      text(
        630,
        610,
        'Concatenate: [range × 3 | bearing × 8 | present]',
        16,
        '#7de2bd',
      ) +
      lines(
        630,
        643,
        [
          'Illustrative near/front-left target; remaining slots repeat.',
          'Absent slot: zero range + bearing + presence channels.',
        ],
        13,
      ) +
      box(610, 709, 562, 411) +
      text(630, 746, 'Optional tails · append only when enabled', 21) +
      lines(
        630,
        784,
        [
          'After the core, before local/general/context outcomes:',
          'Movement history: 2k × (29 + 12H) channels',
          'Scent: 27 × C channels',
          '',
          'H = remembered steps; C = configured scent channels.',
          'Local outcomes: 108; each consequence scope: 216.',
          'One scope or both, as selected by configuration.',
          '',
          'Soft categories are memberships, not probabilities',
          'over actions. Opportunity flags do not mask ordinary PPO.',
          'Unknown evidence has zero confidence; it is not failure.',
          '',
          'Earlier observation versions keep their original layouts.',
        ],
        14,
      ),
  );
}

// Compare representation contents, not allocator or binary-storage formats.
export function observationComparison({
  sheet,
  box,
  text,
  lines,
  semanticLayout,
  rawInputLabels,
  decodeSoft,
  config,
  assert,
}) {
  const template = config.brains[config.species.prey.brain];
  const layout = semanticLayout(template);
  const canonical = rawInputLabels({ ...template, observation_version: 5 });
  assert.equal(canonical.length, 192);
  assert.equal(layout.names.length, 742);
  const sample = Array(layout.names.length).fill(0);
  for (const [label, values] of [
    ['Energy', [0, 0.7, 0.3, 0]],
    ['Health', [0, 0.2, 0.8]],
    ['Food 1', [0.7, 0.3, 0, 0.2, 0, 0, 0, 0, 0, 0, 0.8]],
    ['Food 1 present', [1]],
  ]) {
    const field = layout.fields.find((f) => f.label === label);
    assert.ok(field, label);
    assert.equal(field.names.length, values.length);
    sample.splice(field.offset, values.length, ...values);
  }
  const decoded = decodeSoft(sample, layout);
  const close = (name, value) =>
    assert.ok(Math.abs(decoded[canonical.indexOf(name)] - value) < 1e-12, name);
  close('Energy', 0.4);
  close('Health', 0.9);
  close('Food 1 distance', 0.09);
  close('Food 1 forward', 0.09 * Math.cos(-Math.PI / 5));
  close('Food 1 right', 0.09 * Math.sin(-Math.PI / 5));
  close('Food 1 present', 1);
  const columns = (y, height, before, after) =>
    box(28, y, 550, height) +
    box(622, y, 550, height) +
    lines(48, y + 29, before, 17) +
    lines(642, y + 29, after, 17) +
    text(600, y + height / 2 + 6, '→', 24, '#7de2bd', 'middle');
  const table = [
    [
      'Core (k=3): own state, targets, terrain, mating, opportunities',
      '72',
      '202',
    ],
    [
      'Movement history (H steps; optional)',
      '2k × (7 + 3H)',
      '2k × (29 + 12H)',
    ],
    ['Scent (C channels; optional)', '9C', '27C'],
    ['Local visible outcomes (optional)', '24', '108'],
    ['Each consequence scope (optional)', '48', '216'],
    [
      'Matched prey total: history/scent off, social + both scopes',
      '192',
      '742',
    ],
    ['Matched predator total: same flags, k=10', '276', '994'],
  ]
    .map(([label, before, after], i) => {
      const y = 790 + i * 37;
      return (
        text(48, y, label, 15) +
        text(890, y, before, 15, '#a9bacc', 'middle') +
        text(1080, y, after, 15, '#7de2bd', 'middle')
      );
    })
    .join('');
  return sheet(
    'Observation data: earlier V5 → current V6',
    'The same sensed scene, represented differently · default knots · illustrative values rounded for display',
    1230,
    box(28, 96, 550, 58) +
      text(48, 133, 'EARLIER · V5 normalized numeric inputs', 23) +
      box(622, 96, 550, 58) +
      text(642, 133, 'NOW · V6 overlapping soft categories', 23) +
      columns(
        172,
        92,
        ['Energy: one scalar', '0.40'],
        [
          'Energy: four memberships [0, 0.7, 0.3, 0]',
          'critical · hungry · comfortable · full',
        ],
      ) +
      columns(
        280,
        92,
        ['Health: one scalar', '0.90'],
        [
          'Health: three memberships [0, 0.2, 0.8]',
          'critical · weak · healthy',
        ],
      ) +
      columns(
        388,
        181,
        [
          'One visible food slot: four scalars',
          'forward ≈ 0.0728; right ≈ −0.0529',
          'distance = 0.09; present = 1',
          '',
          'Current body-relative position, not a hidden target.',
        ],
        [
          'Same slot: 12 channels',
          'range [near, medium, far] = [0.7, 0.3, 0]',
          'bearing [F, FR, R, BR, B, BL, L, FL]',
          '= [0.2, 0, 0, 0, 0, 0, 0, 0.8]; present = 1',
          'Same individual slot; no averaging of targets.',
        ],
      ) +
      columns(
        585,
        102,
        [
          'Exact flags / raw values',
          'Presence, heading, sex, blocked cells, readiness,',
          'opportunities, seen and consenting-pair flags',
        ],
        [
          'Exact flags / raw values stay unchanged',
          'Presence still distinguishes an absent slot.',
          'Absent slot: range + bearings + presence are all zero.',
        ],
      ) +
      box(28, 705, 1144, 370) +
      text(
        48,
        741,
        'Channel counts · matched optional settings, not extra sensing',
        22,
      ) +
      text(890, 754, 'V5', 16, '#a9bacc', 'middle') +
      text(1080, 754, 'V6', 16, '#7de2bd', 'middle') +
      table +
      box(28, 1093, 1144, 109) +
      text(48, 1125, 'Two separate changes', 20) +
      lines(
        48,
        1152,
        [
          'V5 → V6 expands/re-encodes the input. In-band scalar hats reconstruct values; out-of-band values saturate.',
          'The later V6 speed/memory optimization changes no input data: identical values, order, widths and replay bytes.',
        ],
        15,
      ),
  );
}

export function compactPolicyVisual({
  sheet,
  box,
  text,
  lines,
  component,
  walk,
  inspectorGraphs,
  compactLayout,
  config,
  assert,
}) {
  const spec = config.species.prey,
    template = {
      ...config.brains[spec.brain],
      observation_actions: spec.allowed_actions,
    };
  const layout = compactLayout(template),
    features = Array(layout.width).fill(0);
  assert.equal(layout.width, 211);
  assert.equal(layout.physical, 107);
  features[0] = 0.35;
  features[1] = 0.8;
  features[2] = 1;
  features.splice(layout.scentStart, 4, 0, -1, 0.9, 0.4);
  const base = spec.allowed_actions.map((a) => (a === 'rest' ? 0.4 : 0)),
    adjust = spec.allowed_actions.map((a) =>
      a === 'eat' ? 0.6 : a === 'swim' ? -0.8 : 0,
    ),
    final = base.map((a, i) => a + adjust[i]);
  const max = Math.max(...final),
    weights = final.map((v) => Math.exp(v - max)),
    sum = weights.reduce((a, b) => a + b, 0);
  const tree = inspectorGraphs({
    diagnostic: {
      tick: 12,
      algorithm: 'ppo',
      features,
      action_probabilities: Object.fromEntries(
        spec.allowed_actions.map((a, i) => [a, weights[i] / sum]),
      ),
      late_consequence: {
        base_logits: base,
        adjustments: adjust,
        final_logits: final,
      },
    },
    template,
    spec,
    hasWater: true,
  });
  const residual = walk(tree).find(
    (n) =>
      n.attributes?.['aria-label'] ===
      'Signed consequence adjustments to action logits',
  );
  assert.ok(residual);
  const compass = walk(tree).find(
    (n) => n.attributes?.class === 'compact-scent-bearing',
  );
  assert.ok(compass);
  return sheet(
    'Compact V7 and late consequence decisions',
    'Controlled viewer specimen · authored values, not learned outcomes or benchmark results',
    940,
    box(28, 95, 545, 810) +
      text(48, 132, 'One saved observation: 211 values', 23) +
      lines(
        48,
        169,
        [
          'Physical input trunk: 107 values',
          '61 core + 42 movement summary + 4 blended scent',
          '',
          'Side-channel evidence: 104 values',
          '24 local + 40 general + 40 context',
          'Recorded with each sample; excluded from trunk weights.',
        ],
        16,
      ) +
      box(48, 315, 505, 200) +
      text(68, 349, 'Four-value anonymous scent', 21) +
      lines(
        68,
        385,
        [
          '[forward=0, right=−1]',
          '[strength=.9, freshness=.4]',
          'Anonymous off-cell patch.',
          'Own older traces remain.',
        ],
        15,
      ) +
      component(compass, 385, 351, 150, 150) +
      box(48, 537, 505, 205) +
      text(68, 573, 'Seven-value movement summary per track', 21) +
      lines(
        68,
        609,
        [
          'Last position: forward/right',
          'Trend velocity: forward/right',
          'Freshness, visible now, motion reliability',
          '',
          'Only personal sightings; no hidden path extrapolation.',
          'The summary does not retain every historical turn.',
        ],
        15,
      ) +
      lines(
        48,
        789,
        [
          'V1–V6 retain their original layouts and artifacts.',
          'V7 requires matching new model weights.',
          'Ordinary PPO remains unmasked.',
          'Scent and history remain optional.',
        ],
        16,
      ) +
      box(601, 95, 571, 810) +
      text(623, 132, 'Inspector: learned consequence adjustment', 23) +
      lines(
        623,
        169,
        [
          'Private physical network proposes base action logits.',
          'A private adapter evaluates saved outcome evidence.',
          'Bounded learned residuals adjust the logits.',
          'Final probabilities drive both sampling and PPO loss.',
        ],
        16,
      ) +
      component(residual, 623, 290, 525, 390) +
      lines(
        623,
        735,
        [
          'Green raises a logit; red lowers it.',
          'These illustrative bars come from the viewer component.',
          'Zero evidence gives an exact zero residual.',
          'No fixed rule says feeding/birth must win over harm.',
          'The actor KL gate commits or rejects both updates.',
          'Published counts are shared; learned weights are private.',
        ],
        15,
      ),
  );
}
