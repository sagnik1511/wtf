// V7 replay-only layout; no world lookup, learned weights or hidden movement.
const ACTIONS = [
  'move_forward',
  'move_backward',
  'strafe_left',
  'strafe_right',
  'turn_left',
  'turn_right',
  'rest',
  'eat',
  'attack',
  'swim',
  'reproduce',
  'challenge',
];
export function buildCompactLayout(raw, template) {
  const k = template.observation_top_k ?? 1,
    h = template.movement_history_steps ?? 0;
  const actions = template.observation_actions ?? ACTIONS;
  const core = 48 + 12 * (k - 1),
    opportunity = {
      'Attack possible now': 'attack',
      'Eat possible now': 'eat',
      'Swim possible now': 'swim',
    };
  const indices = Array.from({ length: core }, (_, i) => i).filter(
    (i) =>
      !raw[i].endsWith(' distance') &&
      (!opportunity[raw[i]] || actions.includes(opportunity[raw[i]])),
  );
  const names = indices.map((i) => raw[i]);
  const tracksStart = names.length;
  if (h)
    for (const category of ['Prey', 'Predator'])
      for (let slot = 1; slot <= k; slot++)
        for (const part of [
          'last forward',
          'last right',
          'trend forward',
          'trend right',
          'freshness',
          'visible',
          'motion reliability',
        ])
          names.push(`${category} track ${slot} ${part}`);
  const scentStart = names.length,
    spec = template.scent_observation,
    channels = [];
  if (spec) {
    if (spec.mode === 'blended') channels.push(['blended', 'any']);
    else
      for (const species of spec.species)
        for (const sex of spec.separate_sexes === false
          ? ['any']
          : ['female', 'male'])
          channels.push([species, sex]);
  }
  for (const [species, sex] of channels)
    for (const part of [
      'direction forward',
      'direction right',
      'strength',
      'freshness',
    ])
      names.push(`Scent ${species} ${sex} ${part}`);
  const physical = names.length,
    rawSocial = core + (h ? 2 * k * (7 + 3 * h) : 0) + 9 * channels.length;
  if (template.social_observation)
    names.push(...raw.slice(rawSocial, rawSocial + 24));
  const consequenceStart = names.length;
  if (template.consequence_observation)
    for (const prefix of template.consequence_scope === 'both'
      ? ['Species action', 'Consequence']
      : [
          template.consequence_scope === 'species_action'
            ? 'Species action'
            : 'Consequence',
        ])
      for (const action of actions)
        for (const part of ['confidence', 'harm', 'feeding', 'birth'])
          names.push(`${prefix} ${action} ${part}`);
  return {
    names,
    width: names.length,
    indices,
    raw,
    core,
    k,
    h,
    actions,
    channels,
    tracksStart,
    scentStart,
    physical,
    rawSocial,
    consequenceStart,
  };
}
export function decodeCompact(features, layout, template) {
  if (features.length !== layout.width || !features.every(Number.isFinite))
    throw Error('Invalid V7 input width');
  const raw = Array(layout.raw.length).fill(0),
    tracks = [],
    scents = [];
  layout.indices.forEach((index, i) => {
    raw[index] = features[i];
  });
  for (let i = 0; i < layout.core; i++)
    if (layout.raw[i].endsWith(' distance'))
      raw[i] = Math.hypot(raw[i - 2], raw[i - 1]);
  let source = layout.tracksStart,
    dest = layout.core;
  if (layout.h)
    for (let i = 0; i < 2 * layout.k; i++) {
      const [f, r, vf, vr, freshness, visible, reliability] = features.slice(
        source,
        source + 7,
      );
      source += 7;
      raw.splice(
        dest,
        7,
        f,
        r,
        vf,
        vr,
        freshness
          ? Math.max(0, ((1 - freshness) * (layout.h + 1)) / layout.h)
          : 0,
        freshness,
        visible,
      );
      tracks.push({
        category: i < layout.k ? 'prey' : 'predator',
        slot: (i % layout.k) + 1,
        reliability,
      });
      dest += 7 + 3 * layout.h;
    }
  for (const [species, sex] of layout.channels) {
    const [forward, right, strength, freshness] = features.slice(
      source,
      source + 4,
    );
    source += 4;
    scents.push({ species, sex, forward, right, strength, freshness });
    dest += 9;
  }
  if (template.social_observation) {
    raw.splice(dest, 24, ...features.slice(source, source + 24));
    source += 24;
    dest += 24;
  }
  if (template.consequence_observation)
    for (
      let scope = 0;
      scope < (template.consequence_scope === 'both' ? 2 : 1);
      scope++
    ) {
      for (const action of layout.actions) {
        const offset = dest + 4 * ACTIONS.indexOf(action);
        raw.splice(offset, 4, ...features.slice(source, source + 4));
        source += 4;
      }
      dest += 48;
    }
  return { features: raw, scents, tracks };
}
