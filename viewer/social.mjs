// Decode saved collective inputs only; no live inference or hidden-world lookup.
export const SOCIAL_ACTIONS = [
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
export const SOCIAL_GROUPS = [
  'walk',
  'swim',
  'feed',
  'combat',
  'reproduce',
  'idle',
];
const PARTS = ['confidence', 'harm', 'feeding', 'birth'];
export function socialWidth(template = {}) {
  const spec = template.social_observation;
  if (
    spec !== undefined &&
    (spec === null ||
      typeof spec !== 'object' ||
      Array.isArray(spec) ||
      Object.keys(spec).some(
        (k) => !['history_steps', 'max_neighbors'].includes(k),
      ) ||
      ![spec.history_steps ?? 4, spec.max_neighbors ?? 8].every(
        (v) => Number.isInteger(v) && v >= 1 && v <= 16,
      ))
  )
    return null;
  if (typeof (template.consequence_observation ?? false) !== 'boolean')
    return null;
  return (
    (spec === undefined ? 0 : 24) + (template.consequence_observation ? 48 : 0)
  );
}
export function socialLabels(template = {}) {
  if (socialWidth(template) === null) return [];
  const names = [];
  if (template.social_observation !== undefined)
    for (const action of SOCIAL_GROUPS)
      for (const part of PARTS) names.push(`Collective ${action} ${part}`);
  if (template.consequence_observation)
    for (const action of SOCIAL_ACTIONS)
      for (const part of PARTS) names.push(`Consequence ${action} ${part}`);
  return names;
}
export function socialView(features, template, offset) {
  if (!socialWidth(template)) return null;
  const rows = (actions) =>
    actions.map((action) => {
      const [confidence, harm, feeding, birth] = features.slice(
        offset,
        offset + 4,
      );
      offset += 4;
      return { action, confidence, harm, feeding, birth };
    });
  return {
    collective:
      template.social_observation === undefined ? [] : rows(SOCIAL_GROUPS),
    consequences: template.consequence_observation ? rows(SOCIAL_ACTIONS) : [],
  };
}
