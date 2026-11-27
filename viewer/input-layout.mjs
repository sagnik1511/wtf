import { buildCompactLayout } from './compact-observations.mjs';
import { OPPORTUNITY_NAMES } from './observation-v5.mjs';
import { socialLabels } from './social.mjs';
import { trackingLabels } from './tracking.mjs';
import { buildSoftLayout } from './soft-observations.mjs';
export function rawInputLabels(template = {}, length) {
  template ??= {};
  const names = [
    'Energy',
    'Health',
    'Heading x',
    'Heading y',
    'Blocked front',
    'Blocked right',
    'Blocked back',
    'Blocked left',
  ];
  for (const category of ['Food', 'Prey', 'Predator'])
    for (let i = 1; i <= (template.observation_top_k ?? 1); i++)
      for (const part of ['dx', 'dy', 'distance', 'present'])
        names.push(`${category} ${i} ${part}`);
  if ((template.observation_version ?? 1) >= 2) {
    names.push('Water depth', 'Flow x', 'Flow y', 'Elevation');
    for (const side of ['Front', 'Right', 'Back', 'Left'])
      names.push(`${side} water`, `${side} climb`);
  }
  if ((template.observation_version ?? 1) >= 3)
    names.push(
      'Sex',
      'Maturity',
      'Cooldown',
      'Self ready',
      'Mate dx',
      'Mate dy',
      'Mate distance',
      'Mate present',
      'Mate ready',
      'Pair opportunity',
    );
  if ((template.observation_version ?? 1) >= 4)
    names.push('Self consent', 'Mate consent', 'Consenting pair');
  if (template.observation_version === 5) {
    for (let i = 0; i < names.length; i++)
      names[i] = names[i]
        .replace(/ dx$/, ' forward')
        .replace(/ dy$/, ' right')
        .replace('Flow x', 'Flow forward')
        .replace('Flow y', 'Flow right');
    names.push(...OPPORTUNITY_NAMES);
  }
  names.push(...trackingLabels(template), ...socialLabels(template));
  return Array.from(
    { length: length ?? names.length },
    (_, i) => names[i] ?? `Input ${i}`,
  );
}
export function semanticLayout(template) {
  return buildSoftLayout(
    rawInputLabels({ ...template, observation_version: 5 }),
    template,
  );
}
export function compactLayout(template) {
  return buildCompactLayout(
    rawInputLabels({ ...template, observation_version: 5 }),
    template,
  );
}
export function inputLabels(template = {}, length = 20) {
  if (template?.observation_version === 7) {
    const names = compactLayout(template).names;
    return Array.from({ length }, (_, i) => names[i] ?? `Input ${i}`);
  }
  if (template?.observation_version !== 6)
    return rawInputLabels(template, length);
  const names = semanticLayout(template).names;
  return Array.from({ length }, (_, i) => names[i] ?? `Input ${i}`);
}
