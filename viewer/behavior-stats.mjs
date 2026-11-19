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
