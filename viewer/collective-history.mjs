// Viewer-only time averages of saved inputs. No learning or extra replay fields.
import { observationView } from './inspector-stats.mjs';

const makeScope = (rows = null) => ({ rows, sums: null, steps: 0 });
const flatten = (social) => [
  ...social.collective,
  ...(social.general_consequences ?? []),
  ...social.consequences,
];
function add(scope, steps) {
  if (!scope.rows || steps <= 0) return;
  scope.sums ??= scope.rows.map(() => [0, 0, 0, 0, 0]);
  scope.steps += steps;
  scope.rows.forEach((row, i) => {
    const sum = scope.sums[i];
    sum[0] += row.confidence * steps;
    if (row.confidence > 0) {
      sum[1] += row.harm * steps;
      sum[2] += row.feeding * steps;
      sum[3] += row.birth * steps;
      sum[4] += steps;
    }
  });
}
function average(scope) {
  if (!scope?.rows || !scope.sums || !scope.steps) return null;
  const rows = scope.rows.map((row, i) => {
    const sum = scope.sums[i],
      known = sum[4];
    return {
      action: row.action,
      confidence: sum[0] / scope.steps,
      harm: known ? sum[1] / known : 0,
      feeding: known ? sum[2] / known : 0,
      birth: known ? sum[3] / known : 0,
    };
  });
  return {
    scope: scope.scope ?? 'context',
    collective: rows.slice(0, scope.collective),
    ...((scope.general ?? 0)
      ? {
          general_consequences: rows.slice(
            scope.collective,
            scope.collective + scope.general,
          ),
        }
      : {}),
    consequences: rows.slice(scope.collective + (scope.general ?? 0)),
    steps: scope.steps,
  };
}
function copyScope(scope) {
  return {
    ...scope,
    rows: scope.rows?.map((r) => ({ ...r })) ?? null,
    sums: scope.sums?.map((r) => r.slice()) ?? null,
  };
}
export class CollectiveTotals {
  constructor(config) {
    this.config = config;
    this.agents = new Map();
    this.species = new Map();
    this.members = new Map(); // Derived row-identity cache; copies rebuild it.
    this.tick = null;
    this.firstTick = null;
  }
  copy() {
    const result = new CollectiveTotals(this.config);
    result.agents = new Map(
      [...this.agents].map(([i, s]) => [i, copyScope(s)]),
    );
    result.species = new Map(
      [...this.species].map(([i, s]) => [i, copyScope(s)]),
    );
    result.tick = this.tick;
    result.firstTick = this.firstTick;
    return result;
  }
  consume(frame) {
    if (
      !Number.isInteger(frame.tick) ||
      frame.tick < 0 ||
      (this.tick !== null && frame.tick <= this.tick)
    )
      throw new Error('Collective history needs increasing replay steps');
    // Saved frames include their own step. Gaps hold only the last seen values;
    // the next sample is never applied retrospectively to unseen steps.
    const gap = this.tick === null ? 0 : frame.tick - this.tick - 1;
    for (const scope of this.agents.values()) add(scope, gap);
    for (const scope of this.species.values())
      if (scope.active) add(scope, gap);
    const living = new Set(frame.animals.map((a) => a.id));
    for (const id of this.agents.keys())
      if (!living.has(id)) this.agents.delete(id);
    const grouped = new Map();
    for (const animal of frame.animals) {
      const spec = this.config.species?.[animal.species];
      const template = this.config.brains?.[spec?.brain];
      const diagnostic = frame.diagnostics?.[String(animal.id)];
      const social =
        diagnostic && diagnostic.tick <= frame.tick
          ? observationView(diagnostic.features, template)?.social
          : null;
      let scope = this.agents.get(animal.id);
      if (social) {
        if (!scope || scope.name !== animal.species) {
          scope = { ...makeScope(), name: animal.species };
          this.agents.set(animal.id, scope);
        }
        scope.rows = flatten(social);
        scope.collective = social.collective.length;
        scope.general = social.general_consequences?.length ?? 0;
        scope.scope = social.scope;
      }
      if (!scope?.rows) continue;
      add(scope, 1);
      if (!grouped.has(animal.species)) grouped.set(animal.species, []);
      grouped.get(animal.species).push(scope);
    }
    for (const [name, scopes] of grouped) {
      const previous = this.members.get(name);
      const unchanged =
        previous?.length === scopes.length &&
        previous.every(
          ([member, rows], i) =>
            member === scopes[i] && rows === scopes[i].rows,
        );
      const rows = unchanged
        ? this.species.get(name).rows
        : scopes[0].rows.map((row, i) => {
            let confidence = 0,
              harm = 0,
              feeding = 0,
              birth = 0,
              known = 0;
            for (const scope of scopes) {
              const current = scope.rows[i];
              confidence += current.confidence;
              if (current.confidence > 0) {
                known++;
                harm += current.harm;
                feeding += current.feeding;
                birth += current.birth;
              }
            }
            return {
              action: row.action,
              confidence: confidence / scopes.length,
              harm: known ? harm / known : 0,
              feeding: known ? feeding / known : 0,
              birth: known ? birth / known : 0,
            };
          });
      if (!unchanged)
        this.members.set(
          name,
          scopes.map((member) => [member, member.rows]),
        );
      let scope = this.species.get(name);
      if (!scope) {
        scope = makeScope();
        this.species.set(name, scope);
      }
      scope.rows = rows;
      scope.active = true;
      scope.collective = scopes[0].collective;
      scope.general = scopes[0].general;
      scope.scope = scopes[0].scope;
      add(scope, 1);
    }
    // Extinct/no-sample species retain prior totals, without inventing new data.
    for (const [name, scope] of this.species)
      if (!grouped.has(name)) {
        scope.active = false;
        this.members.delete(name);
      }
    this.firstTick ??= frame.tick;
    this.tick = frame.tick;
  }
  view(id) {
    const animal = this.agents.get(id);
    return {
      tick: this.tick,
      schedule:
        this.config.consequence_learning?.update_schedule ?? 'continuous',
      first_tick: this.firstTick,
      animal: average(animal),
      by_species: Object.fromEntries(
        [...this.species].map(([name, scope]) => [name, average(scope)]),
      ),
      species_name: animal?.name ?? null,
    };
  }
}

export class CollectiveHistory {
  reset(config, size) {
    if (!Number.isInteger(size) || size < 1)
      throw new Error('Invalid collective chunk size');
    this.request = (this.request ?? 0) + 1;
    this.config = config;
    this.configuredChunkSize = size;
    this.totals = new CollectiveTotals(config);
    this.base = this.totals.copy();
    this.start = -1;
    this.index = -1;
  }
  async move(frames, start, index, loadChunk) {
    if (index === this.index && start === this.start) return;
    const token = ++this.request;
    let totals = this.totals,
      first = this.index + 1;
    if (start !== this.start || index < this.index) {
      if (start === this.start) {
        totals = this.base.copy();
      } else {
        if (this.index !== start - 1) {
          totals = new CollectiveTotals(this.config);
          const size = this.configuredChunkSize;
          for (let offset = 0; offset < start; offset += size) {
            const older = await loadChunk(offset / size);
            if (token !== this.request) return;
            if (!Array.isArray(older) || older.length !== size)
              throw new Error('Collective replay prefix unavailable');
            for (const frame of older) totals.consume(frame);
          }
        }
        this.base = totals.copy();
      }
      first = start;
    }
    for (let frame = first; frame <= index; frame++) {
      if (!frames[frame - start])
        throw new Error('Collective replay frame unavailable');
      totals.consume(frames[frame - start]);
    }
    this.totals = totals;
    this.start = start;
    this.index = index;
  }
}
