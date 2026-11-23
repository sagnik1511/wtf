// Fresh-process replay aggregation probe. Both package roots must contain viewer/.
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const option = (name) => {
  const index = process.argv.indexOf(name);
  return index < 0 ? null : process.argv[index + 1];
};
const source = option('--source');
if (source) {
  const { CollectiveTotals } = await import(
    pathToFileURL(path.resolve(source, 'viewer/collective-history.mjs'))
  );
  const config = {
    brains: {
      test: {
        observation_version: 1,
        social_observation: {},
        consequence_observation: true,
        consequence_scope: 'both',
      },
    },
    species: { forager: { brain: 'test' }, hunter: { brain: 'test' } },
  };
  const animals = Array.from({ length: 70 }, (_, id) => ({
    id,
    species: id < 64 ? 'forager' : 'hunter',
    x: id % 20,
    y: Math.floor(id / 20),
  }));
  const makeFrame = (tick) => ({
    tick,
    animals,
    ...(tick % 50 === 0
      ? {
          diagnostics: Object.fromEntries(
            animals.map(({ id }) => [
              id,
              {
                tick,
                features: [
                  ...Array(20).fill(0),
                  ...Array.from({ length: 30 }, (_, row) => [
                    0.75,
                    ((id + tick / 50) % 7) / 10,
                    (row % 3) / 10,
                    0.1,
                  ]).flat(),
                ],
              },
            ]),
          ),
        }
      : {}),
  });
  function run(checksum) {
    const totals = new CollectiveTotals(config);
    const hash = createHash('sha256');
    let builds = 0;
    for (let tick = 0; tick < 1600; tick++) {
      const previous = new Map(
        [...totals.species].map(([name, scope]) => [name, scope.rows]),
      );
      totals.consume(makeFrame(tick));
      for (const [name, scope] of totals.species)
        if (scope.rows !== previous.get(name)) builds++;
      if (checksum) hash.update(JSON.stringify(totals.view(0)));
    }
    return { totals, builds, digest: checksum ? hash.digest('hex') : null };
  }
  run(false); // Warm JIT paths outside the measurement.
  const start = performance.now();
  const timed = run(false);
  const milliseconds = performance.now() - start;
  const verified = run(true);
  console.log(
    JSON.stringify({
      milliseconds,
      builds: timed.builds,
      members: [...(timed.totals.members?.values() ?? [])].reduce(
        (n, rows) => n + rows.length,
        0,
      ),
      digest: verified.digest,
    }),
  );
} else {
  const baseline = option('--baseline');
  const repeats = Number(option('--repeats') ?? 5);
  const output =
    option('--output') ??
    path.join(root, '.cache/review-opt-0.0.2.26/benchmark.json');
  assert(
    baseline && Number.isInteger(repeats) && repeats > 0,
    'Use --baseline /path/to/baseline-root [--repeats 5] [--output file]',
  );
  const samples = { before: [], after: [] };
  for (let i = 0; i < repeats; i++) {
    for (const side of i % 2 ? ['after', 'before'] : ['before', 'after']) {
      const result = spawnSync(
        process.execPath,
        [
          fileURLToPath(import.meta.url),
          '--source',
          side === 'before' ? path.resolve(baseline) : root,
        ],
        { encoding: 'utf8' },
      );
      assert.equal(result.status, 0, result.stderr);
      samples[side].push(JSON.parse(result.stdout));
    }
  }
  assert.equal(
    new Set(
      Object.values(samples)
        .flat()
        .map((r) => r.digest),
    ).size,
    1,
    'Cumulative output hashes must match',
  );
  const median = (values) => {
    const sorted = values.toSorted((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[middle]
      : (sorted[middle - 1] + sorted[middle]) / 2;
  };
  const report = {
    samples,
    before_ms: median(samples.before.map((r) => r.milliseconds)),
    after_ms: median(samples.after.map((r) => r.milliseconds)),
  };
  report.speedup = report.before_ms / report.after_ms;
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
}
