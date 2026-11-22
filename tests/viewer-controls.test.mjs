import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('All literal app control IDs exist in the shipped viewer HTML', async () => {
  const [app, html] = await Promise.all(
    ['../viewer/app.js', '../viewer/index.html'].map((p) =>
      readFile(new URL(p, import.meta.url), 'utf8'),
    ),
  );
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
  for (const [, id] of app.matchAll(/\$\('([^']+)'\)/g))
    assert(
      ids.has(id),
      `Missing #${id} prevents viewer startup or interaction`,
    );
});
