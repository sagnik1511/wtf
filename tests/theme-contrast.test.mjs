import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const css = await readFile(
  new URL('../viewer/style.css', import.meta.url),
  'utf8',
);
const declarations = (block) =>
  Object.fromEntries(
    [...block.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map((m) => [
      m[1],
      m[2].trim(),
    ]),
  );
const dark = declarations(css.match(/^:root\s*\{([^}]+)\}/m)[1]);
const light = {
  ...dark,
  ...declarations(css.match(/:root\[data-theme='light'\]\s*\{([^}]+)\}/)[1]),
};
function resolve(value, palette) {
  return value.startsWith('var(')
    ? resolve(palette[value.slice(4, -1)], palette)
    : value;
}
function luminance(hex) {
  const raw = hex.slice(1);
  const s = raw.length === 3 ? [...raw].map((c) => c + c).join('') : raw;
  return [0, 2, 4]
    .map((i) => parseInt(s.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
    .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
}
function contrast(a, b) {
  const x = luminance(a),
    y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
for (const [theme, palette] of Object.entries({ dark, light }))
  test(`${theme} semantic text pairs keep readable contrast`, () => {
    const pairs = [
      ['text', 'bg'],
      ['text', 'panel'],
      ['text', 'surface'],
      ['muted', 'bg'],
      ['muted', 'panel'],
      ['muted', 'surface'],
      ['primary-text', 'mint'],
      ['accent', 'panel'],
      ['resource-label', 'bg'],
      ['obstacle-label', 'bg'],
      ...[
        'error',
        'kill',
        'birth',
        'death',
        'prey-death',
        'contest',
        'notice',
      ].map((p) => [p + '-text', p + '-bg']),
    ];
    for (const [ink, bg] of pairs) {
      const ratio = contrast(
        resolve(palette['--' + ink], palette),
        resolve(palette['--' + bg], palette),
      );
      assert(ratio >= 4.5, `${theme}: ${ink} on ${bg} = ${ratio.toFixed(2)}`);
    }
  });

test('Canvas birth, kill and both natural-death label badges contrast in both themes', async () => {
  const app = await readFile(
    new URL('../viewer/app.js', import.meta.url),
    'utf8',
  );
  const source = app.slice(
    app.indexOf('function drawDeathHighlights()'),
    app.indexOf('function drawChart()'),
  );
  for (const theme of ['dark', 'light']) {
    const labels = [];
    const ctx = {
      save() {},
      restore() {},
      beginPath() {},
      arc() {},
      fill() {},
      stroke() {},
      moveTo() {},
      lineTo() {},
      fillRect() {
        this.backdrop = this.fillStyle;
      },
      fillText(label) {
        labels.push([label, this.fillStyle, this.backdrop]);
      },
    };
    const state = {
      width: 10,
      height: 8,
      animals: [],
      species: { prey: { diet: 'resource' }, predator: { diet: 'prey' } },
      natural_death_highlights: [
        { natural: true, species: 'prey', position: [1, 1], age: 0 },
        { natural: true, species: 'predator', position: [2, 2], age: 0 },
      ],
      kill_highlights: [{ position: [3, 3], age: 0 }],
      birth_highlights: [{ position: [4, 4], age: 0 }],
    };
    vm.runInNewContext(
      source + ';drawDeathHighlights();drawBirthHighlights();',
      {
        document: { documentElement: { dataset: { theme } } },
        state,
        $: () => ({ width: 1000, height: 800, getContext: () => ctx }),
      },
    );
    assert.equal(labels.length, 4);
    for (const [label, ink, bg] of labels)
      assert(contrast(ink, bg) >= 4.5, `${theme} ${label}`);
  }
});
