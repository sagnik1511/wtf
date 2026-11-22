import test from 'node:test';
import assert from 'node:assert/strict';
import { savedTheme, installTheme } from '../viewer/theme.mjs';
function fixture(storage) {
  const root = { dataset: {} },
    button = {
      attributes: {},
      setAttribute(k, v) {
        this.attributes[k] = v;
      },
    };
  const changes = [];
  installTheme({
    root,
    button,
    storage,
    onChange: (theme) => changes.push(theme),
  });
  return { root, button, changes };
}
test('theme restores a saved preference and toggles without touching replay state', () => {
  const saved = new Map([['wtf.viewer.theme', 'light']]);
  const storage = {
    getItem: (k) => saved.get(k),
    setItem: (k, v) => saved.set(k, v),
  };
  const { root, button, changes } = fixture(storage);
  assert.equal(root.dataset.theme, 'light');
  assert.equal(button.textContent, 'Dark mode');
  assert.deepEqual(changes, []);
  button.onclick();
  assert.equal(root.dataset.theme, 'dark');
  assert.equal(saved.get('wtf.viewer.theme'), 'dark');
  assert.equal(button.attributes['aria-label'], 'Switch to light mode');
  button.onclick();
  assert.equal(root.dataset.theme, 'light');
  assert.deepEqual(changes, ['dark', 'light']);
});
test('unavailable storage still permits both interface modes', () => {
  const denied = {
    getItem() {
      throw Error('denied');
    },
    setItem() {
      throw Error('denied');
    },
  };
  const { root, button } = fixture(denied);
  assert.equal(root.dataset.theme, 'dark');
  button.onclick();
  assert.equal(root.dataset.theme, 'light');
  button.onclick();
  assert.equal(root.dataset.theme, 'dark');
});
test('invalid stored themes use the existing dark default', () => {
  assert.equal(savedTheme({ getItem: () => '<invalid>' }), 'dark');
  assert.equal(savedTheme(undefined), 'dark');
});

test('light grid changes ground/grid/obstacles without changing dark defaults', async () => {
  const { boardPalette } = await import('../viewer/theme.mjs');
  const dark = boardPalette('dark'),
    light = boardPalette('light');
  assert.equal(dark.background, '#101a22');
  assert.notEqual(light.background, dark.background);
  assert.notEqual(light.grid, dark.grid);
  assert.notEqual(light.obstacle, dark.obstacle);
  assert.deepEqual(boardPalette(undefined), dark);
});

test('actual app theme handler repaints a loaded paused board and events without moving the replay', async () => {
  const { readFile } = await import('node:fs/promises');
  const vm = await import('node:vm');
  const { boardPalette } = await import('../viewer/theme.mjs');
  const source = await readFile(
    new URL('../viewer/app.js', import.meta.url),
    'utf8',
  );
  const start = source.lastIndexOf('installTheme({');
  const end = source.indexOf('operation(refreshCatalog', start);
  assert.ok(start >= 0 && end > start);
  const root = { dataset: {} },
    button = { setAttribute() {} };
  const modal = { open: false },
    paints = [];
  const state = { tick: 123, frame_index: 123, selected: 7 };
  const context = {
    installTheme,
    document: { documentElement: root },
    themeStorage: undefined,
    $: (id) => (id === 'theme-toggle' ? button : modal),
    state,
    drawWorld() {
      paints.push([
        'board',
        boardPalette(root.dataset.theme).background,
        state.tick,
      ]);
    },
    drawBirthHighlights() {
      paints.push(['birth']);
    },
    drawDeathHighlights() {
      paints.push(['death']);
    },
    drawContestHighlights() {
      paints.push(['contest']);
    },
    drawChart() {
      paints.push(['chart']);
    },
    dashboard: {
      draw() {
        paints.push(['dashboard']);
      },
    },
  };
  vm.runInNewContext(source.slice(start, end), context);
  button.onclick();
  assert.deepEqual(
    paints.map((p) => p[0]),
    ['board', 'birth', 'death', 'contest', 'chart'],
  );
  assert.equal(paints[0][1], boardPalette('light').background);
  assert.deepEqual(state, { tick: 123, frame_index: 123, selected: 7 });
  paints.length = 0;
  modal.open = true;
  button.onclick();
  assert.deepEqual(
    paints.map((p) => p[0]),
    ['board', 'birth', 'death', 'contest', 'chart', 'dashboard'],
  );
  assert.equal(paints[0][1], boardPalette('dark').background);
  paints.length = 0;
  context.state = null;
  modal.open = false;
  button.onclick();
  assert.deepEqual(paints, []);
});

test('inspector semantic colors have contrasting light variants and exact dark fallbacks', async () => {
  const { readFile } = await import('node:fs/promises');
  const { visualColor } = await import('../viewer/theme.mjs');
  const { TARGET_COLORS, ACTION_COLORS } = await import(
    '../viewer/inspector-stats.mjs'
  );
  const css = await readFile(
    new URL('../viewer/style.css', import.meta.url),
    'utf8',
  );
  const light = css.match(/:root\[data-theme='light'\]\s*\{([^}]+)\}/)[1];
  const colors = new Map(
    [...light.matchAll(/--viz-([\da-f]+):\s*(#[\da-f]+);/g)].map((m) => [
      m[1],
      m[2],
    ]),
  );
  const luminance = (hex) => {
    const rgb = [1, 3, 5]
      .map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
  };
  for (const color of [...Object.values(TARGET_COLORS), ...ACTION_COLORS]) {
    const [, key, fallback] = color.match(
      /^var\(--viz-([\da-f]+), (#[\da-f]+)\)$/,
    );
    assert.equal(color, visualColor(fallback));
    const mapped = colors.get(key);
    assert.ok(mapped, `missing light variant: ${fallback}`);
    const contrast = (luminance('#eaf0f5') + 0.05) / (luminance(mapped) + 0.05);
    assert.ok(contrast >= 3, `${fallback} light chart contrast: ${contrast}`);
  }
});
