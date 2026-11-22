import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(
  new URL('../viewer/app.js', import.meta.url),
  'utf8',
);
function fixture() {
  const elements = new Map();
  const $ = (id) => {
    if (!elements.has(id))
      elements.set(id, {
        textContent: id === 'connection' ? 'Recorded playback' : '',
        attributes: {},
        setAttribute(key, value) {
          this.attributes[key] = value;
        },
      });
    return elements.get(id);
  };
  const errors = [];
  const context = {
    $,
    busy: false,
    state: null,
    selected: null,
    pause() {},
    showError(e) {
      errors.push(e);
    },
    updateRunButtons() {},
    selectedTracking: () => null,
  };
  vm.createContext(context);
  vm.runInContext(
    source.slice(
      source.indexOf('function setRecordingLoading('),
      source.indexOf('function speciesColor('),
    ),
    context,
  );
  return { context, $, errors };
}
function pending() {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
function assertSymbol($, motion) {
  for (const theme of ['light', 'dark'])
    assert.equal(
      $(`brand-source-${theme}`).attributes.srcset,
      `/branding/wtf-symbol-${motion ? 'motion-' : ''}${theme}.svg`,
    );
  assert.equal($('world').attributes['aria-busy'], String(motion));
}

test('recording load animates until ready, then restores both static theme sources', async () => {
  const { context, $ } = fixture();
  const wait = pending();
  const loading = context.operation(
    async () => {
      await wait.promise;
      $('connection').textContent = 'Recorded playback · train';
    },
    { recording: true },
  );
  assertSymbol($, true);
  assert.equal($('connection').textContent, 'Loading recording…');
  assert.equal($('load').disabled, true);
  let duplicate = false;
  await context.operation(
    () => {
      duplicate = true;
    },
    { recording: true },
  );
  assert.equal(duplicate, false);
  assertSymbol($, true);
  wait.resolve();
  await loading;
  assertSymbol($, false);
  assert.equal($('connection').textContent, 'Recorded playback · train');
  assert.equal($('load').disabled, false);
});

test('failed loads restore static branding and previous status', async () => {
  const { context, $, errors } = fixture();
  const error = new Error('Recording or chunk not found');
  await context.operation(
    async () => {
      throw error;
    },
    { recording: true },
  );
  assertSymbol($, false);
  assert.equal($('connection').textContent, 'Recorded playback');
  assert.deepEqual(errors, [error]);
  assert.equal(context.busy, false);
});

test('playback and seeking do not enable loading animation while waiting for data', async () => {
  const { context, $ } = fixture();
  context.setRecordingLoading(false);
  const wait = pending();
  const stepping = context.operation(() => wait.promise);
  assertSymbol($, false);
  assert.equal($('connection').textContent, 'Recorded playback');
  wait.resolve();
  await stepping;
  assertSymbol($, false);
});

test('loading symbols keep a static reduced-motion fallback without script or raster frames', async () => {
  for (const theme of ['light', 'dark']) {
    const svg = await readFile(
      new URL(
        `../viewer/branding/wtf-symbol-motion-${theme}.svg`,
        import.meta.url,
      ),
      'utf8',
    );
    assert.match(svg, /\.ribbon-motion\{display:none\}/);
    assert.match(svg, /@media\(prefers-reduced-motion:no-preference\)/);
    assert.match(svg, /class="ribbon-static"/);
    assert.doesNotMatch(
      svg,
      /<script|<image|animateTransform|prefers-color-scheme/,
    );
  }
});
