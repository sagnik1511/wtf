import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
const script = await readFile(
  new URL('../docs/javascripts/banner-theme.js', import.meta.url),
  'utf8',
);
function fixture(theme = 'default', readyState = 'complete', present = true) {
  const source = { srcset: 'assets/branding/github-banner-light.svg' };
  const image = { src: 'assets/branding/github-banner-light.png' };
  const element = (attributes) => ({
    getAttribute: (name) => attributes[name],
    setAttribute: (name, value) => {
      attributes[name] = value;
    },
  });
  const body = {
    scheme: theme,
    getAttribute() {
      return this.scheme;
    },
  };
  let update,
    options,
    loaded,
    observers = 0;
  const context = {
    document: {
      body,
      readyState,
      querySelector: () =>
        present
          ? {
              querySelector: (name) =>
                name === 'source' ? element(source) : element(image),
            }
          : null,
      addEventListener: (event, callback) => {
        assert.equal(event, 'DOMContentLoaded');
        loaded = callback;
      },
    },
    MutationObserver: class {
      constructor(callback) {
        update = callback;
        observers++;
      }
      observe(target, config) {
        assert.equal(target, body);
        options = config;
      }
    },
  };
  vm.runInNewContext(script, context);
  return {
    source,
    image,
    body,
    change(theme) {
      body.scheme = theme;
      update();
    },
    load() {
      loaded();
    },
    get observers() {
      return observers;
    },
    get options() {
      return options;
    },
  };
}
test('Single banner follows light/dark/light manual switches for SVG and PNG', () => {
  const f = fixture();
  assert.equal(f.source.srcset, 'assets/branding/github-banner-light.svg');
  for (const [palette, theme] of [
    ['slate', 'dark'],
    ['default', 'light'],
    ['slate', 'dark'],
    ['default', 'light'],
  ]) {
    f.change(palette);
    assert.equal(f.source.srcset, `assets/branding/github-banner-${theme}.svg`);
    assert.equal(f.image.src, `assets/branding/github-banner-${theme}.png`);
  }
  assert.equal(f.observers, 1);
  assert.deepEqual([...f.options.attributeFilter], ['data-md-color-scheme']);
  assert.equal(f.options.subtree, undefined);
});
test('Stored dark palette is applied on first load and deferred initialization works', () => {
  assert.equal(
    fixture('slate').source.srcset,
    'assets/branding/github-banner-dark.svg',
  );
  const f = fixture('slate', 'loading');
  assert.equal(f.observers, 0);
  f.load();
  assert.equal(f.image.src, 'assets/branding/github-banner-dark.png');
});
test('Pages without a banner create no theme observer', () => {
  assert.equal(fixture('slate', 'complete', false).observers, 0);
});
