// Keep one banner in the DOM; follow the palette used by the site's manual toggle.
(() => {
  function initializeBanner() {
    const banner = document.querySelector('[data-theme-banner]');
    if (!banner) return;
    const source = banner.querySelector('source');
    const image = banner.querySelector('img');
    if (!source || !image) return;
    const lightSVG = source.getAttribute('srcset');
    const lightPNG = image.getAttribute('src');
    let selected;
    function updateBanner() {
      const theme =
        document.body.getAttribute('data-md-color-scheme') === 'slate'
          ? 'dark'
          : 'light';
      if (selected === theme) return;
      selected = theme;
      source.setAttribute(
        'srcset',
        lightSVG.replace('-light.svg', `-${theme}.svg`),
      );
      image.setAttribute(
        'src',
        lightPNG.replace('-light.png', `-${theme}.png`),
      );
    }
    updateBanner();
    new MutationObserver(updateBanner).observe(document.body, {
      attributes: true,
      attributeFilter: ['data-md-color-scheme'],
    });
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', initializeBanner, {
      once: true,
    });
  else initializeBanner();
})();
