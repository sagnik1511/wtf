// Interface preference only: never changes saved worlds, policies or replay time.
export function savedTheme(storage, fallback = 'dark') {
  try {
    const value = storage?.getItem('wtf.viewer.theme');
    if (value === 'light' || value === 'dark') return value;
  } catch {
    /* Storage may be disabled in private/embedded browsers. */
  }
  return fallback;
}
export function installTheme({ root, button, storage, onChange = () => {} }) {
  let theme = savedTheme(storage);
  function apply() {
    root.dataset.theme = theme;
    button.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
    button.setAttribute(
      'aria-label',
      `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`,
    );
    button.setAttribute(
      'title',
      'Interface appearance; independent of simulated day and night',
    );
  }
  apply();
  button.onclick = () => {
    theme = theme === 'dark' ? 'light' : 'dark';
    apply();
    try {
      storage?.setItem('wtf.viewer.theme', theme);
    } catch {
      /* Preference is still applied for this page. */
    }
    onChange(theme);
  };
}

export function boardPalette(theme) {
  return theme === 'light'
    ? {
        light: true,
        background: '#dce6d8',
        grid: '#647b6738',
        obstacle: '#6a7d89',
        selection: '#203d50',
      }
    : {
        light: false,
        background: '#101a22',
        grid: '#172630',
        obstacle: '#3a4b5c',
        selection: '#ffffff',
      };
}
