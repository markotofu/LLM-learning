/* UI services: toasts, modals, icons */
(function (global) {
  const MLA = global.MLA;
  const reactive = (global.Vue && global.Vue.reactive) ? global.Vue.reactive : (x => x);

  const toasts = reactive([]);
  function toast(msg, kind, ms) {
    const id = MLA.util.uid('t');
    toasts.push({ id, msg, kind: kind || 'info' });
    if (toasts.length > 5) toasts.splice(0, toasts.length - 5);
    setTimeout(() => { const i = toasts.findIndex(t => t.id === id); if (i >= 0) toasts.splice(i, 1); }, ms || 3200);
  }

  // One modal at a time. Body: markdown, html, or a registered component + props.
  const modal = reactive({ open: false, title: '', html: '', component: null, props: null, actions: [], wide: false, dismissible: true });
  let resolver = null;
  function openModal(opts) {
    closeModal(null);
    modal.open = true;
    modal.title = opts.title || '';
    modal.html = opts.html || (opts.md ? MLA.util.md(opts.md) : '');
    modal.component = opts.component || null;
    modal.props = opts.props || null;
    modal.actions = opts.actions || [{ label: 'Close', kind: '', value: null }];
    modal.wide = !!opts.wide;
    modal.dismissible = opts.dismissible !== false;
    return new Promise(res => { resolver = res; });
  }
  function closeModal(value) {
    if (!modal.open) return;
    modal.open = false; modal.component = null; modal.props = null;
    const r = resolver; resolver = null;
    if (r) r(value);
  }
  function confirm(message, opts) {
    opts = opts || {};
    return openModal({ title: opts.title || 'Please confirm', md: message, actions: [
      { label: opts.cancelLabel || 'Cancel', kind: '', value: false },
      { label: opts.okLabel || 'Confirm', kind: opts.danger ? 'danger' : 'primary', value: true }
    ] });
  }

  // Inline SVG icons (24x24, stroke based)
  const icons = {
    dashboard: '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    map: '<circle cx="12" cy="5" r="2.2"/><circle cx="5" cy="12" r="2.2"/><circle cx="19" cy="12" r="2.2"/><circle cx="8" cy="19.5" r="2.2"/><circle cx="16" cy="19.5" r="2.2"/><path d="M10.4 6.6 6.6 10.4M13.6 6.6l3.8 3.8M6.3 14l1.1 3.5M17.7 14l-1.1 3.5"/>',
    route: '<circle cx="6" cy="19" r="2.5"/><circle cx="18" cy="5" r="2.5"/><path d="M8.5 19H15a3.5 3.5 0 0 0 0-7H9a3.5 3.5 0 0 1 0-7h6.5"/>',
    code: '<path d="m8 7-5 5 5 5M16 7l5 5-5 5M14 4l-4 16"/>',
    terminal: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3M12 15h5"/>',
    cards: '<rect x="3" y="6" width="14" height="14" rx="2"/><path d="M7 3h14v14"/>',
    book: '<path d="M4 4h7a3 3 0 0 1 3 3v13a2 2 0 0 0-2-2H4V4z"/><path d="M20 4h-7a3 3 0 0 0-3 3v13a2 2 0 0 1 2-2h8V4z"/>',
    trophy: '<path d="M8 4h8v4a4 4 0 0 1-8 0V4z"/><path d="M8 5H5v2a3 3 0 0 0 3 3M16 5h3v2a3 3 0 0 1-3 3"/><path d="M12 12v4M9 20h6M10 16h4v4h-4z"/>',
    sync: '<path d="M20 11a8 8 0 0 0-14.3-4.9L4 8"/><path d="M4 4v4h4"/><path d="M4 13a8 8 0 0 0 14.3 4.9L20 16"/><path d="M20 20v-4h-4"/>',
    user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3 3 3 0 0 0 3-3V7a3 3 0 0 0-3-3z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3 3 3 0 0 1-3-3"/><path d="M9 9h1M14 9h1M9 14h1.5M13.5 14H15"/>',
    check: '<path d="m5 12 5 5L20 7"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    play: '<path d="M7 5v14l11-7z"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    send: '<path d="M4 12 20 4l-6 16-3-7-7-1z"/>',
    star: '<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
    flame: '<path d="M12 3s5 4 5 10a5 5 0 0 1-10 0c0-2 1-3.5 2-4.5 0 2 1 3 2 3 0-3 1-6 1-8.5z"/>',
    chevron: '<path d="m9 6 6 6-6 6"/>',
    chevronLeft: '<path d="m15 6-6 6 6 6"/>',
    chevronDown: '<path d="m6 9 6 6 6-6"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    minus: '<path d="M5 12h14"/>',
    trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    download: '<path d="M12 3v12M7 10l5 5 5-5M4 21h16"/>',
    upload: '<path d="M12 21V9M7 14l5-5 5 5M4 3h16"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
    file: '<path d="M6 3h8l5 5v13H6z"/><path d="M14 3v5h5"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    warning: '<path d="M12 3 2 21h20L12 3z"/><path d="M12 10v4M12 18h.01"/>',
    note: '<path d="M5 4h10l4 4v12H5z"/><path d="M15 4v4h4M8 13h8M8 17h5"/>',
    bookmark: '<path d="M6 3h12v18l-6-4-6 4z"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    arrowLeft: '<path d="M19 12H5M12 19l-7-7 7-7"/>',
    arrowRight: '<path d="M5 12h14M12 5l7 7-7 7"/>',
    refresh: '<path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 4v5h-5"/>',
    menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
    external: '<path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>',
    bolt: '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8z"/>',
    clipboard: '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V3h6v1M9 10h6M9 14h6M9 18h3"/>',
    hint: '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.6 1 1.3 1 2.5h6c0-1.2.3-1.9 1-2.5A6 6 0 0 0 12 3z"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    graph: '<path d="M4 20h16M6 16l4-5 4 3 5-7"/>',
    layers: '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5M3 17l9 5 9-5"/>',
    list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    tree: '<rect x="9" y="3" width="6" height="4" rx="1"/><rect x="3" y="17" width="6" height="4" rx="1"/><rect x="15" y="17" width="6" height="4" rx="1"/><path d="M12 7v5M6 17v-2a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v2"/>',
    radial: '<circle cx="12" cy="12" r="2.5"/><circle cx="12" cy="12" r="8" stroke-dasharray="2 3"/><circle cx="12" cy="4" r="1.6"/><circle cx="19" cy="15" r="1.6"/><circle cx="5" cy="15" r="1.6"/>',
    fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    sparkles: '<path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z"/>',
    flag: '<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>',
    save: '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v5h8V3M8 21v-7h8v7"/>',
    keyboard: '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/>',
    eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1l2-1.6-2-3.4-2.4.9a7 7 0 0 0-1.7-1L14.4 3h-4.8l-.4 2.5a7 7 0 0 0-1.7 1L5.1 6 3.1 9.4l2 1.6a7 7 0 0 0 0 2l-2 1.6 2 3.4 2.4-.9a7 7 0 0 0 1.7 1l.4 2.5h4.8l.4-2.5a7 7 0 0 0 1.7-1l2.4.9 2-3.4-2-1.6c.1-.3.1-.7.1-1z"/>',
    flask: '<path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2.2h12.4a1.5 1.5 0 0 0 1.3-2.2L14 9V3"/><path d="M7 15h10"/>',
    palette: '<path d="M12 3a9 9 0 1 0 0 18c1.2 0 1.9-.9 1.9-1.9 0-.5-.2-.9-.5-1.3-.3-.3-.5-.7-.5-1.2 0-1 .8-1.8 1.8-1.8H17a4 4 0 0 0 4-4C21 6.7 17 3 12 3z"/><circle cx="7.5" cy="11" r="1.2"/><circle cx="10.5" cy="7" r="1.2"/><circle cx="15" cy="7.5" r="1.2"/>',
    headphones: '<path d="M4 15v-3a8 8 0 0 1 16 0v3"/><rect x="3" y="14" width="4.5" height="7" rx="1.5"/><rect x="16.5" y="14" width="4.5" height="7" rx="1.5"/>',
    volume: '<path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12"/>',
    pause: '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>',
    skipBack: '<path d="M6 5v14"/><path d="M19 5 9 12l10 7z"/>',
    skipForward: '<path d="M18 5v14"/><path d="M5 5l10 7-10 7z"/>',
    skip: '<path d="m5 7 5 5-5 5M13 7l5 5-5 5"/>',
    dice: '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h.01M15 9h.01M12 12h.01M9 15h.01M15 15h.01"/>'
  };
  function iconSvg(name, size) {
    const p = icons[name] || icons.info;
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="' + (size || 18) + '" height="' + (size || 18) + '" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + p + '</svg>';
  }

  // t: 'light' | 'dark' | 'darker'; style: 'baroque' (default) | 'classic'. The colours match --bg in css/app.css.
  const THEME_BG = { baroque: { light: '#f4ecdc', dark: '#16110b', darker: '#0a1124' }, classic: { light: '#f6f7f9', dark: '#0f1012', darker: '#000000' } };
  function applyTheme(t, style) {
    if (!global.document) return;
    const root = document.documentElement;
    const look = style === 'classic' ? 'classic' : 'baroque', mode = t === 'dark' || t === 'darker' ? t : 'light';
    if (mode !== 'light') root.setAttribute('data-theme', mode); else root.removeAttribute('data-theme');
    if (look === 'classic') root.setAttribute('data-style', 'classic'); else root.removeAttribute('data-style');
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta && meta.setAttribute) meta.setAttribute('content', THEME_BG[look][mode]);
  }

  MLA.ui = { toasts, toast, modal, openModal, closeModal, confirm, icons, iconSvg, applyTheme };
})(typeof globalThis !== 'undefined' ? globalThis : window);
