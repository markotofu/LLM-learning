/* A small built-in accessibility check for the browser tests (no extra packages). It runs inside the page and returns
 * readable problems: controls without an accessible name, links without an address (which the keyboard cannot reach),
 * form fields without a label, pictures without a text
 * alternative, duplicate ids and references to missing ids, a missing document language, main region or top heading,
 * and keyboard-focusable elements that a screen reader could not name. Colour contrast is checked separately, for every
 * theme, by tests/theme.test.js. */
'use strict';

function audit() {
  const out = [];
  const shown = el => {
    if (!el || !el.getClientRects || !el.getClientRects().length) return false;
    const cs = getComputedStyle(el);
    return cs.visibility !== 'hidden' && cs.display !== 'none';
  };
  const text = el => (el.textContent || '').replace(/\s+/g, ' ').trim();
  const labelOf = el => {
    const aria = (el.getAttribute('aria-label') || '').trim();
    if (aria) return aria;
    const by = el.getAttribute('aria-labelledby');
    if (by) { const t = by.split(/\s+/).map(id => document.getElementById(id)).filter(Boolean).map(text).join(' ').trim(); if (t) return t; }
    if (el.id) { const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]'); if (l && text(l)) return text(l); }
    const wrap = el.closest('label');
    if (wrap && text(wrap)) return text(wrap);
    return (el.getAttribute('title') || '').trim();
  };
  const nameOf = el => labelOf(el) || text(el) || [...el.querySelectorAll('img[alt]')].map(i => i.alt).join(' ').trim() ||
    (el.querySelector('svg title') ? text(el.querySelector('svg title')) : '') || (/^(button|submit|reset)$/.test(el.type || '') ? el.value || '' : '');
  const what = el => el.tagName.toLowerCase() + (el.getAttribute('class') ? '.' + el.getAttribute('class').trim().split(/\s+/)[0] : '') + (el.id ? '#' + el.id : '');

  const controls = 'button, a[href], [role="button"], [role="link"], [role="tab"], [role="checkbox"], [role="radio"], [role="switch"], [role="menuitem"], summary';
  document.querySelectorAll(controls).forEach(el => { if (shown(el) && !nameOf(el)) out.push('control without a name: ' + what(el)); });
  document.querySelectorAll('input:not([type="hidden"]):not([type="button"]):not([type="submit"]):not([type="reset"]), select, textarea').forEach(el => {
    if (shown(el) && !labelOf(el)) out.push('form field without a label: ' + what(el) + (el.type ? '[' + el.type + ']' : ''));
  });
  // A link without an address is not a link: the keyboard cannot reach it and screen readers do not announce it.
  document.querySelectorAll('a:not([href])').forEach(el => { if (shown(el)) out.push('link without an address: ' + what(el) + ' "' + text(el).slice(0, 30) + '"'); });
  document.querySelectorAll('[tabindex]:not([tabindex="-1"])').forEach(el => {
    if (!shown(el) || el.matches(controls + ', input, select, textarea')) return;
    if (!labelOf(el) && !(el.getAttribute('role') && text(el))) out.push('focusable element without a name: ' + what(el));
  });
  document.querySelectorAll('img').forEach(el => { if (!el.hasAttribute('alt')) out.push('image without alt text: ' + (el.getAttribute('src') || '').split('/').pop()); });
  document.querySelectorAll('svg[role="img"], [role="img"]:not(svg)').forEach(el => {
    if (!el.getAttribute('aria-label') && !el.getAttribute('aria-labelledby') && !(el.tagName.toLowerCase() === 'svg' && el.querySelector('title'))) out.push('picture without a text alternative: ' + what(el));
  });
  const ids = {};
  document.querySelectorAll('[id]').forEach(el => { ids[el.id] = (ids[el.id] || 0) + 1; });
  Object.keys(ids).filter(k => ids[k] > 1).forEach(k => out.push('duplicate id: ' + k));
  document.querySelectorAll('[aria-labelledby], [aria-describedby], [aria-controls], label[for]').forEach(el => {
    ['aria-labelledby', 'aria-describedby', 'aria-controls', 'for'].forEach(a => (el.getAttribute(a) || '').split(/\s+/).filter(Boolean)
      .forEach(id => { if (!document.getElementById(id)) out.push(a + ' points to a missing id: ' + id); }));
  });
  if (!document.documentElement.getAttribute('lang')) out.push('the page has no language (html lang)');
  if (document.querySelectorAll('main, [role="main"]').length !== 1) out.push('the page should have exactly one main region');
  if (![...document.querySelectorAll('h1')].some(shown)) out.push('the page has no visible h1 heading');
  return [...new Set(out)];
}

module.exports = { audit };
