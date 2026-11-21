/* Theme contrast: reads the colour tokens of every theme from css/app.css and checks them against
 * WCAG 2.x contrast ratios, so a palette change cannot quietly make text unreadable.
 * Run: node tests/theme.test.js */
'use strict';
const fs = require('fs');
const path = require('path');
const { suite, root } = require('./_load.js');
const T = suite('theme');

const css = fs.readFileSync(path.join(root, 'css/app.css'), 'utf8');
function block(selector) {
  const i = css.indexOf(selector + ' {');
  if (i < 0) throw new Error('theme block not found: ' + selector);
  const body = css.slice(i, css.indexOf('}', i));
  const out = {};
  for (const m of body.matchAll(/--([a-z0-9-]+):\s*([^;]+);/g)) out[m[1]] = m[2].trim();
  return out;
}
const base = block(':root');
const themes = {
  fresco: base,
  vault: Object.assign({}, base, block(':root[data-theme="dark"]')),
  lapis: Object.assign({}, base, block(':root[data-theme="darker"]')),
};

function rgb(c) {
  let m = /^#([0-9a-f]{6})$/i.exec(c);
  if (m) return [0, 2, 4].map(i => parseInt(m[1].slice(i, i + 2), 16)).concat(1);
  m = /^rgba?\(\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\s*\)$/i.exec(c);
  if (m) return [+m[1], +m[2], +m[3], m[4] == null ? 1 : +m[4]];
  throw new Error('unsupported colour: ' + c);
}
// Composite a (possibly translucent) colour over an opaque background.
const over = (fg, bg) => { const a = fg[3]; return [0, 1, 2].map(i => fg[i] * a + bg[i] * (1 - a)).concat(1); };
const lum = c => { const v = c.slice(0, 3).map(x => x / 255).map(x => x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4)); return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; };
const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };

const TEXT = ['text', 'muted', 'dim', 'accent2', 'green', 'red', 'amber', 'purple', 'cyan', 'pink'];
const BACKGROUNDS = ['bg', 'bg2', 'panel', 'panel2', 'panel3'];
const SOFT = { green: 'green-soft', red: 'red-soft', amber: 'amber-soft', purple: 'purple-soft', cyan: 'cyan-soft', accent2: 'accent-soft' };
const TOKENS = ['tk-kw', 'tk-str', 'tk-num', 'tk-com', 'tk-fn', 'tk-cls', 'tk-self', 'tk-bi', 'tk-dec', 'code-text'];

for (const [name, t] of Object.entries(themes)) {
  for (const k of TEXT.concat(BACKGROUNDS, Object.values(SOFT), TOKENS, ['accent', 'on-accent', 'on-status', 'editor-bg'])) T.ok(name + ': --' + k + ' is defined', !!t[k]);
  // 1. Every text colour on every background: AA for normal text.
  for (const fg of TEXT) for (const bg of BACKGROUNDS) {
    const r = ratio(rgb(t[fg]), rgb(t[bg]));
    T.ok(name + ': --' + fg + ' on --' + bg + ' ≥ 4.5', r >= 4.5, r.toFixed(2));
  }
  // 2. Status and accent text on its own tinted ("soft") background, e.g. chips and feedback boxes.
  for (const [fg, soft] of Object.entries(SOFT)) for (const bg of ['panel', 'panel2']) {
    const r = ratio(rgb(t[fg]), over(rgb(t[soft]), rgb(t[bg])));
    T.ok(name + ': --' + fg + ' on --' + soft + ' over --' + bg + ' ≥ 4.5', r >= 4.5, r.toFixed(2));
  }
  // 3. Text drawn on solid accent and status fills (buttons, status dots, map markers).
  T.ok(name + ': --on-accent on --accent ≥ 4.5', ratio(rgb(t['on-accent']), rgb(t.accent)) >= 4.5, ratio(rgb(t['on-accent']), rgb(t.accent)).toFixed(2));
  for (const fill of ['green', 'amber', 'accent']) {
    const r = ratio(rgb(t['on-status']), rgb(t[fill]));
    T.ok(name + ': --on-status on --' + fill + ' ≥ 4.5', r >= 4.5, r.toFixed(2));
  }
  // 4. Syntax-highlighting colours in the code editor (comments explain the code, so they count too).
  for (const tk of TOKENS) {
    const r = ratio(rgb(t[tk]), rgb(t['editor-bg']));
    T.ok(name + ': --' + tk + ' on --editor-bg ≥ 4.5', r >= 4.5, r.toFixed(2));
  }
  // 5. The accent must stay distinguishable from the status colours it sits next to on the map.
  for (const st of ['green', 'amber', 'red']) {
    const a = rgb(t.accent), b = rgb(t[st]);
    const hue = c => { const [r, g, bl] = c.slice(0, 3).map(x => x / 255); const mx = Math.max(r, g, bl), mn = Math.min(r, g, bl), d = mx - mn; if (!d) return 0; let h = mx === r ? ((g - bl) / d) % 6 : mx === g ? (bl - r) / d + 2 : (r - g) / d + 4; return (h * 60 + 360) % 360; };
    const dh = Math.abs(hue(a) - hue(b)); const gap = Math.min(dh, 360 - dh);
    T.ok(name + ': accent hue is ≥ 40° away from --' + st, gap >= 40, gap.toFixed(0) + '°');
  }
}

// The colours outside the stylesheet follow the default theme.
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.webmanifest'), 'utf8'));
T.ok('index.html theme-color matches --bg', html.includes('<meta name="theme-color" content="' + base.bg + '">'));
T.ok('manifest theme_color matches --bg', manifest.theme_color === base.bg && manifest.background_color === base.bg);

T.done();
