/* Step-through walkthroughs (js/views/walks*.js). Every walkthrough runs under every setting; its steps point at real
 * boxes and code lines; its numbers are finite; its plot fits the 300 by 210 box with no labels on top of each other;
 * its text follows the writing rules; every symbol in a box, formula or variable name is defined in its legend, and
 * every legend entry is used. Lessons may only embed walkthroughs that exist.
 * Run: node tests/walks.test.js */
'use strict';
const { load, suite } = require('./_load.js');
const MLA = load();
MLA.content.init();
const T = suite('walks');

const names = Object.keys(MLA.walks);
T.ok('at least 12 walkthroughs are registered', names.length >= 12, String(names.length));

// Every parameter combination of a walkthrough
function combos(def) {
  let out = [{}];
  for (const p of def.params || []) out = out.flatMap(c => p.options.map(o => Object.assign({}, c, { [p.key]: o[0] })));
  return out;
}
// Rough extent of a plot shape. Text width uses about 0.55 em per character.
const FS = { 'wp-tick': 9.5, 'wp-txt': 10.5 };
function extent(sh) {
  if (sh.t === 'c') { const r = sh.r || 4; return [sh.x - r, sh.y - r, sh.x + r, sh.y + r]; }
  if (sh.t === 'l') return [Math.min(sh.x1, sh.x2), Math.min(sh.y1, sh.y2), Math.max(sh.x1, sh.x2), Math.max(sh.y1, sh.y2)];
  if (sh.t === 'r') return [sh.x, sh.y, sh.x + Math.max(0, sh.w), sh.y + Math.max(0, sh.h)];
  if (sh.t === 't') {
    const s = String(sh.s == null ? '' : sh.s);
    if (!s) return null;
    const fs = FS[String(sh.cls || '').split(' ')[0]] || 10, w = [...s].length * fs * 0.55;
    const x0 = sh.a === 'middle' ? sh.x - w / 2 : sh.a === 'end' ? sh.x - w : sh.x;
    return [x0, sh.y - fs * 0.8, x0 + w, sh.y + fs * 0.25];
  }
  return null;
}
const SUB = /[₀-ₜᵢ-ᵪⱼ]/g;            // subscripts, as the legend matcher strips them
const tokens = s => s.replace(SUB, '').match(/[\p{L}\p{M}]+/gu) || [];
const single = t => t.replace(/\p{M}/gu, '').length === 1;   // a one-letter symbol, accents allowed (ŷ, m̂)
const STYLE = / — |!|[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u;

// Your own data: generated CSV files for each kind of data a walkthrough can take, including awkward ones.
const WD = MLA.walkData, rnd = MLA.util.seededRng(42);
const gauss = () => { const u = Math.max(1e-12, rnd()), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const csv = (header, rows) => header.join(',') + String.fromCharCode(10) + rows.map(r => r.join(',')).join(String.fromCharCode(10));
const many = (n, fn) => Array.from({ length: n }, (_, i) => fn(i));
const DATA = {
  xy: [['line', csv(['id', 'size_m2', 'price'], many(80, i => { const x = 50 + 30 * gauss(); return [i + 1, x.toFixed(1), (1000 + 12 * x + 90 * gauss()).toFixed(0)]; }))],
    ['noise', csv(['a', 'b'], many(12, () => [gauss().toFixed(2), gauss().toFixed(2)]))]],
  class2: [['blobs', csv(['f1', 'f2', 'label'], many(60, i => { const c = i % 2; return [(c * 3 + gauss()).toFixed(2), (c * 2 + gauss()).toFixed(2), c ? 'yes' : 'no']; }))],
    ['overlap', csv(['f1', 'f2', 'churned'], many(60, i => [gauss().toFixed(2), gauss().toFixed(2), i % 3 ? 0 : 1]))],
    ['rare', csv(['amount', 'distance', 'fraud'], many(200, i => [(40 + 30 * gauss() + (i % 40 ? 0 : 150)).toFixed(2), (5 + 3 * gauss()).toFixed(1), i % 40 ? 0 : 1]))]],
  points2: [['correlated', csv(['height', 'weight'], many(50, () => { const t = gauss(); return [(170 + 9 * t).toFixed(1), (70 + 8 * t + 4 * gauss()).toFixed(1)]; }))],
    ['round', csv(['u', 'v'], many(40, () => [gauss().toFixed(3), gauss().toFixed(3)]))]]
};
{
  const t = WD.parseTable('a,b,"c, d"' + String.fromCharCode(10) + '1,2,"x ""y"""' + String.fromCharCode(13, 10) + '3,4,z' + String.fromCharCode(10));
  T.ok('CSV: quoted fields, doubled quotes and Windows line ends', JSON.stringify(t) === JSON.stringify({ header: ['a', 'b', 'c, d'], rows: [['1', '2', 'x "y"'], ['3', '4', 'z']] }), JSON.stringify(t));
  T.ok('TSV: tabs are detected', WD.parseTable('a\tb' + String.fromCharCode(10) + '1\t2').header.join('|') === 'a|b');
  const bad = WD.parseTable(csv(['x', 'same', 'word', 'three'], many(20, i => [i, 5, 'w' + i, ['a', 'b', 'c'][i % 3]])));
  const err = (kind, cols) => WD.prepareData(kind, bad, cols, 'bad.csv').error || '';
  T.ok('own data: a column of words is refused', /does not hold numbers/.test(err('xy', ['x', 'word'])), err('xy', ['x', 'word']));
  T.ok('own data: a constant column is refused', /same value/.test(err('xy', ['x', 'same'])), err('xy', ['x', 'same']));
  T.ok('own data: a label with three values is refused', /exactly two/.test(err('class2', ['x', 'same', 'three'])), err('class2', ['x', 'same', 'three']));
  T.ok('own data: the same column twice is refused', /different column/.test(err('points2', ['x', 'x'])), err('points2', ['x', 'x']));
  T.ok('own data: too few rows are refused', /At least 5 rows/.test(WD.prepareData('xy', WD.parseTable(csv(['a', 'b'], many(3, i => [i, i * 2]))), ['a', 'b'], 'tiny.csv').error || ''));
  const ok = WD.prepareData('class2', WD.parseTable(DATA.class2[2][1]), ['amount', 'distance', 'fraud'], 'rare.csv');
  T.ok('own data: a rare class gets a balanced sample, rescaled', !ok.error && ok.P.filter(q => q[2]).length === 5 && ok.P.length === 20 && Math.abs(ok.P.reduce((s, q) => s + q[0], 0)) < 1e-9, JSON.stringify(ok).slice(0, 200));
  const line = WD.parseTable(DATA.xy[0][1]);
  T.ok('own data: id columns are not picked by default', WD.defaultColumns('xy', WD.columnsOf(line), line).join('|') === 'size_m2|price');
}

for (const name of names) {
  const def = MLA.walks[name];
  const ids = new Set(def.nodes.map(n => n.id));
  T.ok(name + ': has a component', !!MLA.components['demo-walk-' + name]);
  T.ok(name + ': every arrow joins two boxes', def.edges.every(e => ids.has(e.from) && ids.has(e.to)));
  T.ok(name + ': every box and arrow explains itself', def.nodes.every(n => n.why && n.why.length > 20) && def.edges.every(e => e.why && e.label));
  const prose = [def.title].concat(def.nodes.map(n => n.why + ' ' + n.label.join(' ')), def.edges.map(e => e.why + ' ' + e.label), (def.symbols || []).map(s => s[1]), [def.describe()]);
  const badProse = prose.filter(s => STYLE.test(s));
  T.ok(name + ': boxes, arrows and legend follow the writing rules', !badProse.length, badProse[0]);
  const visited = new Set(), usedSymbols = new Set(), issues = [];
  const datasets = [[null, 'built-in']].concat(def.data ? (DATA[def.data.kind] || []).map(([label, text]) => {
    const t = WD.parseTable(text), d = WD.prepareData(def.data.kind, t, WD.defaultColumns(def.data.kind, WD.columnsOf(t), t), label + '.csv');
    if (d.error) issues.push(label + ': ' + d.error);
    return [d.error ? null : d, label];
  }) : []);
  for (const [data, dname] of datasets) for (const c of combos(def)) {
    const steps = def.run(c, data), tag = dname + ' ' + JSON.stringify(c);
    if (!steps.length) issues.push(tag + ' has no steps');
    steps.forEach((s, i) => {
      const at = tag + ' step ' + (i + 1) + ': ';
      visited.add(s.node);
      if (!ids.has(s.node)) issues.push(at + 'unknown box ' + s.node);
      if ((s.lines || []).some(l => l < 1 || l > def.code.length)) issues.push(at + 'code line out of range');
      // The path follows the flowchart: a named arrow exists, and a change of box without one uses a real arrow.
      const arrow = (a, b) => def.edges.some(e => e.from + '>' + e.to === a + '>' + b);
      if (s.edge && !def.edges.some(e => e.from + '>' + e.to === s.edge)) issues.push(at + 'unknown arrow ' + s.edge);
      const prev = i > 0 ? steps[i - 1].node : null;
      if (!s.edge && prev && prev !== s.node && !arrow(prev, s.node)) issues.push(at + 'jumps from ' + prev + ' to ' + s.node + ' with no arrow');
      const words = [s.say, s.formula, s.ask].concat((s.vars || []).map(v => v[0] + ' ' + v[1])).filter(Boolean).join(' ');
      if (/NaN|undefined|Infinity|null/.test(words)) issues.push(at + 'bad value in "' + words.slice(0, 80) + '"');
      if (STYLE.test([s.say, s.ask].join(' '))) issues.push(at + 'writing rules: ' + s.say.slice(0, 60));
      // The plot
      const shapes = def.draw(s.state || {}, MLA.walkKit, data);
      const keys = new Set();
      for (const sh of shapes) {
        if (!'clprt'.includes(sh.t)) issues.push(at + 'unknown shape type ' + sh.t);
        if (keys.has(sh.k)) issues.push(at + 'duplicate shape key ' + sh.k);
        keys.add(sh.k);
        for (const [k, v] of Object.entries(sh)) if (typeof v === 'number' && !isFinite(v)) issues.push(at + 'shape ' + sh.k + ' has ' + k + ' = ' + v);
        if (sh.t === 'p' && /NaN|Infinity|undefined/.test(sh.d)) issues.push(at + 'path ' + sh.k + ' is broken');
        const box = extent(sh), lim = sh.f ? [0, 0, 300, 210] : sh.t === 't' ? [38, 12, 290, 182] : null;
        if (box && lim && (box[0] < lim[0] - 1 || box[1] < lim[1] - 1 || box[2] > lim[2] + 1 || box[3] > lim[3] + 1))
          issues.push(at + 'label or shape "' + (sh.s || sh.k) + '" leaves the plot');
      }
      const texts = shapes.filter(sh => sh.t === 't').map(sh => [sh, extent(sh)]).filter(x => x[1]);
      for (let a = 0; a < texts.length; a++) for (let b = a + 1; b < texts.length; b++) {
        const A = texts[a][1], B = texts[b][1];
        if (Math.min(A[2], B[2]) - Math.max(A[0], B[0]) > 1.5 && Math.min(A[3], B[3]) - Math.max(A[1], B[1]) > 1.5)
          issues.push(at + 'labels "' + texts[a][0].s + '" and "' + texts[b][0].s + '" overlap');
      }
      // Symbols: every one-letter symbol and Greek letter on screen has a legend entry; record which entries show.
      const node = def.nodes.find(n => n.id === s.node);
      const math = [node ? node.label.join(' ') : '', s.plain ? '' : s.formula || '', (s.vars || []).map(v => v[0]).join(' ')].join('  ');
      const entries = def.symbols || [];
      for (const t of tokens(math)) {
        if (!single(t) || t === 'a' && !entries.some(e => e[0] === 'a') || t === 'A') continue;   // "a" is usually the article
        if (!entries.some(e => tokens(e[0]).includes(t))) issues.push(at + 'symbol "' + t + '" has no legend entry');
      }
      for (const ch of math.match(/[Α-ω]/g) || [])
        if (!entries.some(e => e[0].includes(ch))) issues.push(at + 'Greek letter ' + ch + ' has no legend entry');
      for (const e of def.symbolsFor(s)) usedSymbols.add(e[0]);
      // Read aloud: every step has speakable text, with no raw symbols left, and notation that only works on screen
      // (arrows or dots standing alone, as in "· → → ·") has a spoken version.
      const spoken = def.narrateStep(steps, i).split(MLA.narrate.PAUSE).join(' ');
      if (!spoken.trim()) issues.push(at + 'nothing to read aloud');
      const raw = spoken.match(/[Σ∑√∂∇‖ᵀ²³⁻ⁿ₀-₉ᵢⱼₙ←→≈≤≥×·−∞θσμλαβγδεηπφωτρκ_{}<>]|\s\+\s|times\)/);
      if (raw) issues.push(at + 'read aloud with "' + raw[0] + '" left in: ' + spoken.slice(0, 80));
      if (/(^|\s)[·→←?](\s|$)/.test(s.say) && !s.voice) issues.push(at + 'say uses arrows or dots as notation but has no voice text');
    });
  }
  T.ok(name + ': runs cleanly under every setting', !issues.length, issues.slice(0, 4).join(' | ') + (issues.length > 4 ? ' (+' + (issues.length - 4) + ' more)' : ''));
  T.ok(name + ': every box is reached by some step', def.nodes.every(n => visited.has(n.id)), def.nodes.filter(n => !visited.has(n.id)).map(n => n.id).join(', '));
  const unused = (def.symbols || []).filter(e => !usedSymbols.has(e[0])).map(e => e[0]);
  T.ok(name + ': every legend entry appears on screen', !unused.length, unused.join(', '));
}

// Lessons embed walkthroughs as demo sections named walk-<name>
const embedded = new Set();
for (const c of Object.values(MLA.content.byId)) for (const s of c.sections || []) {
  if (s.type === 'demo' && /^walk-/.test(s.demo || '')) {
    embedded.add(s.demo.slice(5));
    T.ok(c.id + ': walkthrough ' + s.demo + ' exists', !!MLA.walks[s.demo.slice(5)]);
  }
}
// The course template keeps ML Atlas's walkthroughs as examples; lessons decide which ones the learner sees.
const unused = names.filter(n => !embedded.has(n));
if (unused.length) console.log(unused.length + ' walkthroughs are not used by any lesson (examples to adapt or delete): ' + unused.join(', '));
T.done();
