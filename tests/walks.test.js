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
  for (const c of combos(def)) {
    const steps = def.run(c), tag = JSON.stringify(c);
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
      const shapes = def.draw(s.state || {}, MLA.walkKit);
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
      const math = [node ? node.label.join(' ') : '', s.formula || '', (s.vars || []).map(v => v[0]).join(' ')].join('  ');
      const entries = def.symbols || [];
      for (const t of tokens(math)) {
        if (!single(t) || t === 'a' && !entries.some(e => e[0] === 'a') || t === 'A') continue;   // "a" is usually the article
        if (!entries.some(e => tokens(e[0]).includes(t))) issues.push(at + 'symbol "' + t + '" has no legend entry');
      }
      for (const ch of math.match(/[Α-ω]/g) || [])
        if (!entries.some(e => e[0].includes(ch))) issues.push(at + 'Greek letter ' + ch + ' has no legend entry');
      for (const e of def.symbolsFor(s)) usedSymbols.add(e[0]);
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
T.ok('every walkthrough is used in a lesson', names.every(n => embedded.has(n)), names.filter(n => !embedded.has(n)).join(', '));
T.done();
