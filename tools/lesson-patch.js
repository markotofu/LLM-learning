#!/usr/bin/env node
/* Applies lesson patches written in a plain-text format, so new sections (markdown, Python, quiz JSON) never need
 * hand-escaping inside the JavaScript content files. Usage: node tools/lesson-patch.js patch.txt [more.txt …]
 *
 *   ### INSERT concept=knn after="Scaling is not optional"      (or before="…", or at=end)
 *   ### REPLACE concept=knn section="The algorithm"             (the blocks below replace that section)
 *   @@ text | Section title
 *   markdown …
 *   @@ code | Section title
 *   optional markdown intro …
 *   %% code
 *   python …
 *   %% after                                                   (optional markdown shown after the output)
 *   %% predict                                                 (optional "predict first" question, JSON)
 *   @@ check                                                   (a quick check, JSON on the following lines)
 *   @@ demo | Section title | demo-name
 *   @@ quiz                                                    (adds a mastery-quiz question, JSON)
 *
 * Lines starting with "//" directly after a ### line are comments. Text is inserted verbatim (dedent is applied
 * by the app). Run node --check on the content files afterwards; the tests do the rest. */
'use strict';
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const BS = String.fromCharCode(92), BT = String.fromCharCode(96);

const tl = s => BT + String(s).replace(/\\/g, BS + BS).replace(/`/g, BS + BT).replace(/\$\{/g, BS + '${') + BT;
function sectionSrc(s) {
  const p = ["type: '" + s.type + "'"];
  if (s.title) p.push('title: ' + JSON.stringify(s.title));
  if (s.demo) p.push("demo: '" + s.demo + "'");
  if (s.lang) p.push("lang: '" + s.lang + "'");
  if (s.run === false) p.push('run: false');
  if (s.md) p.push('md: ' + tl(s.md));
  if (s.code) p.push('code: ' + tl(s.code));
  if (s.after) p.push('after: ' + tl(s.after));
  if (s.question) p.push('question: ' + JSON.stringify(s.question));
  if (s.predict) p.push('predict: ' + JSON.stringify(s.predict));
  if (s.chart) p.push('chart: ' + JSON.stringify(s.chart));
  return '{ ' + p.join(', ') + ' }';
}
function match(src, open) {
  const pair = { '{': '}', '[': ']' }[src[open]]; let depth = 0, i = open;
  while (i < src.length) {
    const ch = src[i];
    if (ch === "'" || ch === '"' || ch === BT) { const q = ch; i++; while (i < src.length && src[i] !== q) { if (src[i] === BS) i++; i++; } }
    else if (ch === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') i++; }
    else if (ch === src[open]) depth++;
    else if (ch === pair) { depth--; if (depth === 0) return i; }
    i++;
  }
  throw new Error('unbalanced ' + src[open] + ' at ' + open);
}
function findConcept(id) {
  for (const f of fs.readdirSync(path.join(root, 'js/data')).filter(f => /^concepts-.*\.js$/.test(f))) {
    const file = path.join(root, 'js/data', f), src = fs.readFileSync(file, 'utf8');
    const at = src.indexOf("id: '" + id + "'");
    if (at < 0) continue;
    const open = src.lastIndexOf('{', at);
    return { file, src, open, close: match(src, open) };
  }
  throw new Error('concept not found: ' + id);
}
// The array that follows "key: [" inside a concept, as [openIndex, closeIndex].
function arrayIn(src, from, to, key) {
  const re = new RegExp('\\b' + key + ':\\s*\\[', 'g'); re.lastIndex = from; const m = re.exec(src);
  if (!m || m.index > to) throw new Error('no ' + key + ' array');
  const open = m.index + m[0].length - 1; return [open, match(src, open)];
}
// Top-level objects inside an array: [{start, end}] (end inclusive).
function items(src, open, close) {
  const out = []; let i = open + 1;
  while (i < close) {
    if (src[i] === '{') { const e = match(src, i); out.push({ start: i, end: e }); i = e + 1; }
    else if (src[i] === '/' && src[i + 1] === '/') { while (i < close && src[i] !== '\n') i++; }
    else i++;
  }
  return out;
}
const titleOf = s => { const m = /^\{\s*type:\s*'[a-z]+',\s*title:\s*(['"])((?:\\.|(?!\1).)*)\1/.exec(s); return m ? m[2].replace(/\\'/g, "'").replace(/\\"/g, '"') : null; };

function parse(text) {
  const ops = []; let op = null, sec = null, field = null;
  const flush = () => { if (sec) { for (const k of Object.keys(sec)) if (typeof sec[k] === 'string') sec[k] = sec[k].replace(/^\n+|\s+$/g, ''); if (sec._json) { const j = JSON.parse(sec._json); delete sec._json; if (sec.type === 'check') sec.question = j; else if (sec.type === 'quiz') sec.question = j; } if (sec._predict) { sec.predict = JSON.parse(sec._predict); delete sec._predict; } if (sec._spec) { sec.chart = JSON.parse(sec._spec); delete sec._spec; } if (sec.md === '') delete sec.md; if (sec.after === '') delete sec.after; op.sections.push(sec); } sec = null; field = null; };
  for (const line of text.replace(/\r\n/g, '\n').split('\n')) {
    let m;
    if ((m = /^### (INSERT|REPLACE) (.*)$/.exec(line))) {
      flush(); const attrs = {}; for (const a of m[2].matchAll(/(\w+)=("([^"]*)"|\S+)/g)) attrs[a[1]] = a[3] != null ? a[3] : a[2];
      op = { kind: m[1], attrs, sections: [] }; ops.push(op); continue;
    }
    if (!op) continue;
    if (!sec && /^\/\//.test(line)) continue;
    if ((m = /^@@ (text|code|check|demo|quiz|chart)(?: \| (.*))?$/.exec(line))) {
      flush(); const parts = (m[2] || '').split(' | ');
      sec = { type: m[1], md: '' }; if (parts[0]) sec.title = parts[0].trim();
      if (m[1] === 'demo') sec.demo = (parts[1] || '').trim();
      if (m[1] === 'check' || m[1] === 'quiz') { sec._json = ''; field = '_json'; } else field = 'md';
      continue;
    }
    if (sec && (m = /^%% (code|after|predict|lang|norun|spec)\s*(.*)$/.exec(line))) {
      if (m[1] === 'lang') { sec.lang = m[2].trim(); continue; }
      if (m[1] === 'norun') { sec.run = false; continue; }
      field = m[1] === 'predict' ? '_predict' : m[1] === 'spec' ? '_spec' : m[1]; sec[field] = ''; continue;
    }
    if (sec && field) sec[field] += (sec[field] ? '\n' : '') + line;
  }
  flush();
  return ops;
}

function apply(op) {
  const id = op.attrs.concept; const c = findConcept(id); let src = c.src;
  const quiz = op.sections.filter(s => s.type === 'quiz'), secs = op.sections.filter(s => s.type !== 'quiz');
  if (secs.length) {
    const [ao, ac] = arrayIn(src, c.open, c.close, 'sections'); const list = items(src, ao, ac);
    const indent = (() => { const ls = src.lastIndexOf('\n', list.length ? list[0].start : ao); return src.slice(ls + 1, list.length ? list[0].start : ao).replace(/\S.*$/, '') || '      '; })();
    const body = secs.map(sectionSrc).join(',\n' + indent);
    const find = t => { const k = list.findIndex(it => titleOf(src.slice(it.start, it.end + 1)) === t); if (k < 0) throw new Error(id + ': no section titled "' + t + '" (have: ' + list.map(it => titleOf(src.slice(it.start, it.end + 1))).filter(Boolean).join(' | ') + ')'); return k; };
    if (op.kind === 'REPLACE') {
      const it = list[find(op.attrs.section)]; src = src.slice(0, it.start) + body + src.slice(it.end + 1);
    } else if (op.attrs.after) {
      const it = list[find(op.attrs.after)]; src = src.slice(0, it.end + 1) + ',\n' + indent + body + src.slice(it.end + 1);
    } else if (op.attrs.before) {
      const it = list[find(op.attrs.before)]; src = src.slice(0, it.start) + body + ',\n' + indent + src.slice(it.start);
    } else {
      const last = list[list.length - 1]; src = src.slice(0, last.end + 1) + ',\n' + indent + body + src.slice(last.end + 1);
    }
  }
  if (quiz.length) {
    const c2 = { open: c.open, close: match(src, c.open) };
    const [qo, qc] = arrayIn(src, c2.open, c2.close, 'quiz'); const list = items(src, qo, qc); const last = list[list.length - 1];
    const ind = (() => { const ls = src.lastIndexOf('\n', last.start); return src.slice(ls + 1, last.start); })();
    src = src.slice(0, last.end + 1) + ',\n' + ind + quiz.map(q => JSON.stringify(q.question)).join(',\n' + ind) + src.slice(last.end + 1);
  }
  fs.writeFileSync(c.file, src);
  return id + ': ' + op.kind.toLowerCase() + ' ' + secs.length + ' section(s)' + (quiz.length ? ', ' + quiz.length + ' quiz question(s)' : '') + ' in ' + path.basename(c.file);
}

for (const f of process.argv.slice(2)) for (const op of parse(fs.readFileSync(f, 'utf8'))) console.log(apply(op));
