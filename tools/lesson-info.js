#!/usr/bin/env node
/* Compact index of the lessons, so you can find content without opening the large js/data files.
 *   node tools/lesson-info.js                 one line per lesson: id, branch, file:line, sections, quiz, problems, words
 *   node tools/lesson-info.js <id> [<id> …]   a lesson's outline: each section's index, type, title and file:line
 *   node tools/lesson-info.js --find <text>   lessons, sections and glossary terms whose text contains <text>
 * Then read only the line range you need. */
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const MLA = require(path.join(root, 'tests', '_load.js')).load();
MLA.content.init();
const concepts = MLA.data.concepts;

// Where each concept and section starts in the source files.
const files = fs.readdirSync(path.join(root, 'js', 'data')).filter(f => /^concepts-.*\.js$/.test(f));
const src = files.map(f => ({ f: 'js/data/' + f, lines: fs.readFileSync(path.join(root, 'js', 'data', f), 'utf8').split('\n') }));
function locate(id) {
  for (const s of src) {
    const i = s.lines.findIndex(l => l.includes("id: '" + id + "'") && /branch:/.test(l));
    if (i >= 0) return { file: s.f, line: i + 1, lines: s.lines };
  }
  return null;
}
function sectionLines(c, loc) {
  const out = [];
  if (!loc) return c.sections.map(() => null);
  let from = loc.line;                                    // search forward from the concept's first line
  for (const s of c.sections) {
    const key = String(s.title || '').split(/['"\\]/)[0].slice(0, 40);
    let hit = null;
    if (key) for (let i = from; i < loc.lines.length; i++) if (loc.lines[i].includes('title: ') && loc.lines[i].includes(key)) { hit = i + 1; from = i + 1; break; }
    out.push(hit);
  }
  return out;
}
const words = c => c.sections.reduce((n, s) => n + String((s.md || '') + ' ' + (s.after || '')).split(/\s+/).filter(Boolean).length, 0);

const args = process.argv.slice(2);
if (!args.length) {
  for (const c of concepts) {
    const loc = locate(c.id);
    console.log([c.id, c.branch, loc ? loc.file.replace('js/data/', '') + ':' + loc.line : '?', c.sections.length + ' sec', (c.quiz || []).length + ' quiz', (c.problems || []).length + ' prob', words(c) + ' w'].join('  '));
  }
  console.log(concepts.length + ' lessons. Outline: node tools/lesson-info.js <id>');
} else if (args[0] === '--find') {
  const q = args.slice(1).join(' ').toLowerCase();
  if (!q) { console.log('usage: node tools/lesson-info.js --find <text>'); process.exit(1); }
  let n = 0;
  const hit = t => String(t || '').toLowerCase().includes(q);
  for (const c of concepts) {
    if (hit(c.title) || hit(c.id)) { console.log('lesson   ' + c.id + '  ' + c.title); n++; }
    const loc = locate(c.id); const sl = sectionLines(c, loc);
    c.sections.forEach((s, i) => {
      const fields = [s.title, s.md, s.after, s.code, s.question && s.question.q, s.predict && s.predict.q];
      if (fields.some(hit)) { console.log('section  ' + c.id + ' #' + i + ' [' + s.type + '] ' + (s.title || '') + (sl[i] ? '  (' + loc.file + ':' + sl[i] + ')' : '')); n++; }
    });
    for (const p of c.problems || []) if (hit(p.title) || hit(p.prompt) || hit(p.id)) { console.log('problem  ' + c.id + ' / ' + p.id + '  ' + p.title); n++; }
  }
  for (const g of MLA.data.glossary) if (hit(g.term) || hit(g.def)) { console.log('glossary ' + g.term + '  (' + g.concept + ')'); n++; }
  console.log(n + ' matches');
} else {
  for (const id of args) {
    const c = MLA.content.byId[id];
    if (!c) { console.log('no lesson with id ' + id + ' (list them: node tools/lesson-info.js)'); continue; }
    const loc = locate(id); const sl = sectionLines(c, loc);
    console.log('## ' + c.id + ' | ' + c.title + ' | branch ' + c.branch + ' | level ' + c.level + (loc ? ' | ' + loc.file + ':' + loc.line : ''));
    console.log('prereqs: ' + (c.prereqs.join(', ') || 'none') + ((c.softPrereqs || []).length ? ' | recommended: ' + c.softPrereqs.join(', ') : '') + ' | quiz ' + (c.quiz || []).length + ' | problems: ' + (c.problems || []).map(p => p.id).join(', ') + ' | ' + words(c) + ' words');
    c.sections.forEach((s, i) => {
      const extra = s.type === 'code' ? (s.run === false ? ' norun' : '') + (s.predict ? ' predict' : '') + (s.timeoutSec ? ' timeout ' + s.timeoutSec : '') : s.type === 'demo' ? ' ' + s.demo : '';
      console.log(String(i).padStart(3) + '  ' + s.type.padEnd(5) + extra.padEnd(16) + (s.title || '(no title)') + (sl[i] ? '  :' + sl[i] : ''));
    });
  }
}
