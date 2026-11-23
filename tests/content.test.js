/* Content integrity: concepts, prerequisites, problems, quizzes, demos, glossary, tracks.
 * Run: node tests/content.test.js */
'use strict';
const { load, suite } = require('./_load.js');
const MLA = load();
const T = suite('content');
const D = MLA.data;

T.ok('scripts load without errors', MLA.__loadErrors.length === 0, MLA.__loadErrors.join('\n'));
const concepts = D.concepts || [];
const problems = D.problems || [];
const branchIds = new Set(D.branches.map(b => b.id));
T.ok('has concepts', concepts.length > 0);

// Every display formula ($$ … $$, possibly several in a row) must be followed by a :::where legend that defines
// its symbols — full understanding first. Returns a list of problems found in one markdown string.
function formulaLegendIssues(text) {
  const issues = [];
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  let i = 0;
  while (i < lines.length) {
    const t = lines[i].trim();
    if (/^```/.test(t)) { i++; while (i < lines.length && !/^```\s*$/.test(lines[i].trim())) i++; i++; continue; }
    if (!/^\$\$/.test(t)) { i++; continue; }
    const first = t.slice(0, 60);
    while (i < lines.length) { // consume the group of formulas (blank lines allowed between them)
      const u = lines[i].trim();
      if (u === '') { i++; continue; }
      if (!/^\$\$/.test(u)) break;
      if (u.length > 4 && /\$\$$/.test(u)) { i++; continue; }
      i++; while (i < lines.length && !/\$\$\s*$/.test(lines[i].trim())) i++; i++;
    }
    if (i >= lines.length || !/^:::where\b/.test(lines[i].trim())) { issues.push('no :::where legend after ' + first); continue; }
    const defs = []; i++;
    while (i < lines.length && !/^:::\s*$/.test(lines[i].trim())) { if (/ — |: /.test(lines[i])) defs.push(lines[i]); i++; }
    if (!defs.length) issues.push('empty :::where legend after ' + first);
    i++;
  }
  return issues;
}

// ---- concepts ----
const ids = new Set();
for (const c of concepts) {
  const where = 'concept ' + c.id;
  T.ok(where + ': id is kebab-case', /^[a-z0-9]+(-[a-z0-9]+)*$/.test(c.id));
  T.ok(where + ': unique id', !ids.has(c.id)); ids.add(c.id);
  T.ok(where + ': valid branch', branchIds.has(c.branch), c.branch);
  T.ok(where + ': title + summary', !!c.title && !!c.summary && c.summary.length > 20);
  T.ok(where + ': level 1-3', [1, 2, 3].includes(c.level));
  T.ok(where + ': has a "why"', !!c.why && c.why.length > 80);
  T.ok(where + ': objectives', c.objectives.length >= 2);
  T.ok(where + ': at least 3 sections', c.sections.length >= 3);
  T.ok(where + ': takeaways', c.takeaways.length >= 3);
  T.ok(where + ': has "when" and "pitfalls"', !!c.when && !!c.pitfalls);
  T.ok(where + ': quiz has 4+ questions', c.quiz.length >= 4, String(c.quiz.length));
  c.quiz.forEach((q, i) => T.ok(where + ': quiz q' + (i + 1) + ' valid + explained', MLA.quiz.validQuestion(q) && !!q.explain, JSON.stringify(q).slice(0, 120)));
  T.ok(where + ': has practice problems', c.problems.length >= 1);
  c.sections.forEach((s, i) => {
    const sw = where + ' section ' + (i + 1);
    T.ok(sw + ': known type', ['text', 'code', 'demo', 'check', 'chart'].includes(s.type), s.type);
    if (s.type === 'chart') {
      const ch = s.chart || {}, ser = ch.series || [];
      const okKind = ['line', 'scatter', 'bar'].includes(ch.kind || 'line');
      const okSeries = ser.length > 0 && ser.every(x => Array.isArray(x.y) && x.y.length > 0 && x.y.every(v => v === null || Number.isFinite(v)) && (Array.isArray(x.x) ? x.x.length === x.y.length : Array.isArray(ch.x) && ch.x.length === x.y.length));
      const okLog = (!ch.xLog || (ch.x || []).concat(...ser.map(x => x.x || [])).every(v => v > 0)) && (!ch.yLog || ser.every(x => x.y.every(v => v === null || v > 0)));
      T.ok(sw + ': chart spec is valid', okKind && okSeries && okLog && !!s.title && !!ch.xLabel && !!ch.yLabel, JSON.stringify(ch).slice(0, 160));
    }
    if (s.type === 'text') T.ok(sw + ': text has md', !!s.md && s.md.length > 40);
    if (s.type === 'code') { T.ok(sw + ': code present', !!s.code); T.ok(sw + ': known lang', !s.lang || ['python', 'pytorch', 'bash', 'output', 'text', 'json'].includes(s.lang), s.lang); }
    if (s.timeoutSec != null) T.ok(sw + ': timeoutSec is a whole number of seconds (20 to 300) on a runnable Python sample', s.type === 'code' && (!s.lang || s.lang === 'python') && s.run !== false && Number.isInteger(s.timeoutSec) && s.timeoutSec >= 20 && s.timeoutSec <= 300, s.timeoutSec);
    if (s.type === 'demo') T.ok(sw + ': demo "' + s.demo + '" exists', !!MLA.components['demo-' + s.demo]);
    if (s.predict) {
      T.ok(sw + ': prediction is a valid question with an explanation', MLA.quiz.validQuestion(s.predict) && s.predict.type !== 'match' && !!s.predict.explain);
      T.ok(sw + ': prediction names the output text that confirms it', Array.isArray(s.predict.expect) && s.predict.expect.length > 0 && s.predict.expect.every(e => typeof e === 'string' && e.length > 2));
      T.ok(sw + ': prediction sits on a runnable Python sample', s.type === 'code' && (!s.lang || s.lang === 'python') && s.run !== false);
    }
    if (s.type === 'check') T.ok(sw + ': check question valid', MLA.quiz.validQuestion(s.question) && s.question.type !== 'match' && !!s.question.explain);
    if (s.md) { const fences = (s.md.match(/^\s*```/gm) || []).length; T.ok(sw + ': balanced code fences', fences % 2 === 0); }
  });
  for (const f of ['why', 'when', 'pitfalls']) if (c[f]) T.ok(where + ': ' + f + ' fences balanced', ((c[f].match(/^\s*```/gm) || []).length % 2) === 0);
  const mdTexts = [c.why, c.when, c.pitfalls].concat(c.sections.map(s => s.md)).concat(c.problems.map(p => p.prompt));
  const legendIssues = [].concat(...mdTexts.map(formulaLegendIssues));
  T.ok(where + ': every formula defines its symbols (:::where)', legendIssues.length === 0, legendIssues.join(' | '));
  for (const p of c.prereqs) T.ok(where + ': prereq ' + p + ' exists', !!MLA.content.byId[p]);
  if (c.softPrereqs != null) T.ok(where + ': softPrereqs is a list of lesson ids', Array.isArray(c.softPrereqs) && c.softPrereqs.every(p => typeof p === 'string'));
  for (const p of c.softPrereqs || []) T.ok(where + ': recommended prereq ' + p + ' exists, is not also required and is not the lesson itself', !!MLA.content.byId[p] && p !== c.id && !c.prereqs.includes(p));
  T.ok(where + ': no "undefined" leaked into text', !JSON.stringify([c.why, c.when, c.pitfalls, c.sections.map(s => s.md)]).includes('undefined'));
  const sids = c.sections.map(s => s.id);
  T.ok(where + ': section ids are unique, non-numeric strings', sids.every(x => typeof x === 'string' && /^[a-z0-9-]+$/.test(x) && !/^\d+$/.test(x)) && new Set(sids).size === sids.length, sids.join(','));
}

// ---- saved progress from version 6badf8c (positions) must still translate to the right sections ----
{
  const fs = require('fs'), path = require('path');
  const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/legacy-sections.json'), 'utf8'));
  for (const [id, old] of Object.entries(fixture.layouts)) {
    const L = MLA.progress.legacyLayout(id).slice(0, old.length);
    T.ok('legacy layout of ' + id + ' matches version ' + fixture.commit + ' (if a lesson\'s first sections changed, run node tools/legacy-sections.js)', JSON.stringify(L) === JSON.stringify(old), JSON.stringify(L) + ' vs ' + JSON.stringify(old));
  }
}
const g = MLA.content.graph;
const cycle = MLA.graph.findCycle(g);
T.ok('no prerequisite cycles', cycle === null, cycle && cycle.join(' → '));
// Recommended prerequisites (softPrereqs) never lock a lesson, but they must still read as "learn this first":
// no cycles once they count, nothing that is already required on the way, and earlier in the prerequisite tree.
{
  const softCycle = MLA.graph.findCycle(g, { soft: true });
  T.ok('no cycles when recommended prerequisites count too', softCycle === null, softCycle && softCycle.join(' → '));
  const dep = MLA.graph.depths(g);
  for (const c of concepts) for (const p of c.softPrereqs || []) {
    if (!g.nodes[p]) continue;
    T.ok('concept ' + c.id + ': recommended prereq ' + p + ' is not already required on the way', !MLA.graph.ancestors(g, c.id).has(p));
    T.ok('concept ' + c.id + ': recommended prereq ' + p + ' sits earlier in the prerequisite tree', dep[p] < dep[c.id], dep[p] + ' vs ' + dep[c.id]);
  }
  T.ok('some lessons recommend prerequisites', concepts.some(c => (c.softPrereqs || []).length > 0));
}
const roots = concepts.filter(c => c.prereqs.length === 0);
T.ok('exactly one root concept', roots.length === 1, roots.map(r => r.id).join(', '));

// ---- problems ----
const pids = new Set();
for (const p of problems) {
  const where = 'problem ' + p.id;
  T.ok(where + ': id kebab-case', /^[a-z0-9]+(-[a-z0-9]+)*$/.test(p.id));
  T.ok(where + ': unique id', !pids.has(p.id)); pids.add(p.id);
  T.ok(where + ': difficulty', ['easy', 'medium', 'hard'].includes(p.difficulty));
  T.ok(where + ': title + prompt', !!p.title && !!p.prompt && p.prompt.length > 60);
  T.ok(where + ': starter + solution differ', !!p.starter.trim() && !!p.solution.trim() && p.starter !== p.solution);
  T.ok(where + ': hints', p.hints.length >= 1);
  T.ok(where + ': explanation', !!p.explain && p.explain.length > 40);
  T.ok(where + ': 3+ tests', p.tests.length >= 3, String(p.tests.length));
  T.ok(where + ': visible example', p.tests.some(t => !t.hidden));
  T.ok(where + ': hidden test', p.tests.some(t => t.hidden));
  p.tests.forEach((t, i) => T.ok(where + ' test ' + (i + 1) + ': has call or code', (!!t.call) !== (!!t.code)));
  T.ok(where + ': no tabs in code', !/\t/.test(p.starter + p.solution));
}

// ---- glossary, flashcards, tracks, achievements ----
for (const gl of D.glossary || []) {
  T.ok('glossary "' + gl.term + '": branch', branchIds.has(gl.branch));
  T.ok('glossary "' + gl.term + '": concept link', !gl.concept || !!MLA.content.byId[gl.concept], gl.concept);
}
const terms = (D.glossary || []).map(x => x.term.toLowerCase());
T.ok('glossary terms unique', new Set(terms).size === terms.length, terms.filter((t, i) => terms.indexOf(t) !== i).join(', '));
T.ok('flashcards built from glossary', (D.flashcards || []).length === (D.glossary || []).length);
const cardIds = (D.flashcards || []).map(f => f.id);
T.ok('flashcard ids unique', new Set(cardIds).size === cardIds.length, cardIds.filter((t, i) => cardIds.indexOf(t) !== i).join(', '));
for (const c of concepts) T.ok('concept ' + c.id + ': has at least one glossary term', (D.glossary || []).some(gl => gl.concept === c.id));
for (const t of D.tracks) { if (t.custom || t.all) continue; for (const id of t.goals) T.ok('track ' + t.id + ': goal ' + id + ' exists', !!MLA.content.byId[id]); }
T.ok('achievements unique', new Set(D.achievements.map(a => a.id)).size === D.achievements.length);

console.log(concepts.length + ' concepts, ' + problems.length + ' problems, ' + concepts.reduce((s, c) => s + c.quiz.length, 0) + ' quiz questions, ' + (D.glossary || []).length + ' glossary terms');
// Case studies and sample projects (js/data/projects.js): attached to real lessons, complete, and in the house style
// (no em dashes, exclamation marks or emojis in prose; sources named for every case study).
{
  const D2 = MLA.data;
  for (const id of Object.keys(D2.cases || {})) T.ok('case studies: lesson ' + id + ' exists', !!MLA.content.byId[id]);
  for (const id of Object.keys(D2.projects || {})) T.ok('projects: lesson ' + id + ' exists', !!MLA.content.byId[id]);
  const style = (where, t) => {
    t = String(t || '');
    T.ok(where + ': no em dash', !t.includes('—'));
    T.ok(where + ': no exclamation mark', !/!(?!=)/.test(t.replace(/`[^`]*`/g, '')));
    T.ok(where + ': no emoji', !/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u.test(t));
    T.ok(where + ': no "undefined"', !t.includes('undefined'));
  };
  let nc = 0, np = 0;
  for (const c of concepts) {
    for (const k of c.cases) {
      nc++;
      const w = 'case "' + k.title + '" (' + c.id + ')';
      T.ok(w + ': has a title, a date, a story, a lesson and a source', !!(k.title && k.when && k.story && k.lesson && k.source && k.source.title));
      T.ok(w + ': story is a real account (300+ characters)', k.story.length >= 300, String(k.story.length));
      T.ok(w + ': source link, when given, is https or a known http site', !k.source.url || /^https:[/][/]/.test(k.source.url) || /^http:[/][/]www[.]paulgraham[.]com[/]/.test(k.source.url));
      style(w, [k.title, k.when, k.story, k.lesson, k.source.title].join(' '));
    }
    for (const pr of c.projects) {
      np++;
      const w = 'project "' + pr.title + '" (' + c.id + ')';
      T.ok(w + ': level is starter, intermediate or advanced', ['starter', 'intermediate', 'advanced'].includes(pr.level));
      T.ok(w + ': hours between 1 and 40', typeof pr.hours === 'number' && pr.hours >= 1 && pr.hours <= 40);
      T.ok(w + ': has a goal, data, 4+ steps, a finish line, stretch goals and 3+ skills', !!(pr.goal && pr.data && pr.done) && pr.steps.length >= 4 && pr.stretch.length >= 1 && pr.skills.length >= 3);
      style(w, [pr.title, pr.goal, pr.data, pr.done].concat(pr.steps, pr.stretch, pr.skills).join(' '));
    }
  }
  T.ok('there are case studies and projects', nc >= 20 && np >= 20, nc + ' case studies, ' + np + ' projects');
}

{
  const fs2 = require('fs');
  const { generate, OUT } = require('../tools/build-catalog.js');
  T.ok('js/data/catalog.js is up to date (run node tools/build-catalog.js)', fs2.readFileSync(OUT, 'utf8') === generate(MLA));
  const listed = JSON.parse(fs2.readFileSync(require('path').join(__dirname, '../tools/content-files.json'), 'utf8'));
  const onDisk = fs2.readdirSync(require('path').join(__dirname, '../js/data')).filter(f => /^concepts-.*[.]js$/.test(f));
  T.ok('every lesson file is listed in tools/content-files.json', onDisk.every(f => listed.includes(f)) && listed.every(f => onDisk.includes(f)), onDisk.filter(f => !listed.includes(f)).join(', '));
}

T.done();
