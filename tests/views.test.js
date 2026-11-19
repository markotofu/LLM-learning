/* Loads the whole app in Node and compiles every component template with Vue's compiler,
 * catching template syntax errors without a browser. Also checks markdown rendering of all content.
 * Run: node tests/views.test.js */
'use strict';
const { load, suite } = require('./_load.js');
const MLA = load();
const T = suite('views');
const Vue = global.Vue;

T.ok('scripts load without errors', MLA.__loadErrors.length === 0, MLA.__loadErrors.join('\n'));
const required = ['view-home', 'view-map', 'view-concept', 'view-problem', 'view-problems', 'view-playground', 'view-tracks', 'view-track', 'view-review', 'view-glossary', 'view-achievements', 'view-profile', 'view-selftest', 'code-editor', 'run-output', 'code-sample', 'quiz-runner', 'check-q', 'mla-demo', 'mla-md', 'onboard-dialog'];
for (const n of required) T.ok('component ' + n + ' registered', !!MLA.components[n]);
T.ok('every route has a view', MLA.router.routes.every(r => ['home', 'map', 'concept', 'problems', 'problem', 'playground', 'tracks', 'track', 'review', 'glossary', 'achievements', 'profile', 'selftest'].includes(r.name)));

let compiled = 0;
for (const name of Object.keys(MLA.components)) {
  const tpl = MLA.components[name] && MLA.components[name].template;
  if (typeof tpl !== 'string') continue;
  try { const fn = Vue.compile(tpl, { onError(e) { throw e; } }); if (typeof fn !== 'function') throw new Error('not a function'); compiled++; }
  catch (e) { T.ok('template compiles: ' + name, false, String(e.message || e).split('\n')[0]); }
}
T.ok('compiled templates', compiled >= 30, String(compiled));

// render every piece of markdown once (catches renderer crashes) and highlight every code block
let mdCount = 0;
for (const c of MLA.data.concepts) {
  const parts = [c.why, c.when, c.pitfalls].concat(c.sections.map(s => s.md), c.sections.map(s => s.after), c.problems.map(p => p.prompt), c.problems.map(p => p.explain), c.problems.flatMap(p => p.hints)).filter(Boolean);
  for (const m of parts) { try { const html = MLA.util.md(m); if (typeof html !== 'string') throw new Error('no html'); mdCount++; } catch (e) { T.ok('markdown renders in ' + c.id, false, e.message); } }
  for (const s of c.sections) if (s.code) { const h = MLA.highlight.code(s.code, s.lang || 'python'); T.ok('highlight keeps text in ' + c.id, h.replace(/<[^>]+>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&amp;/g, '&') === s.code); }
}
T.ok('markdown rendered', mdCount > 0);

// formula legends: ":::where" attaches to the formulas directly above it and renders symbol/definition pairs
{
  const h = MLA.util.md('Intro text.\n\n$$ a = b + c $$\n$$ d = 2a $$\n:::where\na — the **sum** of b and c\nd: twice a\n  (continued here)\nA plain note line without a separator.\n:::\n\nAfter.');
  T.ok('where: formulas and legend share one card', /<div class="formula-group"><div class="formula">.*a = b \+ c.*<div class="formula">.*d = 2a.*<div class="formula-where">/.test(h), h);
  T.ok('where: em-dash definition', h.includes('<dt>a</dt><dd>the <strong>sum</strong> of b and c</dd>'));
  T.ok('where: colon definition + indented continuation', h.includes('<dt>d</dt><dd>twice a (continued here)</dd>'));
  T.ok('where: note row', h.includes('<dd class="fw-note">A plain note line without a separator.</dd>'));
  T.ok('where: paragraphs around it intact', h.startsWith('<p>Intro text.</p>') && h.endsWith('<p>After.</p>'));
  const solo = MLA.util.md('| x | y |\n| --- | --- |\n| 1 | 2 |\n\n:::where Symbols in this table\nx — input\n:::');
  T.ok('where: standalone legend after a table gets its own label', solo.includes('<div class="formula-group"><div class="formula-where"><div class="fw-label">Symbols in this table</div>'), solo);
}

// store: export → import round-trip and merge never loses solved problems
const S = MLA.store;
const p0 = MLA.data.problems[0];
S.state.problems[p0.id] = { draft: 'x = 1', draftAt: '2026-01-02T00:00:00Z', solved: true, solvedAt: '2026-01-02T00:00:00Z', attempts: 2, runs: 3, assisted: false, solutionViewed: false, hintsShown: 0, lastStatus: 'accepted', accepted: { code: 'x = 1', at: '2026-01-02T00:00:00Z' } };
S.state.xp = 500;
const exported = S.exportJSON();
const other = S.defaultState(); other.xp = 900; other.problems[p0.id] = { draft: 'y = 2', draftAt: '2026-03-01T00:00:00Z', solved: false, attempts: 1 };
const merged = S.mergeStates(JSON.parse(exported), other);
T.ok('merge keeps max XP', merged.xp === 900);
T.ok('merge keeps solved', merged.problems[p0.id].solved === true);
T.ok('merge keeps newest draft', merged.problems[p0.id].draft === 'y = 2');
T.ok('merge keeps accepted code', merged.problems[p0.id].accepted.code === 'x = 1');
S.importJSON(JSON.stringify(other), 'replace');
T.ok('replace import', S.state.xp === 900 && S.state.problems[p0.id].solved === false);
S.importJSON(exported, 'merge');
T.ok('merge import', S.state.xp === 900 && S.state.problems[p0.id].solved === true);
let threw = false; try { S.importJSON('{"hello":1}'); } catch (e) { threw = true; }
T.ok('rejects foreign JSON', threw);
T.done();
