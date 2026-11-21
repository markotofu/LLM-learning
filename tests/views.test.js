/* Loads the whole app in Node and compiles every component template with Vue's compiler,
 * catching template syntax errors without a browser. Also checks markdown rendering of all content.
 * Run: node tests/views.test.js */
'use strict';
const { load, suite } = require('./_load.js');
const MLA = load();
const T = suite('views');
const Vue = global.Vue;

T.ok('scripts load without errors', MLA.__loadErrors.length === 0, MLA.__loadErrors.join('\n'));
const required = ['view-home', 'view-map', 'view-concept', 'view-problem', 'view-problems', 'view-playground', 'view-tracks', 'view-track', 'view-review', 'view-glossary', 'view-projects', 'view-achievements', 'view-profile', 'view-selftest', 'view-terms', 'view-privacy', 'code-editor', 'run-output', 'code-sample', 'quiz-runner', 'check-q', 'mla-demo', 'mla-md', 'onboard-dialog'];
for (const n of required) T.ok('component ' + n + ' registered', !!MLA.components[n]);
T.ok('every route has a view', MLA.router.routes.every(r => ['home', 'map', 'concept', 'problems', 'problem', 'playground', 'tracks', 'track', 'review', 'glossary', 'projects', 'achievements', 'profile', 'selftest', 'terms', 'privacy'].includes(r.name)));

// legal pages: placeholders filled in, key statements present, markdown renders
const L = MLA.data.legal;
T.ok('legal data loaded', !!(L && L.terms && L.privacy && L.owner));
for (const key of ['terms', 'privacy']) {
  const text = L.render(L[key]);
  T.ok('legal ' + key + ': no unfilled placeholders', !/\{(owner|year|updated|contact|codeLicense)\}/.test(text));
  T.ok('legal ' + key + ': renders', /<h[1-6]>/.test(MLA.util.md(text)));
}
T.ok('privacy policy names every localStorage key the app uses', ['mla.state.v1', 'mla.pyMirror'].every(k => L.privacy.includes(k)));
T.ok('copyright notice has year and owner', L.copyright().includes(L.year) && L.copyright().includes(L.owner));

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

// security: the renderer is used for user notes and imported files, so nothing a user types may become live HTML
{
  const attacks = [
    '<svg onload=alert(1)>', '<div><img src=x onerror=alert(1)></div>', '<details open ontoggle=alert(1)>x</details>',
    '<script>alert(1)</script>', 'text <img src=x onerror=alert(1)> more', '- <img src=x onerror=alert(1)>',
    '| <img src=x onerror=alert(1)> | b |\n| --- | --- |\n| c | d |', '# <iframe src=javascript:alert(1)>', '> <svg onload=alert(1)>',
    ':::tip <img src=x onerror=alert(1)>\nbody\n:::', '$$ <img src=x onerror=alert(1)> $$', '```<img src=x onerror=alert(1)>\ncode\n```'
  ];
  for (const a of attacks) {
    const h = MLA.util.md(a);
    // escaped text such as "&lt;svg onload=…&gt;" is harmless; only a real tag (a literal "<") could run anything
    T.ok('md escapes: ' + a.slice(0, 40), !/<(svg|img|script|iframe|details)\b/i.test(h) && !/<[a-z][^>]*\son\w+\s*=/i.test(h), h);
  }
  const link = MLA.util.md('[click](javascript:alert(1)) and [ok](https://example.com) and [route](#/concept/knn)');
  T.ok('md: no javascript: links', !/href="javascript:/i.test(link) && link.includes('href="https://example.com"') && link.includes('href="#/concept/knn"'), link);
  T.ok('md: external links open safely', /target="_blank" rel="noopener"/.test(link));
}

// security: imports drop prototype-tampering keys and refuse absurdly large files
{
  const before = JSON.stringify(MLA.store.exportObject());
  const evil = '{"app":"ml-atlas","version":1,"profile":{"name":"x"},"__proto__":{"polluted":true},"notes":{"constructor":{"prototype":{"bad":1}}}}';
  let threw = null;
  try { MLA.store.importJSON(evil, 'merge'); } catch (e) { threw = e; }
  T.ok('import: prototype keys do not pollute', ({}).polluted === undefined && ({}).bad === undefined && Object.prototype.polluted === undefined);
  T.ok('import: object prototypes untouched', Object.getPrototypeOf(MLA.store.state) === Object.prototype);
  let big = null;
  try { MLA.store.importJSON(' '.repeat(41 * 1024 * 1024), 'merge'); } catch (e) { big = e; }
  T.ok('import: oversized input rejected', big && /too large/.test(big.message));
  MLA.store.importJSON(before, 'replace');    // restore the state for the tests below
  T.ok('import: evil file either imported safely or rejected', threw === null || threw instanceof Error);
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

// Section progress: stored by section id; positions saved by version 6badf8c and earlier are migrated through that
// version's layout. transformer-block gained a diagram after its first section, so position 1 there used to mean
// "The residual stream" (now the third section) and position 3 the first quick check.
{
  const P = MLA.progress;
  const tb = MLA.content.byId['transformer-block'];
  const ids = tb.sections.map(s => s.id);
  const legacy = P.legacyLayout('transformer-block');
  T.ok('sections have string ids', ids.every(x => typeof x === 'string' && x && !/^\d+$/.test(x)));
  T.ok('legacy layout of transformer-block skips the inserted diagram', legacy[1] === 'the-residual-stream' && ids[1] !== 'the-residual-stream', legacy.slice(0, 3).join(','));
  const old = S.defaultState(); old.version = 1; old.xp = 10;
  old.concepts['transformer-block'] = { read: [0, 1, 2], checks: { 3: true }, quizBest: 0, quizAttempts: 0, quizPassed: false };
  old.concepts.attention = { read: [0], checks: {}, quizBest: 0, quizAttempts: 0, quizPassed: false };
  const migrated = JSON.parse(JSON.stringify(old)); P.migrateSections(migrated);
  const m = migrated.concepts['transformer-block'];
  T.ok('migration maps old positions through the old layout', JSON.stringify(m.read) === JSON.stringify([legacy[0], legacy[1], legacy[2]]) && legacy[1] === 'the-residual-stream', JSON.stringify(m.read));
  T.ok('migration maps answered checks', m.checks[legacy[3]] === true && legacy[3].startsWith('check-') && Object.keys(m.checks).length === 1, JSON.stringify(m.checks));
  T.ok('unchanged lessons map positions to the same sections', migrated.concepts.attention.read[0] === MLA.content.byId.attention.sections[0].id);
  T.ok('migration is idempotent', P.migrateSections(migrated) === 0 && JSON.stringify(migrated.concepts['transformer-block'].read) === JSON.stringify(m.read));
  T.ok('migration marks the data version', migrated.version === 2);
  // Importing an old export (positions) into new data (ids) merges cleanly, without numbers left behind.
  S.reset();
  P.markRead('transformer-block', ids[1]);
  S.importJSON(JSON.stringify(old), 'merge');
  const r = S.state.concepts['transformer-block'].read;
  T.ok('merge import of old data converts positions to ids', r.every(x => typeof x === 'string') && r.includes(ids[1]) && r.includes('the-residual-stream') && r.length === 4, JSON.stringify(r));
  T.ok('progress counts migrated sections', P.conceptProgress('transformer-block').sectionsRead === 4);
  P.recordCheck('attention', MLA.content.byId.attention.sections.find(s => s.type === 'check').id, true);
  T.ok('checks are recorded by id', Object.keys(S.state.concepts.attention.checks).every(k => !/^\d+$/.test(k)));
  S.reset();
}

// Mistakes to revisit: missed questions and helped-with problems come back on a spaced schedule.
{
  const P = MLA.progress, M = MLA.mistakes, U = MLA.util;
  const realToday = U.todayKey;
  let day = '2026-03-01';
  U.todayKey = () => day;
  try {
    S.reset();
    const c = MLA.content.byId.attention;
    const check = c.sections.find(s => s.type === 'check');
    P.recordCheck('attention', check.id, false);
    const id = 'check:attention:' + check.id;
    T.ok('a wrong quick check is saved for review', !!S.state.mistakes[id] && S.state.mistakes[id].card.due === '2026-03-02');
    T.ok('it is not due the same day', M.dueItems().length === 0 && M.stats().waiting === 1);
    day = '2026-03-02';
    T.ok('it is due the next day, with its question', M.dueItems().length === 1 && M.dueItems()[0].question === check.question);
    M.grade(id, true); T.ok('first correct review: back in 1 day', S.state.mistakes[id].card.due === '2026-03-03' && !S.state.mistakes[id].graduated);
    day = '2026-03-03'; M.grade(id, true); T.ok('second correct review: back in 6 days', S.state.mistakes[id].card.due === '2026-03-09');
    day = '2026-03-09'; M.grade(id, false); T.ok('a miss keeps it due', S.state.mistakes[id].card.due === '2026-03-09' && !S.state.mistakes[id].graduated);
    M.grade(id, true); day = '2026-03-10'; M.grade(id, true); day = '2026-03-16'; M.grade(id, true);
    T.ok('after a miss, the ease drops and three recalls are not yet enough', !S.state.mistakes[id].graduated && S.state.mistakes[id].card.due === '2026-03-28');
    day = '2026-03-28'; M.grade(id, true);
    T.ok('correct recalls spaced out over weeks retire it', !!S.state.mistakes[id].graduated && M.stats().graduated === 1 && M.dueItems('2027-01-01').length === 0);
    // quiz questions, by the hash of their text (options may be shuffled)
    day = '2026-04-01';
    const q = c.quiz.find(x => x.type === 'mcq');
    P.recordQuiz('attention', { score: 50, results: [{ question: MLA.quiz.shuffled(q, U.seededRng(1)), correct: false }] });
    T.ok('a missed quiz question is saved', !!S.state.mistakes['quiz:attention:' + M.qkey(q)]);
    day = '2026-04-02';
    T.ok('the review shows the original question', M.dueItems().some(x => x.item.kind === 'quiz' && x.question === q));
    // problems solved with the solution come back after 3 days and retire after one unaided re-solve
    const pr = MLA.data.problems.find(p => p.concept === 'attention');
    P.viewSolution(pr.id);
    P.recordSubmit(pr.id, 'x', { tests: [{ passed: true }] });
    const pid = 'problem:attention:' + pr.id;
    T.ok('a problem solved after viewing the solution is saved', !!S.state.mistakes[pid] && S.state.mistakes[pid].card.due === '2026-04-05' && S.state.mistakes[pid].why === 'solution');
    P.recordSubmit(pr.id, 'x', { tests: [{ passed: true }] });
    T.ok('re-solving before it is due does not count', !S.state.mistakes[pid].graduated);
    day = '2026-04-05';
    P.recordSubmit(pr.id, 'x', { tests: [{ passed: true }] });
    T.ok('re-solving on the due date retires it', !!S.state.mistakes[pid].graduated);
    // merging two devices keeps the more-reviewed copy
    const other = S.defaultState(); other.mistakes[id] = { kind: 'check', concept: 'attention', ref: check.id, card: { reps: 0, lapses: 0, due: '2026-03-02', interval: 0, ease: 2.5 }, graduated: null };
    const mm = S.mergeStates(JSON.parse(S.exportJSON()), other);
    T.ok('merge keeps the retired, more-reviewed copy', !!mm.mistakes[id].graduated);
    T.ok('items whose content is gone are ignored', (() => { S.state.mistakes['quiz:attention:nope'] = { kind: 'quiz', concept: 'attention', ref: 'nope', card: { due: '2026-01-01' } }; return !M.all().some(x => x.id === 'quiz:attention:nope'); })());
  } finally { U.todayKey = realToday; S.reset(); }
}
// Search: one box finds lessons, sections, terms, problems, projects and case studies
{
  const res = q => MLA.search.query(q);
  T.ok('search: a lesson title ranks first', (res('gradient descent')[0] || {}).route === '/concept/gradient-descent');
  T.ok('search: finds case studies and projects', res('netflix').some(r => r.kind === 'Case study') && res('churn').some(r => r.kind === 'Project'));
  T.ok('search: finds problems and glossary terms', res('causal mask').some(r => r.kind === 'Problem') && res('softmax').some(r => r.kind === 'Term'));
  T.ok('search: sections link to their place in the lesson', res('causal mask').some(r => r.kind === 'Section' && /^[/]concept[/]attention[?]s=/.test(r.route)));
  T.ok('search: searches inside lesson text', res('blending hundreds').length > 0 || res('wordpieces').some(r => r.kind === 'In the text'));
  T.ok('search: an empty query returns nothing', res('   ').length === 0);
}

T.done();
