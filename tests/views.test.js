/* Loads the whole app in Node and compiles every component template with Vue's compiler,
 * catching template syntax errors without a browser. Also checks markdown rendering of all content.
 * Run: node tests/views.test.js */
'use strict';
const { load, suite } = require('./_load.js');
const MLA = load();
const T = suite('views');
const Vue = global.Vue;

T.ok('scripts load without errors', MLA.__loadErrors.length === 0, MLA.__loadErrors.join('\n'));
const required = ['view-home', 'view-map', 'view-concept', 'view-problem', 'view-problems', 'view-playground', 'view-tracks', 'view-track', 'view-review', 'view-glossary', 'view-projects', 'view-achievements', 'view-profile', 'view-selftest', 'view-terms', 'view-privacy', 'code-editor', 'run-output', 'code-sample', 'quiz-runner', 'check-q', 'mla-demo', 'mla-md', 'onboard-dialog', 'view-placement', 'placement-q', 'view-quiz'];
for (const n of required) T.ok('component ' + n + ' registered', !!MLA.components[n]);
T.ok('every route has a view', MLA.router.routes.every(r => ['home', 'map', 'concept', 'problems', 'problem', 'playground', 'tracks', 'track', 'review', 'glossary', 'projects', 'achievements', 'profile', 'selftest', 'placement', 'quiz', 'terms', 'privacy'].includes(r.name)));

// legal pages: placeholders filled in, key statements present, markdown renders
const L = MLA.data.legal;
T.ok('legal data loaded', !!(L && L.terms && L.privacy && L.owner));
for (const key of ['terms', 'privacy']) {
  const text = L.render(L[key]);
  T.ok('legal ' + key + ': no unfilled placeholders', !/\{(name|subject|key|db|owner|year|updated|contact|codeLicense)\}/.test(text));
  T.ok('legal ' + key + ': renders', /<h[1-6]>/.test(MLA.util.md(text)));
}
T.ok('privacy policy names every localStorage key the app uses', [MLA.course.storageKey, 'mla.pyMirror'].every(k => L.render(L.privacy).includes(k)));
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
  try { MLA.store.importJSON(' '.repeat(MLA.store.MAX_IMPORT_CHARS + 1), 'merge'); } catch (e) { big = e; }
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

// Section progress: stored by section id. Data saved by position (version 1) is migrated through the lesson's legacy
// layout (js/data/legacy-sections.js) or, for lessons not listed there, through its current sections.
{
  const P = MLA.progress;
  const A = MLA.data.concepts.find(c => c.sections.length >= 4 && c.sections.some(s => s.type === 'check'));
  const B = MLA.data.concepts.find(c => c !== A) || A;
  const ids = A.sections.map(s => s.id), ci = A.sections.findIndex(s => s.type === 'check');
  const legacy = P.legacyLayout(A.id);
  T.ok('sections have string ids', ids.every(x => typeof x === 'string' && x && !/^\d+$/.test(x)));
  const old = S.defaultState(); old.version = 1; old.xp = 10;
  old.concepts[A.id] = { read: [0, 1, 2], checks: { [ci]: true }, quizBest: 0, quizAttempts: 0, quizPassed: false };
  old.concepts[B.id] = { read: [0], checks: {}, quizBest: 0, quizAttempts: 0, quizPassed: false };
  const migrated = JSON.parse(JSON.stringify(old)); P.migrateSections(migrated);
  const m = migrated.concepts[A.id];
  T.ok('migration maps old positions through the layout', JSON.stringify(m.read) === JSON.stringify([legacy[0], legacy[1], legacy[2]]), JSON.stringify(m.read));
  T.ok('migration maps answered checks', m.checks[legacy[ci]] === true && Object.keys(m.checks).length === 1, JSON.stringify(m.checks));
  T.ok('unchanged lessons map positions to the same sections', migrated.concepts[B.id].read[0] === B.sections[0].id);
  T.ok('migration is idempotent', P.migrateSections(migrated) === 0 && JSON.stringify(migrated.concepts[A.id].read) === JSON.stringify(m.read));
  T.ok('migration marks the data version', migrated.version === 2);
  // Importing an old export (positions) into new data (ids) merges cleanly, without numbers left behind.
  S.reset();
  P.markRead(A.id, ids[3]);
  S.importJSON(JSON.stringify(old), 'merge');
  const r = S.state.concepts[A.id].read;
  T.ok('merge import of old data converts positions to ids', r.every(x => typeof x === 'string') && r.includes(ids[3]) && r.includes(legacy[1]) && r.length === 4, JSON.stringify(r));
  T.ok('progress counts migrated sections', P.conceptProgress(A.id).sectionsRead === 4);
  P.recordCheck(A.id, A.sections[ci].id, true);
  T.ok('checks are recorded by id', Object.keys(S.state.concepts[A.id].checks).every(k => !/^\d+$/.test(k)));
  S.reset();
}

// Mistakes to revisit: missed questions and helped-with problems come back on a spaced schedule.
{
  const QID = MLA.data.concepts.find(c => c.sections.some(s => s.type === 'check') && c.quiz.some(q => q.type === 'mcq') && MLA.data.problems.some(p => p.concept === c.id)).id;
  const P = MLA.progress, M = MLA.mistakes, U = MLA.util;
  const realToday = U.todayKey;
  let day = '2026-03-01';
  U.todayKey = () => day;
  try {
    S.reset();
    const c = MLA.content.byId[QID];
    const check = c.sections.find(s => s.type === 'check');
    P.recordCheck(QID, check.id, false);
    const id = 'check:' + QID + ':' + check.id;
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
    P.recordQuiz(QID, { score: 50, results: [{ question: MLA.quiz.shuffled(q, U.seededRng(1)), correct: false }] });
    T.ok('a missed quiz question is saved', !!S.state.mistakes['quiz:' + QID + ':' + M.qkey(q)]);
    day = '2026-04-02';
    T.ok('the review shows the original question', M.dueItems().some(x => x.item.kind === 'quiz' && x.question === q));
    // problems solved with the solution come back after 3 days and retire after one unaided re-solve
    const pr = MLA.data.problems.find(p => p.concept === QID);
    P.viewSolution(pr.id);
    P.recordSubmit(pr.id, 'x', { tests: [{ passed: true }] });
    const pid = 'problem:' + QID + ':' + pr.id;
    T.ok('a problem solved after viewing the solution is saved', !!S.state.mistakes[pid] && S.state.mistakes[pid].card.due === '2026-04-05' && S.state.mistakes[pid].why === 'solution');
    P.recordSubmit(pr.id, 'x', { tests: [{ passed: true }] });
    T.ok('re-solving before it is due does not count', !S.state.mistakes[pid].graduated);
    day = '2026-04-05';
    P.recordSubmit(pr.id, 'x', { tests: [{ passed: true }] });
    T.ok('re-solving on the due date retires it', !!S.state.mistakes[pid].graduated);
    // merging two devices keeps the more-reviewed copy
    const other = S.defaultState(); other.mistakes[id] = { kind: 'check', concept: QID, ref: check.id, card: { reps: 0, lapses: 0, due: '2026-03-02', interval: 0, ease: 2.5 }, graduated: null };
    const mm = S.mergeStates(JSON.parse(S.exportJSON()), other);
    T.ok('merge keeps the retired, more-reviewed copy', !!mm.mistakes[id].graduated);
    T.ok('items whose content is gone are ignored', (() => { S.state.mistakes['quiz:' + QID + ':nope'] = { kind: 'quiz', concept: QID, ref: 'nope', card: { due: '2026-01-01' } }; return !M.all().some(x => x.id === 'quiz:' + QID + ':nope'); })());
  } finally { U.todayKey = realToday; S.reset(); }
}
// Sample-project progress (steps ticked, notes) survives export and merges across devices
{
  const S = MLA.store;
  const a = S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, projects: { 'ensembles:x': { steps: [0, 1], notes: 'phone notes', startedAt: '2026-01-02', updatedAt: '2026-01-03', completedAt: null } } });
  const b = S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, projects: { 'ensembles:x': { steps: [1, 2], notes: 'laptop notes', startedAt: '2026-01-01', updatedAt: '2026-01-04', completedAt: null }, 'pca:y': { steps: [0] } } });
  const m = S.mergeStates(a, b);
  T.ok('projects merge: steps are the union', JSON.stringify(m.projects['ensembles:x'].steps) === '[0,1,2]');
  T.ok('projects merge: both sets of notes are kept', m.projects['ensembles:x'].notes.includes('phone notes') && m.projects['ensembles:x'].notes.includes('laptop notes'));
  T.ok('projects merge: earliest start, projects from both sides', m.projects['ensembles:x'].startedAt === '2026-01-01' && !!m.projects['pca:y']);
  T.ok('projects are part of the export', 'projects' in JSON.parse(S.exportJSON()));
}

// Search: one box finds lessons, sections, terms, problems, projects and case studies
{
  const res = q => MLA.search.query(q);
  // These queries match the sample course (js/data/concepts-sample.js); change them along with the lessons.
  T.ok('search: a lesson title ranks first', (res('collision detection')[0] || {}).route === '/concept/collisions');
  T.ok('search: finds projects', res('coin-collecting').some(r => r.kind === 'Project'));
  T.ok('search: finds problems and glossary terms', res('fixed time steps').some(r => r.kind === 'Problem') && res('tunneling').some(r => r.kind === 'Term'));
  T.ok('search: sections link to their place in the lesson', res('tunneling').some(r => r.kind === 'Section' && /^[/]concept[/]collisions[?]s=/.test(r.route)));
  T.ok('search: searches inside lesson text', res('still pictures').length > 0);
  T.ok('search: an empty query returns nothing', res('   ').length === 0);
}

// Uploaded datasets: stored (in memory under Node, IndexedDB in browsers), read back as bytes, exported with their
// contents in the progress file, and restored by an import into a fresh state; sync codes leave them out.
(async () => {
  const D = MLA.datasets, S = MLA.store;
  const csv = 'a,b' + String.fromCharCode(10) + '1,2' + String.fromCharCode(10) + '3,4' + String.fromCharCode(10);
  const file = (name, text) => ({ name, size: Buffer.byteLength(text), arrayBuffer: async () => new Uint8Array(Buffer.from(text)).buffer });
  const meta = await D.add(file('my data.csv', csv));
  T.ok('datasets: a name with spaces becomes a safe path', meta.name === 'my_data.csv' && meta.rows === 2 && meta.columns === 2, JSON.stringify(meta));
  T.ok('datasets: listed after the built-in ones, and found by code', D.list().some(d => d.name === 'my_data.csv' && !d.bundled) && D.referenced('pd.read_csv("data/my_data.csv")').includes('my_data.csv'));
  T.ok('datasets: contents read back exactly', Buffer.from(await D.read('my_data.csv')).toString() === csv);
  let refused = null; try { await D.add(file('bad.exe', 'x')); } catch (e) { refused = e; }
  T.ok('datasets: other file types are refused', !!refused);
  let dup = null; try { await D.add(file('my data.csv', csv)); } catch (e) { dup = e; }
  T.ok('datasets: re-uploading asks before replacing', !!(dup && dup.exists));
  const full = JSON.parse(await S.exportFull());
  T.ok('datasets: the progress file carries the contents', Array.isArray(full.datasetFiles) && full.datasetFiles.some(f => f.name === 'my_data.csv' && Buffer.from(f.data, 'base64').toString() === csv));
  T.ok('datasets: the sync code does not', !('datasetFiles' in S.exportObject()));
  await D.remove('my_data.csv');
  T.ok('datasets: deleting removes it', !D.find('my_data.csv'));
  S.importJSON(JSON.stringify(full), 'merge');
  const n = await S.datasetImport();
  T.ok('datasets: an import restores them', n === 1 && Buffer.from(await D.read('my_data.csv')).toString() === csv, String(n));
  T.ok('datasets: pandas code to load each kind', D.readCode('x.xlsx').includes('read_excel') && D.readCode('y.tsv').includes('sep=') && D.readCode('z.csv').includes('read_csv'));
  await D.remove('my_data.csv');
  T.done();
})();
