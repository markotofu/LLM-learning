/* The placement test (js/engines/placement.js) and the "known" progress state it sets (js/core/progress.js,
 * js/core/store.js): the question pool, how questions are chosen along the prerequisite graph, what the answers
 * imply, how often simulated learners are marked wrongly, and how the marks are undone, exported and merged.
 * Run: node tests/placement.test.js */
'use strict';
const fs = require('fs');
const path = require('path');
const { load, suite, root } = require('./_load.js');
const MLA = load();
const T = suite('placement');
const PL = MLA.placement, P = MLA.progress, S = MLA.store, U = MLA.util;

T.ok('scripts load without errors', MLA.__loadErrors.length === 0, MLA.__loadErrors.join('\n'));
T.ok('engine and page are loaded', !!PL && !!MLA.components['view-placement'] && !!MLA.components['placement-q']);

// ---- the question pool ----
const ids = PL.scope();
const inScope = c => PL.BRANCHES.includes(c.branch) && c.level <= 2 && !c.checkpoints;
T.ok('the placement branches exist', PL.BRANCHES.length > 0 && PL.BRANCHES.every(b => MLA.data.branches.some(x => x.id === b)), PL.BRANCHES.join(', '));
T.ok('the pool covers its lessons', ids.length === PL.POOL.length && ids.length >= 1, ids.length + ' of ' + PL.POOL.length);
for (const [id, openings] of PL.POOL) {
  const c = MLA.content.byId[id];
  T.ok('pool lesson exists: ' + id, !!c);
  if (!c) continue;
  T.ok(id + ': in a placement branch (' + PL.BRANCHES.join(', ') + ')', PL.BRANCHES.includes(c.branch), c.branch);
  T.ok(id + ': beginner or intermediate, not a capstone', c.level <= 2 && !c.checkpoints, 'level ' + c.level);
  const qs = PL.questions(id);
  T.ok(id + ': every opening finds exactly one question of an allowed type', qs.length === openings.length && qs.every(Boolean), openings.filter((o, k) => !qs[k]).join(' | '));
  for (const q of qs.filter(Boolean)) {
    T.ok(id + ': question is one of the lesson\'s own quiz questions', c.quiz.includes(q));
    T.ok(id + ': question is valid and quick to answer', MLA.quiz.validQuestion(q) && ['mcq', 'fill', 'multi'].includes(q.type) && q.q.length < 400, q.type);
  }
  T.ok(id + ': its questions are different', new Set(qs).size === qs.length);
}
for (const [a, b] of PL.SOFT) T.ok('soft link ' + a + ' > ' + b + ' is a real prerequisite', (MLA.content.byId[b].prereqs || []).includes(a));
const lessonsOutside = MLA.data.concepts.filter(c => inScope(c) && !ids.includes(c.id)).map(c => c.id);
T.ok('every beginner and intermediate lesson of those branches is in the pool', lessonsOutside.length === 0, lessonsOutside.join(', '));

// ---- a session: questions along the graph ----
const fresh = PL.create({ seed: 11 });
const first = fresh.next();
T.ok('the first question is chosen, with a usable question', !!first && !!first.q && ids.includes(first.id));
T.ok('a session is deterministic for a seed', JSON.stringify(PL.create({ seed: 11 }).next().id) === JSON.stringify(first.id));
// one answer says something about a group of lessons: a right answer raises what it builds on, a skip lowers what builds on it
{
  const deep = ids.slice().sort((a, b) => fresh.ancestors(b).length - fresh.ancestors(a).length)[0];
  const shallow = ids.slice().sort((a, b) => fresh.descendants(b).length - fresh.descendants(a).length)[0];
  const base = PL.create({ seed: 5 }).marginals();
  const up = PL.create({ seed: 5 }); up.record(deep, 'right', 0);
  const pu = up.marginals(), anc = up.ancestors(deep);
  if (anc.length) T.ok('a right answer raises every lesson it builds on', anc.every(a => pu[a] > base[a]), anc.filter(a => !(pu[a] > base[a])).join(','));
  const down = PL.create({ seed: 5 }); down.record(shallow, 'skip', 0);
  const pd = down.marginals(), desc = down.descendants(shallow);
  if (desc.length) T.ok('"I don\'t know yet" lowers every lesson that builds on it', desc.every(d => pd[d] < base[d]), desc.filter(d => !(pd[d] < base[d])).join(','));
  T.ok('a skipped lesson is not asked again', !down.record(shallow, 'right', 1));
  T.ok('a lesson cannot be answered twice with the same question', !up.record(deep, 'right', 0));
}

// Simulated learners. Known lessons: right 88%, wrong 10%, "I don't know yet" 2%. Unknown lessons: "I don't know
// yet" with probability h, otherwise a guess (1 in 4 for four options, about 1 in 100 for typed answers).
function closed(list) { const out = new Set(list); for (const id of list) fresh.ancestors(id).forEach(a => out.add(a)); return out; }
function simulate(known, h, seed) {
  const s = PL.create({ seed }), rng = U.seededRng(seed * 7919 + 13);
  let nx, prev = null; const seen = {}; let twiceInARow = false, afterSkip = false;
  while ((nx = s.next())) {
    if (prev === nx.id) twiceInARow = true;
    if (seen[nx.id] === 'skip') afterSkip = true;
    const u = rng(); let r;
    if (known.has(nx.id)) r = u < 0.88 ? 'right' : u < 0.98 ? 'wrong' : 'skip';
    else if (u < h) r = 'skip';
    else r = rng() < (nx.q.type === 'fill' ? 0.01 : 1 / ((nx.q.options || []).length || 4)) ? 'right' : 'wrong';
    s.record(nx.id, r, nx.k); seen[nx.id] = r; prev = nx.id;
  }
  const res = s.result(), marked = res.marked.map(m => m.id);
  return { s, res, n: s.answers.length, wrong: marked.filter(id => !known.has(id)).length, right: marked.filter(id => known.has(id)).length, twiceInARow, afterSkip };
}
const RUNS = 12;
const profiles = {                                   // tuned for small pools too; a larger course can tighten them
  'a beginner': { known: new Set(), h: 0.7, maxQ: PL.MAX, maxWrong: 0.5, minRecall: 0 },
  'a guesser who knows nothing': { known: new Set(), h: 0, maxQ: PL.MAX, maxWrong: 1, minRecall: 0 },
  'an expert': { known: new Set(ids), h: 0.5, maxQ: PL.MAX, maxWrong: 0, minRecall: 0.6 }
};
const t0 = Date.now();
for (const [name, pr] of Object.entries(profiles)) {
  let q = 0, wrong = 0, right = 0, rules = true, reasons = true, closedOk = true;
  for (let r = 0; r < RUNS; r++) {
    const o = simulate(pr.known, pr.h, 1 + r);
    q += o.n; wrong += o.wrong; right += o.right;
    if (o.twiceInARow || o.afterSkip || o.n > PL.MAX) rules = false;
    for (const m of o.res.marked) if (!(m.why === 'right' ? m.right > 0 : m.via.length > 0 && m.via.every(v => o.s.ancestors(v).includes(m.id)))) reasons = false;
    // what is marked includes what it builds on, except lessons whose own question was missed
    const marked = new Set(o.res.marked.map(m => m.id)), missed = new Set(o.res.notMarked.filter(n => n.r !== 'right').map(n => n.id));
    for (const m of marked) for (const a of o.s.ancestors(m)) if (!marked.has(a) && !missed.has(a)) closedOk = false;
  }
  T.ok(name + ': at most ' + pr.maxQ + ' questions on average', q / RUNS <= pr.maxQ, (q / RUNS).toFixed(1));
  T.ok(name + ': wrong marks stay rare', wrong / RUNS <= pr.maxWrong, (wrong / RUNS).toFixed(2) + ' per test');
  if (pr.known.size) T.ok(name + ': most of what they know is marked', right / RUNS / pr.known.size >= pr.minRecall, (right / RUNS / pr.known.size).toFixed(2));
  T.ok(name + ': never the same lesson twice in a row, never again after "I don\'t know yet", at most ' + PL.MAX + ' questions', rules);
  T.ok(name + ': every mark has a reason (its own right answer, or a right answer on a lesson that builds on it)', reasons);
  T.ok(name + ': marks include the lessons they build on', closedOk);
}
T.ok('the simulations are quick', Date.now() - t0 < 30000, (Date.now() - t0) + ' ms');
{
  const all = PL.create({ seed: 3 }); let nx, n = 0;
  while ((nx = all.next())) { all.record(nx.id, 'right', nx.k); n++; }
  T.ok('all answers right: at most ' + PL.MAX + ' questions, every lesson marked', n >= 1 && n <= PL.MAX && all.result().marked.length === ids.length, n + ' questions, ' + all.result().marked.length + ' marked');
  const none = PL.create({ seed: 3 }); n = 0;
  while ((nx = none.next())) { none.record(nx.id, 'skip', nx.k); n++; }
  T.ok('all "I don\'t know yet": a short test that marks nothing', n <= PL.MAX && none.result().marked.length === 0 && none.result().notMarked.every(x => x.r === 'skip'), n + ' questions');
  // replaying saved answers gives the same next question and result (how a test in progress resumes)
  const a = PL.create({ seed: 21 }); for (let i = 0; i < Math.min(4, ids.length); i++) { const q = a.next(); if (!q) break; a.record(q.id, i % 2 ? 'skip' : 'right', q.k); }
  const b = PL.create({ seed: 21 }); a.answers.forEach(x => b.record(x.id, x.r, x.k));
  const an = a.next(), bn = b.next();
  T.ok('replayed answers resume at the same question', (an && an.id) === (bn && bn.id) && JSON.stringify(a.result().marked) === JSON.stringify(b.result().marked));
  const retake = PL.create({ seed: 21, avoid: a.answers.map(x => PL.qkey(x.id, x.k)) }).next();
  const asked = new Set(a.answers.map(x => PL.qkey(x.id, x.k)));
  const unasked = PL.questions(retake.id).some((q, k) => q && !asked.has(PL.qkey(retake.id, k)));
  T.ok('a retake asks a question not asked last time, when its lesson has one', !unasked || !asked.has(PL.qkey(retake.id, retake.k)), retake.id + '#' + retake.k);
}

// ---- the known state ----
// X: a lesson that builds on at least one other; its ancestors are marked known.
const graphOf = () => MLA.content.graph;
const X = MLA.data.concepts.find(c => MLA.graph.ancestors(graphOf(), c.id).size >= 2) || MLA.data.concepts.find(c => c.prereqs.length);
const ANC = MLA.graph.topoOrder(graphOf()).filter(id => MLA.graph.ancestors(graphOf(), X.id).has(id));
const A = ANC[0], TR = P.tracks().find(t => !t.hidden && !t.all && !t.custom && P.trackConcepts(t).includes(X.id));
S.reset();
const xp0 = S.state.xp;
T.ok('a lesson starts unmarked', !P.isKnown(X.id) && P.status(X.id) !== 'known');
P.setKnown(A, true, { by: 'placement', why: 'right' });
for (const id of ANC.slice(1)) P.setKnown(id, true, { by: 'placement', why: 'implied', via: [X.id] });
T.ok('known is its own status, not completed', P.status(A) === 'known' && !P.isCompleted(A) && P.isDone(A));
T.ok('known gives no XP and completes nothing', S.state.xp === xp0 && P.overall().conceptsDone === 0 && P.overall().known === ANC.length);
T.ok('known lessons count as prerequisites done', P.prereqsMet(X.id) && P.status(X.id) === 'ready' && P.missingPrereqs(X.id).length === 0);
if (TR) {
  S.state.profile.goal = TR.id;
  T.ok('recommendations skip known lessons', P.recommendNext().concept && !P.isKnown(P.recommendNext().concept.id));
  T.ok('ready lists skip known lessons', P.readyList(20).every(c => !P.isKnown(c.id)));
  const tp = P.trackProgress(P.track(TR.id));
  T.ok('tracks count known lessons apart from completed ones', tp.done === 0 && tp.known === ANC.filter(id => P.trackConcepts(TR).includes(id)).length && tp.knownPct > 0 && tp.pct === 0, JSON.stringify({ done: tp.done, known: tp.known }));
  const sug = P.suggestTrack();
  T.ok('the suggestion is the goal, starting at its first lesson that is neither completed nor known', sug && sug.track.id === TR.id && sug.reason === 'goal' && sug.gap && !P.isDone(sug.gap.id));
  S.state.profile.goal = null;
}
P.setKnown(A, false);
T.ok('undo one lesson', !P.isKnown(A) && P.status(A) !== 'known' && S.state.known[A].on === false);
T.ok('an undo keeps the reason, to mark it again', S.state.known[A].why === 'right');
P.setKnown(A, true, { by: 'you' });
T.ok('mark again', P.isKnown(A) && P.knownInfo(A).by === 'you');
T.ok('known list in learning order', JSON.stringify(P.knownList().map(k => k.id)) === JSON.stringify(MLA.graph.topoOrder(graphOf()).filter(id => P.isKnown(id))));
// a finished test replaces the previous test's marks, but keeps the ones the learner set
P.applyPlacement([{ id: X.id, why: 'right', via: [] }]);
T.ok('a new test replaces the last test\'s marks', P.isKnown(X.id) && ANC.slice(1).every(id => !P.isKnown(id)));
T.ok('marks the learner set themselves stay', P.isKnown(A));
T.ok('undo all', P.unmarkAll() === 2 && P.knownList().length === 0 && P.overall().known === 0);
// a completed lesson stays completed when it is also known
P.setKnown(A, true, { by: 'placement', why: 'right' });
P.cstate(A).completedAt = '2026-01-01T00:00:00Z';
T.ok('completed outranks known', P.status(A) === 'completed' && (!TR || P.trackProgress(P.track(TR.id)).known === 0));
S.reset();

// ---- export, import and merge ----
{
  const ORD = MLA.graph.topoOrder(MLA.content.graph), L1 = ORD[0], L2 = ORD[1], L3 = ORD[2];   // three course lessons
  const at = (d) => '2026-0' + d + 'T10:00:00.000Z';
  const a = S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, known: {
    [L1]: { on: true, at: at('1-02'), by: 'placement', why: 'right', via: [] },
    [L2]: { on: true, at: at('1-05'), by: 'placement', why: 'implied', via: [L3] } } });
  const b = S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, known: {
    [L1]: { on: false, at: at('1-03'), by: 'placement', why: 'right', via: [] },
    [L2]: { on: false, at: at('1-04'), by: 'placement', why: 'implied', via: [] },
    [L3]: { on: true, at: at('1-01'), by: 'you', why: 'you', via: [] } } });
  const ab = S.mergeStates(a, b), ba = S.mergeStates(b, a);
  T.ok('merge: an undo that came later wins', ab.known[L1].on === false && ba.known[L1].on === false);
  T.ok('merge: a mark that came later wins', ab.known[L2].on === true && ba.known[L2].on === true);
  T.ok('merge: marks from both devices are kept', ab.known[L3].on === true && ba.known[L3].on === true);
  T.ok('merge: the same in both directions', JSON.stringify(ab.known) === JSON.stringify(ba.known));
  // placement attempts: the newer finished test wins; a test in progress older than it is dropped
  const done1 = { seed: 1, startedAt: at('2-01'), finishedAt: at('2-02'), answers: [{ id: L1, k: 0, r: 'right', v: '(10,)' }], marked: [{ id: L1, why: 'right', via: [] }], notMarked: [] };
  const done2 = Object.assign({}, done1, { seed: 2, startedAt: at('3-01'), finishedAt: at('3-02') });
  const oldRun = { seed: 3, startedAt: at('2-05'), answers: [{ id: L3, k: 0, r: 'skip', v: '' }] };
  const newRun = { seed: 4, startedAt: at('4-01'), answers: [] };
  const pm = S.mergeStates(S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, placement: { current: oldRun, last: done1 } }), S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, placement: { current: null, last: done2 } }));
  T.ok('merge: the newer finished test is kept', pm.placement.last.seed === 2);
  T.ok('merge: a test in progress that started before the last finished one is dropped', pm.placement.current === null);
  const pn = S.mergeStates(S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, placement: { current: newRun, last: done1 } }), S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, placement: { current: oldRun, last: done2 } }));
  T.ok('merge: a newer test in progress survives', pn.placement.current && pn.placement.current.seed === 4 && pn.placement.last.seed === 2);
  T.ok('defaults: odd placement data is ignored', JSON.stringify(S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, placement: [1, 2] }).placement) === '{"current":null,"last":null}' && JSON.stringify(S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, placement: { current: 'x', last: { answers: 3 } } }).placement) === '{"current":null,"last":null}');
  T.ok('defaults: known must be an object', JSON.stringify(S.mergeDefaults({ xp: 0, concepts: {}, problems: {}, known: [1] }).known) === '{}');
  // round trip through the progress file and the sync code's JSON
  S.reset();
  P.setKnown(L1, true, { by: 'placement', why: 'right' }); S.state.placement.last = done1;
  const file = S.exportJSON();
  T.ok('export: known lessons and the last test are in the progress file', JSON.parse(file).known[L1].on === true && JSON.parse(file).placement.last.seed === 1);
  S.reset();
  S.importJSON(file, 'replace');
  T.ok('import (replace) restores them', P.isKnown(L1) && S.state.placement.last.seed === 1);
  P.setKnown(L1, false);
  const undone = S.exportJSON();
  S.reset(); P.setKnown(L1, true, { by: 'placement', why: 'right' });
  S.state.known[L1].at = '2020-01-01T00:00:00.000Z';                    // marked long before the undo in the file
  S.importJSON(undone, 'merge');
  T.ok('import (merge) applies an undo made on the other device', !P.isKnown(L1));
  S.reset();
}

// ---- the page: a route, findable in search ----
T.ok('the placement test has its route', MLA.router.parseHash('#/placement').name === 'placement');
T.ok('search finds the placement test', MLA.search.query('placement test').some(r => r.route === '/placement'));

// ---- the page's text follows the writing rules ----
for (const f of ['js/views/placement-view.js', 'js/engines/placement.js']) {
  const text = fs.readFileSync(path.join(root, f), 'utf8');
  T.ok(f + ': no em dashes', !text.includes('—'));
  T.ok(f + ': no exclamation marks in sentences', !/[A-Za-z]![\s"'<]/.test(text));
}
T.done();
