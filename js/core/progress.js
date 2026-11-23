/* Progression rules: concept & problem state, completion, recommendations, tracks */
(function (global) {
  const MLA = global.MLA;
  const st = () => MLA.store.state;
  const G = () => MLA.gamification;

  if (!MLA.content.ready) MLA.content.init();

  const LEVELS = { 1: 'Beginner', 2: 'Intermediate', 3: 'Advanced' };
  const XP = { read: 2, check: 5, quizPass: 40, quizPerfect: 20, solve: { easy: 25, medium: 50, hard: 90 }, conceptComplete: 60, conceptMastered: 40, daily: 30 };
  const PASS = 70;

  const concepts = () => MLA.data.concepts;
  const problems = () => MLA.data.problems;
  const concept = id => MLA.content.byId[id] || null;
  const problem = id => MLA.content.problemById[id] || null;
  const branch = id => MLA.content.branchById[id] || null;
  const graph = () => MLA.content.graph;

  function cstate(id) { const s = st(); if (!s.concepts[id]) s.concepts[id] = { read: [], checks: {}, quizBest: 0, quizAttempts: 0, quizPassed: false, completedAt: null, masteredAt: null, visitedAt: null }; return s.concepts[id]; }
  const peekC = id => st().concepts[id] || null;
  function pstate(id) { const s = st(); if (!s.problems[id]) s.problems[id] = { draft: null, draftAt: null, solved: false, solvedAt: null, attempts: 0, runs: 0, assisted: false, solutionViewed: false, hintsShown: 0, lastStatus: null, accepted: null }; return s.problems[id]; }
  const peekP = id => st().problems[id] || null;

  const readable = c => c.sections.filter(s => s.type !== 'check').map(s => s.id);
  const isSolved = pid => { const p = peekP(pid); return !!(p && p.solved); };
  const isCompleted = id => { const p = peekC(id); return !!(p && p.completedAt); };
  const isMastered = id => { const p = peekC(id); return !!(p && p.masteredAt); };
  // "Known": lessons the placement test (js/engines/placement.js) found the learner already knows. Distinct from
  // completed: no XP and no achievements, but they count as done for prerequisites, recommendations and tracks.
  // state.known[id] = { on, at, by: 'placement' | 'you', why: 'right' | 'implied' | 'you', via: [lesson ids] }; undoing
  // keeps the record with on: false, so the newest change wins when two devices are merged (MLA.store.mergeStates).
  const knownInfo = id => { const k = (st().known || {})[id]; return k && k.on ? k : null; };
  const isKnown = id => !!knownInfo(id);
  const isDone = id => isCompleted(id) || isKnown(id);

  // Checklist-based progress: each readable section = 1, quiz = 2, each problem = 2
  function conceptProgress(id) {
    const c = concept(id); if (!c) return null;
    const p = peekC(id) || {};
    const secs = readable(c);
    const read = (p.read || []).filter(sid => secs.includes(sid)).length;
    const solved = c.problems.filter(x => isSolved(x.id)).length;
    const hasQuiz = c.quiz.length > 0;
    const total = secs.length + (hasQuiz ? 2 : 0) + c.problems.length * 2;
    const done = read + (hasQuiz && p.quizPassed ? 2 : 0) + solved * 2;
    const needSolved = Math.min(1, c.problems.length);
    return {
      pct: total ? Math.round((done / total) * 100) : 0,
      sectionsRead: read, sections: secs.length,
      solved, problems: c.problems.length, needSolved,
      quizPassed: !!p.quizPassed, quizBest: p.quizBest || 0, quizAttempts: p.quizAttempts || 0, hasQuiz,
      completed: !!p.completedAt, mastered: !!p.masteredAt, known: isKnown(id),
      started: !!(p.visitedAt || read || p.quizAttempts || solved),
      visitedAt: p.visitedAt || null
    };
  }
  function prereqsMet(id) { const c = concept(id); return !!c && c.prereqs.every(isDone); }
  function missingPrereqs(id) { const c = concept(id); return c ? c.prereqs.filter(p => !isDone(p)).map(concept).filter(Boolean) : []; }
  // mastered | completed | known | started | ready | locked
  function status(id) {
    const pr = conceptProgress(id); if (!pr) return 'locked';
    if (pr.mastered) return 'mastered';
    if (pr.completed) return 'completed';
    if (pr.known) return 'known';
    if (pr.started) return 'started';
    return prereqsMet(id) ? 'ready' : 'locked';
  }
  function requirementsText(id) {
    const pr = conceptProgress(id); const parts = [];
    if (pr.hasQuiz && !pr.quizPassed) parts.push('pass the quiz (' + PASS + '%+)');
    if (pr.solved < pr.needSolved) parts.push('solve at least one practice problem');
    return parts.length ? 'To complete this concept: ' + parts.join(' and ') + '.' : '';
  }

  // ---- concept events ----
  function visit(id) { const p = cstate(id); p.visitedAt = new Date().toISOString(); G().touchStreak(); }
  function markRead(id, sid) {
    const c = concept(id); if (!c || !c.sections.some(s => s.id === sid)) return;
    const p = cstate(id);
    if (!p.read.includes(sid)) { p.read.push(sid); G().touchStreak(); finalize(id); }
  }
  function markAllRead(id) { const c = concept(id); if (!c) return; const p = cstate(id); for (const sid of readable(c)) if (!p.read.includes(sid)) p.read.push(sid); G().touchStreak(); finalize(id); }
  function recordCheck(id, sid, correct) {
    const p = cstate(id);
    if (p.checks[sid] != null) return;
    p.checks[sid] = !!correct;
    if (!correct && MLA.mistakes) { const pred = String(sid).startsWith('predict-'); MLA.mistakes.note(pred ? 'predict' : 'check', id, pred ? String(sid).slice(8) : sid); }
    if (correct) { st().stats.checksCorrect++; G().addXP(XP.check, 'Knowledge check', 'check'); }
  }
  function recordQuiz(id, result) {
    const c = concept(id); const p = cstate(id); const s = st();
    p.quizAttempts++; p.quizBest = Math.max(p.quizBest || 0, result.score);
    s.stats.quizzesTaken++;
    s.quizHistory.unshift({ conceptId: id, score: result.score, at: new Date().toISOString() });
    if (MLA.mistakes) for (const r of (result.results || [])) if (!r.correct && r.question && r.question.type !== 'match') MLA.mistakes.note('quiz', id, MLA.mistakes.qkey(r.question));
    if (s.quizHistory.length > 150) s.quizHistory.length = 150;
    const pass = result.score >= PASS;
    if (result.score === 100) s.stats.perfectQuizzes = (s.stats.perfectQuizzes || 0) + 1;
    if (pass) s.stats.quizzesPassed++;
    if (pass && !p.quizPassed) { p.quizPassed = true; G().addXP(XP.quizPass + (result.score === 100 ? XP.quizPerfect : 0), 'Quiz passed: ' + c.title, 'quiz'); }
    else { G().touchStreak(); G().checkAchievements(); }
    finalize(id);
    return pass;
  }
  function finalize(id) {
    const c = concept(id); const p = cstate(id); if (!c) return;
    const pr = conceptProgress(id);
    if (!p.completedAt && (!pr.hasQuiz || pr.quizPassed) && pr.solved >= pr.needSolved) {
      p.completedAt = new Date().toISOString();
      G().addXP(XP.conceptComplete, 'Concept complete: ' + c.title, 'concept');
      const unlocked = graph().children[id].filter(k => !isDone(k) && prereqsMet(k)).map(concept);
      if (unlocked.length && MLA.ui) setTimeout(() => MLA.ui.toast('Ready next: ' + unlocked.map(u => u.title).slice(0, 3).join(', '), 'success', 5500), 700);
    }
    if (p.completedAt && !p.masteredAt && pr.solved === pr.problems && (!pr.hasQuiz || pr.quizBest >= 90)) {
      p.masteredAt = new Date().toISOString();
      G().addXP(XP.conceptMastered, 'Mastered: ' + c.title, 'mastery');
    }
  }

  // ---- saved data from older versions ----
  // Versions up to 6badf8c stored sections read and answered checks by position. Positions are translated
  // through the layout that version had: js/data/legacy-sections.js lists it for the lessons that changed since,
  // and every other lesson still starts with the same sections. Ids are never purely numeric, so the format is
  // self-describing and this is safe to run on any state, including imported files and merged data.
  const isPos = x => typeof x === 'number' || (typeof x === 'string' && /^\d+$/.test(x));
  function legacyLayout(id) { const L = (MLA.data.legacySections || {})[id]; if (L) return L; const c = concept(id); return c ? c.sections.map(s => s.id) : []; }
  function migrateSections(state) {
    let changed = 0;
    for (const id of Object.keys((state && state.concepts) || {})) {
      const cs = state.concepts[id]; if (!cs || typeof cs !== 'object') continue;
      const L = legacyLayout(id);
      if (Array.isArray(cs.read) && cs.read.some(isPos)) { cs.read = Array.from(new Set(cs.read.map(x => isPos(x) ? L[+x] : x).filter(Boolean))); changed++; }
      if (cs.checks && typeof cs.checks === 'object' && Object.keys(cs.checks).some(isPos)) {
        const out = {};
        for (const k of Object.keys(cs.checks)) { if (!isPos(k)) out[k] = cs.checks[k]; }
        for (const k of Object.keys(cs.checks)) { const key = isPos(k) ? L[+k] : null; if (key && out[key] == null) out[key] = cs.checks[k]; }
        cs.checks = out; changed++;
      }
    }
    if (state) state.version = Math.max(state.version || 1, 2);
    return changed;
  }
  migrateSections(st());

  // ---- problem events ----
  function saveDraft(pid, code) { const p = pstate(pid); p.draft = code; p.draftAt = new Date().toISOString(); }
  function recordRun(pid) { const p = pstate(pid); p.runs = (p.runs || 0) + 1; st().stats.runs++; G().touchStreak(); }
  function viewSolution(pid) { const p = pstate(pid); p.solutionViewed = true; if (!p.solved) p.assisted = true; }
  function showHint(pid, n) { const p = pstate(pid); p.hintsShown = Math.max(p.hintsShown || 0, n); }
  function recordSubmit(pid, code, result) {
    const pr = problem(pid); const p = pstate(pid); const s = st();
    p.attempts = (p.attempts || 0) + 1; s.stats.submissions++;
    const tests = (result && result.tests) || [];
    const accepted = !!result && !result.error && tests.length > 0 && tests.every(t => t.passed);
    p.lastStatus = accepted ? 'accepted' : 'failed';
    if (!accepted) { G().touchStreak(); return { accepted: false, first: false }; }
    s.stats.accepted++;
    p.accepted = { code, at: new Date().toISOString() };
    const first = !p.solved;
    if (MLA.mistakes) {
      MLA.mistakes.problemSolved(pid);
      if (first && (p.solutionViewed || p.attempts >= 4)) MLA.mistakes.note('problem', pr.concept, pid, p.solutionViewed ? 'solution' : 'attempts');
    }
    if (first) {
      p.solved = true; p.solvedAt = new Date().toISOString();
      p.assisted = !!p.solutionViewed;
      const base = XP.solve[pr.difficulty] || 25;
      G().addXP(p.assisted ? Math.round(base / 2) : base, 'Solved: ' + pr.title + (p.assisted ? ' (with solution)' : ''), 'problem');
    } else { G().touchStreak(); G().checkAchievements(); }
    const today = MLA.util.todayKey(); const dly = s.daily[today];
    if (dly && dly.problemId === pid && !dly.done) { dly.done = true; G().addXP(XP.daily, 'Problem of the day', 'daily'); }
    finalize(pr.concept);
    return { accepted: true, first };
  }
  function problemStatus(pid) { const p = peekP(pid); if (!p) return 'new'; if (p.solved) return 'solved'; if (p.attempts || p.runs || p.draft) return 'attempted'; return 'new'; }
  function problemOfTheDay() {
    const s = st(); const today = MLA.util.todayKey();
    const cur = s.daily[today];
    if (cur && problem(cur.problemId)) return { problem: problem(cur.problemId), done: !!cur.done };
    const all = problems(); if (!all.length) return null;
    const near = all.filter(p => !isSolved(p.id) && status(p.concept) !== 'locked');
    const pool = near.length ? near : all.filter(p => !isSolved(p.id)).length ? all.filter(p => !isSolved(p.id)) : all;
    const rng = MLA.util.seededRng(MLA.util.hashString(today));
    const pick = pool[Math.floor(rng() * pool.length)];
    s.daily[today] = { problemId: pick.id, done: false };
    const keys = Object.keys(s.daily).sort(); if (keys.length > 90) keys.slice(0, keys.length - 90).forEach(k => delete s.daily[k]);
    return { problem: pick, done: false };
  }

  // ---- tracks & recommendations ----
  const tracks = () => MLA.data.tracks || [];
  const track = id => tracks().find(t => t.id === id) || null;
  function trackConcepts(t) { if (!t) return []; return t.all ? MLA.graph.topoOrder(graph()) : MLA.graph.pathTo(graph(), t.goals); }
  // done: completed; known: marked as known and not completed (they share the bar as a second, lighter segment)
  function trackProgress(t) {
    const ids = trackConcepts(t), done = ids.filter(isCompleted).length, known = ids.filter(id => !isCompleted(id) && isKnown(id)).length;
    return { ids, done, known, total: ids.length, pct: MLA.util.pct(done, ids.length), knownPct: MLA.util.pct(known, ids.length) };
  }
  function recommendNext() {
    const s = st();
    const goal = s.profile.goal ? track(s.profile.goal) : null;
    const scope = goal ? trackConcepts(goal) : MLA.graph.topoOrder(graph());
    const started = scope.filter(id => { const p = conceptProgress(id); return p.started && !p.completed && !p.known; })
      .sort((a, b) => (conceptProgress(b).visitedAt || '').localeCompare(conceptProgress(a).visitedAt || ''));
    if (started.length) return { concept: concept(started[0]), reason: 'Continue where you left off' };
    const next = scope.find(id => !isDone(id) && prereqsMet(id));
    if (next) return { concept: concept(next), reason: goal ? 'Next step on “' + goal.title + '”' : 'Next in the learning order' };
    const any = scope.find(id => !isDone(id));
    if (any) return { concept: concept(any), reason: 'Next concept' };
    if (goal) {                       // the goal is covered (completed, or known after the placement test): the next track
      const sug = suggestTrack();
      if (sug && sug.gap && sug.track.id !== goal.id) return { concept: sug.gap, reason: 'Goal covered. Next: “' + sug.track.title + '”' };
      return { concept: null, reason: 'Track complete. Pick a new goal.' };
    }
    return { concept: null, reason: 'Everything is complete. Well done.' };
  }
  function readyList(limit) {
    return MLA.graph.topoOrder(graph()).filter(id => !isDone(id) && prereqsMet(id)).slice(0, limit || 6).map(concept);
  }

  // ---- lessons marked as known (placement test) ----
  function setKnown(id, on, info) {
    if (!concept(id)) return false;
    const s = st(), now = new Date().toISOString(), prev = (s.known || {})[id];
    if (!s.known) s.known = {};
    if (on) {
      info = info || {};
      s.known[id] = { on: true, at: now, by: info.by || 'you', why: info.why || 'you', via: (info.via || []).slice(0, 3) };
      if (info.asked) Object.assign(s.known[id], { asked: info.asked, right: info.right || 0 });   // how many of its own questions were asked, and right
    }
    else if (prev && prev.on) s.known[id] = Object.assign({}, prev, { on: false, at: now });
    return true;
  }
  function knownList() { return MLA.graph.topoOrder(graph()).filter(isKnown).map(id => Object.assign({ id, concept: concept(id) }, knownInfo(id))); }
  function unmarkAll() { const ids = knownList().map(k => k.id); ids.forEach(id => setKnown(id, false)); return ids.length; }
  // A finished placement test replaces the marks of the previous one; marks the learner set themselves stay.
  function applyPlacement(marked) {
    const keep = new Set((marked || []).map(m => m.id));
    for (const k of knownList()) if (k.by === 'placement' && !keep.has(k.id)) setKnown(k.id, false);
    for (const m of marked || []) setKnown(m.id, true, { by: 'placement', why: m.why, via: m.via, asked: m.asked, right: m.right });
    return keep.size;
  }
  // The track to follow after the placement test, starting at its first lesson that is neither completed nor known:
  // the learner's goal while it has such a lesson, otherwise the first track, in the listed order from beginner to
  // advanced, that still has one (reason 'next' when the goal is covered, 'first' when there is no goal).
  function suggestTrack() {
    const info = (t, reason) => { const ids = trackConcepts(t), gap = ids.find(id => !isDone(id)); return { track: t, gap: gap ? concept(gap) : null, done: ids.filter(isDone).length, total: ids.length, reason }; };
    const goal = st().profile.goal ? track(st().profile.goal) : null;
    if (goal) { const g = info(goal, 'goal'); if (g.gap) return g; }
    const open = tracks().filter(t => !t.hidden && !t.all && !t.custom).map(t => info(t, goal ? 'next' : 'first')).filter(x => x.gap);
    if (open.length) return Object.assign(open[0], { goal });
    const all = tracks().find(t => t.all), a = all ? info(all, 'all') : null;
    return a && a.gap ? a : null;
  }
  function branchStats() {
    return MLA.data.branches.map(b => {
      const list = concepts().filter(c => c.branch === b.id);
      const probs = problems().filter(p => concept(p.concept).branch === b.id);
      const done = list.filter(c => isCompleted(c.id)).length;
      const solved = probs.filter(p => isSolved(p.id)).length;
      return { branch: b, concepts: list.length, done, problems: probs.length, solved, pct: MLA.util.pct(done * 2 + solved, list.length * 2 + probs.length) };
    }).filter(x => x.concepts > 0);
  }
  function overall() {
    const cs = concepts(), ps = problems();
    const done = cs.filter(c => isCompleted(c.id)).length, solved = ps.filter(p => isSolved(p.id)).length;
    const known = cs.filter(c => !isCompleted(c.id) && isKnown(c.id)).length;
    return { conceptsDone: done, concepts: cs.length, known, solved, problems: ps.length, pct: MLA.util.pct(done + solved, cs.length + ps.length) };
  }

  MLA.progress = { LEVELS, XP, PASS, concepts, problems, concept, problem, branch, graph, cstate, pstate, peekC, peekP, conceptProgress, prereqsMet, missingPrereqs, status, requirementsText, isCompleted, isMastered, isSolved, isKnown, isDone, knownInfo, setKnown, knownList, unmarkAll, applyPlacement, suggestTrack, visit, markRead, markAllRead, recordCheck, migrateSections, legacyLayout, recordQuiz, finalize, saveDraft, recordRun, viewSolution, showHint, recordSubmit, problemStatus, problemOfTheDay, tracks, track, trackConcepts, trackProgress, recommendNext, readyList, branchStats, overall };
})(typeof globalThis !== 'undefined' ? globalThis : window);
