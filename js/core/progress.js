/* ML Atlas – progression rules: concept & problem state, completion, recommendations, tracks */
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

  const readable = c => c.sections.map((s, i) => i).filter(i => c.sections[i].type !== 'check');
  const isSolved = pid => { const p = peekP(pid); return !!(p && p.solved); };
  const isCompleted = id => { const p = peekC(id); return !!(p && p.completedAt); };
  const isMastered = id => { const p = peekC(id); return !!(p && p.masteredAt); };

  // Checklist-based progress: each readable section = 1, quiz = 2, each problem = 2
  function conceptProgress(id) {
    const c = concept(id); if (!c) return null;
    const p = peekC(id) || {};
    const secs = readable(c);
    const read = (p.read || []).filter(i => secs.includes(i)).length;
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
      completed: !!p.completedAt, mastered: !!p.masteredAt,
      started: !!(p.visitedAt || read || p.quizAttempts || solved),
      visitedAt: p.visitedAt || null
    };
  }
  function prereqsMet(id) { const c = concept(id); return !!c && c.prereqs.every(isCompleted); }
  function missingPrereqs(id) { const c = concept(id); return c ? c.prereqs.filter(p => !isCompleted(p)).map(concept).filter(Boolean) : []; }
  // mastered | completed | started | ready | locked
  function status(id) {
    const pr = conceptProgress(id); if (!pr) return 'locked';
    if (pr.mastered) return 'mastered';
    if (pr.completed) return 'completed';
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
  function markRead(id, idx) {
    const c = concept(id); if (!c) return;
    const p = cstate(id);
    if (!p.read.includes(idx)) { p.read.push(idx); G().touchStreak(); finalize(id); }
  }
  function markAllRead(id) { const c = concept(id); if (!c) return; const p = cstate(id); for (const i of readable(c)) if (!p.read.includes(i)) p.read.push(i); G().touchStreak(); finalize(id); }
  function recordCheck(id, idx, correct) {
    const p = cstate(id);
    if (p.checks[idx] != null) return;
    p.checks[idx] = !!correct;
    if (correct) { st().stats.checksCorrect++; G().addXP(XP.check, 'Knowledge check', 'check'); }
  }
  function recordQuiz(id, result) {
    const c = concept(id); const p = cstate(id); const s = st();
    p.quizAttempts++; p.quizBest = Math.max(p.quizBest || 0, result.score);
    s.stats.quizzesTaken++;
    s.quizHistory.unshift({ conceptId: id, score: result.score, at: new Date().toISOString() });
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
      const unlocked = graph().children[id].filter(k => !isCompleted(k) && prereqsMet(k)).map(concept);
      if (unlocked.length && MLA.ui) setTimeout(() => MLA.ui.toast('🔓 Ready next: ' + unlocked.map(u => u.title).slice(0, 3).join(', '), 'success', 5500), 700);
    }
    if (p.completedAt && !p.masteredAt && pr.solved === pr.problems && (!pr.hasQuiz || pr.quizBest >= 90)) {
      p.masteredAt = new Date().toISOString();
      G().addXP(XP.conceptMastered, 'Mastered: ' + c.title, 'mastery');
    }
  }

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
  function trackProgress(t) { const ids = trackConcepts(t); const done = ids.filter(isCompleted).length; return { ids, done, total: ids.length, pct: MLA.util.pct(done, ids.length) }; }
  function recommendNext() {
    const s = st();
    const goal = s.profile.goal ? track(s.profile.goal) : null;
    const scope = goal ? trackConcepts(goal) : MLA.graph.topoOrder(graph());
    const started = scope.filter(id => { const p = conceptProgress(id); return p.started && !p.completed; })
      .sort((a, b) => (conceptProgress(b).visitedAt || '').localeCompare(conceptProgress(a).visitedAt || ''));
    if (started.length) return { concept: concept(started[0]), reason: 'Continue where you left off' };
    const next = scope.find(id => !isCompleted(id) && prereqsMet(id));
    if (next) return { concept: concept(next), reason: goal ? 'Next step on “' + goal.title + '”' : 'Next in the learning order' };
    const any = scope.find(id => !isCompleted(id));
    if (any) return { concept: concept(any), reason: 'Next concept' };
    if (goal) return { concept: null, reason: 'Track complete! Pick a new goal.' };
    return { concept: null, reason: 'Everything is complete — amazing work!' };
  }
  function readyList(limit) {
    return MLA.graph.topoOrder(graph()).filter(id => !isCompleted(id) && prereqsMet(id)).slice(0, limit || 6).map(concept);
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
    return { conceptsDone: done, concepts: cs.length, solved, problems: ps.length, pct: MLA.util.pct(done + solved, cs.length + ps.length) };
  }

  MLA.progress = { LEVELS, XP, PASS, concepts, problems, concept, problem, branch, graph, cstate, pstate, peekC, peekP, conceptProgress, prereqsMet, missingPrereqs, status, requirementsText, isCompleted, isMastered, isSolved, visit, markRead, markAllRead, recordCheck, recordQuiz, finalize, saveDraft, recordRun, viewSolution, showHint, recordSubmit, problemStatus, problemOfTheDay, tracks, track, trackConcepts, trackProgress, recommendNext, readyList, branchStats, overall };
})(typeof globalThis !== 'undefined' ? globalThis : window);
