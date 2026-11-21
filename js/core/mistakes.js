/* ML Atlas – "mistakes to revisit". Questions answered wrongly (mastery quizzes, quick checks, "predict first"
 * questions) and problems solved only with help come back on a spaced-repetition schedule (SM-2, the same rules as
 * the flashcards) until they stick. Stored in state.mistakes:
 *   id ("kind:concept:ref") -> { kind, concept, ref, why, added, misses, card: SM-2 state, graduated: date | null }
 * kind: quiz (ref = hash of the question text) | check | predict (ref = section id) | problem (ref = problem id) */
(function (global) {
  const MLA = global.MLA;
  const st = () => MLA.store.state;
  const today = () => MLA.util.todayKey();
  const GRADUATE_DAYS = 15;          // three correct, spaced recalls (after 1 day, 6 days, then 15+) retire an item
  const FIRST_GAP = { quiz: 1, check: 1, predict: 1, problem: 3 };
  const qkey = q => MLA.util.hashString(String((q && q.q) || '')).toString(36);
  function box() { const s = st(); if (!s.mistakes || typeof s.mistakes !== 'object' || Array.isArray(s.mistakes)) s.mistakes = {}; return s.mistakes; }

  // Record a miss. A miss on an item that is already waiting brings it forward; a miss on a retired item restarts it.
  function note(kind, concept, ref, why) {
    if (!FIRST_GAP[kind] || !concept || ref == null) return null;
    const id = kind + ':' + concept + ':' + ref;
    const m = box(), cur = m[id];
    const due = MLA.util.addDays(today(), FIRST_GAP[kind]);
    if (cur && !cur.graduated) {
      cur.misses = (cur.misses || 1) + 1;
      if (!cur.card.due || cur.card.due > due) cur.card.due = due;
      if (why) cur.why = why;
      return cur;
    }
    m[id] = { kind, concept, ref: String(ref), why: why || null, added: today(), misses: 1, card: Object.assign(MLA.srs.defaultCard(), { due }), graduated: null };
    return m[id];
  }

  // Turn a stored item into what the review screen needs; null if the content no longer exists.
  function resolve(id, item) {
    const c = item && MLA.content.byId[item.concept];
    if (!c || c._stub) return null;
    if (item.kind === 'quiz') { const q = c.quiz.find(x => qkey(x) === item.ref); return q && q.type !== 'match' ? { id, item, concept: c, question: q } : null; }
    if (item.kind === 'check') { const s = c.sections.find(x => x.id === item.ref && x.type === 'check'); return s ? { id, item, concept: c, question: s.question, section: s } : null; }
    if (item.kind === 'predict') { const s = c.sections.find(x => x.id === item.ref && x.predict); return s ? { id, item, concept: c, question: s.predict, section: s } : null; }
    if (item.kind === 'problem') { const p = MLA.content.problemById[item.ref]; return p ? { id, item, concept: c, problem: p } : null; }
    return null;
  }
  function all() { const m = box(); return Object.keys(m).map(id => resolve(id, m[id])).filter(Boolean); }
  // For counting: a mistake in a lesson whose file has not loaded yet still counts (it is resolved when reviewed).
  function counted() {
    const m = box();
    return Object.keys(m).map(id => { const c = MLA.content.byId[m[id].concept]; return !c ? null : c._stub ? { id, item: m[id] } : resolve(id, m[id]); }).filter(Boolean);
  }
  function dueItems(day) { day = day || today(); return all().filter(x => !x.item.graduated && x.item.card.due <= day); }
  function stats(day) {
    day = day || today();
    const items = counted();
    const open = items.filter(x => !x.item.graduated);
    return { due: open.filter(x => x.item.card.due <= day).length, waiting: open.filter(x => x.item.card.due > day).length, graduated: items.length - open.length, total: items.length };
  }

  // Grade a review. Correct answers space the item out; a wrong one keeps it due. Problems retire after one unaided re-solve.
  function grade(id, correct) {
    const it = box()[id];
    if (!it || it.graduated) return null;
    it.card = MLA.srs.review(it.card, correct ? 'good' : 'again');
    if (correct && (it.kind === 'problem' || it.card.interval >= GRADUATE_DAYS)) it.graduated = today();
    const s = st(); s.stats.mistakesReviewed = (s.stats.mistakesReviewed || 0) + 1;
    if (correct) MLA.gamification.addXP(3, it.graduated ? 'Mistake fixed for good' : 'Revisited a mistake', 'review');
    else MLA.gamification.touchStreak();
    return it;
  }
  function snooze(id, days) { const it = box()[id]; if (it && !it.graduated) it.card.due = MLA.util.addDays(today(), days || 1); }
  // A problem in the queue counts as revisited when it is solved again on or after its due date.
  function problemSolved(pid) { const m = box(); for (const id of Object.keys(m)) { const it = m[id]; if (it.kind === 'problem' && it.ref === pid && !it.graduated && it.card.due <= today()) grade(id, true); } }

  MLA.mistakes = { note, resolve, all, dueItems, stats, grade, snooze, problemSolved, qkey, GRADUATE_DAYS };
})(typeof globalThis !== 'undefined' ? globalThis : window);
