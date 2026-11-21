/* ML Atlas – XP, levels, streaks, achievements */
(function (global) {
  const MLA = global.MLA;
  const st = () => MLA.store.state;
  const U = () => MLA.util;

  function xpForLevel(n) { return n <= 1 ? 0 : Math.round(100 * Math.pow(n - 1, 1.55)); }
  function levelInfo(xp) {
    xp = xp == null ? st().xp : xp;
    let level = 1; while (xpForLevel(level + 1) <= xp) level++;
    const cur = xpForLevel(level), next = xpForLevel(level + 1);
    return { level, cur, next, into: xp - cur, needed: next - cur, pct: Math.min(100, Math.round(((xp - cur) / (next - cur)) * 100)) };
  }
  const TITLES = ['Curious Mind', 'Data Explorer', 'Array Wrangler', 'Feature Crafter', 'Gradient Descender', 'Model Trainer', 'Loss Minimizer', 'Neural Tinkerer', 'Backprop Adept', 'Attention Seeker', 'Transformer Tamer', 'Token Whisperer', 'LLM Builder', 'Research Engineer', 'ML Sage'];
  function title(level) { return TITLES[Math.min(TITLES.length - 1, Math.max(0, (level || levelInfo().level) - 1))]; }

  function recordActivity(type, label, xp) {
    const s = st();
    s.activity.unshift({ t: new Date().toISOString(), type, label, xp: xp || 0 });
    if (s.activity.length > 80) s.activity.length = 80;
  }
  function addXP(amount, label, type) {
    if (!amount) return;
    const s = st();
    const before = levelInfo(s.xp).level;
    s.xp += amount;
    recordActivity(type || 'xp', label || 'XP earned', amount);
    if (MLA.ui) MLA.ui.toast('+' + amount + ' XP · ' + (label || ''), 'xp');
    const after = levelInfo(s.xp).level;
    if (after > before && MLA.ui) setTimeout(() => MLA.ui.toast('Level up. You are now level ' + after + ': ' + title(after), 'success', 5000), 400);
    touchStreak();
    checkAchievements();
  }
  function touchStreak() {
    const s = st(); const today = U().todayKey();
    if (s.streak.lastActive === today) { s.streak.days[today] = true; return; }
    const yesterday = U().addDays(today, -1);
    s.streak.current = s.streak.lastActive === yesterday ? (s.streak.current || 0) + 1 : 1;
    s.streak.lastActive = today; s.streak.days[today] = true;
    if (s.streak.current > (s.streak.best || 0)) s.streak.best = s.streak.current;
    const keys = Object.keys(s.streak.days).sort(); if (keys.length > 120) keys.slice(0, keys.length - 120).forEach(k => delete s.streak.days[k]);
  }
  function streakStatus() {
    const s = st(); const today = U().todayKey(); const yesterday = U().addDays(today, -1);
    const alive = s.streak.lastActive === today || s.streak.lastActive === yesterday;
    return { current: alive ? (s.streak.current || 0) : 0, best: s.streak.best || 0, activeToday: s.streak.lastActive === today, days: s.streak.days };
  }

  // Counters used by achievements
  function context() {
    const s = st(); const P = MLA.progress;
    const concepts = MLA.data.concepts || [];
    const done = concepts.filter(c => P.isCompleted(c.id));
    const branchDone = {};
    for (const b of MLA.data.branches) { const list = concepts.filter(c => c.branch === b.id); branchDone[b.id] = list.length > 0 && list.every(c => P.isCompleted(c.id)); }
    const probs = MLA.data.problems || [];
    const solved = probs.filter(p => s.problems[p.id] && s.problems[p.id].solved);
    return {
      xp: s.xp, stats: s.stats,
      conceptsDone: done.length, conceptsTotal: concepts.length,
      mastered: concepts.filter(c => P.isMastered(c.id)).length,
      branchDone,
      solved: solved.length, problemsTotal: probs.length,
      hardSolved: solved.filter(p => p.difficulty === 'hard').length,
      mediumSolved: solved.filter(p => p.difficulty === 'medium').length,
      unassisted: solved.filter(p => !s.problems[p.id].assisted && !(s.problems[p.id].hintsShown > 0)).length,
      perfectQuizzes: s.stats.perfectQuizzes || 0, quizzesPassed: s.stats.quizzesPassed || 0,
      bestStreak: s.streak.best || 0,
      dailyDone: Object.keys(s.daily || {}).filter(k => s.daily[k] && s.daily[k].done).length,
      playgroundRuns: s.stats.playgroundRuns || 0, sampleRuns: s.stats.sampleRuns || 0,
      flashReviews: s.stats.flashReviews || 0, exports: s.stats.exports || 0,
      notes: Object.values(s.notes || {}).filter(n => n && n.trim()).length
    };
  }
  function checkAchievements() {
    const s = st(); if (!MLA.progress || !MLA.content || !MLA.content.ready) return [];
    const ctx = context(); const earned = [];
    for (const a of MLA.data.achievements || []) {
      if (s.achievements[a.id]) continue;
      let hit = false; try { hit = !!a.test(ctx); } catch (e) { hit = false; }
      if (hit) { s.achievements[a.id] = new Date().toISOString(); earned.push(a); }
    }
    for (const a of earned) {
      recordActivity('achievement', 'Achievement unlocked: ' + a.title, a.xp);
      if (a.xp) s.xp += a.xp;
      if (MLA.ui) MLA.ui.toast('Achievement unlocked: ' + a.title + (a.xp ? ' (+' + a.xp + ' XP)' : ''), 'success', 5000);
    }
    return earned;
  }
  function achievementProgress(a) {
    const ctx = context();
    try { const p = a.progress ? a.progress(ctx) : [0, 1]; return { done: Math.min(p[0], p[1]), total: p[1], pct: Math.min(100, Math.round((p[0] / p[1]) * 100)) }; }
    catch (e) { return { done: 0, total: 1, pct: 0 }; }
  }

  MLA.gamification = { xpForLevel, levelInfo, title, addXP, recordActivity, touchStreak, streakStatus, context, checkAchievements, achievementProgress };
})(typeof globalThis !== 'undefined' ? globalThis : window);
