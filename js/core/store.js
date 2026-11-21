/* ML Atlas – persistent reactive state (localStorage) + export / import / merge for multi-device sync */
(function (global) {
  const MLA = global.MLA;
  const KEY = 'mla.state.v1';
  const VERSION = 2; // 2: section progress is keyed by section id (see MLA.progress.migrateSections)
  const APP_ID = 'ml-atlas';

  function defaultState() {
    return {
      app: APP_ID,
      version: VERSION,
      profile: { name: 'Learner', createdAt: new Date().toISOString(), onboarded: false, goal: null, customGoal: null, theme: 'light', style: 'baroque', editorFontSize: 14, timeoutSec: 20, preloadPython: false, mapLayout: 'mindmap', experience: null, voice: '', speechRate: 1, speechFollow: true, speechCaptions: true, speechContinue: false, listenAt: {} },
      xp: 0,
      streak: { current: 0, best: 0, lastActive: null, days: {} },
      concepts: {},   // id -> { read:[], checks:{}, quizBest, quizAttempts, quizPassed, completedAt, masteredAt, visitedAt }
      problems: {},   // id -> { draft, draftAt, solved, solvedAt, attempts, runs, assisted, solutionViewed, hintsShown, lastStatus, accepted:{code,at} }
      notes: {},      // conceptId -> text
      playground: { files: [], active: null },
      flashcards: {}, // cardId -> SRS state
      mistakes: {},   // 'kind:concept:ref' -> { kind, concept, ref, why, added, misses, card, graduated } (see js/core/mistakes.js)
      achievements: {},
      daily: {},      // dateKey -> { problemId, done }
      activity: [],
      stats: { runs: 0, submissions: 0, accepted: 0, playgroundRuns: 0, sampleRuns: 0, quizzesTaken: 0, quizzesPassed: 0, perfectQuizzes: 0, checksCorrect: 0, flashReviews: 0, mistakesReviewed: 0, exports: 0, imports: 0 },
      quizHistory: []
    };
  }

  function mergeDefaults(saved) {
    const base = defaultState();
    if (!saved || typeof saved !== 'object') return base;
    const merged = Object.assign({}, base, saved);
    for (const k of ['profile', 'streak', 'stats', 'playground']) merged[k] = Object.assign({}, base[k], saved[k] || {});
    for (const k of ['concepts', 'problems', 'notes', 'flashcards', 'mistakes', 'achievements', 'daily']) if (!merged[k] || typeof merged[k] !== 'object' || Array.isArray(merged[k])) merged[k] = {};
    for (const k of ['activity', 'quizHistory']) if (!Array.isArray(merged[k])) merged[k] = [];
    if (!Array.isArray(merged.playground.files)) merged.playground.files = [];
    merged.app = APP_ID; merged.version = VERSION;
    return merged;
  }

  function load() {
    let saved = null;
    try { const raw = global.localStorage && localStorage.getItem(KEY); if (raw) saved = JSON.parse(raw); } catch (e) { saved = null; }
    return mergeDefaults(saved);
  }

  const reactive = (global.Vue && global.Vue.reactive) ? global.Vue.reactive : (x => x);
  const state = reactive(load());
  let dirty = false, saveTimer = null, lastError = null;
  const listeners = [];

  function saveNow() {
    dirty = false; clearTimeout(saveTimer);
    try { localStorage.setItem(KEY, JSON.stringify(state)); lastError = null; }
    catch (e) { lastError = e; /* storage full or unavailable (private mode): app keeps working in memory */ }
    listeners.forEach(fn => { try { fn(state); } catch (e) { /* ignore */ } });
  }
  function save() { dirty = true; clearTimeout(saveTimer); saveTimer = setTimeout(saveNow, 300); }
  if (global.Vue && global.Vue.watch) global.Vue.watch(() => state, save, { deep: true });
  if (global.addEventListener) {
    global.addEventListener('beforeunload', () => { if (dirty) saveNow(); });
    global.addEventListener('visibilitychange', () => { if (global.document && document.visibilityState === 'hidden' && dirty) saveNow(); });
  }
  function storageInfo() {
    let bytes = 0; try { bytes = (localStorage.getItem(KEY) || '').length; } catch (e) { /* ignore */ }
    return { bytes, error: lastError ? String(lastError.message || lastError) : null };
  }

  // ---------- export / import ----------
  function exportObject() { return JSON.parse(JSON.stringify(Object.assign({}, state, { exportedAt: new Date().toISOString() }))); }
  function exportJSON() { return JSON.stringify(exportObject(), null, 1); }

  function validate(data) {
    if (!data || typeof data !== 'object') throw new Error('The file is not valid JSON progress data.');
    if (data.app && data.app !== APP_ID) throw new Error('This file was made by a different app (' + data.app + ').');
    if (typeof data.xp !== 'number' || !data.concepts || !data.problems) throw new Error('This does not look like an ML Atlas progress file.');
    return data;
  }
  const newer = (a, b) => (a || '') >= (b || '') ;
  const maxDate = (a, b) => (!a ? b : !b ? a : (a > b ? a : b));
  const minDate = (a, b) => (!a ? b : !b ? a : (a < b ? a : b));

  // Merge two progress objects: union of achievements, best scores, newest drafts. Never loses work.
  function mergeStates(cur, inc) {
    const out = mergeDefaults(JSON.parse(JSON.stringify(cur)));
    inc = mergeDefaults(inc);
    out.xp = Math.max(out.xp || 0, inc.xp || 0);
    out.profile.createdAt = minDate(out.profile.createdAt, inc.profile.createdAt);
    if (!out.profile.goal && inc.profile.goal) out.profile.goal = inc.profile.goal;
    out.profile.onboarded = out.profile.onboarded || inc.profile.onboarded;
    // streak
    out.streak.best = Math.max(out.streak.best || 0, inc.streak.best || 0);
    out.streak.days = Object.assign({}, inc.streak.days || {}, out.streak.days || {});
    if (newer(inc.streak.lastActive, out.streak.lastActive) && inc.streak.lastActive !== out.streak.lastActive) { out.streak.lastActive = inc.streak.lastActive; out.streak.current = inc.streak.current; }
    else if (inc.streak.lastActive === out.streak.lastActive) out.streak.current = Math.max(out.streak.current || 0, inc.streak.current || 0);
    // concepts
    for (const id of Object.keys(inc.concepts)) {
      const a = out.concepts[id], b = inc.concepts[id];
      if (!a) { out.concepts[id] = b; continue; }
      a.read = Array.from(new Set([].concat(a.read || [], b.read || [])));
      a.checks = Object.assign({}, b.checks || {}, a.checks || {});
      a.quizBest = Math.max(a.quizBest || 0, b.quizBest || 0);
      a.quizAttempts = Math.max(a.quizAttempts || 0, b.quizAttempts || 0);
      a.quizPassed = !!(a.quizPassed || b.quizPassed);
      a.completedAt = minDate(a.completedAt, b.completedAt);
      a.masteredAt = minDate(a.masteredAt, b.masteredAt);
      a.visitedAt = maxDate(a.visitedAt, b.visitedAt);
    }
    // problems (code!)
    for (const id of Object.keys(inc.problems)) {
      const a = out.problems[id], b = inc.problems[id];
      if (!a) { out.problems[id] = b; continue; }
      if (b.draft != null && (a.draft == null || newer(b.draftAt, a.draftAt))) { a.draft = b.draft; a.draftAt = b.draftAt; }
      a.solved = !!(a.solved || b.solved);
      a.solvedAt = minDate(a.solvedAt, b.solvedAt);
      a.attempts = Math.max(a.attempts || 0, b.attempts || 0);
      a.runs = Math.max(a.runs || 0, b.runs || 0);
      a.hintsShown = Math.max(a.hintsShown || 0, b.hintsShown || 0);
      a.solutionViewed = !!(a.solutionViewed || b.solutionViewed);
      a.assisted = a.solved && b.solved ? !!(a.assisted && b.assisted) : !!(a.solved ? a.assisted : b.assisted);
      if (b.accepted && (!a.accepted || newer(b.accepted.at, a.accepted.at))) a.accepted = b.accepted;
    }
    // notes: keep both if different
    for (const id of Object.keys(inc.notes)) {
      const a = out.notes[id] || '', b = inc.notes[id] || '';
      if (!a) out.notes[id] = b; else if (b && a !== b && !a.includes(b)) out.notes[id] = a + '\n\n--- merged from another device ---\n' + b;
    }
    // playground files: union by id, newest wins
    const files = {};
    for (const f of out.playground.files.concat(inc.playground.files)) { if (!f || !f.id) continue; if (!files[f.id] || newer(f.updatedAt, files[f.id].updatedAt)) files[f.id] = f; }
    out.playground.files = Object.values(files).sort((x, y) => (y.updatedAt || '').localeCompare(x.updatedAt || ''));
    // flashcards: keep the card with more reviews
    for (const id of Object.keys(inc.flashcards)) { const a = out.flashcards[id], b = inc.flashcards[id]; if (!a || (b.reps || 0) + (b.lapses || 0) > (a.reps || 0) + (a.lapses || 0)) out.flashcards[id] = b; }
    // mistakes to revisit: keep the copy that has been reviewed more (a retired item stays retired)
    const reviews = x => (x && x.card ? (x.card.reps || 0) + (x.card.lapses || 0) : 0) + (x && x.graduated ? 1000 : 0);
    for (const id of Object.keys(inc.mistakes)) { const a = out.mistakes[id], b = inc.mistakes[id]; if (!a || reviews(b) > reviews(a)) out.mistakes[id] = b; }
    for (const id of Object.keys(inc.achievements)) out.achievements[id] = minDate(out.achievements[id], inc.achievements[id]);
    for (const k of Object.keys(inc.daily)) { const a = out.daily[k], b = inc.daily[k]; if (!a || (b && b.done && !a.done)) out.daily[k] = b; }
    for (const k of Object.keys(inc.stats)) if (typeof inc.stats[k] === 'number') out.stats[k] = Math.max(out.stats[k] || 0, inc.stats[k]);
    const seen = new Set(); out.activity = out.activity.concat(inc.activity).filter(a => { const key = a.t + a.label; if (seen.has(key)) return false; seen.add(key); return true; }).sort((x, y) => (y.t || '').localeCompare(x.t || '')).slice(0, 80);
    const seenQ = new Set(); out.quizHistory = out.quizHistory.concat(inc.quizHistory).filter(q => { const key = q.at + q.conceptId; if (seenQ.has(key)) return false; seenQ.add(key); return true; }).sort((x, y) => (y.at || '').localeCompare(x.at || '')).slice(0, 150);
    return out;
  }

  function replaceState(next) {
    Object.keys(state).forEach(k => { delete state[k]; });
    Object.assign(state, next);
    saveNow();
  }
  const MAX_IMPORT_CHARS = 40 * 1024 * 1024;   // far above any real progress file (localStorage itself holds about 5 MB)
  // mode: 'merge' (default, safe) or 'replace'
  function importJSON(text, mode) {
    let data;
    if (String(text).length > MAX_IMPORT_CHARS) throw new Error('That file is too large to be an ML Atlas progress file.');
    // keys that could tamper with object prototypes are dropped while parsing (imports can come from anywhere)
    try { data = JSON.parse(text, (k, v) => (k === '__proto__' || k === 'constructor' || k === 'prototype') ? undefined : v); } catch (e) { throw new Error('Could not read the file: it is not valid JSON.'); }
    validate(data);
    delete data.exportedAt;
    if (MLA.progress && MLA.progress.migrateSections) MLA.progress.migrateSections(data);
    const next = mode === 'replace' ? mergeDefaults(data) : mergeStates(state, data);
    next.stats.imports = (next.stats.imports || 0) + 1;
    replaceState(next);
    return next;
  }
  async function exportSyncCode() { return MLA.util.packText(JSON.stringify(exportObject())); }
  async function importSyncCode(code, mode) { const text = await MLA.util.unpackText(code); return importJSON(text, mode); }

  function reset() { replaceState(defaultState()); }
  function onSave(fn) { listeners.push(fn); }

  MLA.store = { state, save, saveNow, exportJSON, exportObject, importJSON, exportSyncCode, importSyncCode, mergeStates, mergeDefaults, reset, onSave, defaultState, storageInfo, KEY };
})(typeof globalThis !== 'undefined' ? globalThis : window);
