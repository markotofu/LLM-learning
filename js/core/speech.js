/* ML Atlas – read aloud. Plays narration (js/core/narrate.js) with the browser's built-in speech synthesis
 * (the Web Speech API). Only on-device voices are used (SpeechSynthesisVoice.localService): no lesson text is sent
 * to an online or AI voice service, and it works on a static site with no server.
 * Text is spoken a sentence or two at a time. Short utterances avoid browsers cutting long ones off, and they make
 * pause, skip and the reading position precise. Pausing cancels the current sentence; resuming repeats it.
 * While reading, a silent looping clip keeps a media session open, so lock-screen controls and headphone buttons
 * (play, pause, next, previous) reach the player in browsers that support the Media Session API.
 * The part a lesson was left at is remembered (profile.listenAt, by part key), and when a lesson ends the next one
 * in the learner's track is offered (profile.speechContinue plays it automatically).
 *   MLA.speech.play(items, start, { single, source, lessonTitle, gap })   items: [{ key, title, text }]
 *                                            gap: milliseconds of silence between parts (default 500)
 *   MLA.speech.toggle() / pause() / resume() / stop() / next() / prev() / jump(i) / setRate(r) / sample()
 *   MLA.speech.playNext() / dismiss()       act on the "up next" offer after a lesson ends
 *   MLA.speech.resumeKey(lessonId)          the part key a lesson was left at, or null
 *   MLA.speech.state  (reactive) { supported, voices, playing, paused, index, total, key, title, source, lessonTitle,
 *                                  single, caption, upNext: { id, title } | null } */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  const synth = global.speechSynthesis || null;
  const Utter = global.SpeechSynthesisUtterance || null;
  const media = (global.navigator && global.navigator.mediaSession) || null;
  const reactive = global.Vue && global.Vue.reactive ? global.Vue.reactive : (x => x);
  const state = reactive({ supported: !!(synth && Utter), voices: [], playing: false, paused: false, index: -1, total: 0, key: null, title: '',
    source: null, lessonTitle: '', single: false, caption: '', upNext: null });

  let items = [], lists = [], chunk = 0, token = 0, gapTimer = null, watchdog = null, nextTimer = null, current = null, keep = null, gap = 500;
  const profile = () => (MLA.store && MLA.store.state && MLA.store.state.profile) || {};
  const rate = () => { const r = Number(profile().speechRate); return r >= 0.5 && r <= 2 ? r : 1; };
  const english = lang => /^en/i.test(lang || '');

  function localVoices() { return synth ? (synth.getVoices() || []).filter(v => v.localService) : []; }
  function loadVoices() {
    state.voices = localVoices().map(v => ({ uri: v.voiceURI, name: v.name, lang: v.lang }))
      .sort((a, b) => (english(b.lang) - english(a.lang)) || a.lang.localeCompare(b.lang) || a.name.localeCompare(b.name));
  }
  function voice() {
    const all = localVoices(), want = profile().voice;
    return all.find(v => v.voiceURI === want) || all.find(v => v.default && english(v.lang)) || all.find(v => /^en[-_]US/i.test(v.lang)) ||
      all.find(v => english(v.lang)) || all[0] || null;
  }
  if (synth) {
    loadVoices();
    try { synth.addEventListener('voiceschanged', loadVoices); } catch (e) { synth.onvoiceschanged = loadVoices; }
  }

  // ---- media session: lock screen, notification and headphone buttons
  function keepAlive(on) {        // browsers route media keys only to pages that are playing media, hence the silent clip
    if (!global.Audio) return;
    try {
      if (on) {
        if (!keep) { keep = new global.Audio('assets/silence.wav'); keep.loop = true; }
        const p = keep.play();
        if (p && p.catch) p.catch(() => { /* autoplay refused: speech still works, only the media keys do not */ });
      } else if (keep) keep.pause();
    } catch (e) { /* no media support */ }
  }
  function session() {
    if (!media) return;
    try {
      media.metadata = state.playing && global.MediaMetadata ? new global.MediaMetadata({ title: state.title || 'ML Atlas', artist: state.lessonTitle || 'ML Atlas', album: 'ML Atlas',
        artwork: [{ src: 'icons/icon-192.png', sizes: '192x192', type: 'image/png' }, { src: 'icons/icon-512.png', sizes: '512x512', type: 'image/png' }] }) : null;
      media.playbackState = !state.playing ? 'none' : state.paused ? 'paused' : 'playing';
    } catch (e) { /* not supported */ }
  }
  if (media) {
    for (const [action, fn] of [['play', () => resume()], ['pause', () => pause()], ['stop', () => stop()], ['nexttrack', () => next()], ['previoustrack', () => prev()]]) {
      try { media.setActionHandler(action, fn); } catch (e) { /* this action is not supported */ }
    }
  }

  // ---- where each lesson was left, by part key (keys survive edits to the lesson; positions would not)
  function remember() {
    if (state.single || !state.source || !state.key) return;
    const p = profile();
    if (!p.listenAt || p.listenAt[state.source] !== state.key) p.listenAt = Object.assign({}, p.listenAt, { [state.source]: state.key });
  }
  function forget(id) {
    const p = profile();
    if (p.listenAt && p.listenAt[id]) { const o = Object.assign({}, p.listenAt); delete o[id]; p.listenAt = o; }
  }
  function resumeKey(id) { const at = (profile().listenAt || {})[id]; return at && at !== 'intro' ? at : null; }

  // ---- the next lesson: the following one in the learner's track; without a track, a lesson that builds directly on
  // this one (unfinished first, in the authored order), else the next lesson in the catalogue
  function upNext(id) {
    const P = MLA.progress, byId = MLA.content && MLA.content.byId;
    if (!P || !id || !byId || !byId[id]) return null;
    const goal = profile().goal ? P.track(profile().goal) : null;
    const order = goal ? P.trackConcepts(goal) : [];
    const i = order.indexOf(id);
    if (i >= 0) return i + 1 < order.length ? byId[order[i + 1]] : null;
    const kids = ((MLA.content.graph && MLA.content.graph.children[id]) || []).map(k => byId[k]).filter(Boolean).sort((a, b) => a._order - b._order);
    const pick = kids.find(k => !P.isCompleted(k.id)) || kids[0];
    if (pick) return pick;
    const all = MLA.data.concepts, j = all.indexOf(byId[id]);
    return j >= 0 && j + 1 < all.length ? all[j + 1] : null;
  }

  function clearTimers() { clearTimeout(gapTimer); clearTimeout(watchdog); }
  function setCurrent() { const it = items[state.index]; state.key = it ? it.key : null; state.title = it ? it.title : ''; remember(); session(); }
  function silence() { token++; clearTimers(); if (synth) synth.cancel(); }

  function speakNext() {
    clearTimers();
    if (!state.playing || state.paused) return;
    const list = lists[state.index];
    if (!list) return stop();
    if (chunk >= list.length) {                                    // this part is done
      if (state.single) return stop();
      if (state.index + 1 >= items.length) return finish();
      state.index++; chunk = 0; setCurrent();
      gapTimer = setTimeout(speakNext, gap);                     // a breath between parts
      return;
    }
    const text = list[chunk];
    if (text === MLA.narrate.PAUSE) { state.caption = ''; chunk++; gapTimer = setTimeout(speakNext, 2000); return; }   // time to think before an answer
    const v = voice();
    if (!v) {
      stop();
      if (MLA.ui) MLA.ui.toast('This browser has no voice installed on the device, so lessons cannot be read aloud here.', 'error', 6000);
      return;
    }
    const u = new Utter(text);
    u.voice = v; u.lang = v.lang; u.rate = rate(); u.pitch = 1;
    const my = ++token;
    const advance = () => { if (my !== token) return; chunk++; speakNext(); };
    u.onend = advance;
    u.onerror = e => { if (my !== token) return; if (e && (e.error === 'interrupted' || e.error === 'canceled')) return; advance(); };
    current = u;                                                   // keep a reference: some browsers drop events of collected utterances
    state.caption = text;
    // If a browser never reports the end of a sentence, move on after a generous estimate of its length.
    watchdog = setTimeout(() => { if (my === token) { token++; synth.cancel(); chunk++; speakNext(); } }, 6000 + text.length * 110 / rate());
    synth.speak(u);
  }

  function play(list, start, opts) {
    if (!state.supported || !list || !list.length) return;
    silence(); dismiss();
    opts = opts || {};
    items = list;
    lists = list.map(it => MLA.narrate.chunks(it.text));
    state.index = Math.min(Math.max(0, start || 0), list.length - 1); chunk = 0;
    state.total = list.length; state.single = !!opts.single;
    state.source = opts.source || null; state.lessonTitle = opts.lessonTitle || '';
    gap = opts.gap > 0 ? opts.gap : 500;
    state.playing = true; state.paused = false;
    setCurrent();
    keepAlive(true);
    try { synth.resume(); } catch (e) { /* not paused */ }
    speakNext();
  }
  function pause() { if (state.playing && !state.paused) { silence(); state.paused = true; keepAlive(false); session(); } }
  function resume() { if (state.playing && state.paused) { state.paused = false; keepAlive(true); session(); speakNext(); } }
  function toggle() { if (state.paused) resume(); else pause(); }
  function stop() {
    silence(); current = null; keepAlive(false);
    state.playing = false; state.paused = false; state.index = -1; state.key = null; state.title = ''; state.single = false; state.caption = '';
    items = []; lists = [];
    session();
  }
  function finish() {                                              // the whole lesson has been read
    const id = state.source, nx = upNext(id);
    stop();
    if (id) forget(id);
    if (!nx) return;
    state.upNext = { id: nx.id, title: nx.title };
    if (profile().speechContinue) nextTimer = setTimeout(playNext, 4000);
  }
  function playNext() {
    clearTimeout(nextTimer);
    const nx = state.upNext;
    state.upNext = null;
    if (!nx) return;
    const go = () => {
      const c = MLA.content.byId[nx.id];
      if (!c) return;
      if (MLA.router) MLA.router.navigate('/concept/' + c.id);
      play(MLA.narrate.lesson(c), 0, { source: c.id, lessonTitle: c.title });
    };
    if (MLA.content.ensure) MLA.content.ensure(nx.id).then(go); else go();
  }
  function dismiss() { clearTimeout(nextTimer); state.upNext = null; }
  function jump(i) {
    if (!state.playing || i < 0 || i >= items.length) return;
    silence(); state.index = i; chunk = 0; state.single = false; state.paused = false; setCurrent(); keepAlive(true); speakNext();
  }
  function next() { if (state.index + 1 >= items.length) stop(); else jump(state.index + 1); }
  function prev() { jump(chunk > 0 ? state.index : Math.max(0, state.index - 1)); }    // restart this part, or go back one
  function setRate(r) {
    profile().speechRate = r;
    if (state.playing && !state.paused) { silence(); speakNext(); }                       // repeat the sentence at the new speed
  }
  function sample() {
    play([{ key: 'sample', title: 'Voice sample', text: 'This is how lessons will sound when they are read aloud. You can change the voice and the speed at any time.' }], 0, { single: true });
  }

  MLA.speech = { state, play, pause, resume, toggle, stop, next, prev, jump, setRate, sample, loadVoices, playNext, dismiss, resumeKey, upNext };
})(typeof globalThis !== 'undefined' ? globalThis : window);
