/* ML Atlas – read aloud. Plays narration (js/core/narrate.js) with the browser's built-in speech synthesis
 * (the Web Speech API). Only on-device voices are used (SpeechSynthesisVoice.localService): no lesson text is sent
 * to an online or AI voice service, and it works on a static site with no server.
 * Text is spoken a sentence or two at a time. Short utterances avoid browsers cutting long ones off, and they make
 * pause, skip and the reading position precise. Pausing cancels the current sentence; resuming repeats it.
 *   MLA.speech.play(items, start, { single, source, lessonTitle })   items: [{ key, title, text }]
 *   MLA.speech.toggle() / pause() / resume() / stop() / next() / prev() / setRate(r) / sample()
 *   MLA.speech.state  (reactive) { supported, voices, playing, paused, index, total, key, title, source, lessonTitle } */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  const synth = global.speechSynthesis || null;
  const Utter = global.SpeechSynthesisUtterance || null;
  const reactive = global.Vue && global.Vue.reactive ? global.Vue.reactive : (x => x);
  const state = reactive({ supported: !!(synth && Utter), voices: [], playing: false, paused: false, index: -1, total: 0, key: null, title: '', source: null, lessonTitle: '', single: false });

  let items = [], lists = [], chunk = 0, token = 0, gapTimer = null, watchdog = null, current = null;
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

  function clearTimers() { clearTimeout(gapTimer); clearTimeout(watchdog); }
  function setCurrent() { const it = items[state.index]; state.key = it ? it.key : null; state.title = it ? it.title : ''; }
  function silence() { token++; clearTimers(); if (synth) synth.cancel(); }

  function speakNext() {
    clearTimers();
    if (!state.playing || state.paused) return;
    const list = lists[state.index];
    if (!list) return stop();
    if (chunk >= list.length) {                                    // this part is done
      if (state.single || state.index + 1 >= items.length) return stop();
      state.index++; chunk = 0; setCurrent();
      gapTimer = setTimeout(speakNext, 500);                     // a breath between parts
      return;
    }
    const text = list[chunk];
    if (text === MLA.narrate.PAUSE) { chunk++; gapTimer = setTimeout(speakNext, 2000); return; }   // time to think before an answer
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
    // If a browser never reports the end of a sentence, move on after a generous estimate of its length.
    watchdog = setTimeout(() => { if (my === token) { token++; synth.cancel(); chunk++; speakNext(); } }, 6000 + text.length * 110 / rate());
    synth.speak(u);
  }

  function play(list, start, opts) {
    if (!state.supported || !list || !list.length) return;
    silence();
    opts = opts || {};
    items = list;
    lists = list.map(it => MLA.narrate.chunks(it.text));
    state.index = Math.min(Math.max(0, start || 0), list.length - 1); chunk = 0;
    state.total = list.length; state.single = !!opts.single;
    state.source = opts.source || null; state.lessonTitle = opts.lessonTitle || '';
    state.playing = true; state.paused = false;
    setCurrent();
    try { synth.resume(); } catch (e) { /* not paused */ }
    speakNext();
  }
  function pause() { if (state.playing && !state.paused) { silence(); state.paused = true; } }
  function resume() { if (state.playing && state.paused) { state.paused = false; speakNext(); } }
  function toggle() { if (state.paused) resume(); else pause(); }
  function stop() {
    silence(); current = null;
    state.playing = false; state.paused = false; state.index = -1; state.key = null; state.title = ''; state.single = false;
    items = []; lists = [];
  }
  function jump(i) {
    if (!state.playing || i < 0 || i >= items.length) return;
    silence(); state.index = i; chunk = 0; state.single = false; state.paused = false; setCurrent(); speakNext();
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

  MLA.speech = { state, play, pause, resume, toggle, stop, next, prev, jump, setRate, sample, loadVoices };
})(typeof globalThis !== 'undefined' ? globalThis : window);
