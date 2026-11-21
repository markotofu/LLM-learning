/* Read aloud: every lesson turns into speakable text (no markdown, no raw math symbols), formulas are read in words,
 * every chart has a description, and the speech engine works with a stand-in for the browser's speech synthesis.
 * Run: node tests/narrate.test.js */
'use strict';
const { load, suite } = require('./_load.js');
const MLA = load();
MLA.content.init();
const T = suite('narrate');
const N = MLA.narrate;

// 1. Formulas and math in prose, read as words
const has = (input, fn, ...want) => { const out = fn(input); for (const w of want) T.ok('reads "' + input + '" with "' + w + '"', out.includes(w), out); };
has('Regret(T) = Σ_{t=1..T} ( μ* − μ_{a_t} )', N.math, 'the sum over t equals 1 to T of', 'mu star minus mu sub a sub t');
has('L(θ) = (1/n) Σᵢ loss( f(xᵢ; θ), yᵢ )', N.math, 'theta', 'the sum over i of', 'x sub i');
has('z = ( p̂_B − p̂_A ) / √( p̂ (1 − p̂) )', N.math, 'p hat sub B minus p hat sub A', 'the square root of');
has('∂pᵢ/∂zⱼ = pᵢ (δᵢⱼ − pⱼ)', N.math, 'the partial derivative of p sub i with respect to z sub j');
has('w ← w + η (y − ŷ) x', N.math, 'w becomes w plus eta', 'y hat');
has('μ* = K*ᵀ ( K + σ_n² I )⁻¹ y', N.math, 'K star transpose', 'sigma sub n squared', 'inverse');
has('log L(θ) = Σ_i log p(x_i | θ)', N.math, 'given theta');
has('ᾱ_t', N.math, 'alpha bar sub t');
has('(n × k) @ (k × m)', N.math, 'n by k', 'matrix-multiplied by');
has('V̂(π) = (1 / n) Σ_i r_i · 1[π(x_i) = a_i] / p_i', N.math, 'the indicator that');
has('KL(N(μ, σ²) ‖ N(0, 1))', N.math, 'KL');
has('The gradient ∂L/∂W has the shape of W, e.g. 3 × 5.', N.text, 'gradient of L with respect to W', 'for example', '3 by 5');
has('Use `n_estimators` and `cross_val_score`; `**` is the power operator.', N.text, 'n estimators', 'cross val score', 'to the power');
has('About ~10M parameters and x² + y³.', N.text, 'About 10 million', 'x squared', 'y cubed');

// 2. Every lesson narrates cleanly
const LEFT = /\*\*|`|\$\$|:::|\|\s*-{2,}|<\/?[a-z]+[\s>]|[Σ∑√∂∇‖⌈⌉⌊⌋ᵀ²³⁻ⁿ₀₁₂₃ᵢⱼₙ←→≈≤≥×·−θσμλαβγδεηπφωτρκ_{}]/;
let items = 0, chunks = 0, longest = 0;
for (const c of MLA.data.concepts) {
  const list = N.lesson(c);
  T.ok(c.id + ': narration has the intro, every section and the end', list[0].key === 'intro' && list[list.length - 1].key === 'end' && c.sections.every(s => list.some(it => it.key === s.id)));
  // "undefined" and "NaN" are real words in some lessons ("the moving average is undefined", pandas NaN), so compare counts
  const count = t => (t.match(/undefined/g) || []).length;
  const inSource = count(JSON.stringify([c.title, c.summary, c.why, c.objectives, c.sections, c.when, c.pitfalls, c.takeaways]));
  let inNarration = 0;
  for (const it of list) {
    items++;
    const text = it.text.split(N.PAUSE).join(' ');
    const m = text.match(LEFT);
    T.ok(c.id + ' [' + it.key + ']: no markdown or raw symbols left', !m, m && '…' + text.slice(Math.max(0, m.index - 60), m.index + 40) + '…');
    inNarration += count(text);
    for (const ch of N.chunks(it.text)) { chunks++; if (ch !== N.PAUSE) longest = Math.max(longest, ch.length); }
  }
  T.ok(c.id + ': no "undefined" leaks into the narration', inNarration <= inSource, inNarration + ' in the narration, ' + inSource + ' in the lesson');
  for (const s of c.sections) {
    if (s.type === 'check') T.ok(c.id + ': quick checks pause before the answer', N.section(c, s, 0).text.includes(N.PAUSE));
    if (s.type === 'chart') {
      const d = N.chart(s.chart);
      T.ok(c.id + ' | ' + s.title + ': chart described', d.length > 40 && !/NaN|undefined|Infinity/.test(d), d);
    }
  }
}
T.ok('speech chunks stay short (browsers cut off long utterances)', longest <= 260, String(longest));
T.ok('narration covers the lessons', items > 2000 && chunks > items, items + ' items, ' + chunks + ' chunks');

// 3. Chart descriptions: trends, peaks, crossings and reference lines
const d = N.chart({ kind: 'line', x: [1, 2, 3, 4, 5, 6, 7], xLabel: 'k (neighbours)', yLabel: 'accuracy',
  series: [{ name: 'train', y: [1, 0.98, 0.96, 0.95, 0.94, 0.93, 0.92] }, { name: 'test', y: [0.90, 0.93, 0.95, 0.97, 0.95, 0.93, 0.91] }], vlines: [{ x: 4, label: 'best k' }] });
T.ok('a falling line is described as falling', /train falls from 1 to 0\.92/.test(d), d);
T.ok('an interior maximum is described as a peak', /test rises from 0\.9 to a peak of 0\.97 at k 4/.test(d), d);
T.ok('the reference line is mentioned', /reference line marks best k/.test(d), d);
const bar = N.chart({ kind: 'bar', x: ['a', 'b', 'c'], yLabel: 'count', series: [{ name: 'n', y: [3, 5, 1] }] });
T.ok('bar values are listed', /a, 3; b, 5; c, 1/.test(bar), bar);

// 4. The speech engine, with a stand-in for speechSynthesis (fires "end" right away)
{
  const spoken = [];
  const fakeVoice = { voiceURI: 'local-en', name: 'Test voice', lang: 'en-US', localService: true, default: true };
  const online = { voiceURI: 'online-en', name: 'Cloud voice', lang: 'en-US', localService: false, default: false };
  global.SpeechSynthesisUtterance = function (text) { this.text = text; };
  global.speechSynthesis = {
    getVoices: () => [online, fakeVoice], addEventListener() {}, cancel() {}, resume() {},
    speak(u) { spoken.push(u); setTimeout(() => u.onend && u.onend(), 1); }
  };
  delete require.cache[require.resolve('../js/core/speech.js')];
  global.window = global;
  require('../js/core/speech.js');
  const S = global.MLA.speech;
  T.ok('only on-device voices are offered', S.state.voices.length === 1 && S.state.voices[0].uri === 'local-en');
  const c = MLA.content.byId['linear-regression'];
  const list = N.lesson(c);
  S.play(list, 2, { single: true, source: c.id, lessonTitle: c.title });
  T.ok('playing one part starts at that part', S.state.playing && S.state.key === list[2].key);
  setTimeout(() => {
    T.ok('one part plays to its end and stops', !S.state.playing && spoken.length >= 1, spoken.length + ' utterances');
    T.ok('every utterance uses the on-device voice', spoken.every(u => u.voice === fakeVoice));
    T.ok('every utterance is a short chunk', spoken.every(u => u.text.length <= 260));
    S.play(list, 0, { source: c.id });
    S.next(); T.ok('next moves to the following part', S.state.index === 1);
    S.pause(); T.ok('pause holds the position', S.state.paused && S.state.index === 1);
    S.resume(); S.stop(); T.ok('stop clears the player', !S.state.playing && S.state.key === null);
    T.done();
  }, 400);
}
