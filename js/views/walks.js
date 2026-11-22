/* ML Atlas – step-through walkthroughs of algorithms. Each one is an animated flowchart of the algorithm, kept in sync
 * with its code (the running line is highlighted), the formula with the numbers of the current step, the values of the
 * variables, and a small plot of what the model "sees". Every arrow and box carries its reasoning: what the step is
 * for, and why the algorithm goes there next. Some steps ask the reader to predict what happens before it is shown.
 * Numbers are never written by hand: each walkthrough runs its algorithm in the browser on a small fixed dataset.
 * Lessons embed one as a demo section: { type: 'demo', demo: 'walk-<name>' }.
 *   MLA.walk(name, def)
 *     def.title, def.code: [lines], def.params: [{ key, label, options: [[value, text]], value }]
 *     def.nodes: [{ id, label: [lines], why, kind: 'step' | 'dec' | 'end' }]   drawn top to bottom in this order
 *     def.edges: [{ from, to, label, why }]   to the next node: straight down; any other: a lane on the right
 *     def.run(params) -> [{ node, lines: [code line numbers], say, formula?, vars?: [[name, value]], ask?, state }]
 *     def.symbols: [[symbol, meaning]]   each step shows the ones found in its box label, formula or variable names
 *     a step may carry `voice`: the spoken version of `say`, when `say` uses notation that reads badly aloud
 *   Play moves at a chosen speed (Super slow to Fast) or "With the voice": each step is read aloud (MLA.speech) and
 *   the card follows the voice. MLA.walkListen(name) does the same for a lesson's "Listen to this part" button.
 *     def.draw(state, kit) -> shapes for the plot: { k: key, t: 'c' | 'l' | 'p' | 'r' | 't', ...coordinates, cls, f? }
 *       (the plot is 300 by 210; shapes are clipped to the box of kit.frame unless f: 1)
 *   MLA.walks[name].describe() -> the flowchart in words (read aloud and used as the text alternative) */
(function (global) {
  const MLA = global.MLA;
  const C = MLA.components;
  MLA.walks = {};

  // ---------------------------------------------------------------- flowchart layout
  const L = { x: 14, w: 222, gap: 28, top: 10, line: 15, pad: 12, lane0: 252, laneGap: 13 };
  function layout(def) {
    if (def._lay) return def._lay;
    const pos = {};
    let y = L.top;
    def.nodes.forEach((n, i) => {
      const h = L.pad + L.line * n.label.length;
      pos[n.id] = { x: L.x, y, w: L.w, h, i, cx: L.x + L.w / 2, my: y + h / 2 };
      y += h + L.gap;
    });
    let lanes = 0;
    const edges = def.edges.map(e => {
      const a = pos[e.from], b = pos[e.to], id = e.from + '>' + e.to;
      if (!a || !b) throw new Error('walkthrough ' + def.name + ': edge ' + id + ' names a missing node');
      if (b.i === a.i + 1) return Object.assign({ id, lane: false, d: 'M' + a.cx + ' ' + (a.y + a.h) + ' V' + (b.y - 3), lx: a.cx + 8, ly: (a.y + a.h + b.y) / 2 + 4 }, e);
      const x = L.lane0 + (lanes++) * L.laneGap;
      return Object.assign({ id, lane: true, d: 'M' + (a.x + a.w) + ' ' + (a.my - 4) + ' H' + x + ' V' + (b.my + 4) + ' H' + (b.x + b.w + 3), ly: a.my }, e);
    });
    const right = L.lane0 + Math.max(0, lanes - 1) * L.laneGap;
    edges.forEach(e => { if (e.lane) { e.lx = right + 9; e.ly = e.ly + 0; } });
    def._lay = { pos, edges, h: y - L.gap + L.top, w: Math.max(L.x + L.w + 14, right + 104) };
    return def._lay;
  }

  // ---------------------------------------------------------------- plot kit
  const P = { w: 300, h: 210, l: 38, r: 10, t: 12, b: 28 };
  const num = v => {
    if (!isFinite(v)) return String(v);
    const a = Math.abs(v);
    return a >= 1000 ? v.toFixed(0) : a >= 100 ? v.toFixed(1) : a >= 1 || a === 0 ? +v.toFixed(2) + '' : v.toPrecision(2);
  };
  function ticks(d0, d1, n) {
    const raw = (d1 - d0) / (n || 4), mag = Math.pow(10, Math.floor(Math.log10(raw))), step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(d0 / step) * step; v <= d1 + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  // Axes for a data box: returns scale functions and the frame shapes (drawn unclipped)
  function frame(xd, yd, labels) {
    const sx = v => P.l + (v - xd[0]) / (xd[1] - xd[0]) * (P.w - P.l - P.r);
    const sy = v => P.h - P.b - (v - yd[0]) / (yd[1] - yd[0]) * (P.h - P.t - P.b);
    const s = [{ k: 'ax', t: 'l', x1: P.l, y1: P.h - P.b, x2: P.w - P.r, y2: P.h - P.b, cls: 'wp-axis', f: 1 },
      { k: 'ay', t: 'l', x1: P.l, y1: P.t, x2: P.l, y2: P.h - P.b, cls: 'wp-axis', f: 1 }];
    ticks(xd[0], xd[1]).forEach(v => s.push({ k: 'tx' + v, t: 't', x: sx(v), y: P.h - P.b + 13, s: num(v), cls: 'wp-tick', a: 'middle', f: 1 }));
    ticks(yd[0], yd[1]).forEach(v => s.push({ k: 'ty' + v, t: 't', x: P.l - 5, y: sy(v) + 3.5, s: num(v), cls: 'wp-tick', a: 'end', f: 1 }));
    if (labels && labels[0]) s.push({ k: 'xl', t: 't', x: P.w - P.r, y: P.h - 2, s: labels[0], cls: 'wp-tick', a: 'end', f: 1 });
    if (labels && labels[1]) s.push({ k: 'yl', t: 't', x: P.l + 4, y: P.t + 2, s: labels[1], cls: 'wp-tick', a: 'start', f: 1 });
    return { sx, sy, shapes: s };
  }
  const kit = { P, frame, num, ticks };
  MLA.walkKit = kit;

  function describe(def) {
    const lab = id => def.nodes.find(n => n.id === id).label.join(', ');
    const lay = layout(def);
    return 'A flowchart of ' + def.nodes.length + ' steps. ' +
      def.nodes.map((n, i) => 'Step ' + (i + 1) + ': ' + n.label.join(', ') + '. ' + n.why).join(' ') + ' ' +
      lay.edges.filter(e => e.lane).map(e => 'From "' + lab(e.from) + '", the arrow marked "' + e.label + '" goes to "' + lab(e.to) + '": ' + e.why).join(' ');
  }

  // ---------------------------------------------------------------- pace
  // Timed speeds (milliseconds per step, and seconds for each move on screen), then pacing by the read-aloud voice:
  // each step is spoken with the learner's on-device voice and speed, and the card moves on when the voice finishes.
  const PACES = [
    { key: 'super-slow', label: 'Super slow', ms: 9000, move: 1.6 },
    { key: 'sluggish', label: 'Sluggish', ms: 5000, move: 1.1 },
    { key: 'slow', label: 'Slow', ms: 2400, move: 0.7 },
    { key: 'normal', label: 'Normal', ms: 1400, move: 0.45 },
    { key: 'fast', label: 'Fast', ms: 700, move: 0.3 },
    { key: 'voice', label: 'With the voice', ms: 0, move: 0.8 }
  ];
  const paceOf = key => PACES.find(p => p.key === key) || PACES[3];
  const canSpeak = () => !!(MLA.speech && MLA.speech.state && MLA.speech.state.supported);

  // The arrow taken to reach step k: the one the step names, else the one from the previous step's box.
  function arrowOf(def, steps, k) {
    const s = steps[k], lay = layout(def);
    if (!s) return null;
    if (s.edge) return lay.edges.find(e => e.id === s.edge) || null;
    const prev = k > 0 ? steps[k - 1].node : null;
    return prev && prev !== s.node ? lay.edges.find(e => e.from === prev && e.to === s.node) || null : null;
  }

  // ---------------------------------------------------------------- narration
  // What the voice says at step k: the reason for a loop arrow the first time it is taken, the box and its purpose the
  // first time the box is reached, then the step (its `voice` text if it has one, else `say`). A "Think first"
  // question is followed by a few seconds of silence before the walkthrough moves on.
  function narrateStep(def, steps, k) {
    const N = MLA.narrate, s = steps[k], node = def.nodes.find(n => n.id === s.node);
    const say = t => {                                             // spaced comparisons and sums as words, then the usual reading
      t = String(t || '').replace(/ \+ /g, ' plus ').replace(/ < /g, ' is less than ').replace(/ > /g, ' is greater than ')
        .replace(/,\s*·\s*\)/g, ', any action)')                                   // Q(s′, ·): the best over any action
        .replace(/([A-Za-zͰ-Ͽ)̂])\/(?=[A-Za-z(])/g, '$1 over ');               // m/(1 − β₁ᵗ), but not 50/50
      t = N.text(t).trim();
      return !t || /[.!?:;]$/.test(t) ? t : t + '.';
    };
    // Box labels are written in lines: a pause before a line that is a formula, otherwise the phrase simply goes on.
    const label = n => n.label.reduce((a, l) => (!a ? l : a + (!/[:;,.]$/.test(a) && /[=←∂∇]/.test(l) ? ', ' : ' ') + l), '');
    const out = [], e = arrowOf(def, steps, k);
    if (e && e.lane && !steps.slice(0, k).some((x, j) => { const a = arrowOf(def, steps, j); return a && a.id === e.id; })) out.push(say(e.why));
    if (node && steps.findIndex(x => x.node === s.node) === k) out.push(say(label(node)), say(node.why));
    out.push(say(s.voice || s.say));
    if (s.ask) out.push('Think first. ' + say(s.ask) + ' ' + N.PAUSE + ' ' + N.PAUSE);
    return out.join(' ');
  }

  // Mounted cards by name, so a lesson's "Listen to this part" button can have a walkthrough read its steps.
  const cards = {};
  MLA.walkListen = name => { const c = cards[name]; if (!c) return false; c.listen(); return true; };

  // ---------------------------------------------------------------- symbol legends
  // A walkthrough lists its symbols once (def.symbols: [[symbol, meaning]]). Each step shows the ones that appear in
  // its box, formula or variables, so every formula on screen has its symbols defined right under it.
  const SUB = /[₀-ₜᵢ-ᵪⱼ]/g;               // subscript digits and letters, as in x₁ and wⱼ
  const isLetter = ch => !!ch && /[\p{L}\p{M}]/u.test(ch);
  function has(text, sym) {                    // a letter at either end of the symbol must not touch another letter
    const lead = isLetter(sym[0]), tail = isLetter(sym[sym.length - 1]);
    for (const t of [text, text.replace(SUB, '')]) {
      for (let k = t.indexOf(sym); k >= 0; k = t.indexOf(sym, k + 1))
        if (!(lead && isLetter(t[k - 1])) && !(tail && isLetter(t[k + sym.length]))) return true;
    }
    return false;
  }
  function symbolsFor(def, step) {
    const node = def.nodes.find(n => n.id === step.node);
    const text = [node ? node.label.join(' ') : '', step.formula || '', (step.vars || []).map(v => v[0]).join('  ')].join('  ');
    return (def.symbols || []).filter(s => has(text, s[0]));
  }

  MLA.walk = function (name, def) {
    def.name = name;
    def.describe = () => describe(def);
    def.symbolsFor = step => symbolsFor(def, step);
    def.narrateStep = (steps, k) => narrateStep(def, steps, k);
    MLA.walks[name] = def;
    C['demo-walk-' + name] = { template: '<mla-walk name="' + name + '" />' };
  };

  let uid = 0;
  C['mla-walk'] = {
    props: { name: String },
    data() {
      const def = MLA.walks[this.name];
      const params = {};
      (def.params || []).forEach(p => { params[p.key] = p.value; });
      const saved = MLA.store && MLA.store.state && MLA.store.state.profile && MLA.store.state.profile.walkSpeed;
      return { params, i: 0, playing: false, voiceOn: false, said: '', id: ++uid,
        speedKey: PACES.some(p => p.key === saved) && (saved !== 'voice' || canSpeak()) ? saved : 'normal',
        speech: MLA.speech ? MLA.speech.state : {},
        reduced: !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches) };
    },
    computed: {
      def() { return MLA.walks[this.name]; },
      lay() { return layout(this.def); },
      steps() { return this.def.run(Object.assign({}, this.params)); },
      step() { return this.steps[Math.min(this.i, this.steps.length - 1)] || {}; },
      node() { return this.def.nodes.find(n => n.id === this.step.node) || {}; },
      edge() { return arrowOf(this.def, this.steps, Math.min(this.i, this.steps.length - 1)); },   // the arrow taken to reach this step
      pace() { return paceOf(this.speedKey); },
      paces() { return PACES.filter(p => p.key !== 'voice' || canSpeak()); },
      prefix() { return 'walk' + this.id + ':'; },                 // keys of this card's parts in the read-aloud player
      seen() { const s = new Set(); for (let k = 0; k <= this.i && k < this.steps.length; k++) s.add(this.steps[k].node); return s; },
      shapes() {
        const out = this.def.draw ? this.def.draw(this.step.state || {}, kit) : [];
        // Frame shapes (f: 1) are drawn first and unclipped; data shapes on top, clipped to the plot box.
        return [{ k: 'frame', clip: false, shapes: out.filter(s => s.f) }, { k: 'data', clip: true, shapes: out.filter(s => !s.f) }];
      },
      describe() { return this.def.describe(); },
      where() { return this.def.symbolsFor(this.step); },
      lineY() { return (n, k) => this.lay.pos[n.id].y + L.pad / 2 + L.line * (k + 1) - 3; }
    },
    watch: {
      i() { this.$nextTick(this.animate); },
      params: { deep: true, handler() { this.stop(); this.i = 0; } },
      'speech.key'(key) {                          // the voice moved to another step: follow it
        if (!this.voiceOn || !key) return;
        if (key.indexOf(this.prefix) !== 0) { this.voiceOn = false; this.playing = false; return; }   // other narration took over
        const k = Number(key.slice(this.prefix.length));
        if (k >= 0 && k < this.steps.length && k !== this.i) this.i = k;
      },
      'speech.playing'(on) { if (!on && this.voiceOn) { this.voiceOn = false; this.playing = false; } }
    },
    mounted() { cards[this.name] = this; },
    beforeUnmount() { this.stop(); cancelAnimationFrame(this._raf); if (cards[this.name] === this) delete cards[this.name]; },
    methods: {
      go(k, manual) {
        this.i = Math.max(0, Math.min(this.steps.length - 1, k));
        if (!manual) return;
        if (this.voiceOn) MLA.speech.jump(this.i);                // keep listening, from the chosen step
        else { this.stop(); this.said = 'Step ' + (this.i + 1) + ': ' + (this.step.say || ''); }
      },
      next() { this.go(this.i + 1, true); },
      back() { this.go(this.i - 1, true); },
      restart() { this.go(0, true); },
      jump(id) { const k = this.steps.findIndex(s => s.node === id); if (k >= 0) this.go(k, true); },
      toggle() { if (this.playing) this.stop(); else this.play(); },
      play() {
        if (this.i >= this.steps.length - 1) this.i = 0;
        if (this.pace.key === 'voice' && !canSpeak()) this.speedKey = 'normal';
        this.playing = true;
        if (this.pace.key === 'voice') {                           // the read-aloud player sets the pace
          this.voiceOn = true;
          MLA.speech.play(this.voiceItems(), this.i, { lessonTitle: this.def.title, gap: 900 });
          return;
        }
        const tick = () => {
          if (!this.playing) return;
          if (this.i >= this.steps.length - 1) { this.stop(); return; }
          this.i++;
          this._timer = setTimeout(tick, this.pace.ms);
        };
        this._timer = setTimeout(tick, Math.min(600, this.pace.ms));
      },
      stop() {
        this.playing = false; clearTimeout(this._timer);
        if (this.voiceOn) { this.voiceOn = false; MLA.speech.stop(); }
      },
      listen() {                                   // the lesson's "Listen to this part": read every step, from the first
        this.stop();
        this.speedKey = 'voice';
        this.i = 0;
        this.play();
      },
      changeSpeed() {                              // remember the choice, and carry on at the new pace
        if (MLA.store && MLA.store.state && MLA.store.state.profile) MLA.store.state.profile.walkSpeed = this.speedKey;
        if (this.playing) { this.stop(); this.play(); }
      },
      voiceItems() {
        const n = this.steps.length;
        return this.steps.map((s, k) => {
          const node = this.def.nodes.find(x => x.id === s.node);
          return { key: this.prefix + k, title: 'Step ' + (k + 1) + ' of ' + n + (node ? ': ' + node.label.join(' ') : ''), text: narrateStep(this.def, this.steps, k) };
        });
      },
      // A dot travels along the arrow just taken (skipped when the reader prefers reduced motion).
      animate() {
        const dot = this.$refs.dot, flow = this.$refs.flow;
        if (!dot || !flow) return;
        cancelAnimationFrame(this._raf);
        const path = flow.querySelector('path.walk-edge.on');
        if (!path || this.reduced || !path.getTotalLength) { dot.setAttribute('opacity', '0'); return; }
        const len = path.getTotalLength(), t0 = performance.now(), dur = this.pace.move * 1200;
        const frameFn = now => {
          const f = Math.min(1, (now - t0) / dur), p = path.getPointAtLength(len * (1 - Math.pow(1 - f, 2)));
          dot.setAttribute('cx', p.x); dot.setAttribute('cy', p.y); dot.setAttribute('opacity', f < 1 ? '1' : '0');
          if (f < 1) this._raf = requestAnimationFrame(frameFn);
        };
        this._raf = requestAnimationFrame(frameFn);
      },
      key(e) {
        if (/^(SELECT|INPUT|TEXTAREA)$/.test((e.target && e.target.tagName) || '')) return;
        if (e.key === 'ArrowRight') { e.preventDefault(); this.next(); }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); this.back(); }
      }
    },
    template: `
    <div class="walk card" role="group" :aria-label="def.title" @keydown="key" :style="{ '--walk-move': pace.move + 's' }">
      <div class="walk-head">
        <strong>{{ def.title }}</strong>
        <span class="walk-count" aria-hidden="true">step {{ i + 1 }} of {{ steps.length }}</span>
      </div>
      <div class="walk-grid">
        <div class="walk-flow-wrap">
          <svg ref="flow" class="walk-flow" :viewBox="'0 0 ' + lay.w + ' ' + lay.h" :width="lay.w" role="img" :aria-label="describe">
            <defs>
              <marker :id="'wa' + id" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="walk-head-off"/></marker>
              <marker :id="'wb' + id" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="walk-head-on"/></marker>
            </defs>
            <path v-for="e in lay.edges" :key="e.id" :d="e.d" class="walk-edge" :class="{ on: edge && e.id === edge.id }" :marker-end="'url(#' + (edge && e.id === edge.id ? 'wb' : 'wa') + id + ')'"/>
            <text v-for="e in lay.edges" :key="'l' + e.id" :x="e.lx" :y="e.ly" class="walk-elabel" :class="{ on: edge && e.id === edge.id }">{{ e.label }}</text>
            <g v-for="n in def.nodes" :key="n.id" class="walk-node" :class="[n.kind || 'step', { on: n.id === step.node, seen: seen.has(n.id) }]" @click="jump(n.id)">
              <rect :x="lay.pos[n.id].x" :y="lay.pos[n.id].y" :width="lay.pos[n.id].w" :height="lay.pos[n.id].h" rx="8"/>
              <text v-for="(t, k) in n.label" :key="k" :x="lay.pos[n.id].cx" :y="lineY(n, k)" text-anchor="middle">{{ t }}</text>
            </g>
            <circle ref="dot" r="4.5" cx="0" cy="0" opacity="0" class="walk-dot"/>
          </svg>
        </div>
        <div class="walk-side">
          <svg class="walk-plot" :viewBox="'0 0 ' + 300 + ' ' + 210" role="img" :aria-label="'Plot: ' + (step.say || '')">
            <defs><clipPath :id="'wc' + id"><rect x="38" y="12" width="252" height="170"/></clipPath></defs>
            <g v-for="part in shapes" :key="part.k" :clip-path="part.clip ? 'url(#wc' + id + ')' : null">
              <template v-for="s in part.shapes" :key="s.k">
                <circle v-if="s.t === 'c'" cx="0" cy="0" :r="s.r || 4" :class="s.cls" :style="{ transform: 'translate(' + s.x + 'px,' + s.y + 'px)' }"/>
                <line v-else-if="s.t === 'l'" :x1="s.x1" :y1="s.y1" :x2="s.x2" :y2="s.y2" :class="s.cls"/>
                <path v-else-if="s.t === 'p'" :d="s.d" :class="s.cls"/>
                <rect v-else-if="s.t === 'r'" :x="s.x" :y="s.y" :width="Math.max(0, s.w)" :height="Math.max(0, s.h)" :class="s.cls" rx="2"/>
                <text v-else-if="s.t === 't'" :x="s.x" :y="s.y" :text-anchor="s.a || 'start'" :class="s.cls">{{ s.s }}</text>
              </template>
            </g>
          </svg>
          <div class="walk-now">
            <p class="walk-say">{{ step.say }}</p>
            <p v-if="step.ask" class="walk-ask"><b>Think first:</b> {{ step.ask }}</p>
            <div v-if="step.formula" class="walk-formula">{{ step.formula }}</div>
            <div v-if="step.vars && step.vars.length" class="walk-vars"><span v-for="v in step.vars" :key="v[0]" class="walk-var">{{ v[0] }} = <b>{{ v[1] }}</b></span></div>
          </div>
        </div>
      </div>
      <div v-if="where.length" class="formula-where walk-where"><div class="fw-label">where</div>
        <dl><div v-for="s in where" :key="s[0]" class="walk-sym"><dt>{{ s[0] }}</dt><dd>{{ s[1] }}</dd></div></dl>
      </div>
      <div class="walk-reason">
        <p v-if="node.why"><b>Why this step:</b> {{ node.why }}</p>
        <p v-if="edge && edge.why"><b>Why this arrow ({{ edge.label }}):</b> {{ edge.why }}</p>
      </div>
      <ol class="walk-code" :aria-label="'Code, with the line for this step marked'">
        <li v-for="(l, k) in def.code" :key="k" :class="{ on: step.lines && step.lines.includes(k + 1) }"><code>{{ l || ' ' }}</code></li>
      </ol>
      <div class="walk-controls">
        <button class="icon-btn" @click="restart" aria-label="Back to the first step" title="Back to the first step"><mla-icon name="refresh" :size="16"/></button>
        <button class="icon-btn" @click="back" :disabled="i === 0" aria-label="Previous step" title="Previous step (left arrow)"><mla-icon name="skipBack" :size="16"/></button>
        <button class="btn sm primary" @click="toggle" :aria-label="playing ? 'Pause' : pace.key === 'voice' ? 'Read the steps aloud' : 'Play the steps'"><mla-icon :name="playing ? 'pause' : pace.key === 'voice' ? 'volume' : 'play'" :size="14"/> {{ playing ? 'Pause' : pace.key === 'voice' ? 'Listen' : 'Play' }}</button>
        <button class="icon-btn" @click="next" :disabled="i >= steps.length - 1" aria-label="Next step" title="Next step (right arrow)"><mla-icon name="skipForward" :size="16"/></button>
        <select v-model="speedKey" @change="changeSpeed" aria-label="Speed" title="How fast Play moves through the steps"><option v-for="p in paces" :key="p.key" :value="p.key">{{ p.label }}</option></select>
        <label v-for="p in def.params || []" :key="p.key" class="walk-param">{{ p.label }}
          <select v-model="params[p.key]" :aria-label="p.label"><option v-for="o in p.options" :key="String(o[0])" :value="o[0]">{{ o[1] }}</option></select>
        </label>
      </div>
      <p class="sr-only" aria-live="polite">{{ said }}</p>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
