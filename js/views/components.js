/* ML Atlas – shared Vue components (registered globally) */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};
  const C = MLA.components;

  C['mla-icon'] = { props: { name: String, size: { default: 18 } }, computed: { svg() { return MLA.ui.iconSvg(this.name, this.size); } }, template: `<span class="mla-icon" v-html="svg"></span>` };

  // Markdown renderer (trusted authored content). Wires internal links and copy buttons.
  C['mla-md'] = {
    props: { src: String, inline: Boolean },
    computed: { html() { return this.inline ? MLA.util.inline(this.src || '') : MLA.util.md(this.src || ''); } },
    mounted() { this.wire(); }, updated() { this.wire(); },
    methods: {
      wire() {
        const root = this.$el; if (!root || !root.querySelectorAll) return;
        root.querySelectorAll('a[href^="#/"]').forEach(a => { if (a._wired) return; a._wired = true; a.addEventListener('click', e => { e.preventDefault(); MLA.router.navigate(a.getAttribute('href').slice(1)); }); });
        root.querySelectorAll('.code-copy').forEach(b => {
          if (b._wired) return; b._wired = true;
          b.addEventListener('click', () => { const pre = b.closest('.code-block').querySelector('pre'); MLA.util.copyText(pre.textContent).then(ok => { b.textContent = ok ? 'Copied' : 'Copy failed'; setTimeout(() => { b.textContent = 'Copy'; }, 1400); }); });
        });
      }
    },
    template: `<span v-if="inline" v-html="html"></span><div v-else class="md-body" v-html="html"></div>`
  };

  C['mla-progress'] = { props: { value: Number, kind: String, thin: Boolean, label: String, color: String }, template: `
    <div class="progress-wrap">
      <div v-if="label" class="progress-label"><span>{{ label }}</span><span>{{ Math.round(value) }}%</span></div>
      <div class="progress" :class="[kind, {thin}]"><span :style="{width: Math.max(0,Math.min(100,value))+'%', background: color || null}"></span></div>
    </div>` };

  C['stat-card'] = { props: { label: String, value: [String, Number], sub: String, icon: String, color: String }, template: `
    <div class="stat">
      <div class="label">{{ label }}</div>
      <div class="value">{{ value }}</div>
      <div v-if="sub" class="sub">{{ sub }}</div>
      <div v-if="icon" class="stat-icon" :style="color?{color:color,background:'color-mix(in srgb, '+color+' 16%, transparent)'}:null"><mla-icon :name="icon" /></div>
    </div>` };

  C['section-head'] = { props: { title: String, sub: String }, template: `
    <div class="section-head"><div><h2>{{ title }}</h2><p v-if="sub">{{ sub }}</p></div><div class="row"><slot/></div></div>` };

  C['empty-state'] = { props: { icon: { default: 'info' }, text: String }, template: `<div class="empty"><mla-icon :name="icon" :size="28" /><p style="margin-top:8px">{{ text }}</p><slot/></div>` };

  C['branch-chip'] = { props: { id: String, small: Boolean }, computed: { b() { return MLA.progress.branch(this.id) || { name: this.id, color: '#888' }; } }, template: `
    <span class="chip branch-chip" :class="{tiny:small}" :style="{'--bc': b.color}"><span class="dot"></span>{{ small ? (b.short || b.name) : b.name }}</span>` };

  C['level-badge'] = { props: { level: Number }, template: `<span class="badge" :class="'l'+level">{{ ['','Beginner','Intermediate','Advanced'][level] }}</span>` };

  C['diff-chip'] = { props: { d: String }, template: `<span class="diff" :class="d">{{ d==='easy'?'Easy':d==='medium'?'Medium':'Hard' }}</span>` };

  C['status-dot'] = { props: { status: String, size: { default: 18 } }, computed: {
    icon() { return { mastered: 'star', completed: 'check', started: 'play', ready: 'target', locked: 'lock', solved: 'check', attempted: 'play', new: 'code' }[this.status] || 'info'; },
    label() { return { mastered: 'Mastered', completed: 'Completed', started: 'In progress', ready: 'Ready to learn', locked: 'Prerequisites first', solved: 'Solved', attempted: 'Attempted', new: 'Not started' }[this.status] || ''; }
  }, template: `<span class="status-dot" :class="status" :title="label"><mla-icon :name="icon" :size="Math.round(size*0.62)" /></span>` };

  C['mla-toasts'] = { computed: { toasts() { return MLA.ui.toasts; } }, template: `
    <div class="toasts" aria-live="polite"><div v-for="t in toasts" :key="t.id" class="toast" :class="t.kind">{{ t.msg }}</div></div>` };

  // The read-aloud player: shown while a lesson (or one part of it) is being read; it stays while you browse elsewhere.
  // The speaker button on every lesson box: plays that box alone (the lesson page's listenOne).
  C['listen-btn'] = { props: { label: String }, emits: ['play'], data() { return { s: MLA.speech.state }; },
    template: `<button v-if="s.supported" class="icon-btn sec-listen" @click="$emit('play')" :aria-label="'Listen to: ' + label" title="Listen to this part"><mla-icon name="volume" :size="16"/></button>` };

  C['mla-player'] = {
    data() { return { s: MLA.speech.state, profile: MLA.store.state.profile, rates: [0.75, 1, 1.25, 1.5, 1.75, 2] }; },
    computed: {
      rate: { get() { return Number(this.profile.speechRate) || 1; }, set(v) { MLA.speech.setRate(Number(v)); } },
      onLesson() { const r = MLA.router.route; return r.name === 'concept' && r.params.id === this.s.source; },
      captions() { return this.profile.speechCaptions !== false; }
    },
    methods: {
      back() { MLA.router.navigate('/concept/' + this.s.source); }
    },
    template: `
    <div v-if="s.playing" class="player" role="region" aria-label="Read aloud">
      <div class="player-info">
        <div class="tiny dim">{{ s.paused ? 'Paused' : 'Reading aloud' }}<span v-if="s.total > 1"> · part {{ s.index + 1 }} of {{ s.total }}</span></div>
        <div class="player-title">{{ s.title }}</div>
        <div v-if="captions && s.caption" class="player-caption" aria-hidden="true">{{ s.caption }}</div>
        <a v-if="s.source && !onLesson" class="tiny" href="#" @click.prevent="back">Back to {{ s.lessonTitle }}</a>
      </div>
      <div class="player-controls">
        <button class="icon-btn" @click="MLA.speech.prev()" aria-label="Previous part" title="Previous part"><mla-icon name="skipBack" :size="17"/></button>
        <button class="icon-btn player-main" @click="MLA.speech.toggle()" :aria-label="s.paused ? 'Resume' : 'Pause'" :title="s.paused ? 'Resume' : 'Pause'"><mla-icon :name="s.paused ? 'play' : 'pause'" :size="17"/></button>
        <button class="icon-btn" @click="MLA.speech.next()" aria-label="Next part" title="Next part"><mla-icon name="skipForward" :size="17"/></button>
        <select v-model="rate" class="player-rate" aria-label="Reading speed" title="Reading speed"><option v-for="r in rates" :key="r" :value="r">{{ r }}×</option></select>
        <button class="icon-btn" @click="MLA.speech.stop()" aria-label="Stop reading" title="Stop"><mla-icon name="stop" :size="17"/></button>
      </div>
    </div>
    <div v-else-if="s.upNext" class="player" role="region" aria-label="Read aloud">
      <div class="player-info">
        <div class="tiny dim">Lesson finished{{ profile.speechContinue ? '. The next one starts in a moment.' : '' }}</div>
        <div class="player-title">Up next: {{ s.upNext.title }}</div>
      </div>
      <div class="player-controls">
        <button class="btn sm primary" @click="MLA.speech.playNext()"><mla-icon name="play" :size="15"/> Play the next lesson</button>
        <button class="icon-btn" @click="MLA.speech.dismiss()" aria-label="Close" title="Close"><mla-icon name="x" :size="17"/></button>
      </div>
    </div>` };

  C['mla-modal'] = {
    computed: { m() { return MLA.ui.modal; }, comp() { return this.m.component ? MLA.components[this.m.component] : null; } },
    methods: { act(a) { MLA.ui.closeModal(a.value); }, bg(e) { if (e.target === e.currentTarget && this.m.dismissible) MLA.ui.closeModal(null); }, key(e) { if (e.key === 'Escape' && this.m.open && this.m.dismissible) MLA.ui.closeModal(null); } },
    mounted() { document.addEventListener('keydown', this.key); }, unmounted() { document.removeEventListener('keydown', this.key); },
    template: `
    <div v-if="m.open" class="modal-bg" @mousedown="bg">
      <div class="modal" :class="{wide:m.wide}" role="dialog" aria-modal="true">
        <div class="modal-head"><h3>{{ m.title }}</h3><button class="icon-btn" aria-label="Close" @click="act({value:null})"><mla-icon name="x"/></button></div>
        <div class="modal-body">
          <component v-if="comp" :is="comp" v-bind="m.props || {}" />
          <div v-else class="md-body" v-html="m.html"></div>
        </div>
        <div class="modal-foot" v-if="m.actions && m.actions.length">
          <button v-for="a in m.actions" :key="a.label" class="btn" :class="a.kind" @click="act(a)">{{ a.label }}</button>
        </div>
      </div>
    </div>` };

  C['score-ring'] = { props: { value: Number, size: { default: 120 }, color: String }, computed: {
    stroke() { if (this.color) return this.color; return this.value >= 80 ? 'var(--green)' : this.value >= 60 ? 'var(--amber)' : 'var(--red)'; },
    dash() { const r = 50; const c = 2 * Math.PI * r; return { r, c, off: c * (1 - Math.max(0, Math.min(100, this.value)) / 100) }; }
  }, template: `
    <svg :width="size" :height="size" viewBox="0 0 120 120" class="score-ring">
      <circle cx="60" cy="60" :r="dash.r" fill="none" stroke="var(--panel3)" stroke-width="10"/>
      <circle cx="60" cy="60" :r="dash.r" fill="none" :stroke="stroke" stroke-width="10" stroke-linecap="round"
        :stroke-dasharray="dash.c" :stroke-dashoffset="dash.off" transform="rotate(-90 60 60)" style="transition:stroke-dashoffset .6s"/>
      <text x="60" y="60" text-anchor="middle" dominant-baseline="central" font-size="26" font-weight="700" fill="var(--text)">{{ Math.round(value) }}%</text>
    </svg>` };

  // Radar chart of progress per branch
  // Skill radar: one spoke per branch, labelled with the branch and its percentage. Long names wrap onto two lines and the
  // viewBox grows to fit every label, so nothing is cut off at the card's width. Font sizes are in viewBox units.
  C['branch-radar'] = { props: { items: Array, size: { default: 260 } }, computed: {
    pts() {
      const n = this.items.length, R = this.size / 2 - 34, FS = 13, LH = 15;
      const at = (i, r) => { const a = (Math.PI * 2 * i / n) - Math.PI / 2; return { x: r * Math.cos(a), y: r * Math.sin(a), cos: Math.cos(a), sin: Math.sin(a) }; };
      const ring = p => this.items.map((s, i) => { const q = at(i, R * (p != null ? p : Math.max(0.03, s.pct / 100))); return q.x.toFixed(1) + ',' + q.y.toFixed(1); }).join(' ');
      const split = name => {
        if (name.length <= 12 || name.indexOf(' ') < 0) return [name];
        let cut = -1;
        for (let k = 0; k < name.length; k++) if (name[k] === ' ' && (cut < 0 || Math.abs(k - name.length / 2) < Math.abs(cut - name.length / 2))) cut = k;
        return [name.slice(0, cut), name.slice(cut + 1)];
      };
      let x0 = -R, x1 = R, y0 = -R, y1 = R;
      const labels = this.items.map((s, i) => {
        const q = at(i, R + 10);
        const lines = split(s.branch.short || s.branch.name).map(t => ({ t, pct: false })).concat([{ t: Math.round(s.pct) + '%', pct: true }]);
        const anchor = q.cos > 0.3 ? 'start' : q.cos < -0.3 ? 'end' : 'middle';
        const w = Math.max(...lines.map(l => l.t.length)) * FS * 0.62, h = lines.length * LH;
        const top = q.sin < -0.5 ? q.y - h : q.sin > 0.5 ? q.y : q.y - h / 2;
        const x = anchor === 'start' ? q.x + 2 : anchor === 'end' ? q.x - 2 : q.x;
        const left = anchor === 'start' ? x : anchor === 'end' ? x - w : x - w / 2;
        x0 = Math.min(x0, left); x1 = Math.max(x1, left + w); y0 = Math.min(y0, top); y1 = Math.max(y1, top + h);
        return { x, anchor, color: s.branch.color, lines: lines.map((l, k) => Object.assign(l, { y: top + LH * (k + 0.5) })) };
      });
      const pad = 6, box = [x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad].map(v => +v.toFixed(1));
      const summary = 'Skill radar: ' + this.items.map(s => (s.branch.short || s.branch.name) + ' ' + Math.round(s.pct) + '%').join(', ');
      return { data: ring(), grid: [0.25, 0.5, 0.75, 1].map(ring), labels, viewBox: box.join(' '), width: box[2], summary };
    }
  }, template: `
    <svg :width="pts.width" :viewBox="pts.viewBox" class="radar" style="max-width:100%;height:auto" role="img" :aria-label="pts.summary">
      <polygon v-for="(g,i) in pts.grid" :key="i" :points="g" fill="none" stroke="var(--border)" stroke-width="1"/>
      <polygon :points="pts.data" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="2"/>
      <text v-for="(l,i) in pts.labels" :key="i" :text-anchor="l.anchor" dominant-baseline="central" font-size="13">
        <tspan v-for="(ln,k) in l.lines" :key="k" :x="l.x" :y="ln.y" :fill="ln.pct ? 'var(--muted)' : l.color" :font-size="ln.pct ? 11 : 13">{{ ln.t }}</tspan>
      </text>
    </svg>` };

  // A case study (js/data/projects.js): what happened, what to take from it, and where it is documented.
  C['case-study'] = { props: { k: Object, lesson: Object },
    methods: { go(p) { MLA.router.navigate(p); } },
    template: `
    <article class="case">
      <div class="case-head"><h4>{{ k.title }}</h4><span class="chip tiny">{{ k.when }}</span></div>
      <mla-md :src="k.story" />
      <div class="case-lesson"><strong>What to take from it.</strong> <mla-md :src="k.lesson" inline /></div>
      <div class="tiny dim case-source">Source: <a v-if="k.source && k.source.url" :href="k.source.url" target="_blank" rel="noopener">{{ k.source.title }}</a><span v-else-if="k.source">{{ k.source.title }}</span></div>
      <a v-if="lesson" class="tiny case-link" tabindex="0" @click="go('/concept/' + lesson.id)" @keydown.enter="go('/concept/' + lesson.id)">From the lesson: {{ lesson.title }}</a>
    </article>` };

  // A sample project: goal, data, steps, how to know it is done, and stretch goals. A <details> that shows its summary
  // line (title, level, hours) until it is opened.
  // Progress lives in state.projects['lessonId:project-slug']: the steps ticked, the learner's notes and the dates;
  // it is saved, exported and merged with the rest of the progress. Ticking the last step awards XP once.
  C['project-card'] = { props: { p: Object, lesson: Object, open: Boolean },
    data() { return { state: MLA.store.state }; },
    computed: {
      levelClass() { return { starter: 'green', intermediate: 'amber', advanced: 'red' }[this.p.level] || ''; },
      key() { return MLA.util.projectKey(this.p); },
      rec() { return this.state.projects[this.key] || null; },
      done() { return (this.rec && this.rec.steps) || []; },
      complete() { return this.done.length >= this.p.steps.length; },
      practice() { return this.p.practice && MLA.datasets ? MLA.datasets.find(this.p.practice) : null; },
      notes: { get() { return (this.rec && this.rec.notes) || ''; }, set(v) { this.touch().notes = v; } }
    },
    methods: {
      go(p) { MLA.router.navigate(p); },
      tryIt() {                                 // a playground file that already loads the practice dataset
        const d = this.practice;
        MLA.playground.newFile(MLA.util.slug(this.p.title) + '.py', ['# ' + this.p.title + ': practice version, in your browser',
          '# ' + d.title + ' is built into the site. The steps of the project work the same way on it.',
          'import pandas as pd', '', 'df = pd.read_csv("data/' + d.name + '")', 'print(df.shape)', 'print(df.head())', ''].join(String.fromCharCode(10)));
        MLA.router.navigate('/playground');
      },
      touch() {
        const now = new Date().toISOString();
        if (!this.state.projects[this.key]) this.state.projects[this.key] = { steps: [], notes: '', startedAt: now, updatedAt: now, completedAt: null };
        const r = this.state.projects[this.key];
        r.updatedAt = now;
        return r;
      },
      toggle(i, on) {
        const r = this.touch(), set = new Set(r.steps);
        if (on) set.add(i); else set.delete(i);
        r.steps = [...set].sort((a, b) => a - b);
        if (r.steps.length >= this.p.steps.length && !r.completedAt) {
          r.completedAt = new Date().toISOString();
          MLA.gamification.addXP(60, 'Project built: ' + this.p.title, 'project');
          MLA.ui.toast('Project complete: ' + this.p.title + '.', 'success', 5000);
        }
      }
    },
    template: `
    <details class="project" :open="open">
      <summary>
        <span class="project-title">{{ p.title }}</span>
        <span class="project-meta"><span class="chip tiny project-level" :class="levelClass">{{ p.level }}</span><span class="chip tiny"><mla-icon name="clock" :size="12"/> about {{ p.hours }} hours</span><span class="chip tiny" :class="complete ? 'green' : done.length ? 'accent' : ''">{{ complete ? 'Done' : done.length ? done.length + ' of ' + p.steps.length + ' steps' : 'Not started' }}</span></span>
      </summary>
      <div class="project-body">
        <div class="project-row"><strong>Goal.</strong> <mla-md :src="p.goal" inline /></div>
        <div class="project-row"><strong>Data and tools.</strong> <mla-md :src="p.data" inline /></div>
        <div v-if="practice" class="project-row practice-row"><span><strong>Start in your browser.</strong> A synthetic practice version of this data ({{ practice.rows.toLocaleString() }} rows) is built in, so the first steps run in the Playground before you download the real dataset.</span> <button class="btn sm" @click="tryIt"><mla-icon name="terminal" :size="14"/> Open in the Playground</button></div>
        <div class="project-row"><strong>Steps</strong> <span class="tiny dim">tick each one as you finish it</span></div>
        <ol class="project-steps"><li v-for="(s,i) in p.steps" :key="i"><label class="step-check"><input type="checkbox" :checked="done.includes(i)" @change="toggle(i, $event.target.checked)" :aria-label="'Step ' + (i + 1) + ' done'"><span><mla-md :src="s" inline /></span></label></li></ol>
        <div class="project-row"><strong>Done when.</strong> <mla-md :src="p.done" inline /></div>
        <div class="project-row"><strong>Go further</strong></div>
        <ul class="project-steps"><li v-for="(s,i) in p.stretch" :key="i"><mla-md :src="s" inline /></li></ul>
        <div class="chips" aria-label="Skills you practise"><span v-for="s in p.skills" :key="s" class="chip tiny accent">{{ s }}</span></div>
        <div class="project-row"><strong>Your notes</strong></div>
        <textarea class="project-notes" v-model="notes" rows="3" placeholder="What you tried, what you found, where your code lives. Saved in this browser and included when you export your progress." aria-label="Your notes on this project"></textarea>
        <a v-if="lesson" class="tiny case-link" tabindex="0" @click="go('/concept/' + lesson.id)" @keydown.enter="go('/concept/' + lesson.id)">From the lesson: {{ lesson.title }}</a>
      </div>
    </details>` };

  C['kbd-hint'] = { props: { keys: String }, computed: { mac() { try { return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent); } catch (e) { return false; } }, parts() { return this.keys.split('+').map(k => k === 'Mod' ? (this.mac ? '⌘' : 'Ctrl') : k); } }, template: `<span class="kbd-group"><kbd v-for="(k,i) in parts" :key="i" class="kbd">{{ k }}</kbd></span>` };

  MLA.registerComponents = function (app) { Object.keys(C).forEach(name => app.component(name, C[name])); };
})(typeof globalThis !== 'undefined' ? globalThis : window);
