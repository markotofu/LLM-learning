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
  C['branch-radar'] = { props: { items: Array, size: { default: 260 } }, computed: {
    pts() {
      const n = this.items.length; const cx = this.size / 2, cy = this.size / 2, R = this.size / 2 - 42;
      const ring = p => this.items.map((s, i) => { const a = (Math.PI * 2 * i / n) - Math.PI / 2; const r = R * (p != null ? p : Math.max(0.03, s.pct / 100)); return (cx + r * Math.cos(a)).toFixed(1) + ',' + (cy + r * Math.sin(a)).toFixed(1); }).join(' ');
      const labels = this.items.map((s, i) => { const a = (Math.PI * 2 * i / n) - Math.PI / 2; return { x: cx + (R + 20) * Math.cos(a), y: cy + (R + 20) * Math.sin(a), name: s.branch.short || s.branch.name, color: s.branch.color, anchor: Math.cos(a) > 0.3 ? 'start' : Math.cos(a) < -0.3 ? 'end' : 'middle' }; });
      return { data: ring(), grid: [0.25, 0.5, 0.75, 1].map(ring), labels };
    }
  }, template: `
    <svg :width="size" :height="size" :viewBox="'0 0 '+size+' '+size" class="radar" style="max-width:100%;height:auto">
      <polygon v-for="(g,i) in pts.grid" :key="i" :points="g" fill="none" stroke="var(--border)" stroke-width="1"/>
      <polygon :points="pts.data" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="2"/>
      <text v-for="(l,i) in pts.labels" :key="i" :x="l.x" :y="l.y" :text-anchor="l.anchor" dominant-baseline="central" :fill="l.color">{{ l.name }}</text>
    </svg>` };

  C['list-tile'] = { props: { icon: String, title: String, sub: String, done: Boolean, to: String, badge: String, badgeKind: String, color: String },
    methods: { go() { if (this.to) MLA.router.navigate(this.to); } },
    template: `
    <a class="list-item" @click="go" tabindex="0" @keydown.enter="go">
      <div class="li-icon" :class="{done}" :style="color && !done ? {color: color} : null"><mla-icon :name="done?'check':icon" /></div>
      <div class="li-body"><div class="li-title">{{ title }}</div><div class="li-sub"><span v-if="sub">{{ sub }}</span><slot name="sub"/></div></div>
      <div class="li-right"><span v-if="badge" class="badge" :class="badgeKind">{{ badge }}</span><mla-icon name="chevron" :size="16" /></div>
    </a>` };

  C['kbd-hint'] = { props: { keys: String }, computed: { mac() { try { return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent); } catch (e) { return false; } }, parts() { return this.keys.split('+').map(k => k === 'Mod' ? (this.mac ? '⌘' : 'Ctrl') : k); } }, template: `<span class="kbd-group"><kbd v-for="(k,i) in parts" :key="i" class="kbd">{{ k }}</kbd></span>` };

  MLA.registerComponents = function (app) { Object.keys(C).forEach(name => app.component(name, C[name])); };
})(typeof globalThis !== 'undefined' ? globalThis : window);
