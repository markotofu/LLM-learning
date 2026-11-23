/* ML Atlas – application root: navigation, routing, onboarding, mount */
(function (global) {
  const MLA = global.MLA;
  if (!global.Vue) return; // e.g. data-only test runs in Node
  const { createApp } = global.Vue;

  const NAV = [
    { name: 'home', label: 'Dashboard', icon: 'dashboard', path: '/home' },
    { name: 'map', label: 'Concept Map', icon: 'map', path: '/map' },
    { name: 'tracks', label: 'Learning Tracks', icon: 'route', path: '/tracks' },
    { name: 'problems', label: 'Practice Problems', icon: 'code', path: '/problems' },
    { name: 'projects', label: 'Projects & Cases', icon: 'flask', path: '/projects' },
    { name: 'playground', label: 'Python Playground', icon: 'terminal', path: '/playground' },
    { name: 'review', label: 'Review', icon: 'cards', path: '/review' },
    { name: 'glossary', label: 'Glossary', icon: 'book', path: '/glossary' },
    { name: 'achievements', label: 'Achievements', icon: 'trophy', path: '/achievements' },
    { name: 'profile', label: 'Progress & Sync', icon: 'sync', path: '/profile' }
  ];
  MLA.nav = NAV;
  const BOTTOM = ['home', 'map', 'problems', 'playground'];       // on phones; every other page is under More
  const ROUTE_VIEW = {
    home: { comp: 'view-home', nav: 'home' },
    map: { comp: 'view-map', nav: 'map' },
    concept: { comp: 'view-concept', nav: 'map' },
    problems: { comp: 'view-problems', nav: 'problems' },
    problem: { comp: 'view-problem', nav: 'problems' },
    playground: { comp: 'view-playground', nav: 'playground' },
    projects: { comp: 'view-projects', nav: 'projects' },
    tracks: { comp: 'view-tracks', nav: 'tracks' },
    track: { comp: 'view-track', nav: 'tracks' },
    review: { comp: 'view-review', nav: 'review' },
    glossary: { comp: 'view-glossary', nav: 'glossary' },
    achievements: { comp: 'view-achievements', nav: 'achievements' },
    profile: { comp: 'view-profile', nav: 'profile' },
    selftest: { comp: 'view-selftest', nav: 'profile' },
    placement: { comp: 'view-placement', nav: 'home' },
    terms: { comp: 'view-terms', nav: '' },
    privacy: { comp: 'view-privacy', nav: '' }
  };

  const Root = {
    data() { return { sidebarOpen: false, route: MLA.router.route, state: MLA.store.state, speech: MLA.speech.state, loading: false, loadError: '', moreOpen: false }; },
    computed: {
      current() { return ROUTE_VIEW[this.route.name] || { comp: 'view-notfound', nav: '' }; },
      viewProps() { return { route: this.route }; },
      title() {
        if (this.route.name === 'concept') { const c = MLA.progress.concept(this.route.params.id); if (c) return c.title; }
        if (this.route.name === 'problem') { const p = MLA.progress.problem(this.route.params.id); if (p) return p.title; }
        if (this.route.name === 'track') { const t = MLA.progress.track(this.route.params.id); if (t) return t.title; }
        return this.route.title || 'ML Atlas';
      },
      nav() { return NAV; },
      bottomNav() { return NAV.filter(n => BOTTOM.includes(n.name)); },
      moreNav() { return NAV.filter(n => !BOTTOM.includes(n.name)); },
      moreActive() { return this.moreNav.some(n => this.isActive(n.name)); },
      level() { return MLA.gamification.levelInfo(); },
      streak() { return MLA.gamification.streakStatus(); },
      py() { return MLA.python.status; },
      overall() { return MLA.progress.overall(); },
      copyright() { return MLA.data.legal ? MLA.data.legal.copyright() : ''; }
    },
    watch: {
      'route.path': { immediate: true, handler() { this.sidebarOpen = false; this.moreOpen = false; this.loadContent(); } },
      title: { immediate: true, handler(t) { if (global.document) document.title = (t && t !== 'ML Atlas' ? t + ' · ' : '') + 'ML Atlas'; } }
    },
    methods: {
      go(path) { MLA.router.navigate(path); this.sidebarOpen = false; },
      // Lesson files load on demand (js/data/branches.js): pages that show lesson bodies wait for theirs.
      contentNeeded(r) {
        if (r.name === 'concept') return [r.params.id];
        if (r.name === 'problem') { const p = MLA.content.problemById[r.params.id]; return p ? [p.concept] : []; }
        if (r.name === 'review') return Object.values(this.state.mistakes || {}).map(m => m && m.concept);
        if (r.name === 'selftest') return MLA.data.concepts.map(c => c.id);
        return [];
      },
      loadContent() {
        const ids = this.contentNeeded(this.route).filter(id => id && MLA.content.byId[id] && !MLA.content.loaded(id));
        this.loadError = '';
        this.loading = ids.length > 0;
        if (!this.loading) return;
        const path = this.route.path;
        MLA.content.ensure(ids).then(() => { if (this.route.path === path) this.loading = false; },
          e => { if (this.route.path === path) { this.loading = false; this.loadError = e.message; } });
      },
      isActive(name) { return this.current.nav === name; },
      icon(n) { return MLA.ui.iconSvg(n, 18); },
      lookLabel() { return this.state.profile.style === 'classic' ? 'Switch to the Baroque look' : 'Switch to the Classic look'; },
      toggleLook() {
        const s = this.state.profile.style === 'classic' ? 'baroque' : 'classic';
        this.state.profile.style = s; MLA.ui.applyTheme(this.state.profile.theme, s);
        MLA.ui.toast((s === 'classic' ? 'Classic' : 'Baroque') + ' look. Themes for each look are in Progress & Sync.', 'info', 3000);
      },
      pyLabel() { const p = this.py; return p.ready ? (p.state === 'busy' ? 'Running…' : 'Python ready') : p.state === 'loading' ? 'Loading Python…' : p.state === 'error' ? 'Python failed to load' : 'Python (loads on first run)'; }
    },
    mounted() {
      MLA.ui.applyTheme(this.state.profile.theme, this.state.profile.style);
      try { MLA.gamification.touchStreak(); MLA.gamification.checkAchievements(); } catch (e) { /* ignore */ }
      if (!this.state.profile.onboarded) setTimeout(() => MLA.app.onboard(), 350);
      else if (this.state.profile.preloadPython) setTimeout(() => MLA.python.warmup(), 1500);
      setTimeout(() => MLA.content.prefetchAll(), 8000);      // after the first page is up: keep every lesson available offline
      document.addEventListener('keydown', e => {             // Ctrl+K or Cmd+K anywhere, or "/" when not typing: search
        const t = e.target || {}, typing = /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || '') || t.isContentEditable;
        if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === 'k' || e.key === 'K')) { e.preventDefault(); MLA.search.open(); }
        else if (e.key === '/' && !typing && !e.ctrlKey && !e.metaKey) { e.preventDefault(); MLA.search.open(); }
      });
    },
    template: `
    <div class="app" :class="{'full-route': route.full, 'has-player': speech.playing || speech.upNext}">
      <a class="skip-link" href="#main">Skip to content</a>
      <aside class="sidebar" :class="{open: sidebarOpen}">
        <div class="brand" @click="go('/home')">
          <span class="brand-logo" aria-hidden="true"><svg class="logo-classic" viewBox="0 0 32 32" width="34" height="34"><rect width="32" height="32" rx="9" fill="url(#bg-g)"/><defs><linearGradient id="bg-g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0ea5e9"/><stop offset="1" stop-color="#14b8a6"/></linearGradient></defs><g stroke="#fff" stroke-width="1.6" stroke-opacity=".9" fill="none"><path d="M16 9 9 16M16 9l7 7M9 16l7 7M23 16l-7 7M9 16h14"/></g><g fill="#fff"><circle cx="16" cy="9" r="2.6"/><circle cx="9" cy="16" r="2.3"/><circle cx="23" cy="16" r="2.3"/><circle cx="16" cy="23" r="2.6"/></g></svg><svg class="logo-baroque" viewBox="0 0 32 32" width="36" height="36"><defs><radialGradient id="bm-sky" cx="50%" cy="38%" r="70%"><stop offset="0" stop-color="#fff4d6"/><stop offset=".42" stop-color="#a9cbe9"/><stop offset="1" stop-color="#3a669f"/></radialGradient><linearGradient id="bm-gold" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f6dc8c"/><stop offset=".5" stop-color="#c99a32"/><stop offset="1" stop-color="#8a6116"/></linearGradient></defs><circle cx="16" cy="16" r="15.3" fill="url(#bm-gold)"/><circle cx="16" cy="16" r="12.7" fill="url(#bm-sky)"/><g stroke="#fffaf0" stroke-width="1.35" stroke-opacity=".92" fill="none"><path d="M16 9.6 10.2 16M16 9.6l5.8 6.4M10.2 16l5.8 6.4M21.8 16 16 22.4M10.2 16h11.6"/></g><g fill="#fffaf0" stroke="#b8892b" stroke-width=".7"><circle cx="16" cy="9.6" r="2.3"/><circle cx="10.2" cy="16" r="2"/><circle cx="21.8" cy="16" r="2"/><circle cx="16" cy="22.4" r="2.3"/></g></svg></span>
          <div><div class="brand-name">ML Atlas</div><div class="brand-sub">Machine learning → LLMs</div></div>
          <button class="icon-btn sidebar-close" aria-label="Close menu" @click.stop="sidebarOpen=false">&times;</button>
        </div>
        <nav class="nav" aria-label="Main">
          <a v-for="n in nav" :key="n.name" :class="{active: isActive(n.name)}" :href="'#'+n.path" @click.prevent="go(n.path)"><span v-html="icon(n.icon)"></span><span>{{ n.label }}</span></a>
        </nav>
        <div class="sidebar-foot">
          <div class="row between"><span>Level {{ level.level }}</span><span class="tiny dim">{{ MLA.gamification.title(level.level) }}</span></div>
          <mla-progress :value="level.pct" thin style="margin-top:6px" />
          <div class="row between tiny dim" style="margin-top:6px"><span>{{ state.xp.toLocaleString() }} XP</span><span>{{ overall.conceptsDone }}/{{ overall.concepts }} concepts</span></div>
          <div class="legal-links tiny dim">{{ copyright }} · <a href="#/terms" @click.prevent="go('/terms')">Terms</a> · <a href="#/privacy" @click.prevent="go('/privacy')">Privacy</a></div>
        </div>
      </aside>
      <div class="backdrop" :class="{show: sidebarOpen}" @click="sidebarOpen=false"></div>
      <main class="main" id="main">
        <header class="topbar">
          <button class="icon-btn menu-btn" aria-label="Open menu" @click="sidebarOpen=true" v-html="icon('menu')"></button>
          <h1 class="page-title">{{ title }}</h1>
          <div class="topbar-right">
            <button class="icon-btn search-btn" title="Search (Ctrl+K)" aria-label="Search (Ctrl+K)" @click="MLA.search.open()" v-html="icon('search')"></button>
            <button class="icon-btn look-btn" :title="lookLabel()" :aria-label="lookLabel()" @click="toggleLook" v-html="icon('palette')"></button>
            <span class="chip py-chip" :title="py.message || pyLabel()" @click="go('/playground')"><span class="py-dot" :class="py.state"></span><span class="hide-sm">{{ pyLabel() }}</span></span>
            <span class="chip amber" title="Learning streak (days)" :aria-label="'Learning streak: ' + streak.current + ' days'"><span class="chip-ico" v-html="icon('flame')"></span>{{ streak.current }}</span>
            <span class="chip accent" title="Experience points"><mla-icon name="star" :size="13"/> {{ state.xp.toLocaleString() }}</span>
          </div>
        </header>
        <div class="view" :class="{'view-full': route.full}">
          <div v-if="loading" class="content-loading" role="status">Loading…</div>
          <empty-state v-else-if="loadError" icon="warning" :text="loadError"><button class="btn primary mt" @click="loadContent">Try again</button></empty-state>
          <component v-else :is="current.comp" v-bind="viewProps" :key="route.path" />
        </div>
      </main>
      <nav class="bottom-nav" aria-label="Quick">
        <a v-for="n in bottomNav" :key="n.name" :class="{active: isActive(n.name)}" :href="'#'+n.path" @click.prevent="go(n.path)"><span v-html="icon(n.icon)"></span><span>{{ { home: 'Dashboard', map: 'Map', problems: 'Problems', playground: 'Playground' }[n.name] || n.label }}</span></a>
        <a href="#" :class="{active: moreActive || moreOpen}" aria-haspopup="true" :aria-expanded="moreOpen ? 'true' : 'false'" @click.prevent="moreOpen = !moreOpen"><span v-html="icon('menu')"></span><span>More</span></a>
      </nav>
      <div v-if="moreOpen" class="more-sheet" @click.self="moreOpen = false" @keydown.esc="moreOpen = false">
        <div class="more-panel" role="dialog" aria-label="More pages">
          <button class="more-item" @click="moreOpen = false; MLA.search.open()"><span v-html="icon('search')"></span><span>Search</span></button>
          <a v-for="n in moreNav" :key="n.name" class="more-item" :class="{active: isActive(n.name)}" :href="'#'+n.path" @click.prevent="go(n.path)"><span v-html="icon(n.icon)"></span><span>{{ n.label }}</span></a>
        </div>
      </div>
      <mla-player /><mla-search /><mla-toasts /><mla-modal />
    </div>`
  };

  MLA.components['view-notfound'] = { template: `<empty-state icon="info" text="Page not found."><button class="btn primary mt" @click="MLA.router.navigate('/home')">Go to the dashboard</button></empty-state>` };

  MLA.app = {
    Root,
    mount() {
      // The published site precompiles every template (tools/precompile.js writes js/precompiled.js), so its
      // Content-Security-Policy can forbid eval; in development Vue compiles the templates in the browser.
      const R = global.MLA_RENDER;
      if (R) {
        for (const name of Object.keys(MLA.components)) {
          const c = MLA.components[name];
          if (R[name] && typeof c.template === 'string') { c.render = R[name]; delete c.template; }
        }
        if (R.root) { Root.render = R.root; delete Root.template; }
      }
      const app = createApp(Root);
      app.config.globalProperties.MLA = MLA;
      app.config.warnHandler = function () {};
      app.config.errorHandler = function (err) { try { console.error(err); MLA.ui.toast('Something went wrong: ' + (err && err.message ? err.message : err), 'error', 6000); } catch (e) { /* ignore */ } };
      MLA.registerComponents(app);
      app.mount('#app');
      MLA.app._app = app;
    },
    onboard() { MLA.ui.openModal({ title: 'Welcome to ML Atlas', component: 'onboard-dialog', actions: [], dismissible: false }); }
  };

  MLA.components['onboard-dialog'] = {
    data() { return { name: MLA.store.state.profile.name === 'Learner' ? '' : MLA.store.state.profile.name, goal: 'foundations' }; },
    computed: { tracks() { return MLA.progress.tracks().filter(t => !t.hidden); } },
    methods: {
      finish(start) {
        const p = MLA.store.state.profile;
        p.name = (this.name || 'Learner').trim().slice(0, 24) || 'Learner';
        p.goal = this.goal; p.onboarded = true; MLA.store.saveNow(); MLA.ui.closeModal(true);
        MLA.ui.toast('Welcome, ' + p.name + '! Your progress saves automatically in this browser.', 'success', 4500);
        if (start === 'placement') MLA.router.navigate('/placement');
        else if (start) { const r = MLA.progress.recommendNext(); MLA.router.navigate(r.concept ? '/concept/' + r.concept.id : '/map'); }
        else MLA.router.navigate('/map');
      }
    },
    template: `
    <div class="onboard">
      <p>ML Atlas is an interactive <b>map</b> of machine learning: from the maths and Python you need, through classic ML and neural networks, to how large language models are built, fine-tuned and used. Every concept is a lesson with runnable code, demos, a quiz and coding problems that run right here in your browser.</p>
      <div class="field"><label>What should we call you?</label><input type="text" v-model="name" maxlength="24" placeholder="Your name" aria-label="What should we call you?" @keyup.enter="finish(true)" /></div>
      <div class="field"><label>Where do you want to go? (you can change this any time)</label>
        <div class="onboard-tracks"><button v-for="t in tracks" :key="t.id" :class="{active: goal===t.id}" @click="goal=t.id"><span class="ot-icon"><mla-icon :name="t.icon" :size="18"/></span><span><b>{{ t.title }}</b><span class="tiny dim">{{ t.short }}</span></span></button></div>
      </div>
      <div class="callout note"><div class="callout-title">Good to know</div>Progress is saved in this browser automatically. Use <b>Progress &amp; Sync</b> to move it to another device. The first time you run code, about 15 MB of Python downloads once and is then cached.</div>
      <div class="btn-row mt"><button class="btn primary" @click="finish(true)">Start learning</button><button class="btn ghost" @click="finish(false)">Explore the map first</button></div>
      <p class="small muted mt-sm onboard-placement">Know some of this already? <button class="btn sm" @click="finish('placement')"><mla-icon name="target" :size="14"/> Take the placement test</button> About 12 to 15 questions; the lessons you know are marked, so your path starts at the first gap.</p>
    </div>`
  };

  function boot() { MLA.router.sync(); MLA.app.mount(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})(typeof globalThis !== 'undefined' ? globalThis : window);
