/* ML Atlas – learning tracks, glossary, flashcards (spaced repetition), achievements */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  // ---------- Learning tracks ----------
  MLA.components['view-tracks'] = {
    data() { return { state: MLA.store.state }; },
    computed: {
      P() { return MLA.progress; },
      list() { return this.P.tracks().filter(t => !t.hidden).map(t => ({ t, pr: this.P.trackProgress(t) })); }
    },
    methods: {
      go(p) { MLA.router.navigate(p); },
      setGoal(t) { this.state.profile.goal = t.id; MLA.ui.toast('Goal set: ' + t.title, 'success'); }
    },
    template: `
    <div class="stack">
      <section-head title="Learning Tracks" sub="Pick a destination. ML Atlas works out every prerequisite and lines the concepts up in a sensible order. You can switch at any time — progress is shared." />
      <div class="grid cols-2">
        <div v-for="it in list" :key="it.t.id" class="card track-card" :class="{current: state.profile.goal===it.t.id}">
          <div class="card-title"><h3>{{ it.t.icon }} {{ it.t.title }}</h3><span v-if="state.profile.goal===it.t.id" class="chip accent tiny">your goal</span></div>
          <p class="small muted">{{ it.t.desc }}</p>
          <div class="tiny dim mb">{{ it.pr.total }} concepts · about {{ Math.round(it.pr.ids.reduce((s,id)=>s+P.concept(id).minutes,0)/60) }} hours of lessons{{ it.t.level ? ' · ' + it.t.level : '' }}</div>
          <mla-progress :value="it.pr.pct" :label="it.pr.done + '/' + it.pr.total + ' completed'" />
          <div class="btn-row mt"><button class="btn sm" @click="go('/track/'+it.t.id)">View path</button><button v-if="state.profile.goal!==it.t.id" class="btn sm primary" @click="setGoal(it.t)">Set as my goal</button></div>
        </div>
      </div>
    </div>`
  };

  MLA.components['view-track'] = {
    props: { route: Object },
    data() { return { state: MLA.store.state }; },
    computed: {
      P() { return MLA.progress; },
      t() { return this.P.track(this.route.params.id); },
      pr() { return this.t ? this.P.trackProgress(this.t) : null; },
      steps() { return this.pr.ids.map((id, i) => ({ c: this.P.concept(id), n: i + 1, status: this.P.status(id), cp: this.P.conceptProgress(id) })); },
      nextStep() { return this.steps.find(s => !['completed', 'mastered'].includes(s.status)); },
      isGoal() { return this.state.profile.goal === (this.t && this.t.id); }
    },
    methods: {
      go(p) { MLA.router.navigate(p); },
      toggleGoal() { this.state.profile.goal = this.isGoal ? null : this.t.id; }
    },
    template: `
    <div v-if="!t"><empty-state text="Track not found."><button class="btn mt" @click="go('/tracks')">All tracks</button></empty-state></div>
    <div v-else class="stack">
      <button class="btn ghost sm" style="align-self:flex-start" @click="go('/tracks')"><mla-icon name="arrowLeft" :size="16"/> All tracks</button>
      <section-head :title="(t.icon || '') + ' ' + t.title" :sub="t.desc">
        <button class="btn" :class="isGoal ? '' : 'primary'" @click="toggleGoal">{{ isGoal ? 'Clear goal' : 'Set as my goal' }}</button>
        <button v-if="nextStep" class="btn primary" @click="go('/concept/'+nextStep.c.id)"><mla-icon name="play" :size="16"/> {{ pr.done ? 'Continue' : 'Start' }}</button>
      </section-head>
      <mla-progress :value="pr.pct" :label="pr.done + ' of ' + pr.total + ' concepts completed'" />
      <div class="track-steps">
        <a v-for="s in steps" :key="s.c.id" class="track-step" :class="s.status" @click="go('/concept/'+s.c.id)" :style="{'--bc': P.branch(s.c.branch).color}">
          <span class="ts-num">{{ s.n }}</span>
          <status-dot :status="s.status"/>
          <div class="ts-body"><div class="ts-title">{{ s.c.title }}</div><div class="tiny dim">{{ s.c.summary }}</div></div>
          <div class="ts-right"><branch-chip :id="s.c.branch" small/><span class="tiny dim">{{ s.c.minutes }} min</span></div>
        </a>
      </div>
    </div>`
  };

  // ---------- Glossary ----------
  MLA.components['view-glossary'] = {
    data() { return { q: '', cat: 'all' }; },
    computed: {
      P() { return MLA.progress; },
      cats() { return MLA.data.branches.filter(b => (MLA.data.glossary || []).some(g => g.branch === b.id)); },
      items() {
        const q = this.q.trim().toLowerCase();
        return (MLA.data.glossary || []).filter(g => (this.cat === 'all' || g.branch === this.cat) && (!q || g.term.toLowerCase().includes(q) || g.def.toLowerCase().includes(q))).slice().sort((a, b) => a.term.localeCompare(b.term, undefined, { sensitivity: 'base' }));
      }
    },
    methods: { go(p) { MLA.router.navigate(p); }, cname(id) { const c = this.P.concept(id); return c ? c.title : ''; } },
    template: `
    <div class="stack">
      <section-head :title="'Glossary'" :sub="(MLA.data.glossary||[]).length + ' terms, each linked to the concept that teaches it. These also power the flashcards.'" />
      <div class="filters">
        <div class="map-search grow"><mla-icon name="search" :size="15"/><input type="search" v-model="q" placeholder="Search terms…" aria-label="Search glossary"></div>
        <select v-model="cat" aria-label="Branch"><option value="all">All branches</option><option v-for="b in cats" :key="b.id" :value="b.id">{{ b.name }}</option></select>
      </div>
      <div class="glossary">
        <div v-for="g in items" :key="g.term" class="gl-item" :style="{'--bc': P.branch(g.branch) ? P.branch(g.branch).color : 'var(--border)'}">
          <div class="gl-term">{{ g.term }}</div>
          <div class="gl-def"><mla-md :src="g.def" inline /></div>
          <a v-if="g.concept && P.concept(g.concept)" class="tiny gl-link" @click="go('/concept/'+g.concept)">→ {{ cname(g.concept) }}</a>
        </div>
        <empty-state v-if="!items.length" icon="search" text="No terms match." />
      </div>
    </div>`
  };

  // ---------- Flashcards ----------
  MLA.components['view-review'] = {
    data() { return { state: MLA.store.state, deck: 'all', queue: [], idx: 0, flipped: false, done: 0, started: false }; },
    computed: {
      P() { return MLA.progress; },
      cards() { return MLA.data.flashcards || []; },
      decks() { return [{ id: 'all', name: 'All cards' }].concat(MLA.data.branches.filter(b => this.cards.some(c => c.deck === b.id)).map(b => ({ id: b.id, name: b.short || b.name, color: b.color }))); },
      deckCards() { return this.deck === 'all' ? this.cards : this.cards.filter(c => c.deck === this.deck); },
      stats() { return MLA.srs.stats(this.deckCards, this.state.flashcards); },
      card() { return this.queue[this.idx] || null; },
      ivals() { return this.card ? MLA.srs.nextIntervals(this.state.flashcards[this.card.id]) : null; }
    },
    watch: { deck() { this.started = false; } },
    methods: {
      go(p) { MLA.router.navigate(p); },
      start() {
        const due = MLA.util.shuffle(MLA.srs.dueCards(this.deckCards, this.state.flashcards));
        const fresh = MLA.srs.newCards(this.deckCards, this.state.flashcards).slice(0, 10);
        this.queue = due.slice(0, 30).concat(fresh); this.idx = 0; this.flipped = false; this.done = 0; this.started = true;
        if (!this.queue.length) MLA.ui.toast('Nothing due in this deck — come back tomorrow, or pick another deck.', 'info');
      },
      grade(g) {
        const c = this.card; if (!c) return;
        this.state.flashcards[c.id] = MLA.srs.review(this.state.flashcards[c.id], g);
        this.state.stats.flashReviews = (this.state.stats.flashReviews || 0) + 1;
        if (g === 'again') this.queue.push(c);
        this.idx++; this.flipped = false; this.done++;
        MLA.gamification.touchStreak();
        if (this.done % 10 === 0) MLA.gamification.checkAchievements();
      },
      fmt(d) { return d === 0 ? 'now' : d === 1 ? '1 day' : d + ' days'; },
      key(e) { if (!this.started || !this.card) return; if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); this.flipped = !this.flipped; } else if (this.flipped && ['1', '2', '3', '4'].includes(e.key)) this.grade(['again', 'hard', 'good', 'easy'][Number(e.key) - 1]); }
    },
    mounted() { this._k = e => this.key(e); document.addEventListener('keydown', this._k); },
    unmounted() { document.removeEventListener('keydown', this._k); },
    template: `
    <div class="stack">
      <section-head title="Flashcards" sub="Spaced repetition: cards you find easy come back rarely, cards you miss come back soon. Ten minutes a day locks in the vocabulary." />
      <div class="pills"><button v-for="d in decks" :key="d.id" :class="{active:deck===d.id}" @click="deck=d.id">{{ d.name }}</button></div>
      <div class="grid cols-4">
        <div class="stat"><div class="label">Due now</div><div class="value">{{ stats.due }}</div></div>
        <div class="stat"><div class="label">New</div><div class="value">{{ stats.fresh }}</div></div>
        <div class="stat"><div class="label">Learned</div><div class="value">{{ stats.learned }}</div></div>
        <div class="stat"><div class="label">Mastered (21+ days)</div><div class="value">{{ stats.mastered }}</div></div>
      </div>
      <div v-if="!started || !card" class="card center">
        <p v-if="started && !card" style="font-size:1.1rem">Session complete — {{ done }} review{{ done===1?'':'s' }} done. 🎉</p>
        <p class="muted small">A session shows every due card plus up to 10 new ones.</p>
        <button class="btn primary" @click="start"><mla-icon name="play" :size="16"/> {{ started ? 'Start another session' : 'Start reviewing' }}</button>
      </div>
      <template v-else>
        <div class="row between small muted"><span>Card {{ idx+1 }} of {{ queue.length }}</span><span>space = flip · 1–4 = grade</span></div>
        <div class="flashcard" @click="flipped=!flipped" tabindex="0">
          <span class="fc-label">{{ flipped ? 'Definition' : 'Term' }} · {{ P.branch(card.deck) ? P.branch(card.deck).short : '' }}</span>
          <div v-if="!flipped" class="fc-front">{{ card.front }}</div>
          <div v-else class="fc-back"><mla-md :src="card.back" /><a v-if="card.concept && P.concept(card.concept)" class="tiny" @click.stop="go('/concept/'+card.concept)">Open the lesson: {{ P.concept(card.concept).title }} →</a></div>
          <span v-if="!flipped" class="tiny dim fc-hint">Recall it, then click to check</span>
        </div>
        <div v-if="flipped" class="grade-row">
          <button class="btn danger" @click="grade('again')">Again<span class="tiny">{{ fmt(ivals.again) }}</span></button>
          <button class="btn" @click="grade('hard')">Hard<span class="tiny">{{ fmt(ivals.hard) }}</span></button>
          <button class="btn primary" @click="grade('good')">Good<span class="tiny">{{ fmt(ivals.good) }}</span></button>
          <button class="btn success" @click="grade('easy')">Easy<span class="tiny">{{ fmt(ivals.easy) }}</span></button>
        </div>
      </template>
    </div>`
  };

  // ---------- Achievements ----------
  MLA.components['view-achievements'] = {
    data() { return { state: MLA.store.state }; },
    computed: {
      list() { return (MLA.data.achievements || []).map(a => ({ a, at: this.state.achievements[a.id], pr: MLA.gamification.achievementProgress(a) })); },
      earned() { return this.list.filter(x => x.at).length; }
    },
    methods: { date(d) { return MLA.util.fmtDate(d); } },
    template: `
    <div class="stack">
      <section-head title="Achievements" :sub="earned + ' of ' + list.length + ' unlocked'" />
      <div class="grid auto">
        <div v-for="x in list" :key="x.a.id" class="ach" :class="x.at ? 'earned' : 'locked'">
          <div class="ach-icon">{{ x.a.icon }}</div>
          <div style="flex:1;min-width:0">
            <div class="row between"><strong>{{ x.a.title }}</strong><span v-if="x.a.xp" class="tiny dim">+{{ x.a.xp }} XP</span></div>
            <div class="small muted">{{ x.a.desc }}</div>
            <div v-if="x.at" class="tiny ok-text">Unlocked {{ date(x.at) }}</div>
            <mla-progress v-else :value="x.pr.pct" thin :label="x.pr.done + ' / ' + x.pr.total" />
          </div>
        </div>
      </div>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
