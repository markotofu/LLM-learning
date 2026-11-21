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
      <section-head title="Learning Tracks" sub="Pick a destination. ML Atlas works out every prerequisite and lines the concepts up in a sensible order. You can switch at any time; progress is shared." />
      <div class="grid cols-2">
        <div v-for="it in list" :key="it.t.id" class="card track-card" :class="{current: state.profile.goal===it.t.id}">
          <div class="card-title"><h3><mla-icon :name="it.t.icon" :size="18"/> {{ it.t.title }}</h3><span v-if="state.profile.goal===it.t.id" class="chip accent tiny">your goal</span></div>
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
      <section-head :title="t.title" :sub="t.desc">
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

  // ---------- Projects and case studies (js/data/projects.js) ----------
  MLA.components['view-projects'] = {
    props: { route: Object },
    data() { const t = this.route && this.route.query && this.route.query.tab; return { tab: t === 'cases' ? 'cases' : 'projects', q: '', level: 'all', cat: 'all' }; },
    computed: {
      cats() { return MLA.data.branches.filter(b => MLA.data.concepts.some(c => c.branch === b.id && (c.projects.length || c.cases.length))); },
      all() {
        const out = { projects: [], cases: [] };
        for (const c of MLA.data.concepts) {
          c.projects.forEach((p, i) => out.projects.push({ key: c.id + '-p' + i, p, c }));
          c.cases.forEach((k, i) => out.cases.push({ key: c.id + '-c' + i, k, c }));
        }
        return out;
      },
      items() {
        const q = this.q.trim().toLowerCase();
        const keep = (c, words) => (this.cat === 'all' || c.branch === this.cat) && (!q || words.toLowerCase().includes(q));
        if (this.tab === 'cases') return this.all.cases.filter(x => keep(x.c, [x.k.title, x.k.when, x.k.story, x.c.title].join(' ')));
        return this.all.projects.filter(x => (this.level === 'all' || x.p.level === this.level) && keep(x.c, [x.p.title, x.p.goal, x.p.data, x.p.skills.join(' '), x.c.title].join(' ')));
      }
    },
    template: `
    <div class="stack">
      <section-head title="Projects and case studies" :sub="all.projects.length + ' projects to build on your own machine with real data, and ' + all.cases.length + ' documented cases of machine learning in the real world. Each one belongs to a lesson.'" />
      <div class="pills"><button :class="{active: tab==='projects'}" @click="tab='projects'"><mla-icon name="flask" :size="14"/> Sample projects</button><button :class="{active: tab==='cases'}" @click="tab='cases'"><mla-icon name="book" :size="14"/> Case studies</button></div>
      <div class="filters">
        <div class="map-search grow"><mla-icon name="search" :size="15"/><input type="search" v-model="q" :placeholder="tab==='cases' ? 'Search case studies…' : 'Search projects, datasets, skills…'" aria-label="Search"></div>
        <select v-if="tab==='projects'" v-model="level" aria-label="Level"><option value="all">All levels</option><option value="starter">Starter</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option></select>
        <select v-model="cat" aria-label="Branch"><option value="all">All branches</option><option v-for="b in cats" :key="b.id" :value="b.id">{{ b.name }}</option></select>
      </div>
      <div v-if="tab==='projects'" class="stack project-list">
        <project-card v-for="x in items" :key="x.key" :p="x.p" :lesson="x.c" />
      </div>
      <div v-else class="stack">
        <div v-for="x in items" :key="x.key" class="card"><case-study :k="x.k" :lesson="x.c" /></div>
      </div>
      <empty-state v-if="!items.length" icon="search" text="Nothing matches these filters." />
    </div>`
  };

  // ---------- Flashcards ----------
  MLA.components['view-review'] = {
    props: { route: Object },
    data() { return { state: MLA.store.state, tab: (this.route && this.route.query && this.route.query.tab === 'mistakes') ? 'mistakes' : 'cards', deck: 'all', queue: [], idx: 0, flipped: false, done: 0, started: false }; },
    computed: {
      P() { return MLA.progress; },
      cards() { return MLA.data.flashcards || []; },
      decks() { return [{ id: 'all', name: 'All cards' }].concat(MLA.data.branches.filter(b => this.cards.some(c => c.deck === b.id)).map(b => ({ id: b.id, name: b.short || b.name, color: b.color }))); },
      deckCards() { return this.deck === 'all' ? this.cards : this.cards.filter(c => c.deck === this.deck); },
      stats() { return MLA.srs.stats(this.deckCards, this.state.flashcards); },
      card() { return this.queue[this.idx] || null; },
      ivals() { return this.card ? MLA.srs.nextIntervals(this.state.flashcards[this.card.id]) : null; },
      mstats() { return MLA.mistakes.stats(); }
    },
    watch: { deck() { this.started = false; } },
    methods: {
      go(p) { MLA.router.navigate(p); },
      start() {
        const due = MLA.util.shuffle(MLA.srs.dueCards(this.deckCards, this.state.flashcards));
        const fresh = MLA.srs.newCards(this.deckCards, this.state.flashcards).slice(0, 10);
        this.queue = due.slice(0, 30).concat(fresh); this.idx = 0; this.flipped = false; this.done = 0; this.started = true;
        if (!this.queue.length) MLA.ui.toast('Nothing due in this deck: come back tomorrow, or pick another deck.', 'info');
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
      key(e) { if (this.tab !== 'cards' || !this.started || !this.card) return; if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); this.flipped = !this.flipped; } else if (this.flipped && ['1', '2', '3', '4'].includes(e.key)) this.grade(['again', 'hard', 'good', 'easy'][Number(e.key) - 1]); }
    },
    mounted() { this._k = e => this.key(e); document.addEventListener('keydown', this._k); },
    unmounted() { document.removeEventListener('keydown', this._k); },
    template: `
    <div class="stack">
      <section-head title="Review" :sub="tab === 'cards' ? 'Spaced repetition: cards you find easy come back rarely, cards you miss come back soon. Ten minutes a day locks in the vocabulary.' : 'Every question you got wrong and every problem you needed help with comes back here, spaced out over days and weeks, until you have got it right several times in a row.'" />
      <div class="pills"><button :class="{active: tab==='cards'}" @click="tab='cards'"><mla-icon name="cards" :size="14"/> Flashcards</button><button :class="{active: tab==='mistakes'}" @click="tab='mistakes'"><mla-icon name="refresh" :size="14"/> Mistakes to revisit<span v-if="mstats.due" class="chip tiny accent" style="margin-left:6px">{{ mstats.due }}</span></button></div>
      <review-mistakes v-if="tab==='mistakes'" />
      <template v-else>
      <div class="pills"><button v-for="d in decks" :key="d.id" :class="{active:deck===d.id}" @click="deck=d.id">{{ d.name }}</button></div>
      <div class="grid cols-4">
        <div class="stat"><div class="label">Due now</div><div class="value">{{ stats.due }}</div></div>
        <div class="stat"><div class="label">New</div><div class="value">{{ stats.fresh }}</div></div>
        <div class="stat"><div class="label">Learned</div><div class="value">{{ stats.learned }}</div></div>
        <div class="stat"><div class="label">Mastered (21+ days)</div><div class="value">{{ stats.mastered }}</div></div>
      </div>
      <div v-if="!started || !card" class="card center">
        <p v-if="started && !card" style="font-size:1.1rem">Session complete: {{ done }} review{{ done===1?'':'s' }} done.</p>
        <p class="muted small">A session shows every due card plus up to 10 new ones.</p>
        <button class="btn primary" @click="start"><mla-icon name="play" :size="16"/> {{ started ? 'Start another session' : 'Start reviewing' }}</button>
      </div>
      <template v-else>
        <div class="row between small muted"><span>Card {{ idx+1 }} of {{ queue.length }}</span><span>space = flip · 1–4 = grade</span></div>
        <div class="flashcard" role="button" @click="flipped=!flipped" tabindex="0">
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
      </template>
    </div>`
  };

  // ---------- Mistakes to revisit (a tab of the Review page) ----------
  const KIND_LABEL = { quiz: 'Quiz question', check: 'Quick check', predict: 'Predict the output', problem: 'Practice problem' };
  MLA.components['review-mistakes'] = {
    data() { return { state: MLA.store.state, queue: [], idx: 0, started: false, done: 0, fixed: 0, result: null, shown: null }; },
    computed: {
      stats() { return MLA.mistakes.stats(); },
      cur() { return this.queue[this.idx] || null; },
      waitingList() { return MLA.mistakes.all().filter(x => !x.item.graduated).sort((a, b) => a.item.card.due.localeCompare(b.item.card.due)).slice(0, 8); }
    },
    methods: {
      label(x) { return KIND_LABEL[x.item.kind] || x.item.kind; },
      when(d) { const t = MLA.util.todayKey(); if (d <= t) return 'due now'; const n = Math.round((new Date(d) - new Date(t)) / 86400000); return n === 1 ? 'tomorrow' : 'in ' + n + ' days'; },
      start() {
        const items = MLA.util.shuffle(MLA.mistakes.dueItems()).slice(0, 25);
        // Shuffle the options so the position of the right answer is not what gets remembered.
        this.queue = items.map(x => x.question ? Object.assign({}, x, { shown: MLA.quiz.shuffled(x.question, MLA.util.seededRng((Date.now() + Math.random() * 1e9) >>> 0)) }) : x);
        this.idx = 0; this.done = 0; this.fixed = 0; this.result = null; this.started = true;
      },
      answer(val) {
        if (this.result || !this.cur) return;
        const g = MLA.quiz.grade(this.cur.shown, val);
        this.result = { correct: g.correct, explanation: g.explanation, expected: g.expected };
        const it = MLA.mistakes.grade(this.cur.id, g.correct);
        if (it && it.graduated) this.fixed++;
        if (!g.correct) this.queue.push(this.cur);   // again later in this session
      },
      next() { this.idx++; this.done++; this.result = null; },
      later() { MLA.mistakes.snooze(this.cur.id, 1); this.next(); },
      go(p) { MLA.router.navigate(p); },
      lesson(x) { this.go('/concept/' + x.concept.id); }
    },
    template: `
    <div class="stack">
      <div class="grid cols-4">
        <div class="stat"><div class="label">Due now</div><div class="value">{{ stats.due }}</div></div>
        <div class="stat"><div class="label">Coming back later</div><div class="value">{{ stats.waiting }}</div></div>
        <div class="stat"><div class="label">Fixed for good</div><div class="value">{{ stats.graduated }}</div></div>
        <div class="stat"><div class="label">Collected so far</div><div class="value">{{ stats.total }}</div></div>
      </div>
      <div v-if="!started || !cur" class="card center">
        <p v-if="started && !cur" style="font-size:1.1rem">Session complete: {{ done }} revisited{{ fixed ? ', ' + fixed + ' fixed for good' : '' }}.</p>
        <template v-if="stats.due">
          <p class="muted small">{{ stats.due }} item{{ stats.due === 1 ? '' : 's' }} to revisit. Answer each one again: right answers come back after longer and longer gaps (1 day, 6 days, about 2 weeks) until they retire; a miss starts the gaps again.</p>
          <button class="btn primary" @click="start"><mla-icon name="play" :size="16"/> {{ started ? 'Start another session' : 'Start revisiting' }}</button>
        </template>
        <p v-else-if="stats.waiting" class="muted small">Nothing is due today. The next item comes back {{ when(waitingList[0].item.card.due) }}: a gap of a day or more is what makes the second attempt stick.</p>
        <p v-else class="muted small">Nothing here yet. When you miss a quiz question, a quick check or a "predict first" question, or need the solution to solve a problem, it is saved here and comes back a day or more later.</p>
      </div>
      <div v-else class="card">
        <div class="row between small muted"><span>{{ label(cur) }} · <a @click="lesson(cur)">{{ cur.concept.title }}</a></span><span>{{ idx + 1 }} of {{ queue.length }}</span></div>
        <template v-if="cur.problem">
          <h3 class="mt-sm">{{ cur.problem.title }}</h3>
          <p class="small muted">{{ cur.item.why === 'solution' ? 'You solved this one after viewing the solution.' : 'This one took several attempts.' }} Solve it again without help: open it, press <b>Reset to starter code</b>, and write the solution from memory. An accepted submission counts.</p>
          <div class="btn-row"><button class="btn primary" @click="go('/problem/' + cur.problem.id)"><mla-icon name="code" :size="15"/> Open the problem</button><button class="btn ghost" @click="later">Tomorrow</button></div>
        </template>
        <template v-else>
          <div class="q-text mt-sm"><mla-md :src="cur.shown.q" /></div>
          <check-q :key="cur.id + ':' + idx" :q="cur.shown" :done="result" @answer="answer" />
          <div v-if="result" class="feedback" :class="result.correct ? 'ok' : 'bad'"><strong>{{ result.correct ? 'Right this time.' : 'Not yet.' }}</strong> <mla-md :src="result.explanation" inline /><span v-if="!result.correct && result.expected"> Answer: <b>{{ result.expected }}</b></span></div>
          <div class="btn-row mt"><button v-if="result" class="btn primary" @click="next">Next</button><button v-if="!result" class="btn ghost" @click="later">Skip until tomorrow</button><button class="btn ghost" @click="lesson(cur)">Re-read the lesson</button></div>
        </template>
      </div>
      <div v-if="waitingList.length && (!started || !cur)" class="card">
        <h4>In the queue</h4>
        <div v-for="x in waitingList" :key="x.id" class="row between small" style="padding:4px 0"><span>{{ label(x) }} · {{ x.concept.title }}<span v-if="x.problem"> · {{ x.problem.title }}</span></span><span class="muted">{{ when(x.item.card.due) }}</span></div>
      </div>
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
          <div class="ach-icon"><mla-icon :name="x.a.icon" :size="24"/></div>
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
