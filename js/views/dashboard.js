/* Dashboard */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  MLA.components['view-home'] = {
    data() { return { state: MLA.store.state, potd: null }; },
    created() { this.potd = MLA.progress.problemOfTheDay(); },
    computed: {
      P() { return MLA.progress; }, G() { return MLA.gamification; },
      level() { return this.G.levelInfo(); },
      streak() { return this.G.streakStatus(); },
      overall() { return this.P.overall(); },
      rec() { return this.P.recommendNext(); },
      recPr() { return this.rec.concept ? this.P.conceptProgress(this.rec.concept.id) : null; },
      ready() { return this.P.readyList(7).filter(c => !this.rec.concept || c.id !== this.rec.concept.id).slice(0, 5); },
      branches() { return this.P.branchStats().filter(b => b.branch.id !== 'start'); },
      goal() { return this.state.profile.goal ? this.P.track(this.state.profile.goal) : null; },
      goalPr() { return this.goal ? this.P.trackProgress(this.goal) : null; },
      activity() { return (this.state.activity || []).slice(0, 7); },
      days() { const out = []; for (let i = 13; i >= 0; i--) { const k = MLA.util.addDays(MLA.util.todayKey(), -i); out.push({ on: !!this.streak.days[k], today: i === 0, k }); } return out; },
      due() { const cards = MLA.data.flashcards || []; return MLA.srs.stats(cards, this.state.flashcards); },
      mistakes() { return MLA.mistakes.stats(); },
      py() { return MLA.python.status; },
      fresh() { return this.overall.conceptsDone === 0 && this.overall.solved === 0; },
      placementBusy() { const pl = this.state.placement; return !!(pl && pl.current && pl.current.answers.length); }
    },
    methods: {
      go(p) { MLA.router.navigate(p); },
      greeting() { const h = new Date().getHours(); return (h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening') + ', ' + (this.state.profile.name || 'Learner'); },
      cont() { if (this.rec.concept) this.go('/concept/' + this.rec.concept.id); else this.go('/map'); },
      ago(t) { return MLA.util.timeAgo(t); },
      loadPy() { MLA.python.warmup(); }
    },
    template: `
    <div class="stack">
      <section-head class="fresco-banner" :title="greeting()" :sub="'Level ' + level.level + ' · ' + G.title(level.level) + ' · ' + level.into + '/' + level.needed + ' XP to the next level'">
        <button class="btn" @click="go('/map')"><mla-icon name="map" :size="16"/> Concept map</button>
        <button class="btn primary" @click="cont"><mla-icon name="play" :size="16"/> {{ fresh ? 'Start learning' : 'Continue' }}</button>
      </section-head>

      <div v-if="fresh" class="card hero-card">
        <div class="hero-text">
          <h3 style="margin-top:0">Welcome to {{ MLA.course.name }}</h3>
          <p>This is a <b>map</b> of {{ MLA.course.subject }}: {{ P.concepts().length }} concepts. Each concept is a short lesson with runnable code, an interactive demo or two, a quiz, and coding problems you solve right here in the browser.</p>
          <ol class="small"><li>Open the <a @click.prevent="go('/map')" href="#/map">concept map</a> and start at the centre, or pick a <a @click.prevent="go('/tracks')" href="#/tracks">learning track</a>.</li><li>Read a lesson, run the examples, play with the demos.</li><li>Pass the quiz and solve a problem to complete the concept and unlock what builds on it.</li></ol>
          <div class="btn-row"><button class="btn primary" @click="cont"><mla-icon name="play" :size="16"/> Start with “{{ rec.concept ? rec.concept.title : 'the map' }}”</button><button class="btn" @click="go('/tracks')"><mla-icon name="route" :size="16"/> Choose a track</button></div>
          <p v-if="overall.known" class="small mt-sm placement-pitch">{{ overall.known }} {{ overall.known === 1 ? 'lesson is' : 'lessons are' }} marked as known, so your path skips {{ overall.known === 1 ? 'it' : 'them' }}. <a href="#/placement" @click.prevent="go('/placement')">Review or undo the marks</a></p>
          <p v-else class="small muted mt-sm placement-pitch">Know some of this already? <button class="btn sm" @click="go('/placement')"><mla-icon name="target" :size="14"/> {{ placementBusy ? 'Resume the placement test' : 'Take the placement test' }}</button> In 5 to 10 minutes it marks the lessons you know, so your path starts at the first gap.</p>
        </div>
      </div>

      <div class="grid cols-4">
        <stat-card label="Experience" :value="state.xp.toLocaleString() + ' XP'" :sub="'Level ' + level.level" icon="star" color="var(--amber)" />
        <stat-card label="Streak" :value="streak.current + (streak.current===1?' day':' days')" :sub="'Best ' + streak.best" icon="flame" color="var(--red)" />
        <stat-card label="Concepts" :value="overall.conceptsDone + ' / ' + overall.concepts" :sub="'completed' + (overall.known ? ' · ' + overall.known + ' known' : '')" icon="map" color="var(--accent)" />
        <stat-card label="Problems" :value="overall.solved + ' / ' + overall.problems" :sub="'solved'" icon="code" color="var(--green)" />
      </div>

      <div class="grid cols-3">
        <div class="card clickable" style="grid-column:span 2" @click="cont">
          <div class="card-title"><h3><mla-icon name="target" :size="18"/> Up next</h3><span class="chip accent">{{ rec.reason }}</span></div>
          <div v-if="rec.concept" class="row between">
            <div><div class="up-next-title">{{ rec.concept.title }}</div><div class="small muted">{{ rec.concept.summary }}</div>
              <div class="row mt-sm" style="gap:6px"><branch-chip :id="rec.concept.branch" small/><level-badge :level="rec.concept.level"/><span class="chip tiny">{{ rec.concept.minutes }} min</span><span v-if="recPr && recPr.started" class="chip tiny">{{ recPr.pct }}% explored</span></div></div>
            <button class="btn primary">Open <mla-icon name="chevron" :size="16"/></button>
          </div>
          <p v-else class="muted">{{ rec.reason }}</p>
          <div v-if="ready.length" class="mt" @click.stop>
            <div class="tiny dim upper">Also ready for you</div>
            <div class="chips"><a v-for="c in ready" :key="c.id" class="chip clickable" :href="'#/concept/'+c.id" @click.prevent="go('/concept/'+c.id)"><status-dot :status="P.status(c.id)" :size="15"/> {{ c.title }}</a></div>
          </div>
        </div>
        <div class="card">
          <div class="card-title"><h3><mla-icon name="bolt" :size="18"/> Problem of the day</h3><span v-if="potd && potd.done" class="chip green tiny">done</span></div>
          <template v-if="potd">
            <div style="font-weight:650">{{ potd.problem.title }}</div>
            <div class="small muted">{{ P.concept(potd.problem.concept).title }}</div>
            <div class="row mt-sm"><diff-chip :d="potd.problem.difficulty"/><span class="tiny dim">+30 bonus XP</span></div>
            <button class="btn sm block mt" :class="potd.done?'':'primary'" @click="go('/problem/'+potd.problem.id)">{{ potd.done ? 'Review it' : 'Solve it' }}</button>
          </template>
          <div class="streak-days mt"><span v-for="d in days" :key="d.k" :class="{on:d.on, today:d.today}" :title="d.k"></span></div>
          <div class="tiny dim mt-sm">Last 14 days: learn something daily to grow your streak.</div>
        </div>
      </div>

      <div class="grid cols-3">
        <div class="card" style="grid-column:span 2">
          <div class="card-title"><h3>Progress by branch</h3><button class="btn sm ghost" @click="go('/map')">Open map</button></div>
          <div v-for="b in branches" :key="b.branch.id" class="branch-row" @click="go('/map')">
            <span class="branch-name"><span class="branch-dot" :style="{'--bc': b.branch.color}"></span>{{ b.branch.name }}</span>
            <mla-progress :value="b.pct" thin :color="b.branch.color" />
            <span class="tiny dim mono">{{ b.done }}/{{ b.concepts }} · {{ b.solved }}/{{ b.problems }}</span>
          </div>
          <div class="tiny dim">concepts completed · problems solved</div>
        </div>
        <div class="card">
          <div class="card-title"><h3><mla-icon name="route" :size="18"/> Your goal</h3></div>
          <template v-if="goal">
            <div style="font-weight:650">{{ goal.title }}</div>
            <mla-progress :value="goalPr.pct" :extra="goalPr.knownPct" :label="goalPr.done + ' of ' + goalPr.total + ' concepts' + (goalPr.known ? ' · ' + goalPr.known + ' known' : '')" style="margin-top:8px" />
            <button class="btn sm block mt" @click="go('/track/'+goal.id)">View the path</button>
            <div v-if="!fresh" class="tiny muted mt-sm"><a href="#/placement" @click.prevent="go('/placement')">{{ overall.known ? 'Placement test: review or undo the marks' : placementBusy ? 'Resume the placement test' : 'Take the placement test to skip what you know' }}</a></div>
          </template>
          <template v-else>
            <p class="small muted">Pick a destination and {{ MLA.course.name }} will line up exactly the concepts you need, in order.</p>
            <button class="btn sm primary block" @click="go('/tracks')">Choose a learning track</button>
          </template>
          <hr>
          <div class="row between"><span class="small"><mla-icon name="cards" :size="15"/> Flashcards due</span><b>{{ due.due }}</b></div>
          <button class="btn sm block mt-sm" @click="go('/review')">{{ due.due ? 'Review now' : 'Learn new cards' }}</button>
          <div class="row between mt"><span class="small"><mla-icon name="refresh" :size="15"/> Mistakes to revisit</span><b>{{ mistakes.due }}</b></div>
          <button v-if="mistakes.due || mistakes.waiting" class="btn sm block mt-sm" @click="go('/review?tab=mistakes')">{{ mistakes.due ? 'Revisit now' : 'Next ones come back soon' }}</button>
          <p v-else class="tiny muted mt-sm">Questions you miss and problems you need help with will come back here.</p>
        </div>
      </div>

      <div class="grid cols-2">
        <div class="card">
          <div class="card-title"><h3>Recent activity</h3></div>
          <div v-if="activity.length" class="activity">
            <div v-for="(a,i) in activity" :key="i" class="row between small act-row"><span>{{ a.label }}</span><span class="dim tiny">{{ ago(a.t) }}{{ a.xp ? ' · +' + a.xp + ' XP' : '' }}</span></div>
          </div>
          <empty-state v-else icon="clock" text="Nothing yet. Complete a quiz or solve a problem to get started." />
        </div>
        <div class="card">
          <div class="card-title"><h3><mla-icon name="cpu" :size="18"/> Python in your browser</h3><span class="chip tiny"><span class="py-dot" :class="py.state"></span> {{ py.ready ? 'ready' : py.state }}</span></div>
          <p class="small muted">Code runs locally with Pyodide (real CPython compiled to WebAssembly): nothing is sent to a server. The first run downloads about 15 MB from a CDN; after that it is cached and works offline.</p>
          <div class="btn-row"><button class="btn sm" :disabled="py.ready || py.state==='loading'" @click="loadPy">{{ py.ready ? 'Loaded' : py.state==='loading' ? (py.message || 'Loading…') : 'Load Python now' }}</button><button class="btn sm ghost" @click="go('/playground')">Open the playground</button></div>
        </div>
      </div>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
