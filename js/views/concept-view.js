/* ML Atlas – concept page: lesson (sections, runnable code, demos, checks), practice list, mastery quiz, notes */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  MLA.components['view-concept'] = {
    props: { route: Object },
    data() { return { state: MLA.store.state, tab: (this.route && this.route.query.tab) || 'learn', checkDone: {}, quizQs: null, preview: false }; },
    computed: {
      P() { return MLA.progress; },
      c() { return this.P.concept(this.route.params.id); },
      cs() { return (this.c && this.P.peekC(this.c.id)) || { read: [], checks: {} }; },
      pr() { return this.c ? this.P.conceptProgress(this.c.id) : null; },
      branch() { return this.c ? this.P.branch(this.c.branch) : null; },
      status() { return this.P.status(this.c.id); },
      prereqs() { return this.c.prereqs.map(this.P.concept).filter(Boolean); },
      missing() { return this.P.missingPrereqs(this.c.id); },
      next() { return MLA.content.graph.children[this.c.id].map(this.P.concept); },
      toc() { return this.c.sections.map((s, i) => ({ i, title: s.title || (s.type === 'check' ? 'Quick check' : 'Section ' + (i + 1)), type: s.type, read: (this.cs.read || []).includes(s.id) })); },
      reqText() { return this.P.requirementsText(this.c.id); },
      problems() { return this.c.problems.map(p => ({ p, status: this.P.problemStatus(p.id), preview: (p.prompt || '').split('\n').find(l => l.trim() && !/^[#|>`]/.test(l.trim())) || '' })); },
      note: { get() { return this.state.notes[this.c.id] || ''; }, set(v) { this.state.notes[this.c.id] = v; } },
      statusText() { return { mastered: 'Mastered', completed: 'Completed', started: 'In progress', ready: 'Ready to learn', locked: 'Prerequisites recommended first' }[this.status]; }
    },
    watch: {
      tab(v) { if (v === 'quiz' && !this.quizQs) this.startQuiz(); if (v === 'learn') this.$nextTick(this.observe); },
      'route.query.tab'(v) { if (v) this.tab = v; }
    },
    created() { if (this.c) { this.P.visit(this.c.id); if (this.tab === 'quiz') this.startQuiz(); } },
    mounted() { this.$nextTick(this.observe); },
    unmounted() { if (this._io) this._io.disconnect(); },
    methods: {
      go(p) { MLA.router.navigate(p); },
      isRead(sid) { return (this.cs.read || []).includes(sid); },
      isRunnable(s) { return s.run !== false && (!s.lang || s.lang === 'python'); },
      secIcon(s) { if (s.type === 'demo' && String(s.demo).startsWith('dg-')) return 'layers'; return { text: 'book', code: 'code', demo: 'play', check: 'hint', chart: 'graph' }[s.type] || 'book'; },
      // Sections count as read once you reach their end (a sentinel scrolls into view).
      observe() {
        if (!this.c || typeof IntersectionObserver === 'undefined' || !this.$el || !this.$el.querySelectorAll) return;
        if (!this._io) {
          this._timers = {};
          this._io = new IntersectionObserver(entries => {
            for (const e of entries) {
              const sid = e.target.getAttribute('data-sid');
              if (e.isIntersecting) { this._timers[sid] = setTimeout(() => { this.P.markRead(this.c.id, sid); if (this._io) this._io.unobserve(e.target); }, 600); }
              else clearTimeout(this._timers[sid]);
            }
          }, { threshold: 0 });
        }
        this.$el.querySelectorAll('.sec-end').forEach(el => { if (!this.isRead(el.getAttribute('data-sid'))) this._io.observe(el); });
      },
      markAll() { this.P.markAllRead(this.c.id); MLA.ui.toast('All sections marked as read.', 'info'); },
      scrollTo(i) { const el = document.getElementById('sec-' + i); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); },
      answerCheck(i, q, val) {
        if (this.checkDone[i]) return;
        const g = MLA.quiz.grade(q, val);
        this.checkDone[i] = { correct: g.correct, explanation: g.explanation, expected: g.expected };
        this.P.recordCheck(this.c.id, this.c.sections[i].id, g.correct);
      },
      answerPredict(i, s, val) {
        const key = 'p' + i; if (this.checkDone[key]) return;
        const g = MLA.quiz.grade(s.predict, val);
        this.checkDone[key] = { correct: g.correct, explanation: g.explanation, expected: g.expected };
        this.P.recordCheck(this.c.id, 'predict-' + s.id, g.correct);
      },
      startQuiz() { const att = (this.P.peekC(this.c.id) || {}).quizAttempts || 0; this.quizQs = MLA.quiz.prepare(this.c.quiz, MLA.util.seededRng(MLA.util.hashString(this.c.id) + att * 7919)); },
      onQuiz(result) {
        const passed = this.P.recordQuiz(this.c.id, result);
        if (passed) MLA.ui.toast('Quiz passed! ' + (this.pr.solved < this.pr.needSolved ? 'Solve a practice problem to complete the concept.' : ''), 'success', 4500);
      },
      retryQuiz() { this.quizQs = MLA.quiz.prepare(this.c.quiz, MLA.util.seededRng(Date.now() >>> 0)); },
      openProblem(p) { this.go('/problem/' + p.id); }
    },
    template: `
    <div v-if="!c"><empty-state text="Concept not found."><button class="btn primary mt" @click="go('/map')">Open the concept map</button></empty-state></div>
    <div v-else class="stack concept-page" :style="{'--bc': branch.color}">
      <div class="row between">
        <button class="btn ghost sm" @click="go('/map?focus='+c.id)"><mla-icon name="map" :size="16"/> Concept map</button>
        <div class="row" style="gap:6px"><branch-chip :id="c.branch"/><level-badge :level="c.level"/><span class="chip"><mla-icon name="clock" :size="13"/> {{ c.minutes }} min</span></div>
      </div>

      <div class="concept-hero">
        <div class="row" style="gap:10px;align-items:center"><status-dot :status="status" :size="26"/><h2 style="margin:0">{{ c.title }}</h2></div>
        <p class="muted" style="margin:6px 0 10px">{{ c.summary }}</p>
        <mla-progress :value="pr.pct" :color="branch.color" :label="statusText + (pr.completed ? '' : ' · ' + pr.pct + '% explored')" />
        <div v-if="reqText && !pr.completed" class="tiny dim mt-sm">{{ reqText }}</div>
      </div>

      <div class="tabs">
        <button :class="{active:tab==='learn'}" @click="tab='learn'"><mla-icon name="book" :size="15"/> Learn</button>
        <button :class="{active:tab==='practice'}" @click="tab='practice'"><mla-icon name="code" :size="15"/> Practice <span class="tab-count">{{ pr.solved }}/{{ pr.problems }}</span></button>
        <button :class="{active:tab==='quiz'}" @click="tab='quiz'"><mla-icon name="clipboard" :size="15"/> Quiz <span v-if="pr.quizPassed" class="ok-text">✓</span></button>
        <button :class="{active:tab==='notes'}" @click="tab='notes'"><mla-icon name="note" :size="15"/> My notes</button>
      </div>

      <template v-if="tab==='learn'">
        <div class="lesson-layout">
          <div class="lesson-main">
            <div v-if="missing.length" class="callout warn">
              <div class="callout-title">Recommended first</div>
              <p style="margin-bottom:6px">This concept builds on ideas you have not completed yet. You can read on, but these will make it much easier:</p>
              <div class="chips"><a v-for="m in missing" :key="m.id" class="chip clickable" @click="go('/concept/'+m.id)"><status-dot :status="P.status(m.id)" :size="15"/> {{ m.title }}</a></div>
            </div>
            <div v-else-if="prereqs.length" class="builds-on small"><span class="muted">Builds on:</span> <a v-for="m in prereqs" :key="m.id" class="chip clickable" @click="go('/concept/'+m.id)"><status-dot :status="P.status(m.id)" :size="15"/> {{ m.title }}</a></div>

            <div v-if="c.why" class="lesson-section why-card">
              <div class="sec-head"><mla-icon name="sparkles" :size="18"/><h3>Why this matters</h3></div>
              <mla-md :src="c.why" />
            </div>
            <div v-if="c.objectives.length" class="lesson-section">
              <div class="sec-head"><mla-icon name="target" :size="18"/><h3>By the end you will be able to</h3></div>
              <ul class="objectives"><li v-for="(o,i) in c.objectives" :key="i"><mla-md :src="o" inline /></li></ul>
            </div>

            <section v-for="(s,i) in c.sections" :key="i" :id="'sec-'+i" class="lesson-section" :class="['sec-'+s.type, {read: isRead(s.id)}]">
              <div class="sec-head">
                <mla-icon :name="secIcon(s)" :size="17"/>
                <h3>{{ s.title || (s.type==='check' ? 'Quick check' : 'Section ' + (i+1)) }}</h3>
                <span class="spacer"></span>
                <span v-if="isRead(s.id)" class="chip green tiny"><mla-icon name="check" :size="11"/> read</span>
              </div>
              <template v-if="s.type==='text'"><mla-md :src="s.md" /></template>
              <template v-else-if="s.type==='code'">
                <mla-md v-if="s.md" :src="s.md" />
                <div v-if="s.predict" class="predict-box">
                  <div class="predict-head"><mla-icon name="target" :size="15"/> <b>Predict first</b> <span class="small muted">commit to an answer, then run the code to check it</span></div>
                  <div class="q-text"><mla-md :src="s.predict.q" /></div>
                  <check-q :q="s.predict" :done="checkDone['p' + i]" @answer="answerPredict(i, s, $event)" />
                  <div v-if="checkDone['p' + i]" class="feedback" :class="checkDone['p' + i].correct ? 'ok' : 'bad'"><strong>{{ checkDone['p' + i].correct ? 'Good prediction!' : 'Not what happens.' }}</strong> <mla-md :src="checkDone['p' + i].explanation" inline /> Run the code below to see it for yourself.</div>
                  <div v-else-if="cs.checks && cs.checks['predict-' + s.id] != null" class="small muted mt-sm">You made this prediction before{{ cs.checks['predict-' + s.id] ? ', correctly ✓' : '' }}.</div>
                </div>
                <code-sample :code="s.code" :lang="s.lang || 'python'" :runnable="isRunnable(s)" :title="c.title + ' – ' + (s.title || 'example')" :timeout-sec="s.timeoutSec" />
                <div v-if="s.lang==='pytorch'" class="tiny dim mt-sm"><mla-icon name="info" :size="12"/> PyTorch is not available in the in-browser Python, so this example is for reading (run it locally or in Google Colab). The practice problems implement the same ideas in NumPy.</div>
                <mla-md v-if="s.after" :src="s.after" />
              </template>
              <template v-else-if="s.type==='demo'">
                <mla-md v-if="s.md" :src="s.md" />
                <mla-demo :name="s.demo" />
                <mla-md v-if="s.after" :src="s.after" />
              </template>
              <template v-else-if="s.type==='chart'">
                <mla-md v-if="s.md" :src="s.md" />
                <mla-chart :spec="s.chart" :title="s.title" />
                <mla-md v-if="s.after" :src="s.after" />
              </template>
              <template v-else-if="s.type==='check'">
                <div class="q-text"><mla-md :src="s.question.q" /></div>
                <check-q :q="s.question" :done="checkDone[i]" @answer="answerCheck(i, s.question, $event)" />
                <div v-if="checkDone[i]" class="feedback" :class="checkDone[i].correct ? 'ok' : 'bad'"><strong>{{ checkDone[i].correct ? 'Correct!' : 'Not quite.' }}</strong> <mla-md :src="checkDone[i].explanation" inline /><span v-if="!checkDone[i].correct && checkDone[i].expected"> Answer: <b>{{ checkDone[i].expected }}</b></span></div>
                <div v-else-if="cs.checks && cs.checks[s.id] != null" class="tiny dim mt-sm">You answered this before{{ cs.checks[s.id] ? ' correctly ✓' : '' }}.</div>
              </template>
              <div v-if="s.type!=='check'" class="sec-end" :data-sid="s.id"></div>
            </section>

            <div v-if="c.when" class="lesson-section when-card">
              <div class="sec-head"><mla-icon name="route" :size="18"/><h3>When to use it (and when not to)</h3></div>
              <mla-md :src="c.when" />
            </div>
            <div v-if="c.pitfalls" class="lesson-section pitfalls-card">
              <div class="sec-head"><mla-icon name="warning" :size="18"/><h3>Common pitfalls</h3></div>
              <mla-md :src="c.pitfalls" />
            </div>
            <div v-if="c.takeaways.length" class="lesson-section takeaways-card">
              <div class="sec-head"><mla-icon name="star" :size="18"/><h3>Key takeaways</h3></div>
              <ul class="takeaways"><li v-for="(t,i) in c.takeaways" :key="i"><mla-md :src="t" inline /></li></ul>
            </div>
            <div v-if="c.refs && c.refs.length" class="lesson-section">
              <div class="sec-head"><mla-icon name="external" :size="16"/><h3>Go deeper</h3></div>
              <ul><li v-for="r in c.refs" :key="r.url"><a :href="r.url" target="_blank" rel="noopener">{{ r.title }}</a><span v-if="r.note" class="tiny dim"> — {{ r.note }}</span></li></ul>
            </div>

            <div class="card next-card">
              <h3 style="margin-top:0">Prove it</h3>
              <p class="small muted">Understanding shows up when you can use it. {{ reqText || 'This concept is complete — the remaining problems and a 90%+ quiz score earn mastery.' }}</p>
              <div class="btn-row">
                <button v-if="c.problems.length" class="btn primary" @click="tab='practice'"><mla-icon name="code" :size="16"/> Practice problems ({{ pr.solved }}/{{ pr.problems }})</button>
                <button v-if="c.quiz.length" class="btn" @click="tab='quiz'"><mla-icon name="clipboard" :size="16"/> {{ pr.quizPassed ? 'Retake the quiz' : 'Take the quiz' }}</button>
              </div>
              <div v-if="next.length" class="mt">
                <div class="tiny dim upper">This unlocks</div>
                <div class="chips"><a v-for="n in next" :key="n.id" class="chip clickable" @click="go('/concept/'+n.id)"><status-dot :status="P.status(n.id)" :size="15"/> {{ n.title }}</a></div>
              </div>
            </div>
          </div>

          <aside class="lesson-side">
            <div class="card pad-sm">
              <div class="tiny dim upper" style="margin-bottom:6px">In this lesson</div>
              <div class="lesson-toc">
                <a v-for="t in toc" :key="t.i" :class="{read:t.read, check: t.type==='check'}" @click="scrollTo(t.i)"><span class="toc-mark">{{ t.read ? '✓' : (t.type==='check' ? '?' : '•') }}</span>{{ t.title }}</a>
              </div>
              <hr>
              <div class="small">
                <div class="row between"><span class="muted">Sections read</span><span>{{ pr.sectionsRead }}/{{ pr.sections }}</span></div>
                <div class="row between"><span class="muted">Quiz</span><span>{{ pr.quizPassed ? 'passed (' + pr.quizBest + '%)' : (pr.quizAttempts ? 'best ' + pr.quizBest + '%' : '—') }}</span></div>
                <div class="row between"><span class="muted">Problems</span><span>{{ pr.solved }}/{{ pr.problems }}</span></div>
              </div>
              <button v-if="pr.sectionsRead < pr.sections" class="btn xs ghost block mt-sm" @click="markAll">Mark all sections read</button>
            </div>
          </aside>
        </div>
      </template>

      <template v-else-if="tab==='practice'">
        <p class="small muted">Write real Python in the browser. <b>Run</b> checks your code against the visible examples; <b>Submit</b> also runs hidden tests. Solving at least one problem (plus the quiz) completes this concept.</p>
        <div class="problem-cards">
          <div v-for="it in problems" :key="it.p.id" class="card clickable problem-card" @click="openProblem(it.p)" tabindex="0" @keydown.enter="openProblem(it.p)">
            <div class="row between"><div class="row" style="gap:8px"><status-dot :status="it.status"/><strong>{{ it.p.title }}</strong></div><diff-chip :d="it.p.difficulty"/></div>
            <p class="small muted" style="margin:8px 0 0"><mla-md :src="it.preview" inline /></p>
          </div>
        </div>
        <empty-state v-if="!problems.length" icon="code" text="This concept has no coding problems — the quiz completes it." />
      </template>

      <template v-else-if="tab==='quiz'">
        <div class="card">
          <div class="row between mb"><h3 style="margin:0">Mastery quiz — {{ c.title }}</h3><span class="muted small">Pass: {{ P.PASS }}% · best {{ pr.quizBest }}%</span></div>
          <quiz-runner v-if="quizQs && quizQs.length" :key="(quizQs[0] && quizQs[0].q) + pr.quizAttempts" :questions="quizQs" :pass="P.PASS" @complete="onQuiz" @retry="retryQuiz" @close="tab='learn'" />
          <empty-state v-else icon="clipboard" text="No quiz for this concept." />
        </div>
      </template>

      <template v-else-if="tab==='notes'">
        <div class="card">
          <div class="row between mb"><h3 style="margin:0">My notes</h3><div class="pills"><button :class="{active:!preview}" @click="preview=false">Write</button><button :class="{active:preview}" @click="preview=true">Preview</button></div></div>
          <p class="small muted">Explaining an idea in your own words is one of the best ways to learn it. Notes save automatically, support Markdown, and are included when you export your progress.</p>
          <textarea v-if="!preview" v-model="note" class="notes-area" placeholder="e.g. In my own words: gradient descent repeatedly nudges the weights downhill…"></textarea>
          <div v-else class="notes-preview"><mla-md :src="note || '*Nothing yet.*'" /></div>
        </div>
      </template>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
