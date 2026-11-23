/* ML Atlas – concept page: lesson (sections, runnable code, demos, checks), practice list, mastery quiz, notes */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  // The sentence being read aloud, found inside a lesson box so it can be highlighted. Narration rewrites formulas
  // and code into words, so this is best effort: it looks for the sentence's first few words in the box's text.
  function findSentence(root, sentence) {
    const words = (String(sentence).match(/[A-Za-z0-9']+/g) || []).filter(w => w.length > 1).slice(0, 5);
    if (words.length < 3) return null;
    const at = [], chars = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const p = n.parentElement;
      if (!p || p.closest('.sec-head, button, pre, svg, .chart, .editor')) continue;
      for (let i = 0; i < n.nodeValue.length; i++) { at.push([n, i]); chars.push(n.nodeValue[i].toLowerCase()); }
    }
    const hay = chars.join('');
    const m = new RegExp(words.map(w => w.toLowerCase()).join("[^a-z0-9']+")).exec(hay);
    if (!m) return null;
    const rest = hay.slice(m.index + m[0].length), stop = rest.search(/[.?!:;](\s|$)/);
    const end = Math.min(m.index + m[0].length + (stop >= 0 ? stop + 1 : rest.length), m.index + sentence.length + 40, at.length);
    const range = document.createRange();
    range.setStart(at[m.index][0], at[m.index][1]);
    range.setEnd(at[end - 1][0], at[end - 1][1] + 1);
    return range;
  }

  MLA.components['view-concept'] = {
    props: { route: Object },
    data() { return { state: MLA.store.state, tab: (this.route && this.route.query.tab) || 'learn', checkDone: {}, quizQs: null, preview: false, speech: MLA.speech.state }; },
    computed: {
      P() { return MLA.progress; },
      checkpointsDone() { return (this.c && this.c.checkpoints || []).filter(k => this.P.isSolved(k.problem)).length; },
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
      statusText() { return { mastered: 'Mastered', completed: 'Completed', known: 'Already known', started: 'In progress', ready: 'Ready to learn', locked: 'Prerequisites recommended first' }[this.status]; },
      known() { return this.c ? this.P.knownInfo(this.c.id) : null; },     // marked as known by the placement test
      knownWhy() { return this.known && MLA.placementView ? MLA.placementView.markedWhy(this.known) : ''; },
      // Read aloud: the lesson as spoken parts (js/core/narrate.js), about 15 characters a second at normal speed.
      narration() { return this.c ? MLA.narrate.lesson(this.c) : []; },
      listenMinutes() { return Math.max(1, Math.round(this.narration.reduce((n, it) => n + it.text.length, 0) / 15 / 60)); },
      readingHere() { return this.speech.playing && this.c && this.speech.source === this.c.id; },
      resumeIndex() { const k = this.c && MLA.speech.resumeKey(this.c.id); const i = k ? this.narration.findIndex(it => it.key === k) : -1; return i > 0 ? i : 0; }
    },
    watch: {
      'speech.caption'(text) { this.highlight(text); },
      'route.query.s'() { this.toSection(); },
      'speech.key'(key) {                                   // follow along: scroll to the part being read
        if (!key) { this._readBase = null; return; }
        if (!this.readingHere || this.tab !== 'learn' || this.state.profile.speechFollow === false || !this.$el || !this.$el.querySelector) return;
        const base = String(key).split('~')[0];               // a walkthrough's steps (<section>~<step>) stay in their section
        if (base === this._readBase) return;
        this._readBase = base;
        const el = this.$el.querySelector('[data-narr="' + base + '"]');
        if (el && el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      },
      tab(v) { if (v === 'quiz' && !this.quizQs) this.startQuiz(); if (v === 'learn') this.$nextTick(this.observe); },
      'route.query.tab'(v) { if (v) this.tab = v; }
    },
    created() { if (this.c) { this.P.visit(this.c.id); if (this.tab === 'quiz') this.startQuiz(); } },
    mounted() { this.$nextTick(() => { this.observe(); this.toSection(); }); },
    unmounted() { if (this._io) this._io.disconnect(); },
    methods: {
      go(p) { MLA.router.navigate(p); },
      toSection() {                            // ?s=<section id>: open the lesson at that part (links from search)
        const sid = this.route && this.route.query && this.route.query.s;
        if (!sid || !this.$el || !this.$el.querySelector) return;
        if (this.tab !== 'learn') this.tab = 'learn';
        this.$nextTick(() => {
          const el = this.$el.querySelector('[data-narr="' + sid + '"]');
          if (!el) return;
          const open = el.querySelector('details'); if (open && (sid === 'projects' || sid === 'cases')) open.open = true;
          el.scrollIntoView({ block: 'start', behavior: 'smooth' });
          el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 1600);
        });
      },
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
        if (passed) MLA.ui.toast('Quiz passed. ' + (this.pr.solved < this.pr.needSolved ? 'Solve a practice problem to complete the concept.' : ''), 'success', 4500);
      },
      retryQuiz() { this.quizQs = MLA.quiz.prepare(this.c.quiz, MLA.util.seededRng(Date.now() >>> 0)); },
      openProblem(p) { this.go('/problem/' + p.id); },
      listen(from) { MLA.speech.play(this.narration, from || 0, { source: this.c.id, lessonTitle: this.c.title }); },
      copyLink() {                             // the lesson's static page (tools/snapshots.js) previews well in chats and search results
        const url = location.origin + location.pathname.replace(/index\.html$/, '') + 'lessons/' + this.c.id + '.html';
        Promise.resolve(MLA.util.copyText(url)).then(ok => MLA.ui.toast(ok === false ? url : 'Link copied. It opens a readable page for this lesson, with a button into the interactive version.', ok === false ? 'info' : 'success', 5000));
      },
      highlight(text) {                       // the sentence being read, marked without touching the DOM (CSS Custom Highlight API)
        const H = window.CSS && CSS.highlights;
        if (!H || typeof Highlight === 'undefined') return;
        H.delete('mla-reading');
        if (!text || !this.readingHere || this.state.profile.speechCaptions === false || !this.$el || !this.$el.querySelector) return;
        const box = this.$el.querySelector('[data-narr="' + String(this.speech.key).split('~')[0] + '"]');
        const range = box && (findSentence(box, text) || findSentence(box, text.replace(/^[^.:?]*[.:?]\s+/, '')));   // without a leading heading
        if (range) H.set('mla-reading', new Highlight(range));
      },
      listenOne(key) {
        const s = this.c.sections.find(x => x.id === key);       // a walkthrough reads its steps aloud, and its card follows
        if (s && s.type === 'demo' && /^walk-/.test(s.demo || '') && MLA.walkListen && MLA.walkListen(s.demo.slice(5))) return;
        const i = this.narration.findIndex(it => it.key === key);
        if (i >= 0) MLA.speech.play(this.narration, i, { single: true, source: this.c.id, lessonTitle: this.c.title });
      },
      speaking(key) { return this.readingHere && (this.speech.key === key || String(this.speech.key || '').startsWith(key + '~')); }
    },
    template: `
    <div v-if="!c"><empty-state text="Concept not found."><button class="btn primary mt" @click="go('/map')">Open the concept map</button></empty-state></div>
    <div v-else class="stack concept-page" :style="{'--bc': branch.color}">
      <div class="row between">
        <button class="btn ghost sm" @click="go('/map?focus='+c.id)"><mla-icon name="map" :size="16"/> Concept map</button>
        <div class="row" style="gap:6px"><branch-chip :id="c.branch"/><level-badge :level="c.level"/><span class="chip"><mla-icon name="clock" :size="13"/> {{ c.minutes }} min</span></div>
      </div>

      <div class="concept-hero" data-narr="intro" :class="{speaking: speaking('intro')}">
        <div class="row" style="gap:10px;align-items:center"><status-dot :status="status" :size="26"/><h2 style="margin:0">{{ c.title }}</h2></div>
        <p class="muted" style="margin:6px 0 10px">{{ c.summary }}</p>
        <div v-if="speech.supported" class="listen-row">
          <button class="btn sm" @click="listen(readingHere ? 0 : resumeIndex)"><mla-icon name="headphones" :size="16"/> {{ readingHere ? 'Start again from the top' : resumeIndex ? 'Resume listening' : 'Listen to the lesson' }}</button>
          <button v-if="resumeIndex && !readingHere" class="btn sm ghost" @click="listen(0)">From the start</button>
          <button class="btn sm ghost" @click="copyLink" title="Copy a link to this lesson"><mla-icon name="copy" :size="14"/> Copy link</button>
          <span v-if="resumeIndex && !readingHere" class="tiny dim">You stopped at "{{ narration[resumeIndex].title }}".</span>
          <span v-else class="tiny dim">About {{ listenMinutes }} min read aloud, with formulas, charts and quick checks explained in words. Each part also has its own <mla-icon name="volume" :size="12"/> button.</span>
        </div>
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
            <div v-if="known && !pr.completed" class="callout known-note">
              <div class="callout-title">Already known</div>
              <p style="margin-bottom:6px">{{ knownWhy }} It counts as done, so your path skips it. Read it as a refresher, or pass its quiz and solve a problem to complete it.</p>
              <button class="btn sm" @click="P.setKnown(c.id, false)">Not known after all: unmark it</button>
            </div>
            <div v-if="missing.length" class="callout warn">
              <div class="callout-title">Recommended first</div>
              <p style="margin-bottom:6px">This concept builds on ideas you have not completed yet. You can read on, but these will make it much easier:</p>
              <div class="chips"><a v-for="m in missing" :key="m.id" class="chip clickable" :href="'#/concept/'+m.id" @click.prevent="go('/concept/'+m.id)"><status-dot :status="P.status(m.id)" :size="15"/> {{ m.title }}</a></div>
            </div>
            <div v-else-if="prereqs.length" class="builds-on small"><span class="muted">Builds on:</span> <a v-for="m in prereqs" :key="m.id" class="chip clickable" :href="'#/concept/'+m.id" @click.prevent="go('/concept/'+m.id)"><status-dot :status="P.status(m.id)" :size="15"/> {{ m.title }}</a></div>

            <div v-if="c.why" class="lesson-section why-card" data-narr="why" :class="{speaking: speaking('why')}">
              <div class="sec-head"><mla-icon name="sparkles" :size="18"/><h3>Why this matters</h3><span class="spacer"></span><listen-btn label="Why this matters" @play="listenOne('why')"/></div>
              <mla-md :src="c.why" />
            </div>
            <div v-if="c.objectives.length" class="lesson-section" data-narr="objectives" :class="{speaking: speaking('objectives')}">
              <div class="sec-head"><mla-icon name="target" :size="18"/><h3>By the end you will be able to</h3><span class="spacer"></span><listen-btn label="the objectives" @play="listenOne('objectives')"/></div>
              <ul class="objectives"><li v-for="(o,i) in c.objectives" :key="i"><mla-md :src="o" inline /></li></ul>
            </div>
            <div v-if="c.checkpoints && c.checkpoints.length" class="lesson-section checkpoints-card">
              <div class="sec-head"><mla-icon name="flag" :size="18"/><h3>Checkpoints</h3><span class="spacer"></span><span class="chip tiny" :class="checkpointsDone === c.checkpoints.length ? 'green' : 'accent'">{{ checkpointsDone }} of {{ c.checkpoints.length }} passed</span></div>
              <p class="small muted">Each step of this project has a task that is checked automatically. Pass them in order as you work through the steps below.</p>
              <ol class="checkpoints">
                <li v-for="(k,i) in c.checkpoints" :key="k.problem" :class="{done: P.isSolved(k.problem)}">
                  <span class="cp-mark" :aria-label="P.isSolved(k.problem) ? 'Passed' : 'Not passed yet'"><mla-icon :name="P.isSolved(k.problem) ? 'check' : 'target'" :size="16"/></span>
                  <span class="cp-text"><strong>{{ k.step }}</strong><span class="small muted">{{ k.goal }}</span></span>
                  <button class="btn sm" @click="go('/problem/' + k.problem)">{{ P.isSolved(k.problem) ? 'Review' : 'Start' }}</button>
                </li>
              </ol>
            </div>

            <section v-for="(s,i) in c.sections" :key="i" :id="'sec-'+i" class="lesson-section" :data-narr="s.id" :class="['sec-'+s.type, {read: isRead(s.id), speaking: speaking(s.id)}]">
              <div class="sec-head">
                <mla-icon :name="secIcon(s)" :size="17"/>
                <h3>{{ s.title || (s.type==='check' ? 'Quick check' : 'Section ' + (i+1)) }}</h3>
                <span class="spacer"></span>
                <span v-if="isRead(s.id)" class="chip green tiny"><mla-icon name="check" :size="11"/> read</span>
                <listen-btn :label="s.title || 'this quick check'" @play="listenOne(s.id)"/>
              </div>
              <template v-if="s.type==='text'"><mla-md :src="s.md" /></template>
              <template v-else-if="s.type==='code'">
                <mla-md v-if="s.md" :src="s.md" />
                <div v-if="s.predict" class="predict-box">
                  <div class="predict-head"><mla-icon name="target" :size="15"/> <b>Predict first</b> <span class="small muted">commit to an answer, then run the code to check it</span></div>
                  <div class="q-text"><mla-md :src="s.predict.q" /></div>
                  <check-q :q="s.predict" :done="checkDone['p' + i]" @answer="answerPredict(i, s, $event)" />
                  <div v-if="checkDone['p' + i]" class="feedback" :class="checkDone['p' + i].correct ? 'ok' : 'bad'"><strong>{{ checkDone['p' + i].correct ? 'Good prediction.' : 'Not what happens.' }}</strong> <mla-md :src="checkDone['p' + i].explanation" inline /> Run the code below to see it for yourself.</div>
                  <div v-else-if="cs.checks && cs.checks['predict-' + s.id] != null" class="small muted mt-sm">You made this prediction before{{ cs.checks['predict-' + s.id] ? ', correctly' : '' }}.</div>
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
                <div v-if="checkDone[i]" class="feedback" :class="checkDone[i].correct ? 'ok' : 'bad'"><strong>{{ checkDone[i].correct ? 'Correct.' : 'Not quite.' }}</strong> <mla-md :src="checkDone[i].explanation" inline /><span v-if="!checkDone[i].correct && checkDone[i].expected"> Answer: <b>{{ checkDone[i].expected }}</b></span></div>
                <div v-else-if="cs.checks && cs.checks[s.id] != null" class="tiny dim mt-sm">You answered this before{{ cs.checks[s.id] ? ', correctly' : '' }}.</div>
              </template>
              <div v-if="s.type!=='check'" class="sec-end" :data-sid="s.id"></div>
            </section>

            <div v-if="c.when" class="lesson-section when-card" data-narr="when" :class="{speaking: speaking('when')}">
              <div class="sec-head"><mla-icon name="route" :size="18"/><h3>When to use it (and when not to)</h3><span class="spacer"></span><listen-btn label="When to use it" @play="listenOne('when')"/></div>
              <mla-md :src="c.when" />
            </div>
            <div v-if="c.pitfalls" class="lesson-section pitfalls-card" data-narr="pitfalls" :class="{speaking: speaking('pitfalls')}">
              <div class="sec-head"><mla-icon name="warning" :size="18"/><h3>Common pitfalls</h3><span class="spacer"></span><listen-btn label="Common pitfalls" @play="listenOne('pitfalls')"/></div>
              <mla-md :src="c.pitfalls" />
            </div>
            <div v-if="c.takeaways.length" class="lesson-section takeaways-card" data-narr="takeaways" :class="{speaking: speaking('takeaways')}">
              <div class="sec-head"><mla-icon name="star" :size="18"/><h3>Key takeaways</h3><span class="spacer"></span><listen-btn label="Key takeaways" @play="listenOne('takeaways')"/></div>
              <ul class="takeaways"><li v-for="(t,i) in c.takeaways" :key="i"><mla-md :src="t" inline /></li></ul>
            </div>
            <div v-if="c.cases.length" class="lesson-section cases-card" data-narr="cases" :class="{speaking: speaking('cases')}">
              <div class="sec-head"><mla-icon name="book" :size="18"/><h3>{{ c.cases.length > 1 ? 'Case studies' : 'Case study' }}</h3><span class="spacer"></span><listen-btn label="Case studies" @play="listenOne('cases')"/></div>
              <p class="small muted">What happened when these ideas met the real world.</p>
              <case-study v-for="(k,i) in c.cases" :key="i" :k="k" />
            </div>
            <div v-if="c.projects.length" class="lesson-section projects-card" data-narr="projects" :class="{speaking: speaking('projects')}">
              <div class="sec-head"><mla-icon name="flask" :size="18"/><h3>Build it: {{ c.projects.length > 1 ? 'sample projects' : 'a sample project' }}</h3><span class="spacer"></span><listen-btn label="Sample projects" @play="listenOne('projects')"/></div>
              <p class="small muted">A project to do on your own machine with real data, from a first version to stretch goals. <a href="#/projects" @click.prevent="go('/projects')">See all projects</a>.</p>
              <project-card v-for="(p,i) in c.projects" :key="i" :p="p" :open="c.projects.length === 1" />
            </div>
            <div v-if="c.refs && c.refs.length" class="lesson-section">
              <div class="sec-head"><mla-icon name="external" :size="16"/><h3>Go deeper</h3></div>
              <ul><li v-for="r in c.refs" :key="r.url"><a :href="r.url" target="_blank" rel="noopener">{{ r.title }}</a><span v-if="r.note" class="tiny dim">: {{ r.note }}</span></li></ul>
            </div>

            <div class="card next-card" data-narr="end" :class="{speaking: speaking('end')}">
              <h3 style="margin-top:0">Prove it</h3>
              <p class="small muted">Understanding shows up when you can use it. {{ reqText || 'This concept is complete. The remaining problems and a 90%+ quiz score earn mastery.' }}</p>
              <div class="btn-row">
                <button v-if="c.problems.length" class="btn primary" @click="tab='practice'"><mla-icon name="code" :size="16"/> Practice problems ({{ pr.solved }}/{{ pr.problems }})</button>
                <button v-if="c.quiz.length" class="btn" @click="tab='quiz'"><mla-icon name="clipboard" :size="16"/> {{ pr.quizPassed ? 'Retake the quiz' : 'Take the quiz' }}</button>
              </div>
              <div v-if="next.length" class="mt">
                <div class="tiny dim upper">This unlocks</div>
                <div class="chips"><a v-for="n in next" :key="n.id" class="chip clickable" :href="'#/concept/'+n.id" @click.prevent="go('/concept/'+n.id)"><status-dot :status="P.status(n.id)" :size="15"/> {{ n.title }}</a></div>
              </div>
            </div>
          </div>

          <aside class="lesson-side">
            <div class="card pad-sm">
              <div class="tiny dim upper" style="margin-bottom:6px">In this lesson</div>
              <div class="lesson-toc">
                <a v-for="t in toc" :key="t.i" :class="{read:t.read, check: t.type==='check'}" href="#" @click.prevent="scrollTo(t.i)"><span class="toc-mark">{{ t.read ? '✓' : (t.type==='check' ? '?' : '•') }}</span>{{ t.title }}</a>
              </div>
              <hr>
              <div class="small">
                <div class="row between"><span class="muted">Sections read</span><span>{{ pr.sectionsRead }}/{{ pr.sections }}</span></div>
                <div class="row between"><span class="muted">Quiz</span><span>{{ pr.quizPassed ? 'passed (' + pr.quizBest + '%)' : (pr.quizAttempts ? 'best ' + pr.quizBest + '%' : 'not taken yet') }}</span></div>
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
        <empty-state v-if="!problems.length" icon="code" text="This concept has no coding problems; the quiz completes it." />
      </template>

      <template v-else-if="tab==='quiz'">
        <div class="card">
          <div class="row between mb"><h3 style="margin:0">Mastery quiz: {{ c.title }}</h3><span class="muted small">Pass: {{ P.PASS }}% · best {{ pr.quizBest }}%</span></div>
          <quiz-runner v-if="quizQs && quizQs.length" :key="(quizQs[0] && quizQs[0].q) + pr.quizAttempts" :questions="quizQs" :pass="P.PASS" @complete="onQuiz" @retry="retryQuiz" @close="tab='learn'" />
          <empty-state v-else icon="clipboard" text="No quiz for this concept." />
          <p class="small muted mt"><mla-icon name="clipboard" :size="13"/> More practice: <a :href="'#/quiz?ids=' + c.id + '&deps=1'" @click.prevent="go('/quiz?ids=' + c.id + '&deps=1')">a custom quiz on this lesson and the ones it builds on</a>, with extra questions you will not see here.</p>
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
