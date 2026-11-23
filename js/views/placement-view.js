/* ML Atlas – the placement test page (#/placement): an introduction, one question at a time, and the results: a
 * suggested track starting at the first gap, the reason each lesson was marked as known, and undo (per lesson or all
 * at once). The questions and what the answers imply come from js/engines/placement.js; the marks live in
 * state.known (MLA.progress.setKnown), and the attempt itself in state.placement: the test in progress is rebuilt
 * from its saved answers, so leaving the page and coming back resumes it. */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  const title = id => { const c = MLA.progress.concept(id); return c ? c.title : id; };
  const andList = xs => (xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]);
  const SKIP_LABEL = 'I don\'t know yet';
  // Why a lesson was marked as known, in words (a placement result entry or a state.known record).
  function markedWhy(m) {
    if (!m) return '';
    if (m.by === 'you' || m.why === 'you') return 'You marked it yourself.';
    if (m.why === 'right') {
      if ((m.asked || 1) < 2) return 'You answered its question correctly.';
      return m.right > 1 ? 'You answered both of its questions correctly.' : 'You answered one of its two questions correctly, and your other answers back it up.';
    }
    const via = (m.via || []).map(title);
    if (via.length > 1) return 'You answered questions on ' + andList(via) + ' correctly; both lessons build on this one.';
    if (via.length) return 'You answered a question on ' + via[0] + ' correctly; that lesson builds on this one.';
    return 'Your answers on the lessons that build on it show that you know it.';
  }
  function notMarkedWhy(n) {
    if (n.r === 'skip') return 'You chose "' + SKIP_LABEL + '".';
    if (n.r === 'wrong') return (n.asked || 1) > 1 ? 'Neither of your answers was right.' : 'Your answer was not right.';
    const missed = (n.missed || []).slice(0, 2).map(title);
    if (missed.length) return 'Your answer was right, but you missed the question on ' + andList(missed) + ', which it builds on.';
    return (n.asked || 1) > 1 ? 'Your two answers disagreed, so it is not settled.' : 'One right answer was not enough to be sure.';
  }
  MLA.placementView = { markedWhy, notMarkedWhy };     // also used by the lesson page's "Already known" note

  // The text of an answer as the learner saw it (options are shuffled, so indices are not kept).
  function shownAnswer(q, v) {
    if (q.type === 'mcq') return q.options[v] != null ? q.options[v] : '';
    if (q.type === 'multi') return (v || []).map(i => q.options[i]).join('; ');
    return String(v == null ? '' : v);
  }

  // One question: options to pick (or a box to type in), then Answer, or "I don't know yet".
  MLA.components['placement-q'] = {
    props: { q: Object },
    emits: ['answer', 'skip'],
    data() { return { pick: null, picks: [], text: '' }; },
    computed: {
      multi() { return this.q.type === 'multi'; },
      ready() { return this.q.type === 'fill' ? this.text.trim() !== '' : this.multi ? this.picks.length > 0 : this.pick != null; }
    },
    methods: {
      chosen(i) { return this.multi ? this.picks.includes(i) : this.pick === i; },
      choose(i) { if (!this.multi) { this.pick = i; return; } const k = this.picks.indexOf(i); if (k >= 0) this.picks.splice(k, 1); else this.picks.push(i); },
      submit() { if (this.ready) this.$emit('answer', this.q.type === 'fill' ? this.text : this.multi ? this.picks.slice().sort((a, b) => a - b) : this.pick); }
    },
    template: `
    <div class="pq">
      <div class="q-text"><mla-md :src="q.q" /></div>
      <div v-if="q.type==='mcq' || multi" class="options" role="group" :aria-label="multi ? 'Answers: choose every one that applies' : 'Answers: choose one'">
        <button v-for="(o,i) in q.options" :key="i" type="button" class="option" :class="{selected: chosen(i)}" :aria-pressed="chosen(i) ? 'true' : 'false'" @click="choose(i)"><span class="opt-key">{{ String.fromCharCode(65 + i) }}</span><mla-md :src="o" inline /></button>
        <div v-if="multi" class="tiny dim">Select all that apply.</div>
      </div>
      <div v-else class="inline-form"><input type="text" v-model="text" class="pq-input" aria-label="Your answer" placeholder="Your answer" spellcheck="false" autocomplete="off" @keyup.enter="submit" /></div>
      <div class="btn-row mt">
        <button type="button" class="btn primary" :disabled="!ready" @click="submit">Answer</button>
        <button type="button" class="btn" @click="$emit('skip')">I don't know yet</button>
      </div>
    </div>`
  };

  MLA.components['view-placement'] = {
    props: { route: Object },
    data() {
      const pl = MLA.store.state.placement;
      return { state: MLA.store.state, phase: pl && pl.current ? 'intro' : pl && pl.last ? 'results' : 'intro', cur: null, shown: null, error: '', loaded: 0 };
    },
    computed: {
      P() { return MLA.progress; },
      PL() { return MLA.placement; },
      pl() { return this.state.placement || { current: null, last: null }; },
      inProgress() { return this.pl.current; },
      n() { return this.inProgress ? this.inProgress.answers.length + 1 : 1; },
      last() { return this.pl.last; },
      knownCount() { return this.P.knownList().length; },
      // the lessons the last test marked, and those it asked about without marking: each with its current state
      markedRows() { return this.last ? (this.last.marked || []).map(m => { const k = this.P.knownInfo(m.id); return this.row(m.id, markedWhy(k && k.by === 'you' ? k : m)); }) : []; },
      otherRows() { return this.last ? (this.last.notMarked || []).map(n => this.row(n.id, this.P.isKnown(n.id) ? markedWhy(this.P.knownInfo(n.id)) : notMarkedWhy(n))) : []; },
      onCount() { return this.markedRows.concat(this.otherRows).filter(r => r.on).length; },
      rightCount() { return this.last ? this.last.answers.filter(a => a.r === 'right').length : 0; },
      sug() { this.state.known; this.state.profile.goal; return this.P.suggestTrack(); },
      sugText() {
        const s = this.sug; if (!s) return '';
        const end = t => (/[.?]$/.test(t) ? t : t + '.');                 // titles such as "What Is Machine Learning?"
        const name = s.track.custom ? 'the path to ' + s.track.title.replace(/^Goal: /, '') : s.track.title;
        const covered = s.done ? s.done + ' of its ' + s.total + ' lessons ' + (s.done === 1 ? 'is' : 'are') + ' done or known, so it continues at ' + end(s.gap.title) : 'It starts at ' + end(s.gap.title);
        if (s.reason === 'goal') return 'Your goal is ' + end(name) + ' ' + covered;
        if (s.reason === 'next') return 'Your goal, ' + (s.goal.custom ? 'the path to ' + s.goal.title.replace(/^Goal: /, '') : s.goal.title) + ', is covered. The next track is ' + end(name) + ' ' + covered;
        if (s.reason === 'first') return 'The first track with lessons left for you is ' + end(name) + ' ' + covered;
        return 'Every track is covered. The complete atlas continues at ' + end(s.gap.title);
      },
      review() {
        this.loaded;
        if (!this.last) return [];
        return this.last.answers.map((a, i) => {
          const q = this.PL.questions(a.id)[a.k];
          return { i, id: a.id, lesson: title(a.id), q, r: a.r, v: a.v, expected: q ? MLA.quiz.grade(q, null).expected : '' };
        });
      }
    },
    created() { this._s = null; },
    mounted() { if (this.phase === 'results') this.loadReview(); },
    methods: {
      go(p) { MLA.router.navigate(p); },
      title,
      row(id, why) { const on = this.P.isKnown(id); return { id, title: title(id), on, why, status: this.P.status(id) }; },
      loadReview() { const ids = this.last ? [...new Set(this.last.answers.map(a => a.id))] : []; MLA.content.ensure(ids).then(() => { this.loaded++; }, () => { /* the answers list stays without question text */ }); },
      // start (fresh: a new attempt even if one is in progress) or resume
      start(fresh) {
        const pl = this.state.placement;
        if (fresh || !pl.current) pl.current = { seed: ((Date.now() % 2147483647) ^ Math.floor(Math.random() * 1e9)) >>> 0 || 1, startedAt: new Date().toISOString(), answers: [] };
        this.phase = 'loading'; this.error = '';
        MLA.content.ensure(this.PL.scope()).then(() => this.boot(), e => { this.phase = 'error'; this.error = e && e.message ? e.message : String(e); });
      },
      boot() {
        const cur = this.state.placement.current; if (!cur) { this.phase = 'intro'; return; }
        const last = this.state.placement.last;
        const s = this.PL.create({ seed: cur.seed, avoid: last ? last.answers.map(a => this.PL.qkey(a.id, a.k)) : [] });
        for (const a of cur.answers) s.record(a.id, a.r, a.k);
        this._s = s;
        this.advance();
      },
      advance() {
        const nx = this._s && this._s.next();
        if (!nx) { this.finish(); return; }
        const seed = this.state.placement.current.seed;
        this.cur = { id: nx.id, k: nx.k, again: nx.again };
        this.shown = MLA.quiz.shuffled(nx.q, MLA.util.seededRng((seed ^ MLA.util.hashString(nx.id + '#' + nx.k)) >>> 0));
        this.phase = 'question';
        this.$nextTick(() => { const h = this.$refs.qhead; if (h && h.focus) h.focus(); });
      },
      answer(v) { const g = MLA.quiz.grade(this.shown, v); this.record(g.correct ? 'right' : 'wrong', shownAnswer(this.shown, v)); },
      skip() { this.record('skip', ''); },
      record(r, v) {
        const cur = this.cur; if (!cur || !this._s || !this._s.record(cur.id, r, cur.k)) return;
        this.state.placement.current.answers.push({ id: cur.id, k: cur.k, r, v });
        this.advance();
      },
      // A test with no answers yet is simply dropped; otherwise its result replaces the last one's marks.
      stop() {
        const pl = this.state.placement;
        if (!pl.current || !pl.current.answers.length) { pl.current = null; this._s = null; this.cur = null; this.phase = pl.last ? 'results' : 'intro'; return; }
        this.finish();
      },
      finish() {
        const pl = this.state.placement, cur = pl.current;
        if (!cur || !this._s) { this.phase = pl.last ? 'results' : 'intro'; return; }
        const res = this._s.result();
        this.P.applyPlacement(res.marked);
        pl.last = { seed: cur.seed, startedAt: cur.startedAt, finishedAt: new Date().toISOString(), answers: cur.answers.slice(),
          marked: res.marked.map(m => ({ id: m.id, why: m.why, via: m.via, asked: m.asked, right: m.right })),
          notMarked: res.notMarked.map(x => ({ id: x.id, r: x.r, asked: x.asked, right: x.right, missed: x.missed })) };
        pl.current = null; this._s = null; this.cur = null; this.shown = null;
        MLA.store.saveNow();
        this.phase = 'results'; this.loadReview();
        MLA.ui.toast(res.marked.length ? res.marked.length + (res.marked.length === 1 ? ' lesson is' : ' lessons are') + ' now marked as known.' : 'Nothing was marked as known: your path starts at the beginning.', 'success', 4500);
        if (global.scrollTo) try { global.scrollTo({ top: 0 }); } catch (e) { /* ignore */ }
      },
      toggle(r) { if (r.on) this.P.setKnown(r.id, false); else { const m = (this.last.marked || []).find(x => x.id === r.id); this.P.setKnown(r.id, true, m ? { by: 'placement', why: m.why, via: m.via, asked: m.asked, right: m.right } : { by: 'you' }); } },
      undoAll() {
        const n = this.knownCount; if (!n) return;
        MLA.ui.confirm('Unmark all ' + n + (n === 1 ? ' lesson' : ' lessons') + ' marked as known? Your path then goes through ' + (n === 1 ? 'it' : 'them') + ' again. You can mark each one again from this page.', { okLabel: 'Unmark all', title: 'Undo all marks' }).then(ok => { if (ok) { this.P.unmarkAll(); MLA.ui.toast('All marks removed.', 'info'); } });
      },
      markAllAgain() { for (const m of (this.last && this.last.marked) || []) this.P.setKnown(m.id, true, { by: 'placement', why: m.why, via: m.via, asked: m.asked, right: m.right }); },
      startSuggested() {
        const s = this.sug; if (!s) return;
        if (this.state.profile.goal !== s.track.id) { this.state.profile.goal = s.track.id; MLA.ui.toast('Goal set: ' + s.track.title, 'success'); }
        this.go('/concept/' + s.gap.id);
      },
      outcomeLabel(r) { return { right: 'Right', wrong: 'Not right', skip: SKIP_LABEL }[r] || r; }
    },
    template: `
    <div class="stack placement">
      <section-head title="Placement test" sub="Already know some of this? Answer up to 15 questions, and the lessons you know are marked as known, so your path starts at the first gap." />

      <div v-if="phase==='intro'" class="card placement-intro">
        <ul class="placement-points">
          <li>Every question comes from the quiz of a beginner or intermediate lesson in Start Here, Math Foundations, Python &amp; Data or Classical Machine Learning. Each answer stands for a group of lessons: a right answer also covers the lessons it builds on, and a miss tells the test to look earlier on the map.</li>
          <li>If you are not sure, choose <b>I don't know yet</b> instead of guessing. A lucky guess could mark a lesson you still need.</li>
          <li>It takes 5 to 10 minutes, fewer questions if your first answers settle it. Right and wrong are shown at the end, and you can stop at any point.</li>
          <li>Known lessons earn no XP and are not completed. They count as done for prerequisites, tracks and recommendations, and you can undo each one afterwards.</li>
        </ul>
        <p v-if="last && !inProgress" class="small muted">A new test replaces the marks of your last one ({{ MLA.util.fmtDate(last.finishedAt) }}); lessons you marked yourself stay.</p>
        <div class="btn-row mt">
          <template v-if="inProgress && inProgress.answers.length">
            <button class="btn primary" @click="start(false)"><mla-icon name="play" :size="16"/> Resume at question {{ n }}</button>
            <button class="btn" @click="start(true)">Start again</button>
          </template>
          <button v-else class="btn primary" @click="start(true)"><mla-icon name="play" :size="16"/> Start the test</button>
          <button v-if="last" class="btn ghost" @click="phase='results'; loadReview()">See your last results</button>
          <button v-else class="btn ghost" @click="go('/home')">Not now</button>
        </div>
      </div>

      <div v-else-if="phase==='loading'" class="card center" role="status">Loading the questions…</div>
      <empty-state v-else-if="phase==='error'" icon="warning" :text="error"><button class="btn primary mt" @click="start(false)">Try again</button></empty-state>

      <div v-else-if="phase==='question' && cur && shown" class="card placement-card" :data-lesson="cur.id" :data-k="cur.k" :data-n="n">
        <div class="row between small muted placement-meta"><span>Question {{ n }} of at most {{ PL.MAX }}</span><span>From the lesson {{ title(cur.id) }}{{ cur.again ? ', a second question' : '' }}</span></div>
        <mla-progress :value="(n - 1) / PL.MAX * 100" thin />
        <h3 ref="qhead" class="placement-q-head" tabindex="-1">{{ MLA.quiz.describeType(shown.type) }}</h3>
        <placement-q :key="cur.id + '#' + cur.k" :q="shown" @answer="answer" @skip="skip" />
        <div class="row between mt placement-foot"><span class="tiny dim">You see the right answers at the end.</span><button class="btn sm ghost" @click="stop">{{ inProgress && inProgress.answers.length ? 'Finish now and see the results' : 'Cancel' }}</button></div>
      </div>

      <template v-else-if="phase==='results' && last">
        <div class="card">
          <div class="card-title"><h3><mla-icon name="target" :size="18"/> Your results</h3><span class="tiny dim">{{ MLA.util.fmtDate(last.finishedAt) }}</span></div>
          <p>You answered {{ last.answers.length }} {{ last.answers.length === 1 ? 'question' : 'questions' }}, {{ rightCount }} of them right. {{ onCount ? onCount + (onCount === 1 ? ' lesson is' : ' lessons are') + ' marked as known.' : 'No lesson is marked as known.' }}</p>
          <div v-if="sug && sug.gap" class="callout key placement-next">
            <div class="callout-title">Suggested next</div>
            <p>{{ sugText }}</p>
            <div class="btn-row"><button class="btn primary" @click="startSuggested"><mla-icon name="play" :size="16"/> Start with {{ sug.gap.title }}</button><button class="btn" @click="go('/track/' + sug.track.id)"><mla-icon name="route" :size="16"/> See the whole path</button></div>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><h3>Marked as known</h3>
            <button v-if="knownCount" class="btn sm" @click="undoAll"><mla-icon name="refresh" :size="14"/> Undo all</button>
            <button v-else-if="markedRows.length" class="btn sm" @click="markAllAgain">Mark them all again</button>
          </div>
          <p class="small muted">These lessons count as done for prerequisites, tracks and recommendations, without XP. Take a lesson's quiz and solve a problem to complete it as usual.</p>
          <ul v-if="markedRows.length" class="placement-list">
            <li v-for="r in markedRows" :key="r.id" :class="{off: !r.on}">
              <status-dot :status="r.status" />
              <div class="pl-body"><a :href="'#/concept/' + r.id" @click.prevent="go('/concept/' + r.id)">{{ r.title }}</a><div class="small muted">{{ r.on ? r.why : 'Not marked any more.' }}</div></div>
              <button class="btn sm" :aria-label="(r.on ? 'Undo the mark on ' : 'Mark again: ') + r.title" @click="toggle(r)">{{ r.on ? 'Undo' : 'Mark again' }}</button>
            </li>
          </ul>
          <p v-else class="small">None this time. The questions below show where your path starts.</p>
        </div>

        <div v-if="otherRows.length" class="card">
          <div class="card-title"><h3>Asked about, not marked</h3></div>
          <p class="small muted">These are where your path starts. If one is a lesson you know after all, mark it yourself.</p>
          <ul class="placement-list">
            <li v-for="r in otherRows" :key="r.id">
              <status-dot :status="r.status" />
              <div class="pl-body"><a :href="'#/concept/' + r.id" @click.prevent="go('/concept/' + r.id)">{{ r.title }}</a><div class="small muted">{{ r.why }}</div></div>
              <button class="btn sm" :aria-label="(r.on ? 'Undo the mark on ' : 'Mark as known: ') + r.title" @click="P.setKnown(r.id, !r.on, { by: 'you' })">{{ r.on ? 'Undo' : 'Mark as known' }}</button>
            </li>
          </ul>
        </div>

        <details class="card placement-review">
          <summary>Your answers, with the right ones ({{ review.length }})</summary>
          <ol class="placement-answers">
            <li v-for="a in review" :key="a.i">
              <div class="tiny dim">{{ a.lesson }} · <span :class="a.r === 'right' ? 'ok-text' : 'bad-text'">{{ outcomeLabel(a.r) }}</span></div>
              <div v-if="a.q" class="q-text"><mla-md :src="a.q.q" /></div>
              <div class="small"><span class="muted">Your answer:</span> <mla-md v-if="a.v && a.q && a.q.type !== 'fill'" :src="a.v" inline /><span v-else>{{ a.r === 'skip' ? '(none)' : a.v }}</span></div>
              <div v-if="a.q && a.r !== 'right'" class="small"><span class="muted">Right answer:</span> <mla-md :src="a.expected" inline /></div>
            </li>
          </ol>
        </details>

        <div class="btn-row"><button class="btn" @click="phase='intro'"><mla-icon name="refresh" :size="16"/> Retake the test</button><button class="btn ghost" @click="go('/map')"><mla-icon name="map" :size="16"/> Open the concept map</button></div>
      </template>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
