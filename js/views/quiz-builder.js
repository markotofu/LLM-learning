/* ML Atlas – custom quizzes (#/quiz). Pick any lessons (one by one, a whole branch, your track or everything you have
 * completed), optionally with the lessons they build on, and get a quiz drawn from them. Each lesson contributes its own
 * quiz questions and the extra questions of the question bank (js/data/bank-<branch>.js), which appear nowhere else.
 * Questions are drawn in turns, one lesson at a time, so every chosen lesson is asked about. Missed questions from a
 * lesson's own quiz join the review queue, as they do in the lesson; bank questions stay in custom quizzes.
 * A link can preselect lessons: #/quiz?ids=knn,svm&deps=1 */
(function (global) {
  const MLA = global.MLA, C = MLA.components;
  const COUNTS = [10, 20, 30, 50, 0];                     // 0: every available question

  // Draw `count` questions (0 for all) from the lessons `ids`: shuffle each lesson's pool, then take one question from
  // each lesson in turn, in a shuffled lesson order, and shuffle the result.
  function draw(ids, count, rng) {
    const bank = MLA.data.quizBank || {};
    const pools = ids.map(id => {
      const c = MLA.content.byId[id];
      if (!c) return [];
      const tag = extra => q => Object.assign({}, q, { _from: id, _lesson: c.title, _bank: extra });
      return MLA.util.shuffle((c.quiz || []).map(tag(false)).concat((bank[id] || []).map(tag(true))), rng);
    }).filter(p => p.length);
    const out = [], order = MLA.util.shuffle(pools.map((p, i) => i), rng), limit = count > 0 ? count : Infinity;
    for (let round = 0; out.length < limit; round++) {
      let added = false;
      for (const k of order) {
        if (out.length >= limit) break;
        const q = pools[k][round];
        if (q) { out.push(q); added = true; }
      }
      if (!added) break;
    }
    return MLA.util.shuffle(out, rng);
  }
  MLA.customQuiz = { draw, COUNTS };

  C['view-quiz'] = {
    props: { route: Object },
    data() {
      const p = MLA.store.state.profile, query = (this.route && this.route.query) || {};
      const known = id => !!MLA.content.byId[id];
      const linked = query.ids ? String(query.ids).split(',').filter(known) : null;
      return {
        P: MLA.progress, state: MLA.store.state,
        picked: linked || (Array.isArray(p.quizPick) ? p.quizPick.filter(known) : []),
        deps: query.deps != null ? query.deps === '1' : !!p.quizDeps,
        count: COUNTS.includes(p.quizCount) ? p.quizCount : 20,
        loading: false, error: '', quiz: null, result: null, seed: 0
      };
    },
    computed: {
      counts() { return COUNTS; },
      branches() {
        return MLA.data.branches.map(b => ({ b, lessons: MLA.data.concepts.filter(c => c.branch === b.id) })).filter(x => x.lessons.length);
      },
      chosen() {                                           // the picked lessons, and what they build on when asked
        const set = new Set(this.picked);
        if (this.deps && MLA.content.graph) for (const id of this.picked) MLA.graph.ancestors(MLA.content.graph, id).forEach(a => set.add(a));
        return MLA.data.concepts.filter(c => set.has(c.id)).map(c => c.id);
      },
      completedIds() { return MLA.data.concepts.filter(c => this.P.isCompleted(c.id)).map(c => c.id); },
      trackIds() { const g = this.state.profile.goal; return g && this.P.track(g) ? this.P.trackConcepts(g) : []; },
      lessonCount() { return this.quiz ? new Set(this.quiz.map(q => q._from)).size : 0; },
      bankCount() { return this.quiz ? this.quiz.filter(q => q._bank).length : 0; },
      breakdown() {                                        // weakest lessons first
        if (!this.result) return [];
        const by = {};
        for (const r of this.result.results) {
          const q = r.question, b = by[q._from] || (by[q._from] = { id: q._from, title: q._lesson, right: 0, total: 0 });
          b.total++; if (r.correct) b.right++;
        }
        return Object.values(by).sort((a, b) => a.right / a.total - b.right / b.total || a.title.localeCompare(b.title));
      }
    },
    watch: {
      picked: { deep: true, handler() { this.save(); } },
      deps() { this.save(); },
      count() { this.save(); }
    },
    methods: {
      save() { const p = this.state.profile; p.quizPick = this.picked.slice(); p.quizDeps = this.deps; p.quizCount = this.count; },
      countIn(x) { return x.lessons.filter(c => this.picked.includes(c.id)).length; },
      allIn(x) { return this.countIn(x) === x.lessons.length; },
      someIn(x) { const n = this.countIn(x); return n > 0 && n < x.lessons.length; },
      toggleBranch(x, on) {
        const ids = x.lessons.map(c => c.id);
        this.picked = on ? [...new Set(this.picked.concat(ids))] : this.picked.filter(id => !ids.includes(id));
      },
      add(ids) { this.picked = [...new Set(this.picked.concat(ids))]; },
      clear() { this.picked = []; },
      async start() {
        const ids = this.chosen;
        if (!ids.length || this.loading) return;
        this.loading = true; this.error = '';
        try {
          await MLA.content.ensure(ids);
          await MLA.content.ensureBank(ids);
          this.seed = (Date.now() >>> 0) || 1;
          const rng = MLA.util.seededRng(this.seed), qs = draw(ids, this.count, rng);
          if (!qs.length) { this.error = 'These lessons have no quiz questions yet.'; return; }
          this.quiz = MLA.quiz.prepare(qs, rng);
          this.result = null;
          if (global.scrollTo) global.scrollTo(0, 0);
        } catch (e) {
          this.error = (e && e.message) || String(e);
        } finally {
          this.loading = false;
        }
      },
      onDone(result) {
        this.result = result;
        if (MLA.mistakes) for (const r of result.results) {
          const q = r.question;
          if (!r.correct && q && !q._bank && q.type !== 'match') MLA.mistakes.note('quiz', q._from, MLA.mistakes.qkey(q));
        }
      },
      back() { this.quiz = null; this.result = null; },
      go(id) { MLA.router.navigate('/concept/' + id); }
    },
    template: `
    <div class="stack quiz-builder">
      <template v-if="!quiz">
        <div class="card">
          <h2 class="qb-title">Build your own quiz</h2>
          <p class="muted">Choose any lessons, a whole branch, your track or everything you have completed. The quiz mixes each lesson's own questions with extra questions that appear only here, and takes them in turns so that every lesson you chose is asked about.</p>
          <div class="btn-row">
            <button class="btn sm" :disabled="!completedIds.length" @click="add(completedIds)">Lessons you completed ({{ completedIds.length }})</button>
            <button class="btn sm" :disabled="!trackIds.length" @click="add(trackIds)">{{ trackIds.length ? 'Your track (' + trackIds.length + ')' : 'Your track (none chosen yet)' }}</button>
            <button class="btn sm ghost" :disabled="!picked.length" @click="clear">Clear</button>
          </div>
          <label class="checkbox mt"><input type="checkbox" v-model="deps"> Also ask about the lessons these build on<span v-if="deps && chosen.length > picked.length" class="muted"> ({{ chosen.length - picked.length }} more)</span></label>
          <div class="qb-start">
            <label class="qb-count">Questions
              <select v-model.number="count" aria-label="Number of questions"><option v-for="n in counts" :key="n" :value="n">{{ n || 'All of them' }}</option></select>
            </label>
            <button class="btn primary" :disabled="!chosen.length || loading" @click="start"><mla-icon name="play" :size="15"/> {{ loading ? 'Loading the questions' : chosen.length ? 'Start: ' + chosen.length + (chosen.length === 1 ? ' lesson' : ' lessons') : 'Choose lessons to start' }}</button>
          </div>
          <p v-if="error" class="small bad-text" role="alert">{{ error }}</p>
        </div>
        <div v-for="x in branches" :key="x.b.id" class="card qb-branch" :style="{ '--bc': x.b.color }">
          <label class="checkbox qb-branch-head"><input type="checkbox" :checked="allIn(x)" :indeterminate.prop="someIn(x)" @change="toggleBranch(x, $event.target.checked)"> <strong>{{ x.b.name }}</strong> <span class="tiny muted">{{ countIn(x) }} of {{ x.lessons.length }} chosen</span></label>
          <div class="qb-lessons">
            <label v-for="c in x.lessons" :key="c.id" class="checkbox qb-lesson"><input type="checkbox" :value="c.id" v-model="picked"> <status-dot :status="P.status(c.id)" :size="12"/> {{ c.title }}</label>
          </div>
        </div>
      </template>
      <template v-else>
        <div class="card row between qb-head">
          <span><strong>{{ quiz.length }} questions from {{ lessonCount }} {{ lessonCount === 1 ? 'lesson' : 'lessons' }}</strong><span class="small muted">, {{ bankCount }} of them from the extra question bank</span></span>
          <button class="btn sm ghost" @click="back">Change the lessons</button>
        </div>
        <quiz-runner :key="seed" :questions="quiz" :pass="70" back-label="Back to the quiz builder" @complete="onDone" @retry="result = null" @close="back" />
        <div v-if="result" class="card">
          <h3 class="qb-title">By lesson, weakest first</h3>
          <div v-for="b in breakdown" :key="b.id" class="qb-break">
            <a :href="'#/concept/' + b.id" @click.prevent="go(b.id)">{{ b.title }}</a>
            <span class="small" :class="b.right === b.total ? 'ok-text' : 'bad-text'">{{ b.right }} of {{ b.total }}</span>
          </div>
          <p class="small muted">Missed questions from a lesson's own quiz come back in Review; the extra questions stay here.</p>
          <div class="btn-row"><button class="btn" @click="start"><mla-icon name="refresh" :size="15"/> New questions, same lessons</button></div>
        </div>
      </template>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
