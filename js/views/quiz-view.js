/* Quiz runner (concept mastery quizzes) and inline knowledge checks */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  MLA.components['quiz-runner'] = {
    props: { questions: Array, pass: { default: 70 }, backLabel: { type: String, default: 'Back to the lesson' } },
    emits: ['complete', 'close', 'retry'],
    data() { return { answers: {}, submitted: false, result: null }; },
    computed: {
      answeredCount() { return this.questions.filter((q, i) => this.hasAnswer(i)).length; },
      allAnswered() { return this.answeredCount === this.questions.length; }
    },
    methods: {
      typeName(t) { return MLA.quiz.describeType(t); },
      hasAnswer(i) { const a = this.answers[i]; if (a == null) return false; if (Array.isArray(a)) return a.length > 0; if (typeof a === 'object') return Object.keys(a).length > 0; return a !== ''; },
      setMcq(i, opt) { if (!this.submitted) this.answers[i] = opt; },
      toggleMulti(i, opt) { if (this.submitted) return; const a = this.answers[i] = (this.answers[i] || []).slice(); const k = a.indexOf(opt); if (k >= 0) a.splice(k, 1); else a.push(opt); this.answers[i] = a; },
      setFill(i, e) { this.answers[i] = e.target.value; },
      setMatch(i, li, e) { const o = Object.assign({}, this.answers[i] || {}); o[li] = Number(e.target.value); this.answers[i] = o; },
      rightOrder(q) { return q.rightOrder || q.pairs.map((p, k) => k); },      // set by MLA.quiz.prepare, so answers do not sit in row order
      matchLabel(p) { return 'Match for: ' + String(p[0]).replace(/[*_\x60]/g, ''); },
      orderList(i) { const q = this.questions[i]; return (this.answers[i] && this.answers[i].slice()) || (q.start || q.items.map((x, k) => k)).slice(); },
      moveOrder(i, from, dir) { const cur = this.orderList(i); const to = from + dir; if (to < 0 || to >= cur.length) return; [cur[from], cur[to]] = [cur[to], cur[from]]; this.answers[i] = cur; },
      optClass(i, opt) {
        const q = this.questions[i]; const sel = this.answers[i];
        if (!this.submitted) return q.type === 'multi' ? { selected: (sel || []).includes(opt) } : { selected: sel === opt };
        const correct = q.type === 'multi' ? q.answer.includes(opt) : q.type === 'tf' ? q.answer === opt : q.answer === opt;
        const chosen = q.type === 'multi' ? (sel || []).includes(opt) : sel === opt;
        return { correct, wrong: chosen && !correct, selected: chosen };
      },
      res(i) { return this.result ? this.result.results[i] : null; },
      submit() {
        // order questions count as answered even if left in the starting order
        this.questions.forEach((q, i) => { if (q.type === 'order' && !this.answers[i]) this.answers[i] = this.orderList(i); });
        if (!this.allAnswered) { MLA.ui.toast('Answer every question first (' + this.answeredCount + '/' + this.questions.length + ')', 'error'); return; }
        this.result = MLA.quiz.gradeAll(this.questions, this.questions.map((q, i) => this.answers[i]));
        this.submitted = true;
        this.$emit('complete', this.result);
        this.$nextTick(() => { try { this.$el.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (e) { /* ignore */ } });
      },
      retry() { this.answers = {}; this.submitted = false; this.result = null; this.$emit('retry'); }
    },
    template: `
    <div class="quiz">
      <template v-if="submitted">
        <div class="card result-card">
          <div class="result-head">
            <score-ring :value="result.score" />
            <div>
              <h2 style="margin:0">{{ result.score >= pass ? 'Passed' : 'Not yet. Keep going.' }}</h2>
              <p class="muted" style="margin:4px 0 10px">{{ result.correct }} of {{ result.total }} correct ({{ result.score }}%). Passing is {{ pass }}%. Read the explanations below: they are the fastest way to fix a misconception.</p>
              <div class="btn-row"><button class="btn" @click="retry"><mla-icon name="refresh" :size="16"/> Try again</button><button class="btn ghost" @click="$emit('close')">{{ backLabel }}</button></div>
            </div>
          </div>
        </div>
      </template>
      <div v-else class="row between mb">
        <div class="muted small">{{ answeredCount }} / {{ questions.length }} answered</div>
        <mla-progress :value="answeredCount/questions.length*100" thin style="width:160px" />
      </div>

      <div v-for="(q,i) in questions" :key="i" class="quiz-q">
        <div class="q-num">Question {{ i+1 }} · {{ typeName(q.type) }}<span v-if="q._lesson" class="dim"> · {{ q._lesson }}</span><span v-if="submitted" :class="res(i).correct?'ok-text':'bad-text'"> · {{ res(i).correct ? 'Correct' : (res(i).partial>0 ? 'Partly correct' : 'Incorrect') }}</span></div>
        <div class="q-text"><mla-md :src="q.q" /></div>
        <div v-if="q.type==='mcq'||q.type==='multi'" class="options">
          <div v-for="(o,oi) in q.options" :key="oi" class="option" :class="optClass(i,oi)" role="button" @click="q.type==='multi'?toggleMulti(i,oi):setMcq(i,oi)" tabindex="0" @keydown.enter.prevent="q.type==='multi'?toggleMulti(i,oi):setMcq(i,oi)" @keydown.space.prevent="q.type==='multi'?toggleMulti(i,oi):setMcq(i,oi)">
            <span class="opt-key">{{ String.fromCharCode(65+oi) }}</span><mla-md :src="o" inline />
          </div>
          <div v-if="q.type==='multi' && !submitted" class="tiny dim">Select all that apply.</div>
        </div>
        <div v-else-if="q.type==='tf'" class="options">
          <div class="option" :class="optClass(i,true)" role="button" tabindex="0" @click="setMcq(i,true)" @keydown.enter.prevent="setMcq(i,true)" @keydown.space.prevent="setMcq(i,true)"><span class="opt-key">T</span>True</div>
          <div class="option" :class="optClass(i,false)" role="button" tabindex="0" @click="setMcq(i,false)" @keydown.enter.prevent="setMcq(i,false)" @keydown.space.prevent="setMcq(i,false)"><span class="opt-key">F</span>False</div>
        </div>
        <div v-else-if="q.type==='fill'">
          <input type="text" placeholder="your answer…" aria-label="Your answer" @input="setFill(i,$event)" :value="answers[i]||''" spellcheck="false" :disabled="submitted" />
          <div v-if="submitted" class="small mono mt">Expected: {{ res(i).expected }}</div>
        </div>
        <div v-else-if="q.type==='order'" class="list">
          <div v-for="(oi,pos) in orderList(i)" :key="oi" class="list-item order-item">
            <span class="opt-key">{{ pos+1 }}</span><div class="li-body"><mla-md :src="q.items[oi]" inline /></div>
            <div v-if="!submitted" class="li-right"><button class="btn xs" @click="moveOrder(i,pos,-1)" aria-label="Move up">↑</button><button class="btn xs" @click="moveOrder(i,pos,1)" aria-label="Move down">↓</button></div>
          </div>
          <div v-if="submitted && !res(i).correct" class="small mt">Correct order: {{ res(i).expected }}</div>
        </div>
        <div v-else-if="q.type==='match'" class="list">
          <div v-for="(p,pi) in q.pairs" :key="pi" class="match-row">
            <div class="match-left"><mla-md :src="p[0]" inline /></div><span class="dim">→</span>
            <select @change="setMatch(i,pi,$event)" :disabled="submitted" :aria-label="matchLabel(p)">
              <option value="" disabled :selected="(answers[i]||{})[pi]==null">choose…</option>
              <option v-for="pj in rightOrder(q)" :key="pj" :value="pj" :selected="(answers[i]||{})[pi]===pj">{{ q.pairs[pj][1] }}</option>
            </select>
          </div>
          <div v-if="submitted && !res(i).correct" class="small mt">{{ res(i).expected }}</div>
        </div>
        <div v-if="submitted" class="feedback" :class="res(i).correct?'ok':'bad'"><strong>{{ res(i).correct ? 'Correct.' : 'Explanation:' }}</strong> <mla-md :src="res(i).explanation" inline /></div>
      </div>

      <div v-if="!submitted" class="btn-row mt">
        <button class="btn primary" @click="submit"><mla-icon name="check" :size="16"/> Submit answers</button>
        <button class="btn ghost" @click="$emit('close')">Cancel</button>
      </div>
    </div>`
  };

  // Inline knowledge check inside a lesson (instant feedback, single question)
  MLA.components['check-q'] = {
    props: { q: Object, done: Object },
    emits: ['answer'],
    data() { return { multi: [], fillVal: '', pickedVal: null, orderList: null }; },
    created() { if (this.q.type === 'order') this.orderList = MLA.util.shuffle(this.q.items.map((x, i) => i), MLA.util.seededRng(MLA.util.hashString(this.q.q))); },
    methods: {
      pick(v) { if (this.done) return; this.pickedVal = v; this.$emit('answer', v); },
      toggle(i) { if (this.done) return; const k = this.multi.indexOf(i); if (k >= 0) this.multi.splice(k, 1); else this.multi.push(i); },
      move(from, d) { if (this.done) return; const l = this.orderList; const to = from + d; if (to < 0 || to >= l.length) return; [l[from], l[to]] = [l[to], l[from]]; },
      optClass(i) {
        if (!this.done) return this.q.type === 'multi' ? { selected: this.multi.includes(i) } : {};
        const correct = this.q.type === 'multi' ? this.q.answer.includes(i) : this.q.answer === i;
        const chosen = this.q.type === 'multi' ? this.multi.includes(i) : this.pickedVal === i;
        return { correct, wrong: chosen && !correct, selected: chosen };
      }
    },
    template: `
    <div>
      <div v-if="q.type==='mcq'||q.type==='multi'" class="options">
        <div v-for="(o,i) in q.options" :key="i" class="option" :class="optClass(i)" role="button" @click="q.type==='multi'?toggle(i):pick(i)" tabindex="0" @keydown.enter.prevent="q.type==='multi'?toggle(i):pick(i)" @keydown.space.prevent="q.type==='multi'?toggle(i):pick(i)"><span class="opt-key">{{ String.fromCharCode(65+i) }}</span><mla-md :src="o" inline /></div>
        <button v-if="q.type==='multi' && !done" class="btn sm mt" @click="$emit('answer', multi.slice())">Check</button>
      </div>
      <div v-else-if="q.type==='tf'" class="options">
        <div class="option" :class="done ? {correct: q.answer===true, wrong: pickedVal===true && !q.answer} : {}" role="button" tabindex="0" @click="pick(true)" @keydown.enter.prevent="pick(true)" @keydown.space.prevent="pick(true)"><span class="opt-key">T</span>True</div>
        <div class="option" :class="done ? {correct: q.answer===false, wrong: pickedVal===false && q.answer} : {}" role="button" tabindex="0" @click="pick(false)" @keydown.enter.prevent="pick(false)" @keydown.space.prevent="pick(false)"><span class="opt-key">F</span>False</div>
      </div>
      <div v-else-if="q.type==='fill'" class="inline-form">
        <input v-model="fillVal" :disabled="!!done" spellcheck="false" placeholder="your answer…" aria-label="Your answer" @keyup.enter="$emit('answer', fillVal)" />
        <button v-if="!done" class="btn sm primary" @click="$emit('answer', fillVal)">Check</button>
      </div>
      <div v-else-if="q.type==='order'">
        <div class="list"><div v-for="(oi,pos) in orderList" :key="oi" class="list-item order-item"><span class="opt-key">{{ pos+1 }}</span><div class="li-body"><mla-md :src="q.items[oi]" inline /></div><div class="li-right" v-if="!done"><button class="btn xs" @click="move(pos,-1)">↑</button><button class="btn xs" @click="move(pos,1)">↓</button></div></div></div>
        <button v-if="!done" class="btn sm primary mt" @click="$emit('answer', orderList.slice())">Check order</button>
      </div>
      <div v-else-if="q.type==='match'" class="small muted">Matching questions appear in the quiz.</div>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
