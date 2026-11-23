/* LeetCode-style problem page: description / hints / solution on the left,
 * editor + console (test cases, results, output) on the right. Run = visible examples, Submit = all tests. */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  MLA.components['view-problem'] = {
    props: { route: Object },
    data() { return { state: MLA.store.state, left: 'desc', con: 'cases', code: '', running: false, mode: null, result: null, failure: null, openRow: null, accepted: null, compact: false }; },
    computed: {
      P() { return MLA.progress; },
      p() { return this.P.problem(this.route.params.id); },
      c() { return this.p ? this.P.concept(this.p.concept) : null; },
      checkpoint() {                           // capstone checkpoints (concept.checkpoints): where this problem sits, and the next one
        const list = (this.c && this.c.checkpoints) || [], i = list.findIndex(k => k.problem === this.p.id);
        return i < 0 ? null : { n: i + 1, of: list.length, step: list[i].step, next: list[i + 1] ? list[i + 1].problem : null };
      },
      ps() { return (this.p && this.P.peekP(this.p.id)) || {}; },
      py() { return MLA.python.status; },
      siblings() { return this.c ? this.c.problems : []; },
      allProblems() { const order = MLA.graph.topoOrder(MLA.content.graph); const out = []; for (const id of order) for (const q of this.P.concept(id).problems) out.push(q); return out; },
      idx() { return this.allProblems.findIndex(q => q.id === this.p.id); },
      prev() { return this.allProblems[this.idx - 1] || null; },
      next() { return this.allProblems[this.idx + 1] || null; },
      visible() { return this.p.tests.filter(t => !t.hidden); },
      hiddenCount() { return this.p.tests.length - this.visible.length; },
      hintsShown() { return this.ps.hintsShown || 0; },
      solutionOpen() { return !!(this.ps.solutionViewed || this.ps.solved); },
      status() { return this.P.problemStatus(this.p.id); },
      summary() {
        const r = this.result; if (!r) return null;
        if (r.error) return { kind: 'error', label: r.error.type === 'SyntaxError' || r.error.type === 'IndentationError' ? 'Syntax Error' : 'Runtime Error', passed: 0, total: this.mode === 'submit' ? this.p.tests.length : this.visible.length };
        const total = r.tests.length, passed = r.tests.filter(t => t.passed).length;
        const anyErr = r.tests.some(t => t.error && !t.passed);
        if (passed === total) return { kind: 'ok', label: this.mode === 'submit' ? 'Accepted' : 'All examples pass', passed, total };
        return { kind: 'bad', label: anyErr ? 'Runtime Error' : 'Wrong Answer', passed, total };
      },
      errorLine() { const r = this.result; if (!r) return null; if (r.error) return r.error.line; const t = r.tests.find(x => !x.passed && x.error && x.error.line); return t ? t.error.line : null; },
      topError() { return this.result && this.result.error; },
      topHint() { return MLA.python.explain(this.topError); }
    },
    watch: {
      code() { this.queueSave(); },
      'route.params.id'() { this.load(); }
    },
    created() { this.load(); this.queueSave = MLA.util.debounce(() => this.saveDraft(), 600); },
    mounted() { this.measure(); this._onResize = () => this.measure(); global.addEventListener('resize', this._onResize); if (!MLA.python.status.ready && MLA.python.status.state !== 'loading') MLA.python.warmup(); },
    unmounted() { global.removeEventListener('resize', this._onResize); this.saveDraft(); },
    methods: {
      measure() { this.compact = (global.innerWidth || 1200) < 980; },
      load() {
        if (!this.p) return;
        const st = this.P.peekP(this.p.id);
        this.code = st && st.draft != null ? st.draft : this.p.starter;
        this.result = null; this.failure = null; this.left = 'desc'; this.con = 'cases'; this.accepted = null; this.openRow = null;
      },
      saveDraft() { if (!this.p) return; const st = this.P.peekP(this.p.id); if (this.code === this.p.starter && (!st || st.draft == null)) return; if (!st || st.draft !== this.code) this.P.saveDraft(this.p.id, this.code); },
      go(path) { MLA.router.navigate(path); },
      async run(mode) {
        if (this.running) return;
        this.saveDraft();
        this.running = true; this.mode = mode; this.result = null; this.failure = null; this.accepted = null; this.con = 'result'; this.openRow = null;
        const r = await MLA.python.runTests(this.code, this.p, mode);
        this.running = false;
        if (!r.ok) { this.failure = r; return; }
        this.result = r.result;
        const firstBad = (r.result.tests || []).findIndex(t => !t.passed); this.openRow = firstBad >= 0 ? firstBad : null;
        if (mode === 'run') this.P.recordRun(this.p.id);
        else {
          const out = this.P.recordSubmit(this.p.id, this.code, r.result);
          if (out.accepted) { this.accepted = out; if (out.first) this.left = 'solution'; }
        }
      },
      stop() { MLA.python.stop(); this.running = false; },
      reset() { MLA.ui.confirm('Replace your code with the original starter code? Your current code will be lost (download it first if you want to keep it).', { okLabel: 'Reset code', danger: true }).then(ok => { if (ok) { this.code = this.p.starter; this.P.saveDraft(this.p.id, this.p.starter); } }); },
      download() { const head = '# ' + MLA.course.name + ': ' + this.p.title + ' (' + this.p.difficulty + ')\n# Concept: ' + this.c.title + '\n\n'; MLA.util.download(this.p.id + '.py', head + this.code, 'text/x-python'); },
      copyCode() { MLA.util.copyText(this.code).then(ok => MLA.ui.toast(ok ? 'Code copied.' : 'Could not copy.', ok ? 'success' : 'error')); },
      showHint() { const n = this.hintsShown + 1; this.P.showHint(this.p.id, n); },
      revealSolution() {
        if (this.solutionOpen) return;
        MLA.ui.confirm('Stuck? That is part of learning. Before you look, try the hints and re-read the lesson section this builds on.\n\nIf you reveal the solution before solving, solving this problem earns **half XP**.', { okLabel: 'Reveal solution', cancelLabel: 'Keep trying' }).then(ok => { if (ok) this.P.viewSolution(this.p.id); });
      },
      loadSolution() { MLA.ui.confirm('Copy the reference solution into the editor? Your current code will be replaced.', { okLabel: 'Copy into editor' }).then(ok => { if (ok) this.code = this.p.solution; }); },
      loadAccepted() { if (this.ps.accepted) this.code = this.ps.accepted.code; },
      testName(t, i) { return t.name || ('Test ' + (i + 1)); },
      toggleRow(i) { this.openRow = this.openRow === i ? null : i; }
    },
    template: `
    <div v-if="!p" class="view-pad"><empty-state text="Problem not found."><button class="btn primary mt" @click="go('/problems')">All problems</button></empty-state></div>
    <div v-else class="problem-view" :class="{compact}" :style="{'--bc': P.branch(c.branch).color}">
      <div class="pv-left">
        <div class="pv-tabs">
          <button :class="{active:left==='desc'}" @click="left='desc'"><mla-icon name="file" :size="14"/> Description</button>
          <button :class="{active:left==='hints'}" @click="left='hints'"><mla-icon name="hint" :size="14"/> Hints <span class="tab-count">{{ hintsShown }}/{{ p.hints.length }}</span></button>
          <button :class="{active:left==='solution'}" @click="left='solution'"><mla-icon name="eye" :size="14"/> Solution</button>
        </div>
        <div class="pv-scroll">
          <template v-if="left==='desc'">
            <div v-if="checkpoint" class="checkpoint-banner small"><mla-icon name="flag" :size="14"/> Checkpoint {{ checkpoint.n }} of {{ checkpoint.of }}: {{ checkpoint.step }}, in <a :href="'#/concept/' + c.id" @click.prevent="go('/concept/' + c.id)">{{ c.title }}</a><button v-if="checkpoint.next && P.isSolved(p.id)" class="btn sm" @click="go('/problem/' + checkpoint.next)">Next checkpoint</button></div>
            <div class="row between" style="flex-wrap:nowrap;align-items:flex-start"><h2 class="pv-title">{{ p.title }}</h2><status-dot :status="status" :size="24"/></div>
            <div class="row" style="gap:6px;margin-bottom:10px"><diff-chip :d="p.difficulty"/><a class="chip clickable" :href="'#/concept/'+c.id" @click.prevent="go('/concept/'+c.id)"><mla-icon name="book" :size="12"/> {{ c.title }}</a><span v-if="ps.solved" class="chip green tiny">Solved{{ ps.assisted ? ' (with solution)' : '' }}</span></div>
            <mla-md :src="p.prompt" />
            <div class="pv-examples">
              <div class="tiny dim upper">Examples checked by Run</div>
              <div v-for="(t,i) in visible" :key="i" class="pv-example">
                <div class="small"><b>{{ testName(t,i) }}</b></div>
                <pre v-if="t.setup" class="pv-code">{{ t.setup }}</pre>
                <pre v-if="t.call" class="pv-code"><span class="dim"># call</span>
{{ t.call }}<template v-if="t.expect"><span class="dim">
# expected →</span> {{ t.expect }}</template></pre>
                <pre v-if="t.code" class="pv-code">{{ t.code }}</pre>
              </div>
              <div v-if="hiddenCount" class="tiny dim">+ {{ hiddenCount }} hidden test{{ hiddenCount>1?'s':'' }} (random inputs and edge cases) run when you Submit.</div>
            </div>
          </template>
          <template v-else-if="left==='hints'">
            <p class="small muted">Hints reveal one at a time. Using hints does not reduce your XP; only viewing the full solution before solving does.</p>
            <div v-for="(h,i) in p.hints.slice(0, hintsShown)" :key="i" class="callout tip"><div class="callout-title">Hint {{ i+1 }}</div><mla-md :src="h" /></div>
            <button v-if="hintsShown < p.hints.length" class="btn" @click="showHint"><mla-icon name="hint" :size="16"/> Show hint {{ hintsShown+1 }} of {{ p.hints.length }}</button>
            <p v-else-if="p.hints.length" class="small muted">That's every hint. Still stuck? Check the Solution tab.</p>
            <p v-else class="small muted">No hints for this one: try the lesson's code samples.</p>
          </template>
          <template v-else>
            <div v-if="!solutionOpen" class="card">
              <h3 style="margin-top:0">Reference solution</h3>
              <p class="small muted">Give it a real try first: struggling productively is where the learning happens. Revealing before you solve halves the XP for this problem.</p>
              <button class="btn" @click="revealSolution"><mla-icon name="eye" :size="16"/> Reveal solution</button>
            </div>
            <template v-else>
              <div v-if="accepted && accepted.first" class="callout tip"><div class="callout-title">Accepted</div>Nice work. Compare your approach with the reference below: there is often more than one good way.</div>
              <div v-if="p.explain" class="lesson-section" style="padding:16px"><div class="sec-head"><mla-icon name="sparkles" :size="16"/><h3>How it works</h3></div><mla-md :src="p.explain" /></div>
              <code-sample :code="p.solution" lang="python" />
              <div class="btn-row mt"><button class="btn sm" @click="loadSolution">Copy into editor</button><button v-if="ps.accepted" class="btn sm ghost" @click="loadAccepted">Load my accepted code</button></div>
            </template>
          </template>
        </div>
        <div class="pv-nav">
          <button class="btn sm ghost" :disabled="!prev" @click="prev && go('/problem/'+prev.id)"><mla-icon name="chevronLeft" :size="14"/> Prev</button>
          <button class="btn sm ghost" @click="go('/problems')"><mla-icon name="list" :size="14"/> All problems</button>
          <button class="btn sm ghost" :disabled="!next" @click="next && go('/problem/'+next.id)">Next <mla-icon name="chevron" :size="14"/></button>
        </div>
      </div>

      <div class="pv-right">
        <div class="pv-toolbar">
          <span class="chip tiny"><span class="py-dot" :class="py.state"></span> Python {{ py.version || '' }} <span v-if="py.state==='loading'" class="dim">· loading…</span></span>
          <span class="spacer"></span>
          <button class="icon-btn" title="Copy code" @click="copyCode"><mla-icon name="copy" :size="16"/></button>
          <button class="icon-btn" title="Download as .py" @click="download"><mla-icon name="download" :size="16"/></button>
          <button class="icon-btn" title="Reset to starter code" @click="reset"><mla-icon name="refresh" :size="16"/></button>
        </div>
        <div class="pv-editor"><code-editor v-model="code" fill :error-line="errorLine" @run="run('run')" @submit="run('submit')" @save="saveDraft(); MLA.ui.toast('Draft saved in this browser.','info')" /></div>
        <div class="pv-actions">
          <span class="tiny dim hide-sm"><kbd-hint keys="Mod+Enter"/> run · <kbd-hint keys="Mod+Shift+Enter"/> submit</span>
          <span class="spacer"></span>
          <button v-if="running" class="btn sm danger" @click="stop"><mla-icon name="stop" :size="14"/> Stop</button>
          <button class="btn" :disabled="running" @click="run('run')"><mla-icon name="play" :size="15"/> Run</button>
          <button class="btn success" :disabled="running" @click="run('submit')"><mla-icon name="send" :size="15"/> Submit</button>
        </div>
        <div class="pv-console">
          <div class="pv-tabs small-tabs">
            <button :class="{active:con==='cases'}" @click="con='cases'">Test cases</button>
            <button :class="{active:con==='result'}" @click="con='result'">Result <span v-if="summary" :class="summary.kind==='ok'?'ok-text':'bad-text'">●</span></button>
            <button :class="{active:con==='output'}" @click="con='output'">Output</button>
          </div>
          <div class="pv-console-body">
            <template v-if="con==='cases'">
              <div v-for="(t,i) in visible" :key="i" class="case"><div class="tiny dim">{{ testName(t,i) }}</div><pre class="pv-code">{{ [t.setup, t.call || t.code].filter(Boolean).join('\\n') }}</pre></div>
              <div class="tiny dim" v-if="hiddenCount">Submit also runs {{ hiddenCount }} hidden test{{ hiddenCount>1?'s':'' }}.</div>
            </template>
            <template v-else-if="con==='result'">
              <div v-if="running" class="ro-status"><span class="spinner"></span> {{ py.ready ? (mode==='submit' ? 'Running all tests…' : 'Running examples…') : (py.message || 'Starting Python…') }}</div>
              <div v-else-if="failure" class="ro-error"><div class="ro-err-msg">{{ failure.kind==='timeout' ? 'Time Limit Exceeded' : failure.kind==='stopped' ? 'Stopped' : 'Could not run' }}</div><div class="small">{{ failure.message }}</div></div>
              <template v-else-if="result">
                <div class="verdict" :class="summary.kind">
                  <span class="verdict-label">{{ summary.label }}</span>
                  <span class="small">{{ summary.passed }}/{{ summary.total }} tests passed · {{ result.ms }} ms</span>
                </div>
                <div v-if="accepted" class="callout tip" style="margin-top:8px"><div class="callout-title">{{ accepted.first ? 'Solved' : 'Accepted again' }}</div>
                  <div class="btn-row"><button v-if="next" class="btn sm primary" @click="go('/problem/'+next.id)">Next problem <mla-icon name="chevron" :size="14"/></button><button class="btn sm" @click="go('/concept/'+c.id)">Back to {{ c.title }}</button></div></div>
                <div v-if="mode==='run' && summary.kind==='ok' && hiddenCount" class="small muted mt-sm">The visible tests pass. Now <b>Submit</b> to run the {{ hiddenCount }} hidden test{{ hiddenCount>1?'s':'' }} too.</div>
                <div v-if="topError" class="ro-error mt-sm">
                  <div class="ro-err-msg">{{ topError.message }}</div><pre v-if="topError.trace" class="ro-trace">{{ topError.trace }}</pre>
                  <div v-if="topHint" class="ro-hint"><mla-icon name="hint" :size="14"/> <mla-md :src="topHint" inline /></div>
                </div>
                <div v-for="(t,i) in result.tests" :key="i" class="test-row" :class="{pass:t.passed, fail:!t.passed, open: openRow===i}">
                  <button class="test-head" @click="toggleRow(i)"><span class="tr-icon">{{ t.passed ? '✓' : '✗' }}</span><span>{{ t.name }}</span><span v-if="t.hidden" class="chip tiny">hidden</span><span class="spacer"></span><span class="tiny dim">{{ t.ms }} ms</span></button>
                  <div v-if="openRow===i" class="test-body">
                    <div v-if="t.call" class="kv-line"><span class="k">Call</span><code>{{ t.call }}</code></div>
                    <div v-if="t.got != null" class="kv-line"><span class="k">Your output</span><pre>{{ t.got }}</pre></div>
                    <div v-if="t.expected != null" class="kv-line"><span class="k">Expected</span><pre>{{ t.expected }}</pre></div>
                    <div v-if="t.hint" class="ro-hint"><mla-icon name="hint" :size="14"/> {{ t.hint }}</div>
                    <div v-if="t.error" class="ro-error"><div class="ro-err-msg">{{ t.error.message }}</div><pre v-if="t.error.trace" class="ro-trace">{{ t.error.trace }}</pre><div v-if="MLA.python.explain(t.error)" class="ro-hint"><mla-icon name="hint" :size="14"/> <mla-md :src="MLA.python.explain(t.error)" inline /></div></div>
                    <div v-if="t.stdout" class="kv-line"><span class="k">Printed</span><pre>{{ t.stdout }}</pre></div>
                  </div>
                </div>
              </template>
              <div v-else class="tiny dim">Press <b>Run</b> to check your code against the examples, or <b>Submit</b> to run every test.</div>
            </template>
            <template v-else>
              <pre v-if="result && result.stdout" class="ro-stdout">{{ result.stdout }}</pre>
              <div v-else class="tiny dim">Anything your code prints at the top level appears here (prints inside tested functions appear under each test result).</div>
            </template>
          </div>
        </div>
      </div>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
