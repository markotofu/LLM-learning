/* Practice problem list (filter, search, stats) */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  MLA.components['view-problems'] = {
    data() { return { state: MLA.store.state, q: '', branch: 'all', diff: 'all', stat: 'all', potd: null }; },
    created() { this.potd = MLA.progress.problemOfTheDay(); },
    computed: {
      P() { return MLA.progress; },
      branches() { return MLA.data.branches.filter(b => MLA.data.problems.some(p => this.P.concept(p.concept).branch === b.id)); },
      all() {
        const order = MLA.graph.topoOrder(MLA.content.graph); const out = []; let n = 0;
        for (const id of order) { const c = this.P.concept(id); for (const p of c.problems) out.push({ p, c, n: ++n, b: this.P.branch(c.branch), status: this.P.problemStatus(p.id), cstatus: this.P.status(c.id) }); }
        return out;
      },
      rows() {
        const q = this.q.trim().toLowerCase();
        return this.all.filter(r => (this.branch === 'all' || r.c.branch === this.branch) && (this.diff === 'all' || r.p.difficulty === this.diff) && (this.stat === 'all' || (this.stat === 'solved' ? r.status === 'solved' : this.stat === 'attempted' ? r.status === 'attempted' : r.status !== 'solved')) && (!q || r.p.title.toLowerCase().includes(q) || r.c.title.toLowerCase().includes(q) || (r.p.tags || []).some(t => t.includes(q))));
      },
      counts() { const out = {}; for (const d of ['easy', 'medium', 'hard']) { const list = this.all.filter(r => r.p.difficulty === d); out[d] = { total: list.length, solved: list.filter(r => r.status === 'solved').length }; } const s = this.all.filter(r => r.status === 'solved').length; out.all = { total: this.all.length, solved: s }; return out; }
    },
    methods: {
      go(p) { MLA.router.navigate(p); },
      random() { const pool = this.rows.filter(r => r.status !== 'solved'); const list = pool.length ? pool : this.rows; if (!list.length) return; this.go('/problem/' + list[Math.floor(Math.random() * list.length)].p.id); }
    },
    template: `
    <div class="stack">
      <section-head title="Practice Problems" sub="Real Python, graded in your browser. Every problem belongs to a concept on the map. Solve them in map order for a smooth difficulty curve.">
        <button class="btn" @click="random"><mla-icon name="dice" :size="16"/> Random unsolved</button>
      </section-head>
      <div class="grid cols-4">
        <div class="stat"><div class="label">Solved</div><div class="value">{{ counts.all.solved }}<span class="dim small"> / {{ counts.all.total }}</span></div><mla-progress :value="counts.all.solved/Math.max(1,counts.all.total)*100" thin /></div>
        <div class="stat"><div class="label diff-text easy">Easy</div><div class="value">{{ counts.easy.solved }}<span class="dim small"> / {{ counts.easy.total }}</span></div><mla-progress :value="counts.easy.solved/Math.max(1,counts.easy.total)*100" thin color="var(--green)" /></div>
        <div class="stat"><div class="label diff-text medium">Medium</div><div class="value">{{ counts.medium.solved }}<span class="dim small"> / {{ counts.medium.total }}</span></div><mla-progress :value="counts.medium.solved/Math.max(1,counts.medium.total)*100" thin color="var(--amber)" /></div>
        <div class="stat"><div class="label diff-text hard">Hard</div><div class="value">{{ counts.hard.solved }}<span class="dim small"> / {{ counts.hard.total }}</span></div><mla-progress :value="counts.hard.solved/Math.max(1,counts.hard.total)*100" thin color="var(--red)" /></div>
      </div>
      <div v-if="potd" class="card clickable potd" role="link" tabindex="0" @click="go('/problem/'+potd.problem.id)" @keydown.enter="go('/problem/'+potd.problem.id)">
        <div class="row between"><div><div class="tiny dim upper">Problem of the day {{ potd.done ? '· done' : '· +30 bonus XP' }}</div><div style="font-weight:650;font-size:1.05rem">{{ potd.problem.title }}</div><div class="small muted">{{ P.concept(potd.problem.concept).title }}</div></div><diff-chip :d="potd.problem.difficulty"/></div>
      </div>
      <div class="filters">
        <div class="map-search grow"><mla-icon name="search" :size="15"/><input type="search" v-model="q" placeholder="Search problems or concepts…" aria-label="Search problems"></div>
        <select v-model="branch" aria-label="Branch"><option value="all">All branches</option><option v-for="b in branches" :key="b.id" :value="b.id">{{ b.name }}</option></select>
        <select v-model="diff" aria-label="Difficulty"><option value="all">Any difficulty</option><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select>
        <select v-model="stat" aria-label="Status"><option value="all">Any status</option><option value="todo">Unsolved</option><option value="attempted">Attempted</option><option value="solved">Solved</option></select>
      </div>
      <div class="problem-table">
        <div class="pt-head"><span>#</span><span></span><span>Problem</span><span class="hide-sm">Concept</span><span>Difficulty</span></div>
        <a v-for="r in rows" :key="r.p.id" class="pt-row" :href="'#/problem/'+r.p.id" @click.prevent="go('/problem/'+r.p.id)">
          <span class="tiny dim mono">{{ r.n }}</span>
          <status-dot :status="r.status" :size="18"/>
          <span class="pt-title">{{ r.p.title }}<span v-if="r.cstatus==='locked'" class="tiny dim"> · prerequisites recommended</span></span>
          <span class="hide-sm small"><span class="branch-dot" :style="{'--bc': r.b.color}"></span> {{ r.c.title }}</span>
          <diff-chip :d="r.p.difficulty"/>
        </a>
        <empty-state v-if="!rows.length" icon="search" text="No problems match these filters." />
      </div>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
