/* ML Atlas – search everything: pages, lessons, sections, glossary terms, problems, projects, case studies and tracks,
 * from one box (Ctrl+K, Cmd+K or "/", or the search button in the top bar). Titles come from the lesson catalog, so
 * results appear at once; when the query is three letters or more, the lesson files load and their text is searched too.
 *   MLA.search.open() / close()     MLA.search.query(text) -> [{ kind, title, sub, route }]   (used by tests) */
(function (global) {
  const MLA = global.MLA;
  const reactive = global.Vue && global.Vue.reactive ? global.Vue.reactive : (x => x);
  const ui = reactive({ open: false, version: 0, loadingText: false });

  const strip = md => String(md || '').replace(/```[\s\S]*?```/g, ' ').replace(/:::where[\s\S]*?:::/g, ' ').replace(/[`*_#>|[\]()$]/g, ' ').replace(/\s+/g, ' ').trim();
  const lower = s => String(s || '').toLowerCase();
  let index = null, built = -1;

  function build() {
    const out = [];
    const add = (kind, weight, title, sub, route, extra) => out.push({ kind, weight, title, sub, route, tl: lower(title), hay: lower(title + ' ' + (sub || '') + ' ' + (extra || '')) });
    const branchName = id => { const b = MLA.content.branchById && MLA.content.branchById[id]; return b ? b.name : ''; };
    for (const n of MLA.nav || []) add('Page', 6, n.label, '', n.path);
    for (const t of (MLA.data.tracks || []).filter(t => !t.hidden)) add('Track', 7, t.title, t.desc || t.description || '', '/track/' + t.id);
    for (const c of MLA.data.concepts) {
      add('Lesson', 22, c.title, c.summary, '/concept/' + c.id, (c.tags || []).join(' ') + ' ' + branchName(c.branch));
      for (const s of c.sections) {
        if (s.title && s.type !== 'check') add('Section', 9, s.title, c.title, '/concept/' + c.id + '?s=' + s.id);
        if (!c._stub && (s.md || s.after)) out.push({ kind: 'Text', weight: 2, title: s.title || c.title, sub: c.title, route: '/concept/' + c.id + '?s=' + s.id, tl: '', hay: lower(strip(s.md) + ' ' + strip(s.after)), text: strip(s.md) + ' ' + strip(s.after) });
      }
      for (const p of c.problems) add('Problem', 11, p.title, c.title + ' · ' + p.difficulty, '/problem/' + p.id);
      for (const p of c.projects || []) add('Project', 10, p.title, c.title + ' · ' + p.level, '/concept/' + c.id + '?s=projects', p.goal + ' ' + (p.skills || []).join(' '));
      for (const k of c.cases || []) add('Case study', 10, k.title, c.title + ' · ' + k.when, '/concept/' + c.id + '?s=cases', k.story);
    }
    for (const g of MLA.data.glossary || []) add('Term', 12, g.term, strip(g.def), g.concept && MLA.content.byId[g.concept] ? '/concept/' + g.concept : '/glossary');
    return out;
  }

  function snippet(text, words) {
    const t = lower(text);
    let i = -1;
    for (const w of words) { i = t.indexOf(w); if (i >= 0) break; }
    if (i < 0) return '';
    const a = Math.max(0, i - 50), b = Math.min(text.length, i + 90);
    return (a > 0 ? '…' : '') + text.slice(a, b).trim() + (b < text.length ? '…' : '');
  }

  function query(q) {
    q = lower(q).trim();
    const words = q.split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    const stamp = ui.version + MLA.data.concepts.filter(c => !c._stub).length * 1000;
    if (!index || built !== stamp) { index = build(); built = stamp; }
    const hits = [];
    for (const e of index) {
      if (!words.every(w => e.hay.includes(w))) continue;
      let score = e.weight;
      if (e.tl === q) score += 120; else if (e.tl.startsWith(q)) score += 80; else if (e.tl.includes(q)) score += 50;
      for (const w of words) if (e.tl.includes(w)) score += 15;
      hits.push({ e, score });
    }
    hits.sort((x, y) => y.score - x.score);
    const seen = new Set();
    const out = [];
    for (const { e } of hits) {
      const key = e.kind + e.route + e.title;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ kind: e.kind === 'Text' ? 'In the text' : e.kind, title: e.title, sub: e.kind === 'Text' ? e.sub + ': ' + snippet(e.text, words) : e.sub, route: e.route });
      if (out.length >= 40) break;
    }
    return out;
  }

  let lastFocus = null;
  function open() { lastFocus = global.document && document.activeElement; ui.open = true; }
  function close() { ui.open = false; if (lastFocus && lastFocus.focus) try { lastFocus.focus(); } catch (e) { /* gone */ } }
  function loadText() {
    if (ui.loadingText || !MLA.data.concepts.some(c => c._stub)) return;
    ui.loadingText = true;
    MLA.content.ensureAll().then(() => { ui.loadingText = false; ui.version++; }, () => { ui.loadingText = false; });
  }

  MLA.search = { ui, open, close, query };

  MLA.components['mla-search'] = {
    data() { return { ui, q: '', sel: 0 }; },
    computed: {
      results() { this.ui.version; return query(this.q); },
      status() {
        if (this.q.trim().length < 3 || !MLA.data.concepts.some(c => c._stub)) return this.results.length ? '' : (this.q.trim() ? 'Nothing matches.' : 'Search lessons, sections, terms, problems, projects and case studies.');
        return this.ui.loadingText ? 'Loading the lesson text to search inside it…' : '';
      }
    },
    watch: {
      'ui.open'(v) { if (v) { this.sel = 0; this.$nextTick(() => { const el = this.$refs.input; if (el) { el.focus(); el.select(); } }); } },
      q(v) { this.sel = 0; if (v.trim().length >= 3) loadText(); }
    },
    methods: {
      move(d) { const n = this.results.length; if (!n) return; this.sel = (this.sel + d + n) % n; this.$nextTick(() => { const el = document.getElementById('sr-' + this.sel); if (el) el.scrollIntoView({ block: 'nearest' }); }); },
      choose(i) { const r = this.results[i == null ? this.sel : i]; if (!r) return; close(); MLA.router.navigate(r.route); }
    },
    template: `
    <div v-if="ui.open" class="search-overlay" @click.self="MLA.search.close()">
      <div class="search-box" role="dialog" aria-modal="true" aria-label="Search">
        <div class="search-input">
          <mla-icon name="search" :size="18"/>
          <input ref="input" v-model="q" type="search" placeholder="Search lessons, sections, terms, problems, projects…" aria-label="Search" role="combobox" aria-expanded="true" aria-controls="search-results" :aria-activedescendant="results.length ? 'sr-' + sel : null"
            @keydown.down.prevent="move(1)" @keydown.up.prevent="move(-1)" @keydown.enter.prevent="choose()" @keydown.esc.prevent="MLA.search.close()">
          <kbd class="hide-sm">Esc</kbd>
        </div>
        <ul id="search-results" class="search-results" role="listbox" aria-label="Results">
          <li v-for="(r, i) in results" :id="'sr-' + i" :key="r.kind + r.route + r.title" role="option" :aria-selected="i === sel" :class="{ sel: i === sel }" @mousemove="sel = i" @click="choose(i)">
            <span class="chip tiny">{{ r.kind }}</span>
            <span class="sr-text"><span class="sr-title">{{ r.title }}</span><span v-if="r.sub" class="tiny dim sr-sub">{{ r.sub }}</span></span>
          </li>
        </ul>
        <div class="search-foot tiny dim"><span>{{ status }}</span><span class="hide-sm">Up and down to move, Enter to open, Esc to close</span></div>
      </div>
    </div>` };
})(typeof globalThis !== 'undefined' ? globalThis : window);
