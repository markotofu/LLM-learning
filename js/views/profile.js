/* ML Atlas – Progress & Sync (profile, settings, export / import / sync codes) and the developer self-test */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  function codeBundle() {
    const s = MLA.store.state; const P = MLA.progress; const L = [];
    const bar = '# ' + '='.repeat(72);
    L.push('# ML Atlas: all my code (exported ' + new Date().toISOString().slice(0, 16).replace('T', ' ') + ')');
    L.push('# Practice problems (your latest draft, plus your last accepted solution if it differs), then playground files.');
    L.push('# This file is for reading/keeping. To move progress between devices use "Export progress" instead.');
    L.push('');
    for (const id of MLA.graph.topoOrder(MLA.content.graph)) {
      const c = P.concept(id);
      for (const p of c.problems) {
        const st = s.problems[p.id]; if (!st || (st.draft == null && !st.accepted)) continue;
        L.push(bar, '# Problem: ' + p.title + '  [' + p.difficulty + ']  ·  concept: ' + c.title, '# Status: ' + (st.solved ? 'solved' + (st.assisted ? ' (with solution)' : '') : 'in progress') + '   id: ' + p.id, bar);
        L.push((st.draft != null ? st.draft : st.accepted.code).replace(/\s+$/, ''), '');
        if (st.accepted && st.draft != null && st.accepted.code !== st.draft) L.push('# --- last accepted solution ---', st.accepted.code.replace(/\s+$/, ''), '');
      }
    }
    for (const f of s.playground.files) { L.push(bar, '# Playground file: ' + f.name, bar, (f.code || '').replace(/\s+$/, ''), ''); }
    return L.join('\n') + '\n';
  }

  MLA.components['view-profile'] = {
    data() { return { state: MLA.store.state, nameEdit: MLA.store.state.profile.name, mode: 'merge', syncOut: '', syncIn: '', busy: false, speech: MLA.speech.state }; },
    computed: {
      G() { return MLA.gamification; }, P() { return MLA.progress; },
      level() { return this.G.levelInfo(); }, streak() { return this.G.streakStatus(); },
      overall() { return this.P.overall(); },
      branches() { return this.P.branchStats().filter(b => b.branch.id !== 'start'); },
      theme() { return this.state.profile.theme || 'light'; },
      look() { return this.state.profile.style === 'classic' ? 'classic' : 'baroque'; },
      themeOptions() {
        return this.look === 'classic'
          ? [{ id: 'light', label: 'Paper', hint: 'Light' }, { id: 'dark', label: 'Graphite', hint: 'Dark' }, { id: 'darker', label: 'Midnight', hint: 'Near black' }]
          : [{ id: 'light', label: 'Fresco', hint: 'Parchment, lapis and gold' }, { id: 'dark', label: 'Vault', hint: 'Dark umber and gold' }, { id: 'darker', label: 'Lapis', hint: 'Night blue and gold' }];
      },
      storage() { this.state.xp; return MLA.store.storageInfo(); },
      py() { return MLA.python.status; },
      tracks() { return this.P.tracks().filter(t => !t.hidden); },
      datasetCount() { return Object.keys(this.state.datasets || {}).length; },
      codeCount() { return Object.values(this.state.problems).filter(p => p.draft != null || p.accepted).length + this.state.playground.files.length; }
    },
    methods: {
      saveName() { this.state.profile.name = (this.nameEdit || 'Learner').trim().slice(0, 24) || 'Learner'; MLA.ui.toast('Name updated.', 'success'); },
      setTheme(t) { this.state.profile.theme = t; MLA.ui.applyTheme(t, this.state.profile.style); },
      setLook(s) { this.state.profile.style = s; MLA.ui.applyTheme(this.state.profile.theme, s); },
      setRate(r) { MLA.speech.setRate(Number(r)); },
      trySpeech() { MLA.speech.sample(); },
      async exportFile() {                     // everything, including the contents of uploaded datasets
        this.state.stats.exports = (this.state.stats.exports || 0) + 1; MLA.gamification.checkAchievements();
        const name = 'ml-atlas-progress-' + MLA.util.todayKey() + '.json';
        const text = await MLA.store.exportFull();
        if (MLA.util.download(name, text)) MLA.ui.toast('Saved ' + name + ' (' + MLA.util.formatBytes(text.length) + '): import it on your other device.', 'success', 4500);
      },
      exportCode() { MLA.util.download('ml-atlas-code-' + MLA.util.todayKey() + '.py', codeBundle(), 'text/x-python'); },
      pickFile() { this.$refs.file.click(); },
      async confirmReplace() { if (this.mode !== 'replace') return true; return MLA.ui.confirm('**Replace** erases the progress currently in this browser and uses the imported data instead. Use **Merge** to combine both. Continue?', { okLabel: 'Replace', danger: true }); },
      afterImport(label) { this.nameEdit = this.state.profile.name; MLA.ui.applyTheme(this.state.profile.theme, this.state.profile.style); MLA.ui.toast(label + (this.mode === 'merge' ? ' and merged with this device.' : '.'), 'success', 4500); MLA.gamification.checkAchievements(); },
      onFile(e) {
        const f = e.target.files[0]; e.target.value = ''; if (!f) return;
        const r = new FileReader();
        r.onload = async () => { if (!(await this.confirmReplace())) return; try { MLA.store.importJSON(String(r.result), this.mode); const n = await MLA.store.datasetImport(); this.afterImport('Progress imported' + (n ? ' with ' + n + (n > 1 ? ' datasets' : ' dataset') : '')); } catch (err) { MLA.ui.toast('Import failed: ' + err.message, 'error', 6000); } };
        r.readAsText(f);
      },
      async makeSync() {
        this.busy = true;
        try { this.syncOut = await MLA.store.exportSyncCode(); this.state.stats.exports = (this.state.stats.exports || 0) + 1; const ok = await MLA.util.copyText(this.syncOut); MLA.ui.toast(ok ? 'Sync code copied (' + MLA.util.formatBytes(this.syncOut.length) + '). Paste it on your other device.' : 'Sync code created: select and copy it below.', 'success', 5000); }
        catch (e) { MLA.ui.toast('Could not create a sync code: ' + e.message, 'error'); }
        this.busy = false;
      },
      async useSync() {
        if (!this.syncIn.trim()) return;
        if (!(await this.confirmReplace())) return;
        this.busy = true;
        try { await MLA.store.importSyncCode(this.syncIn, this.mode); this.syncIn = ''; this.afterImport('Sync code applied'); }
        catch (e) { MLA.ui.toast('That sync code did not work: ' + e.message, 'error', 6000); }
        this.busy = false;
      },
      reset() { MLA.ui.confirm('This erases **all** progress, code, notes and settings in this browser. It cannot be undone. Export a backup first if you might want it back.', { okLabel: 'Erase everything', danger: true }).then(ok => { if (ok) { MLA.store.reset(); this.nameEdit = this.state.profile.name; MLA.ui.applyTheme('dark'); MLA.ui.toast('All progress erased.', 'success'); MLA.router.navigate('/home'); } }); },
      loadPy() { MLA.python.warmup(); },
      restartPy() { MLA.python.stop(); MLA.ui.toast('Python stopped. It restarts on the next run.', 'info'); },
      go(p) { MLA.router.navigate(p); }
    },
    template: `
    <div class="stack">
      <section-head title="Progress & Sync" :sub="'Learning since ' + MLA.util.fmtDate(state.profile.createdAt) + ' · everything is stored in this browser'" />
      <div class="grid cols-3">
        <div class="card" style="grid-column:span 2">
          <div class="card-title"><h3>Profile</h3></div>
          <div class="row" style="gap:20px;align-items:flex-start">
            <div class="center"><score-ring :value="level.pct" :size="110" color="var(--accent)" /><div class="small muted mt-sm">Level {{ level.level }} · {{ G.title(level.level) }}</div></div>
            <div style="flex:1;min-width:220px">
              <div class="field"><label>Display name</label><div class="inline-form"><input type="text" v-model="nameEdit" maxlength="24" aria-label="Display name" @keyup.enter="saveName"/><button class="btn sm" @click="saveName">Save</button></div></div>
              <div class="field"><label>Learning goal</label><select v-model="state.profile.goal" aria-label="Learning goal"><option :value="null">No goal: follow the whole map</option><option v-for="t in tracks" :key="t.id" :value="t.id">{{ t.title }}</option><option v-if="state.profile.goal==='custom'" value="custom">{{ P.track('custom').title }}</option></select></div>
              <div class="small"><div class="row between"><span class="muted">XP</span><b>{{ state.xp.toLocaleString() }}</b></div><div class="row between"><span class="muted">Concepts completed</span><b>{{ overall.conceptsDone }} / {{ overall.concepts }}</b></div><div class="row between"><span class="muted">Problems solved</span><b>{{ overall.solved }} / {{ overall.problems }}</b></div><div class="row between"><span class="muted">Best streak</span><b>{{ streak.best }} days</b></div></div>
            </div>
          </div>
        </div>
        <div class="card center"><div class="card-title"><h3>Skill radar</h3></div><branch-radar :items="branches" :size="250" /></div>
      </div>

      <div class="card sync-card">
        <div class="card-title"><h3><mla-icon name="sync" :size="18"/> Move your progress between devices</h3><span class="tiny dim">{{ MLA.util.formatBytes(storage.bytes) }} stored · {{ codeCount }} code files<span v-if="datasetCount"> · {{ datasetCount }} datasets ({{ MLA.util.formatBytes(MLA.datasets.usage()) }})</span></span></div>
        <p class="small muted">Your progress, settings, code drafts, accepted solutions, lesson notes, project checklists and notes, playground files, uploaded datasets and review schedules live in this browser. To continue on another device, export here and import there. <b>Merge</b> (recommended) combines both devices and never loses work: best scores, every solved problem and the newest version of each draft are kept.</p>
        <div class="field"><label>When importing</label><div class="pills"><button :class="{active:mode==='merge'}" @click="mode='merge'">Merge with this device</button><button :class="{active:mode==='replace'}" @click="mode='replace'">Replace this device's data</button></div></div>
        <div class="grid cols-2">
          <div class="card pad-sm">
            <h4>Option 1 · Progress file</h4>
            <p class="small muted">Download a <code>.json</code> file with everything, uploaded datasets included, send it to your other device (email, cloud drive, USB), and import it there.</p>
            <div class="btn-row"><button class="btn primary" @click="exportFile"><mla-icon name="download" :size="16"/> Export progress</button><button class="btn" @click="pickFile"><mla-icon name="upload" :size="16"/> Import a file</button></div>
            <input ref="file" type="file" accept="application/json,.json" style="display:none" @change="onFile">
          </div>
          <div class="card pad-sm">
            <h4>Option 2 · Sync code (copy &amp; paste)</h4>
            <p class="small muted">A compressed text version of everything except uploaded datasets (they would make it far too long to paste). Copy it here, paste it into this box on the other device, for example through a note to yourself.</p>
            <div class="btn-row"><button class="btn primary" :disabled="busy" @click="makeSync"><mla-icon name="copy" :size="16"/> Create &amp; copy sync code</button></div>
            <textarea v-if="syncOut" class="sync-box mono" readonly :value="syncOut" aria-label="Your sync code" @focus="$event.target.select()"></textarea>
            <textarea v-model="syncIn" class="sync-box mono" aria-label="Sync code from another device" placeholder="Paste a sync code from another device here…"></textarea>
            <button class="btn" :disabled="busy || !syncIn.trim()" @click="useSync"><mla-icon name="upload" :size="16"/> Apply sync code</button>
          </div>
        </div>
        <div class="row between mt"><div class="small muted">Just want your code? Download every draft, solution and playground file as one readable Python file.</div><button class="btn sm" @click="exportCode"><mla-icon name="file" :size="14"/> Export all my code (.py)</button></div>
        <div v-if="storage.error" class="callout danger mt"><div class="callout-title">Storage problem</div>This browser refused to save ({{ storage.error }}). Your work is only in memory: export it now.</div>
      </div>

      <div class="grid cols-2">
        <div class="card">
          <div class="card-title"><h3><mla-icon name="settings" :size="18"/> Settings</h3></div>
          <div class="field"><label>Look</label><div class="pills"><button :class="{active:look==='baroque'}" @click="setLook('baroque')" title="After baroque ceiling frescoes: gold, clouds and serif type">Baroque</button><button :class="{active:look==='classic'}" @click="setLook('classic')" title="The original design: clean and minimal">Classic</button></div></div>
          <div class="field"><label>Theme</label><div class="pills"><button v-for="o in themeOptions" :key="o.id" :class="{active:theme===o.id}" @click="setTheme(o.id)" :title="o.hint">{{ o.label }}</button></div></div>
          <div class="field"><label>Read-aloud voice</label>
            <select v-model="state.profile.voice" :disabled="!speech.supported || !speech.voices.length" aria-label="Read-aloud voice"><option value="">Automatic (an English voice on this device)</option><option v-for="v in speech.voices" :key="v.uri" :value="v.uri">{{ v.name }} ({{ v.lang }})</option></select>
            <div class="tiny dim mt-sm">{{ !speech.supported ? 'This browser cannot read text aloud.' : speech.voices.length ? 'Only voices installed on this device are offered, so lesson text never leaves it.' : 'No voice is installed on this device for the browser to use.' }}</div>
          </div>
          <div class="field"><label>Reading speed: {{ state.profile.speechRate || 1 }}×</label><input type="range" min="0.5" max="2" step="0.25" :value="state.profile.speechRate || 1" @input="setRate($event.target.value)" aria-label="Reading speed"></div>
          <label class="checkbox"><input type="checkbox" v-model="state.profile.speechFollow"> Scroll to the part being read</label>
          <label class="checkbox"><input type="checkbox" :checked="state.profile.speechCaptions !== false" @change="state.profile.speechCaptions = $event.target.checked"> Show the sentence being read (captions in the player, highlight in the lesson)</label>
          <label class="checkbox"><input type="checkbox" v-model="state.profile.speechContinue"> Go on to the next lesson automatically when one ends</label>
          <p class="tiny dim">Lessons remember where you stopped. In browsers that support media controls for web pages, the lock screen, notification area and headphone buttons can pause, resume and skip; phones may still stop speech when the screen locks.</p>
          <div class="btn-row mt-sm"><button class="btn sm" :disabled="!speech.supported" @click="trySpeech"><mla-icon name="volume" :size="15"/> Try the voice</button></div>
          <div class="field"><label>Code editor font size: {{ state.profile.editorFontSize }}px</label><input type="range" min="11" max="22" step="1" v-model.number="state.profile.editorFontSize" aria-label="Code editor font size"></div>
          <div class="field"><label>Time limit per run: {{ state.profile.timeoutSec }} s (stops infinite loops)</label><input type="range" min="5" max="120" step="5" v-model.number="state.profile.timeoutSec" aria-label="Time limit per run, in seconds"></div>
          <label class="checkbox"><input type="checkbox" v-model="state.profile.preloadPython"> Load Python in the background when the app opens</label>
        </div>
        <div class="card">
          <div class="card-title"><h3><mla-icon name="cpu" :size="18"/> Python runtime</h3><span class="chip tiny"><span class="py-dot" :class="py.state"></span> {{ py.state }}</span></div>
          <p class="small muted">Pyodide {{ MLA.python.VERSION }}: CPython 3.12 compiled to WebAssembly, running in a background worker. Loaded from cdn.jsdelivr.net on first use and cached for offline use. {{ py.message }}</p>
          <div class="btn-row"><button class="btn sm" :disabled="py.ready || py.state==='loading'" @click="loadPy">{{ py.ready ? 'Loaded' : 'Download now' }}</button><button class="btn sm ghost" @click="restartPy">Restart</button><button class="btn sm ghost" @click="go('/selftest')">Run self-test</button></div>
        </div>
      </div>

      <div class="card danger-card">
        <div class="card-title"><h3 style="color:var(--red)">Danger zone</h3></div>
        <div class="row between"><div><strong class="small">Erase all progress</strong><div class="small muted">Removes everything stored by ML Atlas in this browser.</div></div><button class="btn danger" @click="reset"><mla-icon name="trash" :size="16"/> Erase</button></div>
      </div>

      <div class="tiny dim center">{{ MLA.data.legal.copyright() }} · <a href="#/terms">Terms of Use</a> · <a href="#/privacy">Privacy Policy</a> · your data never leaves this browser unless you export it</div>
    </div>`
  };

  // Developer self-test: every reference solution must pass, every starter must fail, every runnable sample must run.
  MLA.components['view-selftest'] = {
    data() { return { running: false, rows: [], samples: [], done: 0, total: 0, only: 'fail' }; },
    computed: {
      problems() { return MLA.data.problems; },
      bad() { return this.rows.filter(r => !r.solOk || !r.starterFails).length + this.samples.filter(s => !s.ok).length; },
      shown() { return this.only === 'fail' ? this.rows.filter(r => !r.solOk || !r.starterFails) : this.rows; },
      shownSamples() { return this.only === 'fail' ? this.samples.filter(s => !s.ok) : this.samples; }
    },
    methods: {
      firstFail(res) { if (!res.ok) return res.kind + ': ' + res.message; const r = res.result; if (r.error) return r.error.message; const t = r.tests.find(x => !x.passed); return t ? t.name + ': got ' + t.got + ' expected ' + t.expected + (t.error ? ': ' + t.error.message : '') : ''; },
      async runProblems() {
        this.running = true; this.rows = []; this.done = 0; this.total = this.problems.length;
        for (const p of this.problems) {
          const a = await MLA.python.runTests(p.solution, p, 'submit', { timeoutMs: 90000 });
          const solOk = a.ok && !a.result.error && a.result.tests.length > 0 && a.result.tests.every(t => t.passed);
          const b = await MLA.python.runTests(p.starter, p, 'submit', { timeoutMs: 30000 });
          const starterFails = !b.ok || !!b.result.error || b.result.tests.some(t => !t.passed);
          this.rows.push({ p, solOk, starterFails, detail: solOk ? '' : this.firstFail(a), ms: a.ok ? a.result.ms : null });
          this.done++;
        }
        this.running = false;
      },
      async runSamples() {
        this.running = true; this.samples = [];
        const list = [];
        for (const c of MLA.data.concepts) c.sections.forEach((s, i) => { if (s.type === 'code' && s.run !== false && (!s.lang || s.lang === 'python')) list.push({ c, s, i }); });
        this.done = 0; this.total = list.length;
        for (const it of list) {
          let out = '';
          const r = await MLA.python.exec(it.s.code, { timeoutMs: Math.max(90, 2 * (it.s.timeoutSec || 0)) * 1000, onStdout: t => { out += t; } });
          if (r.ok && r.result && r.result.value) out += String.fromCharCode(10) + r.result.value;
          // A "predict first" question names output text that confirms its answer: check it in this runtime too.
          const missing = r.ok && !r.result.error && it.s.predict ? [].concat(it.s.predict.expect || []).filter(e => !out.includes(e)) : [];
          this.samples.push({ key: it.c.id + '#' + it.i, title: it.c.title + ': ' + (it.s.title || 'sample'), ok: r.ok && !r.result.error && !missing.length, detail: r.ok ? (r.result.error ? r.result.error.message : missing.length ? 'the output no longer confirms the prediction answer: ' + missing.join(' | ') : '') : r.message });
          this.done++;
        }
        this.running = false;
      }
    },
    template: `
    <div class="stack">
      <section-head title="Self-test" sub="Runs every reference solution (must pass all tests), every starter (must fail at least one) and every runnable code sample in the in-browser Python. Useful after editing content." />
      <div class="btn-row"><button class="btn primary" :disabled="running" @click="runProblems">Test {{ problems.length }} problems</button><button class="btn" :disabled="running" @click="runSamples">Run code samples</button>
        <div class="pills"><button :class="{active:only==='fail'}" @click="only='fail'">Failures only</button><button :class="{active:only==='all'}" @click="only='all'">Everything</button></div></div>
      <div v-if="total" class="small">{{ done }} / {{ total }} {{ running ? '· running…' : '· finished' }} · <b :class="bad ? 'bad-text' : 'ok-text'">{{ bad }} problem(s) found</b></div>
      <div class="problem-table">
        <div v-for="r in shown" :key="r.p.id" class="pt-row" style="grid-template-columns: 26px 1fr auto">
          <span>{{ r.solOk && r.starterFails ? '✓' : '✗' }}</span>
          <span><b>{{ r.p.id }}</b> <span class="tiny dim">{{ r.ms }} ms</span><div v-if="!r.solOk" class="tiny bad-text">solution: {{ r.detail }}</div><div v-if="!r.starterFails" class="tiny bad-text">starter passes every test (tests too weak)</div></span>
          <a class="tiny" :href="'#/problem/'+r.p.id" @click.prevent="MLA.router.navigate('/problem/'+r.p.id)">open</a>
        </div>
        <div v-for="s in shownSamples" :key="s.key" class="pt-row" style="grid-template-columns: 26px 1fr">
          <span>{{ s.ok ? '✓' : '✗' }}</span><span><b>{{ s.title }}</b><div v-if="!s.ok" class="tiny bad-text">{{ s.detail }}</div></span>
        </div>
      </div>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
