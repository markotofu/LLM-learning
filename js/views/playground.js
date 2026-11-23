/* Python playground: free-form scratch files saved in the browser, run with Pyodide */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  const TEMPLATES = [
    { name: 'Blank file', code: '# Write any Python here and press Run (Ctrl/Cmd + Enter).\nimport numpy as np\n\nprint("Hello!")\n' },
    { name: 'Game loop with delta time', code: '# A game loop in miniature: frames of different lengths, movement scaled by delta time.\nimport random\n\nrandom.seed(1)\nx, speed = 0.0, 5.0              # units, units per second\nelapsed = 0.0\nfor frame in range(1, 11):\n    dt = random.uniform(0.010, 0.040)   # this frame took 10 to 40 ms\n    x += speed * dt\n    elapsed += dt\n    print(f"frame {frame:2d}: dt = {dt * 1000:4.1f} ms, x = {x:5.3f}")\nprint(f"after {elapsed:.3f} s: x = {x:.3f}, speed x time = {speed * elapsed:.3f}")\n' },
    { name: 'Vector helpers', code: '# Small 2-D vector helpers, the Python cousins of Unity\'s Vector2.\nimport math\n\ndef sub(a, b): return (a[0] - b[0], a[1] - b[1])\ndef length(v): return math.hypot(v[0], v[1])\ndef normalized(v):\n    n = length(v)\n    return (v[0] / n, v[1] / n) if n > 0 else (0.0, 0.0)\ndef dot(a, b): return a[0] * b[0] + a[1] * b[1]\n\nplayer, enemy, facing = (0.0, 0.0), (3.0, 4.0), (1.0, 0.0)\nto_enemy = sub(enemy, player)\nprint("distance:", length(to_enemy))\nprint("direction:", normalized(to_enemy))\nprint("in front:", dot(facing, normalized(to_enemy)) > 0)\n' },
    { name: 'Plot with matplotlib', code: 'import numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.linspace(-5, 5, 200)\nplt.figure(figsize=(6, 3.5))\nplt.plot(x, 1 / (1 + np.exp(-x)), label="sigmoid")\nplt.plot(x, np.tanh(x), label="tanh")\nplt.plot(x, np.maximum(0, x), label="ReLU")\nplt.ylim(-1.5, 3)\nplt.grid(alpha=0.3)\nplt.legend()\nplt.title("Activation functions")\nplt.show()   # figures appear below the output\n' },
    { name: 'pandas: game scores', code: 'import pandas as pd\n\nruns = pd.DataFrame({\n    "player": ["Ana", "Ben", "Ana", "Cy", "Ben", "Ana"],\n    "level": [1, 1, 2, 1, 2, 3],\n    "seconds": [42.5, 51.0, 63.2, 47.9, 70.4, 88.1],\n    "coins": [12, 9, 15, 11, 14, 20],\n})\nprint(runs)\nprint()\nprint(runs.groupby("player")[["seconds", "coins"]].mean().round(1))\n' }
  ];

  MLA.components['view-playground'] = {
    data() { return { state: MLA.store.state, running: false, stdout: '', result: null, failure: null, showNew: false, renaming: null, renameVal: '', dragging: false, uploading: false }; },
    computed: {
      files() { return this.state.playground.files; },
      active() { return this.files.find(f => f.id === this.state.playground.active) || this.files[0] || null; },
      py() { return MLA.python.status; },
      templates() { return TEMPLATES; },
      ran() { return this.running || this.result || this.failure || this.stdout; },
      datasets() { this.state.datasets; return MLA.datasets.list(); }
    },
    created() { if (!this.files.length) MLA.playground.newFile('scratch.py', TEMPLATES[1].code); else if (!this.active) this.state.playground.active = this.files[0].id; },
    methods: {
      select(f) { this.state.playground.active = f.id; },
      touch() { if (this.active) this.active.updatedAt = new Date().toISOString(); },
      create(t) { MLA.playground.newFile(t.name === 'Blank file' ? 'untitled.py' : MLA.util.slug(t.name) + '.py', t.code); this.showNew = false; this.clear(); },
      startRename(f) { this.renaming = f.id; this.renameVal = f.name; this.$nextTick(() => { const el = this.$refs['rn-' + f.id]; const i = Array.isArray(el) ? el[0] : el; if (i) { i.focus(); i.select(); } }); },
      finishRename(f) { const v = (this.renameVal || '').trim(); if (v) f.name = v.slice(0, 60); this.renaming = null; },
      remove(f) { MLA.ui.confirm('Delete **' + MLA.util.escapeHtml(f.name) + '**? This cannot be undone (download it first to keep a copy).', { okLabel: 'Delete', danger: true }).then(ok => { if (!ok) return; const i = this.files.findIndex(x => x.id === f.id); if (i >= 0) this.files.splice(i, 1); if (!this.files.length) MLA.playground.newFile('untitled.py', TEMPLATES[0].code); this.state.playground.active = this.files[0].id; }); },
      download(f) { MLA.util.download(f.name.endsWith('.py') ? f.name : f.name + '.py', f.code, 'text/x-python'); },
      importPy() { this.$refs.file.click(); },
      onFile(e) { const file = e.target.files[0]; if (!file) return; if (file.size > 500000) { MLA.ui.toast('That file is too large (max 500 KB).', 'error'); return; } const r = new FileReader(); r.onload = () => { MLA.playground.newFile(file.name, String(r.result)); MLA.ui.toast('Imported ' + file.name, 'success'); }; r.readAsText(file); e.target.value = ''; },
      async run() {
        if (this.running || !this.active) return;
        this.running = true; this.stdout = ''; this.result = null; this.failure = null;
        const r = await MLA.python.exec(this.active.code, { onStdout: t => { this.stdout += t; if (this.stdout.length > 200000) this.stdout = this.stdout.slice(-150000); } });
        this.running = false;
        if (!r.ok) { this.failure = { message: r.message, kind: r.kind }; return; }
        this.result = r.result;
        this.state.stats.playgroundRuns = (this.state.stats.playgroundRuns || 0) + 1;
        MLA.gamification.touchStreak(); MLA.gamification.checkAchievements();
      },
      stop() { MLA.python.stop(); this.running = false; this.failure = { message: 'Stopped. Python was restarted.', kind: 'stopped' }; },
      clear() { this.stdout = ''; this.result = null; this.failure = null; },
      ago(f) { return MLA.util.timeAgo(f.updatedAt); },
      // ---- datasets: upload once, then read them in any code as data/<name> (js/core/datasets.js)
      pickData() { this.$refs.dataFile.click(); },
      onData(e) { const files = [...e.target.files]; e.target.value = ''; this.upload(files); },
      dropData(e) { this.dragging = false; this.upload([...((e.dataTransfer && e.dataTransfer.files) || [])]); },
      async upload(files) {
        if (!files.length) return;
        this.uploading = true;
        for (const f of files) {
          try {
            let meta;
            try { meta = await MLA.datasets.add(f); } catch (err) {
              if (!err.exists) throw err;
              if (!(await MLA.ui.confirm('**' + MLA.util.escapeHtml(f.name) + '** is already uploaded. Replace it with this file?', { okLabel: 'Replace' }))) continue;
              meta = await MLA.datasets.add(f, { replace: true });
            }
            MLA.ui.toast('Uploaded ' + meta.name + ': read it in code as data/' + meta.name, 'success', 5000);
          } catch (err) { MLA.ui.toast(err.message, 'error', 6000); }
        }
        this.uploading = false;
      },
      insertData(d) {                          // add the line that loads it to the open file
        if (!this.active) return;
        const code = this.active.code || '', lines = [];
        if (d.type !== 'txt' && !/import pandas as pd/.test(code)) lines.push('import pandas as pd');
        lines.push(MLA.datasets.readCode(d.name));
        if (d.type !== 'txt') lines.push('print(df.shape)', 'print(df.head())');
        this.active.code = code.replace(/\s*$/, '') + (code.trim() ? String.fromCharCode(10, 10) : '') + lines.join(String.fromCharCode(10)) + String.fromCharCode(10);
        this.touch();
      },
      downloadData(d) { MLA.datasets.download(d.name).catch(e => MLA.ui.toast(e.message, 'error')); },
      removeData(d) {
        MLA.ui.confirm('Delete the dataset **' + MLA.util.escapeHtml(d.name) + '** from this browser? Code that reads it will stop working until you upload it again.', { okLabel: 'Delete', danger: true })
          .then(ok => { if (ok) MLA.datasets.remove(d.name); });
      }
    },
    template: `
    <div class="playground">
      <aside class="pg-files">
        <div class="row between"><strong class="small upper dim">Your files</strong><button class="btn xs" @click="showNew=!showNew"><mla-icon name="plus" :size="12"/> New</button></div>
        <div v-if="showNew" class="pg-templates"><div class="tiny dim">Start from a template:</div><a v-for="t in templates" :key="t.name" href="#" @click.prevent="create(t)">{{ t.name }}</a></div>
        <div class="pg-list">
          <div v-for="f in files" :key="f.id" class="pg-file" :class="{active: active && f.id===active.id}" role="button" tabindex="0" :aria-current="active && f.id===active.id ? 'true' : null" :title="'Open ' + f.name + ' (F2 renames it)'" @click="select(f)" @keydown.enter.self="select(f)" @keydown.space.self.prevent="select(f)" @keydown.f2.self.prevent="startRename(f)">
            <mla-icon name="file" :size="14"/>
            <input v-if="renaming===f.id" :ref="'rn-'+f.id" v-model="renameVal" class="pg-rename" @keydown.enter="finishRename(f)" @blur="finishRename(f)" @click.stop>
            <span v-else class="pg-name" @dblclick.stop="startRename(f)">{{ f.name }}</span>
            <span class="tiny dim pg-time">{{ ago(f) }}</span>
          </div>
        </div>
        <button class="btn xs ghost block" @click="importPy"><mla-icon name="upload" :size="12"/> Import a .py file</button>
        <input ref="file" type="file" accept=".py,.txt,text/x-python,text/plain" style="display:none" @change="onFile">
        <p class="tiny dim">Files are saved in this browser and included in progress exports. Double-click a name to rename it.</p>
        <div class="pg-data" :class="{dragging}" @dragover.prevent="dragging = true" @dragleave="dragging = false" @drop.prevent="dropData">
          <div class="row between"><strong class="small upper dim">Datasets</strong><button class="btn xs" :disabled="uploading" @click="pickData"><mla-icon name="upload" :size="12"/> {{ uploading ? 'Uploading…' : 'Upload' }}</button></div>
          <div v-for="d in datasets" :key="d.name" class="pg-ds">
            <div class="pg-ds-name" :title="d.desc || d.name"><mla-icon name="file" :size="13"/> {{ d.name }}</div>
            <div class="tiny dim">{{ d.bundled ? 'built-in practice data' : MLA.util.formatBytes(d.size) }}<span v-if="d.rows"> · {{ d.rows.toLocaleString() }} rows</span></div>
            <div class="pg-ds-actions"><button class="btn xs ghost" @click="insertData(d)" :aria-label="'Use ' + d.name + ' in the open file'">Use in code</button><button v-if="!d.bundled" class="btn xs ghost" @click="downloadData(d)">Download</button><button v-if="!d.bundled" class="btn xs ghost" @click="removeData(d)" :aria-label="'Delete ' + d.name">Delete</button></div>
          </div>
          <input ref="dataFile" type="file" multiple accept=".csv,.tsv,.txt,.json,.jsonl,.xlsx,.xls,.parquet" style="display:none" @change="onData" aria-label="Upload datasets">
          <p class="tiny dim">Upload or drop CSV, Excel, JSON, Parquet or text files (up to 20 MB each). Code reads them as data/ followed by the file name. They stay in this browser and are included when you export your progress.</p>
        </div>
      </aside>
      <section class="pg-main" v-if="active">
        <div class="pv-toolbar">
          <strong class="small">{{ active.name }}</strong>
          <span class="chip tiny"><span class="py-dot" :class="py.state"></span> {{ py.ready ? 'Python ' + py.version : (py.state==='loading' ? 'Loading Python…' : 'Python loads on first run') }}</span>
          <span class="spacer"></span>
          <button class="icon-btn" title="Rename" @click="startRename(active)"><mla-icon name="note" :size="16"/></button>
          <button class="icon-btn" title="Download .py" @click="download(active)"><mla-icon name="download" :size="16"/></button>
          <button class="icon-btn" title="Delete file" @click="remove(active)"><mla-icon name="trash" :size="16"/></button>
          <button v-if="running" class="btn sm danger" @click="stop"><mla-icon name="stop" :size="14"/> Stop</button>
          <button class="btn sm primary" :disabled="running" @click="run"><mla-icon name="play" :size="14"/> Run</button>
        </div>
        <div class="pg-editor"><code-editor :key="active.id" v-model="active.code" fill @update:model-value="touch" @run="run" /></div>
        <div class="pg-output">
          <div class="row between"><span class="tiny dim upper">Output</span><button v-if="ran" class="btn xs ghost" @click="clear">Clear</button></div>
          <run-output v-if="ran" :stdout="stdout" :result="result" :failure="failure" :running="running" />
          <div v-else class="tiny dim">Press Run (or <kbd-hint keys="Mod+Enter"/>). NumPy is preloaded; pandas, scikit-learn, SciPy and matplotlib download automatically the first time you import them. The value of the last line is shown like in a notebook.</div>
        </div>
      </section>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
