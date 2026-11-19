/* ML Atlas – lightweight code editor (textarea + highlighted overlay), run output panel, runnable code samples */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};

  const coarse = () => { try { return global.matchMedia && matchMedia('(pointer: coarse)').matches; } catch (e) { return false; } };

  MLA.components['code-editor'] = {
    props: { modelValue: { type: String, default: '' }, readonly: Boolean, minLines: { default: 6 }, maxLines: { default: 0 }, fill: Boolean, errorLine: { default: null }, label: { default: 'Python code editor' } },
    emits: ['update:modelValue', 'run', 'submit', 'save'],
    data() { return { escapeTab: false, focused: false, showKeys: coarse() }; },
    computed: {
      fs() { return Number(MLA.store.state.profile.editorFontSize) || 14; },
      lh() { return Math.round(this.fs * 1.55); },
      code() { return this.modelValue || ''; },
      lineCount() { return this.code.split('\n').length; },
      html() { return MLA.highlight.python(this.code) + '\n\n\n'; },
      lineNums() { const a = []; for (let i = 1; i <= this.lineCount; i++) a.push(i); return a; },
      boxStyle() {
        const st = { fontSize: this.fs + 'px', lineHeight: this.lh + 'px' };
        if (!this.fill) { const n = Math.max(this.minLines, this.maxLines ? Math.min(this.maxLines, this.lineCount) : this.lineCount); st.height = (n * this.lh + 24 + 16) + 'px'; }
        return st;
      }
    },
    watch: { modelValue(v) { const ta = this.$refs.ta; if (ta && ta.value !== (v || '')) ta.value = v || ''; this.$nextTick(this.sync); } },
    mounted() { this.$refs.ta.value = this.code; this.sync(); },
    methods: {
      sync() { const ta = this.$refs.ta, pre = this.$refs.pre, g = this.$refs.gutter; if (!ta || !pre || !g) return; pre.scrollTop = ta.scrollTop; pre.scrollLeft = ta.scrollLeft; g.scrollTop = ta.scrollTop; },
      onInput(e) { this.$emit('update:modelValue', e.target.value); this.$nextTick(this.sync); },
      focus() { if (this.$refs.ta) this.$refs.ta.focus(); },
      replaceRange(start, end, text) {
        const ta = this.$refs.ta; ta.focus(); ta.setSelectionRange(start, end);
        let ok = false;
        try { ok = !!(document.execCommand && document.execCommand(text ? 'insertText' : 'delete', false, text || null)); } catch (err) { ok = false; }
        if (!ok || ta.value.slice(start, start + text.length) !== text) {
          if (!ok) { ta.setRangeText(text, start, end, 'end'); ta.dispatchEvent(new Event('input', { bubbles: true })); }
        }
      },
      lineBlock() {
        const ta = this.$refs.ta; const s = ta.selectionStart, en = ta.selectionEnd, v = ta.value;
        const ls = v.lastIndexOf('\n', s - 1) + 1;
        let be = en; if (en > s && v[en - 1] === '\n') be = en - 1;
        const le = v.indexOf('\n', be);
        return { s, en, v, ls, le: le === -1 ? v.length : le };
      },
      indent(dir) {
        const ta = this.$refs.ta; const b = this.lineBlock();
        if (dir > 0 && b.s === b.en) { const col = b.s - b.ls; this.replaceRange(b.s, b.en, ' '.repeat(4 - (col % 4))); return; }
        const lines = b.v.slice(b.ls, b.le).split('\n');
        const out = lines.map(l => dir > 0 ? '    ' + l : l.replace(/^ {1,4}/, '')).join('\n');
        if (out === b.v.slice(b.ls, b.le)) return;
        this.replaceRange(b.ls, b.le, out);
        ta.setSelectionRange(b.ls, b.ls + out.length);
      },
      toggleComment() {
        const ta = this.$refs.ta; const b = this.lineBlock();
        const lines = b.v.slice(b.ls, b.le).split('\n');
        const filled = lines.filter(l => l.trim());
        if (!filled.length) return;
        const all = filled.every(l => /^\s*#/.test(l));
        const minIndent = Math.min(...filled.map(l => l.match(/^ */)[0].length));
        const out = lines.map(l => !l.trim() ? l : all ? l.replace(/^(\s*)# ?/, '$1') : l.slice(0, minIndent) + '# ' + l.slice(minIndent)).join('\n');
        this.replaceRange(b.ls, b.le, out);
        ta.setSelectionRange(b.ls, b.ls + out.length);
      },
      insertText(t) { const ta = this.$refs.ta; this.replaceRange(ta.selectionStart, ta.selectionEnd, t); },
      onKey(e) {
        const ta = this.$refs.ta;
        const mod = e.ctrlKey || e.metaKey;
        if (mod && e.key === 'Enter') { e.preventDefault(); this.$emit(e.shiftKey ? 'submit' : 'run'); return; }
        if (mod && (e.key === 's' || e.key === 'S')) { e.preventDefault(); this.$emit('save'); return; }
        if (this.readonly) return;
        if (e.key === 'Escape') { this.escapeTab = true; return; }
        if (e.key === 'Tab') {
          if (this.escapeTab) { this.escapeTab = false; return; }
          e.preventDefault(); this.indent(e.shiftKey ? -1 : 1); return;
        }
        this.escapeTab = false;
        if (mod && e.key === '/') { e.preventDefault(); this.toggleComment(); return; }
        if (e.key === 'Enter' && !e.shiftKey && !mod && !e.altKey && !e.isComposing) {
          e.preventDefault();
          const s = ta.selectionStart, en = ta.selectionEnd, v = ta.value;
          const ls = v.lastIndexOf('\n', s - 1) + 1;
          const before = v.slice(ls, s);
          let ind = before.match(/^ */)[0];
          const trimmed = before.replace(/\s+$/, '');
          if (/[:([{]$/.test(trimmed)) ind += '    ';
          else if (/^\s*(return|pass|break|continue|raise)\b/.test(before) && ind.length >= 4) ind = ind.slice(4);
          this.replaceRange(s, en, '\n' + ind);
          return;
        }
        if (e.key === 'Backspace' && !mod && !e.altKey) {
          const s = ta.selectionStart, en = ta.selectionEnd, v = ta.value;
          if (s !== en) return;
          const ls = v.lastIndexOf('\n', s - 1) + 1;
          const before = v.slice(ls, s);
          if (before.length >= 2 && /^ +$/.test(before)) { e.preventDefault(); const del = before.length % 4 === 0 ? 4 : before.length % 4; this.replaceRange(s - del, s, ''); }
        }
      }
    },
    template: `
    <div class="editor-wrap" :class="{fill}">
      <div v-if="showKeys && !readonly" class="editor-keys">
        <button type="button" @mousedown.prevent @click="indent(1)">Tab ⇥</button>
        <button type="button" @mousedown.prevent @click="indent(-1)">⇤</button>
        <button type="button" v-for="k in ['(', ')', '[', ']', ':', '=', '_', '.', '#', '\\'']" :key="k" @mousedown.prevent @click="insertText(k)">{{ k }}</button>
        <button type="button" @mousedown.prevent @click="toggleComment">#…</button>
      </div>
      <div class="editor" :class="{fill, focused, readonly}" :style="boxStyle">
        <div class="editor-gutter" ref="gutter" aria-hidden="true"><div v-for="n in lineNums" :key="n" :class="{err: n===errorLine}">{{ n }}</div><div class="gutter-pad"></div></div>
        <div class="editor-area">
          <pre class="editor-hl hl" ref="pre" aria-hidden="true"><code v-html="html"></code></pre>
          <textarea ref="ta" class="editor-input" :readonly="readonly" spellcheck="false" autocapitalize="off" autocomplete="off" autocorrect="off" wrap="off" :aria-label="label" @input="onInput" @scroll="sync" @keydown="onKey" @focus="focused=true" @blur="focused=false"></textarea>
        </div>
      </div>
    </div>`
  };

  // Output panel shared by the playground and runnable samples
  MLA.components['run-output'] = {
    props: { stdout: String, result: Object, failure: Object, running: Boolean, compact: Boolean },
    computed: {
      py() { return MLA.python.status; },
      err() { return this.result && this.result.error; },
      hint() { return MLA.python.explain(this.err) || ''; },
      empty() { const r = this.result; return !this.running && r && !this.stdout && r.value == null && !r.error && !(r.figures || []).length; }
    },
    template: `
    <div class="run-output" :class="{compact}">
      <div v-if="running" class="ro-status"><span class="spinner"></span> {{ py.ready ? 'Running…' : (py.message || 'Starting Python…') }}</div>
      <pre v-if="stdout" class="ro-stdout">{{ stdout }}</pre>
      <pre v-if="result && result.value != null" class="ro-value"><span class="ro-label">Out:</span> {{ result.value }}</pre>
      <img v-for="(f,i) in (result && result.figures) || []" :key="i" class="ro-fig" :src="'data:image/png;base64,'+f" alt="Plot produced by the code" />
      <div v-if="err" class="ro-error">
        <div class="ro-err-msg">{{ err.message }}</div>
        <pre v-if="err.trace" class="ro-trace">{{ err.trace }}</pre>
        <div v-if="hint" class="ro-hint"><mla-icon name="hint" :size="14"/> <mla-md :src="hint" inline /></div>
      </div>
      <div v-if="failure" class="ro-error"><div class="ro-err-msg">{{ failure.message }}</div><div v-if="failure.kind==='load'" class="ro-hint">Tip: once Python has loaded successfully one time, it is cached and works offline.</div></div>
      <div v-if="empty" class="ro-empty">Ran successfully with no output — use <code>print()</code> to show values.</div>
      <div v-if="result && !running && result.ms != null" class="ro-meta">{{ result.ms }} ms</div>
    </div>`
  };

  // Highlighted code sample inside a lesson. Runnable samples can be edited and run in place.
  MLA.components['code-sample'] = {
    props: { code: String, lang: { default: 'python' }, runnable: Boolean, title: String },
    data() { return { running: false, stdout: '', result: null, failure: null, ran: false, editing: false, edited: null, copied: false }; },
    computed: {
      src() { return this.edited != null ? this.edited : this.code; },
      html() { return MLA.highlight.code(this.src, this.lang); },
      label() { return this.lang === 'pytorch' ? 'Python · PyTorch (run locally)' : this.lang === 'bash' ? 'Terminal' : this.lang === 'output' ? 'Output' : (this.runnable ? 'Python · runs in your browser' : 'Python'); },
      changed() { return this.edited != null && this.edited !== this.code; }
    },
    methods: {
      async run() {
        if (this.running) return;
        this.running = true; this.ran = true; this.stdout = ''; this.result = null; this.failure = null;
        const r = await MLA.python.exec(this.src, { onStdout: t => { this.stdout += t; } });
        this.running = false;
        if (!r.ok) { this.failure = { message: r.message, kind: r.kind }; return; }
        this.result = r.result;
        MLA.store.state.stats.sampleRuns = (MLA.store.state.stats.sampleRuns || 0) + 1;
        MLA.gamification.checkAchievements();
      },
      copy() { MLA.util.copyText(this.src).then(ok => { this.copied = ok; setTimeout(() => { this.copied = false; }, 1400); }); },
      toggleEdit() { if (!this.editing && this.edited == null) this.edited = this.code; this.editing = !this.editing; },
      reset() { this.edited = null; this.editing = false; },
      toPlayground() { MLA.playground.openWith(this.title || 'Lesson snippet', this.src); }
    },
    template: `
    <div class="code-sample">
      <div class="code-block" :class="{'is-output': lang==='output'}">
        <div class="code-head">
          <span>{{ label }}</span><span class="spacer"></span>
          <button v-if="changed" class="code-btn" @click="reset" title="Restore the original code">Reset</button>
          <button v-if="runnable" class="code-btn" @click="toggleEdit">{{ editing ? 'Done' : 'Edit' }}</button>
          <button class="code-btn" @click="copy">{{ copied ? 'Copied' : 'Copy' }}</button>
          <button v-if="runnable" class="code-btn" @click="toPlayground" title="Open in the Python playground">Playground</button>
          <button v-if="runnable" class="code-btn run" :disabled="running" @click="run"><mla-icon name="play" :size="12"/> Run</button>
        </div>
        <code-editor v-if="editing" v-model="edited" :min-lines="3" :max-lines="30" @run="run" label="Edit this example" />
        <pre v-else class="hl"><code v-html="html"></code></pre>
      </div>
      <run-output v-if="ran" :stdout="stdout" :result="result" :failure="failure" :running="running" compact />
    </div>`
  };

  // Playground helper used by "open in playground" buttons
  MLA.playground = {
    newFile(name, code) {
      const s = MLA.store.state; const now = new Date().toISOString();
      const f = { id: MLA.util.uid('f'), name: (name || 'untitled').slice(0, 60), code: code || '', createdAt: now, updatedAt: now };
      s.playground.files.unshift(f); s.playground.active = f.id;
      return f;
    },
    openWith(name, code) { this.newFile(name, code); MLA.router.navigate('/playground'); MLA.ui.toast('Opened in the playground — edit and run it freely.', 'info'); }
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
