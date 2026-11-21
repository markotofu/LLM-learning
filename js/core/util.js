/* ML Atlas – shared utilities (framework independent; also loaded by the Node tests) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  function uid(prefix) {
    return (prefix || 'id') + '_' + Math.random().toString(36).slice(2, 9) + Date.now().toString(36).slice(-4);
  }
  function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }
  function pad(n, w) { let s = String(n); while (s.length < (w || 2)) s = '0' + s; return s; }

  // Deterministic PRNG (mulberry32) – used for "problem of the day" and quiz shuffles
  function seededRng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hashString(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  function shuffle(arr, rng) {
    const r = rng || Math.random; const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }

  // ---- dates ----
  function todayKey(d) {
    const x = d ? new Date(d) : new Date();
    return x.getFullYear() + '-' + pad(x.getMonth() + 1) + '-' + pad(x.getDate());
  }
  function addDays(key, n) {
    const [y, m, d] = key.split('-').map(Number);
    return todayKey(new Date(y, m - 1, d + n));
  }
  function fmtDate(iso) { if (!iso) return ''; return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }); }
  function timeAgo(iso) {
    if (!iso) return '';
    const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return 'just now';
    if (s < 3600) return Math.floor(s / 60) + ' min ago';
    if (s < 86400) return Math.floor(s / 3600) + ' h ago';
    return Math.floor(s / 86400) + ' d ago';
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // Remove common leading indentation and surrounding blank lines (for code authored inside JS files)
  function dedent(s) {
    if (s == null) return '';
    const lines = String(s).replace(/\r\n/g, '\n').replace(/\t/g, '    ').split('\n');
    while (lines.length && lines[0].trim() === '') lines.shift();
    while (lines.length && lines[lines.length - 1].trim() === '') lines.pop();
    let min = Infinity;
    for (const l of lines) { if (l.trim() === '') continue; const m = l.match(/^ */)[0].length; if (m < min) min = m; }
    if (!isFinite(min) || min === 0) return lines.join('\n');
    return lines.map(l => l.slice(Math.min(min, l.match(/^ */)[0].length))).join('\n');
  }

  // ---------------- Markdown ----------------
  // Every piece of text is HTML-escaped and only a fixed set of tags is generated (no raw-HTML passthrough, links only
  // to #/routes or http(s) URLs), so the renderer is safe for user-written notes and imported files as well as lessons.
  function inline(s) {
    let out = escapeHtml(s);
    const codes = [];
    out = out.replace(/`([^`]+)`/g, (m, c) => { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/(^|[\s(“"'])\*([^*\n]+)\*(?=[\s).,;:!?”"']|$)/g, '$1<em>$2</em>');
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, t, u) => {
      if (/^#/.test(u)) return '<a href="' + u + '">' + t + '</a>';
      if (!/^https?:\/\//.test(u)) return t;
      return '<a href="' + u + '" target="_blank" rel="noopener">' + t + '</a>';
    });
    out = out.replace(/\u0000(\d+)\u0000/g, (m, i) => '<code>' + codes[Number(i)] + '</code>');
    return out;
  }
  const LANG_LABEL = { python: 'Python', py: 'Python', bash: 'Terminal', sh: 'Terminal', shell: 'Terminal', text: 'Text', json: 'JSON', js: 'JavaScript', javascript: 'JavaScript', output: 'Output', pytorch: 'Python · PyTorch' };
  function codeBlockHtml(code, lang) {
    const l = (lang || '').toLowerCase();
    const hl = MLA.highlight ? MLA.highlight.code(code, l) : escapeHtml(code);
    return '<div class="code-block' + (l === 'output' ? ' is-output' : '') + '"><div class="code-head"><span>' + escapeHtml(LANG_LABEL[l] || l || 'Code') + '</span><button class="code-copy" type="button">Copy</button></div><pre class="hl"><code>' + hl + '</code></pre></div>';
  }
  function parseList(lines, i) {
    const items = [];
    while (i < lines.length) {
      const raw = lines[i].replace(/\t/g, '    ');
      const mm = raw.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
      if (mm) { items.push({ indent: mm[1].length, ordered: /\d/.test(mm[2]), text: mm[3] }); i++; continue; }
      if (raw.trim() !== '' && /^\s{2,}/.test(raw) && items.length) { items[items.length - 1].text += ' ' + raw.trim(); i++; continue; }
      break;
    }
    let out = ''; const stack = [];
    for (const it of items) {
      while (stack.length && it.indent < stack[stack.length - 1].indent) out += '</li></' + (stack.pop().ordered ? 'ol' : 'ul') + '>';
      if (!stack.length || it.indent > stack[stack.length - 1].indent) { out += it.ordered ? '<ol>' : '<ul>'; stack.push({ indent: it.indent, ordered: it.ordered }); }
      else out += '</li>';
      out += '<li>' + inline(it.text);
    }
    while (stack.length) out += '</li></' + (stack.pop().ordered ? 'ol' : 'ul') + '>';
    return { html: out, next: i };
  }
  // Lines of a :::where legend → [{sym, def}] (split at the first " — ", else the first ": "); other lines are notes.
  // A line indented by 2+ spaces continues the previous definition.
  function parseWhere(lines) {
    const items = [];
    for (const raw of lines) {
      if (!raw.trim()) continue;
      if (/^\s{2,}\S/.test(raw) && items.length) { const last = items[items.length - 1]; if (last.note) last.note += ' ' + raw.trim(); else last.def += ' ' + raw.trim(); continue; }
      const line = raw.trim().replace(/^[-*]\s+/, '');
      let k = line.indexOf(' — '), w = 3;
      if (k < 0) { k = line.indexOf(': '); w = 2; }
      if (k > 0) items.push({ sym: line.slice(0, k).trim(), def: line.slice(k + w).trim() });
      else items.push({ note: line });
    }
    return items;
  }
  const CALLOUT_TITLES = { tip: 'Tip', warn: 'Common mistake', danger: 'Watch out', real: 'In the real world', note: 'Note', key: 'Key idea', analogy: 'Analogy', math: 'The math' };
  function md(text) {
    if (!text) return '';
    const lines = String(text).replace(/\r\n/g, '\n').split('\n');
    const html = [];
    let i = 0; let para = [];
    const flushPara = () => { if (para.length) { html.push('<p>' + inline(para.join(' ')) + '</p>'); para = []; } };
    while (i < lines.length) {
      const line = lines[i];
      const t = line.trim();
      if (t === '') { flushPara(); i++; continue; }
      if (/^```/.test(t)) { // fenced code
        flushPara();
        const lang = t.slice(3).trim();
        const buf = []; i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
        i++;
        html.push(codeBlockHtml(dedent(buf.join('\n')), lang));
        continue;
      }
      if (/^\$\$/.test(t)) { // display formula block (Unicode math, no LaTeX needed)
        flushPara();
        let body = [];
        if (t.length > 4 && /\$\$$/.test(t)) { body = [t.slice(2, -2)]; i++; }
        else {
          if (t.slice(2).trim()) body.push(t.slice(2).trim());
          i++;
          while (i < lines.length && !/\$\$\s*$/.test(lines[i].trim())) { body.push(lines[i].trim()); i++; }
          if (i < lines.length) { const last = lines[i].trim().replace(/\$\$\s*$/, '').trim(); if (last) body.push(last); i++; }
        }
        // runs of 2+ spaces separate side-by-side formulas: keep them visible
        html.push('<div class="formula">' + body.filter(x => x !== '').map(l => '<div>' + inline(l).replace(/ {2,}/g, '<span class="fgap"></span>') + '</div>').join('') + '</div>');
        continue;
      }
      let m = t.match(/^:::where\s*(.*)$/);
      if (m) { // symbol legend for the formula(s) directly above: one "symbol — meaning" per line
        flushPara();
        const buf = []; i++;
        while (i < lines.length && !/^:::\s*$/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
        i++;
        const legend = '<div class="formula-where"><div class="fw-label">' + escapeHtml(m[1] || 'where') + '</div><dl>' +
          parseWhere(buf).map(it => it.note ? '<dd class="fw-note">' + inline(it.note) + '</dd>' : '<dt>' + inline(it.sym) + '</dt><dd>' + inline(it.def) + '</dd>').join('') + '</dl></div>';
        const group = [];
        while (html.length && html[html.length - 1].indexOf('<div class="formula">') === 0) group.unshift(html.pop());
        html.push('<div class="formula-group">' + group.join('') + legend + '</div>');
        continue;
      }
      m = t.match(/^:::(tip|warn|danger|real|note|key|analogy|math)\s*(.*)$/);
      if (m) { // callouts
        flushPara();
        const kind = m[1]; const title = m[2] || CALLOUT_TITLES[kind];
        const buf = []; i++;
        while (i < lines.length && !/^:::\s*$/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
        i++;
        html.push('<div class="callout ' + kind + '"><div class="callout-title">' + escapeHtml(title) + '</div>' + md(buf.join('\n')) + '</div>');
        continue;
      }
      m = t.match(/^(#{1,4})\s+(.*)$/);
      if (m) { flushPara(); const lvl = Math.min(5, m[1].length + 2); html.push('<h' + lvl + '>' + inline(m[2]) + '</h' + lvl + '>'); i++; continue; }
      if (/^---+$/.test(t)) { flushPara(); html.push('<hr>'); i++; continue; }
      if (/^>\s?/.test(t)) {
        flushPara(); const buf = [];
        while (i < lines.length && /^>\s?/.test(lines[i].trim())) { buf.push(lines[i].trim().replace(/^>\s?/, '')); i++; }
        html.push('<blockquote>' + md(buf.join('\n')) + '</blockquote>');
        continue;
      }
      if (/^\|.*\|$/.test(t) && i + 1 < lines.length && /^\|?\s*:?-{2,}/.test(lines[i + 1].trim())) {
        flushPara();
        const header = t.slice(1, -1).split('|').map(c => c.trim());
        i += 2;
        const rows = [];
        while (i < lines.length && /^\|.*\|$/.test(lines[i].trim())) { rows.push(lines[i].trim().slice(1, -1).split('|').map(c => c.trim())); i++; }
        let tbl = '<div class="scroll-x"><table><thead><tr>' + header.map(c => '<th>' + inline(c) + '</th>').join('') + '</tr></thead><tbody>';
        for (const r of rows) tbl += '<tr>' + r.map(c => '<td>' + inline(c) + '</td>').join('') + '</tr>';
        html.push(tbl + '</tbody></table></div>');
        continue;
      }
      if (/^([-*]|\d+\.)\s+/.test(t)) { flushPara(); const r = parseList(lines, i); html.push(r.html); i = r.next; continue; }
      para.push(t); i++;                 // everything else, including text that looks like HTML, is an escaped paragraph
    }
    flushPara();
    return html.join('\n');
  }

  function debounce(fn, ms) { let t; return function () { clearTimeout(t); const a = arguments, c = this; t = setTimeout(() => fn.apply(c, a), ms); }; }
  function deepClone(o) { return JSON.parse(JSON.stringify(o)); }
  function sum(arr, f) { return arr.reduce((a, x) => a + (f ? f(x) : x), 0); }
  function pct(a, b) { return b ? Math.round((a / b) * 100) : 0; }
  function plural(n, s, p) { return n + ' ' + (n === 1 ? s : (p || s + 's')); }
  function slug(s) { return String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'file'; }
  function formatBytes(n) { if (n < 1024) return n + ' B'; if (n < 1048576) return (n / 1024).toFixed(1) + ' KB'; return (n / 1048576).toFixed(1) + ' MB'; }

  function download(filename, text, type) {
    try {
      const blob = new Blob([text], { type: type || 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a); a.click();
      setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 800);
      return true;
    } catch (e) { return false; }
  }
  function copyText(text) {
    try {
      if (global.navigator && navigator.clipboard && global.isSecureContext) return navigator.clipboard.writeText(text).then(() => true, () => fallbackCopy(text));
    } catch (e) { /* fall through */ }
    return Promise.resolve(fallbackCopy(text));
  }
  function fallbackCopy(text) {
    try {
      const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select(); const ok = document.execCommand('copy'); ta.remove(); return ok;
    } catch (e) { return false; }
  }

  // ---- compact text encoding for "sync codes" (gzip + base64 when the browser supports it) ----
  function bytesToB64(bytes) { let bin = ''; const CH = 0x8000; for (let i = 0; i < bytes.length; i += CH) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH)); return btoa(bin); }
  function b64ToBytes(b64) { const bin = atob(b64); const out = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i); return out; }
  async function packText(text) {
    const bytes = new TextEncoder().encode(text);
    if (typeof CompressionStream !== 'undefined') {
      const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
      const buf = new Uint8Array(await new Response(stream).arrayBuffer());
      return 'MLA1Z.' + bytesToB64(buf);
    }
    return 'MLA1B.' + bytesToB64(bytes);
  }
  async function unpackText(code) {
    const s = String(code || '').trim().replace(/\s+/g, '');
    if (s.startsWith('MLA1Z.')) {
      if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot decompress sync codes. Use a progress file instead.');
      const stream = new Blob([b64ToBytes(s.slice(6))]).stream().pipeThrough(new DecompressionStream('gzip'));
      return new TextDecoder().decode(await new Response(stream).arrayBuffer());
    }
    if (s.startsWith('MLA1B.')) return new TextDecoder().decode(b64ToBytes(s.slice(6)));
    if (s.startsWith('{')) return s;
    throw new Error('That does not look like an ML Atlas sync code.');
  }

  const projectKey = p => p.concept + ':' + slug(p.title);      // where a sample project's progress is stored (state.projects)
  MLA.util = { uid, clamp, pad, seededRng, hashString, shuffle, todayKey, addDays, fmtDate, timeAgo, escapeHtml, dedent, md, inline, codeBlockHtml, debounce, deepClone, sum, pct, plural, slug, projectKey, formatBytes, download, copyText, packText, unpackText };
})(typeof globalThis !== 'undefined' ? globalThis : window);
