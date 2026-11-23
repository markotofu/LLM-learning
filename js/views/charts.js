/* Static, theme-aware charts drawn from data stored in a lesson: { type: 'chart', title, chart: spec }.
 * spec = {
 *   kind: 'line' | 'scatter' | 'bar',
 *   x: [numbers] (line, bar labels as strings for bar),
 *   series: [{ name, y: [numbers] }]              line and bar: one y per x
 *         | [{ name, x: [...], y: [...] }]        scatter (and line with its own x)
 *   xLabel, yLabel, xLog, yLog, yMin, yMax, xMin, xMax,
 *   vlines: [{ x, label }], hlines: [{ y, label }], points: true|false (markers on lines)
 *   A line chart's series can set scatter: true (dots only, e.g. data under a fitted curve) or dash: true.
 * }
 * Colours come from CSS classes (.chart .ch-0 …) so they follow the theme. */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};
  const W = 640, H = 320;
  let clipSeq = 0;                                  // unique clip-path ids when several charts share a page

  function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw))); const f = raw / p; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; }
  function ticks(lo, hi, n, log) {
    if (log) {
      const e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi)), every = Math.max(1, Math.ceil((e1 - e0) / 8));   // wide ranges: label every 2nd, 3rd… decade
      const decades = []; for (let e = e0; e <= e1; e++) if (e % every === 0) decades.push(Math.pow(10, e));
      const mult = Math.log10(hi / lo) < 2.5 ? [1, 2, 5] : [1];      // short ranges also get 2x and 5x ticks
      const out = []; for (const d of decades) for (const m of mult) { const v = +(d * m).toPrecision(6); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
      return out;
    }
    const step = niceStep((hi - lo) / n || 1); const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function fmt(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) { const e = Math.floor(Math.log10(a)); const m = v / Math.pow(10, e); return (Math.abs(m - 1) < 1e-9 ? '' : (+m.toFixed(1)) + '×') + '10' + String(e).split('').map(c => '⁰¹²³⁴⁵⁶⁷⁸⁹'['0123456789'.indexOf(c)] || (c === '-' ? '⁻' : c)).join(''); }
    if (a >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
    return String(+v.toFixed(a >= 10 ? 1 : a >= 1 ? 2 : 3));
  }
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function render(spec) {
    const kind = spec.kind || 'line';
    const series = spec.series || [];
    const bar = kind === 'bar';
    const xs = s => s.x || spec.x || [];
    // ranges
    let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
    for (const s of series) {
      const sx = xs(s);
      s.y.forEach((y, i) => { if (y == null || !isFinite(y)) return; if (!bar) { const x = sx[i]; if (x < xlo) xlo = x; if (x > xhi) xhi = x; } if (y < ylo) ylo = y; if (y > yhi) yhi = y; });
    }
    for (const h of spec.hlines || []) { ylo = Math.min(ylo, h.y); yhi = Math.max(yhi, h.y); }
    if (bar) { ylo = Math.min(0, ylo); }
    if (spec.yMin != null) ylo = spec.yMin; if (spec.yMax != null) yhi = spec.yMax;
    if (spec.xMin != null) xlo = spec.xMin; if (spec.xMax != null) xhi = spec.xMax;
    if (!spec.yLog && spec.yMin == null && !bar) { const pad = (yhi - ylo) * 0.06 || 1; ylo -= pad; yhi += pad; }
    if (bar && spec.yMax == null) yhi += (yhi - ylo) * 0.08;
    if (yhi === ylo) { yhi += 1; ylo -= 1; }
    // Lay out the legend first: long legends wrap onto extra rows, and the plot area moves down to make room.
    const legend = []; let lpos = 66, lrow = 0;
    series.forEach((s, k) => { if (!s.name) return; const w = 30 + String(s.name).length * 6.6; if (lpos + w > W - 18 && lpos > 66) { lrow++; lpos = 66; } legend.push({ x: lpos, row: lrow, name: s.name, k }); lpos += w; });
    // Annotations: each line's last value is written just right of its end (up to four lines), so the right margin
    // grows to fit the widest of those labels.
    const lineIdx = bar || kind === 'scatter' ? [] : series.map((s, k) => k).filter(k => !series[k].scatter);
    const lastPoint = s => { const sx = xs(s); for (let i = s.y.length - 1; i >= 0; i--) if (s.y[i] != null && isFinite(s.y[i]) && isFinite(sx[i])) return { x: sx[i], y: s.y[i] }; return null; };
    const ends = lineIdx.length && lineIdx.length <= 4 ? lineIdx.map(k => ({ k, p: lastPoint(series[k]) })).filter(e => e.p) : [];
    const endW = ends.length ? Math.max(...ends.map(e => fmt(e.p.y).length)) * 6.4 + 10 : 0;
    const PAD = { l: 58, r: 18 + endW, t: legend.length ? 30 + lrow * 16 : 18, b: 50 };
    const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
    const lx = v => spec.xLog ? Math.log10(v) : v, ly = v => spec.yLog ? Math.log10(v) : v;
    const X = v => PAD.l + (lx(v) - lx(xlo)) / ((lx(xhi) - lx(xlo)) || 1) * pw;
    // Values beyond the axis range are not squashed onto the edge (that would bend lines along the border); they are
    // only limited to a generous band to keep coordinates finite, and the data layer is clipped to the plot area.
    const soft = v => spec.yLog ? Math.min(Math.max(v, ylo / 1e3), yhi * 1e3) : Math.min(Math.max(v, ylo - 3 * (yhi - ylo)), yhi + 3 * (yhi - ylo));
    const Y = v => PAD.t + ph - (ly(soft(v)) - ly(ylo)) / ((ly(yhi) - ly(ylo)) || 1) * ph;
    const clipId = 'chclip' + (++clipSeq);
    let out = '<defs><clipPath id="' + clipId + '"><rect x="' + (PAD.l - 4) + '" y="' + (PAD.t - 4) + '" width="' + (pw + 8) + '" height="' + (ph + 8) + '"/></clipPath></defs>';
    // grid and y axis
    for (const t of ticks(ylo, yhi, 5, spec.yLog)) { const y = Y(t); out += '<line x1="' + PAD.l + '" x2="' + (W - PAD.r) + '" y1="' + y + '" y2="' + y + '" class="ch-grid"/><text x="' + (PAD.l - 7) + '" y="' + (y + 4) + '" text-anchor="end" class="ch-tick">' + esc(fmt(t)) + '</text>'; }
    // x axis
    const cats = bar ? (spec.x || []) : null;
    if (bar) {
      const band = pw / cats.length;
      cats.forEach((c, i) => { out += '<text x="' + (PAD.l + band * (i + 0.5)) + '" y="' + (H - PAD.b + 19) + '" text-anchor="middle" class="ch-tick">' + esc(c) + '</text>'; });
    } else {
      for (const t of ticks(xlo, xhi, 7, spec.xLog)) { const x = X(t); out += '<line x1="' + x + '" x2="' + x + '" y1="' + PAD.t + '" y2="' + (PAD.t + ph) + '" class="ch-grid"/><text x="' + x + '" y="' + (H - PAD.b + 19) + '" text-anchor="middle" class="ch-tick">' + esc(fmt(t)) + '</text>'; }
    }
    out += '<line x1="' + PAD.l + '" x2="' + (W - PAD.r) + '" y1="' + (PAD.t + ph) + '" y2="' + (PAD.t + ph) + '" class="ch-axis"/><line x1="' + PAD.l + '" x2="' + PAD.l + '" y1="' + PAD.t + '" y2="' + (PAD.t + ph) + '" class="ch-axis"/>';
    if (!bar && !spec.yLog && ylo < 0 && yhi > 0) out += '<line x1="' + PAD.l + '" x2="' + (W - PAD.r) + '" y1="' + Y(0) + '" y2="' + Y(0) + '" class="ch-zero"/>';
    // reference lines go under the data; their labels go on top of it (with a halo, see .ch-reflabel) so lines never hide them
    let refLabels = '';
    for (const h of spec.hlines || []) { const y = Y(h.y); out += '<line x1="' + PAD.l + '" x2="' + (W - PAD.r) + '" y1="' + y + '" y2="' + y + '" class="ch-ref"/>'; if (h.label) refLabels += '<text x="' + (W - PAD.r - 4) + '" y="' + (y - 5) + '" text-anchor="end" class="ch-reflabel">' + esc(h.label) + '</text>'; }
    for (const v of spec.vlines || []) { const x = X(v.x); out += '<line x1="' + x + '" x2="' + x + '" y1="' + PAD.t + '" y2="' + (PAD.t + ph) + '" class="ch-ref"/>'; if (v.label) refLabels += '<text x="' + (x + 4) + '" y="' + (PAD.t + 12) + '" class="ch-reflabel">' + esc(v.label) + '</text>'; }
    // data (clipped to the plot area)
    out += '<g clip-path="url(#' + clipId + ')">';
    series.forEach((s, k) => {
      const cls = 'ch-' + (k % 6); const sx = xs(s);
      if (bar) {
        const band = pw / cats.length, bw = band * 0.72 / series.length;
        s.y.forEach((y, i) => { const x0 = PAD.l + band * i + band * 0.14 + bw * k; const y0 = Y(Math.max(0, ylo)), y1 = Y(y); out += '<rect x="' + x0 + '" y="' + Math.min(y0, y1) + '" width="' + Math.max(1, bw - 2) + '" height="' + Math.abs(y0 - y1) + '" class="ch-bar ' + cls + '"/>'; });
      } else if (kind === 'scatter' || s.scatter) {
        s.y.forEach((y, i) => { if (y == null || !isFinite(y)) return; out += '<circle cx="' + X(sx[i]) + '" cy="' + Y(y) + '" r="3.4" class="ch-dot ' + cls + '"/>'; });
      } else {
        let d = '', pen = false;   // a gap (null) in the data lifts the pen
        s.y.forEach((y, i) => { if (y == null || !isFinite(y)) { pen = false; return; } d += (pen ? ' L' : ' M') + X(sx[i]).toFixed(1) + ' ' + Y(y).toFixed(1); pen = true; });
        out += '<path d="' + d.trim() + '" class="ch-line ' + cls + (s.dash ? ' ch-dash' : '') + '"/>';
        if (spec.points || (spec.points !== false && s.y.length <= 16)) s.y.forEach((y, i) => { if (y != null && isFinite(y)) out += '<circle cx="' + X(sx[i]) + '" cy="' + Y(y) + '" r="3" class="ch-dot ' + cls + '"/>'; });
      }
    });
    out += '</g>' + refLabels;
    // annotations, on top of the data and with a halo (.ch-annot): end values, notable peaks and dips, bar values
    let annot = '';
    const inPlot = y => Math.min(PAD.t + ph - 2, Math.max(PAD.t + 8, y));
    const placed = [];                                                    // label anchors, to avoid writing labels on top of each other
    const free = (x, y, w) => placed.every(p => Math.abs(p.x - x) > (p.w + w) / 2 + 4 || Math.abs(p.y - y) > 12);
    if (ends.length) {
      const L = ends.map(e => ({ k: e.k, x: X(e.p.x) + 6, y: inPlot(Y(e.p.y) + 4), t: fmt(e.p.y) })).sort((a, b) => a.y - b.y);
      for (let i = 1; i < L.length; i++) if (L[i].y - L[i - 1].y < 13) L[i].y = L[i - 1].y + 13;           // keep the labels apart
      for (const l of L) { annot += '<text x="' + l.x.toFixed(1) + '" y="' + l.y.toFixed(1) + '" class="ch-annot ch-' + (l.k % 6) + '">' + esc(l.t) + '</text>'; placed.push({ x: l.x + l.t.length * 3.2, y: l.y, w: l.t.length * 6.4 }); }
    }
    if (MLA.narrate && lineIdx.length && lineIdx.length <= 4) {
      for (const k of lineIdx) {
        const s = series[k], sx = xs(s), pts = [];
        s.y.forEach((y, i) => { if (y != null && isFinite(y) && isFinite(sx[i])) pts.push({ x: sx[i], y }); });
        if (pts.length < 5) continue;
        const t = MLA.narrate.trend(pts);
        if (t.kind !== 'peak' && t.kind !== 'valley') continue;
        const cx = X(t.at.x), cy = Y(t.at.y), up = t.kind === 'peak';
        const label = (up ? 'peak ' : 'low ') + fmt(t.at.y), ly2 = inPlot(up ? cy - 10 : cy + 18), w = label.length * 6.4;
        if (!free(cx, ly2, w)) continue;                                   // no room: the description still mentions it
        annot += '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="5" class="ch-annot-ring ch-' + (k % 6) + '"/>' +
          '<text x="' + cx.toFixed(1) + '" y="' + ly2.toFixed(1) + '" text-anchor="middle" class="ch-annot ch-' + (k % 6) + '">' + esc(label) + '</text>';
        placed.push({ x: cx, y: ly2, w });
      }
    }
    if (bar && cats.length * series.length <= 16) {
      const band = pw / cats.length, bw = band * 0.72 / series.length;
      series.forEach((s, k) => s.y.forEach((y, i) => {
        if (y == null || !isFinite(y)) return;
        const x = PAD.l + band * i + band * 0.14 + bw * k + Math.max(1, bw - 2) / 2;
        annot += '<text x="' + x.toFixed(1) + '" y="' + inPlot(y >= 0 ? Y(y) - 5 : Y(y) + 13).toFixed(1) + '" text-anchor="middle" class="ch-annot ch-' + (k % 6) + '">' + esc(fmt(y)) + '</text>';
      }));
    }
    out += annot;
    // legend
    for (const it of legend) out += ((kind === 'scatter' || series[it.k].scatter) ? '<circle cx="' + (it.x + 7) + '" cy="' + (13 + it.row * 16) + '" r="3.6" class="ch-key ch-' + (it.k % 6) + '"/>' : '<rect x="' + it.x + '" y="' + (11 + it.row * 16) + '" width="14" height="4" rx="2" class="ch-key ch-' + (it.k % 6) + '"/>') + '<text x="' + (it.x + 19) + '" y="' + (17 + it.row * 16) + '" class="ch-legend">' + esc(it.name) + '</text>';
    if (spec.xLabel) out += '<text x="' + (PAD.l + pw / 2) + '" y="' + (H - 8) + '" text-anchor="middle" class="ch-label">' + esc(spec.xLabel) + '</text>';
    if (spec.yLabel) out += '<text transform="translate(15 ' + (PAD.t + ph / 2) + ') rotate(-90)" text-anchor="middle" class="ch-label">' + esc(spec.yLabel) + '</text>';
    return out;
  }

  MLA.charts = { render };
  let descSeq = 0;
  MLA.components['mla-chart'] = {
    props: { spec: Object, title: String },
    data() { return { descId: 'chdesc' + (++descSeq) }; },
    computed: {
      svg() { try { return render(this.spec || {}); } catch (e) { return '<text x="20" y="40" class="ch-label">Chart error: ' + esc(e.message) + '</text>'; } },
      // The chart in words (js/core/narrate.js): the same text the read-aloud player speaks, and the SVG's description.
      desc() { try { return MLA.narrate ? MLA.narrate.chart(this.spec || {}) : ''; } catch (e) { return ''; } },
      // Series with their own x values (scatter plots, paths) are listed point by point; shared-x data as one row per x.
      long() { return ((this.spec && this.spec.series) || []).some(x => Array.isArray(x.x)); },
      rows() {
        const s = this.spec || {}; const ser = s.series || [];
        const pick = (n, max) => { if (n <= max) return Array.from({ length: n }, (_, i) => i); const step = (n - 1) / (max - 1); return Array.from({ length: max }, (_, i) => Math.round(i * step)); };
        if (this.long) {
          const out = [];
          for (const x of ser) { const xs = x.x || s.x || []; const idx = x.y.map((v, i) => i).filter(i => x.y[i] != null); for (const k of pick(idx.length, 30)) out.push([x.name || '', xs[idx[k]], x.y[idx[k]]]); }
          return out;
        }
        const n = Math.max(0, ...ser.map(x => x.y.length));
        return pick(n, 60).map(i => [(s.x || [])[i]].concat(ser.map(x => x.y[i])));
      }
    },
    methods: { cell(v) { return typeof v === 'number' ? String(+v.toFixed(4)) : v == null ? '' : v; } },
    template: `
    <div class="card demo-card chart-card">
      <div class="diagram-wrap"><svg class="diagram chart" viewBox="0 0 640 320" role="img" :aria-label="title || 'chart'" :aria-describedby="desc ? descId : null" v-html="svg"></svg></div>
      <details v-if="desc" class="chart-desc tiny"><summary>Describe this chart in words</summary><p :id="descId">{{ desc }}</p></details>
      <details class="chart-data tiny"><summary>Show the numbers</summary>
        <table class="mini-table"><thead><tr v-if="long"><th>series</th><th>{{ (spec && spec.xLabel) || 'x' }}</th><th>{{ (spec && spec.yLabel) || 'y' }}</th></tr><tr v-else><th>{{ (spec && spec.xLabel) || 'x' }}</th><th v-for="s in (spec && spec.series) || []" :key="s.name">{{ s.name || (spec && spec.yLabel) || 'y' }}</th></tr></thead>
        <tbody><tr v-for="(r, i) in rows" :key="i"><td v-for="(v, j) in r" :key="j" class="mono">{{ cell(v) }}</td></tr></tbody></table>
      </details>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
