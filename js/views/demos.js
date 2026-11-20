/* ML Atlas – interactive demos embedded in lessons. <mla-demo name="..."> dispatches to demo-<name>. */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};
  const C = MLA.components;

  // ---------- small canvas helpers ----------
  function palette() {
    const cs = getComputedStyle(document.documentElement); const g = n => cs.getPropertyValue(n).trim();
    return { text: g('--text'), muted: g('--muted'), dim: g('--dim'), border: g('--border'), accent: g('--accent'), accent2: g('--accent2'), green: g('--green'), red: g('--red'), amber: g('--amber'), purple: g('--purple'), cyan: g('--cyan'), pink: g('--pink'), panel: g('--panel2'), bg: g('--bg2') };
  }
  function setup(cv, w, h) {
    const dpr = Math.min(2, global.devicePixelRatio || 1);
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    return ctx;
  }
  function frame(ctx, W, H, xr, yr, P, opts) {
    opts = opts || {};
    const pad = { l: 38, r: 12, t: 12, b: 26 };
    const X = x => pad.l + (x - xr[0]) / (xr[1] - xr[0]) * (W - pad.l - pad.r);
    const Y = y => H - pad.b - (y - yr[0]) / (yr[1] - yr[0]) * (H - pad.t - pad.b);
    const inv = (px, py) => [xr[0] + (px - pad.l) / (W - pad.l - pad.r) * (xr[1] - xr[0]), yr[0] + (H - pad.b - py) / (H - pad.t - pad.b) * (yr[1] - yr[0])];
    ctx.strokeStyle = P.border; ctx.lineWidth = 1; ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = P.muted;
    const ticks = (a, b, n) => { const out = []; const step = niceStep((b - a) / n); for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(6)); return out; };
    for (const v of ticks(xr[0], xr[1], opts.xt || 6)) { ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(X(v), pad.t); ctx.lineTo(X(v), H - pad.b); ctx.stroke(); ctx.globalAlpha = 1; ctx.textAlign = 'center'; ctx.fillText(fmt(v), X(v), H - 8); }
    for (const v of ticks(yr[0], yr[1], opts.yt || 4)) { ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(W - pad.r, Y(v)); ctx.stroke(); ctx.globalAlpha = 1; ctx.textAlign = 'right'; ctx.fillText(fmt(v), pad.l - 5, Y(v) + 4); }
    if (opts.axes !== false) { ctx.strokeStyle = P.dim; ctx.lineWidth = 1.2; if (xr[0] < 0 && xr[1] > 0) { ctx.beginPath(); ctx.moveTo(X(0), pad.t); ctx.lineTo(X(0), H - pad.b); ctx.stroke(); } if (yr[0] < 0 && yr[1] > 0) { ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(W - pad.r, Y(0)); ctx.stroke(); } }
    return { X, Y, inv, pad };
  }
  function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw))); const f = raw / p; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; }
  function fmt(v) { return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(0) : (+v.toFixed(2)).toString(); }
  function curve(ctx, f, xr, T, color, width, dash) {
    ctx.strokeStyle = color; ctx.lineWidth = width || 2; ctx.setLineDash(dash || []); ctx.beginPath();
    const n = 240; for (let i = 0; i <= n; i++) { const x = xr[0] + (xr[1] - xr[0]) * i / n; const y = f(x); if (!isFinite(y)) continue; i ? ctx.lineTo(T.X(x), T.Y(y)) : ctx.moveTo(T.X(x), T.Y(y)); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  function dot(ctx, x, y, r, fill, stroke) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); } }
  function gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function canvasPoint(cv, e, W, H) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * W, (e.clientY - r.top) / r.height * H]; }
  function softmax(xs) { const m = Math.max(...xs); const e = xs.map(x => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); }
  const redrawOnChange = { mounted() { this.$nextTick(() => this.draw()); this.$watch(() => JSON.stringify(this.$data), () => this.$nextTick(() => this.draw())); } };
  // Shared with js/views/demos-more.js
  MLA.demoKit = { palette, setup, frame, curve, dot, gauss, canvasPoint, softmax, redrawOnChange, niceStep, fmt };

  C['mla-demo'] = { props: { name: String }, computed: { comp() { return C['demo-' + this.name] ? 'demo-' + this.name : null; } }, template: `<div class="demo"><component v-if="comp" :is="comp" /><div v-else class="callout warn">Unknown demo: {{ name }}</div></div>` };

  // 1 ---------- Gradient descent on a 1-D loss ----------
  C['demo-gradient-descent'] = {
    mixins: [redrawOnChange],
    data() { return { fn: 'bowl', lr: 0.1, start: -4, path: [-4], timer: null }; },
    computed: {
      f() { return this.fn === 'bowl' ? (w => (w - 1) * (w - 1) + 0.5) : (w => 0.15 * w * w + Math.sin(1.5 * w) + 1.2); },
      df() { return this.fn === 'bowl' ? (w => 2 * (w - 1)) : (w => 0.3 * w + 1.5 * Math.cos(1.5 * w)); },
      w() { return this.path[this.path.length - 1]; },
      diverged() { return !isFinite(this.w) || Math.abs(this.w) > 50; }
    },
    watch: { fn() { this.reset(); }, start() { this.reset(); }, lr() { this.reset(); } },
    beforeUnmount() { clearInterval(this.timer); },
    methods: {
      step() { if (this.diverged) return; const w = this.w; this.path.push(w - this.lr * this.df(w)); },
      run() { clearInterval(this.timer); let n = 0; this.timer = setInterval(() => { this.step(); if (++n >= 25 || this.diverged) clearInterval(this.timer); }, 120); },
      reset() { clearInterval(this.timer); this.path = [Number(this.start)]; },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const xr = [-5, 5], yr = this.fn === 'bowl' ? [0, 37] : [0, 6];
        const T = frame(ctx, W, H, xr, yr, P);
        curve(ctx, this.f, xr, T, P.accent, 2.5);
        const pts = this.path.filter(w => isFinite(w) && Math.abs(w) <= 5);
        ctx.strokeStyle = P.amber; ctx.lineWidth = 1.5; ctx.beginPath();
        pts.forEach((w, i) => { const x = T.X(w), y = T.Y(Math.min(yr[1], this.f(w))); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
        pts.forEach((w, i) => dot(ctx, T.X(w), T.Y(Math.min(yr[1], this.f(w))), i === pts.length - 1 ? 6 : 3.5, i === pts.length - 1 ? P.amber : P.panel, P.amber));
        if (!this.diverged && Math.abs(this.w) <= 5) { const w = this.w, g = this.df(w), y0 = this.f(w); curve(ctx, x => y0 + g * (x - w), [w - 1.2, w + 1.2], T, P.pink, 1.5, [4, 4]); }
      }
    },
    template: `
    <div class="card demo-card">
      <div class="row between"><div class="pills"><button :class="{active:fn==='bowl'}" @click="fn='bowl'">Convex bowl</button><button :class="{active:fn==='bumpy'}" @click="fn='bumpy'">Bumpy (local minima)</button></div>
      <div class="small muted">step {{ path.length-1 }} · w = {{ isFinite(w) ? w.toFixed(3) : '∞' }} · loss = {{ isFinite(w) ? f(w).toFixed(3) : '∞' }} · gradient = {{ isFinite(w) ? df(w).toFixed(3) : '—' }}</div></div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <div class="demo-controls">
        <label>Learning rate <b>{{ lr }}</b><input type="range" min="0.01" max="1.1" step="0.01" v-model.number="lr"></label>
        <label>Start at w = <b>{{ start }}</b><input type="range" min="-4.5" max="4.5" step="0.5" v-model.number="start"></label>
      </div>
      <div class="btn-row"><button class="btn sm" @click="step">Take one step</button><button class="btn sm primary" @click="run">Run 25 steps</button><button class="btn sm ghost" @click="reset">Reset</button></div>
      <p v-if="diverged" class="feedback bad">Diverged! The learning rate is so large that every step overshoots the minimum by more than it gains.</p>
      <p class="small muted mt">The dashed pink line is the tangent: its slope is the gradient. Each step moves w by −(learning rate × gradient). Try 0.05 (slow), 0.3 (fast), 0.95 (oscillates) and 1.05 (diverges) on the bowl, then try the bumpy curve from different starting points.</p>
    </div>`
  };

  // 2 ---------- Activation functions ----------
  const ACTS = {
    sigmoid: { f: x => 1 / (1 + Math.exp(-x)), d: x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, color: 'accent' },
    tanh: { f: x => Math.tanh(x), d: x => 1 - Math.tanh(x) ** 2, color: 'purple' },
    relu: { f: x => Math.max(0, x), d: x => (x > 0 ? 1 : 0), color: 'green' },
    'leaky relu': { f: x => (x > 0 ? x : 0.1 * x), d: x => (x > 0 ? 1 : 0.1), color: 'amber' },
    gelu: { f: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))), d: x => { const h = 1e-4; const g = ACTS.gelu.f; return (g(x + h) - g(x - h)) / (2 * h); }, color: 'pink' }
  };
  C['demo-activations'] = {
    mixins: [redrawOnChange],
    data() { return { on: { sigmoid: true, tanh: true, relu: true, 'leaky relu': false, gelu: false }, deriv: false, x: 1 }; },
    computed: { names() { return Object.keys(ACTS); }, rows() { return this.names.filter(n => this.on[n]).map(n => ({ n, f: ACTS[n].f(this.x), d: ACTS[n].d(this.x) })); } },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const xr = [-5, 5], yr = [-1.5, 3]; const T = frame(ctx, W, H, xr, yr, P);
        for (const n of this.names) if (this.on[n]) { curve(ctx, ACTS[n].f, xr, T, P[ACTS[n].color], 2.5); if (this.deriv) curve(ctx, ACTS[n].d, xr, T, P[ACTS[n].color], 1.5, [5, 4]); }
        ctx.strokeStyle = P.muted; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(T.X(this.x), T.Y(yr[0])); ctx.lineTo(T.X(this.x), T.Y(yr[1])); ctx.stroke(); ctx.setLineDash([]);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="pills"><button v-for="n in names" :key="n" :class="{active:on[n]}" @click="on[n]=!on[n]">{{ n }}</button><button :class="{active:deriv}" @click="deriv=!deriv">show derivatives (dashed)</button></div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <label class="demo-controls">Input x = <b>{{ x }}</b><input type="range" min="-5" max="5" step="0.1" v-model.number="x"></label>
      <table class="mini-table"><thead><tr><th>function</th><th>f(x)</th><th>f′(x) (gradient)</th></tr></thead><tbody><tr v-for="r in rows" :key="r.n"><td>{{ r.n }}</td><td class="mono">{{ r.f.toFixed(4) }}</td><td class="mono">{{ r.d.toFixed(4) }}</td></tr></tbody></table>
      <p class="small muted">Notice how the sigmoid and tanh derivatives shrink towards 0 for large |x| (vanishing gradients), while ReLU keeps a gradient of exactly 1 for every positive input.</p>
    </div>`
  };

  // 3 ---------- Fit a line by hand / by gradient descent ----------
  function linData() { const rng = MLA.util.seededRng(7); const pts = []; for (let i = 0; i < 14; i++) { const x = rng() * 8 - 1; pts.push([x, 2 * x + 1 + gauss(rng) * 1.6]); } return pts; }
  C['demo-linear-fit'] = {
    mixins: [redrawOnChange],
    data() { return { pts: linData(), w: 0.5, b: 0, timer: null, residuals: true }; },
    computed: {
      mse() { return this.pts.reduce((s, [x, y]) => s + (this.w * x + this.b - y) ** 2, 0) / this.pts.length; },
      best() { const n = this.pts.length; if (n < 2) return null; const mx = this.pts.reduce((s, p) => s + p[0], 0) / n, my = this.pts.reduce((s, p) => s + p[1], 0) / n; let sxy = 0, sxx = 0; for (const [x, y] of this.pts) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; } const w = sxx ? sxy / sxx : 0; const b = my - w * mx; const mse = this.pts.reduce((s, [x, y]) => s + (w * x + b - y) ** 2, 0) / n; return { w, b, mse }; }
    },
    beforeUnmount() { clearInterval(this.timer); },
    methods: {
      fit() {
        clearInterval(this.timer); let n = 0;
        this.timer = setInterval(() => {
          const N = this.pts.length; let gw = 0, gb = 0;
          for (const [x, y] of this.pts) { const e = this.w * x + this.b - y; gw += 2 * e * x / N; gb += 2 * e / N; }
          this.w = +(this.w - 0.02 * gw).toFixed(4); this.b = +(this.b - 0.02 * gb).toFixed(4);
          if (++n > 160) clearInterval(this.timer);
        }, 30);
      },
      add(e) { const cv = this.$refs.cv; const [px, py] = canvasPoint(cv, e, 560, 300); const [x, y] = this._T.inv(px, py); if (x >= -2 && x <= 8 && y >= -6 && y <= 20) this.pts.push([+x.toFixed(2), +y.toFixed(2)]); },
      resetData() { clearInterval(this.timer); this.pts = linData(); },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const xr = [-2, 8], yr = [-6, 20]; const T = frame(ctx, W, H, xr, yr, P); this._T = T;
        if (this.residuals) { ctx.strokeStyle = P.red; ctx.globalAlpha = 0.6; for (const [x, y] of this.pts) { ctx.beginPath(); ctx.moveTo(T.X(x), T.Y(y)); ctx.lineTo(T.X(x), T.Y(this.w * x + this.b)); ctx.stroke(); } ctx.globalAlpha = 1; }
        curve(ctx, x => this.w * x + this.b, xr, T, P.accent, 2.5);
        for (const [x, y] of this.pts) dot(ctx, T.X(x), T.Y(y), 4.5, P.amber);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="row between"><div class="small">ŷ = <b>{{ w.toFixed(2) }}</b>·x + <b>{{ b.toFixed(2) }}</b> &nbsp; · &nbsp; MSE = <b :style="{color: best && mse < best.mse*1.05 ? 'var(--green)' : ''}">{{ mse.toFixed(3) }}</b></div><div class="small muted" v-if="best">best possible MSE ≈ {{ best.mse.toFixed(3) }} (w={{ best.w.toFixed(2) }}, b={{ best.b.toFixed(2) }})</div></div>
      <canvas ref="cv" class="demo-canvas clickable" @click="add"></canvas>
      <div class="demo-controls">
        <label>Slope w <b>{{ w.toFixed(2) }}</b><input type="range" min="-1" max="4" step="0.01" v-model.number="w"></label>
        <label>Intercept b <b>{{ b.toFixed(2) }}</b><input type="range" min="-4" max="6" step="0.01" v-model.number="b"></label>
      </div>
      <div class="btn-row"><button class="btn sm primary" @click="fit">Fit with gradient descent</button><button class="btn sm ghost" @click="residuals=!residuals">{{ residuals?'Hide':'Show' }} errors</button><button class="btn sm ghost" @click="resetData">Reset data</button></div>
      <p class="small muted mt">Red lines are the errors (residuals). MSE is the average of their squares. Click the chart to add points — try adding an outlier far from the line and watch how much it drags the fit.</p>
    </div>`
  };

  // 4 ---------- Overfitting with polynomials ----------
  function polyData() {
    const rng = MLA.util.seededRng(3); const f = x => Math.sin(x) * 1.5;
    const train = [], test = [];
    for (let i = 0; i < 12; i++) { const x = 0.25 + i * 0.5 + (rng() - 0.5) * 0.3; train.push([x, f(x) + gauss(rng) * 0.35]); }
    for (let i = 0; i < 40; i++) { const x = 0.1 + rng() * 5.8; test.push([x, f(x) + gauss(rng) * 0.35]); }
    return { train, test, f };
  }
  function polyFit(pts, deg) {
    const s = x => (x - 3) / 3; const n = deg + 1;
    const A = Array.from({ length: n }, () => new Array(n + 1).fill(0));
    for (const [x0, y] of pts) { const x = s(x0); const pw = []; for (let j = 0; j < n; j++) pw.push(Math.pow(x, j)); for (let r = 0; r < n; r++) { for (let c = 0; c < n; c++) A[r][c] += pw[r] * pw[c]; A[r][n] += pw[r] * y; } }
    for (let r = 0; r < n; r++) A[r][r] += 1e-9;
    for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r; [A[c], A[p]] = [A[p], A[c]]; const piv = A[c][c] || 1e-12; for (let r = 0; r < n; r++) { if (r === c) continue; const k = A[r][c] / piv; for (let k2 = c; k2 <= n; k2++) A[r][k2] -= k * A[c][k2]; } }
    const coef = A.map((row, i) => row[n] / (row[i] || 1e-12));
    return x0 => { const x = s(x0); let y = 0; for (let j = n - 1; j >= 0; j--) y = y * x + coef[j]; return y; };
  }
  C['demo-overfit-poly'] = {
    mixins: [redrawOnChange],
    data() { return { deg: 1, show: true }; },
    computed: {
      d() { return polyData(); },
      errs() { const out = []; for (let k = 0; k <= 11; k++) { const p = polyFit(this.d.train, k); const m = pts => pts.reduce((s, [x, y]) => s + (p(x) - y) ** 2, 0) / pts.length; out.push({ k, train: m(this.d.train), test: m(this.d.test) }); } return out; },
      cur() { return this.errs[this.deg]; },
      verdict() { const k = this.deg; if (k <= 1) return ['Underfitting', 'The model is too simple to capture the curve: both errors are high (high bias).', 'amber']; if (k >= 8) return ['Overfitting', 'The curve threads through every training point but swings wildly between them: training error is tiny, test error explodes (high variance).', 'red']; return ['Good fit', 'Flexible enough to follow the trend, not so flexible that it memorises the noise.', 'green']; }
    },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 280; const ctx = setup(cv, W, H); const P = palette();
        const xr = [0, 6], yr = [-3, 3]; const T = frame(ctx, W, H, xr, yr, P);
        if (this.show) curve(ctx, this.d.f, xr, T, P.dim, 1.5, [5, 5]);
        curve(ctx, polyFit(this.d.train, this.deg), [0.1, 5.9], T, P.accent, 2.5);
        for (const [x, y] of this.d.test) dot(ctx, T.X(x), T.Y(y), 3, null, P.muted);
        for (const [x, y] of this.d.train) dot(ctx, T.X(x), T.Y(y), 5, P.amber);
        const cv2 = this.$refs.cv2; if (!cv2) return; const W2 = 560, H2 = 150; const c2 = setup(cv2, W2, H2);
        const T2 = frame(c2, W2, H2, [0, 11], [0, 1.6], P, { xt: 11, yt: 3 });
        const line = (key, col) => { c2.strokeStyle = col; c2.lineWidth = 2; c2.beginPath(); this.errs.forEach((e, i) => { const y = T2.Y(Math.min(1.6, e[key])); i ? c2.lineTo(T2.X(e.k), y) : c2.moveTo(T2.X(e.k), y); }); c2.stroke(); };
        line('train', P.amber); line('test', P.purple);
        c2.strokeStyle = P.text; c2.setLineDash([3, 3]); c2.beginPath(); c2.moveTo(T2.X(this.deg), T2.Y(0)); c2.lineTo(T2.X(this.deg), T2.Y(1.6)); c2.stroke(); c2.setLineDash([]);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="row between"><div><span class="chip" :class="verdict[2]">{{ verdict[0] }}</span> <span class="small muted">degree {{ deg }} polynomial</span></div><div class="small">train MSE <b style="color:var(--amber)">{{ cur.train.toFixed(3) }}</b> · test MSE <b style="color:var(--purple)">{{ cur.test > 99 ? '>99' : cur.test.toFixed(3) }}</b></div></div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <label class="demo-controls">Model complexity (polynomial degree) <b>{{ deg }}</b><input type="range" min="0" max="11" step="1" v-model.number="deg"></label>
      <p class="small">{{ verdict[1] }}</p>
      <div class="tiny muted">Error vs. complexity — <span style="color:var(--amber)">training error</span> keeps falling, <span style="color:var(--purple)">test error</span> is U-shaped:</div>
      <canvas ref="cv2" class="demo-canvas short"></canvas>
      <p class="small muted">Orange dots: 12 training points. Hollow dots: unseen test points. Dashed grey: the true function the data came from.</p>
    </div>`
  };

  // 5 ---------- k-nearest neighbours decision regions ----------
  function knnData() { const rng = MLA.util.seededRng(11); const pts = []; for (let i = 0; i < 26; i++) pts.push([2.5 + gauss(rng) * 1.2, 3 + gauss(rng) * 1.1, 0]); for (let i = 0; i < 26; i++) pts.push([5.5 + gauss(rng) * 1.2, 5 + gauss(rng) * 1.1, 1]); for (let i = 0; i < 4; i++) pts.push([4 + gauss(rng) * 2, 4 + gauss(rng) * 2, i % 2]); return pts; }
  C['demo-knn'] = {
    mixins: [redrawOnChange],
    data() { return { pts: knnData(), k: 1, cls: 0 }; },
    methods: {
      predict(x, y) { const d = this.pts.map(p => [(p[0] - x) ** 2 + (p[1] - y) ** 2, p[2]]).sort((a, b) => a[0] - b[0]).slice(0, this.k); const v = d.reduce((s, q) => s + q[1], 0); return v / d.length; },
      add(e) { const [px, py] = canvasPoint(this.$refs.cv, e, 560, 320); const [x, y] = this._T.inv(px, py); this.pts.push([x, y, this.cls]); },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 320; const ctx = setup(cv, W, H); const P = palette();
        const xr = [0, 8], yr = [0, 8]; const T = frame(ctx, W, H, xr, yr, P, { axes: false }); this._T = T;
        const nx = 56, ny = 32; const cw = (T.X(8) - T.X(0)) / nx, ch = (T.Y(0) - T.Y(8)) / ny;
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { const x = (i + 0.5) / nx * 8, y = (j + 0.5) / ny * 8; const p = this.predict(x, y); ctx.fillStyle = p > 0.5 ? P.purple : p < 0.5 ? P.cyan : P.muted; ctx.globalAlpha = 0.14 + Math.abs(p - 0.5) * 0.25; ctx.fillRect(T.X(0) + i * cw, T.Y(8) + (ny - 1 - j) * ch, cw + 0.5, ch + 0.5); }
        ctx.globalAlpha = 1;
        for (const [x, y, c] of this.pts) dot(ctx, T.X(x), T.Y(y), 4.5, c ? P.purple : P.cyan, P.bg);
      }
    },
    template: `
    <div class="card demo-card">
      <canvas ref="cv" class="demo-canvas clickable" @click="add"></canvas>
      <label class="demo-controls">Neighbours k = <b>{{ k }}</b><input type="range" min="1" max="25" step="2" v-model.number="k"></label>
      <div class="row"><span class="small muted">Click to add a point of class:</span><div class="pills"><button :class="{active:cls===0}" @click="cls=0">cyan</button><button :class="{active:cls===1}" @click="cls=1">purple</button></div><button class="btn xs ghost" @click="resetPts">Reset</button></div>
      <p class="small muted mt">Background colour = the class a new point would get. With k = 1 the boundary is jagged and every noisy point carves out its own island (overfitting). Larger k smooths the boundary (more bias, less variance).</p>
    </div>`
  };
  C['demo-knn'].methods.resetPts = function () { this.pts = knnData(); };

  // 6 ---------- k-means step by step ----------
  function blobs() { const rng = MLA.util.seededRng(5); const centers = [[2, 2], [6, 2.5], [4, 6], [6.8, 6.5]]; const pts = []; centers.forEach(c => { for (let i = 0; i < 22; i++) pts.push([c[0] + gauss(rng) * 0.7, c[1] + gauss(rng) * 0.7]); }); return pts; }
  const CL = ['cyan', 'amber', 'purple', 'green', 'pink', 'red'];
  C['demo-kmeans'] = {
    mixins: [redrawOnChange],
    data() { return { pts: blobs(), k: 3, cents: [], assign: [], phase: 'init', iter: 0, seed: 1 }; },
    computed: { inertia() { if (!this.assign.length) return null; return this.pts.reduce((s, p, i) => { const c = this.cents[this.assign[i]]; return s + (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; }, 0); } },
    watch: { k() { this.init(); } },
    mounted() { this.init(); },
    methods: {
      init() { const rng = MLA.util.seededRng(this.seed++); this.cents = MLA.util.shuffle(this.pts, rng).slice(0, this.k).map(p => p.slice()); this.assign = []; this.phase = 'assign'; this.iter = 0; },
      doAssign() { this.assign = this.pts.map(p => { let best = 0, bd = Infinity; this.cents.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; best = j; } }); return best; }); this.phase = 'update'; },
      doUpdate() { let moved = 0; this.cents = this.cents.map((c, j) => { const mine = this.pts.filter((p, i) => this.assign[i] === j); if (!mine.length) return c; const n = [mine.reduce((s, p) => s + p[0], 0) / mine.length, mine.reduce((s, p) => s + p[1], 0) / mine.length]; moved += Math.hypot(n[0] - c[0], n[1] - c[1]); return n; }); this.iter++; this.phase = moved < 1e-6 ? 'done' : 'assign'; },
      next() { if (this.phase === 'assign') this.doAssign(); else if (this.phase === 'update') this.doUpdate(); },
      runAll() { for (let i = 0; i < 100 && this.phase !== 'done'; i++) this.next(); },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 320; const ctx = setup(cv, W, H); const P = palette();
        const T = frame(ctx, W, H, [0, 9], [0, 9], P, { axes: false });
        this.pts.forEach((p, i) => dot(ctx, T.X(p[0]), T.Y(p[1]), 4, this.assign.length ? P[CL[this.assign[i]]] : P.muted));
        this.cents.forEach((c, j) => { const x = T.X(c[0]), y = T.Y(c[1]); ctx.strokeStyle = P.text; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x - 8, y - 8); ctx.lineTo(x + 8, y + 8); ctx.moveTo(x + 8, y - 8); ctx.lineTo(x - 8, y + 8); ctx.stroke(); ctx.strokeStyle = P[CL[j]]; ctx.lineWidth = 1.5; ctx.stroke(); });
      }
    },
    template: `
    <div class="card demo-card">
      <div class="row between"><div class="small">iteration {{ iter }} · next step: <b>{{ phase==='assign' ? 'assign points to nearest centroid' : phase==='update' ? 'move centroids to the mean of their points' : 'converged ✓' }}</b></div><div class="small muted" v-if="inertia!=null">inertia (within-cluster squared distance) = {{ inertia.toFixed(1) }}</div></div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <label class="demo-controls">Number of clusters k = <b>{{ k }}</b><input type="range" min="2" max="6" step="1" v-model.number="k"></label>
      <div class="btn-row"><button class="btn sm primary" :disabled="phase==='done'" @click="next">Next step</button><button class="btn sm" :disabled="phase==='done'" @click="runAll">Run to convergence</button><button class="btn sm ghost" @click="init">New random start</button></div>
      <p class="small muted mt">The data has 4 real groups. Try k = 4 with several random starts: sometimes k-means gets stuck in a worse solution (higher inertia) — which is why libraries run it several times (n_init) and use smarter k-means++ starts.</p>
    </div>`
  };

  // 7 ---------- Threshold → confusion matrix, precision/recall, ROC ----------
  function scoreData() { const rng = MLA.util.seededRng(21); const pos = [], neg = []; for (let i = 0; i < 120; i++) pos.push(Math.min(0.999, Math.max(0.001, 0.66 + gauss(rng) * 0.15))); for (let i = 0; i < 280; i++) neg.push(Math.min(0.999, Math.max(0.001, 0.36 + gauss(rng) * 0.15))); return { pos, neg }; }
  C['demo-threshold'] = {
    mixins: [redrawOnChange],
    data() { return { t: 0.5, curve: 'roc' }; },
    computed: {
      d() { return scoreData(); },
      pr() { const P = this.d.pos.length, pts = []; let ap = 0, prevR = 0; for (let i = 0; i <= 100; i++) { const t = 1 - i / 100; const tp = this.d.pos.filter(s => s >= t).length, fp = this.d.neg.filter(s => s >= t).length; const r = tp / P, p = tp + fp ? tp / (tp + fp) : 1; pts.push([r, p]); ap += (r - prevR) * p; prevR = r; } return { pts, ap, base: P / (P + this.d.neg.length) }; },
      m() { const tp = this.d.pos.filter(s => s >= this.t).length, fn = this.d.pos.length - tp, fp = this.d.neg.filter(s => s >= this.t).length, tn = this.d.neg.length - fp; const prec = tp + fp ? tp / (tp + fp) : 1, rec = tp / (tp + fn), f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0; return { tp, fn, fp, tn, prec, rec, f1, acc: (tp + tn) / (tp + tn + fp + fn), fpr: fp / (fp + tn) }; },
      roc() { const pts = []; for (let i = 0; i <= 100; i++) { const t = 1 - i / 100; pts.push([this.d.neg.filter(s => s >= t).length / this.d.neg.length, this.d.pos.filter(s => s >= t).length / this.d.pos.length]); } let auc = 0; for (let i = 1; i < pts.length; i++) auc += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2; return { pts, auc }; }
    },
    methods: {
      draw() {
        const P = palette();
        const cv = this.$refs.cv; if (!cv) return; const W = 360, H = 220; const ctx = setup(cv, W, H);
        const bins = 25; const hist = arr => { const h = new Array(bins).fill(0); arr.forEach(s => { h[Math.min(bins - 1, Math.floor(s * bins))]++; }); return h; };
        const hp = hist(this.d.pos), hn = hist(this.d.neg); const mx = Math.max(...hp, ...hn);
        const T = frame(ctx, W, H, [0, 1], [0, mx * 1.1], P, { xt: 5, yt: 3 });
        const bw = (T.X(1) - T.X(0)) / bins;
        hn.forEach((v, i) => { ctx.fillStyle = P.cyan; ctx.globalAlpha = 0.55; ctx.fillRect(T.X(i / bins), T.Y(v), bw - 1, T.Y(0) - T.Y(v)); });
        hp.forEach((v, i) => { ctx.fillStyle = P.pink; ctx.globalAlpha = 0.55; ctx.fillRect(T.X(i / bins), T.Y(v), bw - 1, T.Y(0) - T.Y(v)); });
        ctx.globalAlpha = 1; ctx.strokeStyle = P.text; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(T.X(this.t), T.Y(0)); ctx.lineTo(T.X(this.t), T.Y(mx * 1.1)); ctx.stroke();
        const cv2 = this.$refs.cv2; if (!cv2) return; const c2 = setup(cv2, 220, 220); const T2 = frame(c2, 220, 220, [0, 1], [0, 1], P, { xt: 2, yt: 2 });
        const pr = this.curve === 'pr', pts = pr ? this.pr.pts : this.roc.pts;
        c2.strokeStyle = P.dim; c2.setLineDash([4, 4]); c2.beginPath(); if (pr) { c2.moveTo(T2.X(0), T2.Y(this.pr.base)); c2.lineTo(T2.X(1), T2.Y(this.pr.base)); } else { c2.moveTo(T2.X(0), T2.Y(0)); c2.lineTo(T2.X(1), T2.Y(1)); } c2.stroke(); c2.setLineDash([]);
        c2.strokeStyle = P.accent; c2.lineWidth = 2; c2.beginPath(); pts.forEach((p, i) => i ? c2.lineTo(T2.X(p[0]), T2.Y(p[1])) : c2.moveTo(T2.X(p[0]), T2.Y(p[1]))); c2.stroke();
        if (pr) dot(c2, T2.X(this.m.rec), T2.Y(this.m.prec), 5, P.amber, P.text); else dot(c2, T2.X(this.m.fpr), T2.Y(this.m.rec), 5, P.amber, P.text);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="grid cols-2" style="align-items:start;gap:12px">
        <div><div class="tiny muted">Model scores for <span style="color:var(--pink)">actual positives</span> and <span style="color:var(--cyan)">actual negatives</span></div><canvas ref="cv" class="demo-canvas"></canvas></div>
        <div><div class="pills" style="margin-bottom:4px"><button :class="{active: curve==='roc'}" @click="curve='roc'">ROC</button><button :class="{active: curve==='pr'}" @click="curve='pr'">Precision–recall</button></div><div class="tiny muted">{{ curve === 'roc' ? 'ROC curve: recall (true positive rate) against false positive rate. AUC = ' + roc.auc.toFixed(3) : 'Precision against recall. Average precision = ' + pr.ap.toFixed(3) + '; dashed = share of positives (' + pr.base.toFixed(2) + ')' }} · dot = current threshold</div><canvas ref="cv2" class="demo-canvas square"></canvas></div>
      </div>
      <label class="demo-controls">Decision threshold <b>{{ t.toFixed(2) }}</b> — predict “positive” when score ≥ threshold<input type="range" min="0.01" max="0.99" step="0.01" v-model.number="t"></label>
      <div class="grid cols-2" style="gap:12px;align-items:start">
        <table class="mini-table confusion"><thead><tr><th></th><th>predicted +</th><th>predicted −</th></tr></thead><tbody><tr><th>actual +</th><td class="good">TP {{ m.tp }}</td><td class="bad">FN {{ m.fn }}</td></tr><tr><th>actual −</th><td class="bad">FP {{ m.fp }}</td><td class="good">TN {{ m.tn }}</td></tr></tbody></table>
        <table class="mini-table"><tbody><tr><td>Accuracy</td><td class="mono">{{ m.acc.toFixed(3) }}</td></tr><tr><td>Precision = TP/(TP+FP)</td><td class="mono">{{ m.prec.toFixed(3) }}</td></tr><tr><td>Recall = TP/(TP+FN)</td><td class="mono">{{ m.rec.toFixed(3) }}</td></tr><tr><td>F1</td><td class="mono">{{ m.f1.toFixed(3) }}</td></tr><tr><td>False positive rate</td><td class="mono">{{ m.fpr.toFixed(3) }}</td></tr></tbody></table>
      </div>
      <p class="small muted">Raise the threshold: fewer false alarms (precision ↑) but more misses (recall ↓). There is no free lunch — you choose the trade-off your application needs. The ROC curve ignores how rare positives are; the precision–recall curve does not, which is why it is the better picture when positives are rare: a random model's precision sits at the share of positives (the dashed line), not at 0.5.</p>
    </div>`
  };

  // 8 ---------- Softmax with temperature, top-k and top-p ----------
  C['demo-softmax'] = {
    data() { return { tokens: ['cat', 'dog', 'kitten', 'pet', 'car', 'banana', 'quantum'], logits: [3.2, 2.6, 2.2, 1.4, 0.3, -0.5, -1.5], T: 1, k: 7, p: 1, tally: {}, draws: 0 }; },
    computed: {
      probs() { return softmax(this.logits.map(z => z / Math.max(0.05, this.T))); },
      kept() {
        const idx = this.probs.map((p, i) => i).sort((a, b) => this.probs[b] - this.probs[a]);
        const keep = new Set(); let cum = 0;
        for (let r = 0; r < idx.length; r++) { if (r >= this.k) break; keep.add(idx[r]); cum += this.probs[idx[r]]; if (cum >= this.p - 1e-12) break; }
        return keep;
      },
      final() { const s = this.probs.reduce((a, p, i) => a + (this.kept.has(i) ? p : 0), 0); return this.probs.map((p, i) => (this.kept.has(i) ? p / s : 0)); },
      entropy() { return -this.final.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0); }
    },
    methods: {
      sample(n) { for (let j = 0; j < n; j++) { let r = Math.random(), i = 0; for (; i < this.final.length - 1; i++) { r -= this.final[i]; if (r <= 0) break; } const t = this.tokens[i]; this.tally[t] = (this.tally[t] || 0) + 1; this.draws++; } },
      clear() { this.tally = {}; this.draws = 0; }
    },
    template: `
    <div class="card demo-card">
      <div class="small muted">Prompt: “The child wanted to adopt a ___” — the model's raw scores (logits) for 7 candidate tokens:</div>
      <div class="bars mt">
        <div v-for="(t,i) in tokens" :key="t" class="bar-row" :class="{off: !kept.has(i)}">
          <span class="mono bar-label">{{ t }}</span><span class="tiny muted mono" style="width:44px">{{ logits[i].toFixed(1) }}</span>
          <div class="bar"><span :style="{width: (final[i]*100)+'%'}"></span></div><span class="mono tiny" style="width:52px;text-align:right">{{ (final[i]*100).toFixed(1) }}%</span>
        </div>
      </div>
      <div class="demo-controls">
        <label>Temperature T = <b>{{ T.toFixed(2) }}</b><input type="range" min="0.1" max="3" step="0.05" v-model.number="T"></label>
        <label>top-k = <b>{{ k }}</b><input type="range" min="1" max="7" step="1" v-model.number="k"></label>
        <label>top-p = <b>{{ p.toFixed(2) }}</b><input type="range" min="0.05" max="1" step="0.05" v-model.number="p"></label>
      </div>
      <div class="row between"><div class="btn-row"><button class="btn sm primary" @click="sample(1)">Sample 1 token</button><button class="btn sm" @click="sample(100)">Sample 100</button><button class="btn sm ghost" @click="clear">Clear</button></div><span class="small muted">entropy {{ entropy.toFixed(2) }} bits</span></div>
      <div v-if="draws" class="small mt">Sampled ({{ draws }}): <span v-for="(c,t) in tally" :key="t" class="chip" style="margin:2px">{{ t }} × {{ c }}</span></div>
      <p class="small muted mt">Low temperature sharpens the distribution (T → 0 is greedy decoding); high temperature flattens it. top-k keeps the k most likely tokens; top-p keeps the smallest set whose probability adds up to p. Greyed tokens can no longer be sampled.</p>
    </div>`
  };

  // 9 ---------- Attention heat map (toy, hand-built scores) ----------
  const ATT_TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired'];
  const ATT_TYPE = ['det', 'animal', 'verb', 'prep', 'det', 'object', 'conj', 'pron', 'verb', 'adj'];
  function attScores(i, j) {
    const a = ATT_TYPE[i], b = ATT_TYPE[j]; let s = 0;
    if (a === 'pron' && b === 'animal') s += 3.2; if (a === 'pron' && b === 'object') s += 1.2;
    if (a === 'adj' && (b === 'animal' || b === 'pron')) s += 2.4;
    if (a === 'verb' && (b === 'animal' || b === 'pron')) s += 2.0; if (a === 'verb' && b === 'object') s += 0.8;
    if (a === 'det' && j === i + 1) s += 2.2; if (a === 'prep' && b === 'object') s += 2.0;
    if (a === 'object' && b === 'prep') s += 1.4; if (a === 'animal' && b === 'verb') s += 1.0;
    if (a === 'conj' && (b === 'verb')) s += 1.2;
    if (i === j) s += 1.0; s -= 0.12 * Math.abs(i - j);
    return s;
  }
  C['demo-attention'] = {
    data() { return { q: 7, causal: true, scale: 1 }; },
    computed: {
      toks() { return ATT_TOKENS; },
      W() { return this.toks.map((t, i) => { const row = this.toks.map((u, j) => (this.causal && j > i ? -Infinity : attScores(i, j) * this.scale)); const m = Math.max(...row.filter(isFinite)); const e = row.map(x => (isFinite(x) ? Math.exp(x - m) : 0)); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); }); },
      row() { return this.W[this.q]; }
    },
    methods: { cell(v) { return { background: 'color-mix(in srgb, var(--purple) ' + Math.round(v * 100) + '%, transparent)' }; } },
    template: `
    <div class="card demo-card">
      <div class="small">Click a token to make it the <b>query</b>. Bars show how much it attends to each <b>key</b> (each row of weights sums to 1).</div>
      <div class="att-tokens mt"><button v-for="(t,i) in toks" :key="i" class="att-tok" :class="{q: i===q, masked: causal && i>q}" @click="q=i">{{ t }}<span class="att-w" :style="{height: Math.round(row[i]*48)+'px'}"></span><span class="tiny mono">{{ (row[i]*100).toFixed(0) }}%</span></button></div>
      <div class="row mt"><label class="checkbox"><input type="checkbox" v-model="causal"> causal mask (a token may only look at itself and earlier tokens — what GPT uses)</label></div>
      <label class="demo-controls">Score scale (like 1/√d or inverse temperature) <b>{{ scale.toFixed(1) }}</b><input type="range" min="0.2" max="3" step="0.1" v-model.number="scale"></label>
      <div class="scroll-x"><table class="att-matrix"><thead><tr><th class="tiny muted">query ↓ / key →</th><th v-for="(t,j) in toks" :key="j" class="tiny">{{ t }}</th></tr></thead>
      <tbody><tr v-for="(t,i) in toks" :key="i" :class="{sel:i===q}" @click="q=i"><th class="tiny">{{ t }}</th><td v-for="(v,j) in W[i]" :key="j" :style="cell(v)" class="tiny mono">{{ v>0.005 ? (v*100).toFixed(0) : '' }}</td></tr></tbody></table></div>
      <p class="small muted">These scores are hand-made to illustrate the mechanism (a trained model learns them from data via the query and key matrices). Note how “it” attends mostly to “cat” — attention is how a transformer resolves which earlier words matter.</p>
    </div>`
  };

  // 10 ---------- Byte-pair encoding step by step ----------
  C['demo-bpe'] = {
    data() { return { text: 'low lower lowest slow slower newest widest', words: [], merges: [] }; },
    created() { this.reset(); },
    computed: {
      pairs() { const c = {}; for (const w of this.words) for (let i = 0; i < w.length - 1; i++) { const k = w[i] + '\u0001' + w[i + 1]; c[k] = (c[k] || 0) + 1; } return Object.entries(c).map(([k, n]) => ({ a: k.split('\u0001')[0], b: k.split('\u0001')[1], n })).sort((x, y) => y.n - x.n || (x.a + x.b).localeCompare(y.a + y.b)); },
      vocab() { const v = new Set(); for (const w of this.words) for (const t of w) v.add(t); return Array.from(v).sort(); },
      tokenCount() { return this.words.reduce((s, w) => s + w.length, 0); }
    },
    methods: {
      reset() { this.words = this.text.trim().split(/\s+/).filter(Boolean).map(w => w.split('').concat(['_'])); this.merges = []; },
      merge() { const best = this.pairs[0]; if (!best || best.n < 2) { MLA.ui.toast('No pair occurs twice — merging further would just memorise words.', 'info'); return; } const tok = best.a + best.b; this.words = this.words.map(w => { const out = []; for (let i = 0; i < w.length; i++) { if (i < w.length - 1 && w[i] === best.a && w[i + 1] === best.b) { out.push(tok); i++; } else out.push(w[i]); } return out; }); this.merges.push(best.a + ' + ' + best.b + ' → ' + tok + '  (' + best.n + '×)'); }
    },
    template: `
    <div class="card demo-card">
      <div class="inline-form"><input type="text" v-model="text" @change="reset" aria-label="Training text"><button class="btn sm" @click="reset">Restart</button><button class="btn sm primary" @click="merge">Merge most frequent pair</button></div>
      <div class="bpe-words mt"><span v-for="(w,i) in words" :key="i" class="bpe-word"><span v-for="(t,j) in w" :key="j" class="bpe-tok" :class="{end: t.endsWith('_')}">{{ t }}</span></span></div>
      <div class="grid cols-2 mt" style="gap:12px;align-items:start">
        <div><div class="tiny muted">Most frequent adjacent pairs</div><table class="mini-table"><tbody><tr v-for="p in pairs.slice(0,6)" :key="p.a+p.b"><td class="mono">{{ p.a }} + {{ p.b }}</td><td class="mono">{{ p.n }}×</td></tr></tbody></table></div>
        <div><div class="tiny muted">Merges learned ({{ merges.length }}) · vocab {{ vocab.length }} · tokens {{ tokenCount }}</div><ol class="small mono" style="margin:4px 0 0"><li v-for="(m,i) in merges" :key="i">{{ m }}</li></ol></div>
      </div>
      <p class="small muted mt">“_” marks the end of a word. Each merge adds one token to the vocabulary and makes the text shorter. Real tokenizers learn ~50,000–200,000 merges from terabytes of text, on bytes rather than characters.</p>
    </div>`
  };

  // 11 ---------- 2-D convolution ----------
  const KERNELS = { 'vertical edge': [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], 'horizontal edge': [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map(r => r.map(v => v / 9)), sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] };
  function convImage() { const g = []; for (let i = 0; i < 8; i++) { const r = []; for (let j = 0; j < 8; j++) r.push(j >= 2 && j <= 5 && i >= 1 && i <= 6 ? 1 : 0); g.push(r); } g[3][3] = 0; g[4][4] = 0; return g; }
  C['demo-convolution'] = {
    data() { return { img: convImage(), kname: 'vertical edge', hover: null }; },
    computed: {
      K() { return KERNELS[this.kname]; }, names() { return Object.keys(KERNELS); },
      out() { const o = []; for (let i = 0; i < 6; i++) { const r = []; for (let j = 0; j < 6; j++) { let s = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += this.img[i + a][j + b] * this.K[a][b]; r.push(s); } o.push(r); } return o; },
      calc() { if (!this.hover) return ''; const [i, j] = this.hover; const parts = []; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) if (this.K[a][b]) parts.push(this.img[i + a][j + b] + '×' + (+this.K[a][b].toFixed(2))); return parts.join(' + ') + ' = ' + (+this.out[i][j].toFixed(2)); }
    },
    methods: {
      inField(i, j) { if (!this.hover) return false; const [a, b] = this.hover; return i >= a && i < a + 3 && j >= b && j < b + 3; },
      outStyle(v) { const m = 3; const t = Math.max(-1, Math.min(1, v / m)); return { background: t >= 0 ? 'color-mix(in srgb, var(--green) ' + Math.round(t * 100) + '%, var(--panel2))' : 'color-mix(in srgb, var(--red) ' + Math.round(-t * 100) + '%, var(--panel2))' }; }
    },
    template: `
    <div class="card demo-card">
      <div class="pills"><button v-for="n in names" :key="n" :class="{active:kname===n}" @click="kname=n">{{ n }}</button></div>
      <div class="conv-layout mt">
        <div><div class="tiny muted">Input 8×8 (click to toggle pixels)</div><div class="conv-grid g8"><span v-for="(v,k) in img.flat()" :key="k" :class="{on: v, field: inField(Math.floor(k/8), k%8)}" @click="img[Math.floor(k/8)][k%8] = v ? 0 : 1"></span></div></div>
        <div><div class="tiny muted">Kernel 3×3</div><div class="conv-grid g3"><span v-for="(v,k) in K.flat()" :key="k" class="mono">{{ +v.toFixed(2) }}</span></div></div>
        <div><div class="tiny muted">Output 6×6 (hover a cell)</div><div class="conv-grid g6"><span v-for="(v,k) in out.flat()" :key="k" class="mono" :style="outStyle(v)" @mouseenter="hover=[Math.floor(k/6), k%6]" @mouseleave="hover=null">{{ Math.round(v*10)/10 }}</span></div></div>
      </div>
      <p class="small mono mt" style="min-height:1.4em">{{ calc || 'Hover an output cell to see the multiply-and-add that produced it.' }}</p>
      <p class="small muted">The same 9 weights slide over every position. An edge kernel lights up (green/red) only where brightness changes — that is a learned “feature detector” in a CNN.</p>
    </div>`
  };

  // 12 ---------- A single neuron / perceptron on logic gates ----------
  C['demo-neuron'] = {
    mixins: [redrawOnChange],
    data() { return { gate: 'AND', w1: 1, w2: 1, b: -1.5 }; },
    computed: {
      targets() { return { AND: [0, 0, 0, 1], OR: [0, 1, 1, 1], NAND: [1, 1, 1, 0], XOR: [0, 1, 1, 0] }[this.gate]; },
      rows() { return [[0, 0], [0, 1], [1, 0], [1, 1]].map((x, i) => { const z = this.w1 * x[0] + this.w2 * x[1] + this.b; const y = z >= 0 ? 1 : 0; return { x, z, y, t: this.targets[i], ok: y === this.targets[i] }; }); },
      allOk() { return this.rows.every(r => r.ok); }
    },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 300, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const T = frame(ctx, W, H, [-0.5, 1.5], [-0.5, 1.5], P, { xt: 2, yt: 2 });
        const n = 40; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const x = -0.5 + 2 * (i + 0.5) / n, y = -0.5 + 2 * (j + 0.5) / n; const on = this.w1 * x + this.w2 * y + this.b >= 0; ctx.fillStyle = on ? P.green : P.red; ctx.globalAlpha = 0.1; ctx.fillRect(T.X(-0.5 + 2 * i / n), T.Y(-0.5 + 2 * (j + 1) / n), (T.X(1.5) - T.X(-0.5)) / n + 0.5, (T.Y(-0.5) - T.Y(1.5)) / n + 0.5); }
        ctx.globalAlpha = 1;
        this.rows.forEach(r => dot(ctx, T.X(r.x[0]), T.Y(r.x[1]), 9, r.t ? P.green : P.red, r.ok ? P.text : P.amber));
      }
    },
    template: `
    <div class="card demo-card">
      <div class="pills"><button v-for="g in ['AND','OR','NAND','XOR']" :key="g" :class="{active:gate===g}" @click="gate=g">{{ g }}</button></div>
      <div class="grid cols-2 mt" style="gap:14px;align-items:start">
        <canvas ref="cv" class="demo-canvas square"></canvas>
        <div>
          <div class="formula small">z = {{ w1.toFixed(1) }}·x₁ + {{ w2.toFixed(1) }}·x₂ + ({{ b.toFixed(1) }}) &nbsp;→&nbsp; output 1 if z ≥ 0</div>
          <label class="demo-controls">w₁ <b>{{ w1.toFixed(1) }}</b><input type="range" min="-3" max="3" step="0.1" v-model.number="w1"></label>
          <label class="demo-controls">w₂ <b>{{ w2.toFixed(1) }}</b><input type="range" min="-3" max="3" step="0.1" v-model.number="w2"></label>
          <label class="demo-controls">bias b <b>{{ b.toFixed(1) }}</b><input type="range" min="-3" max="3" step="0.1" v-model.number="b"></label>
          <table class="mini-table"><thead><tr><th>x₁</th><th>x₂</th><th>z</th><th>output</th><th>target</th></tr></thead><tbody><tr v-for="(r,i) in rows" :key="i" :class="r.ok?'':'bad-row'"><td>{{ r.x[0] }}</td><td>{{ r.x[1] }}</td><td class="mono">{{ r.z.toFixed(1) }}</td><td>{{ r.y }}</td><td>{{ r.t }}</td></tr></tbody></table>
          <p class="small" :style="{color: allOk ? 'var(--green)' : ''}">{{ allOk ? '✓ All four cases correct!' : (gate==='XOR' ? 'Try as long as you like: no single straight line separates XOR. That is why we need hidden layers.' : 'Adjust the weights until every row matches its target.') }}</p>
        </div>
      </div>
    </div>`
  };

  // 13 ---------- Sinusoidal positional encoding heat map ----------
  C['demo-positional'] = {
    mixins: [redrawOnChange],
    data() { return { d: 32, len: 48, dim: 4 }; },
    methods: {
      pe(pos, i) { const k = Math.floor(i / 2); const ang = pos / Math.pow(10000, (2 * k) / this.d); return i % 2 === 0 ? Math.sin(ang) : Math.cos(ang); },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 240; const ctx = setup(cv, W, H); const P = palette();
        const cw = W / this.d, ch = (H - 20) / this.len;
        for (let p = 0; p < this.len; p++) for (let i = 0; i < this.d; i++) { const v = this.pe(p, i); ctx.fillStyle = v >= 0 ? P.accent : P.amber; ctx.globalAlpha = Math.abs(v); ctx.fillRect(i * cw, p * ch, cw, ch + 0.5); }
        ctx.globalAlpha = 1; ctx.strokeStyle = P.text; ctx.lineWidth = 1.5; ctx.strokeRect(this.dim * cw, 0, cw, this.len * ch);
        ctx.fillStyle = P.muted; ctx.font = '11px system-ui'; ctx.fillText('dimension → (each column is one sine/cosine)   ·   position ↓', 4, H - 5);
        const cv2 = this.$refs.cv2; if (!cv2) return; const c2 = setup(cv2, 560, 140); const T = frame(c2, 560, 140, [0, this.len - 1], [-1.1, 1.1], P, { yt: 2 });
        curve(c2, x => this.pe(x, this.dim), [0, this.len - 1], T, P.accent, 2);
        curve(c2, x => this.pe(x, this.dim + 1 < this.d ? this.dim + 1 : this.dim), [0, this.len - 1], T, P.amber, 1.5, [4, 3]);
      }
    },
    template: `
    <div class="card demo-card">
      <canvas ref="cv" class="demo-canvas"></canvas>
      <label class="demo-controls">Inspect dimension <b>{{ dim }}</b> (and {{ dim+1 }}, dashed)<input type="range" :min="0" :max="d-2" step="2" v-model.number="dim"></label>
      <canvas ref="cv2" class="demo-canvas short"></canvas>
      <p class="small muted">Low dimensions oscillate fast, high dimensions slowly — like the hands of a clock. Every position gets a unique pattern, and the pattern for position p+k is a fixed rotation of the pattern for p, which makes relative positions easy to learn.</p>
    </div>`
  };

  // 14 ---------- Quantization ----------
  function qWeights(outlier) { const rng = MLA.util.seededRng(9); const w = []; for (let i = 0; i < 32; i++) w.push(gauss(rng) * 0.4); if (outlier) w[17] = 6.0; return w; }
  C['demo-quantization'] = {
    mixins: [redrawOnChange],
    data() { return { bits: 4, outlier: false }; },
    computed: {
      w() { return qWeights(this.outlier); },
      q() { const qmax = Math.pow(2, this.bits - 1) - 1; const scale = Math.max(...this.w.map(Math.abs)) / qmax; const ints = this.w.map(v => Math.max(-qmax, Math.min(qmax, Math.round(v / scale)))); return { qmax, scale, ints, deq: ints.map(i => i * scale) }; },
      err() { return this.w.reduce((s, v, i) => s + Math.abs(v - this.q.deq[i]), 0) / this.w.length; },
      levels() { return new Set(this.q.ints).size; }
    },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 240; const ctx = setup(cv, W, H); const P = palette();
        const m = Math.max(1.4, ...this.w.map(Math.abs)) * 1.1; const T = frame(ctx, W, H, [-1, 32], [-m, m], P, { xt: 8, yt: 4 });
        this.w.forEach((v, i) => { ctx.strokeStyle = P.dim; ctx.beginPath(); ctx.moveTo(T.X(i), T.Y(v)); ctx.lineTo(T.X(i), T.Y(this.q.deq[i])); ctx.stroke(); dot(ctx, T.X(i), T.Y(v), 4, P.accent); dot(ctx, T.X(i), T.Y(this.q.deq[i]), 3.5, null, P.amber); });
      }
    },
    template: `
    <div class="card demo-card">
      <canvas ref="cv" class="demo-canvas"></canvas>
      <div class="demo-controls"><label>Bits per weight <b>{{ bits }}</b> ({{ 2*q.qmax+1 }} levels)<input type="range" min="2" max="8" step="1" v-model.number="bits"></label></div>
      <label class="checkbox"><input type="checkbox" v-model="outlier"> add one outlier weight (6.0)</label>
      <div class="small mt">scale = max|w| / {{ q.qmax }} = <b>{{ q.scale.toFixed(4) }}</b> · mean absolute error <b>{{ err.toFixed(4) }}</b> · distinct values used <b>{{ levels }}</b> · memory vs FP32: <b>{{ (bits/32*100).toFixed(1) }}%</b></div>
      <p class="small muted mt"><span style="color:var(--accent)">●</span> original weight, <span style="color:var(--amber)">○</span> after quantize → dequantize. Turn on the outlier: the scale is set by the largest value, so all the small weights collapse onto just a few levels. That is why real schemes quantize in small blocks with their own scale (e.g. 32–128 weights per block).</p>
    </div>`
  };

  // 15 ---------- Embedding space ----------
  const EMB = { man: [1, 1], woman: [1, 3], king: [4, 1.2], queen: [4, 3.2], boy: [0.2, 1.3], girl: [0.2, 3.3], prince: [3.1, 1.4], princess: [3.1, 3.4], apple: [7.5, 6.8], banana: [8.2, 7.4], cherry: [7.9, 6.1], car: [1.5, 7.6], bus: [2.4, 8.2], truck: [2.0, 7.0], dog: [6.2, 2.0], cat: [6.8, 2.6], puppy: [5.6, 1.6], kitten: [6.3, 3.1] };
  C['demo-embeddings'] = {
    mixins: [redrawOnChange],
    data() { return { sel: 'king', a: 'king', b: 'man', c: 'woman', mode: 'near' }; },
    computed: {
      words() { return Object.keys(EMB); },
      cos() { return (u, v) => { const d = u[0] * v[0] + u[1] * v[1]; return d / (Math.hypot(u[0], u[1]) * Math.hypot(v[0], v[1])); }; },
      near() { const v = EMB[this.sel]; return this.words.filter(w => w !== this.sel).map(w => ({ w, d: Math.hypot(EMB[w][0] - v[0], EMB[w][1] - v[1]), cos: this.cos(EMB[w], v) })).sort((x, y) => x.d - y.d).slice(0, 4); },
      analogy() { const A = EMB[this.a], B = EMB[this.b], Cc = EMB[this.c]; const p = [A[0] - B[0] + Cc[0], A[1] - B[1] + Cc[1]]; const best = this.words.filter(w => ![this.a, this.b, this.c].includes(w)).map(w => ({ w, d: Math.hypot(EMB[w][0] - p[0], EMB[w][1] - p[1]) })).sort((x, y) => x.d - y.d)[0]; return { p, best }; }
    },
    methods: {
      pick(e) { const [px, py] = canvasPoint(this.$refs.cv, e, 560, 320); let best = null, bd = 1e9; for (const w of this.words) { const x = this._T.X(EMB[w][0]), y = this._T.Y(EMB[w][1]); const d = Math.hypot(x - px, y - py); if (d < bd) { bd = d; best = w; } } if (bd < 30) this.sel = best; },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 320; const ctx = setup(cv, W, H); const P = palette();
        const T = frame(ctx, W, H, [-0.5, 9], [0, 9], P, { axes: false }); this._T = T;
        if (this.mode === 'near') { const v = EMB[this.sel]; this.near.forEach(n => { ctx.strokeStyle = P.amber; ctx.globalAlpha = 0.7; ctx.beginPath(); ctx.moveTo(T.X(v[0]), T.Y(v[1])); ctx.lineTo(T.X(EMB[n.w][0]), T.Y(EMB[n.w][1])); ctx.stroke(); ctx.globalAlpha = 1; }); }
        else { const A = EMB[this.a], B = EMB[this.b], Cc = EMB[this.c], p = this.analogy.p; const arrow = (u, v, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(T.X(u[0]), T.Y(u[1])); ctx.lineTo(T.X(v[0]), T.Y(v[1])); ctx.stroke(); }; arrow(B, A, P.pink); arrow(Cc, p, P.pink); dot(ctx, T.X(p[0]), T.Y(p[1]), 7, null, P.pink); ctx.lineWidth = 1; }
        ctx.font = '12px system-ui'; ctx.textAlign = 'left';
        for (const w of this.words) { const [x, y] = EMB[w]; const hl = w === this.sel && this.mode === 'near'; dot(ctx, T.X(x), T.Y(y), hl ? 6 : 4, hl ? P.amber : P.accent); ctx.fillStyle = hl ? P.amber : P.text; ctx.fillText(w, T.X(x) + 7, T.Y(y) + 4); }
      }
    },
    template: `
    <div class="card demo-card">
      <div class="pills"><button :class="{active:mode==='near'}" @click="mode='near'">Nearest neighbours</button><button :class="{active:mode==='analogy'}" @click="mode='analogy'">Vector arithmetic</button></div>
      <canvas ref="cv" class="demo-canvas clickable" @click="pick"></canvas>
      <div v-if="mode==='near'" class="small">Click a word. Nearest to <b>{{ sel }}</b>: <span v-for="n in near" :key="n.w" class="chip" style="margin:2px">{{ n.w }} · cos {{ n.cos.toFixed(2) }}</span></div>
      <div v-else class="inline-form small"><select v-model="a"><option v-for="w in words" :key="w">{{ w }}</option></select> − <select v-model="b"><option v-for="w in words" :key="w">{{ w }}</option></select> + <select v-model="c"><option v-for="w in words" :key="w">{{ w }}</option></select> ≈ <b>{{ analogy.best.w }}</b></div>
      <p class="small muted mt">A toy 2-D space (real embeddings have hundreds of dimensions and are learned, not hand-placed). Similar meanings sit close together, and directions can encode relationships: the arrow man → woman is roughly the same as king → queen.</p>
    </div>`
  };

  // 17 ---------- Calibration: a reliability diagram and temperature scaling ----------
  function calibrationData() {
    const rng = MLA.util.seededRng(21); const rows = [];
    for (let i = 0; i < 3000; i++) {
      const z = [0, 1, 2, 3].map(() => gauss(rng) * 1.5);                 // true logits of a 4-class problem
      const p = softmax(z); let u = rng(), y = 0;
      while (y < 3 && u > p[y]) { u -= p[y]; y++; }                          // draw the true label
      rows.push({ z: z.map(v => v * 2.5), y });                              // an overconfident model: logits 2.5x too large
    }
    return rows;
  }
  C['demo-calibration'] = {
    mixins: [redrawOnChange],
    data() { return { T: 1, rows: calibrationData() }; },
    computed: {
      stats() {
        const bins = Array.from({ length: 10 }, () => ({ n: 0, conf: 0, acc: 0 })); let correct = 0;
        for (const r of this.rows) {
          const p = softmax(r.z.map(v => v / this.T)); const conf = Math.max(...p); const pred = p.indexOf(conf);
          const b = bins[Math.min(9, Math.floor(conf * 10))]; b.n++; b.conf += conf; b.acc += pred === r.y ? 1 : 0; correct += pred === r.y ? 1 : 0;
        }
        let ece = 0; bins.forEach(b => { if (b.n) { b.conf /= b.n; b.acc /= b.n; ece += b.n / this.rows.length * Math.abs(b.acc - b.conf); } });
        return { bins, ece, accuracy: correct / this.rows.length };
      }
    },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const T = frame(ctx, W, H, [0, 1], [0, 1], P, { axes: false, xt: 5, yt: 5 });
        this.stats.bins.forEach((b, i) => {
          if (!b.n) return; const x0 = T.X(i / 10) + 2, x1 = T.X((i + 1) / 10) - 2;
          ctx.fillStyle = P.accent; ctx.globalAlpha = 0.75; ctx.fillRect(x0, T.Y(b.acc), x1 - x0, T.Y(0) - T.Y(b.acc)); ctx.globalAlpha = 1;
          ctx.strokeStyle = P.red; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x0, T.Y(b.conf)); ctx.lineTo(x1, T.Y(b.conf)); ctx.stroke();
        });
        curve(ctx, x => x, [0, 1], T, P.muted, 1.5, [5, 4]);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="small muted">accuracy <b>{{ (stats.accuracy * 100).toFixed(1) }}%</b> · expected calibration error <b>{{ stats.ece.toFixed(3) }}</b> · temperature T = <b>{{ T.toFixed(2) }}</b></div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <div class="demo-controls"><label>Temperature T (logits are divided by T) <b>{{ T.toFixed(2) }}</b><input type="range" min="0.5" max="5" step="0.05" v-model.number="T"></label></div>
      <p class="small muted mt">Blue bars: accuracy of the predictions in each confidence bin (0–10%, …, 90–100%). Red ticks: their average confidence. Dashed line: perfect calibration. This model is overconfident (red above blue). Raise T until the ticks meet the bars — the error is smallest near T ≈ 2.5 — and notice that accuracy never changes, because dividing all logits by T keeps their order.</p>
    </div>`
  };

  // 18 ---------- Diffusion: the forward (noising) process ----------
  function diffusionData() {
    const rng = MLA.util.seededRng(4); const pts = [];
    for (let i = 0; i < 300; i++) { const c = i % 3; const ang = 2 * Math.PI * c / 3; pts.push({ x: [2 * Math.cos(ang) + gauss(rng) * 0.25, 2 * Math.sin(ang) + gauss(rng) * 0.25], e: [gauss(rng), gauss(rng)], c }); }
    return pts;
  }
  const DIFF_BETAS = Array.from({ length: 1000 }, (_, i) => 1e-4 + (0.02 - 1e-4) * i / 999);
  const DIFF_ABAR = (() => { let a = 1; return DIFF_BETAS.map(b => (a *= 1 - b)); })();
  C['demo-diffusion'] = {
    mixins: [redrawOnChange],
    data() { return { t: 0, pts: diffusionData() }; },
    computed: { abar() { return DIFF_ABAR[this.t]; } },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 300, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const T = frame(ctx, W, H, [-4, 4], [-4, 4], P, { xt: 4, yt: 4 });
        const s = Math.sqrt(this.abar), n = Math.sqrt(1 - this.abar); const cols = [P.accent, P.amber, P.green];
        for (const p of this.pts) dot(ctx, T.X(s * p.x[0] + n * p.e[0]), T.Y(s * p.x[1] + n * p.e[1]), 2.6, cols[p.c]);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="small muted">step t = <b>{{ t }}</b> · signal weight √ᾱ = <b>{{ Math.sqrt(abar).toFixed(3) }}</b> · noise weight √(1 − ᾱ) = <b>{{ Math.sqrt(1 - abar).toFixed(3) }}</b></div>
      <canvas ref="cv" class="demo-canvas square"></canvas>
      <div class="demo-controls"><label>Diffusion step t <b>{{ t }}</b><input type="range" min="0" max="999" step="1" v-model.number="t"></label></div>
      <p class="small muted mt">Three clusters of 2-D data, noised with the linear schedule of the original DDPM paper (β from 0.0001 to 0.02 over 1,000 steps). Each point keeps its own noise sample, so you can watch the clusters dissolve into one Gaussian cloud. A diffusion model learns to run this film backwards.</p>
    </div>`
  };

  // 19 ---------- LLM serving: batching on the roofline ----------
  C['demo-roofline'] = {
    mixins: [redrawOnChange],
    data() { return { batch: 16, ctxK: 4, paramsB: 8 }; },
    computed: {
      model() {
        const peak = 990e12, bw = 3.35e12, kvTok = 2 * 32 * 8 * 128 * 2; const params = this.paramsB * 1e9;
        const step = b => Math.max(2 * params * b / peak, (params * 2 + b * this.ctxK * 1000 * kvTok) / bw);
        const memBound = b => (params * 2 + b * this.ctxK * 1000 * kvTok) / bw > 2 * params * b / peak;
        return { step, memBound };
      },
      now() { const t = this.model.step(this.batch); return { ms: t * 1000, user: 1 / t, total: this.batch / t, mem: this.model.memBound(this.batch) }; }
    },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const bs = Array.from({ length: 10 }, (_, i) => 2 ** i); const tot = bs.map(b => b / this.model.step(b)); const ymax = Math.max(...tot) * 1.1;
        const T = frame(ctx, W, H, [0, 9], [0, ymax], P, { xt: 9 });
        ctx.strokeStyle = P.accent; ctx.lineWidth = 2.5; ctx.beginPath(); bs.forEach((b, i) => { const x = T.X(i), y = T.Y(tot[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
        bs.forEach((b, i) => { dot(ctx, T.X(i), T.Y(tot[i]), b === this.batch ? 7 : 3.5, b === this.batch ? P.amber : (this.model.memBound(b) ? P.accent : P.red)); });
      }
    },
    template: `
    <div class="card demo-card">
      <div class="small muted">batch <b>{{ batch }}</b>: one decode step takes <b>{{ now.ms.toFixed(2) }} ms</b> ({{ now.mem ? 'memory-bound' : 'compute-bound' }}) · each user gets <b>{{ now.user.toFixed(0) }}</b> tokens/s · total <b>{{ now.total.toFixed(0) }}</b> tokens/s</div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <div class="demo-controls">
        <label>Batch size (concurrent requests) <b>{{ batch }}</b><input type="range" min="0" max="9" step="1" :value="Math.log2(batch)" @input="batch = 2 ** Number($event.target.value)"></label>
        <label>Context already in each sequence <b>{{ ctxK }}k tokens</b><input type="range" min="1" max="32" step="1" v-model.number="ctxK"></label>
        <label>Model size <b>{{ paramsB }}B parameters</b><input type="range" min="1" max="70" step="1" v-model.number="paramsB"></label>
      </div>
      <p class="small muted mt">Total throughput (tokens per second, vertical axis) against log₂ of the batch size (horizontal: 0 = 1 request, 9 = 512) for decoding on a GPU like an H100 (≈ 990 TFLOP/s, 3.35 TB/s), with bf16 weights and a grouped-query-attention KV cache of 128 KiB per token. Blue points are memory-bound, red ones compute-bound. Batching shares each read of the weights across many users — until the growing KV cache or the compute limit catches up. A simplified model: real servers add scheduling, prefill and kernel overheads.</p>
    </div>`
  };

  // 20 ---------- Deep RL: PPO's clipped objective ----------
  C['demo-ppo-clip'] = {
    mixins: [redrawOnChange],
    data() { return { eps: 0.2, adv: 1 }; },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const A = this.adv, e = this.eps; const yr = A > 0 ? [-0.2, 2.2] : [-2.2, 0.2];
        const T = frame(ctx, W, H, [0, 2.2], yr, P);
        curve(ctx, r => r * A, [0, 2.2], T, P.muted, 1.5, [5, 4]);
        curve(ctx, r => Math.min(r * A, Math.min(Math.max(r, 1 - e), 1 + e) * A), [0, 2.2], T, P.accent, 3);
        ctx.strokeStyle = P.amber; ctx.setLineDash([3, 3]); [1 - e, 1 + e].forEach(r => { ctx.beginPath(); ctx.moveTo(T.X(r), T.Y(yr[0])); ctx.lineTo(T.X(r), T.Y(yr[1])); ctx.stroke(); }); ctx.setLineDash([]);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="row between"><div class="pills"><button :class="{active:adv>0}" @click="adv=1">Good action (A = +1)</button><button :class="{active:adv<0}" @click="adv=-1">Bad action (A = −1)</button></div>
      <div class="small muted">clip range: ratio between <b>{{ (1 - eps).toFixed(2) }}</b> and <b>{{ (1 + eps).toFixed(2) }}</b></div></div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <div class="demo-controls"><label>Clip parameter ε <b>{{ eps.toFixed(2) }}</b><input type="range" min="0.05" max="0.5" step="0.01" v-model.number="eps"></label></div>
      <p class="small muted mt">Horizontal axis: the ratio r = π_new(a|s) / π_old(a|s) for the action taken. Dashed grey: the unclipped objective r·A. Blue: PPO's min(r·A, clip(r, 1 − ε, 1 + ε)·A). Where the blue line is flat, the gradient is zero — the update has no reason to move the policy further. For a bad action the objective still falls as r grows: PPO never ignores a change that makes a bad action more likely.</p>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
