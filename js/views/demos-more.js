/* ML Atlas – more interactive demos (linear algebra, optimization, probabilistic ML, PCA, schedules).
 * Uses the canvas helpers that js/views/demos.js exports as MLA.demoKit. */
(function (global) {
  const MLA = global.MLA;
  const C = MLA.components;
  const { palette, setup, frame, curve, dot, gauss, canvasPoint, redrawOnChange } = MLA.demoKit;

  // Equal-aspect axes centred on the canvas, `scale` pixels per unit, a grid line every unit.
  function eqFrame(ctx, W, H, scale, P) {
    const cx = W / 2, cy = H / 2;
    const X = x => cx + x * scale, Y = y => cy - y * scale;
    ctx.strokeStyle = P.border; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
    const nx = Math.ceil(cx / scale), ny = Math.ceil(cy / scale);
    for (let i = -nx; i <= nx; i++) { ctx.beginPath(); ctx.moveTo(X(i), 0); ctx.lineTo(X(i), H); ctx.stroke(); }
    for (let j = -ny; j <= ny; j++) { ctx.beginPath(); ctx.moveTo(0, Y(j)); ctx.lineTo(W, Y(j)); ctx.stroke(); }
    ctx.globalAlpha = 1; ctx.strokeStyle = P.dim; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), H); ctx.stroke();
    return { X, Y };
  }
  function arrow(ctx, x0, y0, x1, y1, color, width) {
    const len = Math.hypot(x1 - x0, y1 - y0); if (len < 1) return;
    const a = Math.atan2(y1 - y0, x1 - x0), h = Math.min(10, len * 0.6);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width || 2.5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * h * 0.7, y1 - Math.sin(a) * h * 0.7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - h * Math.cos(a - 0.42), y1 - h * Math.sin(a - 0.42)); ctx.lineTo(x1 - h * Math.cos(a + 0.42), y1 - h * Math.sin(a + 0.42)); ctx.closePath(); ctx.fill();
  }
  function text(ctx, s, x, y, color, align) { ctx.font = '12px system-ui, sans-serif'; ctx.fillStyle = color; ctx.textAlign = align || 'left'; ctx.fillText(s, x, y); }
  const f2 = v => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
  const deg = (y, x) => ((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360;

  // 21 ---------- Eigenvectors: what a 2×2 matrix does to directions ----------
  const EIG_PRESETS = { Symmetric: [2, 1, 1, 2], Stretch: [2, 0, 0, 0.5], Shear: [1, 1, 0, 1], Rotation: [0, -1, 1, 0], 'Rank 1': [1, 1, 1, 1] };
  function eig2(a, b, c, d) {
    const tr = a + d, det = a * d - b * c, disc = tr * tr / 4 - det;
    if (disc < -1e-12) return { real: false, re: tr / 2, im: Math.sqrt(-disc), det, ls: [], vecs: [] };
    const s = Math.sqrt(Math.max(0, disc)), ls = s < 1e-9 ? [tr / 2] : [tr / 2 + s, tr / 2 - s];
    if (Math.abs(b) < 1e-9 && Math.abs(c) < 1e-9 && Math.abs(a - d) < 1e-9) return { real: true, all: true, ls, det, vecs: [] };
    const vec = l => {
      let v;
      if (Math.abs(b) > 1e-9) v = [b, l - a]; else if (Math.abs(c) > 1e-9) v = [l - d, c]; else v = Math.abs(l - a) < 1e-9 ? [1, 0] : [0, 1];
      const n = Math.hypot(v[0], v[1]); return [v[0] / n, v[1] / n];
    };
    return { real: true, all: false, ls, det, vecs: ls.map(l => ({ l, v: vec(l) })) };
  }
  // Singular values of [[a, b], [c, d]] and the input directions (right singular vectors) that become the ellipse's axes.
  function svd2(a, b, c, d) {
    const m11 = a * a + c * c, m22 = b * b + d * d, m12 = a * b + c * d;
    const p = (m11 + m22) / 2, q = Math.sqrt((m11 - m22) * (m11 - m22) / 4 + m12 * m12);
    const phi = 0.5 * Math.atan2(2 * m12, m11 - m22);
    return { s1: Math.sqrt(p + q), s2: Math.sqrt(Math.max(0, p - q)), v1: [Math.cos(phi), Math.sin(phi)], v2: [-Math.sin(phi), Math.cos(phi)] };
  }
  C['demo-eigen'] = {
    mixins: [redrawOnChange],
    data() { return { a: 2, b: 1, c: 1, d: 2, theta: 20 }; },
    computed: {
      presets() { return EIG_PRESETS; },
      eig() { return eig2(this.a, this.b, this.c, this.d); },
      sv() { return svd2(this.a, this.b, this.c, this.d); },
      v() { const t = this.theta * Math.PI / 180; return [Math.cos(t), Math.sin(t)]; },
      Av() { const [x, y] = this.v; return [this.a * x + this.b * y, this.c * x + this.d * y]; },
      rel() {
        const [x, y] = this.v, [p, q] = this.Av, n = Math.hypot(p, q);
        if (n < 1e-9) return { parallel: true, l: 0, n };
        const cross = x * q - y * p, dt = x * p + y * q;
        return { parallel: Math.abs(cross) / n < 0.0175, l: dt, n, angle: Math.atan2(Math.abs(cross), dt) * 180 / Math.PI };
      },
      eigText() {
        const e = this.eig;
        if (!e.real) return 'complex eigenvalues ' + f2(e.re) + ' ± ' + f2(e.im) + 'i: A turns every direction, so there is no real eigenvector';
        if (e.all) return 'λ = ' + f2(e.ls[0]) + ' in every direction (A is a multiple of the identity)';
        if (e.ls.length === 1) return 'a repeated eigenvalue λ = ' + f2(e.ls[0]) + ' with only one eigen-direction';
        return 'eigenvalues λ₁ = ' + f2(e.ls[0]) + ', λ₂ = ' + f2(e.ls[1]);
      }
    },
    methods: {
      f2,
      setPreset(k) { const m = EIG_PRESETS[k]; this.a = m[0]; this.b = m[1]; this.c = m[2]; this.d = m[3]; },
      isPreset(k) { const m = EIG_PRESETS[k]; return this.a === m[0] && this.b === m[1] && this.c === m[2] && this.d === m[3]; },
      pick(e) { const [px, py] = canvasPoint(this.$refs.cv, e, 300, 300); if (Math.hypot(px - 150, py - 150) < 4) return; this.theta = deg(150 - py, px - 150); },
      powerStep() { const [p, q] = this.Av; if (Math.hypot(p, q) < 1e-9) return; this.theta = deg(q, p); },
      snap() {
        const [x, y] = this.v; let best = null, bestDot = -2;
        for (const { v } of this.eig.vecs) for (const s of [1, -1]) { const dd = s * (v[0] * x + v[1] * y); if (dd > bestDot) { bestDot = dd; best = [s * v[0], s * v[1]]; } }
        if (best) this.theta = deg(best[1], best[0]);
      },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 300, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const T = eqFrame(ctx, W, H, 40, P), A = this;
        const map = (x, y) => [A.a * x + A.b * y, A.c * x + A.d * y];
        ctx.setLineDash([4, 4]); ctx.strokeStyle = P.muted; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(T.X(0), T.Y(0), 40, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
        ctx.strokeStyle = P.accent; ctx.lineWidth = 2; ctx.beginPath();
        for (let i = 0; i <= 180; i++) { const t = i / 180 * 2 * Math.PI; const [p, q] = map(Math.cos(t), Math.sin(t)); i ? ctx.lineTo(T.X(p), T.Y(q)) : ctx.moveTo(T.X(p), T.Y(q)); }
        ctx.stroke();
        ctx.strokeStyle = P.purple; ctx.lineWidth = 1.5;
        for (const v of [this.sv.v1, this.sv.v2]) { const [p, q] = map(v[0], v[1]); ctx.beginPath(); ctx.moveTo(T.X(-p), T.Y(-q)); ctx.lineTo(T.X(p), T.Y(q)); ctx.stroke(); }
        ctx.strokeStyle = P.green; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
        for (const { v } of this.eig.vecs) { ctx.beginPath(); ctx.moveTo(T.X(-5 * v[0]), T.Y(-5 * v[1])); ctx.lineTo(T.X(5 * v[0]), T.Y(5 * v[1])); ctx.stroke(); }
        ctx.setLineDash([]);
        for (const { l, v } of this.eig.vecs) { const s = v[1] < -0.2 ? -1 : 1; text(ctx, 'λ = ' + f2(l), T.X(s * 3 * v[0]) + 5, T.Y(s * 3 * v[1]) - 5, P.green); }
        arrow(ctx, T.X(0), T.Y(0), T.X(this.Av[0]), T.Y(this.Av[1]), P.pink, 3);
        arrow(ctx, T.X(0), T.Y(0), T.X(this.v[0]), T.Y(this.v[1]), P.amber, 3);
        text(ctx, 'v', T.X(this.v[0] * 1.3), T.Y(this.v[1] * 1.3) + 4, P.amber, 'center');
        if (this.rel.n > 0.05) text(ctx, 'Av', T.X(this.Av[0]) + 7, T.Y(this.Av[1]) - 7, P.pink);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="pills"><button v-for="(m, k) in presets" :key="k" :class="{active: isPreset(k)}" @click="setPreset(k)">{{ k }}</button></div>
      <div class="small muted">A = [[{{ a.toFixed(1) }}, {{ b.toFixed(1) }}], [{{ c.toFixed(1) }}, {{ d.toFixed(1) }}]] · {{ eigText }} · det(A) = <b>{{ f2(eig.det) }}</b> · singular values σ₁ = <b>{{ f2(sv.s1) }}</b>, σ₂ = <b>{{ f2(sv.s2) }}</b></div>
      <canvas ref="cv" class="demo-canvas square clickable" @click="pick"></canvas>
      <p v-if="rel.parallel" class="feedback ok">A v = {{ f2(rel.l) }} · v, so v is an eigenvector{{ rel.n < 1e-6 ? ': A squashes it to zero (λ = 0)' : rel.l < 0 ? ': A flips it' : '' }}.</p>
      <p v-else class="small muted">A v is turned {{ rel.angle.toFixed(0) }}° away from v (and has length {{ f2(rel.n) }}), so v is not an eigenvector.</p>
      <div class="demo-controls">
        <label>A₁₁ <b>{{ a.toFixed(1) }}</b><input type="range" min="-2" max="2" step="0.1" v-model.number="a"></label>
        <label>A₁₂ <b>{{ b.toFixed(1) }}</b><input type="range" min="-2" max="2" step="0.1" v-model.number="b"></label>
        <label>A₂₁ <b>{{ c.toFixed(1) }}</b><input type="range" min="-2" max="2" step="0.1" v-model.number="c"></label>
        <label>A₂₂ <b>{{ d.toFixed(1) }}</b><input type="range" min="-2" max="2" step="0.1" v-model.number="d"></label>
        <label>Direction of v <b>{{ theta.toFixed(0) }}°</b><input type="range" min="0" max="359" step="1" v-model.number="theta"></label>
      </div>
      <div class="btn-row"><button class="btn sm primary" @click="powerStep">Power-iteration step: v ← Av / ‖Av‖</button><button class="btn sm" :disabled="!eig.vecs.length" @click="snap">Snap v to the nearest eigenvector</button></div>
      <p class="small muted mt">The dashed circle holds every unit vector; the blue ellipse is where A sends them. Amber: a unit vector v (use the slider or click the plot); pink: A v. Green dashed lines are the eigenvector directions, where A v points along v, scaled by λ. Purple: the ellipse's semi-axes, whose lengths are the singular values σ₁ ≥ σ₂ of the SVD; |det A| = σ₁σ₂ is how much A scales areas. Press the power-iteration button a few times: v swings towards the eigenvector with the largest |λ|. Then try the presets: a rotation has no real eigenvectors, a shear has only one, and a rank-1 matrix squashes the circle onto a line (σ₂ = 0, det = 0).</p>
    </div>`
  };

  // 22 ---------- Conditioning: gradient descent in a narrow valley ----------
  C['demo-conditioning'] = {
    mixins: [redrawOnChange],
    data() { return { kappa: 10, c: 1.8, beta: 0 }; },
    computed: {
      eta() { return this.c / this.kappa; },
      run() {
        let x = -4.5, y = 2, vx = 0, vy = 0; const pts = [[x, y]];
        for (let k = 1; k <= 5000; k++) {
          vx = this.beta * vx - this.eta * x; vy = this.beta * vy - this.eta * this.kappa * y;
          x += vx; y += vy;
          if (pts.length < 400) pts.push([x, y]);
          if (!isFinite(x) || !isFinite(y) || Math.abs(x) + Math.abs(y) > 1e6) return { pts, steps: null, diverged: true };
          if (Math.hypot(x, y) < 0.01) return { pts, steps: k, diverged: false };
        }
        return { pts, steps: null, diverged: false };
      },
      verdict() {
        const r = this.run;
        if (r.diverged) return 'Diverged: the step is too large for the steep direction, so every step overshoots by more than it gains.';
        return r.steps ? 'Reached the minimum (within 0.01) in ' + r.steps + ' steps.' : 'Still not within 0.01 of the minimum after 5,000 steps.';
      }
    },
    methods: {
      bestPlain() { this.beta = 0; this.c = +(2 * this.kappa / (this.kappa + 1)).toFixed(2); },
      bestMomentum() { const r = Math.sqrt(this.kappa); this.beta = +(((r - 1) / (r + 1)) ** 2).toFixed(2); this.c = +(4 * this.kappa / ((r + 1) ** 2)).toFixed(2); },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const T = frame(ctx, W, H, [-5, 5], [-2.6, 2.6], P, { xt: 10, yt: 4 });
        const sx = T.X(1) - T.X(0), sy = T.Y(0) - T.Y(1);
        ctx.save(); ctx.beginPath(); ctx.rect(T.pad.l, T.pad.t, W - T.pad.l - T.pad.r, H - T.pad.t - T.pad.b); ctx.clip();
        ctx.strokeStyle = P.accent; ctx.lineWidth = 1; ctx.globalAlpha = 0.4;
        for (let i = 0; i < 22; i++) { const f = 0.05 * Math.pow(1.7, i); ctx.beginPath(); ctx.ellipse(T.X(0), T.Y(0), Math.sqrt(2 * f) * sx, Math.sqrt(2 * f / this.kappa) * sy, 0, 0, 2 * Math.PI); ctx.stroke(); }
        ctx.globalAlpha = 1;
        const pts = this.run.pts, cl = v => Math.max(-50, Math.min(50, v));
        ctx.strokeStyle = P.amber; ctx.lineWidth = 1.5; ctx.beginPath();
        pts.forEach(([x, y], i) => { i ? ctx.lineTo(T.X(cl(x)), T.Y(cl(y))) : ctx.moveTo(T.X(x), T.Y(y)); });
        ctx.stroke();
        pts.slice(0, 150).forEach(([x, y]) => { if (Math.abs(x) < 6 && Math.abs(y) < 3) dot(ctx, T.X(x), T.Y(y), 2.5, P.amber); });
        dot(ctx, T.X(0), T.Y(0), 5, P.green);
        dot(ctx, T.X(pts[0][0]), T.Y(pts[0][1]), 5, P.panel, P.amber);
        ctx.restore();
      }
    },
    template: `
    <div class="card demo-card">
      <div class="small muted">condition number κ = λ_max / λ_min = <b>{{ kappa }}</b> · step size η = <b>{{ eta.toPrecision(3) }}</b> (stable only below 2 / λ_max = {{ (2 / kappa).toPrecision(3) }}{{ beta > 0 ? '; with momentum, below ' + (2 * (1 + beta) / kappa).toPrecision(3) : '' }}) · momentum β = <b>{{ beta.toFixed(2) }}</b></div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <p :class="run.diverged ? 'feedback bad' : 'small muted'">{{ verdict }}</p>
      <div class="demo-controls">
        <label>Condition number κ <b>{{ kappa }}</b><input type="range" min="1" max="50" step="1" v-model.number="kappa"></label>
        <label>Step size, as η × λ_max <b>{{ c.toFixed(2) }}</b><input type="range" min="0.05" max="3.9" step="0.05" v-model.number="c"></label>
        <label>Momentum β <b>{{ beta.toFixed(2) }}</b><input type="range" min="0" max="0.95" step="0.01" v-model.number="beta"></label>
      </div>
      <div class="btn-row"><button class="btn sm" @click="bestPlain">Best fixed step, no momentum</button><button class="btn sm" @click="bestMomentum">Best step with momentum</button></div>
      <p class="small muted mt">The loss is f(x, y) = ½ (x² + κ·y²): curvature 1 along x and κ along y, so its condition number is exactly κ. Blue ellipses are contour lines; the amber path is gradient descent from the hollow dot towards the minimum (green). The steep direction limits the step size (η·λ_max must stay below 2) while progress along the flat direction is proportional to η·λ_min, so a large κ means zigzagging across the valley and crawling along it. With the best fixed step the error shrinks by (κ − 1)/(κ + 1) per step; well-tuned momentum improves this to about (√κ − 1)/(√κ + 1). Compare the step counts of the two buttons at κ = 50.</p>
    </div>`
  };

  // 23 ---------- Bayesian updating: the Beta–Binomial model ----------
  // Beta(a, b) density on the grid x = (i + 0.5) / N, normalized numerically (no gamma functions needed).
  function betaPdf(a, b, N) {
    const lp = []; for (let i = 0; i < N; i++) { const x = (i + 0.5) / N; lp.push((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)); }
    const m = Math.max(...lp), p = lp.map(v => Math.exp(v - m)), s = p.reduce((u, v) => u + v, 0) / N;
    return p.map(v => v / s);
  }
  const BETA_PRIORS = { 'Flat: Beta(1, 1)': [1, 1], 'Weak: Beta(2, 8)': [2, 8], 'Strong: Beta(20, 80)': [20, 80] };
  C['demo-beta'] = {
    mixins: [redrawOnChange],
    data() { return { a: 2, b: 8, n: 10, k: 4 }; },
    watch: { n(v) { if (this.k > v) this.k = v; } },
    computed: {
      priors() { return BETA_PRIORS; },
      post() {
        const N = 1000, pa = this.a + this.k, pb = this.b + this.n - this.k, pdf = betaPdf(pa, pb, N);
        let cum = 0, lo = null, hi = null;
        for (let i = 0; i < N; i++) { cum += pdf[i] / N; if (lo === null && cum >= 0.025) lo = (i + 0.5) / N; if (hi === null && cum >= 0.975) hi = (i + 0.5) / N; }
        return { pa, pb, pdf, lo, hi, mean: pa / (pa + pb) };
      }
    },
    methods: {
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const N = 1000, prior = betaPdf(this.a, this.b, N), lik = betaPdf(this.k + 1, this.n - this.k + 1, N), post = this.post.pdf;
        let peak = 0; for (const arr of [prior, lik, post]) for (const v of arr) if (v > peak) peak = v;
        const top = Math.min(30, peak * 1.12), xs = i => (i + 0.5) / N;
        const T = frame(ctx, W, H, [0, 1], [0, top], P, { axes: false, xt: 10, yt: 4 });
        ctx.fillStyle = P.accent; ctx.globalAlpha = 0.18; ctx.beginPath(); ctx.moveTo(T.X(this.post.lo), T.Y(0));
        for (let i = 0; i < N; i++) { const x = xs(i); if (x >= this.post.lo && x <= this.post.hi) ctx.lineTo(T.X(x), T.Y(Math.min(top, post[i]))); }
        ctx.lineTo(T.X(this.post.hi), T.Y(0)); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
        const line = (arr, color, width, dash) => { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath(); arr.forEach((v, i) => { const px = T.X(xs(i)), py = T.Y(Math.min(top, v)); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }); ctx.stroke(); ctx.setLineDash([]); };
        line(prior, P.muted, 1.8, [6, 4]);
        if (this.n > 0) line(lik, P.amber, 2, [2, 3]);
        line(post, P.accent, 2.8);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="pills"><button v-for="(ab, name) in priors" :key="name" :class="{active: a === ab[0] && b === ab[1]}" @click="a = ab[0]; b = ab[1]">{{ name }}</button></div>
      <div class="small muted">posterior Beta({{ post.pa }}, {{ post.pb }}) · mean <b>{{ post.mean.toFixed(3) }}</b> · 95% credible interval <b>{{ post.lo.toFixed(3) }} – {{ post.hi.toFixed(3) }}</b> · the data alone (k / n) <b>{{ n ? (k / n).toFixed(3) : '—' }}</b></div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <div class="demo-controls">
        <label>Prior a (like earlier successes) <b>{{ a }}</b><input type="range" min="0.5" max="40" step="0.5" v-model.number="a"></label>
        <label>Prior b (like earlier failures) <b>{{ b }}</b><input type="range" min="0.5" max="100" step="0.5" v-model.number="b"></label>
        <label>Trials n <b>{{ n }}</b><input type="range" min="0" max="200" step="1" v-model.number="n"></label>
        <label>Successes k <b>{{ k }}</b><input type="range" min="0" :max="n" step="1" v-model.number="k"></label>
      </div>
      <p class="small muted mt">Grey dashed: the prior. Amber dotted: what the data alone say (the likelihood θᵏ(1 − θ)ⁿ⁻ᵏ, rescaled into a density). Blue: the posterior Beta(a + k, b + n − k), with its 95% credible interval shaded. With few trials the posterior stays close to the prior; as n grows it follows the data and narrows roughly like 1/√n. A strong prior (large a + b) behaves like many earlier observations, so it takes a lot of data to move it. Try n = 10, k = 4 with each prior, then raise n to 200 with k = 80.</p>
    </div>`
  };

  // 24 ---------- Gaussian process regression ----------
  const GP_START = [[1, 0.94], [2.2, 1.1], [3.1, 0.41], [5, -1.17], [6.3, -0.69], [8.4, 1.15]];
  const GP_XS = Array.from({ length: 100 }, (_, i) => i / 99 * 10);
  const GP_Z = (() => { const rng = MLA.util.seededRng(3); return [0, 1, 2].map(() => GP_XS.map(() => gauss(rng))); })();
  function cholesky(M) {
    const n = M.length, L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = M[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
    }
    return L;
  }
  function solveLower(L, b) { const n = b.length, x = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * x[k]; x[i] = s / L[i][i]; } return x; }
  function solveUpper(L, b) { const n = b.length, x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; }
  // GP posterior with an RBF kernel: mean, standard deviation, log marginal likelihood, and (optionally) the full covariance.
  function gpFit(pts, ell, sf, sn, xs, wantCov) {
    const kern = (u, v) => sf * sf * Math.exp(-(u - v) * (u - v) / (2 * ell * ell));
    const n = pts.length, m = xs.length, V = [];
    let L = null, alpha = null, lml = null;
    if (n) {
      const K = pts.map(p => pts.map(q => kern(p[0], q[0]))); for (let i = 0; i < n; i++) K[i][i] += sn * sn;
      L = cholesky(K); const y = pts.map(p => p[1]); alpha = solveUpper(L, solveLower(L, y));
      let fit = 0, halfLogDet = 0; for (let i = 0; i < n; i++) { fit += y[i] * alpha[i]; halfLogDet += Math.log(L[i][i]); }
      lml = -0.5 * fit - halfLogDet - n / 2 * Math.log(2 * Math.PI);
    }
    const mean = new Float64Array(m), sd = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      if (!n) { sd[j] = sf; continue; }
      const ks = pts.map(p => kern(p[0], xs[j])); let mu = 0; for (let i = 0; i < n; i++) mu += ks[i] * alpha[i];
      const v = solveLower(L, ks); let s = 0; for (let i = 0; i < n; i++) s += v[i] * v[i];
      mean[j] = mu; sd[j] = Math.sqrt(Math.max(0, sf * sf - s)); V.push(v);
    }
    let cov = null;
    if (wantCov) {
      cov = [];
      for (let i = 0; i < m; i++) {
        const row = new Float64Array(m);
        for (let j = 0; j < m; j++) { let s = kern(xs[i], xs[j]); if (n) for (let k = 0; k < n; k++) s -= V[i][k] * V[j][k]; row[j] = s + (i === j ? 1e-5 * sf * sf : 0); }
        cov.push(row);
      }
    }
    return { mean, sd, lml, cov };
  }
  C['demo-gp'] = {
    mixins: [redrawOnChange],
    data() { return { pts: GP_START.map(p => p.slice()), ell: 1, sf: 1, sn: 0.1, samples: false }; },
    computed: { fit() { return gpFit(this.pts, this.ell, this.sf, this.sn, GP_XS, this.samples); } },
    methods: {
      add(e) {
        if (!this._T || this.pts.length >= 40) return;
        const [px, py] = canvasPoint(this.$refs.cv, e, 560, 300); const [x, y] = this._T.inv(px, py);
        if (x < 0 || x > 10 || y < -3 || y > 3) return;
        this.pts.push([+x.toFixed(2), +y.toFixed(2)]);
      },
      clear() { this.pts = []; },
      reset() { this.pts = GP_START.map(p => p.slice()); this.ell = 1; this.sf = 1; this.sn = 0.1; },
      optimize() {
        if (!this.pts.length) return;
        let best = null;
        for (let i = 0; i < 40; i++) {
          const ell = 0.1 * Math.pow(30, i / 39);
          for (const sf of [0.5, 0.75, 1, 1.5, 2]) for (const sn of [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]) {
            const r = gpFit(this.pts, ell, sf, sn, [], false); if (!best || r.lml > best.lml) best = { lml: r.lml, ell, sf, sn };
          }
        }
        this.ell = +best.ell.toFixed(2); this.sf = best.sf; this.sn = best.sn;
      },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const T = frame(ctx, W, H, [0, 10], [-3, 3], P, { xt: 10, yt: 6 }); this._T = T;
        const { mean, sd, cov } = this.fit, xs = GP_XS, cl = v => Math.max(-3.3, Math.min(3.3, v));
        ctx.fillStyle = P.accent; ctx.globalAlpha = 0.16; ctx.beginPath();
        xs.forEach((x, i) => { const py = T.Y(cl(mean[i] + 2 * sd[i])); i ? ctx.lineTo(T.X(x), py) : ctx.moveTo(T.X(x), py); });
        for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(T.X(xs[i]), T.Y(cl(mean[i] - 2 * sd[i])));
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
        if (cov) {
          const L = cholesky(cov), cols = [P.green, P.purple, P.cyan];
          GP_Z.forEach((z, s) => {
            ctx.strokeStyle = cols[s]; ctx.lineWidth = 1.2; ctx.beginPath();
            xs.forEach((x, i) => { let v = mean[i]; for (let k = 0; k <= i; k++) v += L[i][k] * z[k]; const py = T.Y(cl(v)); i ? ctx.lineTo(T.X(x), py) : ctx.moveTo(T.X(x), py); });
            ctx.stroke();
          });
        }
        ctx.strokeStyle = P.accent; ctx.lineWidth = 2.5; ctx.beginPath();
        xs.forEach((x, i) => { const py = T.Y(cl(mean[i])); i ? ctx.lineTo(T.X(x), py) : ctx.moveTo(T.X(x), py); });
        ctx.stroke();
        for (const [x, y] of this.pts) dot(ctx, T.X(x), T.Y(y), 4.5, P.amber, P.panel);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="small muted">{{ pts.length }} observations · length scale ℓ = <b>{{ ell.toFixed(2) }}</b> · signal σ_f = <b>{{ sf.toFixed(2) }}</b> · noise σ_n = <b>{{ sn.toFixed(2) }}</b> · log marginal likelihood <b>{{ fit.lml === null ? '—' : fit.lml.toFixed(2) }}</b></div>
      <canvas ref="cv" class="demo-canvas clickable" @click="add"></canvas>
      <div class="demo-controls">
        <label>Length scale ℓ <b>{{ ell.toFixed(2) }}</b><input type="range" min="0.1" max="3" step="0.05" v-model.number="ell"></label>
        <label>Signal σ_f <b>{{ sf.toFixed(2) }}</b><input type="range" min="0.2" max="2" step="0.05" v-model.number="sf"></label>
        <label>Noise σ_n <b>{{ sn.toFixed(2) }}</b><input type="range" min="0.01" max="1" step="0.01" v-model.number="sn"></label>
      </div>
      <label class="checkbox"><input type="checkbox" v-model="samples"> show three functions sampled from the posterior</label>
      <div class="btn-row"><button class="btn sm primary" @click="optimize">Fit ℓ, σ_f and σ_n (maximize the marginal likelihood)</button><button class="btn sm" @click="reset">Reset example</button><button class="btn sm ghost" @click="clear">Clear points</button></div>
      <p class="small muted mt">Click the plot to add observations (the prior mean is 0). Blue line: the posterior mean; shaded band: ±2 posterior standard deviations of the function. The band pinches near the data and widens back to the prior (±2σ_f) far from it. A short ℓ wiggles through every point, a long ℓ gives a smooth, almost straight fit, and a larger σ_n treats the points as noisy and stops passing through them. The fit button tries a grid of settings and keeps the one with the highest log marginal likelihood: the GP's built-in Occam's razor.</p>
    </div>`
  };

  // 25 ---------- PCA: the direction of greatest variance ----------
  function pcaData() {
    const rng = MLA.util.seededRng(13), a = Math.PI / 6, pts = [];
    for (let i = 0; i < 120; i++) { const u = gauss(rng) * 1.8, v = gauss(rng) * 0.55; pts.push([u * Math.cos(a) - v * Math.sin(a), u * Math.sin(a) + v * Math.cos(a)]); }
    const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length, my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    return pts.map(p => [p[0] - mx, p[1] - my]);
  }
  const PCA_PTS = pcaData();
  C['demo-pca'] = {
    mixins: [redrawOnChange],
    data() { return { theta: 100, showPC: false }; },
    computed: {
      cov() { const n = PCA_PTS.length; let xx = 0, yy = 0, xy = 0; for (const [x, y] of PCA_PTS) { xx += x * x; yy += y * y; xy += x * y; } return { xx: xx / n, yy: yy / n, xy: xy / n }; },
      total() { return this.cov.xx + this.cov.yy; },
      pc1() { const c = this.cov; return (0.5 * Math.atan2(2 * c.xy, c.xx - c.yy) * 180 / Math.PI + 180) % 180; },
      kept() { return this.varAt(this.theta); }
    },
    methods: {
      varAt(dg) { const t = dg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t), v = this.cov; return v.xx * c * c + 2 * v.xy * c * s + v.yy * s * s; },
      snap(k) { this.theta = +((this.pc1 + (k === 2 ? 90 : 0)) % 180).toFixed(1); },
      pick(e) { const [px, py] = canvasPoint(this.$refs.cv, e, 560, 300); if (Math.hypot(px - 280, py - 150) < 6) return; this.theta = +(deg(150 - py, px - 280) % 180).toFixed(1); },
      draw() {
        const P = palette(), cv = this.$refs.cv, cv2 = this.$refs.cv2;
        if (cv) {
          const W = 560, H = 300, ctx = setup(cv, W, H), T = eqFrame(ctx, W, H, 38, P);
          const t = this.theta * Math.PI / 180, u = [Math.cos(t), Math.sin(t)];
          ctx.strokeStyle = P.pink; ctx.lineWidth = 1; ctx.globalAlpha = 0.6;
          for (const [x, y] of PCA_PTS) { const s = x * u[0] + y * u[1]; ctx.beginPath(); ctx.moveTo(T.X(x), T.Y(y)); ctx.lineTo(T.X(s * u[0]), T.Y(s * u[1])); ctx.stroke(); }
          ctx.globalAlpha = 1;
          if (this.showPC) { const p = this.pc1 * Math.PI / 180; ctx.strokeStyle = P.green; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(T.X(-9 * Math.cos(p)), T.Y(-9 * Math.sin(p))); ctx.lineTo(T.X(9 * Math.cos(p)), T.Y(9 * Math.sin(p))); ctx.stroke(); ctx.setLineDash([]); }
          ctx.strokeStyle = P.amber; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(T.X(-9 * u[0]), T.Y(-9 * u[1])); ctx.lineTo(T.X(9 * u[0]), T.Y(9 * u[1])); ctx.stroke();
          for (const [x, y] of PCA_PTS) dot(ctx, T.X(x), T.Y(y), 2.8, P.accent);
          for (const [x, y] of PCA_PTS) { const s = x * u[0] + y * u[1]; dot(ctx, T.X(s * u[0]), T.Y(s * u[1]), 2, P.amber); }
        }
        if (cv2) {
          const W = 560, H = 150, ctx = setup(cv2, W, H), T = frame(ctx, W, H, [0, 180], [0, 100], P, { axes: false, xt: 6, yt: 2 });
          curve(ctx, d => 100 * this.varAt(d) / this.total, [0, 180], T, P.accent, 2);
          ctx.strokeStyle = P.amber; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(T.X(this.theta), T.Y(0)); ctx.lineTo(T.X(this.theta), T.Y(100)); ctx.stroke(); ctx.setLineDash([]);
          dot(ctx, T.X(this.theta), T.Y(100 * this.kept / this.total), 5, P.amber);
        }
      }
    },
    template: `
    <div class="card demo-card">
      <div class="small muted">direction θ = <b>{{ theta.toFixed(0) }}°</b> · variance kept <b>{{ kept.toFixed(2) }}</b> ({{ (100 * kept / total).toFixed(1) }}%) · mean squared reconstruction error <b>{{ (total - kept).toFixed(2) }}</b> ({{ (100 - 100 * kept / total).toFixed(1) }}%) · total variance {{ total.toFixed(2) }}</div>
      <canvas ref="cv" class="demo-canvas clickable" @click="pick"></canvas>
      <canvas ref="cv2" class="demo-canvas short"></canvas>
      <div class="demo-controls"><label>Direction of the line θ <b>{{ theta.toFixed(0) }}°</b><input type="range" min="0" max="179" step="1" v-model.number="theta"></label></div>
      <label class="checkbox"><input type="checkbox" v-model="showPC"> show PC1 (dashed green)</label>
      <div class="btn-row"><button class="btn sm primary" @click="snap(1)">Snap to PC1</button><button class="btn sm" @click="snap(2)">Snap to PC2</button></div>
      <p class="small muted mt">Each blue point is projected onto the amber line (its score is the position along the line); the pink segments are the reconstruction errors, what the projection throws away. Turn the line with the slider or by clicking the plot: the variance kept and the squared error always add up to the total variance, so the direction that keeps the most variance (PC1) is exactly the one with the smallest reconstruction error. The lower plot shows the percentage of variance kept for every direction θ (horizontal axis, in degrees): one peak at PC1 and the minimum 90° away, at PC2.</p>
    </div>`
  };

  // 26 ---------- Learning-rate schedules ----------
  C['demo-lr-schedule'] = {
    mixins: [redrawOnChange],
    data() { return { kind: 'cosine', warmup: 500, minPct: 10, at: 2000 }; },
    computed: {
      now() { return this.lr(this.at); },
      explain() {
        const n = this.now;
        if (n.phase === 'warmup') return 'warmup: lr_max · (step + 1) / warmup_steps = (' + (this.at + 1) + ' / ' + this.warmup + ') · lr_max';
        if (n.phase === 'constant') return 'constant after the warmup';
        if (n.phase === 'step') return 'step decay: divided by 10 at step 5,000 and again at step 7,500';
        return 'cosine decay, progress = ' + n.progress.toFixed(2) + ': lr_min + ½ (lr_max − lr_min)(1 + cos(π · progress))';
      }
    },
    methods: {
      lr(step) {
        const w = this.warmup, mn = this.minPct / 100;
        if (step < w) return { v: (step + 1) / w, phase: 'warmup' };
        if (this.kind === 'constant') return { v: 1, phase: 'constant' };
        if (this.kind === 'step') return { v: step < 5000 ? 1 : step < 7500 ? 0.1 : 0.01, phase: 'step' };
        const progress = (step - w) / Math.max(1, 10000 - w);
        return { v: mn + 0.5 * (1 - mn) * (1 + Math.cos(Math.PI * progress)), phase: 'cosine', progress };
      },
      draw() {
        const cv = this.$refs.cv; if (!cv) return; const W = 560, H = 300; const ctx = setup(cv, W, H); const P = palette();
        const T0 = frame(ctx, W, H, [0, 10], [0, 1.05], P, { axes: false, xt: 5, yt: 4 }), T = { X: s => T0.X(s / 1000), Y: T0.Y };
        if (this.warmup > 0) { ctx.fillStyle = P.amber; ctx.globalAlpha = 0.12; ctx.fillRect(T.X(0), T.Y(1.05), T.X(this.warmup) - T.X(0), T.Y(0) - T.Y(1.05)); ctx.globalAlpha = 1; }
        ctx.strokeStyle = P.accent; ctx.lineWidth = 2.5; ctx.beginPath();
        for (let s = 0; s < 10000; s += 10) { const y = T.Y(this.lr(s).v); s ? ctx.lineTo(T.X(s), y) : ctx.moveTo(T.X(s), y); }
        ctx.stroke();
        ctx.strokeStyle = P.pink; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(T.X(this.at), T.Y(0)); ctx.lineTo(T.X(this.at), T.Y(1.05)); ctx.stroke(); ctx.setLineDash([]);
        dot(ctx, T.X(this.at), T.Y(this.now.v), 5.5, P.pink);
      }
    },
    template: `
    <div class="card demo-card">
      <div class="pills"><button :class="{active: kind==='constant'}" @click="kind='constant'">Constant</button><button :class="{active: kind==='step'}" @click="kind='step'">Step decay</button><button :class="{active: kind==='cosine'}" @click="kind='cosine'">Cosine decay</button></div>
      <div class="small muted">step <b>{{ at }}</b> of 10,000: lr = <b>{{ now.v.toFixed(3) }}</b> × lr_max · {{ explain }}</div>
      <canvas ref="cv" class="demo-canvas"></canvas>
      <div class="demo-controls">
        <label>Warmup steps <b>{{ warmup }}</b><input type="range" min="0" max="3000" step="50" v-model.number="warmup"></label>
        <label>lr_min for cosine decay <b>{{ minPct }}% of lr_max</b><input type="range" min="0" max="50" step="1" v-model.number="minPct"></label>
        <label>Inspect step <b>{{ at }}</b><input type="range" min="0" max="9999" step="1" v-model.number="at"></label>
      </div>
      <p class="small muted mt">The vertical axis is the learning rate as a fraction of its peak, lr_max, over a 10,000-step run (horizontal axis: thousands of steps); the shaded strip is the warmup. Warmup ramps the rate up linearly, protecting the start of training while Adam's running statistics are still unreliable. Step decay divides the rate by 10 at fixed points; cosine decay glides smoothly from lr_max down to lr_min. Warmup followed by cosine decay is the standard recipe for transformers and LLMs. Set the warmup to 0 to switch it off.</p>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
