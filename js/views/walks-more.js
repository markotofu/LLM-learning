/* ML Atlas – more step-through walkthroughs (drawn by js/views/walks.js): a small network learning XOR, a linear SVM,
 * PCA by power iteration, an RNN's memory, retrieval for RAG, and the KV cache. Every number on screen comes from
 * running the algorithm below. The SVM and PCA walkthroughs can also run on the learner's own data (def.data). */
(function (global) {
  const MLA = global.MLA, W = MLA.walk;
  const f = v => MLA.walkKit.num(v);
  const vec = v => '(' + v.map(f).join(', ') + ')';
  const list = a => '[' + a.map(f).join(', ') + ']';
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const deg = v => Math.atan2(v[1], v[0]) * 180 / Math.PI;

  // A straight line a·x + b·y = c, drawn across a plot box [x0, x1] × [y0, y1] (clipped by the plot).
  function lineThrough(a, b, c, xd, yd, sx, sy) {
    if (Math.abs(b) > Math.abs(a)) return 'M' + sx(xd[0]) + ' ' + sy((c - a * xd[0]) / b) + ' L' + sx(xd[1]) + ' ' + sy((c - a * xd[1]) / b);
    if (Math.abs(a) > 1e-12) return 'M' + sx((c - b * yd[0]) / a) + ' ' + sy(yd[0]) + ' L' + sx((c - b * yd[1]) / a) + ' ' + sy(yd[1]);
    return null;
  }

  // ---------------------------------------------------------------- a small network learns XOR
  (function () {
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]], Y = [0, 1, 1, 0], FOCUS = 2;        // the example followed in detail: (1, 0)
    const sig = z => 1 / (1 + Math.exp(-z));
    const normal = rnd => { const u = Math.max(1e-12, rnd()), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const fwd = (net, x) => {
      const h = net.b1.map((b, j) => Math.tanh(x[0] * net.W1[0][j] + x[1] * net.W1[1][j] + b));
      const z = h.reduce((s, v, j) => s + v * net.W2[j], net.b2);
      return { h, z, p: sig(z) };
    };
    const lossOf = net => -mean(X.map((x, i) => { const p = Math.min(1 - 1e-12, Math.max(1e-12, fwd(net, x).p)); return Y[i] ? Math.log(p) : Math.log(1 - p); }));
    const right = net => X.filter((x, i) => (fwd(net, x).p > 0.5 ? 1 : 0) === Y[i]).length;
    const clone = net => ({ W1: net.W1.map(r => r.slice()), b1: net.b1.slice(), W2: net.W2.slice(), b2: net.b2 });
    W('mlp-xor', {
      title: 'A small network learns XOR by gradient descent',
      params: [{ key: 'H', label: 'Hidden units', value: 4, options: [[1, '1'], [2, '2'], [4, '4']] },
        { key: 'seed', label: 'Random start', value: 1, options: [[1, 'A'], [2, 'B']] }],
      symbols: [['x', 'the input: two numbers (x₁, x₂), each 0 or 1'], ['y', 'the answer: 1 when exactly one input is 1 (XOR), else 0'],
        ['h', 'the hidden layer: one value per hidden unit, between −1 and 1'], ['tanh', 'the hyperbolic tangent, which squashes any number into (−1, 1)'],
        ['W₁', 'the weights from the inputs to the hidden units'], ['b₁', 'the hidden units\' biases'],
        ['W₂', 'the weights from the hidden units to the output'], ['b₂', 'the output\'s bias'],
        ['p', 'the predicted probability that the answer is 1'], ['σ', 'the sigmoid, which turns a score into a probability'],
        ['L', 'the loss: cross-entropy, averaged over the four examples'], ['δ', 'an error signal: how much the loss changes as a unit\'s input changes'],
        ['⊙', 'multiply element by element'], ['W', 'any weight of the network'], ['η', 'the learning rate (1 here)'],
        ['←', 'replace the value on the left with the value on the right']],
      code: [
        'W1, b1 = rng.normal(size=(2, H)), np.zeros(H)    # random start',
        'W2, b2 = rng.normal(size=H), 0.0',
        'for epoch in range(1, 2001):',
        '    h = np.tanh(X @ W1 + b1)                     # hidden layer',
        '    p = 1 / (1 + np.exp(-(h @ W2 + b2)))         # output probability',
        '    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))',
        '    d_out = (p - y) / 4                          # error at the output',
        '    d_hid = np.outer(d_out, W2) * (1 - h ** 2)   # sent back through W2 and tanh',
        '    W2 -= lr * h.T @ d_out;  b2 -= lr * d_out.sum()',
        '    W1 -= lr * X.T @ d_hid;  b1 -= lr * d_hid.sum(0)',
        '    if loss < 0.05: break'
      ],
      nodes: [
        { id: 'init', label: ['Random start'], why: 'If every weight started equal, every hidden unit would compute the same thing and receive the same updates forever. Random weights break the symmetry.' },
        { id: 'fwd', label: ['Forward: h = tanh(x·W₁ + b₁)', 'then p = σ(h·W₂ + b₂)'], why: 'The hidden units turn the two inputs into new features; the output unit combines those features into one probability.' },
        { id: 'loss', label: ['Loss: cross-entropy', 'of p against y'], why: 'Cross-entropy punishes confident wrong answers hardest, and its gradient at the output is simply p − y.' },
        { id: 'back', label: ['Backward: error signals', 'from the output to each unit'], why: 'The chain rule passes the output error back through W₂ and the slope of tanh, so every weight learns how much it contributed.' },
        { id: 'update', label: ['Step downhill', 'W ← W − η·∂L/∂W'], why: 'Every weight moves a little against its gradient, so the loss on these four examples goes down.' },
        { id: 'check', label: ['Loss below 0.05?'], kind: 'dec', why: 'Training stops once the network is confident on all four examples, or when the epoch budget runs out.' },
        { id: 'done', label: ['Done: XOR, learned', 'or stuck'], kind: 'end', why: 'Successful hidden units become features that make XOR easy for the output unit, like the hand-set OR and AND units above.' }
      ],
      edges: [
        { from: 'init', to: 'fwd', label: 'epoch 1', why: 'With weights in place, the network can make its first predictions.' },
        { from: 'fwd', to: 'loss', label: 'compare', why: 'A prediction only means something next to the right answer.' },
        { from: 'loss', to: 'back', label: 'chain rule', why: 'The loss decides which way each weight should move.' },
        { from: 'back', to: 'update', label: 'gradients', why: 'Each weight\'s gradient is its error signal times the input that reached it.' },
        { from: 'update', to: 'check', label: 'new loss', why: 'After every step, check whether the network is good enough.' },
        { from: 'check', to: 'fwd', label: 'no: again', why: 'Still unsure about some examples: run another epoch with the new weights.' },
        { from: 'check', to: 'done', label: 'yes: stop', why: 'Confident on every example, or out of epochs.' }
      ],
      run(p) {
        const H = +p.H, start = +p.seed === 2 ? 'B' : 'A', rnd = MLA.util.seededRng(+p.seed === 2 ? 2 : 1), LR = 1, BUDGET = 2000;
        const net = { W1: [0, 1].map(() => Array.from({ length: H }, () => normal(rnd))), b1: Array(H).fill(0), W2: Array.from({ length: H }, () => normal(rnd)), b2: 0 };
        const steps = [], hist = [];
        let L = lossOf(net), since = 2;
        steps.push({ node: 'init', lines: [1, 2], say: 'Start ' + start + ': ' + H + ' hidden unit' + (H > 1 ? 's' : '') + ' with random weights. The four examples are the XOR table: the answer is 1 when exactly one input is 1. Right now the network gets ' + right(net) + ' of 4 right.',
          vars: [['hidden units', H], ['L', f(L)]], state: { net: clone(net) }, ask: 'Could one straight line put both 1s on one side and both 0s on the other?' });
        for (let ep = 1; ep <= BUDGET; ep++) {
          const g = { W1: [0, 1].map(() => Array(H).fill(0)), b1: Array(H).fill(0), W2: Array(H).fill(0), b2: 0 };
          let seen = null;
          X.forEach((x, i) => {
            const o = fwd(net, x), dOut = (o.p - Y[i]) / 4, dHid = o.h.map((v, j) => dOut * net.W2[j] * (1 - v * v));
            g.b2 += dOut;
            o.h.forEach((v, j) => { g.W2[j] += dOut * v; g.b1[j] += dHid[j]; g.W1[0][j] += dHid[j] * x[0]; g.W1[1][j] += dHid[j] * x[1]; });
            if (i === FOCUS) seen = { o, dOut, dHid };
          });
          const before = clone(net), L0 = L;
          for (let j = 0; j < H; j++) { net.W2[j] -= LR * g.W2[j]; net.b1[j] -= LR * g.b1[j]; net.W1[0][j] -= LR * g.W1[0][j]; net.W1[1][j] -= LR * g.W1[1][j]; }
          net.b2 -= LR * g.b2;
          L = lossOf(net); hist.push(L0);
          if (ep === 1) {
            const o = seen.o;
            steps.push({ node: 'fwd', lines: [4, 5], say: 'Epoch 1, following the input (1, 0), whose answer is 1. The hidden units give h = ' + vec(o.h) + ', and the output is p = ' + f(o.p) + ': a ' + Math.round(o.p * 100) + '% chance of 1.',
              formula: 'h = tanh(x·W₁ + b₁) = ' + vec(o.h) + '\np = σ(h·W₂ + b₂) = σ(' + f(o.z) + ') = ' + f(o.p), vars: [['epoch', 1], ['L', f(L0)]], state: { net: before, focus: FOCUS } });
            steps.push({ node: 'loss', lines: [6], say: 'Over all four examples the cross-entropy loss is ' + f(L0) + '. Answering 50% everywhere would score ' + f(Math.log(2)) + '.',
              formula: 'L = −mean(y·log p + (1 − y)·log(1 − p)) = ' + f(L0), vars: [['epoch', 1], ['L', f(L0)]], state: { net: before } });
            steps.push({ node: 'back', lines: [7, 8], say: 'For (1, 0) the output error is p − y = ' + f(o.p - 1) + '. The chain rule passes it back through W₂ and the slope of tanh, so each hidden unit gets its share of the blame.',
              formula: 'δ_out = (p − y) / 4 = ' + f(seen.dOut) + '\nδ_hid = δ_out·W₂ ⊙ (1 − h²) = ' + vec(seen.dHid), vars: [['epoch', 1], ['L', f(L0)]], state: { net: before, focus: FOCUS } });
            steps.push({ node: 'update', lines: [9, 10], say: 'Every weight takes a step against its gradient, summed over the four examples. The loss goes from ' + f(L0) + ' to ' + f(L) + '.',
              formula: 'W ← W − η·∂L/∂W,  η = 1\nL: ' + f(L0) + ' → ' + f(L), vars: [['epoch', 1], ['L', f(L)]], state: { net: clone(net), hist: hist.concat([L]) } });
          }
          const done = L < 0.05 || ep === BUDGET;
          if (ep === 1 || done || [10, 50, 100, 200, 500, 1000].includes(ep)) {
            steps.push({ node: 'check', lines: [11], say: done && L < 0.05 ? 'The loss is ' + f(L) + ', below 0.05: training stops after ' + ep + ' epochs.'
              : ep === 1 ? 'The loss is ' + f(L) + ', not below 0.05 yet, so another epoch runs.'
                : (ep === BUDGET ? 'The budget of ' + BUDGET + ' epochs is used up. ' : '') + 'Epochs ' + since + ' to ' + ep + ' repeated the loop. The loss is ' + f(L) + ', and ' + right(net) + ' of the 4 examples are right.',
            vars: [['epoch', ep], ['L', f(L)], ['right', right(net) + ' of 4']], state: { net: clone(net), hist: hist.concat([L]) } });
            since = ep + 1;
          }
          if (done) {
            const preds = X.map(x => f(fwd(net, x).p)).join(', ');
            steps.push({ node: 'done', lines: [], say: L < 0.05
              ? 'Learned after ' + ep + ' epochs: for the inputs (0, 0), (0, 1), (1, 0) and (1, 1) the network answers ' + preds + '. The dashed lines show where each hidden unit changes sign: together they cut the square into regions that no single line could make.'
              : H === 1
                ? 'After ' + BUDGET + ' epochs the loss is stuck at ' + f(L) + ', with ' + right(net) + ' of 4 right. A single hidden unit gives the output only one dividing line to work with, and one line cannot separate XOR.'
                : 'After ' + BUDGET + ' epochs the loss is stuck at ' + f(L) + ', with ' + right(net) + ' of 4 right. ' + H + ' hidden units are enough in principle (the hand-set solution above uses two), but from this random start gradient descent settles in a poor local minimum. Try the other start, or more hidden units.',
            vars: [['epochs', ep], ['L', f(L)], ['right', right(net) + ' of 4']], state: { net: clone(net), hist: hist.concat([L]) } });
            break;
          }
        }
        return steps;
      },
      draw(s, kit) {
        const xd = [-0.5, 1.5], fr = kit.frame(xd, xd, ['x₁', 'x₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy, net = s.net;
        if (!net) return out;
        const N = 12, w = 2 / N;
        for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {           // the network's answer over the whole square
          const p = fwd(net, [xd[0] + (a + 0.5) * w, xd[0] + (b + 0.5) * w]).p;
          out.push({ k: 'g' + a + '-' + b, t: 'r', x: sx(xd[0] + a * w), y: sy(xd[0] + (b + 1) * w), w: sx(w) - sx(0), h: sy(0) - sy(w), cls: p > 0.5 ? 'wp-area2' : 'wp-area' });
        }
        net.b1.forEach((b, j) => { const d = lineThrough(net.W1[0][j], net.W1[1][j], -b, [-1, 2], [-1, 2], sx, sy); if (d) out.push({ k: 'u' + j, t: 'p', d, cls: 'wp-dash' }); });
        X.forEach((x, i) => out.push({ k: 'p' + i, t: 'c', x: sx(x[0]), y: sy(x[1]), r: 6, cls: Y[i] ? 'wp-c1' : 'wp-c0' }));
        if (s.focus !== undefined) out.push({ k: 'hl', t: 'c', x: sx(X[s.focus][0]), y: sy(X[s.focus][1]), r: 10, cls: 'wp-hl' });
        if (s.hist && s.hist.length > 1) {                                // the loss so far, as a small inset
          const h = s.hist, mx = Math.max(...h), x0 = 124, y0 = 22, ww = 80, hh = 26;          // between the two upper points
          out.push({ k: 'ib', t: 'r', x: x0 - 4, y: y0 - 12, w: ww + 8, h: hh + 16, cls: 'wp-cell' });
          out.push({ k: 'it', t: 't', x: x0, y: y0 - 2, s: 'loss so far', cls: 'wp-tick' });
          out.push({ k: 'ic', t: 'p', d: h.map((v, i) => (i ? 'L' : 'M') + (x0 + i / (h.length - 1) * ww).toFixed(1) + ' ' + (y0 + hh - Math.min(1, v / mx) * (hh - 6)).toFixed(1)).join(' '), cls: 'wp-line' });
        }
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- a linear SVM by subgradient descent (the lesson's code)
  (function () {
    const PTS = [[-2.2, -1.0, 0], [-1.6, 0.4, 0], [-0.9, -1.6, 0], [-2.6, 0.9, 0], [-0.3, -0.2, 0], [-1.3, -0.4, 0],
      [1.8, 1.2, 1], [0.9, 2.2, 1], [2.4, 0.3, 1], [0.6, 0.8, 1], [1.5, 2.6, 1], [2.9, 1.7, 1]];          // class 0: y = −1, class 1: y = +1
    const LAMS = [0.001, 0.01, 0.1], LR = 0.1, EPOCHS = 200;
    function train(D, lam, each) {
      const n = D.length;
      let w = [0, 0], b = 0;
      for (let ep = 1; ep <= EPOCHS; ep++) {
        const m = D.map(q => q[2] * (w[0] * q[0] + w[1] * q[1] + b)), viol = D.filter((q, i) => m[i] < 1);
        const gw = [2 * lam * w[0] - viol.reduce((s, q) => s + q[2] * q[0], 0) / n, 2 * lam * w[1] - viol.reduce((s, q) => s + q[2] * q[1], 0) / n];
        const gb = -viol.reduce((s, q) => s + q[2], 0) / n, w0 = w, b0 = b;
        w = [w[0] - LR * gw[0], w[1] - LR * gw[1]]; b = b - LR * gb;
        if (each) each({ ep, m, viol, gw, gb, w0, b0, w, b });
      }
      return { w, b, m: D.map(q => q[2] * (w[0] * q[0] + w[1] * q[1] + b)) };
    }
    const width = w => 2 / Math.hypot(w[0], w[1]);
    const prep = data => (data ? data.P : PTS).map(q => [q[0], q[1], q[2] ? 1 : -1]);
    W('svm-hinge', {
      title: 'A linear SVM finds its street, epoch by epoch',
      data: { kind: 'class2' },
      params: [{ key: 'lam', label: 'λ (regularization)', value: 0.01, options: LAMS.map(l => [l, String(l)]) }],
      symbols: [['x', 'a point, (x₁, x₂)'], ['y', 'its class: −1 (blue) or +1 (orange)'], ['w', 'the weights: they set the line\'s direction'], ['b', 'the bias: it shifts the line'],
        ['m', 'a point\'s margin, y·(w·x + b): at least 1 means safely outside the street, below 0 means on the wrong side'],
        ['λ', 'the regularization strength (set below): a larger λ prefers a wider street; it plays the part of 1/C'],
        ['n', 'the number of points'], ['Σ', 'the sum over the points with a margin below 1'],
        ['∂/∂w', 'the gradient with respect to w'], ['∂/∂b', 'the gradient with respect to b'], ['η', 'the learning rate (0.1, as in the lesson\'s code)'],
        ['←', 'replace the value on the left with the value on the right']],
      code: [
        'w, b = np.zeros(2), 0.0',
        'for epoch in range(200):',
        '    margins = y * (X @ w + b)',
        '    viol = margins < 1                   # inside the street or on the wrong side',
        '    grad_w = 2 * lam * w - (X[viol].T @ y[viol]) / len(y)',
        '    grad_b = -np.sum(y[viol]) / len(y)',
        '    w -= lr * grad_w',
        '    b -= lr * grad_b'
      ],
      nodes: [
        { id: 'init', label: ['Start: w = (0, 0), b = 0'], why: 'With zero weights every point scores 0, so every point starts inside the street and pulls on the line.' },
        { id: 'margin', label: ['Margins: m = y·(w·x + b)'], why: 'The margin says how safely a point sits on its own side. The hinge loss only cares about points with a margin below 1.' },
        { id: 'grad', label: ['Gradient: λ·w minus the', 'pull of points with m < 1'], why: 'The λ part shrinks w, which widens the street; each violating point pulls the line away from itself.' },
        { id: 'update', label: ['Step: w ← w − η·∂/∂w'], why: 'A small step against the gradient balances the two forces a little better.' },
        { id: 'check', label: ['More epochs?'], kind: 'dec', why: 'The lesson\'s code runs a fixed 200 epochs.' },
        { id: 'done', label: ['Done: the street and', 'its support vectors'], kind: 'end', why: 'Only the points on or inside the street fixed the line; every other point could move without changing it.' }
      ],
      edges: [
        { from: 'init', to: 'margin', label: 'epoch 1', why: 'Measure where every point stands relative to the current street.' },
        { from: 'margin', to: 'grad', label: 'find violators', why: 'Points with a margin of at least 1 contribute nothing to the gradient.' },
        { from: 'grad', to: 'update', label: 'scale by η', why: 'The gradient gives the direction; the learning rate sets the step.' },
        { from: 'update', to: 'check', label: 'next', why: 'One epoch is one pass over all the points.' },
        { from: 'check', to: 'margin', label: 'yes: again', why: 'Epochs remain: measure the margins against the moved line.' },
        { from: 'check', to: 'done', label: 'no: stop', why: 'All 200 epochs have run.' }
      ],
      run(p, data) {
        const D = prep(data), n = D.length, lam = +p.lam, steps = [];
        const st = (w, b, extra) => Object.assign({ w: w.slice(), b, m: D.map(q => q[2] * (w[0] * q[0] + w[1] * q[1] + b)) }, extra || {});
        const counts = [D.filter(q => q[2] < 0).length, D.filter(q => q[2] > 0).length];
        steps.push({ node: 'init', lines: [1], say: (data ? data.note + ' ' : n + ' points, ' + counts[0] + ' of each class. ') + 'The model starts with w = (0, 0) and b = 0: every point scores 0, so every point is inside the street.',
          vars: [['λ', lam], ['η', LR]], state: st([0, 0], 0) });
        let since = 3;
        const fin = train(D, lam, e => {
          const inside = e.m.filter(v => v < 1).length;
          if (e.ep <= 2) {
            steps.push({ node: 'margin', lines: [3, 4], say: 'Epoch ' + e.ep + ': ' + e.viol.length + ' of ' + n + ' points have a margin below 1 (ringed): inside the street or on the wrong side.', formula: 'm = y·(w·x + b) = ' + list(e.m.slice(0, 12)) + (n > 12 ? ' …' : ''), vars: [['epoch', e.ep], ['inside', e.viol.length]], state: st(e.w0, e.b0, { ring: true }) });
            steps.push({ node: 'grad', lines: [5, 6], say: 'Only the ringed points pull on the line. ' + (e.ep === 1 ? 'With w still zero, the λ part is zero too: the whole gradient is their pull.' : 'The λ part now pushes back, shrinking w a little.'),
              formula: '∂/∂w = 2λ·w − Σ y·x / n = ' + vec(e.gw) + '\n∂/∂b = −Σ y / n = ' + f(e.gb), vars: [['epoch', e.ep], ['inside', e.viol.length]], state: st(e.w0, e.b0, { ring: true }) });
            steps.push({ node: 'update', lines: [7, 8], say: 'One step against the gradient: w goes from ' + vec(e.w0) + ' to ' + vec(e.w) + ', and b from ' + f(e.b0) + ' to ' + f(e.b) + '.',
              formula: 'w ← w − η·∂/∂w = ' + vec(e.w) + '\nb ← b − η·∂/∂b = ' + f(e.b), vars: [['epoch', e.ep], ['w', vec(e.w)], ['b', f(e.b)]], state: st(e.w, e.b, { ghost: [e.w0, e.b0] }) });
          }
          if (e.ep <= 2 || [5, 20, 50, 100, EPOCHS].includes(e.ep)) {
            const m2 = D.map(q => q[2] * (e.w[0] * q[0] + e.w[1] * q[1] + e.b)), in2 = m2.filter(v => v < 1).length;
            steps.push({ node: 'check', lines: [2], say: (e.ep <= 2 ? 'Epoch ' + e.ep + ' is done' : 'Epochs ' + since + ' to ' + e.ep + ' repeated the loop') + ': the street is ' + f(width(e.w)) + ' wide, and ' + in2 + ' point' + (in2 === 1 ? ' sits' : 's sit') + ' inside it or on the wrong side.' + (e.ep < EPOCHS ? ' More epochs remain.' : ''),
              vars: [['epoch', e.ep], ['inside', in2], ['width', f(width(e.w))]], state: st(e.w, e.b, { ring: true }) });
            if (e.ep > 2) since = e.ep + 1;
          }
          return inside;
        });
        const sv = fin.m.filter(v => v <= 1.0001).length, acc = fin.m.filter(v => v > 0).length;
        const others = LAMS.map(l => { const r = train(D, l); return 'λ = ' + l + ': ' + f(width(r.w)) + ' wide, ' + r.m.filter(v => v <= 1.0001).length + ' inside'; });
        steps.push({ node: 'done', lines: [], say: 'After ' + EPOCHS + ' epochs: w = ' + vec(fin.w) + ', b = ' + f(fin.b) + '. The street is ' + f(width(fin.w)) + ' wide, with ' + (sv ? sv + ' point' + (sv === 1 ? '' : 's') : 'no point') + ' on or inside it. The points nearest the street, the support vectors, are the only ones that push on it: any other point could move a little without changing the line. Training accuracy: ' + acc + ' of ' + n + '. For comparison, ' + others.join('; ') + '. A larger λ, like a smaller C, buys a wider street at the price of more points inside it.',
          vars: [['w', vec(fin.w)], ['b', f(fin.b)], ['width', f(width(fin.w))]], state: st(fin.w, fin.b, { ring: true }) });
        return steps;
      },
      draw(s, kit, data) {
        const D = prep(data);
        const xd = data ? kit.span(D.map(q => q[0])) : [-3.2, 3.4], yd = data ? kit.span(D.map(q => q[1])) : [-2.2, 3.2];
        const fr = kit.frame(xd, yd, data ? data.cols.slice(0, 2) : ['x₁', 'x₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        const draw = (w, b, c, cls, k) => { const d = lineThrough(w[0], w[1], c - b, xd, yd, sx, sy); if (d) out.push({ k, t: 'p', d, cls }); };
        if (s.ghost && (s.ghost[0][0] || s.ghost[0][1])) draw(s.ghost[0], s.ghost[1], 0, 'wp-line2', 'ghost');
        if (s.w && (s.w[0] || s.w[1])) { draw(s.w, s.b, 1, 'wp-dash', 'up'); draw(s.w, s.b, -1, 'wp-dash', 'lo'); draw(s.w, s.b, 0, 'wp-line', 'mid'); }
        D.forEach((q, i) => {
          out.push({ k: 'p' + i, t: 'c', x: sx(q[0]), y: sy(q[1]), r: 4.5, cls: q[2] > 0 ? 'wp-c1' : 'wp-c0' });
          if (s.ring && s.m && s.m[i] < 1) out.push({ k: 'h' + i, t: 'c', x: sx(q[0]), y: sy(q[1]), r: 8, cls: 'wp-hl' });
        });
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- PCA: the first component by power iteration
  (function () {
    const PTS = [[-1.4, 0.5], [0.2, -1], [-0.3, 0.8], [0.8, -0.2], [0.1, 1.8], [1.1, 0.5], [0.8, 1.8], [2.3, -0.1], [1.8, 1.6], [2.7, 0.8], [2, 2.8], [3.4, 1.5]];
    const prep = data => {
      const P = data ? data.P.map(q => [q[0], q[1]]) : PTS, n = P.length, mu = [0, 1].map(j => mean(P.map(q => q[j])));
      const C = P.map(q => [q[0] - mu[0], q[1] - mu[1]]), S = [[0, 0], [0, 0]];
      C.forEach(q => { S[0][0] += q[0] * q[0]; S[0][1] += q[0] * q[1]; S[1][1] += q[1] * q[1]; });
      S[0][0] /= n - 1; S[0][1] /= n - 1; S[1][1] /= n - 1; S[1][0] = S[0][1];
      return { P, n, mu, C, S };
    };
    const times = (S, v) => [S[0][0] * v[0] + S[0][1] * v[1], S[1][0] * v[0] + S[1][1] * v[1]];
    W('pca-power', {
      title: 'Finding the first principal component by power iteration',
      data: { kind: 'points2' },
      symbols: [['x', 'a data point, (x₁, x₂)'], ['x_c', 'a centred point: x minus the mean of all points'], ['X_c', 'all the centred points, one per row'], ['n', 'the number of points'],
        ['Σ', 'the covariance matrix: the variances on its diagonal, the covariance off it'], ['ᵀ', 'transpose: rows become columns'],
        ['v', 'the current guess for the direction of greatest variance, a vector of length 1'], ['‖Σv‖', 'the length of the vector Σv'],
        ['θ', 'the angle of v, measured from the x₁ axis'], ['λ', 'an eigenvalue: the variance of the data along its direction'],
        ['←', 'replace the value on the left with the value on the right']],
      code: [
        'Xc = X - X.mean(axis=0)                  # centre the data',
        'S = Xc.T @ Xc / (len(X) - 1)              # covariance matrix (2 x 2)',
        'v = np.array([0.0, 1.0])                  # any unit vector to start',
        'while True:',
        '    v_new = S @ v / np.linalg.norm(S @ v)  # stretch toward PC1, rescale to length 1',
        '    turned = np.degrees(np.arccos(min(1, abs(v_new @ v))))',
        '    v = v_new',
        '    if turned < 0.5: break                # it no longer turns: converged',
        'lam = v @ S @ v                           # the variance along PC1',
        'scores = Xc @ v                           # each point\'s position along PC1'
      ],
      nodes: [
        { id: 'data', label: ['The data'], why: 'PCA looks for the direction along which the points vary most.' },
        { id: 'center', label: ['Centre: subtract the mean'], why: 'Variance is spread around the mean, so PCA works on the data shifted to the origin.' },
        { id: 'cov', label: ['Covariance matrix Σ'], why: 'Σ summarizes the spread of the cloud in every direction at once. Its eigenvector with the largest eigenvalue is PC1.' },
        { id: 'start', label: ['Start: any unit vector v'], why: 'Power iteration can start almost anywhere: only a vector exactly perpendicular to PC1 would never get there.' },
        { id: 'mult', label: ['Power step: v ← Σv / ‖Σv‖'], why: 'Multiplying by Σ stretches each direction by its variance, so the direction of greatest variance grows fastest and v swings toward it.' },
        { id: 'turn', label: ['Did v still turn?'], kind: 'dec', why: 'Once another multiplication no longer changes the direction, v is an eigenvector of Σ.' },
        { id: 'var', label: ['Variance along v: λ = vᵀΣv'], why: 'The eigenvalue measures how much of the total spread lies along PC1.' },
        { id: 'project', label: ['Project: score = x_c · v'], kind: 'end', why: 'Each point is replaced by one number, its position along PC1: two features become one, keeping as much variance as any single direction can.' }
      ],
      edges: [
        { from: 'data', to: 'center', label: 'shift', why: 'Only the shape of the cloud matters, not where it sits.' },
        { from: 'center', to: 'cov', label: 'measure spread', why: 'The covariance matrix turns the centred cloud into four numbers.' },
        { from: 'cov', to: 'start', label: 'guess', why: 'Power iteration needs a first direction to improve.' },
        { from: 'start', to: 'mult', label: 'multiply', why: 'Each multiplication pulls v toward the stretchiest direction.' },
        { from: 'mult', to: 'turn', label: 'compare', why: 'Measure the angle between the old and the new v.' },
        { from: 'turn', to: 'mult', label: 'yes: again', why: 'v is still turning, so it has not reached PC1 yet: multiply again.' },
        { from: 'turn', to: 'var', label: 'no: settled', why: 'v has stopped turning: it is PC1.' },
        { from: 'var', to: 'project', label: 'project', why: 'With the direction known, every point gets its coordinate along it.' }
      ],
      run(p, data) {
        const D = prep(data), S = D.S, steps = [];
        steps.push({ node: 'data', lines: [], say: (data ? data.note : D.n + ' points, two features each.') + ' Their mean is ' + vec(D.mu) + '.', vars: [['n', D.n], ['mean', vec(D.mu)]], state: { stage: 'raw' } });
        steps.push({ node: 'center', lines: [1], say: data ? 'The rescaling already gave every column a mean of 0, so the cloud already sits on the origin.' : 'Subtracting the mean moves the cloud onto the origin. PCA only cares about the spread, not the position.',
          formula: 'x_c = x − mean = x − ' + vec(D.mu), state: { stage: 'centered' } });
        const r = S[0][1] / Math.sqrt(S[0][0] * S[1][1]);
        steps.push({ node: 'cov', lines: [2], say: 'The variances are ' + f(S[0][0]) + ' and ' + f(S[1][1]) + ', and the covariance is ' + f(S[0][1]) + ': ' + (S[0][1] >= 0 ? 'the two features tend to rise together.' : 'when one feature rises, the other tends to fall.'),
          formula: 'Σ = X_cᵀ X_c / (n − 1) = [[' + f(S[0][0]) + ', ' + f(S[0][1]) + '], [' + f(S[1][0]) + ', ' + f(S[1][1]) + ']]', state: { stage: 'centered' },
          ask: 'In which direction do you expect the cloud to be longest?' });
        let v = [0, 1];
        steps.push({ node: 'start', lines: [3], say: 'Start from v = (0, 1), straight up: θ = 90°.', vars: [['v', vec(v)], ['θ', '90°']], state: { stage: 'centered', v } });
        for (let k = 1; k <= 60; k++) {
          const u = times(S, v), len = Math.hypot(u[0], u[1]), nv = [u[0] / len, u[1] / len];
          const turned = Math.acos(Math.min(1, Math.abs(nv[0] * v[0] + nv[1] * v[1]))) * 180 / Math.PI, old = v;
          v = nv;
          if (k <= 3) steps.push({ node: 'mult', lines: [5], say: 'Iteration ' + k + ': Σv = ' + vec(u) + ', rescaled to length 1. The direction moved from θ = ' + f(deg(old)) + '° to ' + f(deg(v)) + '°.',
            formula: 'v ← Σv / ‖Σv‖ = ' + vec(u) + ' / ' + f(len) + ' = ' + vec(v), vars: [['v', vec(v)], ['θ', f(deg(v)) + '°']], state: { stage: 'centered', v, ghost: old } });
          else steps.push({ node: 'mult', edge: 'turn>mult', lines: [5, 6, 7], say: 'Iteration ' + k + ': θ = ' + f(deg(v)) + '°, a turn of ' + f(turned) + '°. Each turn is about the same fraction of the one before: the ratio of the two eigenvalues.',
            vars: [['v', vec(v)], ['θ', f(deg(v)) + '°']], state: { stage: 'centered', v, ghost: old } });
          if (turned < 0.5 || k === 60) {
            steps.push({ node: 'turn', lines: [8], say: (turned < 0.5 ? 'It turned only ' + f(turned) + '°: converged after ' + k + ' iteration' + (k > 1 ? 's' : '') : 'Still turning after 60 iterations, so the loop gives up') + ', at θ = ' + f(deg(v)) + '°.', vars: [['turned', f(turned) + '°']], state: { stage: 'centered', v } });
            break;
          }
          if (k <= 3) steps.push({ node: 'turn', lines: [6, 8], say: 'It turned ' + f(turned) + '°, more than 0.5°, so another iteration follows.', vars: [['turned', f(turned) + '°']], state: { stage: 'centered', v } });
        }
        const lam1 = v[0] * times(S, v)[0] + v[1] * times(S, v)[1], total = S[0][0] + S[1][1], lam2 = total - lam1;
        steps.push({ node: 'var', lines: [9], say: 'The variance along v is λ₁ = ' + f(lam1) + ' of a total of ' + f(total) + ': PC1 holds ' + Math.round(lam1 / total * 100) + '% of the variance, and the direction at right angles holds the other ' + f(lam2) + '.',
          formula: 'λ₁ = vᵀΣv = ' + f(lam1) + ',  λ₁ / (λ₁ + λ₂) = ' + f(lam1 / total), vars: [['λ₁', f(lam1)], ['share', Math.round(lam1 / total * 100) + '%']], state: { stage: 'centered', v } });
        const scores = D.C.map(q => q[0] * v[0] + q[1] * v[1]);
        const tr = total, det = S[0][0] * S[1][1] - S[0][1] * S[0][1], l1 = tr / 2 + Math.sqrt(Math.max(0, tr * tr / 4 - det));
        let e = Math.abs(S[0][1]) > 1e-12 ? [S[0][1], l1 - S[0][0]] : (S[0][0] >= S[1][1] ? [1, 0] : [0, 1]);
        const el = Math.hypot(e[0], e[1]); e = [e[0] / el, e[1] / el];
        if (e[0] * v[0] + e[1] * v[1] < 0) e = [-e[0], -e[1]];
        const off = Math.acos(Math.min(1, Math.abs(e[0] * v[0] + e[1] * v[1]))) * 180 / Math.PI, l2 = tr - l1;
        steps.push({ node: 'project', lines: [10], say: 'Each point\'s score is its position along PC1; the dashed lines drop the points onto it. Keeping only the scores turns two numbers per point into one and keeps ' + Math.round(lam1 / total * 100) + '% of the variance. ' +
          (off < 1 ? 'Solving the eigenvalue problem exactly, as NumPy\'s SVD does, gives the same direction, ' + vec(e) + ', up to its sign.'
            : 'The exact answer, which NumPy\'s SVD would give, is ' + vec(e) + ': power iteration stopped ' + f(off) + '° short of it, because the two eigenvalues are close (their ratio is ' + f(l2 / l1) + ') and each step turned v only a little.') +
          (data ? ' With two standardized features, PC1 always lies on a diagonal, because both variances are 1: the data only decides which diagonal, and how much variance it holds, which is (1 + |r|) / 2 = ' + f((1 + Math.abs(r)) / 2) + ' for the correlation r = ' + f(r) + '.' : ''),
        formula: 'scores = x_c · v = ' + list(scores.slice(0, 12)) + (scores.length > 12 ? ' …' : ''), vars: [['PC1', vec(v)]], state: { stage: 'centered', v, project: true } });
        return steps;
      },
      draw(s, kit, data) {
        const D = prep(data), all = D.P.concat(D.C);
        const xd = data ? kit.span(all.map(q => q[0])) : [-2.8, 3.8], yd = data ? kit.span(all.map(q => q[1])) : [-2.2, 3.0];
        const fr = kit.frame(xd, yd, data ? data.cols.slice(0, 2) : ['x₁', 'x₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        const pts = s.stage === 'raw' ? D.P : D.C, far = 2 * Math.max(xd[1] - xd[0], yd[1] - yd[0]);
        const axis = (v, cls, k) => out.push({ k, t: 'p', d: 'M' + sx(-far * v[0]) + ' ' + sy(-far * v[1]) + ' L' + sx(far * v[0]) + ' ' + sy(far * v[1]), cls });
        if (s.ghost) axis(s.ghost, 'wp-line2', 'ghost');
        if (s.v) axis(s.v, 'wp-line', 'v');
        pts.forEach((q, i) => {
          if (s.project && s.v) { const t = q[0] * s.v[0] + q[1] * s.v[1]; out.push({ k: 'r' + i, t: 'l', x1: sx(q[0]), y1: sy(q[1]), x2: sx(t * s.v[0]), y2: sy(t * s.v[1]), cls: 'wp-res' }); }
          out.push({ k: 'p' + i, t: 'c', x: sx(q[0]), y: sy(q[1]), r: 4.5, cls: 'wp-c0' });
        });
        if (s.stage === 'raw') out.push({ k: 'mu', t: 'c', x: sx(D.mu[0]), y: sy(D.mu[1]), r: 5, cls: 'wp-q' });
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- an RNN carries a signal in its memory
  (function () {
    const XS = [1, 0, 0, 0, 0, 0], WX = [1.2, 0.4], TH = 40 * Math.PI / 180, SCALES = [0.5, 1, 1.5];
    const weights = s => [[s * Math.cos(TH), s * Math.sin(TH)], [-s * Math.sin(TH), s * Math.cos(TH)]];
    const mixOf = (x, h, Wh) => [x * WX[0] + h[0] * Wh[0][0] + h[1] * Wh[1][0], x * WX[1] + h[0] * Wh[0][1] + h[1] * Wh[1][1]];
    function unroll(s) {                       // every memory state, and how much the last one still depends on x₁
      const Wh = weights(s), hs = [[0, 0]], zs = [];
      XS.forEach(x => { const z = mixOf(x, hs[hs.length - 1], Wh); zs.push(z); hs.push(z.map(Math.tanh)); });
      let g = [(1 - hs[1][0] ** 2) * WX[0], (1 - hs[1][1] ** 2) * WX[1]];          // ∂h₁/∂x₁
      for (let t = 2; t < hs.length; t++) { const m = [g[0] * Wh[0][0] + g[1] * Wh[1][0], g[0] * Wh[0][1] + g[1] * Wh[1][1]]; g = [(1 - hs[t][0] ** 2) * m[0], (1 - hs[t][1] ** 2) * m[1]]; }
      return { Wh, hs, zs, g };
    }
    W('rnn-memory', {
      title: 'An RNN carries one input through time',
      params: [{ key: 's', label: 'Recurrent weight scale', value: 1, options: SCALES.map(s => [s, String(s)]) }],
      symbols: [['xₜ', 'the input at step t, a single number here'], ['hₜ', 'the hidden state after step t: the memory, two numbers between −1 and 1'],
        ['hₜ₋₁', 'the memory before this step; h₀ is all zeros'], ['Wₓ', 'the input-to-hidden weights'], ['Wₕ', 'the hidden-to-hidden (recurrent) weights, here a rotation scaled by the setting below'],
        ['b', 'the bias, zero here'], ['z', 'the mixed signal before squashing'], ['tanh', 'the hyperbolic tangent, which squashes each number into (−1, 1)'],
        ['∂h₆/∂x₁', 'how much the final memory moves when the first input changes a little'], ['t', 'the step number']],
      code: [
        'Wx = np.array([1.2, 0.4])                  # input to hidden',
        'Wh = s * np.array([[c, si], [-si, c]])     # hidden to hidden: a rotation by 40 degrees, scaled by s',
        'h = np.zeros(2)                            # empty memory',
        'for x in [1, 0, 0, 0, 0, 0]:               # one pulse, then silence',
        '    z = x * Wx + h @ Wh                    # mix the input with the memory',
        '    h = np.tanh(z)                         # squash into (-1, 1)',
        'print(h)                                   # what is left of the pulse'
      ],
      nodes: [
        { id: 'init', label: ['Start: h₀ = (0, 0)'], why: 'Before the sequence begins, the network remembers nothing.' },
        { id: 'read', label: ['Read the next input xₜ'], why: 'An RNN sees one element of the sequence at a time, in order.' },
        { id: 'mix', label: ['Mix: z = xₜ·Wₓ + hₜ₋₁·Wₕ'], why: 'The new input and the old memory are combined by the same weights at every step: weight sharing across time.' },
        { id: 'squash', label: ['Squash: hₜ = tanh(z)'], why: 'tanh keeps the memory between −1 and 1, so it cannot grow without bound.' },
        { id: 'more', label: ['More inputs?'], kind: 'dec', why: 'The loop runs once per element of the sequence.' },
        { id: 'out', label: ['The final memory'], kind: 'end', why: 'Whatever the network will say about the sequence must come from this last hidden state.' }
      ],
      edges: [
        { from: 'init', to: 'read', label: 'first input', why: 'The sequence starts.' },
        { from: 'read', to: 'mix', label: 'combine', why: 'Input and memory meet in one weighted sum.' },
        { from: 'mix', to: 'squash', label: 'tanh', why: 'Squash the sum into the memory\'s range.' },
        { from: 'squash', to: 'more', label: 'store hₜ', why: 'The new memory replaces the old one.' },
        { from: 'more', to: 'read', label: 'yes: next', why: 'More of the sequence is left: read it with the updated memory.' },
        { from: 'more', to: 'out', label: 'no', why: 'The sequence is over.' }
      ],
      run(p) {
        const s = +p.s, U = unroll(s), steps = [], st = extra => Object.assign({ hs: U.hs }, extra);
        steps.push({ node: 'init', lines: [1, 2, 3], say: 'The memory starts empty, h₀ = (0, 0). The input is a single 1 followed by five 0s, so everything the network knows about the 1 has to be carried in its memory.', vars: [['t', 0], ['hₜ', '(0, 0)']], state: st({ upto: 0 }) });
        XS.forEach((x, k) => {
          const t = k + 1, h = U.hs[t], z = U.zs[k], prev = U.hs[k];
          if (t <= 2) {
            steps.push({ node: 'read', lines: [4], say: 'Step ' + t + ': the input is ' + x + (t === 1 ? ', the pulse.' : '. From now on only the memory carries the pulse.'), vars: [['t', t], ['xₜ', x]], state: st({ upto: k }) });
            steps.push({ node: 'mix', lines: [5], say: 'The input and the previous memory are mixed by the weights into z = ' + vec(z) + '.',
              formula: 'z = xₜ·Wₓ + hₜ₋₁·Wₕ + b = ' + x + '·' + vec(WX) + ' + ' + vec(prev) + '·Wₕ = ' + vec(z), vars: [['t', t], ['z', vec(z)]], state: st({ upto: k, z }) });
            steps.push({ node: 'squash', lines: [6], say: 'tanh squashes each number into the range from −1 to 1. The new memory is ' + vec(h) + ', at a distance of ' + f(Math.hypot(h[0], h[1])) + ' from empty.',
              formula: 'hₜ = tanh(z) = ' + vec(h), vars: [['t', t], ['hₜ', vec(h)]], state: st({ upto: t }) });
            steps.push({ node: 'more', lines: [4], say: 'More inputs follow, so the loop continues from this memory.', vars: [['t', t], ['hₜ', vec(h)]], state: st({ upto: t }) });
          } else {
            steps.push({ node: 'squash', edge: 'mix>squash', lines: [5, 6], say: 'Step ' + t + ': the input is 0, so only the memory drives the update. hₜ = ' + vec(h) + ', at a distance of ' + f(Math.hypot(h[0], h[1])) + ' from empty.',
              formula: 'hₜ = tanh(hₜ₋₁·Wₕ) = ' + vec(h), vars: [['t', t], ['hₜ', vec(h)]], state: st({ upto: t }) });
          }
        });
        steps.push({ node: 'more', lines: [4], say: 'All six inputs have been read.', vars: [['t', XS.length], ['hₜ', vec(U.hs[XS.length])]], state: st({ upto: XS.length }) });
        const size = g => Math.hypot(g[0], g[1]), all = SCALES.map(v => [v, size(unroll(v).g)]), mine = size(U.g);
        const saturates = size(unroll(1.5).g) < size(unroll(1).g);
        steps.push({ node: 'out', lines: [7], say: 'The final memory is ' + vec(U.hs[XS.length]) + '. How much does it still depend on the first input? By the chain rule, a small change in x₁ moves h₆ by ' + f(mine) + ' times as much. For the three scales: ' + all.map(a => a[0] + ' gives ' + f(a[1])).join(', ') + '. A small recurrent weight lets the pulse fade, the vanishing gradient of the next section' + (saturates ? '; a large one saturates tanh, whose slope then flattens, so its gradient shrinks too.' : '.'),
          formula: '∂h₆/∂x₁ = ' + vec(U.g) + ',  size ' + f(mine), vars: [['∂h₆/∂x₁', f(mine)]], state: st({ upto: XS.length }) });
        return steps;
      },
      draw(s, kit) {
        const xd = [-1.4, 1.4], fr = kit.frame(xd, xd, ['h₁', 'h₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        out.push({ k: 'box', t: 'r', x: sx(-1), y: sy(1), w: sx(1) - sx(-1), h: sy(-1) - sy(1), cls: 'wp-cell' });
        const hs = s.hs || [], upto = Math.min(s.upto || 0, hs.length - 1);
        if (upto > 0) out.push({ k: 'path', t: 'p', d: hs.slice(0, upto + 1).map((h, i) => (i ? 'L' : 'M') + sx(h[0]).toFixed(1) + ' ' + sy(h[1]).toFixed(1)).join(' '), cls: 'wp-line' });
        const placed = [[sx(hs[upto][0]), sy(hs[upto][1])]];          // label the current step, then any step with room for it
        hs.slice(0, upto + 1).forEach((h, i) => {
          const x = sx(h[0]), y = sy(h[1]), room = i === upto || placed.every(q => Math.hypot(q[0] - x, q[1] - y) > 16);
          out.push({ k: 'h' + i, t: 'c', x, y, r: i === upto ? 6 : 3.5, cls: i === upto ? 'wp-c1' : 'wp-c0' });
          if (room) { out.push({ k: 'n' + i, t: 't', x: x + 8, y: y - 6, s: String(i), cls: 'wp-tick' }); if (i !== upto) placed.push([x, y]); }
        });
        if (s.z) { out.push({ k: 'z', t: 'c', x: sx(s.z[0]), y: sy(s.z[1]), r: 5, cls: 'wp-hl' }); out.push({ k: 'zl', t: 't', x: 40, y: 208, s: 'ring: z, before tanh', cls: 'wp-tick', f: 1 }); }
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- retrieval for RAG (the lesson's code, in the browser)
  (function () {
    const DOCS = [
      ['returns.md', 'Items can be returned within 30 days of delivery. Sale items can only be exchanged, not refunded. Opened electronics have a 14 day return window.'],
      ['shipping.md', 'Standard shipping takes 3 to 5 business days. Express shipping arrives the next business day if ordered before 2pm.'],
      ['warranty.md', 'All electronics include a two year warranty covering manufacturing defects. Accidental damage is not covered.'],
      ['account.md', 'You can reset your password from the login page. Two factor authentication can be enabled in account settings.']];
    const QUESTION = 'How long do I have to return opened electronics?';
    const chunk = (text, size, overlap) => {
      const words = text.split(/\s+/), step = size - overlap, out = [];
      for (let start = 0; start < words.length; start += step) { out.push(words.slice(start, start + size).join(' ')); if (start + size >= words.length) break; }
      return out;
    };
    const clean = w => w.replace(/^[.,?!]+|[.,?!]+$/g, '').toLowerCase();
    const CHUNKS = [];
    DOCS.forEach(([src, text]) => chunk(text, 12, 4).forEach(c => CHUNKS.push({ src, text: c })));
    const VOCAB = [...new Set(CHUNKS.flatMap(c => c.text.split(/\s+/).map(clean)))].sort();
    const counts = text => { const m = {}; text.split(/\s+/).map(clean).filter(w => VOCAB.includes(w)).forEach(w => { m[w] = (m[w] || 0) + 1; }); return m; };
    const norm = m => Math.sqrt(Object.values(m).reduce((s, v) => s + v * v, 0));
    const cosine = (a, b) => { const na = norm(a), nb = norm(b); return na && nb ? Object.keys(a).reduce((s, w) => s + a[w] * (b[w] || 0), 0) / (na * nb) : 0; };
    const tag = i => '#' + (i + 1);
    const short = (t, n) => (t.length > n ? t.slice(0, n - 1).replace(/\s+\S*$/, '') + ' …' : t);
    W('rag-retrieve', {
      title: 'Retrieval for RAG: find the chunks that answer a question',
      params: [{ key: 'k', label: 'k (chunks kept)', value: 2, options: [[1, '1'], [2, '2'], [3, '3']] }],
      symbols: [['q', 'the question\'s vector: its word counts, scaled to length 1'], ['c', 'a chunk\'s vector, made the same way'],
        ['cos', 'cosine similarity: 1 when two vectors point the same way, 0 when they share no words'], ['‖', 'the length of a vector, as in ‖q‖'],
        ['V', 'the vocabulary size: every distinct word in the chunks'], ['k', 'how many chunks are kept (set below)']],
      code: [
        'chunks = [(name, c) for name, text in docs.items() for c in chunk(text, 12, 4)]',
        'vocab = sorted({clean(w) for _, c in chunks for w in c.split()})',
        'E = np.stack([embed(c) for _, c in chunks])        # one unit vector per chunk',
        'q = embed(question)                                 # the same "embedding"',
        'scores = E @ q                                      # cosine similarity with every chunk',
        'top = np.argsort(-scores)[:k]                       # the k best chunks',
        'prompt = instructions + documents(top) + question  # what the model reads'
      ],
      nodes: [
        { id: 'chunk', label: ['Split the documents', 'into overlapping chunks'], why: 'One vector for a whole document blurs its topics together. Small chunks give sharp vectors and fit in the prompt.' },
        { id: 'embed', label: ['Embed every chunk'], why: 'Each chunk becomes a vector, so that similar meaning means nearby vectors. Here the vector is word counts; a real system uses an embedding model.' },
        { id: 'query', label: ['Embed the question', 'the same way'], why: 'Question and chunks must live in the same vector space to be compared.' },
        { id: 'score', label: ['Score each chunk: cos(q, c)'], why: 'Cosine similarity compares directions, not lengths, so a long chunk is not favoured just for being long.' },
        { id: 'rank', label: ['Rank, keep the top k'], why: 'Only the best few chunks fit in the prompt, and more text would dilute the model\'s attention.' },
        { id: 'prompt', label: ['Build the prompt:', 'instructions, chunks, question'], kind: 'end', why: 'The model answers from the retrieved text and cites it. If retrieval missed the right chunk, no prompt wording can save the answer.' }
      ],
      edges: [
        { from: 'chunk', to: 'embed', label: 'index', why: 'Indexing happens once, before any question arrives.' },
        { from: 'embed', to: 'query', label: 'a question', why: 'At question time, only the question needs embedding.' },
        { from: 'query', to: 'score', label: 'compare', why: 'Score the question against every chunk in the index.' },
        { from: 'score', to: 'rank', label: 'sort', why: 'The highest scores are the most similar chunks.' },
        { from: 'rank', to: 'prompt', label: 'assemble', why: 'The kept chunks go into the prompt, each with its source.' }
      ],
      run(p) {
        const k = +p.k, steps = [], q = counts(QUESTION), cm = CHUNKS.map(c => counts(c.text)), sc = cm.map(m => cosine(q, m));
        const bySrc = DOCS.map(([src]) => src + ' ' + CHUNKS.map((c, i) => (c.src === src ? tag(i) : null)).filter(Boolean).join(', ')).join('; ');
        steps.push({ node: 'chunk', lines: [1], say: 'Four help-centre documents become ' + CHUNKS.length + ' chunks of up to 12 words. Neighbouring chunks share 4 words, so text near a cut appears in both. The chunks are ' + bySrc + '.',
          formula: CHUNKS.map((c, i) => tag(i) + '  ' + short(c.text, 46)).join('\n'), state: { scores: [] } });
        const m0 = cm[0];
        steps.push({ node: 'embed', lines: [2, 3], say: 'The vocabulary holds every distinct word of the chunks: V = ' + VOCAB.length + ' words. Each chunk becomes a vector of its word counts, divided by its length so that every vector has length 1.',
          formula: '#1 → ' + Object.keys(m0).slice(0, 8).map(w => w + ' ' + m0[w]).join(', ') + ' … ÷ ' + f(norm(m0)), vars: [['V', VOCAB.length]], state: { scores: [] } });
        const qWords = QUESTION.split(/\s+/).map(clean), missing = qWords.filter(w => !VOCAB.includes(w));
        steps.push({ node: 'query', lines: [4], say: 'The question becomes a vector the same way. ' + Object.keys(q).length + ' of its words are in the vocabulary; ' + (missing.length ? 'the others (' + missing.map(w => '"' + w + '"').join(', ') + ') match nothing and are ignored.' : 'all of them count.'),
          formula: 'q → ' + Object.keys(q).map(w => w + ' ' + q[w]).join(', ') + ' ÷ ' + f(norm(q)), state: { scores: [] }, ask: 'Which document do you expect to score highest?' });
        [0, 1].forEach(i => {
          const shared = Object.keys(q).filter(w => cm[i][w]);
          steps.push({ node: 'score', lines: [5], say: 'Chunk ' + tag(i) + ' shares ' + (shared.length ? shared.map(w => '"' + w + '"').join(', ') : 'no words') + ' with the question, so its score is ' + f(sc[i]) + '.',
            formula: 'cos(q, c) = q·c / (‖q‖ ‖c‖) = ' + shared.reduce((s, w) => s + q[w] * cm[i][w], 0) + ' / (' + f(norm(q)) + ' × ' + f(norm(cm[i])) + ') = ' + f(sc[i]), state: { scores: sc.slice(0, i + 1) } });
        });
        steps.push({ node: 'score', lines: [5], say: 'One matrix product scores every chunk at once.', formula: sc.map((v, i) => tag(i) + ' ' + f(v)).join(',  '), state: { scores: sc } });
        const order = sc.map((v, i) => i).sort((a, b) => sc[b] - sc[a]), top = order.slice(0, k);
        steps.push({ node: 'rank', lines: [6], say: 'Sorted by score, the best chunk is ' + tag(order[0]) + ' from ' + CHUNKS[order[0]].src + ': "' + CHUNKS[order[0]].text + '" With k = ' + k + ', the kept chunk' + (k > 1 ? 's are ' : ' is ') + top.map(tag).join(', ') + '.',
          formula: order.map(i => tag(i) + ' ' + f(sc[i])).join(',  '), vars: [['k', k]], state: { scores: sc, top } });
        const context = top.map((i, n) => '<doc id="' + (n + 1) + '" source="' + CHUNKS[i].src + '">' + CHUNKS[i].text + '</doc>').join('\n');
        const hasAnswer = top.some(i => /14 day return/.test(CHUNKS[i].text));
        steps.push({ node: 'prompt', lines: [7], say: 'The prompt tells the model to answer only from the documents and to cite them. ' + (hasAnswer ? (k === 1 ? 'The one kept chunk' : 'One of the kept chunks') + ' holds the 14-day return window for opened electronics, so the model can answer correctly.' : 'No kept chunk holds the answer, so the model should say it does not know.') + ' Word counts only match shared words: a real embedding model would also match a paraphrase such as "send back my laptop".',
          formula: 'Answer using only the documents below and cite them like [1]. If the answer is not in the documents, say that you don\'t know.\n\n' + context + '\n\nQuestion: ' + QUESTION, plain: true, state: { scores: sc, top } });
        return steps;
      },
      draw(s, kit) {
        const out = [], base = 168, h = 120, n = CHUNKS.length, bw = Math.min(24, 240 / n - 4);
        out.push({ k: 'ttl', t: 't', x: 12, y: 16, s: 'cosine similarity to the question', cls: 'wp-tick' });
        out.push({ k: 'base', t: 'l', x1: 10, y1: base, x2: 292, y2: base, cls: 'wp-axis' });
        CHUNKS.forEach((c, i) => {
          const v = s.scores && s.scores[i] !== undefined ? s.scores[i] : null, x = 18 + i * (bw + 6), hh = v === null ? 0 : v * h, top = s.top && s.top.includes(i);
          out.push({ k: 'b' + i, t: 'r', x, y: base - hh, w: bw, h: hh, cls: top ? 'wp-bar2' : 'wp-bar' });
          if (v !== null) out.push({ k: 'v' + i, t: 't', x: x + bw / 2, y: base - hh - 4, s: f(v), cls: 'wp-tick', a: 'middle' });
          out.push({ k: 'n' + i, t: 't', x: x + bw / 2, y: base + 13, s: tag(i), cls: top ? 'wp-txt' : 'wp-tick', a: 'middle' });
          out.push({ k: 'd' + i, t: 't', x: x + bw / 2, y: base + 25, s: c.src[0], cls: 'wp-tick', a: 'middle' });
        });
        out.push({ k: 'key', t: 't', x: 12, y: 205, s: 'r returns, s shipping, w warranty, a account', cls: 'wp-tick' });
        return out.map(o => Object.assign(o, { f: 1 }));
      }
    });
  })();

  // ---------------------------------------------------------------- the KV cache while generating
  (function () {
    const TOK = ['the', 'cat', 'sat', 'on', 'the'], EMB = { the: [1.0, 0.2], cat: [0.3, 1.0], sat: [0.9, 0.7], on: [-0.5, 0.8] };
    const WQ = [[0.9, 0.3], [-0.2, 1.1]], WK = [[1.0, -0.4], [0.5, 0.8]], WV = [[0.6, 0.9], [1.2, -0.3]];
    const mul = (x, M) => [x[0] * M[0][0] + x[1] * M[1][0], x[0] * M[0][1] + x[1] * M[1][1]];
    const withCache = t => 3 * t, without = t => t * t + 2 * t;            // projections done so far
    W('kv-cache', {
      title: 'The KV cache: what generating one more token costs',
      symbols: [['x', 'the new token\'s embedding'], ['q', 'the new token\'s query'], ['k', 'a key; one is stored per token'], ['v', 'a value; one is stored per token'],
        ['Wq', 'the query weights'], ['Wk', 'the key weights'], ['Wv', 'the value weights'], ['d', 'the vector size, 2 here'], ['√', 'the square root'],
        ['w', 'the attention weights over the cached tokens, summing to 1'], ['K', 'the cached keys, one row per token'], ['Σ', 'the sum over the cached tokens'],
        ['o', 'the attention output for the new token'], ['t', 'the number of tokens so far']],
      code: [
        'K, V = [], []                               # the cache starts empty',
        'for x in token_embeddings:                  # one new token at a time',
        '    q, k, v = x @ Wq, x @ Wk, x @ Wv        # project only the new token',
        '    K.append(k); V.append(v)                # keep its key and value',
        '    w = softmax(np.array(K) @ q / np.sqrt(d))',
        '    o = w @ np.array(V)                     # attend over everything cached'
      ],
      nodes: [
        { id: 'token', label: ['A new token arrives'], why: 'A language model generates one token at a time, and each new token must attend to all the earlier ones.' },
        { id: 'proj', label: ['Project it: q, k, v'], why: 'Only the new token needs projecting. The keys and values of earlier tokens never change, because a token cannot see what comes after it.' },
        { id: 'cache', label: ['Append k and v to the cache'], why: 'Keeping old keys and values saves recomputing them at every step. The cost is memory, one key and one value per token.' },
        { id: 'attend', label: ['Attend: o = Σ wⱼ vⱼ', 'over every cached token'], why: 'The new token\'s query is compared with every cached key; the output mixes the cached values.' },
        { id: 'more', label: ['Another token?'], kind: 'dec', why: 'Generation goes on until an end token or a length limit.' },
        { id: 'done', label: ['Done: the work saved'], kind: 'end', why: 'Without a cache, step t would recompute t keys and t values. With it, every step does the same small amount of new work.' }
      ],
      edges: [
        { from: 'token', to: 'proj', label: 'embed', why: 'The token\'s embedding is multiplied by three weight matrices.' },
        { from: 'proj', to: 'cache', label: 'store', why: 'The new key and value join those of all earlier tokens.' },
        { from: 'cache', to: 'attend', label: 'look back', why: 'Attention reads the whole cache.' },
        { from: 'attend', to: 'more', label: 'output', why: 'The output feeds the rest of the layer and, eventually, the next-token prediction.' },
        { from: 'more', to: 'token', label: 'yes: next', why: 'The next token repeats the same small amount of work.' },
        { from: 'more', to: 'done', label: 'no', why: 'Generation has stopped.' }
      ],
      run() {
        const steps = [], K = [], V = [];
        TOK.forEach((tk, n) => {
          const t = n + 1, x = EMB[tk], q = mul(x, WQ), k = mul(x, WK), v = mul(x, WV);
          K.push(k); V.push(v);
          const s = K.map(kk => (q[0] * kk[0] + q[1] * kk[1]) / Math.SQRT2), mx = Math.max(...s), ex = s.map(z => Math.exp(z - mx)), Z = ex.reduce((a, b) => a + b, 0), w = ex.map(z => z / Z);
          const o = [0, 1].map(c => w.reduce((acc, wj, j) => acc + wj * V[j][c], 0));
          const vars = [['t', t], ['with cache', withCache(t)], ['without', without(t)]];
          const st = extra => Object.assign({ t, K: K.map(r => r.slice()), V: V.map(r => r.slice()) }, extra || {});
          if (t <= 2) {
            steps.push({ node: 'token', lines: [2], say: 'Token ' + t + ', "' + tk + '", arrives with the embedding x = ' + vec(x) + '.', vars, state: st({ t: t - 1, K: K.slice(0, -1), V: V.slice(0, -1) }) });
            steps.push({ node: 'proj', lines: [3], say: 'Three small matrix products give its query, key and value. That is all the projection work this step needs.', formula: 'q = x·Wq = ' + vec(q) + '\nk = x·Wk = ' + vec(k) + '\nv = x·Wv = ' + vec(v), vars, state: st({ t: t - 1, K: K.slice(0, -1), V: V.slice(0, -1) }) });
            steps.push({ node: 'cache', lines: [4], say: 'k and v join the cache, which now holds ' + t + ' key' + (t > 1 ? 's' : '') + ' and ' + t + ' value' + (t > 1 ? 's' : '') + '. Without a cache, this step would also recompute the keys and values of the ' + (t - 1) + ' earlier token' + (t === 2 ? '' : 's') + '.', vars, state: st() });
            steps.push({ node: 'attend', lines: [5, 6], say: 'The query is compared with every cached key. The weights are ' + w.map(f).join(', ') + ', and the output is o = ' + vec(o) + '.',
              formula: 'w = softmax(K·q / √d) = ' + list(w) + '\no = Σ wⱼ vⱼ = ' + vec(o), vars, state: st({ w }), ask: t === 2 ? 'At the next token, how many keys must be computed with the cache, and how many without it?' : null });
            steps.push({ node: 'more', lines: [2], say: 'More tokens follow.', vars, state: st({ w }) });
          } else {
            steps.push({ node: 'attend', edge: 'cache>attend', lines: [2, 3, 4, 5, 6], say: 'Token ' + t + ', "' + tk + '": one new q, k and v, the cache grows to ' + t + ' rows, and attention reads all of them. Projections so far: ' + withCache(t) + ' with the cache, ' + without(t) + ' without.',
              formula: 'w = ' + list(w) + '\no = ' + vec(o), vars, state: st({ w }) });
          }
        });
        const T = TOK.length;
        steps.push({ node: 'more', lines: [2], say: 'The ' + T + ' tokens are done.', vars: [['t', T], ['with cache', withCache(T)], ['without', without(T)]], state: { t: T, K: K.map(r => r.slice()), V: V.map(r => r.slice()) } });
        steps.push({ node: 'done', lines: [], say: 'For ' + T + ' tokens the cache needed ' + withCache(T) + ' projections where recomputing would need ' + without(T) + '. The gap grows with length: for 1,000 tokens it is ' + withCache(1000).toLocaleString('en-US') + ' against ' + without(1000).toLocaleString('en-US') + '. The price is memory: one key and one value per token, per head and per layer, which is what the formula above measures.',
          vars: [['with cache', withCache(T)], ['without', without(T)]], state: { t: T, K: K.map(r => r.slice()), V: V.map(r => r.slice()), done: true } });
        return steps;
      },
      draw(s) {
        const out = [], t = s.t || 0, rowY = j => 38 + j * 21;
        out.push({ k: 'h0', t: 't', x: 12, y: 18, s: 'cache', cls: 'wp-tick' });
        out.push({ k: 'h1', t: 't', x: 62, y: 18, s: 'key k', cls: 'wp-tick' });
        out.push({ k: 'h2', t: 't', x: 132, y: 18, s: 'value v', cls: 'wp-tick' });
        out.push({ k: 'h3', t: 't', x: 210, y: 18, s: 'weight', cls: 'wp-tick' });
        TOK.forEach((tk, j) => {
          const on = j < t, y = rowY(j);
          out.push({ k: 'row' + j, t: 'r', x: 8, y: y - 12, w: 284, h: 17, cls: on ? (j === t - 1 ? 'wp-cell-on' : 'wp-cell') : 'wp-bar-off' });
          out.push({ k: 'tk' + j, t: 't', x: 14, y, s: on ? tk : '', cls: 'wp-txt' });
          out.push({ k: 'k' + j, t: 't', x: 62, y, s: on && s.K && s.K[j] ? vec(s.K[j]) : '', cls: 'wp-tick' });
          out.push({ k: 'v' + j, t: 't', x: 132, y, s: on && s.V && s.V[j] ? vec(s.V[j]) : '', cls: 'wp-tick' });
          const wj = s.w && on ? s.w[j] : 0;
          out.push({ k: 'w' + j, t: 'r', x: 210, y: y - 9, w: wj * 50, h: 11, cls: 'wp-bar2' });
          out.push({ k: 'wl' + j, t: 't', x: 214 + wj * 50, y, s: wj ? f(wj) : '', cls: 'wp-tick' });
        });
        const c1 = withCache(t), c2 = without(t), scale = 170 / without(TOK.length);
        out.push({ k: 'c1l', t: 't', x: 12, y: 160, s: 'with cache', cls: 'wp-tick' });
        out.push({ k: 'c1', t: 'r', x: 70, y: 151, w: c1 * scale, h: 12, cls: 'wp-bar' });
        out.push({ k: 'c1v', t: 't', x: 74 + c1 * scale, y: 160, s: String(c1), cls: 'wp-tick' });
        out.push({ k: 'c2l', t: 't', x: 12, y: 182, s: 'without', cls: 'wp-tick' });
        out.push({ k: 'c2', t: 'r', x: 70, y: 173, w: c2 * scale, h: 12, cls: 'wp-bar-off' });
        out.push({ k: 'c2v', t: 't', x: 74 + c2 * scale, y: 182, s: String(c2), cls: 'wp-tick' });
        out.push({ k: 'cap', t: 't', x: 12, y: 202, s: 'projections computed so far', cls: 'wp-tick' });
        return out.map(o => Object.assign(o, { f: 1 }));
      }
    });
  })();
})(typeof globalThis !== 'undefined' ? globalThis : window);
