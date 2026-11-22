/* ML Atlas – step-through walkthroughs for deep learning, LLMs and reinforcement learning (drawn by js/views/walks.js).
 * Every number on screen comes from running the algorithm below on its small fixed example. */
(function (global) {
  const MLA = global.MLA, W = MLA.walk;
  const f = v => MLA.walkKit.num(v);
  const vec = v => '(' + v.map(f).join(', ') + ')';

  // ---------------------------------------------------------------- Adam on an elongated bowl
  (function () {
    const fn = t => t[0] ** 2 + 10 * t[1] ** 2, grad = t => [2 * t[0], 20 * t[1]];
    const B1 = 0.9, B2 = 0.999, EPS = 1e-8, START = [-4, 2];
    W('adam', {
      title: 'One Adam update at a time',
      symbols: [['θ', 'the two parameters (θ₁, θ₂): the point being moved downhill'],
        ['f(θ)', 'the function being minimized, θ₁² + 10·θ₂²: a valley ten times steeper along θ₂ than along θ₁'],
        ['g', 'the gradient ∇f(θ): the slope of f along each parameter'],
        ['m', 'the running average of recent gradients (momentum): it keeps the direction steady'],
        ['v', 'the running average of squared gradients: how large the gradients of each parameter tend to be'],
        ['β₁', 'the share of the old m kept at each step (0.9)'], ['β₂', 'the share of the old v kept at each step (0.999)'],
        ['m̂', 'm corrected for starting at zero, which makes the first averages too small'], ['v̂', 'v corrected the same way'],
        ['t', 'the step number'], ['η', 'the learning rate (set below)'], ['ε', 'a tiny number (10⁻⁸) that prevents division by zero'],
        ['√', 'the square root, taken for each parameter separately'], ['←', 'replace the value on the left with the value on the right']],
      params: [{ key: 'lr', label: 'Learning rate η', value: 0.3, options: [[0.1, '0.1'], [0.3, '0.3'], [0.6, '0.6']] }],
      code: [
        'theta = np.array([-4.0, 2.0]); m = np.zeros(2); v = np.zeros(2)',
        'for t in range(1, 201):',
        '    g = grad(theta)                          # which way is uphill',
        '    m = b1 * m + (1 - b1) * g                # running mean of gradients',
        '    v = b2 * v + (1 - b2) * g ** 2           # running mean of squares',
        '    m_hat = m / (1 - b1 ** t)                # undo the bias toward 0',
        '    v_hat = v / (1 - b2 ** t)',
        '    theta -= lr * m_hat / (np.sqrt(v_hat) + eps)',
        '    if f(theta) < 1e-4: break'
      ],
      nodes: [
        { id: 'init', label: ['Start: θ = (−4, 2), m = v = 0'], why: 'The loss f(θ) = θ₁² + 10·θ₂² is a narrow valley: ten times steeper across θ₂ than along θ₁. Plain gradient descent zig-zags in valleys like this.' },
        { id: 'grad', label: ['Gradient g = ∇f(θ)'], why: 'As in every gradient method, the gradient says which way is uphill and how steep it is, for each parameter.' },
        { id: 'm', label: ['Momentum: m ← β₁·m + (1 − β₁)·g'], why: 'A running average of recent gradients. Directions that keep agreeing build up speed; directions that flip back and forth cancel out.' },
        { id: 'v', label: ['Scale: v ← β₂·v + (1 − β₂)·g²'], why: 'A running average of squared gradients: a per-parameter measure of how large the gradients usually are.' },
        { id: 'corr', label: ['Bias-correct both:', 'm̂ = m/(1 − β₁ᵗ), v̂ = v/(1 − β₂ᵗ)'], why: 'm and v start at zero, so early on they underestimate. Dividing by 1 − βᵗ removes that bias; the correction fades as t grows.' },
        { id: 'step', label: ['Step: θ ← θ − η·m̂ / (√v̂ + ε)'], why: 'Dividing by √v̂ gives every parameter a step of about η no matter how steep its direction: the steep θ₂ no longer dominates.' },
        { id: 'check', label: ['Close enough to the minimum?'], kind: 'dec', why: 'Stop when the loss is tiny, or the step budget runs out.' },
        { id: 'done', label: ['Done'], kind: 'end', why: 'θ has reached the bottom of the valley at (0, 0).' }
      ],
      edges: [
        { from: 'init', to: 'grad', label: 'first step', why: 'Everything Adam does starts from the gradient.' },
        { from: 'grad', to: 'm', label: 'smooth', why: 'Average the direction over recent steps.' },
        { from: 'm', to: 'v', label: 'measure size', why: 'Track how large each parameter\'s gradients are.' },
        { from: 'v', to: 'corr', label: 'fix the start', why: 'Both averages started at zero and need correcting.' },
        { from: 'corr', to: 'step', label: 'normalize', why: 'Divide the direction by its typical size.' },
        { from: 'step', to: 'check', label: 'new θ', why: 'Check whether the minimum is reached.' },
        { from: 'check', to: 'grad', label: 'no: again', why: 'Not there yet: compute a fresh gradient at the new point.' },
        { from: 'check', to: 'done', label: 'yes', why: 'The loss is below the tolerance.' }
      ],
      run(p) {
        const lr = +p.lr, steps = [], path = [START.slice()];
        let th = START.slice(), m = [0, 0], v = [0, 0], since = 1;
        const st = () => ({ path: path.map(q => q.slice()) });
        steps.push({ node: 'init', lines: [1], say: 'θ starts at (−4, 2), up the side of a narrow valley whose bottom is at (0, 0). The loss there is ' + f(fn(th)) + '.', vars: [['t', 0], ['θ', vec(th)], ['f(θ)', f(fn(th))]], state: st() });
        for (let t = 1; t <= 200; t++) {
          const g = grad(th);
          const m2 = m.map((x, i) => B1 * x + (1 - B1) * g[i]), v2 = v.map((x, i) => B2 * x + (1 - B2) * g[i] ** 2);
          const mh = m2.map(x => x / (1 - B1 ** t)), vh = v2.map(x => x / (1 - B2 ** t));
          const th2 = th.map((x, i) => x - lr * mh[i] / (Math.sqrt(vh[i]) + EPS));
          if (t <= 2) {
            steps.push({ node: 'grad', lines: [3], say: 'Step ' + t + ': the gradient at ' + vec(th) + ' is ' + vec(g) + '. The θ₂ part is ' + f(Math.abs(g[1] / g[0])) + ' times larger: the valley is much steeper across than along.', formula: 'g = ∇f(θ) = (2·θ₁, 20·θ₂) = ' + vec(g), vars: [['t', t], ['θ', vec(th)], ['g', vec(g)]], state: st() });
            steps.push({ node: 'm', lines: [4], say: 'The momentum average moves 10% of the way toward the new gradient: m = ' + vec(m2) + '.', formula: 'm ← β₁·m + (1 − β₁)·g\n  = 0.9 × ' + vec(m) + ' + 0.1 × ' + vec(g) + ' = ' + vec(m2), vars: [['t', t], ['m', vec(m2)]], state: st() });
            steps.push({ node: 'v', lines: [5], say: 'The average of squared gradients moves 0.1% of the way: v = ' + vec(v2) + '. It records how big each parameter\'s gradients tend to be.', formula: 'v ← β₂·v + (1 − β₂)·g²\n  = 0.999 × ' + vec(v) + ' + 0.001 × ' + vec(g.map(x => x * x)) + ' = ' + vec(v2), vars: [['t', t], ['v', vec(v2)]], state: st() });
            steps.push({ node: 'corr', lines: [6, 7], say: 'Both averages started at zero, so they are far too small at step ' + t + '. The correction rescales them: m̂ = ' + vec(mh) + ', v̂ = ' + vec(vh) + '.', formula: 'm̂ = m / (1 − β₁ᵗ) = m / (1 − 0.9^' + t + ') = ' + vec(mh) + '\nv̂ = v / (1 − β₂ᵗ) = v / (1 − 0.999^' + t + ') = ' + vec(vh), vars: [['t', t], ['m̂', vec(mh)], ['v̂', vec(vh)]], state: st(),
              ask: t === 1 ? 'The θ₂ gradient is ten times bigger. Will θ₂ take a ten times bigger step?' : null });
            steps.push({ node: 'step', lines: [8], say: 'Each parameter moves by ' + vec(th2.map((x, i) => x - th[i])) + ': about η = ' + lr + ' each, even though their gradients differ tenfold. Dividing by √v̂ evens out the steps.', formula: 'θ ← θ − η·m̂ / (√v̂ + ε)\n  = ' + vec(th) + ' − ' + lr + ' × m̂ / (√v̂ + ε) = ' + vec(th2), vars: [['t', t], ['θ', vec(th2)], ['f(θ)', f(fn(th2))]], state: { path: path.concat([th2]).map(q => q.slice()) } });
          }
          th = th2; m = m2; v = v2; path.push(th.slice());
          const done = fn(th) < 1e-4 || t === 200;
          if (t <= 2 || done || [5, 10, 20, 40, 80, 150].includes(t)) {
            steps.push({ node: 'check', lines: [9], say: done ? (fn(th) < 1e-4 ? 'The loss is ' + f(fn(th)) + ', below the tolerance, after ' + t + ' steps.' : 'The step budget is used up with the loss at ' + f(fn(th)) + '.') : t <= 2 ? 'The loss is ' + f(fn(th)) + ': not there yet, so another step follows.' : 'Steps ' + since + ' to ' + t + ' repeated the loop. θ = ' + vec(th) + ', loss ' + f(fn(th)) + '.', vars: [['t', t], ['θ', vec(th)], ['f(θ)', f(fn(th))]], state: st() });
            since = t + 1;
          }
          if (done) {
            const over = path.some(q => q[0] * START[0] < 0 || q[1] * START[1] < 0);
            steps.push({ node: 'done', lines: [], say: (fn(th) < 1e-4 ? 'Adam reached the bottom in ' + t + ' steps. ' : 'After ' + t + ' steps the loss is ' + f(fn(th)) + '. ') +
              'Dividing by √v̂ gave both parameters steps of similar size, so the steep θ₂ direction did not dominate the way it does in plain gradient descent. ' +
              (over ? 'Momentum also carried θ past the bottom, after which it swung back and settled: the price of moving fast.' : 'It approached the bottom without overshooting it.'), vars: [['t', t], ['θ', vec(th)], ['f(θ)', f(fn(th))]], state: st() });
            break;
          }
        }
        return steps;
      },
      draw(s, kit) {
        const fr = kit.frame([-5, 2], [-2.5, 2.5], ['θ₁', 'θ₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        [1, 4, 9, 16, 25].forEach(c => {
          const rx = Math.sqrt(c) * (sx(1) - sx(0)), ry = Math.sqrt(c / 10) * (sy(0) - sy(1));
          out.push({ k: 'c' + c, t: 'p', d: 'M' + (sx(0) - rx) + ' ' + sy(0) + ' a' + rx + ' ' + ry + ' 0 1 0 ' + (2 * rx) + ' 0 a' + rx + ' ' + ry + ' 0 1 0 ' + (-2 * rx) + ' 0', cls: 'wp-faint' });
        });
        out.push({ k: 'min', t: 'c', x: sx(0), y: sy(0), r: 3, cls: 'wp-c2' });
        const pts = s.path || [START];
        out.push({ k: 'path', t: 'p', d: pts.map((q, i) => (i ? 'L' : 'M') + sx(q[0]).toFixed(1) + ' ' + sy(q[1]).toFixed(1)).join(' '), cls: 'wp-line' });
        const last = pts[pts.length - 1];
        out.push({ k: 'theta', t: 'c', x: sx(last[0]), y: sy(last[1]), r: 5.5, cls: 'wp-c1' });
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- one attention query, computed step by step
  (function () {
    const TOK = ['the', 'cat', 'sat', 'down'];
    const K = [[0.2, 0.1], [1.0, 0.8], [0.4, 0.9], [-0.3, 0.6]], V = [[0.1, 0.0], [1.0, 0.2], [0.3, 0.9], [-0.4, 0.5]];
    const Q = [1.2, 0.9], QI = 2;                                    // the query belongs to "sat"
    W('attention', {
      title: 'Attention for one token, computed step by step',
      symbols: [['q', 'the query vector of "sat": what this token is looking for'],
        ['k', 'a key vector: what a token offers to be matched against. kⱼ belongs to token j'],
        ['d', 'the length of the query and key vectors: 2 here, 64 or 128 in a real model'],
        ['s', 'the scaled score of a token, q·k / √d'], ['√', 'the square root'],
        ['w', 'the attention weight of a token: its share of the attention, between 0 and 1'], ['e', 'Euler\'s number, about 2.718'],
        ['Σ', 'the sum over all tokens'], ['v', 'a value vector: the information a token passes on. vⱼ belongs to token j'],
        ['out', 'the result: the weighted average of the value vectors'], ['output', 'the new vector for "sat", passed on to the next layer']],
      params: [{ key: 'mask', label: 'Causal mask', value: 'off', options: [['off', 'off (an encoder)'], ['on', 'on (a decoder)']] }],
      code: [
        'q = W_q @ x_sat                          # what "sat" is looking for',
        'scores = K @ q                           # match q against every key',
        'scores = scores / np.sqrt(d)             # keep the softmax from saturating',
        'if causal: scores[j > i] = -np.inf       # no peeking at later tokens',
        'w = np.exp(scores) / np.exp(scores).sum()   # softmax: weights summing to 1',
        'out = w @ V                              # a weighted mix of the values'
      ],
      nodes: [
        { id: 'query', label: ['Query q for the token "sat"'], why: 'Each token asks a question of the others. Its query vector is learned so that it matches the keys of the tokens it needs.' },
        { id: 'score', label: ['Score every key: q·kⱼ'], why: 'A dot product is large when two vectors point the same way, so it measures how well each token\'s key answers the query.' },
        { id: 'scale', label: ['Divide by √d'], why: 'Dot products grow with the vector size d. Scaling keeps the softmax from becoming nearly one-hot, which would make its gradients vanish.' },
        { id: 'mask', label: ['Causal mask: hide later tokens'], kind: 'dec', why: 'A decoder predicts the next token, so a token must not see the ones after it. An encoder like BERT skips this step.' },
        { id: 'soft', label: ['Softmax: weights that sum to 1'], why: 'The scores become a probability distribution over the tokens: how much attention each one gets.' },
        { id: 'mix', label: ['Mix the values: Σ wⱼ·vⱼ'], why: 'The output is a weighted average of the tokens\' value vectors, so the token takes in information mostly from the tokens it attends to.' },
        { id: 'out', label: ['Output: a context-aware vector'], kind: 'end', why: 'The new vector for "sat" now carries information from the tokens that matter to it. A transformer does this for every token, in every head and layer.' }
      ],
      edges: [
        { from: 'query', to: 'score', label: 'compare', why: 'The query is compared with the key of every token, itself included.' },
        { from: 'score', to: 'scale', label: '÷ √d', why: 'Raw dot products are too large for a well-behaved softmax.' },
        { from: 'scale', to: 'mask', label: 'decoder?', why: 'Whether later tokens may be seen depends on the model type.' },
        { from: 'mask', to: 'soft', label: 'normalize', why: 'Turn the remaining scores into weights.' },
        { from: 'soft', to: 'mix', label: 'weights', why: 'The weights decide how much of each value goes into the result.' },
        { from: 'mix', to: 'out', label: 'result', why: 'The mix is the attention output for this token.' }
      ],
      run(p) {
        const causal = p.mask === 'on', steps = [], d = 2;
        const sc = K.map(k => k[0] * Q[0] + k[1] * Q[1]), ss = sc.map(x => x / Math.sqrt(d));
        const masked = ss.map((x, j) => (causal && j > QI ? -Infinity : x));
        const ex = masked.map(x => (x === -Infinity ? 0 : Math.exp(x))), Z = ex.reduce((a, b) => a + b, 0), w = ex.map(x => x / Z);
        const out = [0, 1].map(c => w.reduce((s, wj, j) => s + wj * V[j][c], 0));
        const show = arr => TOK.map((t, j) => t + ' ' + (arr[j] === -Infinity ? '−∞' : f(arr[j]))).join(',  ');
        steps.push({ node: 'query', lines: [1], say: 'The sentence is "the cat sat down", and we follow the token "sat". Its query vector is q = ' + vec(Q) + '. The keys and values of all four tokens are 2-dimensional here, so they can be drawn.', vars: [['q', vec(Q)], ['d', d]], state: { bars: null } });
        TOK.forEach((t, j) => { if (j < 2) steps.push({ node: 'score', lines: [2], say: '"sat" against "' + t + '": the dot product of q with k = ' + vec(K[j]) + ' is ' + f(sc[j]) + '.', formula: 'q·k = q₁k₁ + q₂k₂ = ' + Q[0] + '×' + K[j][0] + ' + ' + Q[1] + '×' + K[j][1] + ' = ' + f(sc[j]), state: { bars: sc.map((x, i) => (i <= j ? x : null)), label: 'scores q·k' } }); });
        steps.push({ node: 'score', lines: [2], say: 'All four scores. "cat" matches the query best: its key points in nearly the same direction as q.', formula: 'q·kⱼ:  ' + show(sc), state: { bars: sc, label: 'scores q·k' }, ask: 'After the softmax, will "cat" get all of the attention, or only most of it?' });
        steps.push({ node: 'scale', lines: [3], say: 'Every score is divided by √2 = ' + f(Math.sqrt(2)) + '. With 2 dimensions this changes little; with the 64 or 128 dimensions of a real head, it matters a lot.', formula: 'sⱼ = q·kⱼ / √d:  ' + show(ss), state: { bars: ss, label: 'scores / √d' } });
        steps.push({ node: 'mask', lines: [4], say: causal ? 'With the causal mask on, "down" comes after "sat", so its score becomes −∞: it will get exactly zero weight.' : 'With the mask off, as in an encoder, "sat" may also look at "down", the token after it.', formula: (causal ? 'sⱼ after the mask:  ' : 'sⱼ, nothing masked:  ') + show(masked), state: { bars: masked, label: causal ? 'after masking' : 'scores / √d' } });
        steps.push({ node: 'soft', lines: [5], say: 'The softmax turns the scores into weights that sum to 1: ' + TOK.map((t, j) => t + ' ' + Math.round(w[j] * 100) + '%').join(', ') + '. "cat" gets the most attention, but not all of it: the softmax is soft.', formula: 'wⱼ = e^(sⱼ) / Σ e^(s) → ' + TOK.map((t, j) => f(w[j])).join(', '), state: { bars: w, label: 'attention weights', weights: true } });
        steps.push({ node: 'mix', lines: [6], say: 'The output is the weighted average of the value vectors. In the right panel, each value is drawn with a size that matches its weight, and the output (ringed) lands closest to the heavy ones.', formula: 'out = Σ wⱼ·vⱼ = ' + TOK.map((t, j) => f(w[j]) + '·' + vec(V[j])).join(' + ') + ' = ' + vec(out), state: { bars: w, label: 'attention weights', weights: true, mix: out } });
        steps.push({ node: 'out', lines: [6], say: 'The new vector for "sat" is ' + vec(out) + '. It now mixes in what "cat" offered, the subject that sat. Running the same steps for every token at once is one matrix product: softmax(QKᵀ/√d)·V.', vars: [['output', vec(out)]], state: { bars: w, label: 'attention weights', weights: true, mix: out } });
        return steps;
      },
      draw(s, kit) {
        const out = [], y0 = 34, h = 114, base = y0 + h, bw = 26;
        if (s.bars) {
          const fin = s.bars.filter(x => x !== null && isFinite(x)), top = Math.max(1, ...fin.map(Math.abs));
          out.push({ k: 'ttl', t: 't', x: 12, y: 16, s: s.label, cls: 'wp-tick' });
          out.push({ k: 'base', t: 'l', x1: 10, y1: base, x2: 150, y2: base, cls: 'wp-axis' });
          TOK.forEach((t, j) => {
            const v = s.bars[j], x = 16 + j * 34, hh = v === null || !isFinite(v) ? 0 : Math.abs(v) / top * h;
            out.push({ k: 'b' + j, t: 'r', x, y: base - hh, w: bw, h: hh, cls: j === QI ? 'wp-bar2' : 'wp-bar' });
            out.push({ k: 'v' + j, t: 't', x: x + bw / 2, y: base - hh - 4, s: v === null ? '' : v === -Infinity ? '−∞' : f(v), cls: 'wp-tick', a: 'middle' });
            out.push({ k: 'n' + j, t: 't', x: x + bw / 2, y: base + 14, s: t, cls: 'wp-txt', a: 'middle' });
          });
        }
        if (s.weights) {                                            // the value vectors as points; size = weight
          const ox = 222, oy = 110, sc = 62;
          out.push({ k: 'ax1', t: 'l', x1: ox - 70, y1: oy, x2: ox + 70, y2: oy, cls: 'wp-axis' });
          out.push({ k: 'ax2', t: 'l', x1: ox, y1: oy - 70, x2: ox, y2: oy + 70, cls: 'wp-axis' });
          out.push({ k: 'vt', t: 't', x: ox, y: 30, s: 'values', cls: 'wp-tick', a: 'middle' });
          V.forEach((v, j) => {
            const r = 3 + 9 * s.bars[j];
            out.push({ k: 'pv' + j, t: 'c', x: ox + v[0] * sc, y: oy - v[1] * sc, r, cls: j === QI ? 'wp-c1' : 'wp-c0' });
            out.push({ k: 'pl' + j, t: 't', x: ox + v[0] * sc, y: oy - v[1] * sc - r - 4, s: TOK[j], cls: 'wp-tick', a: 'middle' });
          });
          if (s.mix) {
            out.push({ k: 'mix', t: 'c', x: ox + s.mix[0] * sc, y: oy - s.mix[1] * sc, r: 5, cls: 'wp-q' });
            out.push({ k: 'mixr', t: 'c', x: ox + s.mix[0] * sc, y: oy - s.mix[1] * sc, r: 10, cls: 'wp-hl' });
          }
        }
        return out.map(o => Object.assign(o, { f: 1 }));
      }
    });
  })();

  // ---------------------------------------------------------------- the generation loop of a language model
  (function () {
    const VOC = ['the', 'cat', 'dog', 'sat', 'ran', 'on', 'mat', '.'];
    // The toy model: the scores (logits) it gives each next word after a word. Everything downstream is computed.
    const LOGITS = {
      the: [-2, 2.0, 1.6, -2, -2, -2, 1.0, -3], cat: [-3, -3, -3, 2.2, 1.4, -2, -3, 0.2], dog: [-3, -3, -3, 1.2, 2.0, -2, -3, 0.2],
      sat: [-2, -3, -3, -3, -3, 2.4, -3, 0.8], ran: [-2, -3, -3, -3, -3, 1.6, -3, 1.2], on: [2.6, -2, -2, -3, -3, -3, -1, -3], mat: [-2, -3, -3, -2, -2, -1, -3, 2.4]
    };
    const softmax = z => { const m = Math.max(...z), e = z.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
    W('decoding', {
      title: 'How a language model writes, one token at a time',
      symbols: [['z', 'the logits: the raw score the model gives each word of its vocabulary'],
        ['T', 'the temperature (set below): below 1 it sharpens the distribution, above 1 it flattens it'],
        ['softmax', 'turns scores into probabilities: e raised to each score, divided by the sum over all words'],
        ['probs', 'the probability of each word being next'],
        ['p', 'the top-p threshold (set below): keep the most likely words until their probabilities add up to p']],
      params: [{ key: 'T', label: 'Temperature', value: 1, options: [[0.5, '0.5'], [1, '1'], [1.5, '1.5']] }, { key: 'p', label: 'top-p', value: 0.9, options: [[0.9, '0.9'], [1, '1 (off)']] }],
      code: [
        'tokens = ["the"]',
        'while True:',
        '    logits = model(tokens)[-1]               # a score for every word',
        '    probs = softmax(logits / T)              # temperature, then probabilities',
        '    keep = smallest_set_with_mass(probs, p)  # top-p: drop the long tail',
        '    probs = renormalize(probs * keep)',
        '    nxt = rng.choice(vocab, p=probs)         # sample one token',
        '    tokens.append(nxt)',
        '    if nxt == "." or len(tokens) == 8: break'
      ],
      nodes: [
        { id: 'logits', label: ['Score every word in the', 'vocabulary (the logits)'], why: 'The model reads everything so far and outputs one score per vocabulary entry: how plausible each is as the next token.' },
        { id: 'temp', label: ['Divide by the temperature T'], why: 'Below 1, differences between scores grow and the model gets more predictable; above 1 they shrink and it gets more varied.' },
        { id: 'soft', label: ['Softmax: probabilities'], why: 'Exponentiating and normalizing turns any set of scores into a probability distribution.' },
        { id: 'topp', label: ['top-p: keep the smallest set', 'with probability ≥ p'], why: 'The long tail of unlikely tokens is where nonsense comes from. Top-p cuts it, adapting to how confident the model is.' },
        { id: 'sample', label: ['Sample one token'], why: 'Drawing at random from the kept probabilities is what makes generation vary from run to run. Greedy decoding would always take the top one.' },
        { id: 'end', label: ['End token, or too long?'], kind: 'dec', why: 'Generation stops at an end-of-sequence token or a length limit. Otherwise the new token joins the context.' },
        { id: 'done', label: ['Done: the generated text'], kind: 'end', why: 'Everything the model wrote came from repeating this loop, one token at a time.' }
      ],
      edges: [
        { from: 'logits', to: 'temp', label: 'reshape', why: 'The temperature sharpens or flattens the distribution before sampling.' },
        { from: 'temp', to: 'soft', label: 'normalize', why: 'Scores become probabilities.' },
        { from: 'soft', to: 'topp', label: 'trim the tail', why: 'Remove the unlikely tokens before choosing.' },
        { from: 'topp', to: 'sample', label: 'choose', why: 'Pick one token from what remains.' },
        { from: 'sample', to: 'end', label: 'append it', why: 'The chosen token is added to the text.' },
        { from: 'end', to: 'logits', label: 'no: next', why: 'The longer text goes back into the model for the next token.' },
        { from: 'end', to: 'done', label: 'yes: stop', why: 'An end token appeared, or the length limit was reached.' }
      ],
      run(p) {
        const T = +p.T, P = +p.p, rnd = MLA.util.seededRng(7), steps = [], toks = ['the'], chosen = [];
        for (let n = 0; n < 7; n++) {
          const z = LOGITS[toks[toks.length - 1]], zt = z.map(x => x / T), pr = softmax(zt);
          const order = pr.map((x, i) => i).sort((a, b) => pr[b] - pr[a]);
          const keep = new Set();
          let mass = 0;
          for (const i of order) { keep.add(i); mass += pr[i]; if (mass >= P - 1e-12) break; }
          const kept = pr.map((x, i) => (keep.has(i) ? x / mass : 0));
          const r = rnd();
          let acc = 0, pick = order.find(i => keep.has(i));
          for (const i of order) { if (!keep.has(i)) continue; acc += kept[i]; if (r < acc) { pick = i; break; } }
          const word = VOC[pick], text = toks.join(' ');
          if (n === 0) {
            steps.push({ node: 'logits', lines: [3], say: 'The text so far is "' + text + '". The model scores every word it knows as the next one; "cat" and "dog" score highest.', formula: 'z:  ' + VOC.map((w, i) => w + ' ' + f(z[i])).join(',  '), state: { bars: z, label: 'logits', text } });
            steps.push({ node: 'temp', lines: [4], say: 'Dividing by T = ' + T + (T < 1 ? ' stretches the scores apart: the favourites pull further ahead.' : T > 1 ? ' squeezes the scores together: unlikely words gain ground.' : ' leaves the scores unchanged.'), formula: 'z / T:  ' + VOC.map((w, i) => w + ' ' + f(zt[i])).join(',  '), state: { bars: zt, label: 'logits / T', text } });
            steps.push({ node: 'soft', lines: [4], say: 'The softmax turns them into probabilities: "cat" ' + Math.round(pr[1] * 100) + '%, "dog" ' + Math.round(pr[2] * 100) + '%, "mat" ' + Math.round(pr[6] * 100) + '%.', formula: 'probs = softmax(z / T):  ' + VOC.map((w, i) => w + ' ' + f(pr[i])).join(',  '), state: { bars: pr, label: 'probabilities', text, prob: true }, ask: 'Will the model always pick "cat" here?' });
            steps.push({ node: 'topp', lines: [5, 6], say: P >= 1 ? 'top-p is off, so every word stays in the running, however unlikely.' : 'top-p = ' + P + ' keeps the most likely words until their probabilities add up to ' + P + ': ' + [...keep].map(i => '"' + VOC[i] + '"').join(', ') + '. The rest (grey) can no longer be chosen.', formula: 'kept: ' + [...keep].map(i => VOC[i] + ' ' + f(kept[i])).join(',  '), state: { bars: kept, label: 'after top-p (renormalized)', text, prob: true, keep: [...keep] } });
            steps.push({ node: 'sample', lines: [7], say: 'A random number, ' + f(r) + ', falls in the slice of "' + word + '", so "' + word + '" is chosen. With a different random draw, a different word could follow.', formula: 'random draw ' + f(r) + ' → "' + word + '"', state: { bars: kept, label: 'after top-p (renormalized)', text, prob: true, keep: [...keep], pick } });
          } else steps.push({ node: 'sample', edge: 'topp>sample', lines: [3, 4, 5, 6, 7], say: 'After "' + text + '", the same steps give "' + word + '" (probability ' + f(kept[pick]) + ' after temperature and top-p).', formula: 'probs after T and top-p:  ' + [...keep].map(i => VOC[i] + ' ' + f(kept[i])).join(',  '), state: { bars: kept, label: 'next-word probabilities', text, prob: true, keep: [...keep], pick } });
          toks.push(word);
          chosen.push({ word, p: kept[pick] });
          const stop = word === '.' || toks.length === 8;
          steps.push({ node: 'end', lines: [8, 9], say: '"' + word + '" is appended: "' + toks.join(' ') + '". ' + (stop ? (word === '.' ? 'It is the end token, so generation stops.' : 'The length limit is reached.') : 'It is not the end token, so the loop runs again with the longer text.'), state: { bars: kept, label: 'next-word probabilities', text: toks.join(' '), prob: true, keep: [...keep], pick } });
          if (stop) break;
        }
        steps.push({ node: 'done', lines: [], say: 'The model wrote "' + toks.join(' ') + '". The random draws are the same for every setting, so changing the temperature or top-p and stepping through again shows what the settings alone do: low temperatures stick to the likeliest words (and can fall into a loop), while high temperatures without top-p let unlikely words in. The bars show the probability each chosen word had when it was drawn.', state: { text: toks.join(' '), chosen } });
        return steps;
      },
      draw(s, kit) {
        const out = [], base = 160, h = 110;
        out.push({ k: 'txt', t: 't', x: 12, y: 16, s: 'text: ' + (s.text || ''), cls: 'wp-txt' });
        if (s.bars) {
          const top = s.prob ? 1 : Math.max(...s.bars.map(Math.abs)), neg = !s.prob && s.bars.some(x => x < 0);
          const zero = neg ? base - h / 2 : base, scale = neg ? h / 2 / top : h / top;
          out.push({ k: 'ttl', t: 't', x: 12, y: 32, s: s.label, cls: 'wp-tick' });
          out.push({ k: 'base', t: 'l', x1: 8, y1: zero, x2: 292, y2: zero, cls: 'wp-axis' });
          VOC.forEach((w, i) => {
            const v = s.bars[i], x = 12 + i * 35, hh = Math.abs(v) * scale, off = s.keep && !s.keep.includes(i);
            out.push({ k: 'b' + i, t: 'r', x, y: v >= 0 ? zero - hh : zero, w: 26, h: hh, cls: off ? 'wp-bar-off' : s.pick === i ? 'wp-bar2' : 'wp-bar' });
            const lab = s.prob ? (v > 0 ? Math.round(v * 100) + '%' : '') : f(v);     // the number behind each bar
            out.push({ k: 'bv' + i, t: 't', x: x + 13, y: v >= 0 ? zero - hh - 3 : zero + hh + 10, s: lab, cls: 'wp-tick', a: 'middle' });
            out.push({ k: 'n' + i, t: 't', x: x + 13, y: 192, s: w, cls: s.pick === i ? 'wp-txt' : 'wp-tick', a: 'middle' });
          });
        }
        if (s.chosen) {                                              // the finished run: how likely each pick was
          out.push({ k: 'ttl', t: 't', x: 12, y: 32, s: 'probability of each chosen word', cls: 'wp-tick' });
          out.push({ k: 'base', t: 'l', x1: 8, y1: base, x2: 292, y2: base, cls: 'wp-axis' });
          s.chosen.forEach((c, i) => {
            const x = 12 + i * 35, hh = c.p * h;
            out.push({ k: 'b' + i, t: 'r', x, y: base - hh, w: 26, h: hh, cls: 'wp-bar2' });
            out.push({ k: 'bv' + i, t: 't', x: x + 13, y: base - hh - 3, s: Math.round(c.p * 100) + '%', cls: 'wp-tick', a: 'middle' });
            out.push({ k: 'n' + i, t: 't', x: x + 13, y: 192, s: c.word, cls: 'wp-txt', a: 'middle' });
          });
        }
        return out.map(o => Object.assign(o, { f: 1 }));
      }
    });
  })();

  // ---------------------------------------------------------------- Q-learning in a corridor
  (function () {
    const NS = 5, START = 2, A = ['left', 'right'], ALPHA = 0.5, GAMMA = 0.9, EPS = 0.3;
    const reward = s => (s === 4 ? 1 : s === 0 ? -1 : 0), terminal = s => s === 0 || s === 4;
    W('q-learning', {
      title: 'Q-learning: an agent learns a corridor from rewards',
      symbols: [['Q', 'the estimated total future reward: Q(s, a) is the value of taking action a in cell s'],
        ['s', 'the current cell'], ['a', 'the action: left or right'], ['s′', 'the cell the move leads to'],
        ['r', 'the reward for this move: +1 at the goal, −1 in the pit, 0 elsewhere'],
        ['γ', 'the discount (0.9): a reward one step later counts 90% as much'],
        ['max Q(s′, ·)', 'the value of the best action from the new cell'],
        ['α', 'the learning rate (0.5): how far each estimate moves toward its target'],
        ['ε', 'the exploration rate (0.3): the chance of a random move instead of the best known one'],
        ['←', 'replace the value on the left with the value on the right']],
      code: [
        'Q = np.zeros((5, 2))                       # value of each action in each cell',
        'for episode in range(30):',
        '    s = 2                                  # start in the middle',
        '    while s not in (0, 4):',
        '        a = rng.integers(2) if rng.random() < eps else Q[s].argmax()',
        '        s2 = s - 1 if a == 0 else s + 1        # move left or right',
        '        r = 1 if s2 == 4 else -1 if s2 == 0 else 0',
        '        target = r if s2 in (0, 4) else r + gamma * Q[s2].max()',
        '        Q[s, a] += alpha * (target - Q[s, a])  # nudge toward the target',
        '        s = s2'
      ],
      nodes: [
        { id: 'init', label: ['Start: every Q(s, a) = 0'], why: 'Q(s, a) estimates the total future reward of taking action a in cell s. At first the agent knows nothing, so all estimates are zero.' },
        { id: 'act', label: ['Choose an action:', 'ε-greedy'], why: 'Usually take the action with the highest Q (exploit), but with probability ε try a random one (explore), or it might never discover a better path.' },
        { id: 'env', label: ['Move; receive reward r', 'and the next cell s′'], why: 'The environment answers: the goal on the right pays +1, the pit on the left −1, every other move 0.' },
        { id: 'update', label: ['Nudge Q(s, a) toward', 'r + γ·max Q(s′, ·)'], why: 'The target says: what I just got, plus the discounted value of the best move from where I landed. That is how reward spreads backward from the goal.' },
        { id: 'term', label: ['Goal or pit reached?'], kind: 'dec', why: 'An episode ends in either end cell. Until then, the agent keeps moving from where it is.' },
        { id: 'more', label: ['More episodes?'], kind: 'dec', why: 'One lucky episode is not enough: values settle only after many runs.' },
        { id: 'done', label: ['Done: act greedily on Q'], kind: 'end', why: 'Following the highest Q in every cell is now the learned policy.' }
      ],
      edges: [
        { from: 'init', to: 'act', label: 'episode 1', why: 'The agent starts in the middle cell.' },
        { from: 'act', to: 'env', label: 'act', why: 'The chosen action is carried out.' },
        { from: 'env', to: 'update', label: 'learn', why: 'Every single move is a lesson, rewarded or not.' },
        { from: 'update', to: 'term', label: 's ← s′', why: 'The agent is now in the new cell.' },
        { from: 'term', to: 'act', label: 'no: move on', why: 'Not finished: choose the next action from the new cell.' },
        { from: 'term', to: 'more', label: 'yes', why: 'The episode is over.' },
        { from: 'more', to: 'act', label: 'yes: reset', why: 'Start a new episode from the middle, with everything learned so far.' },
        { from: 'more', to: 'done', label: 'no: stop', why: 'Training is over.' }
      ],
      run() {
        const steps = [], Q = Array.from({ length: NS }, () => [0, 0]), rnd = MLA.util.seededRng(3);
        const copy = () => Q.map(q => q.slice());
        const pol = () => Q.map((q, s) => (terminal(s) ? '·' : q[1] > q[0] ? '→' : q[0] > q[1] ? '←' : '?')).join(' ');
        steps.push({ node: 'init', lines: [1], say: 'Five cells: a pit (−1) on the far left, the goal (+1) on the far right. The agent starts in the middle and knows nothing: every Q value is 0.', state: { Q: copy(), s: START } });
        for (let ep = 1; ep <= 30; ep++) {
          let s = START, n = 0;
          while (!terminal(s) && n < 50) {
            n++;
            const explore = rnd() < EPS, best = Q[s][1] > Q[s][0] ? 1 : Q[s][0] > Q[s][1] ? 0 : (rnd() < 0.5 ? 0 : 1);
            const a = explore ? (rnd() < 0.5 ? 0 : 1) : best;
            const s2 = a ? s + 1 : s - 1, r = reward(s2), target = terminal(s2) ? r : r + GAMMA * Math.max(...Q[s2]);
            const old = Q[s][a], tie = Q[s][0] === Q[s][1], before = copy();   // the plot shows Q before the update
            Q[s][a] = old + ALPHA * (target - old);
            if (ep === 1) {
              steps.push({ node: 'act', lines: [5], say: 'In cell ' + s + ', the agent ' + (explore ? 'explores: a random move, ' + A[a] + '.' : tie ? 'has no preference yet, so a tie is broken at random: ' + A[a] + '.' : 'exploits its estimates: ' + A[a] + '.'), state: { Q: before, s, a } });
              steps.push({ node: 'env', lines: [6, 7], say: 'Moving ' + A[a] + ' lands in cell ' + s2 + (r ? ', ' + (r > 0 ? 'the goal: reward +1.' : 'the pit: reward −1.') : ', with reward 0.'), state: { Q: before, s: s2, a } });
              const still = r === 0 && old === 0 && Math.max(...Q[s2]) === 0;
              steps.push({ node: 'update', lines: [8, 9], say: still ? 'Q(' + s + ', ' + A[a] + ') stays at 0: the reward was 0 and nothing is known yet about cell ' + s2 + ', so the target is 0 as well.' : 'Q(' + s + ', ' + A[a] + ') moves halfway toward its target ' + f(target) + ': from ' + f(old) + ' to ' + f(Q[s][a]) + '.', formula: 'target = ' + (terminal(s2) ? 'r = ' + r + ' (the episode ends here)' : 'r + γ·max Q(s′, ·) = ' + r + ' + 0.9 × ' + f(Math.max(...Q[s2])) + ' = ' + f(target)) + '\nQ(s, a) ← Q(s, a) + α·(target − Q(s, a))\n  = ' + f(old) + ' + 0.5 × (' + f(target) + ' − ' + f(old) + ') = ' + f(Q[s][a]), state: { Q: copy(), s: s2, hl: [s, a] },
                ask: n === 1 ? 'Rewards only appear at the two ends. How can a middle cell ever learn which way is better?' : null });
              steps.push({ node: 'term', lines: [4], say: terminal(s2) ? 'Cell ' + s2 + ' ends the episode.' : 'Cell ' + s2 + ' is not an end, so the agent moves on from there.', state: { Q: copy(), s: s2 } });
            }
            s = s2;
          }
          if (ep === 1 || [2, 5, 10, 30].includes(ep)) steps.push({ node: 'more', lines: [2, 3], say: ep === 1 ? 'Episode 1 is over. Only the cell next to where it ended learned anything: reward has to spread backward one step per visit.' : 'After episode ' + ep + ', the greedy choice per cell is ' + pol() + (pol().includes('?') ? '. A ? marks a cell with no preference yet: no reward has reached it.' : '. Q(right) in cells 1 to 3: ' + [1, 2, 3].map(c => f(Q[c][1])).join(', ') + '.'), state: { Q: copy(), s: START } });
        }
        steps.push({ node: 'done', lines: [], say: 'After 30 episodes, the best action in every middle cell is to go right, toward the goal: ' + pol() + '. The values shrink with distance from the goal (by γ = 0.9 per step): the agent prefers rewards that come sooner.', state: { Q: copy(), s: START, final: true } });
        return steps;
      },
      draw(s, kit) {
        const out = [], x0 = 12, y0 = 70, cw = 54, ch = 70;
        for (let c = 0; c < NS; c++) {
          const x = x0 + c * (cw + 2), end = c === 0 ? 'wp-bar2' : c === 4 ? 'wp-area' : 'wp-cell';
          out.push({ k: 'cell' + c, t: 'r', x, y: y0, w: cw, h: ch, cls: s.hl && s.hl[0] === c ? 'wp-cell-on' : end });
          out.push({ k: 'lab' + c, t: 't', x: x + cw / 2, y: y0 - 8, s: c === 0 ? 'pit −1' : c === 4 ? 'goal +1' : 'cell ' + c, cls: 'wp-tick', a: 'middle' });
          if (c > 0 && c < 4) {
            out.push({ k: 'ql' + c, t: 't', x: x + 4, y: y0 + ch - 10, s: '← ' + f(s.Q[c][0]), cls: 'wp-tick' });
            out.push({ k: 'qr' + c, t: 't', x: x + cw - 4, y: y0 + 16, s: f(s.Q[c][1]) + ' →', cls: 'wp-tick', a: 'end' });
          }
        }
        if (s.s !== undefined) out.push({ k: 'agent', t: 'c', x: x0 + s.s * (cw + 2) + cw / 2, y: y0 + ch / 2, r: 9, cls: 'wp-q' });
        out.push({ k: 'key', t: 't', x: 12, y: 172, s: 'in each cell: Q(left) bottom left, Q(right) top right', cls: 'wp-tick' });
        return out.map(o => Object.assign(o, { f: 1 }));
      }
    });
  })();
})(typeof globalThis !== 'undefined' ? globalThis : window);
