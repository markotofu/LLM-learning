/* ML Atlas – step-through walkthroughs for classical machine learning (drawn by js/views/walks.js).
 * Every number on screen comes from running the algorithm below on its small fixed dataset. */
(function (global) {
  const MLA = global.MLA, W = MLA.walk;
  const f = v => MLA.walkKit.num(v);
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const list = a => '[' + a.slice(0, 10).map(f).join(', ') + (a.length > 10 ? ', …' : '') + ']';   // long lists show their start

  // ---------------------------------------------------------------- linear regression by gradient descent
  (function () {
    const BX = [1, 2, 3, 4, 5, 6], BY = [2.2, 3.8, 6.1, 8.2, 9.7, 12.1];
    function setup(data) {                    // the points of this run, and the exact least-squares line for comparison
      const X = data ? data.X : BX, Y = data ? data.Y : BY, mx = mean(X), my = mean(Y);
      const wOpt = X.reduce((s, x, i) => s + (x - mx) * (Y[i] - my), 0) / X.reduce((s, x) => s + (x - mx) ** 2, 0), bOpt = my - wOpt * mx;
      return { X, Y, wOpt, bOpt, loss: (w, b) => mean(X.map((x, i) => (w * x + b - Y[i]) ** 2)) };
    }
    W('linreg-gd', {
      title: 'Training a line by gradient descent, one step at a time',
      data: { kind: 'xy' },
      symbols: [['x', 'the input of each point (its horizontal position)'], ['y', 'the true value of each point'],
        ['ŷ', 'the prediction of the current line for each point'], ['w', 'the slope of the line'], ['b', 'the intercept: where the line crosses x = 0'],
        ['e', 'the error of each prediction, ŷ − y'], ['L', 'the loss: the mean of the squared errors'],
        ['∂L/∂w', 'the gradient for w: how fast the loss grows as w grows'], ['∂L/∂b', 'the gradient for b, the same idea'],
        ['η', 'the learning rate: how big a step to take (set below)'], ['←', 'replace the value on the left with the value on the right']],
      params: [{ key: 'lr', label: 'Learning rate η', value: 0.03, options: [[0.01, '0.01 (slow)'], [0.03, '0.03'], [0.06, '0.06 (near the limit)'], [0.07, '0.07 (too big)']] }],
      code: [
        'w, b = 0.0, 0.0                    # start from a flat line',
        'for it in range(1, 601):',
        '    y_hat = w * x + b              # predict every point',
        '    err = y_hat - y                # how far off each one is',
        '    dw = 2 * np.mean(err * x)      # slope of the loss in w',
        '    db = 2 * np.mean(err)          # slope of the loss in b',
        '    w -= lr * dw                   # step downhill in w',
        '    b -= lr * db                   # step downhill in b',
        '    loss = np.mean((w * x + b - y) ** 2)',
        '    if prev - loss < 1e-7: break   # stop when progress stalls'
      ],
      nodes: [
        { id: 'init', label: ['Start: w = 0, b = 0'], why: 'Gradient descent needs a starting guess. The loss of a line is a bowl with one bottom, so any start reaches the same answer; zero is simplest.' },
        { id: 'predict', label: ['Predict every point', 'ŷ = w·x + b'], why: 'The current line makes a guess for each training point. Everything after this judges those guesses.' },
        { id: 'error', label: ['Measure the errors', 'e = ŷ − y, loss = mean(e²)'], why: 'The errors say how wrong each guess is; the loss turns them into one number to make smaller.' },
        { id: 'grad', label: ['Compute the gradients', '∂L/∂w and ∂L/∂b'], why: 'A gradient is the slope of the loss along one parameter: its sign says which way is uphill, its size how steep.' },
        { id: 'update', label: ['Step downhill', 'w ← w − η·∂L/∂w'], why: 'Moving against the gradient lowers the loss. The learning rate η sets the step size: too small is slow, too big overshoots.' },
        { id: 'check', label: ['Is the loss still falling?'], kind: 'dec', why: 'Training stops when another round barely changes the loss, or when the loss rises, a sign that the steps are too big.' },
        { id: 'done', label: ['Done: the fitted line'], kind: 'end', why: 'The parameters now sit at, or very near, the bottom of the loss bowl.' }
      ],
      edges: [
        { from: 'init', to: 'predict', label: 'first guess', why: 'With parameters in hand, the model can make its first predictions.' },
        { from: 'predict', to: 'error', label: 'compare', why: 'A prediction only means something next to the true value.' },
        { from: 'error', to: 'grad', label: 'average', why: 'The loss averages over all points, so its slopes do too: every point votes on the direction.' },
        { from: 'grad', to: 'update', label: 'scale by η', why: 'The gradient gives the direction; the learning rate decides how far to go.' },
        { from: 'update', to: 'check', label: 'new loss', why: 'After every step, check whether it helped.' },
        { from: 'check', to: 'predict', label: 'yes: again', why: 'The loss is still dropping, so the line can improve further: run the loop again with the new w and b.' },
        { from: 'check', to: 'done', label: 'no: stop', why: 'Progress has stalled (converged), or the loss rose (the learning rate is too big). Either way, stop.' }
      ],
      run(p, data) {
        const { X, Y, wOpt, bOpt, loss } = setup(data), lr = +p.lr, steps = [], hist = [];
        let w = 0, b = 0, since = 1;
        const vars = (it, L) => [['iteration', it], ['w', f(w)], ['b', f(b)], ['loss', f(L)]];
        steps.push({ node: 'init', lines: [1], say: data ? data.note + ' The model starts knowing nothing: slope w = 0 and intercept b = 0, a flat line at height 0, through the middle of the data.' : 'The model starts knowing nothing: slope w = 0 and intercept b = 0, a flat line along the bottom of the plot.', vars: vars(0, loss(0, 0)), state: { w, b } });
        for (let it = 1; it <= 600; it++) {
          const yh = X.map(x => w * x + b), e = yh.map((v, i) => v - Y[i]), L = mean(e.map(v => v * v));
          const dw = 2 * mean(e.map((v, i) => v * X[i])), db = 2 * mean(e);
          const w2 = w - lr * dw, b2 = b - lr * db, L2 = loss(w2, b2);
          if (it <= 2) {
            steps.push({ node: 'predict', lines: [3], say: 'Iteration ' + it + ': the line ŷ = ' + f(w) + '·x + ' + f(b) + ' makes a prediction for each of the ' + (data ? X.length : 'six') + ' points.', formula: 'ŷ = w·x + b = ' + list(yh), vars: vars(it, L), state: { w, b } });
            steps.push({ node: 'error', lines: [4], say: 'The errors (red) show how far each prediction falls from its point. Their mean square, the loss, is ' + f(L) + '.', formula: 'e = ŷ − y = ' + list(e) + '\nL = mean(e²) = ' + f(L), vars: vars(it, L), state: { w, b, res: true },
              ask: it === 1 ? (data ? 'Look at the red errors. To lower the loss, should w go up or down?' : 'Every prediction is below its point. To lower the loss, should w go up or down?') : null });
            steps.push({ node: 'grad', lines: [5, 6], say: 'The gradients turn the errors into directions. ∂L/∂w = ' + f(dw) + ' is ' + (dw < 0 ? 'negative: raising w lowers the loss.' : 'positive: lowering w lowers the loss.'), formula: '∂L/∂w = 2·mean(e·x) = ' + f(dw) + '\n∂L/∂b = 2·mean(e) = ' + f(db), vars: vars(it, L), state: { w, b, res: true } });
            steps.push({ node: 'update', lines: [7, 8], say: 'Each parameter steps against its gradient, scaled by η = ' + lr + ': w goes from ' + f(w) + ' to ' + f(w2) + ', and b from ' + f(b) + ' to ' + f(b2) + '. The dashed line is where it was.', formula: 'w ← w − η·∂L/∂w = ' + f(w) + ' − ' + lr + ' × (' + f(dw) + ') = ' + f(w2) + '\nb ← b − η·∂L/∂b = ' + f(b) + ' − ' + lr + ' × (' + f(db) + ') = ' + f(b2), vars: [['iteration', it], ['w', f(w2)], ['b', f(b2)], ['loss', f(L2)]], state: { w: w2, b: b2, ghost: [w, b] } });
          }
          hist.push(L);
          w = w2; b = b2;
          const rising = L2 > L, flat = Math.abs(L - L2) < 1e-7, done = rising || flat || it === 600;
          if (it <= 2 || done || [5, 20, 50, 100, 200, 400].includes(it)) {
            const say = rising ? 'The loss rose from ' + f(L) + ' to ' + f(L2) + ': each step overshoots the bottom of the bowl and lands higher on the other side. Stop, and lower the learning rate.'
              : flat ? 'The loss changed by less than 0.0000001: converged after ' + it + ' iterations.'
              : it <= 2 ? 'The loss went from ' + f(L) + ' to ' + f(L2) + '. Still falling, so the loop runs again.'
              : (it === 600 ? 'The iteration budget is used up. ' : '') + 'Iterations ' + since + ' to ' + it + ' ran the same loop. The loss is now ' + f(L2) + '.';
            steps.push({ node: 'check', lines: [9, 10], say, formula: 'loss: ' + f(L) + ' → ' + f(L2), vars: [['iteration', it], ['w', f(w)], ['b', f(b)], ['loss', f(L2)]], state: { w, b, hist: hist.concat([L2]) } });
            since = it + 1;
          }
          if (done) {
            steps.push({ node: 'done', lines: [], say: rising ? 'Stopped because the loss was rising. With η = ' + lr + ' the updates grow instead of settling: the learning rate has to stay below a limit set by how steep the loss bowl is.'
              : w.toFixed(2) === wOpt.toFixed(2) && b.toFixed(2) === bOpt.toFixed(2)
                ? 'The fitted line is ŷ = ' + w.toFixed(2) + '·x + ' + b.toFixed(2) + ', the same, to two decimals, as the exact least-squares answer from this lesson\'s formula. Gradient descent got there by small steps instead of solving an equation.'
                : 'After ' + it + ' iterations the line is ŷ = ' + f(w) + '·x + ' + f(b) + ', while the exact least-squares answer is ŷ = ' + f(wOpt) + '·x + ' + f(bOpt) + '. ' + (data ? 'More iterations, or a larger learning rate, would close the gap.' : 'The slope settled quickly, but the intercept is still creeping: x runs from 1 to 6 and is not centred, which stretches the loss bowl into a long, narrow valley. A larger learning rate, or centring x, gets there much faster.'),
            vars: [['w', f(w)], ['b', f(b)], ['loss', f(loss(w, b))]], state: { w, b, hist: hist.concat([loss(w, b)]) } });
            break;
          }
        }
        return steps;
      },
      draw(s, kit, data) {
        const { X, Y } = setup(data), xd = data ? kit.span(X) : [0, 7], yd = data ? kit.span(Y.concat([0])) : [0, 14];
        const fr = kit.frame(xd, yd, data ? data.cols : ['x', 'y']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        const line = (w, b) => 'M' + sx(xd[0]) + ' ' + sy(w * xd[0] + b) + ' L' + sx(xd[1]) + ' ' + sy(w * xd[1] + b);
        if (s.ghost) out.push({ k: 'ghost', t: 'p', d: line(s.ghost[0], s.ghost[1]), cls: 'wp-line2' });
        out.push({ k: 'fit', t: 'p', d: line(s.w || 0, s.b || 0), cls: 'wp-line' });
        X.forEach((x, i) => {
          if (s.res) out.push({ k: 'r' + i, t: 'l', x1: sx(x), y1: sy(Y[i]), x2: sx(x), y2: sy(s.w * x + s.b), cls: 'wp-res' });
          out.push({ k: 'p' + i, t: 'c', x: sx(x), y: sy(Y[i]), r: 4.5, cls: 'wp-c0' });
        });
        if (s.hist && s.hist.length > 1) {                         // the loss so far, as a small inset
          const h = s.hist, mx = Math.max(...h), x0 = 200, y0 = 22, ww = 86, hh = 34;
          out.push({ k: 'ib', t: 'r', x: x0 - 4, y: y0 - 12, w: ww + 8, h: hh + 18, cls: 'wp-cell' });
          out.push({ k: 'it', t: 't', x: x0, y: y0 - 2, s: 'loss so far', cls: 'wp-tick' });
          out.push({ k: 'ic', t: 'p', d: h.map((v, i) => (i ? 'L' : 'M') + (x0 + i / (h.length - 1) * ww).toFixed(1) + ' ' + (y0 + hh - Math.min(1, v / mx) * (hh - 6)).toFixed(1)).join(' '), cls: 'wp-line' });
        }
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- k-nearest neighbours, at prediction time
  (function () {
    const BP = [[1, 1, 0], [1.5, 2, 0], [1.4, 1, 0], [2.5, 3.2, 0], [3, 2.2, 1], [3.6, 3.1, 1], [3.2, 1, 1], [4.2, 2.8, 1]];
    const NAME = ['blue', 'orange'];
    const setup = data => (data ? { PTS: data.P, Q: [0, 0] } : { PTS: BP, Q: [2.6, 2.3] });   // with your data, the new point is the centre
    W('knn', {
      title: 'How k-NN decides the class of a new point',
      data: { kind: 'class2' },
      symbols: [['d', 'the straight-line distance between the new point and a training point'], ['x', 'a training point, with coordinates x₁ and x₂'],
        ['q', 'the new point, with coordinates q₁ and q₂'], ['√', 'the square root'], ['k', 'how many of the nearest points get a vote (set below)']],
      params: [{ key: 'k', label: 'k', value: 3, options: [[1, '1'], [3, '3'], [5, '5']] }],
      code: [
        'def predict(q, X, y, k):',
        '    dists = []',
        '    for xi, yi in zip(X, y):                 # every stored example',
        '        dists.append((np.linalg.norm(q - xi), yi))',
        '    dists.sort()                             # nearest first',
        '    nearest = dists[:k]                      # the k closest',
        '    votes = Counter(label for _, label in nearest)',
        '    return votes.most_common(1)[0][0]        # majority class'
      ],
      nodes: [
        { id: 'arrive', label: ['A new point arrives'], why: 'k-NN does no work at training time: it only stores the examples. All the thinking happens now, when a point needs a label.' },
        { id: 'dist', label: ['Distance to the next', 'training point'], why: 'Closeness is the only notion of similarity k-NN has, so it measures it to every stored example. This is why features must be on comparable scales.' },
        { id: 'more', label: ['More training points?'], kind: 'dec', why: 'Plain k-NN compares with every example, so a prediction costs time in proportion to the size of the training set.' },
        { id: 'sort', label: ['Sort by distance'], why: 'Ordering the distances puts the most similar examples first.' },
        { id: 'keep', label: ['Keep the k nearest'], why: 'k is the one real setting: a small k follows local detail (and noise), a large k smooths it away.' },
        { id: 'vote', label: ['Vote: count each class'], why: 'Each neighbour backs its own class. A majority vote turns the neighbourhood into a single answer.' },
        { id: 'predict', label: ['Predict the winning class'], kind: 'end', why: 'The new point takes the class most common among its neighbours.' }
      ],
      edges: [
        { from: 'arrive', to: 'dist', label: 'compare', why: 'The new point is judged only by the company it keeps.' },
        { from: 'dist', to: 'more', label: 'store it', why: 'Each distance is kept with that example\'s label.' },
        { from: 'more', to: 'dist', label: 'yes: next', why: 'There are examples left to compare with.' },
        { from: 'more', to: 'sort', label: 'no', why: 'All distances are known; now rank them.' },
        { from: 'sort', to: 'keep', label: 'take the top k', why: 'Only the closest examples get a say.' },
        { from: 'keep', to: 'vote', label: 'count labels', why: 'The neighbours\' labels decide the answer.' },
        { from: 'vote', to: 'predict', label: 'majority', why: 'The class with the most votes wins.' }
      ],
      run(p, data) {
        const { PTS, Q } = setup(data), d = pt => Math.hypot(pt[0] - Q[0], pt[1] - Q[1]), k = +p.k, steps = [], done = [];
        steps.push({ node: 'arrive', lines: [1], say: (data ? data.note + ' The red point at (0, 0), the centre of the data,' : 'The red point at (' + Q[0] + ', ' + Q[1] + ')') + ' has no label. k-NN will label it by looking at the ' + k + ' stored example' + (k > 1 ? 's' : '') + ' closest to it.', state: { lines: [] } });
        PTS.forEach((pt, i) => {
          done.push(i);
          if (i < 3) steps.push({ node: 'dist', lines: [3, 4], say: 'Example ' + (i + 1) + ' at (' + f(pt[0]) + ', ' + f(pt[1]) + ') is ' + NAME[pt[2]] + '. Its straight-line distance to the new point is ' + f(d(pt)) + '.', formula: 'd = √((x₁ − q₁)² + (x₂ − q₂)²)\n  = √((' + f(pt[0]) + ' − ' + f(Q[0]) + ')² + (' + f(pt[1]) + ' − ' + f(Q[1]) + ')²) = ' + f(d(pt)), state: { lines: done.slice(), cur: i } });
        });
        steps.push({ node: 'dist', lines: [3, 4], say: 'The same calculation for the other ' + (PTS.length - 3) + ' examples gives every distance.', formula: PTS.slice(0, 12).map((pt, i) => 'd' + (i + 1) + ' = ' + f(d(pt))).join(',  ') + (PTS.length > 12 ? ',  …' : ''), state: { lines: done.slice() } });
        steps.push({ node: 'more', lines: [3], say: 'All ' + PTS.length + ' examples have been measured, so the loop ends.', state: { lines: done.slice() } });
        const order = PTS.map((pt, i) => ({ i, d: d(pt), c: pt[2] })).sort((a, b) => a.d - b.d);
        steps.push({ node: 'sort', lines: [5], say: 'Sorted from nearest to farthest: ' + order.slice(0, 8).map(o => NAME[o.c] + ' ' + f(o.d)).join(', ') + (order.length > 8 ? ', and so on' : '') + '.', state: { lines: done.slice(), order: order.map(o => o.i) } });
        const near = order.slice(0, k);
        steps.push({ node: 'keep', lines: [6], say: 'The ' + k + ' nearest lie inside the circle of radius ' + f(near[k - 1].d) + '. Everything outside it is ignored.', state: { lines: near.map(o => o.i), ring: near.map(o => o.i), radius: near[k - 1].d } });
        const votes = [0, 0];
        near.forEach(o => { votes[o.c]++; });
        const win = votes[1] > votes[0] ? 1 : 0;
        steps.push({ node: 'vote', lines: [7], say: 'The votes: ' + votes[0] + ' blue and ' + votes[1] + ' orange.', formula: 'votes = {blue: ' + votes[0] + ', orange: ' + votes[1] + '}', ask: 'Which class wins, and would it change with a different k? Try the k setting below afterwards.', state: { lines: near.map(o => o.i), ring: near.map(o => o.i), radius: near[k - 1].d } });
        const byK = [1, 3, 5].map(kk => { const v = [0, 0]; order.slice(0, kk).forEach(o => { v[o.c]++; }); return NAME[v[1] > v[0] ? 1 : 0]; });
        const agree = byK.every(n => n === byK[0]);
        steps.push({ node: 'predict', lines: [8], say: 'The new point is labelled ' + NAME[win] + ' (' + votes[win] + ' of ' + k + ' votes). For comparison, k = 1 gives ' + byK[0] + ', k = 3 gives ' + byK[1] + ' and k = 5 gives ' + byK[2] + (agree ? ': here every k agrees.' : ': this point sits near the boundary between the classes, where the choice of k decides the answer.'), state: { lines: near.map(o => o.i), ring: near.map(o => o.i), radius: near[k - 1].d, label: win } });
        return steps;
      },
      draw(s, kit, data) {
        const { PTS, Q } = setup(data);
        const xd = data ? kit.span(PTS.map(q => q[0]).concat([Q[0]])) : [0.5, 4.7], yd = data ? kit.span(PTS.map(q => q[1]).concat([Q[1]])) : [0.5, 3.7];
        const fr = kit.frame(xd, yd, data ? data.cols.slice(0, 2) : ['x₁', 'x₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        if (s.radius) {
          const rx = s.radius * (sx(1) - sx(0)), ry = s.radius * (sy(0) - sy(1));
          out.push({ k: 'ring', t: 'p', d: 'M' + (sx(Q[0]) - rx) + ' ' + sy(Q[1]) + ' a' + rx + ' ' + ry + ' 0 1 0 ' + (2 * rx) + ' 0 a' + rx + ' ' + ry + ' 0 1 0 ' + (-2 * rx) + ' 0', cls: 'wp-faint' });
        }
        (s.lines || []).forEach(i => out.push({ k: 'd' + i, t: 'l', x1: sx(Q[0]), y1: sy(Q[1]), x2: sx(PTS[i][0]), y2: sy(PTS[i][1]), cls: s.cur === i ? 'wp-res' : 'wp-faint' }));
        PTS.forEach((pt, i) => {
          out.push({ k: 'p' + i, t: 'c', x: sx(pt[0]), y: sy(pt[1]), r: 5, cls: pt[2] ? 'wp-c1' : 'wp-c0' });
          if (s.ring && s.ring.includes(i)) out.push({ k: 'h' + i, t: 'c', x: sx(pt[0]), y: sy(pt[1]), r: 8.5, cls: 'wp-hl' });
        });
        out.push({ k: 'q', t: 'c', x: sx(Q[0]), y: sy(Q[1]), r: 6, cls: s.label === undefined ? 'wp-q' : (s.label ? 'wp-c1' : 'wp-c0') });
        if (s.label !== undefined) out.push({ k: 'qh', t: 'c', x: sx(Q[0]), y: sy(Q[1]), r: 10, cls: 'wp-hl' });
        return out;
      }
    });
  })();
  // ---------------------------------------------------------------- logistic regression by gradient descent
  (function () {
    const BD = [[1, 1, 0], [1.5, 2, 0], [2, 1, 0], [2.5, 2.5, 0], [3, 3.5, 1], [3.5, 2.5, 1], [4, 4, 1], [4.5, 3, 1]];
    const sig = z => 1 / (1 + Math.exp(-z));
    const setup = data => {                   // the points, and the blue point followed in detail
      const D = data ? data.P : BD, H = data ? Math.max(0, D.findIndex(d => d[2] === 0)) : 3;
      return { D, H, h: D[H] };
    };
    W('logreg-gd', {
      title: 'Training logistic regression, one step at a time',
      data: { kind: 'class2' },
      symbols: [['x', 'the two features of a point, (x₁, x₂)'], ['y', 'the true class of a point: 1 for orange, 0 for blue'],
        ['w', 'the weights, one per feature'], ['b', 'the bias'], ['z', 'the score of a point: positive on the orange side of the boundary, negative on the blue side'],
        ['σ', 'the sigmoid, which squashes any score into a probability between 0 and 1'], ['e', 'Euler\'s number, about 2.718'],
        ['p', 'the predicted probability that the point is orange'], ['log', 'the natural logarithm'],
        ['∇w', 'the gradient for w: one slope per weight, saying how the loss changes as that weight grows'], ['∂L/∂b', 'the slope of the loss L with respect to b'],
        ['η', 'the learning rate: how big a step to take (set below)'], ['←', 'replace the value on the left with the value on the right']],
      params: [{ key: 'lr', label: 'Learning rate η', value: 0.5, options: [[0.1, '0.1'], [0.5, '0.5'], [2, '2']] }],
      code: [
        'w, b = np.zeros(2), 0.0',
        'for it in range(1, 501):',
        '    z = X @ w + b                        # a score for every point',
        '    p = 1 / (1 + np.exp(-z))             # scores into probabilities',
        '    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))',
        '    gw = X.T @ (p - y) / len(y)          # how each weight should move',
        '    gb = np.mean(p - y)',
        '    w -= lr * gw',
        '    b -= lr * gb',
        '    if np.abs(gw).max() < 0.01: break    # nearly flat: stop'
      ],
      nodes: [
        { id: 'init', label: ['Start: w = (0, 0), b = 0'], why: 'With every weight at zero, every point scores 0 and gets probability 0.5: the model starts undecided.' },
        { id: 'score', label: ['Score each point', 'z = w·x + b'], why: 'A weighted sum of the features. Positive scores lean toward class 1, negative toward class 0; the boundary is where z = 0.' },
        { id: 'prob', label: ['Turn scores into probabilities', 'p = σ(z) = 1 / (1 + e^(−z))'], why: 'The sigmoid squashes any score into a probability between 0 and 1, so the model can say how sure it is.' },
        { id: 'loss', label: ['Log loss: the mean of', '−log p(true class)'], why: 'Log loss punishes confident mistakes hardest, and it is smooth, so its gradient always points somewhere useful.' },
        { id: 'grad', label: ['Gradients', '∇w = mean((p − y)·x)'], why: 'Each point pulls on the weights in proportion to its error p − y: a point the model gets badly wrong pulls hardest.' },
        { id: 'update', label: ['Step downhill', 'w ← w − η·∇w'], why: 'The same downhill step as in any gradient descent. The boundary turns and shifts toward separating the classes.' },
        { id: 'check', label: ['Still improving?'], kind: 'dec', why: 'Stop when the gradient is nearly zero, or when the iteration budget runs out.' },
        { id: 'done', label: ['Done: a decision boundary'], kind: 'end', why: 'The boundary z = 0 splits the plane into the two classes, and the probabilities say how sure the model is on each side.' }
      ],
      edges: [
        { from: 'init', to: 'score', label: 'first guess', why: 'Scoring needs weights. Zeros are a fair start because the log loss has a single bowl-shaped valley.' },
        { from: 'score', to: 'prob', label: 'squash', why: 'Scores are unbounded; probabilities are what the loss can judge.' },
        { from: 'prob', to: 'loss', label: 'judge', why: 'Each probability is compared with the true class.' },
        { from: 'loss', to: 'grad', label: 'differentiate', why: 'For the sigmoid with log loss, the gradient simplifies to the error times the input.' },
        { from: 'grad', to: 'update', label: 'scale by η', why: 'Direction from the gradient, distance from the learning rate.' },
        { from: 'update', to: 'check', label: 'new weights', why: 'After every step, check whether it is worth continuing.' },
        { from: 'check', to: 'score', label: 'yes: again', why: 'The gradient is not yet flat, so the boundary can still improve.' },
        { from: 'check', to: 'done', label: 'no: stop', why: 'The gradient is nearly zero, or the budget is used up.' }
      ],
      run(p, data) {
        const { D, H, h } = setup(data), n = D.length, lr = +p.lr, steps = [];
        let w = [0, 0], b = 0, since = 1;
        const ev = () => {
          const z = D.map(d => w[0] * d[0] + w[1] * d[1] + b), pr = z.map(sig);
          const L = -mean(D.map((d, i) => Math.log(Math.max(d[2] ? pr[i] : 1 - pr[i], 1e-12))));
          return { z, pr, L, gw: [0, 1].map(j => mean(D.map((d, i) => (pr[i] - d[2]) * d[j]))), gb: mean(D.map((d, i) => pr[i] - d[2])) };
        };
        const vec = v => '(' + f(v[0]) + ', ' + f(v[1]) + ')';
        const vars = (it, L) => [['iteration', it], ['w', vec(w)], ['b', f(b)], ['loss', f(L)]];
        const st = extra => Object.assign({ w: w.slice(), b }, extra || {});
        steps.push({ node: 'init', lines: [1], say: (data ? data.note + ' ' : '') + 'All weights start at zero. Every point scores 0, so every probability is 0.5 and there is no boundary yet.', vars: vars(0, Math.log(2)), state: st() });
        for (let it = 1; it <= 500; it++) {
          const e = ev();
          if (it <= 2) {
            steps.push({ node: 'score', lines: [3], say: 'Iteration ' + it + '. The ringed point at (' + f(h[0]) + ', ' + f(h[1]) + ') is blue, class 0. With the current weights its score is ' + f(e.z[H]) + '.', formula: 'z = w·x + b = ' + f(w[0]) + '·' + f(h[0]) + ' + ' + f(w[1]) + '·' + f(h[1]) + ' + ' + f(b) + ' = ' + f(e.z[H]), vars: vars(it, e.L), state: st({ hl: true }) });
            steps.push({ node: 'prob', lines: [4], say: 'The sigmoid turns that score into p = ' + f(e.pr[H]) + ': a ' + Math.round(e.pr[H] * 100) + '% chance of orange for a point that is really blue.', formula: 'p = σ(z) = 1 / (1 + e^(−z))\n  = 1 / (1 + e^(' + f(-e.z[H]) + ')) = ' + f(e.pr[H]), vars: vars(it, e.L), state: st({ hl: true }) });
            steps.push({ node: 'loss', lines: [5], say: 'Averaged over all ' + (data ? n : 'eight') + ' points, the log loss is ' + f(e.L) + (it === 1 ? '. That is ln 2, the cost of answering 0.5 for everything.' : '.'), formula: 'loss = −mean(y·log p + (1 − y)·log(1 − p)) = ' + f(e.L), vars: vars(it, e.L), state: st({ hl: true }),
              ask: it === 1 ? (data ? 'Where do the orange points sit compared with the blue ones? Should each weight grow or shrink?' : 'The orange points sit up and to the right. Should the two weights grow or shrink?') : null });
            steps.push({ node: 'grad', lines: [6, 7], say: 'The gradient is ∇w = ' + vec(e.gw) + '. ' + (e.gw[0] < 0 && e.gw[1] < 0 ? 'Both parts are negative, so growing both weights lowers the loss: the orange points need higher scores.' : 'Its signs say which way each weight should move to lower the loss.'), formula: '∇w = mean((p − y)·x) = ' + vec(e.gw) + '\n∂L/∂b = mean(p − y) = ' + f(e.gb), vars: vars(it, e.L), state: st() });
            const w2 = [w[0] - lr * e.gw[0], w[1] - lr * e.gw[1]], b2 = b - lr * e.gb;
            steps.push({ node: 'update', lines: [8, 9], say: 'The weights move against the gradient, scaled by η = ' + lr + '. ' + (it === 1 ? 'Now a boundary appears: the line where z = 0. The dashed lines are where p = 0.2 and p = 0.8.' : 'The boundary turns and shifts.'), formula: 'w ← w − η·∇w = ' + vec(w) + ' − ' + lr + ' × ' + vec(e.gw) + ' = ' + vec(w2) + '\nb ← b − η·∂L/∂b = ' + f(b) + ' − ' + lr + ' × (' + f(e.gb) + ') = ' + f(b2), vars: [['iteration', it], ['w', vec(w2)], ['b', f(b2)]], state: { w: w2, b: b2 } });
          }
          w = [w[0] - lr * e.gw[0], w[1] - lr * e.gw[1]]; b -= lr * e.gb;
          const e2 = ev(), flat = Math.max(Math.abs(e2.gw[0]), Math.abs(e2.gw[1])) < 0.01, done = flat || it === 500;
          if (it <= 2 || done || [5, 20, 50, 100, 200].includes(it)) {
            steps.push({ node: 'check', lines: [10], say: flat ? 'The gradient is nearly zero: the boundary has settled after ' + it + ' iterations.' : it <= 2 ? 'The loss fell from ' + f(e.L) + ' to ' + f(e2.L) + ' and the gradient is not flat yet, so the loop runs again.' : 'Iterations ' + since + ' to ' + it + ' repeated the loop. The loss is now ' + f(e2.L) + '.', vars: vars(it, e2.L), state: st() });
            since = it + 1;
          }
          if (done) {
            const right = D.filter((d, i) => (e2.pr[i] > 0.5 ? 1 : 0) === d[2]).length;
            steps.push({ node: 'done', lines: [], say: 'The boundary classifies ' + right + ' of ' + n + ' points correctly' + (flat ? '.' : right === n ? ', and the loss is still creeping down: these classes can be separated perfectly, so the weights could grow forever to make the probabilities ever more extreme. Regularization (the C setting in scikit-learn) is what stops that in practice.' : ', and after 500 iterations the loss is still creeping down.'), vars: vars(it, e2.L), state: st({ final: true }) });
            break;
          }
        }
        return steps;
      },
      draw(s, kit, data) {
        const { D, h } = setup(data), xd = data ? kit.span(D.map(d => d[0])) : [0.5, 5], yd = data ? kit.span(D.map(d => d[1])) : [0.5, 4.5];
        const fr = kit.frame(xd, yd, data ? data.cols.slice(0, 2) : ['x₁', 'x₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy, w = s.w || [0, 0], b = s.b || 0;
        const at = z => {                                   // the line where w·x + b = z, across the plot
          if (Math.abs(w[1]) > Math.abs(w[0])) return 'M' + sx(xd[0]) + ' ' + sy((z - b - w[0] * xd[0]) / w[1]) + ' L' + sx(xd[1]) + ' ' + sy((z - b - w[0] * xd[1]) / w[1]);
          if (Math.abs(w[0]) > 1e-9) return 'M' + sx((z - b - w[1] * yd[0]) / w[0]) + ' ' + sy(yd[0]) + ' L' + sx((z - b - w[1] * yd[1]) / w[0]) + ' ' + sy(yd[1]);
          return null;
        };
        const lo = at(-1.386), mid = at(0), hi = at(1.386);
        if (lo) out.push({ k: 'lo', t: 'p', d: lo, cls: 'wp-line2' });
        if (hi) out.push({ k: 'hi', t: 'p', d: hi, cls: 'wp-line2' });
        if (mid) out.push({ k: 'mid', t: 'p', d: mid, cls: 'wp-line' });
        D.forEach((d, i) => out.push({ k: 'p' + i, t: 'c', x: sx(d[0]), y: sy(d[1]), r: 5, cls: d[2] ? 'wp-c1' : 'wp-c0' }));
        if (s.hl) out.push({ k: 'h', t: 'c', x: sx(h[0]), y: sy(h[1]), r: 9, cls: 'wp-hl' });
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- the perceptron learning rule
  (function () {
    const D = [[1, 3, 1], [2, 1, -1], [2, 3.5, 1], [3, 1.5, -1], [3, 4.5, 1], [4, 3, -1]];
    const cls = y => (y > 0 ? 'orange (+1)' : 'blue (−1)');
    W('perceptron', {
      title: 'The perceptron learning rule, point by point',
      symbols: [['x', 'the point, (x₁, x₂)'], ['y', 'the true class of the point, +1 or −1'], ['w', 'the weights: they set the direction of the line'],
        ['b', 'the bias: it shifts the line'], ['s', 'the score w·x + b: its sign is the predicted class'],
        ['←', 'replace the value on the left with the value on the right'], ['epoch', 'one full pass over all the points']],
      code: [
        'w, b = np.zeros(2), 0.0',
        'for epoch in range(100):',
        '    mistakes = 0',
        '    for xi, yi in zip(X, y):               # one point at a time',
        '        s = w @ xi + b                       # which side is it on?',
        '        if yi * s <= 0:                      # wrong side (or on the line)',
        '            w += yi * xi                     # tilt the line toward it',
        '            b += yi',
        '            mistakes += 1',
        '    if mistakes == 0: break                  # a clean pass: done'
      ],
      nodes: [
        { id: 'init', label: ['Start: w = (0, 0), b = 0'], why: 'The perceptron starts with no line at all and learns only from its mistakes.' },
        { id: 'score', label: ['Take the next point:', 'score s = w·x + b'], why: 'The sign of the score says which side of the current line the point is on.' },
        { id: 'wrong', label: ['Wrong side? (y·s ≤ 0)'], kind: 'dec', why: 'y·s is positive exactly when the point is on its own side. Zero counts as wrong: a point on the line is not yet classified.' },
        { id: 'update', label: ['Move the line toward it:', 'w ← w + y·x, b ← b + y'], why: 'Adding y·x to w raises this point\'s score if it should be positive and lowers it if it should be negative, so the line tilts toward fixing it.' },
        { id: 'epoch', label: ['Last point of the epoch?'], kind: 'dec', why: 'An epoch is one pass over all the training points, in a fixed order.' },
        { id: 'any', label: ['Any mistakes this epoch?'], kind: 'dec', why: 'If a whole pass needs no update, every point is on its correct side and the line is final.' },
        { id: 'done', label: ['Done: every point', 'is on its side'], kind: 'end', why: 'For data a straight line can separate, the perceptron is guaranteed to get here in a finite number of updates.' }
      ],
      edges: [
        { from: 'init', to: 'score', label: 'first point', why: 'Learning happens one example at a time.' },
        { from: 'score', to: 'wrong', label: 'check', why: 'Compare the predicted side with the true label.' },
        { from: 'wrong', to: 'update', label: 'yes: fix', why: 'The perceptron only learns from mistakes.' },
        { from: 'wrong', to: 'epoch', label: 'no: keep w', why: 'A correct point changes nothing.' },
        { from: 'update', to: 'epoch', label: 'then', why: 'After the fix, move on.' },
        { from: 'epoch', to: 'score', label: 'no: next', why: 'More points remain in this pass.' },
        { from: 'epoch', to: 'any', label: 'yes', why: 'The pass is over; count its mistakes.' },
        { from: 'any', to: 'score', label: 'yes: again', why: 'Some points were wrong, so another pass is needed.' },
        { from: 'any', to: 'done', label: 'no: stop', why: 'A clean pass: training is finished.' }
      ],
      run() {
        const steps = [];
        let w = [0, 0], b = 0;
        const vec = v => '(' + f(v[0]) + ', ' + f(v[1]) + ')';
        const st = extra => Object.assign({ w: w.slice(), b }, extra || {});
        const shown = (ww, bb) => {                                // does the line w·x + b = 0 cross the plotted area?
          const v = [[0, 0], [5, 0], [5, 5.5], [0, 5.5]].map(p => ww[0] * p[0] + ww[1] * p[1] + bb);
          return v.some(x => x > 0) && v.some(x => x < 0);
        };
        steps.push({ node: 'init', lines: [1], say: 'The weights start at zero, so there is no line yet. Six points, three of each class, are visited in the same order every epoch.', vars: [['w', vec(w)], ['b', f(b)]], state: st() });
        for (let epoch = 1; epoch <= 20; epoch++) {
          const wrongs = [];
          D.forEach((d, i) => {
            const s = w[0] * d[0] + w[1] * d[1] + b, bad = d[2] * s <= 0;
            if (epoch === 1) {
              steps.push({ node: 'score', lines: [4, 5], say: 'Point ' + (i + 1) + ' at (' + d[0] + ', ' + d[1] + ') is ' + cls(d[2]) + '. Its score is ' + f(s) + '.', formula: 's = w·x + b = ' + f(w[0]) + '·' + d[0] + ' + ' + f(w[1]) + '·' + d[1] + ' + ' + f(b) + ' = ' + f(s), vars: [['epoch', epoch], ['w', vec(w)], ['b', f(b)]], state: st({ cur: i }),
                edge: i >= 2 ? 'epoch>score' : undefined,            // arrived through "Last point of the epoch?"
                ask: i === 2 ? 'Is point 3 on the right side of the current line?' : null });
              steps.push({ node: 'wrong', lines: [6], say: 'y·s = ' + f(d[2] * s) + (bad ? (s === 0 ? (!w[0] && !w[1] && !b ? ': with every weight at zero, every point scores 0, and a score of 0 counts as a mistake.' : ': the point is exactly on the line, which counts as a mistake.') : ': the point is on the wrong side, a mistake.') : ': positive, so the point is on its correct side and nothing changes.'), vars: [['epoch', epoch], ['w', vec(w)], ['b', f(b)]], state: st({ cur: i }) });
            }
            if (bad) {
              const w2 = [w[0] + d[2] * d[0], w[1] + d[2] * d[1]], b2 = b + d[2];
              const first = !w[0] && !w[1] && !b, side = (ww, bb) => (ww[0] * 2.5 + ww[1] * 2.75 + bb > 0 ? 'orange' : 'blue');
              const say = (first ? 'The first line: w = ' + vec(w2) + ' and b = ' + f(b2) + '. The shading shows what it predicts: orange where the score is positive, blue where it is negative.'
                : 'The line moves toward point ' + (i + 1) + ': w goes from ' + vec(w) + ' to ' + vec(w2) + (shown(w, b) ? ', and the dashed line shows where it was.' : '.'))
                + (shown(w2, b2) ? '' : ' The line itself lies outside the plot, so it calls every point shown ' + side(w2, b2) + '.');
              if (epoch === 1) steps.push({ node: 'update', lines: [7, 8, 9], say, formula: 'w ← w + y·x = ' + vec(w) + ' + (' + d[2] + ') × (' + d[0] + ', ' + d[1] + ') = ' + vec(w2) + '\nb ← b + y = ' + f(b) + ' + (' + d[2] + ') = ' + f(b2), vars: [['epoch', epoch], ['w', vec(w2)], ['b', f(b2)]], state: { w: w2, b: b2, cur: i, ghost: [w.slice(), b] } });
              w = w2; b = b2; wrongs.push(i + 1);
            }
            if (epoch === 1 && (i === 0 || i === D.length - 1)) steps.push({ node: 'epoch', lines: [4], say: i === 0 ? 'Point 1 was not the last of the epoch, so the loop moves on to point 2. The same check follows every point.' : 'Point ' + D.length + ' was the last one, so this pass over the data is over.', vars: [['epoch', epoch], ['w', vec(w)], ['b', f(b)]], state: st({ cur: i }) });
          });
          steps.push({ node: 'any', lines: [10], say: epoch === 1 ? 'Epoch 1 is over, with ' + wrongs.length + ' mistakes (points ' + wrongs.join(', ') + '). The line moved after each one, so another pass is needed.'
            : wrongs.length ? 'Epoch ' + epoch + ': ' + wrongs.length + ' mistake' + (wrongs.length > 1 ? 's' : '') + ' (' + (wrongs.length > 1 ? 'points ' : 'point ') + wrongs.join(', ') + '). After the update' + (wrongs.length > 1 ? 's' : '') + ', w = ' + vec(w) + ' and b = ' + f(b) + '.'
            : 'Epoch ' + epoch + ': a full pass with no mistakes.', vars: [['epoch', epoch], ['mistakes', wrongs.length], ['w', vec(w)], ['b', f(b)]], state: st() });
          if (!wrongs.length) {
            steps.push({ node: 'done', lines: [], say: 'Training stops after ' + epoch + ' epochs: the line ' + f(w[0]) + '·x₁ + ' + f(w[1]) + '·x₂ + ' + f(b) + ' = 0 puts every point on its own side. It is a separating line, not the best one: the perceptron stops at the first line that works.', vars: [['w', vec(w)], ['b', f(b)]], state: st() });
            break;
          }
        }
        return steps;
      },
      draw(s, kit) {
        const fr = kit.frame([0, 5], [0, 5.5], ['x₁', 'x₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        const line = (w, b) => Math.abs(w[1]) > 1e-9 ? 'M' + sx(-1) + ' ' + sy((-b + w[0]) / w[1]) + ' L' + sx(6) + ' ' + sy((-b - 6 * w[0]) / w[1])
          : Math.abs(w[0]) > 1e-9 ? 'M' + sx(-b / w[0]) + ' ' + sy(-1) + ' L' + sx(-b / w[0]) + ' ' + sy(7) : null;
        // Shade each side of the line in the colour of the class it predicts (visible even when the line is off the plot)
        const half = (w, b, sign) => {
          const val = p => sign * (w[0] * p[0] + w[1] * p[1] + b), box = [[0, 0], [5, 0], [5, 5.5], [0, 5.5]], poly = [];
          box.forEach((p, i) => {
            const q = box[(i + 1) % 4], a = val(p), c = val(q);
            if (a >= 0) poly.push(p);
            if ((a >= 0) !== (c >= 0)) { const t = a / (a - c); poly.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
          });
          return poly.length > 2 ? 'M' + poly.map(p => sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join(' L') + ' Z' : null;
        };
        if (s.w && (s.w[0] || s.w[1])) {
          const pos = half(s.w, s.b, 1), neg = half(s.w, s.b, -1);
          if (pos) out.push({ k: 'hp', t: 'p', d: pos, cls: 'wp-area2' });
          if (neg) out.push({ k: 'hn', t: 'p', d: neg, cls: 'wp-area' });
        }
        if (s.ghost) { const g = line(s.ghost[0], s.ghost[1]); if (g) out.push({ k: 'g', t: 'p', d: g, cls: 'wp-line2' }); }
        const l = line(s.w || [0, 0], s.b || 0);
        if (l) out.push({ k: 'l', t: 'p', d: l, cls: 'wp-line' });
        D.forEach((d, i) => out.push({ k: 'p' + i, t: 'c', x: sx(d[0]), y: sy(d[1]), r: 5, cls: d[2] > 0 ? 'wp-c1' : 'wp-c0' }));
        D.forEach((d, i) => out.push({ k: 'n' + i, t: 't', x: sx(d[0]) + 8, y: sy(d[1]) - 6, s: String(i + 1), cls: 'wp-tick' }));
        if (s.cur !== undefined) out.push({ k: 'h', t: 'c', x: sx(D[s.cur][0]), y: sy(D[s.cur][1]), r: 9, cls: 'wp-hl' });
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- decision tree: the greedy split search
  (function () {
    const D = [[1, 1, 0], [1.5, 3, 0], [2, 2, 0], [2.5, 1, 0], [3, 3.5, 1], [3.5, 2, 1], [4, 3, 1], [2.2, 3.6, 1]];
    const NAME = ['blue', 'orange'], FEAT = ['x₁', 'x₂'];
    const gini = c => { const n = c[0] + c[1]; return n ? 1 - (c[0] / n) ** 2 - (c[1] / n) ** 2 : 0; };
    const counts = pts => [pts.filter(p => !p[2]).length, pts.filter(p => p[2]).length];
    function candidates(pts) {
      const out = [];
      [0, 1].forEach(j => {
        const v = [...new Set(pts.map(p => p[j]))].sort((a, b) => a - b);
        for (let k = 0; k + 1 < v.length; k++) {
          const t = (v[k] + v[k + 1]) / 2, L = pts.filter(p => p[j] < t), R = pts.filter(p => p[j] >= t), cl = counts(L), cr = counts(R);
          out.push({ j, t, cl, cr, w: (L.length * gini(cl) + R.length * gini(cr)) / pts.length });
        }
      });
      return out;
    }
    W('tree-split', {
      title: 'How a decision tree chooses its splits',
      symbols: [['G', 'the Gini impurity of a group of points: 1 − (share of blue)² − (share of orange)². It is 0 when the group holds one class and 0.5 for an even mix'],
        ['n', 'a number of points: nₗ left of the cut, nᵣ right of it, n in the whole node'], ['Gini', 'the Gini impurity G of the current node'],
        ['x₁', 'the first feature (horizontal axis)'], ['x₂', 'the second feature (vertical axis)']],
      code: [
        'def best_split(X, y):',
        '    best = None',
        '    for j in range(X.shape[1]):                  # every feature',
        '        for t in midpoints(np.unique(X[:, j])):  # every threshold',
        '            left = X[:, j] < t',
        '            score = weighted_gini(y[left], y[~left])',
        '            if best is None or score < best[0]:',
        '                best = (score, j, t)',
        '    return best',
        '',
        'def grow(X, y, depth):',
        '    if gini(y) == 0 or depth == max_depth: return leaf(y)',
        '    _, j, t = best_split(X, y)',
        '    left = X[:, j] < t',
        '    return node(j, t, grow(X[left], y[left], depth + 1), grow(X[~left], y[~left], depth + 1))'
      ],
      nodes: [
        { id: 'node', label: ['Take a node and its points'], why: 'A tree grows from the top: the root holds every training point, and each split hands its points down to two children.' },
        { id: 'cand', label: ['Next candidate split:', 'feature < threshold'], why: 'Candidates are midpoints between neighbouring values of each feature. Any other threshold would split the training points the same way as one of these.' },
        { id: 'score', label: ['Score it: weighted Gini', 'of the two sides'], why: 'Gini impurity measures how mixed a side is: 0 is pure, 0.5 is a 50/50 mix. Weighting by size rewards splits that make large, pure groups.' },
        { id: 'more', label: ['More candidates?'], kind: 'dec', why: 'The search is greedy and exhaustive: every feature and every threshold is tried before choosing.' },
        { id: 'split', label: ['Split on the best one'], why: 'The best candidate has the lowest weighted impurity: the biggest drop from the parent node.' },
        { id: 'stop', label: ['Each child: pure, or too deep?'], kind: 'dec', why: 'Growth stops at pure nodes, or at limits such as max_depth that keep the tree from memorizing noise.' },
        { id: 'leaf', label: ['Leaf: predict the majority'], kind: 'end', why: 'A leaf answers with the most common class among the training points that reached it.' }
      ],
      edges: [
        { from: 'node', to: 'cand', label: 'search', why: 'Every possible way to cut this node is about to be tried.' },
        { from: 'cand', to: 'score', label: 'cut the points', why: 'The candidate splits the node\'s points into a left and a right group.' },
        { from: 'score', to: 'more', label: 'keep the best', why: 'Only the lowest score seen so far is remembered.' },
        { from: 'more', to: 'cand', label: 'yes: next', why: 'Some candidates have not been tried yet.' },
        { from: 'more', to: 'split', label: 'no', why: 'Every candidate has been scored.' },
        { from: 'split', to: 'stop', label: 'each child', why: 'Each child is a smaller version of the same problem.' },
        { from: 'stop', to: 'node', label: 'no: grow it', why: 'A mixed child gets its own split search: the tree is built recursively.' },
        { from: 'stop', to: 'leaf', label: 'yes', why: 'Nothing is left to separate, or the depth limit is reached.' }
      ],
      run() {
        const steps = [], all = D, c0 = counts(all);
        steps.push({ node: 'node', lines: [11, 12], say: 'The root holds all 8 points: ' + c0[0] + ' blue and ' + c0[1] + ' orange. Its Gini impurity is ' + f(gini(c0)) + ', a perfect mix.', vars: [['node', 'root'], ['Gini', f(gini(c0))]], state: { region: null } });
        const cands = candidates(all);
        let best = null;
        cands.forEach((c, k) => {
          const better = !best || c.w < best.w;
          if (better) best = c;
          if (k === 0) steps.push({ node: 'cand', lines: [3, 4], say: 'The first candidate cuts at ' + FEAT[c.j] + ' < ' + f(c.t) + ', halfway between the two smallest ' + FEAT[c.j] + ' values.', state: { cand: c } });
          steps.push({ node: 'score', lines: [5, 6, 7, 8], say: FEAT[c.j] + ' < ' + f(c.t) + ': the left side gets ' + c.cl[0] + ' blue and ' + c.cl[1] + ' orange, the right ' + c.cr[0] + ' blue and ' + c.cr[1] + ' orange. Weighted Gini ' + f(c.w) + (better ? ', the best so far.' : ', not better than ' + f(best.w) + '.'),
            formula: 'G = (nₗ·Gₗ + nᵣ·Gᵣ) / n\n  = (' + (c.cl[0] + c.cl[1]) + ' × ' + f(gini(c.cl)) + ' + ' + (c.cr[0] + c.cr[1]) + ' × ' + f(gini(c.cr)) + ') / 8 = ' + f(c.w), vars: [['candidate', (k + 1) + ' of ' + cands.length], ['best so far', FEAT[best.j] + ' < ' + f(best.t)]], state: { cand: c, best },
            ask: k === 2 ? 'Can any split of these points reach a weighted Gini of 0?' : null });
        });
        steps.push({ node: 'more', lines: [3, 4], say: 'All ' + cands.length + ' candidates are scored. The winner is ' + FEAT[best.j] + ' < ' + f(best.t) + ' with weighted Gini ' + f(best.w) + '.', state: { best } });
        steps.push({ node: 'split', lines: [13, 14], say: 'The root splits on ' + FEAT[best.j] + ' < ' + f(best.t) + '. Left: ' + best.cl[0] + ' blue, ' + best.cl[1] + ' orange. Right: ' + best.cr[0] + ' blue, ' + best.cr[1] + ' orange.', state: { best, split: true } });
        const kids = [{ name: 'left', pts: D.filter(p => p[best.j] < best.t) }, { name: 'right', pts: D.filter(p => p[best.j] >= best.t) }];
        const pure = kids.map(k => gini(counts(k.pts)) === 0);
        steps.push({ node: 'stop', lines: [12], say: kids.map((k, i) => 'The ' + k.name + ' child is ' + (pure[i] ? 'pure: a leaf.' : 'still mixed (Gini ' + f(gini(counts(k.pts))) + '), so it grows.')).join(' '), state: { best, split: true } });
        const mixed = kids.find((k, i) => !pure[i]);
        if (mixed) {
          const cc = candidates(mixed.pts), b2 = cc.reduce((a, c) => (c.w < a.w ? c : a));
          steps.push({ node: 'node', lines: [11, 12], say: 'The ' + mixed.name + ' child becomes a node of its own, with ' + mixed.pts.length + ' points. The same search runs on it alone.', state: { best, split: true, focus: mixed.name } });
          steps.push({ node: 'cand', lines: [3, 4], say: 'Its ' + cc.length + ' candidate splits are tried one at a time, each scored exactly as at the root.', state: { best, split: true, focus: mixed.name } });
          steps.push({ node: 'more', lines: [3, 4], edge: 'score>more', say: 'All ' + cc.length + ' candidates are scored. The best is ' + FEAT[b2.j] + ' < ' + f(b2.t) + ', with weighted Gini ' + f(b2.w) + (b2.w === 0 ? ': both sides pure.' : '.'), formula: 'G for each candidate:\n' + cc.map(c => FEAT[c.j] + ' < ' + f(c.t) + ': ' + f(c.w)).join(',  '), state: { best, split: true, best2: b2, side: mixed.name } });
          steps.push({ node: 'split', lines: [13, 14], say: 'The ' + mixed.name + ' child splits on ' + FEAT[b2.j] + ' < ' + f(b2.t) + '.', state: { best, split: true, best2: b2, side: mixed.name, split2: true } });
          steps.push({ node: 'stop', lines: [12], say: 'Every child is now pure, so all of them become leaves. The tree has depth 2 and classifies every training point correctly.', state: { best, split: true, best2: b2, side: mixed.name, split2: true } });
        }
        steps.push({ node: 'leaf', lines: [12], say: 'Each leaf predicts its majority class; a new point follows the questions from the root down to one leaf. The coloured regions are the tree\'s answer for every possible point: always boxes with edges parallel to the axes.', state: { best, split: true, best2: mixed ? candidates(mixed.pts).reduce((a, c) => (c.w < a.w ? c : a)) : null, side: mixed && mixed.name, split2: !!mixed, regions: true } });
        return steps;
      },
      draw(s, kit) {
        const X0 = 0.5, X1 = 4.5, Y0 = 0.5, Y1 = 4.2;
        const fr = kit.frame([X0, X1], [Y0, Y1], ['x₁', 'x₂']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        const cut = (c, box, cls, key) => {                     // a split line inside a box [x0, x1, y0, y1]
          if (c.j === 0) out.push({ k: key, t: 'l', x1: sx(c.t), y1: sy(box[2]), x2: sx(c.t), y2: sy(box[3]), cls });
          else out.push({ k: key, t: 'l', x1: sx(box[0]), y1: sy(c.t), x2: sx(box[1]), y2: sy(c.t), cls });
        };
        const full = [X0, X1, Y0, Y1];
        if (s.regions && s.best) {
          const b = s.best, maj = pts => { const c = counts(pts); return c[1] > c[0] ? 'wp-area2' : 'wp-area'; };
          const boxes = [];
          const lbox = b.j === 0 ? [X0, b.t, Y0, Y1] : [X0, X1, Y0, b.t], rbox = b.j === 0 ? [b.t, X1, Y0, Y1] : [X0, X1, b.t, Y1];
          const inBox = (p, bx) => p[0] >= bx[0] && p[0] < bx[1] && p[1] >= bx[2] && p[1] < bx[3];
          [[lbox, 'left'], [rbox, 'right']].forEach(([bx, name]) => {
            if (s.best2 && s.side === name) {
              const c = s.best2, a = c.j === 0 ? [bx[0], c.t, bx[2], bx[3]] : [bx[0], bx[1], bx[2], c.t], z = c.j === 0 ? [c.t, bx[1], bx[2], bx[3]] : [bx[0], bx[1], c.t, bx[3]];
              boxes.push(a, z);
            } else boxes.push(bx);
          });
          boxes.forEach((bx, i) => out.push({ k: 'reg' + i, t: 'r', x: sx(bx[0]), y: sy(bx[3]), w: sx(bx[1]) - sx(bx[0]), h: sy(bx[2]) - sy(bx[3]), cls: maj(D.filter(p => inBox(p, bx))) }));
        }
        if (s.cand) cut(s.cand, full, 'wp-line2', 'cand');
        if (s.best && !s.split && s.best !== s.cand) cut(s.best, full, 'wp-faint', 'best');
        if (s.split && s.best) cut(s.best, full, 'wp-line', 'best');
        if (s.split2 && s.best2) {
          const b = s.best, bx = (s.side === 'left') === true ? (b.j === 0 ? [X0, b.t, Y0, Y1] : [X0, X1, Y0, b.t]) : (b.j === 0 ? [b.t, X1, Y0, Y1] : [X0, X1, b.t, Y1]);
          cut(s.best2, bx, 'wp-line', 'best2');
        } else if (s.best2 && s.side) {
          const b = s.best, bx = s.side === 'left' ? (b.j === 0 ? [X0, b.t, Y0, Y1] : [X0, X1, Y0, b.t]) : (b.j === 0 ? [b.t, X1, Y0, Y1] : [X0, X1, b.t, Y1]);
          cut(s.best2, bx, 'wp-line2', 'cand2');
        }
        D.forEach((d, i) => out.push({ k: 'p' + i, t: 'c', x: sx(d[0]), y: sy(d[1]), r: 5, cls: d[2] ? 'wp-c1' : 'wp-c0' }));
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- gradient boosting with stumps
  (function () {
    const X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], Y = [1.0, 1.3, 2.2, 3.8, 4.1, 4.0, 5.9, 6.2, 6.0, 7.8];
    const mse = F => mean(F.map((v, i) => (Y[i] - v) ** 2));
    function stump(r) {                                    // the single split on x that best fits the residuals r
      let best = null;
      for (let k = 0; k + 1 < X.length; k++) {
        const t = (X[k] + X[k + 1]) / 2, L = r.filter((v, i) => X[i] < t), R = r.filter((v, i) => X[i] >= t);
        const ml = mean(L), mr = mean(R), sse = L.reduce((s, v) => s + (v - ml) ** 2, 0) + R.reduce((s, v) => s + (v - mr) ** 2, 0);
        if (!best || sse < best.sse) best = { t, ml, mr, sse };
      }
      return best;
    }
    W('boosting', {
      title: 'Gradient boosting: each tree fixes what the others got wrong',
      symbols: [['x', 'the input of each point'], ['y', 'the true value of each point'],
        ['F', 'the model so far: its prediction at each x. F₀ is the first model, the mean of y'], ['r', 'the residuals y − F: what the model still gets wrong at each point'],
        ['stump', 'a tree with a single split: one threshold on x and an output value on each side'],
        ['η', 'the learning rate, or shrinkage: the share of each new tree\'s correction that is added (set below)'], ['Σ', 'the sum over all the trees'],
        ['←', 'replace the value on the left with the value on the right'], ['MSE', 'the mean squared error of the model on the training points']],
      params: [{ key: 'lr', label: 'Learning rate η', value: 0.3, options: [[0.3, '0.3'], [1, '1 (no shrinkage)']] }],
      code: [
        'F = np.full(len(y), y.mean())             # start from the average',
        'trees = []',
        'for m in range(30):',
        '    r = y - F                               # residuals: what is still wrong',
        '    tree = DecisionTreeRegressor(max_depth=1).fit(x, r)',
        '    F += lr * tree.predict(x)               # add a shrunken correction',
        '    trees.append(tree)',
        '# prediction = y.mean() + lr * (sum of every tree)'
      ],
      nodes: [
        { id: 'init', label: ['Start: F₀ = the mean of y'], why: 'The simplest possible model predicts the same number everywhere. Every tree after this one only corrects it.' },
        { id: 'resid', label: ['Residuals r = y − F'], why: 'The residuals are exactly what the current ensemble still gets wrong. For squared error they are also the negative gradient, hence the name.' },
        { id: 'fit', label: ['Fit a small tree to r'], why: 'A one-split tree (a stump) is a weak learner: it captures just the biggest remaining pattern.' },
        { id: 'add', label: ['Add it, shrunk:', 'F ← F + η·tree'], why: 'Adding only part of each tree (η below 1) needs more rounds but generalizes better: no single tree can dominate.' },
        { id: 'more', label: ['More rounds?'], kind: 'dec', why: 'The number of rounds is the main knob: too few underfits, too many starts fitting noise. In practice, early stopping on validation data picks it.' },
        { id: 'done', label: ['Done: F = F₀ + η·Σ trees'], kind: 'end', why: 'The final model is the starting mean plus the sum of every shrunken tree.' }
      ],
      edges: [
        { from: 'init', to: 'resid', label: 'what is left', why: 'Everything the mean cannot explain is in the residuals.' },
        { from: 'resid', to: 'fit', label: 'target = r', why: 'Each new tree learns the errors, not the original y.' },
        { from: 'fit', to: 'add', label: 'shrink by η', why: 'A small step in the direction that reduces the error most.' },
        { from: 'add', to: 'more', label: 'new F', why: 'The ensemble has improved; check the budget.' },
        { from: 'more', to: 'resid', label: 'yes: again', why: 'There are rounds left, and new residuals to fit.' },
        { from: 'more', to: 'done', label: 'no: stop', why: 'The planned number of trees is reached.' }
      ],
      run(p) {
        const lr = +p.lr, steps = [], m0 = mean(Y);
        let F = X.map(() => m0), since = 1;
        steps.push({ node: 'init', lines: [1], say: 'The first model predicts the average, ' + f(m0) + ', for every x. Its mean squared error is ' + f(mse(F)) + '.', vars: [['trees', 0], ['MSE', f(mse(F))]], state: { F: F.slice() } });
        for (let m = 1; m <= 30; m++) {
          const r = Y.map((v, i) => v - F[i]), s = stump(r), tree = X.map(x => (x < s.t ? s.ml : s.mr));
          const F2 = F.map((v, i) => v + lr * tree[i]);
          if (m <= 2) {
            steps.push({ node: 'resid', lines: [4], say: 'Round ' + m + ': the red segments are the residuals, what the model still gets wrong at each point.', formula: 'r = y − F = ' + list(r), vars: [['trees', m - 1], ['MSE', f(mse(F))]], state: { F: F.slice(), res: true },
              ask: m === 1 ? 'Where would a single split of x help most?' : null });
            steps.push({ node: 'fit', lines: [5], say: 'The best single split of x is at x < ' + f(s.t) + ': the stump predicts ' + f(s.ml) + ' on the left and ' + f(s.mr) + ' on the right. The dashed line shows the model with the whole correction added.', formula: 'stump(x) = ' + f(s.ml) + ' if x < ' + f(s.t) + ', else ' + f(s.mr), vars: [['trees', m - 1], ['MSE', f(mse(F))]], state: { F: F.slice(), prop: F.map((v, i) => v + tree[i]) } });
            steps.push({ node: 'add', lines: [6, 7], say: 'Only ' + (lr === 1 ? 'all' : Math.round(lr * 100) + '%') + ' of the correction is added. The error drops from ' + f(mse(F)) + ' to ' + f(mse(F2)) + '.', formula: 'F ← F + η·stump = F + ' + lr + ' × stump', vars: [['trees', m], ['MSE', f(mse(F2))]], state: { F: F2.slice(), ghost: F.slice() } });
          }
          F = F2;
          if (m <= 2 || [5, 10, 20, 30].includes(m)) {
            steps.push({ node: 'more', lines: [3], say: m <= 2 ? 'Rounds are left, so the loop continues with new residuals.' : 'Rounds ' + since + ' to ' + m + ' repeated the loop: after ' + m + ' trees the error is ' + f(mse(F)) + '.', vars: [['trees', m], ['MSE', f(mse(F))]], state: { F: F.slice() } });
            since = m + 1;
          }
        }
        steps.push({ node: 'done', lines: [8], say: 'After 30 stumps the model is a staircase that follows the data, with training error ' + f(mse(F)) + '. ' + (lr === 1 ? 'Without shrinkage it fits the training points fast, noise included: on new data that usually costs accuracy.' : 'With η = 0.3 each tree only nudges the fit, which is slower but usually generalizes better.'), vars: [['trees', 30], ['MSE', f(mse(F))]], state: { F: F.slice() } });
        return steps;
      },
      draw(s, kit) {
        const fr = kit.frame([-0.5, 9.5], [0, 9], ['x', 'y']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        const stair = F => F.map((v, i) => (i ? 'L' : 'M') + sx(X[i] - 0.5) + ' ' + sy(v) + ' L' + sx(X[i] + 0.5) + ' ' + sy(v)).join(' ');
        if (s.ghost) out.push({ k: 'g', t: 'p', d: stair(s.ghost), cls: 'wp-faint' });
        if (s.prop) out.push({ k: 'prop', t: 'p', d: stair(s.prop), cls: 'wp-line2' });
        if (s.F) out.push({ k: 'F', t: 'p', d: stair(s.F), cls: 'wp-line' });
        X.forEach((x, i) => {
          if (s.res && s.F) out.push({ k: 'r' + i, t: 'l', x1: sx(x), y1: sy(Y[i]), x2: sx(x), y2: sy(s.F[i]), cls: 'wp-res' });
          out.push({ k: 'p' + i, t: 'c', x: sx(x), y: sy(Y[i]), r: 4.5, cls: 'wp-c0' });
        });
        return out;
      }
    });
  })();

  // ---------------------------------------------------------------- naive Bayes: counting, then adding evidence
  (function () {
    const SPAM = ['win money now', 'win a free prize', 'free money offer', 'claim your free prize now'];
    const HAM = ['lunch at noon', 'see you at lunch', 'the meeting is at noon', 'can you send the notes'];
    const MSG = 'free lunch now'.split(' ');
    const toks = docs => docs.join(' ').split(' ');
    const ts = toks(SPAM), th = toks(HAM), V = new Set(ts.concat(th)).size;
    const cnt = (t, w) => t.filter(x => x === w).length;
    const prob = (t, w) => (cnt(t, w) + 1) / (t.length + V);
    W('naive-bayes', {
      title: 'How naive Bayes weighs the words of a message',
      symbols: [['P(class)', 'the prior: the share of training messages in that class, before any word is read'],
        ['P(spam)', 'the prior for spam'], ['P(normal)', 'the prior for normal mail'],
        ['P(w | spam)', 'the probability that a word picked from a spam message is the word w; the bar reads "given"'],
        ['P(word | class)', 'the probability of the word in a message of that class; the bar reads "given"'],
        ['V', 'the vocabulary size: how many different words the training messages contain'],
        ['log', 'the natural logarithm. Adding logs is the same as multiplying probabilities, without the product becoming too small for the computer to store'],
        ['e', 'Euler\'s number, about 2.718: raising e to a log gives the probability back'],
        ['P(spam | message)', 'the probability that this message is spam, given its words']],
      code: [
        'prior = {c: n_docs[c] / n_total for c in classes}',
        'count = {c: Counter(words in class c) for c in classes}',
        'def p_word(w, c):                         # with add-one smoothing',
        '    return (count[c][w] + 1) / (total_words[c] + V)',
        'def classify(message):',
        '    score = {c: np.log(prior[c]) for c in classes}',
        '    for w in message.split():',
        '        for c in classes:',
        '            score[c] += np.log(p_word(w, c))   # add the word\'s evidence',
        '    return max(score, key=score.get)'
      ],
      nodes: [
        { id: 'prior', label: ['Count messages per class:', 'the priors P(class)'], why: 'Before reading a single word, the model only knows how common each class is in the training data.' },
        { id: 'count', label: ['Count each word in each class'], why: 'Training is just counting: how often every word appears in spam and in normal mail. That is why naive Bayes trains so fast.' },
        { id: 'smooth', label: ['Add one to every count', '(Laplace smoothing)'], why: 'Without it, a word never seen in a class would give probability 0 and veto that class no matter what else the message says.' },
        { id: 'start', label: ['New message: each score', 'starts at log P(class)'], why: 'Working with log probabilities turns a long product of small numbers into a sum, which never underflows to zero.' },
        { id: 'word', label: ['Add log P(word | class)', 'for the next word'], why: 'The naive assumption: each word counts as independent evidence, so its log probability is simply added.' },
        { id: 'more', label: ['More words?'], kind: 'dec', why: 'Every word of the message gets its say, one at a time.' },
        { id: 'pick', label: ['Pick the higher score;', 'normalize to probabilities'], kind: 'end', why: 'The class with the larger total wins. Normalizing the two scores gives the probability of each class.' }
      ],
      edges: [
        { from: 'prior', to: 'count', label: 'then words', why: 'The words are the evidence that can move the model away from the priors.' },
        { from: 'count', to: 'smooth', label: 'fix zeros', why: 'Unseen words would otherwise have probability zero.' },
        { from: 'smooth', to: 'start', label: 'trained', why: 'Training is finished. A message can now be classified.' },
        { from: 'start', to: 'word', label: 'read', why: 'Evidence is added one word at a time.' },
        { from: 'word', to: 'more', label: 'next?', why: 'Check whether the message has more words.' },
        { from: 'more', to: 'word', label: 'yes', why: 'There is more evidence to add.' },
        { from: 'more', to: 'pick', label: 'no: decide', why: 'All the evidence is in.' }
      ],
      run() {
        const steps = [], sc = [Math.log(0.5), Math.log(0.5)], hist = [];
        steps.push({ node: 'prior', lines: [1], say: 'The training set has 4 spam and 4 normal messages, so each class starts with prior probability 0.5.', formula: 'P(spam) = 4/8 = 0.5,  P(normal) = 4/8 = 0.5', state: { scores: null } });
        steps.push({ node: 'count', lines: [2], say: 'The spam messages contain ' + ts.length + ' words and the normal ones ' + th.length + ', with ' + V + ' different words in all. For the words of the message to classify: ' + MSG.map(w => '"' + w + '" appears ' + cnt(ts, w) + ' times in spam and ' + cnt(th, w) + ' in normal mail').join('; ') + '.', state: { scores: null } });
        steps.push({ node: 'smooth', lines: [3, 4], say: '"lunch" never appears in spam, and "free" never in normal mail. Adding one to every count keeps such words from forcing a probability of zero.', formula: 'P(w | spam) = (count of w + 1) / (spam words + V)\n' + MSG.map(w => 'P(' + w + ' | spam) = ' + (cnt(ts, w) + 1) + '/' + (ts.length + V) + ' = ' + f(prob(ts, w))).join('\n'), state: { scores: null } });
        steps.push({ node: 'start', lines: [5, 6], say: 'To classify "' + MSG.join(' ') + '", both scores start at log 0.5 = ' + f(sc[0]) + '.', vars: [['spam', f(sc[0])], ['normal', f(sc[1])]], state: { scores: sc.slice(), words: [] } });
        MSG.forEach((w, i) => {
          const a = Math.log(prob(ts, w)), b = Math.log(prob(th, w));
          sc[0] += a; sc[1] += b; hist.push(w);
          steps.push({ node: 'word', lines: [7, 8, 9], say: '"' + w + '" is ' + f(prob(ts, w) / prob(th, w)) + ' times as likely in spam as in normal mail' + (a > b ? ', so it pushes toward spam.' : ', so it pushes toward normal mail.'),
            formula: 'spam:   ' + f(sc[0] - a) + ' + log ' + f(prob(ts, w)) + ' = ' + f(sc[0]) + '\nnormal: ' + f(sc[1] - b) + ' + log ' + f(prob(th, w)) + ' = ' + f(sc[1]), vars: [['spam', f(sc[0])], ['normal', f(sc[1])]], state: { scores: sc.slice(), words: hist.slice() },
            ask: i === 1 ? 'One more word, "now". Will it help spam or normal mail?' : null });
          if (i < MSG.length - 1) steps.push({ node: 'more', lines: [7], say: 'More words remain.', vars: [['spam', f(sc[0])], ['normal', f(sc[1])]], state: { scores: sc.slice(), words: hist.slice() } });
        });
        steps.push({ node: 'more', lines: [7], say: 'Every word has been read.', vars: [['spam', f(sc[0])], ['normal', f(sc[1])]], state: { scores: sc.slice(), words: hist.slice() } });
        const ps = 1 / (1 + Math.exp(sc[1] - sc[0]));
        steps.push({ node: 'pick', lines: [10], say: 'Spam scores ' + f(sc[0]) + ' against ' + f(sc[1]) + ' for normal mail, so the message is classified as ' + (ps > 0.5 ? 'spam' : 'normal') + ', with probability ' + f(ps > 0.5 ? ps : 1 - ps) + '. "lunch" argued the other way, but "free" and "now" together outweighed it.',
          formula: 'P(spam | message) = 1 / (1 + e^(' + f(sc[1]) + ' − (' + f(sc[0]) + '))) = ' + f(ps), vars: [['P(spam | message)', f(ps)]], state: { scores: sc.slice(), words: hist.slice(), ps } });
        return steps;
      },
      draw(s, kit) {
        const out = [];
        out.push({ k: 'ttl', t: 't', x: 18, y: 14, s: s.scores ? 'log scores (closer to 0 = more likely)' : 'training: counting words', cls: 'wp-tick' });
        if (!s.scores) {
          ['spam', 'normal'].forEach((name, c) => {
            const docs = c ? HAM : SPAM;
            out.push({ k: 'h' + c, t: 't', x: 18 + c * 142, y: 38, s: name + ' (' + docs.length + ' messages)', cls: 'wp-txt' });
            docs.forEach((d, i) => out.push({ k: 'd' + c + i, t: 't', x: 18 + c * 142, y: 58 + i * 18, s: d, cls: 'wp-tick' }));
          });
          return out.map(o => Object.assign(o, { f: 1 }));
        }
        const y0 = 44, base = 160, lo = -12, scale = v => y0 + (v / lo) * (base - y0);
        out.push({ k: 'w', t: 't', x: 18, y: 29, s: s.words.length ? 'words read: ' + s.words.join(', ') : 'no words read yet', cls: 'wp-tick' });
        if (s.ps !== undefined) out.push({ k: 'ps', t: 't', x: 290, y: 29, s: 'P(spam) = ' + f(s.ps), cls: 'wp-txt', a: 'end' });
        out.push({ k: 'zero', t: 'l', x1: 50, y1: y0, x2: 280, y2: y0, cls: 'wp-axis' });
        ['spam', 'normal'].forEach((name, c) => {
          const x = 80 + c * 110, v = s.scores[c];
          out.push({ k: 'b' + c, t: 'r', x, y: y0, w: 60, h: scale(v) - y0, cls: c ? 'wp-bar2' : 'wp-bar' });
          out.push({ k: 'v' + c, t: 't', x: x + 30, y: scale(v) + 14, s: f(v), cls: 'wp-txt', a: 'middle' });
          out.push({ k: 'n' + c, t: 't', x: x + 30, y: 196, s: name, cls: 'wp-txt', a: 'middle' });
        });
        return out.map(o => Object.assign(o, { f: 1 }));
      }
    });
  })();

  // ---------------------------------------------------------------- EM for a mixture of two Gaussians
  (function () {
    const X = [1.2, 1.8, 2.1, 2.5, 2.9, 3.3, 5.9, 6.4, 6.9, 7.2, 7.8, 8.4];
    const N = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
    W('em-gmm', {
      title: 'Expectation-maximization, round by round',
      symbols: [['x', 'a data point'], ['μ', 'the mean of a Gaussian: where its bell curve is centred'], ['σ²', 'the variance of a Gaussian: how wide its bell curve is'],
        ['π', 'the weight of a Gaussian: the share of all points it accounts for. The weights add up to 1'],
        ['N', 'the bell curve (normal density): N(x | μ, σ²) is high when x is near μ'],
        ['r', 'the responsibility: the share of point x that belongs to the first Gaussian. The second gets 1 − r'],
        ['Σ', 'the sum over all points'], ['n', 'the number of points (12)'],
        ['E-step', 'the expectation step: share each point between the Gaussians, given their current shapes'],
        ['M-step', 'the maximization step: refit each Gaussian to its share of the points'],
        ['log-likelihood', 'how well the mixture explains the data: the sum over the points of the log of their density. EM never lets it fall']],
      code: [
        'mu, var, pi = np.array([3.5, 5.0]), np.array([4.0, 4.0]), np.array([0.5, 0.5])',
        'for it in range(50):',
        '    dens = pi * norm.pdf(x[:, None], mu, np.sqrt(var))   # each point, each component',
        '    r = dens / dens.sum(axis=1, keepdims=True)           # E-step: responsibilities',
        '    nk = r.sum(axis=0)',
        '    pi = nk / len(x)                                       # M-step: weights,',
        '    mu = (r * x[:, None]).sum(axis=0) / nk                 # means,',
        '    var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / nk   # and variances',
        '    ll = np.log(dens.sum(axis=1)).sum()',
        '    if ll - prev < 1e-4: break'
      ],
      nodes: [
        { id: 'init', label: ['Start: guess two Gaussians'], why: 'EM needs a starting guess for each component: a mean, a spread and a weight. A poor guess only costs extra rounds here, though in general it can lead to a worse answer.' },
        { id: 'e', label: ['E-step: how much does each', 'Gaussian explain each point?'], why: 'Each point is shared between the components in proportion to how likely each one makes it. These shares are the responsibilities.' },
        { id: 'm', label: ['M-step: refit each Gaussian', 'to its share of the points'], why: 'Each component\'s new mean, variance and weight are weighted averages, with the responsibilities as weights.' },
        { id: 'check', label: ['Did the likelihood still rise?'], kind: 'dec', why: 'Each E-step plus M-step can only raise (or keep) the log-likelihood, so EM stops when the gain becomes tiny.' },
        { id: 'done', label: ['Done: two clusters,', 'each with its own shape'], kind: 'end', why: 'Unlike k-means, every point keeps a soft membership, and each cluster has its own spread.' }
      ],
      edges: [
        { from: 'init', to: 'e', label: 'given the guess', why: 'With the parameters fixed, responsibilities can be computed exactly.' },
        { from: 'e', to: 'm', label: 'given the shares', why: 'With the responsibilities fixed, the best parameters have a closed form.' },
        { from: 'm', to: 'check', label: 'new parameters', why: 'Measure how well the new mixture explains the data.' },
        { from: 'check', to: 'e', label: 'yes: again', why: 'The fit is still improving: recompute the shares with the new Gaussians.' },
        { from: 'check', to: 'done', label: 'no: stop', why: 'The log-likelihood has stopped rising: a local optimum.' }
      ],
      run() {
        const steps = [];
        let mu = [3.5, 5.0], vr = [4, 4], pi = [0.5, 0.5], prev = -Infinity, since = 1;
        const ll = () => X.reduce((s, x) => s + Math.log(pi[0] * N(x, mu[0], vr[0]) + pi[1] * N(x, mu[1], vr[1])), 0);
        const resp = () => X.map(x => { const a = pi[0] * N(x, mu[0], vr[0]), b = pi[1] * N(x, mu[1], vr[1]); return a / (a + b); });
        const vars = L => [['μ', f(mu[0]) + ', ' + f(mu[1])], ['σ²', f(vr[0]) + ', ' + f(vr[1])], ['π', f(pi[0]) + ', ' + f(pi[1])], ['log-likelihood', f(L)]];
        const st = r => ({ mu: mu.slice(), vr: vr.slice(), pi: pi.slice(), r });
        steps.push({ node: 'init', lines: [1], say: 'The guess: two Gaussians with means 3.5 and 5, both wide (variance 4), each with half the weight. They overlap a lot and sit between the two real groups.', vars: vars(ll()), state: st(null) });
        for (let it = 1; it <= 50; it++) {
          const r = resp(), i6 = 6;
          if (it <= 2) steps.push({ node: 'e', lines: [3, 4], say: 'Round ' + it + ', E-step. Each point is shared between the two Gaussians. The point at ' + X[i6] + ', for example, belongs ' + Math.round(r[i6] * 100) + '% to the first (blue) and ' + Math.round((1 - r[i6]) * 100) + '% to the second (orange).', formula: 'r = π₁N(x | μ₁, σ₁²) / (π₁N(x | μ₁, σ₁²) + π₂N(x | μ₂, σ₂²)) = ' + f(r[i6]) + '  for x = ' + X[i6], vars: vars(ll()), state: st(r),
            ask: it === 1 ? 'After the M-step, will the orange Gaussian move left or right?' : null });
          const nk = [r.reduce((s, v) => s + v, 0), r.reduce((s, v) => s + 1 - v, 0)];
          const mu2 = [X.reduce((s, x, i) => s + r[i] * x, 0) / nk[0], X.reduce((s, x, i) => s + (1 - r[i]) * x, 0) / nk[1]];
          const vr2 = [Math.max(0.05, X.reduce((s, x, i) => s + r[i] * (x - mu2[0]) ** 2, 0) / nk[0]), Math.max(0.05, X.reduce((s, x, i) => s + (1 - r[i]) * (x - mu2[1]) ** 2, 0) / nk[1])];
          const old = { mu: mu.slice() };
          mu = mu2; vr = vr2; pi = [nk[0] / X.length, nk[1] / X.length];
          const L = ll();
          if (it <= 2) steps.push({ node: 'm', lines: [5, 6, 7, 8], say: 'M-step: each Gaussian is refitted to its share of the points. The means move from ' + f(old.mu[0]) + ' and ' + f(old.mu[1]) + ' to ' + f(mu[0]) + ' and ' + f(mu[1]) + '.', formula: 'μ₁ = Σ r·x / Σ r = ' + f(mu[0]) + ',  μ₂ = ' + f(mu[1]) + '\nσ₁² = Σ r·(x − μ₁)² / Σ r = ' + f(vr[0]) + ',  σ₂² = ' + f(vr[1]) + '\nπ₁ = Σ r / n = ' + f(pi[0]) + ',  π₂ = ' + f(pi[1]), vars: vars(L), state: st(r) });
          const flat = L - prev < 1e-4, done = flat || it === 50;
          if (it <= 2 || done || [3, 5, 10].includes(it)) {
            steps.push({ node: 'check', lines: [9, 10], say: flat ? 'The log-likelihood rose by less than 0.0001: converged after ' + it + ' rounds.' : it <= 2 ? 'The log-likelihood rose to ' + f(L) + ', so another round follows.' : 'Rounds ' + since + ' to ' + it + ' repeated the E-step and M-step. The log-likelihood is ' + f(L) + '.', vars: vars(L), state: st(resp()) });
            since = it + 1;
          }
          prev = L;
          if (done) {
            steps.push({ node: 'done', lines: [], say: 'The two Gaussians now sit on the two groups, at ' + f(mu[0]) + ' and ' + f(mu[1]) + ', with variances ' + f(vr[0]) + ' and ' + f(vr[1]) + '. Points far from both boundaries belong almost entirely to one component; a point between them would be shared.', vars: vars(L), state: st(resp()) });
            break;
          }
        }
        return steps;
      },
      draw(s, kit) {
        const fr = kit.frame([0, 10], [0, 0.4], ['x', 'density']), out = fr.shapes, sx = fr.sx, sy = fr.sy;
        [0, 1].forEach(k => {
          let d = '';
          for (let i = 0; i <= 80; i++) { const x = i / 8; d += (i ? 'L' : 'M') + sx(x).toFixed(1) + ' ' + sy(s.pi[k] * N(x, s.mu[k], s.vr[k])).toFixed(1) + ' '; }
          out.push({ k: 'g' + k, t: 'p', d, cls: k ? 'wp-line2' : 'wp-line' });
        });
        X.forEach((x, i) => out.push({ k: 'p' + i, t: 'c', x: sx(x), y: sy(0.012), r: 5, cls: !s.r ? 'wp-pt' : s.r[i] > 0.5 ? 'wp-c0' : 'wp-c1' }));
        return out;
      }
    });
  })();
})(typeof globalThis !== 'undefined' ? globalThis : window);
