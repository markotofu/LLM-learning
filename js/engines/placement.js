/* ML Atlas – placement test engine (framework independent, unit tested in Node: tests/placement.test.js).
 * An optional adaptive test of about 12 to 15 questions, so newcomers with some background can skip what they know.
 * Every question is one of a lesson's own quiz questions, from the beginner and intermediate lessons of Start Here,
 * Math Foundations, Python & Data and Classical Machine Learning (POOL). An answer says something about a group of
 * lessons along the prerequisite graph: knowing a lesson implies knowing what it builds on, and not knowing a lesson
 * yet implies not knowing what builds on it.
 *
 * How it decides. A session keeps a few thousand possible knowledge states, each a set of lessons that includes the
 * prerequisites of every lesson in it, drawn from a prior in which background varies by branch. Every answer reweighs
 * them by how likely that answer is under each state, allowing for slips (a wrong answer on a lesson you know) and
 * lucky guesses (a right answer on one you do not, likelier on a four-option question than on one you type). The next
 * question is the one whose answer is expected to remove the most uncertainty about all the lessons (the sum of their
 * binary entropies); that can be a lesson's second question, when one answer left it unsettled. The test stops after
 * MAX questions, or earlier when no question would tell much more (a beginner who answers "I don't know yet" is done
 * after about seven). A lesson is marked as known when it is known in at least 90% of the weighted states, its own
 * questions were not missed, and a right answer on it or on a lesson that builds on it supports it, so every mark
 * has a reason to show. Simulated learners (tests/placement.test.js) keep wrong marks rare.
 * Randomness comes from a seeded generator: the same seed and answers always give the same questions and results,
 * which is how a test in progress is restored from the saved answers (MLA.store.state.placement). */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  // The questions, by lesson: the opening words of one or two of the lesson's own quiz questions, in order of
  // preference (a retake asks the other one when it can). Chosen to test the core idea of the lesson, to make sense
  // outside it, and to be quick to answer: multiple choice or a short number. tests/placement.test.js checks that
  // every opening still finds exactly one question of an allowed type, so rewording a quiz question means updating it.
  const POOL = [
    ['what-is-ml', ['In machine learning, what does', 'Which of these tasks is **unsupervised**?']],
    ['vectors-matrices', ['Compute the dot product [2, −1,', 'Which shapes can be multiplied']],
    ['derivatives', ['If ∂L/∂w = −3 at the current', 'At a point where f′(x) = 0 and']],
    ['probability', ['For a Gaussian with mean 100', 'What is the probability of rolling']],
    ['python-essentials', ['After a = [1, 2]; b = a; b.append(3),', 'A function has no `return` statement.']],
    ['statistics', ['What does a z-score of −2 mean?', 'What is the median of [7, 1,']],
    ['information-theory', ['Which distribution over 4 outcomes', 'What is the entropy, in bits,']],
    ['eigen-svd', ['For A = [[3, 0], [0, 1]], which', 'What does the number of non-zero']],
    ['matrix-calculus', ['W is a 3 × 5 matrix and L is', 'Why does deep learning use reverse-mode']],
    ['discrete-math', ['By De Morgan\'s laws,', 'Which algorithm finds the shortest']],
    ['numpy', ['X has shape (64, 10). What is', 'Which shapes broadcast together?']],
    ['supervised-learning', ['What is a decision boundary?', 'Which is a multi-class classification']],
    ['stat-inference', ['You test 20 independent metrics', 'How does the bootstrap build']],
    ['pandas-eda', ['Which expression selects the', 'What is the difference between']],
    ['loss-functions', ['Targets [2, 4] and predictions', 'If you must predict one constant']],
    ['knn', ['kNN with k = 1 achieves 100%', 'What is the main problem with']],
    ['decision-trees', ['What is the Gini impurity of', 'How does a classification tree']],
    ['naive-bayes', ['In Naive Bayes, what does the', 'Class "spam" has 8 words in total,']],
    ['kmeans', ['What does k-means minimise?', 'Which data would k-means handle']],
    ['pca', ['The first principal component', 'Why standardize before PCA when']],
    ['data-preparation', ['A model scores 99.8% accuracy', 'A sensor writes −999 when a reading']],
    ['data-visualization', ['You need to compare the distribution', 'After `fig, axes = plt.subplots(2,']],
    ['sql-data-engineering', ['Which clause filters groups after', 'How do you find users with no']],
    ['gradient-descent', ['w = 2, the gradient is 4 and', 'On L(w) = w², gradient descent']],
    ['mle-map', ['What is the likelihood L(θ)?', 'Which prior on the weights corresponds']],
    ['feature-engineering', ['One-hot encoding a "country"', 'A feature has training mean 50']],
    ['linear-regression', ['Why add a column of ones to X', 'Residual sum of squares is 20']],
    ['logistic-regression', ['Why is log loss preferred over', 'A logistic model has a weight']],
    ['overfitting', ['A degree-15 polynomial fits 16', 'Which error component can no']],
    ['regularization', ['In ridge regression, what happens', 'Which regularizer tends to produce']],
    ['evaluation-metrics', ['Raising the decision threshold', 'TP = 40, FN = 10. What is the']],
    ['cross-validation', ['You are forecasting electricity', 'You tried 50 hyperparameter combinations']],
    ['ensembles', ['Bagging mainly reduces', 'What is a bootstrap sample?']],
    ['glm-survival', ['Which link function does a Poisson', 'What is a censored observation']],
    ['svm', ['What are support vectors?', 'What does the kernel trick allow?']],
    ['sklearn-workflow', ['What should you save to deploy', 'What does a ColumnTransformer']],
    ['data-centric', ['How do you find likely mislabelled', 'A fraud model has 99% accuracy']]
  ];
  // Prerequisites that set a reading order rather than knowledge: the maths and Python lessons do not depend on the
  // introduction to machine learning, and discrete maths does not depend on Python. The test ignores these links.
  const SOFT = [['what-is-ml', 'vectors-matrices'], ['what-is-ml', 'derivatives'], ['what-is-ml', 'probability'], ['what-is-ml', 'python-essentials'], ['python-essentials', 'discrete-math']];
  const TYPES = ['mcq', 'fill', 'multi'];
  const MAX = 15, MIN = 5, PARTICLES = 4000, MARK = 0.9, STOP_GAIN = 0.1;
  // The answer model: how likely each answer is when the learner knows the lesson, and when they do not yet.
  // "skip" is the "I don't know yet" button; the test asks for it instead of a guess.
  const IF_KNOWN = { right: 0.9, wrong: 0.08, skip: 0.02 };
  const SKIP_IF_UNKNOWN = 0.5, OUTCOMES = ['right', 'wrong', 'skip'];
  // Chance of a lucky guess: options can often be narrowed down, so a four-option question counts as 2 in 5.
  function guessRate(q) {
    if (!q) return 0.4;
    if (q.type === 'fill') return 0.03;
    if (q.type === 'multi') return 0.1;
    return Math.min(0.5, 1.6 / Math.max(2, (q.options || []).length));
  }
  function likelihood(outcome, known, g) {
    if (known) return IF_KNOWN[outcome] || 0.05;
    if (outcome === 'skip') return SKIP_IF_UNKNOWN;
    return (1 - SKIP_IF_UNKNOWN) * (outcome === 'right' ? g : 1 - g);
  }
  const H = p => (p <= 1e-9 || p >= 1 - 1e-9 ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p)));

  const norm = s => String(s == null ? '' : s).toLowerCase().replace(/\s+/g, ' ').trim();
  const poolById = {};
  POOL.forEach(([id, openings]) => { poolById[id] = openings; });

  function concept(id) { return (MLA.content && MLA.content.byId && MLA.content.byId[id]) || null; }
  // The lessons the test covers, in learning order (only those on the map).
  function scope() {
    const g = MLA.content && MLA.content.graph;
    if (!g) return POOL.map(e => e[0]);
    return MLA.graph.topoOrder(g).filter(id => poolById[id]);
  }
  // A lesson's placement questions, aligned with its openings: the question object, or null when the opening finds
  // no single question of an allowed type (or the lesson file has not loaded yet).
  function questions(id) {
    const c = concept(id), openings = poolById[id];
    if (!c || c._stub || !openings) return [];
    return openings.map(o => {
      const hits = (c.quiz || []).filter(q => q && typeof q.q === 'string' && norm(q.q).startsWith(norm(o)));
      return hits.length === 1 && TYPES.includes(hits[0].type) && (!MLA.quiz || MLA.quiz.validQuestion(hits[0])) ? hits[0] : null;
    });
  }
  const qkey = (id, k) => id + '#' + k;
  const isSoft = (p, c) => SOFT.some(e => e[0] === p && e[1] === c);

  function create(opts) {
    opts = opts || {};
    const ids = scope(), N = ids.length, index = {};
    ids.forEach((id, i) => { index[id] = i; });
    // Prerequisites that count, within the test's lessons (through any lesson outside it), as indices.
    const parents = ids.map(id => {
      const out = new Set(), seen = new Set(), stack = (((concept(id) || {}).prereqs) || []).filter(p => !isSoft(p, id));
      while (stack.length) {
        const p = stack.pop(); if (seen.has(p)) continue; seen.add(p);
        if (index[p] != null) out.add(index[p]);
        else for (const q of ((concept(p) || {}).prereqs || [])) if (!isSoft(q, p)) stack.push(q);
      }
      return [...out];
    });
    const children = ids.map(() => []);
    parents.forEach((ps, i) => ps.forEach(p => children[p].push(i)));
    const closure = next => ids.map((x, i) => { const out = new Set(), st = [i]; while (st.length) { const y = st.pop(); for (const z of next[y]) if (!out.has(z)) { out.add(z); st.push(z); } } return out; });
    const anc = closure(parents), desc = closure(children);
    const level = ids.map(id => (concept(id) || {}).level || 1);
    const branch = ids.map(id => (concept(id) || {}).branch || 'ml');
    const branches = [...new Set(branch)];

    // Prior: background varies by branch around an overall level. Ids are in learning order, so prerequisites
    // are decided before the lessons that need them.
    const seed = (opts.seed >>> 0) || 1;
    const rng = MLA.util.seededRng(seed);
    const M = opts.particles || PARTICLES;
    const lists = new Array(M), K = new Uint8Array(M * N);
    for (let m = 0; m < M; m++) {
      const overall = rng(), ability = {};
      for (const b of branches) ability[b] = Math.min(0.97, Math.max(0.03, overall + (rng() - 0.5) * 0.9));
      const row = m * N, list = [];
      for (let i = 0; i < N; i++) {
        if (!parents[i].every(p => K[row + p])) continue;
        const a = ability[branch[i]];
        if (rng() < (level[i] > 1 ? a * Math.sqrt(a) : a)) { K[row + i] = 1; list.push(i); }
      }
      lists[m] = Int16Array.from(list);
    }
    const w = new Float64Array(M).fill(1 / M);
    const answers = [], mine = ids.map(() => []), avoid = new Set(opts.avoid || []);
    let J = null;                                          // J[x * N + i]: weight of the states that know x and i

    function joint() {
      if (J) return J;
      J = new Float64Array(N * N);
      for (let m = 0; m < M; m++) {
        const L = lists[m], wm = w[m];
        if (!wm) continue;
        for (let a = 0; a < L.length; a++) { const row = L[a] * N; for (let b = 0; b < L.length; b++) J[row + L[b]] += wm; }
      }
      return J;
    }
    function marginals() { const j = joint(), out = {}; ids.forEach((id, i) => { out[id] = Math.min(1, Math.max(0, j[i * N + i])); }); return out; }
    // The question to ask next about lesson x, or -1. A lesson not asked yet gets its first usable question that was
    // not asked last time; a lesson answered once, right or wrong, can get its other question to settle it, but not
    // straight after the first one. "I don't know yet" is taken at its word: that lesson is not asked again.
    function pickQuestion(x) {
      const id = ids[x], had = mine[x], qs = questions(id);
      if (had.length >= 2 || had.some(a => a.r === 'skip')) return -1;
      if (had.length && answers.length && answers[answers.length - 1].id === id) return -1;
      const usable = []; qs.forEach((q, k) => { if (q && !had.some(a => a.k === k)) usable.push(k); });
      if (!usable.length) return -1;
      if (had.length) return usable[0];
      const fresh = usable.filter(k => !avoid.has(qkey(id, k)));
      return fresh.length ? fresh[0] : usable[0];
    }
    function gainOf(x, j, g) {
      const W1 = j[x * N + x], W0 = 1 - W1;
      let before = 0; for (let i = 0; i < N; i++) before += H(j[i * N + i]);
      let after = 0;
      for (const y of OUTCOMES) {
        const l1 = likelihood(y, true, g), l0 = likelihood(y, false, g), py = l1 * W1 + l0 * W0;
        if (py <= 0) continue;
        let h = 0;
        for (let i = 0; i < N; i++) { const A = j[x * N + i], B = j[i * N + i] - A; h += H((l1 * A + l0 * B) / py); }
        after += py * h;
      }
      return before - after;
    }
    // The next question { id, k, q, gain, again }, or null when the test is over.
    function next() {
      if (answers.length >= MAX) return null;
      const j = joint(); let best = null;
      for (let x = 0; x < N; x++) {
        const k = pickQuestion(x); if (k < 0) continue;
        const id = ids[x], q = questions(id)[k], gain = gainOf(x, j, guessRate(q));
        if (!best || gain > best.gain + 1e-9) best = { id, k, q, gain, again: mine[x].length > 0 };
      }
      if (!best || (answers.length >= MIN && best.gain < STOP_GAIN)) return null;
      return best;
    }
    // outcome: 'right', 'wrong' or 'skip' ("I don't know yet"); k: which of the lesson's questions was asked.
    function record(id, outcome, k) {
      const x = index[id]; k = k || 0;
      if (x == null || !OUTCOMES.includes(outcome) || mine[x].length >= 2 || mine[x].some(a => a.k === k || a.r === 'skip')) return false;
      const g = guessRate(questions(id)[k] || null);
      const l1 = likelihood(outcome, true, g), l0 = likelihood(outcome, false, g);
      let total = 0;
      for (let m = 0; m < M; m++) { w[m] *= K[m * N + x] ? l1 : l0; total += w[m]; }
      if (total > 0) for (let m = 0; m < M; m++) w[m] /= total;
      const a = { id, k, r: outcome };
      answers.push(a); mine[x].push(a); J = null;
      return true;
    }
    const rights = i => mine[i].filter(a => a.r === 'right').length;
    const missed = i => mine[i].some(a => a.r === 'skip') || (mine[i].length > 0 && !rights(i));
    // The lessons whose right answers support lesson i: itself, or else the nearest lessons that build on it (two at most).
    function support(i) {
      if (rights(i)) return [ids[i]];
      const out = [], seen = new Set([i]), order = id => answers.findIndex(a => a.id === id);
      let frontier = [i];
      while (frontier.length && out.length < 2) {
        const level = [];
        for (const y of frontier) for (const z of children[y]) if (!seen.has(z)) { seen.add(z); level.push(z); }
        level.filter(z => rights(z)).map(z => ids[z]).sort((a, b) => order(a) - order(b)).forEach(id => { if (out.length < 2) out.push(id); });
        frontier = level;
      }
      return out;
    }
    // What the answers say: the lessons to mark as known, each with its reason (why: 'right' when its own question was
    // answered correctly, 'implied' with via: the lessons building on it that were), and the lessons asked about but
    // not marked (r: 'skip', 'wrong', or 'right' when other answers outweigh it; missed: prerequisites whose questions
    // were missed).
    function result() {
      const p = marginals(), marked = [], notMarked = [];
      ids.forEach((id, i) => {
        const via = support(i), own = mine[i];
        if (p[id] >= MARK && via.length && !missed(i)) marked.push({ id, p: p[id], why: rights(i) ? 'right' : 'implied', via: rights(i) ? [] : via, asked: own.length, right: rights(i) });
        else if (own.length) notMarked.push({ id, p: p[id], r: own.some(a => a.r === 'skip') ? 'skip' : rights(i) ? 'right' : 'wrong', asked: own.length, right: rights(i), missed: [...anc[i]].filter(missed).map(a => ids[a]) });
      });
      return { marked, notMarked, answers: answers.map(a => Object.assign({}, a)) };
    }
    return { ids, seed, answers, record, next, marginals, result, ancestors: id => [...anc[index[id]]].map(i => ids[i]), descendants: id => [...desc[index[id]]].map(i => ids[i]) };
  }

  MLA.placement = { POOL, SOFT, MAX, MIN, MARK, scope, questions, create, guessRate, likelihood, qkey };
})(typeof globalThis !== 'undefined' ? globalThis : window);
