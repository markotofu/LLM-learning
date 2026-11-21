/* ML Atlas – branches of the concept map + the content registry every concept file plugs into.
 * See docs/CONTENT_GUIDE.md for the concept / problem format. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};

  MLA.data.branches = [
    { id: 'start', name: 'Start Here', short: 'Start', color: '#cbd5e1', icon: 'flag', desc: 'What machine learning is, and how this map is organised.' },
    { id: 'math', name: 'Math Foundations', short: 'Math', color: '#818cf8', icon: 'graph', desc: 'Vectors, derivatives, probability, statistics and information theory: the language machine learning is written in. Only the parts you actually need, explained from scratch.' },
    { id: 'code', name: 'Python & Data', short: 'Python & Data', color: '#22d3ee', icon: 'code', desc: 'Python, NumPy and the craft of turning messy real-world data into something a model can learn from.' },
    { id: 'ml', name: 'Classical Machine Learning', short: 'Classical ML', color: '#4ade80', icon: 'target', desc: 'Regression, classification, trees, clustering and dimensionality reduction, plus how to train, tune and evaluate models honestly.' },
    { id: 'dl', name: 'Neural Networks & Deep Learning', short: 'Neural Networks', color: '#fbbf24', icon: 'brain', desc: 'From a single neuron to deep networks: backpropagation, training, optimizers, CNNs, RNNs and embeddings.' },
    { id: 'llm', name: 'Transformers & LLMs', short: 'Transformers & LLMs', color: '#a78bfa', icon: 'sparkles', desc: 'Tokenization, attention, the transformer, GPT, pretraining, fine-tuning, alignment and efficient inference: how large language models are built.' },
    { id: 'apps', name: 'Building with LLMs', short: 'LLM Apps', color: '#f472b6', icon: 'bolt', desc: 'Using LLMs in real software: prompting, retrieval-augmented generation (RAG), agents with tools, and evaluation.' },
    { id: 'spec', name: 'Specializations', short: 'Specializations', color: '#fb923c', icon: 'layers', desc: 'Branch-out areas that build on the core: reinforcement learning, generative models, vision transformers, recommender systems, MLOps and responsible AI.' }
  ];

  MLA.data.concepts = MLA.data.concepts || [];

  // Content registry -------------------------------------------------------
  const MD_FIELDS = ['why', 'when', 'pitfalls', 'summaryMd'];
  function d(s) { return MLA.util ? MLA.util.dedent(s) : String(s || '').trim(); }

  function normalizeProblem(p, c, i) {
    p.concept = c.id;
    p.order = i;
    p.difficulty = p.difficulty || 'easy';
    p.prompt = d(p.prompt);
    p.starter = d(p.starter) + '\n';
    p.solution = d(p.solution) + '\n';
    if (p.prelude) p.prelude = d(p.prelude) + '\n';
    if (p.explain) p.explain = d(p.explain);
    p.hints = p.hints || [];
    p.tests = (p.tests || []).map((t, k) => {
      const o = Object.assign({}, t);
      if (o.setup) o.setup = d(o.setup);
      if (o.code) o.code = d(o.code);
      if (!o.name) o.name = (o.hidden ? 'Hidden test ' : 'Example ') + (k + 1);
      return o;
    });
    return p;
  }
  // Stable section ids. Saved progress (sections read, answered checks) is keyed by these ids, so lessons can
  // gain or reorder sections without shifting anyone's progress. Ids come from the title (checks: the question).
  function slugify(s) { return String(s || '').toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64).replace(/-+$/, ''); }
  function sectionIds(sections) {
    const seen = {};
    return (sections || []).map(s => {
      let base = s.id || slugify(s.title) || (s.type === 'check' && s.question ? 'check-' + MLA.util.hashString(String(s.question.q || '')).toString(36) : s.type || 'section');
      if (/^\d+$/.test(base)) base = 's-' + base;
      seen[base] = (seen[base] || 0) + 1;
      return seen[base] > 1 ? base + '-' + seen[base] : base;
    });
  }
  function normalize(c, i) {
    if (c._normalized) return c;
    c._order = i;
    c.level = c.level || 1;
    c.minutes = c.minutes || 20;
    c.prereqs = c.prereqs || [];
    for (const f of MD_FIELDS) if (c[f]) c[f] = d(c[f]);
    c.sections = (c.sections || []).map(s => {
      const o = Object.assign({}, s);
      if (o.md) o.md = d(o.md);
      if (o.code) o.code = d(o.code);
      if (o.after) o.after = d(o.after);
      return o;
    });
    sectionIds(c.sections).forEach((id, k) => { c.sections[k].id = id; });
    c.quiz = c.quiz || [];
    c.objectives = c.objectives || [];
    c.takeaways = c.takeaways || [];
    // case studies and sample projects live in js/data/projects.js, keyed by lesson id
    c.cases = ((MLA.data.cases || {})[c.id] || c.cases || []).map(k => Object.assign({}, k, { story: d(k.story), lesson: d(k.lesson) }));
    c.projects = ((MLA.data.projects || {})[c.id] || c.projects || []).map(p => Object.assign({}, p));
    c.problems = (c.problems || []).map((p, k) => normalizeProblem(p, c, k));
    c._normalized = true;
    return c;
  }

  MLA.content = {
    add(list) { for (const c of list) MLA.data.concepts.push(c); },
    sectionIds,
    ready: false,
    init() {
      const concepts = MLA.data.concepts;
      concepts.forEach(normalize);
      const problems = [];
      for (const c of concepts) for (const p of c.problems) problems.push(p);
      MLA.data.problems = problems;
      MLA.content.byId = {}; concepts.forEach(c => { MLA.content.byId[c.id] = c; });
      MLA.content.problemById = {}; problems.forEach(p => { MLA.content.problemById[p.id] = p; });
      MLA.content.branchById = {}; MLA.data.branches.forEach(b => { MLA.content.branchById[b.id] = b; });
      MLA.content.graph = MLA.graph.build(concepts, MLA.data.branches);
      MLA.content.ready = true;
      return MLA.content;
    }
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
