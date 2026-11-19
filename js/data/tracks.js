/* ML Atlas – learning tracks: a goal set of concepts; the app adds every prerequisite automatically */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};

  MLA.data.tracks = [
    { id: 'foundations', icon: '📐', title: 'Machine Learning Foundations', short: 'The maths, Python and first models', level: 'Beginner', goals: ['linear-regression', 'logistic-regression', 'overfitting', 'evaluation-metrics'],
      desc: 'Start here if you are new. Vectors, derivatives, probability and Python with NumPy, then your first real models — linear and logistic regression — and how to tell whether a model is actually any good.' },
    { id: 'classical', icon: '🌳', title: 'Classical ML Practitioner', short: 'Trees, ensembles, clustering, scikit-learn', level: 'Beginner → Intermediate', goals: ['sklearn-workflow', 'ensembles', 'svm', 'naive-bayes', 'kmeans', 'pca', 'cross-validation', 'regularization'],
      desc: 'The toolbox that still wins on spreadsheets and tabular data: decision trees, random forests and boosting, SVMs, Naive Bayes, clustering, PCA, honest validation and the scikit-learn workflow.' },
    { id: 'deep-learning', icon: '🧠', title: 'Deep Learning from Scratch', short: 'Neurons → backprop → CNNs & RNNs', level: 'Intermediate', goals: ['autograd', 'optimizers', 'normalization', 'pytorch', 'cnn', 'rnn', 'embeddings'],
      desc: 'Build neural networks with nothing but NumPy: the forward pass, backpropagation, a tiny autograd engine, training loops, Adam, normalization, then convolutional and recurrent networks — and how the same ideas look in PyTorch.' },
    { id: 'build-llm', icon: '🤖', title: 'Build an LLM from Scratch', short: 'Tokens → attention → a tiny GPT', level: 'Intermediate → Advanced', goals: ['mini-gpt', 'pretraining'],
      desc: 'Everything a GPT-style model is made of: tokenization, embeddings, attention, multi-head attention, positional encodings, transformer blocks, next-token training and decoding — ending with a tiny GPT you assemble yourself in NumPy.' },
    { id: 'llm-apps', icon: '🛠️', title: 'LLM Application Engineer', short: 'Prompting, RAG, agents, evaluation', level: 'Intermediate', goals: ['rag', 'agents', 'llm-evaluation'],
      desc: 'Use language models in real software: how decoding and prompting work, retrieval-augmented generation over your own documents, tool-using agents, and how to measure whether your LLM feature is getting better.' },
    { id: 'fine-tuning', icon: '🎯', title: 'Fine-tuning & Alignment', short: 'SFT, LoRA, RLHF/DPO, quantization', level: 'Advanced', goals: ['lora', 'alignment', 'quantization'],
      desc: 'Adapt and ship models: pretraining and scaling laws, supervised fine-tuning, parameter-efficient LoRA, preference tuning with RLHF and DPO, and quantization for cheap, fast inference.' },
    { id: 'everything', icon: '🗺️', title: 'The Complete Atlas', short: 'Every concept, in learning order', level: 'All levels', all: true, goals: [],
      desc: 'Every concept on the map, ordered so that each one comes after everything it builds on. Includes the specializations: reinforcement learning, generative models, vision transformers, recommender systems, MLOps and responsible AI.' },
    { id: 'custom', custom: true, hidden: true, icon: '📍', short: 'A goal you picked on the map',
      get title() { const id = MLA.store && MLA.store.state.profile.customGoal; const c = id && MLA.content && MLA.content.byId ? MLA.content.byId[id] : null; return 'Goal: ' + (c ? c.title : 'custom'); },
      get goals() { const id = MLA.store && MLA.store.state.profile.customGoal; return id ? [id] : []; },
      desc: 'A personal goal chosen from the concept map: that concept plus everything it depends on.' }
  ];
})(typeof globalThis !== 'undefined' ? globalThis : window);
