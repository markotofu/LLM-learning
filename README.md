# ML Atlas

An interactive mind map for learning machine learning from first principles, through neural networks, to building and using large language models. Every concept has a lesson with plain-language explanations, formulas with every symbol defined, runnable code, interactive demos, a quiz, and LeetCode-style coding problems that run **in your browser**: no installation and no server.

It is a static site: it runs on GitHub Pages, from any web server, or straight from the files on your computer.

## What's inside

| | |
| --- | --- |
| **Concept map** | 123 concepts in 8 branches, including three capstone projects, connected by their prerequisites. Three views: mind map, prerequisite lanes, and outline. Search, highlight the chain of prerequisites for any concept, and set a learning goal. Zoom to one branch and the lessons it builds on, hide the branches you have finished, or turn on a heat map that colours each lesson by how far you got. Recommended prerequisites are drawn dotted, apart from the required ones. |
| **Lessons** | Each concept explains *why* it matters, how it works, when to use it, common pitfalls and key takeaways, with worked examples, runnable code and quick checks along the way (a typical lesson is about 1,200 words). Every formula comes with a "where" legend that defines each symbol. |
| **Runnable code** | 356 code samples run in the browser with real Python 3.12 + NumPy (via [Pyodide](https://pyodide.org)). Samples that load a dataset print a few rows of it first, so you can see what you are working with. Another 30 examples use libraries the browser cannot run (PyTorch, torchvision, Hugging Face, PyTorch Geometric, Stable-Baselines3, vLLM) and are marked "run locally": they show the real library calls behind each lesson's ideas, such as fine-tuning a pretrained CNN, zero-shot classification with CLIP, a GCN on a citation graph, counting objects with a pretrained detector, transcribing audio, training PPO, loading a model in 4 bits and serving one with vLLM. |
| **Interactive demos** | 28 visual demos: gradient descent, activation functions, line fitting, overfitting, k-NN, k-means, classification thresholds, softmax/temperature/top-p, attention, BPE tokenization, convolution, a single neuron, positional encodings, quantization, embeddings, calibration and temperature scaling, the diffusion forward process, LLM batching on the roofline, PPO's clipped objective, eigenvectors and the SVD of a 2×2 matrix, gradient descent and momentum against the condition number, Bayesian updating (Beta–Binomial), Gaussian-process regression, PCA, learning-rate schedules, the bias–variance trade-off over many training sets, a decision-tree split explorer, and backpropagation step by step. The threshold demo also shows the precision–recall curve. Plots that take clicks also work from the keyboard: focus the plot with Tab, move the cross-hair with the arrow keys and press Enter. |
| **Diagrams** | 26 theme-aware diagrams: rules versus learning, names and objects in Python, random and time-based splits, a PyTorch training step, documents as a matrix of word counts, the four families of generative models, a vision transformer, split-apply-combine, choosing a chart, an encoder-decoder with attention, bidirectional and causal attention masks, how joins duplicate or drop rows, MCP hosts and servers, equivariance, the transformer block, the RAG pipeline, the agent loop, diffusion training and sampling, the U-Net, the ML lifecycle, cross-validation folds, a LoRA adapter, the LLM training pipeline, an LSTM cell, mixture-of-experts routing, and data / tensor / pipeline parallelism. |
| **Step-through walkthroughs** | 18 algorithms you can follow one step at a time: gradient descent on a line, k-NN, logistic regression, the perceptron, how a decision tree picks its splits, gradient boosting, naive Bayes, EM for Gaussian mixtures, a linear SVM finding its margin, PCA by power iteration, a small network learning XOR, an RNN carrying a signal through time, Adam, attention for one token, the KV cache, writing text one token at a time, retrieval for RAG, and Q-learning. Five of them (linear and logistic regression, k-NN, the SVM and PCA) also run on **your own data**: a practice dataset or a CSV file you uploaded, with sensible columns picked for you. Each is an animated flowchart kept in step with its code (the running line is highlighted), the formula with the numbers of the current step, the variables, a small plot of what the model sees, and a legend for every symbol on screen. Every box says what the step is for and every arrow why the algorithm goes there next; at a few moments you are asked to predict what happens before you see it. Change a setting such as the learning rate, k or the temperature and the whole run is recomputed: every number comes from running the algorithm in your browser. **Play** moves at the speed you pick, from Sluggish (9 seconds a step) and Super slow (5 seconds) up to Fast, and the moves on screen slow down with it; or pick **With the voice**, and each step is read aloud with your on-device voice while the card moves on as the voice finishes. On a phone, the code shows only the lines of the current step and the step buttons stay at the bottom of the screen. |
| **Charts** | 155 charts in 109 lessons plot numbers computed by real experiments: loss curves and learning rates, decision boundaries, bias and variance, precision and recall, scaling laws, attention cost, quantization error, and many more. Each has a "Show the numbers" table with the underlying data, labels on the end of each line and on peaks and lows, and a "Describe this chart in words" summary that screen readers also announce. |
| **Projects and case studies** | 27 sample projects to build on your own machine with real public data (house prices, a spam filter, churn, fraud, forecasting, a CNN, a BPE tokenizer, semantic search, a small GPT, an LLM evaluation suite, a CSV agent, a monitored model service and more), each with a goal, the data, steps you tick off, a finish line, stretch goals and your own notes (saved, exported and merged like the rest of your progress); three of them start in the browser on built-in practice datasets; and 31 documented case studies of machine learning in the real world (the Netflix Prize, Google Flu Trends, AlexNet, Chinchilla, InstructGPT, AlphaFold, Gender Shades, a chatbot ruling against an airline and others), each with what to take from it and its source. They appear at the end of the lessons they belong to, and all together on the **Projects & Cases** page. |
| **Read aloud** | Listen to a whole lesson, or to any single box, while you do something else. Formulas are read in words ("the sum over i of x sub i squared"), each symbol legend is read with its formula, tables row by row, quick checks pause before the answer, and charts are described: what each line does, where it peaks, where lines cross. Step-through walkthroughs are read step by step, whether you listen to the whole lesson or press the walkthrough's own speaker button, and the card follows the voice. The player shows the sentence being read and the lesson highlights it; each lesson remembers where you stopped; when a lesson ends, the next one in your track is offered (or plays by itself). Lock-screen, notification and headphone buttons work where the browser supports media controls. It uses only the voices installed on your device (the Web Speech API): no lesson text leaves the browser and no AI service is involved. Pick a voice and speed in **Progress & Sync → Settings**. |
| **Predict first** | 158 code samples ask you to predict their output before you run them; committing to a guess first is one of the most reliable ways to remember the result. |
| **Practice problems** | 270 problems (easy / medium / hard) with a LeetCode-style editor: **Run** checks the visible examples, **Submit** runs hidden tests too. Hints, worked solutions and explanations. Drafts are saved automatically. The three capstones are split into checkpoints: one checked task per project step, with progress per step. |
| **Quizzes** | 877 questions (multiple choice, multi-select, true/false, fill-in, ordering, matching) with explanations. Passing a quiz and solving a problem completes a concept. |
| **Search** | Press Ctrl+K (Cmd+K on a Mac) or "/" anywhere, or use the search button: one box searches pages, lessons, sections, glossary terms, problems, projects, case studies and tracks, and the full text of the lessons, and opens results at the right section. |
| **Learning tracks** | Guided paths: ML Foundations, Classical ML Practitioner, Data Science & Experiments, ML Theory & Probabilistic Methods, Deep Learning from Scratch, Build an LLM from Scratch, LLM Application Engineer, Fine-tuning, Alignment & Serving, Frontier Deep Learning, the Complete Atlas, or set any concept as a custom goal. |
| **Placement test** | Optional, from the welcome dialog or the dashboard: about 12 to 15 questions from the lessons' own quizzes, chosen one at a time along the prerequisite graph. Lessons you already know are marked as known (no XP, not completed), so tracks and recommendations start at the first gap; each mark says why, and every mark can be undone. |
| **Review** | Flashcards for 683 glossary terms, plus **mistakes to revisit**: every quiz question, quick check or prediction you get wrong, and every problem you could only solve with the solution, comes back after a day, then a week, then a few weeks (spaced repetition), until it sticks. |
| **Playground** | A multi-file Python scratchpad with templates (NumPy, linear regression, matplotlib, softmax & attention, pandas, scikit-learn). **Upload your own data** (CSV, TSV, Excel .xlsx and .xls, JSON, Parquet or text, up to 20 MB per file and 60 MB in total) and read it in any code as `data/<file name>`, for example `pd.read_excel("data/sales.xlsx")`; three built-in practice datasets are there too. |
| **Progress** | XP, levels, streaks and achievements. Everything is stored in your browser (localStorage). |
| **Looks and themes** | Two looks: **Baroque**, after baroque ceiling frescoes (gold, clouds and serif type), and **Classic**, the original clean design. Each has a light, a dark and a night theme. Switch looks with the palette button in the top bar, or pick both in **Progress & Sync → Settings**. |
| **Sync across devices** | Export your progress and code as a file (or a copyable sync code) and import it on another device: imports *merge* by default, so nothing is lost. You can also download all your code as one `.py` file. The export file carries everything you did, uploaded datasets included; a sync code carries everything except the datasets. |
| **Works offline** | Installable as an app (PWA). The first page loads a small lesson index (about 60 KB compressed) instead of every lesson; each lesson loads when you open it (the walkthrough code with the first lesson that has one), and the rest download quietly in the background so that, after the first visit, the site and the Python runtime work offline. |

### The branches

1. **Start Here**: what machine learning is.
2. **Math Foundations**: vectors & matrices, eigenvectors & the SVD, derivatives & the chain rule, matrix calculus, optimization theory (convexity, conditioning, Newton's method, Lagrange multipliers), probability, statistics, statistical inference (confidence intervals, hypothesis tests, A/B tests), entropy & cross-entropy, discrete maths (sets, counting, graphs, complexity).
3. **Python & Data**: Python essentials, NumPy, pandas & exploratory data analysis, SQL & data engineering, data preparation, feature engineering.
4. **Classical Machine Learning**: supervised learning, loss functions, maximum likelihood & MAP, Bayesian inference (MCMC, variational inference), gradient descent, linear & logistic regression, overfitting, generalization theory (PAC, VC dimension, double descent), regularization, evaluation metrics, calibration, uncertainty & conformal prediction, cross-validation, data-centric ML (augmentation, imbalance, few labels), k-NN, decision trees, ensembles, SVMs, Gaussian processes & Bayesian optimization, naive Bayes, k-means, Gaussian mixtures & EM (with DBSCAN and hierarchical clustering), PCA, graphical models & hidden Markov models, GLMs & survival analysis, the scikit-learn workflow, and a capstone: an end-to-end tabular project.
5. **Neural Networks & Deep Learning**: perceptrons, activations, MLPs, backpropagation, autograd, training loops, optimizers, normalization & dropout, PyTorch, CNNs, RNNs, embeddings, sequence-to-sequence models (where attention began), knowledge distillation & pruning, self-supervised representation learning.
6. **Transformers & LLMs**: text representation, language models, tokenization (BPE), attention, multi-head attention, positional encoding (incl. RoPE), the transformer block, GPT, BERT-style encoders, text embedding models, decoding, pretraining & scaling laws, training at scale (mixed precision, parallelism), efficient attention & long context, state-space models & hybrids (Mamba, linear attention), mixture of experts, multimodal models (vision, audio, any-to-any), fine-tuning, LoRA, RLHF & DPO, reasoning models, quantization, LLM inference & serving, AI hardware (GPUs, memory, kernels and compilers), and two capstones: build a tiny GPT, and fine-tune a model with LoRA while measuring adaptation and forgetting.
7. **Building with LLMs**: AI engineering (choosing models, APIs and SDKs, open models, frameworks), prompting & structured output, vector search, retrieval-augmented generation (RAG), agents & tool use, the Model Context Protocol (MCP), advanced agents (context engineering, planning, multi-agent systems, agent evaluation), evaluating LLM applications, LLMs in production (observability, monitoring, safety), and a capstone: build and evaluate a RAG system.
8. **Specializations**: reinforcement learning, deep RL (DQN, actor–critic, PPO), generative models (VAEs, GANs), diffusion models & flow matching, video understanding & generation, world models, robot learning & vision-language-action models, vision transformers & CLIP, object detection & segmentation, speech & audio, graph neural networks, recommender systems, learning to rank, time series & forecasting, anomaly detection, causal inference, contextual bandits & online experiments, 3D vision & neural rendering, audio & music generation, geometric & equivariant deep learning, AI for science, mechanistic interpretability (with a hands-on lesson), adversarial ML, security & privacy, MLOps, responsible AI, AI governance & regulation.

## Run it

**On the web:** once deployed (below), open `https://<your-user>.github.io/<your-repo>/`.

**On your computer**, either:

- double-click **`Open ML Atlas.bat`** (Windows) or open `index.html` in a browser: no install needed; or
- run a tiny local server (recommended: it also enables offline mode and the installable app):

  ```bash
  node serve.js          # or: npm start
  ```

  then open <http://localhost:8080>. The server also prints an address you can open on a phone or tablet on the same Wi-Fi.

The first time you run Python, the browser downloads the Python runtime (about 10 MB, from the jsDelivr CDN); after that it is cached. Everything else works without an internet connection.

## Deploy to GitHub Pages

1. Push this repository to GitHub (branch `main`).
2. In the repository, open **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**.
3. Every push to `main` now runs the full test suite and, if it passes, publishes the site (`.github/workflows/deploy.yml`).

No build step is involved in development: the site is plain HTML, CSS and JavaScript (Vue 3 is vendored in `vendor/`), and it uses hash-based URLs, so it works under any sub-path. The deploy job adds two steps to its own copy: `tools/snapshots.js` writes a readable static page for every lesson (`lessons/<id>.html`), a sitemap and robots.txt, so lessons can be found by search engines and previewed in chats; `tools/precompile.js` compiles the Vue templates ahead of time, so the published page's Content-Security-Policy can forbid eval. A separate weekly workflow checks every external link (`tools/check-links.js`).

## Your progress and data

- Progress, notes, code drafts and settings are saved in your browser's `localStorage` under the key `mla.state.v1`. Nothing is sent anywhere.
- Uploaded datasets are kept in the browser's IndexedDB (database `ml-atlas-datasets`); their names and sizes are part of the progress data. Python reads them from its in-browser file system, so the files never leave your device. (Reading an .xlsx file downloads the openpyxl package from PyPI once per session; the Python runtime itself comes from jsDelivr, never from this site's host.)
- **Progress & Sync → Export progress / Sync code**: download a backup file, or copy a compact *sync code* and paste it on another device. Import offers **Merge** (keeps the best of both: solved problems, best quiz scores, newest drafts) or **Replace**. **Export all my code (.py)** downloads every solution and playground file as one Python file.
- Clearing your browser's site data erases local progress, so export a backup first.

**Security.** The page sets a Content-Security-Policy (in a `<meta>` tag, because GitHub Pages cannot send headers) that allows scripts only from the site itself and from jsDelivr for the Python runtime, and the page has no inline scripts. The markdown renderer escapes all text and never passes raw HTML through, so notes and imported progress files are always displayed as plain text; imports are size-limited and checked before they are merged. Python runs in a Web Worker with no access to the page or to saved progress. `serve.js` serves only the files inside the project folder (no hidden files, no `node_modules`) and answers only GET and HEAD requests.

## Project structure

```text
index.html               the single page; loads every script in order (no bundler)
css/app.css              all styles: two looks, Baroque (default; themes Fresco, Vault and Lapis, with
                         light rays, clouds and gilding in CSS and inline SVG) and Classic (the original
                         design; themes Paper, Graphite and Midnight)
vendor/                  Vue 3 (global build)
js/core/                 utilities & markdown renderer, syntax highlighter, store (localStorage,
                         export/import/merge), router, UI helpers, gamification, progress logic,
                         narration (lesson to spoken text) and the read-aloud player engine
js/engines/              graph layout (DAG, mind map, lanes), Python runner (Pyodide in a Web
                         Worker + the test harness), quiz grading, spaced repetition (SM-2)
js/data/                 content: branches (and the lesson loader), catalog.js (generated index,
                         loaded with the page), concepts-*.js (lessons, quizzes, problems; loaded
                         on demand), projects.js (case studies, sample projects), glossary,
                         learning tracks, achievements
data/                    small synthetic practice datasets for the in-browser Python
js/views/                Vue components: map, lesson, problem editor, practice list,
                         playground, dashboard, tracks, glossary, flashcards, profile, self-test,
                         and the demos, diagrams, step-through walkthroughs and charts shown
                         inside lessons
js/app.js                app shell: navigation, onboarding, Python status
sw.js, manifest.webmanifest   offline support and installable app
serve.js                 zero-dependency local web server
tests/                   Node + CPython test suites (see below)
tools/                   lesson-patch.js (adds lesson sections, charts and quiz questions from a
                         plain-text patch), build-catalog.js, lesson-info.js, legacy-sections.js,
                         make-datasets.py, snapshots.js and precompile.js (deploy time),
                         check-links.js (weekly), visual-diff.js, make-icons.js
docs/CONTENT_GUIDE.md    how to write and test new lessons and problems
```

## Tests

Requirements: Node.js 18+ and Python 3 with `numpy`, `pandas` and `scikit-learn`.

```bash
npm test
```

| Suite | Checks |
| --- | --- |
| `tests/graph.test.js` | the prerequisite graph: cycles, depths, ancestors, both map layouts, recommended prerequisites, sub-graphs and zoom to a branch |
| `tests/content.test.js` | every concept, quiz question, problem, demo, chart, glossary term and track is well-formed; prerequisites exist; **every formula has a symbol legend**; every concept has glossary terms |
| `tests/views.test.js` | every Vue template compiles; all markdown renders, and hostile markdown (scripts, event handlers, `javascript:` links) comes out inert; store export/import/merge round-trips, and malformed, oversized or prototype-polluting imports are rejected |
| `tests/theme.test.js` | every text colour of every theme meets WCAG AA contrast (4.5:1) on every background |
| `tests/narrate.test.js` | every lesson can be read aloud: no markdown or raw math symbols are left in the spoken text, quick checks pause before the answer, every chart has a description, and the player engine plays, skips, pauses and stops with on-device voices only |
| `tests/walks.test.js` | every step-through walkthrough runs under every setting, and those that take your own data also run on generated datasets (including awkward ones), while unusable files and columns are refused with a clear message: its steps follow the flowchart's arrows and highlight real code lines, its numbers are finite, its plot labels stay inside the plot without overlapping, and every symbol on screen has a legend entry |
| `tests/serve.test.js` | the local server refuses path traversal (including encoded variants), hidden files, `node_modules`, directories and non-GET requests, and sends its security headers |
| `tests/problems.test.js` | runs in CPython: every reference solution passes all of its tests, every starter fails at least one, every code sample compiles, every runnable sample executes, and every "predict first" answer is confirmed by its sample's real output |

You can limit the problem checks to some concepts: `node tests/problems.test.js attention gpt`.

**Browser tests** (Playwright + Chromium; one-time setup: `npm install` and `npx playwright install chromium`):

- `npm run test:e2e` opens every lesson and page at phone width, runs a built-in accessibility check on each (controls and links with names and addresses, labelled form fields, text alternatives, unique ids; `tests/e2e/a11y.js`), checks that demos keep their proportions and diagram labels fit, and exercises theme switching, sync codes, saved-progress migration, "predict first", the mistakes queue and the in-browser Python. CI runs it before every deploy.
- `npm run test:visual [ref]` screenshots every demo and diagram in the dark and light themes, first from a git ref (default: the last commit) and then from your working tree, and reports what changed (`npx playwright show-report`). CI runs it on every push as a report-only job.

Inside the app, **Progress & Sync → Run self-test** runs every reference solution and runnable sample in the in-browser Python, to confirm the live runtime agrees.

## Adding or editing content

See [docs/CONTENT_GUIDE.md](docs/CONTENT_GUIDE.md) for the concept format, markdown features (formulas with legends, callouts, tables), quiz types, the problem/test format, how to add demos and diagrams, and the authoring checklist.

Planned and possible improvements, ranked by impact and effort, are in [docs/ROADMAP.md](docs/ROADMAP.md).

## Copyright, license, terms and privacy

© 2026 markotofu. See [LICENSE](LICENSE) for the details:

- **Source code** (the app's JavaScript, HTML, CSS, tests and tools) is released under the **MIT License**.
- **Educational content** (lessons, quizzes, practice problems, lesson code samples, glossary and flashcards in `js/data/`) is **all rights reserved**: free for personal, non-commercial learning in the app, but not for republishing without permission.
- **Third-party software** (Vue in `vendor/`, and Pyodide with its Python packages loaded at run time) keeps its own license: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The app includes a **Terms of Use** page (`#/terms`) and a **Privacy Policy** page (`#/privacy`), linked from the sidebar and from Progress & Sync. Their texts, the copyright holder, the contact link and the "last updated" date live in one place: `js/data/legal.js`. The privacy policy describes what the app really does: progress stays in the browser's local storage, no cookies or analytics, and only the web host and the Python CDN (jsDelivr) see standard request data, so update it if you add analytics, accounts, a backend or another third-party service. These texts are a plain-language starting point, not legal advice; have them reviewed if you need certainty for your jurisdiction.
