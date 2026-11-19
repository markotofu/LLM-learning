# ML Atlas

An interactive mind map for learning machine learning from first principles, through neural networks, to building and using large language models. Every concept has a lesson with plain-language explanations, formulas with every symbol defined, runnable code, interactive demos, a quiz, and LeetCode-style coding problems that run **in your browser** — no installation and no server.

It is a static site: it runs on GitHub Pages, from any web server, or straight from the files on your computer.

## What's inside

| | |
| --- | --- |
| **Concept map** | 64 concepts in 8 branches, connected by their prerequisites. Three views: mind map, prerequisite lanes, and outline. Search, highlight the chain of prerequisites for any concept, and set a learning goal. |
| **Lessons** | Each concept explains *why* it matters, how it works, when to use it, common pitfalls and key takeaways. Every formula comes with a "where" legend that defines each symbol. |
| **Runnable code** | Code samples run in the browser with real Python 3.12 + NumPy (via [Pyodide](https://pyodide.org)). PyTorch and API examples are marked "run locally". |
| **Interactive demos** | 15 visual demos: gradient descent, activation functions, line fitting, overfitting, k-NN, k-means, classification thresholds, softmax/temperature/top-p, attention, BPE tokenization, convolution, a single neuron, positional encodings, quantization, embeddings. |
| **Practice problems** | 143 problems (easy / medium / hard) with a LeetCode-style editor: **Run** checks the visible examples, **Submit** runs hidden tests too. Hints, worked solutions and explanations. Drafts are saved automatically. |
| **Quizzes** | 327 questions (multiple choice, multi-select, true/false, fill-in, ordering, matching) with explanations. Passing a quiz and solving a problem completes a concept. |
| **Learning tracks** | Guided paths: ML Foundations, Classical ML Practitioner, Deep Learning from Scratch, Build an LLM from Scratch, LLM Application Engineer, Fine-tuning & Alignment, the Complete Atlas — or set any concept as a custom goal. |
| **Flashcards** | 259 glossary terms, reviewed with spaced repetition. |
| **Playground** | A multi-file Python scratchpad with templates (NumPy, linear regression, matplotlib, softmax & attention, pandas, scikit-learn). |
| **Progress** | XP, levels, streaks and achievements. Everything is stored in your browser (localStorage). |
| **Sync across devices** | Export your progress and code as a file (or a copyable sync code) and import it on another device — imports *merge* by default, so nothing is lost. You can also download all your code as one `.py` file. |
| **Works offline** | Installable as an app (PWA). After the first visit the site and the Python runtime are cached. |

### The branches

1. **Start Here** — what machine learning is.
2. **Math Foundations** — vectors & matrices, derivatives & the chain rule, probability, statistics, entropy & cross-entropy.
3. **Python & Data** — Python essentials, NumPy, data preparation, feature engineering.
4. **Classical Machine Learning** — supervised learning, loss functions, gradient descent, linear & logistic regression, overfitting, regularization, evaluation metrics, cross-validation, k-NN, decision trees, ensembles, SVMs, naive Bayes, k-means, PCA, the scikit-learn workflow.
5. **Neural Networks & Deep Learning** — perceptrons, activations, MLPs, backpropagation, autograd, training loops, optimizers, normalization & dropout, PyTorch, CNNs, RNNs, embeddings.
6. **Transformers & LLMs** — text representation, language models, tokenization (BPE), attention, multi-head attention, positional encoding (incl. RoPE), the transformer block, GPT, decoding, pretraining & scaling laws, fine-tuning, LoRA, RLHF & DPO, quantization, and a capstone: build a tiny GPT.
7. **Building with LLMs** — prompting & structured output, retrieval-augmented generation (RAG), agents & tool use, evaluating LLM applications.
8. **Specializations** — reinforcement learning, generative models (VAEs, GANs, diffusion), vision transformers & CLIP, recommender systems, MLOps, responsible AI.

## Run it

**On the web:** once deployed (below), open `https://<your-user>.github.io/<your-repo>/`.

**On your computer**, either:

- double-click **`Open ML Atlas.bat`** (Windows) or open `index.html` in a browser — no install needed; or
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

No build step is involved: the site is plain HTML, CSS and JavaScript (Vue 3 is vendored in `vendor/`), and it uses hash-based URLs, so it works under any sub-path.

## Your progress and data

- Progress, notes, code drafts and settings are saved in your browser's `localStorage` under the key `mla.state.v1`. Nothing is sent anywhere.
- **Progress & Sync → Export progress / Sync code**: download a backup file, or copy a compact *sync code* and paste it on another device. Import offers **Merge** (keeps the best of both: solved problems, best quiz scores, newest drafts) or **Replace**. **Export all my code (.py)** downloads every solution and playground file as one Python file.
- Clearing your browser's site data erases local progress, so export a backup first.

## Project structure

```text
index.html               the single page; loads every script in order (no bundler)
css/app.css              all styles, with dark / darker / light themes
vendor/                  Vue 3 (global build)
js/core/                 utilities & markdown renderer, syntax highlighter, store (localStorage,
                         export/import/merge), router, UI helpers, gamification, progress logic
js/engines/              graph layout (DAG, mind map, lanes), Python runner (Pyodide in a Web
                         Worker + the test harness), quiz grading, spaced repetition (SM-2)
js/data/                 content: branches, concepts-*.js (lessons, quizzes, problems),
                         glossary, learning tracks, achievements
js/views/                Vue components: map, lesson, problem editor, practice list,
                         playground, dashboard, tracks, glossary, flashcards, profile, self-test
js/app.js                app shell: navigation, onboarding, Python status
sw.js, manifest.webmanifest   offline support and installable app
serve.js                 zero-dependency local web server
tests/                   Node + CPython test suites (see below)
tools/make-icons.js      regenerates the app icons
docs/CONTENT_GUIDE.md    how to write and test new lessons and problems
```

## Tests

Requirements: Node.js 18+ and Python 3 with `numpy`, `pandas` and `scikit-learn`.

```bash
npm test
```

| Suite | Checks |
| --- | --- |
| `tests/graph.test.js` | the prerequisite graph: cycles, depths, ancestors, both map layouts |
| `tests/content.test.js` | every concept, quiz question, problem, demo, glossary term and track is well-formed; prerequisites exist; **every formula has a symbol legend**; every concept has glossary terms |
| `tests/views.test.js` | every Vue template compiles; all markdown renders; store export/import/merge round-trips |
| `tests/problems.test.js` | runs in CPython: every reference solution passes all of its tests, every starter fails at least one, every code sample compiles and every runnable sample executes |

You can limit the problem checks to some concepts: `node tests/problems.test.js attention gpt`.

Inside the app, **Progress & Sync → Run self-test** runs every reference solution and runnable sample in the in-browser Python, to confirm the live runtime agrees.

## Adding or editing content

See [docs/CONTENT_GUIDE.md](docs/CONTENT_GUIDE.md) for the concept format, markdown features (formulas with legends, callouts, tables), quiz types, the problem/test format, and the authoring checklist.

## License

The package is declared MIT in `package.json`. Add a `LICENSE` file before publishing if you want to make that explicit.
