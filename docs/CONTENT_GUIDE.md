# Content guide

How to add or edit lessons, quizzes and practice problems in ML Atlas. All content lives in plain JavaScript files in `js/data/`; there is no build step. After any change, run `npm test` (see the end of this guide).

## 1. Where content lives

| File | Contents |
| --- | --- |
| `js/data/branches.js` | the 8 branches (id, name, colour, icon) and the content registry `MLA.content.add([...])` |
| `js/data/concepts-*.js` | the concepts, in topic files: `start-math`, `python-data`, `ml-core`, `ml-more`, `ml-unsup`, `nn-core`, `nn-more`, `llm-core`, `llm-more`, `apps`, `spec`, plus the later additions `foundations-plus` (eigenvectors & SVD, MLE & MAP, statistical inference, pandas & EDA, mixtures & EM), `llm-plus` (seq2seq, BERT, text embeddings, training at scale, reasoning models, efficient attention, mixture of experts), `spec-plus` (interpretability, time series, anomaly detection, detection & segmentation, GNNs, speech & audio, causal inference), `theory-plus` (matrix calculus, optimization theory, generalization theory, Bayesian inference, Gaussian processes), `practice-plus` (calibration & uncertainty, data-centric ML, SQL & data engineering, learning to rank, vector search) and `deep-plus` (distillation & pruning, self-supervised learning, diffusion models, deep RL, ML security & privacy, LLM serving), `frontier-plus` (multimodal models, advanced agents, state-space models) and `capstones` (end-to-end tabular project, RAG system with evaluation, LoRA fine-tuning). A concept's `branch` field, not its file, decides where it appears on the map. |
| `js/data/glossary.js` | glossary terms (also the flashcard deck) |
| `js/data/legal.js` | Terms of Use, Privacy Policy and the copyright notice (owner, year, contact, last-updated date) |
| `js/data/tracks.js` | learning tracks (goal concepts; the path is computed from prerequisites) |
| `js/data/achievements.js` | achievements |

A **new content file** must be added to `index.html` with the other `concepts-*.js` scripts, **before** `glossary.js`. The order of concepts across files does not matter: the prerequisite graph is built after everything has loaded.

## 2. The concept object

```js
{
  id: 'attention',                 // kebab-case, unique; used in URLs (#/concept/attention)
  branch: 'llm',                   // start | math | code | ml | dl | llm | apps | spec
  title: 'Attention & Self-Attention',
  short: 'Attention',              // optional shorter label for the map
  level: 2,                        // 1 beginner, 2 intermediate, 3 advanced
  minutes: 50,                     // rough reading + practice time
  prereqs: ['embeddings', 'activation-functions'],   // concept ids; must exist; no cycles
  tags: ['attention', 'query', 'key', 'value'],      // used by search
  summary: 'One or two sentences shown on the map and in lists.',
  why: `Why this matters — markdown.`,
  objectives: ['What the learner will be able to do', '...'],
  sections: [ /* see below */ ],
  when: `- When to use it (markdown list)`,
  pitfalls: `- Common mistakes (markdown list)`,
  takeaways: ['3–5 one-line key points'],
  quiz: [ /* 4+ questions, see below */ ],
  problems: [ /* 1+ problems, see below */ ],
  refs: [{ title: 'Paper or tutorial', url: 'https://…', note: 'optional' }]
}
```

### Section types

```js
{ type: 'text', title: 'How it works', md: `markdown…` }
{ type: 'code', title: 'Example', md: `optional intro`, code: `python…`, after: `optional note shown after the output` }
{ type: 'code', title: 'Run locally', lang: 'pytorch', code: `…` }   // shown, not run (PyTorch is not in the browser)
{ type: 'code', title: 'Real API call', run: false, md: `…`, code: `…` }   // plain Python, shown but not runnable
{ type: 'demo', title: 'Interactive: attention', md: `optional intro`, demo: 'attention' }
{ type: 'check', question: { /* a quiz question, any type except match */ } }
{ type: 'chart', title: 'Loss per step for different learning rates', md: `optional intro`, chart: { /* spec */ }, after: `what to notice` }
```

**Charts** are static, theme-aware SVG plots drawn by `js/views/charts.js` from numbers stored in the lesson. Compute the numbers with a real experiment or formula (keep the script that generated them in `tools/charts/`), never invent them, and say in `after` what the reader should notice. The spec:

```js
{ kind: 'line' | 'scatter' | 'bar',
  x: [ … ],                                   // shared x values (category labels for bars)
  series: [{ name: 'legend text', y: [ … ] },  // one y per x; null leaves a gap in a line
           { name: '…', x: [ … ], y: [ … ], dash: true, scatter: true }],   // own x values; dashed; dots only
  xLabel: '…', yLabel: '…',                   // required
  xLog: true, yLog: true, xMin, xMax, yMin, yMax, points: false,
  vlines: [{ x, label }], hlines: [{ y, label }] }
```

`tests/content.test.js` checks every spec (known kind, matching lengths, finite values, positive values on log axes, a title and both axis labels). Data outside the axis limits is clipped at the plot edge, long legends wrap, and wide log ranges label every second or third decade. Under each chart a "Show the numbers" table lists the data for screen readers and for anyone who wants the values. Check new charts in the browser: no overlapping labels in light and dark themes and at phone width.

**Writing many sections at once.** `node tools/lesson-patch.js patch.txt` inserts sections, quiz questions and charts from a plain-text patch file, so Markdown, Python and JSON never need escaping by hand. A patch is a list of operations:

```text
### INSERT concept=knn after="Scaling is not optional"      (or before="…", or at=end)
@@ text | Section title
Markdown…
@@ code | Section title
optional intro
%% code
python…
%% after
what the output shows
%% predict
{"type": "mcq", "q": "…", "options": ["…"], "answer": 0, "explain": "…", "expect": ["…"]}
@@ chart | Chart title
%% spec
{"kind": "line", "x": [ … ], "series": [ … ], "xLabel": "…", "yLabel": "…"}
@@ check
{"type": "tf", "q": "…", "answer": true, "explain": "…"}
@@ quiz
{"type": "mcq", "q": "…", "options": ["…"], "answer": 1, "explain": "…"}
### REPLACE concept=knn section="Old title"                  (the blocks below replace that section)
```

Afterwards run `node --check` on the changed content files, `node tools/legacy-sections.js`, `node tests/content.test.js` and `MLA_SAMPLE_OUTPUT=out.json node tests/problems.test.js <concepts>`, and read the sample outputs against the text you wrote.

**A new lesson.** Write its metadata (id, branch, title, summary, why, objectives, when, pitfalls, takeaways, refs), an empty `quiz: []`, its practice problems and a single text section titled `placeholder` in a content file, and add the file to `index.html`. Then write the sections in a patch that starts with `### REPLACE concept=<id> section="placeholder"`, and add quiz questions with `### INSERT concept=<id> at=end`. Every lesson also needs at least one glossary term in `js/data/glossary.js`, linked to it by its id. See `js/data/concepts-roadmap.js` for examples.

**Predict first.** A runnable Python code section can carry a `predict` question, shown above the code. The learner commits to an answer, reads the explanation, then runs the code to see it happen:

```js
{ type: 'code', title: 'Bayes by counting', code: `…`, predict: {
    type: 'mcq', q: 'If you test positive, how likely is it that you have the condition?',
    options: ['About 99%', 'About 95%', 'About 50%', 'Under 20%'], answer: 3,
    explain: 'Only about 17%: …',
    expect: ['P(condition | positive) ~ 0.167'] } }   // text that must appear in the sample's real output
```

`expect` is checked by `tests/problems.test.js` (CPython) and by the in-app self-test (Pyodide), so a prediction cannot silently drift away from what the code prints. Keep each expected string short and robust: avoid digits that depend on a library version.

**Section ids and saved progress.** Every section gets an id from its title (quick checks: from a hash of the question). Progress (sections read, answered checks and predictions, the mistakes queue) is stored by these ids, so you can insert or reorder sections freely. Keep section titles unique within a lesson, and do not rename a section without a reason: a renamed section starts again as unread. Old saved data that used positions is migrated through `js/data/legacy-sections.js`; if `tests/content.test.js` reports that a lesson's legacy layout no longer matches (because you changed its first sections), run `node tools/legacy-sections.js` and commit the result.

Runnable code samples execute in the in-browser Python (Pyodide 0.27, Python 3.12, NumPy; pandas, scikit-learn and matplotlib load automatically when imported). Keep them fast (a few seconds at most in the browser) and print something meaningful. A run stops after the learner's time limit (20 s by default, set under Progress & Sync). A sample that genuinely needs longer, such as the LoRA capstone's training runs, can declare `timeoutSec: 120` (20 to 300; in a patch file, `%% timeout 120`), and the in-app self-test allows it twice that. Say in the text that the sample is slow, and expect the browser to be about three times slower than desktop Python.

Available demos (28):

- In `js/views/demos.js`: `gradient-descent`, `activations`, `linear-fit`, `overfit-poly`, `knn`, `kmeans`, `threshold`, `softmax`, `attention`, `bpe`, `convolution`, `neuron`, `positional`, `quantization`, `embeddings`, `calibration`, `diffusion`, `roofline` and `ppo-clip`.
- In `js/views/demos-more.js`: `eigen`, `conditioning`, `beta`, `gp`, `pca`, `lr-schedule`, `backprop`, `bias-variance` and `tree-split`. This file uses the canvas helpers that `demos.js` exports as `MLA.demoKit`.

Available diagrams (12), defined in `js/views/diagrams.js`: `dg-transformer`, `dg-rag`, `dg-agent`, `dg-diffusion`, `dg-unet`, `dg-lifecycle`, `dg-cv`, `dg-lora`, `dg-pipeline`, `dg-lstm`, `dg-moe` and `dg-parallel`. Use them like demos, for example `{ type: 'demo', title: 'Diagram: the agent loop', demo: 'dg-agent' }`. A section can also show only a diagram followed by text, using `after` (see the `rag` and `mlops` lessons).

Use a **demo** when the learner should move a slider and watch something change, a **diagram** for the structure of a system, and a **chart** for numbers: a curve, a comparison or a scaling law.

### Adding a demo

Add a component named `C['demo-<name>']` to `demos-more.js`:

1. Include the `redrawOnChange` mixin and a `draw()` method that calls `setup(canvas, W, H)` and `palette()`, so the colours follow the theme.
2. Put a short caption in `<p class="small muted mt">`. The lighter `dim` colour fails WCAG contrast on demo cards.
3. Check it in the browser: in all three themes, and at a 375 px phone width with no sideways page scrolling.

### Adding a diagram

Call `diagram(name, viewBox, ariaLabel, body, caption)` in `diagrams.js`:

- Build `body` from the `box()`, `arrow()`, `label()` and `plus()` helpers, inside the fixed 820-wide coordinate system.
- Use the theme classes (`dg-box`, `dg-acc`, `dg-grn`, `dg-amb`, `dg-pur`, `dg-red`, `dg-cyn`), never fixed colours.
- Keep every label inside its box. Diagram text is 14 px bold for box titles and 12 px for small lines, so allow about 8 px per character for titles.

## 3. Markdown features

The renderer (`MLA.util.md` in `js/core/util.js`) also displays learners' notes and imported files, so it is deliberately strict: all text is HTML-escaped, raw HTML is never passed through (a line such as `<div>…</div>` appears as plain text), and links work only for `#/…` app routes and `http(s)` URLs. Use a diagram, a chart or a demo instead of hand-written HTML or SVG. It supports:

- paragraphs, `#`/`##` headings, **bold**, *italic*, `inline code`, `> quotes`, `---` rules
- nested bullet and numbered lists, and tables (`| a | b |` with a `| --- |` separator row)
- fenced code blocks with a language (` ```python `, ` ```text `, ` ```json `…), with a copy button
- links: `[text](https://…)` opens in a new tab; `[text](#/concept/gpt)` links inside the app
- callouts:

  ```text
  :::tip Optional custom title
  Content (markdown).
  :::
  ```

  Kinds: `tip`, `warn`, `danger`, `real` (real-world example), `note`, `key` (key idea), `analogy`, `math`.

### Formulas, every symbol must be defined

Formulas are written in Unicode (no LaTeX) between `$$` markers, one formula per line. Two or more spaces inside a formula become a visible gap, so side-by-side formulas stay readable.

**Every formula (or group of consecutive formulas) must be followed by a `:::where` legend** that defines every symbol in plain language. Full understanding is the point of the app, so a formula with an undefined symbol counts as a bug, and `tests/content.test.js` fails if a legend is missing.

```text
$$ θ ← θ − η · ∇L(θ) $$
:::where
θ — "theta": all the parameters, e.g. the weights w and the bias b
η — "eta": the learning rate, how big a step to take
∇L(θ) — the gradient of the loss at the current parameters; it points uphill
:::
```

- One definition per line: `symbol — meaning`. A plain `symbol: meaning` also works.
- A line without either separator is shown as a note across the full width (avoid a ": " in notes).
- A line indented by two or more spaces continues the previous definition.
- `:::where Custom label` changes the "where" heading. A legend placed after a table (for example `:::where Symbols in this table`) renders on its own.
- Give shapes or units where they help, e.g. "X — the data matrix, shape (n, d)".
- The same rule applies to formulas inside problem prompts.

## 4. Quiz questions

```js
{ type: 'mcq',   q: 'Question?', options: ['A', 'B', 'C'], answer: 1, explain: 'Why B.' }
{ type: 'multi', q: 'Select all that apply.', options: ['A', 'B', 'C', 'D'], answer: [0, 2], explain: '…' }
{ type: 'tf',    q: 'Statement.', answer: false, explain: '…' }
{ type: 'fill',  q: 'Numeric or short answer?', answer: ['0.8', '0.80'], display: 'optional pretty answer', explain: '…' }
{ type: 'order', q: 'Put the steps in order.', items: ['first', 'second', 'third'], explain: '…' }   // items in the correct order
{ type: 'match', q: 'Match the pairs.', pairs: [['left', 'right'], ['left 2', 'right 2']], explain: '…' }
```

Every question needs an `explain`. Options and items are shuffled when shown. `fill` answers are compared after normalization (case, extra spaces, a trailing full stop), numbers are compared as numbers, and an accepted answer written as `'/pattern/i'` is treated as a regular expression; still, list the common spellings.

## 5. Practice problems

```js
{
  id: 'scaled-dot-product-attention',      // kebab-case, unique across all concepts
  title: 'Scaled Dot-Product Attention',
  difficulty: 'medium',                    // easy | medium | hard
  prompt: `Markdown task description with a small example.`,
  starter: `the code the learner starts from (must fail at least one test)`,
  solution: `the reference solution (must pass every test)`,
  prelude: `optional hidden code run before both, e.g. test helpers`,
  hints: ['progressive hints'],
  explain: `why the solution works / what it connects to`,
  tests: [ /* 3+ tests, at least one visible and one hidden */ ]
}
```

### Tests

Each test either evaluates an expression or runs assertions:

```js
{ call: 'attention(Q, K, V)', expect: '(np.array([[1.0, 1.0]]), np.full((1, 3), 1 / 3))' }
{ name: 'random inputs', setup: 'rng = np.random.default_rng(0)\nQ = rng.normal(size=(4, 3))', call: 'f(Q)', hidden: true }
{ name: 'property', setup: '…', code: 'out = f(x)\nassert out.shape == (3, 3)', hidden: true }
```

- `call`: a Python expression evaluated in the learner's namespace. The result is compared with `expect`.
- `expect`: a Python expression for the expected value. **If omitted, it is computed by running the reference solution** on the same `setup` (convenient for random inputs).
- `setup`: Python statements run first; `np` and `math` are always available.
- `code`: statements with `assert`s, instead of `call`/`expect`.
- `hidden: true`: only run on **Submit**, not on **Run**.
- `tol`: tolerance for numeric comparisons (default 1e-6). Comparison is smart: NumPy arrays with `allclose`, ints and floats interchangeably, lists/tuples/dicts element by element, `None` distinguished from everything else.

Good problems give a small worked example in the prompt, a visible test that matches it, and hidden tests with edge cases and seeded random inputs. When a function uses a random generator, state the exact order of the generator calls in the prompt so results are reproducible.

## 6. Authoring gotchas

Content sits inside JavaScript template literals (backtick strings):

- Escape markdown backticks as `` \` `` (e.g. code fences become `` \`\`\` ``).
- Never write `${` anywhere in content.
- A backslash in the source becomes an escape: write `\\n` to get the two characters `\n` (a Python newline escape in code, or a literal `\n` shown in a prompt).
- Tests and quiz strings are single-quoted JavaScript strings: escape apostrophes as `\'`.
- Avoid tabs in Python code.
- The in-browser Python is **32-bit** (WebAssembly), unlike desktop CPython. Some NumPy functions, such as `np.bincount`, reject 64-bit integer arrays (which `rng.integers` returns by default) there. Prefer `np.unique(x, return_counts=True)` or `x.astype(np.intp)`, and always run the in-browser self-test as well as the CPython tests.

### Writing style

The text should read as if a careful teacher wrote it:

- **No em dashes in prose.** Use a colon, a comma, a semicolon or parentheses instead. The one exception is the `symbol — meaning` lines of a `:::where` legend, which the renderer uses as a separator.
- **No emojis** in lessons or in the interface; use the line icons in `js/core/ui.js` (`<mla-icon name="…">`).
- **No exclamation marks, and no hype words** ("powerful", "beautiful", "superpower", "seamless"). State what something does and let the numbers show how much it matters.
- **Show the data.** A sample that loads or builds a dataset prints a few rows of it (`df.head()`, a short loop over `X[:3]`, an 8 × 8 digit grid) before using it, so the learner can see what the model works with.
- **Keyboard first.** A new demo that reacts to clicks needs a keyboard path too: the `kbCursor` mixin in `js/views/demos.js` gives a canvas an arrow-key cross-hair with Enter to act and a live region for screen readers.

## 7. Glossary, tracks and achievements

- Add 2–4 glossary terms for every new concept: `G('Term', 'One- or two-sentence definition.', 'branch', 'concept-id')`. Terms must be unique; each one becomes a flashcard. The tests require at least one glossary term per concept.
- Tracks list goal concepts; the app adds all prerequisites automatically.

## 8. Checklist before you commit

```bash
node --check js/data/concepts-<file>.js     # JavaScript syntax
node tests/content.test.js                   # structure, links, formula legends, glossary coverage
node tests/views.test.js                     # templates and markdown render
node tests/problems.test.js <concept ids>    # solutions pass, starters fail, samples run (CPython)
npm test                                     # everything
```

Then open the lesson in the browser (`node serve.js`) to check the layout, run the samples, and try the problems. **Progress & Sync → Run self-test** repeats the problem checks inside the in-browser Python.

Writing style: explain for a motivated beginner. Start from the intuition, give a concrete example, define every symbol, show runnable code, and end with when to use it, pitfalls and takeaways.
