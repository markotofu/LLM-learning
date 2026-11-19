# Content guide

How to add or edit lessons, quizzes and practice problems in ML Atlas. All content lives in plain JavaScript files in `js/data/`; there is no build step. After any change, run `npm test` (see the end of this guide).

## 1. Where content lives

| File | Contents |
| --- | --- |
| `js/data/branches.js` | the 8 branches (id, name, colour, icon) and the content registry `MLA.content.add([...])` |
| `js/data/concepts-*.js` | the concepts, grouped by branch: `start-math`, `python-data`, `ml-core`, `ml-more`, `ml-unsup`, `nn-core`, `nn-more`, `llm-core`, `llm-more`, `apps`, `spec` |
| `js/data/glossary.js` | glossary terms (also the flashcard deck) |
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
```

Runnable code samples execute in the in-browser Python (Pyodide 0.27, Python 3.12, NumPy; pandas, scikit-learn and matplotlib load automatically when imported). Keep them fast (a few seconds at most in the browser) and print something meaningful.

Available demos: `gradient-descent`, `activations`, `linear-fit`, `overfit-poly`, `knn`, `kmeans`, `threshold`, `softmax`, `attention`, `bpe`, `convolution`, `neuron`, `positional`, `quantization`, `embeddings` (defined in `js/views/demos.js`).

## 3. Markdown features

The renderer (`MLA.util.md` in `js/core/util.js`) supports:

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

### Formulas — every symbol must be defined

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

Every question needs an `explain`. Options and items are shuffled when shown. `fill` answers are compared after normalization (case, extra spaces, a trailing full stop), numbers are compared as numbers, and an accepted answer written as `'/pattern/i'` is treated as a regular expression — still, list the common spellings.

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

- `call` — a Python expression evaluated in the learner's namespace. The result is compared with `expect`.
- `expect` — a Python expression for the expected value. **If omitted, it is computed by running the reference solution** on the same `setup` (convenient for random inputs).
- `setup` — Python statements run first; `np` and `math` are always available.
- `code` — statements with `assert`s, instead of `call`/`expect`.
- `hidden: true` — only run on **Submit**, not on **Run**.
- `tol` — tolerance for numeric comparisons (default 1e-6). Comparison is smart: NumPy arrays with `allclose`, ints and floats interchangeably, lists/tuples/dicts element by element, `None` distinguished from everything else.

Good problems give a small worked example in the prompt, a visible test that matches it, and hidden tests with edge cases and seeded random inputs. When a function uses a random generator, state the exact order of the generator calls in the prompt so results are reproducible.

## 6. Authoring gotchas

Content sits inside JavaScript template literals (backtick strings):

- Escape markdown backticks as `` \` `` (e.g. code fences become `` \`\`\` ``).
- Never write `${` anywhere in content.
- A backslash in the source becomes an escape: write `\\n` to get the two characters `\n` (a Python newline escape in code, or a literal `\n` shown in a prompt).
- Tests and quiz strings are single-quoted JavaScript strings: escape apostrophes as `\'`.
- Avoid tabs in Python code.
- The in-browser Python is **32-bit** (WebAssembly), unlike desktop CPython. Some NumPy functions, such as `np.bincount`, reject 64-bit integer arrays (which `rng.integers` returns by default) there. Prefer `np.unique(x, return_counts=True)` or `x.astype(np.intp)`, and always run the in-browser self-test as well as the CPython tests.

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
