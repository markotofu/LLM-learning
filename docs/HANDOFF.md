# ML Atlas: build handoff

## STATUS UPDATE (session 9, end): two looks, Baroque and Classic

The rebrand is on `main`; the state before it is the branch `pre-rebrand`. The user then asked for a theme setting that lets learners switch between both designs.

- `profile.style` is `'baroque'` (default) or `'classic'`; `profile.theme` stays `'light' | 'dark' | 'darker'`. `MLA.ui.applyTheme(theme, style)` sets `data-theme` (dark, darker) and `data-style="classic"` on `<html>`; `js/core/boot-theme.js` does the same before the first paint.
- `css/app.css`: the Baroque tokens are `:root`, `[data-theme="dark"]` and `[data-theme="darker"]`; the Classic tokens (the original palette) are `[data-style="classic"]`, `[data-style="classic"][data-theme="dark"]` and `[data-style="classic"][data-theme="darker"]`. `tests/theme.test.js` checks all six themes (686 checks).
- **Every ornament rule is scoped** with `:where(:root:not([data-style="classic"]))`, which adds no specificity. New ornaments must use the same prefix, or they leak into the Classic look.
- The gilt `border-color` rule on cards was removed: it overrode the coloured borders of the "when to use", "pitfalls" and "takeaways" cards and of the danger zone.
- Both logos are in the sidebar markup (`.logo-classic`, `.logo-baroque`); CSS shows the one for the current look. The palette button in the top bar (`toggleLook` in `js/app.js`) switches looks; Progress & Sync → Settings has Look and Theme, with theme names per look (Fresco, Vault, Lapis or Paper, Graphite, Midnight).

## STATUS UPDATE (session 9, rebrand): branch `rebrand-baroque`

The user asked for a rebrand "in another branch" after the content round, from four reference images of baroque ceiling frescoes (golden light, angels, clouds, gilded white architecture, sepia, gold and sky blue), and asked whether it could be done without generated images. It is done with CSS and inline SVG only.

- **Themes** (`css/app.css` top): `:root` is now **Fresco** (light); `[data-theme="dark"]` is **Vault** and `[data-theme="darker"]` is **Lapis**, each a complete token block. `MLA.ui.applyTheme` and `js/core/boot-theme.js` set the attribute for "dark" and "darker"; "light" needs none. New profiles default to "light"; saved profiles keep their theme ("dark" was the old default, so existing learners see Vault). `index.html` and the manifest use the Fresco background as theme colour. The tests were updated (`tests/theme.test.js` reads the three blocks; the e2e theme test clicks Vault, Lapis and Fresco).
- **Decorative tokens:** `--gold`, `--gold-light`, `--gold-deep`, `--sky`, `--sky-light`, `--glow`, `--ray`, `--cloud`, `--banner-text`, plus `--font-display` and `--font-read`. Gold is never used for body text.
- **Ornaments** (end of `css/app.css`, "Fresco ornaments"): rays (`repeating-conic-gradient` masked by a radial gradient), clouds (an `feTurbulence` SVG used as a CSS mask, so its colour follows `--cloud`), dentils under the top bar, the lozenge rule under `.section-head h2`, and `.section-head.fresco-banner` on the dashboard.
- **Logo and icons:** the medallion SVG is inline in `js/app.js`; `node tools/make-icons.js` redraws `icons/`.
- Also on `main` (commit `43fb46f`): long button labels wrap on narrow screens.

## STATUS UPDATE (session 9, 2026-11-21): ten topic lessons, keyboard access, row previews, style pass, loading screen

The user asked for: the lessons missing from the "full mastery" review, keyboard access, practice problems ("sample problems"), short row previews of the data, no emojis and fewer other signs of machine-written text, and a real loading screen. Then, on a separate branch, a rebrand inspired by baroque ceiling frescoes (CSS and SVG only, no generated images).

**New lessons** (`js/data/concepts-topics-1.js`, `concepts-topics-2.js`, loaded after `concepts-roadmap.js`): graphical-models, glm-survival, vision-3d, ai-for-science, audio-generation, ai-governance, contextual-bandits, geometric-dl, interpretability-practice, ai-hardware. Each has 7 to 10 sections, runnable samples with predictions, 2 problems (20 in all) and 6 quiz questions. Governance is dated "as of mid-2026" and says it is not legal advice. The Interpretability lesson has one `run: false` TransformerLens sample.

**Charts:** `tools/charts/charts_z1.py` (9 charts); it replays the lessons' own samples through Node like `charts_y1.py`.

**Glossary and tracks:** 49 new terms; the lessons were added to the theory, data-science, build-llm, llm-apps and frontier-dl tracks. The geometric-dl problem `pairwise-distances` clashed with an existing id and was replaced by `symmetry-check`.

**Keyboard access:** `MLA.demoKit.kbCursor` (in `js/views/demos.js`) is a mixin for canvases that take clicks: arrow keys move a cross-hair, Enter or Space acts there, and a polite live region (`said`) reports positions; a component can define `kbDescribe(x, y)`. Used by linear-fit, knn and gp. `turnKey` in `demos-more.js` turns the eigen and PCA directions. The convolution grid uses a roving tabindex. Quiz options, map hubs, the problem of the day and playground files are focusable. `.sr-only` is in `css/app.css`.

**Row previews:** 31 samples print a few rows of their data (the first sample per lesson and dataset).

**Style pass:** emojis replaced by `MLA.ui` line icons (tracks and achievements now store icon names); em dashes in prose rewritten by rule (pairs to parentheses, one dash to a colon, or to a comma before words such as "and", "which", "including"). Formula legends keep " — " because `MLA.util.md` splits legend lines on it. The rewrite touched no code, tests, expectations or quiz answers, and all 1,499 section ids are unchanged. CONTENT_GUIDE has the style rules.

**Loading screen:** markup in `index.html`, styles under "Loading screen" in `css/app.css`.

**Totals:** 123 concepts, 266 problems, 877 quiz questions, 683 glossary terms, 155 charts in 109 lessons, 158 predictions, 356 browser-run samples, median lesson 1,180 words.

## STATUS UPDATE (session 8, later on 2026-11-21): seven new lessons, roadmap.sh coverage, slow samples

The user asked for three things after the capstone round (committed as `23df452`):
- longer time limits for the slow samples;
- lessons for the topics missing from the "full mastery" rating;
- a check against the roadmap.sh [AI Engineer](https://roadmap.sh/ai-engineer) and [Machine Learning](https://roadmap.sh/machine-learning) roadmaps, adding whatever was missing.

**Coverage check.**
- Both roadmaps' topic lists (about 400 entries) were read from the pages in the browser and matched to lesson text with `scratchpad/coverage.js`, a keyword search. Keyword hits include false positives, such as "permutation importance" for discrete maths, so every gap was judged by hand.
- The real gaps were:
  - MCP;
  - the AI-engineering stack: model choice, APIs, open models, frameworks, agent SDKs and coding assistants;
  - production observability and safety practices;
  - discrete maths;
  - NoSQL and other data sources, and file formats;
  - TensorFlow and Keras;
  - video recognition;
  - context failure modes.
- Product names from the roadmaps (vector databases, LangChain and LlamaIndex, LangSmith and Langfuse, DeepEval and RAGAS, Cursor and Claude Code) are introduced inside the relevant lessons.

**New lessons:**
- In `js/data/concepts-frontier-more.js`: `video-generation`, `world-models`, `robotics-vla`.
- In `js/data/concepts-roadmap.js`: `ai-engineering`, `llm-observability`, `mcp`, `discrete-math`.
- Each lesson was built as a skeleton (metadata, two practice problems, a placeholder section), then filled by REPLACEing the placeholder with the patch tool. `lesson-patch.js` can now add quiz questions to an empty `quiz: []`.
- Each has 10–11 sections, 2–4 runnable samples with predictions, 1–2 charts (MCP has none), 6 quiz questions, 2 practice problems with hidden tests, and glossary terms.
- They were added to the `frontier-dl` and `llm-apps` tracks.

**New sections in existing lessons:**
- SQL & Data Engineering: "Beyond SQL tables" and "File formats", with a CSV round-trip sample.
- PyTorch: "TensorFlow, Keras and JAX", with a Keras snippet that runs only locally.
- Advanced Agents: "When context goes wrong, and memory that lasts".
- A hierarchical-clustering and DBSCAN section was drafted for k-means but removed, because `gmm-em` already covers it.

**Slow samples.**
- A new optional field, `timeoutSec` (20–300; `%% timeout N` in patch files), applies to these samples:
  - `code-sample` uses the larger of it and the learner's limit;
  - the in-app self-test allows twice the value;
  - `tests/content.test.js` validates it.
- 16 samples use it: the LoRA experiments (120 s) and 13 scikit-learn-heavy or loop-heavy samples (90 s).

**Charts.** `tools/charts/charts_y1.py` computes 8 charts; it runs the lessons' own sample code through Node, as `charts_x3.py` and `charts_x4.py` do.

**Regression results:**
- `npm test`: graph 21, content 12,881, theme 344, views 443, serve 26, problems 1,316.
- Every new chart is free of overlapping or clipped labels at desktop width and at 375 px, and no page overflows sideways.
- The new samples all run in Pyodide, with their predictions confirmed.

**Totals:** 113 concepts, 246 problems, 817 quiz questions, 634 glossary terms, 146 charts in 100 lessons, 136 predictions, 334 browser-run samples.

## STATUS UPDATE (session 8, 2026-11-21): Advanced Agents and the three capstones deepened

The user asked to deepen the four lessons that session 7 left out. Session 7 is committed as `f07dc59` and `987198f`.

| Lesson | Sections | Words (section text) | Charts | Predictions | Quiz |
| --- | --- | --- | --- | --- | --- |
| `advanced-agents` | 11 → 19 | 1,191 → 2,466 | 0 → 2 | 0 → 5 | 5 → 11 |
| `capstone-tabular` | 10 → 18 | 829 → 1,763 | 0 → 4 | 0 → 4 | 5 → 10 |
| `capstone-rag` | 7 → 13 | 569 → 1,499 | 0 → 2 | 0 → 3 | 5 → 10 |
| `capstone-lora` | 6 → 13 | 563 → 1,346 | 0 → 3 | 0 → 3 | 5 → 10 |

**What was added:**
- **Advanced Agents.** It now avoids repeating the basic `agents` lesson, which already covers compounding errors and checks with retries. New material:
  - the cost of long runs: quadratic token growth, compaction, prompt caching, and cost per solved task;
  - tool design for agents;
  - truncating tool results so that failures survive;
  - a permission gate against the "lethal trifecta" (taint tracking in the harness);
  - a chart of pass@k against pass^k;
  - predictions on the reliability sample and on the evaluation harness, which now prints whether the two agents' intervals overlap.
- **Tabular capstone:**
  - a reliability diagram and a chart of cost against the threshold;
  - the break-even threshold t* = c_offer / (s · c_lost);
  - permutation importance;
  - a learning curve;
  - a drift step with two cases. Under data drift (the contract mix shifts) the model stays calibrated. Under concept drift (calls hurt more) PSI on the inputs sees nothing.
- **RAG capstone:**
  - a recall@k chart for BM25, embeddings and reciprocal rank fusion over matched candidates;
  - evaluation-set sizing (confidence-interval half-width, paired comparisons);
  - offline answer grading (token F1, citations, numbers, abstention). It shows a correct answer failing on F1.
  - an error-analysis table.
- **LoRA capstone:**
  - a chart of each method on the two skills;
  - forgetting curves: 59% of full fine-tuning's forgetting happens in the first 60 of 300 steps;
  - a rank sweep from rank 2 to 32. Rank 1 diverges at lr 0.5, so the sweep leaves it out.
  - hyperparameters that matter.

**Chart scripts:**
- `tools/charts/charts_x1.py`–`charts_x4.py` produced the 11 new charts.
- `charts_x3.py` and `charts_x4.py` read the sample code straight from the lesson through Node, so the charts always match the samples.

**Gotcha: timing on this laptop.** The CPU throttles about 7× after a few seconds of sustained load: a plain NumPy benchmark goes from 0.6 s to 4.5 s per run. Slow sample timings measured here (for example 40 s for the LoRA lesson in `problems.test.js`) are therefore mostly the machine, not the code. At full speed each new LoRA sample takes about 2 s in CPython.

**Regression results:**
- `npm test`: graph 21, content 12,071, theme 344, views 423, serve 26, problems 1,232.
- All 11 new charts have no overlapping or clipped labels at desktop width and at 375 px.
- Every sample in the four lessons runs in Pyodide, and every prediction's `expect` is present.

**Totals:** 138 charts in 94 lessons, 118 predictions, 775 quiz questions, 316 browser-run samples, median lesson 1,173 words.

## STATUS UPDATE (session 7, 2026-11-21): deeper lessons, 127 charts, security review

The user asked for five things:
1. Work the ideas (not the text) of their ten textbook PDFs into the matching lessons, and make every lesson more detailed and readable.
2. Deepen every other lesson the same way, with graphs and charts rather than only text and bullet points.
3. Review the site's security, and check whether the terms and privacy policy need changes.
4. Re-rate the app, adding a 1–10 score for how much a learner would know about current AI after studying it.
5. Push to `main` automatically when finished.

Session 6 is committed as `d5d9d82`; this round is committed on top of it.

**Deeper lessons.**
- Every lesson except the three capstones and `advanced-agents` gained sections: worked examples, runnable samples, predictions, quick checks and quiz questions.
- The PDF chapters fed these lessons: k-NN, supervised learning, linear models, naive Bayes, MLPs, trees, ensembles, SVMs, metrics, grid search and pipelines, pandas, text and plotting. The PDFs were read as untrusted data (Python with `-E -P`, from a scratch folder), and no text was copied.
- New lesson `data-visualization` (Python & Data, `js/data/concepts-viz.js`), so there are 106 concepts.
- Before → after (section text only):
  - words 78,414 → 127,382 (+62%); median lesson 710 → 1,130 words; lessons under 800 words 70 → 5;
  - sections 894 → 1,295;
  - code samples 202 → 323 (185 → 306 run in the browser);
  - predictions 24 → 104;
  - quiz questions 532 → 754; problems 230 → 232; glossary 594 → 598.
- Duplicate material was removed where a new section repeated another lesson. For example, the initialization part of `activation-functions` now covers dead ReLUs instead of repeating `normalization`.

**Charts.**
- New `chart` section type (`js/views/charts.js`): line, scatter and bar charts drawn as theme-aware SVG. It supports:
  - log axes, with decade labels thinned on wide ranges;
  - reference lines, whose labels are drawn on top of the data with a halo;
  - series with their own x values, dashed series and dot-only series, and `null` gaps;
  - clipping to the plot area;
  - a "Show the numbers" table under each chart.
- `tests/content.test.js` validates every spec.
- 127 charts in 90 lessons. The numbers come from the scripts in `tools/charts/` (see its README).
  - Every caption was checked against the data, and many first drafts were corrected.
  - A chart that would give away a prediction is placed after the code sample.
  - Every chart was checked in the browser: no overlapping labels and nothing outside the plot.
- `tools/lesson-patch.js` applies plain-text patches (INSERT or REPLACE; text, code, check, demo, quiz and chart blocks); see CONTENT_GUIDE §2.

**Security review.**
- **Fixed (stored XSS).** The markdown renderer passed raw HTML blocks (`<div>`, `<svg>`, `<details>`…) through unescaped. It also renders learners' notes and imported progress files, so a crafted import could run script. The passthrough is gone (no lesson used it), and `tests/views.test.js` covers 12 attack strings.
- **Fixed (`serve.js`).** It allowed path traversal (`/../`, including encoded variants) and served dotfiles and `node_modules`.
  - It now checks containment with `path.relative` and allows only GET and HEAD.
  - It sends nosniff, no-referrer and SAMEORIGIN headers.
  - `tests/serve.test.js` has 26 checks; the old server failed 18 of them.
- **Hardened (imports).** Imports drop `__proto__`, `constructor` and `prototype` keys and refuse files over 40 MB.
- **CSP meta tag.** The two inline scripts moved to `js/core/boot-theme.js` and `js/core/boot-sw.js`.
  - The self-test ran every problem and sample with no CSP violations, and an injected inline handler is blocked.
  - What it still allows: `'unsafe-eval'` (Vue compiles templates in the browser), `'wasm-unsafe-eval'` and `blob:` workers (Pyodide), and `https://*.jsdelivr.net`.
- **Known limits** (ROADMAP item 18):
  - Pyodide's `importScripts` cannot use SRI; it runs in a worker with no access to the page.
  - `serve.js` listens on the LAN on purpose, for phone testing.
  - Removing `'unsafe-eval'` needs precompiled templates, which means a build step.

**Terms and privacy** (`js/data/legal.js`, dated 21 November 2026):
- Terms §3: import only files you trust.
- Terms §6: referenced works remain their authors'.
- Privacy §1 lists the mistakes queue and the activity log.
- Privacy §2 says exports contain notes and code in readable form.
- New "Security" section.

**Regression results:**
- `npm test`: graph 21, content 11,921, theme 344, views 413, serve 26, problems 1,198.
- Browser tests: 10/10. The prediction test now finds the Bayes section by its heading, because the probability lesson has an earlier prediction.
- In-app self-test: 232/232 problems and 306/306 samples.

**Next:**
- The capstones and `advanced-agents` were not deepened (ROADMAP item 9).
- Open roadmap items: 5, 9 and 10–18 (10, 11 and 14 are partly done).

## STATUS UPDATE (session 6, 2026-11-20): colour scheme, section ids, browser tests, predictions, mistakes queue

The user asked for roadmap items 2–6, a new colour scheme, a re-rating and a push to `main` (the site is live on GitHub Pages). Session 5 is committed as `bdce3a6`; this round is committed on top of it.

**Colour scheme.**
- New palette: "Graphite" (neutral dark), "Midnight" (near-black) and "Paper" (light), with a sky-blue accent.
- New tokens `--on-accent` (text on accent fills) and `--on-status` (text and glyphs on green, amber or accent fills).
- `--cyan` is now teal, so it stays distinct from the accent.
- Updated to match: the logo gradient, the regenerated icons (`npm run icons`), `theme-color` and the manifest.
- `tests/theme.test.js` checks every text token on every background in all three themes (WCAG AA, 4.5:1). It also checks syntax colours on the editor background, text on solid fills, that the accent hue stays at least 40° from green, amber and red, and that `theme-color` and the manifest match.

**Section ids** (roadmap item 2).
- `MLA.content.sectionIds` (in `branches.js`) gives every section an id: a slug of its title, or `check-<hash>` for quick checks.
- `progress.js` stores sections read and checks by id. `migrateSections` converts positions saved by older versions, running at startup and on import.
- Old positions are mapped through the `6badf8c` layout:
  - `js/data/legacy-sections.js` lists the 18 lessons that changed since; it is generated by `tools/legacy-sections.js` from git history;
  - `tests/fixtures/legacy-sections.json` holds the full old layout, and the content test checks every lesson against it.
- The store's data version is now 2.

**Browser tests** (item 3).
- Setup: `@playwright/test` 1.49.1 is a devDependency (package-lock stays git-ignored); `playwright.config.js`.
- `tests/e2e/app.spec.js` runs 10 tests: every lesson and page at 375 px, demo proportions and diagram labels, themes, legal pages, migration, predictions and mistakes, sync code, and the Python sample and problem flow.
- `tests/e2e/visual.spec.js` and `tools/visual-diff.js` (`npm run test:visual [ref]`) render a git ref in a temporary worktree and the working tree on two ports, then compare 80+ screenshots.
- `serve.js` now takes an optional folder argument.
- CI jobs:
  - `browser` gates deployment;
  - `visual` compares with `HEAD~1` and only reports;
  - matplotlib was added to the pip install.

**Predict first** (item 5a).
- 24 code samples have a `predict` question with `expect` strings.
- Checked in two places: `tests/problems.test.js` (via `run_problems.py`, `expectMissing`) and the in-app self-test (Pyodide).
- Answers are saved as checks keyed `predict-<section id>`.
- `MLA_SAMPLE_OUTPUT=<file> node tests/problems.test.js` dumps every sample's output, which helps when writing predictions.

**Mistakes to revisit** (item 5b).
- `js/core/mistakes.js` stores items in `state.mistakes`, scheduled by SM-2. The store's merge keeps the copy that has been reviewed more.
- What gets saved:
  - wrong quick checks and predictions (in `recordCheck`);
  - wrong quiz questions (in `recordQuiz`, keyed by a hash of the question text; match questions are skipped);
  - problems solved after viewing the solution or after 4+ attempts (in `recordSubmit`). They retire after one re-solve on or after their due date.
- Question items retire at an interval of 15 days or more; after a miss, the ease drops and more recalls are needed.
- UI: the Review page (renamed from Flashcards) has a second tab, "Mistakes to revisit" (`#/review?tab=mistakes`), and the dashboard shows a count.

**More demos and diagrams** (item 6).
- Demos in `demos-more.js`:
  - `bias-variance` (overfitting): 60 training sets, with a finite-sample-corrected bias²;
  - `tree-split` (decision trees);
  - `backprop` (backpropagation): step by step, with a gradient-descent step and a gradient check.
- The threshold demo has an ROC / precision–recall toggle.
- Diagrams:
  - `dg-pipeline`: replaced the fine-tuning ASCII sketch;
  - `dg-lstm` (rnn);
  - `dg-moe` (mixture-of-experts);
  - `dg-parallel` (training-at-scale).

**Gotcha:** never put backticks inside a double-quoted bash string: bash runs them as commands. Write helper scripts with the Write tool, and build backslashes and backticks with `String.fromCharCode(92 / 96)`.

**Regression results:**
- `npm test`: content 10210, theme 344, views 274, problems 870.
- Browser tests: 10/10.
- In-app self-test: 230/230 problems and 185/185 samples.
- Visual comparison: shows only the intended colour change.

**Totals.** 105 concepts, 230 problems, 532 quiz questions, 594 glossary terms, 28 demos, 12 diagrams, 24 predictions.

**Next (the user's latest request).** Use the ten textbook PDFs the user supplied (in their Downloads folder) to deepen the lessons:
- IMLP chapters 1, 2 and 7, plus "10 Grid Search" and "11 Algorithm Chains and Pipelines";
- PDA chapters 7–8 and "7 Plotting and Visualization";
- "8 Decision Trees";
- HMLSK chapter 3 (classification) and chapter 5 (SVM).

Cover their ideas (not their text), and make every lesson more detailed and readable.

## STATUS UPDATE (session 5, 2026-11-20): diagrams, 6 new demos, regression fixes, roadmap

Session 4's work is committed as `6badf8c` and pushed to `origin/main`. Session 5 is committed as `bdce3a6` (not pushed until session 6).

**Diagrams.** `js/views/diagrams.js` registers static SVG components named `demo-dg-<name>`, used in lessons as `{ type: 'demo', demo: 'dg-<name>' }`.
- They are styled by the `.diagram .dg-*` classes in `app.css`, so they follow the theme.
- Each uses a fixed 820-wide viewBox. Below about 560 px the diagram scrolls sideways, with a hint.
- The 8 diagrams:
  - transformer-block;
  - rag: replaces the old ASCII sketch; "The pipeline" is now a demo section with `after` text;
  - agents;
  - diffusion-models;
  - detection-segmentation (U-Net);
  - mlops: replaces the ASCII sketch;
  - cross-validation: replaces the ASCII fold sketch;
  - lora.
- Diagram sections get a "layers" icon in the lesson's table of contents.
- Layout was verified with an in-page checker: `getBBox()` finds text overflowing its box or overlapping other text, and `getPointAtLength()` finds lines crossing text.

**Demos.** `js/views/demos-more.js` uses `MLA.demoKit`, which `demos.js` now exports. It holds 6 demos, each placed in one lesson:
- `eigen` in eigen-svd: verified against hand calculations, including shear and rotation.
- `conditioning` in optimization-theory: at κ = 50, the best fixed step takes 155 steps and tuned momentum takes 28.
- `beta` in bayesian-methods: credible intervals match SciPy.
- `gp` in gaussian-processes: the log marginal likelihood matches NumPy; the "fit" button grid-searches ℓ, σ_f and σ_n.
- `pca` in pca.
- `lr-schedule` in optimizers.

That makes 25 demos in total.

**Bugs found and fixed:**
1. `setup()` only compared canvas widths. A 300-px square canvas kept the default 150-px height on screens with a device-pixel ratio of 1, which squashed the diffusion demo.
2. `.demo-canvas` forced a 560:300 aspect ratio, distorting 8 older demos drawn at other heights. It now uses `aspect-ratio: auto 560 / 300`.
3. Two things widened the page on phones: a long "Builds on" chip (on gaussian-processes) and long inline code (on optimizers). Chips now wrap below 600 px, and inline code may break anywhere.
4. Demo captions used `tiny dim`, which measured 3.1:1 contrast on the dark theme and 2.8:1 on the light theme. They now use `small muted` (5.6:1). App-wide `--dim` is still low contrast: ROADMAP item 4.

**Regression results:**
- `npm test`: 0 failures. Content 9917, views 249, problems 846. That covers 230 problems and 202 samples, with 1 skipped because matplotlib is not installed locally.
- Browser self-test: 230/230 problems and 185/185 samples.
- Smoke tests:
  - problem Run, and Submit (rejected for the starter, accepted for the solution);
  - the playground;
  - sync-code export and Merge import (merging a state with itself changes only `stats`);
  - switching between the three themes;
  - every route renders;
  - all 34 demo sections render.

  The preview browser's saved progress was backed up first and restored afterwards.
- At 375 px wide, none of the 105 lessons or 15 other routes scrolls sideways.

**Roadmap.** `docs/ROADMAP.md` lists improvements by priority, with pros, cons and effort.

**Known limitation.** "Sections read" and answered checks are stored by section position. Inserting sections, as this session did in 12 lessons, shifts them. ROADMAP item 2 proposes stable section ids.

**Totals.** 105 concepts, 230 problems, 532 quiz questions, 594 glossary terms, 25 demos, 8 diagrams.

## STATUS UPDATE (session 4, end of 2026-11-20): frontier lessons, capstones, legal pages, review pass (105 concepts)
- New lessons in `concepts-frontier-plus.js`: multimodal-models, advanced-agents, ssm-architectures.
- Capstones in `concepts-capstones.js` (each code cell is self-contained, since samples run in isolation):
  capstone-tabular (churn: leakage, pipeline, CV, calibration, cost threshold, bootstrap and slices),
  capstone-rag (eval-driven iteration: BM25 → + toy embeddings → overlapping chunks; context recall per question
  type; a run:false Claude API cited-answer step), capstone-lora (NumPy char-level model: full FT vs LoRA r2/r8,
  forgetting and replay, merge, int8, generation; plus a PEFT script marked "run locally").
- Legal: Terms of Use and Privacy Policy pages (`#/terms`, `#/privacy`; texts and owner/contact in
  `js/data/legal.js`, views in `js/views/legal.js`), copyright notice in the sidebar footer and on Progress & Sync,
  `LICENSE` (code MIT; educational content all rights reserved; third-party excluded), `THIRD_PARTY_NOTICES.md`,
  package.json license "SEE LICENSE IN LICENSE". Owner is the GitHub handle "markotofu" (change in legal.js/LICENSE).
- Review pass: 4 new demos (calibration, diffusion, roofline, ppo-clip; demos.js), a second problem for what-is-ml,
  naive-bayes, sklearn-workflow and autograd, "Going further" sections for the 8 shortest core lessons, and a second
  reference for 17 single-reference lessons.
- Totals: 105 concepts, 230 problems, 532 quiz questions, 594 glossary terms, 19 interactive demos.
- Tests: views.test.js now checks the legal pages (placeholders filled, localStorage keys named in the policy).

## STATUS UPDATE (session 4, later on 2026-11-20): 16 MORE CONCEPTS (99 total) ("full mastery" pass)
- The user asked to add anything else needed for full mastery of modern ML plus the four lower-priority topics.
  16 new concepts in three new files (index.html, after concepts-spec-plus.js, before glossary.js):
  - `concepts-theory-plus.js`: matrix-calculus, optimization-theory, learning-theory, bayesian-methods,
    gaussian-processes (10 problems)
  - `concepts-practice-plus.js`: calibration-uncertainty, data-centric, sql-data-engineering (problems use
    sqlite3 through a `prelude` with `make_db()`; sqlite3 3.39 works in Pyodide, window functions included),
    learning-to-rank, vector-search (10 problems)
  - `concepts-deep-plus.js`: distillation-pruning, self-supervised, diffusion-models, deep-rl, ml-security,
    llm-serving (12 problems)
- Considered but not added (already covered): LLM data curation (pretraining), t-SNE/UMAP (pca), basic
  hyperparameter search (cross-validation).
- Prereqs: backpropagation ← matrix-calculus; alignment ← deep-rl (instead of reinforcement-learning);
  mlops ← sql-data-engineering.
- Tracks: new `theory` (ML Theory & Probabilistic Methods) and `frontier-dl` (Frontier Deep Learning);
  classical += data-centric; data-science += sql-data-engineering; llm-apps += vector-search; fine-tuning
  (renamed "Fine-tuning, Alignment & Serving") += distillation-pruning, llm-serving.
- Glossary: +153 terms (565 total); "Distillation" now links to distillation-pruning and "Speculative decoding"
  to llm-serving.
- Totals: 99 concepts, 214 problems, 502 quiz questions, 565 glossary terms. `npm test` passes with 0 failures
  (content 9240, views 222, problems 780 with 184 samples).
- Authoring gotcha: when writing content files with the Write tool, check escapes afterwards (an apostrophe)
  inside a single-quoted JS string sometimes lost its backslash, and a doubled backslash sometimes appeared.
  `node --check` catches both; repair scripts lived in the session scratchpad (fix_quotes.js / fix_double.js).
- Not committed yet.

## STATUS UPDATE (session 4, 2026-11-20): 19 MORE CONCEPTS (83 total)
- Session 3's work is committed (`570e053`) and pushed to `origin/main` (the repo is private, so GitHub Pages
  needs a public repo or a paid plan). The additions below are NOT committed yet.
- The user asked which concepts were missing, then "add all of them": 19 new concepts in three new files
  (all in index.html after concepts-spec.js and before glossary.js):
  - `concepts-foundations-plus.js`: eigen-svd, mle-map, stat-inference, pandas-eda, gmm-em (10 problems)
  - `concepts-llm-plus.js`: seq2seq, bert-encoders, text-embeddings, training-at-scale, reasoning-models,
    efficient-attention, mixture-of-experts (15 problems)
  - `concepts-spec-plus.js`: interpretability, time-series, anomaly-detection, detection-segmentation, gnn,
    speech-audio, causal-inference (14 problems)
- Wired in as prerequisites: language-models ← mle-map, pca ← eigen-svd, lora ← eigen-svd,
  data-preparation ← pandas-eda, llm-evaluation ← stat-inference, attention ← seq2seq,
  rag ← text-embeddings (instead of embeddings).
- Tracks: new `data-science` track (pandas-eda, stat-inference, causal-inference, time-series,
  anomaly-detection); classical += gmm-em; deep-learning += seq2seq; build-llm += training-at-scale,
  efficient-attention, mixture-of-experts; fine-tuning += reasoning-models; descriptions updated.
- Glossary: 153 new terms (412 total); the existing "Mixture of Experts (MoE)" term now links to the new
  mixture-of-experts concept.
- Totals: 83 concepts, 182 problems, 422 quiz questions, 412 glossary terms. `npm test` passes with 0 failures
  (content 7619 checks, views 180, problems 632 with 142 samples). In-browser self-test (`#/selftest`):
  182/182 problems and 127/127 runnable samples pass in Pyodide. One earlier sample run reported
  "No module named 'matplotlib'" for the pandas-eda plotting sample; two reruns (one fresh, one after the
  problem run in the same worker) passed, so it was a one-off package download failure. Note: the worker
  swallows package-download errors, so a network failure surfaces as ModuleNotFoundError and the app's hint
  wrongly says the package is unavailable.
- Every code sample's printed output was checked against its lesson text (scratch script that runs each
  sample in CPython). Avoid `np.bincount` on `rng.integers` output (fails in 32-bit Pyodide) and prefer
  building correlated data by hand over `rng.multivariate_normal` (it goes through an SVD whose signs may
  differ between LAPACK builds, so printed numbers could differ from CPython).
- Not added, offered as possible next steps: Bayesian methods, learning-to-rank, SQL & data engineering,
  matrix calculus.

## STATUS UPDATE (session 3, 2026-11-20): BUILD COMPLETE
- All 64 planned concepts are written: 143 problems, 327 quiz questions, 259 glossary terms (every concept has
  at least one). Content files: concepts-start-math, -python-data, -ml-core, -ml-more, -ml-unsup, -nn-core,
  -nn-more, -llm-core, -llm-more, -apps, -spec (all in index.html before glossary.js).
- Formula legends are done: util.js renders `:::where` (formula + legend share one `.formula-group` card;
  `:::where Label` after a table renders on its own); tests/content.test.js fails any `$$` group without a
  legend (lesson text and problem prompts); tests/views.test.js checks the renderer. Every existing formula
  was retrofitted.
- README.md and docs/CONTENT_GUIDE.md are written (the guide documents the legend rule and every format).
- `npm test` passes with 0 failures (graph, content, views, problems: 143 solutions pass and starters fail in
  CPython, 105 samples compile and the runnable ones run). The in-browser self-test (Progress & Sync → Run
  self-test) passes all 143 problems in Pyodide; its sample run caught one 32-bit-only failure (np.bincount on
  int64 in the probability lesson), fixed with np.unique. Pyodide is wasm32: always run the browser self-test.
- LLM API examples (prompting, RAG, agents, evaluation) use the Anthropic Python SDK with `claude-opus-5-5`,
  `client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"], fallbacks="default")` and a
  `stop_reason == "refusal"` check; they are `run: false` samples (compile-checked only).
- Still nothing committed. Possible next steps if the user wants: commit + push to GitHub and enable Pages
  (Settings → Pages → Source: GitHub Actions), more demos, more problems per concept.

## STATUS UPDATE (session 2)
- DONE and verified in a real browser (`node serve.js` → built-in browser): every view, app root, CSS (dark/midnight/light),
  index.html, sw.js, manifest, icons (tools/make-icons.js), serve.js, package.json, .nojekyll, .gitignore, .gitattributes,
  .github/workflows/deploy.yml, tracks.js, achievements.js, glossary.js (36 terms so far: extend at the end for every concept),
  tests (graph / content / views / problems + run_problems.py, shared loader tests/_load.js), and the first content file
  `js/data/concepts-start-math.js` (what-is-ml + 5 math concepts, 12 problems: all solutions pass in CPython).
- Map, drawer, lesson page, runnable samples and the problem page (Run → Wrong Answer, Submit → Accepted + XP) all work.
- Python loader: worker prefetches runtime files with progress + stall detection, races 4 jsDelivr hosts
  (cdn / testingcf / fastly / gcore) on first load, remembers the winner in localStorage `mla.pyMirror`.
  The service worker must NOT intercept cross-origin requests (doing so deadlocked the 10 MB wasm download).
- CONTENT DONE (all tests pass): concepts-start-math, -python-data, -ml-core, -ml-more, -ml-unsup, -nn-core, -nn-more,
  -llm-core, -llm-more (54 concepts, 120 problems, all in index.html). Expected content-test failures until the
  rest exists: alignment's prereq reinforcement-learning, and track llm-apps goals rag/agents/llm-evaluation.
- NEW USER REQUIREMENT (2026-11-19): every math formula must define all of its symbols ("full understanding is the
  most important part"). Do this NEXT, before more content:
  1. util.js markdown: support a `:::where` block right after a `$$` group, with one `symbol — meaning` per line
     (fall back to splitting on the first ": "). Render it as a compact <dl class="formula-where"> attached under
     the formula, with inline markdown in the meanings. Add CSS in app.css for all three themes.
  2. tests/content.test.js: fail any text section whose `$$` group is not followed by a `:::where` block.
  3. Retrofit every existing `$$` formula in all 9 content files. Also add definitions to formulas inside
     problem prompts. Give shapes and units where helpful.
  Then write concepts-apps.js and concepts-spec.js with legends from the start.
- Gotcha: `python -I` can't see the user's numpy (user site-packages). To eyeball sample outputs run plain
  `python` (tests/problems.test.js already runs every sample successfully).
  After each: `node tests/content.test.js && node tests/views.test.js && node tests/problems.test.js`.
  Then extend glossary.js, README.md, docs/CONTENT_GUIDE.md, run /selftest in the browser.

Read this first when resuming. It records the requirements, every design decision, what is
already written, and exactly what is left. Nothing has been committed yet (the repo has no
commits; remote `origin` = https://github.com/markotofu/LLM-learning). Do not commit unless asked.

## What the user asked for

1. Use `C:\Users\Mark\Downloads\learning-app-development.zip` as the basis (it is "NetSec Academy":
   an offline Vue 3 learning app with lessons, quizzes, flashcards, XP, labs).
2. A learning app built as a **tree / mind map of concept nodes**: start from core concepts, branch
   out to the sub-concepts needed for the big topics: machine learning → neural networks → LLMs
   and how to build them, plus other branching concepts (not limited to ML/NN/LLM).
3. **Full understanding of every concept is the top priority**: as detailed as possible, yet
   understandable for complete beginners.
4. Code samples everywhere: how to do things in code, and *why, when and how*.
5. **Runnable on GitHub Pages** (static site, no build step).
6. **LeetCode-style practice problems in every concept**: the student types code in an in-browser
   editor, it runs in the browser, and tests check the output.
7. Progress remembered in **localStorage**.
8. **Export / import** of user code + progress so it can move between devices.

## Key decisions

- App name **ML Atlas**; global namespace `MLA` (base used `NSA`). Same no-build architecture as the
  base: vendored `vendor/vue.global.prod.js` (Vue 3.5.13, includes the template compiler), IIFE
  modules, string templates, hash router (works on any GitHub Pages sub-path and from file://).
- **Python in the browser = Pyodide v0.27.2** from `https://cdn.jsdelivr.net/pyodide/v0.27.2/full/`
  (verified reachable, CORS `*`). Runs in a Blob-URL Web Worker (works on https and file://).
  NumPy is loaded at init; other packages (pandas, scikit-learn, matplotlib) load on demand via
  `loadPackagesFromImports`. A run that exceeds the time limit terminates the worker.
  PyTorch is NOT available in Pyodide → PyTorch samples use `lang: 'pytorch'` (display only);
  all practice problems use plain Python + NumPy.
- **One test harness for browser and CI**: the Python source lives in `HARNESS` inside
  `js/engines/python.js`; `tests/run_problems.py` (to write) must exec that same string under
  CPython to verify every problem.
- Map has three views: **Mind map** (centre = root concept, one hub per branch, each concept hangs
  under its deepest same-branch prerequisite; cross-branch prerequisite links shown on focus or via
  "all links"), **Prerequisite tree** (layered DAG, one horizontal lane per branch), **Outline**.
- Completion rule: quiz passed (≥70%) + at least one problem solved. Mastery: all problems solved +
  quiz best ≥90%. Reading sections only feeds the progress bar. Locked concepts are still openable
  (soft gating: shows "recommended first").
- XP: read 2, check 5, quiz pass 40 (+20 perfect), solve easy 25 / medium 50 / hard 90 (halved if the
  solution was viewed first), concept complete 60, mastered 40, problem of the day 30.

## Files already written (all in this repo)

| File | Status / notes |
| --- | --- |
| `vendor/vue.global.prod.js` | copied from the zip |
| `js/core/util.js` | md renderer (code fences w/ highlight + copy button, `$$` formula blocks, callouts `:::tip/warn/danger/real/note/key/analogy/math`, nested lists, tables), dedent, copyText, download, packText/unpackText (gzip+base64 sync codes `MLA1Z.`/`MLA1B.`) |
| `js/core/highlight.js` | Python + generic highlighter → `tk-*` span classes |
| `js/core/store.js` | key `mla.state.v1`; state shape; `exportJSON`, `importJSON(text, 'merge'|'replace')`, `mergeStates` (never loses work), `exportSyncCode`/`importSyncCode`, `storageInfo` |
| `js/core/router.js` | routes: home, map (full), concept/:id, problems, problem/:id (full), playground (full), tracks, track/:id, review, glossary, achievements, profile, selftest |
| `js/core/ui.js` | toasts, modal (`openModal/confirm/alert`), icons (incl. map, route, code, brain, sync, radial, tree, fit, sparkles…), `applyTheme('dark'|'darker'|'light')` |
| `js/core/gamification.js` | levels/titles, addXP, streaks, `context()` counters, achievements check |
| `js/core/progress.js` | calls `MLA.content.init()`; concept/problem state, `conceptProgress`, `status` (mastered/completed/started/ready/locked), events (visit, markRead, markAllRead, recordCheck, recordQuiz, saveDraft, recordRun, viewSolution, showHint, recordSubmit), problemOfTheDay, tracks/trackConcepts/trackProgress, recommendNext, readyList, branchStats, overall |
| `js/engines/graph.js` | build, findCycle, depths, topoOrder, ancestors, descendants, pathTo, edgesWithin, layoutLanes, layoutMindmap |
| `js/engines/python.js` | HARNESS (`_mla_run_tests`, `_mla_exec`, smart equality, hints like "returned None"), worker, API `MLA.python.{status, warmup, exec, runTests(code, problem, 'run'|'submit'), stop, explain}` |
| `js/engines/quiz.js`, `srs.js` | quiz types mcq/multi/tf/fill/order/match (+`validQuestion`); SM-2 flashcards |
| `js/data/branches.js` | branches (start, math, code, ml, dl, llm, apps, spec + colours) and `MLA.content.add([...])` / `init()` registry with normalisation (dedent of md/code/tests) |
| `js/views/components.js` | mla-icon, mla-md, mla-progress, stat-card, section-head, empty-state, branch-chip, level-badge, diff-chip, status-dot, mla-toasts, mla-modal, score-ring, branch-radar, list-tile, kbd-hint |
| `js/views/editor.js` | `code-editor` (textarea + highlighted overlay, gutter, Tab/Shift+Tab, auto-indent, Ctrl/Cmd+Enter run, Ctrl+Shift+Enter submit, Ctrl+/ comment, Esc then Tab leaves, mobile key bar), `run-output`, `code-sample` (Run/Edit/Copy/Playground), `MLA.playground.newFile/openWith` |
| `js/views/demos.js` | `<mla-demo name>`: gradient-descent, activations, linear-fit, overfit-poly, knn, kmeans, threshold, softmax, attention, bpe, convolution, neuron, positional, quantization, embeddings |
| `js/views/quiz-view.js` | `quiz-runner`, `check-q` |
| `js/views/map-view.js` | `view-map` (3 layouts, pan/zoom/pinch, search, chain highlight, drawer, goal banner). `setGoalFor` sets `profile.customGoal` + `profile.goal='custom'` |

## Content format (concept files call `MLA.content.add([...])`)

Concept: `{ id, branch, title, short?, level 1-3, minutes, prereqs:[ids], parent?, tags?, summary,
why (md), objectives:[], sections:[...], when (md), pitfalls (md), takeaways:[], quiz:[questions],
problems:[...], refs:[{title,url}] }`.

Sections: `{type:'text', title, md}` · `{type:'code', title, md?, code, lang?('python'|'pytorch'|'bash'|'output'), run?(default true for python), after?}` ·
`{type:'demo', title, md?, demo:'<name>'}` · `{type:'check', question:{...quiz question...}}`.

Problem: `{ id (globally unique), title, difficulty 'easy'|'medium'|'hard', prompt (md), starter,
solution, prelude?, hints:[], explain (md), tests:[...] }`.
Test: `{ name?, setup? (py code), call (py expression), expect? (py expression; omitted ⇒ computed by
running the reference solution on the same setup), tol?, hidden? }` or `{ name?, setup?, code: 'assert ...' }`.

Authoring gotchas: content lives in JS template literals → escape markdown backticks as \`, never
write `${`, and avoid backslashes in Python (or double them). Every problem's solution must pass all
tests and the starter must fail at least one (enforced by the problems test).

## Planned concept tree (64 concepts; prereqs in brackets; planned problems after →)

- **start**: what-is-ml "What Is Machine Learning?" [] → learn-threshold
- **math**: vectors-matrices [what-is-ml] → dot-product, matmul-loops, cosine-similarity · derivatives "Derivatives, Gradients & the Chain Rule" [what-is-ml] → numerical-derivative, numerical-gradient · probability [what-is-ml] → bayes-rule, gaussian-pdf · statistics [probability] → mean-var-std, pearson-correlation · information-theory "Entropy, Cross-Entropy & KL" [probability] → entropy-bits, cross-entropy-kl
- **code**: python-essentials [what-is-ml] → word-frequencies, moving-average · numpy [python-essentials, vectors-matrices] → normalize-rows, pairwise-distances · data-preparation [numpy, statistics] → train-test-split, impute-mean · feature-engineering [data-preparation] → standard-scaler, one-hot-encode
- **ml**: supervised-learning [what-is-ml, vectors-matrices] → nearest-centroid · loss-functions [supervised-learning, statistics] → mse-mae, binary-cross-entropy · gradient-descent [loss-functions, derivatives] (demo gradient-descent) → gd-minimize, minibatch-indices · linear-regression [gradient-descent, numpy] (demo linear-fit) → linreg-gd, normal-equation, r2-score · logistic-regression [linear-regression] → sigmoid-stable, logreg-train · overfitting [linear-regression] (demo overfit-poly) → polynomial-features, choose-degree · regularization [overfitting] → ridge-closed-form, l2-gradient-step · evaluation-metrics [logistic-regression] (demo threshold) → confusion-counts, precision-recall-f1, roc-auc · cross-validation [overfitting, data-preparation] → k-fold-splits, grid-search · knn [supervised-learning, numpy] (demo knn) → knn-classify · decision-trees [supervised-learning, information-theory] → gini-impurity, best-split · ensembles [decision-trees, overfitting] → bootstrap-majority, boosting-stumps · svm [logistic-regression, regularization] → hinge-loss-svm · naive-bayes [probability, supervised-learning] → naive-bayes-train-predict · kmeans [numpy, supervised-learning] (demo kmeans) → kmeans-assign-update, kmeans-full · pca [statistics, numpy] → pca-svd, explained-variance · sklearn-workflow [evaluation-metrics, cross-validation, feature-engineering] → sklearn-pipeline (scikit-learn)
- **dl**: perceptron [logistic-regression] (demo neuron) → perceptron-train · activation-functions [perceptron, derivatives] (demo activations) → relu-leaky, softmax-stable, activation-derivatives · mlp [activation-functions, numpy] → dense-forward, mlp-forward · backpropagation [mlp, derivatives, loss-functions] → backprop-2layer, gradient-check · autograd [backpropagation] → value-autograd · training-loop [backpropagation, gradient-descent] → train-xor, early-stopping-epoch · optimizers [training-loop] → momentum-step, adam-step, warmup-cosine · normalization "Initialization, Normalization & Dropout" [training-loop, statistics] → layer-norm, inverted-dropout, he-init · pytorch [training-loop, optimizers] → linear-layer-class, sequential-model · cnn [mlp] (demo convolution) → conv-output-size, conv2d, max-pool · rnn [mlp, backpropagation] → rnn-forward, lstm-cell · embeddings [mlp, vectors-matrices] (demo embeddings) → embedding-lookup, most-similar, word-analogy
- **llm**: text-representation [feature-engineering, python-essentials] → bag-of-words, tf-idf · language-models [text-representation, probability, information-theory] → bigram-model, perplexity · tokenization [text-representation] (demo bpe) → bpe-pair-counts, bpe-merge, bpe-train · attention [embeddings, activation-functions] (demo attention) → scaled-dot-product-attention, causal-mask · multi-head-attention [attention] → split-merge-heads, mha-forward · positional-encoding [attention] (demo positional) → sinusoidal-pe · transformer-block [multi-head-attention, positional-encoding, normalization] → ffn-gelu, transformer-block-forward · gpt [transformer-block, language-models, tokenization] → gpt-param-count, next-token-loss · decoding [gpt] (demo softmax) → temperature-softmax, top-k-top-p, greedy-generate · pretraining "Pretraining & Scaling Laws" [gpt] → training-compute, chinchilla · fine-tuning [pretraining] → chat-template, loss-mask · lora [fine-tuning] → lora-forward, lora-params · alignment "RLHF, DPO & Alignment" [fine-tuning, reinforcement-learning] → bradley-terry, dpo-loss · quantization [gpt] (demo quantization) → absmax-int8, kv-cache-memory · mini-gpt "Capstone: A Tiny GPT in NumPy" [decoding, gpt] → char-tokenizer, tiny-gpt-forward
- **apps**: prompting [decoding] → few-shot-prompt, parse-json-output · rag [prompting, embeddings] → chunk-text, retrieve-top-k, build-rag-prompt · agents [prompting] → tool-dispatch, agent-loop (scripted fake LLM) · llm-evaluation [prompting, evaluation-metrics] → exact-match-f1, win-rate
- **spec**: reinforcement-learning [probability, supervised-learning] → epsilon-greedy-bandit, q-learning-update, discounted-return · generative-models "Autoencoders, GANs & Diffusion" [mlp, probability] → forward-diffusion, reparameterization · vision-transformers [cnn, transformer-block] → image-to-patches · recommender-systems [embeddings, regularization] → mf-predict-sgd · mlops [evaluation-metrics, sklearn-workflow] → psi-drift · responsible-ai [evaluation-metrics] → fairness-metrics

Each lesson: big idea/analogy → how it works (Unicode formulas in `$$` blocks) → code (runnable
NumPy) → demo where listed → inline checks → when to use it → pitfalls → takeaways; quiz of ~5
questions; problems with visible examples + hidden randomised tests.

## Still to do (in order)

1. `js/views/concept-view.js`: tabs Learn / Practice / Quiz / Notes (`?tab=` query); prereq card
   with "recommended first" warning; why / objectives / sections / when / pitfalls / takeaways / refs;
   auto-mark sections read via an IntersectionObserver sentinel at each section end; side TOC;
   `P.visit(id)` on mount; quiz prepared with a seeded shuffle.
2. `js/views/problem-view.js`: LeetCode layout: left tabs Description / Hints / Solution
   (confirm before reveal → assisted) / Explanation (after solve); right `code-editor` (fill) +
   console tabs Test cases / Result / Output; Run (visible tests) + Submit (all tests); draft
   autosave per problem; reset to starter; download `.py`; prev/next problem; XP toast on accept.
3. `practice-view.js` (problem list: search, branch/difficulty/status filters, solved counts by
   difficulty, random unsolved, problem of the day), `playground.js` (files in
   `state.playground.files`, new/rename/delete/import .py/download, Run/Stop, streamed output,
   starter templates), `dashboard.js`, `tracks-view.js`, `review-view.js` (flashcards from the
   glossary, SRS), `glossary-view.js`, `achievements-view.js`, `profile.js` (name, goal, theme
   dark/darker/light, editor font size, run timeout, preload Python; export/import file with
   Merge/Replace, copy/paste sync code, export all code as one `.py`, reset), `selftest` view (runs
   every reference solution + starter in Pyodide), `js/app.js` (sidebar nav, topbar with Python
   status chip, bottom nav, onboarding: name + goal track, theme apply, preload option).
4. `js/data/tracks.js` (ML Foundations, Classical ML Practitioner, Deep Learning from Scratch,
   Build an LLM from Scratch → mini-gpt, LLM Application Engineer, Fine-tuning & Alignment, Complete
   Atlas `all:true`, plus hidden `custom` track whose `title`/`goals` are **getters** reading
   `profile.customGoal`), `achievements.js`, `glossary.js` (~150 terms with `concept` links; doubles
   as flashcards).
5. `css/app.css`: port the base stylesheet (in the zip) and add styles for every class used in the
   views (map-*, mnode*, hub*, edge*, lane*, editor*, ro-*, code-sample/code-block/code-btn,
   demo-*, mini-table, att-*, bpe-*, conv-*, formula, diff, status-dot, branch-chip, light theme,
   `--pink`, `--accent-soft`).
6. `index.html` (script order: util, highlight, store, router, ui, graph, python, quiz, srs, data
   files, gamification, progress, views, app), `manifest.webmanifest`, `sw.js` (cache app shell +
   runtime-cache the Pyodide CDN so Python works offline after first load), `serve.js`,
   `package.json`, `.nojekyll`, `.gitignore`, `.gitattributes`, `.github/workflows/deploy.yml`
   (run tests, then deploy to GitHub Pages), `tools/make-icons.js` + icons, `README.md`,
   `docs/CONTENT_GUIDE.md`.
7. Content: write the 64 concepts in ~9 files (`js/data/concepts-*.js`), one branch at a time.
8. Tests: `tests/graph.test.js`, `tests/content.test.js` (ids unique, prereqs exist, no cycles,
   demos exist, quiz questions valid, tests well-formed), `tests/views.test.js` (load app in Node
   with DOM stubs, compile every template: adapt the base's version), `tests/problems.test.js` +
   `tests/run_problems.py` (CPython: solution passes all, starter fails ≥1, every Python code sample
   compiles; runnable samples execute).
9. Run in the browser via `node serve.js` + the built-in browser: map, lesson, a problem Run/Submit
   with Pyodide, playground, export/import; then `/selftest`.

Local environment: Node 24, Python 3.14.5 with numpy 2.5.3 and scikit-learn 1.9.0 (sklearn imports
slowly: run with a long timeout or in the background).
