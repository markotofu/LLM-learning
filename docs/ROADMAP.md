# ML Atlas roadmap

Possible improvements, ranked by how much they help a learner per unit of effort. Status as of 22 November 2026:

- 123 concepts, including 3 capstones; a typical (median) lesson has about 1,200 words;
- 270 practice problems and 877 quiz questions;
- 683 glossary terms;
- 28 interactive demos, 26 diagrams, 18 step-through walkthroughs and 155 charts of computed data;
- 356 code samples that run in the browser, 158 of them with a "predict first" question, and a spaced-repetition queue of mistakes to revisit;
- Terms of Use and Privacy Policy pages, a Content-Security-Policy and a hardened local server.

**Effort:**

- **S**: under a day.
- **M**: a few days.
- **L**: a week or more.

**Impact** is the expected gain in learning, reach or safety.

## Done in the ninth round (22 November 2026)

- **Your own data in Python:** upload CSV, TSV, Excel, JSON, Parquet or text files, read them as `data/<name>`; they stay in the browser and travel in the export file.
- **12 step-through walkthroughs** (gradient descent, k-NN, logistic regression, the perceptron, tree splits, boosting, naive Bayes, EM, Adam, attention, decoding, Q-learning): an animated flowchart with the reason for every box and arrow, the running code line, the formula with real numbers, a legend for every symbol and a plot, all computed live under each setting. `tests/walks.test.js` checks them.
- **Walkthroughs, round two:** speeds from Sluggish to Fast and a pace set by the read-aloud voice; six more (a network learning XOR, a linear SVM, PCA by power iteration, an RNN's memory, retrieval for RAG, the KV cache), 18 in all; five run on the learner's own data; a shorter card on phones.

## Done in the eighth round (21 November 2026)

- **Read aloud, round two:** media keys and lock-screen controls, captions, the sentence highlighted, resume where you stopped, the next lesson offered or played.
- **Faster first load:** a 62 KB (gzipped) lesson catalog instead of 1.1 MB of lessons; lessons load on demand and the rest prefetch for offline use (item 16).
- **Search everywhere** (Ctrl+K), a **More** menu on phones, **project checklists and notes**, **practice datasets** that Python reads in the browser.
- **Capstone checkpoints** with their own tests and progress per step (item 9).
- **Screen-reader support for demos** (item 12), **clearer package-download errors** (item 5), a **strict CSP without 'unsafe-eval'** on the published site (item 18), **static lesson pages and a sitemap**, and a **weekly link check**.
- **14 new diagrams**, so every lesson has a picture, and a cleanup of dead code.

## Done in the seventh round (21 November 2026)

- **Read aloud:** a lesson, or any single box, read by an on-device voice, with a player bar (back, pause, next, speed, stop). Formulas, legends, tables, quick checks and charts are turned into sentences (`js/core/narrate.js`, `js/core/speech.js`).
- **Projects and case studies:** 27 sample projects and 31 documented case studies, at the end of their lessons and on a Projects & Cases page.
- **Skill radar** labels no longer get cut off.
- **Charts explain themselves:** end labels, peak and low markers, bar values and a description in words under every chart.

## Rebrand (merged into `main`; the pre-rebrand state is kept on the `pre-rebrand` branch)

- **Two looks, switchable by learners:** Baroque (the default) and Classic (the original design), each with three themes, chosen in Progress & Sync → Settings or with the palette button in the top bar.

- **A look taken from baroque ceiling frescoes**, made with CSS and inline SVG only (no images, no web fonts): golden light rays behind the top of every page, `feTurbulence` clouds, a gilded cornice of dentils under the top bar, a marble sidebar with a double gilt edge, gold hairlines on cards, lozenge rules under page titles, and a fresco-sky banner on the dashboard.
- **Three themes**, still checked by `tests/theme.test.js`: **Fresco** (parchment, lapis and gold; now the default for new visitors), **Vault** (dark umber and gold; the saved choice of everyone who had the old dark theme) and **Lapis** (night blue and gold). The interactive accent stays blue in all three, because gold sits too close in hue to the amber status colour.
- **Serif type** (Palatino or Iowan Old Style where installed, Georgia otherwise) for headings and lesson text; the interface stays in the system sans-serif.
- **A gilded medallion** as the logo and app icon (`tools/make-icons.js`), and the loading screen in the same sky.

## Done in the sixth round (also 21 November 2026)

- **Ten topic lessons** for the gaps in the "full mastery" review, each with runnable samples that ask for a prediction first, two practice problems, six quiz questions, glossary terms and a computed chart (except Geometric Deep Learning): Graphical Models & Hidden Markov Models; GLMs & Survival Analysis; 3D Vision & Neural Rendering; AI for Science; Audio & Music Generation; AI Governance & Regulation; Contextual Bandits & Online Experimentation; Geometric & Equivariant Deep Learning; Mechanistic Interpretability in Practice; AI Hardware & Systems. They are placed in the Theory, Data Science, Build an LLM, LLM Application Engineer and Frontier Deep Learning tracks.
- **Row previews.** 31 samples that load a dataset (scikit-learn datasets, generated DataFrames, an SQLite table, the capstone's churn data) now print a few rows first; digit images print as an 8 × 8 grid.
- **Keyboard access.** The plots that take clicks (line fitting, k-NN, Gaussian processes) take a cross-hair moved with the arrow keys and Enter, and announce positions and results to screen readers; the direction plots (eigenvectors, PCA) turn with the arrow keys; the embedding demo has a word menu; the convolution grid and all quiz options (including true/false) work with Tab, the arrow keys, Enter and Space; map hubs and a few cards became focusable.
- **Less machine-made style.** Emojis were replaced by the app's line icons (tracks, achievements, streak) or removed from messages; about 1,980 em dashes in lesson and interface text became colons, commas or parentheses (formula legends keep theirs, since the renderer uses them); exclamation marks and a few hype words went. Section ids did not change, so saved progress is unaffected.
- **Loading screen.** A centred concept-map animation with the name, a progress bar and a note that appears if loading takes more than 12 seconds; plain HTML and CSS, so it works under the Content-Security-Policy, follows the saved theme and respects reduced motion.

## Done in the fifth round (also 21 November 2026)

- **Coverage checked against the roadmap.sh [AI Engineer](https://roadmap.sh/ai-engineer) and [Machine Learning](https://roadmap.sh/machine-learning) roadmaps.** Every topic on both is now taught. Product names on the roadmaps (vector databases, frameworks, observability tools, coding assistants) are introduced inside the lessons they belong to.
- **Seven new lessons:**
  - Video Models: Understanding and Generation;
  - World Models;
  - Robot Learning & Vision-Language-Action Models;
  - AI Engineering: Models, APIs and the LLM Stack;
  - LLMs in Production: Observability, Monitoring and Safety;
  - The Model Context Protocol (MCP);
  - Discrete Math for ML.
- **New sections in existing lessons:** NoSQL, APIs and other data sources, and file formats (SQL & Data Engineering); TensorFlow, Keras and JAX (PyTorch); context failure modes and long-term memory (Advanced Agents).
- **Slow samples** can declare `timeoutSec`. The 16 heaviest samples now have 90–120 seconds instead of the default 20.

## Done in the fourth round (also 21 November 2026)

- **Advanced Agents and the three capstones deepened**, the lessons the third round had left out. Advanced Agents gained the cost of long runs (with prompt caching and compaction), tool design, truncating tool results, a permission gate against prompt injection, and a pass@k against pass^k chart. The tabular capstone gained calibration and cost charts, a break-even threshold, permutation importance, a learning curve, and data drift against concept drift. The RAG capstone gained a fusion chart, evaluation-set sizing, offline answer grading and an error-analysis table. The LoRA capstone gained forgetting curves, a rank sweep and the hyperparameters that matter. Together: 11 more charts, 14 more predictions and 21 more quiz questions. Item 9's per-step checks are still open.

## Done in the third round (21 November 2026)

- **Deeper lessons.** Every lesson except the three capstones and Advanced Agents gained worked examples, runnable samples, quick checks and quiz questions. Lesson text grew by 62% (median lesson from 710 to 1,130 words; lessons under 800 words from 70 to 5), and "predict first" questions went from 24 to 104. The classical-ML, pandas and plotting lessons now cover the ideas of the textbook chapters the owner supplied, in the app's own words.
- **Charts** (a new `chart` section type): 127 charts in 90 lessons, each computed by a script in `tools/charts/`, with a data table under every chart.
- **New lesson:** Data Visualization with matplotlib & pandas (from item 14).
- **Security review:** fixed a stored-XSS path (the markdown renderer passed raw HTML through, and it also renders notes and imported files) and path traversal in `serve.js`; added a Content-Security-Policy, removed inline scripts, and hardened imports. New tests: `tests/serve.test.js` and hostile-input cases in `tests/views.test.js`. The terms and privacy policy now describe the mistakes queue, the activity log, readable notes in exports, trusted imports and the security measures.

## Done in the second round (also 20 November 2026)

Items 1–4, 6, 7, 8 and part of 10–11 below are done:

- **Published** on GitHub Pages.
- **Stable section ids** (item 2): progress is stored by section id; positions saved by version `6badf8c` are migrated through that version's layout (`js/data/legacy-sections.js`, generated by `tools/legacy-sections.js` and checked by `tests/content.test.js`).
- **Browser tests** (item 3): `npm run test:e2e` (Playwright, also a CI gate before deploying) and `npm run test:visual` (screenshots of every demo and diagram compared with a git ref; a report-only CI job).
- **Contrast** (item 4): a new colour scheme (graphite with a sky-blue accent; "Paper" light theme) in which every text colour passes WCAG AA on every background, enforced by `tests/theme.test.js`.
- **matplotlib in CI** (item 6).
- **Predict first** (item 7): 24 code samples ask for a prediction before running; the CPython tests and the in-app self-test check that each sample's output still confirms the answer.
- **Mistakes to revisit** (item 8): missed quiz questions, quick checks and predictions, and problems solved only with the solution or after many attempts, come back on an SM-2 schedule (Review page, second tab; dashboard count).
- **More demos and diagrams** (items 10–11, part): bias–variance across many training sets, a decision-tree split explorer, backpropagation step by step, a precision–recall view in the threshold demo; diagrams of the LLM training pipeline, an LSTM cell, mixture-of-experts routing and data / tensor / pipeline parallelism.

## Done in the first round

- **8 lesson diagrams** (theme-aware SVG):
  - the transformer block;
  - the RAG pipeline;
  - the agent loop;
  - diffusion training and sampling;
  - the U-Net;
  - the ML lifecycle;
  - cross-validation folds;
  - a LoRA adapter.

  Three ASCII sketches were replaced by diagrams.
- **6 new interactive demos:**
  - eigenvectors and the SVD of a 2×2 matrix, with power iteration;
  - gradient descent and momentum against the condition number;
  - Beta–Binomial updating with credible intervals;
  - Gaussian-process regression with click-to-add points and marginal-likelihood fitting;
  - PCA as "the direction of greatest variance";
  - learning-rate schedules.
- **A canvas bug**: square demos were squashed to half height on screens with a device-pixel ratio of 1. The size check in `setup()` only compared widths, and 300 px is also the browser's default canvas width.
- **Demo captions** moved from the `dim` text colour to `muted`:
  - dim on a demo card measured 3.1:1 (dark theme) and 2.8:1 (light theme), below the WCAG AA minimum of 4.5:1;
  - muted measures 5.6:1.

## Priority 1: do next

| # | Improvement | Impact | Effort |
| --- | --- | --- | --- |
| 1 | Publish the site: done | High | S |
| 2 | Stable section ids for progress: done | Medium | S |
| 3 | Browser end-to-end and visual tests: done | High | M |
| 4 | Raise the contrast of `--dim` app-wide: done | Medium | S |
| 5 | Clearer error when a Python package download fails | Low | S |
| 6 | Add matplotlib to CI: done | Low | S |

1. **Publish the site.** The GitHub Actions workflow already tests and deploys on every push to `main`. GitHub Pages needs **Settings → Pages → Source: GitHub Actions**, and either a public repository or a paid plan for a private one.
   - *Pros:* learners get a URL; offline use and installing as an app work over HTTPS.
   - *Cons:* making the repository public publishes the content source. The licence keeps it "all rights reserved", but copying becomes easy.
   - *Alternative:* Netlify, Cloudflare Pages or Vercel deploy a private repository for free (no build step, publish directory `.`).
2. **Stable section ids for progress.** "Sections read" and "knowledge checks answered" are stored by position (`read: [0, 1, 4]`, `checks: {3: true}`). Inserting a section, as this round did in 12 lessons, shifts them, so a learner may see a read section as unread or answer a check again.
   - *Fix:* give each section an id (a slug of its title, or explicit), store ids, and migrate stored indices once by mapping them through the old title order.
   - *Pros:* content can be edited safely forever.
   - *Cons:* one-off migration code; titles must stay unique within a lesson. The content test can enforce that.
3. **Browser end-to-end and visual tests** (Playwright in CI). Cover the core flows:
   - open a lesson;
   - run a code sample;
   - Run and Submit a problem;
   - export and import progress;
   - switch the theme;
   - open the legal pages.

   Add screenshot comparisons of every demo and diagram in all three themes.
   - *Pros:* catches the class of bugs the Node tests cannot see. The canvas bug above is an example.
   - *Cons:* 2–4 extra CI minutes, a cached Pyodide download, and screenshot baselines to maintain.
4. **Raise the contrast of `--dim` app-wide.** Use about `#7c88a8` in the dark theme and `#6b7590` in the light theme, then check every `.dim` use. Many labels and meta lines still use it.
   - *Pros:* readability, accessibility and compliance.
   - *Cons:* a slightly busier look; needs a visual review.
5. *(Done in the eighth round.)* **Clearer error when a Python package download fails.** The hint says the package is "not available" even when the network merely failed. Tell those two cases apart and offer a retry.
6. **Add matplotlib to CI** (`pip install matplotlib`), so the one skipped plotting sample is checked in CPython too.

## Priority 2: learning quality

| # | Improvement | Impact | Effort |
| --- | --- | --- | --- |
| 7 | Predict-then-run exercises: done | High | M |
| 8 | Spaced review of missed quiz questions and problems: done | High | M |
| 9 | Guided capstones with per-step checks | High | M–L |
| 10 | More demos where a picture teaches best | Medium–High | S each |
| 11 | More diagrams | Medium | S each |
| 12 | Accessibility pass on the demos | Medium | M |

7. **Predict-then-run exercises.** Before a code sample runs, ask the learner to predict its output (multiple choice or a number), then reveal it. Committing to a prediction measurably improves retention.
   - *Pros:* turns passive reading of samples into practice.
   - *Cons:* each prediction must be authored and kept in sync with the sample output. The CPython test can verify this automatically.
8. **Spaced review of missed quiz questions and problems.** Flashcards already use SM-2 for the glossary. Extend the same queue to quiz questions answered wrongly and problems that needed hints.
   - *Cons:* a busier review queue; needs per-item caps.
9. *(Done in the eighth round.)* **Guided capstones.** Today each capstone is a long lesson plus problems (deepened in the fourth round, but still without per-step checks). Split each one into checkpoints that each have their own tests (for example "your leak detector must flag `refund_issued`"), with progress per step.
   - *Pros:* closest to real project work.
   - *Cons:* the largest authoring effort on this list.
10. **More demos where a picture teaches best.** Candidates, in rough order of value:
    - the bias–variance trade-off (many bootstrap fits);
    - a decision-tree split explorer;
    - building a ROC and precision–recall curve point by point;
    - SVM margins and kernels;
    - stepping through backpropagation on a small computation graph;
    - beam search against sampling;
    - KV-cache memory as context grows;
    - a Metropolis sampler random-walking over a posterior.
11. **More diagrams:**
    - an LSTM cell;
    - seq2seq with attention;
    - the LLM training pipeline (pretraining, SFT, preference tuning);
    - mixture-of-experts routing;
    - speculative decoding;
    - data, tensor and pipeline parallelism;
    - the CNN feature hierarchy.
12. *(Done in the eighth round.)* **Accessibility pass on the demos.** Most demos are slider-driven, and the click-only interactions now have keyboard alternatives (sixth round). Still to do:
    - give the remaining canvases `role="img"` with an updating label;
    - respect `prefers-reduced-motion` in animated demos (the loading screen already does).

## Priority 3: reach and depth

| # | Improvement | Impact | Effort |
| --- | --- | --- | --- |
| 13 | Optional AI tutor (bring your own API key) | High | L |
| 14 | Niche content | Medium | M each |
| 15 | Map at scale | Medium | M |
| 16 | Faster first load | Low–Medium | M |
| 17 | Legal hardening | Medium | S + external review |
| 18 | A stricter Content-Security-Policy | Low–Medium | M |

13. **Optional AI tutor.** An "Ask about this lesson" panel that sends the lesson text and the learner's question to the Claude API, using the learner's own API key stored locally.
    - *Pros:* adaptive explanations, hints tailored to the learner's code.
    - *Cons:* the privacy policy must change, since data would leave the browser; API costs fall on the learner; answers can be wrong, so ground them in the lesson and show sources. Never ship a shared key in a static site.
14. **Niche content** (one lesson each):
    - contextual bandits;
    - probabilistic graphical models and HMMs;
    - survival analysis;
    - generalized linear models;
    - equivariant networks;
    - robotics and vision-language-action models: done;
    - text-generation metrics (BLEU, ROUGE, BERTScore);
    - data visualization: done;
    - "open in Colab" links for the PyTorch examples that need a GPU.
15. **Map at scale.** With 113 nodes:
    - mark *recommended* prerequisites separately from *required* ones;
    - add a "zoom to branch" view and a progress heat map;
    - let a learner hide finished branches.
16. *(Done in the eighth round.)* **Faster first load.** All content scripts (about 2 MB uncompressed) load up front.
    - *Fix:* lazy-load lessons per branch, or ship a JSON content bundle. GitHub Pages serves gzip, so the real transfer is much smaller and this matters mainly on slow mobile connections.
    - *Cons:* the tests and the no-build design rely on loading everything at once.
17. **Legal hardening:**
    - have a lawyer review the Terms of Use and Privacy Policy;
    - add a governing-law clause;
    - offer an email contact as well as the GitHub profile;
    - confirm you hold the rights to the original "NetSec Academy" code this app was built from;
    - decide whether the lesson content stays "all rights reserved" or moves to a licence such as CC BY-NC-SA, which invites contributions.
18. *(Done in the eighth round.)* **A stricter Content-Security-Policy.** The policy still allows `'unsafe-eval'`, because the vendored Vue build compiles templates in the browser, and it loads Pyodide from jsDelivr without integrity hashes (`importScripts` in a worker cannot check them).
    - *Fix:* precompile the templates (or move to Vue's runtime-only build), and self-host a pinned copy of Pyodide.
    - *Pros:* removes the last ways a script could be injected or swapped.
    - *Cons:* adds a build step, which the project has avoided so far, and about 10 MB of Pyodide files to the repository. The current risk is small: no user text reaches a template, and the worker has no access to the page.

## Suggested order

1. Items 5 and 12: a quick fix and the demo accessibility pass.
2. Item 9: per-step checks for the capstones.
3. Items 10–11: in small batches, alongside the content they illustrate.
4. Item 17 before promoting the site widely; item 18 if a build step becomes acceptable.
5. Everything else as time allows.
