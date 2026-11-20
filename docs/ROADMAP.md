# ML Atlas roadmap

Possible improvements, ranked by how much they help a learner per unit of effort. Status as of 20 November 2026:

- 105 concepts, including 3 capstones;
- 230 practice problems and 532 quiz questions;
- 594 glossary terms;
- 25 interactive demos and 8 diagrams;
- Terms of Use and Privacy Policy pages.

**Effort:**

- **S** — under a day.
- **M** — a few days.
- **L** — a week or more.

**Impact** is the expected gain in learning, reach or safety.

## Recently done

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
| 1 | Publish the site | High | S |
| 2 | Stable section ids for progress | Medium | S |
| 3 | Browser end-to-end and visual tests | High | M |
| 4 | Raise the contrast of `--dim` app-wide | Medium | S |
| 5 | Clearer error when a Python package download fails | Low | S |
| 6 | Add matplotlib to CI | Low | S |

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
5. **Clearer error when a Python package download fails.** The hint says the package is "not available" even when the network merely failed. Tell those two cases apart and offer a retry.
6. **Add matplotlib to CI** (`pip install matplotlib`), so the one skipped plotting sample is checked in CPython too.

## Priority 2: learning quality

| # | Improvement | Impact | Effort |
| --- | --- | --- | --- |
| 7 | Predict-then-run exercises | High | M |
| 8 | Spaced review of missed quiz questions and problems | High | M |
| 9 | Guided capstones with per-step checks | High | M–L |
| 10 | More demos where a picture teaches best | Medium–High | S each |
| 11 | More diagrams | Medium | S each |
| 12 | Accessibility pass on the demos | Medium | M |

7. **Predict-then-run exercises.** Before a code sample runs, ask the learner to predict its output (multiple choice or a number), then reveal it. Committing to a prediction measurably improves retention.
   - *Pros:* turns passive reading of samples into practice.
   - *Cons:* each prediction must be authored and kept in sync with the sample output. The CPython test can verify this automatically.
8. **Spaced review of missed quiz questions and problems.** Flashcards already use SM-2 for the glossary. Extend the same queue to quiz questions answered wrongly and problems that needed hints.
   - *Cons:* a busier review queue; needs per-item caps.
9. **Guided capstones.** Today each capstone is a long lesson plus problems. Split each one into checkpoints that each have their own tests (for example "your leak detector must flag `refund_issued`"), with progress per step.
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
12. **Accessibility pass on the demos.** Most demos are slider-driven, so keyboard use works, and each has a live text summary. Still to do:
    - give canvases `role="img"` with an updating label;
    - make sure click-only interactions (GP points, k-NN points, PCA direction) have a keyboard or slider alternative;
    - respect `prefers-reduced-motion` in animated demos.

## Priority 3: reach and depth

| # | Improvement | Impact | Effort |
| --- | --- | --- | --- |
| 13 | Optional AI tutor (bring your own API key) | High | L |
| 14 | Niche content | Medium | M each |
| 15 | Map at scale | Medium | M |
| 16 | Faster first load | Low–Medium | M |
| 17 | Legal hardening | Medium | S + external review |

13. **Optional AI tutor.** An "Ask about this lesson" panel that sends the lesson text and the learner's question to the Claude API, using the learner's own API key stored locally.
    - *Pros:* adaptive explanations, hints tailored to the learner's code.
    - *Cons:* the privacy policy must change, since data would leave the browser; API costs fall on the learner; answers can be wrong, so ground them in the lesson and show sources. Never ship a shared key in a static site.
14. **Niche content** (one lesson each):
    - contextual bandits;
    - probabilistic graphical models and HMMs;
    - survival analysis;
    - generalized linear models;
    - equivariant networks;
    - robotics and vision-language-action models;
    - text-generation metrics (BLEU, ROUGE, BERTScore);
    - data visualization;
    - "open in Colab" links for the PyTorch examples that need a GPU.
15. **Map at scale.** With 105 nodes:
    - mark *recommended* prerequisites separately from *required* ones;
    - add a "zoom to branch" view and a progress heat map;
    - let a learner hide finished branches.
16. **Faster first load.** All content scripts (about 2 MB uncompressed) load up front.
    - *Fix:* lazy-load lessons per branch, or ship a JSON content bundle. GitHub Pages serves gzip, so the real transfer is much smaller and this matters mainly on slow mobile connections.
    - *Cons:* the tests and the no-build design rely on loading everything at once.
17. **Legal hardening:**
    - have a lawyer review the Terms of Use and Privacy Policy;
    - add a governing-law clause;
    - offer an email contact as well as the GitHub profile;
    - confirm you hold the rights to the original "NetSec Academy" code this app was built from;
    - decide whether the lesson content stays "all rights reserved" or moves to a licence such as CC BY-NC-SA, which invites contributions.

## Suggested order

1. Items 1–2: publish, and make progress robust before more content edits.
2. Item 3: tests that would have caught today's canvas bug.
3. Items 4–6: quick fixes.
4. Items 7–8: the biggest learning gains for moderate effort.
5. Items 10–11: in small batches, alongside the content they illustrate.
6. Everything else as time allows.
