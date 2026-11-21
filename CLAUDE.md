# ML Atlas: notes for Claude

An interactive learning site, from ML basics to LLMs. It is static (GitHub Pages, no build step): a vendored Vue 3 global build, a hash router, Pyodide for in-browser Python and localStorage for progress. Every push to `main` deploys through CI to https://markotofu.github.io/LLM-learning/.

## Save tokens
- Do not read whole `js/data/*.js` files: they hold about 37,000 lines, up to 2,500 per file. Find things first, then read only the line range you need:
  - `node tools/lesson-info.js` lists every lesson (id, branch, file:line, counts);
  - `node tools/lesson-info.js <id>` prints one lesson's sections, with file:line for each;
  - `node tools/lesson-info.js --find <text>` searches lesson and section titles, text, code, problems and glossary.
- To inspect content in Node instead of parsing files: `const MLA = require('./tests/_load.js').load(); MLA.content.init();`, then `MLA.content.byId[id]`, `MLA.data.glossary`, `MLA.data.tracks`.
- `docs/HANDOFF.md` is a long session log, newest first: read only its first section.
- Write multi-line scripts to a file in the scratchpad instead of inline `node -e` / `python -c` (bash mangles `$`, backticks and `\n`).

## Commands
- Run: `npm start` (node serve.js, port 8080), or the `ml-atlas` preview in `.claude/launch.json`. If the browser shows stale code, the dev server is probably down and the service worker is serving its cache.
- `npm test` runs graph, content, theme, views, serve, narrate and problems. Narrate reads every lesson through `MLA.narrate` and fails on leftover markdown or symbols: when it flags new content, fix the rule in `js/core/narrate.js` rather than the lesson. Problems runs every reference solution and code sample in CPython, about 45 s. Limit it with `node tests/problems.test.js <id> …`; `MLA_SAMPLE_OUTPUT=out.json` saves each sample's output (keys are `<id>#<section index + 1>`).
- `npm run test:e2e` (Playwright, about a minute; downloads Pyodide from the CDN).
- `node --check <file>` after editing any JS file.

## Map
| Path | What |
| --- | --- |
| `index.html` | the only page: CSP, loading screen, and the script list in load order (the Node tests load the same list) |
| `css/app.css` | all styles. Top: theme tokens for six themes. Baroque look: `:root` Fresco, `[data-theme="dark"]` Vault, `[data-theme="darker"]` Lapis. Classic look: the same under `[data-style="classic"]` (Paper, Graphite, Midnight). End: "Fresco ornaments", every rule prefixed `:where(:root:not([data-style="classic"]))` |
| `js/core/` | `util.js` (markdown renderer, slug, seeded RNG), `store.js` (state, export, import, merge), `router.js`, `ui.js` (icons, toasts, modal, `applyTheme(theme, style)`), `progress.js`, `gamification.js`, `mistakes.js`, `highlight.js`, `boot-theme.js`, `narrate.js` (lesson, formula and chart to spoken text), `speech.js` (read-aloud player on the Web Speech API, on-device voices only) |
| `js/engines/` | `python.js` (Pyodide worker, `MLA.python.exec`), `quiz.js` (question types and grading), `graph.js` (prerequisite graph), `srs.js` |
| `js/data/` | `concepts-*.js` (lessons, grouped by topic rather than by branch), `projects.js` (case studies and sample projects, keyed by lesson id), `glossary.js`, `tracks.js`, `achievements.js`, `branches.js`, `legal.js`, `legacy-sections.js` (generated) |
| `js/views/` | `concept-view.js` (lesson page), `components.js`, `charts.js` (chart renderer), `demos.js` and `demos-more.js` (canvas demos), `diagrams.js` (SVG diagrams), `problem-view.js`, `map-view.js`, `profile.js` (settings), `dashboard.js`, `pages.js`, `quiz-view.js`, `playground.js`, `editor.js` |
| `tools/` | `lesson-patch.js` (applies lesson patch files), `lesson-info.js`, `legacy-sections.js`, `make-icons.js`, `visual-diff.js`, `charts/*.py` (the scripts that compute chart data) |
| `tests/` | `*.test.js` (Node), `_load.js` (loads the app in Node), `e2e/` (Playwright) |
| `docs/` | `CONTENT_GUIDE.md` (authoring rules: read before writing content), `ROADMAP.md`, `HANDOFF.md` |

## Content model
- A concept has id, branch (start, math, code, ml, dl, llm, apps, spec), title, level, minutes, prereqs, tags, summary, why, objectives, sections, when, pitfalls, takeaways, quiz, problems and refs.
- Optional `cases` and `projects` come from `js/data/projects.js` (keyed by lesson id, attached in `MLA.content.init`); case studies must be real, documented events with a named source.
- Section types: `text` (md); `code` (md, code, after, predict, `run: false`, lang, timeoutSec); `demo` (a canvas demo name, or `dg-<name>` for a diagram); `chart` (spec, md, after); `check` (question).
- Section ids are slugs of the titles, with punctuation ignored. Saved progress uses them, so keep a title's words when rewording it.
- Add or replace sections with patch files: `node tools/lesson-patch.js patch.txt` (`### INSERT` / `### REPLACE` blocks; see the content guide).
- The content tests require a `why` over 80 characters, at least 2 objectives, at least 4 quiz questions and 1 problem, a glossary term per concept, existing prereqs, and a `:::where` legend after every formula.

## Rules
- Writing: no em dashes in prose, no emojis, no exclamation marks, no hype words. Legend lines (`symbol — meaning`) keep their dash, because the markdown renderer splits on " — ".
- Every formula gets a `:::where` legend defining each symbol. Chart numbers come from real computations (`tools/charts`), never invented. A prediction's `expect` strings must appear in the sample's real output.
- Content sits in JS template literals: escape backticks as `` \` ``, never write `${`, and write `\\n` to get a literal `\n`.
- CSP: no inline scripts; scripts only from this site and `*.jsdelivr.net`; fonts and images only from this site or `data:`.
- Colours come from theme tokens; `node tests/theme.test.js` checks WCAG AA contrast in all six themes. Decoration that belongs to the Baroque look goes in the ornaments section with the `:where` prefix.
- Python runs in Pyodide (32-bit WebAssembly, 3 to 7 times slower than CPython): slow samples get `timeoutSec`.
- Commit or push only when the user asks.
- This laptop throttles its CPU under sustained load, so local timings are noisy.
