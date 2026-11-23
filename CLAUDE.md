# Course template: notes for Claude

An interactive learning site template: a concept map of lessons with runnable Python, quizzes, custom quizzes, a placement test, practice problems, review and read-aloud. It is the engine of ML Atlas with its content replaced by a three-lesson sample course on game development (`js/data/concepts-sample.js`). Static (GitHub Pages, no build step in development): a vendored Vue 3 global build, a hash router, Pyodide for in-browser Python and localStorage for progress. Every push to `main` runs CI, which also writes static lesson pages and precompiles the templates, then deploys.

Start with `docs/MAKING_A_COURSE.md` (turning the template into a course) and `docs/CONTENT_GUIDE.md` (lesson format and writing rules). A ready-made brief for a Unity game-development course is in `docs/UNITY_COURSE_PROMPT.md`.

## Save tokens
- Find content before reading it: `node tools/lesson-info.js` lists every lesson; `node tools/lesson-info.js <id>` prints one lesson's sections with file:line; `node tools/lesson-info.js --find <text>` searches titles, text, code, problems and glossary.
- To inspect content in Node: `const MLA = require('./tests/_load.js').load(); MLA.content.init();`, then `MLA.content.byId[id]`, `MLA.data.glossary`, `MLA.data.tracks`, `MLA.data.quizBank`, `MLA.course`.
- Write multi-line scripts to a file in the scratchpad instead of inline `node -e` / `python -c`. The Bash tool can halve `\\` and decode `\u` even inside quoted heredocs: build backslashes with `chr(92)` or `String.fromCharCode(92)`, or edit with the Edit tool.

## Commands
- Run: `npm start` (node serve.js, port 8080). If the browser shows stale code, reload; if it persists, the dev server is probably down and the service worker is serving its cache.
- `npm test` runs graph, content, theme, views, serve, narrate, walks, placement and problems. Problems runs every reference solution (must pass), starter (must fail) and code sample in CPython; limit it with `node tests/problems.test.js <id> …`; `MLA_SAMPLE_OUTPUT=out.json` saves each sample's output.
- `npm run test:e2e` (Playwright, a few minutes; downloads Pyodide from the CDN). It includes `tests/e2e/a11y.js`, a built-in accessibility check on every lesson and page: give every control a name, every link an `href` (in-page actions: `href="#"` with `@click.prevent`), every form field a label, every image alt text.
- `node --check <file>` after editing any JS file.
- After changing lessons: `node tools/build-catalog.js` (the content test fails when the catalog is stale).
- Screenshots (Windows): `powershell -NoProfile -ExecutionPolicy Bypass -File tools/capture-window.ps1 -Title "<part of window title>" [-Crop "x,y,w,h"] -Out assets/images/<topic>/<name>.png`; `-List` shows open windows and sizes.

## Map
| Path | What |
| --- | --- |
| `js/course.js` | the course settings: id (names saved progress, uploads, progress files), name, tagline, subject, owner, contact, dates, siteUrl. Loads first |
| `index.html` | the only page: CSP, loading screen (repeats the course name), and the script list in load order (Node tests load the same list) |
| `css/app.css` | all styles. Top: theme tokens for six themes (Baroque look: Fresco, Vault, Lapis; Classic look under `[data-style="classic"]`: Paper, Graphite, Midnight). End: "Fresco ornaments", every rule prefixed `:where(:root:not([data-style="classic"]))` |
| `js/core/` | `util.js` (markdown renderer), `store.js` (state, export, import, merge; key from `MLA.course`), `router.js`, `ui.js`, `progress.js`, `gamification.js`, `mistakes.js`, `highlight.js` (Python, C#, generic), `narrate.js` (lesson to spoken text), `speech.js`, `datasets.js` (bundled and uploaded files for Python) |
| `js/engines/` | `python.js` (Pyodide worker), `quiz.js`, `graph.js` (prerequisite graph), `srs.js`, `placement.js` (placement test: `BRANCHES`, `POOL`, `SOFT`) |
| `js/data/` | `branches.js` (the branches and `MLA.content`: lesson stubs, `ensure`, `ensureBank`, `WALK_FILES`), `catalog.js` (generated), `concepts-*.js` (lessons, listed in `tools/content-files.json`), `bank-<branch>.js` (extra custom-quiz questions, one file per branch), `glossary.js`, `tracks.js`, `projects.js`, `achievements.js` (one per branch, automatic), `legal.js`, `legacy-sections.js` (empty) |
| `js/views/` | `concept-view.js` (lesson page), `components.js`, `search.js`, `charts.js`, `demos.js` and `demos-more.js` (canvas demos), `diagrams.js` (SVG diagrams), `walks.js` (walkthrough engine) with `walks-ml.js`, `walks-dl.js`, `walks-more.js` (ML Atlas walkthroughs kept as examples; no sample lesson uses them), `problem-view.js`, `practice-view.js`, `map-view.js`, `profile.js`, `dashboard.js`, `pages.js`, `quiz-view.js`, `quiz-builder.js` (custom quizzes), `playground.js`, `editor.js`, `placement-view.js` |
| `assets/images/` | pictures for image sections (screenshots, figures) |
| `data/` | bundled datasets for Python (none in the sample course) |
| `tools/` | `lesson-patch.js`, `lesson-info.js`, `build-catalog.js`, `capture-window.ps1` (window screenshots), `snapshots.js` and `precompile.js` (deploy-time), `check-links.js`, `make-icons.js`, `visual-diff.js`, `charts/*.py` (chart data) |
| `tests/` | `*.test.js` (Node), `_load.js` (loads the app in Node), `e2e/` (Playwright) |
| `docs/` | `MAKING_A_COURSE.md`, `CONTENT_GUIDE.md`, `UNITY_COURSE_PROMPT.md`, `HANDOFF.md`, `ROADMAP.md` |

## Content model
- A lesson has id, branch (an id from `js/data/branches.js`), title, level, minutes, prereqs, tags, summary, why, objectives, sections, when, pitfalls, takeaways, quiz, problems and refs; capstones also have `checkpoints`.
- Section types: `text` (md); `code` (md, code, after, predict, `run: false`, lang, timeoutSec; `lang: 'csharp'` for Unity scripts, shown not run); `image` (src under `assets/images/`, alt of 20+ characters, md, after, width; files under 600 KB); `demo` (a canvas demo name, `dg-<name>`, or `walk-<name>`); `chart` (spec, md, after); `check` (question).
- Section ids are slugs of the titles. Saved progress uses them, so keep a title's words when rewording it.
- Every lesson has 3 or more extra questions in `js/data/bank-<branch>.js`, used only by custom quizzes (`#/quiz`).
- The placement test asks quiz questions listed by their opening words in `POOL` for every beginner or intermediate lesson of the branches in `BRANCHES` (`js/engines/placement.js`); rewording a listed question means updating its opening.
- Only Python runs in the browser: runnable samples and practice problems are Python. A Unity lesson pairs a Python simulation (real numbers) with the C# script.
- The content tests require a `why` over 80 characters, at least 2 objectives, 4 quiz questions and 1 problem (3+ tests, a hidden one, hints, an explanation), a glossary term per lesson, existing prereqs, exactly one root lesson, and a `:::where` legend after every formula.

## Rules
- Writing: no em dashes in prose, no emojis, no exclamation marks, no hype words. Legend lines (`symbol — meaning`) keep their dash, because the markdown renderer splits on " — ".
- Every formula gets a `:::where` legend. Chart numbers come from real computations (scripts in `tools/charts`), never invented. A prediction's `expect` strings must appear in the sample's real output.
- Content sits in JS template literals: escape backticks as `` \` ``, never write `${`, and write `\\n` to get a literal `\n`.
- Screenshots: check every capture by reading the file before using it; crop away personal information (account names, user names in paths, other windows).
- CSP: no inline scripts or inline event handlers; scripts only from this site and `*.jsdelivr.net`; images only from this site or `data:`. The published page forbids eval, so templates must compile (`node tools/precompile.js --index strict.html` builds a strict copy to try locally; delete it afterwards).
- Colours come from theme tokens; `node tests/theme.test.js` checks contrast in all six themes.
- Python runs in Pyodide (32-bit WebAssembly, slower than CPython): slow samples get `timeoutSec`.
- Commit or push only when the user asks.
