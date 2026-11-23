# Making a course from this template

This branch (`course-template`) is the engine of ML Atlas with its machine-learning content replaced by a three-lesson
sample course on game development. Everything a learner sees comes from a few data files, so a new course on any
subject means editing data, not the app.

What every course gets, unchanged:

- a **concept map** of the lessons and their prerequisites (mind map, prerequisite tree, outline), with zoom to a branch,
  a progress heat map and "hide finished branches";
- **lessons** with text, formulas with a symbol legend, runnable Python (in the browser, through Pyodide), code in other
  languages (C# is highlighted for Unity), **images and screenshots**, charts, quick checks and "predict first" questions,
  key takeaways and references; each lesson can be **read aloud** with an on-device voice;
- a **quiz** per lesson, **custom quizzes** over any set of lessons with extra questions that appear only there, a
  **placement test** that marks what a learner already knows, and **practice problems** with hidden tests;
- **Review** (flashcards from the glossary, and missed questions on a spaced schedule), **learning tracks**, sample
  **projects** and case studies, **search** (Ctrl+K), achievements and XP;
- progress kept in the browser, with export, import and sync codes; six themes; works offline; checked for accessibility.

## 1. Start the new course

Make a separate repository from this branch, so the course has its own history and its own site:

```bash
git clone --branch course-template --single-branch <this repository's path or URL> my-course
cd my-course
git switch -c main                  # the deploy workflow publishes from main
git remote remove origin            # then add your new repository: git remote add origin <url>
npm install                         # test tools only; the site itself needs no build
npm start                           # http://localhost:8080
```

`npm test` and `npm run test:e2e` should pass before you change anything.

## 2. Name it

| File | What to set |
| --- | --- |
| `js/course.js` | `id` (short and unique among your courses: it names the saved progress, uploads and progress files, so courses published on one site never mix data), `name`, `tagline`, `subject` (completes "an interactive map of …"), `owner`, `contact`, `year`, `updated`, `siteUrl` |
| `index.html` | the `<title>`, the description meta tag and the loading screen's title (shown before any script runs) |
| `manifest.webmanifest` | `name`, `short_name`, `description` (the installed app) |
| `icons/` | `icon.svg`, then `node tools/make-icons.js` for the PNG sizes |
| `js/data/legal.js` | read the terms of use and privacy policy; they take the name, owner and contact from `js/course.js`. They are a plain-language starting point, not legal advice |

## 3. Plan the branches

A **branch** is one area of the course, a colour on the map. Edit `MLA.data.branches` in `js/data/branches.js`:
`{ id, name, short, color, icon, desc }`. Each branch needs a question-bank file `js/data/bank-<id>.js` (copy
`bank-start.js`); the test fails without it. Achievements for finishing a branch are made automatically. Keep the
colours from the template or check new ones with `node tests/theme.test.js` (contrast in all six themes).

## 4. Write the lessons

Lessons live in `js/data/concepts-<topic>.js` files that call `MLA.content.add([...])`. List every file in
`tools/content-files.json` and run `node tools/build-catalog.js` after any change (the page loads lessons on demand
through `js/data/catalog.js`). `js/data/concepts-sample.js` shows every kind of section; `docs/CONTENT_GUIDE.md`
describes the format and the writing rules.

Each lesson needs, or the content test fails:

- a `why` of 80 or more characters, 2 or more objectives, 4 or more quiz questions;
- 1 or more practice problems with 3 or more tests (one hidden), hints and an explanation; the reference solution
  must pass and the starter must fail (`node tests/problems.test.js <id>` runs them in CPython);
- a `:::where` legend after every formula, and at least one glossary term;
- 3 or more extra questions in its branch's `bank-<branch>.js` (custom quizzes only);
- the house style: no em dashes in prose, no exclamation marks, no emoji.

When your own lessons exist, delete `concepts-sample.js`, the sample glossary terms, track, project and bank entries,
and `assets/images/game-loop.svg`, then update the tests that name sample lessons (section 9).

## 5. Glossary, tracks, projects and the placement test

- `js/data/glossary.js`: terms with a short definition, a branch and a lesson; they also become flashcards.
- `js/data/tracks.js`: guided paths, each a list of goal lessons; prerequisites are added and ordered automatically.
  Keep the `everything` and `custom` entries.
- `js/data/projects.js`: sample projects (for the learner's own machine, such as a small Unity game) and case studies
  (real, documented events with a named source only).
- `js/engines/placement.js`: `BRANCHES` names the branches the placement test covers; `POOL` lists one or two quiz
  questions (by their opening words: multiple choice or a short answer) for each beginner or intermediate lesson in
  them; `SOFT` lists prerequisite links that are only a reading order. `node tests/placement.test.js` checks it all.

## 6. Pictures and screenshots

Add an image section: `{ type: 'image', title, src: 'assets/images/<topic>/<name>.png', alt, md?, after?, width? }`.
The file must exist, stay under 600 KB, and `alt` must describe what the picture shows (20 characters or more).

On Windows, `tools/capture-window.ps1` saves a window, or a region of it, as a PNG:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/capture-window.ps1 -Title "Unity 6" -Out assets/images/unity/editor.png
powershell -NoProfile -ExecutionPolicy Bypass -File tools/capture-window.ps1 -Title "Unity 6" -Crop "1180,90,420,640" -Out assets/images/unity/inspector.png
```

`-Title` matches part of the window title, `-Crop` is x, y, width and height inside the window, and captures wider
than 1600 pixels are scaled down. `-List` shows the open windows and their sizes. Before committing a screenshot, look
at it and crop away anything personal: account names, the user name in file paths, other windows.

## 7. Code: Python runs, other languages are shown

Only Python runs in the browser, so runnable samples and practice problems are Python. Show code in another language
with `lang` and `run: false`: `lang: 'csharp'` highlights Unity scripts and labels them "C# · runs in Unity, not in
the browser". A good pattern for a game-development course is a short Python simulation that shows the idea with real
numbers, followed by the Unity script that does the same thing (see the sample lessons).

## 8. Demos, diagrams and walkthroughs

Interactive canvas demos (`js/views/demos.js`, `demos-more.js`), SVG diagrams (`js/views/diagrams.js`) and
step-through walkthroughs (`js/views/walks-ml.js`, `walks-dl.js`, `walks-more.js`, drawn by `js/views/walks.js`) are
ML Atlas's, kept as working examples of each format. Lessons decide what learners see (a `demo` section names
`<demo>`, `dg-<diagram>` or `walk-<walkthrough>`), so unused examples cost nothing at run time. Write your own next to
them, and delete the examples and their checks in `tests/walks.test.js` when you no longer need them.

## 9. Tests that name the sample lessons

Most tests pick lessons by what they contain, so they keep working as lessons change. These name sample content and
need new values when the sample lessons go:

- `tests/views.test.js`: the search checks (`collision detection`, `coin-collecting`, `fixed time steps`, `tunneling`,
  `still pictures`).
- `tests/e2e/app.spec.js`: the page list (`/problem/fixed-time-steps`, `/track/basics`), the "predict first" test
  (`game-loop`, *Moving with and without delta time*, `5.00 units`), the Python sample test (its printed line), the
  map tests (branches `start`, `math`, `play`; lessons `vectors` and `collisions`), the placement test
  (`/track/basics`) and the custom-quiz test (`vectors,collisions`).

Run `npm test` (about a minute) and `npm run test:e2e` (a few minutes; downloads Pyodide once) before publishing.

## 10. Publish

The workflow in `.github/workflows/deploy.yml` tests every push to `main`, writes a static page per lesson (for search
engines and link previews), compiles the templates ahead of time so the published page can forbid `eval`, and deploys
to GitHub Pages. In the repository's Settings → Pages, choose **GitHub Actions** as the source. Set `siteUrl` in
`js/course.js` to the published address. Several courses can share one site (`you.github.io/ml-course`,
`you.github.io/unity-course`): each keeps its own progress, uploads and offline copy, as long as their `id`s differ.
