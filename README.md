# Course Atlas: a template for interactive courses

An interactive concept map of a subject, with lessons, runnable Python, quizzes and coding problems that run in the browser: a static site for GitHub Pages, with no build step and no server. It is the engine of [ML Atlas](https://markotofu.github.io/LLM-learning/) with the machine-learning content replaced by a three-lesson sample course on game development, ready to become a course on anything else.

```bash
npm install      # test tools only
npm start        # http://localhost:8080
```

To build a course on it, read **[docs/MAKING_A_COURSE.md](docs/MAKING_A_COURSE.md)**. A ready-made brief for a Unity game-development course, to paste into a new Claude Code chat, is in [docs/UNITY_COURSE_PROMPT.md](docs/UNITY_COURSE_PROMPT.md).

## What a course gets

| Feature | What it does |
| --- | --- |
| **Concept map** | Every lesson on a mind map, a prerequisite tree or an outline, coloured by branch. Zoom to a branch, hide finished branches, colour lessons by progress. Locked lessons show what they need first. |
| **Lessons** | Text with formulas and a legend for every symbol, runnable Python samples (Pyodide, in the browser), code in other languages (C# highlighted for Unity), images and screenshots, charts, quick checks, "predict first" questions, demos, diagrams and step-through walkthroughs. |
| **Read aloud** | Any lesson, or one part of it, read with an on-device voice, with media keys and resume. |
| **Quizzes** | A quiz per lesson (multiple choice, select all, true or false, fill in, order, match) with explanations. |
| **Custom quizzes** | A quiz on any lessons, branches or track, optionally with what they build on, mixing each lesson's questions with extra questions that appear only there; results by lesson. |
| **Placement test** | A short adaptive test that marks lessons a learner already knows, so their path starts at the first gap; every mark can be undone. |
| **Practice problems** | A code editor with visible and hidden tests, hints and worked solutions. |
| **Review** | Flashcards from the glossary, and every missed question back on a spaced schedule. |
| **Tracks, projects, search** | Guided paths through the map, sample projects and case studies, and Ctrl+K search over everything. |
| **Progress** | Saved in the browser, exported or moved between devices with a sync code; each course on a site keeps its own. |
| **Looks and access** | Six themes in two looks, works offline, fits phones, checked for accessibility. |

## Layout

| Path | What |
| --- | --- |
| `js/course.js` | the course's name, id, owner and address |
| `js/data/` | the content: branches, lessons (`concepts-*.js`), question banks, glossary, tracks, projects |
| `js/views/`, `js/core/`, `js/engines/` | the app |
| `assets/images/` | pictures and screenshots used by lessons |
| `tools/` | authoring tools: lesson lookup and patching, the catalog builder, window screenshots, chart scripts |
| `tests/` | Node tests (`npm test`) and browser tests (`npm run test:e2e`) |
| `docs/` | how to make a course, the content guide, the Unity course brief |

## Tests

`npm test` checks the content (structure, formulas with legends, quizzes, problems against CPython, images, the question banks, the house style), the views, narration, the placement test and more. `npm run test:e2e` drives the site in Chromium: every lesson and page on a phone screen with an accessibility check, quizzes, the placement test, the map, the in-browser Python and read-aloud.

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
