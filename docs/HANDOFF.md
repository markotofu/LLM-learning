# Handoff

Newest first. Read the first section when resuming work.

## The course template (23 November 2026)

This branch, `course-template`, was made from ML Atlas (`main` at commit `cc09088`) to start other courses, the first
being a Unity game-development course (`docs/UNITY_COURSE_PROMPT.md`).

- **Settings.** `js/course.js` holds the course id, name, subject, owner, contact, dates and address. Every place
  learners see the name reads it; the storage key, the uploads database, progress file names and the app id in
  exports come from the id, so two courses on one site (`you.github.io/a`, `you.github.io/b`) keep separate progress,
  and a progress file from one course is refused by another. The service worker names its cache after its address and
  deletes only its own old caches.
- **Content.** The 123 ML lessons, their question banks, glossary, tracks, projects, datasets, chart scripts and the
  old-progress migration were removed. A sample course on game development takes their place: three lessons (the game
  loop and delta time, vectors for movement, collision detection) in three branches, with a quiz, a check, a
  prediction, a chart, an image, Python samples, Unity C# scripts and a practice problem each, plus glossary terms,
  bank questions, a track, a sample project and placement questions.
- **New in the engine.** An `image` section type (screenshots and figures from `assets/images/`, alt text required,
  read aloud, checked by the content test), C# highlighting (`lang: 'csharp'`), `tools/capture-window.ps1` for window
  screenshots on Windows, branch achievements and question-bank files derived from the branch list, and `BRANCHES`
  in the placement engine.
- **Tests.** The Node tests and browser tests no longer name ML lessons: they pick course lessons by what they contain,
  except a short list of checks that name sample lessons (`docs/MAKING_A_COURSE.md`, section 9).
- **Kept as examples.** ML Atlas's canvas demos, diagrams and 18 walkthroughs stay in `js/views/`; no sample lesson
  uses them, so learners never load them.
