# Brief: a Unity game-development course built on the course template

Paste everything below the line into a new Claude Code chat (Claude desktop app, Code tab, on this computer).

---

Build an interactive website for learning game development with Unity, from a complete beginner to shipping a small
game. Use the course template on the `course-template` branch of the repository at
`C:\Users\Mark\Desktop\LLM-learning` (GitHub: `markotofu/LLM-learning`, private). The template already has everything
the site needs: a concept map of lessons, runnable Python in the browser, quizzes, custom quizzes, a placement test,
practice problems, review, read-aloud, image sections, C# highlighting and a screenshot tool. Your work is the course
content, done to the same standard as the template's sample lessons.

## 1. Set up (first)

1. Make the course its own repository, next to the template:
   ```powershell
   git clone --branch course-template --single-branch "C:\Users\Mark\Desktop\LLM-learning" "C:\Users\Mark\Desktop\unity-atlas"
   cd "C:\Users\Mark\Desktop\unity-atlas"
   git switch -c main
   git remote remove origin
   npm install
   ```
   Open that folder as the working directory. Run `npm test` once; it must pass before you change anything.
2. Read `CLAUDE.md`, `docs/MAKING_A_COURSE.md` and `docs/CONTENT_GUIDE.md` (especially "Writing style" and the
   section types). Look at `js/data/concepts-sample.js`: its three lessons are the model for every lesson you write.
3. Ask me for the course name (suggestion: "Unity Atlas"), then set `js/course.js` (`id: 'unity-atlas'`, name,
   tagline, `subject: 'game development with Unity'`, owner, contact, `siteUrl`), the title, description and loading
   screen in `index.html`, and `manifest.webmanifest`.
4. Find the installed Unity versions (`Get-ChildItem "C:\Program Files\Unity\Hub\Editor"`) and ask me which one the
   course targets. Write every C# script and every editor instruction for that version.

## 2. Plan the course, then wait for my approval

Propose branches (each with a colour from the template's palette and an icon) and a lesson list with prerequisites,
levels and one-line summaries. A starting point, about 40 to 60 lessons:

- **Start Here**: what a game engine does, installing Unity Hub and the editor, a tour of the editor windows, a first scene.
- **C# for Unity**: variables and types, methods, classes and objects, the MonoBehaviour lifecycle (Awake, Start,
  Update, FixedUpdate, OnEnable), lists and dictionaries, events and delegates, coroutines.
- **Game Math**: vectors, transforms and parent-child hierarchies, rotations and quaternions, trigonometry for aiming,
  interpolation and easing, randomness.
- **Unity Fundamentals**: GameObjects and components, prefabs, scenes, the Input System, ScriptableObjects, assets and import.
- **Physics**: Rigidbody, colliders and triggers, raycasts, layers and collision masks, 2D physics, joints.
- **Gameplay Systems**: character controllers, cameras (Cinemachine), game state and saving, UI, audio, animation and
  the Animator, enemy AI with state machines and NavMesh.
- **Graphics**: the Universal Render Pipeline, materials and shaders, lighting, particles, post-processing.
- **Shipping**: debugging and profiling, performance, builds for each platform, version control for Unity projects,
  tests with the Unity Test Framework.

Keep the three sample lessons (game loop, vectors, collisions) if they fit the plan, rewritten into the course. Wait
for my approval before writing lessons.

## 3. Write lessons in batches

Write 3 to 5 lessons per batch, then run the checks in section 6 and show me the batch in the preview before the
next one. Each lesson follows the template's rules (`docs/MAKING_A_COURSE.md`, section 4) and this recipe:

1. The idea in plain words, for a beginner, with a concrete game example.
2. A **runnable Python sample** that shows the idea with real numbers (only Python runs in the browser): delta time,
   vector maths, easing curves, a state machine, a collision test, A* on a grid, weighted random drops, save data as
   JSON. Add a "predict first" question to samples whose output is worth guessing.
3. The **Unity C# script** that does the same (`lang: 'csharp', run: false`): complete, with its `using` lines, and
   correct for the chosen Unity version. Check API names in the Unity Manual and Scripting API for that version (for
   example, Unity 6 renamed `Rigidbody.velocity` to `Rigidbody.linearVelocity`, and `FindObjectOfType` gave way to
   `FindFirstObjectByType`). Say which input system a script needs (the Input System package or the old Input Manager).
4. **Editor screenshots** wherever the lesson uses the editor (section 4), as image sections with a precise alt text
   and an `after` line that says what to notice.
5. A quick check, a quiz of 4 or more questions, a **Python practice problem** on game logic (health and damage,
   cooldown timers, an inventory, grid movement, pathfinding, wave spawning) with visible and hidden tests, glossary
   terms, 3 or more questions in the branch's question bank, and references to the Unity documentation (check that
   every link opens).
6. Charts only from real computations, with the script kept in `tools/charts/`.

Also add tracks (for example "Your first 2D game", "Your first 3D game", "C# for Unity") and sample projects (small
complete games to build in Unity, each with steps, a finish line and stretch goals), and keep the placement test's
`BRANCHES` and `POOL` in step with the beginner and intermediate lessons.

## 4. Screenshots of the Unity Editor

Take the screenshots yourself:

1. **A separate screenshot project.** Never open or change my own Unity projects. Create one for screenshots, once:
   ```powershell
   & "C:\Program Files\Unity\Hub\Editor\<version>\Editor\Unity.exe" -createProject "C:\Users\Mark\Desktop\UnityAtlasShots"
   ```
   Later, open it with `-projectPath "C:\Users\Mark\Desktop\UnityAtlasShots"`. It takes a while to load: wait until
   a window whose title contains `UnityAtlasShots` exists (`tools/capture-window.ps1 -List -Titles`). If Unity asks
   me to sign in, accept a licence or install anything, stop and ask me.
2. **Set up each view with computer use.** Read the computer-use skill first, then operate the Unity Editor like a
   person: create and select objects, add components, open windows (Window menu), switch Scene and Game views, play
   the scene. Start each session from the default layout (Window, then Layouts, then Default) so screenshots look
   alike. For complex setups, write a small editor script in the screenshot project (`Assets/Editor/ShotSetup.cs`,
   a menu item per screenshot) and click its menu item.
3. **Capture to a file** with the template's script, using the project name as the title:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File tools/capture-window.ps1 -Title "UnityAtlasShots" -Out assets/images/unity/<lesson-id>-<what>.png
   powershell -NoProfile -ExecutionPolicy Bypass -File tools/capture-window.ps1 -Title "UnityAtlasShots" -Crop "x,y,width,height" -Out assets/images/unity/<lesson-id>-inspector.png
   ```
   Crop to the panel the lesson talks about (Inspector, Hierarchy, Project window, a menu), so it stays readable on a
   phone. Captures wider than 1600 pixels are scaled down; files must stay under 600 KB.
4. **Check every file** by reading it (the Read tool shows the picture): the right panel, readable text, and nothing
   personal. Crop away the account button and initials in the toolbar, my user name in any path, and any other window.
   Retake when in doubt.
5. **Use it** in an image section: `{ type: 'image', title, src: 'assets/images/unity/<file>.png', alt, after }`.

Close the screenshot project when you finish a batch.

## 5. Rules

- The template's house style: no em dashes in prose, no exclamation marks, no emoji, no hype words; every formula
  has a `:::where` legend that defines every symbol; full understanding for beginners comes first.
- Content sits in JavaScript template literals: escape backticks, never write `${`, and write `\\n` for a literal `\n`.
- Ask me before downloading or installing anything (Unity versions, packages, Asset Store content, tools).
- Commit or push only when I ask.

## 6. Checks after every batch

- `node tools/build-catalog.js`, then `npm test`: every lesson, quiz, problem, image, bank question and the placement
  pool are checked, and the problems run in CPython.
- When the sample lessons are gone, update the tests that name them (`docs/MAKING_A_COURSE.md`, section 9).
- `npm start` and look at the new lessons in the browser preview, at desktop and phone width, in a light and a dark
  theme; run their Python samples in the page.
- Every few batches, `npm run test:e2e`.

Start with section 1, then show me the plan from section 2.
