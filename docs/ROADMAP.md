# Roadmap

Ideas for the template itself; a course built on it keeps its own list.

1. **A C# runner.** Only Python runs in the browser. A WebAssembly C# runtime would let Unity-style problems and samples
   run in C#, at the cost of a large download; until then, problems are Python and C# is shown.
2. **Game-development walkthroughs and demos.** Step-through walkthroughs (`js/views/walks.js`) of a fixed-step game
   loop, a state machine or A* pathfinding, and canvas demos of delta time and collisions, to replace the ML examples.
3. **Images in more places.** Image sections exist; pictures inside quiz questions and problem prompts do not yet.
4. **A course generator.** A script that asks for the name, id and branches and writes `js/course.js`,
   `js/data/branches.js`, the bank files and an empty lesson file in one go.
