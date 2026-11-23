/* Case studies and sample projects, attached to lessons by id in MLA.content.init (js/data/branches.js).
 *   Case study: { title, when, story, lesson, source: { title, url? } }        story and lesson are markdown
 *   Project:    { title, level: 'starter' | 'intermediate' | 'advanced', hours, goal, data, steps: [md], done, stretch: [md], skills: [text],
 *                 practice?: a bundled dataset (js/core/datasets.js) to start with in the browser }
 * Case studies must be real, documented events: every number comes from the cited source (a story of 300 or more
 * characters; tests/content.test.js checks the fields). Projects are for the learner's own machine, such as a small
 * game built in Unity. Lessons list both after the key takeaways; the Projects page (#/projects) lists them all. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  const C = (title, when, story, lesson, source) => ({ title, when, story, lesson, source });
  const P = (o) => o;

  // Add case studies by lesson id, for example:  'collisions': [C('Title', 'Year', `story`, `lesson`, { title: 'Source', url: 'https://…' })]
  MLA.data.cases = {
  };
  void C;

  MLA.data.projects = {
    'collisions': [P({ title: 'A coin-collecting game in Unity', level: 'starter', hours: 3,
      goal: `Build a small top-down game in which the player moves with the keyboard, collects coins and cannot walk through walls.`,
      data: `No dataset: you build the scene yourself in the Unity editor (any recent Unity version; the free Personal licence is enough).`,
      steps: [
        `Create a new 3D project, add a plane for the floor, a capsule for the player and a few cubes as walls.`,
        `Write a script that reads the arrow keys or WASD and moves the player at a speed in units per second, scaled by \`Time.deltaTime\`.`,
        `Give the player a Rigidbody (freeze its rotation) and move it with \`Rigidbody.MovePosition\` in \`FixedUpdate\`, so the walls block it.`,
        `Add coins: small spheres with a collider marked *Is Trigger* and the coin script from the lesson. Tag the player "Player".`,
        `Show a counter of collected coins with a UI text element, and print a message when every coin is gone.`],
      done: `You can walk around without passing through walls, every coin disappears when touched, and the counter matches the coins collected.`,
      stretch: [`Make the coins spin with \`transform.Rotate\` scaled by delta time.`, `Add an enemy that chases the player with \`Vector3.MoveTowards\` and ends the game on contact.`, `Fire fast projectiles and fix tunneling with continuous collision detection.`],
      skills: ['frame-rate independent movement', 'Rigidbody and colliders', 'trigger events', 'tags', 'simple UI'] })]
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
