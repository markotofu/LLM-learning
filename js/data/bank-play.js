/* Extra quiz questions for the Gameplay Systems lessons, seen only in custom quizzes (#/quiz). See js/data/bank-start.js. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  MLA.data.quizBank = MLA.data.quizBank || {};
  Object.assign(MLA.data.quizBank, {
    'collisions': [
      { type: 'mcq', q: "Box a spans x from 0 to 4 and y from 0 to 4. Box b spans x from 4 to 6 and y from 1 to 2. Under the rule that touching counts, do they collide?",
        options: ["Yes: they touch at x = 4 and overlap on y", "No: they only touch", "No: b is smaller", "Yes: but only because b is inside a"], answer: 0,
        explain: "On x the spans meet at 4 (0 to 4 and 4 to 6), and on y the span 1 to 2 lies within 0 to 4, so the boxes touch along an edge." },
      { type: 'fill', q: "A ball of radius 0.5 sits at (0, 0) and another of radius 0.5 at (0.6, 0.8). What is the squared distance between their centres?", answer: "1",
        explain: "0.6² + 0.8² = 0.36 + 0.64 = 1. The squared sum of the radii is (0.5 + 0.5)² = 1 too, so the balls just touch." },
      { type: 'tf', q: "In Unity, two objects that both have colliders but no Rigidbody receive no OnTriggerEnter events from each other.", answer: true,
        explain: "Collision and trigger events need at least one of the two objects to have a Rigidbody." },
      { type: 'mcq', q: "A bullet moves 2 units per physics step and keeps tunneling through 0.5-unit walls. Which change fixes it most directly?",
        options: ["Continuous collision detection on the bullet's Rigidbody", "A larger frame rate cap", "A trigger collider on the wall", "A smaller bullet"], answer: 0,
        explain: "Continuous detection tests the path swept during each step, so a thin wall in between is found even when no step ends inside it." }
    ]
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
