/* Glossary: one or more terms per lesson. It also builds the spaced-repetition flashcards (#/review).
 * G(term, definition, branch, lesson id). tests/content.test.js requires at least one term per lesson. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  const G = (term, def, branch, concept) => ({ term, def, branch, concept });

  MLA.data.glossary = [
    G('Game loop', 'The cycle a game repeats for as long as it runs: read the input, update the world, draw the result.', 'start', 'game-loop'),
    G('Frame', 'One pass through the game loop, ending with a new picture on the screen.', 'start', 'game-loop'),
    G('Frame rate', 'How many frames a game shows per second (fps). It varies with the hardware and with how busy the scene is.', 'start', 'game-loop'),
    G('Delta time', 'The time the previous frame took, in seconds (Time.deltaTime in Unity). Multiplying speeds by it makes movement independent of the frame rate.', 'start', 'game-loop'),
    G('Fixed time step', 'A constant step of game time (0.02 s by default in Unity) at which physics advances, however fast the frames are.', 'start', 'game-loop'),
    G('Vector', 'A list of numbers such as (3, 4) that stands for a point or an arrow: a position, a velocity or a direction.', 'math', 'vectors'),
    G('Magnitude', 'The length of a vector: the square root of the sum of its squared coordinates.', 'math', 'vectors'),
    G('Normalized vector', 'A vector divided by its own length, so that it has length 1 and keeps its direction.', 'math', 'vectors'),
    G('Dot product', 'The sum of the products of two vectors\' coordinates. For two directions it is the cosine of the angle between them: positive when they point roughly the same way.', 'math', 'vectors'),
    G('Collider', 'A simple shape (box, sphere, capsule or mesh) that stands in for an object when testing collisions.', 'play', 'collisions'),
    G('AABB', 'Axis-aligned bounding box: a box whose sides are parallel to the axes. Two of them overlap when their spans overlap on every axis.', 'play', 'collisions'),
    G('Trigger', 'A collider that reports overlaps without blocking movement, used for pickups, zones and checkpoints.', 'play', 'collisions'),
    G('Tunneling', 'A fast object passing through a thin collider because no step ends while the two overlap. Continuous collision detection prevents it.', 'play', 'collisions')
  ];

  MLA.data.flashcards = MLA.data.glossary.map(g => ({ id: 'g-' + g.term.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''), deck: g.branch, front: g.term, back: g.def, concept: g.concept }));
})(typeof globalThis !== 'undefined' ? globalThis : window);
