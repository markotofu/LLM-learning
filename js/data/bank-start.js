/* Extra quiz questions for the Start Here lessons. They appear only in custom quizzes (#/quiz), never in a lesson's own
   quiz or the review queue. Keyed by lesson id; the format is the lesson quiz format (docs/CONTENT_GUIDE.md). Every
   branch has a file like this one: js/data/bank-<branch id>.js. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  MLA.data.quizBank = MLA.data.quizBank || {};
  Object.assign(MLA.data.quizBank, {
    'game-loop': [
      { type: 'fill', q: "A game runs at a steady 50 frames per second. How long is each frame, in seconds?", answer: ["0.02", "1/50"],
        explain: "One second divided by 50 frames is 0.02 s per frame, the same length as Unity's default fixed time step." },
      { type: 'mcq', q: "A frame takes 25 ms and the fixed time step is 20 ms. Nothing was owed before the frame. How many fixed steps run after it?",
        options: ["1, with 5 ms carried over", "2", "0, with 25 ms carried over", "1, and the extra 5 ms is dropped"], answer: 0,
        explain: "20 ms fits once into 25 ms. The remaining 5 ms is kept and added to the next frame's time." },
      { type: 'tf', q: "Unity calls Update once per rendered frame.", answer: true,
        explain: "Update runs once each frame before the frame is drawn, so how often it runs depends on the frame rate. FixedUpdate follows the fixed time step instead." },
      { type: 'mcq', q: "Which value should a timer subtract each frame to count down 3 seconds correctly at any frame rate?",
        options: ["Time.deltaTime", "1", "The frame number", "Time.fixedDeltaTime, inside Update"], answer: 0,
        explain: "The frame times add up to the elapsed time, so subtracting Time.deltaTime each frame reaches zero after 3 seconds whatever the frame rate." }
    ]
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
