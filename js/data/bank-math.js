/* Extra quiz questions for the Game Math lessons, seen only in custom quizzes (#/quiz). See js/data/bank-start.js. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  MLA.data.quizBank = MLA.data.quizBank || {};
  Object.assign(MLA.data.quizBank, {
    'vectors': [
      { type: 'mcq', q: "What is the normalized version of the vector (0, 5)?",
        options: ["(0, 1)", "(0, 5)", "(1, 1)", "(0, 0.2)"], answer: 0,
        explain: "Its length is 5, and (0, 5) / 5 = (0, 1): the same direction with length 1." },
      { type: 'fill', q: "What is the dot product of (2, 3) and (4, −1)?", answer: "5",
        explain: "2 · 4 + 3 · (−1) = 8 − 3 = 5." },
      { type: 'tf', q: "Two directions at right angles to each other have a dot product of 0.", answer: true,
        explain: "The dot product of two directions is the cosine of the angle between them, and cos 90° = 0." },
      { type: 'mcq', q: "An enemy moves along the direction (0.6, 0.8) at 10 units per second. The frame took 0.1 s. Which step does it take this frame?",
        options: ["(0.6, 0.8)", "(6, 8)", "(0.06, 0.08)", "(1, 1)"], answer: 0,
        explain: "Step = direction · speed · Δt = (0.6, 0.8) · 10 · 0.1 = (0.6, 0.8): a step 1 unit long." }
    ]
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
