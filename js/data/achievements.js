/* ML Atlas – achievements. test(ctx) and progress(ctx) receive the counters from gamification.context() */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  const A = (id, icon, title, desc, xp, test, progress) => ({ id, icon, title, desc, xp, test, progress });
  const branch = (id, icon, title, name) => A('branch-' + id, icon, title, 'Complete every concept in ' + name + '.', 150, c => c.branchDone[id], c => { const list = MLA.data.concepts.filter(x => x.branch === id); return [list.filter(x => MLA.progress.isCompleted(x.id)).length, Math.max(1, list.length)]; });

  MLA.data.achievements = [
    A('first-concept', '🌱', 'First Step', 'Complete your first concept.', 25, c => c.conceptsDone >= 1, c => [c.conceptsDone, 1]),
    A('concepts-5', '📚', 'Getting Serious', 'Complete 5 concepts.', 50, c => c.conceptsDone >= 5, c => [c.conceptsDone, 5]),
    A('concepts-15', '🧭', 'Navigator', 'Complete 15 concepts.', 100, c => c.conceptsDone >= 15, c => [c.conceptsDone, 15]),
    A('concepts-30', '🗺️', 'Cartographer', 'Complete 30 concepts.', 200, c => c.conceptsDone >= 30, c => [c.conceptsDone, 30]),
    A('concepts-all', '🌌', 'The Whole Atlas', 'Complete every concept on the map.', 500, c => c.conceptsDone >= c.conceptsTotal && c.conceptsTotal > 0, c => [c.conceptsDone, c.conceptsTotal]),
    A('first-solve', '✅', 'Hello, Tests', 'Solve your first practice problem.', 25, c => c.solved >= 1, c => [c.solved, 1]),
    A('solves-10', '⌨️', 'Problem Solver', 'Solve 10 practice problems.', 75, c => c.solved >= 10, c => [c.solved, 10]),
    A('solves-25', '🧮', 'Algorithm Builder', 'Solve 25 practice problems.', 125, c => c.solved >= 25, c => [c.solved, 25]),
    A('solves-50', '⚙️', 'Implementation Machine', 'Solve 50 practice problems.', 200, c => c.solved >= 50, c => [c.solved, 50]),
    A('solves-all', '🏆', 'Grandmaster', 'Solve every practice problem.', 500, c => c.solved >= c.problemsTotal && c.problemsTotal > 0, c => [c.solved, c.problemsTotal]),
    A('first-hard', '🔥', 'Hard Mode', 'Solve a hard problem.', 75, c => c.hardSolved >= 1, c => [c.hardSolved, 1]),
    A('hard-10', '💎', 'Diamond Mind', 'Solve 10 hard problems.', 200, c => c.hardSolved >= 10, c => [c.hardSolved, 10]),
    A('unassisted-10', '🦾', 'No Training Wheels', 'Solve 10 problems without hints or solutions.', 100, c => c.unassisted >= 10, c => [c.unassisted, 10]),
    A('mastered-5', '⭐', 'Mastery', 'Master 5 concepts (all problems + 90% quiz).', 120, c => c.mastered >= 5, c => [c.mastered, 5]),
    A('perfect-quiz', '💯', 'Perfect Score', 'Score 100% on a mastery quiz.', 40, c => c.perfectQuizzes >= 1, c => [c.perfectQuizzes, 1]),
    A('quizzes-10', '🧠', 'Quiz Whiz', 'Pass 10 mastery quizzes.', 80, c => c.quizzesPassed >= 10, c => [c.quizzesPassed, 10]),
    branch('math', '📐', 'Math Wizard', 'Math Foundations'),
    branch('code', '🐍', 'Data Wrangler', 'Python & Data'),
    branch('ml', '📈', 'Classical Scholar', 'Classical Machine Learning'),
    branch('dl', '🕸️', 'Neural Architect', 'Neural Networks & Deep Learning'),
    branch('llm', '🤖', 'Transformer Tamer', 'Transformers & LLMs'),
    branch('apps', '🛠️', 'LLM Engineer', 'Building with LLMs'),
    branch('spec', '🧪', 'Explorer', 'Specializations'),
    A('capstone', '🚀', 'I Built a GPT', 'Complete the capstone: A Tiny GPT in NumPy.', 250, c => !!(MLA.progress && MLA.progress.isCompleted('mini-gpt')), c => [MLA.progress && MLA.progress.isCompleted('mini-gpt') ? 1 : 0, 1]),
    A('streak-3', '📅', 'Habit Forming', 'Learn 3 days in a row.', 30, c => c.bestStreak >= 3, c => [c.bestStreak, 3]),
    A('streak-7', '🗓️', 'One Week Strong', 'Learn 7 days in a row.', 75, c => c.bestStreak >= 7, c => [c.bestStreak, 7]),
    A('streak-30', '🏅', 'Unstoppable', 'Learn 30 days in a row.', 300, c => c.bestStreak >= 30, c => [c.bestStreak, 30]),
    A('daily-5', '☀️', 'Daily Practice', 'Solve 5 problems of the day.', 80, c => c.dailyDone >= 5, c => [c.dailyDone, 5]),
    A('tinkerer', '🔧', 'Tinkerer', 'Run code in the playground 10 times.', 30, c => c.playgroundRuns >= 10, c => [c.playgroundRuns, 10]),
    A('experimenter', '🧫', 'Experimenter', 'Run 25 lesson code samples.', 40, c => c.sampleRuns >= 25, c => [c.sampleRuns, 25]),
    A('flash-50', '🃏', 'Memory Palace', 'Review 50 flashcards.', 50, c => c.flashReviews >= 50, c => [c.flashReviews, 50]),
    A('note-taker', '📝', 'In My Own Words', 'Write notes on 5 concepts.', 40, c => c.notes >= 5, c => [c.notes, 5]),
    A('backup', '💾', 'Safe Keeper', 'Export your progress (file or sync code).', 20, c => c.exports >= 1, c => [c.exports, 1])
  ];
})(typeof globalThis !== 'undefined' ? globalThis : window);
