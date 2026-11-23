/* ML Atlas – extra quiz questions for the Start Here lessons. They appear only in custom quizzes (#/quiz), never in a
   lesson's own quiz or the review queue. Keyed by lesson id; the format is the lesson quiz format (docs/CONTENT_GUIDE.md). */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  MLA.data.quizBank = MLA.data.quizBank || {};
  Object.assign(MLA.data.quizBank, {
    'what-is-ml': [
      { type: 'mcq', q: "A model scores well on its training examples but poorly on new ones. What is this called?",
        options: ["Overfitting", "Underfitting", "Generalization", "Inference"], answer: 0,
        explain: "The model memorised its training examples, noise included, instead of a pattern that carries over. Evaluating on data it has never seen is how you notice." },
      { type: 'tf', q: "Inference means using a trained model to make predictions for new inputs.", answer: true,
        explain: "Training adjusts the parameters. Inference keeps them fixed and applies the model to inputs it has not seen." },
      { type: 'match', q: "Match each term with its meaning.",
        pairs: [["Feature", "A measurable input, such as a house's floor area"], ["Label", "The correct output the model should learn to predict"], ["Parameter", "A number inside the model that training adjusts"], ["Hyperparameter", "A setting chosen before training, such as the learning rate"]],
        explain: "Features go in, labels are the answers to learn, parameters are learned, and hyperparameters are chosen by you." },
      { type: 'mcq', q: "A spam filter learns from emails that users marked as spam or not spam. What is the label?",
        options: ["The words in each email", "Whether each email is spam", "The number of emails", "The filter's weights"], answer: 1,
        explain: "The label is the answer the model should predict for each example. The words are the features it uses to predict it." },
      { type: 'fill', q: "Learning from rewards for actions, rather than from labelled examples, is called ___ learning.", answer: ["reinforcement", "reinforcement learning"],
        explain: "In reinforcement learning an agent acts, receives rewards and learns which actions lead to high total reward." }
    ]
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
