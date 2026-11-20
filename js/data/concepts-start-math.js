/* ML Atlas – content: Start Here + Math Foundations
 * Format: see docs/CONTENT_GUIDE.md. Markdown lives in template literals, so inline code uses \` */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'what-is-ml', branch: 'start', title: 'What Is Machine Learning?', level: 1, minutes: 20, prereqs: [],
    tags: ['introduction', 'overview', 'ai'],
    summary: 'Programs that learn patterns from examples instead of following hand-written rules — and the five-step recipe that every model on this map follows.',
    why: `Machine learning is behind search engines, spam filters, recommendations, translation, self-driving perception and large language models like ChatGPT. It can look like magic, but every one of those systems — from a two-parameter line fit to a trillion-parameter LLM — is built from the **same five ingredients**: data, a model with adjustable numbers, a way to measure mistakes, a procedure that reduces those mistakes, and an honest test on new data.

Learn that recipe once and the rest of this map becomes a tour of better ingredients, rather than a pile of unrelated tricks.`,
    objectives: [
      'Explain the difference between traditional programming (writing rules) and machine learning (learning rules from examples)',
      'Use the core vocabulary: features, labels, model, parameters, training, inference, generalization',
      'Name the main kinds of learning — supervised, unsupervised, reinforcement and self-supervised — with an example of each',
      'Describe the universal recipe: data → model → loss → optimization → evaluation',
      'Place AI, machine learning, deep learning and LLMs relative to each other'
    ],
    sections: [
      { type: 'text', title: 'Rules versus learning', md: `In **traditional programming** a person writes the rules: *if the email contains "FREE MONEY" and comes from an unknown sender, mark it as spam.* The computer just follows them.

That works until the rules get too complicated to write down. Spammers change their wording every week; there are thousands of subtle clues; and nobody can write a rule for "does this photo contain a dog?".

**Machine learning flips the arrangement.** Instead of rules, you give the computer *examples* — 100,000 emails each labelled "spam" or "not spam" — and an algorithm *finds* the rules (patterns) that best explain them.

| | Traditional programming | Machine learning |
| --- | --- | --- |
| You provide | rules + input | examples (inputs **and** correct outputs) |
| Computer produces | output | the rules (a trained model) |
| Good when | logic is clear and fixed | patterns are complex, fuzzy or changing |

:::analogy How you learned what a dog is
Nobody gave you a rulebook ("four legs, fur, a tail..." — cats have those too). You saw many dogs and many not-dogs, someone told you which was which, and your brain tuned itself until it could recognise dogs it had never seen. That is supervised machine learning.
:::` },
      { type: 'text', title: 'The vocabulary you will use everywhere', md: `Suppose we want to predict house prices.

| Term | Meaning | In the example |
| --- | --- | --- |
| **Example / sample** | one data point | one house |
| **Feature** (input, \`x\`) | a measurable property | size, bedrooms, age |
| **Label / target** (\`y\`) | the answer we want to predict | the sale price |
| **Dataset** | many examples | 10,000 past sales |
| **Model** | a function from features to a prediction | price ≈ w₁·size + w₂·bedrooms + b |
| **Parameters / weights** | the adjustable numbers inside the model | w₁, w₂, b |
| **Training** | finding parameter values that fit the data | adjusting w₁, w₂, b |
| **Inference / prediction** | using the trained model on new inputs | pricing a house that is for sale today |
| **Generalization** | doing well on data the model has never seen | accurate prices for next year's houses |

The model's prediction is usually written **ŷ** ("y-hat") to distinguish it from the true value **y**.` },
      { type: 'code', title: 'Your first learning program', md: `Here is machine learning stripped to the bone. We have hours studied and whether each student passed. The "model" is a single rule — *pass if hours ≥ t* — with one parameter, the threshold **t**. "Training" means trying candidate values of t and keeping the one that makes the fewest mistakes. Press **Run**.`, code: `
# Hours studied and whether each student passed (1) or failed (0)
hours  = [1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0]
passed = [0,   0,   0,   1,   0,   1,   1,   1]

def accuracy(t):
    """How often does the rule 'pass if hours >= t' match reality?"""
    hits = sum((h >= t) == bool(p) for h, p in zip(hours, passed))
    return hits / len(hours)

# Training = searching for the parameter value that makes the fewest mistakes
best = max(sorted(set(hours)), key=accuracy)
print("learned rule: pass if hours >=", best)
print("training accuracy:", accuracy(best))

# Inference = applying the learned rule to new, unseen students
for h in [1.5, 3.5, 5.5]:
    print(h, "hours ->", "pass" if h >= best else "fail")
`, after: `Notice that even the best threshold gets one student wrong (the one who studied 4 hours and failed). Real data is noisy; models make mistakes. Our job is to make *few* mistakes on *new* data — not zero mistakes on old data.` },
      { type: 'text', title: 'The kinds of learning', md: `| Kind | What you have | Typical tasks | Examples |
| --- | --- | --- | --- |
| **Supervised** | inputs **with** correct labels | *regression* (predict a number), *classification* (predict a category) | house prices, spam filters, medical image diagnosis |
| **Unsupervised** | inputs **without** labels | clustering, dimensionality reduction, anomaly detection | customer segments, compressing data, fraud alerts |
| **Reinforcement** | an environment that gives **rewards** | learning a sequence of actions | game-playing agents, robotics, tuning chatbots with human feedback (RLHF) |
| **Self-supervised** | raw data where the label hides inside the data | predict a missing or next piece | **LLM pretraining**: predict the next word of internet text |

Self-supervised learning is the secret behind LLMs: nobody labels trillions of words by hand. The "label" for every position in a sentence is simply the word that comes next — free supervision at enormous scale.` },
      { type: 'check', question: { type: 'mcq', q: 'An online shop wants to group its customers by purchasing behaviour, but has no predefined groups. Which kind of learning fits?', options: ['Supervised classification', 'Unsupervised learning (clustering)', 'Reinforcement learning', 'Supervised regression'], answer: 1, explain: 'There are no labels — nobody has told us which group each customer belongs to — so we look for structure in the data itself: clustering, a form of unsupervised learning.' } },
      { type: 'text', title: 'The universal recipe', md: `Every model on this map — linear regression, decision trees, neural networks, GPT — is an instance of the same five steps:

1. **Data** — collect examples and turn them into numbers (features).
2. **Model** — choose a family of functions with adjustable parameters θ (theta): ŷ = f(x; θ).
3. **Loss** — a single number measuring how wrong the predictions are (for example, the average squared error).
4. **Optimization** — adjust θ to make the loss smaller, usually by *gradient descent*.
5. **Evaluation** — measure performance on data that was **not** used for training.

$$ find θ that minimises  L(θ) = (1/n) Σᵢ loss( f(xᵢ; θ), yᵢ ) $$
:::where
θ — "theta": all of the model's adjustable parameters (its weights), the numbers training searches for
L(θ) — the training loss: one number saying how bad the parameters θ are on the whole training set
n — the number of training examples
Σᵢ — "sum over i": add up the term that follows for every example i = 1 … n
xᵢ, yᵢ — the i-th example's input and its correct answer (label)
f(xᵢ; θ) — the model's prediction for input xᵢ when it uses parameters θ
loss(prediction, y) — the mistake on a single example, for instance (prediction − y)²
:::

Read that formula as: *average the mistakes over all n training examples, and search for the parameters that make that average as small as possible.*

:::key Everything else is detail
When you meet a new technique, ask: which of the five boxes does it change? A *transformer* is a new **model**. *Cross-entropy* is a **loss**. *Adam* is an **optimizer**. *Cross-validation* is about **evaluation**. *Tokenization* is about **data**.
:::` },
      { type: 'text', title: 'AI, machine learning, deep learning and LLMs', md: `These terms nest inside each other:

- **Artificial intelligence (AI)** — any technique that makes computers do things we consider intelligent. Includes old-school rule systems and search.
  - **Machine learning (ML)** — AI that *learns from data*.
    - **Deep learning** — ML with *neural networks* of many layers. Dominant for images, audio and text since about 2012.
      - **Large language models (LLMs)** — very large deep-learning models (transformers) trained on huge amounts of text to predict the next token. ChatGPT, Claude, Gemini and Llama are LLMs.

A whirlwind history: the **perceptron** (1958) → **backpropagation** popularised (1986) → **support vector machines** and **random forests** dominate (1990s–2000s) → **AlexNet** wins ImageNet with a deep CNN on GPUs (2012) → the **Transformer** paper "Attention Is All You Need" (2017) → **GPT-2/3** show that scale works (2019–2020) → **ChatGPT** brings LLMs to everyone (2022).` },
      { type: 'text', title: 'How to use this map', md: `The **concept map** is organised like a tree that grows outward from this page:

- **Math Foundations** and **Python & Data** are the roots: the language everything else is written in.
- **Classical ML** teaches the recipe with simple, interpretable models.
- **Neural Networks** generalise those models into deep learning.
- **Transformers & LLMs** builds a GPT-style model piece by piece.
- **Building with LLMs** and **Specializations** branch out into applications and neighbouring fields.

Each concept lists its **prerequisites**. You can open anything at any time, but following the arrows means you always have the background you need. A concept is **completed** when you pass its quiz and solve at least one practice problem; **mastered** when you solve them all and score 90%+ on the quiz.` },
      { type: 'check', question: { type: 'tf', q: 'A model that scores 100% accuracy on its training data is guaranteed to work well on new data.', answer: false, explain: 'It might simply have memorised the training examples (overfitting). Only performance on unseen data tells you whether it generalises.' } }
    ],
    when: `**Machine learning is a good fit when:**
- the rules are too complex or fuzzy to write by hand (vision, speech, language);
- the patterns change over time and the system must adapt (fraud, spam, recommendations);
- you have (or can collect) enough examples of the behaviour you want;
- an occasional mistake is acceptable and can be monitored.

**Prefer ordinary code when:**
- a clear rule exists (tax calculations, unit conversion, sorting);
- you have little or no data;
- every answer must be exactly right and fully explainable;
- a simple heuristic already works — start with the simplest thing that could work, then use ML to beat it.`,
    pitfalls: `- **Garbage in, garbage out.** A model can only learn what is in its data — including its biases and errors.
- **Testing on the training data.** It measures memory, not understanding. Always keep a separate test set.
- **Mistaking correlation for causation.** A model that predicts well has found useful patterns, not necessarily causes.
- **Reaching for deep learning first.** On small tabular datasets a simple model is often as good and far easier to trust.
- **Forgetting the baseline.** Always compare against something trivial (predict the average, or the most common class).`,
    takeaways: [
      'ML learns rules (a model) from examples instead of having them written by hand.',
      'Features (x) go in, predictions (ŷ) come out; training adjusts the parameters to match the labels (y).',
      'Supervised, unsupervised, reinforcement and self-supervised learning differ in what feedback is available.',
      'Every model follows the recipe: data → model → loss → optimization → evaluation on unseen data.',
      'Deep learning ⊂ machine learning ⊂ AI, and LLMs are very large deep-learning models trained to predict the next token.'
    ],
    quiz: [
      { type: 'mcq', q: 'In machine learning, what does the computer produce from the training examples?', options: ['The labels', 'The rules (a trained model)', 'The features', 'The raw data'], answer: 1, explain: 'We supply inputs and correct outputs; the learning algorithm produces the rules — the trained model — that map one to the other.' },
      { type: 'match', q: 'Match each task with the kind of learning it uses.', pairs: [['Predict tomorrow\'s temperature from historical weather data', 'Supervised regression'], ['Decide whether an email is spam', 'Supervised classification'], ['Group news articles by topic without labels', 'Unsupervised clustering'], ['Teach an agent to win a game from scores', 'Reinforcement learning']], explain: 'Numbers → regression; categories → classification; no labels → unsupervised; rewards from actions → reinforcement learning.' },
      { type: 'order', q: 'Put the universal ML recipe in order.', items: ['Collect data and turn it into features', 'Choose a model with adjustable parameters', 'Define a loss that measures mistakes', 'Optimise the parameters to reduce the loss', 'Evaluate on data the model has never seen'], explain: 'Data → model → loss → optimization → evaluation. Evaluation must use unseen data.' },
      { type: 'mcq', q: 'How are large language models mainly pretrained?', options: ['On millions of hand-labelled question/answer pairs', 'By self-supervised next-token prediction on huge amounts of text', 'With hand-written grammar rules', 'Purely by reinforcement learning from scratch'], answer: 1, explain: 'Pretraining predicts the next token of real text; the labels come from the text itself, so it scales to trillions of tokens. Later stages (fine-tuning, RLHF) add smaller amounts of curated data.' },
      { type: 'multi', q: 'Which situations are good candidates for machine learning? (select all)', options: ['Recognising handwritten digits', 'Converting Celsius to Fahrenheit', 'Flagging unusual credit-card transactions', 'Recommending songs a user might like'], answer: [0, 2, 3], explain: 'Temperature conversion is an exact formula — just code it. The others involve fuzzy, data-driven patterns.' },
      { type: 'tf', q: 'Parameters are the numbers inside a model that are adjusted during training.', answer: true, explain: 'Yes — weights and biases are parameters. Settings you choose before training, like the learning rate, are called hyperparameters.' }
    ],
    problems: [
      {
        id: 'learn-threshold', title: 'Learn a Threshold from Data', difficulty: 'easy',
        prompt: `Welcome to your first practice problem! Write code in the editor on the right, press **Run** to check it against the examples, then **Submit** to run the hidden tests too.

We will "train" the simplest possible classifier: **predict 1 if x ≥ t, otherwise 0**, where the threshold **t** is the only parameter.

Implement two functions:

1. \`accuracy(xs, ys, t)\` — the fraction of examples (between 0 and 1) where the rule's prediction equals the label \`ys[i]\`.
2. \`fit_threshold(xs, ys)\` — try every distinct value in \`xs\` as a candidate threshold and return the one with the **highest accuracy**. If several tie, return the **smallest**.

**Example**

\`\`\`text
fit_threshold([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1])  ->  4
\`\`\`
because "x ≥ 4" classifies every example correctly.

Inputs are plain Python lists; \`ys\` contains only 0 and 1. \`xs\` is not necessarily sorted.`,
        starter: `
def accuracy(xs, ys, t):
    """Fraction of examples where (1 if x >= t else 0) equals the label."""
    # TODO: count the correct predictions and divide by len(xs)
    pass


def fit_threshold(xs, ys):
    """Return the candidate threshold (a value from xs) with the highest accuracy.
    Ties go to the smallest threshold."""
    # TODO: try each distinct value of xs (smallest first) and keep the best
    pass
`,
        solution: `
def accuracy(xs, ys, t):
    correct = 0
    for x, y in zip(xs, ys):
        prediction = 1 if x >= t else 0
        if prediction == y:
            correct += 1
    return correct / len(xs)


def fit_threshold(xs, ys):
    best_t, best_acc = None, -1.0
    for t in sorted(set(xs)):           # smallest first
        acc = accuracy(xs, ys, t)
        if acc > best_acc:              # strictly better, so ties keep the smaller t
            best_t, best_acc = t, acc
    return best_t
`,
        hints: ['In `accuracy`, loop with `for x, y in zip(xs, ys):` and compare `(1 if x >= t else 0)` with `y`.', '`sorted(set(xs))` gives the distinct candidate thresholds from smallest to largest.', 'Only replace your best threshold when the new accuracy is *strictly* greater — that automatically keeps the smallest one on ties.'],
        explain: `This is a complete, if tiny, learning algorithm: a **model family** (threshold rules), a **performance measure** (accuracy) and an **optimizer** (exhaustive search over candidate parameters). Exhaustive search works because there is one parameter with few candidates. Real models have millions of parameters, so instead of trying every value we follow the slope of the loss — gradient descent — which you will meet soon.`,
        tests: [
          { name: 'accuracy: perfect rule', call: 'accuracy([1, 2, 3, 4], [0, 0, 1, 1], 3)', expect: '1.0' },
          { name: 'accuracy: one mistake', call: 'accuracy([1, 2, 3, 4], [0, 1, 0, 1], 2)', expect: '0.75' },
          { name: 'fit: clean data', call: 'fit_threshold([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1])', expect: '4' },
          { name: 'fit: tie goes to the smallest', call: 'fit_threshold([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 1, 0, 1, 1, 1, 1])', expect: '3' },
          { name: 'unsorted input', call: 'fit_threshold([2.5, 0.5, 4.0, 3.0, 1.0], [1, 0, 1, 1, 0])', expect: '2.5', hidden: true },
          { name: 'all positive labels', call: 'fit_threshold([3, 1, 2], [1, 1, 1])', expect: '1', hidden: true },
          { name: 'noisy random data', setup: 'rng = np.random.default_rng(3)\nxs = [float(v) for v in rng.uniform(0, 10, 40).round(2)]\nys = [int(x + rng.normal(0, 1.5) > 5) for x in xs]', call: 'fit_threshold(xs, ys)', hidden: true }
        ]
      },
      {
        id: 'train-then-test', title: 'Train on One Part, Test on Another', difficulty: 'easy',
        prompt: `The universal recipe of machine learning: **learn from training data, then check on data the model has never seen.** Using the threshold rule from the first problem (predict 1 if x ≥ t, else 0), implement:

1. \`best_threshold(xs, ys)\` — try every value in \`xs\` as the threshold t and return the one with the highest accuracy on (xs, ys). If several tie, return the smallest.
2. \`holdout_split(xs, ys, test_fraction)\` — put the **last** \`round(len(xs) * test_fraction)\` examples in the test set and the rest in the training set; return \`(xs_train, ys_train, xs_test, ys_test)\`.
3. \`train_and_test(xs, ys, test_fraction=0.25)\` — split, choose t on the training part only, and return \`(t, train_accuracy, test_accuracy)\`.

\`\`\`text
best_threshold([1, 2, 3, 4], [0, 0, 1, 1])  ->  3
holdout_split([1, 2, 3, 4], [0, 0, 1, 1], 0.25)  ->  ([1, 2, 3], [0, 0, 1], [4], [1])
\`\`\``,
        starter: `
def accuracy(xs, ys, t):
    return sum((1 if x >= t else 0) == y for x, y in zip(xs, ys)) / len(xs)

def best_threshold(xs, ys):
    # TODO: try each x as a threshold; keep the best (smallest on ties)
    pass

def holdout_split(xs, ys, test_fraction):
    # TODO
    pass

def train_and_test(xs, ys, test_fraction=0.25):
    # TODO
    pass
`,
        solution: `
def accuracy(xs, ys, t):
    return sum((1 if x >= t else 0) == y for x, y in zip(xs, ys)) / len(xs)

def best_threshold(xs, ys):
    best_t, best_acc = None, -1.0
    for t in sorted(xs):
        acc = accuracy(xs, ys, t)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t

def holdout_split(xs, ys, test_fraction):
    n_test = round(len(xs) * test_fraction)
    cut = len(xs) - n_test
    return list(xs[:cut]), list(ys[:cut]), list(xs[cut:]), list(ys[cut:])

def train_and_test(xs, ys, test_fraction=0.25):
    xtr, ytr, xte, yte = holdout_split(xs, ys, test_fraction)
    t = best_threshold(xtr, ytr)
    return t, accuracy(xtr, ytr, t), accuracy(xte, yte, t)
`,
        hints: ['Loop over the thresholds in ascending order and only replace the best when the accuracy is strictly higher.', 'Slicing with a cut point: xs[:cut] and xs[cut:].', 'The threshold must be chosen without looking at the test part.'],
        explain: `Training accuracy measures how well the rule fits the examples it was chosen on; test accuracy estimates how it will do on new data. The gap between them — here usually small, because a threshold has only one parameter — is the generalization gap you will meet again in *Overfitting* and *Generalization Theory*.`,
        tests: [
          { call: 'best_threshold([1, 2, 3, 4], [0, 0, 1, 1])', expect: '3' },
          { call: 'holdout_split([1, 2, 3, 4], [0, 0, 1, 1], 0.25)', expect: '([1, 2, 3], [0, 0, 1], [4], [1])' },
          { call: 'train_and_test([1, 5, 2, 6, 3, 7, 4, 8], [0, 1, 0, 1, 0, 1, 0, 1], 0.25)', expect: '(5, 1.0, 1.0)' },
          { name: 'ties go to the smallest threshold', call: 'best_threshold([2, 4, 6], [1, 1, 1])', expect: '2', hidden: true },
          { name: 'noisy data vs reference', setup: 'import random\nrandom.seed(0)\nxs = [random.uniform(0, 10) for _ in range(40)]\nys = [1 if x + random.gauss(0, 1.5) > 5 else 0 for x in xs]', call: 'train_and_test(xs, ys, 0.3)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Google — Machine Learning Crash Course', url: 'https://developers.google.com/machine-learning/crash-course', note: 'free, practical introduction' },
      { title: 'Dive into Deep Learning (d2l.ai)', url: 'https://d2l.ai/', note: 'free interactive textbook' }
    ]
  },

  // =====================================================================================
  {
    id: 'vectors-matrices', branch: 'math', title: 'Vectors, Matrices & the Dot Product', short: 'Vectors & Matrices', level: 1, minutes: 35, prereqs: ['what-is-ml'],
    tags: ['linear algebra', 'dot product', 'matrix multiplication', 'cosine similarity'],
    summary: 'How data becomes numbers: vectors for single examples, matrices for whole datasets, and the dot product that powers almost every model.',
    why: `Every input to a model is a **vector** of numbers: a house is [size, bedrooms, age], a 28×28 image is 784 pixel brightnesses, and each word inside an LLM is a vector of thousands of numbers. Whole datasets are **matrices**, and most of what a neural network does is **multiply matrices**. That is literally why GPUs — chips built to multiply matrices fast — power modern AI.

If you understand the dot product and how matrix shapes fit together, you can read the code of any model, debug the most common errors ("shapes not aligned"), and understand why attention in a transformer is "just" a big batch of dot products.`,
    objectives: [
      'Represent an example as a vector and a dataset as a matrix with shape (rows, columns)',
      'Compute a dot product by hand and in NumPy, and interpret it both as a weighted sum and as similarity',
      'Compute vector lengths (norms) and cosine similarity',
      'Multiply matrices and predict the shape of the result before running the code'
    ],
    sections: [
      { type: 'text', title: 'Vectors: lists of numbers with meaning', md: `A **vector** is an ordered list of numbers. The position matters: in \`[120, 3, 10]\` the first number might be size in m², the second the bedrooms and the third the age of a house. The number of entries is its **dimension** (here 3).

Geometrically, a 2-D vector like [3, 4] is an arrow from the origin to the point (3, 4). That picture generalises — we can't *draw* 4,096 dimensions, but distances, angles and directions still work exactly the same way.

Two basic operations, both done **element by element**:

$$ [1, 2] + [10, 20] = [11, 22]          3 · [1, 2] = [3, 6] $$
:::where
[a, b] — a vector: an ordered list of numbers (here with two elements)
+ — vector addition: add matching positions, first with first and second with second
3 · x — scalar multiplication: multiply every element of the vector x by the number 3
:::

The **length** (or *norm*) of a vector comes from Pythagoras:

$$ ‖x‖ = √(x₁² + x₂² + … + xₙ²)          ‖[3, 4]‖ = √(9 + 16) = 5 $$
:::where
x — a vector with n elements x₁, x₂, …, xₙ
xᵢ — the i-th element (component) of x; the small subscript is a position, not a power
‖x‖ — the length of x (its Euclidean or L2 norm): the straight-line distance from the origin to the arrow's tip
√ — the square root
n — the number of elements, also called the dimension of the vector
:::` },
      { type: 'text', title: 'The dot product', md: `The **dot product** multiplies two vectors of the same length element by element and adds everything up:

$$ a · b = a₁b₁ + a₂b₂ + … + aₙbₙ $$
:::where
a, b — two vectors with the same number of elements, n
aᵢ, bᵢ — their i-th elements, multiplied pairwise
a · b — the dot product: a single number (a scalar), not a vector
:::

Example: [1, 2, 3] · [4, 5, 6] = 1·4 + 2·5 + 3·6 = **32**.

It has two interpretations, and you will use both constantly:

**1. A weighted sum.** A linear model predicts with *weights · features + bias*. If the weights are [2000, 15000, −800] (€ per m², € per bedroom, € per year of age), the dot product with a house's features adds up each feature's contribution. A single artificial neuron is exactly this.

**2. A similarity score.** The dot product also equals

$$ a · b = ‖a‖ ‖b‖ cos θ $$
:::where
‖a‖, ‖b‖ — the lengths (norms) of the two vectors
θ — the angle between the two vectors, drawn as arrows from the origin
cos θ — the cosine of that angle: 1 when they point the same way, 0 at a right angle, −1 when they point in opposite directions
:::

where θ is the angle between the arrows. Pointing the same way → large and positive; perpendicular → 0; opposite → negative. Dividing out the lengths gives **cosine similarity**, a pure measure of direction between −1 and 1:

$$ cos_sim(a, b) = (a · b) / (‖a‖ ‖b‖) $$
:::where
cos_sim(a, b) — cosine similarity: a score from −1 (opposite directions) to 1 (same direction) that ignores how long the vectors are
a · b — the dot product of the two vectors
‖a‖ ‖b‖ — the product of their lengths; dividing by it removes the effect of size
:::

:::real Where you will meet this
Search engines and RAG systems find relevant documents by cosine similarity between embedding vectors. In a transformer, a token decides what to pay attention to by taking dot products between its *query* vector and every other token's *key* vector.
:::` },
      { type: 'code', title: 'Dot products in Python and NumPy', md: `NumPy is Python's array library (you will study it properly soon). The \`@\` operator computes dot products and matrix products.`, code: `
import numpy as np

# Two houses described by the same 3 features: [size m², bedrooms, age in years]
house_a = np.array([120.0, 3.0, 10.0])
house_b = np.array([ 80.0, 2.0, 35.0])

# A linear model is a dot product: weights · features + bias
weights = np.array([2000.0, 15000.0, -800.0])   # € per m², € per bedroom, € per year of age
bias = 50000.0
print("price A:", house_a @ weights + bias)      # @ is the dot product
print("price B:", np.dot(house_b, weights) + bias)

# The same thing written out by hand
manual = sum(f * w for f, w in zip(house_a, weights)) + bias
print("by hand:", manual)

# Dot product as similarity: cosine similarity keeps the direction, ignores the length
def cosine(u, v):
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))

cat    = np.array([0.9, 0.8, 0.1])   # toy "meaning" vectors: [animal, pet, vehicle]
kitten = np.array([0.8, 0.9, 0.0])
truck  = np.array([0.0, 0.1, 0.9])
print("cat ~ kitten:", round(cosine(cat, kitten), 3))
print("cat ~ truck: ", round(cosine(cat, truck), 3))
` },
      { type: 'check', question: { type: 'mcq', q: 'What is the cosine similarity between [2, 0] and [−5, 0]?', options: ['1', '0', '−1', '−10'], answer: 2, explain: 'They point in exactly opposite directions (180°), and cos 180° = −1. The lengths do not matter for cosine similarity.' } },
      { type: 'text', title: 'Matrices: a whole dataset at once', md: `A **matrix** is a 2-D grid of numbers with a **shape** written (rows, columns). The convention in ML: **one row per example, one column per feature**. Four houses with three features each form a 4×3 matrix **X**.

Multiplying the matrix by a weight vector computes the dot product of *every row* with the weights — a prediction for every house in one operation:

$$ X w  =  [ x⁽¹⁾·w,  x⁽²⁾·w,  x⁽³⁾·w,  x⁽⁴⁾·w ] $$
:::where
X — the data matrix, shape (4, 3): one row per house, one column per feature
w — the weight vector: one weight per feature (3 numbers)
x⁽ⁱ⁾ — row i of X, the feature vector of house i (the bracketed superscript is an index, not a power)
x⁽ⁱ⁾·w — the dot product of that row with the weights, which is the prediction for house i
:::

This is why ML code rarely loops over examples: one matrix operation handles the whole batch, and NumPy or a GPU does it far faster than a Python loop.

The **transpose** Xᵀ flips rows and columns: a (4, 3) matrix becomes (3, 4).` },
      { type: 'text', title: 'Matrix multiplication and the shape rule', md: `To multiply **A** (shape n×k) by **B** (shape k×m):

$$ (n × k) @ (k × m)  →  (n × m)          C[i][j] = row i of A · column j of B $$
:::where
A, B — the two matrices being multiplied; C = A @ B is the result
(n × k) — a shape: n rows and k columns
@ — matrix multiplication (the Python operator for it)
k — the shared inner dimension: A's number of columns must equal B's number of rows
C[i][j] — the entry in row i, column j of the result, which is the dot product of row i of A with column j of B
:::

- The **inner** dimensions (k and k) must match — every row of A must be as long as every column of B.
- The **outer** dimensions give the result's shape.

Worked example:

$$ [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[1·5+2·7, 1·6+2·8], [3·5+4·7, 3·6+4·8]] = [[19, 22], [43, 50]] $$
:::where
[[1, 2], [3, 4]] — a 2 × 2 matrix written row by row: the first row is [1, 2], the second is [3, 4]
1·5 + 2·7 — row 1 of the left matrix · column 1 of the right matrix, which gives the top-left entry, 19
:::

Matrix multiplication is **not commutative**: A @ B usually differs from B @ A (and often one of them is not even allowed).

:::key Why neural networks are matrix multiplications
A layer with 784 inputs and 128 outputs stores a weight matrix W of shape (784, 128). Feeding a batch of 32 images, X of shape (32, 784), through the layer is X @ W → (32, 128): 128 dot products for each of 32 images, in one call.
:::` },
      { type: 'code', title: 'Shapes in practice', md: `Get in the habit of printing \`.shape\`. Most bugs in ML code are shape bugs.`, code: `
import numpy as np

X = np.array([[120, 3, 10],      # 4 houses (rows) x 3 features (columns)
              [ 80, 2, 35],
              [200, 5,  2],
              [ 60, 1, 50]], dtype=float)
w = np.array([2000.0, 15000.0, -800.0])
print("X shape:", X.shape, " w shape:", w.shape)

prices = X @ w + 50000            # (4, 3) @ (3,) -> (4,): one prediction per house
print("predictions:", prices)

W = np.random.default_rng(0).normal(size=(3, 2))   # a "layer" with 2 outputs
H = X @ W                         # (4, 3) @ (3, 2) -> (4, 2)
print("H shape:", H.shape)
print("X.T shape:", X.T.shape)    # transpose swaps rows and columns

try:
    X @ X                         # (4, 3) @ (4, 3): inner dimensions 3 and 4 differ
except ValueError as err:
    print("shape error:", err)
`, predict: {"type":"mcq","q":"X has shape (4, 3) and W has shape (3, 2). What is the shape of H = X @ W?","options":["(4, 2)","(3, 3)","(4, 3)","It raises a shape error"],"answer":0,"explain":"(4, 3) @ (3, 2) → (4, 2): the inner dimensions (3 and 3) must match and disappear, the outer ones remain. One row per example, one column per output.","expect":["H shape: (4, 2)"]} },
      { type: 'check', question: { type: 'fill', q: 'A batch of 32 flattened images has shape (32, 784) and a weight matrix has shape (784, 128). What is the shape of their product? (write it like (a, b))', answer: ['(32, 128)', '(32,128)', '32, 128', '32x128', '32 x 128', '32×128'], display: '(32, 128)', explain: 'Inner dimensions 784 and 784 match and cancel; the outer dimensions give (32, 128).' } },
      { type: 'text', title: 'Element-wise versus matrix products', md: `A frequent confusion in NumPy:

| Code | What it does | Shapes |
| --- | --- | --- |
| \`a * b\` | **element-wise** product | same shape (or broadcastable) |
| \`a @ b\` or \`np.dot(a, b)\` | dot product / matrix product | inner dimensions must match |
| \`np.sum(a * b)\` | dot product of two vectors, the long way | same shape |

\`[1, 2, 3] * [4, 5, 6]\` gives \`[4, 10, 18]\`; adding those up gives the dot product, 32.` }
    ],
    when: `- Use **vectors** whenever you represent one example, one word, one image or one set of weights.
- Use **matrix products** instead of Python loops whenever you apply the same operation to many examples — it is clearer and often 100× faster.
- Use **cosine similarity** when only direction matters (comparing texts or embeddings of different lengths); use the plain **dot product** when magnitude carries meaning (a model's score), and **Euclidean distance** when absolute position matters (k-nearest neighbours on scaled features).`,
    pitfalls: `- **Shape mismatches.** Before multiplying, write the shapes down: (n×k)@(k×m). If they don't line up, you may need a transpose.
- **\`*\` is not \`@\`.** \`*\` multiplies element by element.
- **1-D vs 2-D arrays.** NumPy treats a shape \`(3,)\` vector differently from a \`(3, 1)\` column. When something broadcasts strangely, check for a stray extra dimension.
- **Comparing raw dot products of vectors with very different lengths.** Longer vectors get bigger dot products regardless of direction — normalise first if you mean "similar direction".`,
    takeaways: [
      'An example is a vector of features; a dataset is a matrix with one row per example.',
      'The dot product a·b = Σ aᵢbᵢ is both a weighted sum (a linear model, a neuron) and a similarity score.',
      'Cosine similarity = dot product ÷ product of lengths; it ranges from −1 to 1.',
      '(n×k) @ (k×m) → (n×m): inner dimensions must match.',
      'Neural network layers, attention and embedding search are all built from dot products and matrix multiplications.'
    ],
    quiz: [
      { type: 'fill', q: 'Compute the dot product [2, −1, 3] · [4, 5, −2].', answer: ['-3'], explain: '2·4 + (−1)·5 + 3·(−2) = 8 − 5 − 6 = −3.' },
      { type: 'mcq', q: 'Which shapes can be multiplied as A @ B?', options: ['A (5, 3) and B (5, 3)', 'A (5, 3) and B (3, 7)', 'A (3, 5) and B (3, 7)', 'A (7, 1) and B (5, 1)'], answer: 1, explain: 'The inner dimensions must match: (5, 3) @ (3, 7) → (5, 7).' },
      { type: 'tf', q: 'For matrices, A @ B always equals B @ A.', answer: false, explain: 'Matrix multiplication is not commutative. The products are usually different, and often only one of them has compatible shapes.' },
      { type: 'mcq', q: 'Two embedding vectors have cosine similarity 0.98. What does that tell you?', options: ['They are almost identical in length', 'They point in almost the same direction — likely similar meanings', 'They are perpendicular', 'One is the negative of the other'], answer: 1, explain: 'Cosine similarity measures the angle. 0.98 means a tiny angle, which for embeddings indicates very similar meaning.' },
      { type: 'fill', q: 'What is the length (L2 norm) of the vector [6, 8]?', answer: ['10', '10.0'], explain: '√(6² + 8²) = √100 = 10.' },
      { type: 'mcq', q: 'In NumPy, what does `a * b` compute for two vectors of the same length?', options: ['Their dot product', 'Their element-wise product (a new vector)', 'Their cosine similarity', 'An error'], answer: 1, explain: '`*` multiplies element by element. Use `a @ b` or `np.dot(a, b)` for the dot product.' }
    ],
    problems: [
      {
        id: 'dot-product', title: 'Dot Product from Scratch', difficulty: 'easy',
        prompt: `Implement \`dot(a, b)\` for two Python lists of numbers **without NumPy**.

- Return the sum of the element-wise products: a₁b₁ + a₂b₂ + … + aₙbₙ.
- If the lists have different lengths, \`raise ValueError(...)\` with any message.
- The dot product of two empty lists is 0.

\`\`\`text
dot([1, 2, 3], [4, 5, 6])  ->  32
dot([1, 0], [0, 1])        ->  0      (perpendicular)
\`\`\``,
        starter: `
def dot(a, b):
    """Dot product of two equal-length lists. Raise ValueError if the lengths differ."""
    # TODO
    pass
`,
        solution: `
def dot(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    total = 0
    for x, y in zip(a, b):
        total += x * y
    return total
`,
        hints: ['Check `len(a) != len(b)` first and `raise ValueError("...")`.', 'Start a running `total = 0`, then loop over `zip(a, b)` adding `x * y`.'],
        explain: `\`zip\` walks both lists in step. Starting the total at 0 makes the empty case work for free. In real code you would call \`np.dot\` or use \`@\` — they run in optimised C and are roughly 100× faster — but writing it once by hand makes it obvious what every neuron is computing.`,
        tests: [
          { call: 'dot([1, 2, 3], [4, 5, 6])', expect: '32' },
          { call: 'dot([1, 0], [0, 1])', expect: '0' },
          { call: 'dot([2.5, -1], [2, 4])', expect: '1.0' },
          { name: 'empty vectors', call: 'dot([], [])', expect: '0', hidden: true },
          { name: 'random vectors', setup: 'rng = np.random.default_rng(7)\na = rng.normal(size=25).tolist()\nb = rng.normal(size=25).tolist()', call: 'dot(a, b)', expect: 'float(np.dot(a, b))', hidden: true },
          { name: 'different lengths raise ValueError', code: 'try:\n    dot([1, 2], [1, 2, 3])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError("dot() should raise ValueError when the lengths differ")', hidden: true }
        ]
      },
      {
        id: 'matmul-loops', title: 'Matrix Multiplication with Loops', difficulty: 'medium',
        prompt: `Implement \`matmul(A, B)\` for matrices stored as **lists of lists** (no NumPy).

- \`A\` has shape n×k (n rows of length k) and \`B\` has shape k×m.
- Return the n×m product as a list of lists, where \`C[i][j]\` is the dot product of row i of A with column j of B.
- If the inner dimensions do not match, \`raise ValueError\`.

\`\`\`text
matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])  ->  [[19, 22], [43, 50]]
\`\`\``,
        starter: `
def matmul(A, B):
    """Multiply an n x k matrix A by a k x m matrix B (lists of lists)."""
    n, k = len(A), len(A[0])
    # TODO: check that B has k rows, then build the n x m result with three nested loops
    pass
`,
        solution: `
def matmul(A, B):
    n, k = len(A), len(A[0])
    if len(B) != k:
        raise ValueError("inner dimensions do not match: %d vs %d" % (k, len(B)))
    m = len(B[0])
    C = [[0] * m for _ in range(n)]
    for i in range(n):            # each row of A
        for j in range(m):        # each column of B
            s = 0
            for t in range(k):    # dot product of row i and column j
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C
`,
        hints: ['The number of rows of B must equal the number of columns of A (`len(A[0])`).', 'Create the result with `[[0] * m for _ in range(n)]` — not `[[0] * m] * n`, which repeats the *same* row object.', 'Column j of B is `B[0][j], B[1][j], ...` — loop over the shared index t.'],
        explain: `Three nested loops: for each output cell (i, j), take the dot product of row i of A with column j of B. That is n·m·k multiply-adds — for two 4096×4096 matrices about 69 billion. Libraries like NumPy and GPUs use cache-friendly blocking and parallel hardware to do this enormously faster, which is why you should never write this loop in real code — but now you know exactly what \`A @ B\` means.`,
        tests: [
          { call: 'matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])', expect: '[[19, 22], [43, 50]]' },
          { name: 'row times column', call: 'matmul([[1, 2, 3]], [[1], [0], [-1]])', expect: '[[-2]]' },
          { name: '(3x2) @ (2x3)', call: 'matmul([[2, 0], [0, 2], [1, 1]], [[1, 2, 3], [4, 5, 6]])', expect: '[[2, 4, 6], [8, 10, 12], [5, 7, 9]]' },
          { name: 'random integer matrices', setup: 'rng = np.random.default_rng(11)\nA = rng.integers(-5, 6, size=(4, 3)).tolist()\nB = rng.integers(-5, 6, size=(3, 5)).tolist()', call: 'matmul(A, B)', expect: '(np.array(A) @ np.array(B)).tolist()', hidden: true },
          { name: 'identity matrix', call: 'matmul([[1, 0], [0, 1]], [[7, 8], [9, 10]])', expect: '[[7, 8], [9, 10]]', hidden: true },
          { name: 'shape mismatch raises ValueError', code: 'try:\n    matmul([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError("(2x3) @ (2x3) should raise ValueError")', hidden: true }
        ]
      },
      {
        id: 'cosine-similarity', title: 'Cosine Similarity', difficulty: 'easy',
        prompt: `Implement \`cosine_similarity(a, b)\` with NumPy:

$$ cos_sim(a, b) = (a · b) / (‖a‖ ‖b‖) $$
:::where
a · b — the dot product, Σ aᵢbᵢ (np.dot(a, b) or a @ b)
‖a‖ — the length of a, √(Σ aᵢ²) (np.linalg.norm(a))
cos_sim(a, b) — the result, a number between −1 and 1
:::

- Inputs may be lists or NumPy arrays of the same length.
- If either vector has length (norm) 0, return \`0.0\` instead of dividing by zero.
- Return a plain Python \`float\`.

\`\`\`text
cosine_similarity([1, 2, 3], [2, 4, 6])  ->  1.0    (same direction)
cosine_similarity([1, 0], [0, 1])        ->  0.0    (perpendicular)
\`\`\``,
        starter: `
import numpy as np

def cosine_similarity(a, b):
    """Cosine of the angle between a and b. Return 0.0 if either has zero length."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # TODO: dot product divided by the product of the norms (np.linalg.norm)
    pass
`,
        solution: `
import numpy as np

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))
`,
        hints: ['`np.linalg.norm(a)` computes √(Σ aᵢ²).', 'Check for a zero norm *before* dividing.'],
        explain: `Cosine similarity is the dot product of the two **unit** vectors a/‖a‖ and b/‖b‖, so length is divided out and only direction remains. It is the default similarity measure for text embeddings: a long document and a short one about the same topic should still count as similar.`,
        tests: [
          { call: 'cosine_similarity([1, 2, 3], [2, 4, 6])', expect: '1.0' },
          { call: 'cosine_similarity([1, 0], [0, 1])', expect: '0.0' },
          { call: 'cosine_similarity([3, 4], [4, 3])', expect: '0.96' },
          { name: 'opposite directions', call: 'cosine_similarity([1, 1], [-2, -2])', expect: '-1.0', hidden: true },
          { name: 'zero vector', call: 'cosine_similarity([0, 0, 0], [1, 2, 3])', expect: '0.0', hidden: true },
          { name: 'random 50-d vectors', setup: 'rng = np.random.default_rng(5)\na = rng.normal(size=50)\nb = rng.normal(size=50)', call: 'cosine_similarity(a, b)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: '3Blue1Brown — Essence of Linear Algebra', url: 'https://www.3blue1brown.com/topics/linear-algebra', note: 'beautiful visual intuition' },
      { title: 'NumPy: the absolute basics for beginners', url: 'https://numpy.org/doc/stable/user/absolute_beginners.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'derivatives', branch: 'math', title: 'Derivatives, Gradients & the Chain Rule', short: 'Derivatives & Gradients', level: 1, minutes: 40, prereqs: ['what-is-ml'],
    tags: ['calculus', 'gradient', 'chain rule', 'partial derivative'],
    summary: 'The mathematics of "which way is downhill": slopes, partial derivatives, gradients and the chain rule that makes backpropagation possible.',
    why: `Training a model means adjusting its parameters to make the loss smaller. The **derivative** answers the key question: *if I nudge this parameter a little, does the loss go up or down, and how fast?* The **gradient** collects that answer for every parameter at once — millions of them — and points uphill, so stepping the opposite way reduces the loss. The **chain rule** lets us compute gradients through long chains of computations, which is exactly what **backpropagation** does in every neural network.

You do not need to become a calculus expert. You need the intuition (slope = sensitivity), a handful of rules, and the chain rule. This lesson gives you all three, with code to check every claim numerically.`,
    objectives: [
      'Explain a derivative as a slope / sensitivity and estimate it numerically',
      'Apply the basic derivative rules, including those of sigmoid and ReLU',
      'Compute partial derivatives and assemble them into a gradient vector',
      'Use the chain rule on a composition of functions and connect it to backpropagation'
    ],
    sections: [
      { type: 'text', title: 'A derivative is a slope', md: `Take f(x) = x². Between x = 3 and x = 3.001 the output changes from 9 to 9.006001. The ratio "change in output ÷ change in input" is

$$ (9.006001 − 9) / 0.001 = 6.001 $$
:::where
9.006001 − 9 — the change in output, f(3.001) − f(3)
0.001 — the change in input (the step size)
6.001 — the average slope over that tiny step: change in output ÷ change in input
:::

As the step gets smaller the ratio settles on **6**. That limit is the **derivative** of f at 3, written **f′(3)** or **df/dx**:

$$ f′(x) = lim (h → 0) of  [ f(x + h) − f(x) ] / h $$
:::where
f′(x) — "f prime of x": the derivative, the slope of f exactly at the point x
h — a small step added to the input
f(x + h) − f(x) — how much the output changes over that step
lim (h → 0) — "the limit as h goes to 0": the value the ratio settles on as the step shrinks towards zero
:::

For x² the derivative is **2x** — at x = 3 that is 6. ✓

How to read it:
- **f′(x) > 0** — increasing x increases f (the curve goes uphill to the right).
- **f′(x) < 0** — increasing x decreases f.
- **f′(x) = 0** — the curve is flat: a minimum, a maximum or a plateau.
- The **size** tells you how *sensitive* the output is: a derivative of 6 means "a tiny nudge of ε in x changes f by about 6ε".

:::key Why ML cares
Replace x with a model parameter and f with the loss. The derivative says which direction to move the parameter to reduce the loss, and how strongly the loss reacts. That is all an optimizer needs.
:::` },
      { type: 'text', title: 'The rules you actually need', md: `You rarely differentiate by hand in practice (libraries do it), but these rules let you understand what the libraries compute:

| Function | Derivative | Note |
| --- | --- | --- |
| c (a constant) | 0 | constants don't change |
| xⁿ | n·xⁿ⁻¹ | x² → 2x, x³ → 3x² |
| eˣ | eˣ | its own derivative |
| ln x | 1/x | used in log-likelihood losses |
| a·f(x) + g(x) | a·f′(x) + g′(x) | derivatives add and scale |
| f(x)·g(x) | f′g + f g′ | product rule |
| sigmoid σ(x) = 1/(1+e⁻ˣ) | σ(x)·(1 − σ(x)) | at most 0.25 |
| ReLU max(0, x) | 1 if x > 0 else 0 | the most common activation |

Example: the squared error (ŷ − y)² has derivative **2(ŷ − y)** with respect to ŷ — large errors produce large corrections.` },
      { type: 'code', title: 'Estimating derivatives numerically', md: `You can always *estimate* a derivative by nudging the input. The **central difference** (look a little left *and* a little right) is far more accurate than the one-sided version, and it is how ML engineers sanity-check gradient code (*gradient checking*).`, code: `
def f(x):
    return x ** 2

def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

x = 3.0
print("exact derivative 2x =", 2 * x)
for h in [0.1, 0.01, 0.001]:
    print(f"h={h:<6} forward={forward_diff(f, x, h):.6f}   central={central_diff(f, x, h):.6f}")
`, after: `The forward difference has an error proportional to h; the central difference's error shrinks like h² — for a parabola it is exact. Don't make h *too* small, though (like 1e-15): floating-point rounding then dominates.`, predict: {"type":"mcq","q":"Both formulas estimate the derivative of x² at x = 3 (exactly 6) with the same step h = 0.1. Which one prints the more accurate value?","options":["The forward difference (f(x+h) − f(x)) / h","The central difference (f(x+h) − f(x−h)) / 2h","Both give exactly the same number","Neither: h = 0.1 is far too large for either to be close"],"answer":1,"explain":"For x² the central difference is exact (6.000000) at every h, while the forward difference is off by h (6.1 at h = 0.1). In general the central error shrinks like h², the forward error only like h.","expect":["h=0.1    forward=6.100000   central=6.000000"]} },
      { type: 'check', question: { type: 'fill', q: 'f(x) = x³. What is f′(2)?', answer: ['12', '12.0'], explain: 'f′(x) = 3x², so f′(2) = 3·4 = 12.' } },
      { type: 'text', title: 'Partial derivatives and the gradient', md: `Models have many parameters, so the loss is a function of many variables. Take a one-feature linear model with weight w and bias b, and a single training example x = 2, y = 7:

$$ L(w, b) = (w·2 + b − 7)² $$
:::where
w, b — the model's two parameters: the weight and the bias
w·2 + b — the model's prediction for the input x = 2
7 — the correct answer y for that input
L(w, b) — the squared error (prediction − y)², which depends on both parameters
:::

A **partial derivative** measures the effect of one variable while holding the others fixed:

$$ ∂L/∂w = 2(2w + b − 7) · 2          ∂L/∂b = 2(2w + b − 7) · 1 $$
:::where
∂L/∂w — the partial derivative of L with respect to w: how fast the loss changes when only w moves and b is held fixed
∂ — "partial": the curly d used when a function has several inputs
2(2w + b − 7) — the derivative of the square, 2 × the error (the chain rule's outer part)
· 2 and · 1 — how the prediction 2w + b changes with w (by 2) and with b (by 1)
:::

At w = 1, b = 1 the error is 2 + 1 − 7 = −4, so ∂L/∂w = −16 and ∂L/∂b = −8. Both are negative: increasing w or b would *reduce* the loss.

The **gradient** stacks all partial derivatives into one vector:

$$ ∇L = [ ∂L/∂w, ∂L/∂b ] = [−16, −8] $$
:::where
∇L — "nabla L" or "grad L": the gradient, a vector holding one partial derivative per parameter
[−16, −8] — its value at w = 1, b = 1; it points in the direction in which the loss increases fastest
:::

Two facts make the gradient the workhorse of ML:
1. It points in the direction of **steepest increase** of the function.
2. So **−∇L points downhill**. Gradient descent repeatedly takes a small step in that direction: θ ← θ − η·∇L(θ).

:::analogy Lost in fog on a hillside
You can't see the valley, but you can feel the slope under your feet. Stepping in the steepest downhill direction, again and again, eventually brings you to low ground. The gradient is "the slope under your feet" in every direction at once.
:::` },
      { type: 'check', question: { type: 'mcq', q: 'The gradient of the loss with respect to the parameters points in which direction?', options: ['Towards the minimum of the loss', 'The direction of steepest increase of the loss', 'Always towards zero', 'Perpendicular to the steepest direction'], answer: 1, explain: 'The gradient points uphill (steepest ascent). To reduce the loss we step in the opposite direction: −gradient.' } },
      { type: 'text', title: 'The chain rule', md: `Models are **compositions**: the input goes through one function, then another, then another. The chain rule tells us how to differentiate a chain:

$$ if  y = f(g(x))  then  dy/dx = f′(g(x)) · g′(x) $$
:::where
g — the inner function, applied first: u = g(x)
f — the outer function, applied to g's output: y = f(u)
dy/dx — how fast the final output y changes when the input x changes
f′(g(x)) — the slope of the outer function, measured at the point the inner function produced
g′(x) — the slope of the inner function at x
:::

In words: **multiply the local rates of change along the chain.**

Example: y = (3x + 1)². Let u = 3x + 1 (inner) and y = u² (outer). Then dy/du = 2u and du/dx = 3, so

$$ dy/dx = 2u · 3 = 6(3x + 1) $$
:::where
u — the inner expression, 3x + 1
2u — dy/du, the derivative of the outer function u²
3 — du/dx, the derivative of the inner function 3x + 1
:::

:::analogy Gears
If gear A turns 3 times for each turn of gear B, and B turns 2 times for each turn of C, then A turns 3 × 2 = 6 times per turn of C. Rates along a chain multiply.
:::

**This is backpropagation.** A neural network computes a long chain: input → layer 1 → activation → layer 2 → … → loss. To find how the loss depends on a weight deep inside, backprop starts at the loss and walks backwards, multiplying each step's local derivative — the chain rule, applied efficiently to millions of parameters at once.` },
      { type: 'code', title: 'Checking the chain rule on a neuron', md: `A single neuron with a sigmoid computes ŷ = σ(w·x + b). By the chain rule, dŷ/dw = σ′(z)·dz/dw = σ(z)(1 − σ(z))·x. Let's confirm numerically.`, code: `
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

x, w, b = 2.0, 0.5, -0.3

# chain rule: d/dw sigmoid(w*x + b) = sigmoid'(z) * dz/dw = sigmoid(z) * (1 - sigmoid(z)) * x
z = w * x + b
analytic = sigmoid(z) * (1 - sigmoid(z)) * x

# numerical check: nudge w a tiny bit in both directions
h = 1e-6
numeric = (sigmoid((w + h) * x + b) - sigmoid((w - h) * x + b)) / (2 * h)

print("chain rule:", analytic)
print("numerical :", numeric)
print("difference:", abs(analytic - numeric))
` },
      { type: 'check', question: { type: 'mcq', q: 'y = sin(x²). Using the chain rule, dy/dx equals…', options: ['cos(x²)', '2x·cos(x²)', 'cos(2x)', '2x·sin(x²)'], answer: 1, explain: 'Outer function sin(u) has derivative cos(u); inner u = x² has derivative 2x. Multiply: cos(x²)·2x.' } }
    ],
    when: `- Think in **derivatives** whenever you ask "how sensitive is the output to this input or parameter?" — tuning, debugging, feature importance.
- Use **numerical derivatives** to *check* gradient code (gradient checking) or for a handful of parameters; they are too slow for training big models (two function evaluations per parameter).
- Use **analytic gradients via the chain rule** — in practice, automatic differentiation in PyTorch/JAX — for training.`,
    pitfalls: `- **Confusing the gradient's direction.** It points *uphill*; gradient descent subtracts it.
- **Step size in numerical derivatives.** Too large → inaccurate; too small (≲1e-10) → floating-point noise. 1e-5 to 1e-6 with central differences is a good default.
- **Forgetting the inner derivative** in the chain rule (the "·g′(x)" part) — the classic error.
- **Non-differentiable points.** ReLU has a kink at 0; libraries simply pick 0 (or 1) there and it works fine in practice.`,
    takeaways: [
      'A derivative is a slope: how much the output changes per tiny change in the input.',
      'Positive derivative → increasing; negative → decreasing; zero → flat (possibly a minimum).',
      'The gradient is the vector of partial derivatives and points in the direction of steepest increase.',
      'Gradient descent steps against the gradient: θ ← θ − η·∇L.',
      'The chain rule multiplies local derivatives along a chain of functions — this is backpropagation.'
    ],
    quiz: [
      { type: 'mcq', q: 'If ∂L/∂w = −3 at the current weights, what happens to the loss if we increase w slightly?', options: ['It increases', 'It decreases', 'It stays the same', 'It cannot be determined'], answer: 1, explain: 'A negative derivative means the loss goes down as w goes up (approximately by 3 × the change).' },
      { type: 'fill', q: 'What is the derivative of f(x) = 5x² + 3x − 7?', answer: ['10x + 3', '10x+3', '3 + 10x', '3+10x', '10*x + 3', '10*x+3'], display: '10x + 3', explain: 'Differentiate term by term: 5·2x + 3 + 0 = 10x + 3.' },
      { type: 'mcq', q: 'What is the derivative of the sigmoid σ(x) in terms of σ itself?', options: ['σ(x)²', 'σ(x)·(1 − σ(x))', '1 − σ(x)', 'e⁻ˣ'], answer: 1, explain: 'σ′(x) = σ(x)(1 − σ(x)). Its maximum is 0.25 at x = 0, which matters for vanishing gradients.' },
      { type: 'tf', q: 'The central difference (f(x+h) − f(x−h)) / 2h is usually more accurate than the forward difference (f(x+h) − f(x)) / h.', answer: true, explain: 'Its error shrinks like h² instead of h.' },
      { type: 'mcq', q: 'Why is the chain rule central to training neural networks?', options: ['It removes the need for a loss function', 'It lets us compute how the loss depends on every weight by multiplying local derivatives backwards through the network', 'It guarantees finding the global minimum', 'It makes networks linear'], answer: 1, explain: 'Backpropagation is the chain rule applied systematically from the loss back to every parameter.' },
      { type: 'fill', q: 'L(w) = (2w − 6)². What is dL/dw at w = 1?', answer: ['-16', '-16.0'], explain: 'dL/dw = 2(2w − 6)·2. At w = 1: 2·(−4)·2 = −16.' }
    ],
    problems: [
      {
        id: 'numerical-derivative', title: 'Numerical Derivative', difficulty: 'easy',
        prompt: `Implement \`numerical_derivative(f, x, h=1e-5)\` that estimates f′(x) with the **central difference**:

$$ f′(x) ≈ ( f(x + h) − f(x − h) ) / (2h) $$
:::where
f′(x) — the derivative (slope) of f at x, which you are estimating
h — a small step size, such as 0.00001
f(x + h) − f(x − h) — the change in output between a point just right of x and a point just left of x
2h — the distance between those two points, so the ratio is the slope of the line joining them
≈ — "approximately equal": an estimate that becomes very accurate when h is small
:::

\`f\` is any Python function of one number. Use the \`h\` you are given.

\`\`\`text
numerical_derivative(lambda x: x ** 2, 3.0)   ->  6.0   (approximately)
\`\`\`

The tests use a tight tolerance (1e-6), which the forward difference (f(x+h) − f(x)) / h is not accurate enough to meet — another reason the central difference is the standard choice.`,
        starter: `
def numerical_derivative(f, x, h=1e-5):
    """Central-difference estimate of f'(x)."""
    # TODO
    pass
`,
        solution: `
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
`,
        hints: ['Evaluate f a little to the right and a little to the left of x.', 'Divide the difference by the total distance between those points, 2h.'],
        explain: `The central difference averages the slopes to the left and right of x, so the first-order errors cancel and the error shrinks like h². This one line is the basis of **gradient checking**: compare your backprop gradients against it to catch bugs.`,
        tests: [
          { call: 'numerical_derivative(lambda x: x ** 2, 3.0)', expect: '6.0' },
          { call: 'numerical_derivative(lambda x: x ** 3, 2.0)', expect: '12.0' },
          { call: 'numerical_derivative(math.sin, 0.0)', expect: '1.0' },
          { name: 'exponential', call: 'numerical_derivative(math.exp, 1.0)', expect: 'math.e', hidden: true },
          { name: 'straight line', call: 'numerical_derivative(lambda x: 3 * x + 2, -7.0)', expect: '3.0', hidden: true },
          { name: 'uses the given h', call: 'numerical_derivative(lambda x: x ** 3, 1.0, h=0.1)', expect: '3.01', hidden: true }
        ]
      },
      {
        id: 'numerical-gradient', title: 'Numerical Gradient of a Multivariable Function', difficulty: 'medium',
        prompt: `Implement \`numerical_gradient(f, x, h=1e-5)\`:

- \`x\` is a 1-D NumPy array of floats (a point, e.g. a model's parameters).
- \`f(x)\` returns a single number (e.g. the loss).
- Return a NumPy array of the same shape where entry **i** is the partial derivative ∂f/∂xᵢ, estimated with the central difference: nudge **only** coordinate i up by h, then down by h.
- When you return, \`x\` must hold its **original values** (put back anything you changed).

\`\`\`text
numerical_gradient(lambda v: np.sum(v ** 2), np.array([1.0, -2.0, 3.0]))  ->  [ 2., -4.,  6.]
\`\`\``,
        starter: `
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """Central-difference gradient of f at x (1-D float array). Leave x unchanged."""
    grad = np.zeros_like(x, dtype=float)
    # TODO: for each index i, nudge x[i] up and down by h, evaluate f, and restore x[i]
    return grad
`,
        solution: `
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        original = x[i]
        x[i] = original + h
        f_plus = f(x)
        x[i] = original - h
        f_minus = f(x)
        x[i] = original                  # restore before moving on
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad
`,
        hints: ['Loop `for i in range(x.size):` and save `original = x[i]` first.', 'Set `x[i] = original + h`, call f, then `x[i] = original - h`, call f again.', 'Restore `x[i] = original` before the next coordinate — otherwise later partials are measured at the wrong point.'],
        explain: `Each partial derivative needs two evaluations of f, so a model with a million parameters would need two million loss evaluations per gradient — hopelessly slow for training, which is why backpropagation (one backward pass for *all* partials) matters so much. Numerical gradients remain the gold standard for *testing* backprop code on small examples.`,
        tests: [
          { name: 'sum of squares', call: 'numerical_gradient(lambda v: np.sum(v ** 2), np.array([1.0, -2.0, 3.0]))', expect: 'np.array([2.0, -4.0, 6.0])' },
          { name: 'the lesson\'s loss L(w, b)', call: 'numerical_gradient(lambda p: (p[0] * 2 + p[1] - 7) ** 2, np.array([1.0, 1.0]))', expect: 'np.array([-16.0, -8.0])' },
          { name: 'x is restored', code: 'x = np.array([0.5, 1.5])\nbefore = x.copy()\nnumerical_gradient(lambda v: np.sum(np.sin(v)), x)\nassert np.allclose(x, before), "numerical_gradient must leave x unchanged"' },
          { name: 'quadratic form vᵀAv', setup: 'rng = np.random.default_rng(2)\nM = rng.normal(size=(4, 4))\nA = M @ M.T\nx0 = rng.normal(size=4)', call: 'numerical_gradient(lambda v: v @ A @ v, x0.copy())', expect: '2 * A @ x0', tol: 1e-5, hidden: true },
          { name: 'non-polynomial function', call: 'numerical_gradient(lambda v: np.exp(v[0]) * np.cos(v[1]), np.array([0.0, 0.0]))', expect: 'np.array([1.0, 0.0])', hidden: true }
        ]
      }
    ],
    refs: [
      { title: '3Blue1Brown — Essence of Calculus', url: 'https://www.3blue1brown.com/topics/calculus' },
      { title: 'CS231n notes — Optimization and gradients', url: 'https://cs231n.github.io/optimization-1/' }
    ]
  },

  // =====================================================================================
  {
    id: 'probability', branch: 'math', title: 'Probability & Distributions', level: 1, minutes: 40, prereqs: ['what-is-ml'],
    tags: ['probability', 'bayes', 'gaussian', 'expectation', 'distribution'],
    summary: 'Reasoning under uncertainty: probabilities, random variables, the Gaussian, expected values, conditional probability and Bayes\' rule.',
    why: `Models rarely know for sure. A spam filter says *"97% spam"*; a medical model says *"12% chance of disease"*; an LLM assigns a probability to every one of ~100,000 possible next tokens and samples from that distribution. Training objectives are written in the language of probability (*maximise the likelihood of the data*), and many classic algorithms — Naive Bayes, Gaussian models, probabilistic generative models — are probability turned into code.

Probability also protects you from very human reasoning errors. Bayes' rule explains why a "99% accurate" test can still be wrong most of the time — a lesson that applies directly to evaluating classifiers.`,
    objectives: [
      'Use the basic rules of probability (complement, sum, product for independent events)',
      'Describe discrete and continuous distributions, including the Gaussian and the categorical distribution an LLM outputs',
      'Compute expected values and variance, and simulate randomness with NumPy',
      'Apply conditional probability and Bayes\' rule, and explain the base-rate fallacy'
    ],
    sections: [
      { type: 'text', title: 'The basics', md: `A **probability** is a number between 0 (impossible) and 1 (certain). Two useful ways to think about it:
- **Long-run frequency** — a fair coin lands heads with probability 0.5: flip it a million times and about half are heads.
- **Degree of belief** — "there is a 70% chance it rains tomorrow" even though tomorrow happens only once.

The handful of rules you need:

| Rule | Formula | Example |
| --- | --- | --- |
| Probabilities of all outcomes sum to 1 | Σ P(outcome) = 1 | die: 6 × 1/6 = 1 |
| Complement | P(not A) = 1 − P(A) | P(not rolling a 6) = 5/6 |
| Either of two exclusive events | P(A or B) = P(A) + P(B) | P(1 or 2) = 2/6 |
| Both of two *independent* events | P(A and B) = P(A) · P(B) | two heads in a row: 0.25 |

**Independent** means knowing one outcome tells you nothing about the other. Coin flips are independent; "it is cloudy" and "it rains" are not.` },
      { type: 'text', title: 'Random variables and distributions', md: `A **random variable** is a quantity whose value is uncertain; its **distribution** says how likely each value is.

**Discrete** distributions list a probability for each value:
- **Bernoulli** — one yes/no trial with P(yes) = p. Binary classification models output a Bernoulli probability.
- **Categorical** — one of K categories with probabilities that sum to 1. **An LLM's output at each step is a categorical distribution over its entire vocabulary.**

**Continuous** distributions are described by a **probability density function (PDF)**. The most important is the **Gaussian** (normal) bell curve with mean μ (centre) and standard deviation σ (width):

$$ p(x) = 1 / (σ√(2π)) · exp( −(x − μ)² / (2σ²) ) $$
:::where
p(x) — the probability density at the value x: the height of the bell curve (a density, not a probability)
μ — "mu": the mean, where the bell is centred
σ — "sigma": the standard deviation, how wide the bell is (σ² is the variance)
exp(z) — e raised to the power z, where e ≈ 2.718
(x − μ)² — the squared distance from the centre; values far from μ get a tiny density
1 / (σ√(2π)) — a normalizing constant that makes the total area under the curve exactly 1
π — pi ≈ 3.14159
:::

Rule of thumb: about **68%** of values fall within 1σ of the mean, **95%** within 2σ and **99.7%** within 3σ. Measurement noise, heights and many sums of small effects are roughly Gaussian (the *central limit theorem*), which is why it appears everywhere — from weight initialisation to the noise in diffusion models.

:::warn A density is not a probability
For continuous variables, P(x = exactly 1.70000…) is 0. The PDF gives probability *per unit*, and you integrate it over a range to get a probability. Densities can therefore exceed 1 (a very narrow Gaussian has a tall peak).
:::` },
      { type: 'code', title: 'Simulating randomness with NumPy', md: `Simulation is a superpower: when you are unsure about a probability, generate a million random trials and count. Always use a seeded generator so results are reproducible.`, code: `
import numpy as np
rng = np.random.default_rng(42)          # seeded: the same "random" numbers on every run

flips = rng.integers(0, 2, size=10)      # 10 coin flips (0 = tails, 1 = heads)
print("10 flips:", flips, "-> fraction of heads", flips.mean())
for n in [100, 10_000, 1_000_000]:       # law of large numbers: the average settles down
    print(f"{n:>9,} flips -> fraction of heads {rng.integers(0, 2, size=n).mean():.4f}")

rolls = rng.integers(1, 7, size=600_000) # a fair die
print("average roll:", rolls.mean().round(3), "(theory: 3.5)")
faces, counts = np.unique(rolls, return_counts=True)
print("count per face:", dict(zip(faces.tolist(), counts.tolist())))

heights = rng.normal(loc=170, scale=8, size=100_000)   # Gaussian: mean 170 cm, std 8 cm
within1 = np.mean(np.abs(heights - 170) < 8)
within2 = np.mean(np.abs(heights - 170) < 16)
print(f"within 1 std: {within1:.3f}   within 2 std: {within2:.3f}   (theory: 0.683, 0.954)")
` },
      { type: 'text', title: 'Expected value and variance', md: `The **expected value** E[X] is the probability-weighted average of the outcomes — the long-run average if you repeated the experiment forever:

$$ E[X] = Σ x · P(x)          die: (1 + 2 + 3 + 4 + 5 + 6) · 1/6 = 3.5 $$
:::where
X — a random variable: a quantity whose value is uncertain, such as the result of a die roll
x — one possible outcome (1, 2, …, 6 for a die)
P(x) — the probability of that outcome (1/6 for each face of a fair die)
Σ — add up over all possible outcomes
E[X] — the expected value: the probability-weighted average outcome
:::

You never roll a 3.5; the expectation is an average, not a typical single outcome.

The **variance** measures spread around the expectation:

$$ Var(X) = E[(X − E[X])²]          σ = √Var(X) $$
:::where
Var(X) — the variance: the expected squared distance of X from its mean
E[X] — the expected value (mean) of X
(X − E[X])² — how far one outcome lands from the mean, squared so that misses on either side count as positive
σ — the standard deviation: the square root of the variance, in the same units as X
:::

:::key The loss is an expectation
When we train a model we want to minimise the *expected* loss over all the data the model will ever see. We can't compute that, so we estimate it with the average over the training set — and that average gets more reliable as the dataset grows.
:::` },
      { type: 'text', title: 'Conditional probability and Bayes\' rule', md: `**P(A | B)** — "the probability of A *given* B" — restricts attention to the cases where B happened:

$$ P(A | B) = P(A and B) / P(B) $$
:::where
A, B — two events, things that may or may not happen
P(A | B) — the probability of A given that B happened (read the bar "|" as "given")
P(A and B) — the probability that both happen
P(B) — the probability of B; dividing by it restricts attention to the cases where B happened
:::

**Bayes' rule** turns a conditional around:

$$ P(A | B) = P(B | A) · P(A) / P(B) $$
:::where
P(A) — the prior: how likely A was before seeing the evidence B
P(B | A) — the likelihood: how likely the evidence B is when A is true
P(B) — the evidence: how likely B is overall, counting both the cases where A is true and where it is false
P(A | B) — the posterior: the updated probability of A after seeing B
:::

**The classic example.** A condition affects 1% of people. A test detects it 99% of the time (sensitivity), but also says "positive" for 5% of healthy people (false positive rate). You test positive. How likely is it that you have the condition?

Imagine 10,000 people:

| | Have it (100) | Don't have it (9,900) |
| --- | --- | --- |
| Test positive | 99 | 495 |
| Test negative | 1 | 9,405 |

Of the 99 + 495 = 594 positive results, only 99 are real: **P(condition | positive) = 99 / 594 ≈ 16.7%**, not 99%!

The rare condition's low **base rate** (prior) means false positives from the huge healthy group swamp the true positives. Ignoring it is the **base-rate fallacy** — and the same arithmetic explains why a fraud model with "99% accuracy" can still produce mostly false alarms.` },
      { type: 'code', title: 'Bayes by counting', md: `Let's verify the 16.7% by simulating a million people, then compute it directly with Bayes' rule.`, code: `
import numpy as np
rng = np.random.default_rng(0)
n = 1_000_000

has_it = rng.random(n) < 0.01                       # 1% of people have the condition
# people with it test positive 99% of the time; people without it 5% of the time
positive = np.where(has_it, rng.random(n) < 0.99, rng.random(n) < 0.05)

print("positive tests:          ", positive.sum())
print("...that are real cases:  ", (positive & has_it).sum())
print("P(condition | positive) ~", round((positive & has_it).sum() / positive.sum(), 4))

# Bayes' rule: P(c | +) = P(+ | c) P(c) / [ P(+ | c) P(c) + P(+ | not c) P(not c) ]
p = 0.99 * 0.01 / (0.99 * 0.01 + 0.05 * 0.99)
print("Bayes' rule:             ", round(p, 4))
`, predict: {"type":"mcq","q":"1% of people have a condition. The test catches 99% of real cases but also flags 5% of healthy people. If you test positive, roughly how likely is it that you really have the condition?","options":["About 99%","About 95%","About 50%","Under 20%"],"answer":3,"explain":"Only about 17%. Healthy people vastly outnumber sick ones, so the 5% false alarms (about 49,500 per million people) swamp the 9,900 true positives. This is the base-rate fallacy; Bayes’ rule gives exactly 0.1667.","expect":["P(condition | positive) ~ 0.167"]} },
      { type: 'check', question: { type: 'mcq', q: 'A test for a very rare condition (1 in 10,000) is 99.9% accurate in both directions. Someone tests positive. The probability they have the condition is…', options: ['About 99.9%', 'About 50%', 'About 9%', 'Exactly 0.01%'], answer: 2, explain: 'In 10,000,000 people: 1,000 have it (999 test positive) and 9,999,000 don\'t (about 9,999 false positives). 999 / (999 + 9,999) ≈ 9%. The tiny base rate dominates.' } },
      { type: 'text', title: 'Independence, products and log-probabilities', md: `When events are independent, the probability of all of them is the product of their probabilities. Models often *assume* independence to keep things tractable — Naive Bayes assumes the words of an email are independent given its class; an LLM writes the probability of a whole sentence as a product of next-token probabilities:

$$ P(w₁, w₂, …, wₙ) = P(w₁) · P(w₂ | w₁) · P(w₃ | w₁, w₂) · … $$
:::where
w₁, …, wₙ — a sequence of n words (or tokens)
P(w₁, …, wₙ) — the probability of the whole sequence
P(w₂ | w₁) — the probability of the second word given the first, and so on for every later word
This chain rule holds for any sequence. If the words were independent, each factor would simplify to just P(wᵢ).
:::

Multiplying hundreds of numbers below 1 quickly **underflows** to 0.0 in floating point. The fix used everywhere in ML: work with **log-probabilities**, because the log of a product is a sum of logs:

$$ log P(w₁, …, wₙ) = Σ log P(wᵢ | earlier words) $$
:::where
log — the logarithm, which turns products into sums, log(a·b) = log a + log b
P(wᵢ | earlier words) — the probability of word i given all the words before it
Σ — add up over all positions i = 1 … n
:::

Sums don't underflow, and maximising a log-probability is the same as maximising the probability. This is why loss functions are full of logarithms.` },
      { type: 'check', question: { type: 'tf', q: 'A probability density function can take values greater than 1.', answer: true, explain: 'Densities are probability per unit. A narrow Gaussian (say σ = 0.1) peaks at about 3.99. Only the area under the curve must equal 1.' } }
    ],
    when: `- Use **probabilities** (not hard yes/no labels) as model outputs whenever decisions depend on confidence or costs — e.g. only auto-block transactions above 95% fraud probability.
- Use **Bayes' rule** whenever you update a belief with new evidence, and whenever you interpret a test or classifier on a rare class.
- Use **simulation** to check probability reasoning you are unsure about.
- Use **log-probabilities** whenever you multiply many probabilities.`,
    pitfalls: `- **The base-rate fallacy** — ignoring how rare something is to begin with.
- **Assuming independence** that doesn't hold (e.g. treating consecutive days of weather as independent).
- **Confusing P(A | B) with P(B | A)** — "most positives are sick" is not "most sick people test positive".
- **Treating a density as a probability.**
- **Unseeded randomness** — results you can't reproduce. Use \`np.random.default_rng(seed)\`.`,
    takeaways: [
      'Probabilities lie between 0 and 1 and sum to 1 over all outcomes.',
      'An LLM outputs a categorical distribution over its vocabulary at every step.',
      'The Gaussian is described by its mean and standard deviation; ~68% / 95% / 99.7% lie within 1 / 2 / 3 σ.',
      'Bayes\' rule: P(A|B) = P(B|A)·P(A)/P(B). Low base rates make many positives false.',
      'Work with log-probabilities to turn products into sums and avoid underflow.'
    ],
    quiz: [
      { type: 'fill', q: 'What is the probability of rolling two sixes with two fair dice? (as a fraction like a/b or a decimal)', answer: ['1/36', '0.0278', '0.028', '/^0\\.027[78]\\d*$/'], display: '1/36', explain: 'The dice are independent: 1/6 · 1/6 = 1/36 ≈ 0.0278.' },
      { type: 'mcq', q: 'What does an LLM produce at each generation step?', options: ['A single word with certainty', 'A probability distribution over every token in its vocabulary', 'A Gaussian over word lengths', 'A yes/no answer'], answer: 1, explain: 'A categorical distribution over the vocabulary; decoding then picks or samples a token from it.' },
      { type: 'mcq', q: 'For a Gaussian with mean 100 and standard deviation 15, roughly what fraction of values lies between 70 and 130?', options: ['50%', '68%', '95%', '99.7%'], answer: 2, explain: '70 to 130 is mean ± 2σ, which contains about 95% of the values.' },
      { type: 'tf', q: 'If P(positive | sick) = 0.99, then P(sick | positive) must also be about 0.99.', answer: false, explain: 'Those are different conditionals. With a rare condition, P(sick | positive) can be far lower (the base-rate fallacy).' },
      { type: 'mcq', q: 'Why do ML libraries work with log-probabilities?', options: ['Logs are faster to compute than products', 'Products of many small probabilities underflow to zero; sums of logs do not', 'Log-probabilities are always positive', 'It makes the model more accurate by itself'], answer: 1, explain: 'log(a·b) = log a + log b. Summing logs is numerically stable even for thousands of terms.' },
      { type: 'fill', q: 'A game pays €10 with probability 0.2 and €0 otherwise. What is the expected payout in euros?', answer: ['2', '2.0', '€2', '2 euros'], explain: 'E = 10·0.2 + 0·0.8 = 2.' }
    ],
    problems: [
      {
        id: 'bayes-rule', title: 'Bayes\' Rule for a Diagnostic Test', difficulty: 'easy',
        prompt: `Implement \`posterior(prior, sensitivity, false_positive_rate)\` returning **P(condition | positive test)**:

- \`prior\` = P(condition), the base rate
- \`sensitivity\` = P(positive | condition)
- \`false_positive_rate\` = P(positive | no condition)

Use Bayes' rule, with the total probability of a positive test in the denominator:

$$ P(+) = P(+ | c)·P(c) + P(+ | not c)·P(not c) $$
:::where
+ — a positive test result
c — having the condition (not c — not having it)
P(+ | c) — the sensitivity: the chance that a person with the condition tests positive
P(+ | not c) — the false-positive rate: the chance that a healthy person tests positive
P(c) — the prior (base rate); P(not c) = 1 − P(c)
P(+) — the overall chance of a positive test, counting both groups
:::

\`\`\`text
posterior(0.01, 0.99, 0.05)  ->  0.1667 (approximately)
\`\`\``,
        starter: `
def posterior(prior, sensitivity, false_positive_rate):
    """P(condition | positive test) via Bayes' rule."""
    # TODO
    pass
`,
        solution: `
def posterior(prior, sensitivity, false_positive_rate):
    p_positive = sensitivity * prior + false_positive_rate * (1 - prior)
    return sensitivity * prior / p_positive
`,
        hints: ['P(not condition) = 1 − prior.', 'Numerator: sensitivity × prior. Denominator: the probability of a positive test from either group.'],
        explain: `The denominator adds the two ways to get a positive result: real cases (sensitivity × prior) and false alarms (false positive rate × (1 − prior)). When the prior is tiny, the false alarms dominate — the arithmetic behind the base-rate fallacy, and behind low precision on rare classes.`,
        tests: [
          { call: 'posterior(0.01, 0.99, 0.05)', expect: '0.0099 / (0.0099 + 0.0495)' },
          { call: 'posterior(0.5, 0.9, 0.1)', expect: '0.9' },
          { call: 'posterior(0.001, 1.0, 0.01)', expect: '0.001 / (0.001 + 0.00999)' },
          { name: 'moderate prior', call: 'posterior(0.3, 0.8, 0.2)', expect: '0.24 / 0.38', hidden: true },
          { name: 'no false positives', call: 'posterior(0.02, 0.7, 0.0)', expect: '1.0', hidden: true }
        ]
      },
      {
        id: 'gaussian-pdf', title: 'The Gaussian Density', difficulty: 'easy',
        prompt: `Implement \`gaussian_pdf(x, mu=0.0, sigma=1.0)\`, the density of the normal distribution:

$$ p(x) = 1 / (σ√(2π)) · exp( −(x − μ)² / (2σ²) ) $$
:::where
x — the value (or array of values) at which to evaluate the density
μ — the mean, mu
σ — the standard deviation, sigma
exp(z) — e to the power z (np.exp)
π — pi (np.pi)
:::

It must work when \`x\` is a single number **or** a NumPy array (use NumPy functions like \`np.exp\` and \`np.sqrt\`, which work on both).

\`\`\`text
gaussian_pdf(0.0)                     ->  0.3989...
gaussian_pdf(np.array([-1.0, 0, 1]))  ->  [0.2420, 0.3989, 0.2420]
\`\`\``,
        starter: `
import numpy as np

def gaussian_pdf(x, mu=0.0, sigma=1.0):
    """Normal density N(mu, sigma^2) evaluated at x (number or array)."""
    # TODO
    pass
`,
        solution: `
import numpy as np

def gaussian_pdf(x, mu=0.0, sigma=1.0):
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
`,
        hints: ['`np.pi` is π and `np.sqrt` is the square root.', 'NumPy functions apply element-wise, so the same formula works for arrays.'],
        explain: `Because NumPy operations are element-wise, one formula handles a single value or a million at once ("vectorization"). The 1/(σ√(2π)) factor makes the total area exactly 1 — the hidden test checks that numerically.`,
        tests: [
          { call: 'gaussian_pdf(0.0)', expect: '1 / np.sqrt(2 * np.pi)' },
          { call: 'gaussian_pdf(1.0, 1.0, 2.0)', expect: '1 / (2 * np.sqrt(2 * np.pi))' },
          { call: 'gaussian_pdf(np.array([-1.0, 0.0, 1.0]))', expect: 'np.array([np.exp(-0.5), 1.0, np.exp(-0.5)]) / np.sqrt(2 * np.pi)' },
          { name: 'area under the curve is 1', code: 'xs = np.linspace(-12, 12, 24001)\narea = np.sum(gaussian_pdf(xs, 0.5, 1.7)) * (xs[1] - xs[0])\nassert abs(area - 1) < 1e-3, "the density should integrate to 1, got %.4f" % area', hidden: true },
          { name: 'random points, other mu/sigma', setup: 'rng = np.random.default_rng(4)\nxs = rng.normal(3, 2, size=6)', call: 'gaussian_pdf(xs, 3.0, 0.5)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Seeing Theory — a visual introduction to probability', url: 'https://seeing-theory.brown.edu/' },
      { title: '3Blue1Brown — Bayes theorem, the geometry of changing beliefs', url: 'https://www.youtube.com/watch?v=HZGCoVF3YvM' }
    ]
  },

  // =====================================================================================
  {
    id: 'statistics', branch: 'math', title: 'Statistics: Mean, Variance & Correlation', short: 'Descriptive Statistics', level: 1, minutes: 35, prereqs: ['probability'],
    tags: ['mean', 'median', 'variance', 'standard deviation', 'correlation', 'sampling'],
    summary: 'Summarising data with a few numbers — mean, median, variance, standard deviation, covariance and correlation — and the traps hidden in each.',
    why: `Before you model data you must *understand* it: what is typical, how spread out it is, where the outliers are, and which features move together. These few statistics are also building blocks of ML itself: **standardization** (used by almost every model) subtracts the mean and divides by the standard deviation; **PCA** is built on the covariance matrix; **batch and layer normalization** in neural networks compute means and variances on the fly; and judging whether "model B is better than model A" needs a sense of sampling noise.`,
    objectives: [
      'Compute and interpret the mean, median, variance and standard deviation — and know when the median is better',
      'Explain the difference between population (÷n) and sample (÷(n−1)) variance and why NumPy and pandas disagree',
      'Compute covariance and Pearson correlation and explain why correlation is not causation',
      'Reason about sampling noise and the standard error when comparing models'
    ],
    sections: [
      { type: 'text', title: 'Centre: mean and median', md: `The **mean** is the average: x̄ = (x₁ + … + xₙ) / n. The **median** is the middle value of the sorted data (the average of the two middle values when n is even).

Ten salaries in k€: 38, 41, 42, 45, 47, 50, 52, 55, 58, **950** (the CEO).
- Mean = **137.8** — no one earns anything close to that.
- Median = **48.5** — a much better summary of a typical salary.

The mean is pulled by **outliers**; the median is **robust**. Use the median for skewed data (incomes, house prices, response times) and report both when in doubt.` },
      { type: 'text', title: 'Spread: variance and standard deviation', md: `Two datasets can share a mean but differ wildly in spread: [49, 50, 51] versus [0, 50, 100]. The **variance** is the average squared distance from the mean; the **standard deviation** (std, σ) is its square root, in the original units:

$$ Var = (1/n) Σ (xᵢ − x̄)²          σ = √Var $$
:::where
n — the number of data values
xᵢ — the i-th value
x̄ — "x bar": the mean of the values
(xᵢ − x̄)² — each value's squared distance from the mean
Var — the variance, the average squared distance (in squared units)
σ — the standard deviation, √Var, back in the data's own units
:::

**Population vs sample.** If the data is a *sample* from a bigger population, dividing by **n − 1** instead of n gives an unbiased estimate of the population variance (Bessel's correction).

:::warn NumPy and pandas use different defaults
\`np.std(x)\` divides by **n** (\`ddof=0\`); pandas' \`df.std()\` divides by **n − 1** (\`ddof=1\`). For big datasets the difference is negligible; for small ones it can surprise you.
:::

**z-scores** express a value as "standard deviations from the mean": z = (x − x̄) / σ. Subtracting the mean and dividing by the std for every feature is called **standardization**, and it puts features measured in m² and in years on a comparable scale.` },
      { type: 'code', title: 'Describing a dataset', code: `
import numpy as np

salaries = np.array([38, 41, 42, 45, 47, 50, 52, 55, 58, 950], dtype=float)  # k€, one CEO
print("mean  :", salaries.mean())           # dragged up by the outlier
print("median:", np.median(salaries))       # robust: the middle value
print("std (population, ddof=0):", salaries.std().round(2))
print("std (sample,     ddof=1):", salaries.std(ddof=1).round(2))
print("25th / 75th percentiles :", np.percentile(salaries, [25, 75]))

z = (salaries - salaries.mean()) / salaries.std()
print("z-scores:", z.round(2))              # the CEO sits 3 std above the mean
`, predict: {"type":"mcq","q":"Ten salaries in k€: nine between 38 and 58, plus one CEO at 950. How do the mean and the median compare?","options":["They are almost equal, around 50","The median is much larger than the mean","The mean is much larger than the median","Both are pulled up to about 140"],"answer":2,"explain":"The mean is 137.8, nearly three times the median of 48.5: one extreme value drags the mean but barely moves the middle value. For skewed data, report the median (or both).","expect":["mean  : 137.8","median: 48.5"]} },
      { type: 'check', question: { type: 'mcq', q: 'Website response times are mostly 100–200 ms with occasional 30-second timeouts. Which single number best describes a typical response?', options: ['The mean', 'The median', 'The maximum', 'The variance'], answer: 1, explain: 'The rare huge timeouts inflate the mean. The median reflects the typical experience (and percentiles like p95 describe the slow tail).' } },
      { type: 'text', title: 'Covariance and correlation', md: `Do two variables move together? **Covariance** averages the product of their deviations from their means:

$$ cov(x, y) = (1/n) Σ (xᵢ − x̄)(yᵢ − ȳ) $$
:::where
x, y — two variables measured on the same n examples, giving pairs (xᵢ, yᵢ)
x̄, ȳ — their means
(xᵢ − x̄)(yᵢ − ȳ) — positive when both values are on the same side of their means, negative when on opposite sides
cov(x, y) — the covariance: the average of those products
:::

Positive when both tend to be above (or below) their means together, negative when one goes up as the other goes down. Its size depends on the units, so we normalise it to get **Pearson's correlation** r, which always lies between −1 and 1:

$$ r = cov(x, y) / (σₓ σᵧ) $$
:::where
r — Pearson's correlation coefficient, always between −1 and 1
cov(x, y) — the covariance of x and y
σₓ, σᵧ — the standard deviations of x and of y; dividing by them removes the units
:::

| r | Meaning |
| --- | --- |
| +1 | perfect increasing straight line |
| 0 | no *linear* relationship |
| −1 | perfect decreasing straight line |

Two crucial caveats:
1. **Correlation is not causation.** Ice-cream sales correlate with drownings — both are driven by hot weather.
2. **r only measures straight-line relationships.** y = x² for x spread symmetrically around 0 is perfectly determined by x, yet r ≈ 0.` },
      { type: 'code', title: 'Correlation in practice', code: `
import numpy as np
rng = np.random.default_rng(1)

x = rng.uniform(-3, 3, size=500)
linear    = 2 * x + rng.normal(0, 1, size=500)
quadratic = x ** 2 + rng.normal(0, 0.3, size=500)
noise     = rng.normal(0, 1, size=500)

for name, y in [("linear", linear), ("quadratic", quadratic), ("noise", noise)]:
    r = np.corrcoef(x, y)[0, 1]
    print(f"{name:<9} r = {r:+.3f}")
# 'quadratic' depends strongly on x, yet r is near 0: correlation only sees straight-line patterns
`, predict: {"type":"mcq","q":"x is spread evenly between −3 and 3, and y = x² plus a little noise, so y depends strongly on x. What correlation r will NumPy report?","options":["Close to +1","Close to −1","Close to 0","Exactly +0.5"],"answer":2,"explain":"r is about 0 (+0.000 here). Pearson correlation only measures straight-line association: the left half of the parabola goes down, the right half goes up, and the two cancel.","expect":["quadratic r = +0.000"]} },
      { type: 'check', question: { type: 'tf', q: 'If two variables have a correlation of 0, they are unrelated.', answer: false, explain: 'r = 0 only rules out a *linear* relationship. y = x² on symmetric x has r ≈ 0 but is completely determined by x.' } },
      { type: 'text', title: 'Samples, noise and the standard error', md: `Every statistic computed from a sample is itself noisy: a different sample gives a slightly different answer. The **standard error** of a mean tells you how noisy:

$$ SE = σ / √n $$
:::where
SE — the standard error: the typical size of the error in a sample mean
σ — the standard deviation of the individual values
n — the sample size (the number of observations)
:::

To halve the noise you need **four times** the data.

This matters every time you evaluate a model. Accuracy on a test set of 100 examples is a sample statistic: at 90% accuracy its standard error is about √(0.9·0.1/100) = **3 percentage points**. "Model B scored 91%, model A 89%" on 100 examples is well within noise — don't celebrate yet. On 10,000 examples the standard error drops to 0.3 points and the same gap becomes meaningful.` }
    ],
    when: `- Compute **mean, median, std and percentiles** for every feature before modelling ("know your data").
- Prefer the **median** and percentiles for skewed data or data with outliers.
- Use **correlation** to spot redundant features and possible data leakage (a feature correlated ~1.0 with the label is suspicious).
- Think about the **standard error** before trusting a small difference between two models.`,
    pitfalls: `- Reporting only the mean of skewed data.
- Forgetting that \`np.std\` and pandas' \`.std()\` use different denominators.
- Reading causation into correlation.
- Concluding "no relationship" from r ≈ 0 — always plot the data.
- Comparing models on tiny test sets and trusting differences smaller than the noise.`,
    takeaways: [
      'The mean is sensitive to outliers; the median is robust.',
      'Variance = average squared deviation; standard deviation is its square root, in the data\'s units.',
      'Standardization (z-scores) puts features on a common scale: (x − mean) / std.',
      'Pearson r ∈ [−1, 1] measures linear association only — and is not causation.',
      'The standard error shrinks like 1/√n: small test sets give noisy scores.'
    ],
    quiz: [
      { type: 'fill', q: 'What is the median of [7, 1, 9, 3, 5, 11]?', answer: ['6', '6.0'], explain: 'Sorted: 1, 3, 5, 7, 9, 11. Two middle values 5 and 7 → median 6.' },
      { type: 'mcq', q: 'What does a z-score of −2 mean?', options: ['The value is 2 units below the mean', 'The value is 2 standard deviations below the mean', 'The value is in the bottom 2%', 'The value is negative'], answer: 1, explain: 'z = (x − mean)/std, so z = −2 means two standard deviations below the mean.' },
      { type: 'mcq', q: 'Two features have Pearson correlation 0.98. What is the most reasonable conclusion?', options: ['One causes the other', 'They are nearly linearly related — possibly redundant features', 'They are independent', 'The model will be perfectly accurate'], answer: 1, explain: 'A very high correlation means they carry almost the same linear information; it says nothing about causation.' },
      { type: 'tf', q: '`np.std(x)` and pandas `pd.Series(x).std()` give slightly different answers by default.', answer: true, explain: 'NumPy uses ddof=0 (divide by n); pandas uses ddof=1 (divide by n − 1).' },
      { type: 'mcq', q: 'To halve the standard error of an estimated mean, you need…', options: ['twice as much data', 'four times as much data', 'half as much data', 'the same data with a different seed'], answer: 1, explain: 'SE = σ/√n, so halving it requires √n to double, i.e. n × 4.' }
    ],
    problems: [
      {
        id: 'mean-var-std', title: 'Describe a Dataset from Scratch', difficulty: 'easy',
        prompt: `Implement \`describe(xs)\` for a non-empty list of numbers **using plain Python** (no NumPy, no \`statistics\` module). Return a dictionary:

| key | value |
| --- | --- |
| \`"mean"\` | the average |
| \`"var"\` | the **population** variance (divide by n) |
| \`"std"\` | the square root of that variance |
| \`"median"\` | the middle of the sorted values (average of the two middle values when n is even) |

\`\`\`text
describe([2, 4, 4, 4, 5, 5, 7, 9])  ->  {"mean": 5.0, "var": 4.0, "std": 2.0, "median": 4.5}
\`\`\``,
        starter: `
def describe(xs):
    """Mean, population variance, std and median of a list (plain Python)."""
    n = len(xs)
    # TODO: compute each statistic
    return {"mean": None, "var": None, "std": None, "median": None}
`,
        solution: `
def describe(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = var ** 0.5
    s = sorted(xs)
    mid = n // 2
    median = s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2
    return {"mean": mean, "var": var, "std": std, "median": median}
`,
        hints: ['`sum(xs) / len(xs)` is the mean.', 'For the variance, average `(x - mean) ** 2` over all x.', 'For the median, sort first; `n // 2` is the middle index. Handle even n separately.'],
        explain: `Writing these by hand makes the definitions concrete: variance is literally "the average squared distance from the mean". In real projects use \`np.mean\`, \`np.var\`, \`np.std\` and \`np.median\` (or pandas' \`describe()\`), and remember the ddof difference.`,
        tests: [
          { call: 'describe([2, 4, 4, 4, 5, 5, 7, 9])', expect: '{"mean": 5.0, "var": 4.0, "std": 2.0, "median": 4.5}' },
          { call: 'describe([1, 2, 3])', expect: '{"mean": 2.0, "var": 2 / 3, "std": (2 / 3) ** 0.5, "median": 2}' },
          { name: 'outlier', call: 'describe([10, 1, 1000])', expect: '{"mean": 337.0, "var": 219798.0, "std": 219798.0 ** 0.5, "median": 10}' },
          { name: 'single value', call: 'describe([42])', expect: '{"mean": 42.0, "var": 0.0, "std": 0.0, "median": 42}', hidden: true },
          { name: 'random data', setup: 'rng = np.random.default_rng(8)\nxs = [float(v) for v in rng.normal(10, 3, size=31).round(3)]', call: 'describe(xs)', expect: '{"mean": float(np.mean(xs)), "var": float(np.var(xs)), "std": float(np.std(xs)), "median": float(np.median(xs))}', hidden: true }
        ]
      },
      {
        id: 'pearson-correlation', title: 'Pearson Correlation', difficulty: 'medium',
        prompt: `Implement \`pearson(x, y)\` from the formula (please don't call \`np.corrcoef\` — that's what we compare against):

$$ r = Σ (xᵢ − x̄)(yᵢ − ȳ) / √( Σ (xᵢ − x̄)² · Σ (yᵢ − ȳ)² ) $$
:::where
xᵢ, yᵢ — the i-th pair of values
x̄, ȳ — the means of x and of y
Σ — add up over all pairs i = 1 … n
r — the correlation, between −1 and 1 (the 1/n factors cancel, so they are left out)
:::

- \`x\` and \`y\` are equal-length lists or arrays.
- If either has zero variance (all values equal), return \`0.0\`.
- Return a Python \`float\`.

\`\`\`text
pearson([1, 2, 3], [1, 3, 2])  ->  0.5
\`\`\``,
        starter: `
import numpy as np

def pearson(x, y):
    """Pearson correlation coefficient of x and y. Return 0.0 if either is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # TODO: centre both arrays, then apply the formula
    pass
`,
        solution: `
import numpy as np

def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x.mean()
    dy = y - y.mean()
    denom = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(dx * dy) / denom)
`,
        hints: ['Centre each array: `dx = x - x.mean()`.', 'Numerator: `np.sum(dx * dy)`. Denominator: the square root of the product of the sums of squares.', 'The 1/n factors in the covariance and the standard deviations cancel, so you can leave them out.'],
        explain: `Centring removes each variable's mean, so only co-movement remains; dividing by the square roots of the sums of squares removes the units and bounds r between −1 and 1. Geometrically, r is the **cosine similarity of the centred vectors** — the same idea as in the vectors lesson.`,
        tests: [
          { call: 'pearson([1, 2, 3, 4], [2, 4, 6, 8])', expect: '1.0' },
          { call: 'pearson([1, 2, 3, 4], [8, 6, 4, 2])', expect: '-1.0' },
          { call: 'pearson([1, 2, 3], [1, 3, 2])', expect: '0.5' },
          { name: 'symmetric quadratic', call: 'pearson([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4])', expect: '0.0', hidden: true },
          { name: 'constant input', call: 'pearson([1, 2, 3], [5, 5, 5])', expect: '0.0', hidden: true },
          { name: 'random noisy data', setup: 'rng = np.random.default_rng(12)\nx = rng.normal(size=200)\ny = 0.6 * x + rng.normal(size=200)', call: 'pearson(x, y)', expect: 'float(np.corrcoef(x, y)[0, 1])', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Spurious Correlations (Tyler Vigen)', url: 'https://www.tylervigen.com/spurious-correlations', note: 'correlation is not causation, illustrated' },
      { title: 'OpenIntro Statistics (free textbook)', url: 'https://www.openintro.org/book/os/' }
    ]
  },

  // =====================================================================================
  {
    id: 'information-theory', branch: 'math', title: 'Entropy, Cross-Entropy & KL Divergence', short: 'Information Theory', level: 2, minutes: 35, prereqs: ['probability'],
    tags: ['entropy', 'cross-entropy', 'kl divergence', 'perplexity', 'bits'],
    summary: 'Measuring surprise and uncertainty — the foundation of classification losses, decision-tree splits and LLM perplexity.',
    why: `**Cross-entropy is the loss function of modern AI.** Every image classifier and every LLM — trained to predict the next token — minimises cross-entropy. **Entropy** decides where decision trees split. **KL divergence** keeps an RLHF-tuned chatbot close to its original model, powers VAEs and knowledge distillation. And **perplexity**, the headline metric of language models, is just cross-entropy exponentiated.

Understanding these few formulas turns the loss curve of a training run from a mysterious number into something you can read: "the model is about as uncertain as if it were choosing among 20 equally likely words".`,
    objectives: [
      'Quantify the surprise of an event as −log p and explain bits versus nats',
      'Compute the entropy of a distribution and say when it is high or low',
      'Compute cross-entropy and explain why it is the classification / next-token loss',
      'Compute KL divergence and relate it to cross-entropy; convert a loss into perplexity'
    ],
    sections: [
      { type: 'text', title: 'Surprise', md: `Information theory starts with one idea: **rare events are more surprising — they carry more information.** The surprise (information content) of an event with probability p is

$$ surprise(p) = −log₂ p   bits $$
:::where
p — the probability of the event that happened
log₂ — the logarithm base 2; −log₂ p counts how many halvings take you from 1 down to p
bits — the unit of information for base-2 logs; one bit is the surprise of a fair coin flip
:::

| Event | p | Surprise |
| --- | --- | --- |
| certain event | 1 | 0 bits |
| fair coin lands heads | 1/2 | 1 bit |
| a fair die shows 6 | 1/6 | 2.58 bits |
| one specific card from a deck | 1/52 | 5.7 bits |

Logs base 2 give **bits**; natural logs (ln) give **nats** (1 nat ≈ 1.44 bits). Machine learning libraries use natural logs, so losses are in nats.` },
      { type: 'text', title: 'Entropy: average surprise', md: `The **entropy** of a distribution is its *expected* surprise:

$$ H(p) = − Σ pᵢ log pᵢ $$
:::where
p — a probability distribution over outcomes i; its probabilities pᵢ sum to 1
−log pᵢ — the surprise of outcome i
Σ — add up over all outcomes
H(p) — the entropy: the probability-weighted average surprise, highest when all outcomes are equally likely
:::

(with the convention 0 · log 0 = 0).

- A **fair coin**: H = 1 bit — maximum uncertainty for two outcomes.
- A **coin that lands heads 90% of the time**: H ≈ 0.47 bits — usually unsurprising.
- A **certain outcome**: H = 0.
- **Uniform over K outcomes**: H = log₂ K — the maximum possible for K outcomes.

Intuition: entropy is roughly the average number of yes/no questions needed to identify the outcome. It measures *how unpredictable* a distribution is.

:::real Where entropy shows up
Decision trees choose the split that reduces the entropy (impurity) of the labels the most — that reduction is called *information gain*. An LLM's next-token distribution has low entropy after "The capital of France is" and high entropy after "My favourite colour is".
:::` },
      { type: 'code', title: 'Computing entropy', md: `Writing entropy as Σ p · log(1/p) — "probability × surprise" — avoids an awkward −0.0 for certain events.`, code: `
import numpy as np

def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                          # 0 * log(0) counts as 0
    return float(np.sum(p * np.log2(1 / p)))

print("fair coin       :", entropy_bits([0.5, 0.5]))
print("90/10 coin      :", round(entropy_bits([0.9, 0.1]), 3))
print("fair die        :", round(entropy_bits([1/6] * 6), 3))
print("certain outcome :", entropy_bits([1.0]))
print("uniform over 1024 tokens:", entropy_bits(np.ones(1024) / 1024), "bits")
`, predict: {"type":"mcq","q":"A fair coin has an entropy of 1 bit. What is the entropy of a fair six-sided die, in bits?","options":["1","About 2.6","3","6"],"answer":1,"explain":"log₂ 6 ≈ 2.585 bits. Six equally likely outcomes need more than 2 bits (enough for 4 outcomes) but less than 3 bits (enough for 8).","expect":["fair die        : 2.585"]} },
      { type: 'text', title: 'Cross-entropy: the classification loss', md: `Now suppose the true distribution is **p** but our model predicts **q**. The **cross-entropy** is the average surprise we experience when outcomes come from p but we judge them with q:

$$ H(p, q) = − Σ pᵢ log qᵢ $$
:::where
p — the true distribution (how often each outcome really happens)
q — the model's predicted distribution
−log qᵢ — the surprise the model feels when outcome i happens
H(p, q) — the cross-entropy: the model's average surprise, weighted by how often each outcome truly occurs
:::

In classification, the true label is a single class k, so p is **one-hot** (1 for class k, 0 elsewhere) and the sum collapses to one term:

$$ loss = − log q_k        (the negative log-probability the model gave the correct class) $$
:::where
k — the index of the correct class
q_k — the probability the model assigned to that correct class
:::

| Model's probability for the correct class | Loss (nats) |
| --- | --- |
| 0.99 (confident and right) | 0.01 |
| 0.9 | 0.105 |
| 0.5 (unsure) | 0.693 |
| 0.1 | 2.30 |
| 0.01 (confident and wrong) | 4.61 |

The loss explodes as the probability of the truth approaches 0, so cross-entropy **punishes confident mistakes** severely. This is exactly how an LLM is trained: at every position, the loss is −log(probability the model assigned to the actual next token), averaged over trillions of tokens.` },
      { type: 'check', question: { type: 'mcq', q: 'A model assigns probability 1.0 to the correct class. What is its cross-entropy loss on that example?', options: ['1', '0', '−∞', 'log 2'], answer: 1, explain: '−log(1) = 0. A perfectly confident, correct prediction costs nothing.' } },
      { type: 'text', title: 'KL divergence: the extra cost', md: `How much worse is using q instead of the true p? The difference between cross-entropy and entropy is the **Kullback–Leibler (KL) divergence**:

$$ D_KL(p ‖ q) = Σ pᵢ log(pᵢ / qᵢ) = H(p, q) − H(p) $$
:::where
D_KL(p ‖ q) — the KL divergence of q from p: the extra surprise you pay for using q when the truth is p (never negative)
pᵢ, qᵢ — the true and the predicted probability of outcome i
H(p, q) — the cross-entropy
H(p) — the entropy of the true distribution, the unavoidable part of the surprise
:::

Properties:
- **D_KL ≥ 0**, and it is 0 only when q = p.
- It is **not symmetric**: D_KL(p‖q) ≠ D_KL(q‖p), so it is not a true distance.
- Because H(p) does not depend on the model, **minimising cross-entropy is the same as minimising KL divergence** to the true distribution.

KL appears in VAEs (keeping the latent distribution close to a Gaussian), knowledge distillation (a small student matching a big teacher's probabilities), and RLHF (a penalty that stops the tuned chatbot drifting too far from the pretrained model).` },
      { type: 'code', title: 'Cross-entropy, KL and perplexity in NumPy', code: `
import numpy as np

def cross_entropy(p, q, eps=1e-12):
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)   # avoid log(0)
    return float(-np.sum(np.asarray(p, dtype=float) * np.log(q)))

truth = [0, 1, 0]                                       # the correct class is index 1
for q in ([0.05, 0.90, 0.05], [0.3, 0.4, 0.3], [0.80, 0.01, 0.19]):
    print(q, "-> loss", round(cross_entropy(truth, q), 3))

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.5, 0.3, 0.2])
H_p = float(-np.sum(p * np.log(p)))
H_pq = cross_entropy(p, q)
print("entropy H(p)          =", round(H_p, 4))
print("cross-entropy H(p, q) =", round(H_pq, 4))
print("KL(p || q)            =", round(H_pq - H_p, 4))

loss = 2.0                                              # average loss per token, in nats
print("perplexity of a loss of 2.0 nats:", round(float(np.exp(loss)), 2))
`, predict: {"type":"mcq","q":"The correct class is index 1, but the model gives it a probability of only 0.01. What cross-entropy loss (natural log) will be printed for that prediction?","options":["About 0.01","About 0.99","About 4.6","About 100"],"answer":2,"explain":"−ln(0.01) ≈ 4.605. Cross-entropy only looks at the probability of the correct class, and it punishes confident mistakes hard: compare 0.105 for a confident correct prediction.","expect":["[0.8, 0.01, 0.19] -> loss 4.605"]} },
      { type: 'text', title: 'Perplexity: the LLM connection', md: `Language models report **perplexity** alongside the loss:

$$ perplexity = exp( average cross-entropy per token ) $$
:::where
average cross-entropy per token — the mean of −ln(probability given to the correct next token) over all tokens, in nats
exp — e to the power of that number, which undoes the natural logarithm
perplexity — the effective number of equally likely choices the model is torn between at each step
:::

Intuition: a perplexity of 20 means the model is, on average, as uncertain as if it had to choose uniformly among **20** equally likely tokens at each step. A model that knew nothing, choosing uniformly from a 50,000-token vocabulary, would have perplexity 50,000; a perfect model would have perplexity 1.

A loss of 2.0 nats ↔ perplexity e² ≈ 7.4. Lowering the loss from 2.0 to 1.8 looks small but cuts perplexity from 7.4 to 6.0 — a meaningful gain.` },
      { type: 'check', question: { type: 'fill', q: 'A model is exactly as uncertain as a uniform choice among 50,000 tokens. What is its perplexity?', answer: ['50000', '50,000', '50 000', '5e4'], display: '50,000', explain: 'Uniform over K options has cross-entropy log K, and exp(log K) = K.' } }
    ],
    when: `- Use **cross-entropy** as the loss for any classification problem, binary or multi-class, and for language modelling.
- Use **entropy** to measure uncertainty — of a model's prediction (e.g. to flag uncertain cases for human review) or of labels when choosing decision-tree splits.
- Use **KL divergence** when you want one distribution to match or stay close to another (distillation, VAEs, RLHF penalties).
- Report **perplexity** for language models; compare it only between models with the same tokenizer.`,
    pitfalls: `- **log(0)** — always clip probabilities (e.g. to at least 1e-12) or, better, compute the loss from raw logits with a numerically stable function such as PyTorch's \`cross_entropy\`.
- **Mixing bits and nats** — be explicit about the log base.
- **Treating KL as symmetric** — D_KL(p‖q) and D_KL(q‖p) behave differently.
- **Comparing perplexities across tokenizers** — a model with bigger tokens faces a harder per-token task; the numbers are not comparable.`,
    takeaways: [
      'Surprise of an event = −log p; rare events carry more information.',
      'Entropy H(p) = −Σ p log p is the average surprise — maximal for uniform distributions, 0 for certainty.',
      'Cross-entropy H(p, q) = −Σ p log q; with one-hot labels it is −log(probability of the correct class).',
      'KL(p‖q) = H(p, q) − H(p) ≥ 0, so minimising cross-entropy matches the model to the data.',
      'Perplexity = exp(cross-entropy per token): the effective number of choices an LLM is unsure between.'
    ],
    quiz: [
      { type: 'fill', q: 'What is the entropy, in bits, of a fair 8-sided die?', answer: ['3', '3.0', '3 bits'], explain: 'Uniform over 8 outcomes: log₂ 8 = 3 bits.' },
      { type: 'mcq', q: 'Which distribution over 4 outcomes has the highest entropy?', options: ['[1, 0, 0, 0]', '[0.7, 0.1, 0.1, 0.1]', '[0.25, 0.25, 0.25, 0.25]', '[0.5, 0.5, 0, 0]'], answer: 2, explain: 'The uniform distribution is the most unpredictable: 2 bits. The others are partly or fully predictable.' },
      { type: 'mcq', q: 'For a single example whose correct class received probability 0.1, the cross-entropy loss is…', options: ['0.1', '−log(0.1) ≈ 2.30 nats', 'log(0.9)', '0.9'], answer: 1, explain: 'With a one-hot label the loss is −log(q_correct) = −ln 0.1 ≈ 2.30.' },
      { type: 'tf', q: 'KL divergence is symmetric: D_KL(p‖q) = D_KL(q‖p).', answer: false, explain: 'It is asymmetric in general, which is why it is called a divergence rather than a distance.' },
      { type: 'mcq', q: 'An LLM\'s average loss falls from 3.0 to 2.0 nats per token. What happens to its perplexity?', options: ['It falls from about 20.1 to about 7.4', 'It falls from 3 to 2', 'It rises', 'It is unchanged'], answer: 0, explain: 'Perplexity = e^loss: e³ ≈ 20.1 and e² ≈ 7.4.' },
      { type: 'mcq', q: 'Why does minimising cross-entropy H(p, q) also minimise D_KL(p‖q)?', options: ['Because they are the same formula', 'Because H(p, q) = H(p) + D_KL(p‖q), and H(p) does not depend on the model', 'Because KL is always zero', 'It does not'], answer: 1, explain: 'The true entropy H(p) is a constant with respect to the model, so the two objectives differ only by that constant.' }
    ],
    problems: [
      {
        id: 'entropy-bits', title: 'Entropy in Bits', difficulty: 'easy',
        prompt: `Implement \`entropy(p)\` returning the entropy, **in bits**, of a discrete distribution \`p\` (a list of probabilities that sum to 1):

$$ H(p) = − Σ pᵢ log₂ pᵢ $$
:::where
pᵢ — the probability of outcome i (the list p sums to 1)
log₂ — the logarithm base 2 (np.log2), so the answer is in bits
Σ — add up over every outcome with pᵢ > 0
:::

Treat 0 · log 0 as 0 (skip zero probabilities).

\`\`\`text
entropy([0.5, 0.5])   ->  1.0
entropy([1.0, 0.0])   ->  0.0
\`\`\``,
        starter: `
import math

def entropy(p):
    """Entropy in bits; zero probabilities contribute nothing."""
    # TODO
    pass
`,
        solution: `
import math

def entropy(p):
    return sum(pi * math.log2(1 / pi) for pi in p if pi > 0)
`,
        hints: ['`math.log2(x)` is log base 2.', 'Skip entries that are 0 — log(0) is undefined.', '−p·log₂(p) equals p·log₂(1/p), which avoids returning −0.0.'],
        explain: `Entropy is just "probability × surprise", summed. Skipping zeros implements the limit p·log p → 0 as p → 0. Writing it as p·log₂(1/p) also avoids the cosmetic −0.0 for certain events.`,
        tests: [
          { call: 'entropy([0.5, 0.5])', expect: '1.0' },
          { call: 'entropy([0.25, 0.25, 0.25, 0.25])', expect: '2.0' },
          { call: 'entropy([1.0, 0.0])', expect: '0.0' },
          { call: 'entropy([0.9, 0.1])', expect: '-(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))' },
          { name: 'uniform over 8', call: 'entropy([1 / 8] * 8)', expect: '3.0', hidden: true },
          { name: 'random distribution with zeros', setup: 'rng = np.random.default_rng(9)\nw = rng.random(10)\nw[[2, 5]] = 0\np = (w / w.sum()).tolist()', call: 'entropy(p)', hidden: true }
        ]
      },
      {
        id: 'cross-entropy-kl', title: 'Cross-Entropy and KL Divergence', difficulty: 'medium',
        prompt: `Implement two functions using **natural logarithms** (nats), with \`EPS = 1e-12\`:

1. \`cross_entropy(p, q)\` — H(p, q) = −Σ pᵢ log qᵢ. First clip q into [EPS, 1] (\`np.clip\`) so log(0) can't happen.
2. \`kl_divergence(p, q)\` — D_KL(p‖q) = Σ pᵢ log(pᵢ / qᵢ), where terms with pᵢ = 0 contribute 0. Clip q the same way.

Inputs are lists or arrays of equal length; return Python floats.

\`\`\`text
cross_entropy([0, 1, 0], [0.1, 0.8, 0.1])  ->  0.2231   (= −ln 0.8)
kl_divergence([0.5, 0.5], [0.5, 0.5])      ->  0.0
\`\`\``,
        starter: `
import numpy as np

EPS = 1e-12

def cross_entropy(p, q):
    """H(p, q) in nats, with q clipped to [EPS, 1]."""
    # TODO
    pass


def kl_divergence(p, q):
    """D_KL(p || q) in nats; entries where p == 0 contribute nothing."""
    # TODO
    pass
`,
        solution: `
import numpy as np

EPS = 1e-12

def cross_entropy(p, q):
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), EPS, 1.0)
    return float(-np.sum(p * np.log(q)))


def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), EPS, 1.0)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
`,
        hints: ['`np.clip(q, EPS, 1.0)` keeps every probability at least EPS.', 'For KL, build a boolean mask `p > 0` and index both arrays with it.', 'Check your work: KL(p‖q) should equal cross_entropy(p, q) − entropy(p).'],
        explain: `Clipping is the crude version of what deep-learning libraries do carefully: they compute cross-entropy directly from logits using the log-sum-exp trick so that log(0) never occurs. The hidden test with q = [1, 0] and the true class at index 1 shows why clipping matters — without it you get infinity (and NaN gradients) during training.`,
        tests: [
          { call: 'cross_entropy([0, 1, 0], [0.1, 0.8, 0.1])', expect: '-np.log(0.8)' },
          { call: 'cross_entropy([1, 0], [0.5, 0.5])', expect: 'np.log(2)' },
          { call: 'kl_divergence([0.5, 0.5], [0.5, 0.5])', expect: '0.0' },
          { call: 'kl_divergence([0.5, 0.5], [0.9, 0.1])', expect: '0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)' },
          { name: 'clipping prevents infinity', call: 'cross_entropy([0, 1], [1, 0])', expect: '-np.log(1e-12)', hidden: true },
          { name: 'zeros in p', call: 'kl_divergence([0.0, 1.0], [0.3, 0.7])', expect: '-np.log(0.7)', hidden: true },
          { name: 'KL = cross-entropy - entropy', setup: 'rng = np.random.default_rng(6)\np = rng.random(7); p = p / p.sum()\nq = rng.random(7); q = q / q.sum()', code: 'h = -np.sum(p * np.log(p))\nassert abs(kl_divergence(p, q) - (cross_entropy(p, q) - h)) < 1e-9, "KL should equal cross-entropy minus entropy"\nassert kl_divergence(p, q) > 0', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Chris Olah — Visual Information Theory', url: 'https://colah.github.io/posts/2015-09-Visual-Information/', note: 'the best visual explanation of entropy and cross-entropy' },
      { title: "Wikipedia — Entropy (information theory)", url: "https://en.wikipedia.org/wiki/Entropy_(information_theory)" }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
