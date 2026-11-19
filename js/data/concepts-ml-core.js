/* ML Atlas – content: Classical ML (core) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'supervised-learning', branch: 'ml', title: 'Supervised Learning: The Big Picture', short: 'Supervised Learning', level: 1, minutes: 30, prereqs: ['what-is-ml', 'vectors-matrices'],
    tags: ['regression', 'classification', 'baseline', 'decision boundary', 'generalization'],
    summary: 'Learning a function from labelled examples: regression vs classification, model families, decision boundaries, baselines and what "generalization" really requires.',
    why: `Most deployed machine learning is supervised: predict a price, a diagnosis, a click, a translation, the next token. Before diving into specific algorithms it pays to understand the shared frame — inputs and labels, a family of candidate functions, the fit/predict workflow, and the assumption that tomorrow's data looks like today's. With that frame, every later model is just a different answer to "which family of functions, and how do we pick the best member?"`,
    objectives: [
      'Distinguish regression from classification and recognise each in real problems',
      'Describe a model family, its parameters and its inductive bias',
      'Explain decision boundaries and what makes a boundary linear or non-linear',
      'Build and use simple baselines to judge whether a model is any good',
      'State the i.i.d. assumption and when it breaks (distribution shift)'
    ],
    sections: [
      { type: 'text', title: 'The setup', md: `In supervised learning we have **n** labelled examples (x⁽¹⁾, y⁽¹⁾), …, (x⁽ⁿ⁾, y⁽ⁿ⁾). Each **x** is a feature vector; each **y** is the correct answer. We want a function **f** so that f(x) ≈ y — not just for these examples, but for **new** ones.

The type of y decides the task:

| Task | y is… | Examples | Typical output |
| --- | --- | --- | --- |
| **Regression** | a number | price, temperature, time-to-failure | a real value ŷ |
| **Binary classification** | one of two classes | spam / not spam, fraud / legit | a probability p(y = 1) |
| **Multi-class classification** | one of K classes | digit 0–9, species, next token | a probability for each class |
| **Multi-label** | any subset of labels | tags on a photo | an independent probability per label |

Classifiers usually output **probabilities**, and you choose a threshold to turn them into decisions. An LLM is a multi-class classifier whose classes are the ~100,000 tokens of its vocabulary.` },
      { type: 'text', title: 'Model families and inductive bias', md: `We never search over *all* possible functions. We pick a **model family** — a set of functions described by parameters — and learning picks the best member:

- **Linear models**: f(x) = w·x + b. Parameters w, b. Assume effects add up.
- **Decision trees**: a cascade of if/else questions. Parameters: the questions and thresholds.
- **k-nearest neighbours**: "look like your closest training examples". Almost no parameters — it stores the data.
- **Neural networks**: stacked linear maps and non-linear functions. Millions of parameters.

The assumptions baked into a family are its **inductive bias**. A linear model assumes straight-line relationships; a CNN assumes nearby pixels matter together; a transformer assumes any token may be relevant to any other. With a good bias you learn from less data; with a wrong bias no amount of data helps (a straight line can never fit a circle).

:::key The "no free lunch" idea
No model family is best for every problem. That is why practitioners try a few sensible models and let validation data decide.
:::` },
      { type: 'text', title: 'Decision boundaries', md: `A classifier splits the feature space into regions, one per predicted class. The border between regions is the **decision boundary**.

- A **linear classifier** (logistic regression, a perceptron, a linear SVM) has a straight-line boundary in 2-D, a flat plane in 3-D, a hyperplane in general: the points where w·x + b = 0.
- **Non-linear** models (trees, kNN, neural networks, kernel SVMs) can draw curved or jagged boundaries.

More flexible boundaries can fit more patterns — but also more noise. Balancing that trade-off (*Overfitting*) is the central skill of the field. Play with the boundary of k-nearest neighbours below: k = 1 traces every point; larger k smooths it.` },
      { type: 'demo', title: 'Interactive: decision regions of k-nearest neighbours', demo: 'knn' },
      { type: 'text', title: 'Always start with a baseline', md: `A **baseline** is the score of a trivial strategy. Any real model must beat it to be worth anything:

- **Regression**: predict the training mean (or median) for everything.
- **Classification**: predict the most frequent training class.
- **Simple heuristic**: "tomorrow's sales = today's sales"; "flag a transaction if amount > €5,000".

Why it matters: on a dataset with 95% legitimate transactions, a model with 95% accuracy may have learned *nothing* — the majority baseline gets 95% too.` },
      { type: 'code', title: 'Fit, predict and compare with a baseline', md: `Almost every ML library uses the same two-method interface: \`fit(X, y)\` learns from data, \`predict(X)\` answers new questions. Here is a tiny nearest-centroid classifier — it represents each class by the average of its examples — compared with the majority-class baseline.`, code: `
import numpy as np
rng = np.random.default_rng(0)

# two classes of 2-D points: 70% class 0, 30% class 1
X0 = rng.normal([0, 0], 1.0, size=(140, 2))
X1 = rng.normal([2.5, 2.0], 1.0, size=(60, 2))
X = np.vstack([X0, X1]); y = np.array([0] * 140 + [1] * 60)
perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
X_train, y_train, X_test, y_test = X[:150], y[:150], X[150:], y[150:]

class NearestCentroid:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self
    def predict(self, X):
        dists = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=-1)
        return self.classes_[dists.argmin(axis=1)]

model = NearestCentroid().fit(X_train, y_train)
majority = np.bincount(y_train).argmax()
print("centroids:", model.centroids_.round(2).tolist())
print("baseline accuracy (always predict %d): %.3f" % (majority, np.mean(y_test == majority)))
print("nearest-centroid accuracy:         %.3f" % np.mean(model.predict(X_test) == y_test))
` },
      { type: 'check', question: { type: 'mcq', q: 'A fraud detector reaches 99% accuracy on a dataset where 99% of transactions are legitimate. What can you conclude?', options: ['It is an excellent model', 'Nothing yet — always predicting "legitimate" also gets 99%', 'It must be overfitting', 'It has 1% false positives'], answer: 1, explain: 'Compare against the majority baseline. Here accuracy alone says nothing; look at precision and recall on the fraud class.' } },
      { type: 'text', title: 'The i.i.d. assumption and distribution shift', md: `Supervised learning silently assumes that training and future data are **i.i.d.** — *independent and identically distributed*: drawn from the same underlying distribution. Then good performance on held-out data predicts good performance in production.

Reality breaks this all the time — **distribution shift**:
- **Covariate shift** — inputs change (a new camera, a new customer segment, a new slang word).
- **Label/concept shift** — the relationship itself changes (what counts as spam evolves; fraudsters adapt).
- **Feedback loops** — your model's decisions change the data you collect next.

That's why deployed models are monitored and retrained (see *MLOps*), and why test sets should be as close as possible to the data the model will really face.` },
      { type: 'check', question: { type: 'mcq', q: 'Predicting the number of bikes rented tomorrow (e.g. 312) is which kind of task?', options: ['Binary classification', 'Regression', 'Clustering', 'Reinforcement learning'], answer: 1, explain: 'The target is a quantity, so it is regression (even though it happens to be an integer).' } }
    ],
    when: `- Use **supervised learning** when you have — or can collect — labelled examples of the exact decision you want to automate.
- Frame the task carefully: regression vs classification, the unit of prediction (per user? per day?), and what information exists *at prediction time*.
- If labels are scarce, consider transfer learning, self-supervised pretraining or active labelling before collecting thousands of labels by hand.`,
    pitfalls: `- Skipping the baseline — and celebrating a model that is no better than predicting the majority class.
- Treating classification outputs as certain; use probabilities and choose thresholds deliberately.
- Assuming the future looks like the past without monitoring for shift.
- Framing the wrong target (predicting what is easy to measure instead of what matters).`,
    takeaways: [
      'Supervised learning fits f(x) ≈ y from labelled examples and must generalize to new x.',
      'Regression predicts numbers; classification predicts classes (usually as probabilities).',
      'A model family plus its inductive bias determines what can be learned.',
      'Decision boundaries can be linear or flexible; flexibility trades off against overfitting.',
      'Always compare with a baseline, and remember the i.i.d. assumption can break.'
    ],
    quiz: [
      { type: 'mcq', q: 'Which is a multi-class classification task?', options: ['Predicting tomorrow\'s rainfall in mm', 'Recognising which of 10 digits is in an image', 'Grouping songs without labels', 'Estimating a house price'], answer: 1, explain: 'Ten possible categories → multi-class classification.' },
      { type: 'mcq', q: 'What is a decision boundary?', options: ['The maximum value a feature can take', 'The border in feature space where the predicted class changes', 'The size of the training set', 'The learning rate limit'], answer: 1, explain: 'It separates the regions assigned to different classes.' },
      { type: 'tf', q: 'A linear classifier\'s decision boundary in 2-D is always a straight line.', answer: true, explain: 'The boundary is the set of points with w·x + b = 0 — a line in 2-D (a hyperplane in general).' },
      { type: 'mcq', q: 'What does "inductive bias" mean?', options: ['A bug in the training data', 'The assumptions a model family makes about the kind of function it can represent', 'Prejudice in model predictions', 'The bias term b'], answer: 1, explain: 'It is the built-in preference that lets a model generalize — e.g. linearity, locality, smoothness.' },
      { type: 'mcq', q: 'A model trained on photos from one hospital performs badly at another hospital with different scanners. This is…', options: ['Underfitting', 'Distribution (covariate) shift', 'A learning-rate problem', 'Label leakage'], answer: 1, explain: 'The input distribution changed between training and deployment.' }
    ],
    problems: [
      {
        id: 'nearest-centroid', title: 'Nearest-Centroid Classifier', difficulty: 'easy',
        prompt: `Implement a \`NearestCentroid\` classifier with the standard \`fit\` / \`predict\` API:

- \`fit(X, y)\`:
  - \`self.classes_\` = the sorted unique labels (\`np.unique(y)\`),
  - \`self.centroids_\` = an array with one row per class: the mean of that class's training rows (in the same order as \`classes_\`),
  - return \`self\`.
- \`predict(X)\` → for each row of X, the label of the **closest centroid** (Euclidean distance).

\`\`\`text
m = NearestCentroid().fit(np.array([[0, 0], [0, 2], [10, 10]]), np.array([0, 0, 1]))
m.centroids_                      ->  [[0., 1.], [10., 10.]]
m.predict(np.array([[1, 1], [9, 8]]))  ->  [0, 1]
\`\`\`

Labels may be strings too.`,
        starter: `
import numpy as np

class NearestCentroid:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        # TODO: classes_ and centroids_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # TODO: distance from each row to each centroid, then the closest class
        pass
`,
        solution: `
import numpy as np

class NearestCentroid:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=-1)
        return self.classes_[np.argmin(dists, axis=1)]
`,
        hints: ['`X[y == c]` selects the rows of class c; `.mean(axis=0)` averages them.', 'Pairwise distances: `np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=-1)` has shape (n, n_classes).', '`self.classes_[np.argmin(dists, axis=1)]` maps the closest column back to its label.'],
        explain: `Nearest-centroid is a fast, surprisingly strong baseline: each class becomes one prototype point. Its decision boundaries are straight lines halfway between centroids. It struggles when a class is spread out or has several clusters — which motivates k-nearest neighbours (many prototypes) and learned models.`,
        tests: [
          { call: 'NearestCentroid().fit(np.array([[0, 0], [0, 2], [10, 10]]), np.array([0, 0, 1])).centroids_', expect: 'np.array([[0.0, 1.0], [10.0, 10.0]])' },
          { call: 'NearestCentroid().fit(np.array([[0, 0], [0, 2], [10, 10]]), np.array([0, 0, 1])).predict(np.array([[1, 1], [9, 8]]))', expect: 'np.array([0, 1])' },
          { name: 'string labels', call: 'NearestCentroid().fit(np.array([[1.0], [2.0], [9.0], [11.0]]), np.array(["low", "low", "high", "high"])).predict(np.array([[0.0], [7.0]]))', expect: 'np.array(["low", "high"])' },
          { name: 'three random blobs', setup: 'rng = np.random.default_rng(2)\ncenters = np.array([[0, 0], [6, 0], [0, 6]])\nX = np.vstack([rng.normal(c, 1.0, size=(40, 2)) for c in centers])\ny = np.repeat([0, 1, 2], 40)\nXt = np.vstack([rng.normal(c, 1.0, size=(10, 2)) for c in centers])', call: 'NearestCentroid().fit(X, y).predict(Xt)', hidden: true },
          { name: 'classes_ sorted', call: 'NearestCentroid().fit(np.array([[0.0], [1.0], [2.0]]), np.array([5, 3, 9])).classes_', expect: 'np.array([3, 5, 9])', hidden: true }
        ]
      },
      {
        id: 'majority-baseline', title: 'The Majority-Class Baseline', difficulty: 'easy',
        prompt: `Implement \`majority_baseline_accuracy(y_train, y_test)\`:

1. Find the most frequent label in \`y_train\` (if several tie, choose the **smallest** label).
2. Return the accuracy (a float between 0 and 1) of predicting that label for every example of \`y_test\`.

\`\`\`text
majority_baseline_accuracy([0, 0, 1], [0, 1, 0, 0])  ->  0.75
\`\`\`

Any model you build must beat this number to be worth deploying.`,
        starter: `
import numpy as np

def majority_baseline_accuracy(y_train, y_test):
    """Accuracy of always predicting the most common training label (ties -> smallest label)."""
    # TODO: np.unique(..., return_counts=True) is handy
    pass
`,
        solution: `
import numpy as np

def majority_baseline_accuracy(y_train, y_test):
    values, counts = np.unique(np.asarray(y_train), return_counts=True)
    majority = values[np.argmax(counts)]          # np.unique is sorted, argmax takes the first max
    return float(np.mean(np.asarray(y_test) == majority))
`,
        hints: ['`values, counts = np.unique(y_train, return_counts=True)` gives sorted labels and their counts.', '`np.argmax(counts)` returns the first maximum — which is the smallest label on ties because `values` is sorted.'],
        explain: `\`np.unique\` returns labels in sorted order, so \`argmax\` (which picks the first maximum) automatically breaks ties toward the smallest label. Comparing \`y_test == majority\` gives booleans whose mean is the accuracy.`,
        tests: [
          { call: 'majority_baseline_accuracy([0, 0, 1], [0, 1, 0, 0])', expect: '0.75' },
          { call: 'majority_baseline_accuracy(["spam", "ham", "ham"], ["ham", "spam"])', expect: '0.5' },
          { name: 'tie goes to the smallest label', call: 'majority_baseline_accuracy([2, 1, 2, 1], [1, 1, 2])', expect: '2 / 3', hidden: true },
          { name: 'imbalanced random labels', setup: 'rng = np.random.default_rng(0)\nytr = (rng.random(500) < 0.1).astype(int)\nyte = (rng.random(200) < 0.1).astype(int)', call: 'majority_baseline_accuracy(ytr, yte)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Google ML Crash Course — Framing', url: 'https://developers.google.com/machine-learning/crash-course' }
    ]
  },

  // =====================================================================================
  {
    id: 'loss-functions', branch: 'ml', title: 'Loss Functions: Measuring Mistakes', short: 'Loss Functions', level: 1, minutes: 35, prereqs: ['supervised-learning', 'statistics'],
    tags: ['mse', 'mae', 'huber', 'log loss', 'cross-entropy', 'hinge'],
    summary: 'The single number a model tries to minimise: squared, absolute and Huber errors for regression, log loss and hinge loss for classification — and why the choice matters.',
    why: `A model only learns what its loss function rewards. Choose mean squared error and a few outliers will dominate training; choose absolute error and the model predicts medians instead of means; choose cross-entropy and a classifier learns calibrated probabilities. The loss is also what gradient descent differentiates, so it must be smooth enough to give useful gradients. Understanding loss functions means understanding *what your model is actually optimising* — which is often subtly different from what you care about.`,
    objectives: [
      'Compute MSE, MAE and Huber loss and explain how each treats outliers',
      'Explain why accuracy (0-1 loss) is not used for training',
      'Compute binary cross-entropy (log loss) and connect it to information theory',
      'Choose a loss that matches the task and the cost of different mistakes'
    ],
    sections: [
      { type: 'text', title: 'What a loss function is', md: `A **loss function** ℓ(ŷ, y) measures how bad one prediction is. Averaging over the training set gives the quantity we minimise, often called the **cost** or **objective**:

$$ L(θ) = (1/n) Σᵢ ℓ( f(xᵢ; θ), yᵢ ) $$
:::where
ℓ(ŷ, y) — the per-example loss: how bad one prediction ŷ is when the truth is y
f(xᵢ; θ) — the model's prediction for example i, using the parameters θ
yᵢ — the correct answer for example i
n — the number of training examples; (1/n) Σᵢ averages over all of them
L(θ) — the training loss (also called the cost or objective): the single number that training minimizes
:::

A good training loss is:
1. **Aligned** with what you care about (small when predictions are good);
2. **Differentiable** (or nearly), so gradient descent can follow its slope;
3. **Well-behaved** — not dominated by a few weird examples unless you want it to be.

The **evaluation metric** you report (accuracy, F1, revenue…) can differ from the training loss. We train on smooth surrogates and evaluate on what matters.` },
      { type: 'text', title: 'Regression losses', md: `For a single prediction with error e = ŷ − y:

| Loss | Formula | Behaviour |
| --- | --- | --- |
| **Squared error** (MSE when averaged) | e² | big errors dominate; optimal constant prediction = the **mean** |
| **Absolute error** (MAE) | \|e\| | every unit of error costs the same; optimal constant = the **median**; robust to outliers |
| **Huber** | e²/2 if \|e\| ≤ δ, else δ(\|e\| − δ/2) | quadratic for small errors, linear for large — the best of both |

Why squared error is so popular: it is smooth, its gradient 2e is proportional to the error (big mistakes get big corrections), it gives closed-form solutions for linear models, and it corresponds to assuming Gaussian noise. Its weakness: one wildly wrong label (a mistyped price of €10,000,000) can drag the whole model toward it, because 1000² = 1,000,000.` },
      { type: 'code', title: 'MSE vs MAE vs Huber with an outlier', code: `
import numpy as np

def mse(y, yhat):   return np.mean((yhat - y) ** 2)
def mae(y, yhat):   return np.mean(np.abs(yhat - y))
def huber(y, yhat, delta=1.0):
    e = np.abs(yhat - y)
    return np.mean(np.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta)))

y = np.array([10.0, 11.0, 9.5, 10.5, 10.0])
good = np.array([10.2, 10.8, 9.7, 10.4, 9.9])
print("clean data      MSE %.3f  MAE %.3f  Huber %.3f" % (mse(y, good), mae(y, good), huber(y, good)))

y_out = y.copy(); y_out[0] = 100.0               # one corrupted label
print("with an outlier MSE %.3f  MAE %.3f  Huber %.3f" % (mse(y_out, good), mae(y_out, good), huber(y_out, good)))

# Which single constant c minimises each loss over the corrupted labels?
cs = np.linspace(0, 100, 10001)
best_mse = cs[np.argmin([mse(y_out, np.full(5, c)) for c in cs])]
best_mae = cs[np.argmin([mae(y_out, np.full(5, c)) for c in cs])]
print("best constant under MSE:", round(best_mse, 2), " (the mean is", y_out.mean(), ")")
print("best constant under MAE:", round(best_mae, 2), " (the median is", np.median(y_out), ")")
` },
      { type: 'check', question: { type: 'mcq', q: 'Your house-price data contains a few wildly mistyped prices you cannot remove. Which loss is the most robust choice?', options: ['Mean squared error', 'Mean absolute error or Huber loss', 'Squared error with a larger learning rate', 'Cubic error'], answer: 1, explain: 'Squared error amplifies large errors; absolute/Huber losses grow only linearly, so outliers have bounded influence.' } },
      { type: 'text', title: 'Classification losses', md: `The obvious classification loss is the **0-1 loss**: 1 for a wrong prediction, 0 for a right one — that is just (1 − accuracy). We don't train on it because it is **flat almost everywhere**: nudging a weight slightly rarely flips a prediction, so its gradient is zero and gives no direction.

Instead we use smooth surrogates on the model's *scores* or *probabilities*:

**Binary cross-entropy (log loss)** — for a predicted probability p that y = 1:

$$ ℓ(p, y) = −[ y·log p + (1 − y)·log(1 − p) ] $$
:::where
y — the true label: 1 (positive) or 0 (negative)
p — the model's predicted probability that y = 1
log — the natural logarithm; log p becomes very negative as p approaches 0
y·log p — only switched on when y = 1, while (1 − y)·log(1 − p) is only switched on when y = 0
:::

If y = 1 the loss is −log p; if y = 0 it is −log(1 − p). This is exactly the cross-entropy from information theory. It rewards confident correct answers, punishes confident wrong ones heavily, and produces **calibrated probabilities** (when the model says 70%, it is right about 70% of the time).

**Hinge loss** — used by SVMs, on a raw score s and a label y ∈ {−1, +1}:

$$ ℓ(s, y) = max(0, 1 − y·s) $$
:::where
s — the model's raw score: positive means "class +1", negative means "class −1"
y — the true label, written as −1 or +1
y·s — positive when the score has the right sign; 1 or more means correct with a safety margin
max(0, …) — the larger of 0 and the expression: zero loss once y·s ≥ 1, growing linearly as the point moves towards the wrong side
:::

Zero once a point is on the correct side *with a margin*; it doesn't care about probabilities.

**Categorical cross-entropy** generalises log loss to K classes: −log(probability assigned to the true class). It trains image classifiers and every LLM.` },
      { type: 'code', title: 'Log loss punishes confident mistakes', code: `
import numpy as np

def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

y = np.array([1, 1, 0, 0])
print("confident & right :", round(log_loss(y, np.array([0.95, 0.90, 0.05, 0.10])), 4))
print("unsure everywhere :", round(log_loss(y, np.array([0.55, 0.60, 0.45, 0.40])), 4))
print("one confident miss:", round(log_loss(y, np.array([0.95, 0.90, 0.05, 0.999])), 4))
print("accuracy in all three cases is 100%, 100% and 75% -- log loss sees much more")
` },
      { type: 'check', question: { type: 'mcq', q: 'Why don\'t we train classifiers by directly minimising the 0-1 loss (error rate)?', options: ['It is too slow to compute', 'Its gradient is zero almost everywhere, so gradient descent gets no direction', 'It is always larger than log loss', 'It only works for regression'], answer: 1, explain: 'Small parameter changes rarely flip any prediction, so the 0-1 loss is piecewise constant — no useful slope.' } },
      { type: 'text', title: 'Choosing a loss', md: `| Situation | Good choice |
| --- | --- |
| Regression, roughly Gaussian noise | MSE |
| Regression with outliers / heavy tails | MAE or Huber |
| You care about *relative* errors (prices from €10 to €10M) | MSE on log(y), or a percentage error |
| Binary or multi-class classification with probabilities | (binary) cross-entropy |
| Margin classifiers (SVM) | hinge loss |
| Very imbalanced classes | class-weighted cross-entropy, or focal loss |
| Different mistakes have different costs | weight the loss by cost, or choose the threshold afterwards |

Losses can also be **combined**: training objectives often add a regularization term (*Regularization*) or several task losses with weights.` }
    ],
    when: `- **MSE** is the default for regression; switch to **MAE/Huber** when outliers or heavy-tailed noise appear.
- **Cross-entropy** is the default for classification and language modelling — it gives good gradients and calibrated probabilities.
- Use **weighted losses** when classes are imbalanced or mistakes differ in cost.
- Evaluate with the **business metric**, train with a smooth loss that correlates with it.`,
    pitfalls: `- Using MSE on data with corrupted labels — the outliers take over.
- Computing log(0) — clip probabilities or compute losses from logits.
- Reporting the training loss as if it were the evaluation metric.
- Optimising a loss that ignores what you care about (e.g. accuracy on imbalanced data).
- Forgetting the loss scale changes with units — MSE in € vs k€ differs by a factor of a million.`,
    takeaways: [
      'The loss is the single number training minimises; models learn what it rewards.',
      'MSE punishes large errors heavily (→ mean); MAE treats errors linearly (→ median); Huber blends both.',
      '0-1 loss has no useful gradient, so we train classifiers on log loss (cross-entropy) or hinge loss.',
      'Log loss = −log(probability of the true class): confident mistakes are very expensive.',
      'Pick the loss to match the noise, the task and the cost of mistakes.'
    ],
    quiz: [
      { type: 'fill', q: 'Targets [2, 4] and predictions [3, 1]. What is the mean squared error?', answer: ['5', '5.0'], explain: 'Errors 1 and −3 → squares 1 and 9 → mean 5.' },
      { type: 'fill', q: 'Same data (targets [2, 4], predictions [3, 1]). What is the mean absolute error?', answer: ['2', '2.0'], explain: '|1| and |−3| → mean (1 + 3)/2 = 2.' },
      { type: 'mcq', q: 'If you must predict one constant value for a dataset, which constant minimises the MAE?', options: ['The mean', 'The median', 'The mode', 'The maximum'], answer: 1, explain: 'The median minimises the sum of absolute deviations; the mean minimises the sum of squared deviations.' },
      { type: 'mcq', q: 'For a positive example (y = 1), the model predicts p = 0.01. The binary cross-entropy is about…', options: ['0.01', '0.99', '4.6', '0'], answer: 2, explain: '−ln(0.01) ≈ 4.6 — a confident mistake is heavily penalised.' },
      { type: 'tf', q: 'Hinge loss is zero for a correctly classified point that lies beyond the margin.', answer: true, explain: 'max(0, 1 − y·s) = 0 when y·s ≥ 1.' },
      { type: 'mcq', q: 'Which loss is standard for training large language models?', options: ['Mean squared error between token ids', 'Cross-entropy on the next token', 'Hinge loss', 'Mean absolute error'], answer: 1, explain: 'LLMs minimise the cross-entropy of the true next token under the predicted distribution.' }
    ],
    problems: [
      {
        id: 'mse-mae', title: 'MSE, MAE and Huber Loss', difficulty: 'easy',
        prompt: `Implement three regression losses. Each takes equal-length arrays (or lists) of targets \`y\` and predictions \`y_hat\` and returns a Python float, averaged over the examples:

1. \`mse(y, y_hat)\` — mean of (ŷ − y)²
2. \`mae(y, y_hat)\` — mean of |ŷ − y|
3. \`huber(y, y_hat, delta=1.0)\` — mean of 0.5·e² where |e| ≤ δ, and δ·(|e| − 0.5·δ) elsewhere (e = ŷ − y)

\`\`\`text
mse([2, 4], [3, 1])    ->  5.0
mae([2, 4], [3, 1])    ->  2.0
huber([2, 4], [3, 1])  ->  (0.5 + 2.5) / 2 = 1.5
\`\`\``,
        starter: `
import numpy as np

def mse(y, y_hat):
    # TODO
    pass

def mae(y, y_hat):
    # TODO
    pass

def huber(y, y_hat, delta=1.0):
    # TODO: quadratic inside [-delta, delta], linear outside
    pass
`,
        solution: `
import numpy as np

def mse(y, y_hat):
    e = np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float)
    return float(np.mean(e ** 2))

def mae(y, y_hat):
    e = np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float)
    return float(np.mean(np.abs(e)))

def huber(y, y_hat, delta=1.0):
    e = np.abs(np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float))
    return float(np.mean(np.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta))))
`,
        hints: ['Convert to arrays first so that subtraction is element-wise.', 'For Huber, compute `e = np.abs(y_hat - y)` and use `np.where(e <= delta, ..., ...)`.'],
        explain: `All three are an element-wise penalty followed by a mean. Huber's two branches meet smoothly at |e| = δ (same value and same slope), so it stays differentiable while capping the influence of large errors to a slope of δ.`,
        tests: [
          { call: 'mse([2, 4], [3, 1])', expect: '5.0' },
          { call: 'mae([2, 4], [3, 1])', expect: '2.0' },
          { call: 'huber([2, 4], [3, 1])', expect: '1.5' },
          { name: 'huber with a different delta', call: 'huber([0, 0, 0], [0.5, -2.0, 4.0], delta=2.0)', expect: '(0.125 + 2.0 + 6.0) / 3', hidden: true },
          { name: 'random arrays', setup: 'rng = np.random.default_rng(4)\ny = rng.normal(size=50)\nyh = y + rng.normal(0, 2, size=50)', call: '(mse(y, yh), mae(y, yh), huber(y, yh, 1.5))', hidden: true }
        ]
      },
      {
        id: 'binary-cross-entropy', title: 'Binary Cross-Entropy (Log Loss)', difficulty: 'medium',
        prompt: `Implement \`binary_cross_entropy(y, p, eps=1e-12)\`:

- \`y\` holds true labels (0 or 1), \`p\` the predicted probabilities that the label is 1.
- First clip p into [eps, 1 − eps] so log(0) can never happen.
- Return the **mean** over examples of −[y·log p + (1 − y)·log(1 − p)], using natural logs, as a Python float.

\`\`\`text
binary_cross_entropy([1, 0], [0.9, 0.2])  ->  (−ln 0.9 − ln 0.8) / 2 ≈ 0.1643
\`\`\``,
        starter: `
import numpy as np

def binary_cross_entropy(y, p, eps=1e-12):
    """Mean log loss for labels y in {0, 1} and probabilities p."""
    # TODO
    pass
`,
        solution: `
import numpy as np

def binary_cross_entropy(y, p, eps=1e-12):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
`,
        hints: ['`np.clip(p, eps, 1 - eps)` keeps probabilities away from exactly 0 and 1.', 'Only one of the two terms is non-zero for each example because y is 0 or 1.'],
        explain: `For each example only one log term survives: −log p for positives and −log(1 − p) for negatives. Clipping keeps the loss finite when a model is (wrongly) 100% certain. Deep-learning libraries go further and compute this from the raw score with a numerically stable formula (\`BCEWithLogitsLoss\`).`,
        tests: [
          { call: 'binary_cross_entropy([1, 0], [0.9, 0.2])', expect: '(-np.log(0.9) - np.log(0.8)) / 2' },
          { call: 'binary_cross_entropy([1, 1, 0], [0.5, 0.5, 0.5])', expect: 'np.log(2)' },
          { name: 'perfect predictions', call: 'binary_cross_entropy([1, 0], [1.0, 0.0])', expect: '-np.log(1 - 1e-12)', hidden: true },
          { name: 'confident mistake is finite', call: 'binary_cross_entropy([1], [0.0])', expect: '-np.log(1e-12)', hidden: true },
          { name: 'random probabilities', setup: 'rng = np.random.default_rng(1)\ny = (rng.random(100) < 0.4).astype(int)\np = rng.random(100)', call: 'binary_cross_entropy(y, p)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Metrics and scoring', url: 'https://scikit-learn.org/stable/modules/model_evaluation.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'gradient-descent', branch: 'ml', title: 'Gradient Descent', level: 1, minutes: 40, prereqs: ['loss-functions', 'derivatives'],
    tags: ['optimization', 'learning rate', 'sgd', 'mini-batch', 'convergence'],
    summary: 'The algorithm that trains almost every model: repeatedly step against the gradient. Learning rates, convergence, local minima, and stochastic mini-batch training.',
    why: `Gradient descent is the engine of modern machine learning. Linear regression can be solved with a formula, but logistic regression, neural networks and every LLM cannot — they are trained by gradient descent or one of its descendants (momentum, Adam). GPT-style models take hundreds of thousands of gradient steps over trillions of tokens. If you understand what one step does, why the learning rate matters so much, and why we use small random mini-batches, you understand the heartbeat of every training run.`,
    objectives: [
      'Write the gradient-descent update rule and explain each symbol',
      'Predict the effect of too-small, good and too-large learning rates',
      'Explain local minima, saddle points and convergence criteria',
      'Compare batch, stochastic and mini-batch gradient descent and define an epoch'
    ],
    sections: [
      { type: 'text', title: 'Walking downhill', md: `We want parameters θ that make the loss L(θ) small. The gradient ∇L(θ) points uphill, so we take a small step the opposite way, and repeat:

$$ θ ← θ − η · ∇L(θ) $$
:::where
θ — "theta": all the parameters, e.g. the weights w and the bias b
← — "becomes": replace the old value on the left with the value computed on the right
∇L(θ) — the gradient of the loss at the current parameters; it points uphill
η — "eta": the **learning rate**, how big a step to take (a small positive number such as 0.01)
−η · ∇L(θ) — a small step downhill; the minus sign turns "uphill" into "downhill"
:::

Each repetition is a **step** or **iteration**. Near a minimum the gradient shrinks towards zero, so the steps naturally get smaller and the parameters settle.

:::analogy The blindfolded hiker
Blindfolded on a hillside, you feel the slope with your feet and take a step downhill — a small one if you are careful, a big one if you are in a hurry. Too big and you may leap over the valley entirely.
:::` },
      { type: 'demo', title: 'Interactive: gradient descent on a 1-D loss', md: `Try learning rates 0.05, 0.3, 0.95 and 1.05 on the bowl. Then switch to the bumpy curve and start from different places.`, demo: 'gradient-descent' },
      { type: 'text', title: 'The learning rate', md: `The learning rate is the most important **hyperparameter** (a setting you choose, not learned):

| Learning rate | What happens |
| --- | --- |
| Too small | progress is painfully slow; may stall on plateaus |
| Just right | loss falls quickly and steadily |
| A bit too large | the parameters oscillate around the minimum |
| Far too large | steps overshoot further each time: the loss **diverges** (explodes to inf/NaN) |

For the bowl L(w) = (w − 1)² the gradient is 2(w − 1), so each step multiplies the distance to the minimum by (1 − 2η). With η = 0.1 the distance shrinks by 0.8 per step; with η = 1 it flips sign forever; with η > 1 it grows. Real loss surfaces are more complicated, but the lesson holds: if your training loss suddenly becomes NaN, **lower the learning rate first**.

In practice people try values on a log scale (1e-1, 3e-2, 1e-2, 3e-3, 1e-3…) and often change the rate during training (*learning-rate schedules*, see *Optimizers*).` },
      { type: 'code', title: 'Learning rates on a simple bowl', code: `
def loss(w):            # minimum at w = 3
    return (w - 3) ** 2 + 1

def grad(w):
    return 2 * (w - 3)

for lr in [0.01, 0.1, 0.5, 0.9, 1.05]:
    w = 0.0
    for step in range(30):
        w = w - lr * grad(w)
    print(f"lr={lr:<5} after 30 steps: w = {w:12.4f}   loss = {loss(w):.4g}")
` },
      { type: 'check', question: { type: 'mcq', q: 'Your training loss decreases for a few steps, then shoots up to NaN. What is the most likely fix?', options: ['Increase the learning rate', 'Decrease the learning rate', 'Train for more epochs', 'Use a bigger model'], answer: 1, explain: 'Exploding loss is the classic symptom of a learning rate that is too large: steps overshoot and grow.' } },
      { type: 'text', title: 'Minima, saddle points and when to stop', md: `For a **convex** loss — a single bowl, as in linear and logistic regression — gradient descent with a sensible learning rate reaches the global minimum.

Neural network losses are **non-convex**: many valleys, plateaus and **saddle points** (minimum in some directions, maximum in others). In very high dimensions, true bad local minima turn out to be rare; saddle points and flat plateaus are the bigger nuisance, and the noise of mini-batch training helps escape them. Remarkably, the many different minima found in big networks tend to perform similarly well.

When to stop:
- after a fixed number of steps/epochs (most common in deep learning);
- when the improvement becomes tiny;
- when the **validation** loss stops improving (**early stopping** — also a regularizer).` },
      { type: 'text', title: 'Batch, stochastic and mini-batch', md: `The true gradient averages over **all** n training examples. That is expensive when n is millions.

| Variant | Gradient computed on | Pros | Cons |
| --- | --- | --- | --- |
| **Batch GD** | the whole training set | exact, smooth | one step per full pass — slow for big data |
| **Stochastic GD (SGD)** | one random example | many cheap steps | very noisy |
| **Mini-batch GD** | a random batch of B examples (e.g. 32–4096) | fast on GPUs, moderate noise | batch size is another hyperparameter |

Mini-batch is the standard. An **epoch** is one full pass through the training data: with n = 50,000 and B = 100 an epoch is 500 steps. Shuffle the data every epoch so batches differ.

The mini-batch gradient is a noisy but unbiased estimate of the full gradient. The noise is not just tolerable — it helps escape saddle points and tends to find solutions that generalize well.` },
      { type: 'code', title: 'Mini-batch SGD for linear regression', code: `
import numpy as np
rng = np.random.default_rng(0)
n = 1000
X = rng.uniform(-1, 1, size=(n, 2))
y = X @ np.array([3.0, -2.0]) + 0.5 + rng.normal(0, 0.1, size=n)   # true w = [3, -2], b = 0.5

w, b, lr, batch = np.zeros(2), 0.0, 0.1, 32
for epoch in range(5):
    perm = rng.permutation(n)                      # reshuffle every epoch
    for start in range(0, n, batch):
        idx = perm[start:start + batch]
        err = X[idx] @ w + b - y[idx]              # prediction error on this mini-batch
        w -= lr * 2 * X[idx].T @ err / len(idx)    # gradient of the mini-batch MSE
        b -= lr * 2 * err.mean()
    print(f"epoch {epoch + 1}: full-data MSE {np.mean((X @ w + b - y) ** 2):.4f}  w {w.round(3)}  b {b:.3f}")
`, after: `Each epoch performs ⌈1000 / 32⌉ = 32 updates instead of one, so the model converges after very few passes over the data.` },
      { type: 'check', question: { type: 'fill', q: 'A dataset has 60,000 examples and the batch size is 100. How many gradient steps are in one epoch?', answer: ['600'], explain: '60,000 / 100 = 600 mini-batches per pass over the data.' } }
    ],
    when: `- Use gradient descent whenever the loss is differentiable and there is no closed-form solution (logistic regression, neural networks, anything large).
- Use **mini-batches** for anything beyond tiny datasets; full-batch only for small, convex problems.
- For linear regression with modest feature counts, the closed-form solution is simpler; for huge datasets SGD still wins.
- In deep learning use an improved variant — momentum or **Adam** — rather than plain SGD as a first choice.`,
    pitfalls: `- A learning rate that is too large (divergence, NaN) or too small (no progress).
- Forgetting to shuffle between epochs — batches stay correlated with the file order.
- Unscaled features — a narrow, elongated loss valley forces a tiny learning rate (standardize first).
- Judging convergence on the training loss alone; watch the validation loss.
- Mixing up gradient *ascent* and *descent* signs.`,
    takeaways: [
      'Update rule: θ ← θ − η·∇L(θ); repeat until the loss stops improving.',
      'The learning rate is the key hyperparameter: too small is slow, too large diverges.',
      'Convex losses have one minimum; deep-network losses are non-convex but GD works well in practice.',
      'Mini-batch SGD estimates the gradient from a random batch — fast and helpfully noisy.',
      'An epoch is one full pass over the training data.'
    ],
    quiz: [
      { type: 'fill', q: 'w = 2, the gradient is 4 and the learning rate is 0.1. What is w after one gradient-descent step?', answer: ['1.6'], explain: 'w − η·g = 2 − 0.1·4 = 1.6.' },
      { type: 'mcq', q: 'Why is mini-batch gradient descent preferred over full-batch for large datasets?', options: ['It computes the exact gradient', 'It makes many cheap, reasonably accurate updates per pass and uses hardware efficiently', 'It never needs a learning rate', 'It guarantees the global minimum'], answer: 1, explain: 'Many noisy but useful steps per epoch beat one expensive exact step.' },
      { type: 'tf', q: 'For a convex loss like linear regression\'s MSE, gradient descent with a small enough learning rate reaches the global minimum.', answer: true, explain: 'Convex functions have no bad local minima.' },
      { type: 'mcq', q: 'What is an epoch?', options: ['One gradient step', 'One full pass over the training data', 'The time to compute one gradient', 'A learning-rate setting'], answer: 1, explain: 'An epoch contains as many steps as there are mini-batches in the dataset.' },
      { type: 'mcq', q: 'On L(w) = w², gradient descent with learning rate 1.0 starting from w = 1 will…', options: ['Converge in one step', 'Oscillate between 1 and −1 forever', 'Diverge to infinity', 'Stay at 1'], answer: 1, explain: 'Gradient 2w: w ← w − 2w = −w. It flips sign forever without approaching 0.' }
    ],
    problems: [
      {
        id: 'gd-minimize', title: 'Gradient Descent from Scratch', difficulty: 'easy',
        prompt: `Implement \`gradient_descent(grad, x0, lr=0.1, steps=100)\`:

- \`grad(x)\` returns the gradient (a NumPy array) at point \`x\`.
- Start at \`x0\` (a list or array) and apply \`x ← x − lr · grad(x)\` exactly \`steps\` times.
- Return the **whole history** as a NumPy array of shape \`(steps + 1, d)\`: the starting point followed by the point after every step.
- Don't modify \`x0\`.

\`\`\`text
gradient_descent(lambda x: 2 * (x - 3), [0.0], lr=0.1, steps=3)
->  [[0.   ],
     [0.6  ],
     [1.08 ],
     [1.464]]
\`\`\``,
        starter: `
import numpy as np

def gradient_descent(grad, x0, lr=0.1, steps=100):
    """Return the (steps + 1, d) array of visited points, starting with x0."""
    x = np.array(x0, dtype=float)     # np.array makes a copy
    history = [x.copy()]
    # TODO: take 'steps' steps and record each new point
    return np.array(history)
`,
        solution: `
import numpy as np

def gradient_descent(grad, x0, lr=0.1, steps=100):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(steps):
        x = x - lr * grad(x)
        history.append(x.copy())
    return np.array(history)
`,
        hints: ['Loop `for _ in range(steps):` and update `x = x - lr * grad(x)`.', 'Append `x.copy()` after each update so later changes cannot alter stored points.'],
        explain: `That loop *is* gradient descent. Recording the history lets you plot the path or the loss curve — exactly what training dashboards show. Using \`x = x - ...\` (which creates a new array) rather than \`x -= ...\` also guarantees the caller's \`x0\` is untouched.`,
        tests: [
          { call: 'gradient_descent(lambda x: 2 * (x - 3), [0.0], lr=0.1, steps=3)', expect: 'np.array([[0.0], [0.6], [1.08], [1.464]])' },
          { name: 'converges to the minimum', call: 'gradient_descent(lambda x: 2 * (x - 3), [0.0], lr=0.1, steps=100)[-1]', expect: 'np.array([3.0])' },
          { name: 'history shape', code: 'h = gradient_descent(lambda x: x, [1.0, 2.0, 3.0], lr=0.5, steps=10)\nassert h.shape == (11, 3), "expected shape (11, 3), got %s" % (h.shape,)' },
          { name: 'elongated 2-D bowl', call: 'gradient_descent(lambda v: np.array([2 * v[0], 20 * v[1]]), [5.0, 1.0], lr=0.04, steps=25)', hidden: true },
          { name: 'x0 not modified', code: 'x0 = np.array([4.0])\ngradient_descent(lambda x: 2 * x, x0, 0.1, 5)\nassert x0[0] == 4.0, "gradient_descent must not modify x0"', hidden: true }
        ]
      },
      {
        id: 'minibatch-indices', title: 'Shuffled Mini-Batches', difficulty: 'medium',
        prompt: `Training loops need a fresh shuffled set of mini-batches every epoch. Implement \`minibatches(n, batch_size, seed=0)\`:

1. \`rng = np.random.default_rng(seed)\`, \`perm = rng.permutation(n)\`.
2. Cut \`perm\` into consecutive chunks of \`batch_size\` (the last chunk may be smaller).
3. Return the list of chunks (each a NumPy array of indices).

\`\`\`text
len(minibatches(10, 4))  ->  3      (sizes 4, 4, 2)
\`\`\`

Every index from 0 to n − 1 must appear exactly once across the batches.`,
        starter: `
import numpy as np

def minibatches(n, batch_size, seed=0):
    """Shuffled index batches covering range(n) exactly once."""
    # TODO
    pass
`,
        solution: `
import numpy as np

def minibatches(n, batch_size, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[i:i + batch_size] for i in range(0, n, batch_size)]
`,
        hints: ['`range(0, n, batch_size)` gives the start of each batch.', 'Slicing past the end of an array is fine: `perm[8:12]` on 10 items returns the last 2.'],
        explain: `Shuffling indices (not data) is cheap and keeps features and labels aligned. Re-seeding per epoch (e.g. seed = epoch) gives a different but reproducible order each pass. PyTorch's \`DataLoader(shuffle=True, batch_size=B)\` does exactly this under the hood.`,
        tests: [
          { name: 'batch sizes', code: 'b = minibatches(10, 4)\nassert [len(x) for x in b] == [4, 4, 2], "expected sizes [4, 4, 2], got %s" % [len(x) for x in b]' },
          { name: 'every index exactly once', code: 'b = minibatches(37, 5, seed=3)\nallidx = np.concatenate(b)\nassert sorted(allidx.tolist()) == list(range(37))' },
          { name: 'seeded order', call: 'minibatches(9, 4, seed=5)' },
          { name: 'batch larger than n', call: 'minibatches(3, 10, seed=1)', hidden: true },
          { name: 'another seed', call: 'minibatches(20, 6, seed=42)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Distill — Why Momentum Really Works', url: 'https://distill.pub/2017/momentum/', note: 'beautiful interactive look at gradient descent dynamics' },
      { title: 'CS231n — Optimization', url: 'https://cs231n.github.io/optimization-1/' }
    ]
  },

  // =====================================================================================
  {
    id: 'linear-regression', branch: 'ml', title: 'Linear Regression', level: 1, minutes: 45, prereqs: ['gradient-descent', 'numpy'],
    tags: ['regression', 'least squares', 'normal equation', 'r squared', 'ols'],
    summary: 'Fit a straight line (or plane) to data — the "hello world" of ML. Two ways to solve it, how to interpret the weights, and how to measure the fit.',
    why: `Linear regression is the first model every practitioner learns, and for good reason. It shows the whole recipe in its simplest form — a model, a loss, an optimizer — and it can be solved both with a formula and with gradient descent, so you can check one against the other. It is also genuinely useful: fast, interpretable, hard to overfit when features are few, and the right baseline for any regression problem. And it lives on inside every neural network: each layer is a linear regression (Xw + b) followed by a non-linearity.`,
    objectives: [
      'Write the linear model ŷ = Xw + b and interpret its weights',
      'Fit it with the closed-form least-squares solution',
      'Fit it with gradient descent, deriving the gradient of the MSE',
      'Evaluate the fit with MSE and R², and recognise when a linear model is the wrong tool'
    ],
    sections: [
      { type: 'text', title: 'The model', md: `With d features, linear regression predicts

$$ ŷ = w₁x₁ + w₂x₂ + … + w_d x_d + b = w · x + b $$
:::where
ŷ — "y hat": the model's prediction (the hat marks an estimate)
x₁ … x_d — the d input features of one example; x is the vector of all of them
w₁ … w_d — one weight per feature: how much the prediction changes per unit of that feature; w is the weight vector
b — the bias (intercept): the prediction when every feature is 0
w · x — the dot product, w₁x₁ + … + w_d x_d
:::

For all n examples at once, stacked in a matrix X of shape (n, d):

$$ ŷ = X w + b $$
:::where
X — the data matrix, shape (n, d): one row per example, one column per feature
w — the weight vector, shape (d,)
b — the bias, a single number added to every prediction
ŷ — the vector of n predictions, one per row of X
:::

**Interpreting the weights.** If a house-price model has w_size = 1.5 (k€ per m²), then holding everything else fixed, one extra m² adds €1,500 to the predicted price. The bias b is the prediction when all features are 0. (Interpretation is only fair when features are not strongly correlated with each other, and comparing weights' sizes only makes sense if features are on the same scale.)

Training means choosing w and b to minimise the mean squared error:

$$ L(w, b) = (1/n) Σᵢ (w · xᵢ + b − yᵢ)² $$
:::where
xᵢ, yᵢ — the features and the true target of example i
w · xᵢ + b — the model's prediction for example i
(… − yᵢ)² — the squared error of that prediction
(1/n) Σᵢ — the average over all n examples, which makes L the mean squared error (MSE)
:::` },
      { type: 'demo', title: 'Interactive: fit a line by hand, then by gradient descent', demo: 'linear-fit' },
      { type: 'text', title: 'Solution 1: the closed form (least squares)', md: `MSE for a linear model is a convex bowl, so its minimum is where the gradient is exactly zero. Put a column of ones in front of X (so the bias becomes just another weight) and call the result X̃. Setting the gradient to zero gives the **normal equation**:

$$ θ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y $$
:::where
X̃ — "X tilde": the data matrix with an extra column of ones in front, shape (n, d + 1)
θ — all d + 1 parameters at once: the bias first, then the weights
X̃ᵀ — the transpose of X̃ (rows and columns swapped)
(…)⁻¹ — the matrix inverse, the matrix version of "divide by"
y — the vector of the n true targets
:::

In code you should *not* compute that inverse explicitly (it is slow and numerically fragile). Use a least-squares solver: \`np.linalg.lstsq(X_tilde, y)\`. It works even when features are redundant.

Cost: roughly O(n·d²) — fine for thousands of features, impractical for millions, which is when gradient descent takes over.` },
      { type: 'code', title: 'Closed-form fit and R²', code: `
import numpy as np
rng = np.random.default_rng(42)
n = 200
size = rng.uniform(40, 200, n)                        # m²
rooms = rng.integers(1, 6, n)
price = 1.5 * size + 20 * rooms + 30 + rng.normal(0, 15, n)   # k€; true weights 1.5, 20 and bias 30

X = np.column_stack([size, rooms])
X_tilde = np.column_stack([np.ones(n), X])            # column of ones -> bias becomes theta[0]
theta, *_ = np.linalg.lstsq(X_tilde, price, rcond=None)
b, w = theta[0], theta[1:]
print("bias:", b.round(2), " weights:", w.round(3))

pred = X_tilde @ theta
ss_res = np.sum((price - pred) ** 2)
ss_tot = np.sum((price - price.mean()) ** 2)
print("MSE:", round(np.mean((price - pred) ** 2), 2), "  R^2:", round(1 - ss_res / ss_tot, 4))
print("predicted price of 100 m², 3 rooms:", round(b + w @ [100, 3], 1), "k€")
` },
      { type: 'text', title: 'Solution 2: gradient descent', md: `Differentiating the MSE gives the gradients (e = Xw + b − y is the vector of errors):

$$ ∂L/∂w = (2/n) Xᵀ e          ∂L/∂b = (2/n) Σ eᵢ $$
:::where
e — the error vector Xw + b − y: one entry per example, positive where the model over-predicts
Xᵀ e — for each feature, the sum over all examples of (feature value × error)
n — the number of examples
∂L/∂w — the gradient for the weights, a vector of shape (d,)
∂L/∂b — the gradient for the bias, a single number
:::

Read the first one as: *each weight moves in proportion to how strongly its feature correlates with the current errors.* The training loop is then:

\`\`\`python
for epoch in range(epochs):
    e = X @ w + b - y
    w -= lr * (2 / n) * X.T @ e
    b -= lr * (2 / n) * e.sum()
\`\`\`

Standardize the features first; otherwise one large-scale feature forces a tiny learning rate for all of them.

:::key Same pattern, bigger models
Logistic regression has the *same* gradient form, X·(prediction − target). A neural network's last layer too. Learning this once pays off repeatedly.
:::` },
      { type: 'check', question: { type: 'mcq', q: 'The gradient of the MSE with respect to w is (2/n)·Xᵀe. If every error eᵢ is positive (the model over-predicts everywhere) and all features are positive, the weights will…', options: ['increase', 'decrease', 'stay the same', 'become negative immediately'], answer: 1, explain: 'Xᵀe is positive, so w − η·(positive) decreases the weights, lowering the predictions.' } },
      { type: 'text', title: 'Measuring the fit: R²', md: `MSE depends on the units of y. **R²** (the coefficient of determination) is unit-free:

$$ R² = 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)² $$
:::where
yᵢ — the true value for example i
ŷᵢ — the model's prediction for example i
ȳ — the mean of the true values; always predicting ȳ is the "know-nothing" baseline
Σ (yᵢ − ŷᵢ)² — the model's total squared error
Σ (yᵢ − ȳ)² — the baseline's total squared error (the total variation in y)
:::

- R² = 1: perfect predictions.
- R² = 0: no better than always predicting the mean (the baseline!).
- R² < 0: worse than the mean — possible on test data with a bad model.

R² on the training set never decreases when you add features, even useless ones, so judge it on validation data.` },
      { type: 'code', title: 'The same model in scikit-learn', md: `In practice you would use a library. (Running this downloads scikit-learn the first time — it is a big package.)`, code: `
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(100, 2))
y = 4 * X[:, 0] - 2 * X[:, 1] + 7 + rng.normal(0, 1, 100)

model = LinearRegression().fit(X, y)
print("weights:", model.coef_.round(3), " bias:", round(model.intercept_, 3))
print("R^2:", round(r2_score(y, model.predict(X)), 4))
` },
      { type: 'check', question: { type: 'tf', q: 'An R² of 0 on test data means the model is no better than predicting the mean of y for everything.', answer: true, explain: 'R² compares the model\'s squared error to that of the constant mean prediction.' } }
    ],
    when: `- **Use it** as the first model and baseline for any regression task; when the relationship is roughly additive; when you need interpretable effects or very fast predictions.
- **Extend it** with feature engineering (logs, interactions, polynomial terms) and regularization (ridge/lasso) before reaching for complex models.
- **Avoid it** when the relationship is strongly non-linear with interactions you can't engineer, or when outliers dominate (use a robust loss).`,
    pitfalls: `- Interpreting weights when features are highly correlated (multicollinearity) — individual weights become unstable.
- Comparing weight magnitudes across unscaled features.
- Extrapolating far outside the training range — a line keeps going forever.
- Reading high training R² as proof of a good model.
- Computing matrix inverses explicitly instead of using \`lstsq\` or \`solve\`.`,
    takeaways: [
      'Linear regression predicts ŷ = Xw + b and minimises the mean squared error.',
      'Closed form: least squares via np.linalg.lstsq on [1, X].',
      'Gradient: ∂L/∂w = (2/n)·Xᵀ(ŷ − y) — the "features × errors" pattern reused everywhere.',
      'R² = 1 − SS_res / SS_tot measures fit relative to the mean baseline.',
      'Every dense layer of a neural network is a linear model followed by a non-linearity.'
    ],
    quiz: [
      { type: 'mcq', q: 'A model predicts salary = 2000·years_experience + 30000. What does 2000 mean?', options: ['The average salary', 'Each extra year of experience adds 2000 to the predicted salary', 'The model\'s error', 'The starting salary'], answer: 1, explain: 'A weight is the change in prediction per unit change of its feature (others held fixed).' },
      { type: 'mcq', q: 'Why add a column of ones to X before solving least squares?', options: ['To avoid division by zero', 'So the bias b is learned as an ordinary weight', 'To scale the features', 'It is required by NumPy'], answer: 1, explain: 'The ones column multiplies the extra weight, which then plays the role of b.' },
      { type: 'fill', q: 'Residual sum of squares is 20 and total sum of squares is 80. What is R²?', answer: ['0.75', '.75', '3/4'], explain: '1 − 20/80 = 0.75.' },
      { type: 'tf', q: 'You should compute (XᵀX)⁻¹ explicitly with np.linalg.inv in production code.', answer: false, explain: 'Use lstsq or solve — they are faster and numerically far more stable.' },
      { type: 'mcq', q: 'When does gradient descent beat the closed-form solution for linear regression?', options: ['Never', 'With huge numbers of examples or features, or streaming data', 'Only with one feature', 'When R² is low'], answer: 1, explain: 'The closed form costs O(n·d²) and needs all data at once; SGD scales to massive or streaming data.' }
    ],
    problems: [
      {
        id: 'normal-equation', title: 'Closed-Form Linear Regression', difficulty: 'easy',
        prompt: `Implement \`fit_closed_form(X, y)\` returning \`(w, b)\` — the least-squares weights (a 1-D array of length d) and bias (a float).

Steps:
1. Build \`X_tilde\` by adding a column of ones in front of X: \`np.column_stack([np.ones(n), X])\`.
2. Solve least squares: \`theta = np.linalg.lstsq(X_tilde, y, rcond=None)[0]\`.
3. \`b = theta[0]\`, \`w = theta[1:]\`.

\`\`\`text
X = [[0], [1], [2]], y = [1, 3, 5]      (exactly y = 2x + 1)
fit_closed_form(X, y)  ->  (array([2.]), 1.0)
\`\`\``,
        starter: `
import numpy as np

def fit_closed_form(X, y):
    """Least-squares linear regression. Returns (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def fit_closed_form(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_tilde = np.column_stack([np.ones(len(X)), X])
    theta = np.linalg.lstsq(X_tilde, y, rcond=None)[0]
    return theta[1:], float(theta[0])
`,
        hints: ['`np.column_stack([np.ones(n), X])` puts the ones column first.', '`np.linalg.lstsq` returns several values; the solution is the first one.'],
        explain: `Folding the bias into θ via a ones column turns "fit a line with an intercept" into "solve one least-squares problem". \`lstsq\` uses a stable decomposition (SVD) instead of inverting XᵀX, so it also copes with redundant features.`,
        tests: [
          { call: 'fit_closed_form([[0], [1], [2]], [1, 3, 5])', expect: '(np.array([2.0]), 1.0)' },
          { call: 'fit_closed_form([[1, 0], [0, 1], [1, 1], [2, 3]], [3, 4, 6, 14])', expect: '(np.array([2.0, 3.0]), 1.0)' },
          { name: 'noisy random data', setup: 'rng = np.random.default_rng(6)\nX = rng.normal(size=(80, 3))\ny = X @ np.array([1.5, -2.0, 0.5]) + 4 + rng.normal(0, 0.3, 80)', call: 'fit_closed_form(X, y)', hidden: true },
          { name: 'returns a float bias', code: 'w, b = fit_closed_form([[0.0], [2.0]], [1.0, 2.0])\nassert isinstance(b, float), "b should be a Python float"\nassert np.allclose(w, [0.5]) and abs(b - 1.0) < 1e-9', hidden: true }
        ]
      },
      {
        id: 'linreg-gd', title: 'Linear Regression by Gradient Descent', difficulty: 'medium',
        prompt: `Train linear regression with **full-batch gradient descent**. Implement \`train_linear_gd(X, y, lr=0.1, epochs=100)\` returning \`(w, b)\`:

- Start from \`w = np.zeros(d)\` and \`b = 0.0\`.
- Each epoch, with \`e = X @ w + b - y\` and n examples:
  - \`grad_w = (2 / n) * X.T @ e\`
  - \`grad_b = (2 / n) * e.sum()\`
  - update **both** using the gradients computed from the same e: \`w -= lr * grad_w\`, \`b -= lr * grad_b\`.
- Return \`w\` (array) and \`b\` (float).

Follow this exactly — the tests compare against the same procedure, and also check that with enough epochs you reach the closed-form solution.`,
        starter: `
import numpy as np

def train_linear_gd(X, y, lr=0.1, epochs=100):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    # TODO
    return w, b
`,
        solution: `
import numpy as np

def train_linear_gd(X, y, lr=0.1, epochs=100):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        e = X @ w + b - y
        grad_w = (2 / n) * X.T @ e
        grad_b = (2 / n) * e.sum()
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, float(b)
`,
        hints: ['Compute the error vector once per epoch: `e = X @ w + b - y`.', 'Compute both gradients **before** updating either parameter.', 'Return `float(b)` so the bias is a plain number.'],
        explain: `Each epoch computes all predictions, the error vector, and the two gradients, then steps downhill. Updating w before computing grad_b from a *new* error would subtly change the algorithm — always compute every gradient from the same forward pass, as deep-learning frameworks do.`,
        tests: [
          { name: 'three epochs, exact trajectory', call: 'train_linear_gd([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0], lr=0.1, epochs=3)' },
          { name: 'converges on y = 2x + 1', call: 'train_linear_gd([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0], lr=0.1, epochs=2000)', expect: '(np.array([2.0]), 1.0)', tol: 1e-4 },
          { name: 'matches least squares on standardized data', setup: 'rng = np.random.default_rng(10)\nX = rng.normal(size=(200, 3))\ny = X @ np.array([2.0, -1.0, 0.5]) + 3 + rng.normal(0, 0.2, 200)\nXt = np.column_stack([np.ones(200), X])\ntheta = np.linalg.lstsq(Xt, y, rcond=None)[0]', call: 'train_linear_gd(X, y, lr=0.1, epochs=500)', expect: '(theta[1:], float(theta[0]))', tol: 1e-4, hidden: true },
          { name: 'exact trajectory, two features', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(30, 2))\ny = rng.normal(size=30)', call: 'train_linear_gd(X, y, lr=0.05, epochs=17)', hidden: true }
        ]
      },
      {
        id: 'r2-score', title: 'The R² Score', difficulty: 'easy',
        prompt: `Implement \`r2_score(y, y_hat)\`:

$$ R² = 1 − SS_res / SS_tot,   SS_res = Σ (y − ŷ)²,   SS_tot = Σ (y − ȳ)² $$
:::where
y — the array of true values
ŷ — the array of predictions (y_hat)
ȳ — the mean of y
SS_res — the residual sum of squares: the model's total squared error
SS_tot — the total sum of squares: the squared error of always predicting the mean
:::

Edge case: if \`SS_tot == 0\` (all targets equal), return \`1.0\` if the predictions are perfect (\`SS_res == 0\`), otherwise \`0.0\`.

\`\`\`text
r2_score([1, 2, 3], [1, 2, 3])  ->  1.0
r2_score([1, 2, 3], [2, 2, 2])  ->  0.0    (predicting the mean)
\`\`\``,
        starter: `
import numpy as np

def r2_score(y, y_hat):
    # TODO
    pass
`,
        solution: `
import numpy as np

def r2_score(y, y_hat):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)
`,
        hints: ['SS_tot is the squared error of the mean baseline.', 'Handle the constant-target case before dividing.'],
        explain: `R² rescales the model's squared error by the baseline's. It is 0 for the "predict the mean" baseline and negative for anything worse — a handy sanity check on test data.`,
        tests: [
          { call: 'r2_score([1, 2, 3], [1, 2, 3])', expect: '1.0' },
          { call: 'r2_score([1, 2, 3], [2, 2, 2])', expect: '0.0' },
          { call: 'r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])', expect: '1 - 1.5 / 29.1875' },
          { name: 'worse than the mean is negative', call: 'r2_score([1, 2, 3], [3, 2, 1])', expect: '-3.0', hidden: true },
          { name: 'constant targets', call: '(r2_score([5, 5], [5, 5]), r2_score([5, 5], [4, 6]))', expect: '(1.0, 0.0)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Google ML Crash Course — Linear regression', url: 'https://developers.google.com/machine-learning/crash-course/linear-regression' }
    ]
  },

  // =====================================================================================
  {
    id: 'logistic-regression', branch: 'ml', title: 'Logistic Regression & Classification', short: 'Logistic Regression', level: 1, minutes: 45, prereqs: ['linear-regression'],
    tags: ['classification', 'sigmoid', 'log loss', 'probability', 'decision boundary', 'softmax'],
    summary: 'Turn a linear score into a probability with the sigmoid, train it with log loss, and get the most widely used classifier — and the building block of neural networks.',
    why: `Logistic regression is the workhorse classifier of industry: credit scoring, medicine, ad click prediction and countless baselines. It outputs **probabilities**, trains quickly, and its weights are interpretable. It is also, quite literally, a single artificial neuron: a weighted sum followed by a sigmoid. Once you understand it, a neural network is "many logistic regressions stacked and trained together", and the softmax output layer of an LLM is its multi-class generalisation.`,
    objectives: [
      'Map a linear score to a probability with the sigmoid function',
      'Explain the decision boundary and the role of the threshold',
      'Derive why log loss is used and its gradient (p − y)·x',
      'Train logistic regression with gradient descent in NumPy',
      'Extend to multiple classes with softmax'
    ],
    sections: [
      { type: 'text', title: 'From scores to probabilities', md: `A linear model's score z = w·x + b can be any real number, but a probability must lie between 0 and 1. The **sigmoid** (logistic) function squashes any number into (0, 1):

$$ σ(z) = 1 / (1 + e^(−z)) $$
:::where
z — the linear score w · x + b, which can be any real number
e — Euler's number ≈ 2.718; e^(−z) is tiny for large z and huge for very negative z
σ(z) — "sigma of z": the sigmoid, which squashes z into a probability between 0 and 1
:::

| z | σ(z) |
| --- | --- |
| −∞ | → 0 |
| −2 | 0.12 |
| 0 | 0.5 |
| 2 | 0.88 |
| +∞ | → 1 |

So logistic regression predicts **p = σ(w·x + b)** = the probability that y = 1.

**Log-odds view.** Inverting the sigmoid, z = log(p / (1 − p)) — the *log-odds*. A weight wⱼ means: increasing xⱼ by 1 adds wⱼ to the log-odds, i.e. multiplies the odds by e^(wⱼ). That's how doctors read logistic models: "smoking multiplies the odds of the disease by 2.3".` },
      { type: 'text', title: 'Decision boundary and threshold', md: `To make a decision, compare p with a **threshold**, usually 0.5. Since σ(z) ≥ 0.5 exactly when z ≥ 0, the decision boundary is

$$ w · x + b = 0 $$
:::where
x — a point in feature space
w — the weight vector, which points perpendicular to the boundary
b — the bias, which shifts the boundary away from the origin
w · x + b — the score z; the boundary is where z = 0, which is where σ(z) = 0.5
Points with a positive score are classified positive, the rest negative.
:::

— a straight line (hyperplane): logistic regression is a **linear classifier**.

The threshold is a business decision, not part of training. A cancer screen might flag anything above p = 0.1 (missing a case is costly); a spam filter might require p > 0.99 before deleting mail (a false positive is costly). See *Evaluation Metrics* for choosing it.` },
      { type: 'text', title: 'Training with log loss', md: `Why not MSE? With a sigmoid, the MSE becomes non-convex and its gradients vanish when the model is confidently wrong — learning stalls exactly when it should be fastest. Instead we use **log loss** (binary cross-entropy), which comes from maximising the likelihood of the labels:

$$ L = −(1/n) Σ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ] $$
:::where
yᵢ — the true label of example i (0 or 1)
pᵢ — the predicted probability σ(w · xᵢ + b) that example i is positive
log — the natural logarithm
n — the number of examples
L — the average binary cross-entropy (log loss)
:::

Its gradient is astonishingly simple — the sigmoid's derivative cancels out:

$$ ∂L/∂w = (1/n) Xᵀ (p − y)          ∂L/∂b = (1/n) Σ (pᵢ − yᵢ) $$
:::where
p — the vector of predicted probabilities for the n examples
y — the vector of true labels (0 or 1)
p − y — the error of each prediction, between −1 and 1
Xᵀ (p − y) — for each feature, the sum over examples of (feature value × error)
:::

Exactly the shape of linear regression's gradient: **features × (prediction − target)**. The loss is convex, so gradient descent finds the global optimum.` },
      { type: 'code', title: 'Logistic regression from scratch', code: `
import numpy as np
rng = np.random.default_rng(1)
# two classes: class 0 around (-1, -1), class 1 around (1.5, 1)
X = np.vstack([rng.normal([-1, -1], 0.8, size=(100, 2)), rng.normal([1.5, 1], 0.8, size=(100, 2))])
y = np.array([0] * 100 + [1] * 100)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w, b, lr = np.zeros(2), 0.0, 0.5
for step in range(301):
    p = sigmoid(X @ w + b)                  # probability of class 1 for every example
    w -= lr * X.T @ (p - y) / len(y)        # gradient of the mean log loss
    b -= lr * np.mean(p - y)
    if step % 100 == 0:
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        print(f"step {step:3d}  log loss {loss:.4f}  accuracy {np.mean((p >= 0.5) == y):.3f}")

print("weights:", w.round(3), " bias:", round(b, 3))
print("P(class 1) at the point (0, 0):", round(float(sigmoid(np.array([0.0, 0.0]) @ w + b)), 3))
` },
      { type: 'check', question: { type: 'mcq', q: 'In logistic regression, what is the predicted probability when w·x + b = 0?', options: ['0', '0.5', '1', 'It depends on the learning rate'], answer: 1, explain: 'σ(0) = 1 / (1 + e⁰) = 0.5 — points on the decision boundary are maximally uncertain.' } },
      { type: 'text', title: 'Numerical stability', md: `For a large negative z, \`np.exp(-z)\` overflows: e^1000 is too big for a float. The usual trick evaluates the sigmoid differently on each side of zero:

- for z ≥ 0: σ(z) = 1 / (1 + e^(−z)) — safe, e^(−z) ≤ 1;
- for z < 0: σ(z) = e^z / (1 + e^z) — safe, e^z < 1.

Both formulas are mathematically identical; each is used where it cannot overflow. Libraries also compute log loss directly from z ("from logits") to avoid log(0). You will implement the stable sigmoid in the practice problems.` },
      { type: 'text', title: 'More than two classes: softmax', md: `For K classes, compute one score per class, z_k = w_k·x + b_k, and turn the scores into a probability distribution with **softmax**:

$$ p_k = e^(z_k) / Σⱼ e^(z_j) $$
:::where
K — the number of classes
z_k — the score (logit) for class k, z_k = w_k · x + b_k
e^(z_k) — exponentiating makes every score positive and exaggerates the differences between them
Σⱼ e^(z_j) — the sum over all K classes; dividing by it makes the probabilities add up to 1
p_k — the predicted probability of class k
:::

All p_k are positive and sum to 1; the largest score gets the largest probability. The loss is the categorical cross-entropy, −log p_(true class), and the gradient keeps the same friendly form, (p − one_hot(y))·x. This is **multinomial logistic regression** — and it is precisely the output layer of every neural classifier and every LLM (with K = vocabulary size).` },
      { type: 'check', question: { type: 'tf', q: 'Logistic regression can only produce straight-line (linear) decision boundaries in the original feature space.', answer: true, explain: 'The boundary is w·x + b = 0. Curved boundaries need engineered features (x², x₁x₂…) or a non-linear model.' } }
    ],
    when: `- **First choice** for binary classification on tabular data, and the baseline for any classifier.
- When you need **probabilities**, interpretability (odds ratios) or very fast inference.
- With many sparse features (text bag-of-words, one-hot categories) it is hard to beat.
- **Move on** when the boundary is clearly non-linear and feature engineering doesn't fix it — try trees/boosting or neural networks.`,
    pitfalls: `- Unscaled features slow training and make regularization uneven.
- Perfectly separable data makes weights grow without bound — use regularization (scikit-learn does by default).
- Assuming 0.5 is the right threshold for every application.
- Naively computing \`log(sigmoid(z))\` — use stable formulas.
- Reading weights as importances without scaling the features.`,
    takeaways: [
      'p = σ(w·x + b): a linear score squashed into a probability.',
      'The decision boundary w·x + b = 0 is linear; the threshold is chosen per application.',
      'Train with log loss (convex); the gradient is Xᵀ(p − y)/n.',
      'Weights act on the log-odds: e^w is an odds ratio.',
      'Softmax + cross-entropy generalise it to K classes — the output layer of neural nets and LLMs.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the sigmoid function do?', options: ['Sorts the inputs', 'Maps any real number to a value between 0 and 1', 'Computes the gradient', 'Normalizes features'], answer: 1, explain: 'σ(z) = 1 / (1 + e^−z) ∈ (0, 1), usable as a probability.' },
      { type: 'mcq', q: 'Why is log loss preferred over MSE for logistic regression?', options: ['It is faster to type', 'It gives a convex objective with strong gradients when the model is confidently wrong', 'MSE cannot be computed for probabilities', 'Log loss never needs a learning rate'], answer: 1, explain: 'MSE through a sigmoid is non-convex and its gradients vanish for confident mistakes.' },
      { type: 'fill', q: 'A logistic model has a weight of 0.693 (≈ ln 2) on "number of late payments". By what factor do the odds of default multiply for each additional late payment?', answer: ['2', '2.0', '2x', '×2'], explain: 'Odds multiply by e^w = e^0.693 ≈ 2.' },
      { type: 'mcq', q: 'For multi-class problems, which function turns K scores into probabilities that sum to 1?', options: ['ReLU', 'Softmax', 'Sigmoid applied once', 'argmax'], answer: 1, explain: 'Softmax exponentiates and normalizes. argmax picks one class but gives no probabilities.' },
      { type: 'tf', q: 'The gradient of log loss for logistic regression has the form (1/n)·Xᵀ(p − y).', answer: true, explain: 'The same "features × error" form as linear regression.' }
    ],
    problems: [
      {
        id: 'sigmoid-stable', title: 'A Numerically Stable Sigmoid', difficulty: 'easy',
        prompt: `Implement \`sigmoid(z)\` for a NumPy array \`z\` that never overflows, even for z = ±1000:

- where z ≥ 0 use 1 / (1 + e^(−z))
- where z < 0 use e^z / (1 + e^z)

Return an array of the same shape. The hidden tests run your function with NumPy set to **raise an error on overflow**, so the naive \`1 / (1 + np.exp(-z))\` will fail for large negative z.

\`\`\`text
sigmoid(np.array([-1000.0, 0.0, 1000.0]))  ->  [0. , 0.5, 1. ]
\`\`\``,
        starter: `
import numpy as np

def sigmoid(z):
    """Stable logistic function for arrays."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    # TODO: fill out[pos] and out[~pos] with the two safe formulas
    return out
`,
        solution: `
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
`,
        hints: ['Boolean masks: `out[pos] = ...` only touches the elements where pos is True.', 'For the negative part compute `ez = np.exp(z[~pos])` — every value there is negative, so ez < 1.'],
        explain: `Each branch only ever exponentiates a non-positive number, so \`np.exp\` stays between 0 and 1 and can't overflow. \`np.where\` would *also* give the right values but evaluates both formulas on every element — which overflows (and raises under \`np.errstate(over="raise")\`), so masks are the clean solution.`,
        tests: [
          { call: 'sigmoid(np.array([-1000.0, 0.0, 1000.0]))', expect: 'np.array([0.0, 0.5, 1.0])' },
          { call: 'sigmoid(np.array([-2.0, 2.0]))', expect: '1 / (1 + np.exp(np.array([2.0, -2.0])))' },
          { name: 'no overflow allowed', code: 'with np.errstate(over="raise"):\n    out = sigmoid(np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0]))\nassert np.allclose(out, [0.0, 1 / (1 + np.exp(50.0)), 0.5, 1 / (1 + np.exp(-50.0)), 1.0])', hidden: true },
          { name: '2-D input keeps its shape', setup: 'rng = np.random.default_rng(0)\nZ = rng.normal(0, 5, size=(4, 3))', call: 'sigmoid(Z)', expect: '1 / (1 + np.exp(-Z))', hidden: true }
        ]
      },
      {
        id: 'logreg-train', title: 'Train Logistic Regression', difficulty: 'medium',
        prompt: `Implement full-batch gradient-descent training for logistic regression.

\`train_logistic(X, y, lr=0.1, epochs=200)\` → \`(w, b)\`:
- start from \`w = np.zeros(d)\`, \`b = 0.0\`;
- each epoch: \`p = sigmoid(X @ w + b)\`, then
  \`w -= lr * X.T @ (p - y) / n\` and \`b -= lr * np.mean(p - y)\` (both from the same p);
- return \`w\` and \`float(b)\`.

Also implement \`predict(X, w, b, threshold=0.5)\` → an int array of 0/1 predictions (1 where the probability is ≥ threshold).

A \`sigmoid\` helper is provided in the starter.`,
        starter: `
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic(X, y, lr=0.1, epochs=200):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    # TODO
    return w, float(b)

def predict(X, w, b, threshold=0.5):
    # TODO
    pass
`,
        solution: `
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic(X, y, lr=0.1, epochs=200):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        w = w - lr * X.T @ (p - y) / n
        b = b - lr * np.mean(p - y)
    return w, float(b)

def predict(X, w, b, threshold=0.5):
    p = sigmoid(np.asarray(X, dtype=float) @ w + b)
    return (p >= threshold).astype(int)
`,
        hints: ['Compute `p` once per epoch, then both gradients from it.', '`(p >= threshold).astype(int)` converts booleans to 0/1.'],
        explain: `The loop is linear regression's loop with one change — predictions pass through the sigmoid — and the gradient keeps the same Xᵀ(prediction − target) form. Because the loss is convex, longer training only gets closer to the unique optimum (unless the data is perfectly separable, where the weights keep growing).`,
        tests: [
          { name: 'exact trajectory (5 epochs)', call: 'train_logistic([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]], [0, 0, 1, 1], lr=0.5, epochs=5)' },
          { name: 'predict uses the threshold', call: 'predict(np.array([[-2.0], [0.0], [2.0]]), np.array([1.0]), 0.0)', expect: 'np.array([0, 1, 1])' },
          { name: 'learns separable blobs', setup: 'rng = np.random.default_rng(3)\nX = np.vstack([rng.normal([-2, -2], 1.0, size=(60, 2)), rng.normal([2, 2], 1.0, size=(60, 2))])\ny = np.array([0] * 60 + [1] * 60)', code: 'w, b = train_logistic(X, y, lr=0.5, epochs=300)\nacc = np.mean(predict(X, w, b) == y)\nassert acc > 0.95, "training accuracy should exceed 0.95, got %.3f" % acc', hidden: true },
          { name: 'exact trajectory, random data', setup: 'rng = np.random.default_rng(8)\nX = rng.normal(size=(40, 3))\ny = (rng.random(40) < 0.5).astype(int)', call: 'train_logistic(X, y, lr=0.2, epochs=25)', hidden: true },
          { name: 'custom threshold', call: 'predict(np.array([[0.0], [1.0]]), np.array([1.0]), 0.0, threshold=0.7)', expect: 'np.array([0, 1])', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Logistic regression', url: 'https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression' }
    ]
  },

  // =====================================================================================
  {
    id: 'overfitting', branch: 'ml', title: 'Overfitting, Underfitting & the Bias–Variance Trade-off', short: 'Overfitting & Bias–Variance', level: 1, minutes: 40, prereqs: ['linear-regression'],
    tags: ['overfitting', 'underfitting', 'bias', 'variance', 'generalization', 'capacity'],
    summary: 'The central tension of machine learning: models too simple to capture the pattern, or so flexible they memorise the noise — and how to find the sweet spot.',
    why: `Every model you will ever train faces this tension. Too little flexibility and the model misses real structure (**underfitting**); too much and it memorises noise in the training data and fails on new data (**overfitting**). Recognising which situation you are in — from training and validation curves — tells you what to do next: add capacity, add data, regularize, or stop training earlier. Getting this right matters more than choosing between fashionable algorithms.`,
    objectives: [
      'Define training error, validation error and the generalization gap',
      'Diagnose underfitting and overfitting from train/validation errors and learning curves',
      'Explain the bias–variance trade-off with an intuitive picture',
      'List the main remedies for each problem'
    ],
    sections: [
      { type: 'text', title: 'Two errors, one gap', md: `Every model has two errors that matter:

- **Training error** — how wrong it is on the data it learned from.
- **Validation (or test) error** — how wrong it is on data it has never seen.

The difference is the **generalization gap**. Training error is almost always lower, because the model was tuned to that data. What matters is the validation error.

| Training error | Validation error | Diagnosis |
| --- | --- | --- |
| high | high (similar) | **underfitting** — the model is too simple |
| low | much higher | **overfitting** — the model memorised noise |
| low | low (similar) | a good fit |` },
      { type: 'demo', title: 'Interactive: polynomial degree vs. train and test error', md: `Fit polynomials of increasing degree to 12 noisy points from a sine curve. Watch the training error keep falling while the test error forms a U.`, demo: 'overfit-poly' },
      { type: 'text', title: 'Bias and variance', md: `Imagine training the same model type on many different training sets drawn from the same source, and look at the predictions for one fixed input:

- **Bias** — how far the *average* prediction is from the truth. High when the model family can't represent the true pattern (a line for a sine curve).
- **Variance** — how much the predictions *jump around* between training sets. High when the model is so flexible that it follows each sample's noise.

For squared error there is an exact decomposition:

$$ expected error = bias² + variance + irreducible noise $$
:::where
expected error — the squared error on new data, averaged over many possible training sets
bias — how far the average prediction is from the truth; error from assumptions that are too simple
variance — how much the predictions change when the model is trained on a different sample; error from sensitivity to noise
irreducible noise — randomness in the targets themselves that no model can predict
:::

Simple models: high bias, low variance. Flexible models: low bias, high variance. The best model balances the two for the amount of data you have — more data lets you afford more flexibility, because it reduces variance.

:::analogy Darts
High bias: every dart lands in a tight cluster — but off-centre. High variance: darts are centred on average but scattered all over the board. You want a tight cluster on the bullseye.
:::` },
      { type: 'code', title: 'Train vs validation error as complexity grows', code: `
import numpy as np, warnings
warnings.simplefilter("ignore")                    # polyfit warns about ill-conditioning at high degrees
rng = np.random.default_rng(0)

x_train = rng.uniform(0, 6, 12);  y_train = np.sin(x_train) + rng.normal(0, 0.3, 12)
x_val = rng.uniform(0, 6, 300);   y_val = np.sin(x_val) + rng.normal(0, 0.3, 300)

for degree in [0, 1, 3, 5, 8, 11]:
    coefs = np.polyfit(x_train, y_train, degree)
    train_mse = np.mean((np.polyval(coefs, x_train) - y_train) ** 2)
    val_mse = np.mean((np.polyval(coefs, x_val) - y_val) ** 2)
    print(f"degree {degree:2d}:  train MSE {train_mse:8.4f}   validation MSE {val_mse:12.4f}")
` },
      { type: 'check', question: { type: 'mcq', q: 'Training accuracy 99%, validation accuracy 71%. What is happening, and what helps?', options: ['Underfitting — use a bigger model', 'Overfitting — get more data, regularize, simplify or stop earlier', 'The learning rate is too small', 'Nothing is wrong'], answer: 1, explain: 'A large gap between training and validation performance is the signature of overfitting.' } },
      { type: 'text', title: 'Learning curves', md: `Plot training and validation error against the **number of training examples** (or against training epochs):

- **Underfitting**: both curves plateau close together at a high error. More data won't help; more capacity or better features will.
- **Overfitting**: a big gap that narrows slowly as data grows. More data helps; so do regularization and simpler models.
- **Over training epochs**: validation loss falls, then starts rising while training loss keeps falling — the moment to stop (**early stopping**).

:::real Deep learning's twist
Very large neural networks can fit random labels perfectly, yet generalize well on real data — and sometimes test error falls again as models grow past the point of interpolating the training data ("double descent"). Regularization, data augmentation, the implicit bias of SGD and huge datasets all play a role. The classical U-curve is still the right instinct for most problems you will meet.
:::` },
      { type: 'text', title: 'Remedies', md: `| If you are… | Try |
| --- | --- |
| **Underfitting** | a more flexible model; better or more features; less regularization; train longer |
| **Overfitting** | more training data; data augmentation; regularization (L1/L2, dropout); a simpler model; fewer features; early stopping; ensembles |

And always: diagnose with a **validation set** (or cross-validation), never with the training set.` },
      { type: 'check', question: { type: 'tf', q: 'Adding more training data is a good remedy for high bias (underfitting).', answer: false, explain: 'More data reduces variance. If the model family is too simple, more data won\'t fix it — increase capacity or improve features.' } }
    ],
    when: `- Check for under/overfitting **every time** you train: compare training and validation metrics.
- Use **learning curves** to decide whether to collect more data (helps overfitting) or change the model (needed for underfitting).
- Use **early stopping** by default for iterative training.`,
    pitfalls: `- Judging a model by training error alone.
- Tuning hyperparameters on the test set (it becomes overfit too).
- Assuming a bigger model is always better — or always worse.
- Confusing a bad learning rate (training loss not decreasing) with underfitting.`,
    takeaways: [
      'Overfitting = low training error, high validation error; underfitting = both high.',
      'Error = bias² + variance + noise: flexibility trades bias for variance.',
      'More data reduces variance; more capacity or features reduce bias.',
      'Learning curves and train/validation gaps tell you what to fix.',
      'Validation data — not training data — is the judge.'
    ],
    quiz: [
      { type: 'mcq', q: 'A degree-15 polynomial fits 16 training points perfectly. On new data it is wildly off. This is…', options: ['high bias', 'high variance (overfitting)', 'irreducible noise', 'underfitting'], answer: 1, explain: 'It memorised the training noise; tiny changes in the data would change it drastically.' },
      { type: 'match', q: 'Match each situation with the most useful remedy.', pairs: [['Training and validation errors both high', 'Use a more flexible model or better features'], ['Large gap between training and validation errors', 'Regularize or collect more data'], ['Validation loss rising while training loss keeps falling', 'Stop training earlier'], ['Model performs well everywhere', 'Keep it, evaluate once on the test set']], explain: 'Underfitting needs capacity; overfitting needs data or regularization; a rising validation curve calls for early stopping.' },
      { type: 'tf', q: 'Bias measures how much a model\'s predictions vary between different training sets.', answer: false, explain: 'That is variance. Bias is the systematic error of the average prediction.' },
      { type: 'mcq', q: 'Which change usually reduces variance?', options: ['Adding many polynomial features', 'Training on more data', 'Removing regularization', 'Using k = 1 in kNN'], answer: 1, explain: 'More data averages out the noise that a flexible model would otherwise fit.' },
      { type: 'mcq', q: 'Which error component can no model remove?', options: ['Bias', 'Variance', 'Irreducible noise in the labels', 'Optimization error'], answer: 2, explain: 'Noise in the data itself sets a floor on the achievable error.' }
    ],
    problems: [
      {
        id: 'polynomial-features', title: 'Polynomial Features', difficulty: 'easy',
        prompt: `Polynomial features let a linear model fit curves. Implement \`poly_features(x, degree)\` for a 1-D array \`x\`:

Return a 2-D array of shape \`(len(x), degree)\` whose columns are x¹, x², …, x^degree (no column of ones).

\`\`\`text
poly_features(np.array([1.0, 2.0, 3.0]), 3)
-> [[ 1.,  1.,  1.],
    [ 2.,  4.,  8.],
    [ 3.,  9., 27.]]
\`\`\``,
        starter: `
import numpy as np

def poly_features(x, degree):
    """Columns x, x**2, ..., x**degree."""
    x = np.asarray(x, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def poly_features(x, degree):
    x = np.asarray(x, dtype=float)
    return np.column_stack([x ** k for k in range(1, degree + 1)])
`,
        hints: ['`range(1, degree + 1)` gives the powers 1..degree.', '`np.column_stack([...])` stacks 1-D arrays as columns.'],
        explain: `With these columns, "linear regression" fits ŷ = w₁x + w₂x² + … + b — a curve in x, but still linear in the weights, so the same solvers work. The catch: high degrees make it easy to overfit, and the columns have wildly different scales (x¹⁰ vs x), so standardize them before gradient descent.`,
        tests: [
          { call: 'poly_features(np.array([1.0, 2.0, 3.0]), 3)', expect: 'np.array([[1.0, 1.0, 1.0], [2.0, 4.0, 8.0], [3.0, 9.0, 27.0]])' },
          { call: 'poly_features(np.array([-1.0, 0.5]), 2)', expect: 'np.array([[-1.0, 1.0], [0.5, 0.25]])' },
          { name: 'degree 1 is a single column', call: 'poly_features(np.array([4.0, 5.0]), 1)', expect: 'np.array([[4.0], [5.0]])', hidden: true },
          { name: 'random inputs', setup: 'rng = np.random.default_rng(2)\nx = rng.normal(size=7)', call: 'poly_features(x, 5)', hidden: true }
        ]
      },
      {
        id: 'choose-degree', title: 'Pick the Best Polynomial Degree', difficulty: 'medium',
        prompt: `Use a validation set to choose model complexity. Implement \`choose_degree(x_train, y_train, x_val, y_val, max_degree)\`:

For each degree d = 1, 2, …, max_degree:
1. Build the design matrix with a ones column and powers up to d: columns 1, x, x², …, x^d.
2. Fit by least squares on the **training** data (\`np.linalg.lstsq\`).
3. Compute the **validation** MSE.

Return the degree (an int) with the lowest validation MSE; on ties return the smaller degree.

The hidden tests use data from a cubic curve plus noise — the best degree should not be the largest one!`,
        starter: `
import numpy as np

def design(x, d):
    """Columns 1, x, x**2, ..., x**d."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x ** k for k in range(d + 1)])

def choose_degree(x_train, y_train, x_val, y_val, max_degree):
    # TODO: fit each degree on train, score on validation, keep the best
    pass
`,
        solution: `
import numpy as np

def design(x, d):
    x = np.asarray(x, dtype=float)
    return np.column_stack([x ** k for k in range(d + 1)])

def choose_degree(x_train, y_train, x_val, y_val, max_degree):
    best_d, best_mse = None, np.inf
    for d in range(1, max_degree + 1):
        coef = np.linalg.lstsq(design(x_train, d), y_train, rcond=None)[0]
        mse = np.mean((design(x_val, d) @ coef - y_val) ** 2)
        if mse < best_mse:
            best_d, best_mse = d, mse
    return best_d
`,
        hints: ['Use the `design` helper for both training and validation inputs with the same d.', 'Keep the degree with the smallest validation MSE; using `<` (not `<=`) keeps the smaller degree on ties.'],
        explain: `This is model selection in miniature: every candidate is trained on the same training data and judged on held-out data. The training error would always favour the biggest degree; the validation error reveals where extra flexibility starts fitting noise. Cross-validation (later) makes the choice more reliable when data is scarce.`,
        tests: [
          { name: 'a straight line wins on linear data', setup: 'xt = np.array([0.0, 1, 2, 3, 4, 5]); yt = 2 * xt + 1 + np.array([0.1, -0.1, 0.05, 0.0, -0.05, 0.1])\nxv = np.array([0.5, 2.5, 4.5]); yv = 2 * xv + 1', call: 'choose_degree(xt, yt, xv, yv, 4)', expect: '1' },
          { name: 'cubic data', setup: 'rng = np.random.default_rng(2)\nxt = rng.uniform(-2, 2, 40); yt = xt ** 3 - 2 * xt + rng.normal(0, 0.3, 40)\nxv = rng.uniform(-2, 2, 200); yv = xv ** 3 - 2 * xv + rng.normal(0, 0.3, 200)', call: 'choose_degree(xt, yt, xv, yv, 9)', expect: '3' },
          { name: 'another noisy cubic', setup: 'rng = np.random.default_rng(5)\nxt = rng.uniform(-3, 3, 30); yt = 0.5 * xt ** 3 - xt ** 2 + rng.normal(0, 1.0, 30)\nxv = rng.uniform(-3, 3, 300); yv = 0.5 * xv ** 3 - xv ** 2 + rng.normal(0, 1.0, 300)', call: 'choose_degree(xt, yt, xv, yv, 8)', hidden: true },
          { name: 'returns an int in range', setup: 'rng = np.random.default_rng(9)\nxt = rng.uniform(0, 6, 15); yt = np.sin(xt) + rng.normal(0, 0.2, 15)\nxv = rng.uniform(0, 6, 100); yv = np.sin(xv) + rng.normal(0, 0.2, 100)', code: 'd = choose_degree(xt, yt, xv, yv, 10)\nassert int(d) == d and 1 <= d <= 10, "expected a degree between 1 and 10, got %r" % (d,)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Underfitting vs. overfitting example', url: 'https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'regularization', branch: 'ml', title: 'Regularization: L1, L2 & Early Stopping', short: 'Regularization', level: 2, minutes: 40, prereqs: ['overfitting'],
    tags: ['ridge', 'lasso', 'l1', 'l2', 'weight decay', 'early stopping', 'sparsity'],
    summary: 'Fighting overfitting by preferring simpler models: L2 (ridge) shrinks weights, L1 (lasso) zeroes them out, and early stopping halts training at the right time.',
    why: `Collecting more data is the best cure for overfitting, but it is often expensive or impossible. **Regularization** is the next best thing: it adds a preference for simpler models directly into training. L2 regularization (called *weight decay* in deep learning) is used when training virtually every neural network including LLMs; L1 performs automatic feature selection; early stopping is the cheapest regularizer of all. Knowing how they work — and how strength λ trades bias for variance — is essential for getting models that generalize.`,
    objectives: [
      'Write the L2 (ridge) and L1 (lasso) regularized objectives',
      'Compute ridge regression in closed form, leaving the bias unpenalized',
      'Explain why L1 produces sparse weights and L2 does not',
      'Choose the regularization strength with validation data and use early stopping'
    ],
    sections: [
      { type: 'text', title: 'The idea: penalise complexity', md: `Overfit models typically have **large, finely balanced weights** that bend the fit to pass through noisy points. Regularization adds a penalty on the size of the weights:

$$ L_reg(w) = L_data(w) + λ · Ω(w) $$
:::where
L_reg(w) — the regularized loss that training actually minimizes
L_data(w) — the usual data loss, e.g. the MSE
Ω(w) — "omega": a penalty measuring the model's complexity, such as the size of its weights
λ — "lambda": the regularization strength, a hyperparameter; λ = 0 means no regularization, a large λ strongly prefers small weights (more bias, less variance)
:::

We usually **don't penalise the bias** b: shifting all predictions up or down isn't "complexity".` },
      { type: 'text', title: 'L2 regularization (ridge, weight decay)', md: `$$ Ω(w) = ‖w‖² = Σ wⱼ² $$
:::where
wⱼ — the j-th weight
‖w‖² — the squared L2 norm (squared length) of the weight vector
Σ wⱼ² — the sum of the squared weights; squaring punishes large weights far more than small ones
:::

Effects:
- Shrinks all weights smoothly towards zero, but rarely *to exactly* zero.
- Spreads weight across correlated features instead of putting it all on one.
- Has a closed form for linear regression (with centred data):

$$ w = (XᵀX + λI)⁻¹ Xᵀ y $$
:::where
X — the centred data matrix, shape (n, d)
y — the vector of targets
λ — the regularization strength
I — the identity matrix (d × d): ones on the diagonal, zeros everywhere else
(…)⁻¹ — the matrix inverse
:::

Adding λI also makes the matrix invertible even when features are redundant — a numerical bonus.

In gradient descent the penalty adds 2λw to the gradient, so every step also multiplies the weights by (1 − 2ηλ): they **decay** a little each step. That's why deep-learning libraries call it **weight decay** (AdamW applies it in a slightly different, decoupled way).` },
      { type: 'code', title: 'Ridge shrinks the weights of an overfit polynomial', code: `
import numpy as np
rng = np.random.default_rng(3)
x = rng.uniform(-1, 1, 15)
y = np.sin(3 * x) + rng.normal(0, 0.2, 15)
X = np.column_stack([x ** k for k in range(1, 10)])     # degree-9 polynomial features
Xc, yc = X - X.mean(axis=0), y - y.mean()               # centre so the bias is not penalized

x_test = np.linspace(-1, 1, 200); y_test = np.sin(3 * x_test)
Xt = np.column_stack([x_test ** k for k in range(1, 10)])

for lam in [0.0, 1e-4, 1e-2, 1.0, 100.0]:
    w = np.linalg.lstsq(Xc.T @ Xc + lam * np.eye(9), Xc.T @ yc, rcond=None)[0]
    b = y.mean() - X.mean(axis=0) @ w
    test_mse = np.mean((Xt @ w + b - y_test) ** 2)
    print(f"lambda={lam:<7} ||w|| = {np.linalg.norm(w):10.2f}   test MSE = {test_mse:.4f}")
`, after: `Without regularization the weights are huge and cancel each other out; moderate λ gives small weights and the best test error; very large λ flattens the curve (underfitting).` },
      { type: 'text', title: 'L1 regularization (lasso) and sparsity', md: `$$ Ω(w) = ‖w‖₁ = Σ |wⱼ| $$
:::where
|wⱼ| — the absolute value of the j-th weight (its size without the sign)
‖w‖₁ — the L1 norm: the sum of the absolute values of all the weights
:::

L1 pushes many weights to **exactly zero**, performing **feature selection** automatically — great when you suspect only a few of many features matter, and for interpretability.

Why the difference? The L1 penalty's slope stays constant (±λ) all the way down to zero, so a weight that is only weakly useful is pushed all the way to 0. The L2 penalty's slope (2λw) fades as the weight shrinks, so weights get small but never quite zero. Geometrically, the L1 constraint region is a diamond with corners on the axes, and the optimum tends to land on a corner (some weights 0); the L2 region is a circle with no corners.

**Elastic net** mixes both: λ₁‖w‖₁ + λ₂‖w‖², combining sparsity with L2's stability when features are correlated.` },
      { type: 'check', question: { type: 'mcq', q: 'You have 500 features and suspect only about 20 matter. Which regularizer helps identify them?', options: ['L2 (ridge)', 'L1 (lasso)', 'No regularization', 'A larger learning rate'], answer: 1, explain: 'L1 drives the weights of unhelpful features to exactly zero.' } },
      { type: 'text', title: 'Choosing λ and early stopping', md: `λ is a hyperparameter: try values on a log scale (1e-4, 1e-3, …, 10) and pick the one with the best **validation** score (or cross-validation score). Remember to **standardize features first** — otherwise the penalty hits features unevenly simply because of their units.

**Early stopping** is regularization by time: with gradient descent, weights start small and grow as training proceeds. Monitor the validation loss each epoch; keep the parameters from the best epoch and stop once it hasn't improved for a while ("patience"). It is free, effective, and standard in deep learning.

Other regularizers you'll meet later: **dropout** (randomly disabling neurons), **data augmentation** (training on modified copies of the data), **label smoothing**, and simply using **smaller models**.` },
      { type: 'check', question: { type: 'tf', q: 'Increasing the regularization strength λ generally increases bias and decreases variance.', answer: true, explain: 'Stronger penalties restrict the model more: it becomes simpler (more bias) and less sensitive to the particular training sample (less variance).' } }
    ],
    when: `- **L2 / weight decay**: the default for linear models with many features and for neural networks.
- **L1**: when you want a sparse, interpretable model or automatic feature selection.
- **Elastic net**: many correlated features *and* a desire for sparsity.
- **Early stopping**: any iterative training — essentially always in deep learning.
- Skip heavy regularization when you have abundant data relative to model size (but some weight decay rarely hurts).`,
    pitfalls: `- Regularizing unscaled features.
- Penalising the bias term.
- Picking λ on the test set.
- Comparing λ values across libraries without checking their conventions (scikit-learn's \`C\` is 1/λ, and penalties may be scaled by n).
- Expecting L2 to produce exact zeros.`,
    takeaways: [
      'Regularized loss = data loss + λ·penalty; λ trades bias for variance.',
      'L2 (ridge / weight decay) shrinks weights smoothly; closed form w = (XᵀX + λI)⁻¹Xᵀy on centred data.',
      'L1 (lasso) sets many weights exactly to zero → feature selection.',
      'Don\'t penalise the bias; standardize features first.',
      'Early stopping on validation loss is a free and powerful regularizer.'
    ],
    quiz: [
      { type: 'mcq', q: 'In ridge regression, what happens to the weights as λ → ∞?', options: ['They grow without bound', 'They shrink towards zero', 'They become exactly the least-squares solution', 'They become random'], answer: 1, explain: 'An infinitely strong L2 penalty forces all (penalized) weights to zero, leaving only the bias.' },
      { type: 'mcq', q: 'Which regularizer tends to produce exactly-zero weights?', options: ['L2', 'L1', 'Early stopping', 'Data augmentation'], answer: 1, explain: 'L1\'s constant-slope penalty pushes weak weights all the way to zero.' },
      { type: 'fill', q: 'With plain gradient descent and L2 penalty λ‖w‖², each step multiplies the weights by (1 − 2ηλ) before the data-gradient update. With η = 0.1 and λ = 0.05, what is that factor?', answer: ['0.99'], explain: '1 − 2 · 0.1 · 0.05 = 0.99 — the weights "decay" by 1% each step.' },
      { type: 'tf', q: 'You should choose the regularization strength using the test set.', answer: false, explain: 'Use a validation set or cross-validation; the test set stays untouched until the end.' },
      { type: 'mcq', q: 'Why standardize features before applying L1/L2 regularization?', options: ['It makes training deterministic', 'So the penalty treats features equally instead of depending on their units', 'Regularization only works on integers', 'It removes outliers'], answer: 1, explain: 'A feature measured in millimetres needs a tiny weight and would barely be penalized compared with one in kilometres.' }
    ],
    problems: [
      {
        id: 'ridge-closed-form', title: 'Ridge Regression in Closed Form', difficulty: 'medium',
        prompt: `Implement \`ridge(X, y, lam)\` returning \`(w, b)\` for the objective Σ(Xw + b − y)² + λ‖w‖², **without** penalising b:

1. Centre the data: \`x_mean = X.mean(axis=0)\`, \`y_mean = y.mean()\`, \`Xc = X - x_mean\`, \`yc = y - y_mean\`.
2. Solve \`(Xcᵀ Xc + λ I) w = Xcᵀ yc\` with \`np.linalg.solve\`.
3. Recover the bias: \`b = y_mean - x_mean @ w\`.

With λ = 0 this must match ordinary least squares; as λ grows, ‖w‖ must shrink.`,
        starter: `
import numpy as np

def ridge(X, y, lam):
    """Ridge regression with an unpenalized bias. Returns (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def ridge(X, y, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    d = X.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    b = y_mean - x_mean @ w
    return w, float(b)
`,
        hints: ['After centring, the bias drops out of the problem; solve for w alone.', '`np.linalg.solve(A, v)` solves A·w = v without forming an inverse.', 'The bias makes the model pass through the point (x_mean, y_mean).'],
        explain: `Centring X and y removes the intercept from the optimisation, so the penalty applies only to w; afterwards b is set so the fitted plane passes through the data's centre. Adding λI to XᵀX also guarantees a solvable system even with redundant features.`,
        tests: [
          { name: 'lambda = 0 equals least squares', call: 'ridge([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0], 0.0)', expect: '(np.array([2.0]), 1.0)' },
          { name: 'lambda shrinks the weight', call: 'ridge([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0], 5.0)', expect: '(np.array([1.0]), 2.5)' },
          { name: 'random data vs reference', setup: 'rng = np.random.default_rng(12)\nX = rng.normal(size=(60, 4))\ny = X @ np.array([1.0, -2.0, 0.0, 3.0]) + 0.5 + rng.normal(0, 0.5, 60)', call: 'ridge(X, y, 3.0)', hidden: true },
          { name: 'norm decreases with lambda', setup: 'rng = np.random.default_rng(2)\nX = rng.normal(size=(40, 6))\ny = rng.normal(size=40)', code: 'norms = [np.linalg.norm(ridge(X, y, lam)[0]) for lam in (0.0, 1.0, 10.0, 100.0)]\nassert all(a > b for a, b in zip(norms, norms[1:])), "weight norms should shrink as lambda grows: %s" % norms', hidden: true }
        ]
      },
      {
        id: 'l2-gradient-step', title: 'One Gradient Step with Weight Decay', difficulty: 'easy',
        prompt: `Implement one gradient-descent step on the L2-regularized MSE

$$ L(w, b) = mean( (Xw + b − y)² ) + λ‖w‖² $$
:::where
X — the data matrix, shape (n, d)
w, b — the weight vector, shape (d,), and the bias
y — the targets, shape (n,)
λ — lam, the regularization strength
‖w‖² — the sum of the squared weights (the bias is not included)
:::

\`ridge_step(X, y, w, b, lr, lam)\` → \`(w_new, b_new)\` where, with e = Xw + b − y and n examples:

- grad_w = (2/n)·Xᵀe + 2·λ·w
- grad_b = (2/n)·Σe   (the bias is not penalized)
- w_new = w − lr·grad_w, b_new = b − lr·grad_b

Don't modify the input \`w\`.`,
        starter: `
import numpy as np

def ridge_step(X, y, w, b, lr, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def ridge_step(X, y, w, b, lr, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    e = X @ w + b - y
    grad_w = (2 / n) * X.T @ e + 2 * lam * w
    grad_b = (2 / n) * e.sum()
    return w - lr * grad_w, float(b - lr * grad_b)
`,
        hints: ['The only difference from plain linear regression is the extra `2 * lam * w` term in grad_w.', 'Return new arrays (`w - lr * grad_w`) rather than updating in place.'],
        explain: `Rewrite the update as w_new = (1 − 2·lr·λ)·w − lr·(data gradient): each step first shrinks ("decays") the weights by a constant factor, then applies the usual correction. That's literally weight decay.`,
        tests: [
          { name: 'no data error, only decay', call: 'ridge_step([[1.0], [2.0]], [2.0, 4.0], [2.0], 0.0, 0.1, 0.5)', expect: '(np.array([1.8]), 0.0)' },
          { name: 'lam = 0 is plain gradient descent', call: 'ridge_step([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0], [0.0], 0.0, 0.1, 0.0)', expect: '(np.array([2.6 / 3]), 0.6)' },
          { name: 'random step', setup: 'rng = np.random.default_rng(4)\nX = rng.normal(size=(25, 3))\ny = rng.normal(size=25)\nw0 = rng.normal(size=3)', call: 'ridge_step(X, y, w0, 0.3, 0.05, 0.1)', hidden: true },
          { name: 'input w unchanged', code: 'w0 = np.array([1.0, -1.0])\nridge_step(np.eye(2), np.zeros(2), w0, 0.0, 0.1, 1.0)\nassert np.allclose(w0, [1.0, -1.0]), "ridge_step must not modify w"', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Ridge, Lasso and Elastic-Net', url: 'https://scikit-learn.org/stable/modules/linear_model.html' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
