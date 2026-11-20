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
      { type: 'text', title: "Generalization, overfitting and underfitting", md: `A model that is accurate on new data **generalizes**. We can only check that on data the model did not see while training, which is why every evaluation in this app uses a held-out test set.

Imagine a shop that wants to predict who will buy a boat, and has records of 12 past customers. A rule like *"customers older than 45 with fewer than 3 children, who are not divorced and live within 30 km of a lake, buy boats"* may fit all 12 records perfectly, yet it is almost certainly a coincidence of this tiny sample. A rule like *"people who own a house buy boats"* is too simple and ignores obvious signals. Somewhere between lies the model that will do best on next month's customers.

| | What it looks like | Training score | Test score |
| --- | --- | --- | --- |
| **Underfitting** | the model is too simple to capture the pattern | low | low |
| **Good fit** | captures the pattern, ignores the noise | high | close to training |
| **Overfitting** | the model memorizes quirks of the training sample | very high | clearly lower |

Making a model more flexible (fewer neighbours in k-NN, deeper trees, more features, weaker regularization) moves it from underfitting towards overfitting. The goal is the **sweet spot** where the test score peaks. See [Overfitting & Bias–Variance](#/concept/overfitting) for the full picture.` },
      { type: 'text', title: "Model complexity and dataset size", md: `How complex a model you can afford depends on **how much varied data** you have. With 12 boat customers, any detailed rule overfits; with 10,000 customers, the detailed rule can be checked against thousands of cases, and if it still holds it is probably real.

- **More data lets you use more flexible models** without overfitting, because coincidences that fit a small sample rarely survive a large one.
- **Variety matters, not just volume.** Copying the same customers ten times, or collecting thousands of near-identical examples, adds no information.
- **Collecting more data often beats clever modelling.** If you can cheaply gather more labelled examples, it is frequently the single best improvement available.

:::tip A useful habit
When a model overfits, ask first "can I get more (varied) data?", then "can I simplify the model or regularize it?". When it underfits, ask "do my features contain the signal?" before reaching for a more complex model.
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
      { type: 'code', title: "A first look at many model families on one dataset", timeoutSec: 90, md: `The breast-cancer dataset has 569 tumours, each described by 30 measurements of cell nuclei, with the diagnosis (malignant or benign) as the label. Here every major family on the map is trained with default settings and scored on the same held-out test set. Models that care about the scale of the features (k-NN, SVM, neural network, logistic regression) get standardized features, an idea explained in *Feature Engineering*.`, code: `import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
scaled = lambda model: make_pipeline(StandardScaler(), model)
models = {
    "baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
    "k-nearest neighbours": scaled(KNeighborsClassifier()),
    "logistic regression": scaled(LogisticRegression()),
    "Gaussian naive Bayes": GaussianNB(),
    "decision tree": DecisionTreeClassifier(random_state=0),
    "random forest": RandomForestClassifier(random_state=0),
    "gradient boosting": GradientBoostingClassifier(random_state=0),
    "support vector machine": scaled(SVC()),
    "neural network (MLP)": scaled(MLPClassifier(max_iter=1000, random_state=0)),
}
print(f"{'model':<26}{'train':>7}{'test':>7}")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name:<26}{model.score(X_train, y_train):>7.3f}{model.score(X_test, y_test):>7.3f}")`, after: `Reading the table:

- Every real model beats the baseline (which always predicts "benign") by a wide margin, so they have all learned something.
- The **single decision tree** reaches 100% on its training data but noticeably less on the test set: the classic signature of overfitting. Averaging many trees (random forest, gradient boosting) fixes much of that.
- Several very different families land within a percentage point or two of each other. On a small, clean dataset like this, the differences between good models are often smaller than the uncertainty of the test score itself, which is why [cross-validation](#/concept/cross-validation) matters.` },
      { type: 'chart', title: "Test accuracy of each model family", after: `Every real model clears the baseline easily, and the best ones are within about one test example of each other. Note the value axis starts at 0.5 here to make small differences visible; with bars that is a deliberate exception, flagged so it does not mislead.`, chart: {"kind":"bar","x":["baseline","k-NN","logistic","naive Bayes","tree","forest","boosting","SVM","MLP"],"series":[{"name":"test accuracy","y":[0.629,0.951,0.958,0.923,0.902,0.944,0.958,0.958,0.958]}],"xLabel":"model (default settings)","yLabel":"test accuracy","yMin":0.5,"yMax":1} },
      { type: 'text', title: "Which model should I try first?", md: `A practical summary of the supervised families on this map:

| Model | Reach for it when… | Watch out for |
| --- | --- | --- |
| [k-nearest neighbours](#/concept/knn) | the dataset is small and you want a simple baseline that is easy to explain | slow predictions on large data; needs scaled features; poor with many features |
| [Linear models](#/concept/logistic-regression) | always a good first try; very large or very high-dimensional data (text) | low-dimensional data with curved relationships |
| [Naive Bayes](#/concept/naive-bayes) | classification on huge or high-dimensional data, when speed matters most | usually less accurate than linear models; poor probabilities |
| [Decision trees](#/concept/decision-trees) | you need a model a person can read and audit; no scaling needed | a single tree overfits easily |
| [Random forests](#/concept/ensembles) | a strong default for tables of numbers and categories; little tuning | slower and bigger; weak on very sparse, high-dimensional data |
| [Gradient boosting](#/concept/ensembles) | squeezing out the best accuracy on tabular data | needs more careful tuning than a random forest |
| [Support vector machines](#/concept/svm) | medium-sized data where all features are measurements of the same kind | needs scaling and tuning; hard to inspect; slow on large data |
| [Neural networks](#/concept/mlp) | large datasets, images, sound, text | long training, many settings, scaling required |

:::tip A sensible order
Start with a baseline, then a simple model (linear model, naive Bayes or k-NN). Look at the data and the errors. Only then move to random forests, gradient boosting, SVMs or neural networks, and keep the simple model as a yardstick.
:::` },
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
      { type: 'mcq', q: 'A model trained on photos from one hospital performs badly at another hospital with different scanners. This is…', options: ['Underfitting', 'Distribution (covariate) shift', 'A learning-rate problem', 'Label leakage'], answer: 1, explain: 'The input distribution changed between training and deployment.' },
      {"type":"mcq","q":"A model scores 99% on its training data but 74% on the test set. What is the most likely diagnosis?","options":["Underfitting: the model is too simple","Overfitting: the model memorized the training data","The test set is too large","The model generalizes well"],"answer":1,"explain":"A large gap between a high training score and a much lower test score is the signature of overfitting."},
      {"type":"mcq","q":"Which change usually lets you use a more flexible model without overfitting?","options":["Duplicating every training example","Collecting more varied training data","Evaluating on the training set instead","Removing the test set"],"answer":1,"explain":"More varied data makes coincidences in the training sample less likely, so flexible models can be checked against many more cases. Duplicates add no new information."}
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
      { title: 'Google ML Crash Course — Framing', url: 'https://developers.google.com/machine-learning/crash-course' },
      { title: "Stanford CS229 — Machine Learning lecture notes", url: "https://cs229.stanford.edu/main_notes.pdf" },
      { title: "Andreas Müller & Sarah Guido — Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
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
      { type: 'chart', title: "Regression losses as a function of the error", after: `Squared error grows quadratically, so a single large error dominates the total: great when errors are Gaussian, bad with outliers. Absolute error grows linearly and is robust, but has a kink at 0. Huber is quadratic near zero and linear far away: the best of both.`, chart: {"kind":"line","x":[-3,-2.95,-2.9,-2.85,-2.8,-2.75,-2.7,-2.65,-2.6,-2.55,-2.5,-2.45,-2.4,-2.35,-2.3,-2.25,-2.2,-2.15,-2.1,-2.05,-2,-1.95,-1.9,-1.85,-1.8,-1.75,-1.7,-1.65,-1.6,-1.55,-1.5,-1.45,-1.4,-1.35,-1.3,-1.25,-1.2,-1.15,-1.1,-1.05,-1,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,2.55,2.6,2.65,2.7,2.75,2.8,2.85,2.9,2.95,3],"points":false,"series":[{"name":"squared error e²","y":[9,8.7025,8.41,8.1225,7.84,7.5625,7.29,7.0225,6.76,6.5025,6.25,6.0025,5.76,5.5225,5.29,5.0625,4.84,4.6225,4.41,4.2025,4,3.8025,3.61,3.4225,3.24,3.0625,2.89,2.7225,2.56,2.4025,2.25,2.1025,1.96,1.8225,1.69,1.5625,1.44,1.3225,1.21,1.1025,1,0.9025,0.81,0.7225,0.64,0.5625,0.49,0.4225,0.36,0.3025,0.25,0.2025,0.16,0.1225,0.09,0.0625,0.04,0.0225,0.01,0.0025,0,0.0025,0.01,0.0225,0.04,0.0625,0.09,0.1225,0.16,0.2025,0.25,0.3025,0.36,0.4225,0.49,0.5625,0.64,0.7225,0.81,0.9025,1,1.1025,1.21,1.3225,1.44,1.5625,1.69,1.8225,1.96,2.1025,2.25,2.4025,2.56,2.7225,2.89,3.0625,3.24,3.4225,3.61,3.8025,4,4.2025,4.41,4.6225,4.84,5.0625,5.29,5.5225,5.76,6.0025,6.25,6.5025,6.76,7.0225,7.29,7.5625,7.84,8.1225,8.41,8.7025,9]},{"name":"absolute error |e|","y":[3,2.95,2.9,2.85,2.8,2.75,2.7,2.65,2.6,2.55,2.5,2.45,2.4,2.35,2.3,2.25,2.2,2.15,2.1,2.05,2,1.95,1.9,1.85,1.8,1.75,1.7,1.65,1.6,1.55,1.5,1.45,1.4,1.35,1.3,1.25,1.2,1.15,1.1,1.05,1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,2.55,2.6,2.65,2.7,2.75,2.8,2.85,2.9,2.95,3]},{"name":"Huber (δ = 1)","y":[2.5,2.45,2.4,2.35,2.3,2.25,2.2,2.15,2.1,2.05,2,1.95,1.9,1.85,1.8,1.75,1.7,1.65,1.6,1.55,1.5,1.45,1.4,1.35,1.3,1.25,1.2,1.15,1.1,1.05,1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.4512,0.405,0.3613,0.32,0.2812,0.245,0.2112,0.18,0.1512,0.125,0.1012,0.08,0.0612,0.045,0.0312,0.02,0.0112,0.005,0.0012,0,0.0013,0.005,0.0113,0.02,0.0312,0.045,0.0613,0.08,0.1013,0.125,0.1513,0.18,0.2113,0.245,0.2812,0.32,0.3613,0.405,0.4513,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5]}],"xLabel":"error e = prediction − target","yLabel":"loss","yMax":9} },
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
`, predict: {"type":"mcq","q":"One of five labels is corrupted from 10 to 100. Which loss jumps the most compared with the clean data?","options":["MSE","MAE","Huber","All three jump by about the same factor"],"answer":0,"explain":"MSE goes from 0.028 to about 1,613 because the error of 90 is squared; MAE and Huber grow only linearly (to about 18). That is why models trained with MSE chase outliers.","expect":["with an outlier MSE 1612.828"]} },
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
      { type: 'chart', title: "Classification losses as a function of the margin", md: `For a label y ∈ {−1, +1} and a raw score s, the **margin** m = y · s is positive when the prediction is on the correct side and large when it is confidently correct. Every classification loss is a function of it:`, after: `The 0-1 loss is a step: flat everywhere except at the boundary, so it has no useful gradient. The three surrogates are convex and lie on or above it, so pushing them down also pushes the error rate down. Hinge loss stops caring once the margin passes 1; log loss keeps rewarding larger margins, ever more weakly; the exponential loss punishes badly wrong points so hard that it is sensitive to mislabelled data.`, chart: {"kind":"line","x":[-2,-1.95,-1.9,-1.85,-1.8,-1.75,-1.7,-1.65,-1.6,-1.55,-1.5,-1.45,-1.4,-1.35,-1.3,-1.25,-1.2,-1.15,-1.1,-1.05,-1,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,2.55,2.6,2.65,2.7,2.75,2.8,2.85,2.9,2.95,3],"points":false,"series":[{"name":"0-1 loss (error)","y":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"name":"hinge max(0, 1 − m)","y":[3,2.95,2.9,2.85,2.8,2.75,2.7,2.65,2.6,2.55,2.5,2.45,2.4,2.35,2.3,2.25,2.2,2.15,2.1,2.05,2,1.95,1.9,1.85,1.8,1.75,1.7,1.65,1.6,1.55,1.5,1.45,1.4,1.35,1.3,1.25,1.2,1.15,1.1,1.05,1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"name":"log loss log₂(1 + e⁻ᵐ)","y":[3.0685,3.0052,2.9422,2.8797,2.8176,2.7559,2.6946,2.6339,2.5736,2.5139,2.4546,2.3959,2.3378,2.2802,2.2232,2.1668,2.1111,2.056,2.0015,1.9477,1.8946,1.8423,1.7906,1.7397,1.6895,1.6402,1.5916,1.5438,1.4968,1.4506,1.4053,1.3608,1.3172,1.2745,1.2326,1.1916,1.1515,1.1123,1.0739,1.0365,1,0.9644,0.9297,0.8959,0.8629,0.8309,0.7998,0.7695,0.7401,0.7116,0.6839,0.6571,0.6312,0.606,0.5817,0.5581,0.5354,0.5134,0.4922,0.4717,0.4519,0.4329,0.4145,0.3969,0.3798,0.3635,0.3477,0.3326,0.318,0.304,0.2906,0.2777,0.2653,0.2534,0.2421,0.2312,0.2207,0.2107,0.2011,0.1919,0.1831,0.1747,0.1667,0.159,0.1516,0.1446,0.1378,0.1314,0.1253,0.1194,0.1138,0.1085,0.1034,0.0985,0.0938,0.0894,0.0852,0.0811,0.0773,0.0736,0.0701]},{"name":"exponential e⁻ᵐ (AdaBoost)","y":[null,null,null,null,null,null,null,null,null,null,null,null,null,3.8574,3.6693,3.4903,3.3201,3.1582,3.0042,2.8577,2.7183,2.5857,2.4596,2.3396,2.2255,2.117,2.0138,1.9155,1.8221,1.7333,1.6487,1.5683,1.4918,1.4191,1.3499,1.284,1.2214,1.1618,1.1052,1.0513,1,0.9512,0.9048,0.8607,0.8187,0.7788,0.7408,0.7047,0.6703,0.6376,0.6065,0.5769,0.5488,0.522,0.4966,0.4724,0.4493,0.4274,0.4066,0.3867,0.3679,0.3499,0.3329,0.3166,0.3012,0.2865,0.2725,0.2592,0.2466,0.2346,0.2231,0.2122,0.2019,0.192,0.1827,0.1738,0.1653,0.1572,0.1496,0.1423,0.1353,0.1287,0.1225,0.1165,0.1108,0.1054,0.1003,0.0954,0.0907,0.0863,0.0821,0.0781,0.0743,0.0707,0.0672,0.0639,0.0608,0.0578,0.055,0.0523,0.0498]}],"xLabel":"margin m = y · score (positive = correct side)","yLabel":"loss","yMin":0,"vlines":[{"x":0,"label":"decision boundary"}]} },
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
      { type: 'text', title: "Losses for imbalanced data and overconfident models", md: `Plain cross-entropy treats every example equally. Three standard modifications change that.

**Class weights.** With 1 fraud case per 1,000 transactions, a model can reach a low average loss while ignoring fraud entirely. Weighting each example's loss by its class restores the balance:

$$ L = −(1/n) Σᵢ w_{yᵢ} · log q_{yᵢ}          w_c = n / (K · n_c) $$
:::where
yᵢ — the true class of example i; q_{yᵢ} — the probability the model gave that class
w_c — the weight for class c; this "balanced" choice makes every class contribute equally in total
n_c — the number of training examples in class c; K — the number of classes; n — the number of examples
:::

In scikit-learn this is \`class_weight="balanced"\`. Weighting distorts the predicted probabilities (the model now over-predicts the rare class), so recalibrate if you need honest probabilities, or keep the plain loss and simply move the decision threshold.

**Focal loss** down-weights the examples the model already gets right, so training concentrates on the hard ones. It was designed for object detection, where most candidate boxes are easy background:

$$ FL(p_t) = −(1 − p_t)^γ · log p_t $$
:::where
p_t — the probability the model assigns to the true class
γ — "gamma", the focusing parameter (2 is common); γ = 0 gives plain cross-entropy
(1 − p_t)^γ — close to 0 for easy, confidently correct examples and close to 1 for hard ones
:::

**Label smoothing** trains on softened targets: instead of 1 for the correct class and 0 for the rest, use 1 − ε and spread ε over the other classes (ε = 0.1 is typical). The model can no longer drive its loss to zero by becoming infinitely confident, which tends to improve calibration and generalisation. Many image classifiers and transformers are trained this way.` },
      { type: 'chart', title: "Focal loss against cross-entropy", after: `For an easy example (p_t = 0.9) the γ = 2 focal loss is 100 times smaller than cross-entropy, while for a badly misclassified one (p_t = 0.1) it is only about 20% smaller. Thousands of easy examples no longer drown out the few hard ones.`, chart: {"kind":"line","x":[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1],"points":false,"series":[{"name":"cross-entropy (γ = 0)","y":[4.6052,3.912,3.5066,3.2189,2.9957,2.8134,2.6593,2.5257,2.4079,2.3026,2.2073,2.1203,2.0402,1.9661,1.8971,1.8326,1.772,1.7148,1.6607,1.6094,1.5606,1.5141,1.4697,1.4271,1.3863,1.3471,1.3093,1.273,1.2379,1.204,1.1712,1.1394,1.1087,1.0788,1.0498,1.0217,0.9943,0.9676,0.9416,0.9163,0.8916,0.8675,0.844,0.821,0.7985,0.7765,0.755,0.734,0.7133,0.6931,0.6733,0.6539,0.6349,0.6162,0.5978,0.5798,0.5621,0.5447,0.5276,0.5108,0.4943,0.478,0.462,0.4463,0.4308,0.4155,0.4005,0.3857,0.3711,0.3567,0.3425,0.3285,0.3147,0.3011,0.2877,0.2744,0.2614,0.2485,0.2357,0.2231,0.2107,0.1985,0.1863,0.1744,0.1625,0.1508,0.1393,0.1278,0.1165,0.1054,0.0943,0.0834,0.0726,0.0619,0.0513,0.0408,0.0305,0.0202,0.0101,0]},{"name":"focal, γ = 1","y":[4.5591,3.8338,3.4014,3.0901,2.8459,2.6446,2.4731,2.3237,2.1912,2.0723,1.9645,1.8658,1.775,1.6909,1.6126,1.5394,1.4707,1.4061,1.3452,1.2876,1.2329,1.181,1.1317,1.0846,1.0397,0.9968,0.9558,0.9165,0.8789,0.8428,0.8081,0.7748,0.7428,0.712,0.6824,0.6539,0.6264,0.5999,0.5744,0.5498,0.526,0.5032,0.4811,0.4597,0.4392,0.4193,0.4002,0.3817,0.3638,0.3466,0.3299,0.3139,0.2984,0.2834,0.269,0.2551,0.2417,0.2288,0.2163,0.2043,0.1928,0.1817,0.171,0.1607,0.1508,0.1413,0.1322,0.1234,0.115,0.107,0.0993,0.092,0.085,0.0783,0.0719,0.0659,0.0601,0.0547,0.0495,0.0446,0.04,0.0357,0.0317,0.0279,0.0244,0.0211,0.0181,0.0153,0.0128,0.0105,0.0085,0.0067,0.0051,0.0037,0.0026,0.0016,0.0009,0.0004,0.0001,0]},{"name":"focal, γ = 2","y":[4.5135,3.7571,3.2993,2.9665,2.7036,2.4859,2.3,2.1378,1.994,1.8651,1.7484,1.6419,1.5442,1.4541,1.3707,1.2931,1.2207,1.153,1.0896,1.03,0.974,0.9212,0.8714,0.8243,0.7798,0.7377,0.6977,0.6599,0.624,0.5899,0.5576,0.5269,0.4977,0.4699,0.4435,0.4185,0.3946,0.3719,0.3504,0.3299,0.3104,0.2918,0.2742,0.2575,0.2415,0.2264,0.2121,0.1985,0.1855,0.1733,0.1617,0.1507,0.1402,0.1304,0.1211,0.1123,0.1039,0.0961,0.0887,0.0817,0.0752,0.069,0.0633,0.0578,0.0528,0.048,0.0436,0.0395,0.0357,0.0321,0.0288,0.0258,0.0229,0.0204,0.018,0.0158,0.0138,0.012,0.0104,0.0089,0.0076,0.0064,0.0054,0.0045,0.0037,0.003,0.0024,0.0018,0.0014,0.0011,0.0008,0.0005,0.0004,0.0002,0.0001,0.0001,0,0,0,0]}],"xLabel":"p_t, the probability given to the true class","yLabel":"loss","yMin":0} },
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

Losses can also be **combined**: training objectives often add a regularization term (*Regularization*) or several task losses with weights.` },
      { type: 'text', title: "The quantile loss: predicting a range, not a point", md: `MSE makes a model predict the **mean** of y given x, and MAE the **median**. Sometimes you need a different point of the distribution: a delivery time you will beat 90% of the time, or the stock needed to cover demand on 95% of days. The **quantile loss** (also called the pinball loss) targets any quantile τ by charging under- and over-predictions differently:

$$ ℓ_τ(y, ŷ) = τ · (y − ŷ)  if y ≥ ŷ,      (1 − τ) · (ŷ − y)  if y < ŷ $$
:::where
y — the true value; ŷ — the prediction
τ — "tau": the target quantile between 0 and 1 (0.5 is the median, 0.9 the 90th percentile)
τ · (y − ŷ) — the cost of predicting too low
(1 − τ) · (ŷ − y) — the cost of predicting too high
:::

With τ = 0.5 both mistakes cost the same and the loss is half the absolute error, so the model predicts the median. With τ = 0.9 under-predicting costs 9 times as much, so the model moves up until only 10% of outcomes lie above its prediction. Training one model at τ = 0.1 and another at τ = 0.9 gives an **80% prediction interval** for every input: a range instead of a single number, and one that can be wider where the data is noisier.` },
      { type: 'chart', title: "The quantile (pinball) loss", after: `Read the τ = 0.9 line: predicting too low (left) costs 9 times more per unit than predicting too high (right), so the best prediction is high enough that 90% of outcomes fall below it. τ = 0.5 is symmetric: half the absolute error, which targets the median.`, chart: {"kind":"line","x":[-3,-2.95,-2.9,-2.85,-2.8,-2.75,-2.7,-2.65,-2.6,-2.55,-2.5,-2.45,-2.4,-2.35,-2.3,-2.25,-2.2,-2.15,-2.1,-2.05,-2,-1.95,-1.9,-1.85,-1.8,-1.75,-1.7,-1.65,-1.6,-1.55,-1.5,-1.45,-1.4,-1.35,-1.3,-1.25,-1.2,-1.15,-1.1,-1.05,-1,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,2.55,2.6,2.65,2.7,2.75,2.8,2.85,2.9,2.95,3],"points":false,"series":[{"name":"τ = 0.1","y":[0.3,0.295,0.29,0.285,0.28,0.275,0.27,0.265,0.26,0.255,0.25,0.245,0.24,0.235,0.23,0.225,0.22,0.215,0.21,0.205,0.2,0.195,0.19,0.185,0.18,0.175,0.17,0.165,0.16,0.155,0.15,0.145,0.14,0.135,0.13,0.125,0.12,0.115,0.11,0.105,0.1,0.095,0.09,0.085,0.08,0.075,0.07,0.065,0.06,0.055,0.05,0.045,0.04,0.035,0.03,0.025,0.02,0.015,0.01,0.005,0,0.045,0.09,0.135,0.18,0.225,0.27,0.315,0.36,0.405,0.45,0.495,0.54,0.585,0.63,0.675,0.72,0.765,0.81,0.855,0.9,0.945,0.99,1.035,1.08,1.125,1.17,1.215,1.26,1.305,1.35,1.395,1.44,1.485,1.53,1.575,1.62,1.665,1.71,1.755,1.8,1.845,1.89,1.935,1.98,2.025,2.07,2.115,2.16,2.205,2.25,2.295,2.34,2.385,2.43,2.475,2.52,2.565,2.61,2.655,2.7]},{"name":"τ = 0.5","y":[1.5,1.475,1.45,1.425,1.4,1.375,1.35,1.325,1.3,1.275,1.25,1.225,1.2,1.175,1.15,1.125,1.1,1.075,1.05,1.025,1,0.975,0.95,0.925,0.9,0.875,0.85,0.825,0.8,0.775,0.75,0.725,0.7,0.675,0.65,0.625,0.6,0.575,0.55,0.525,0.5,0.475,0.45,0.425,0.4,0.375,0.35,0.325,0.3,0.275,0.25,0.225,0.2,0.175,0.15,0.125,0.1,0.075,0.05,0.025,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1,1.125,1.15,1.175,1.2,1.225,1.25,1.275,1.3,1.325,1.35,1.375,1.4,1.425,1.45,1.475,1.5]},{"name":"τ = 0.9","y":[2.7,2.655,2.61,2.565,2.52,2.475,2.43,2.385,2.34,2.295,2.25,2.205,2.16,2.115,2.07,2.025,1.98,1.935,1.89,1.845,1.8,1.755,1.71,1.665,1.62,1.575,1.53,1.485,1.44,1.395,1.35,1.305,1.26,1.215,1.17,1.125,1.08,1.035,0.99,0.945,0.9,0.855,0.81,0.765,0.72,0.675,0.63,0.585,0.54,0.495,0.45,0.405,0.36,0.315,0.27,0.225,0.18,0.135,0.09,0.045,0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.125,0.13,0.135,0.14,0.145,0.15,0.155,0.16,0.165,0.17,0.175,0.18,0.185,0.19,0.195,0.2,0.205,0.21,0.215,0.22,0.225,0.23,0.235,0.24,0.245,0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.285,0.29,0.295,0.3]}],"xLabel":"error e = prediction − target","yLabel":"loss"} },
      { type: 'code', title: "Prediction intervals with quantile gradient boosting", timeoutSec: 90, code: `import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 1600)
y = np.sin(x) + rng.normal(0, 0.1 + 0.05 * x)        # the noise grows with x
X = x.reshape(-1, 1)
train, test = slice(0, 1200), slice(1200, None)

models = {}
for tau in (0.1, 0.5, 0.9):
    models[tau] = GradientBoostingRegressor(loss="quantile", alpha=tau, n_estimators=150, max_depth=3,
                                            learning_rate=0.1, random_state=0).fit(X[train], y[train])
lo, hi = models[0.1].predict(X[test]), models[0.9].predict(X[test])
inside = np.mean((y[test] >= lo) & (y[test] <= hi))
print(f"test points below the interval {np.mean(y[test] < lo):.1%}, inside {inside:.1%}, above {np.mean(y[test] > hi):.1%}")
print("coverage roughly 80% (between 70% and 90%):", 0.70 <= inside <= 0.90)
for xv in (1.0, 5.0, 9.0):
    l, m, h = (models[t].predict([[xv]])[0] for t in (0.1, 0.5, 0.9))
    print(f"x = {xv}: median {m:+.2f}, 80% interval [{l:+.2f}, {h:+.2f}], width {h - l:.2f}")`, after: `About three quarters of the unseen points land inside the interval, with similar misses on each side: close to the 80% target, but a little short. Quantile models fitted on training data are often slightly overconfident on new data, which is why intervals should be checked, or adjusted with conformal prediction (see [Calibration & Uncertainty](#/concept/calibration-uncertainty)), on held-out data. The interval widens as x grows because the noise does; a single MSE model would give only the middle line. \`HistGradientBoostingRegressor(loss="quantile", quantile=…)\` and LightGBM's \`objective="quantile"\` do the same on large datasets.`, predict: {"type":"mcq","q":"Two models are trained with the quantile loss at τ = 0.1 and τ = 0.9. What share of new points should fall between their predictions?","options":["About 10%","About 50%","About 80%","About 90%"],"answer":2,"explain":"About 10% of outcomes should fall below the τ = 0.1 prediction and 10% above the τ = 0.9 one, leaving about 80% in between. In practice the share is often a little lower on new data.","expect":["coverage roughly 80% (between 70% and 90%): True"]} }
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
      { type: 'mcq', q: 'Which loss is standard for training large language models?', options: ['Mean squared error between token ids', 'Cross-entropy on the next token', 'Hinge loss', 'Mean absolute error'], answer: 1, explain: 'LLMs minimise the cross-entropy of the true next token under the predicted distribution.' },
      {"type":"mcq","q":"With the quantile loss at τ = 0.9, under-predicting by 1 costs 0.9. What does over-predicting by 1 cost?","options":["0.9","0.1","1.0","9"],"answer":1,"explain":"Over-predictions cost (1 − τ) per unit, here 0.1, so the model is pushed upwards until only about 10% of outcomes lie above its prediction."},
      {"type":"mcq","q":"What does the factor (1 − p_t)^γ in the focal loss do?","options":["It increases the loss of easy examples","It shrinks the loss of examples the model already classifies confidently, so training focuses on the hard ones","It makes the loss symmetric","It removes the need for a learning rate"],"answer":1,"explain":"For an easy example p_t is close to 1, so (1 − p_t)^γ is close to 0 and the example contributes little to the loss."},
      {"type":"tf","q":"The hinge loss, the log loss (in bits) and the exponential loss are all convex functions of the margin that lie on or above the 0-1 loss.","answer":true,"explain":"That is what makes them useful surrogates: minimising any of them pushes down an upper bound on the error rate, and their convexity makes them easy to optimise."}
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
      { title: 'scikit-learn — Metrics and scoring', url: 'https://scikit-learn.org/stable/modules/model_evaluation.html' },
      { title: "PyTorch — Loss functions reference", url: "https://pytorch.org/docs/stable/nn.html#loss-functions" }
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
      { type: 'chart', title: "Loss per step for different learning rates", md: `The loss (w − 3)², starting from w = 0, minimised with five different learning rates:`, after: `On a log scale, steady progress is a straight downward line. Too small a step crawls; a well-chosen one converges in a few steps (0.45 reaches the bottom of the chart in four); 0.95 overshoots back and forth but still converges slowly; past the critical value (here 1.0) every step overshoots by more than the last and the loss explodes. In real training you watch exactly this curve.`, chart: {"kind":"line","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"yLog":true,"points":false,"series":[{"name":"η = 0.01 (too small)","y":[9,8.644,8.301,7.973,7.657,7.354,7.062,6.783,6.514,6.256,6.008,5.771,5.542,5.323,5.112,4.909,4.715,4.528,4.349,4.177,4.011,3.852,3.7,3.553,3.413,3.278,3.148,3.023,2.903,2.788,2.678]},{"name":"η = 0.1 (good)","y":[9,5.76,3.686,2.359,1.51,0.9664,0.6185,0.3958,0.2533,0.1621,0.1038,0.06641,0.0425,0.0272,0.01741,0.01114,0.007131,0.004564,0.002921,0.001869,0.001196,0.0007656,0.00049,0.0003136,0.0002007,0.0001285,0.00008221,0.00005261,0.00003367,0.00002155,0.00001379]},{"name":"η = 0.45 (fast)","y":[9,0.09,0.0009,0.000009,9e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8]},{"name":"η = 0.95 (oscillates)","y":[9,7.29,5.905,4.783,3.874,3.138,2.542,2.059,1.668,1.351,1.094,0.8863,0.7179,0.5815,0.471,0.3815,0.309,0.2503,0.2028,0.1642,0.133,0.1078,0.08728,0.0707,0.05726,0.04638,0.03757,0.03043,0.02465,0.01997,0.01617]},{"name":"η = 1.05 (diverges)","y":[9,10.89,13.18,15.94,19.29,23.34,28.25,34.18,41.35,50.04,60.55,73.26,88.65,107.3,129.8,157,190,229.9,278.2,336.6,407.3,492.9,596.4,721.6,873.2,1000,1000,1000,1000,1000,1000]}],"xLabel":"step","yLabel":"loss (w − 3)² (log scale)","yMin":1e-8,"yMax":1000} },
      { type: 'check', question: { type: 'mcq', q: 'Your training loss decreases for a few steps, then shoots up to NaN. What is the most likely fix?', options: ['Increase the learning rate', 'Decrease the learning rate', 'Train for more epochs', 'Use a bigger model'], answer: 1, explain: 'Exploding loss is the classic symptom of a learning rate that is too large: steps overshoot and grow.' } },
      { type: 'text', title: "Gradients in many dimensions", md: `With many parameters the gradient is a **vector** of partial derivatives, one per parameter: each entry says how fast the loss changes when you nudge that parameter alone.

$$ ∇L(w) = [ ∂L/∂w₁, ∂L/∂w₂, …, ∂L/∂w_d ] $$
:::where
w = (w₁, …, w_d) — the d parameters
∂L/∂wⱼ — the partial derivative: the slope of the loss along parameter j, holding the others fixed
∇L(w) — the gradient: it points in the direction of steepest increase, perpendicular to the contour line through w
:::

Picture the loss as a landscape seen from above, with contour lines joining points of equal loss. Each step moves perpendicular to the contour through the current point. On a round bowl that heads straight for the minimum. On an **elongated** bowl it does not. Take L(w₁, w₂) = w₁² + 10·w₂², whose gradient is (2w₁, 20w₂). The w₂ direction is ten times steeper, so:

- the learning rate must be small enough for the steep direction, or w₂ overshoots and diverges; for a quadratic bowl the limit is η < 2 / (largest curvature), here 2 / 20 = 0.1;
- at such a learning rate the shallow w₁ direction crawls, and the path zig-zags across the valley.

$$ stable if  η < 2 / λ_max          number of steps grows with  κ = λ_max / λ_min $$
:::where
λ_max, λ_min — the largest and smallest curvatures of the loss: the eigenvalues of its matrix of second derivatives, the Hessian (here 20 and 2)
η — the learning rate
κ — the condition number of the problem; the larger it is, the more steps plain gradient descent needs
:::

Two practical fixes follow. **Standardise your features**: inputs on wildly different scales (metres and millimetres, age and income) create exactly this elongated landscape for linear models. And use **momentum** (which averages away the zig-zag) or **adaptive optimizers** such as Adam (which scale each parameter's step by its own gradient history); see [Optimizers](#/concept/optimizers).` },
      { type: 'chart', title: "Gradient descent zig-zags on an elongated bowl", md: `Fifteen steps from w = (−9, 1.5) with η = 0.09, just under the stability limit of 0.1:`, after: `Each step is perpendicular to the contour it starts on, so instead of heading for the centre the path bounces between the steep walls (w₂ flips sign every step) while creeping along the valley floor in w₁. After 15 steps w₁ is still at −0.46. Rescaling w₂, or an optimizer with momentum, would remove the zig-zag.`, chart: {"kind":"line","series":[{"name":"contours of w₁² + 10·w₂² (loss 1, 5, 20, 50, 100)","x":[1,0.995,0.978,0.951,0.914,0.866,0.809,0.743,0.669,0.588,0.5,0.407,0.309,0.208,0.105,0,-0.105,-0.208,-0.309,-0.407,-0.5,-0.588,-0.669,-0.743,-0.809,-0.866,-0.914,-0.951,-0.978,-0.995,-1,-0.995,-0.978,-0.951,-0.914,-0.866,-0.809,-0.743,-0.669,-0.588,-0.5,-0.407,-0.309,-0.208,-0.105,0,0.105,0.208,0.309,0.407,0.5,0.588,0.669,0.743,0.809,0.866,0.914,0.951,0.978,0.995,1,null,2.236,2.224,2.187,2.127,2.043,1.936,1.809,1.662,1.496,1.314,1.118,0.909,0.691,0.465,0.234,0,-0.234,-0.465,-0.691,-0.909,-1.118,-1.314,-1.496,-1.662,-1.809,-1.936,-2.043,-2.127,-2.187,-2.224,-2.236,-2.224,-2.187,-2.127,-2.043,-1.936,-1.809,-1.662,-1.496,-1.314,-1.118,-0.909,-0.691,-0.465,-0.234,0,0.234,0.465,0.691,0.909,1.118,1.314,1.496,1.662,1.809,1.936,2.043,2.127,2.187,2.224,2.236,null,4.472,4.448,4.374,4.253,4.085,3.873,3.618,3.323,2.992,2.629,2.236,1.819,1.382,0.93,0.467,0,-0.467,-0.93,-1.382,-1.819,-2.236,-2.629,-2.992,-3.323,-3.618,-3.873,-4.085,-4.253,-4.374,-4.448,-4.472,-4.448,-4.374,-4.253,-4.085,-3.873,-3.618,-3.323,-2.992,-2.629,-2.236,-1.819,-1.382,-0.93,-0.467,0,0.467,0.93,1.382,1.819,2.236,2.629,2.992,3.323,3.618,3.873,4.085,4.253,4.374,4.448,4.472,null,7.071,7.032,6.917,6.725,6.46,6.124,5.721,5.255,4.731,4.156,3.536,2.876,2.185,1.47,0.739,0,-0.739,-1.47,-2.185,-2.876,-3.536,-4.156,-4.731,-5.255,-5.721,-6.124,-6.46,-6.725,-6.917,-7.032,-7.071,-7.032,-6.917,-6.725,-6.46,-6.124,-5.721,-5.255,-4.731,-4.156,-3.536,-2.876,-2.185,-1.47,-0.739,0,0.739,1.47,2.185,2.876,3.536,4.156,4.731,5.255,5.721,6.124,6.46,6.725,6.917,7.032,7.071,null,10,9.945,9.781,9.511,9.135,8.66,8.09,7.431,6.691,5.878,5,4.067,3.09,2.079,1.045,0,-1.045,-2.079,-3.09,-4.067,-5,-5.878,-6.691,-7.431,-8.09,-8.66,-9.135,-9.511,-9.781,-9.945,-10,-9.945,-9.781,-9.511,-9.135,-8.66,-8.09,-7.431,-6.691,-5.878,-5,-4.067,-3.09,-2.079,-1.045,0,1.045,2.079,3.09,4.067,5,5.878,6.691,7.431,8.09,8.66,9.135,9.511,9.781,9.945,10,null],"y":[0,0.033,0.066,0.098,0.129,0.158,0.186,0.212,0.235,0.256,0.274,0.289,0.301,0.309,0.314,0.316,0.314,0.309,0.301,0.289,0.274,0.256,0.235,0.212,0.186,0.158,0.129,0.098,0.066,0.033,0,-0.033,-0.066,-0.098,-0.129,-0.158,-0.186,-0.212,-0.235,-0.256,-0.274,-0.289,-0.301,-0.309,-0.314,-0.316,-0.314,-0.309,-0.301,-0.289,-0.274,-0.256,-0.235,-0.212,-0.186,-0.158,-0.129,-0.098,-0.066,-0.033,0,null,0,0.074,0.147,0.219,0.288,0.354,0.416,0.473,0.525,0.572,0.612,0.646,0.672,0.692,0.703,0.707,0.703,0.692,0.672,0.646,0.612,0.572,0.525,0.473,0.416,0.354,0.288,0.219,0.147,0.074,0,-0.074,-0.147,-0.219,-0.288,-0.354,-0.416,-0.473,-0.525,-0.572,-0.612,-0.646,-0.672,-0.692,-0.703,-0.707,-0.703,-0.692,-0.672,-0.646,-0.612,-0.572,-0.525,-0.473,-0.416,-0.354,-0.288,-0.219,-0.147,-0.074,0,null,0,0.148,0.294,0.437,0.575,0.707,0.831,0.946,1.051,1.144,1.225,1.292,1.345,1.383,1.406,1.414,1.406,1.383,1.345,1.292,1.225,1.144,1.051,0.946,0.831,0.707,0.575,0.437,0.294,0.148,0,-0.148,-0.294,-0.437,-0.575,-0.707,-0.831,-0.946,-1.051,-1.144,-1.225,-1.292,-1.345,-1.383,-1.406,-1.414,-1.406,-1.383,-1.345,-1.292,-1.225,-1.144,-1.051,-0.946,-0.831,-0.707,-0.575,-0.437,-0.294,-0.148,0,null,0,0.234,0.465,0.691,0.909,1.118,1.314,1.496,1.662,1.809,1.936,2.043,2.127,2.187,2.224,2.236,2.224,2.187,2.127,2.043,1.936,1.809,1.662,1.496,1.314,1.118,0.909,0.691,0.465,0.234,0,-0.234,-0.465,-0.691,-0.909,-1.118,-1.314,-1.496,-1.662,-1.809,-1.936,-2.043,-2.127,-2.187,-2.224,-2.236,-2.224,-2.187,-2.127,-2.043,-1.936,-1.809,-1.662,-1.496,-1.314,-1.118,-0.909,-0.691,-0.465,-0.234,0,null,0,0.331,0.657,0.977,1.286,1.581,1.859,2.116,2.35,2.558,2.739,2.889,3.008,3.093,3.145,3.162,3.145,3.093,3.008,2.889,2.739,2.558,2.35,2.116,1.859,1.581,1.286,0.977,0.657,0.331,0,-0.331,-0.657,-0.977,-1.286,-1.581,-1.859,-2.116,-2.35,-2.558,-2.739,-2.889,-3.008,-3.093,-3.145,-3.162,-3.145,-3.093,-3.008,-2.889,-2.739,-2.558,-2.35,-2.116,-1.859,-1.581,-1.286,-0.977,-0.657,-0.331,0,null],"dash":true},{"name":"15 steps of gradient descent, η = 0.09","x":[-9,-7.38,-6.052,-4.962,-4.069,-3.337,-2.736,-2.244,-1.84,-1.509,-1.237,-1.014,-0.832,-0.682,-0.559,-0.459],"y":[1.5,-1.2,0.96,-0.768,0.614,-0.492,0.393,-0.315,0.252,-0.201,0.161,-0.129,0.103,-0.082,0.066,-0.053]}],"xLabel":"w₁ (the shallow direction)","yLabel":"w₂ (the steep direction)","xMin":-10.5,"xMax":10.5,"yMin":-3.3,"yMax":3.3} },
      { type: 'code', title: "Why feature scaling speeds up gradient descent", md: `The same linear regression, once with raw features (area in m² and number of rooms) and once with standardised ones. Each run uses a safe learning rate for its own landscape and counts the steps until the training MSE is within 1% of the best possible value.`, code: `import numpy as np
rng = np.random.default_rng(0)
n = 300
area = rng.uniform(40, 200, n)                     # square metres
rooms = rng.integers(1, 6, n).astype(float)        # 1 to 5 rooms
y = 3000 * area + 20000 * rooms + rng.normal(0, 20000, n)
y = (y - y.mean()) / y.std()                       # put the target on a unit scale
MAX_STEPS = 5000

def steps_to_converge(X):
    Xb = np.column_stack([X, np.ones(len(X))])     # add an intercept column
    best = np.mean((Xb @ np.linalg.lstsq(Xb, y, rcond=None)[0] - y) ** 2)
    H = 2 * Xb.T @ Xb / len(y)                     # the curvature (Hessian) of the MSE
    lr = 1.0 / np.linalg.eigvalsh(H).max()         # a safe learning rate: half the stability limit
    w = np.zeros(Xb.shape[1])
    for step in range(MAX_STEPS + 1):
        residual = Xb @ w - y
        if np.mean(residual ** 2) <= 1.01 * best:  # within 1% of the best possible MSE
            return step, np.linalg.cond(H)
        w -= lr * 2 * Xb.T @ residual / len(y)
    return None, np.linalg.cond(H)

raw = np.column_stack([area, rooms])
scaled = (raw - raw.mean(axis=0)) / raw.std(axis=0)
steps = {}
for name, X in [("raw features", raw), ("standardised", scaled)]:
    steps[name], cond = steps_to_converge(X)
    result = f"{steps[name]} steps" if steps[name] is not None else f"not within {MAX_STEPS:,} steps"
    print(f"{name:<13} condition number {cond:>12,.1f}   converged: {result}")
s_raw, s_scaled = steps["raw features"], steps["standardised"]
print("scaling needs far fewer steps:", s_scaled is not None and (s_raw is None or s_raw > 10 * s_scaled))`, after: `With raw features the curvature differs enormously between directions (the area column is in the hundreds, the intercept column is 1), so the condition number is in the hundreds of thousands and gradient descent does not get there within 5,000 steps. Standardised features make the bowl nearly round and a couple of steps suffice. Closed-form solvers don't care about scaling, but every gradient-based learner does.`, predict: {"type":"mcq","q":"One feature is an area in m² (40 to 200), the other a room count (1 to 5). Compared with standardised features, plain gradient descent on the raw features will need…","options":["About the same number of steps","Far more steps","Fewer steps","Exactly one step"],"answer":1,"explain":"Very different feature scales stretch the loss bowl, so the safe learning rate is set by the steep direction and the flat directions crawl.","expect":["scaling needs far fewer steps: True"]} },
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
      { type: 'chart', title: "Progress per pass over the data: full batch vs mini-batches", after: `With the same learning rate, full-batch gradient descent has made only 5 updates after 5 passes and is still far from the minimum. Mini-batches of 32 make 32 updates per pass and reach the noise floor in under two epochs. Single-example SGD gets there sooner still, but its loss never settles: each step follows one noisy example, which is why its learning rate is usually decayed over time.`, chart: {"kind":"line","yLog":true,"points":false,"series":[{"name":"full batch (1 update per pass)","x":[0,1,2,3,4,5],"y":[4.958,4.202,3.578,3.057,2.619,2.248]},{"name":"mini-batches of 32","x":[0,0.032,0.064,0.096,0.128,0.16,0.192,0.224,0.256,0.288,0.32,0.352,0.384,0.416,0.448,0.48,0.512,0.544,0.576,0.608,0.64,0.672,0.704,0.736,0.768,0.8,0.832,0.864,0.896,0.928,0.96,0.992,1,1.032,1.064,1.096,1.128,1.16,1.192,1.224,1.256,1.288,1.32,1.352,1.384,1.416,1.448,1.48,1.512,1.544,1.576,1.608,1.64,1.672,1.704,1.736,1.768,1.8,1.832,1.864,1.896,1.928,1.96,1.992,2,2.032,2.064,2.096,2.128,2.16,2.192,2.224,2.256,2.288,2.32,2.352,2.384,2.416,2.448,2.48,2.512,2.544,2.576,2.608,2.64,2.672,2.704,2.736,2.768,2.8,2.832,2.864,2.896,2.928,2.96,2.992,3,3.032,3.064,3.096,3.128,3.16,3.192,3.224,3.256,3.288,3.32,3.352,3.384,3.416,3.448,3.48,3.512,3.544,3.576,3.608,3.64,3.672,3.704,3.736,3.768,3.8,3.832,3.864,3.896,3.928,3.96,3.992,4,4.032,4.064,4.096,4.128,4.16,4.192,4.224,4.256,4.288,4.32,4.352,4.384,4.416,4.448,4.48,4.512,4.544,4.576,4.608,4.64,4.672,4.704,4.736,4.768,4.8,4.832,4.864,4.896,4.928,4.96,4.992,5],"y":[4.958,4.03,3.371,3.018,2.619,2.172,1.761,1.5,1.339,1.099,0.9169,0.7911,0.6835,0.5932,0.5134,0.4362,0.3867,0.333,0.2969,0.2599,0.2266,0.1998,0.1759,0.1593,0.142,0.1209,0.101,0.08923,0.07933,0.07066,0.06291,0.05622,0.0522,0.04432,0.04015,0.03556,0.03289,0.03101,0.02887,0.02652,0.02444,0.02302,0.02117,0.01898,0.01755,0.01693,0.01595,0.01473,0.01415,0.01344,0.01314,0.01247,0.01227,0.01178,0.01136,0.01102,0.01063,0.01054,0.01036,0.01024,0.01015,0.01011,0.009961,0.01002,0.01002,0.009935,0.009949,0.009844,0.009809,0.009782,0.009762,0.009763,0.009847,0.009679,0.009701,0.009671,0.009619,0.009653,0.009639,0.009635,0.009756,0.009642,0.009588,0.009571,0.009566,0.00961,0.009579,0.009606,0.009595,0.00957,0.009619,0.009584,0.009597,0.009649,0.009646,0.009552,0.009697,0.009551,0.009569,0.009658,0.009706,0.00959,0.00956,0.009573,0.009613,0.00959,0.009546,0.009574,0.00955,0.009558,0.00955,0.009593,0.009567,0.009551,0.009551,0.009576,0.00955,0.009544,0.00955,0.009555,0.009552,0.009573,0.009565,0.00958,0.009565,0.009551,0.009544,0.009544,0.009731,0.009734,0.009797,0.009752,0.009662,0.009581,0.009624,0.00957,0.009624,0.009696,0.009606,0.009653,0.009635,0.009619,0.009608,0.00959,0.009583,0.009565,0.009585,0.009599,0.009616,0.009602,0.009587,0.009555,0.009552,0.009556,0.009568,0.009574,0.009546,0.009545,0.009556,0.009574,0.009713]},{"name":"single examples (SGD)","x":[0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1,1.125,1.15,1.175,1.2,1.225,1.25,1.275,1.3,1.325,1.35,1.375,1.4,1.425,1.45,1.475,1.5,1.525,1.55,1.575,1.6,1.625,1.65,1.675,1.7,1.725,1.75,1.775,1.8,1.825,1.85,1.875,1.9,1.925,1.95,1.975,2,2.025,2.05,2.075,2.1,2.125,2.15,2.175,2.2,2.225,2.25,2.275,2.3,2.325,2.35,2.375,2.4,2.425,2.45,2.475,2.5,2.525,2.55,2.575,2.6,2.625,2.65,2.675,2.7,2.725,2.75,2.775,2.8,2.825,2.85,2.875,2.9,2.925,2.95,2.975,3,3.025,3.05,3.075,3.1,3.125,3.15,3.175,3.2,3.225,3.25,3.275,3.3,3.325,3.35,3.375,3.4,3.425,3.45,3.475,3.5,3.525,3.55,3.575,3.6,3.625,3.65,3.675,3.7,3.725,3.75,3.775,3.8,3.825,3.85,3.875,3.9,3.925,3.95,3.975,4,4.025,4.05,4.075,4.1,4.125,4.15,4.175,4.2,4.225,4.25,4.275,4.3,4.325,4.35,4.375,4.4,4.425,4.45,4.475,4.5,4.525,4.55,4.575,4.6,4.625,4.65,4.675,4.7,4.725,4.75,4.775,4.8,4.825,4.85,4.875,4.9,4.925,4.95,4.975,5],"y":[4.958,0.09161,0.0112,0.01098,0.01371,0.009727,0.01082,0.0119,0.01041,0.01034,0.01008,0.01284,0.01176,0.01149,0.01119,0.01119,0.0119,0.01205,0.01053,0.01066,0.01106,0.01071,0.009565,0.01251,0.01339,0.01138,0.01006,0.01194,0.009633,0.01016,0.01117,0.01395,0.01162,0.01038,0.01149,0.009795,0.01133,0.01105,0.01555,0.01099,0.01406,0.01279,0.01294,0.01125,0.009567,0.01064,0.0112,0.01763,0.01376,0.01093,0.009838,0.01007,0.009857,0.01027,0.01247,0.009737,0.01102,0.01087,0.009746,0.009825,0.01258,0.01254,0.01013,0.01162,0.01216,0.01003,0.009758,0.01427,0.00979,0.01038,0.01519,0.01062,0.009852,0.01119,0.01079,0.01165,0.0136,0.01043,0.0111,0.01461,0.01289,0.01129,0.01269,0.01153,0.01097,0.0173,0.01169,0.00967,0.01066,0.01056,0.01131,0.00996,0.01024,0.01014,0.009815,0.0142,0.01,0.01146,0.014,0.01043,0.01162,0.01136,0.01089,0.01042,0.01022,0.01142,0.01072,0.01036,0.01066,0.01312,0.01177,0.01011,0.01057,0.01079,0.01135,0.01774,0.01439,0.01156,0.009984,0.01025,0.01675,0.01317,0.01032,0.009805,0.01751,0.00994,0.01088,0.01001,0.009905,0.014,0.01015,0.01178,0.01334,0.01171,0.01172,0.01282,0.01066,0.01215,0.01006,0.009573,0.01179,0.009822,0.01067,0.01038,0.01002,0.01254,0.01117,0.01384,0.009759,0.01019,0.01038,0.01088,0.01424,0.0189,0.0108,0.009597,0.009992,0.01089,0.0153,0.01075,0.01354,0.009626,0.01107,0.009869,0.01168,0.01291,0.01215,0.0115,0.009804,0.009654,0.01615,0.01296,0.01188,0.01076,0.01225,0.01124,0.009706,0.01056,0.009762,0.01125,0.009686,0.009837,0.009818,0.0143,0.01011,0.01026,0.01075,0.01032,0.009818,0.01362,0.01251,0.01102,0.011,0.01144,0.01099,0.01152,0.01073,0.009929,0.0136,0.01034,0.01295]}],"xLabel":"epochs (passes over the 1,000 training examples)","yLabel":"full-data MSE (log scale)","hlines":[{"y":0.01,"label":"noise floor (σ² = 0.01)"}]} },
      { type: 'check', question: { type: 'fill', q: 'A dataset has 60,000 examples and the batch size is 100. How many gradient steps are in one epoch?', answer: ['600'], explain: '60,000 / 100 = 600 mini-batches per pass over the data.' } },
      { type: 'text', title: "Checking a gradient with finite differences", md: `Gradient descent is only as good as its gradients. When you derive and code a gradient by hand, a cheap safety net is to compare it with a **numerical** estimate: nudge one parameter a tiny amount in each direction and measure how the loss changes.

$$ ∂L/∂wⱼ ≈ ( L(w + h·eⱼ) − L(w − h·eⱼ) ) / (2h) $$
:::where
h — a small step, such as 1e-5
eⱼ — the vector with 1 in position j and 0 elsewhere, so only parameter j is nudged
L(w ± h·eⱼ) — the loss after nudging parameter j up or down
2h — the distance between the two nudged points; the ratio is the slope of the line through them
:::

This **central difference** has an error proportional to h², far better than the one-sided version (L(w + h·eⱼ) − L(w)) / h, whose error is proportional to h. Too small an h runs into floating-point rounding, so values from 1e-4 to 1e-6 are usual in double precision. Compare the two gradients with a **relative error**, |a − b| / max(|a|, |b|): around 1e-7 or less means the analytic gradient is right, while something like 1e-2 means a bug.

Each numerical partial derivative costs two evaluations of the loss, so checking a network with millions of parameters this way would be far too slow for training. That is why frameworks compute exact gradients with [backpropagation](#/concept/backpropagation) and [automatic differentiation](#/concept/autograd), and why gradient checking is a debugging tool, run on a few parameters of a small model.` },
      { type: 'code', title: "Gradient checking a logistic-regression loss", code: `import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 3))
y = (rng.random(50) < 0.5).astype(float)

def loss(w):
    p = 1 / (1 + np.exp(-X @ w))
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def grad(w):                                       # the analytic gradient: X^T (p - y) / n
    p = 1 / (1 + np.exp(-X @ w))
    return X.T @ (p - y) / len(y)

def numeric_grad(w, h):
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (loss(w + e) - loss(w - e)) / (2 * h)
    return g

w = np.array([0.5, -1.0, 2.0])
exact = grad(w)
approx = numeric_grad(w, 1e-5)
print("analytic gradient: ", exact.round(6))
print("numerical, h=1e-5: ", approx.round(6))
rel = np.abs(approx - exact).max() / np.abs(exact).max()
print(f"relative error {rel:.1e} -> gradient correct: {rel < 1e-7}")

err1 = np.abs(numeric_grad(w, 1e-2) - exact).max()
err2 = np.abs(numeric_grad(w, 5e-3) - exact).max()
print(f"error with h = 0.01: {err1:.2e}   with h = 0.005: {err2:.2e}   ratio {err1 / err2:.2f}")
print("error ratio about 4:", 3.5 < err1 / err2 < 4.5)`, after: `The analytic and numerical gradients agree to many decimal places, and halving h divides the error by about 4: the signature of an error proportional to h².`, predict: {"type":"mcq","q":"You halve the step h in a central finite difference. Ignoring rounding, what happens to its approximation error?","options":["It stays the same","It halves","It drops to about a quarter","It doubles"],"answer":2,"explain":"The central difference's error is proportional to h², so halving h divides it by about 4.","expect":["error ratio about 4: True"]} }
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
      { type: 'mcq', q: 'On L(w) = w², gradient descent with learning rate 1.0 starting from w = 1 will…', options: ['Converge in one step', 'Oscillate between 1 and −1 forever', 'Diverge to infinity', 'Stay at 1'], answer: 1, explain: 'Gradient 2w: w ← w − 2w = −w. It flips sign forever without approaching 0.' },
      {"type":"mcq","q":"On L(w₁, w₂) = w₁² + 10·w₂², plain gradient descent with a learning rate just below its stability limit tends to…","options":["Go straight to the minimum","Zig-zag across the steep direction while crawling along the shallow one","Diverge in the shallow direction","Stop after one step"],"answer":1,"explain":"The learning rate is limited by the steep w₂ direction (η < 2/20 = 0.1), so w₂ oscillates while w₁ shrinks only slowly."},
      {"type":"tf","q":"Standardising the input features can make gradient descent converge in far fewer steps.","answer":true,"explain":"Features on very different scales create an elongated loss bowl (a large condition number). Standardising makes it rounder, so one learning rate suits every direction."},
      {"type":"fill","q":"For a quadratic loss whose largest curvature (Hessian eigenvalue) is 50, what is the largest learning rate for which gradient descent is stable? (decimal)","answer":["0.04",".04","0.040","1/25"],"explain":"Gradient descent on a quadratic is stable when η < 2 / λ_max = 2 / 50 = 0.04."}
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
      { type: 'chart', title: "Least squares minimises the squared vertical distances", after: `Each dashed segment is one residual. Least squares picks the line that makes the sum of their squares as small as possible; any other line, tilted or shifted, would make that sum larger. Note that the distances are vertical (errors in y), not perpendicular to the line: the model predicts y from x, it does not treat the two symmetrically.`, chart: {"kind":"line","points":false,"series":[{"name":"data","x":[0.015,0.856,0.941,1.137,1.597,2.368,2.842,2.927,2.984,3.14,3.912,4.306,4.331,4.791,5.167,5.822,5.868,6.485,6.962,7.346,7.378,8.013,8.917,9.563,9.735],"y":[2.161,4.507,1.929,2.117,3.981,2.331,3.424,4.901,4.584,4.122,5.434,1.551,6.191,4.181,3.632,6.489,7.035,6.155,5.778,7.408,7.339,9.597,9.531,9.383,10.622],"scatter":true},{"name":"least-squares line ŷ = 1.61 + 0.80x","x":[0,10],"y":[1.611,9.626]},{"name":"residuals yᵢ − ŷᵢ","x":[0.015,0.015,null,0.856,0.856,null,0.941,0.941,null,1.137,1.137,null,1.597,1.597,null,2.368,2.368,null,2.842,2.842,null,2.927,2.927,null,2.984,2.984,null,3.14,3.14,null,3.912,3.912,null,4.306,4.306,null,4.331,4.331,null,4.791,4.791,null,5.167,5.167,null,5.822,5.822,null,5.868,5.868,null,6.485,6.485,null,6.962,6.962,null,7.346,7.346,null,7.378,7.378,null,8.013,8.013,null,8.917,8.917,null,9.563,9.563,null,9.735,9.735,null],"y":[2.161,1.623,null,4.507,2.298,null,1.929,2.366,null,2.117,2.523,null,3.981,2.892,null,2.331,3.509,null,3.424,3.889,null,4.901,3.957,null,4.584,4.003,null,4.122,4.128,null,5.434,4.747,null,1.551,5.063,null,6.191,5.083,null,4.181,5.451,null,3.632,5.753,null,6.489,6.277,null,7.035,6.314,null,6.155,6.809,null,5.778,7.191,null,7.408,7.499,null,7.339,7.525,null,9.597,8.033,null,9.531,8.758,null,9.383,9.275,null,10.622,9.413,null],"dash":true}],"xLabel":"feature x","yLabel":"target y"} },
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
      { type: 'text', title: "Linear models with many features", md: `With one feature, a linear model is a line; with two, a plane; with many, a flat "hyperplane" in a high-dimensional space. Written out for p features:

$$ ŷ = w₁·x₁ + w₂·x₂ + … + w_p·x_p + b $$
:::where
ŷ — the prediction
x₁ … x_p — the p input features of one example
w₁ … w_p — the weights (coefficients), one per feature; scikit-learn stores them in \`coef_\`
b — the intercept (bias), stored in \`intercept_\`
p — the number of features
:::

A straight line sounds restrictive, and with one or two features it is. With many features, linear models become remarkably powerful: if there are at least as many features as training examples, a linear model can fit **any** training targets perfectly, noise included. That is exactly when it overfits, and why the next lesson, [regularization](#/concept/regularization), exists.

:::note The trailing underscore
In scikit-learn, attributes that are *learned from the data* end with an underscore: \`coef_\`, \`intercept_\`, \`classes_\`, \`feature_importances_\`. Settings you choose yourself, such as \`alpha\`, do not. You can only read the underscore attributes after calling \`fit\`.
:::` },
      { type: 'code', title: "When ordinary least squares overfits", md: `Ordinary least squares (\`LinearRegression\`) has no setting that controls its complexity. Here it gets 80 features but only 70 training examples, and only 10 of the features really matter.`, code: `from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=150, n_features=80, n_informative=10, noise=25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=70, random_state=0)

ols = LinearRegression().fit(X_train, y_train)
print("weights learned:", ols.coef_.shape[0])
print(f"training R2: {ols.score(X_train, y_train):.3f}")
print(f"test R2:     {ols.score(X_test, y_test):.3f}")`, after: `With more features than examples, the fit on the training set is perfect, but every one of the 80 weights also absorbs some of the noise, so the test score is clearly lower. The model has no way to know that 70 of its features are useless. Regularized versions (ridge and lasso) fix this by preferring small or zero weights.` },
      { type: 'text', title: "Reading coefficients without fooling yourself", md: `Coefficients are tempting to interpret ("each extra bedroom adds €12,000"), but be careful:

- **Scale matters.** A weight's size depends on the feature's units: price per metre is a thousand times larger than price per millimetre. Standardize the features before comparing weights.
- **Correlated features share credit.** If two features carry the same information (size in m² and number of rooms), the model can split the weight between them arbitrarily, and either can even get the "wrong" sign.
- **A weight describes the model, not the world.** It says how the *prediction* changes when one feature changes and the others stay fixed. That is not a causal effect: see [Causal Inference](#/concept/causal-inference).` },
      { type: 'text', title: "Checking the assumptions: residual plots", md: `A linear model can be fitted to any data; whether its answers mean anything depends on a few assumptions. Plot the residuals (y − ŷ) against the fitted values and look for patterns:

| What you see | What it means | What to do |
| --- | --- | --- |
| a curve (a U or an arch) | the relationship is not linear | add transformed features (x², log x) or use a non-linear model |
| a funnel: the spread grows with ŷ | the noise is not constant (heteroscedasticity) | model log(y) or use weighted least squares; trust the standard errors less |
| a few points far from the rest | outliers or high-leverage points | check them for data errors; consider a robust loss such as Huber |
| clusters, or trends over time or by group | the errors are not independent | add the missing feature (time, group) or use a model for grouped or time-series data |
| a shapeless horizontal band | the assumptions look fine | carry on |

**Leverage** deserves a word: a point with an unusual x value, far from the others, pulls the line towards itself much harder than a point in the middle. One extreme point can tilt the whole fit, so look at points that are unusual in x as well as in y.

For predictions alone, mild violations are often harmless. For **inference** (standard errors, p-values and confidence intervals of the coefficients) they matter a lot, because those calculations assume independent errors with constant variance.` },
      { type: 'chart', title: "Residuals against fitted values", md: `The data follows a curve. Two linear models, one with a straight line and one with an added x² feature:`, after: `The straight-line model's residuals form a U: positive at both ends and negative in the middle, so the errors are predictable from the fitted value and the model is missing structure. With the x² feature the residuals are a shapeless band around 0, which is what a well-specified model leaves behind.`, chart: {"kind":"scatter","series":[{"name":"straight-line model","x":[-0.446,-0.24,-0.226,-0.029,0.024,0.229,0.238,0.363,0.507,0.58,0.664,0.684,0.693,0.739,0.742,0.793,0.794,1.048,1.099,1.186,1.292,1.323,1.457,1.616,1.65,1.655,1.779,1.785,1.857,1.909,1.936,1.941,1.979,2.496,2.497,2.504,2.574,2.589,2.766,2.796,2.85,2.906,2.917,2.968,3.014,3.021,3.047,3.076,3.192,3.301,3.34,3.408,3.698,3.701,3.767,3.885,3.933,3.947,4.158,4.175,4.176,4.194,4.228,4.368,4.45,4.469,4.617,4.662,4.802,4.849,4.858,4.907,4.987,5.232,5.306,5.308,5.472,5.582,5.587,5.621],"y":[4.604,3.334,3.666,1.614,1.993,1.066,2.435,1.611,0.69,0.439,0.817,0.063,-0.095,0.658,1.84,0.113,-0.076,-0.919,-1.103,-0.129,-0.095,-0.644,-0.634,-0.956,-0.979,-1.984,-1.469,-1.134,-1.735,-1.595,-1.823,-1.535,-0.852,-1.848,-1.696,-1.117,-1.229,-0.6,-2.871,-1.099,-1.894,-1.512,-1.087,-0.925,-0.937,-1.465,-0.582,-1.706,-1.564,-0.933,-1.718,-0.039,-0.454,0.243,-1.005,-1.074,-0.681,-0.322,-0.806,-0.199,-0.436,0.573,-0.739,0.527,-0.335,-0.49,-0.046,-0.241,0.867,1.505,1.006,1.397,2.194,2.009,2.168,1.892,1.565,3.318,1.825,3.376]},{"name":"model with an x² term","x":[3.175,2.725,2.696,2.312,2.214,1.868,1.855,1.667,1.47,1.38,1.281,1.259,1.249,1.2,1.197,1.146,1.145,0.928,0.893,0.839,0.784,0.771,0.722,0.69,0.687,0.686,0.685,0.685,0.692,0.7,0.706,0.707,0.716,0.993,0.993,0.999,1.059,1.072,1.249,1.283,1.345,1.413,1.427,1.492,1.552,1.561,1.597,1.638,1.809,1.984,2.048,2.168,2.724,2.73,2.869,3.13,3.241,3.273,3.79,3.833,3.835,3.882,3.971,4.349,4.582,4.635,5.074,5.21,5.654,5.806,5.837,6.001,6.271,7.141,7.417,7.426,8.058,8.497,8.52,8.659],"y":[0.984,0.369,0.744,-0.727,-0.197,-0.573,0.818,0.307,-0.272,-0.362,0.199,-0.512,-0.65,0.197,1.386,-0.241,-0.426,-0.798,-0.897,0.218,0.413,-0.092,0.101,-0.03,-0.017,-1.016,-0.375,-0.033,-0.57,-0.387,-0.592,-0.301,0.411,-0.345,-0.193,0.388,0.286,0.917,-1.354,0.415,-0.389,-0.018,0.404,0.551,0.525,-0.005,0.868,-0.268,-0.182,0.384,-0.426,1.202,0.52,1.214,-0.107,-0.319,0.012,0.352,-0.438,0.143,-0.095,0.885,-0.482,0.546,-0.466,-0.656,-0.503,-0.79,0.015,0.547,0.027,0.304,0.911,0.099,0.056,-0.226,-1.021,0.402,-1.107,0.339]}],"xLabel":"fitted value ŷ","yLabel":"residual y − ŷ"} },
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
      { type: 'mcq', q: 'When does gradient descent beat the closed-form solution for linear regression?', options: ['Never', 'With huge numbers of examples or features, or streaming data', 'Only with one feature', 'When R² is low'], answer: 1, explain: 'The closed form costs O(n·d²) and needs all data at once; SGD scales to massive or streaming data.' },
      {"type":"mcq","q":"A linear regression has 200 features and is trained on 150 examples. What is most likely to happen?","options":["It underfits because a linear model is too simple","It fits the training data (almost) perfectly and generalizes worse","It ignores half of the features automatically","It cannot be trained at all"],"answer":1,"explain":"With at least as many features as examples, least squares can match every training target, noise included, which is classic overfitting. Regularization (ridge or lasso) is the usual remedy."},
      {"type":"mcq","q":"The residual plot of a linear regression shows a clear U shape. What does it suggest?","options":["The model is perfect","The relationship is curved: add a feature such as x², or use a non-linear model","The noise variance is constant","There are too many features"],"answer":1,"explain":"Systematic structure in the residuals means the model is missing a pattern; a U shape is the signature of an unmodelled curve."},
      {"type":"tf","q":"Least squares minimises the sum of the perpendicular distances from the points to the line.","answer":false,"explain":"It minimises the squared vertical distances, the errors in y, because the model predicts y from x."}
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
      { title: 'Google ML Crash Course — Linear regression', url: 'https://developers.google.com/machine-learning/crash-course/linear-regression' },
      { title: "scikit-learn — Linear models", url: "https://scikit-learn.org/stable/modules/linear_model.html" },
      { title: "Andreas Müller & Sarah Guido — Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
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
      { type: 'chart', title: "Log loss punishes confident mistakes", after: `When the true label is 1, predicting p = 0.9 costs only 0.105, but predicting p = 0.01 costs 4.6, and the loss grows without limit as p approaches 0. A model trained with log loss is therefore pushed hard to avoid confident wrong answers.`, chart: {"kind":"line","x":[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99],"points":false,"series":[{"name":"true label 1:  −log p","y":[4.605,3.912,3.507,3.219,2.996,2.813,2.659,2.526,2.408,2.303,2.207,2.12,2.04,1.966,1.897,1.833,1.772,1.715,1.661,1.609,1.561,1.514,1.47,1.427,1.386,1.347,1.309,1.273,1.238,1.204,1.171,1.139,1.109,1.079,1.05,1.022,0.994,0.968,0.942,0.916,0.892,0.868,0.844,0.821,0.799,0.777,0.755,0.734,0.713,0.693,0.673,0.654,0.635,0.616,0.598,0.58,0.562,0.545,0.528,0.511,0.494,0.478,0.462,0.446,0.431,0.416,0.4,0.386,0.371,0.357,0.342,0.329,0.315,0.301,0.288,0.274,0.261,0.248,0.236,0.223,0.211,0.198,0.186,0.174,0.163,0.151,0.139,0.128,0.117,0.105,0.094,0.083,0.073,0.062,0.051,0.041,0.03,0.02,0.01]},{"name":"true label 0:  −log(1 − p)","y":[0.01,0.02,0.03,0.041,0.051,0.062,0.073,0.083,0.094,0.105,0.117,0.128,0.139,0.151,0.163,0.174,0.186,0.198,0.211,0.223,0.236,0.248,0.261,0.274,0.288,0.301,0.315,0.329,0.342,0.357,0.371,0.386,0.4,0.416,0.431,0.446,0.462,0.478,0.494,0.511,0.528,0.545,0.562,0.58,0.598,0.616,0.635,0.654,0.673,0.693,0.713,0.734,0.755,0.777,0.799,0.821,0.844,0.868,0.892,0.916,0.942,0.968,0.994,1.022,1.05,1.079,1.109,1.139,1.171,1.204,1.238,1.273,1.309,1.347,1.386,1.427,1.47,1.514,1.561,1.609,1.661,1.715,1.772,1.833,1.897,1.966,2.04,2.12,2.207,2.303,2.408,2.526,2.659,2.813,2.996,3.219,3.507,3.912,4.605]}],"xLabel":"predicted probability p of class 1","yLabel":"loss for one example"} },
      { type: 'text', title: "Linear classifiers in scikit-learn", md: `A linear classifier predicts the positive class when a weighted sum of the features is above zero:

$$ ŷ = 1  if  w₁·x₁ + … + w_p·x_p + b > 0,  else  0 $$
:::where
ŷ — the predicted class (1 = positive, 0 = negative)
x₁ … x_p — the features of one example
w₁ … w_p, b — the learned weights and intercept
:::

The boundary between the classes is therefore a line (two features), a plane (three) or a hyperplane (more). The two most common linear classifiers differ only in the loss they minimise:

| Model | Loss | scikit-learn class |
| --- | --- | --- |
| Logistic regression | log loss (this lesson) | \`LogisticRegression\` |
| Linear support vector machine | hinge loss (see [SVMs](#/concept/svm)) | \`LinearSVC\` |

Both use L2 regularization by default, controlled by **C**, the *inverse* of the regularization strength:

- **Large C** = weak regularization: the model tries hard to classify every training point correctly.
- **Small C** = strong regularization: the model prefers small weights and settles for getting the majority right.` },
      { type: 'code', title: "C in action on the breast-cancer data", code: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

train_scores = []
for C in [0.001, 0.01, 1, 100]:
    model = LogisticRegression(C=C, max_iter=10_000).fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    print(f"C={C:<6} train {train_scores[-1]:.3f}   test {model.score(X_test, y_test):.3f}"
          f"   total weight size {np.abs(model.coef_).sum():6.1f}")
print("training accuracy rises with C:", all(a < b for a, b in zip(train_scores, train_scores[1:])))`, after: `Training accuracy rises steadily with C, because a weakly regularized model can bend to fit more training points. Test accuracy rises too at first, peaks around C = 1, then falls when C = 100 lets the weights grow large enough to fit noise. The weights' total size grows by a factor of almost 100 along the way.`, predict: {"type":"mcq","q":"Raising C from 1 to 100 weakens the regularization. What happens to the **training** accuracy?","options":["It goes up","It goes down","It stays exactly the same","It drops to chance level"],"answer":0,"explain":"Weaker regularization lets the model fit the training data more closely, so training accuracy goes up (0.988 to 0.998 here), while test accuracy goes down (0.986 to 0.944): the beginning of overfitting.","expect":["training accuracy rises with C: True"]} },
      { type: 'chart', title: "Accuracy against C", after: `Moving right means less regularization. Training accuracy climbs steadily; test accuracy peaks around C = 1 and then falls: the same complexity trade-off as k in k-NN, seen through a different setting.`, chart: {"kind":"line","x":[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100],"xLog":true,"series":[{"name":"training","y":[0.892,0.937,0.958,0.981,0.984,0.986,0.988,0.988,0.991,0.995,0.998]},{"name":"test","y":[0.916,0.944,0.937,0.951,0.979,0.986,0.986,0.986,0.972,0.972,0.944]}],"xLabel":"C (inverse regularization strength, log scale)","yLabel":"accuracy"} },
      { type: 'text', title: "Many classes: one-vs-rest", md: `Many binary classifiers, such as the linear SVM, extend to K classes with **one-vs-rest**: train K separate classifiers, each separating one class from all the others. To classify a new point, compute all K scores and pick the class whose classifier is most confident (the highest score).

Logistic regression can instead fit all classes at once with the [softmax](#/concept/activation-functions), which is scikit-learn's default for multi-class problems. In both cases there is one weight vector per class, so \`coef_\` has shape (K, p).` },
      { type: 'text', title: "Strengths and weaknesses of linear models", md: `| Strengths | Weaknesses |
| --- | --- |
| very fast to train and to predict, even on millions of examples | in low-dimensional data with curved boundaries, other models do better |
| work well with very many features and with sparse data such as text | features must be scaled for regularization to treat them fairly |
| a prediction is a weighted sum, so it is easy to explain | coefficients of correlated features are hard to interpret |
| a strong first model to try on almost any problem | interactions (x₁·x₂) and curves must be added as features by hand |

For very large datasets, use a solver designed for them (\`solver="saga"\`) or \`SGDClassifier\`, which learns from small batches with [stochastic gradient descent](#/concept/gradient-descent).` },
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
      { type: 'tf', q: 'The gradient of log loss for logistic regression has the form (1/n)·Xᵀ(p − y).', answer: true, explain: 'The same "features × error" form as linear regression.' },
      {"type":"mcq","q":"In scikit-learn's LogisticRegression, what does a **smaller** C do?","options":["Weakens regularization, so the model fits the training data more closely","Strengthens regularization, so the weights are pushed towards zero","Changes the loss to the hinge loss","Adds more classes"],"answer":1,"explain":"C is the inverse of the regularization strength: smaller C means a stronger penalty on large weights and a simpler model."},
      {"type":"mcq","q":"A one-vs-rest classifier is trained on a problem with 4 classes. How many binary classifiers does it contain?","options":["1","3","4","6"],"answer":2,"explain":"One classifier per class, each separating that class from all the others; the class with the highest score wins. (One-vs-one would need 4·3/2 = 6.)"}
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
      { title: 'scikit-learn — Logistic regression', url: 'https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression' },
      { title: "Stanford CS229 — Machine Learning lecture notes", url: "https://cs229.stanford.edu/main_notes.pdf" },
      { title: "Andreas Müller & Sarah Guido — Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
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
      { type: 'chart', title: "Training and test error against model complexity", md: `Forty noisy points from a sine wave, fitted with polynomials of increasing degree. This is the most important picture in machine learning:`, after: `Training error only ever goes down as the model gets more flexible. Test error falls while the model is still underfitting (low degrees cannot bend like a sine), bottoms out at the sweet spot, then climbs as the polynomial starts fitting the noise. The dashed line is the noise level (σ² = 0.0625): no model can do better on new data.`, chart: {"kind":"line","x":[0,1,2,3,4,5,6,7,8,9,10,11,12],"yLog":true,"series":[{"name":"training error","y":[0.4403,0.1492,0.1488,0.0291,0.0291,0.0241,0.0236,0.0235,0.0234,0.0235,0.0235,0.0235,0.0235]},{"name":"test error","y":[0.5684,0.2702,0.2694,0.0843,0.0841,0.0809,0.0824,0.0824,0.0825,0.0824,0.0824,0.0825,0.0824]}],"xLabel":"polynomial degree (model complexity)","yLabel":"mean squared error (log scale)","hlines":[{"y":0.0625,"label":"noise level"}],"vlines":[{"x":5,"label":"sweet spot"}]} },
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
      { type: 'demo', title: 'Interactive: bias and variance across many training sets', demo: 'bias-variance' },
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
      { type: 'chart', title: "Learning curves for a simple and a flexible model", after: `Dashed lines are training error, solid lines test error. The **simple** model (a straight line) converges quickly to a high error: more data cannot fix high bias. The **flexible** model is terrible with little data (high variance) but keeps improving as data grows, until it beats the simple model by a wide margin. That is the practical meaning of "more data lets you afford more complex models".`, chart: {"kind":"line","x":[10,15,20,30,45,70,100,150,250,400],"xLog":true,"yLog":true,"yMax":2,"series":[{"name":"simple (degree 1) train","y":[0.2745,0.3575,0.3206,0.2778,0.275,0.2779,0.269,0.2977,0.281,0.2808],"dash":true},{"name":"simple (degree 1) test","y":[0.3049,0.2699,0.2876,0.2784,0.2994,0.2827,0.2842,0.277,0.2734,0.2698],"dash":false},{"name":"flexible (degree 9) train","y":[0.0158,0.0193,0.0392,0.0454,0.0567,0.0604,0.0674,0.0756,0.0742,0.0702],"dash":true},{"name":"flexible (degree 9) test","y":[0.4817,0.139,0.0931,0.0971,0.0843,0.0781,0.078,0.0756,0.0733,0.0732],"dash":false}],"xLabel":"number of training examples (log scale)","yLabel":"mean squared error (log scale)"} },
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
      { title: 'scikit-learn — Underfitting vs. overfitting example', url: 'https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html' },
      { title: "Hastie, Tibshirani & Friedman — The Elements of Statistical Learning (free book)", url: "https://hastie.su.domains/ElemStatLearn/" }
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
      { type: 'code', title: "Ridge and lasso in scikit-learn", md: `The same data as in the linear-regression lesson: 80 features, 70 training examples, and only 10 features that matter. In scikit-learn the regularization strength is called **alpha** (λ in the formulas above). For ridge the code prints the overall size of the weights; for lasso, how many weights are not exactly zero.`, code: `import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

X, y = make_regression(n_samples=150, n_features=80, n_informative=10, noise=25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=70, random_state=0)

ols = LinearRegression().fit(X_train, y_train)
print(f"least squares        train {ols.score(X_train, y_train):.3f}  test {ols.score(X_test, y_test):.3f}")
for alpha in [0.1, 1, 10, 100]:
    m = Ridge(alpha=alpha).fit(X_train, y_train)
    print(f"ridge alpha={alpha:<6} train {m.score(X_train, y_train):.3f}  test {m.score(X_test, y_test):.3f}  weight size {np.linalg.norm(m.coef_):6.1f}")
used = []
for alpha in [0.1, 1, 5, 20]:
    m = Lasso(alpha=alpha, max_iter=100_000).fit(X_train, y_train)
    used.append(int(np.sum(m.coef_ != 0)))
    print(f"lasso alpha={alpha:<6} train {m.score(X_train, y_train):.3f}  test {m.score(X_test, y_test):.3f}  features used {used[-1]:3d} of 80")
print("fewer features as alpha grows:", all(a > b for a, b in zip(used, used[1:])))`, after: `- **Ridge** shrinks all weights a little. Moderate alpha helps slightly here; large alpha shrinks the useful weights too and the test score drops: the model now underfits.
- **Lasso** is the clear winner on this data, because only a few features truly matter. As alpha grows it switches whole features off: with alpha = 5 it keeps about 18 features and scores far better than least squares. With alpha = 20 it keeps exactly as many features as are informative, but shrinks them so much that the score falls again.
- There is always a sweet spot. You find it with [cross-validation](#/concept/cross-validation), never with the test set.`, predict: {"type":"mcq","q":"As lasso's alpha increases from 0.1 to 20, what happens to the number of features it uses (weights that are not exactly zero)?","options":["It stays at 80","It goes down","It goes up","It first goes down, then back up to 80"],"answer":1,"explain":"A stronger L1 penalty pushes more weights to exactly zero, so lasso uses fewer and fewer features: from 66 at alpha = 0.1 down to 10 at alpha = 20.","expect":["fewer features as alpha grows: True"]} },
      { type: 'chart', title: "Test R² against regularization strength", after: `Both penalties have a sweet spot. Lasso's is much higher on this data because only 10 of the 80 features matter and it can switch the other 70 off; ridge can only shrink them. Too much of either penalty underfits and the score collapses.`, chart: {"kind":"line","x":[0.01,0.03,0.1,0.3,1,3,10,30,100],"xLog":true,"series":[{"name":"ridge","y":[0.859,0.861,0.867,0.876,0.88,0.855,0.763,0.621,0.433]},{"name":"lasso","y":[0.922,0.952,0.964,0.975,0.98,0.981,0.934,0.541,0.143]}],"xLabel":"alpha (log scale)","yLabel":"test R²","hlines":[{"y":0.858,"label":"least squares"}],"yMin":0} },
      { type: 'chart', title: "How many features lasso keeps", after: `As alpha grows, lasso sets more and more weights exactly to zero: automatic feature selection.`, chart: {"kind":"line","x":[0.01,0.03,0.1,0.3,1,3,10,30,100],"xLog":true,"series":[{"name":"features with a non-zero weight","y":[76,71,66,56,46,24,15,8,2]}],"xLabel":"alpha (log scale)","yLabel":"features used (of 80)","hlines":[{"y":10,"label":"truly informative features"}],"yMin":0} },
      { type: 'text', title: "Regularization matters most when data is scarce", md: `Plot training and test scores against the number of training examples and you get a **learning curve**. For least squares on the data above:

- With few examples, the training score is perfect and the test score is poor: the model memorizes.
- As examples are added, memorizing becomes impossible. The training score drops a little, the test score rises, and the two curves meet.
- With enough data, plain least squares does about as well as its regularized versions. **Regularization is most valuable when data is scarce relative to the number of features**, which is common in practice.` },
      { type: 'code', title: "A learning curve for least squares and lasso", code: `from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso

X, y = make_regression(n_samples=1140, n_features=80, n_informative=10, noise=25, random_state=1)
X_pool, y_pool, X_test, y_test = X[:640], y[:640], X[640:], y[640:]
print(" n     least squares (train / test)    lasso alpha=1 (train / test)")
for n in [40, 80, 160, 320, 640]:
    a = LinearRegression().fit(X_pool[:n], y_pool[:n])
    b = Lasso(alpha=1, max_iter=100_000).fit(X_pool[:n], y_pool[:n])
    print(f"{n:4d}        {a.score(X_pool[:n], y_pool[:n]):.3f} / {a.score(X_test, y_test):.3f}"
          f"                  {b.score(X_pool[:n], y_pool[:n]):.3f} / {b.score(X_test, y_test):.3f}")`, after: `With 40 examples (fewer than the 80 features), least squares cannot tell signal from noise, while lasso already does far better. By 320 examples the gap has almost closed: given enough data, the simple unregularized model catches up.` },
      { type: 'text', title: "Elastic net, and which penalty to choose", md: `**Elastic net** (\`ElasticNet\` in scikit-learn) adds both penalties: an L1 part that can switch features off and an L2 part that keeps groups of correlated features together. Its \`l1_ratio\` setting blends the two.

| Situation | Good first choice |
| --- | --- |
| You expect most features to matter a little | **Ridge** (L2) |
| You expect only a few features to matter, or want a model that is easy to read | **Lasso** (L1) |
| Many correlated features and you still want sparsity | **Elastic net** |

:::warn Scale before you penalize
Penalties treat all weights alike, so a feature measured in large units (salary in euros) needs only a tiny weight and is hardly penalized, while one measured in small units is penalized heavily. Standardize the features first, inside a [pipeline](#/concept/sklearn-workflow).
:::` },
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
      { type: 'mcq', q: 'Why standardize features before applying L1/L2 regularization?', options: ['It makes training deterministic', 'So the penalty treats features equally instead of depending on their units', 'Regularization only works on integers', 'It removes outliers'], answer: 1, explain: 'A feature measured in millimetres needs a tiny weight and would barely be penalized compared with one in kilometres.' },
      {"type":"mcq","q":"You train lasso and many coefficients come out **exactly** zero. What does that mean?","options":["The training failed","Those features are ignored by the model","The features have negative effects","The data must be standardized again"],"answer":1,"explain":"The L1 penalty drives some weights exactly to zero, which removes those features from the model: built-in feature selection."},
      {"type":"tf","q":"With a very large training set, an unregularized linear model usually performs about as well as a regularized one.","answer":true,"explain":"Regularization protects against fitting noise, which matters most when data is scarce relative to the number of features. With plenty of data the curves converge."}
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
      { title: 'scikit-learn — Ridge, Lasso and Elastic-Net', url: 'https://scikit-learn.org/stable/modules/linear_model.html' },
      { title: "Hastie, Tibshirani & Friedman — The Elements of Statistical Learning (free book)", url: "https://hastie.su.domains/ElemStatLearn/" },
      { title: "Andreas Müller & Sarah Guido — Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
