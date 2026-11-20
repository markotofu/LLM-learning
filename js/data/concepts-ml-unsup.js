/* ML Atlas – content: Classical ML (SVM, Naive Bayes, clustering, PCA, scikit-learn) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'svm', branch: 'ml', title: 'Support Vector Machines', short: 'Support Vector Machines', level: 2, minutes: 35, prereqs: ['logistic-regression', 'regularization'],
    tags: ['svm', 'margin', 'hinge loss', 'kernel trick', 'rbf'],
    summary: 'Classify with the widest possible margin: support vectors, the hinge loss, the C trade-off, and the kernel trick for curved boundaries.',
    why: `Before deep learning took over, support vector machines were the go-to classifier for text, images and bioinformatics — and they are still excellent for small-to-medium datasets with many features. SVMs introduce two ideas you will meet again and again: **maximising a margin** (robustness through confidence) and the **kernel trick** (working implicitly in a richer feature space via similarities). The hinge loss is also a close cousin of the losses used to train ranking and embedding models.`,
    objectives: [
      'Explain the maximum-margin idea and what support vectors are',
      'Write the soft-margin SVM objective with hinge loss and L2 regularization',
      'Explain the role of C (or λ) in trading margin width against violations',
      'Describe the kernel trick and when an RBF kernel helps'
    ],
    sections: [
      { type: 'text', title: 'The widest street', md: `When two classes are linearly separable, infinitely many lines separate them. Which is best? An SVM chooses the line with the **largest margin** — the widest "street" between the classes, with the boundary down the middle.

Intuition: a boundary that passes far from all training points is robust — small perturbations of the inputs won't flip predictions — and tends to generalize better.

Only the points on the edges of the street determine it: the **support vectors**. Move any other point (without crossing the street) and the boundary doesn't change. That makes SVMs memory-efficient at prediction time.` },
      { type: 'text', title: 'Soft margins and the hinge loss', md: `Real data overlaps, so we allow some points inside the street or on the wrong side, at a price. With labels y ∈ {−1, +1} and score s = w·x + b, the **soft-margin SVM** minimises

$$ λ ‖w‖² + (1/n) Σ max(0, 1 − yᵢ sᵢ) $$
:::where
yᵢ — the label of example i, −1 or +1
sᵢ — the score of example i, w · xᵢ + b
max(0, 1 − yᵢ sᵢ) — the **hinge loss**: zero for points classified correctly with a margin (yᵢsᵢ ≥ 1), growing linearly for points inside the street or on the wrong side
λ‖w‖² — the regularization term; a small ‖w‖ means a *wide* margin (the street's width is 2/‖w‖)
λ — "lambda": how much a wide margin matters compared with avoiding violations
n — the number of training examples
:::

In scikit-learn the trade-off is written with **C ≈ 1/λ**: large C → violations are expensive → narrow margin, fits the training data tightly (risk of overfitting); small C → wide margin, more violations tolerated (more regularization).

Compare with logistic regression: same linear score, different loss. Log loss never reaches zero (it keeps pushing confident points further), while hinge loss stops caring once a point is beyond the margin.` },
      { type: 'code', title: 'A linear SVM trained by subgradient descent', code: `
import numpy as np
rng = np.random.default_rng(0)
X = np.vstack([rng.normal([-2, -1], 1.0, (50, 2)), rng.normal([2, 1.5], 1.0, (50, 2))])
y = np.array([-1] * 50 + [1] * 50)

w, b, lam, lr = np.zeros(2), 0.0, 0.01, 0.1
for epoch in range(200):
    margins = y * (X @ w + b)
    viol = margins < 1                                   # points inside the street or misclassified
    grad_w = 2 * lam * w - (X[viol].T @ y[viol]) / len(y)
    grad_b = -np.sum(y[viol]) / len(y)
    w -= lr * grad_w
    b -= lr * grad_b

margins = y * (X @ w + b)
print("w:", w.round(3), " b:", round(b, 3))
print("training accuracy:", np.mean(np.sign(X @ w + b) == y))
print("margin width 2/||w|| =", round(2 / np.linalg.norm(w), 3))
print("points on or inside the margin (support vectors):", int(np.sum(margins <= 1.0001)))
` },
      { type: 'check', question: { type: 'mcq', q: 'For a correctly classified point with y·s = 2.5, what is its hinge loss?', options: ['2.5', '1', '0', '−1.5'], answer: 2, explain: 'max(0, 1 − 2.5) = 0. Points beyond the margin contribute nothing.' } },
      { type: 'text', title: 'The kernel trick', md: `What if no straight line separates the classes — say, one class forms a ring around the other? Mapping the data into a higher-dimensional space (add x₁², x₂², x₁x₂…) can make it linearly separable there. But explicit feature maps get huge.

The SVM's solution only ever needs **dot products between examples**. A **kernel** k(a, b) computes the dot product in a rich feature space *without* building that space:
- **Polynomial kernel** (a·b + c)^d.
- **RBF (Gaussian) kernel** exp(−γ‖a − b‖²) — a similarity that is 1 for identical points and decays with distance; it corresponds to an *infinite*-dimensional feature space.

With an RBF kernel the decision function becomes a weighted sum of similarities to the support vectors — a smooth, flexible boundary. Hyperparameters: C and γ (large γ = very local, wiggly boundary).

The cost: kernel SVMs scale roughly quadratically with the number of examples, so they suit datasets up to tens of thousands of points.` },
      { type: 'code', title: 'Linear vs RBF kernel on rings (scikit-learn)', code: `
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ["linear", "rbf"]:
    clf = SVC(kernel=kernel, C=1.0).fit(X_tr, y_tr)
    print(f"{kernel:>6} kernel: test accuracy {clf.score(X_te, y_te):.3f}, support vectors {len(clf.support_)}")
` },
      { type: 'check', question: { type: 'tf', q: 'Increasing C in scikit-learn\'s SVC makes margin violations more expensive, which tends to narrow the margin and fit the training data more tightly.', answer: true, explain: 'C acts like 1/λ: less regularization, more risk of overfitting.' } }
    ],
    when: `- Small/medium datasets with many features (text classification with TF-IDF features is a classic fit for linear SVMs).
- When a robust, max-margin linear classifier is wanted, or a smooth non-linear boundary via the RBF kernel on modest data sizes.
- Avoid kernel SVMs on very large datasets (training cost), and when you need calibrated probabilities (use logistic regression, or calibrate).
- Always **scale features** — the margin and RBF distances depend on them.`,
    pitfalls: `- Forgetting feature scaling.
- Poorly chosen C and γ — tune them together on a log grid with cross-validation.
- Expecting probabilities: \`SVC(probability=True)\` fits an extra calibration model and is slower.
- Labels must be ±1 in the math (libraries handle 0/1 for you).`,
    takeaways: [
      'An SVM picks the separating boundary with the widest margin; support vectors define it.',
      'Soft-margin objective = λ‖w‖² + mean hinge loss max(0, 1 − y·s).',
      'C ≈ 1/λ trades margin width for training errors.',
      'Kernels compute similarities in rich feature spaces implicitly; RBF gives smooth non-linear boundaries.',
      'Scale features; kernel SVMs suit small-to-medium datasets.'
    ],
    quiz: [
      { type: 'mcq', q: 'What are support vectors?', options: ['All training points', 'The training points on or inside the margin that determine the boundary', 'The weight vectors', 'Points far from the boundary'], answer: 1, explain: 'Only margin and violating points affect the solution.' },
      { type: 'fill', q: 'A point has label y = −1 and score s = 0.4. What is its hinge loss?', answer: ['1.4'], explain: 'max(0, 1 − (−1)(0.4)) = max(0, 1.4) = 1.4 — it is on the wrong side.' },
      { type: 'mcq', q: 'What does the kernel trick allow?', options: ['Training without labels', 'Computing dot products in a high-dimensional feature space without constructing it', 'Removing the need for regularization', 'Making SVMs output probabilities'], answer: 1, explain: 'Kernels give similarities that equal dot products of implicit feature maps.' },
      { type: 'mcq', q: 'How does SVM\'s hinge loss differ from logistic regression\'s log loss?', options: ['Hinge loss is zero for points beyond the margin; log loss is never exactly zero', 'They are identical', 'Hinge loss is used for regression', 'Log loss ignores misclassified points'], answer: 0, explain: 'Log loss keeps pushing even confident correct points; hinge loss is satisfied once the margin is met.' },
      { type: 'tf', q: 'A wider margin corresponds to a smaller ‖w‖.', answer: true, explain: 'The margin width is 2/‖w‖.' }
    ],
    problems: [
      {
        id: 'hinge-loss', title: 'The SVM Objective', difficulty: 'easy',
        prompt: `Implement \`svm_objective(X, y, w, b, lam)\` for labels \`y\` in {−1, +1}:

$$ λ‖w‖² + (1/n) Σ max(0, 1 − yᵢ (w·xᵢ + b)) $$
:::where
xᵢ — row i of X, the features of example i (X has shape (n, d))
yᵢ — the label of example i, −1 or +1
w, b — the weight vector and the bias
λ — lam, the regularization strength
max(0, 1 − yᵢ(w·xᵢ + b)) — the hinge loss of example i
:::

Return a Python float.

\`\`\`text
X = [[1, 0], [0, 1]], y = [1, -1], w = [1, -1], b = 0, lam = 0.5
scores = [1, -1] -> margins y*s = [1, 1] -> hinge = [0, 0]
objective = 0.5 * (1 + 1) + 0 = 1.0
\`\`\``,
        starter: `
import numpy as np

def svm_objective(X, y, w, b, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def svm_objective(X, y, w, b, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    hinge = np.maximum(0.0, 1.0 - y * (X @ w + b))
    return float(lam * np.sum(w ** 2) + np.mean(hinge))
`,
        hints: ['Scores: `X @ w + b`; margins: `y * scores`.', '`np.maximum(0, 1 - margins)` gives the hinge loss of every example.'],
        explain: `\`np.maximum\` is the element-wise max, perfect for the hinge. The objective balances a wide margin (small ‖w‖) against the average violation; the subgradient used in training follows directly from it.`,
        tests: [
          { call: 'svm_objective([[1, 0], [0, 1]], [1, -1], [1, -1], 0.0, 0.5)', expect: '1.0' },
          { name: 'violations count', call: 'svm_objective([[1.0], [-1.0], [0.2]], [1, -1, 1], [1.0], 0.0, 0.0)', expect: '0.8 / 3' },
          { name: 'random data', setup: 'rng = np.random.default_rng(2)\nX = rng.normal(size=(30, 3))\ny = np.where(rng.random(30) < 0.5, -1, 1)\nw = rng.normal(size=3)', call: 'svm_objective(X, y, w, 0.3, 0.05)', hidden: true }
        ]
      },
      {
        id: 'linear-svm', title: 'Train a Linear SVM', difficulty: 'medium',
        prompt: `Train a linear SVM with full-batch **subgradient descent**. Implement \`train_linear_svm(X, y, lam=0.01, lr=0.1, epochs=200)\` → \`(w, b)\` for labels in {−1, +1}:

- start at \`w = zeros(d)\`, \`b = 0.0\`;
- each epoch: \`m = y * (X @ w + b)\`, \`viol = m < 1\`, then
  - \`grad_w = 2 * lam * w - (X[viol].T @ y[viol]) / n\`
  - \`grad_b = -np.sum(y[viol]) / n\`
  - \`w -= lr * grad_w\`, \`b -= lr * grad_b\`;
- return \`w\` and \`float(b)\`.

The tests check the exact trajectory and that the result separates easy data.`,
        starter: `
import numpy as np

def train_linear_svm(X, y, lam=0.01, lr=0.1, epochs=200):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    # TODO
    return w, float(b)
`,
        solution: `
import numpy as np

def train_linear_svm(X, y, lam=0.01, lr=0.1, epochs=200):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        m = y * (X @ w + b)
        viol = m < 1
        grad_w = 2 * lam * w - (X[viol].T @ y[viol]) / n
        grad_b = -np.sum(y[viol]) / n
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, float(b)
`,
        hints: ['Only the margin violators (y·s < 1) contribute to the data part of the gradient.', 'The regularization part 2·lam·w applies every step.'],
        explain: `The hinge loss has a kink at y·s = 1, so we use a *subgradient*: −y·x for violators and 0 otherwise. Points comfortably beyond the margin stop influencing training — the algorithmic version of "only support vectors matter".`,
        tests: [
          { name: 'exact trajectory (3 epochs)', call: 'train_linear_svm([[2.0, 0.0], [0.0, -1.0], [-2.0, 1.0]], [1, -1, -1], lam=0.1, lr=0.5, epochs=3)' },
          { name: 'separates easy data', setup: 'rng = np.random.default_rng(1)\nX = np.vstack([rng.normal([-3, -3], 1.0, (40, 2)), rng.normal([3, 3], 1.0, (40, 2))])\ny = np.array([-1] * 40 + [1] * 40)', code: 'w, b = train_linear_svm(X, y)\nacc = np.mean(np.sign(X @ w + b) == y)\nassert acc >= 0.97, "accuracy %.3f too low" % acc', hidden: true },
          { name: 'exact trajectory, random data', setup: 'rng = np.random.default_rng(6)\nX = rng.normal(size=(25, 3))\ny = np.where(X[:, 0] + 0.3 * rng.normal(size=25) > 0, 1, -1)', call: 'train_linear_svm(X, y, lam=0.05, lr=0.2, epochs=30)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Support Vector Machines', url: 'https://scikit-learn.org/stable/modules/svm.html' },
      { title: "Stanford CS229 — Machine Learning lecture notes (SVMs and kernels)", url: "https://cs229.stanford.edu/main_notes.pdf" }
    ]
  },

  // =====================================================================================
  {
    id: 'naive-bayes', branch: 'ml', title: 'Naive Bayes for Text Classification', short: 'Naive Bayes', level: 2, minutes: 35, prereqs: ['probability', 'supervised-learning'],
    tags: ['naive bayes', 'spam', 'text classification', 'laplace smoothing', 'bayes rule'],
    summary: 'Bayes\' rule turned into a fast classifier: the "naive" independence assumption, word-count likelihoods, Laplace smoothing and log probabilities — the classic spam filter.',
    why: `Naive Bayes filtered the world's spam for years, and it remains a strong, extremely fast baseline for text classification: it trains in one pass over the data, needs little data, and handles huge vocabularies. It is also the clearest example of a **generative** classifier — it models how each class generates its data — and it puts Bayes' rule, independence assumptions and log-probabilities to work in a few lines of code.`,
    objectives: [
      'Apply Bayes\' rule to classification: posterior ∝ likelihood × prior',
      'State the naive independence assumption and why it still works',
      'Estimate word likelihoods with Laplace (add-one) smoothing',
      'Classify with summed log-probabilities to avoid underflow'
    ],
    sections: [
      { type: 'text', title: 'Classification with Bayes\' rule', md: `We want P(class | document). Bayes' rule turns it around:

$$ P(c | doc) = P(doc | c) · P(c) / P(doc) $$
:::where
c — a class, such as "spam"
doc — the document being classified (its words)
P(c | doc) — the posterior: the probability of class c after reading the document
P(doc | c) — the likelihood: how probable the document is if it came from class c
P(c) — the prior: how common class c is before reading anything
P(doc) — the overall probability of the document, which is the same for every class
:::

P(doc) is the same for every class, so to pick the most likely class we only compare

$$ score(c) = P(c) · P(doc | c) $$
:::where
c — a candidate class, such as "spam" or "not spam"
P(c) — the **prior**: how common the class is (e.g. 30% of emails are spam)
P(doc | c) — the **likelihood**: how probable this document is if it came from class c
score(c) — proportional to P(c | doc), so the class with the highest score wins
:::

The hard part is P(doc | c) — there are astronomically many possible documents.` },
      { type: 'text', title: 'The naive assumption', md: `Naive Bayes assumes that, given the class, **each word occurs independently** of the others. Then the likelihood of a document is a product over its words:

$$ P(doc | c) = Π_words P(word | c) $$
:::where
Π — "product": multiply everything together (the multiplication version of Σ)
Π_words — multiply over every word in the document; a repeated word contributes once per occurrence
P(word | c) — how likely that word is to appear in documents of class c
:::

This is obviously false ("New" and "York" are not independent!), yet the classifier works surprisingly well: we only need the *ranking* of classes to be right, not calibrated probabilities, and the errors from the assumption often affect all classes similarly.

**Multinomial Naive Bayes** estimates each P(word | c) from counts:

$$ P(w | c) = (count(w, c) + α) / (total words in c + α · |V|) $$
:::where
w — a word in the vocabulary
count(w, c) — how many times word w appears in the training documents of class c
total words in c — the total number of words in all training documents of class c
|V| — the vocabulary size, the number of distinct words
α — "alpha": the smoothing amount added to every count
:::

Setting α = 1 is called **Laplace (add-one) smoothing**. Without smoothing, a word never seen in spam during training would get probability 0 — and a single unseen word would make P(doc | spam) = 0 no matter what else the email says.` },
      { type: 'text', title: 'Log-probabilities', md: `Multiplying hundreds of small probabilities underflows to 0.0. So we work in logs, and products become sums:

$$ log score(c) = log P(c) + Σ_words log P(word | c) $$
:::where
log — the natural logarithm, which turns the product of probabilities into a sum
log P(c) — the log prior of class c
Σ_words — add up over every word in the document
log P(word | c) — the log-likelihood of each word under class c (always ≤ 0)
:::

Predict the class with the highest log score. Words not in the training vocabulary are simply skipped.

:::real Variants
**Bernoulli NB** models presence/absence of words (good for short texts); **Gaussian NB** models continuous features with a normal distribution per class; **Complement NB** is more robust on imbalanced text data.
:::` },
      { type: 'code', title: 'A tiny spam filter from scratch', code: `
import math
from collections import Counter

train = [("win money now", "spam"), ("cheap money offer", "spam"), ("win a cheap prize now", "spam"),
         ("meeting at noon", "ham"), ("lunch money tomorrow", "ham"), ("project meeting notes", "ham")]

classes = sorted({c for _, c in train})
vocab = sorted({w for doc, _ in train for w in doc.split()})
docs_per_class = Counter(c for _, c in train)
word_counts = {c: Counter() for c in classes}
for doc, c in train:
    word_counts[c].update(doc.split())

def log_score(doc, c, alpha=1.0):
    total = sum(word_counts[c].values())
    s = math.log(docs_per_class[c] / len(train))                      # log prior
    for w in doc.split():
        if w in vocab:
            s += math.log((word_counts[c][w] + alpha) / (total + alpha * len(vocab)))
    return s

for msg in ["win cheap money", "meeting tomorrow at noon", "money for the project"]:
    scores = {c: round(log_score(msg, c), 2) for c in classes}
    print(f"{msg!r:30} -> {max(scores, key=scores.get):4}  {scores}")
` },
      { type: 'check', question: { type: 'mcq', q: 'Why is Laplace (add-one) smoothing needed in Naive Bayes?', options: ['To speed up training', 'So a word never seen with a class doesn\'t force that class\'s probability to zero', 'To remove stop words', 'To balance the classes'], answer: 1, explain: 'One zero factor in a product makes the whole product zero; adding α keeps every word probability positive.' } }
    ],
    when: `- **Text classification baselines** (spam, sentiment, topic, language detection), especially with little data or a need for speed.
- When features are counts or presence indicators.
- Streaming or online settings — updating counts is trivial.
- Prefer logistic regression or fine-tuned transformers when accuracy matters more than speed, or when you need calibrated probabilities.`,
    pitfalls: `- Forgetting smoothing (zero probabilities).
- Multiplying raw probabilities (underflow) instead of summing logs.
- Trusting the output probabilities — they are typically over-confident because of the independence assumption.
- Strongly correlated features get "double counted".`,
    takeaways: [
      'Naive Bayes picks argmax_c P(c)·Π P(word | c).',
      'The naive assumption: words are independent given the class — false, but effective.',
      'Laplace smoothing: (count + α) / (total + α|V|) prevents zero probabilities.',
      'Sum log-probabilities to avoid underflow.',
      'Fast, data-efficient, and a strong text baseline.'
    ],
    quiz: [
      { type: 'mcq', q: 'In Naive Bayes, what does the prior P(c) represent?', options: ['The probability of a word', 'How common class c is before seeing the document', 'The smoothing constant', 'The accuracy of the classifier'], answer: 1, explain: 'It is the base rate of each class in the training data.' },
      { type: 'fill', q: 'Class "spam" has 8 words in total, the word "free" appears 3 times, and the vocabulary has 10 words. With α = 1, what is P(free | spam)? (fraction or decimal)', answer: ['4/18', '2/9', '0.222', '0.2222', '0.22'], display: '4/18 = 2/9', explain: '(3 + 1) / (8 + 1·10) = 4/18.' },
      { type: 'tf', q: 'Naive Bayes assumes that all words in a document are independent of each other given the class.', answer: true, explain: 'That conditional-independence assumption is the "naive" part.' },
      { type: 'mcq', q: 'Why does the implementation add log-probabilities instead of multiplying probabilities?', options: ['Logs are more accurate for big numbers', 'Products of many small numbers underflow to zero', 'It is required by Bayes\' rule', 'To make the classifier linear'], answer: 1, explain: 'Sums of logs are numerically stable.' },
      { type: 'mcq', q: 'How should Naive Bayes treat a word that never appeared in the training data?', options: ['Give it probability 0', 'Skip it (it is not in the vocabulary)', 'Reject the document', 'Retrain the model'], answer: 1, explain: 'Out-of-vocabulary words carry no learned information, so they are ignored.' }
    ],
    problems: [
      {
        id: 'naive-bayes-train-predict', title: 'Multinomial Naive Bayes', difficulty: 'medium',
        prompt: `Implement a \`MultinomialNB\` text classifier. Tokenize every document with \`doc.lower().split()\`.

**fit(docs, labels, alpha=1.0)** sets:
- \`classes_\` — sorted list of distinct labels;
- \`vocab_\` — sorted list of distinct training tokens;
- \`log_prior_\` — NumPy array, log(number of docs of the class / number of docs), in \`classes_\` order;
- \`log_likelihood_\` — NumPy array of shape (n_classes, |V|): log((count of word in class + α) / (total words in class + α·|V|)), columns in \`vocab_\` order.

**predict(docs)** → a list of labels: for each document, the class maximising log_prior + Σ log_likelihood over its tokens (repeated tokens count each time, unknown tokens are skipped). Ties → the earlier class in \`classes_\`.`,
        starter: `
import numpy as np

class MultinomialNB:
    def fit(self, docs, labels, alpha=1.0):
        # TODO
        return self

    def predict(self, docs):
        # TODO
        pass
`,
        solution: `
import numpy as np

class MultinomialNB:
    def fit(self, docs, labels, alpha=1.0):
        tokens = [d.lower().split() for d in docs]
        self.classes_ = sorted(set(labels))
        self.vocab_ = sorted({w for t in tokens for w in t})
        index = {w: j for j, w in enumerate(self.vocab_)}
        counts = np.zeros((len(self.classes_), len(self.vocab_)))
        n_docs = np.zeros(len(self.classes_))
        for t, lab in zip(tokens, labels):
            c = self.classes_.index(lab)
            n_docs[c] += 1
            for w in t:
                counts[c, index[w]] += 1
        self.log_prior_ = np.log(n_docs / len(docs))
        totals = counts.sum(axis=1, keepdims=True)
        self.log_likelihood_ = np.log((counts + alpha) / (totals + alpha * len(self.vocab_)))
        self._index = index
        return self

    def predict(self, docs):
        out = []
        for d in docs:
            scores = self.log_prior_.copy()
            for w in d.lower().split():
                j = self._index.get(w)
                if j is not None:
                    scores = scores + self.log_likelihood_[:, j]
            out.append(self.classes_[int(np.argmax(scores))])
        return out
`,
        hints: ['Build a word → column index dict from the sorted vocabulary.', 'Fill a (n_classes, |V|) count matrix in one pass over the documents.', 'In predict, start from log_prior_ and add a column of log_likelihood_ for every known token.'],
        explain: `Training is just counting — one pass, no iterations — which is why Naive Bayes is so fast. Keeping a (classes × vocabulary) matrix of log-probabilities makes prediction a sum of columns. \`np.argmax\` returns the first maximum, giving the required tie-breaking.`,
        tests: [
          { name: 'log priors', setup: 'docs = ["win money", "win prize", "meeting notes"]\nlabels = ["spam", "spam", "ham"]', call: 'MultinomialNB().fit(docs, labels).log_prior_', expect: 'np.log(np.array([1 / 3, 2 / 3]))' },
          { name: 'predictions', setup: 'docs = ["win money now", "cheap money offer", "win a cheap prize now", "meeting at noon", "lunch money tomorrow", "project meeting notes"]\nlabels = ["spam", "spam", "spam", "ham", "ham", "ham"]', call: 'MultinomialNB().fit(docs, labels).predict(["win cheap money", "meeting tomorrow at noon", "totally unknown words"])', expect: '["spam", "ham", "ham"]' },
          { name: 'likelihood matrix', setup: 'docs = ["a b a", "b c"]\nlabels = [1, 2]', call: 'MultinomialNB().fit(docs, labels).log_likelihood_', expect: 'np.log(np.array([[3 / 6, 2 / 6, 1 / 6], [1 / 5, 2 / 5, 2 / 5]]))', hidden: true },
          { name: 'random corpus vs reference', setup: 'rng = np.random.default_rng(4)\nwords = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]\ndocs = [" ".join(rng.choice(words[:4] if i % 2 else words[2:], size=5)) for i in range(20)]\nlabels = ["odd" if i % 2 else "even" for i in range(20)]\ntests = [" ".join(rng.choice(words, size=4)) for _ in range(8)]', call: 'MultinomialNB().fit(docs, labels, alpha=0.5).predict(tests)', hidden: true }
        ]
      },
      {
        id: 'nb-explain', title: 'Explaining a Naive Bayes Decision', difficulty: 'medium',
        prompt: `A trained two-class Naive Bayes model is fully transparent: every word adds a fixed amount of evidence. Given \`log_prior\` (shape 2: class 0, class 1), \`log_likelihood\` (shape 2 × |V|, log P(word | class)) and the matching list \`vocab\`, implement:

1. \`word_log_odds(log_likelihood, vocab)\` → a dict {word: log P(word | class 1) − log P(word | class 0)}.
2. \`top_words(log_likelihood, vocab, k)\` → the k words with the **largest** log-odds (strongest evidence for class 1), largest first; ties keep vocabulary order.
3. \`doc_log_odds(log_prior, log_likelihood, vocab, doc)\` → the log-odds of class 1 versus class 0 for a document: (log_prior[1] − log_prior[0]) plus the word log-odds of every token of \`doc.lower().split()\` that is in the vocabulary (repeated tokens count every time; unknown tokens are ignored). A positive value means class 1 wins.`,
        starter: `
import numpy as np

def word_log_odds(log_likelihood, vocab):
    # TODO
    pass

def top_words(log_likelihood, vocab, k):
    # TODO
    pass

def doc_log_odds(log_prior, log_likelihood, vocab, doc):
    # TODO
    pass
`,
        solution: `
import numpy as np

def word_log_odds(log_likelihood, vocab):
    L = np.asarray(log_likelihood, dtype=float)
    return {w: float(L[1, j] - L[0, j]) for j, w in enumerate(vocab)}

def top_words(log_likelihood, vocab, k):
    odds = word_log_odds(log_likelihood, vocab)
    order = sorted(range(len(vocab)), key=lambda j: -odds[vocab[j]])
    return [vocab[j] for j in order[:k]]

def doc_log_odds(log_prior, log_likelihood, vocab, doc):
    odds = word_log_odds(log_likelihood, vocab)
    total = float(log_prior[1] - log_prior[0])
    for tok in doc.lower().split():
        if tok in odds:
            total += odds[tok]
    return total
`,
        hints: ['Log-odds of a word: subtract its two log-likelihoods.', 'Python\'s `sorted` is stable, so sorting indices by −odds keeps vocabulary order on ties.', 'Start the document total from the prior log-odds.'],
        explain: `Because Naive Bayes multiplies independent word probabilities, its log-odds are a sum of per-word contributions — a built-in explanation of every decision. The same additive view underlies logistic regression, which learns the per-word weights directly instead of from counts.`,
        tests: [
          { setup: 'L = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]])); V = ["meeting", "report", "prize"]', call: 'top_words(L, V, 2)', expect: '["prize", "report"]' },
          { setup: 'L = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]])); V = ["meeting", "report", "prize"]; P = np.log(np.array([0.5, 0.5]))', call: 'doc_log_odds(P, L, V, "Prize prize unknown")', expect: '2 * np.log(0.8 / 0.25)' },
          { setup: 'L = np.log(np.array([[0.5, 0.5], [0.25, 0.75]])); V = ["a", "b"]', call: 'word_log_odds(L, V)', expect: '{"a": np.log(0.5), "b": np.log(1.5)}' },
          { name: 'random model vs reference', setup: 'rng = np.random.default_rng(0)\nraw = rng.random((2, 6)); L = np.log(raw / raw.sum(1, keepdims=True)); V = ["w%d" % i for i in range(6)]; P = np.log(np.array([0.7, 0.3]))', call: '[top_words(L, V, 3), doc_log_odds(P, L, V, "w1 w5 w5 zzz W2")]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Naive Bayes', url: 'https://scikit-learn.org/stable/modules/naive_bayes.html' },
      { title: 'Paul Graham — A Plan for Spam (2002)', url: 'http://www.paulgraham.com/spam.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'kmeans', branch: 'ml', title: 'Clustering with k-Means', short: 'k-Means Clustering', level: 1, minutes: 35, prereqs: ['numpy', 'supervised-learning'],
    tags: ['unsupervised', 'clustering', 'k-means', 'inertia', 'k-means++'],
    summary: 'Find groups in unlabelled data: the assign-update loop of k-means, its objective, initialization with k-means++, choosing k, and its limits.',
    why: `Most data in the world has no labels. Clustering finds structure anyway: customer segments, groups of similar documents, anomalies, compressed representations. k-means is the simplest and most widely used clustering algorithm, and its "alternate between assigning and updating" loop is a pattern you will see again (expectation–maximization, vector quantization in image and audio models, product quantization in vector databases).`,
    objectives: [
      'Explain the goal of clustering and how it differs from classification',
      'Run Lloyd\'s algorithm: assign points to the nearest centroid, move centroids to the mean',
      'Compute the inertia objective and explain why initialization matters (k-means++)',
      'Choose k with the elbow method or silhouette scores and know k-means\' assumptions'
    ],
    sections: [
      { type: 'text', title: 'Clustering', md: `In **clustering**, there are no labels: we want to group similar examples together so that points in the same cluster are close and points in different clusters are far apart.

Uses: customer segmentation, grouping news articles by topic, compressing colours in an image (each pixel replaced by its cluster's colour), detecting anomalies (points far from every cluster), and pre-organising data for search.

Clustering is **exploratory**: there is no single correct answer, and the result depends on the features, their scaling and the algorithm.` },
      { type: 'text', title: 'The k-means algorithm', md: `Choose the number of clusters **k**. Then:

1. **Initialize** k centroids (e.g. k randomly chosen data points).
2. **Assign**: put each point in the cluster of its nearest centroid.
3. **Update**: move each centroid to the mean of the points assigned to it.
4. Repeat 2–3 until the assignments stop changing.

It minimises the **inertia** — the total squared distance from each point to its centroid:

$$ inertia = Σᵢ ‖xᵢ − μ_{cluster(i)}‖² $$
:::where
xᵢ — the i-th data point
cluster(i) — the index of the cluster that point i is currently assigned to
μ_{cluster(i)} — that cluster's centroid (the mean of its points)
‖…‖² — the squared Euclidean distance
Σᵢ — add up over all the points
:::

Both steps can only decrease the inertia, so the algorithm always converges — but to a **local** minimum that depends on the starting centroids.` },
      { type: 'demo', title: 'Interactive: k-means step by step', md: `Step through assign/update; then try different random starts with k = 4 and compare the inertia.`, demo: 'kmeans' },
      { type: 'code', title: 'k-means in NumPy', code: `
import numpy as np
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 0.6, (50, 2)) for c in ([0, 0], [4, 0], [2, 3.5])])

def kmeans(X, k, seed, iters=100):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)   # assign
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC                                                                # update
    return C, labels, ((X - C[labels]) ** 2).sum()

for k in range(1, 7):
    best = min(kmeans(X, k, s)[2] for s in range(5))      # best of 5 random starts
    print(f"k={k}: inertia {best:8.1f}")
`, after: `Inertia always falls as k grows (k = n gives inertia 0), so you can't pick k by minimising it. Look for the **elbow** where adding clusters stops helping much — here at k = 3.` },
      { type: 'check', question: { type: 'mcq', q: 'Why do libraries run k-means several times with different initializations?', options: ['To use more CPU cores', 'Because k-means converges to a local minimum that depends on the starting centroids', 'To choose k automatically', 'Because one run never converges'], answer: 1, explain: 'Different starts can give different (worse or better) solutions; keep the run with the lowest inertia.' } },
      { type: 'text', title: 'Initialization, choosing k, and limitations', md: `**k-means++ initialization** picks the first centroid at random, then each next one with probability proportional to its squared distance from the nearest existing centroid — spreading the starting points out. It's the default in scikit-learn and gives better, more consistent results.

**Choosing k:**
- **Elbow method** — plot inertia vs k, look for the bend.
- **Silhouette score** — for each point compares the mean distance to its own cluster with the distance to the nearest other cluster (−1 to 1, higher is better).
- **Domain needs** — sometimes the business needs exactly 5 segments.

**Assumptions and limits:**
- Clusters are roughly **spherical** and of similar size (it uses Euclidean distance to a mean).
- Sensitive to **feature scaling** and to **outliers** (means get dragged).
- Every point must belong to some cluster.

Alternatives: **DBSCAN** (density-based, finds arbitrary shapes, labels noise), **hierarchical clustering** (a tree of merges), **Gaussian mixture models** (soft, elliptical clusters).` },
      { type: 'check', question: { type: 'tf', q: 'Increasing k always decreases (or keeps equal) the best achievable inertia.', answer: true, explain: 'More centroids can only fit the points as well or better; that is why inertia alone can\'t choose k.' } },
      { type: 'text', title: 'Going further: k-means++, choosing k and scaling up', md: `- **k-means++ initialization** picks the first centroid at random and each next one with probability proportional to its squared distance from the nearest centroid already chosen. Spreading the starting points out gives faster convergence and better solutions; scikit-learn uses it by default and keeps the best of several runs.
- **Choosing k**: look for the elbow of the inertia curve, compare silhouette scores, or let the downstream use decide; Gaussian mixtures add the BIC (see *Gaussian Mixtures & EM*). The silhouette of a point compares its own cluster with the nearest other one:

$$ s(i) = ( b(i) − a(i) ) / max( a(i), b(i) ) $$
:::where
a(i) — the average distance from point i to the other points of its own cluster
b(i) — the average distance from point i to the points of the nearest other cluster
s(i) — the silhouette of point i, from −1 (probably in the wrong cluster) to +1 (well inside its cluster); average it over all points to compare values of k
:::

- **Scale**: mini-batch k-means updates the centroids from small random batches and handles millions of points.
- **Limits**: k-means prefers round clusters of similar size and is sensitive to feature scaling and outliers; DBSCAN and Gaussian mixtures fit other shapes.
- **Beyond clustering**: k-means codebooks compress vectors (IVF and product quantization in *Vector Search*) and turn audio into discrete units for speech models.` }
    ],
    when: `- Quick segmentation of data with compact, similar-sized groups; colour/vector quantization; a first exploratory look.
- Scale features first and remove or cap outliers.
- Use DBSCAN for irregular shapes or when you need noise detection; Gaussian mixtures for soft assignments.`,
    pitfalls: `- Unscaled features (one feature dominates the distance).
- Trusting a single random initialization.
- Choosing k by minimum inertia (always the largest k).
- Forcing k-means onto elongated or ring-shaped clusters.
- Over-interpreting clusters as "real" categories without validation.`,
    takeaways: [
      'k-means alternates two steps: assign to the nearest centroid, move centroids to the mean.',
      'It minimises inertia (within-cluster squared distance) and converges to a local minimum.',
      'Use k-means++ initialization and several restarts.',
      'Choose k with the elbow method, silhouette scores or domain needs — not raw inertia.',
      'Assumes spherical, similar-sized clusters; scale features.'
    ],
    quiz: [
      { type: 'order', q: 'Put one iteration of k-means in order (after initialization).', items: ['Compute the distance from every point to every centroid', 'Assign each point to its nearest centroid', 'Move each centroid to the mean of its assigned points', 'Check whether the assignments changed'], explain: 'Assign (via distances), update the means, then test for convergence.' },
      { type: 'mcq', q: 'What does k-means minimise?', options: ['The number of clusters', 'The sum of squared distances from points to their cluster centroid (inertia)', 'The classification error', 'The distance between centroids'], answer: 1, explain: 'Both steps of the algorithm reduce the inertia.' },
      { type: 'mcq', q: 'Which data would k-means handle poorly?', options: ['Three compact round blobs', 'Two concentric rings', 'Well-separated equal-size groups', 'Standardized customer features with clear segments'], answer: 1, explain: 'Rings are not spherical clusters around a mean; density-based methods like DBSCAN fit them better.' },
      { type: 'tf', q: 'k-means is a supervised learning algorithm.', answer: false, explain: 'It uses no labels: it is unsupervised.' },
      { type: 'mcq', q: 'What does k-means++ improve?', options: ['The choice of k', 'The initial centroids, spreading them out for better and more consistent results', 'The distance metric', 'Training speed on GPUs'], answer: 1, explain: 'Smart seeding avoids starting several centroids in the same cluster.' }
    ],
    problems: [
      {
        id: 'kmeans-assign-update', title: 'k-Means: Assign and Update', difficulty: 'medium',
        prompt: `Implement the two steps of k-means:

1. \`assign(X, C)\` → int array of length n: for each row of X, the index of the nearest centroid (row of C) by Euclidean distance. Ties → the lowest index.
2. \`update(X, labels, C_old)\` → new centroid array, same shape as \`C_old\`: row j is the mean of the points with label j; if cluster j is **empty**, keep \`C_old[j]\`.

\`\`\`text
X = [[0, 0], [0, 1], [5, 5], [6, 5]],  C = [[0, 0], [5, 5]]
assign(X, C)                -> [0, 0, 1, 1]
update(X, [0, 0, 1, 1], C)  -> [[0. , 0.5], [5.5, 5. ]]
\`\`\``,
        starter: `
import numpy as np

def assign(X, C):
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    # TODO
    pass

def update(X, labels, C_old):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    C_old = np.asarray(C_old, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def assign(X, C):
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d2, axis=1)

def update(X, labels, C_old):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    C_old = np.asarray(C_old, dtype=float)
    C = C_old.copy()
    for j in range(len(C_old)):
        mask = labels == j
        if np.any(mask):
            C[j] = X[mask].mean(axis=0)
    return C
`,
        hints: ['Squared distances are enough for choosing the nearest centroid — no square root needed.', '`np.argmin` returns the first (lowest) index on ties.', 'Start from a copy of `C_old` so empty clusters keep their old centroid.'],
        explain: `Comparing squared distances avoids the square root without changing which centroid is nearest. Handling empty clusters matters in practice: with unlucky initialization a centroid can lose all its points, and dividing by zero would produce NaNs.`,
        tests: [
          { call: 'assign([[0, 0], [0, 1], [5, 5], [6, 5]], [[0, 0], [5, 5]])', expect: 'np.array([0, 0, 1, 1])' },
          { call: 'update([[0, 0], [0, 1], [5, 5], [6, 5]], [0, 0, 1, 1], [[0, 0], [5, 5]])', expect: 'np.array([[0.0, 0.5], [5.5, 5.0]])' },
          { name: 'empty cluster keeps its centroid', call: 'update([[1.0, 1.0], [3.0, 3.0]], [0, 0], [[0.0, 0.0], [9.0, 9.0]])', expect: 'np.array([[2.0, 2.0], [9.0, 9.0]])' },
          { name: 'tie goes to the lowest index', call: 'assign([[1.0, 0.0]], [[0.0, 0.0], [2.0, 0.0]])', expect: 'np.array([0])', hidden: true },
          { name: 'random data', setup: 'rng = np.random.default_rng(3)\nX = rng.normal(size=(40, 3))\nC = rng.normal(size=(4, 3))', call: 'update(X, assign(X, C), C)', hidden: true }
        ]
      },
      {
        id: 'kmeans-full', title: 'k-Means from Start to Finish', difficulty: 'medium',
        prompt: `Implement \`kmeans(X, k, seed=0, max_iter=100)\` → \`(C, labels, inertia)\` using \`assign\` and \`update\` (provided in the starter) and this exact procedure:

\`\`\`text
rng = np.random.default_rng(seed)
C = X[rng.choice(n, k, replace=False)]        (as floats)
labels = None
repeat up to max_iter times:
    new_labels = assign(X, C)
    if labels is not None and new_labels equals labels: stop
    labels = new_labels
    C = update(X, labels, C)
inertia = sum of squared distances from each point to its centroid
\`\`\`

Return the final centroids, labels and inertia (a float).`,
        starter: `
import numpy as np

def assign(X, C):
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d2, axis=1)

def update(X, labels, C_old):
    C = C_old.copy()
    for j in range(len(C_old)):
        mask = labels == j
        if np.any(mask):
            C[j] = X[mask].mean(axis=0)
    return C

def kmeans(X, k, seed=0, max_iter=100):
    X = np.asarray(X, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def assign(X, C):
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d2, axis=1)

def update(X, labels, C_old):
    C = C_old.copy()
    for j in range(len(C_old)):
        mask = labels == j
        if np.any(mask):
            C[j] = X[mask].mean(axis=0)
    return C

def kmeans(X, k, seed=0, max_iter=100):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    labels = None
    for _ in range(max_iter):
        new_labels = assign(X, C)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        C = update(X, labels, C)
    inertia = float(((X - C[labels]) ** 2).sum())
    return C, labels, inertia
`,
        hints: ['`rng.choice(n, k, replace=False)` picks k distinct starting rows.', 'Stop when `np.array_equal(new_labels, labels)`.', 'Inertia: `((X - C[labels]) ** 2).sum()`.'],
        explain: `Stopping when the assignments no longer change is a clean convergence test: if labels are stable, the means (and therefore the centroids) are stable too. \`C[labels]\` gathers each point's own centroid for the inertia in one vectorized expression.`,
        tests: [
          { name: 'two obvious clusters', setup: 'X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])', code: 'C, labels, inertia = kmeans(X, 2, seed=0)\nassert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]\nassert abs(inertia - 1.0) < 1e-9, "inertia should be 1.0, got %r" % inertia' },
          { name: 'seeded blobs, exact result', setup: 'rng = np.random.default_rng(0)\nX = np.vstack([rng.normal(c, 0.6, (30, 2)) for c in ([0, 0], [4, 0], [2, 3.5])])', call: 'kmeans(X, 3, seed=1)' },
          { name: 'another seed', setup: 'rng = np.random.default_rng(7)\nX = rng.normal(size=(60, 3))', call: 'kmeans(X, 4, seed=5)', hidden: true },
          { name: 'inertia never increases with more clusters (best of seeds)', setup: 'rng = np.random.default_rng(2)\nX = np.vstack([rng.normal(c, 0.5, (25, 2)) for c in ([0, 0], [3, 3])])', code: 'best = [min(kmeans(X, k, seed=s)[2] for s in range(4)) for k in (1, 2, 3)]\nassert best[0] >= best[1] >= best[2], best', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Clustering', url: 'https://scikit-learn.org/stable/modules/clustering.html' },
      { title: "Arthur & Vassilvitskii — k-means++: The Advantages of Careful Seeding", url: "https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf" }
    ]
  },

  // =====================================================================================
  {
    id: 'pca', branch: 'ml', title: 'Dimensionality Reduction with PCA', short: 'PCA', level: 2, minutes: 40, prereqs: ['statistics', 'numpy', 'eigen-svd'],
    tags: ['pca', 'svd', 'eigenvectors', 'explained variance', 'dimensionality reduction'],
    summary: 'Compress many correlated features into a few directions of maximum variance: the intuition, the SVD computation, explained variance, and how to use and misuse PCA.',
    why: `Real datasets often have hundreds or thousands of features that are heavily correlated — pixels next to each other, survey questions that ask nearly the same thing, gene expression levels. **Principal Component Analysis** finds the few directions that capture most of the variation, letting you visualize high-dimensional data in 2-D, speed up and regularize downstream models, remove noise, and understand structure. It is also the cleanest introduction to the linear-algebra machinery (eigenvectors, SVD) behind embeddings and low-rank tricks like LoRA.`,
    objectives: [
      'Explain PCA as finding orthogonal directions of maximum variance',
      'Compute principal components with the SVD of the centred data',
      'Use explained variance ratios to choose the number of components',
      'Project data onto components, reconstruct it, and know PCA\'s limitations'
    ],
    sections: [
      { type: 'text', title: 'The idea', md: `Picture a cloud of 2-D points shaped like a tilted cigar. Most of the variation lies along the cigar's long axis; the short axis is mostly noise. PCA rotates the coordinate system so that:

1. the **first principal component (PC1)** points along the direction of **greatest variance**;
2. **PC2** is perpendicular to PC1 and captures the most remaining variance;
3. and so on — each component orthogonal to the previous ones.

Keeping only the first k components gives the best k-dimensional linear approximation of the data (smallest squared reconstruction error). If two features are 95% redundant, one component captures almost everything both carried.` },
      { type: 'text', title: 'How it is computed', md: `1. **Centre** the data: subtract each feature's mean (and usually **standardize**, so units don't dominate).
2. The principal components are the eigenvectors of the covariance matrix Σ = XᵀX / (n − 1), sorted by eigenvalue; each eigenvalue is the variance along its component.
3. In practice we use the **singular value decomposition** of the centred data instead (more stable, no need to form Σ):

$$ X_c = U S Vᵀ $$
:::where
X_c — the centred data (each column minus its mean), shape (n, d)
U — an (n, r) matrix with orthonormal columns, where r = min(n, d)
S — the singular values S₁ ≥ S₂ ≥ … ≥ 0, one per component
Vᵀ — an (r, d) matrix whose rows are the **principal directions** (components), each a unit vector
Sᵢ² / (n − 1) — the variance of the data along component i
Sᵢ² / Σⱼ Sⱼ² — the **explained variance ratio** of component i
V_k — the first k rows of Vᵀ, shape (k, d)
X_c V_kᵀ — the projected data (the "scores"), an (n, k) matrix
:::

**Reconstruction**: map back with X̂ = scores · V_k + mean. The reconstruction error measures what was thrown away.

Note: each component's sign is arbitrary — v and −v describe the same direction.` },
      { type: 'code', title: 'PCA with NumPy\'s SVD', code: `
import numpy as np
rng = np.random.default_rng(0)

# 3 features, but really only ~1 underlying factor plus noise
t = rng.normal(size=300)
X = np.column_stack([2 * t + rng.normal(0, 0.3, 300),
                     -t + rng.normal(0, 0.3, 300),
                     0.5 * t + rng.normal(0, 0.3, 300)])

mean = X.mean(axis=0)
Xc = X - mean
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ratio = S ** 2 / np.sum(S ** 2)
print("explained variance ratio:", ratio.round(4))
print("first component:", Vt[0].round(3))

scores = Xc @ Vt[:1].T                     # keep 1 component: (300, 1)
X_rec = scores @ Vt[:1] + mean             # reconstruct back to 3 features
print("reconstruction MSE with 1 component:", round(float(np.mean((X - X_rec) ** 2)), 4))
print("original feature variance:          ", X.var(axis=0).round(3))
` },
      { type: 'check', question: { type: 'mcq', q: 'Explained variance ratios are [0.72, 0.18, 0.06, 0.04]. How many components keep at least 90% of the variance?', options: ['1', '2', '3', '4'], answer: 1, explain: '0.72 + 0.18 = 0.90 — two components.' } },
      { type: 'text', title: 'Using PCA well', md: `**Good uses:**
- **Visualization** — project to 2-D or 3-D to eyeball clusters and outliers.
- **Preprocessing** — fewer, decorrelated features for models that struggle with many correlated ones; faster training.
- **Denoising/compression** — drop low-variance components, which are often noise.

**Choosing k:** keep enough components for ~90–99% of the variance, look for an elbow in the explained-variance plot, or choose k by validation performance of the downstream model.

**Limitations:**
- **Linear** — curved structure (a "Swiss roll") needs non-linear methods: t-SNE or UMAP for visualization, autoencoders for learned compression.
- **Variance ≠ importance** — the highest-variance direction may be irrelevant to your label (PCA is unsupervised).
- **Scale-sensitive** — standardize unless features share units.
- **Fit on training data only**, like any preprocessing.

:::real The low-rank idea everywhere
"Most of the information lives in a few directions" is the same insight behind LoRA fine-tuning of LLMs (weight updates are low-rank) and embedding compression.
:::` },
      { type: 'check', question: { type: 'tf', q: 'PCA requires class labels to find the principal components.', answer: false, explain: 'PCA is unsupervised; it only uses the features\' variance.' } }
    ],
    when: `- Many correlated numeric features; visualization of high-dimensional data; speeding up or regularizing downstream models; noise reduction.
- Standardize first (unless features share units and scale matters).
- Prefer feature selection when you need to keep original, interpretable features; non-linear methods for curved structure.`,
    pitfalls: `- Forgetting to centre (and usually standardize) the data.
- Fitting PCA on train + test together (leakage).
- Interpreting component signs or treating components as physical causes.
- Assuming high-variance components are the predictive ones.
- Using PCA plots (or t-SNE plots) as proof of separability without validation.`,
    takeaways: [
      'PCA finds orthogonal directions of maximal variance; the first k give the best k-dimensional linear approximation.',
      'Compute it with the SVD of the centred data: components = rows of Vᵀ.',
      'Explained variance ratio of component i = Sᵢ² / ΣSⱼ².',
      'Project with X_c Vᵀ_k; reconstruct with scores·V_k + mean.',
      'Standardize first; PCA is linear and unsupervised.'
    ],
    quiz: [
      { type: 'mcq', q: 'The first principal component is the direction that…', options: ['best separates the classes', 'has the greatest variance of the projected data', 'is aligned with the first feature', 'has the smallest variance'], answer: 1, explain: 'PC1 maximises the variance of the data projected onto it.' },
      { type: 'tf', q: 'Principal components are orthogonal (perpendicular) to each other.', answer: true, explain: 'Each component is orthogonal to all previous ones.' },
      { type: 'mcq', q: 'Why standardize before PCA when features have different units?', options: ['PCA only works on integers', 'Otherwise features with large numeric ranges dominate the variance and therefore the components', 'It makes the SVD faster', 'It is not needed'], answer: 1, explain: 'A feature in grams has 1000× the variance of the same feature in kilograms.' },
      { type: 'fill', q: 'Singular values are S = [3, 2, 1]. What fraction of the variance does the first component explain? (fraction or decimal)', answer: ['9/14', '0.643', '0.64', '0.6429'], display: '9/14 ≈ 0.643', explain: 'Sᵢ²/ΣSⱼ² = 9 / (9 + 4 + 1) = 9/14.' },
      { type: 'mcq', q: 'Which method would better reveal curved, non-linear structure for visualization?', options: ['PCA with more components', 'UMAP or t-SNE', 'Linear regression', 'One-hot encoding'], answer: 1, explain: 'They are non-linear embedding methods designed for visualization.' }
    ],
    problems: [
      {
        id: 'pca-svd', title: 'PCA via SVD', difficulty: 'medium',
        prompt: `Implement \`pca(X, k)\` returning \`(components, projected, ratio)\`:

1. Centre X: \`Xc = X - X.mean(axis=0)\`.
2. \`U, S, Vt = np.linalg.svd(Xc, full_matrices=False)\`.
3. \`components = Vt[:k]\` — then fix the arbitrary signs: **flip each component so that its entry with the largest absolute value is positive**.
4. \`projected = Xc @ components.T\` (shape (n, k)).
5. \`ratio = S[:k] ** 2 / np.sum(S ** 2)\`.

The sign rule makes the output deterministic so it can be checked.`,
        starter: `
import numpy as np

def pca(X, k):
    X = np.asarray(X, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def pca(X, k):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k].copy()
    for i in range(k):
        j = np.argmax(np.abs(components[i]))
        if components[i, j] < 0:
            components[i] = -components[i]
    projected = Xc @ components.T
    ratio = S[:k] ** 2 / np.sum(S ** 2)
    return components, projected, ratio
`,
        hints: ['`np.argmax(np.abs(v))` finds the position of the largest-magnitude entry.', 'Flip the whole component (multiply by −1) if that entry is negative.', 'Project the *centred* data.'],
        explain: `Because v and −v span the same direction, SVD routines may return either; a sign convention makes results reproducible (scikit-learn applies a similar one). Everything else is a direct translation of X_c = USVᵀ.`,
        tests: [
          { name: 'line along a diagonal', call: 'pca(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), 1)[2]', expect: 'np.array([1.0])' },
          { name: 'components are unit length and orthogonal', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(50, 4)) @ rng.normal(size=(4, 4))', code: 'comps, proj, ratio = pca(X, 3)\nassert comps.shape == (3, 4) and proj.shape == (50, 3)\nassert np.allclose(comps @ comps.T, np.eye(3))\nassert np.all(np.diff(ratio) <= 1e-12), "ratios should be sorted in decreasing order"' },
          { name: 'exact output on correlated data', setup: 'rng = np.random.default_rng(0)\nt = rng.normal(size=100)\nX = np.column_stack([2 * t, -t, 0.5 * t]) + rng.normal(0, 0.2, (100, 3))', call: 'pca(X, 2)' },
          { name: 'random data vs reference', setup: 'rng = np.random.default_rng(9)\nX = rng.normal(size=(30, 5))', call: 'pca(X, 3)', hidden: true }
        ]
      },
      {
        id: 'explained-variance', title: 'How Many Components?', difficulty: 'easy',
        prompt: `Implement \`components_for_variance(ratios, threshold)\`: given explained variance ratios sorted from largest to smallest, return the **smallest k** whose cumulative sum reaches at least \`threshold\` (allow a tiny floating-point tolerance of 1e-12). If even all of them fall short, return \`len(ratios)\`.

\`\`\`text
components_for_variance([0.72, 0.18, 0.06, 0.04], 0.90)  ->  2
\`\`\``,
        starter: `
import numpy as np

def components_for_variance(ratios, threshold):
    # TODO: cumulative sum, then the first index that reaches the threshold
    pass
`,
        solution: `
import numpy as np

def components_for_variance(ratios, threshold):
    cum = np.cumsum(ratios)
    hits = np.nonzero(cum >= threshold - 1e-12)[0]
    return int(hits[0]) + 1 if len(hits) else len(ratios)
`,
        hints: ['`np.cumsum` gives running totals.', 'The first position where the running total reaches the threshold, plus one, is k.'],
        explain: `The cumulative explained variance curve is the standard tool for picking k. The tolerance guards against cases like 0.72 + 0.18 = 0.8999999999999999 in floating point.`,
        tests: [
          { call: 'components_for_variance([0.72, 0.18, 0.06, 0.04], 0.90)', expect: '2' },
          { call: 'components_for_variance([0.5, 0.3, 0.2], 0.4)', expect: '1' },
          { name: 'needs everything', call: 'components_for_variance([0.25, 0.25, 0.25, 0.25], 1.0)', expect: '4', hidden: true },
          { name: 'threshold not reachable', call: 'components_for_variance([0.4, 0.3], 0.95)', expect: '2', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Decomposing signals in components (PCA)', url: 'https://scikit-learn.org/stable/modules/decomposition.html#pca' },
      { title: 'Setosa — Principal Component Analysis explained visually', url: 'https://setosa.io/ev/principal-component-analysis/' }
    ]
  },

  // =====================================================================================
  {
    id: 'sklearn-workflow', branch: 'ml', title: 'The scikit-learn Workflow: Pipelines & Model Selection', short: 'scikit-learn Workflow', level: 2, minutes: 35, prereqs: ['evaluation-metrics', 'cross-validation', 'feature-engineering'],
    tags: ['scikit-learn', 'pipeline', 'columntransformer', 'gridsearchcv', 'joblib'],
    summary: 'How classical ML is done in practice: the estimator API, leak-proof pipelines, cross-validation and grid search in a few lines, and saving models.',
    why: `scikit-learn is the standard library for classical machine learning in Python, and its design — every model with \`fit\`, \`predict\` and \`transform\` — has been copied by countless other libraries. Its **Pipeline** object solves the data-leakage problem structurally: preprocessing is fitted inside each cross-validation fold automatically. Knowing this workflow turns everything you have learned so far into reliable, production-ready code.`,
    objectives: [
      'Use the estimator API: fit, predict, predict_proba, transform, score',
      'Build a Pipeline (and a ColumnTransformer for mixed column types)',
      'Evaluate with cross_val_score and tune with GridSearchCV',
      'Save and load a trained pipeline'
    ],
    sections: [
      { type: 'text', title: 'One API for everything', md: `Every scikit-learn object follows the same conventions:

| Method | Who has it | What it does |
| --- | --- | --- |
| \`fit(X, y)\` | all estimators | learn from data (y optional for unsupervised) |
| \`predict(X)\` | models | predictions for new rows |
| \`predict_proba(X)\` | most classifiers | class probabilities |
| \`transform(X)\` | preprocessors | e.g. scaled or encoded features |
| \`fit_transform(X)\` | preprocessors | fit and transform in one go |
| \`score(X, y)\` | models | default metric (accuracy / R²) |

Learned attributes end with an underscore (\`coef_\`, \`mean_\`, \`classes_\`); hyperparameters are constructor arguments. Inputs are NumPy arrays or pandas DataFrames with shape (n_samples, n_features).` },
      { type: 'text', title: 'Pipelines prevent leakage', md: `A **Pipeline** chains preprocessing steps and a final model into one estimator:

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)      # fits the scaler on X_train, then the classifier
model.predict(X_test)            # transforms X_test with the *training* statistics
\`\`\`

Pass the pipeline to \`cross_val_score\` or \`GridSearchCV\` and the scaler is re-fitted on each fold's training part — no leakage, by construction.

For tables with mixed types, a **ColumnTransformer** routes columns to different preprocessing: standardize numeric columns, one-hot encode categorical ones (with \`handle_unknown="ignore"\`), impute missing values — all inside the pipeline.` },
      { type: 'code', title: 'A complete, leak-free workflow', md: `(The first run downloads scikit-learn into the browser.)`, code: `
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
scores = cross_val_score(pipe, X_train, y_train, cv=5)
print("5-fold CV accuracy: %.3f ± %.3f" % (scores.mean(), scores.std()))

search = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=5)   # step__parameter
search.fit(X_train, y_train)
print("best C:", search.best_params_["clf__C"], " CV accuracy: %.3f" % search.best_score_)
print("final test accuracy: %.3f" % search.score(X_test, y_test))     # touched once
` },
      { type: 'check', question: { type: 'mcq', q: 'Why pass a whole Pipeline (scaler + model) to cross_val_score instead of scaling X first?', options: ['It runs faster', 'The scaler is then fitted only on each fold\'s training part, preventing leakage', 'cross_val_score cannot handle scaled data', 'It changes the model type'], answer: 1, explain: 'Scaling before splitting lets validation-fold statistics leak into training.' } },
      { type: 'text', title: 'Saving models and choosing among them', md: `**Persisting**: save the whole fitted pipeline so preprocessing and model travel together.

\`\`\`python
import joblib
joblib.dump(search.best_estimator_, "model.joblib")
model = joblib.load("model.joblib")          # same library versions recommended
\`\`\`

**A sensible model-selection routine for tabular data:**
1. Baseline: \`DummyClassifier\` / \`DummyRegressor\`.
2. A regularized linear model in a pipeline with scaling.
3. A tree ensemble (\`RandomForest\`, \`HistGradientBoosting\`).
4. Compare with the same cross-validation folds; tune the best one or two.
5. Evaluate the winner once on the test set; save the pipeline with its version info.

Set \`random_state\` everywhere randomness appears so results are reproducible.` },
      { type: 'text', title: 'Going further: mixed columns, tuning and saving models', md: `- **Mixed column types**: \`ColumnTransformer\` applies different preprocessing to numeric and categorical columns inside one pipeline (the tabular capstone uses it end to end).
- **Tuning**: \`GridSearchCV\` and \`RandomizedSearchCV\` search pipeline parameters named step__parameter, such as logisticregression__C; successive-halving searches drop weak candidates early.
- **Nested cross-validation**: an inner loop tunes, an outer loop estimates how well the whole tuning procedure generalizes — the honest number to report when there is no separate test set.
- **Your own steps**: \`FunctionTransformer\`, or a small class with fit and transform, puts custom feature code inside the pipeline so it runs identically in training and in production.
- **Saving**: \`joblib.dump(pipeline, "model.joblib")\` stores the fitted pipeline. Pickle-based files can run code when loaded, so only load files you trust, and record the library versions used.` }
    ],
    when: `- Any classical ML project on tabular or text-feature data — the default toolbox.
- Use pipelines always; ColumnTransformer for mixed column types.
- Move to PyTorch/JAX for deep learning, and to XGBoost/LightGBM/CatBoost for maximum tabular accuracy (they offer scikit-learn-compatible APIs too).`,
    pitfalls: `- Preprocessing outside the pipeline (leakage in CV).
- Forgetting \`handle_unknown="ignore"\` in OneHotEncoder → errors on new categories in production.
- Hyperparameter names in grids must be \`step__param\`.
- Loading a pickled model with a different scikit-learn version.
- Comparing models evaluated on different splits.`,
    takeaways: [
      'Every estimator shares fit / predict / transform / score.',
      'Pipelines bundle preprocessing with the model and prevent leakage in CV.',
      'ColumnTransformer handles numeric and categorical columns differently.',
      'cross_val_score and GridSearchCV make honest evaluation and tuning short.',
      'Save the whole fitted pipeline (joblib) and pin library versions.'
    ],
    quiz: [
      { type: 'mcq', q: 'In GridSearchCV over a Pipeline with a step named "clf", how do you refer to the classifier\'s C parameter?', options: ['C', 'clf.C', 'clf__C', 'pipeline.C'], answer: 2, explain: 'scikit-learn uses step__parameter with a double underscore.' },
      { type: 'mcq', q: 'Which method returns class probabilities?', options: ['predict', 'predict_proba', 'transform', 'score'], answer: 1, explain: 'predict returns labels; predict_proba returns probabilities.' },
      { type: 'tf', q: 'A fitted attribute like `coef_` ends with an underscore by convention.', answer: true, explain: 'Trailing underscores mark values learned during fit.' },
      { type: 'mcq', q: 'What does a ColumnTransformer let you do?', options: ['Train several models at once', 'Apply different preprocessing to different columns within one pipeline', 'Plot columns', 'Remove the need for cross-validation'], answer: 1, explain: 'E.g. scale numeric columns and one-hot encode categorical ones.' },
      { type: 'mcq', q: 'What should you save to deploy a model trained with scaling?', options: ['Only the classifier', 'The whole fitted pipeline (scaler + model)', 'Only the training data', 'The test set predictions'], answer: 1, explain: 'New data must be preprocessed exactly like the training data.' }
    ],
    problems: [
      {
        id: 'sklearn-pipeline', title: 'Build and Evaluate a Pipeline', difficulty: 'medium',
        prompt: `This problem uses real **scikit-learn** (it downloads into the browser the first time — about 20 MB — then it's cached).

Implement:
1. \`make_model(C=1.0)\` → a \`Pipeline\` with two steps named **"scale"** (\`StandardScaler()\`) and **"clf"** (\`LogisticRegression(C=C, max_iter=5000)\`).
2. \`cv_accuracy(model, X, y)\` → the **mean** of \`cross_val_score(model, X, y, cv=5)\` as a float.
3. \`best_C(X, y, Cs)\` → the C from \`Cs\` with the highest mean CV accuracy (ties → the first).

The tests use the breast-cancer dataset that ships with scikit-learn.`,
        starter: `
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def make_model(C=1.0):
    # TODO
    pass

def cv_accuracy(model, X, y):
    # TODO
    pass

def best_C(X, y, Cs):
    # TODO
    pass
`,
        solution: `
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def make_model(C=1.0):
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=5000))])

def cv_accuracy(model, X, y):
    return float(cross_val_score(model, X, y, cv=5).mean())

def best_C(X, y, Cs):
    best, best_score = None, -1.0
    for C in Cs:
        score = cv_accuracy(make_model(C), X, y)
        if score > best_score:
            best, best_score = C, score
    return best
`,
        hints: ['`Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(...))])`.', '`cross_val_score` returns an array of 5 scores — take `.mean()`.', 'Loop over Cs, keep the strictly best score so ties keep the first.'],
        explain: `Because the scaler lives inside the pipeline, \`cross_val_score\` refits it on each fold's training portion — your CV estimate stays honest. \`best_C\` is a hand-written GridSearchCV; in practice you would call \`GridSearchCV(make_model(), {"clf__C": Cs}, cv=5)\`.`,
        tests: [
          { name: 'pipeline structure', code: 'm = make_model(0.5)\nfrom sklearn.pipeline import Pipeline\nassert isinstance(m, Pipeline), "make_model must return a Pipeline"\nassert [name for name, _ in m.steps] == ["scale", "clf"]\nassert type(m.named_steps["scale"]).__name__ == "StandardScaler"\nassert m.named_steps["clf"].C == 0.5' },
          { name: 'CV accuracy on breast cancer', setup: 'from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)', call: 'cv_accuracy(make_model(1.0), X, y)', tol: 1e-9 },
          { name: 'accuracy is high', setup: 'from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)', code: 'acc = cv_accuracy(make_model(1.0), X, y)\nassert acc > 0.95, "expected CV accuracy above 0.95, got %.3f" % acc', hidden: true },
          { name: 'best C', setup: 'from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)', call: 'best_C(X, y, [0.001, 0.1, 1.0])', hidden: true }
        ]
      },
      {
        id: 'grid-search-pipeline', title: 'Tune a Pipeline with GridSearchCV', difficulty: 'medium',
        prompt: `Tuning inside a pipeline keeps preprocessing inside the cross-validation folds. With scikit-learn, implement:

1. \`make_pipe()\` → \`make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))\`.
2. \`tune(X, y, Cs)\` → \`(best_C, best_score)\`: run \`GridSearchCV(make_pipe(), {"logisticregression__C": Cs}, cv=5)\` on (X, y) and return \`best_params_["logisticregression__C"]\` and \`best_score_\` (a float).
3. \`final_test_accuracy(X_train, y_train, X_test, y_test, Cs)\` → tune on the training data only, refit is done for you by GridSearchCV, and return the accuracy of the tuned search object on the test data (\`search.score(X_test, y_test)\`).

Step names in a \`make_pipeline\` are the lower-cased class names, which is why the parameter is called \`logisticregression__C\`.`,
        starter: `
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def make_pipe():
    # TODO
    pass

def tune(X, y, Cs):
    # TODO
    pass

def final_test_accuracy(X_train, y_train, X_test, y_test, Cs):
    # TODO
    pass
`,
        solution: `
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def make_pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

def _search(X, y, Cs):
    search = GridSearchCV(make_pipe(), {"logisticregression__C": list(Cs)}, cv=5)
    return search.fit(X, y)

def tune(X, y, Cs):
    search = _search(X, y, Cs)
    return search.best_params_["logisticregression__C"], float(search.best_score_)

def final_test_accuracy(X_train, y_train, X_test, y_test, Cs):
    return float(_search(X_train, y_train, Cs).score(X_test, y_test))
`,
        hints: ['Grid keys use the step name, two underscores, then the parameter name.', 'GridSearchCV refits the best configuration on all the data it was given (refit=True by default).', 'Never pass the test set to the search.'],
        explain: `GridSearchCV performs k-fold cross-validation for every candidate, keeps the best, and refits it on all the training data. Its best_score_ is still optimistic (it is the maximum over candidates), which is why the final, untouched test score is reported separately.`,
        tests: [
          { setup: 'from sklearn.datasets import make_classification\nX, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)', call: '[type(make_pipe()).__name__, [name for name, _ in make_pipe().steps]]', expect: '["Pipeline", ["standardscaler", "logisticregression"]]' },
          { setup: 'from sklearn.datasets import make_classification\nX, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)', call: 'tune(X, y, [0.01, 0.1, 1.0, 10.0])' },
          { name: 'tuned on train, scored on test', setup: 'from sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nX, y = load_breast_cancer(return_X_y=True)\nXtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)', call: 'final_test_accuracy(Xtr, ytr, Xte, yte, [0.01, 0.1, 1.0])', hidden: true },
          { name: 'the best score is a cross-validated accuracy', setup: 'from sklearn.datasets import make_classification\nX, y = make_classification(n_samples=150, n_features=6, random_state=1)', code: 'C, s = tune(X, y, [0.1, 1.0])\nassert C in (0.1, 1.0) and 0.5 < s <= 1.0', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Getting started', url: 'https://scikit-learn.org/stable/getting_started.html' },
      { title: 'scikit-learn — Pipelines and composite estimators', url: 'https://scikit-learn.org/stable/modules/compose.html' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
