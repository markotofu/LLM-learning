/* ML Atlas – content: Classical ML (evaluation, validation, kNN, trees, ensembles) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'evaluation-metrics', branch: 'ml', title: 'Evaluation Metrics: Beyond Accuracy', short: 'Evaluation Metrics', level: 1, minutes: 45, prereqs: ['logistic-regression'],
    tags: ['accuracy', 'precision', 'recall', 'f1', 'roc', 'auc', 'confusion matrix', 'rmse'],
    summary: 'How to measure whether a model is actually good: confusion matrices, precision and recall, F1, ROC/AUC, thresholds, and the right metric for regression.',
    why: `"95% accurate" can mean an excellent model or a useless one that always predicts the majority class. Real decisions — approving loans, flagging fraud, screening for disease, filtering spam — have asymmetric costs: a missed cancer is not the same as a false alarm. Choosing the right metric, and the right decision threshold, is how you align a model with what actually matters. It is also how you compare models honestly, report results credibly, and avoid shipping something that looks good on paper only.`,
    objectives: [
      'Build a confusion matrix and compute accuracy, precision, recall and F1',
      'Explain why accuracy misleads on imbalanced data',
      'Use the decision threshold to trade precision against recall',
      'Interpret ROC curves and AUC, and know when precision-recall curves are better',
      'Pick appropriate regression metrics (MAE, RMSE, R², MAPE)'
    ],
    sections: [
      { type: 'text', title: 'The confusion matrix', md: `For binary classification every prediction falls into one of four boxes:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actually positive** | **TP** — true positive (caught it) | **FN** — false negative (missed it) |
| **Actually negative** | **FP** — false positive (false alarm) | **TN** — true negative (correctly ignored) |

Everything else is computed from these four counts:

| Metric | Formula | Question it answers |
| --- | --- | --- |
| **Accuracy** | (TP + TN) / all | How often is the model right? |
| **Precision** | TP / (TP + FP) | When it says "positive", how often is it right? |
| **Recall** (sensitivity, TPR) | TP / (TP + FN) | Of all real positives, how many does it catch? |
| **Specificity** | TN / (TN + FP) | Of all real negatives, how many does it leave alone? |
| **False positive rate** | FP / (FP + TN) | How often does it raise a false alarm? |
| **F1 score** | 2·P·R / (P + R) | A single balance of precision and recall (harmonic mean) |` },
      { type: 'text', title: 'Why accuracy misleads', md: `A fraud dataset with 1% fraud: a "model" that always says "legitimate" scores **99% accuracy** — and catches zero fraud (recall 0). Whenever classes are imbalanced, accuracy is dominated by the majority class.

Which metric to emphasise depends on the cost of each mistake:

- **Recall matters most** when missing a positive is costly: disease screening, fraud, safety defects. (Better to send a few healthy people for a second test.)
- **Precision matters most** when false alarms are costly: spam filtering (losing a real email), auto-blocking accounts, recommending something to a user.
- **F1** when you need one number balancing both and have no clear cost ratio.

:::analogy A fishing net
Recall: what fraction of the fish in the lake did your net catch? Precision: what fraction of what's in your net is actually fish (rather than boots and seaweed)? A bigger net raises recall and lowers precision.
:::` },
      { type: 'demo', title: 'Interactive: move the threshold, watch the trade-off', md: `A classifier outputs scores; you choose the threshold. Slide it and watch the confusion matrix, precision, recall and the point on the ROC curve.`, demo: 'threshold' },
      { type: 'text', title: 'ROC curves and AUC', md: `Sweeping the threshold from 1 down to 0 traces the **ROC curve**: the true positive rate (recall) against the false positive rate. The diagonal is random guessing; the top-left corner is perfection.

**AUC** (area under the ROC curve) summarises it in one threshold-free number:
- 1.0 = perfect ranking, 0.5 = random, below 0.5 = worse than random (flip the predictions!).
- Elegant interpretation: **AUC is the probability that a randomly chosen positive gets a higher score than a randomly chosen negative.** It measures how well the model *ranks*, regardless of threshold or calibration.

On **highly imbalanced** data ROC can look rosy because the false positive rate stays small even with many false alarms (the negatives are so numerous). The **precision-recall curve** and its area (average precision) are more informative there.` },
      { type: 'code', title: 'Metrics from scratch, then with scikit-learn', code: `
import numpy as np

y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
scores = np.array([0.9, 0.8, 0.35, 0.6, 0.2, 0.1, 0.4, 0.05, 0.7, 0.3])

def metrics_at(threshold):
    pred = (scores >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1)); fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1)); tn = np.sum((pred == 0) & (y_true == 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, tn, precision, recall, f1

for t in [0.75, 0.5, 0.3]:
    tp, fp, fn, tn, p, r, f1 = metrics_at(t)
    print(f"threshold {t}: TP={tp} FP={fp} FN={fn} TN={tn}  precision={p:.2f} recall={r:.2f} F1={f1:.2f}")

# AUC as "probability a random positive outranks a random negative"
pos, neg = scores[y_true == 1], scores[y_true == 0]
auc = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
print("AUC:", round(auc, 3))
` },
      { type: 'check', question: { type: 'mcq', q: 'A cancer screening test should miss as few cancers as possible, even at the cost of extra follow-up tests. Which metric should it prioritise?', options: ['Precision', 'Recall', 'Accuracy', 'Specificity'], answer: 1, explain: 'Missing a positive (false negative) is the costly error, so maximise recall — accept lower precision.' } },
      { type: 'text', title: 'Regression metrics and multi-class averaging', md: `**Regression:**

| Metric | Meaning | Note |
| --- | --- | --- |
| **MAE** | average absolute error | in the target's units; robust; easy to explain |
| **RMSE** | √MSE | same units; punishes large errors more |
| **R²** | 1 − SS_res/SS_tot | unit-free; 0 = mean baseline |
| **MAPE** | mean of \|error\| / \|y\| | relative error; explodes when y ≈ 0 |

**Multi-class** precision/recall/F1 are computed per class (one-vs-rest) and then averaged:
- **macro average** — the plain mean over classes: every class counts equally (good for rare classes);
- **weighted average** — weighted by class frequency;
- **micro average** — pool all decisions first (equals accuracy for single-label problems).

Report a confusion matrix too: it shows *which* classes get confused with which.` },
      { type: 'check', question: { type: 'fill', q: 'TP = 30, FP = 10, FN = 20. What is the precision? (decimal)', answer: ['0.75', '.75', '3/4'], explain: 'TP / (TP + FP) = 30 / 40 = 0.75.' } }
    ],
    when: `- **Balanced classes, equal costs**: accuracy is fine (plus a confusion matrix).
- **Imbalanced classes**: precision, recall, F1, PR-AUC; never accuracy alone.
- **Ranking quality / comparing models independently of threshold**: ROC-AUC (or PR-AUC when positives are rare).
- **Choosing the operating point**: pick the threshold on validation data by cost, or by a target such as "recall ≥ 95%".
- **Regression**: MAE for interpretability, RMSE when big errors are especially bad, R² for a unit-free summary.`,
    pitfalls: `- Reporting accuracy on imbalanced data.
- Choosing the threshold on the test set.
- Comparing F1 scores computed with different averaging (macro vs weighted).
- Treating AUC as accuracy — a model can have high AUC and still be badly calibrated or use a poor threshold.
- Ignoring the confidence interval: a 1-point difference on a small test set is noise.`,
    takeaways: [
      'Everything starts from the confusion matrix: TP, FP, FN, TN.',
      'Precision = TP/(TP+FP): trust in positive predictions. Recall = TP/(TP+FN): coverage of real positives.',
      'The threshold trades precision against recall — choose it by the cost of each mistake.',
      'AUC = probability a random positive outranks a random negative; PR curves suit rare positives.',
      'Accuracy alone is misleading on imbalanced data.'
    ],
    quiz: [
      { type: 'fill', q: 'TP = 40, FN = 10. What is the recall? (decimal)', answer: ['0.8', '.8', '0.80', '4/5'], explain: 'TP / (TP + FN) = 40 / 50 = 0.8.' },
      { type: 'mcq', q: 'Raising the decision threshold of a classifier usually…', options: ['raises recall and lowers precision', 'raises precision and lowers recall', 'raises both', 'changes nothing'], answer: 1, explain: 'Fewer, more confident positive predictions: fewer false alarms (precision ↑) but more misses (recall ↓).' },
      { type: 'mcq', q: 'An AUC of 0.5 means…', options: ['the model is perfect', 'the model ranks positives above negatives no better than chance', 'half the predictions are correct', 'precision is 0.5'], answer: 1, explain: 'AUC is the probability that a random positive outranks a random negative; 0.5 is a coin flip.' },
      { type: 'fill', q: 'Precision 0.5 and recall 1.0. What is the F1 score? (decimal, 2 places)', answer: ['0.67', '0.667', '2/3', '0.6667'], explain: '2·0.5·1/(0.5 + 1) = 1/1.5 ≈ 0.67 — the harmonic mean is pulled toward the smaller value.' },
      { type: 'mcq', q: 'For a dataset with 0.1% positives, which evaluation is most informative?', options: ['Accuracy', 'Precision-recall curve / average precision', 'R²', 'Training loss'], answer: 1, explain: 'With extreme imbalance accuracy and even ROC can look great; precision-recall focuses on the rare positive class.' }
    ],
    problems: [
      {
        id: 'confusion-counts', title: 'Confusion Matrix Counts', difficulty: 'easy',
        prompt: `Implement \`confusion(y_true, y_pred)\` for binary labels (0/1). Return a dict with integer counts:

\`{"tp": ..., "fp": ..., "fn": ..., "tn": ...}\`

\`\`\`text
confusion([1, 0, 1, 1, 0], [1, 1, 0, 1, 0])  ->  {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
\`\`\``,
        starter: `
import numpy as np

def confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # TODO: count each combination with boolean masks
    return {"tp": None, "fp": None, "fn": None, "tn": None}
`,
        solution: `
import numpy as np

def confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "tp": int(np.sum((y_pred == 1) & (y_true == 1))),
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1))),
        "tn": int(np.sum((y_pred == 0) & (y_true == 0))),
    }
`,
        hints: ['Combine masks with `&`, e.g. `(y_pred == 1) & (y_true == 1)`, and remember the parentheses.', '`np.sum` of a boolean array counts the Trues; wrap in `int(...)`.'],
        explain: `Each count is the size of an intersection of two boolean masks. Every other classification metric is a ratio of these four numbers, so getting them right — including the operator precedence trap without parentheses — is the foundation.`,
        tests: [
          { call: 'confusion([1, 0, 1, 1, 0], [1, 1, 0, 1, 0])', expect: '{"tp": 2, "fp": 1, "fn": 1, "tn": 1}' },
          { call: 'confusion([0, 0, 0], [0, 0, 0])', expect: '{"tp": 0, "fp": 0, "fn": 0, "tn": 3}' },
          { name: 'random labels', setup: 'rng = np.random.default_rng(3)\nyt = rng.integers(0, 2, 200)\nyp = rng.integers(0, 2, 200)', call: 'confusion(yt, yp)', hidden: true },
          { name: 'counts add up', setup: 'rng = np.random.default_rng(4)\nyt = rng.integers(0, 2, 77)\nyp = rng.integers(0, 2, 77)', code: 'c = confusion(yt, yp)\nassert sum(c.values()) == 77', hidden: true }
        ]
      },
      {
        id: 'precision-recall-f1', title: 'Precision, Recall and F1', difficulty: 'easy',
        prompt: `Implement \`precision_recall_f1(y_true, y_pred)\` for binary labels, returning a tuple of floats \`(precision, recall, f1)\`.

- precision = TP / (TP + FP), recall = TP / (TP + FN), f1 = 2·P·R / (P + R)
- Whenever a denominator is 0, that metric is **0.0** (this is scikit-learn's convention with \`zero_division=0\`).

\`\`\`text
precision_recall_f1([1, 0, 1, 1, 0], [1, 1, 0, 1, 0])  ->  (0.6667, 0.6667, 0.6667)
\`\`\``,
        starter: `
import numpy as np

def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # TODO
    pass
`,
        solution: `
import numpy as np

def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)
`,
        hints: ['Compute TP, FP and FN first.', 'Guard each division: `x / y if y > 0 else 0.0`.'],
        explain: `F1 is the harmonic mean of precision and recall: it is high only when **both** are high, unlike the arithmetic mean which a single strong number can prop up. The zero-division guards matter in practice — a model that never predicts positive has undefined precision.`,
        tests: [
          { call: 'precision_recall_f1([1, 0, 1, 1, 0], [1, 1, 0, 1, 0])', expect: '(2 / 3, 2 / 3, 2 / 3)' },
          { call: 'precision_recall_f1([1, 1, 0, 0], [1, 0, 0, 0])', expect: '(1.0, 0.5, 2 / 3)' },
          { name: 'never predicts positive', call: 'precision_recall_f1([1, 0, 1], [0, 0, 0])', expect: '(0.0, 0.0, 0.0)', hidden: true },
          { name: 'random labels', setup: 'rng = np.random.default_rng(6)\nyt = rng.integers(0, 2, 300)\nyp = rng.integers(0, 2, 300)', call: 'precision_recall_f1(yt, yp)', hidden: true }
        ]
      },
      {
        id: 'roc-auc', title: 'ROC AUC by Ranking', difficulty: 'medium',
        prompt: `Implement \`roc_auc(y_true, scores)\` using AUC's probabilistic meaning:

> AUC = the probability that a randomly chosen positive example has a higher score than a randomly chosen negative one, counting ties as ½.

Compare every positive with every negative (NumPy broadcasting makes this one line: \`pos[:, None]\` vs \`neg[None, :]\`) and return the average as a float.

\`\`\`text
roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])  ->  0.75
\`\`\`
(Positive 0.35 beats one of the two negatives; 0.8 beats both → 3 of 4 pairs.)`,
        starter: `
import numpy as np

def roc_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    # TODO: fraction of (positive, negative) pairs where the positive scores higher (ties = 0.5)
    pass
`,
        solution: `
import numpy as np

def roc_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    greater = (pos[:, None] > neg[None, :]).astype(float)
    ties = (pos[:, None] == neg[None, :]).astype(float)
    return float(np.mean(greater + 0.5 * ties))
`,
        hints: ['`pos[:, None] > neg[None, :]` is a (n_pos, n_neg) table of comparisons.', 'Add 0.5 for every tie, then take the mean over all pairs.'],
        explain: `This pairwise definition is mathematically equal to the area under the ROC curve (it is the Mann–Whitney U statistic, normalised). It makes clear that AUC only cares about **ordering** — multiplying all scores by 10 or squaring them doesn't change it. For large datasets, a sort-based O(n log n) version is used instead of all pairs.`,
        tests: [
          { call: 'roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])', expect: '0.75' },
          { name: 'perfect ranking', call: 'roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])', expect: '1.0' },
          { name: 'ties count half', call: 'roc_auc([0, 1], [0.5, 0.5])', expect: '0.5' },
          { name: 'ordering only matters', setup: 'rng = np.random.default_rng(11)\ny = rng.integers(0, 2, 120)\ns = rng.random(120) + 0.3 * y', code: 'a = roc_auc(y, s)\nassert abs(a - roc_auc(y, s ** 3 * 10)) < 1e-12, "AUC must not change under a monotonic transform"', hidden: true },
          { name: 'random scores', setup: 'rng = np.random.default_rng(2)\ny = rng.integers(0, 2, 150)\ns = np.round(rng.random(150) + 0.4 * y, 1)', call: 'roc_auc(y, s)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Google ML Crash Course — Classification: accuracy, precision, recall, ROC/AUC', url: 'https://developers.google.com/machine-learning/crash-course/classification' },
      { title: "scikit-learn — Metrics and scoring", url: "https://scikit-learn.org/stable/modules/model_evaluation.html" }
    ]
  },

  // =====================================================================================
  {
    id: 'cross-validation', branch: 'ml', title: 'Validation, Cross-Validation & Hyperparameter Tuning', short: 'Cross-Validation & Tuning', level: 2, minutes: 40, prereqs: ['overfitting', 'data-preparation'],
    tags: ['k-fold', 'hyperparameters', 'grid search', 'random search', 'model selection'],
    summary: 'Choosing models and hyperparameters without fooling yourself: validation sets, k-fold cross-validation, grid and random search, and the final test.',
    why: `Every model has knobs you set before training — learning rate, regularization strength, tree depth, number of neighbours. Pick them by looking at the test set and your reported score becomes optimistic fiction; pick them on a single small validation split and you may be choosing by luck. **Cross-validation** gives stable estimates from limited data, and a disciplined tuning procedure gives you numbers you can trust. This is how real projects choose models.`,
    objectives: [
      'Distinguish parameters (learned) from hyperparameters (chosen)',
      'Run k-fold cross-validation and report mean ± standard deviation',
      'Use stratified, group and time-series variants appropriately',
      'Tune hyperparameters with grid or random search without leaking the test set'
    ],
    sections: [
      { type: 'text', title: 'Parameters vs hyperparameters', md: `- **Parameters** are learned from data by the training algorithm: weights, biases, tree split thresholds.
- **Hyperparameters** are chosen *before* training and control the learning process or model capacity: learning rate, λ, max tree depth, k in kNN, number of layers, batch size.

You can't learn hyperparameters from the training loss (it would always prefer the most flexible setting, i.e. overfitting), so you choose them by **validation performance**.` },
      { type: 'text', title: 'k-fold cross-validation', md: `With a single validation split, your score depends on which examples happened to land in it — noisy, especially for small datasets. **k-fold cross-validation** uses all the data for validation, once each:

1. Shuffle and split the training data into **k folds** of roughly equal size (k = 5 or 10 is typical).
2. For each fold i: train on the other k − 1 folds, evaluate on fold i.
3. Report the **mean** of the k scores (and their standard deviation as a sense of uncertainty).

Cost: k training runs. Cheap for classical models; often too expensive for big neural networks, which use a single validation set instead.

Variants:
- **Stratified k-fold** — each fold keeps the class proportions (default for classification).
- **Group k-fold** — all examples from one patient/user/document stay in the same fold.
- **Time-series split** — train on the past, validate on the next period, roll forward. Never shuffle time.
- **Leave-one-out** — k = n; nearly unbiased but expensive and high-variance.` },
      { type: 'demo', title: 'Diagram: five folds and a locked test set', demo: 'dg-cv' },
      { type: 'code', title: 'k-fold cross-validation from scratch', code: `
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(60, 3)); y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.5, 60)

def ridge(X, y, lam):
    xm, ym = X.mean(axis=0), y.mean()
    w = np.linalg.solve((X - xm).T @ (X - xm) + lam * np.eye(X.shape[1]), (X - xm).T @ (y - ym))
    return w, ym - xm @ w

folds = np.array_split(np.random.default_rng(1).permutation(len(X)), 5)
for lam in [0.0, 1.0, 10.0, 100.0]:
    scores = []
    for i in range(5):
        val = folds[i]; train = np.concatenate([folds[j] for j in range(5) if j != i])
        w, b = ridge(X[train], y[train], lam)
        scores.append(np.mean((X[val] @ w + b - y[val]) ** 2))
    print(f"lambda={lam:<6} CV MSE = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
` },
      { type: 'check', question: { type: 'mcq', q: 'Why use k-fold cross-validation instead of one validation split on a small dataset?', options: ['It trains the final model faster', 'Every example is used for validation once, giving a less noisy estimate', 'It removes the need for a test set', 'It prevents overfitting during training'], answer: 1, explain: 'Averaging over k different validation folds reduces the luck of a single split.' } },
      { type: 'text', title: 'Searching for good hyperparameters', md: `- **Grid search** — try every combination from lists of values (λ ∈ {0.01, 0.1, 1, 10} × depth ∈ {3, 5, 7}). Simple, exhaustive, explodes combinatorially.
- **Random search** — sample combinations at random. With the same budget it usually beats grid search, because only a few hyperparameters really matter and random sampling tries more distinct values of each.
- **Bayesian optimization / successive halving** (e.g. Optuna, Hyperband) — use earlier results to decide what to try next, or stop bad runs early.

Search on a **log scale** for things like learning rates and λ (1e-4 … 1).

**The full procedure:**
1. Split off a test set and lock it away.
2. On the remaining data, compare candidates with cross-validation (or a validation split).
3. Retrain the best configuration on all non-test data.
4. Evaluate **once** on the test set and report that number.

If you also need an unbiased estimate of the tuning procedure itself, use **nested cross-validation** (an outer CV around the inner search).` },
      { type: 'check', question: { type: 'tf', q: 'It is fine to try 50 hyperparameter settings, pick the one with the best test-set score, and report that score.', answer: false, explain: 'Choosing by test score makes that score optimistically biased — the test set has effectively been used for training decisions.' } },
      { type: 'text', title: 'Going further: choosing the right split', md: `- **Stratified k-fold** keeps the class proportions in every fold — the default for classification.
- **Group k-fold**: when examples come in groups — several scans of one patient, many sessions of one user — keep each group inside a single fold, or the model is tested on people it has already seen.
- **Time-ordered splits**: train on the past and validate on the future (see *Time Series & Forecasting*).
- **Nested cross-validation**: an inner loop tunes the hyperparameters, an outer loop measures how well the whole tuning procedure generalizes.
- **Repeated k-fold** averages several random splits to steady the estimate on small datasets; leave-one-out (k = n) is costly and noisy, so 5 or 10 folds are the usual compromise.
- **Report the spread**: give the mean and the standard deviation across folds — two models whose difference is smaller than that spread are not reliably different.` }
    ],
    when: `- **k-fold CV** for small to medium datasets and fast models; **single validation split** for large datasets or expensive models (deep learning).
- **Stratified** folds for classification, **group** folds for related examples, **time-series** splits for temporal data.
- **Random search** (or smarter tuners) when there are more than two or three hyperparameters.`,
    pitfalls: `- Preprocessing (scaling, imputation, feature selection) fitted on all data before CV — must happen inside each fold (use pipelines).
- Shuffled CV on time series.
- Tuning against the test set.
- Over-interpreting tiny CV differences smaller than the standard deviation across folds.
- Forgetting to retrain on all training data after choosing the hyperparameters.`,
    takeaways: [
      'Hyperparameters are chosen by validation performance, never by training loss or test score.',
      'k-fold CV: train k times, validate on each fold once, report mean ± std.',
      'Use stratified, group or time-series splits to match the data\'s structure.',
      'Random search usually beats grid search for the same budget; search on log scales.',
      'Retrain the chosen configuration on all training data; touch the test set once.'
    ],
    quiz: [
      { type: 'mcq', q: 'Which of these is a hyperparameter?', options: ['A weight in a linear model', 'The split threshold learned at a tree node', 'The regularization strength λ', 'A neural network bias'], answer: 2, explain: 'λ is set before training; the others are learned.' },
      { type: 'fill', q: 'With 5-fold cross-validation, how many times is the model trained?', answer: ['5', 'five'], explain: 'Once per fold.' },
      { type: 'mcq', q: 'You are forecasting electricity demand. Which validation scheme is appropriate?', options: ['Shuffled k-fold', 'Time-series (rolling) splits', 'Leave-one-out', 'Stratified k-fold'], answer: 1, explain: 'Each validation fold must lie after its training data in time.' },
      { type: 'mcq', q: 'Why does random search often beat grid search with the same number of trials?', options: ['It uses more compute', 'It tries more distinct values of the few hyperparameters that actually matter', 'It always finds the global optimum', 'It doesn\'t need validation data'], answer: 1, explain: 'A grid repeats the same few values of each hyperparameter; random sampling explores each dimension more.' },
      { type: 'tf', q: 'When scaling features, the scaler should be fitted inside each cross-validation fold on that fold\'s training part.', answer: true, explain: 'Otherwise the validation fold\'s statistics leak into training.' }
    ],
    problems: [
      {
        id: 'k-fold-splits', title: 'k-Fold Split Indices', difficulty: 'medium',
        prompt: `Implement \`k_fold_indices(n, k, seed=0)\` returning a list of k \`(train_idx, val_idx)\` pairs, using this exact recipe:

1. \`perm = np.random.default_rng(seed).permutation(n)\`
2. \`folds = np.array_split(perm, k)\` (sizes differ by at most one)
3. For fold i: \`val_idx = folds[i]\`, \`train_idx\` = all the other folds concatenated in order.

Every index must appear in exactly one validation fold, and never in its own training set.

\`\`\`text
pairs = k_fold_indices(10, 3)
[len(v) for t, v in pairs]  ->  [4, 3, 3]
\`\`\``,
        starter: `
import numpy as np

def k_fold_indices(n, k, seed=0):
    """List of (train_idx, val_idx) for k-fold cross-validation."""
    # TODO
    pass
`,
        solution: `
import numpy as np

def k_fold_indices(n, k, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    pairs = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        pairs.append((train_idx, val_idx))
    return pairs
`,
        hints: ['`np.array_split(perm, k)` handles sizes that don\'t divide evenly.', 'Concatenate the other folds with `np.concatenate([folds[j] for j in range(k) if j != i])`.'],
        explain: `Splitting a single shuffled permutation guarantees the validation folds are disjoint and cover every example exactly once. scikit-learn's \`KFold(n_splits=k, shuffle=True, random_state=seed)\` does the same.`,
        tests: [
          { name: 'fold sizes', code: 'pairs = k_fold_indices(10, 3)\nassert [len(v) for t, v in pairs] == [4, 3, 3], [len(v) for t, v in pairs]' },
          { name: 'seeded result', call: 'k_fold_indices(7, 3, seed=2)' },
          { name: 'validation folds cover everything once', code: 'pairs = k_fold_indices(23, 5, seed=4)\nall_val = np.concatenate([v for t, v in pairs])\nassert sorted(all_val.tolist()) == list(range(23))\nfor t, v in pairs:\n    assert len(set(t.tolist()) & set(v.tolist())) == 0, "train and validation overlap"\n    assert len(t) + len(v) == 23', hidden: true },
          { name: 'another seed', call: 'k_fold_indices(12, 4, seed=9)', hidden: true }
        ]
      },
      {
        id: 'grid-search', title: 'Grid Search with Cross-Validation', difficulty: 'medium',
        prompt: `Choose the ridge regularization strength by 5-fold cross-validation. A \`ridge(X, y, lam)\` function (returns \`(w, b)\`) and \`k_fold_indices(n, k, seed)\` are provided in the starter.

Implement \`best_lambda(X, y, lambdas, k=5, seed=0)\`:
1. Build the folds **once**: \`k_fold_indices(len(X), k, seed)\`.
2. For each λ: train on each training split, compute the MSE on its validation split, and average over the k folds.
3. Return the λ with the lowest mean validation MSE (ties → the earlier one in \`lambdas\`).`,
        starter: `
import numpy as np

def ridge(X, y, lam):
    xm, ym = X.mean(axis=0), y.mean()
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, ym - xm @ w

def k_fold_indices(n, k, seed=0):
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]

def best_lambda(X, y, lambdas, k=5, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def ridge(X, y, lam):
    xm, ym = X.mean(axis=0), y.mean()
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, ym - xm @ w

def k_fold_indices(n, k, seed=0):
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]

def best_lambda(X, y, lambdas, k=5, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    pairs = k_fold_indices(len(X), k, seed)
    best, best_score = None, np.inf
    for lam in lambdas:
        scores = []
        for tr, va in pairs:
            w, b = ridge(X[tr], y[tr], lam)
            scores.append(np.mean((X[va] @ w + b - y[va]) ** 2))
        score = np.mean(scores)
        if score < best_score:
            best, best_score = lam, score
    return best
`,
        hints: ['Compute the folds once, outside the λ loop, so every λ is judged on identical splits.', 'Average the k validation MSEs per λ and keep the smallest (strict `<` keeps the earlier λ on ties).'],
        explain: `Using the same folds for every candidate makes the comparison fair — differences come from λ, not from which examples landed where. This loop is exactly what \`GridSearchCV\` automates (plus parallelism and refitting the winner on all data).`,
        tests: [
          { name: 'noisy, many features: some regularization wins', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(40, 15))\ny = X[:, 0] - X[:, 1] + rng.normal(0, 1.0, 40)', call: 'best_lambda(X, y, [0.0, 1.0, 10.0, 100.0, 1000.0])' },
          { name: 'clean data: little regularization wins', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(100, 3))\ny = X @ np.array([3.0, -2.0, 1.0]) + rng.normal(0, 0.05, 100)', call: 'best_lambda(X, y, [0.0, 10.0, 100.0])', expect: '0.0' },
          { name: 'another dataset', setup: 'rng = np.random.default_rng(7)\nX = rng.normal(size=(60, 8))\ny = X @ rng.normal(size=8) * 0.3 + rng.normal(0, 1.5, 60)', call: 'best_lambda(X, y, [0.01, 0.1, 1.0, 10.0, 100.0], k=4, seed=3)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Cross-validation: evaluating estimator performance', url: 'https://scikit-learn.org/stable/modules/cross_validation.html' },
      { title: 'Bergstra & Bengio — Random Search for Hyper-Parameter Optimization (2012)', url: 'https://www.jmlr.org/papers/v13/bergstra12a.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'knn', branch: 'ml', title: 'k-Nearest Neighbours', short: 'k-Nearest Neighbours', level: 1, minutes: 30, prereqs: ['supervised-learning', 'numpy'],
    tags: ['knn', 'distance', 'lazy learning', 'curse of dimensionality', 'nearest neighbor search'],
    summary: 'Predict by looking at the most similar training examples: distances, choosing k, feature scaling, and the curse of dimensionality.',
    why: `k-nearest neighbours (kNN) is the most intuitive learning algorithm there is — "you are like your closest neighbours" — and it needs no training at all. It is a strong baseline, works for classification and regression, and makes the importance of **distance** and **feature scaling** impossible to ignore. The same nearest-neighbour search, scaled to billions of vectors, powers semantic search, recommendation engines and retrieval-augmented generation (RAG) for LLMs.`,
    objectives: [
      'Describe the kNN algorithm for classification and regression',
      'Choose a distance metric and explain why feature scaling matters',
      'Explain how k controls the bias–variance trade-off',
      'Describe the curse of dimensionality and the cost of prediction'
    ],
    sections: [
      { type: 'text', title: 'The algorithm', md: `kNN does no work at training time: it just stores the training data. To predict for a new point x:

1. Compute the distance from x to every training example.
2. Take the **k** closest ones.
3. **Classification:** majority vote among their labels. **Regression:** average of their targets.

It is called a **lazy** or **instance-based** learner: the "model" *is* the data.

Common distances:
- **Euclidean** √Σ(aᵢ − bᵢ)² — the default.
- **Manhattan** Σ|aᵢ − bᵢ| — city blocks; more robust to single large differences.
- **Cosine distance** 1 − cos_sim — for text and embedding vectors where direction matters.` },
      { type: 'demo', title: 'Interactive: choosing k', md: `Move the slider: small k gives jagged boundaries that chase individual noisy points (low bias, high variance); large k gives smooth boundaries (higher bias, lower variance). Click to add points.`, demo: 'knn' },
      { type: 'text', title: 'Scaling is not optional', md: `Distances add up differences across features. If one feature is "income in euros" (differences of thousands) and another is "number of children" (differences of 1–3), the income completely dominates — the model effectively ignores the children feature.

**Always standardize features before kNN** (fit the scaler on training data). Similarly, irrelevant features add noise to every distance; feature selection helps kNN more than most models.` },
      { type: 'code', title: 'kNN from scratch (and why scaling matters)', code: `
import numpy as np
rng = np.random.default_rng(0)

# feature 0: income in euros (huge scale, irrelevant); feature 1: a small-scale but informative signal
n = 300
signal = rng.normal(size=n)
income = rng.normal(50_000, 15_000, size=n)
X = np.column_stack([income, signal]); y = (signal > 0).astype(int)
Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]

def knn_predict(Xtr, ytr, Xte, k):
    d = np.linalg.norm(Xte[:, None, :] - Xtr[None, :, :], axis=-1)   # (n_test, n_train)
    nearest = np.argsort(d, axis=1)[:, :k]                            # indices of the k closest
    return (ytr[nearest].mean(axis=1) >= 0.5).astype(int)            # majority vote (binary)

print("accuracy, raw features:   ", np.mean(knn_predict(Xtr, ytr, Xte, 5) == yte))
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
print("accuracy, standardized:   ", np.mean(knn_predict((Xtr - mu) / sd, ytr, (Xte - mu) / sd, 5) == yte))
`, predict: {"type":"mcq","q":"The label depends only on the signal feature (standard deviation 1), but each point also has an income feature (standard deviation 15,000). What accuracy does 5-nearest-neighbours reach on the raw features?","options":["About 95%, the same as after scaling","About 50%: no better than guessing","About 75%","Exactly 100%"],"answer":1,"explain":"0.47, a coin flip. Distances are dominated by income, which is pure noise here, so the neighbours are effectively random. After standardizing, accuracy jumps to 0.95.","expect":["accuracy, raw features:    0.47","accuracy, standardized:    0.95"]} },
      { type: 'check', question: { type: 'mcq', q: 'In kNN, what happens as k becomes very large (close to the number of training examples)?', options: ['The model overfits more', 'Predictions approach the overall majority class / mean (underfitting)', 'Training becomes slower', 'Distances become more accurate'], answer: 1, explain: 'With k ≈ n every prediction averages almost the whole dataset, ignoring the local structure: high bias.' } },
      { type: 'text', title: 'The curse of dimensionality and the cost of prediction', md: `In high dimensions, intuition about "nearby" breaks down. With hundreds of features, data points become sparse and **distances concentrate**: the nearest and farthest neighbours are almost equally far away, so "nearest" carries little information. To keep the same density of points, the data needed grows exponentially with the number of dimensions. Remedies: feature selection, dimensionality reduction (PCA), or learned representations (embeddings) where distances are meaningful.

**Cost:** prediction compares the query with every training example — O(n·d) per query. For millions of examples, systems use **approximate nearest-neighbour** indexes (HNSW graphs, IVF, product quantization) that trade a little accuracy for enormous speed. Those indexes are what "vector databases" are made of, and they are a core part of RAG systems.` },
      { type: 'check', question: { type: 'tf', q: 'kNN has an expensive training phase and a cheap prediction phase.', answer: false, explain: 'The reverse: training just stores the data; prediction must search it.' } },
      { type: 'text', title: 'Going further: weights, regression and speed', md: `- **Distance-weighted votes**: closer neighbours count more (for example with weight 1/distance), which makes the result less sensitive to the exact choice of k.
- **kNN regression**: predict the average — or the distance-weighted average — of the neighbours' target values.
- **Choosing k and the distance** by cross-validation: a small k gives jagged, high-variance boundaries; a large k smooths them but blurs real structure. An odd k avoids ties in two-class problems. Euclidean distance suits scaled numeric features, cosine similarity suits text and embeddings.
- **Speed**: brute force costs about n · d operations per query. KD-trees and ball trees speed this up in low dimensions; for embeddings with hundreds of dimensions, approximate nearest-neighbour indexes take over (see *Vector Search*).
- **The idea lives on**: retrieval in RAG, "customers like you" recommendations and few-shot classification on embeddings are all nearest-neighbour methods.` }
    ],
    when: `- **Good for**: small to medium datasets with a meaningful distance (after scaling), low-to-moderate dimensionality, quick baselines, and problems where similar inputs truly have similar outputs.
- **Similarity search** over learned embeddings (images, text, products) — kNN is the retrieval step of RAG and recommendations.
- **Avoid** for very high-dimensional raw features, huge datasets without an ANN index, or when you need fast predictions on modest hardware.`,
    pitfalls: `- Forgetting to scale features.
- Even k for binary classification causes ties — prefer odd k (or weight votes by distance).
- Irrelevant features silently destroy performance.
- Leaving the test point inside the training set during evaluation (distance 0 to itself).
- Slow predictions at scale without an index.`,
    takeaways: [
      'kNN predicts from the k most similar training examples: vote (classification) or average (regression).',
      'Small k = flexible (high variance); large k = smooth (high bias). Tune k with validation.',
      'Distances are meaningless without feature scaling.',
      'High dimensions make neighbours less meaningful (curse of dimensionality).',
      'Large-scale nearest-neighbour search (vector databases) powers RAG and recommendations.'
    ],
    quiz: [
      { type: 'mcq', q: 'kNN with k = 1 achieves 100% training accuracy. Why is that not impressive?', options: ['Because k = 1 is always optimal', 'Each training point\'s nearest neighbour is itself — it memorised the data', 'Because accuracy is the wrong metric', 'Because distances were not scaled'], answer: 1, explain: 'Evaluate on held-out data; 1-NN memorises and is very sensitive to noise.' },
      { type: 'mcq', q: 'Why prefer an odd k for binary classification?', options: ['It is faster', 'It avoids tied votes', 'It reduces the dimensionality', 'Even k is not allowed'], answer: 1, explain: 'With two classes and odd k, the vote cannot tie.' },
      { type: 'tf', q: 'For kNN regression, the prediction is typically the average target of the k nearest neighbours.', answer: true, explain: 'Optionally weighted by inverse distance.' },
      { type: 'mcq', q: 'What is the main problem with kNN on 1,000 raw features?', options: ['It cannot compute distances in 1,000 dimensions', 'Distances become nearly equal, so neighbours carry little information', 'It needs a GPU', 'It always underfits'], answer: 1, explain: 'This is the curse of dimensionality: distances concentrate.' },
      { type: 'mcq', q: 'Which technology makes nearest-neighbour search fast over billions of embedding vectors?', options: ['Gradient boosting', 'Approximate nearest-neighbour indexes (e.g. HNSW) in vector databases', 'One-hot encoding', 'Dropout'], answer: 1, explain: 'ANN indexes trade a little accuracy for orders-of-magnitude speedups.' }
    ],
    problems: [
      {
        id: 'knn-classify', title: 'k-NN Classifier', difficulty: 'medium',
        prompt: `Implement \`knn_predict(X_train, y_train, X_test, k)\`:

For each test row:
1. compute the Euclidean distance to every training row;
2. take the indices of the k smallest distances — use \`np.argsort(d, kind="stable")[:k]\` so equal distances keep training order;
3. return the most common label among those k neighbours; if labels tie, return the **smallest** tied label.

Return a NumPy array of predictions.

\`\`\`text
X_train = [[0], [1], [2], [10], [11]], y_train = [0, 0, 0, 1, 1]
knn_predict(X_train, y_train, [[1.5], [10.5]], k=3)  ->  [0, 1]
\`\`\``,
        starter: `
import numpy as np

def knn_predict(X_train, y_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    preds = []
    for x in X_test:
        # TODO: distances, k nearest indices, vote (ties -> smallest label)
        pass
    return np.array(preds)
`,
        solution: `
import numpy as np

def knn_predict(X_train, y_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    preds = []
    for x in X_test:
        d = np.linalg.norm(X_train - x, axis=1)
        nearest = np.argsort(d, kind="stable")[:k]
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        preds.append(labels[np.argmax(counts)])   # unique is sorted -> ties go to the smallest label
    return np.array(preds)
`,
        hints: ['`np.linalg.norm(X_train - x, axis=1)` gives the distance from x to every training row.', '`np.unique(labels, return_counts=True)` returns sorted labels, so `argmax(counts)` breaks ties toward the smallest.'],
        explain: `The loop handles one query at a time: compute distances, sort, vote. Sorted labels from \`np.unique\` plus \`argmax\` (first maximum) gives deterministic tie-breaking. A fully vectorized version computes the whole (n_test, n_train) distance matrix at once, as in the lesson — faster, but memory grows with n_test × n_train.`,
        tests: [
          { call: 'knn_predict([[0], [1], [2], [10], [11]], [0, 0, 0, 1, 1], [[1.5], [10.5]], 3)', expect: 'np.array([0, 1])' },
          { name: 'k = 1', call: 'knn_predict([[0, 0], [5, 5]], ["a", "b"], [[1, 1], [4, 4], [6, 5]], 1)', expect: 'np.array(["a", "b", "b"])' },
          { name: 'tie goes to the smallest label', call: 'knn_predict([[0.0], [2.0]], [1, 0], [[1.0]], 2)', expect: 'np.array([0])' },
          { name: 'random blobs vs reference', setup: 'rng = np.random.default_rng(13)\nXtr = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(2.5, 1, (30, 2))])\nytr = np.repeat([0, 1], 30)\nXte = rng.normal(1.2, 1.5, (15, 2))', call: 'knn_predict(Xtr, ytr, Xte, 5)', hidden: true },
          { name: 'three classes', setup: 'rng = np.random.default_rng(1)\nXtr = rng.normal(size=(50, 3))\nytr = rng.integers(0, 3, 50)\nXte = rng.normal(size=(10, 3))', call: 'knn_predict(Xtr, ytr, Xte, 4)', hidden: true }
        ]
      },
      {
        id: 'knn-regress', title: 'k-NN Regression', difficulty: 'easy',
        prompt: `Implement \`knn_regress(X_train, y_train, X_test, k)\`: for each test row, return the **mean target** of its k nearest training rows (Euclidean distance; use a stable argsort as in the classifier).

\`\`\`text
knn_regress([[0], [1], [2], [3]], [0.0, 10.0, 20.0, 30.0], [[0.9], [2.6]], 2)  ->  [5.0, 25.0]
\`\`\``,
        starter: `
import numpy as np

def knn_regress(X_train, y_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def knn_regress(X_train, y_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    d = np.linalg.norm(X_test[:, None, :] - X_train[None, :, :], axis=-1)
    nearest = np.argsort(d, axis=1, kind="stable")[:, :k]
    return y_train[nearest].mean(axis=1)
`,
        hints: ['Build the full distance matrix with broadcasting: shape (n_test, n_train).', '`y_train[nearest]` has shape (n_test, k); average along axis 1.'],
        explain: `Fancy indexing \`y_train[nearest]\` gathers the k neighbour targets for every query in one go. kNN regression produces piecewise-constant predictions; distance-weighted averaging smooths them.`,
        tests: [
          { call: 'knn_regress([[0], [1], [2], [3]], [0.0, 10.0, 20.0, 30.0], [[0.9], [2.6]], 2)', expect: 'np.array([5.0, 25.0])' },
          { name: 'k = 1 copies the nearest target', call: 'knn_regress([[0, 0], [3, 4]], [1.0, 7.0], [[2.5, 4.0]], 1)', expect: 'np.array([7.0])' },
          { name: 'random data', setup: 'rng = np.random.default_rng(21)\nXtr = rng.uniform(0, 10, (40, 1))\nytr = np.sin(Xtr[:, 0]) + rng.normal(0, 0.1, 40)\nXte = rng.uniform(0, 10, (12, 1))', call: 'knn_regress(Xtr, ytr, Xte, 3)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Nearest Neighbors', url: 'https://scikit-learn.org/stable/modules/neighbors.html' },
      { title: "Stanford CS231n — Image classification and nearest neighbours", url: "https://cs231n.github.io/classification/" }
    ]
  },

  // =====================================================================================
  {
    id: 'decision-trees', branch: 'ml', title: 'Decision Trees', level: 2, minutes: 40, prereqs: ['supervised-learning', 'information-theory'],
    tags: ['tree', 'gini', 'entropy', 'information gain', 'cart', 'pruning'],
    summary: 'A flowchart of yes/no questions learned from data: how trees choose splits with Gini impurity or entropy, how they overfit, and why they are the foundation of the strongest tabular models.',
    why: `Decision trees are among the most interpretable models — you can print one and follow its reasoning — and they handle mixed feature types, non-linear interactions and unscaled data with no fuss. On their own they overfit easily, but combined into **random forests** and **gradient-boosted trees** (XGBoost, LightGBM) they are still the state of the art for spreadsheet-style tabular data, often beating deep learning there. Understanding how a single tree picks its splits is the key to all of them.`,
    objectives: [
      'Explain how a decision tree makes predictions',
      'Compute Gini impurity and entropy and the information gain of a split',
      'Describe greedy tree growing and the stopping/pruning controls',
      'Explain why single trees overfit and why ensembles fix that'
    ],
    sections: [
      { type: 'text', title: 'A tree is a flowchart', md: `A decision tree asks a sequence of yes/no questions about the features:

\`\`\`text
                 [income > 40k?]
                 /             \\
              no                yes
        [debt > 10k?]          → approve (92% repaid)
         /         \\
       no          yes
  → approve      → reject (61% defaulted)
  (80% repaid)
\`\`\`

- **Internal nodes** test one feature against a threshold (or a category).
- **Leaves** hold a prediction: the majority class (or class proportions) for classification, the mean target for regression.
- To predict, start at the root and follow the answers down to a leaf.

The boundaries are **axis-aligned**: each split cuts the feature space with a horizontal or vertical line (in 2-D), producing rectangular regions.` },
      { type: 'text', title: 'Measuring impurity', md: `A good split produces children that are **purer** — more dominated by a single class — than the parent. Two standard impurity measures for a node whose class proportions are p₁, …, p_K:

$$ Gini = 1 − Σ p_k²          Entropy = − Σ p_k log₂ p_k $$
:::where
k — a class (the sums run over every class present in the node)
p_k — the fraction of the node's training examples that belong to class k
Gini — the chance that two examples drawn at random from the node have different classes; 0 for a pure node
Entropy — the average surprise in bits; 0 for a pure node and 1 for a 50/50 two-class node
:::

| Node contents | Gini | Entropy |
| --- | --- | --- |
| all one class | 0 | 0 |
| 50 / 50 | 0.5 | 1 bit |
| 90 / 10 | 0.18 | 0.47 bits |

A split's quality is the **weighted average impurity** of its children (weighted by how many examples go each way). The decrease from the parent's impurity is the **information gain** (for entropy) or Gini gain. Both measures usually pick very similar splits; Gini is slightly cheaper and is scikit-learn's default.

For **regression trees** impurity is the variance (MSE) of the targets in the node, and leaves predict the mean.` },
      { type: 'demo', title: 'Interactive: find the best split', demo: 'tree-split' },
      { type: 'code', title: 'Finding the best split by hand', code: `
import numpy as np

def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

# one feature (hours of sleep) and a label (felt rested?)
x = np.array([4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9])
y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1])
print("parent Gini:", round(gini(y), 3))

values = np.unique(x)
for t in (values[:-1] + values[1:]) / 2:          # candidate thresholds between neighbours
    left, right = y[x <= t], y[x > t]
    weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
    print(f"split x <= {t:4.2f}: weighted Gini {weighted:.3f}")
` },
      { type: 'check', question: { type: 'fill', q: 'A node has 6 examples of class A and 2 of class B. What is its Gini impurity? (decimal)', answer: ['0.375', '.375', '3/8'], explain: '1 − (0.75² + 0.25²) = 1 − (0.5625 + 0.0625) = 0.375.' } },
      { type: 'text', title: 'Growing and pruning', md: `Trees are grown **greedily** (the CART algorithm):
1. For every feature and every candidate threshold, compute the weighted impurity of the resulting split.
2. Apply the best split.
3. Recurse on each child until a stopping rule triggers.

Grown without limits, a tree keeps splitting until every leaf is pure — memorising the training data, noise included. Controls (hyperparameters):
- \`max_depth\` — limit the number of questions in a row;
- \`min_samples_leaf\` / \`min_samples_split\` — don't create tiny leaves;
- \`max_leaf_nodes\`;
- **cost-complexity pruning** (\`ccp_alpha\`) — grow fully, then cut back branches that don't pay for their complexity.

Even well-tuned single trees are **unstable**: a small change in the data can change the top split and therefore the whole tree. That high variance is exactly what ensembles average away.` },
      { type: 'code', title: 'A tree in scikit-learn', md: `\`export_text\` prints the learned rules — interpretability for free.`, code: `
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

for depth in [1, 2, 3, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"max_depth={depth}: train {tree.score(X_tr, y_tr):.3f}  test {tree.score(X_te, y_te):.3f}")

print(export_text(DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr), feature_names=names))
` },
      { type: 'check', question: { type: 'mcq', q: 'Why don\'t decision trees need feature scaling?', options: ['They normalize features internally', 'Splits compare one feature against a threshold, which is unaffected by rescaling that feature', 'They only work with binary features', 'Scaling makes them slower'], answer: 1, explain: 'Multiplying a feature by 1000 just multiplies its thresholds by 1000; the same examples go left and right.' } }
    ],
    when: `- When you need an **interpretable** model or explicit rules.
- Mixed numeric/categorical features, missing values (in many implementations), non-linear interactions, no scaling needed.
- As the building block of **random forests** and **gradient boosting**, which are usually what you actually deploy on tabular data.
- Avoid a single deep tree when accuracy matters — use an ensemble.`,
    pitfalls: `- Unlimited depth → memorisation (overfitting).
- Instability — small data changes produce very different trees; don't over-interpret one tree's structure.
- Axis-aligned splits approximate diagonal boundaries with staircases.
- Impurity-based feature importances are biased toward features with many unique values; prefer permutation importance.`,
    takeaways: [
      'A tree predicts by following yes/no feature tests down to a leaf.',
      'Splits are chosen greedily to minimise weighted Gini impurity or entropy (maximise information gain).',
      'Regression trees split to reduce variance and predict leaf means.',
      'Depth and leaf-size limits (or pruning) control overfitting.',
      'Single trees are high-variance; ensembles of trees are the tabular-data champions.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the Gini impurity of a node containing only one class?', options: ['1', '0.5', '0', 'It is undefined'], answer: 2, explain: '1 − 1² = 0: a pure node.' },
      { type: 'mcq', q: 'How does a classification tree predict at a leaf?', options: ['By averaging all training labels', 'By the majority class (or class proportions) of the training examples in that leaf', 'By a linear model', 'Randomly'], answer: 1, explain: 'Leaves store the class distribution of the examples that reached them.' },
      { type: 'tf', q: 'A decision tree grown until all leaves are pure will usually generalize well.', answer: false, explain: 'It memorises noise; limit depth or leaf size, or use an ensemble.' },
      { type: 'mcq', q: 'Information gain of a split is…', options: ['the entropy of the parent minus the weighted entropy of the children', 'the number of features used', 'the depth of the tree', 'the accuracy of the leaf'], answer: 0, explain: 'It measures how much a split reduces uncertainty about the label.' },
      { type: 'mcq', q: 'What do regression trees use as impurity?', options: ['Gini', 'Variance / MSE of the targets in the node', 'Accuracy', 'Cosine similarity'], answer: 1, explain: 'They split to reduce the spread of target values; leaves predict the mean.' }
    ],
    problems: [
      {
        id: 'gini-impurity', title: 'Gini Impurity', difficulty: 'easy',
        prompt: `Implement \`gini(labels)\` = 1 − Σ p_k², where p_k is the fraction of examples with class k. Labels can be any hashable values (ints or strings). The Gini of an empty list is 0.0.

\`\`\`text
gini([0, 0, 1, 1])       ->  0.5
gini(["a", "a", "a"])    ->  0.0
gini([0, 0, 0, 1])       ->  0.375
\`\`\``,
        starter: `
import numpy as np

def gini(labels):
    """Gini impurity of a list of class labels."""
    # TODO
    pass
`,
        solution: `
import numpy as np

def gini(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(1 - np.sum(p ** 2))
`,
        hints: ['`np.unique(labels, return_counts=True)` counts each class.', 'Convert counts to proportions, square, sum, subtract from 1.'],
        explain: `Gini impurity is the probability that two examples drawn at random (with replacement) from the node have different labels: 1 − Σ p_k² is "1 minus the probability they match". It is 0 for a pure node and largest when classes are evenly mixed.`,
        tests: [
          { call: 'gini([0, 0, 1, 1])', expect: '0.5' },
          { call: 'gini(["a", "a", "a"])', expect: '0.0' },
          { call: 'gini([0, 0, 0, 1])', expect: '0.375' },
          { name: 'three classes', call: 'gini([1, 2, 3, 1, 2, 3])', expect: '1 - 3 * (1 / 3) ** 2', hidden: true },
          { name: 'empty', call: 'gini([])', expect: '0.0', hidden: true }
        ]
      },
      {
        id: 'best-split', title: 'Find the Best Split', difficulty: 'medium',
        prompt: `Implement \`best_split(x, y)\` for a single numeric feature \`x\` and class labels \`y\`:

- Candidate thresholds are the **midpoints between consecutive distinct sorted values** of x.
- A threshold t sends \`x <= t\` left and \`x > t\` right.
- Score = weighted Gini = (n_left·Gini(left) + n_right·Gini(right)) / n.
- Return \`(best_threshold, best_score)\` with the **lowest** score; on ties, the smaller threshold. If x has fewer than 2 distinct values, return \`(None, gini(y))\`.

A \`gini\` helper is provided.

\`\`\`text
best_split([1, 2, 3, 4], [0, 0, 1, 1])  ->  (2.5, 0.0)
\`\`\``,
        starter: `
import numpy as np

def gini(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(1 - np.sum(p ** 2))

def best_split(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    # TODO
    pass
`,
        solution: `
import numpy as np

def gini(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(1 - np.sum(p ** 2))

def best_split(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    values = np.unique(x)
    if len(values) < 2:
        return None, gini(y)
    best_t, best_score = None, np.inf
    for t in (values[:-1] + values[1:]) / 2:
        left, right = y[x <= t], y[x > t]
        score = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
        if score < best_score:
            best_t, best_score = float(t), float(score)
    return best_t, best_score
`,
        hints: ['`values = np.unique(x)` is sorted; midpoints are `(values[:-1] + values[1:]) / 2`.', 'Use a strict `<` while scanning thresholds in increasing order to keep the smaller one on ties.'],
        explain: `This is the inner loop of every tree-learning algorithm (CART), run for every feature at every node. Real implementations sort once and update class counts incrementally while sweeping the thresholds, making each feature O(n log n) instead of O(n²).`,
        tests: [
          { call: 'best_split([1, 2, 3, 4], [0, 0, 1, 1])', expect: '(2.5, 0.0)' },
          { name: 'imperfect split', call: 'best_split([4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9], [0, 0, 0, 0, 1, 1, 0, 1, 1])' },
          { name: 'no possible split', call: 'best_split([3, 3, 3], [0, 1, 1])', expect: '(None, 4 / 9)', hidden: true },
          { name: 'unsorted input, string labels', call: 'best_split([10, 1, 7, 3], ["b", "a", "b", "a"])', expect: '(5.0, 0.0)', hidden: true },
          { name: 'random data', setup: 'rng = np.random.default_rng(4)\nx = np.round(rng.uniform(0, 10, 40), 1)\ny = (x + rng.normal(0, 2, 40) > 5).astype(int)', call: 'best_split(x, y)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Decision Trees', url: 'https://scikit-learn.org/stable/modules/tree.html' },
      { title: 'R2D3 — A visual introduction to machine learning', url: 'http://www.r2d3.us/visual-intro-to-machine-learning-part-1/' }
    ]
  },

  // =====================================================================================
  {
    id: 'ensembles', branch: 'ml', title: 'Ensembles: Random Forests & Gradient Boosting', short: 'Ensembles & Boosting', level: 2, minutes: 45, prereqs: ['decision-trees', 'overfitting'],
    tags: ['random forest', 'bagging', 'boosting', 'gradient boosting', 'xgboost', 'lightgbm'],
    summary: 'Many weak models beat one strong one: bagging and random forests reduce variance; boosting builds models that fix each other\'s mistakes. The best tools for tabular data.',
    why: `For structured, spreadsheet-like data — customer records, transactions, sensor logs — **gradient-boosted trees** (XGBoost, LightGBM, CatBoost) and **random forests** remain the strongest practical models, regularly winning competitions and powering production systems in finance, retail and logistics. They need little preprocessing, handle non-linear interactions, and are robust. The ideas behind them — averaging to cut variance, and sequentially correcting errors — reappear throughout ML (dropout, model averaging, residual connections).`,
    objectives: [
      'Explain why averaging many models reduces variance',
      'Describe bagging, bootstrap samples, random forests and out-of-bag error',
      'Describe boosting and gradient boosting as fitting residuals step by step',
      'Choose between random forests and gradient boosting and tune their key knobs'
    ],
    sections: [
      { type: 'text', title: 'The wisdom of crowds', md: `Average the guesses of many people at the weight of an ox and the average is often closer than nearly any individual guess — *if* their errors are independent. The same holds for models: averaging B models whose errors are uncorrelated divides the variance by B.

Two ensemble families exploit this differently:
- **Bagging** trains many models **in parallel** on random variations of the data and averages them → reduces **variance**. (Best with low-bias, high-variance models like deep trees.)
- **Boosting** trains models **sequentially**, each focusing on the errors of the ensemble so far → reduces **bias** (and with care, variance).` },
      { type: 'text', title: 'Bagging and random forests', md: `**Bagging** (bootstrap aggregating):
1. Draw a **bootstrap sample**: n examples sampled *with replacement* from the n training examples (about 63% of distinct examples appear; some repeat).
2. Train a model (typically a deep, unpruned tree) on it.
3. Repeat B times; predict by majority vote (classification) or averaging (regression).

**Random forests** add one twist: at each split, a tree may only consider a **random subset of features** (e.g. √d of them). This *decorrelates* the trees — otherwise every tree would split on the same dominant feature first — and decorrelated errors average out much better.

A free bonus: each tree never saw ~37% of examples (**out-of-bag** examples), so evaluating each example only with the trees that didn't train on it gives a built-in validation estimate (OOB score).

Random forests are hard to break: robust defaults, little tuning, no scaling, parallelizable.` },
      { type: 'code', title: 'Averaging reduces variance', md: `Fit a very flexible model (a high-degree polynomial) on many bootstrap samples. Each fit is wild; their average is far steadier.`, code: `
import numpy as np, warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 6, 30)); y = np.sin(x) + rng.normal(0, 0.3, 30)
x_test = np.linspace(0.3, 5.7, 100); truth = np.sin(x_test)

single_errors, preds = [], []
for b in range(50):
    idx = rng.integers(0, 30, 30)                 # bootstrap: sample with replacement
    coefs = np.polyfit(x[idx], y[idx], 9)         # a high-variance model
    p = np.clip(np.polyval(coefs, x_test), -3, 3)
    preds.append(p)
    single_errors.append(np.mean((p - truth) ** 2))

bagged = np.mean(preds, axis=0)
print("average error of a single model:", round(float(np.mean(single_errors)), 4))
print("error of the bagged average:   ", round(float(np.mean((bagged - truth) ** 2)), 4))
` },
      { type: 'text', title: 'Boosting and gradient boosting', md: `**Boosting** builds an ensemble one small model at a time, each new model concentrating on what the current ensemble gets wrong.

**Gradient boosting** for regression with squared error is easy to state:

1. Start with a constant prediction F₀(x) = mean(y).
2. For m = 1 … M:
   - compute the **residuals** rᵢ = yᵢ − F_{m−1}(xᵢ) — what is still unexplained;
   - fit a small tree h_m (depth 1–6) to predict those residuals;
   - update F_m(x) = F_{m−1}(x) + η · h_m(x), with a small **learning rate** η (e.g. 0.05–0.1).

Why "gradient"? The residuals are exactly the negative gradient of the squared-error loss with respect to the predictions. Gradient boosting is **gradient descent in function space**: each tree is a step downhill, and other losses (log loss for classification, quantile loss, ranking losses) simply use their own gradients as targets.

Key hyperparameters: number of trees M, learning rate η (smaller needs more trees but generalizes better), tree depth, subsampling of rows/columns, and **early stopping** on validation loss. Modern libraries (XGBoost, LightGBM, CatBoost) add regularization, clever split finding and native categorical handling.` },
      { type: 'check', question: { type: 'mcq', q: 'In gradient boosting with squared-error loss, what does each new tree try to predict?', options: ['The original targets y', 'The residuals of the current ensemble', 'A random subset of labels', 'The feature importances'], answer: 1, explain: 'Each tree fits what the ensemble still gets wrong — the residuals, i.e. the negative gradient.' } },
      { type: 'code', title: 'Random forest vs gradient boosting in scikit-learn', code: `
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
models = {
    "single tree": DecisionTreeClassifier(random_state=0),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0),
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:<18} accuracy {scores.mean():.3f} ± {scores.std():.3f}")
` },
      { type: 'check', question: { type: 'tf', q: 'Random forests decorrelate their trees by letting each split consider only a random subset of features.', answer: true, explain: 'Feature subsampling (plus bootstrap samples) makes the trees\' errors less correlated, so averaging helps more.' } }
    ],
    when: `- **Tabular data**: start with gradient boosting (LightGBM/XGBoost/CatBoost or scikit-learn's \`HistGradientBoosting\`) and a random forest as a robust baseline.
- **Random forest** when you want strong results with almost no tuning, or parallel training.
- **Gradient boosting** when you want the last few points of accuracy and can tune (learning rate, depth, early stopping).
- **Prefer neural networks** for images, audio, text, and very large unstructured datasets.`,
    pitfalls: `- Boosting with too many trees and no early stopping overfits.
- A high learning rate in boosting needs fewer trees but generalizes worse.
- Interpreting impurity-based feature importance as causal — use permutation importance or SHAP, and still be careful.
- Large forests are slow to predict and big to store.
- Extrapolation: tree ensembles predict constants outside the training range.`,
    takeaways: [
      'Averaging many decorrelated models reduces variance (bagging, random forests).',
      'Random forests = bootstrap samples + random feature subsets per split; OOB gives free validation.',
      'Gradient boosting adds small trees sequentially, each fitting the residuals (negative gradient).',
      'Learning rate × number of trees and early stopping control boosting\'s fit.',
      'For tabular data, boosted trees are usually the strongest starting point.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is a bootstrap sample?', options: ['The first n/2 examples', 'n examples drawn with replacement from the training set', 'A sample with all duplicates removed', 'The validation set'], answer: 1, explain: 'Sampling with replacement produces repeats and leaves out about 37% of examples.' },
      { type: 'mcq', q: 'Bagging mainly reduces…', options: ['bias', 'variance', 'irreducible noise', 'training time'], answer: 1, explain: 'Averaging high-variance models smooths out their individual fluctuations.' },
      { type: 'mcq', q: 'Why do boosting libraries use a small learning rate (shrinkage)?', options: ['To train faster', 'Each tree contributes a little, which generalizes better (with more trees)', 'Because trees cannot be large', 'To avoid needing validation data'], answer: 1, explain: 'Many small steps act as regularization, like a small learning rate in gradient descent.' },
      { type: 'tf', q: 'Out-of-bag examples are the ones a given tree did not see during training, so they can be used to estimate generalization.', answer: true, explain: 'Each example is evaluated only by trees that didn\'t train on it.' },
      { type: 'mcq', q: 'For a dataset of customer records with 40 mixed numeric/categorical columns, a strong first choice is…', options: ['A convolutional neural network', 'Gradient-boosted trees', 'k-means', 'A transformer trained from scratch'], answer: 1, explain: 'Boosted trees dominate tabular data with minimal preprocessing.' }
    ],
    problems: [
      {
        id: 'bootstrap-majority', title: 'Majority Vote of an Ensemble', difficulty: 'easy',
        prompt: `An ensemble classifier combines its members by voting. Implement \`majority_vote(predictions)\`:

- \`predictions\` is a 2-D array of shape (n_models, n_samples) with integer class labels.
- Return a 1-D array with, for each sample (column), the most frequent label; ties go to the **smallest** label.

\`\`\`text
majority_vote(np.array([[0, 1, 2],
                        [0, 2, 2],
                        [1, 1, 0]]))  ->  [0, 1, 2]
\`\`\``,
        starter: `
import numpy as np

def majority_vote(predictions):
    predictions = np.asarray(predictions)
    # TODO: vote per column
    pass
`,
        solution: `
import numpy as np

def majority_vote(predictions):
    predictions = np.asarray(predictions)
    out = []
    for col in predictions.T:
        labels, counts = np.unique(col, return_counts=True)
        out.append(labels[np.argmax(counts)])
    return np.array(out)
`,
        hints: ['Iterate over columns with `predictions.T`.', '`np.unique(..., return_counts=True)` + `argmax` breaks ties toward the smallest label.'],
        explain: `Voting is the classification version of averaging. Random forests in scikit-learn actually average predicted probabilities ("soft voting"), which uses each tree's confidence and usually works a little better than counting hard votes.`,
        tests: [
          { call: 'majority_vote(np.array([[0, 1, 2], [0, 2, 2], [1, 1, 0]]))', expect: 'np.array([0, 1, 2])' },
          { name: 'tie goes to the smallest label', call: 'majority_vote(np.array([[3, 1], [2, 1]]))', expect: 'np.array([2, 1])' },
          { name: 'random ensemble', setup: 'rng = np.random.default_rng(5)\nP = rng.integers(0, 4, size=(9, 25))', call: 'majority_vote(P)', hidden: true }
        ]
      },
      {
        id: 'boosting-stumps', title: 'Gradient Boosting with Decision Stumps', difficulty: 'hard',
        prompt: `Build gradient boosting for 1-D regression from scratch, using **decision stumps** (one-split trees) as the weak learners.

\`fit_boosting(x, y, n_rounds, lr)\` returns \`(f0, stumps)\`:
1. \`f0 = mean(y)\`; current predictions \`F = f0\` for every example.
2. Repeat \`n_rounds\` times:
   - residuals \`r = y - F\`;
   - fit a stump to r: try every threshold t at the midpoints between consecutive distinct sorted x values; a stump predicts \`mean(r[x <= t])\` on the left and \`mean(r[x > t])\` on the right; choose the t with the **lowest sum of squared errors** on r (ties → smaller t);
   - append \`(t, left_value, right_value)\` to \`stumps\`;
   - update \`F += lr * stump_prediction(x)\`.

\`predict_boosting(f0, stumps, x_new, lr)\` returns \`f0 + lr * Σ stump(x_new)\` as an array.

The tests check exact agreement with this procedure and that the training error falls as rounds increase.`,
        starter: `
import numpy as np

def fit_stump(x, r):
    """Best (t, left_value, right_value) for residuals r."""
    # TODO
    pass

def fit_boosting(x, y, n_rounds=50, lr=0.1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f0 = float(y.mean())
    F = np.full(len(y), f0)
    stumps = []
    # TODO
    return f0, stumps

def predict_boosting(f0, stumps, x_new, lr=0.1):
    # TODO
    pass
`,
        solution: `
import numpy as np

def fit_stump(x, r):
    values = np.unique(x)
    best = None
    best_sse = np.inf
    for t in (values[:-1] + values[1:]) / 2:
        left, right = r[x <= t], r[x > t]
        lv, rv = left.mean(), right.mean()
        sse = np.sum((left - lv) ** 2) + np.sum((right - rv) ** 2)
        if sse < best_sse:
            best, best_sse = (float(t), float(lv), float(rv)), sse
    return best

def fit_boosting(x, y, n_rounds=50, lr=0.1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f0 = float(y.mean())
    F = np.full(len(y), f0)
    stumps = []
    for _ in range(n_rounds):
        r = y - F
        t, lv, rv = fit_stump(x, r)
        stumps.append((t, lv, rv))
        F = F + lr * np.where(x <= t, lv, rv)
    return f0, stumps

def predict_boosting(f0, stumps, x_new, lr=0.1):
    x_new = np.asarray(x_new, dtype=float)
    out = np.full(len(x_new), f0)
    for t, lv, rv in stumps:
        out = out + lr * np.where(x_new <= t, lv, rv)
    return out
`,
        hints: ['A stump\'s best leaf values are just the means of the residuals on each side.', 'Compare stumps by the sum of squared errors of residuals around those means.', 'Keep F as an array and add `lr * np.where(x <= t, left, right)` each round.'],
        explain: `Each round fits the current residuals — the negative gradient of the squared error — with the best one-split function, then takes a small step (lr) in that direction. With many rounds, the sum of stumps approximates any 1-D curve as a staircase. Real gradient boosting uses deeper trees (capturing interactions), row/column subsampling and early stopping, but the loop is the same.`,
        tests: [
          { name: 'first stump on a step function', call: 'fit_stump(np.array([1.0, 2.0, 3.0, 4.0]), np.array([-1.0, -1.0, 1.0, 1.0]))', expect: '(2.5, -1.0, 1.0)' },
          { name: 'two rounds, exact', call: 'fit_boosting([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 3.0, 5.0], n_rounds=2, lr=0.5)' },
          { name: 'training error decreases', setup: 'rng = np.random.default_rng(0)\nx = np.sort(rng.uniform(0, 6, 60))\ny = np.sin(x) + rng.normal(0, 0.1, 60)', code: 'errs = []\nfor m in (1, 10, 50):\n    f0, st = fit_boosting(x, y, n_rounds=m, lr=0.3)\n    errs.append(np.mean((predict_boosting(f0, st, x, lr=0.3) - y) ** 2))\nassert errs[0] > errs[1] > errs[2], "training MSE should fall as rounds increase: %s" % errs', hidden: true },
          { name: 'predictions match the reference', setup: 'rng = np.random.default_rng(3)\nx = rng.uniform(0, 10, 40)\ny = 0.5 * x + np.sin(x) + rng.normal(0, 0.2, 40)\nxn = np.linspace(0, 10, 7)', call: 'predict_boosting(*fit_boosting(x, y, n_rounds=20, lr=0.2), xn, lr=0.2)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Ensemble methods', url: 'https://scikit-learn.org/stable/modules/ensemble.html' },
      { title: 'XGBoost — Introduction to boosted trees', url: 'https://xgboost.readthedocs.io/en/stable/tutorials/model.html' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
