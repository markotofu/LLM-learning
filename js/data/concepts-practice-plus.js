/* ML Atlas – content: applied additions (calibration & uncertainty, data-centric ML, SQL & data engineering, learning to rank, vector search) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'calibration-uncertainty', branch: 'ml', title: 'Calibration, Uncertainty & Conformal Prediction', short: 'Calibration & Uncertainty', level: 3, minutes: 45, prereqs: ['evaluation-metrics', 'logistic-regression'],
    tags: ['calibration', 'reliability diagram', 'expected calibration error', 'ece', 'temperature scaling', 'platt scaling', 'isotonic regression', 'uncertainty', 'aleatoric', 'epistemic', 'deep ensembles', 'mc dropout', 'conformal prediction', 'prediction sets', 'coverage'],
    summary: 'Can you trust a model\'s probabilities? Calibration and reliability diagrams, the expected calibration error, fixing overconfidence with temperature, Platt or isotonic scaling, aleatoric versus epistemic uncertainty and how ensembles estimate it, and conformal prediction — prediction sets and intervals with a guaranteed coverage rate for any model.',
    why: `A prediction is only useful for decisions if you know how much to trust it. A fraud model that says "90%" should be right about 90% of the time; a medical model should say "I don't know" for inputs unlike anything it has seen; an LLM application should know when to hand over to a human. Modern networks are often **overconfident**, so their raw probabilities mislead. This lesson shows how to measure that, fix it cheaply, separate the different kinds of uncertainty, and wrap any model in **conformal prediction** to get answers that come with a guaranteed error rate.`,
    objectives: [
      'Explain calibration and measure it with a reliability table and the expected calibration error',
      'Recalibrate a model with temperature scaling, Platt scaling or isotonic regression',
      'Distinguish aleatoric from epistemic uncertainty and estimate the latter with ensembles',
      'Build split-conformal prediction intervals and sets, and state what their guarantee does and does not cover'
    ],
    sections: [
      { type: 'text', title: 'Is "80% sure" right 80% of the time?', md: `A classifier is **calibrated** if, among all predictions made with confidence p, a fraction p turn out to be correct. Calibration is separate from accuracy: two models with the same accuracy can be calibrated very differently. Deep networks trained to minimize log loss on large datasets are often **overconfident** (Guo et al., 2017), and the confidence an LLM states in words is frequently off too.

A **reliability diagram** groups predictions into bins by confidence and compares each bin's average confidence with its accuracy; a perfectly calibrated model lies on the diagonal. The **expected calibration error** summarizes the gaps:

$$ ECE = Σ_b ( n_b / n ) · | acc(b) − conf(b) | $$
:::where
b — a confidence bin, e.g. all predictions with confidence between 0.8 and 0.9
n_b — the number of predictions in bin b
n — the total number of predictions
acc(b) — the fraction of predictions in bin b that were correct
conf(b) — the average confidence of the predictions in bin b
:::

Log loss and the **Brier score** (the mean squared error of the probabilities) are *proper scoring rules*: they reward calibration and sharpness together, and are good companions to ECE.` },
      { type: 'code', title: 'A reliability table for an overconfident model', code: `
import numpy as np
rng = np.random.default_rng(0)
n, k = 6000, 4
true_logits = rng.normal(size=(n, k)) * 1.5

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

p_true = softmax(true_logits)                                           # the real class probabilities
labels = (rng.random(n)[:, None] > np.cumsum(p_true, axis=1)).sum(axis=1)   # draw one label per row
model_logits = 2.5 * true_logits                                        # right ranking, but logits far too large

def reliability(P, y, bins=10):
    conf, correct = P.max(axis=1), P.argmax(axis=1) == y
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    ece, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        if m.sum() >= 20:
            rows.append((b / bins, (b + 1) / bins, m.sum(), conf[m].mean(), correct[m].mean()))
    return ece, rows

P = softmax(model_logits)
ece, rows = reliability(P, labels)
print(f"accuracy {np.mean(P.argmax(axis=1) == labels):.3f}   ECE {ece:.3f}")
print("confidence bin   count   mean confidence   accuracy")
for lo, hi, cnt, c, a in rows:
    print(f"  {lo:.1f} - {hi:.1f}    {cnt:5d}   {c:15.3f}   {a:8.3f}")
`, after: `In every bin accuracy falls well short of confidence: when this model says about 97%, it is right only about 75% of the time. The expected calibration error of about 0.21 summarizes that gap.` },
      { type: 'text', title: 'Fixing calibration after training', md: `Recalibration methods learn a small mapping from the model's scores to better probabilities, on **held-out data the model was not trained on**:

- **Temperature scaling** divides all logits by one number T, chosen to minimize log loss on a validation set:

$$ p = softmax( z / T ) $$
:::where
z — the model's logits for one example
T — the temperature, a single number fitted on held-out data; T > 1 softens overconfident predictions
:::

  Because dividing by T never changes which logit is largest, predictions and accuracy stay exactly the same — only the confidence changes. It is the standard first fix for neural networks.
- **Platt scaling** fits a logistic regression σ(a·s + b) on the score s of a binary classifier (classically for SVMs).
- **Isotonic regression** fits any increasing step function from score to probability — more flexible, but it needs more calibration data.` },
      { type: 'code', title: 'Temperature scaling fixes it with one number', code: `
import numpy as np
rng = np.random.default_rng(0)                         # the same overconfident model as above
n, k = 6000, 4
true_logits = rng.normal(size=(n, k)) * 1.5
softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
labels = (rng.random(n)[:, None] > np.cumsum(softmax(true_logits), axis=1)).sum(axis=1)
model_logits = 2.5 * true_logits

def ece(P, y, bins=10):
    conf, correct = P.max(1), P.argmax(1) == y
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(np.mean(idx == b) * abs(correct[idx == b].mean() - conf[idx == b].mean()) for b in range(bins) if np.any(idx == b))

def nll(P, y):
    return -np.mean(np.log(P[np.arange(len(y)), y]))

val, test = slice(0, 3000), slice(3000, None)          # fit T on one half, evaluate on the other
temps = np.linspace(0.5, 5, 91)
T = min(temps, key=lambda t: nll(softmax(model_logits[val] / t), labels[val]))
before, after = softmax(model_logits[test]), softmax(model_logits[test] / T)
print(f"fitted temperature T = {T:.2f}")
print(f"test ECE       before {ece(before, labels[test]):.3f}   after {ece(after, labels[test]):.3f}")
print(f"test accuracy  before {np.mean(before.argmax(1) == labels[test]):.3f}   after {np.mean(after.argmax(1) == labels[test]):.3f}")
`, after: `The fitted temperature recovers the factor by which the logits were inflated, the calibration error drops by an order of magnitude, and accuracy does not move at all.`, predict: {"type":"mcq","q":"Temperature scaling divides every logit by the same number T (about 2.5 here). What happens to the test accuracy?","options":["It rises, because the probabilities are better","It falls, because the model becomes less confident","It stays exactly the same","It drops to chance"],"answer":2,"explain":"Accuracy stays at 0.609. Dividing all logits by the same positive number never changes which one is largest, so every prediction is unchanged; only the confidence changes (ECE 0.206 → 0.013).","expect":["test accuracy  before 0.609   after 0.609"]} },
      { type: 'chart', title: "Reliability diagram: before and after temperature scaling", md: `The 4-class model from the reliability table above, binned by confidence:`, after: `The original model's curve sags far below the diagonal: in every bin it is more confident than it is accurate. Dividing all logits by a single temperature T = 2.54 (fitted by minimising the log loss) brings every bin onto the diagonal, without changing a single prediction.`, chart: {"kind":"line","series":[{"name":"perfect calibration","x":[0.25,1],"y":[0.25,1],"dash":true},{"name":"original model","x":[0.373,0.459,0.549,0.649,0.751,0.854,0.97],"y":[0.274,0.368,0.413,0.44,0.531,0.592,0.754]},{"name":"after temperature scaling (T = 2.54)","x":[0.362,0.453,0.548,0.648,0.748,0.847,0.937],"y":[0.353,0.454,0.554,0.622,0.754,0.863,0.933]}],"xLabel":"mean confidence in the bin","yLabel":"accuracy in the bin","xMin":0.25,"xMax":1,"yMin":0.25,"yMax":1} },
      { type: 'demo', title: 'Interactive: a reliability diagram and the temperature knob', demo: 'calibration' },
      { type: 'check', question: { type: 'tf', q: 'Temperature scaling can change which class a model predicts.', answer: false, explain: 'Dividing all logits by the same positive number keeps their order.' } },
      { type: 'text', title: "Scoring probabilities: the Brier score and log loss", md: `Accuracy ignores the probabilities entirely, and the ECE checks only calibration: a model that always predicts the base rate is perfectly calibrated and useless. To compare probabilistic predictions as a whole, use a **proper scoring rule**: a score that is optimised, in expectation, only by reporting the true probabilities, so no model can game it by hedging or exaggerating.

$$ Brier = (1/n) Σᵢ Σ_k ( p_ik − y_ik )²          log loss = −(1/n) Σᵢ log p_i,yᵢ $$
:::where
p_ik — the probability the model gave class k for example i
y_ik — 1 if example i's true class is k, otherwise 0
p_i,yᵢ — the probability the model gave the true class of example i
n — the number of examples
:::

The **Brier score** is the mean squared error of the probabilities: bounded, easy to explain and gentle with confident mistakes. **Log loss** (cross-entropy) punishes a confident mistake without limit, so it is the stricter judge, and the one models are trained on. Both reward calibration *and* sharpness (confident predictions that turn out right), which is why temperature scaling improves them while leaving the accuracy untouched.` },
      { type: 'code', title: "Temperature scaling improves the scores, not the accuracy", md: `The overconfident 4-class model from above. T is fitted here on the same data for brevity; in practice, fit it on a held-out calibration set.`, code: `import numpy as np
rng = np.random.default_rng(0)
n, k = 6000, 4
true_logits = rng.normal(size=(n, k)) * 1.5

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

labels = (rng.random(n)[:, None] > np.cumsum(softmax(true_logits), axis=1)).sum(axis=1)
model_logits = 2.5 * true_logits                          # right ranking, logits far too large
onehot = np.eye(k)[labels]

def scores(P):
    accuracy = np.mean(P.argmax(axis=1) == labels)
    brier = np.mean(np.sum((P - onehot) ** 2, axis=1))
    log_loss = -np.mean(np.log(P[np.arange(n), labels]))
    return accuracy, brier, log_loss

temps = np.linspace(0.5, 5, 451)
best_T = temps[np.argmin([scores(softmax(model_logits / T))[2] for T in temps])]
results = {}
for name, P in [("original", softmax(model_logits)), (f"T = {best_T:.2f}", softmax(model_logits / best_T))]:
    results[name] = scores(P)
    a, b, l = results[name]
    print(f"{name:<9}  accuracy {a:.3f}   Brier {b:.3f}   log loss {l:.3f}")
acc = [v[0] for v in results.values()]
print("accuracy unchanged:", acc[0] == acc[1])`, after: `Same accuracy, but a clearly better Brier score and a much better log loss: the original model's confident mistakes were costing it dearly. Proper scoring rules see the difference that accuracy cannot.`, predict: {"type":"mcq","q":"Temperature scaling divides every logit by the same T. Which of these will change?","options":["Only the accuracy","Only the Brier score and the log loss","All three","None of them"],"answer":1,"explain":"Dividing all logits by a positive T never changes which class has the largest logit, so the predictions and the accuracy stay the same; the probabilities, and therefore the scores, change.","expect":["accuracy unchanged: True"]} },
      { type: 'text', title: 'Two kinds of uncertainty', md: `- **Aleatoric** uncertainty is noise in the data itself — a blurry photo, an ambiguous question, a coin flip. More data does not remove it.
- **Epistemic** uncertainty is the model's lack of knowledge — inputs unlike anything in the training data. More (relevant) data reduces it, and it is what should make a model say "I don't know" on out-of-distribution inputs.

Estimating epistemic uncertainty: exact Bayesian methods for small models (see *Bayesian Inference* and *Gaussian Processes*); **deep ensembles** — train several networks from different random initializations and measure how much they disagree (simple and strong); and **MC dropout** — keep dropout active at test time and average several stochastic passes. For an ensemble of M regressors, each predicting a mean and a noise variance, the total predictive variance splits neatly:

$$ Var[y] ≈ (1/M) Σ_m σ_m²  +  (1/M) Σ_m ( μ_m − μ̄ )² $$
:::where
M — the number of models in the ensemble
μ_m, σ_m² — the mean and the noise variance predicted by model m
μ̄ — "mu bar": the average of the models' means, used as the prediction
the first term — aleatoric, the noise the models expect in the data
the second term — epistemic, how much the models disagree with each other
:::` },
      { type: 'text', title: 'Conformal prediction: coverage guarantees for any model', md: `**Split conformal prediction** turns any trained model into one that outputs sets or intervals with a guaranteed hit rate:

1. Hold out a **calibration set** the model has not seen. For each calibration example compute a **nonconformity score** — for regression |y − ŷ|; for classification 1 − p̂(true class).
2. Take the quantile q of those scores, and output every answer whose score would be at most q:

$$ q = the ⌈(n + 1)(1 − α)⌉-th smallest calibration score          C(x) = { y : score(x, y) ≤ q } $$
:::where
n — the number of calibration examples
α — "alpha": the allowed error rate, e.g. 0.1 for 90% coverage
⌈ … ⌉ — rounding up to a whole number
C(x) — the prediction set for a new input x, an interval ŷ ± q in regression, or every class with 1 − p̂(class) ≤ q in classification
:::

**The guarantee:** if calibration and test examples are exchangeable (for instance i.i.d.), the set contains the true answer with probability at least 1 − α — however good or bad the model is. A bad model simply produces wide intervals or large sets. **The caveat:** the guarantee is *marginal*, an average over all inputs; it does not promise 90% coverage for every subgroup or region. Adaptive scores (for example conformalized quantile regression) produce intervals that widen where the data is noisier, and the guarantee breaks under distribution shift.` },
      { type: 'code', title: 'Split conformal intervals for regression', code: `
import numpy as np
rng = np.random.default_rng(1)

def make(n):                                               # the noise grows with x
    x = rng.uniform(0, 10, n)
    return x, np.sin(x) + 0.1 * x + rng.normal(0, 0.3 + 0.05 * x, n)

x_train, y_train = make(500)
x_cal, y_cal = make(500)
x_test, y_test = make(5000)
coef = np.polyfit(x_train, y_train, 5)                     # any model would do
predict = lambda x: np.polyval(coef, x)

scores = np.sort(np.abs(y_cal - predict(x_cal)))           # nonconformity scores on the calibration set
for alpha in (0.2, 0.1, 0.05):
    q = scores[int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1]
    inside = np.abs(y_test - predict(x_test)) <= q
    print(f"target {1 - alpha:.0%}: interval ±{q:.2f}  coverage {inside.mean():.1%}  "
          f"(where x < 3: {inside[x_test < 3].mean():.1%},  where x > 7: {inside[x_test > 7].mean():.1%})")
`, after: `Overall coverage meets the target every time. But one fixed width over-covers where the data is quiet (small x) and under-covers where it is noisy (large x): the guarantee is about the average, not every region.` },
      { type: 'code', title: "Uncertainty from scikit-learn classifiers", md: `scikit-learn classifiers offer two ways to see how sure they are, besides \`predict\`:

- **\`decision_function\`** returns a raw score. For two classes it is one number per example: positive means the second class in \`classes_\`, negative the first, and its size says how confident the model is, on a scale that depends on the model.
- **\`predict_proba\`** returns one probability per class; each row adds up to 1, and the predicted class is the column with the largest value.`, code: `import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
names = iris.target_names[iris.target]                 # use the species names as labels
X_train, X_test, y_train, y_test = train_test_split(iris.data, names, random_state=42)
gbc = GradientBoostingClassifier(learning_rate=0.01, random_state=0).fit(X_train, y_train)

print("classes_:", gbc.classes_)
scores = gbc.decision_function(X_test[:4])
print("decision_function shape:", gbc.decision_function(X_test).shape)   # one column per class
print(scores.round(2))
proba = gbc.predict_proba(X_test[:4])
print("predict_proba:")
print(proba.round(3))
print("rows sum to 1:", np.allclose(proba.sum(axis=1), 1))
print("argmax of scores, mapped through classes_:", gbc.classes_[np.argmax(scores, axis=1)])
print("predict:                                  ", gbc.predict(X_test[:4]))`, after: `- With three classes, both methods return one column per class, in the order of \`classes_\`. Always map column numbers through \`classes_\`: the classes are sorted, and with string labels the column number is **not** the label.
- The argmax of either output reproduces \`predict\`.
- Whether these probabilities can be trusted as real frequencies is exactly the calibration question of this lesson: a model can be confidently wrong, so check a reliability diagram before relying on the numbers.` }
    ],
    when: `- Any decision that uses probabilities: thresholds, expected costs, ranking by risk, deferring to a human.
- Combining models or feeding probabilities into downstream systems (they must mean what they say).
- Safety-critical or regulated settings that need error bars or guaranteed coverage (medicine, finance, autonomous systems).
- Detecting inputs the model should not be trusted on (out-of-distribution data), using epistemic uncertainty.`,
    pitfalls: `- Fitting calibration on the training set — use held-out data.
- Reporting ECE alone; it depends on the binning and can be gamed. Pair it with log loss or the Brier score.
- Assuming calibration transfers to new data distributions — recalibrate after shifts.
- Reading a conformal guarantee as conditional: 90% on average can mean much less for a hard subgroup.
- Treating softmax confidence as a measure of epistemic uncertainty — networks can be confidently wrong far from the data.`,
    takeaways: [
      'Calibrated means confidence p is right a fraction p of the time; measure with reliability tables and ECE.',
      'Temperature scaling fits one number on held-out data and never changes accuracy.',
      'Aleatoric uncertainty is noise in the data; epistemic uncertainty is lack of knowledge — ensembles estimate it.',
      'Split conformal: the ⌈(n + 1)(1 − α)⌉-th smallest calibration score gives sets with ≥ 1 − α coverage for any model.',
      'Conformal coverage is marginal, not per subgroup, and assumes exchangeable data.'
    ],
    quiz: [
      { type: 'mcq', q: 'Predictions a model makes with 90% confidence are correct 70% of the time. The model is…', options: ['Underconfident', 'Overconfident', 'Perfectly calibrated', 'Calibrated but inaccurate'], answer: 1, explain: 'Its confidence exceeds its accuracy.' },
      { type: 'mcq', q: 'Which kind of uncertainty can be reduced by collecting more relevant data?', options: ['Aleatoric', 'Epistemic', 'Both equally', 'Neither'], answer: 1, explain: 'Epistemic uncertainty comes from lack of knowledge; aleatoric is noise.' },
      { type: 'fill', q: 'Split conformal with n = 99 calibration scores and α = 0.1: which sorted score (its rank, counting from 1) is used as q?', answer: ['90', '90th'], explain: '⌈(99 + 1) × 0.9⌉ = 90.' },
      { type: 'mcq', q: 'What does split conformal prediction guarantee?', options: ['Every prediction set is small', 'The true answer is in the set with probability at least 1 − α, on average over new data', 'The model becomes calibrated', 'Coverage of 1 − α in every subgroup'], answer: 1, explain: 'Marginal coverage, for any model, under exchangeability.' },
      { type: 'tf', q: 'Deep ensembles estimate epistemic uncertainty by measuring how much independently trained models disagree.', answer: true, explain: 'Disagreement grows where the data did not pin the function down.' },
      {"type":"mcq","q":"A model predicts the base rate (say 30%) for every example. Which statement is true?","options":["It is perfectly calibrated but useless for ranking","It has the best possible log loss","It is badly calibrated","Its accuracy is 100%"],"answer":0,"explain":"Among all examples given 30%, about 30% are positive, so it is calibrated; but it cannot tell any two examples apart. Proper scoring rules penalise this lack of sharpness."},
      {"type":"mcq","q":"Why is log loss called a proper scoring rule?","options":["It is always between 0 and 1","Its expected value is optimised only by predicting the true probabilities","It ignores calibration","It counts correct predictions"],"answer":1,"explain":"No model can improve its expected log loss by reporting probabilities other than the ones it believes are true, so hedging and exaggerating are both punished."}
    ],
    problems: [
      {
        id: 'calibration-ece', title: 'Expected Calibration Error and Temperature Scaling', difficulty: 'medium',
        prompt: `Implement:

- \`ece(probs, labels, n_bins=10)\` → the expected calibration error of a classifier. \`probs\` is (n, k): each row's confidence is its largest probability and its prediction the argmax. Put each prediction in bin min(⌊confidence · n_bins⌋, n_bins − 1) and sum, over non-empty bins, (n_b / n) · |accuracy in the bin − mean confidence in the bin|.
- \`temperature_scale(logits, T)\` → softmax(logits / T) row by row (subtract each row's maximum for stability).
- \`fit_temperature(logits, labels, temps)\` → the temperature in \`temps\` with the lowest average negative log-likelihood −mean(log p[i, labels[i]]); the first one if tied.`,
        starter: `
import numpy as np

def ece(probs, labels, n_bins=10):
    # TODO
    pass

def temperature_scale(logits, T):
    # TODO
    pass

def fit_temperature(logits, labels, temps):
    # TODO
    pass
`,
        solution: `
import numpy as np

def ece(probs, labels, n_bins=10):
    P = np.asarray(probs, dtype=float)
    y = np.asarray(labels)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)

def temperature_scale(logits, T):
    Z = np.asarray(logits, dtype=float) / T
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

def fit_temperature(logits, labels, temps):
    y = np.asarray(labels)
    best, best_nll = None, np.inf
    for T in temps:
        P = temperature_scale(logits, T)
        nll = -np.mean(np.log(P[np.arange(len(y)), y]))
        if nll < best_nll:
            best, best_nll = T, nll
    return best
`,
        hints: ['Compute confidences and correctness once, then loop over bins with a boolean mask.', 'Keep the first temperature that achieves a strictly lower loss.'],
        explain: `ECE is a weighted average gap between confidence and accuracy. Temperature scaling is the cheapest effective fix: one parameter, fitted by log loss on held-out data, with no change to accuracy.`,
        tests: [
          { name: 'a perfectly calibrated toy case', call: 'ece(np.array([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]), [0, 0, 0, 1])', expect: '0.0' },
          { call: 'ece(np.array([[0.9, 0.1], [0.9, 0.1]]), [0, 1])', expect: '0.4' },
          { call: 'temperature_scale(np.array([[2.0, 0.0]]), 2.0)', expect: 'np.array([[np.e / (np.e + 1), 1 / (np.e + 1)]])' },
          { name: 'recovers an inflation factor', setup: 'rng = np.random.default_rng(0)\nZ = rng.normal(size=(4000, 3))\nP = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)\ny = (rng.random(4000)[:, None] > np.cumsum(P, axis=1)).sum(axis=1)', code: 'T = fit_temperature(3 * Z, y, np.linspace(1, 5, 41))\nassert abs(T - 3) < 0.4, T', hidden: true },
          { name: 'random data vs reference', setup: 'rng = np.random.default_rng(1)\nL = rng.normal(size=(200, 4)) * 2; y = rng.integers(0, 4, 200)', call: '[ece(temperature_scale(L, 1.0), y), fit_temperature(L, y, [0.5, 1.0, 2.0, 4.0])]', hidden: true }
        ]
      },
      {
        id: 'conformal-sets', title: 'Conformal Intervals and Prediction Sets', difficulty: 'medium',
        prompt: `Implement split conformal prediction:

- \`conformal_quantile(scores, alpha)\` → the ⌈(n + 1)(1 − alpha)⌉-th smallest calibration score (counting from 1). If that rank exceeds n, return \`float('inf')\`.
- \`conformal_interval(pred, q)\` → \`(pred - q, pred + q)\` for a prediction (a number or an array).
- \`prediction_sets(probs, q)\` → for each row of class probabilities (n, k), the **list** of class indices c whose score 1 − probs[i, c] is at most q. Return a list of lists.

\`\`\`text
conformal_quantile([0.1, 0.4, 0.2, 0.3], 0.5)  ->  rank ⌈5 · 0.5⌉ = 3  ->  0.3
\`\`\``,
        starter: `
import math
import numpy as np

def conformal_quantile(scores, alpha):
    # TODO
    pass

def conformal_interval(pred, q):
    # TODO
    pass

def prediction_sets(probs, q):
    # TODO
    pass
`,
        solution: `
import math
import numpy as np

def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, dtype=float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    if k > len(s):
        return float('inf')
    return float(s[k - 1])

def conformal_interval(pred, q):
    return pred - q, pred + q

def prediction_sets(probs, q):
    P = np.asarray(probs, dtype=float)
    return [[int(c) for c in range(P.shape[1]) if 1 - P[i, c] <= q] for i in range(P.shape[0])]
`,
        hints: ['Sort the scores and index at rank − 1.', 'The +1 in (n + 1) accounts for the new test point itself.'],
        explain: `The quantile with the (n + 1) correction is what makes the coverage guarantee exact in finite samples. With too few calibration points for the requested α the honest answer is an infinite interval — "anything is possible".`,
        tests: [
          { call: 'conformal_quantile([0.1, 0.4, 0.2, 0.3], 0.5)', expect: '0.3' },
          { call: 'conformal_quantile([0.5, 0.1], 0.1)', expect: 'float("inf")' },
          { call: 'prediction_sets(np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]]), 0.7)', expect: '[[0], [0, 1]]' },
          { name: 'coverage on simulated data', setup: 'rng = np.random.default_rng(0)\ncal = np.abs(rng.normal(size=999)); test = np.abs(rng.normal(size=20000))', code: 'q = conformal_quantile(cal, 0.1)\nlo, hi = conformal_interval(0.0, q)\ncov = np.mean(test <= hi)\nassert 0.88 < cov < 0.92, cov', hidden: true },
          { name: 'random sets vs reference', setup: 'rng = np.random.default_rng(1)\nP = rng.dirichlet(np.ones(4), size=6)', call: 'prediction_sets(P, 0.75)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Guo et al. — On Calibration of Modern Neural Networks', url: 'https://arxiv.org/abs/1706.04599' },
      { title: 'Angelopoulos & Bates — A Gentle Introduction to Conformal Prediction', url: 'https://arxiv.org/abs/2107.07511' },
      { title: 'Lakshminarayanan et al. — Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles', url: 'https://arxiv.org/abs/1612.01474' },
      { title: 'scikit-learn — Probability calibration', url: 'https://scikit-learn.org/stable/modules/calibration.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'data-centric', branch: 'ml', title: 'Data-Centric ML: Augmentation, Imbalance & Learning with Few Labels', short: 'Data-Centric ML', level: 2, minutes: 45, prereqs: ['evaluation-metrics', 'cross-validation'],
    tags: ['data-centric ai', 'data augmentation', 'mixup', 'class imbalance', 'class weights', 'resampling', 'smote', 'threshold moving', 'label noise', 'semi-supervised learning', 'pseudo-labeling', 'active learning', 'uncertainty sampling', 'weak supervision', 'synthetic data'],
    summary: 'Improving models by improving their data: label-preserving data augmentation and mixup, handling imbalanced classes with weights, resampling and threshold moving, finding noisy labels, and learning with few labels through semi-supervised learning, active learning, weak supervision and synthetic data.',
    why: `On most real projects the model architecture is a solved problem and the **data** is the bottleneck: too few labels, rare classes, wrong labels, and training data that doesn't cover what the model will see. A data-centric approach holds the model fixed and improves the dataset — and often gains more than any model change. These techniques are everyday tools for practitioners, from image augmentation in every vision pipeline to class weights in fraud detection and active learning when labels cost money.`,
    objectives: [
      'Use data augmentation and mixup to teach invariances and regularize models',
      'Handle class imbalance with class weights, resampling and threshold moving — and evaluate it correctly',
      'Find likely label errors with out-of-fold predictions',
      'Choose between semi-supervised learning, active learning, weak supervision and synthetic data when labels are scarce'
    ],
    sections: [
      { type: 'text', title: 'Better data often beats a better model', md: `**Data-centric AI** flips the usual loop: keep the model fixed and iterate on the data. The workflow is error analysis — look at the mistakes, slice the metrics by data source, class, length or region, and find *why* the model fails — followed by targeted fixes: correct wrong labels, add examples of the failing slice, clarify ambiguous labelling guidelines, remove duplicates and leakage. A few hundred carefully chosen or corrected examples routinely help more than a new architecture.` },
      { type: 'text', title: 'Data augmentation', md: `**Augmentation** creates new training examples by transforming existing ones in ways that do not change the label:

- images: flips, crops, small rotations and shifts, colour jitter, blur;
- text: back-translation (translate and translate back), synonym replacement, LLM paraphrases;
- audio: time shifts, speed changes, added noise, masking blocks of the spectrogram (SpecAugment).

It teaches the model **invariances** you know are true and acts as a strong regularizer. **Mixup** goes further and blends two examples and their labels:

$$ x̃ = λ x_i + (1 − λ) x_j          ỹ = λ y_i + (1 − λ) y_j          λ ~ Beta(α, α) $$
:::where
x_i, x_j — two training examples
y_i, y_j — their one-hot labels
λ — "lambda": a random mixing weight between 0 and 1
Beta(α, α) — the distribution λ is drawn from; a small α such as 0.2 keeps most mixes close to one of the two originals
x̃, ỹ — the mixed example and its soft label
:::

Rules: a transformation must really preserve the label (a horizontally flipped "6" may become something else, and a rotated "6" a "9"); augment the training data only, never the validation or test sets.` },
      { type: 'code', title: 'Augmentation teaches invariance', code: `
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
X, y = digits.images / 16.0, digits.target                 # 8 x 8 images of handwritten digits
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
train, test = order[:300], order[300:]                     # only 300 training images

def shift(img, dx, dy):                                    # move an image by whole pixels, filling with blank
    out = np.zeros_like(img)
    h, w = img.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = img[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out

moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
picks = rng.integers(len(moves), size=len(test))
X_test_shifted = np.array([shift(X[i], *moves[m]) for i, m in zip(test, picks)])   # test digits nudged by up to 1 pixel
X_aug = np.array([shift(X[i], dx, dy) for i in train for dx, dy in moves])        # 9 shifted copies of each training image
y_aug = np.repeat(y[train], len(moves))

flat = lambda A: A.reshape(len(A), -1)
for name, (A, b) in [("original 300 images", (X[train], y[train])), ("with shifted copies", (X_aug, y_aug))]:
    clf = KNeighborsClassifier(n_neighbors=1).fit(flat(A), b)
    print(f"{name:20s}  accuracy on centred test digits {clf.score(flat(X[test]), y[test]):.3f}   "
          f"on shifted test digits {clf.score(flat(X_test_shifted), y[test]):.3f}")
`, after: `Nudging the test digits by a single pixel halves the accuracy of the model trained on the originals — nothing told it that position doesn't matter. Training on shifted copies teaches exactly that invariance and restores accuracy on shifted digits without hurting the centred ones. Convolutional networks build part of this invariance into their architecture; augmentation supplies the rest.` },
      { type: 'text', title: 'Imbalanced classes', md: `When one class is rare (fraud, disease, defects), a model can score high accuracy by ignoring it. First fix the **evaluation**: precision, recall, the precision–recall curve and costs, not accuracy. Then the main tools:

- **Class weights** make each mistake on a rare example count more in the loss:

$$ w_c = n / ( k · n_c ) $$
:::where
w_c — the weight of every example of class c in the loss
n — the total number of training examples
k — the number of classes
n_c — the number of examples of class c, so rare classes get large weights
:::

- **Resampling**: oversample the minority class (repeat examples, or **SMOTE**, which synthesizes new points by interpolating between minority neighbours) or undersample the majority. Resample *only the training folds*, after splitting — never the validation or test data.
- **Threshold moving**: keep the model, but choose the decision threshold on validation data to hit the precision/recall trade-off the application needs.
- **Focal loss** down-weights easy, well-classified examples so training concentrates on hard ones.` },
      { type: 'code', title: 'Class weights versus moving the threshold', code: `
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=20000, n_features=10, n_informative=4, weights=[0.97, 0.03], class_sep=0.8, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)

def report(name, pred):
    tp = np.sum((pred == 1) & (y_te == 1))
    fp = np.sum((pred == 1) & (y_te == 0))
    fn = np.sum((pred == 0) & (y_te == 1))
    print(f"{name:26s} accuracy {np.mean(pred == y_te):.3f}   recall {tp / (tp + fn):.3f}   precision {tp / max(tp + fp, 1):.3f}")

plain = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
weighted = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)
print(f"share of positives in the test set: {y_te.mean():.3f}")
report("plain, threshold 0.5", plain.predict(X_te))
report("class weights, 0.5", weighted.predict(X_te))
report("plain, threshold 0.2", (plain.predict_proba(X_te)[:, 1] >= 0.2).astype(int))
report("plain, threshold 0.1", (plain.predict_proba(X_te)[:, 1] >= 0.1).astype(int))
`, after: `At the default threshold the plain model never predicts the rare class, and its 96.6% accuracy is exactly what always answering "negative" would score. Class weights and lower thresholds both trade precision for recall; they mostly change where the model's ranking is cut, so choose the operating point from the real costs of misses and false alarms.`, predict: {"type":"mcq","q":"Only 3.4% of the test examples are positive. A plain logistic regression with threshold 0.5 reaches 96.6% accuracy. What is its recall on the positive class?","options":["About 97%","About 50%","About 3%","0%"],"answer":3,"explain":"Recall is 0.000: the model predicts \"negative\" for everyone, and that alone earns 96.6% accuracy. On imbalanced data accuracy hides failure, so look at recall and precision.","expect":["plain, threshold 0.5       accuracy 0.966   recall 0.000"]} },
      { type: 'check', question: { type: 'mcq', q: 'When should you oversample the minority class?', options: ['Before splitting into train and test sets', 'Only on the training folds, after splitting', 'Only on the test set', 'On all data, then split'], answer: 1, explain: 'Resampling before splitting leaks copies of training examples into the evaluation data.' } },
      { type: 'text', title: 'Noisy labels', md: `Labels are wrong more often than people think — even widely used benchmark test sets contain label errors (Northcutt et al., 2021). Noisy labels cap accuracy and corrupt evaluation. To find them:

1. Get **out-of-fold** predicted probabilities: with k-fold cross-validation, every example is scored by a model that never saw it.
2. Flag examples where that model confidently predicts a different class than the label (the idea behind *confident learning*).
3. Review the flagged examples by hand, starting with the most confident disagreements, and fix or remove them.

Also: measure agreement between annotators, write precise labelling guidelines, and consider robust losses when noise cannot be cleaned.` },
      { type: 'chart', title: "Test accuracy as training labels get noisier", md: `A synthetic two-class problem; a growing share of the training labels is flipped, and every model is tested on clean labels:`, after: `The 1-nearest-neighbour classifier copies the label of a single training point, so every flipped label becomes a wrong prediction around it: its accuracy falls almost in step with the noise. Averaging over 25 neighbours outvotes isolated errors until about 30% noise, and the simple logistic regression barely moves. Flexible models memorise noise, which is why label quality matters most exactly where models are most powerful.`, chart: {"kind":"line","x":[0,5,10,15,20,25,30,35,40],"series":[{"name":"1-nearest neighbour","y":[0.924,0.893,0.851,0.81,0.762,0.719,0.674,0.622,0.58]},{"name":"25-nearest neighbours","y":[0.924,0.925,0.924,0.923,0.923,0.912,0.894,0.833,0.76]},{"name":"logistic regression","y":[0.883,0.883,0.88,0.874,0.871,0.869,0.872,0.852,0.835]}],"xLabel":"share of training labels flipped at random (%)","yLabel":"accuracy on clean test labels"} },
      { type: 'text', title: "Finding label errors with out-of-fold predictions", md: `Real datasets contain mistakes; studies of popular benchmarks have found that a few percent of their test labels are wrong. Checking every label by hand is too expensive, but a model can point to the suspicious ones, provided it never judges an example it was trained on:

1. Get **out-of-fold** predicted probabilities with cross-validation (\`cross_val_predict(..., method="predict_proba")\`), so every example is scored by a model that did not see it.
2. For each example, look at the probability the model gives to its **given** label.
3. Review the examples with the lowest such probability first: there the model confidently disagrees with the label.

This is the core of **confident learning** (the cleanlab library), which also estimates how often each class is mislabelled as each other class. The flagged examples go to a human, not straight into the bin: some will be genuinely hard or rare cases, the last examples you want to delete.` },
      { type: 'code', title: "Flagging suspicious labels", code: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
X, y = make_classification(2000, n_features=10, n_informative=5, random_state=0)
rng = np.random.default_rng(3)
noisy = y.copy()
wrong = rng.choice(len(y), 100, replace=False)                   # corrupt 100 of the 2,000 labels (5%)
noisy[wrong] = 1 - noisy[wrong]
is_wrong = np.zeros(len(y), dtype=bool)
is_wrong[wrong] = True

proba = cross_val_predict(KNeighborsClassifier(n_neighbors=15), X, noisy, cv=5, method="predict_proba")
p_given = proba[np.arange(len(noisy)), noisy]                     # how much the model believes each given label
suspects = np.argsort(p_given)[:50]                               # the 50 least believable labels
print("truly wrong among the 50 flagged labels:", int(is_wrong[suspects].sum()))
print("expected if we picked 50 at random:      ", 50 * 100 // 2000)
print("most flagged labels are real errors:", bool(is_wrong[suspects].mean() > 0.6))`, after: `A random sample of 50 would contain about 2 or 3 of the corrupted labels; the 50 labels the out-of-fold model believes least are mostly real errors. Reviewing those first gives a human checker the most value per minute.`, predict: {"type":"mcq","q":"5% of the labels were flipped at random. Among the 50 labels the model finds least believable, roughly how many are real errors?","options":["About 2 or 3, as for a random pick","About 10","Most of them","Always exactly 50"],"answer":2,"explain":"A flipped label usually contradicts everything similar examples say, so a model that never saw it gives it a very low probability.","expect":["most flagged labels are real errors: True"]} },
      { type: 'text', title: 'Learning with few labels', md: `When labelled data is scarce but unlabelled data is plentiful:

- **Transfer learning**: start from a pretrained model, or train a small classifier on embeddings from a foundation model — often the strongest first step.
- **Semi-supervised learning**: **pseudo-labelling** trains a model, labels the unlabelled examples it is confident about, and retrains on both; **consistency regularization** asks for the same prediction on differently augmented versions of an unlabelled example.
- **Active learning**: let the model choose which examples a human should label next — typically the ones it is most uncertain about (least confident, smallest margin between the top two classes, or highest entropy).
- **Weak supervision**: write several noisy labelling functions (keyword rules, heuristics, existing databases) and combine their votes into training labels.
- **Synthetic data**: generate examples with simulators or LLMs (paraphrases, new instructions, rare cases). Filter for quality and diversity, and keep real data in the mix — models trained repeatedly on their own outputs degrade ("model collapse").` },
      { type: 'code', title: 'Active learning: ask for the labels that teach the most', code: `
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=3000, n_features=20, n_informative=5, random_state=1)
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=1000, random_state=1)

def run(strategy, seed, budget=80):
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(X_pool), 10, replace=False))     # start with 10 random labels
    scores = {}
    while True:
        clf = LogisticRegression(max_iter=1000).fit(X_pool[labeled], y_pool[labeled])
        if len(labeled) in (20, 40, 80):
            scores[len(labeled)] = clf.score(X_test, y_test)
        if len(labeled) == budget:
            return scores
        rest = np.setdiff1d(np.arange(len(X_pool)), labeled)
        if strategy == "random":
            labeled.append(rng.choice(rest))
        else:                                                        # uncertainty sampling
            p = clf.predict_proba(X_pool[rest])[:, 1]
            labeled.append(rest[np.argmin(np.abs(p - 0.5))])

for strategy in ("random", "uncertainty"):
    runs = [run(strategy, seed) for seed in range(5)]
    print(f"{strategy:12s}", "   ".join(f"{n} labels: {np.mean([r[n] for r in runs]):.3f}" for n in (20, 40, 80)))
`, after: `Averaged over five runs, asking for the labels the model is least sure about gives clearly better accuracy for the same budget once a few dozen labels are in. With very few labels the advantage is small — the model's uncertainty is itself unreliable — which is why practical systems often mix in some random picks.` }
    ],
    when: `- Small or expensive datasets: augmentation, transfer learning, active learning.
- Rare-event problems (fraud, defects, disease): class weights, resampling and threshold moving with proper metrics.
- When accuracy plateaus: error analysis and label cleaning before new architectures.
- Large unlabelled pools: semi-supervised learning, weak supervision, synthetic data.`,
    pitfalls: `- Augmentations that change the label, or augmenting the evaluation data.
- Resampling or SMOTE before the train/test split (leakage), and reporting accuracy on imbalanced data.
- Trusting pseudo-labels from a weak model — errors reinforce themselves.
- Active-learning datasets that become biased towards the boundary and unrepresentative for evaluation — keep a random test set.
- Synthetic data without filtering, or without real data, drifting away from reality.`,
    takeaways: [
      'Hold the model fixed and improve the data: error analysis, label fixes and targeted examples.',
      'Augmentation teaches invariances and regularizes; mixup blends examples and labels.',
      'Imbalance: evaluate with precision/recall, then use class weights, resampling (training folds only) or threshold moving.',
      'Out-of-fold predictions that confidently disagree with a label point to label errors.',
      'Few labels: transfer learning, pseudo-labelling, active learning, weak supervision and filtered synthetic data.'
    ],
    quiz: [
      { type: 'mcq', q: 'A fraud model has 99% accuracy on data with 1% fraud. What should you check first?', options: ['Nothing, 99% is excellent', 'Recall and precision on the fraud class — it may never predict fraud', 'The learning rate', 'The number of features'], answer: 1, explain: 'Always answering "not fraud" scores 99% accuracy.' },
      { type: 'fill', q: 'A dataset has 1,000 examples: 900 of class A and 100 of class B. With balanced class weights w_c = n / (k · n_c), what is the weight of class B?', answer: ['5', '5.0'], explain: '1000 / (2 × 100) = 5.' },
      { type: 'tf', q: 'Data augmentation should also be applied to the test set so that it matches the training data.', answer: false, explain: 'Evaluation must reflect real inputs; augment only the training data.' },
      { type: 'mcq', q: 'Which active-learning strategy picks the example whose top two class probabilities are closest?', options: ['Random sampling', 'Margin sampling', 'Pseudo-labelling', 'SMOTE'], answer: 1, explain: 'A small margin means the model can hardly decide.' },
      { type: 'mcq', q: 'How do you find likely mislabelled training examples?', options: ['Look at the examples with the highest training accuracy', 'Find examples where out-of-fold predictions confidently disagree with the label', 'Delete the rare classes', 'Shuffle the labels'], answer: 1, explain: 'A model that never saw the example disagrees with its label confidently.' },
      {"type":"mcq","q":"Why must label-error detection use out-of-fold predictions?","options":["They are faster","A model scoring its own training examples has partly memorised their labels, including the wrong ones","Cross-validation removes noise automatically","Out-of-fold predictions are always calibrated"],"answer":1,"explain":"A model that saw an example during training tends to agree with its label, right or wrong, so only predictions from models that never saw it reveal disagreement."},
      {"type":"mcq","q":"As training labels get noisier, which model typically degrades fastest?","options":["Logistic regression","25-nearest neighbours","1-nearest neighbour","A model that always predicts the majority class"],"answer":2,"explain":"1-nearest neighbour copies the label of a single training point, so every flipped label turns directly into wrong predictions around it."}
    ],
    problems: [
      {
        id: 'imbalance-tools', title: 'Class Weights, Oversampling and Thresholds', difficulty: 'medium',
        prompt: `Implement:

- \`balanced_class_weights(y)\` → a dict {class: n / (k · n_c)} for each class present in y (use plain Python ints as keys).
- \`oversample_indices(y)\` → a NumPy array of row indices that balances the classes by repetition: for each class in ascending order, take its indices in ascending order and repeat them cyclically until there are as many as in the largest class (\`np.resize\` does this); concatenate the classes in order.
- \`threshold_for_precision(probs, y, min_precision)\` → the smallest threshold t among the sorted unique values of probs such that predicting "positive" for probs ≥ t gives precision ≥ min_precision; return None if no threshold reaches it.

\`\`\`text
oversample_indices([0, 1, 0, 0, 1, 2])  ->  [0, 2, 3, 1, 4, 1, 5, 5, 5]
\`\`\``,
        starter: `
import numpy as np

def balanced_class_weights(y):
    # TODO
    pass

def oversample_indices(y):
    # TODO
    pass

def threshold_for_precision(probs, y, min_precision):
    # TODO
    pass
`,
        solution: `
import numpy as np

def balanced_class_weights(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): len(y) / (len(classes) * n) for c, n in zip(classes, counts)}

def oversample_indices(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    parts = [np.resize(np.nonzero(y == c)[0], target) for c in classes]
    return np.concatenate(parts)

def threshold_for_precision(probs, y, min_precision):
    probs, y = np.asarray(probs, dtype=float), np.asarray(y)
    for t in np.unique(probs):
        pred = probs >= t
        precision = np.sum(pred & (y == 1)) / pred.sum()
        if precision >= min_precision:
            return float(t)
    return None
`,
        hints: ['`np.unique(y, return_counts=True)` gives the classes and their counts (it also works where `np.bincount` cannot).', '`np.resize(a, n)` repeats a cyclically to length n.', 'Recall only falls as the threshold rises, so the smallest qualifying threshold gives the best recall.'],
        explain: `Weights and oversampling both make each class contribute equally to the loss; weights do it without duplicating rows. Choosing the smallest threshold that meets a precision target maximizes recall under that constraint — a common way to set operating points in fraud and safety systems.`,
        tests: [
          { call: 'balanced_class_weights([0, 0, 0, 1])', expect: '{0: 4 / 6, 1: 2.0}' },
          { call: 'oversample_indices([0, 1, 0, 0, 1, 2])', expect: 'np.array([0, 2, 3, 1, 4, 1, 5, 5, 5])' },
          { call: 'threshold_for_precision([0.9, 0.8, 0.3, 0.2], [1, 0, 1, 0], 0.6)', expect: '0.3' },
          { name: 'unreachable precision', call: 'threshold_for_precision([0.9, 0.1], [0, 1], 0.9)', expect: 'None', hidden: true },
          { name: 'random labels vs reference', setup: 'rng = np.random.default_rng(0)\ny = rng.choice([0, 1, 2], size=40, p=[0.7, 0.2, 0.1])\nprobs = rng.random(40); yb = (rng.random(40) < probs).astype(int)', call: '[balanced_class_weights(y), oversample_indices(y), threshold_for_precision(probs, yb, 0.8)]', hidden: true }
        ]
      },
      {
        id: 'mixup-queries', title: 'Mixup and Uncertainty Sampling', difficulty: 'easy',
        prompt: `Implement:

- \`mixup(x1, y1, x2, y2, lam)\` → \`(lam·x1 + (1 − lam)·x2, lam·y1 + (1 − lam)·y2)\` for NumPy arrays.
- \`entropy(probs)\` → the entropy −Σ p·ln p of each row of a (n, k) probability array, treating 0·ln 0 as 0.
- \`query(probs, k, strategy)\` → the indices (a NumPy array) of the k most uncertain rows, most uncertain first, with ties broken by the lower index (use a stable sort). Strategies: \`"least_confident"\` (uncertainty 1 − max p), \`"entropy"\` (highest entropy) and \`"margin"\` (smallest difference between the two largest probabilities).`,
        starter: `
import numpy as np

def mixup(x1, y1, x2, y2, lam):
    # TODO
    pass

def entropy(probs):
    # TODO
    pass

def query(probs, k, strategy):
    # TODO
    pass
`,
        solution: `
import numpy as np

def mixup(x1, y1, x2, y2, lam):
    return lam * np.asarray(x1) + (1 - lam) * np.asarray(x2), lam * np.asarray(y1) + (1 - lam) * np.asarray(y2)

def entropy(probs):
    P = np.asarray(probs, dtype=float)
    logs = np.log(np.where(P > 0, P, 1.0))
    return -np.sum(P * logs, axis=1)

def query(probs, k, strategy):
    P = np.asarray(probs, dtype=float)
    if strategy == "least_confident":
        return np.argsort(-(1 - P.max(axis=1)), kind="stable")[:k]
    if strategy == "entropy":
        return np.argsort(-entropy(P), kind="stable")[:k]
    top2 = np.sort(P, axis=1)[:, -2:]
    return np.argsort(top2[:, 1] - top2[:, 0], kind="stable")[:k]
`,
        hints: ['Replace zeros by 1 inside the log so that 0·log 1 = 0.', 'Sort the negated uncertainty to get the most uncertain first; margins are sorted ascending.'],
        explain: `All three strategies agree for two classes; with more classes, margin sampling focuses on the hardest decision between the top two, while entropy considers the whole distribution.`,
        tests: [
          { call: 'mixup(np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.7)', expect: '(np.array([0.7, 0.3]), np.array([0.7, 0.3]))' },
          { call: 'entropy(np.array([[0.5, 0.5], [1.0, 0.0]]))', expect: 'np.array([np.log(2), 0.0])' },
          { call: 'query(np.array([[0.9, 0.1], [0.55, 0.45], [0.7, 0.3]]), 1, "margin")', expect: 'np.array([1])' },
          { name: 'random probabilities vs reference', setup: 'rng = np.random.default_rng(0)\nP = rng.dirichlet(np.ones(4), size=12)', call: '[query(P, 3, "least_confident"), query(P, 3, "entropy"), query(P, 3, "margin")]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Zhang et al. — mixup: Beyond Empirical Risk Minimization', url: 'https://arxiv.org/abs/1710.09412' },
      { title: 'Chawla et al. — SMOTE: Synthetic Minority Over-sampling Technique', url: 'https://arxiv.org/abs/1106.1813' },
      { title: 'Northcutt, Athalye & Mueller — Pervasive Label Errors in Test Sets', url: 'https://arxiv.org/abs/2103.14749' },
      { title: 'Settles — Active Learning Literature Survey', url: 'https://burrsettles.com/pub/settles.activelearning.pdf' }
    ]
  },

  // =====================================================================================
  {
    id: 'sql-data-engineering', branch: 'code', title: 'SQL & Data Engineering for ML', short: 'SQL & Data Engineering', level: 2, minutes: 45, prereqs: ['pandas-eda'],
    tags: ['sql', 'select', 'where', 'group by', 'join', 'left join', 'window functions', 'cte', 'sqlite', 'data pipeline', 'etl', 'elt', 'parquet', 'data warehouse', 'data quality', 'point-in-time correctness', 'feature store'],
    summary: 'Getting training data out of databases and into models: the relational model and SQL — SELECT, WHERE, GROUP BY, joins and their traps, window functions and CTEs — run from Python with sqlite3, then the engineering around it: ETL and ELT pipelines, columnar files, data validation, and point-in-time correct features that don\'t leak the future.',
    why: `Most of the data that ML models learn from lives in databases and warehouses, not in CSV files. **SQL** is how you filter, join and aggregate it — often far faster than pulling everything into pandas — and it is asked in nearly every data and ML job interview. Just as important is the engineering around it: pipelines that are reproducible, validated and free of **leakage**, especially the subtle kind where a feature accidentally uses information from after the moment you want to predict.`,
    objectives: [
      'Write SELECT queries with filtering, grouping, aggregation and ordering, and know their evaluation order',
      'Use inner, left and anti-joins correctly and detect join fan-out',
      'Compute per-entity features with window functions and structure queries with CTEs',
      'Build point-in-time correct features and describe a reliable data pipeline'
    ],
    sections: [
      { type: 'text', title: 'Tables, keys and SQL', md: `A relational database stores data in **tables**: each row is a record, each column has a type. A **primary key** identifies each row (users.id); a **foreign key** points to a row in another table (orders.user_id → users.id). Keeping each fact in exactly one place avoids contradictions; **joins** bring the facts back together.

SQL is **declarative**: you describe the result, and the database's query planner decides how to compute it — using indexes and running in parallel next to the data. That is why "push the computation to the database" usually beats downloading millions of rows into Python.` },
      { type: 'text', title: 'The shape of a query', md: `\`\`\`sql
SELECT   u.country, COUNT(*) AS n_orders, SUM(o.amount) AS revenue   -- columns and aggregates
FROM     orders AS o
JOIN     users  AS u ON u.id = o.user_id                              -- combine tables
WHERE    o.day >= '2026-01-04'                                        -- filter rows
GROUP BY u.country                                                    -- one output row per group
HAVING   COUNT(*) >= 1                                                -- filter groups
ORDER BY revenue DESC                                                 -- sort
LIMIT    10;
\`\`\`

The clauses are written in this order but **evaluated** as FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT. That explains classic errors: WHERE cannot use an aggregate (use HAVING), and some databases do not let WHERE refer to a column alias defined in SELECT.

**NULL** means "unknown": any comparison with NULL is unknown, so \`WHERE x = NULL\` matches nothing (use \`IS NULL\`); \`COUNT(*)\` counts rows while \`COUNT(col)\` skips NULLs; aggregates ignore NULLs.` },
      { type: 'code', title: 'SQL from Python with sqlite3', code: `
import sqlite3
con = sqlite3.connect(":memory:")                       # a throwaway database that lives in memory
con.executescript("""
CREATE TABLE users  (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, day TEXT);
INSERT INTO users VALUES (1, 'Ana', 'PT'), (2, 'Ben', 'UK'), (3, 'Chloe', 'FR'), (4, 'Dev', 'UK');
INSERT INTO orders VALUES
  (1, 1, 30.0, '2026-01-03'), (2, 2, 12.5, '2026-01-04'), (3, 1, 55.0, '2026-01-10'),
  (4, 3, 8.0, '2026-01-11'), (5, 2, 40.0, '2026-01-15'), (6, 1, 15.0, '2026-01-20');
""")

query = """
SELECT u.country, COUNT(*) AS n_orders, SUM(o.amount) AS revenue, ROUND(AVG(o.amount), 2) AS avg_order
FROM orders AS o
JOIN users AS u ON u.id = o.user_id
WHERE o.day >= '2026-01-04'
GROUP BY u.country
ORDER BY revenue DESC
"""
for row in con.execute(query):
    print(row)

print("users with no orders:", con.execute("""
    SELECT u.name FROM users AS u
    LEFT JOIN orders AS o ON o.user_id = u.id
    WHERE o.id IS NULL
""").fetchall())
`, after: `Each result row is a Python tuple. The LEFT JOIN keeps every user and fills the order columns with NULL when there is no match, so "WHERE o.id IS NULL" keeps exactly the users without orders — the standard **anti-join**.` },
      { type: 'text', title: 'Joins and their traps', md: `- **INNER JOIN** keeps only rows with a match on both sides; **LEFT JOIN** keeps every row of the left table (NULLs where nothing matches). pandas' \`merge(how="inner" / "left")\` does the same.
- **Fan-out**: if the key is not unique on the right side, each left row is repeated once per match — sums and counts silently inflate. Joining orders to a table with two addresses per user doubles revenue. Always check row counts before and after a join, and aggregate the "many" side *before* joining when you need one row per entity.
- **Filtering a LEFT JOIN in WHERE** (on a right-table column) quietly turns it into an inner join; put such conditions in the ON clause instead.
- Pass values with **placeholders** — \`con.execute("SELECT * FROM users WHERE country = ?", (country,))\` — never by formatting strings into SQL, which invites SQL injection.` },
      { type: 'check', question: { type: 'mcq', q: 'You join 1,000 orders to a table that has two rows for some users. What can happen to SUM(amount)?', options: ['Nothing, joins never change sums', 'It can be inflated, because matching orders are duplicated (fan-out)', 'It always halves', 'The query fails'], answer: 1, explain: 'Each order is repeated once per matching row on the other side.' } },
      { type: 'text', title: 'Window functions and CTEs', md: `**Window functions** compute values across related rows *without* collapsing them the way GROUP BY does — perfect for per-entity history features:

\`\`\`sql
WITH user_orders AS (                                   -- a CTE: a named sub-query
  SELECT user_id, day, amount,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY day) AS order_number,
         SUM(amount)  OVER (PARTITION BY user_id ORDER BY day) AS running_total,
         LAG(day)     OVER (PARTITION BY user_id ORDER BY day) AS previous_day
  FROM orders
)
SELECT * FROM user_orders WHERE order_number <= 3;      -- each user's first three orders
\`\`\`

PARTITION BY defines the groups, ORDER BY the order inside each group, and the function (ROW_NUMBER, RANK, SUM, AVG, LAG, LEAD) is computed for every row. **CTEs** (WITH … AS) name intermediate results so that long queries read top to bottom like a pipeline.` },
      { type: 'code', title: 'Per-user history features with window functions', code: `
import sqlite3
con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, day TEXT);
INSERT INTO orders VALUES
  (1, 1, 30.0, '2026-01-03'), (2, 2, 12.5, '2026-01-04'), (3, 1, 55.0, '2026-01-10'),
  (4, 3, 8.0, '2026-01-11'), (5, 2, 40.0, '2026-01-15'), (6, 1, 15.0, '2026-01-20');
""")
rows = con.execute("""
SELECT user_id, day, amount,
       ROW_NUMBER() OVER w                              AS order_number,
       SUM(amount)  OVER w                              AS running_total,
       julianday(day) - julianday(LAG(day) OVER w)      AS days_since_previous
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY day)
ORDER BY user_id, day
""").fetchall()
print("user  day          amount  order#  running total  days since previous")
for r in rows:
    print(f"{r[0]:4d}  {r[1]}  {r[2]:6.1f}  {r[3]:6d}  {r[4]:13.1f}  {r[5]}")
`, after: `Every order keeps its own row and gains features computed from that user's earlier orders. The first order of each user has no previous one, so its gap is NULL (None in Python).` },
      { type: 'text', title: 'Point-in-time correctness', md: `A model that predicts at time t may only use information available **before t**. A feature such as "total spend" computed over *all* of a user's orders includes purchases made after the prediction time — sometimes the very event being predicted. Offline the model looks brilliant; in production, where the future is not available, it fails.

The fix is an **as-of join**: when building the training row for (user, t), only aggregate events with timestamp < t. Feature stores (Feast, Tecton and the feature stores of the big cloud platforms) automate these point-in-time joins and serve the same feature definitions online, avoiding **training–serving skew**.` },
      { type: 'code', title: 'A leaky feature and its point-in-time fix', code: `
import sqlite3
con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE orders (user_id INTEGER, amount REAL, day TEXT);
INSERT INTO orders VALUES (1, 30.0, '2026-01-03'), (1, 55.0, '2026-01-10'), (1, 15.0, '2026-01-20'),
                          (2, 12.5, '2026-01-04'), (2, 40.0, '2026-01-15');
CREATE TABLE labels (user_id INTEGER, as_of TEXT, will_buy_again INTEGER);   -- what we predict, and when
INSERT INTO labels VALUES (1, '2026-01-12', 1), (2, '2026-01-16', 0);
""")
leaky = con.execute("""
SELECT l.user_id, l.as_of, COUNT(o.amount) AS n_orders, SUM(o.amount) AS spend, l.will_buy_again
FROM labels AS l LEFT JOIN orders AS o ON o.user_id = l.user_id
GROUP BY l.user_id ORDER BY l.user_id
""").fetchall()
point_in_time = con.execute("""
SELECT l.user_id, l.as_of, COUNT(o.amount) AS n_orders, SUM(o.amount) AS spend, l.will_buy_again
FROM labels AS l LEFT JOIN orders AS o ON o.user_id = l.user_id AND o.day < l.as_of
GROUP BY l.user_id ORDER BY l.user_id
""").fetchall()
print("leaky (all orders, even future ones):  ", leaky)
print("point-in-time (orders before as_of):   ", point_in_time)
`, after: `For user 1, the leaky version counts the order of 20 January — the very purchase the label says will happen. Moving the time condition into the ON clause keeps every label row while counting only the past.` },
      { type: 'text', title: "Splitting data reproducibly", md: `How you split data into training and test sets is a data-engineering decision, and the wrong one quietly inflates every metric.

- **Split by entity, not by row.** If one user has 50 rows, a random row split puts some of them in training and some in test, and the model is rewarded for recognising users rather than for generalising. Assign whole users (patients, devices, documents) to one side.
- **Split by time when you predict the future.** Train on January to June and test on July; never let the model learn from data after the moment it would have been used (see point-in-time correctness above).
- **Make the split deterministic.** \`ORDER BY RANDOM()\` gives a different split on every run, so results are not reproducible and rows can move between sides as data is added. Instead, hash a stable id and let the hash choose the side: the same user lands in the same bucket today and next month, in SQL or in Python.

$$ bucket(id) = hash(id) mod 100;     training if bucket < 80, otherwise test $$
:::where
hash(id) — a deterministic hash of the entity's id (for example MD5), the same on every run and every machine
mod 100 — the remainder after dividing by 100: a bucket from 0 to 99
bucket < 80 — about 80% of entities go to training; changing the threshold changes the share without reshuffling everyone else
:::

Be careful with built-in hash functions: Python's \`hash()\` of a string changes from run to run unless \`PYTHONHASHSEED\` is fixed, so use \`hashlib\`.` },
      { type: 'code', title: "A deterministic split by user", code: `import hashlib
import sqlite3

def bucket(user_id, buckets=100):
    digest = hashlib.md5(str(user_id).encode()).hexdigest()
    return int(digest, 16) % buckets             # the same id always gives the same bucket

con = sqlite3.connect(":memory:")
con.execute("CREATE TABLE events (user_id INTEGER, day INTEGER, clicks INTEGER)")
rows = [(u, d, (u * 7 + d * 3) % 5) for u in range(1, 201) for d in range(1, 6)]
con.executemany("INSERT INTO events VALUES (?, ?, ?)", rows)
con.create_function("bucket", 1, bucket)         # make the Python function callable from SQL

train = {u for (u,) in con.execute("SELECT DISTINCT user_id FROM events WHERE bucket(user_id) < 80")}
test = {u for (u,) in con.execute("SELECT DISTINCT user_id FROM events WHERE bucket(user_id) >= 80")}
print("users in train:", len(train), "  users in test:", len(test))
print("users in both splits:", len(train & test))
print("user 42 is always in bucket", bucket(42), "-> stable:", bucket(42) == bucket(42))`, after: `About 80% of the 200 users land in training (165 here), and each user's 5 rows stay together, so no user appears on both sides. Because the bucket depends only on the id, rerunning the query next month puts every existing user on the same side, and new users are assigned without disturbing anyone else.`, predict: {"type":"mcq","q":"Rows are assigned to train or test by hashing the user id. How many users can appear in both splits?","options":["About 20%","About half","None","It changes on every run"],"answer":2,"explain":"The bucket depends only on the user id, so all of a user's rows land on the same side, every time.","expect":["users in both splits: 0"]} },
      { type: 'text', title: 'Pipelines and data quality', md: `- **ETL vs ELT**: extract, transform, load (transform before the warehouse) or load raw data first and transform inside the warehouse with SQL (dbt is a popular tool) — the modern default.
- **Batch vs streaming**: nightly jobs for most training data; streams (Kafka, Flink) for real-time features.
- **Storage**: columnar formats such as **Parquet** are typed, compressed and let queries read only the needed columns — far better than CSV for analytics. Warehouses (BigQuery, Snowflake, Redshift) and lakehouse tables (Delta Lake, Iceberg) hold the data; engines such as DuckDB and Polars bring the same ideas to a laptop.
- **Orchestration**: Airflow, Dagster or Prefect schedule pipelines as graphs of tasks with retries.
- **Validation**: check schemas, null rates, ranges, uniqueness and row counts at every stage, and stop the pipeline when they fail (tools such as Great Expectations or dbt tests).
- **Reproducibility**: make jobs idempotent (re-running gives the same result), version data snapshots alongside code, and record which data trained which model.` }
    ],
    when: `- Extracting and joining training data that lives in relational databases or warehouses.
- Building per-entity history features (counts, sums, recency) with window functions.
- Exploratory analysis on data too big for pandas — run SQL in the warehouse or in DuckDB.
- Any time series of events feeds a model: enforce point-in-time correctness.`,
    pitfalls: `- Join fan-out inflating sums and counts; check row counts around every join.
- WHERE conditions on the right table that turn a LEFT JOIN into an inner join.
- NULL comparisons with = instead of IS NULL; COUNT(col) versus COUNT(*).
- Features computed with future data, and different feature code for training and serving (skew).
- Building SQL with string formatting (injection, quoting bugs) instead of placeholders.`,
    takeaways: [
      'SQL is declarative: SELECT … FROM … WHERE … GROUP BY … HAVING … ORDER BY, evaluated FROM-first.',
      'Inner, left and anti-joins combine tables; watch for fan-out and NULL semantics.',
      'Window functions (ROW_NUMBER, SUM, LAG … OVER (PARTITION BY … ORDER BY …)) build history features without collapsing rows.',
      'Point-in-time correct features use only events before the prediction time.',
      'Reliable pipelines: columnar storage, orchestration, validation at every step, versioned and reproducible runs.'
    ],
    quiz: [
      { type: 'mcq', q: 'Which clause filters groups after aggregation?', options: ['WHERE', 'HAVING', 'ORDER BY', 'LIMIT'], answer: 1, explain: 'WHERE filters rows before grouping; HAVING filters groups.' },
      { type: 'mcq', q: 'How do you find users with no orders?', options: ['INNER JOIN orders and keep everything', 'LEFT JOIN orders and keep rows where the order id IS NULL', 'GROUP BY user and ORDER BY count', 'SELECT DISTINCT users'], answer: 1, explain: 'That is the anti-join pattern.' },
      { type: 'tf', q: 'A window function such as SUM(amount) OVER (PARTITION BY user_id ORDER BY day) returns one row per user.', answer: false, explain: 'Window functions keep every row; GROUP BY would collapse them.' },
      { type: 'mcq', q: 'Which feature is point-in-time correct for a prediction made on 1 March?', options: ['Total spend over all of the user\'s orders', 'Total spend over orders placed before 1 March', 'Spend in the 30 days after 1 March', 'Whether the user churned in March'], answer: 1, explain: 'Only information available before the prediction time may be used.' },
      { type: 'fill', q: 'What does SQL\'s COUNT(*) do differently from COUNT(column)? COUNT(column) skips rows where the column is … (one word)', answer: ['null', 'NULL'], explain: 'COUNT(*) counts every row.' },
      {"type":"mcq","q":"Why is ORDER BY RANDOM() a poor way to create a train/test split?","options":["It is too slow","It gives a different split on every run, so results are not reproducible and rows move between sides","It only works in SQLite","It always puts the newest rows in the test set"],"answer":1,"explain":"A split should be a stable property of each entity. Hashing a stable id gives the same assignment on every run."},
      {"type":"tf","q":"With several rows per user, a random split by row can let a model score well simply by recognising users it saw in training.","answer":true,"explain":"The same user's behaviour appears on both sides, so the test set no longer measures generalisation to new users. Split by user instead."}
    ],
    problems: [
      {
        id: 'sql-aggregates', title: 'Revenue by Country with SQL', difficulty: 'medium',
        prelude: `
import sqlite3

def make_db():
    con = sqlite3.connect(":memory:")
    con.executescript("""
CREATE TABLE users  (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, day TEXT);
INSERT INTO users VALUES (1, 'Ana', 'PT'), (2, 'Ben', 'UK'), (3, 'Chloe', 'FR'), (4, 'Dev', 'UK'), (5, 'Eva', 'PT');
INSERT INTO orders VALUES
  (1, 1, 30.0, '2026-01-03'), (2, 2, 12.5, '2026-01-04'), (3, 1, 55.0, '2026-01-10'),
  (4, 3, 8.0, '2026-01-11'), (5, 2, 40.0, '2026-01-15'), (6, 1, 15.0, '2026-01-20'),
  (7, 5, 22.0, '2026-01-21');
""")
    return con
`,
        prompt: `A helper \`make_db()\` (already defined) returns a sqlite3 connection with two tables: \`users(id, name, country)\` and \`orders(id, user_id, amount, day)\`. Write SQL inside these functions:

- \`revenue_by_country(con)\` → a list of tuples \`(country, n_orders, revenue)\` for countries that have orders, highest revenue first (ties by country name).
- \`users_without_orders(con)\` → the names of users with no orders, sorted alphabetically.
- \`top_customer(con, country)\` → the name of the user in that country with the highest total spend, or None if nobody there has ordered. Pass the country as a **placeholder** parameter (\`?\`), not by formatting it into the SQL string.`,
        starter: `
def revenue_by_country(con):
    # TODO: return con.execute("SELECT ...").fetchall()
    pass

def users_without_orders(con):
    # TODO
    pass

def top_customer(con, country):
    # TODO
    pass
`,
        solution: `
def revenue_by_country(con):
    return con.execute("""
        SELECT u.country, COUNT(*) AS n_orders, SUM(o.amount) AS revenue
        FROM orders AS o JOIN users AS u ON u.id = o.user_id
        GROUP BY u.country
        ORDER BY revenue DESC, u.country
    """).fetchall()

def users_without_orders(con):
    rows = con.execute("""
        SELECT u.name FROM users AS u
        LEFT JOIN orders AS o ON o.user_id = u.id
        WHERE o.id IS NULL
        ORDER BY u.name
    """).fetchall()
    return [r[0] for r in rows]

def top_customer(con, country):
    row = con.execute("""
        SELECT u.name FROM orders AS o JOIN users AS u ON u.id = o.user_id
        WHERE u.country = ?
        GROUP BY u.id
        ORDER BY SUM(o.amount) DESC
        LIMIT 1
    """, (country,)).fetchone()
    return row[0] if row else None
`,
        hints: ['`con.execute(sql).fetchall()` returns a list of tuples.', 'For "no orders" use a LEFT JOIN and keep rows where the order id IS NULL.', 'Placeholders: `con.execute("... WHERE u.country = ?", (country,))`.'],
        explain: `Letting the database aggregate means only the small result travels to Python. Placeholders let the driver handle quoting, which is what prevents SQL injection when a value comes from a user.`,
        tests: [
          { setup: 'con = make_db()', call: 'revenue_by_country(con)', expect: '[("PT", 4, 122.0), ("UK", 2, 52.5), ("FR", 1, 8.0)]' },
          { setup: 'con = make_db()', call: 'users_without_orders(con)', expect: '["Dev"]' },
          { setup: 'con = make_db()', call: '[top_customer(con, "UK"), top_customer(con, "FR")]', expect: '["Ben", "Chloe"]' },
          { name: 'a country without orders', setup: 'con = make_db()', call: 'top_customer(con, "DE")', expect: 'None', hidden: true },
          { name: 'after new orders arrive', setup: 'con = make_db()\ncon.execute("INSERT INTO orders VALUES (?, ?, ?, ?)", (8, 4, 100.0, "2026-01-22"))', call: '[revenue_by_country(con), users_without_orders(con), top_customer(con, "UK")]', hidden: true }
        ]
      },
      {
        id: 'sql-window-features', title: 'History Features with Window Functions', difficulty: 'medium',
        prelude: `
import sqlite3

def make_db():
    con = sqlite3.connect(":memory:")
    con.executescript("""
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, day TEXT);
INSERT INTO orders VALUES
  (1, 1, 30.0, '2026-01-03'), (2, 2, 12.5, '2026-01-04'), (3, 1, 55.0, '2026-01-10'),
  (4, 3, 8.0, '2026-01-11'), (5, 2, 40.0, '2026-01-15'), (6, 1, 15.0, '2026-01-20');
""")
    return con
`,
        prompt: `\`make_db()\` returns a connection with the table \`orders(id, user_id, amount, day)\`, days written as 'YYYY-MM-DD'. Implement:

- \`order_features(con)\` → a list of tuples \`(order_id, user_id, order_number, running_total, days_since_previous)\`, sorted by order id, where — per user, ordered by day — order_number counts from 1, running_total is the user's total spend up to and including this order, and days_since_previous is the number of days since the user's previous order (None for a user's first order). Window functions make this short; \`julianday(day)\` converts a date to a day number.
- \`spend_before(con, user_id, day)\` → the user's total spend on orders strictly **before** the given day (0.0 if there are none) — a point-in-time feature. Use placeholders.`,
        starter: `
def order_features(con):
    # TODO
    pass

def spend_before(con, user_id, day):
    # TODO
    pass
`,
        solution: `
def order_features(con):
    return con.execute("""
        SELECT id, user_id,
               ROW_NUMBER() OVER w,
               SUM(amount) OVER w,
               julianday(day) - julianday(LAG(day) OVER w)
        FROM orders
        WINDOW w AS (PARTITION BY user_id ORDER BY day)
        ORDER BY id
    """).fetchall()

def spend_before(con, user_id, day):
    row = con.execute(
        "SELECT COALESCE(SUM(amount), 0.0) FROM orders WHERE user_id = ? AND day < ?",
        (user_id, day)).fetchone()
    return float(row[0])
`,
        hints: ['Define the window once: `WINDOW w AS (PARTITION BY user_id ORDER BY day)` and use `OVER w`.', '`LAG(day) OVER w` is the previous order\'s day; it is NULL for the first one.', '`COALESCE(SUM(amount), 0.0)` turns "no rows" into 0.'],
        explain: `Window functions compute history features for every row in one pass, which is how real feature pipelines do it at scale. Comparing ISO-formatted date strings works because they sort in the same order as the dates.`,
        tests: [
          { setup: 'con = make_db()', call: 'order_features(con)[:3]', expect: '[(1, 1, 1, 30.0, None), (2, 2, 1, 12.5, None), (3, 1, 2, 85.0, 7.0)]' },
          { setup: 'con = make_db()', call: 'spend_before(con, 1, "2026-01-12")', expect: '85.0' },
          { setup: 'con = make_db()', call: 'spend_before(con, 3, "2026-01-01")', expect: '0.0' },
          { name: 'all orders vs reference', setup: 'con = make_db()', call: 'order_features(con)', hidden: true },
          { name: 'more data vs reference', setup: 'con = make_db()\ncon.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", [(7, 3, 20.0, "2026-01-12"), (8, 1, 5.0, "2026-01-25"), (9, 3, 1.0, "2026-02-02")])', call: '[order_features(con), spend_before(con, 3, "2026-02-02"), spend_before(con, 1, "2026-12-31")]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'SQLite — SQL language reference', url: 'https://www.sqlite.org/lang.html' },
      { title: 'SQLite — Window functions', url: 'https://www.sqlite.org/windowfunctions.html' },
      { title: 'Python docs — sqlite3', url: 'https://docs.python.org/3/library/sqlite3.html' },
      { title: 'Kleppmann — Designing Data-Intensive Applications', url: 'https://dataintensive.net/' }
    ]
  },

  // =====================================================================================
  {
    id: 'learning-to-rank', branch: 'spec', title: 'Learning to Rank', short: 'Learning to Rank', level: 3, minutes: 40, prereqs: ['recommender-systems', 'ensembles'],
    tags: ['learning to rank', 'ranking', 'ndcg', 'dcg', 'mrr', 'precision at k', 'pointwise', 'pairwise', 'listwise', 'ranknet', 'lambdamart', 'bm25', 'two-stage retrieval', 'position bias', 'interleaving'],
    summary: 'Ordering results so the best come first: ranking problems in search, feeds and RAG; metrics that care about position (precision@k, MRR, DCG and NDCG); pointwise, pairwise (RankNet) and listwise (LambdaMART) training; BM25 as the classic text score; and production concerns — two-stage retrieval and reranking, click data and position bias, and interleaving experiments.',
    why: `Search engines, product listings, social feeds, job matching, recommendation carousels and the retrieval step of every RAG system share one problem: given a query (or a user), put the most relevant items **at the top**. Users look at the first few results, so mistakes near the top cost far more than mistakes further down. **Learning to rank** trains models directly for that goal and measures them with position-aware metrics.`,
    objectives: [
      'Compute precision@k, MRR, DCG and NDCG and explain why position matters',
      'Contrast pointwise, pairwise and listwise approaches; derive the RankNet loss',
      'Describe LambdaMART and BM25 and where they fit in a ranking stack',
      'Explain two-stage retrieval, position bias in click data and interleaving'
    ],
    sections: [
      { type: 'text', title: 'Ranking problems', md: `In learning to rank, data comes in **groups**: a query (or a user session) and a list of candidate items, each with features and a **graded relevance** label — for example 0 = irrelevant, 1 = somewhat, 2 = relevant, 3 = perfect — from human judges or from behaviour such as clicks and purchases. A model assigns each item a score; sorting by score gives the ranking. Only the order *within* a query matters, not the absolute scores, and comparing scores across different queries is meaningless.` },
      { type: 'text', title: 'Measuring a ranking', md: `- **Precision@k**: the fraction of the top k items that are relevant.
- **MRR** (mean reciprocal rank): the average over queries of 1 / (position of the first relevant item) — good when one right answer is enough.
- **DCG and NDCG**: graded relevance with a discount for lower positions:

$$ DCG@k = Σ_{i=1}^{k} ( 2^{rel_i} − 1 ) / log₂(i + 1)          NDCG@k = DCG@k / IDCG@k $$
:::where
i — the position in the ranked list (1 is the top)
rel_i — the graded relevance of the item at position i
2^{rel_i} − 1 — the gain, so highly relevant items count much more
log₂(i + 1) — the position discount, so lower positions count less
IDCG@k — the DCG of the ideal ordering (items sorted by relevance), so NDCG lies between 0 and 1
:::` },
      { type: 'chart', title: "How much each position counts", after: `Both metrics care most about the top of the list, but differently. MRR's weight drops by half from position 1 to 2 and is almost nothing by position 10: only the first relevant result matters. DCG's logarithmic discount is gentler (position 10 still counts 0.29), so it rewards getting several relevant items into the top positions.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10],"series":[{"name":"DCG discount 1 / log₂(i + 1)","y":[1,0.631,0.5,0.431,0.387,0.356,0.333,0.315,0.301,0.289]},{"name":"reciprocal rank 1 / i (MRR)","y":[1,0.5,0.333,0.25,0.2,0.167,0.143,0.125,0.111,0.1]}],"xLabel":"position i in the ranking","yLabel":"weight","yMin":0,"yMax":1.05} },
      { type: 'code', title: 'NDCG and MRR by hand', code: `
import numpy as np

def dcg(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    return float(np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))

def ndcg(rels, k):
    ideal = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0

rankings = {
    "perfect":          [3, 2, 2, 1, 0, 0],
    "swap top two":     [2, 3, 2, 1, 0, 0],
    "swap 4th and 5th": [3, 2, 2, 0, 1, 0],
    "reversed":         [0, 0, 1, 2, 2, 3],
}
for name, rels in rankings.items():
    first_relevant = next(i for i, r in enumerate(rels, start=1) if r > 0)
    print(f"{name:18s} NDCG@5 = {ndcg(rels, 5):.3f}   reciprocal rank = {1 / first_relevant:.2f}")
`, after: `Swapping the top two items costs over 13% of NDCG, while swapping the 4th and 5th costs less than 1% — position matters most at the top. The reciprocal rank only notices the first relevant item, so it cannot tell the first three rankings apart.` },
      { type: 'text', title: 'Pointwise, pairwise and listwise learning', md: `- **Pointwise**: predict each item's relevance with ordinary regression or classification, then sort. Simple, but it spends effort on getting absolute values right, which ranking doesn't need.
- **Pairwise**: learn which of two items should come first. **RankNet** turns score differences into probabilities and uses cross-entropy:

$$ P(i ≻ j) = σ( s_i − s_j )          L = −log σ( s_i − s_j )   for every pair with rel_i > rel_j $$
:::where
s_i, s_j — the model's scores for items i and j of the same query
σ — the sigmoid function
i ≻ j — "i should be ranked above j"
L — the pairwise loss, small when the more relevant item already scores higher by a margin
:::

- **Listwise**: optimize a whole-list objective. **LambdaRank** keeps RankNet's pair gradients but multiplies each by |ΔNDCG|, how much swapping the two items would change NDCG, so effort goes where the metric cares. **LambdaMART** — LambdaRank with gradient-boosted trees — has long been the strongest method on tabular ranking features (LightGBM's \`lambdarank\` objective, XGBoost's \`rank:ndcg\`).` },
      { type: 'code', title: 'Training a pairwise ranker', code: `
import numpy as np
rng = np.random.default_rng(0)
n_queries, n_docs, d = 60, 20, 5
true_w = np.array([2.0, -1.0, 0.5, 0.0, 1.0])
X = rng.normal(size=(n_queries, n_docs, d))                    # features of each (query, document) pair
rel = np.digitize(X @ true_w + rng.normal(0, 1.0, (n_queries, n_docs)), [1.0, 2.5, 4.0])   # graded relevance 0-3

def ndcg_at(scores, rels, k=5):
    disc = np.log2(np.arange(2, k + 2))
    gains = (2.0 ** rels[np.argsort(-scores)][:k] - 1) / disc
    ideal = (2.0 ** np.sort(rels)[::-1][:k] - 1) / disc
    return gains.sum() / ideal.sum() if ideal.sum() > 0 else 0.0

train, test = range(40), range(40, 60)
w = np.zeros(d)
for epoch in range(50):
    grad = np.zeros(d)
    for q in train:
        s = X[q] @ w
        i, j = np.nonzero(rel[q][:, None] > rel[q][None, :])     # all pairs where i should rank above j
        p = 1 / (1 + np.exp(-(s[i] - s[j])))                     # the model's P(i above j)
        grad += (-(1 - p)[:, None] * (X[q][i] - X[q][j])).sum(axis=0)
    w -= 0.01 * grad / len(train)

mean_ndcg = lambda scorer: np.mean([ndcg_at(scorer(q), rel[q]) for q in test])
print("learned weights:", w.round(2), "  true weights:", true_w)
print(f"NDCG@5 on unseen queries: random order {mean_ndcg(lambda q: rng.normal(size=n_docs)):.3f}   "
      f"RankNet {mean_ndcg(lambda q: X[q] @ w):.3f}   true weights {mean_ndcg(lambda q: X[q] @ true_w):.3f}")
`, after: `Trained only on which document beats which, the ranker recovers the direction of the true weights (scaled up — only the ordering matters) and ranks unseen queries about as well as the true scoring function.` },
      { type: 'text', title: "Learning from clicks: position bias", md: `Human relevance labels are expensive, so most ranking systems learn from **clicks**. But a click needs two things: the user must **see** the item and find it relevant. Items near the top are seen far more often, so they collect more clicks whether or not they are better. Train on raw clicks and the model learns to repeat whatever the old ranking did: a feedback loop.

$$ P(click at position k) = P(seen at k) · P(click once seen) $$
$$ relevance estimate = (1/n) Σ clicks / P(seen at the position where the item was shown) $$
:::where
P(seen at k) — the propensity: the chance that a user looks at position k at all; it falls steeply with k
P(click once seen) — the item's relevance, which is what we want to learn
clicks / P(seen …) — each click is up-weighted by how unlikely it was to be seen: inverse propensity weighting (IPW)
n — the number of times the item was shown
:::

Propensities are measured by randomising: swap or shuffle results for a small share of traffic and see how the click rate of the *same* item changes with its position. **Interleaving**, which mixes the results of two rankers into one list and counts which ranker's items get clicked, compares rankers with far less traffic than an A/B test.` },
      { type: 'code', title: "Correcting click data for position bias", code: `import numpy as np
rng = np.random.default_rng(0)
true_relevance = np.array([0.30, 0.50, 0.20, 0.45, 0.10])   # P(click | seen) for items 0..4
seen_prob = np.array([1.0, 0.6, 0.4, 0.3, 0.2])             # P(seen) at positions 1..5
old_ranking = np.array([0, 2, 4, 1, 3])                     # the current, poor ranking

shown, clicks, weighted = np.zeros(5), np.zeros(5), np.zeros(5)
for _ in range(20_000):
    order = old_ranking if rng.random() < 0.9 else rng.permutation(5)   # 10% of traffic randomised
    for pos, item in enumerate(order):
        shown[item] += 1
        if rng.random() < seen_prob[pos] and rng.random() < true_relevance[item]:
            clicks[item] += 1
            weighted[item] += 1 / seen_prob[pos]                 # inverse propensity weight

raw, ipw = clicks / shown, weighted / shown
print("raw click rate:", raw.round(3), "-> ranking", np.argsort(-raw))
print("IPW estimate:  ", ipw.round(3), "-> ranking", np.argsort(-ipw))
print("true relevance:", true_relevance, "-> ranking", np.argsort(-true_relevance))
print("IPW recovers the true order:", np.array_equal(np.argsort(-ipw), np.argsort(-true_relevance)))`, after: `Raw click rates crown item 0, which nearly always sits at the top, and rate item 2 above item 3, although item 3 is more than twice as relevant. Dividing each click by the chance that it was seen recovers the true relevances and the true order. Here the propensities were known; in practice the randomised share of traffic is how you measure them.`, predict: {"type":"mcq","q":"Item 1 is the most relevant item but is usually shown 4th; item 0 is less relevant and usually shown 1st. Which has the higher raw click rate?","options":["Item 1, because it is more relevant","Item 0, because it is seen far more often","They are equal","It is impossible to say"],"answer":1,"explain":"At position 4 an item is seen only 30% of the time, so even a very relevant item collects few clicks there. Inverse propensity weighting corrects for this.","expect":["IPW recovers the true order: True"]} },
      { type: 'text', title: 'Ranking in production', md: `- **Two stages**: a cheap **retrieval** stage narrows millions of items to a few hundred candidates — lexical search with **BM25**, embedding search (see *Vector Search*) or both — and a heavier **reranker** (LambdaMART on many features, a cross-encoder, or an LLM) orders the top candidates.
- **BM25**, the classic lexical relevance score, still a strong baseline and a common reranker feature:

$$ BM25(q, d) = Σ_{t ∈ q} IDF(t) · f(t, d) · (k₁ + 1) / ( f(t, d) + k₁ · (1 − b + b · |d| / avgdl) ) $$
:::where
t ∈ q — each term of the query
f(t, d) — how often term t occurs in document d
IDF(t) — the inverse document frequency, so rare terms count more
|d| — the length of document d
avgdl — the average document length in the collection
k₁ — saturation (typically 1.2–2), so extra occurrences of a term help less and less
b — length normalization (typically 0.75), so long documents are not favoured just for containing more words
:::

- **Learning from clicks**: clicks are cheap labels but biased — people click higher positions more regardless of relevance (**position bias**). Correct it with inverse propensity weighting (see *Causal Inference*) or randomization experiments.
- **Online evaluation**: **interleaving** mixes the results of two rankers in one list and credits whichever ranker's items get clicked — far more sensitive than a standard A/B test.` },
      { type: 'check', question: { type: 'tf', q: 'In a two-stage ranking system, the first stage optimizes recall (find the candidates) and the second precision at the top.', answer: true, explain: 'A cheap retriever casts a wide net; an expensive reranker orders the best few.' } }
    ],
    when: `- Search, e-commerce listings, feeds, ads, job and dating matches, recommendation carousels.
- Reranking retrieved passages in RAG systems.
- Any task where users see an ordered list and the top positions matter most.
- Use pointwise models when absolute scores are needed (e.g. predicted click probability for an auction); pairwise or listwise for pure ordering.`,
    pitfalls: `- Splitting train and test by rows instead of by query (leaks the same query into both).
- Comparing scores across queries, or evaluating with accuracy instead of NDCG/MRR.
- Training on raw clicks without correcting for position bias.
- Tuning the reranker while ignoring the candidates the retriever never returns (recall is capped by stage one).
- Offline NDCG gains that don't translate online — confirm with interleaving or A/B tests.`,
    takeaways: [
      'Ranking data comes in query groups with graded relevance; only the order within a query matters.',
      'NDCG discounts gains by position and normalizes by the ideal ranking; MRR looks at the first relevant item.',
      'Pairwise RankNet: −log σ(s_i − s_j) for each correctly ordered pair; LambdaMART weights pairs by |ΔNDCG|.',
      'Production stacks: cheap retrieval (BM25, embeddings) followed by a strong reranker.',
      'Clicks are biased by position; correct with propensities and evaluate online with interleaving.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why does DCG divide each gain by log₂(i + 1)?', options: ['To normalize between 0 and 1', 'To discount items at lower positions, which users are less likely to see', 'To make the computation faster', 'To count only relevant items'], answer: 1, explain: 'The discount encodes that the top positions matter most.' },
      { type: 'fill', q: 'For three queries the first relevant result is at positions 1, 2 and 4. What is the MRR? (decimal, about 2–3 places)', answer: ['0.583', '0.58', '7/12'], explain: '(1 + 1/2 + 1/4) / 3 = 0.583.' },
      { type: 'mcq', q: 'RankNet is an example of which approach?', options: ['Pointwise', 'Pairwise', 'Unsupervised', 'Clustering'], answer: 1, explain: 'Its loss is defined on pairs of items.' },
      { type: 'tf', q: 'Users click higher-ranked results more often even when lower results are equally relevant.', answer: true, explain: 'That is position bias, which raw click labels inherit.' },
      { type: 'mcq', q: 'How should you split ranking data into training and test sets?', options: ['Randomly by rows', 'By query, so each query\'s items are all in one set', 'By relevance label', 'By document length'], answer: 1, explain: 'Otherwise information about a test query leaks into training.' },
      {"type":"mcq","q":"Why can't you train a ranker directly on raw click-through rates?","options":["Clicks are too rare","Items near the top are seen more often, so they collect more clicks regardless of their relevance","Click data is always private","Rankers need numeric labels"],"answer":1,"explain":"A click requires the item to be seen, and being seen depends on the old ranking. Correct for position (inverse propensity weighting) or randomise."},
      {"type":"fill","q":"With the DCG discount 1 / log₂(i + 1), what is the weight of position 3? (decimal)","answer":["0.5",".5","1/2"],"explain":"log₂(3 + 1) = log₂ 4 = 2, so the weight is 1/2."}
    ],
    problems: [
      {
        id: 'ranking-metrics', title: 'DCG, NDCG and MRR', difficulty: 'easy',
        prompt: `Relevance lists are given in ranked order (position 1 first). Implement:

- \`dcg_at_k(rels, k)\` → Σ over the first k positions of (2^rel − 1) / log₂(position + 1).
- \`ndcg_at_k(rels, k)\` → DCG@k divided by the DCG@k of the same relevances sorted in descending order; return 0.0 when that ideal DCG is 0.
- \`mean_reciprocal_rank(rankings)\` → the average over a list of relevance lists of 1 / (position of the first item with relevance > 0), counting 0 for a list without any relevant item.`,
        starter: `
import numpy as np

def dcg_at_k(rels, k):
    # TODO
    pass

def ndcg_at_k(rels, k):
    # TODO
    pass

def mean_reciprocal_rank(rankings):
    # TODO
    pass
`,
        solution: `
import numpy as np

def dcg_at_k(rels, k):
    r = np.asarray(rels, dtype=float)[:k]
    return float(np.sum((2 ** r - 1) / np.log2(np.arange(2, len(r) + 2))))

def ndcg_at_k(rels, k):
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg_at_k(rels, k) / ideal if ideal > 0 else 0.0

def mean_reciprocal_rank(rankings):
    total = 0.0
    for rels in rankings:
        for pos, r in enumerate(rels, start=1):
            if r > 0:
                total += 1 / pos
                break
    return total / len(rankings)
`,
        hints: ['Positions start at 1, so the discounts are log₂(2), log₂(3), …', 'Sorting the relevances in descending order gives the ideal ranking.'],
        explain: `NDCG makes rankings of different queries comparable by dividing by the best achievable score, and the exponential gain rewards putting the very best items first. MRR is simpler and suits "one right answer" tasks such as question answering.`,
        tests: [
          { call: 'dcg_at_k([3, 2, 0], 3)', expect: '7 + 3 / np.log2(3)' },
          { call: '[ndcg_at_k([3, 2, 2, 1], 4), ndcg_at_k([0, 0, 0], 3)]', expect: '[1.0, 0.0]' },
          { call: 'mean_reciprocal_rank([[0, 1, 0], [1, 0], [0, 0, 0]])', expect: '0.5' },
          { name: 'random relevances vs reference', setup: 'rng = np.random.default_rng(0)\nR = rng.integers(0, 4, size=(5, 8)).tolist()', call: '[[ndcg_at_k(r, 5) for r in R], mean_reciprocal_rank(R)]', hidden: true }
        ]
      },
      {
        id: 'ranknet-gradient', title: 'The RankNet Loss and Its Gradient', difficulty: 'medium',
        prompt: `For one query with model scores \`scores\` and graded relevances \`rels\`, implement \`ranknet_loss_grad(scores, rels)\` → \`(loss, grad)\`:

- loss = the **mean** over all pairs (i, j) with rels[i] > rels[j] of log(1 + exp(−(s_i − s_j))) (use \`np.logaddexp(0, -diff)\` for numerical stability);
- grad = the gradient of that mean loss with respect to each score (an array shaped like \`scores\`).

If there are no such pairs, return \`(0.0, np.zeros(len(scores)))\`.`,
        starter: `
import numpy as np

def ranknet_loss_grad(scores, rels):
    # TODO
    pass
`,
        solution: `
import numpy as np

def ranknet_loss_grad(scores, rels):
    s = np.asarray(scores, dtype=float)
    r = np.asarray(rels)
    i, j = np.nonzero(r[:, None] > r[None, :])
    grad = np.zeros(len(s))
    if len(i) == 0:
        return 0.0, grad
    diff = s[i] - s[j]
    loss = float(np.mean(np.logaddexp(0, -diff)))
    g = -1 / (1 + np.exp(diff)) / len(i)          # d loss / d diff for each pair
    np.add.at(grad, i, g)
    np.add.at(grad, j, -g)
    return loss, grad
`,
        hints: ['`np.nonzero(r[:, None] > r[None, :])` lists every correctly ordered pair.', 'The derivative of log(1 + e^(−d)) with respect to d is −1 / (1 + e^d).', '`np.add.at` accumulates contributions when an index appears in several pairs.'],
        explain: `Each pair pushes the more relevant item up and the less relevant one down, more strongly when the model currently orders them wrongly. LambdaRank keeps exactly these gradients but rescales each pair's push by how much swapping the pair would change NDCG.`,
        tests: [
          { call: 'ranknet_loss_grad([0.0, 0.0], [1, 0])', expect: '(np.log(2), np.array([-0.5, 0.5]))' },
          { name: 'no informative pairs', call: 'ranknet_loss_grad([1.0, 2.0], [1, 1])', expect: '(0.0, np.zeros(2))' },
          { name: 'matches finite differences', setup: 'rng = np.random.default_rng(0)\ns = rng.normal(size=6); r = np.array([3, 0, 1, 2, 0, 1])', code: 'loss, g = ranknet_loss_grad(s, r)\nnum = np.zeros(6)\nfor k in range(6):\n    d = np.zeros(6); d[k] = 1e-6\n    num[k] = (ranknet_loss_grad(s + d, r)[0] - ranknet_loss_grad(s - d, r)[0]) / 2e-6\nassert np.allclose(g, num, atol=1e-6)', hidden: true },
          { name: 'random query vs reference', setup: 'rng = np.random.default_rng(1)\ns = rng.normal(size=8); r = rng.integers(0, 3, 8)', call: 'ranknet_loss_grad(s, r)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Burges — From RankNet to LambdaRank to LambdaMART: An Overview', url: 'https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/' },
      { title: 'Järvelin & Kekäläinen — Cumulated gain-based evaluation of IR techniques (NDCG)', url: 'https://doi.org/10.1145/582415.582418' },
      { title: 'Robertson & Zaragoza — The Probabilistic Relevance Framework: BM25 and Beyond', url: 'https://doi.org/10.1561/1500000019' },
      { title: 'LightGBM — learning-to-rank objectives', url: 'https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective' }
    ]
  },

  // =====================================================================================
  {
    id: 'vector-search', branch: 'apps', title: 'Vector Search & Approximate Nearest Neighbours', short: 'Vector Search', level: 3, minutes: 40, prereqs: ['text-embeddings', 'kmeans'],
    tags: ['vector search', 'nearest neighbour search', 'approximate nearest neighbours', 'ann', 'recall at k', 'ivf', 'inverted file index', 'product quantization', 'hnsw', 'vector database', 'faiss', 'hybrid search', 'reciprocal rank fusion', 'metadata filtering'],
    summary: 'Finding the most similar embeddings among millions: exact search as a matrix product and when it stops scaling, recall@k and the speed–accuracy trade-off, IVF indexes that search only a few k-means clusters, product quantization that compresses vectors into a few bytes, HNSW graph indexes, and practical vector-database concerns — filtering, hybrid search with reciprocal rank fusion, and re-embedding.',
    why: `Retrieval-augmented generation, semantic search, recommendations, deduplication and image search all reduce to the same operation: **find the stored vectors most similar to a query vector**. With a few thousand vectors a brute-force matrix product is perfect; with hundreds of millions it becomes too slow and too memory-hungry. **Approximate nearest-neighbour (ANN)** indexes trade a little accuracy for orders of magnitude in speed and memory — and understanding how they work tells you how to tune them and what can go wrong.`,
    objectives: [
      'Implement exact top-k search and relate cosine similarity to Euclidean distance',
      'Measure ANN quality with recall@k and reason about speed, memory and accuracy trade-offs',
      'Explain IVF, product quantization and HNSW and their main parameters',
      'Combine vector and keyword search with reciprocal rank fusion and handle filters and updates'
    ],
    sections: [
      { type: 'text', title: 'Exact nearest-neighbour search', md: `Exact search compares the query with every stored vector — one matrix product for a batch of queries — and keeps the top k. For normalized vectors, the usual similarity measures agree:

$$ ‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b = 2 − 2 cos(a, b)     when ‖a‖ = ‖b‖ = 1 $$
:::where
a, b — two embedding vectors
‖a − b‖² — their squared Euclidean distance
cos(a, b) — their cosine similarity, a·b / (‖a‖‖b‖), which equals a·b for unit vectors
:::

So after normalizing, "smallest distance", "largest cosine" and "largest dot product" give the same ranking. (Unnormalized dot-product search, "maximum inner product search", is different and is what some recommendation models need.)

Cost: n·d multiply-adds per query. With 100 million 768-dimensional float32 vectors that is about 300 GB of memory and roughly 77 billion operations per query — too slow for interactive search on a CPU, which is where approximate indexes come in.` },
      { type: 'code', title: 'Exact top-k search with one matrix product', code: `
import numpy as np, time
rng = np.random.default_rng(0)
n, d = 20000, 64
base = rng.normal(size=(n, d)).astype(np.float32)
base /= np.linalg.norm(base, axis=1, keepdims=True)                      # unit vectors
queries = base[:5] + 0.1 * rng.normal(size=(5, d)).astype(np.float32)   # queries close to items 0..4
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

t0 = time.perf_counter()
sims = queries @ base.T                                                  # cosine similarities, shape (5, n)
top = np.argsort(-sims, axis=1)[:, :5]
elapsed = time.perf_counter() - t0
dist = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(axis=2)       # squared Euclidean distances
print("top-5 by cosine:   ", top[0])
print("top-5 by distance: ", np.argsort(dist, axis=1)[0, :5])
print(f"searched {n} vectors for 5 queries in {elapsed * 1000:.1f} ms; nearest neighbour of query i is item i: {bool(np.all(top[:, 0] == np.arange(5)))}")
`, after: `For unit vectors, ranking by cosine and by Euclidean distance gives the same neighbours. At this size brute force is fast and exact — approximate indexes only pay off with millions of vectors or very tight latency budgets.` },
      { type: 'text', title: 'Recall@k and the trade-off', md: `An approximate index returns *almost* the right neighbours. Its quality is measured against exact search:

$$ recall@k = | ANN top-k ∩ exact top-k | / k $$
:::where
ANN top-k — the k results returned by the approximate index
exact top-k — the true k nearest neighbours from brute-force search
:::

Every index has knobs that trade recall against queries per second, memory and build time. Benchmarks (such as ann-benchmarks) plot recall against throughput; pick the operating point your application needs — and measure on your own data, because embeddings from different models behave differently.` },
      { type: 'text', title: 'IVF: cluster first, then search a few clusters', md: `An **inverted file (IVF)** index runs k-means (see *Clustering with k-Means*) on the vectors to get n_list centroids, and stores each vector in the list of its nearest centroid. At query time it finds the n_probe centroids closest to the query and searches only their lists. With n_list = 1,000 and n_probe = 10, each query scans about 1% of the data.

The failure mode: a true neighbour can sit in a list whose centroid is not among the n_probe closest, especially near cluster boundaries. Raising n_probe recovers recall at the cost of speed.` },
      { type: 'code', title: 'An IVF index from scratch', code: `
import numpy as np
rng = np.random.default_rng(1)
n, d, n_lists = 20000, 32, 64
centers = rng.normal(size=(200, d))                           # 200 overlapping clusters, like real embeddings
base = centers[rng.integers(0, 200, n)] + rng.normal(size=(n, d))
queries = centers[rng.integers(0, 200, 50)] + rng.normal(size=(50, d))

def sqdist(A, B):
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T

C = base[rng.choice(n, n_lists, replace=False)]               # k-means for the coarse centroids
for _ in range(10):
    assign = sqdist(base, C).argmin(axis=1)
    C = np.array([base[assign == c].mean(axis=0) if np.any(assign == c) else C[c] for c in range(n_lists)])
assign = sqdist(base, C).argmin(axis=1)
lists = [np.nonzero(assign == c)[0] for c in range(n_lists)]  # the inverted lists

exact = np.argsort(sqdist(queries, base), axis=1)[:, :10]
for n_probe in (1, 2, 4, 8, 16):
    recalls, scanned = [], []
    for qi, q in enumerate(queries):
        probe = np.argsort(sqdist(q[None], C)[0])[:n_probe]    # the n_probe closest centroids
        cand = np.concatenate([lists[c] for c in probe])
        found = cand[np.argsort(sqdist(q[None], base[cand])[0])[:10]]
        recalls.append(len(set(found) & set(exact[qi])) / 10)
        scanned.append(len(cand) / n)
    print(f"n_probe = {n_probe:2d}: recall@10 = {np.mean(recalls):.3f}, scanning {np.mean(scanned):.1%} of the vectors")
`, after: `Searching only the single closest list finds about half of the true neighbours — where clusters overlap, neighbours spill into adjacent lists. Each doubling of n_probe recovers more, and by 16 lists recall reaches about 96% while scanning roughly a quarter of the data. The right n_probe depends on how your own embeddings are distributed, so tune it on your data.` },
      { type: 'chart', title: "IVF: recall against the share of vectors scanned", md: `20,000 clustered vectors in an IVF index with 100 lists, 200 queries:`, after: `Searching a single list scans about 1% of the database and finds about 80% of the true top 10: true neighbours often sit just across a list boundary. Each doubling of n_probe recovers more of them: with 4 lists recall is essentially 100% while scanning only about 4% of the vectors. That steep early rise is why ANN search is so effective: most of the recall comes from a small part of the data.`, chart: {"kind":"line","x":[1,2,4,8,16,32,64],"xLog":true,"series":[{"name":"recall@10 (versus exact search)","y":[0.801,0.95,0.999,1,1,1,1]},{"name":"share of the database scanned","y":[0.013,0.024,0.043,0.083,0.161,0.315,0.629]}],"xLabel":"n_probe: lists searched (of 100, log scale)","yLabel":"share","yMin":0,"yMax":1} },
      { type: 'text', title: 'Product quantization: compress the vectors', md: `Memory is often the real limit. **Product quantization (PQ)** splits each vector into m sub-vectors, runs k-means separately in each sub-space (typically 256 centroids, so each code fits in one byte), and stores every vector as m small codes:

$$ x ≈ [ c¹_{j₁}, c²_{j₂}, …, c^m_{j_m} ]          ‖q − x‖² ≈ Σ_s ‖ q^s − c^s_{j_s} ‖² $$
:::where
x — a database vector, split into sub-vectors x¹ … x^m
c^s_j — centroid j of the codebook learned for sub-space s
j_s — the stored code for sub-vector s, the index of its nearest centroid
q^s — the matching part of the query; the distances ‖q^s − c^s_j‖² are computed once per query into small lookup tables
:::

A 768-dimensional float32 vector (3,072 bytes) becomes 96 one-byte codes — 32× smaller — and each distance becomes 96 table lookups. **IVF-PQ** combines both ideas (the workhorse of the FAISS library); when accuracy matters, the top candidates are re-ranked with the exact vectors.` },
      { type: 'text', title: 'Graph indexes: HNSW', md: `**Hierarchical Navigable Small World (HNSW)** graphs connect every vector to a few dozen near neighbours. A search starts at an entry point and greedily moves to whichever neighbour is closer to the query, until no neighbour improves. Upper layers contain only a sample of the points with long-range links — like an express train — so the search first jumps across the space, then descends to finer layers for the local walk.

Knobs: M (links per node: more means higher recall and more memory), efConstruction (build quality) and efSearch (how many candidates a query keeps; the main recall/speed knob). HNSW gives excellent recall and latency and supports inserts, but it keeps full vectors plus the graph in memory. It is the default index in many vector databases (pgvector, Qdrant, Weaviate, Milvus, Elasticsearch/OpenSearch).` },
      { type: 'check', question: { type: 'mcq', q: 'An IVF index has poor recall. What is the most direct fix?', options: ['Reduce n_probe', 'Increase n_probe, so more lists are searched', 'Use fewer training vectors', 'Remove normalization'], answer: 1, explain: 'Searching more lists recovers neighbours near cluster boundaries, at some cost in speed.' } },
      { type: 'text', title: 'Vector search in practice', md: `- **Metadata filters** ("only documents this user may see", "only 2025"): filtering *before* the vector search can break graph traversal; filtering *after* can return fewer than k results. Good systems integrate filters into the index search.
- **Hybrid search**: combine keyword search (BM25, great for names, codes and rare words) with vector search (great for meaning). **Reciprocal rank fusion** merges ranked lists without needing comparable scores:

$$ RRF(d) = Σ_r 1 / ( k + rank_r(d) ) $$
:::where
r — one of the rankings being fused (for example BM25 and vector search)
rank_r(d) — the position of document d in ranking r (1 is the top); a document missing from a ranking contributes nothing for it
k — a constant, usually 60, that damps the influence of the very top positions
:::

- **Updates and deletes**: graph indexes handle inserts, but heavy churn degrades them; many systems rebuild periodically.
- **Re-embedding**: vectors from different embedding models (or versions) are not comparable — changing models means re-embedding the whole collection.
- **Evaluate end to end**: measure recall@k of the index *and* the quality of the final answers on your own queries.` },
      { type: 'text', title: "Choosing an index", md: `| Index | Memory | Speed | Recall | Good for |
| --- | --- | --- | --- | --- |
| flat (exact) | the full vectors | slow: scans everything | 100% | up to about a million vectors, and as the ground truth for testing |
| IVF | the full vectors plus centroids | fast; tune n_probe | high, tunable | millions of vectors; quick to build |
| IVF-PQ | compressed codes, 10 to 50 times smaller | fast | lower: rescore the top candidates with full vectors | hundreds of millions to billions of vectors |
| HNSW | the full vectors plus a graph | very fast | very high | the default in most vector databases when memory allows |
| disk-based graphs (DiskANN) | mostly on SSD | fast | high | billions of vectors on a single machine |

Whatever you choose, measure recall@k against exact search on a sample of your own queries, together with the latency at your expected load: published benchmarks rarely match the distribution of your data.` }
    ],
    when: `- Retrieval for RAG, semantic and image search, recommendations, deduplication and clustering at scale.
- Brute force (a matrix product, on GPU if needed) up to around a million vectors or when exactness matters.
- IVF or IVF-PQ for very large or memory-constrained collections; HNSW for low-latency search with frequent inserts.
- Hybrid keyword + vector search when queries contain names, codes or rare terms.`,
    pitfalls: `- Mixing normalized and unnormalized vectors, or cosine and dot-product metrics, between indexing and querying.
- Reporting index speed without recall — an index that returns the wrong neighbours fast is useless.
- Mixing embeddings from different models or versions in one index.
- Post-filtering that silently returns too few results.
- Tuning on random vectors instead of your real embeddings and queries.`,
    takeaways: [
      'For unit vectors, cosine similarity, dot product and Euclidean distance give the same ranking.',
      'Exact search is one matrix product — fine up to millions; beyond that, approximate indexes.',
      'IVF searches the n_probe nearest k-means lists; PQ compresses vectors into byte codes with lookup-table distances.',
      'HNSW navigates a layered neighbour graph; efSearch trades recall for speed.',
      'Measure recall@k; combine keyword and vector search with reciprocal rank fusion; re-embed when models change.'
    ],
    quiz: [
      { type: 'mcq', q: 'Two unit vectors have cosine similarity 0.8. What is their squared Euclidean distance?', options: ['0.2', '0.4', '0.8', '1.6'], answer: 1, explain: '2 − 2 × 0.8 = 0.4.' },
      { type: 'fill', q: 'An ANN index returns 10 results, 8 of which are among the true 10 nearest neighbours. What is recall@10?', answer: ['0.8', '80%'], explain: '8 / 10.' },
      { type: 'mcq', q: 'What does product quantization mainly reduce?', options: ['The number of queries', 'The memory per vector, by storing small codes instead of floats', 'The embedding dimension of the model', 'The number of clusters'], answer: 1, explain: 'For example 3,072 bytes become 96 bytes.' },
      { type: 'tf', q: 'Vectors produced by two different embedding models can be mixed in one index without problems.', answer: false, explain: 'Their spaces are unrelated; similarities across models are meaningless.' },
      { type: 'mcq', q: 'Why use reciprocal rank fusion to combine BM25 and vector search?', options: ['It needs no scores at all, only ranks, so the scales do not have to match', 'It is the only exact method', 'It removes the need for embeddings', 'It makes BM25 faster'], answer: 0, explain: 'BM25 scores and cosine similarities are not on comparable scales; ranks are.' },
      {"type":"mcq","q":"An IVF index has 100 lists of similar size. Roughly what share of the database does a query scan with n_probe = 8?","options":["1%","8%","50%","100%"],"answer":1,"explain":"Each list holds about 1% of the vectors, so probing 8 lists scans about 8%."},
      {"type":"tf","q":"For a few hundred thousand vectors, exact (flat) search can be a perfectly reasonable choice.","answer":true,"explain":"One matrix product over a few hundred thousand vectors is fast on modern hardware, and exact search needs no tuning and has perfect recall."}
    ],
    problems: [
      {
        id: 'topk-recall-rrf', title: 'Top-k Search, Recall and Rank Fusion', difficulty: 'easy',
        prompt: `Implement:

- \`top_k_cosine(queries, base, k)\` → an integer array (n_queries, k) with the indices of the k most cosine-similar rows of base for each query, most similar first (normalize the rows; break ties by lower index with \`np.argsort(-sims, axis=1, kind="stable")\`).
- \`recall_at_k(found, truth)\` → the mean over queries of |found_i ∩ truth_i| / k, where both are (n_queries, k) arrays.
- \`rrf(rankings, k=60)\` → reciprocal rank fusion: given a list of rankings (lists of integer document ids, best first), return all documents sorted by Σ 1 / (k + rank) (rank starting at 1) in descending order, breaking ties by the smaller id.`,
        starter: `
import numpy as np

def top_k_cosine(queries, base, k):
    # TODO
    pass

def recall_at_k(found, truth):
    # TODO
    pass

def rrf(rankings, k=60):
    # TODO
    pass
`,
        solution: `
import numpy as np

def top_k_cosine(queries, base, k):
    Q = np.asarray(queries, dtype=float)
    B = np.asarray(base, dtype=float)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.argsort(-(Q @ B.T), axis=1, kind="stable")[:, :k]

def recall_at_k(found, truth):
    found, truth = np.asarray(found), np.asarray(truth)
    k = truth.shape[1]
    return float(np.mean([len(set(f.tolist()) & set(t.tolist())) / k for f, t in zip(found, truth)]))

def rrf(rankings, k=60):
    score = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1 / (k + rank)
    return sorted(score, key=lambda doc: (-score[doc], doc))
`,
        hints: ['Normalize rows with `np.linalg.norm(..., axis=1, keepdims=True)`.', 'Sets make the overlap easy: `len(set(a) & set(b))`.', 'Sort by a key of (−score, id).'],
        explain: `Exact top-k is the ground truth every approximate index is judged against. RRF needs no score calibration between systems — documents that rank well in several lists rise to the top.`,
        tests: [
          { call: 'top_k_cosine(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0], [2.0, 0.1], [-1.0, 0.0], [1.0, 1.0]]), 2)', expect: 'np.array([[1, 3]])' },
          { call: 'recall_at_k(np.array([[1, 2, 3]]), np.array([[3, 4, 1]]))', expect: '2 / 3' },
          { call: 'rrf([[1, 2, 3], [3, 1, 4]])', expect: '[1, 3, 2, 4]' },
          { name: 'random vectors vs reference', setup: 'rng = np.random.default_rng(0)\nQ = rng.normal(size=(4, 8)); B = rng.normal(size=(30, 8))', call: 'top_k_cosine(Q, B, 5)', hidden: true },
          { name: 'fusion with a small k', call: 'rrf([[5, 2, 9], [9, 5], [2, 7, 5]], k=1)', hidden: true }
        ]
      },
      {
        id: 'ivf-search', title: 'Searching an IVF Index', difficulty: 'medium',
        prompt: `Implement a tiny inverted-file index (Euclidean distance):

- \`build_lists(base, centroids)\` → a list with one entry per centroid: a NumPy array (sorted ascending) of the indices of the base vectors whose nearest centroid it is (ties go to the lower centroid index, as \`argmin\` does).
- \`ivf_search(query, base, centroids, lists, n_probe, k)\` → a NumPy array with the indices of the k nearest base vectors among those stored in the n_probe lists whose centroids are closest to the query. Order the candidates by index first, then use a **stable** sort by distance, so equal distances keep the lower index first.`,
        starter: `
import numpy as np

def build_lists(base, centroids):
    # TODO
    pass

def ivf_search(query, base, centroids, lists, n_probe, k):
    # TODO
    pass
`,
        solution: `
import numpy as np

def build_lists(base, centroids):
    B, C = np.asarray(base, dtype=float), np.asarray(centroids, dtype=float)
    d2 = ((B[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    assign = d2.argmin(axis=1)
    return [np.nonzero(assign == c)[0] for c in range(len(C))]

def ivf_search(query, base, centroids, lists, n_probe, k):
    q = np.asarray(query, dtype=float)
    B, C = np.asarray(base, dtype=float), np.asarray(centroids, dtype=float)
    probe = np.argsort(((C - q) ** 2).sum(axis=1), kind="stable")[:n_probe]
    cand = np.sort(np.concatenate([lists[c] for c in probe]))
    dist = ((B[cand] - q) ** 2).sum(axis=1)
    return cand[np.argsort(dist, kind="stable")[:k]]
`,
        hints: ['Squared distances to all centroids: broadcast (n, 1, d) against (1, c, d) and sum over the last axis.', 'Sort the concatenated candidate indices before the stable distance sort.'],
        explain: `Only the probed lists are scanned, so the cost is about n · n_probe / n_lists distance computations instead of n. With n_probe equal to the number of lists, IVF becomes exact brute-force search.`,
        tests: [
          { call: '[a.tolist() for a in build_lists(np.array([[0.0], [1.0], [10.0], [11.0]]), np.array([[0.5], [10.5]]))]', expect: '[[0, 1], [2, 3]]' },
          { setup: 'base = np.array([[0.0], [1.0], [10.0], [11.0]]); C = np.array([[0.5], [10.5]])\nL = build_lists(base, C)', call: 'ivf_search(np.array([0.9]), base, C, L, 1, 2)', expect: 'np.array([1, 0])' },
          { name: 'probing every list equals brute force', setup: 'rng = np.random.default_rng(0)\nB = rng.normal(size=(200, 4)); C = B[:8].copy(); q = rng.normal(size=4)\nL = build_lists(B, C)', code: 'got = ivf_search(q, B, C, L, 8, 5)\nexact = np.argsort(((B - q) ** 2).sum(axis=1), kind="stable")[:5]\nassert np.array_equal(got, exact)', hidden: true },
          { name: 'random index vs reference', setup: 'rng = np.random.default_rng(1)\nB = rng.normal(size=(150, 3)); C = rng.normal(size=(6, 3)); q = rng.normal(size=3)\nL = build_lists(B, C)', call: 'ivf_search(q, B, C, L, 2, 4)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Johnson, Douze & Jégou — Billion-scale similarity search with GPUs (FAISS)', url: 'https://arxiv.org/abs/1702.08734' },
      { title: 'Malkov & Yashunin — Efficient and robust approximate nearest neighbor search using HNSW graphs', url: 'https://arxiv.org/abs/1603.09320' },
      { title: 'Jégou, Douze & Schmid — Product Quantization for Nearest Neighbor Search', url: 'https://doi.org/10.1109/TPAMI.2010.57' },
      { title: 'ANN-Benchmarks', url: 'https://ann-benchmarks.com/' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
