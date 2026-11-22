/* ML Atlas – content: Classical ML (evaluation, validation, kNN, trees, ensembles) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'evaluation-metrics', branch: 'ml', title: 'Evaluation Metrics: Beyond Accuracy', short: 'Evaluation Metrics', level: 1, minutes: 45, prereqs: ['logistic-regression'],
    tags: ['accuracy', 'precision', 'recall', 'f1', 'roc', 'auc', 'confusion matrix', 'rmse'],
    summary: 'How to measure whether a model is actually good: confusion matrices, precision and recall, F1, ROC/AUC, thresholds, and the right metric for regression.',
    why: `"95% accurate" can mean an excellent model or a useless one that always predicts the majority class. Real decisions (approving loans, flagging fraud, screening for disease, filtering spam) have asymmetric costs: a missed cancer is not the same as a false alarm. Choosing the right metric, and the right decision threshold, is how you align a model with what actually matters. It is also how you compare models honestly, report results credibly, and avoid shipping something that looks good on paper only.`,
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
| **Actually positive** | **TP**: true positive (caught it) | **FN**, false negative (missed it) |
| **Actually negative** | **FP**: false positive (false alarm) | **TN**, true negative (correctly ignored) |

Everything else is computed from these four counts:

| Metric | Formula | Question it answers |
| --- | --- | --- |
| **Accuracy** | (TP + TN) / all | How often is the model right? |
| **Precision** | TP / (TP + FP) | When it says "positive", how often is it right? |
| **Recall** (sensitivity, TPR) | TP / (TP + FN) | Of all real positives, how many does it catch? |
| **Specificity** | TN / (TN + FP) | Of all real negatives, how many does it leave alone? |
| **False positive rate** | FP / (FP + TN) | How often does it raise a false alarm? |
| **F1 score** | 2·P·R / (P + R) | A single balance of precision and recall (harmonic mean) |` },
      { type: 'text', title: 'Why accuracy misleads', md: `A fraud dataset with 1% fraud: a "model" that always says "legitimate" scores **99% accuracy**, and catches zero fraud (recall 0). Whenever classes are imbalanced, accuracy is dominated by the majority class.

Which metric to emphasise depends on the cost of each mistake:

- **Recall matters most** when missing a positive is costly: disease screening, fraud, safety defects. (Better to send a few healthy people for a second test.)
- **Precision matters most** when false alarms are costly: spam filtering (losing a real email), auto-blocking accounts, recommending something to a user.
- **F1** when you need one number balancing both and have no clear cost ratio.

:::analogy A fishing net
Recall: what fraction of the fish in the lake did your net catch? Precision: what fraction of what's in your net is actually fish (rather than boots and seaweed)? A bigger net raises recall and lowers precision.
:::` },
      { type: 'code', title: "A skewed task: is this digit a 5?", md: `scikit-learn's digits dataset holds 1,797 small 8×8 images of handwritten digits. Turn it into a **binary** task, "is this a 5?", and only about 10% of the images are positive. Accuracy is measured with 3-fold [cross-validation](#/concept/cross-validation) for a real linear classifier and for a "classifier" that always says *no*.`, code: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

digits = load_digits()
print(f"{len(digits.images)} images of 8 x 8 grey levels (0 to 16). The first one, a {digits.target[0]}:")
for row in digits.images[0].astype(int):
    print("  ", " ".join(f"{v:2d}" for v in row))
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=0)
y_train_5 = (y_train == 5)                       # True for 5s, False for every other digit
print(f"share of 5s: {y_train_5.mean():.1%}")

sgd = SGDClassifier(random_state=42)
print("linear classifier, accuracy per fold:", cross_val_score(sgd, X_train, y_train_5, cv=3).round(3))
never5 = cross_val_score(DummyClassifier(strategy="most_frequent"), X_train, y_train_5, cv=3).mean()
print(f"a model that never says 5 is right {never5:.0%} of the time")

pred = cross_val_predict(sgd, X_train, y_train_5, cv=3)   # a clean prediction for every training image
print("confusion matrix (rows: actual not-5 / 5, columns: predicted not-5 / 5):")
print(confusion_matrix(y_train_5, pred))
print(f"precision {precision_score(y_train_5, pred):.3f}   recall {recall_score(y_train_5, pred):.3f}   F1 {f1_score(y_train_5, pred):.3f}")`, after: `\`cross_val_predict\` returns, for every training example, a prediction made by a model that did **not** see that example, so the confusion matrix is honest without touching the test set. The real classifier is excellent, but notice how little its accuracy (about 99%) says on its own: the useless "never 5" model already scores 90%. Precision and recall show what matters: when it says "5" it is right about 95% of the time (precision), and it finds about 93% of the real 5s (recall).`, predict: {"type":"mcq","q":"Only about 10% of the digits are 5s. What accuracy does a model that **always** answers \"not a 5\" get?","options":["About 10%","About 50%","About 90%","0%, because it never finds a 5"],"answer":2,"explain":"It is right on every image that is not a 5, which is about 90% of them. That is why accuracy alone is misleading on skewed classes.","expect":["a model that never says 5 is right 90% of the time"]} },
      { type: 'demo', title: 'Interactive: move the threshold, watch the trade-off', md: `A classifier outputs scores; you choose the threshold. Slide it and watch the confusion matrix, precision, recall and the point on the ROC curve.`, demo: 'threshold' },
      { type: 'text', title: 'ROC curves and AUC', md: `Sweeping the threshold from 1 down to 0 traces the **ROC curve**: the true positive rate (recall) against the false positive rate. The diagonal is random guessing; the top-left corner is perfection.

**AUC** (area under the ROC curve) summarises it in one threshold-free number:
- 1.0 = perfect ranking, 0.5 = random, below 0.5 = worse than random (flip the predictions).
- Elegant interpretation: **AUC is the probability that a randomly chosen positive gets a higher score than a randomly chosen negative.** It measures how well the model *ranks*, regardless of threshold or calibration.

On **highly imbalanced** data ROC can look rosy because the false positive rate stays small even with many false alarms (the negatives are so numerous). The **precision-recall curve** and its area (average precision) are more informative there.` },
      { type: 'code', title: "Choosing a threshold for a goal", md: `Classifiers produce a **score** (\`decision_function\`) or a probability, and the threshold turns it into a decision. Suppose a wrong "5" is costly and you need 99% precision. Find the lowest threshold that achieves it, and see what recall remains:`, code: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=0)
y_train_5 = (y_train == 5)
scores = cross_val_predict(SGDClassifier(random_state=42), X_train, y_train_5, cv=3, method="decision_function")
precision, recall, thresholds = precision_recall_curve(y_train_5, scores)

i = np.argmax(precision >= 0.99)          # the first (lowest) threshold with at least 99% precision
print(f"threshold {thresholds[i]:.1f}: precision {precision[i]:.3f}, recall {recall[i]:.3f}")
print(f"default threshold 0: precision {np.mean(y_train_5[scores >= 0]):.3f}, recall {np.mean(scores[y_train_5] >= 0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_train_5, scores):.3f}")`, after: `Reaching 99% precision costs recall: some genuine 5s now score below the threshold. That is the trade-off you choose deliberately, using the curve. And beware of claims like "our model has 99% precision": any classifier can reach high precision by raising its threshold. Always ask **"at what recall?"**

Two practical details: precision usually, but not always, rises as the threshold goes up (removing one true positive can lower it); recall can only fall.` },
      { type: 'chart', title: "Precision and recall against the threshold", after: `Moving the threshold right makes the classifier more cautious: precision rises towards 1 while recall falls, because fewer real 5s clear the bar. Every threshold is a different trade-off; the default of 0 is just one of them.`, chart: {"kind":"line","x":[-300,-275,-250,-225,-200,-175,-150,-125,-100,-75,-50,-25,0,25,50,75,100,125,150,175,200,225,250,275,300],"points":false,"series":[{"name":"precision","y":[0.344,0.384,0.433,0.496,0.56,0.643,0.703,0.757,0.831,0.868,0.903,0.934,0.955,0.962,0.976,0.984,0.983,0.991,1,1,1,1,1,1,1]},{"name":"recall","y":[0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.985,0.978,0.971,0.963,0.941,0.934,0.926,0.904,0.89,0.838,0.816,0.787,0.728,0.647,0.618,0.581,0.507,0.463]}],"xLabel":"decision threshold on the classifier's score","yLabel":"value","vlines":[{"x":0,"label":"default threshold 0"}]} },
      { type: 'chart', title: "The precision–recall curve", after: `The same trade-off as one curve: each point is one threshold. A perfect classifier would sit in the top-right corner. Random guessing gives a precision equal to the share of positives, about 0.10 here, which is why a precision–recall curve is so revealing when positives are rare.`, chart: {"kind":"line","points":false,"series":[{"name":"linear classifier: is it a 5?","x":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.993,0.985,0.985,0.985,0.985,0.985,0.978,0.971,0.971,0.971,0.971,0.956,0.941,0.941,0.926,0.904,0.882,0.853,0.831,0.809,0.779,0.75,0.721,0.691,0.662,0.632,0.603,0.574,0.544,0.515,0.485,0.456,0.426,0.397,0.368,0.338,0.309,0.279,0.25,0.221,0.191,0.162,0.132,0.103,0.074,0.044,0.015],"y":[0.101,0.101,0.102,0.102,0.102,0.102,0.103,0.103,0.103,0.104,0.104,0.104,0.105,0.105,0.105,0.106,0.106,0.106,0.107,0.107,0.107,0.108,0.108,0.108,0.109,0.109,0.109,0.11,0.11,0.11,0.111,0.111,0.112,0.112,0.112,0.113,0.113,0.113,0.114,0.114,0.115,0.115,0.115,0.116,0.116,0.117,0.117,0.117,0.118,0.118,0.119,0.119,0.119,0.12,0.12,0.121,0.121,0.122,0.122,0.122,0.123,0.123,0.124,0.124,0.125,0.125,0.126,0.126,0.127,0.127,0.127,0.128,0.128,0.129,0.129,0.13,0.13,0.131,0.131,0.132,0.132,0.133,0.133,0.134,0.135,0.135,0.136,0.136,0.137,0.137,0.138,0.138,0.139,0.139,0.14,0.141,0.141,0.142,0.142,0.143,0.144,0.144,0.145,0.145,0.146,0.147,0.147,0.148,0.149,0.149,0.15,0.151,0.151,0.152,0.153,0.153,0.154,0.155,0.155,0.156,0.157,0.158,0.158,0.159,0.16,0.161,0.161,0.162,0.163,0.164,0.164,0.165,0.166,0.167,0.168,0.169,0.169,0.17,0.171,0.172,0.173,0.174,0.175,0.175,0.176,0.177,0.178,0.179,0.18,0.181,0.182,0.183,0.184,0.185,0.186,0.187,0.188,0.189,0.19,0.191,0.192,0.193,0.195,0.196,0.197,0.198,0.199,0.2,0.201,0.203,0.204,0.205,0.206,0.208,0.209,0.21,0.212,0.213,0.214,0.216,0.217,0.218,0.218,0.22,0.221,0.222,0.224,0.225,0.227,0.228,0.23,0.232,0.233,0.235,0.236,0.238,0.24,0.242,0.243,0.245,0.247,0.249,0.25,0.252,0.254,0.256,0.258,0.26,0.262,0.264,0.266,0.268,0.271,0.273,0.275,0.277,0.28,0.282,0.284,0.287,0.289,0.292,0.294,0.297,0.299,0.302,0.305,0.308,0.31,0.313,0.316,0.319,0.322,0.325,0.328,0.332,0.335,0.338,0.342,0.345,0.349,0.352,0.356,0.36,0.364,0.368,0.372,0.376,0.38,0.385,0.389,0.394,0.398,0.403,0.408,0.413,0.418,0.423,0.429,0.434,0.44,0.447,0.453,0.459,0.466,0.472,0.479,0.486,0.493,0.5,0.508,0.515,0.523,0.531,0.54,0.549,0.558,0.567,0.577,0.587,0.597,0.608,0.619,0.631,0.643,0.655,0.668,0.682,0.696,0.711,0.726,0.736,0.753,0.77,0.788,0.807,0.821,0.835,0.857,0.88,0.904,0.915,0.928,0.955,0.969,0.976,0.984,0.983,0.991,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}],"xLabel":"recall","yLabel":"precision","hlines":[{"y":0.101,"label":"random guessing (share of 5s)"}],"yMin":0} },
      { type: 'text', title: "ROC curve or precision–recall curve?", md: `Both curves sweep the threshold, but they answer different questions:

- The **ROC curve** plots recall against the false-positive rate, the share of *negatives* wrongly flagged. Its AUC is 1 for a perfect model and 0.5 for random guessing.
- The **precision–recall curve** plots precision against recall, so it shows how many of the *flagged* items are real.

When positives are **rare**, or false positives matter more than false negatives, prefer the **precision–recall curve**. A detector that flags 100 innocent transactions per real fraud can still have an excellent false-positive rate (100 out of a million is tiny) and a great-looking ROC curve, while its precision is a dismal 1%. When classes are balanced, the ROC curve is a fine summary.` },
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
      { type: 'check', question: { type: 'mcq', q: 'A cancer screening test should miss as few cancers as possible, even at the cost of extra follow-up tests. Which metric should it prioritise?', options: ['Precision', 'Recall', 'Accuracy', 'Specificity'], answer: 1, explain: 'Missing a positive (false negative) is the costly error, so maximise recall: accept lower precision.' } },
      { type: 'text', title: 'Regression metrics and multi-class averaging', md: `**Regression:**

| Metric | Meaning | Note |
| --- | --- | --- |
| **MAE** | average absolute error | in the target's units; robust; easy to explain |
| **RMSE** | √MSE | same units; punishes large errors more |
| **R²** | 1 − SS_res/SS_tot | unit-free; 0 = mean baseline |
| **MAPE** | mean of \|error\| / \|y\| | relative error; explodes when y ≈ 0 |

**Multi-class** precision/recall/F1 are computed per class (one-vs-rest) and then averaged:
- **macro average**: the plain mean over classes: every class counts equally (good for rare classes);
- **weighted average**: weighted by class frequency;
- **micro average**: pool all decisions first (equals accuracy for single-label problems).

Report a confusion matrix too: it shows *which* classes get confused with which.` },
      { type: 'code', title: "Many classes: one-versus-rest and one-versus-one", md: `Some algorithms handle many classes natively (logistic regression with softmax, trees, naive Bayes). Binary-only algorithms are extended in one of two ways:

- **One-versus-rest (OvR):** one classifier per class, "this class versus all others": 10 classifiers for 10 digits. Predict the class whose classifier gives the highest score.
- **One-versus-one (OvO):** one classifier for every *pair* of classes: N·(N − 1)/2 of them, 45 for 10 digits. Predict the class that wins the most duels. Each classifier trains on only two classes' data, which suits algorithms that slow down badly with more data, such as kernel SVMs.

scikit-learn picks a strategy automatically; \`OneVsRestClassifier\` and \`OneVsOneClassifier\` force one.`, code: `from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=0)
svm = SVC().fit(X_train, y_train)                            # trains 45 one-versus-one classifiers internally
print(f"test accuracy {svm.score(X_test, y_test):.3f}")
print("scores per image, default (one per class):", svm.decision_function(X_test[:1]).shape[1])
ovo = SVC(decision_function_shape="ovo").fit(X_train, y_train)
print("ovo scores per digit:", ovo.decision_function(X_test[:1]).shape[1])`, predict: {"type":"mcq","q":"How many binary classifiers does one-versus-one train for the 10 digit classes?","options":["10","20","45","100"],"answer":2,"explain":"One per pair of classes: 10 · 9 / 2 = 45. One-versus-rest would need only 10.","expect":["ovo scores per digit: 45"]} },
      { type: 'code', title: "Error analysis with the confusion matrix", md: `A good model makes mistakes too; studying **which** mistakes points to the next improvement. Here a linear classifier is trained on all ten digits (with scaled features), and the cross-validated confusion matrix is searched for the most frequent confusions.`, code: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=0)
model = make_pipeline(StandardScaler(), SGDClassifier(random_state=42))
pred = cross_val_predict(model, X_train, y_train, cv=3)
cm = confusion_matrix(y_train, pred)
print("actual  predicted: " + " ".join(f"{p:4d}" for p in range(10)))
for a in range(10):
    print(f"{a:6d}             " + " ".join(f"{v:4d}" for v in cm[a]))
print(f"accuracy {np.trace(cm) / cm.sum():.3f}")
errors = cm.copy()
np.fill_diagonal(errors, 0)                                  # keep only the mistakes
pairs = sorted(((errors[a, p], a, p) for a in range(10) for p in range(10)), reverse=True)[:4]
for n, actual, predicted in pairs:
    print(f"{n} images of a {actual} were read as a {predicted}")`, after: `Each **row** is the true digit and each **column** the prediction, so the diagonal holds the correct answers and everything else is a mistake. Rows are worth reading as proportions too (divide each row by its sum): a class with few examples can hide a high error *rate* behind small counts.

Ideas that come out of error analysis are concrete: collect or synthesize more examples of the confused digits, add features that separate them (for instance, how many closed loops a digit has: 8 has two, 9 one, 1 none), or preprocess the images (centre and straighten them), since a linear model is sensitive to small shifts and rotations.` },
      { type: 'code', title: "Multilabel classification", md: `Sometimes each example has **several** yes/no labels at once: a photo can contain a beach *and* a dog; a face-recognition system may find three known people in one picture. Here every digit gets two labels: "is it large (7, 8 or 9)?" and "is it odd?". \`KNeighborsClassifier\` supports multilabel targets directly.`, code: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=0)
y_multi = np.c_[y_train >= 7, y_train % 2 == 1]             # two labels per image
knn = KNeighborsClassifier().fit(X_train, y_multi)
print("digit 9 -> [large, odd]:", knn.predict(X_test[y_test == 9][:1])[0])
print("digit 2 -> [large, odd]:", knn.predict(X_test[y_test == 2][:1])[0])
pred = cross_val_predict(KNeighborsClassifier(), X_train, y_multi, cv=3)
print(f"F1 averaged over the two labels: {f1_score(y_multi, pred, average='macro'):.3f}")`, after: `To evaluate a multilabel model, compute a metric per label and average: \`average="macro"\` treats every label as equally important, while \`average="weighted"\` weights each label by how many examples carry it.

A generalization, **multioutput** classification, lets each label take more than two values. For example, a model that removes noise from an image outputs one label per pixel, and each label is the pixel's intensity from 0 to 255.` },
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
- Treating AUC as accuracy: a model can have high AUC and still be badly calibrated or use a poor threshold.
- Ignoring the confidence interval: a 1-point difference on a small test set is noise.`,
    takeaways: [
      'Everything starts from the confusion matrix: TP, FP, FN, TN.',
      'Precision = TP/(TP+FP): trust in positive predictions. Recall = TP/(TP+FN): coverage of real positives.',
      'The threshold trades precision against recall: choose it by the cost of each mistake.',
      'AUC = probability a random positive outranks a random negative; PR curves suit rare positives.',
      'Accuracy alone is misleading on imbalanced data.'
    ],
    quiz: [
      { type: 'fill', q: 'TP = 40, FN = 10. What is the recall? (decimal)', answer: ['0.8', '.8', '0.80', '4/5'], explain: 'TP / (TP + FN) = 40 / 50 = 0.8.' },
      { type: 'mcq', q: 'Raising the decision threshold of a classifier usually…', options: ['raises recall and lowers precision', 'raises precision and lowers recall', 'raises both', 'changes nothing'], answer: 1, explain: 'Fewer, more confident positive predictions: fewer false alarms (precision ↑) but more misses (recall ↓).' },
      { type: 'mcq', q: 'An AUC of 0.5 means…', options: ['the model is perfect', 'the model ranks positives above negatives no better than chance', 'half the predictions are correct', 'precision is 0.5'], answer: 1, explain: 'AUC is the probability that a random positive outranks a random negative; 0.5 is a coin flip.' },
      { type: 'fill', q: 'Precision 0.5 and recall 1.0. What is the F1 score? (decimal, 2 places)', answer: ['0.67', '0.667', '2/3', '0.6667'], explain: '2·0.5·1/(0.5 + 1) = 1/1.5 ≈ 0.67: the harmonic mean is pulled toward the smaller value.' },
      { type: 'mcq', q: 'For a dataset with 0.1% positives, which evaluation is most informative?', options: ['Accuracy', 'Precision-recall curve / average precision', 'R²', 'Training loss'], answer: 1, explain: 'With extreme imbalance accuracy and even ROC can look great; precision-recall focuses on the rare positive class.' },
      {"type":"mcq","q":"Why use `cross_val_predict` to build a confusion matrix on the training set?","options":["It is faster than predict","Every prediction comes from a model that did not train on that example, so the errors are honest","It removes the need for labels","It automatically chooses the best threshold"],"answer":1,"explain":"Each fold's examples are predicted by a model trained on the other folds, so the confusion matrix reflects generalization without spending the test set."},
      {"type":"mcq","q":"When should you prefer the precision–recall curve over the ROC curve?","options":["When the classes are perfectly balanced","When positives are rare or false positives are what you care about most","When the model outputs no scores","Never: the ROC curve is always better"],"answer":1,"explain":"With rare positives, the false-positive rate stays tiny even when false alarms swamp the true hits, so the ROC curve looks good while precision is poor. The PR curve exposes that."},
      {"type":"mcq","q":"A model must label each photo with any of 20 tags (beach, dog, sunset…). What kind of problem is this?","options":["Binary classification","Multiclass classification","Multilabel classification","Regression"],"answer":2,"explain":"Each photo can carry several tags at once, so there is one yes/no label per tag: multilabel classification."}
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
        explain: `Each count is the size of an intersection of two boolean masks. Every other classification metric is a ratio of these four numbers, so getting them right (including the operator precedence trap without parentheses) is the foundation.`,
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
        explain: `F1 is the harmonic mean of precision and recall: it is high only when **both** are high, unlike the arithmetic mean which a single strong number can prop up. The zero-division guards matter in practice: a model that never predicts positive has undefined precision.`,
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
        explain: `This pairwise definition is mathematically equal to the area under the ROC curve (it is the Mann–Whitney U statistic, normalised). It makes clear that AUC only cares about **ordering**: multiplying all scores by 10 or squaring them doesn't change it. For large datasets, a sort-based O(n log n) version is used instead of all pairs.`,
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
      { title: 'Google ML Crash Course, Classification: accuracy, precision, recall, ROC/AUC', url: 'https://developers.google.com/machine-learning/crash-course/classification' },
      { title: "scikit-learn, Metrics and scoring", url: "https://scikit-learn.org/stable/modules/model_evaluation.html" },
      { title: "Aurélien Géron, Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow (book; code and notebooks)", url: "https://github.com/ageron/handson-ml3" }
    ]
  },

  // =====================================================================================
  {
    id: 'cross-validation', branch: 'ml', title: 'Validation, Cross-Validation & Hyperparameter Tuning', short: 'Cross-Validation & Tuning', level: 2, minutes: 40, prereqs: ['overfitting', 'data-preparation'],
    tags: ['k-fold', 'hyperparameters', 'grid search', 'random search', 'model selection'],
    summary: 'Choosing models and hyperparameters without fooling yourself: validation sets, k-fold cross-validation, grid and random search, and the final test.',
    why: `Every model has knobs you set before training: learning rate, regularization strength, tree depth, number of neighbours. Pick them by looking at the test set and your reported score becomes optimistic fiction; pick them on a single small validation split and you may be choosing by luck. **Cross-validation** gives stable estimates from limited data, and a disciplined tuning procedure gives you numbers you can trust. This is how real projects choose models.`,
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
      { type: 'text', title: 'k-fold cross-validation', md: `With a single validation split, your score depends on which examples happened to land in it: noisy, especially for small datasets. **k-fold cross-validation** uses all the data for validation, once each:

1. Shuffle and split the training data into **k folds** of roughly equal size (k = 5 or 10 is typical).
2. For each fold i: train on the other k − 1 folds, evaluate on fold i.
3. Report the **mean** of the k scores (and their standard deviation as a sense of uncertainty).

Cost: k training runs. Cheap for classical models; often too expensive for big neural networks, which use a single validation set instead.

Variants:
- **Stratified k-fold**: each fold keeps the class proportions (default for classification).
- **Group k-fold**: all examples from one patient/user/document stay in the same fold.
- **Time-series split**: train on the past, validate on the next period, roll forward. Never shuffle time.
- **Leave-one-out**: k = n; nearly unbiased but expensive and high-variance.` },
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
      { type: 'text', title: 'Searching for good hyperparameters', md: `- **Grid search**: try every combination from lists of values (λ ∈ {0.01, 0.1, 1, 10} × depth ∈ {3, 5, 7}). Simple, exhaustive, explodes combinatorially.
- **Random search**: sample combinations at random. With the same budget it usually beats grid search, because only a few hyperparameters really matter and random sampling tries more distinct values of each.
- **Bayesian optimization / successive halving** (e.g. Optuna, Hyperband): use earlier results to decide what to try next, or stop bad runs early.

Search on a **log scale** for things like learning rates and λ (1e-4 … 1).

**The full procedure:**
1. Split off a test set and lock it away.
2. On the remaining data, compare candidates with cross-validation (or a validation split).
3. Retrain the best configuration on all non-test data.
4. Evaluate **once** on the test set and report that number.

If you also need an unbiased estimate of the tuning procedure itself, use **nested cross-validation** (an outer CV around the inner search).` },
      { type: 'code', title: "A simple grid search, and its hidden flaw", md: `A kernel SVM has two important settings, C and gamma. The simplest grid search tries every combination, trains on a training set and keeps the combination that scores best on a separate **validation** set. A third split, the **test** set, is kept aside to measure the chosen model at the end.`, code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
print("X: 150 flowers x 4 measurements in cm (sepal length, sepal width, petal length, petal width). One flower of each species:")
for row, label in zip(X[::50], y[::50]):
    print("  ", row.tolist(), "->", label)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)                 # test set: locked away
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)   # train + validation
print(f"train {len(X_train)}   validation {len(X_valid)}   test {len(X_test)}")

best_score, best = 0, None
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        score = SVC(gamma=gamma, C=C).fit(X_train, y_train).score(X_valid, y_valid)
        if score > best_score:
            best_score, best = score, {"gamma": gamma, "C": C}
print(f"best validation score {best_score:.3f} with {best}")

final = SVC(**best).fit(X_trainval, y_trainval)          # retrain on train + validation
print(f"test score of the chosen model: {final.score(X_test, y_test):.3f}")`, after: `The validation score of the winner (0.964) is higher than its test score (0.921). That is not bad luck; it is **selection bias**. Out of 36 combinations, the winner is partly the one that happened to suit these particular 28 validation flowers. Any score used to *choose* among models is optimistic, which is why the test set must never be used for choosing anything: the number you report must come from data that played no part in any decision.

A single small validation set also makes the choice itself noisy. Cross-validation fixes that, next.` },
      { type: 'code', title: "Grid search with cross-validation: GridSearchCV", md: `\`GridSearchCV\` runs the whole procedure: for every combination in the grid it runs k-fold cross-validation on the training data, picks the combination with the best **mean** cross-validation score, and (by default) retrains it on all the training data, ready to predict.`, code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}

search = GridSearchCV(SVC(), param_grid, cv=5)     # 36 combinations x 5 folds = 180 fits
search.fit(X_train, y_train)
print("best parameters:", search.best_params_)
print(f"best mean cross-validation accuracy: {search.best_score_:.3f}")
print(f"test accuracy of the refitted best model: {search.score(X_test, y_test):.3f}")
print("the refitted model:", search.best_estimator_)`, after: `- \`best_params_\` and \`best_score_\` describe the winner; \`best_score_\` is still a *selection* score, so report \`search.score(X_test, y_test)\` instead.
- \`best_estimator_\` is the winning model retrained on the whole training set; \`search.predict\` uses it directly.
- The number of fits is the size of the grid times the number of folds. Add \`n_jobs=-1\` to use all CPU cores.` },
      { type: 'code', title: "Reading the results: is the grid in the right place?", md: `\`cv_results_\` holds every combination's scores. A pivot table, or a colour-coded heat map, shows at a glance whether the search covered the right region.`, code: `import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
search = GridSearchCV(SVC(), {"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5)
search.fit(X_train, y_train)

results = pd.DataFrame(search.cv_results_)
table = results.pivot_table(index="param_C", columns="param_gamma", values="mean_test_score")
print("mean cross-validation accuracy (rows: C, columns: gamma)")
print(table.round(2).to_string())`, after: `How to read a grid like this:

- **The good region is inside the grid**, surrounded by worse scores on every side. Good: the search covered the interesting range.
- **Very small C or very large gamma** give near-chance accuracy (0.37 is the share of the largest class). Those rows and columns are just wasted compute, so the next search can skip them.
- **If the best score sat on the edge** of the grid (say at the largest C), the true optimum might lie beyond it: extend the grid in that direction.
- **If the scores barely changed across a parameter**, either that parameter does not matter or its range is wrong. Try values orders of magnitude apart before concluding it is unimportant.

Search over **powers of ten** first, then refine around the best region.` },
      { type: 'chart', title: "Mean cross-validation accuracy across the grid", after: `Each line is one value of gamma. Small C underfits for every gamma; gamma = 10 never gets close to the best because its bells are too narrow for this data. The best region (C between 1 and 100 with gamma between 0.01 and 1) is surrounded by worse values, so the grid covers it well.`, chart: {"kind":"line","x":[0.001,0.01,0.1,1,10,100],"xLog":true,"series":[{"name":"gamma = 0.01","y":[0.366,0.366,0.696,0.938,0.964,0.964]},{"name":"gamma = 0.1","y":[0.366,0.366,0.902,0.964,0.973,0.955]},{"name":"gamma = 1","y":[0.366,0.366,0.955,0.947,0.946,0.946]},{"name":"gamma = 10","y":[0.366,0.366,0.366,0.928,0.911,0.911]}],"xLabel":"C (log scale)","yLabel":"mean CV accuracy","hlines":[{"y":0.37,"label":"share of the largest class"}]} },
      { type: 'text', title: "Search spaces that are not grids", md: `Some settings only make sense together: an SVM with \`kernel="linear"\` has no gamma. Instead of one grid, pass a **list of grids**; each is searched separately:

\`\`\`python
param_grid = [
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
]
search = GridSearchCV(SVC(), param_grid, cv=5)
\`\`\`

When the grid would be huge, sample it instead: \`RandomizedSearchCV\` tries a fixed number of random combinations, drawn from lists or from distributions such as \`scipy.stats.loguniform(1e-3, 1e3)\`. For very expensive models, \`HalvingGridSearchCV\` (successive halving) gives every candidate a little data, keeps the best fraction, and repeats with more data.` },
      { type: 'code', title: "Nested cross-validation: evaluating the tuning procedure itself", timeoutSec: 90, md: `A single train/test split gives one noisy number for "a tuned SVM". **Nested cross-validation** wraps the whole grid search in an outer cross-validation: each outer fold runs a complete inner search on its training part and is scored on its held-out part.`, code: `from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), X, y, cv=5)   # 5 outer x 36 x 5 inner fits
print("outer-fold accuracies:", scores.round(3))
print(f"mean {scores.mean():.3f} +/- {scores.std():.3f}")`, after: `Nested cross-validation does not produce a model; it answers "how well does *this whole procedure* (an SVM tuned by this grid search) generalize?" It is expensive (here 900 fits), so it is mostly used to compare procedures or to report a trustworthy number on small datasets.` },
      { type: 'check', question: { type: 'tf', q: 'It is fine to try 50 hyperparameter settings, pick the one with the best test-set score, and report that score.', answer: false, explain: 'Choosing by test score makes that score optimistically biased: the test set has effectively been used for training decisions.' } },
      { type: 'text', title: 'Going further: choosing the right split', md: `- **Stratified k-fold** keeps the class proportions in every fold: the default for classification.
- **Group k-fold**: when examples come in groups (several scans of one patient, many sessions of one user) keep each group inside a single fold, or the model is tested on people it has already seen.
- **Time-ordered splits**: train on the past and validate on the future (see *Time Series & Forecasting*).
- **Nested cross-validation**: an inner loop tunes the hyperparameters, an outer loop measures how well the whole tuning procedure generalizes.
- **Repeated k-fold** averages several random splits to steady the estimate on small datasets; leave-one-out (k = n) is costly and noisy, so 5 or 10 folds are the usual compromise.
- **Report the spread**: give the mean and the standard deviation across folds (two models whose difference is smaller than that spread are not reliably different).` }
    ],
    when: `- **k-fold CV** for small to medium datasets and fast models; **single validation split** for large datasets or expensive models (deep learning).
- **Stratified** folds for classification, **group** folds for related examples, **time-series** splits for temporal data.
- **Random search** (or smarter tuners) when there are more than two or three hyperparameters.`,
    pitfalls: `- Preprocessing (scaling, imputation, feature selection) fitted on all data before CV: must happen inside each fold (use pipelines).
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
      { type: 'tf', q: 'When scaling features, the scaler should be fitted inside each cross-validation fold on that fold\'s training part.', answer: true, explain: 'Otherwise the validation fold\'s statistics leak into training.' },
      {"type":"mcq","q":"You tried 50 hyperparameter combinations and picked the one with the best validation score. Why is that score likely to be too optimistic?","options":["Validation sets are always too small","The winner was selected partly because it happened to suit this validation data","Cross-validation was not used","The model was retrained"],"answer":1,"explain":"Selecting the maximum of many noisy scores favours lucky combinations. Only data that played no part in any choice, the test set, gives an unbiased estimate."},
      {"type":"mcq","q":"A grid search's best value of C is the largest value in the grid. What should you do next?","options":["Nothing: the grid found the optimum","Extend the grid to larger values of C","Remove C from the search","Evaluate on the test set and pick a new C"],"answer":1,"explain":"A best value on the edge of the grid suggests the optimum may lie beyond it, so extend the search in that direction."},
      {"type":"mcq","q":"GridSearchCV with a grid of 4 × 5 values and cv=10 trains how many models during the search (ignoring the final refit)?","options":["20","45","200","2,000"],"answer":2,"explain":"20 combinations, each cross-validated with 10 folds: 200 fits (plus one final refit on all the training data)."}
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
        explain: `Using the same folds for every candidate makes the comparison fair: differences come from λ, not from which examples landed where. This loop is exactly what \`GridSearchCV\` automates (plus parallelism and refitting the winner on all data).`,
        tests: [
          { name: 'noisy, many features: some regularization wins', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(40, 15))\ny = X[:, 0] - X[:, 1] + rng.normal(0, 1.0, 40)', call: 'best_lambda(X, y, [0.0, 1.0, 10.0, 100.0, 1000.0])' },
          { name: 'clean data: little regularization wins', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(100, 3))\ny = X @ np.array([3.0, -2.0, 1.0]) + rng.normal(0, 0.05, 100)', call: 'best_lambda(X, y, [0.0, 10.0, 100.0])', expect: '0.0' },
          { name: 'another dataset', setup: 'rng = np.random.default_rng(7)\nX = rng.normal(size=(60, 8))\ny = X @ rng.normal(size=8) * 0.3 + rng.normal(0, 1.5, 60)', call: 'best_lambda(X, y, [0.01, 0.1, 1.0, 10.0, 100.0], k=4, seed=3)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn, Cross-validation: evaluating estimator performance', url: 'https://scikit-learn.org/stable/modules/cross_validation.html' },
      { title: 'Bergstra & Bengio, Random Search for Hyper-Parameter Optimization (2012)', url: 'https://www.jmlr.org/papers/v13/bergstra12a.html' },
      { title: "Andreas Müller & Sarah Guido, Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
    ]
  },

  // =====================================================================================
  {
    id: 'knn', branch: 'ml', title: 'k-Nearest Neighbours', short: 'k-Nearest Neighbours', level: 1, minutes: 30, prereqs: ['supervised-learning', 'numpy'],
    tags: ['knn', 'distance', 'lazy learning', 'curse of dimensionality', 'nearest neighbor search'],
    summary: 'Predict by looking at the most similar training examples: distances, choosing k, feature scaling, and the curse of dimensionality.',
    why: `k-nearest neighbours (kNN) is the most intuitive learning algorithm there is ("you are like your closest neighbours") and it needs no training at all. It is a strong baseline, works for classification and regression, and makes the importance of **distance** and **feature scaling** impossible to ignore. The same nearest-neighbour search, scaled to billions of vectors, powers semantic search, recommendation engines and retrieval-augmented generation (RAG) for LLMs.`,
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
- **Euclidean** √Σ(aᵢ − bᵢ)²: the default.
- **Manhattan** Σ|aᵢ − bᵢ|: city blocks; more robust to single large differences.
- **Cosine distance** 1 − cos_sim, for text and embedding vectors where direction matters.` },
      { type: 'demo', title: "Step through: how k-NN labels a new point", demo: 'walk-knn' },
      { type: 'demo', title: 'Interactive: choosing k', md: `Move the slider: small k gives jagged boundaries that chase individual noisy points (low bias, high variance); large k gives smooth boundaries (higher bias, lower variance). Click to add points.`, demo: 'knn' },
      { type: 'text', title: 'Scaling is not optional', md: `Distances add up differences across features. If one feature is "income in euros" (differences of thousands) and another is "number of children" (differences of 1–3), the income completely dominates: the model effectively ignores the children feature.

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
      { type: 'chart', title: "Training and test accuracy as k grows", md: `The breast-cancer data again, with k from 1 to 150. Read the chart from left to right: from a very complex model to a very simple one.`, after: `On the left, k = 1 memorizes the training set (accuracy 1.0) but generalizes less well. As k grows the two curves approach each other; past the best k, both fall together because the model has become too simple. The gap between the curves is the overfitting.`, chart: {"kind":"line","x":[1,2,3,5,7,10,15,20,30,50,75,100,150],"xLog":true,"series":[{"name":"training","y":[1,0.981,0.988,0.981,0.977,0.972,0.967,0.955,0.953,0.955,0.944,0.932,0.915]},{"name":"test","y":[0.944,0.923,0.944,0.958,0.958,0.958,0.972,0.965,0.958,0.944,0.937,0.93,0.902]}],"xLabel":"k (number of neighbours, log scale)","yLabel":"accuracy","vlines":[{"x":15,"label":"best test k = 15"}]} },
      { type: 'code', title: "Model complexity in k-NN: training versus test accuracy", md: `How does the number of neighbours change the model? Here k-NN is trained on the breast-cancer data with k from 1 to 15, and accuracy is measured on the training set and on a held-out test set.`, code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
print(f"X: {X.shape[0]} tumours x {X.shape[1]} measurements. The first 3 rows (first 5 columns) and labels (0 = malignant, 1 = benign):")
for row, label in zip(X[:3], y[:3]):
    print("  ", row[:5].round(2).tolist(), "->", label)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)
scaler = StandardScaler().fit(X_train)            # learn the scaling on the training data only
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

for k in [1, 2, 3, 5, 10, 20, 50, 100]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print(f"k={k:3d}   train {knn.score(X_train, y_train):.3f}   test {knn.score(X_test, y_test):.3f}")`, after: `With **k = 1**, every training point is its own nearest neighbour, so the training accuracy is perfect, but the decision boundary follows every noisy point and the test accuracy is lower. As k grows, the boundary becomes smoother: training accuracy falls, while test accuracy first rises and then levels off or falls once the model becomes too simple. This trade-off between fitting the training data and generalizing appears in every model family; for k-NN, **fewer neighbours means a more complex model**.`, predict: {"type":"mcq","q":"With k = 1, what accuracy will k-NN score on its own training data?","options":["About 90%","About 95%","Exactly 100%","It depends on the random split"],"answer":2,"explain":"Each training point's nearest neighbour is itself (at distance 0), so with k = 1 every training prediction copies the point's own label: 100%. This says nothing about new data, which is why the test column matters.","expect":["k=  1   train 1.000"]} },
      { type: 'code', title: "k-NN for regression", md: `For a numeric target, k-NN predicts the **average** target of the k nearest training points. Here the target is a noisy wave, y = sin(4x) plus noise, and the score is **R²**: 1 is perfect, 0 is no better than always predicting the mean.`, code: `import numpy as np
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, size=(60, 1))
y = np.sin(4 * x[:, 0]) + x[:, 0] + rng.normal(0, 0.4, 60)
x_train, y_train, x_test, y_test = x[:40], y[:40], x[40:], y[40:]

for k in [1, 3, 9, 30]:
    reg = KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train)
    print(f"k={k:2d}   train R2 {reg.score(x_train, y_train):5.2f}   test R2 {reg.score(x_test, y_test):5.2f}")
print("prediction at x = 0.5 with k = 3:", round(float(KNeighborsRegressor(n_neighbors=3).fit(x_train, y_train).predict([[0.5]])[0]), 2))`, after: `k = 1 reproduces every training point exactly (train R² = 1), including the noise, so its predictions jump around. k = 30 averages most of the training set, which flattens the curve until it underfits. A moderate k follows the wave while smoothing out the noise.` },
      { type: 'text', title: "Strengths, weaknesses and parameters", md: `k-NN has two important settings:

- **The number of neighbours, k.** Small values (1–3) give jagged boundaries that follow the noise; large values give smooth boundaries that eventually ignore real structure. Values from 3 to 10 work well surprisingly often; choose with cross-validation.
- **The distance measure.** Euclidean distance (straight-line distance) is the default and works when all features are on comparable scales. That is why scaling matters so much: an unscaled feature measured in thousands drowns out features measured in units.

| Strengths | Weaknesses |
| --- | --- |
| very easy to understand and explain ("these five similar cases decided") | predictions are slow on big training sets, since every prediction compares against all stored points |
| no training at all: adding data is instant | memory grows with the data: the model *is* the data |
| decent accuracy without much tuning: a good baseline | struggles with many features (hundreds or more), where all points become roughly equally far apart |
| works for classification and regression, with any number of classes | poor on sparse data (mostly zeros), such as word counts |

In practice k-NN is mostly used as a quick, honest baseline and as a building block (for example in recommendation and in [vector search](#/concept/vector-search)), rather than as the final model on large tabular datasets.` },
      { type: 'check', question: { type: 'mcq', q: 'In kNN, what happens as k becomes very large (close to the number of training examples)?', options: ['The model overfits more', 'Predictions approach the overall majority class / mean (underfitting)', 'Training becomes slower', 'Distances become more accurate'], answer: 1, explain: 'With k ≈ n every prediction averages almost the whole dataset, ignoring the local structure: high bias.' } },
      { type: 'text', title: 'The curse of dimensionality and the cost of prediction', md: `In high dimensions, intuition about "nearby" breaks down. With hundreds of features, data points become sparse and **distances concentrate**: the nearest and farthest neighbours are almost equally far away, so "nearest" carries little information. To keep the same density of points, the data needed grows exponentially with the number of dimensions. Remedies: feature selection, dimensionality reduction (PCA), or learned representations (embeddings) where distances are meaningful.

**Cost:** prediction compares the query with every training example, O(n·d) per query. For millions of examples, systems use **approximate nearest-neighbour** indexes (HNSW graphs, IVF, product quantization) that trade a little accuracy for enormous speed. Those indexes are what "vector databases" are made of, and they are a core part of RAG systems.` },
      { type: 'check', question: { type: 'tf', q: 'kNN has an expensive training phase and a cheap prediction phase.', answer: false, explain: 'The reverse: training just stores the data; prediction must search it.' } },
      { type: 'text', title: 'Going further: weights, regression and speed', md: `- **Distance-weighted votes**: closer neighbours count more (for example with weight 1/distance), which makes the result less sensitive to the exact choice of k.
- **kNN regression**: predict the average (or the distance-weighted average) of the neighbours' target values.
- **Choosing k and the distance** by cross-validation: a small k gives jagged, high-variance boundaries; a large k smooths them but blurs real structure. An odd k avoids ties in two-class problems. Euclidean distance suits scaled numeric features, cosine similarity suits text and embeddings.
- **Speed**: brute force costs about n · d operations per query. KD-trees and ball trees speed this up in low dimensions; for embeddings with hundreds of dimensions, approximate nearest-neighbour indexes take over (see *Vector Search*).
- **The idea lives on**: retrieval in RAG, "customers like you" recommendations and few-shot classification on embeddings are all nearest-neighbour methods.` }
    ],
    when: `- **Good for**: small to medium datasets with a meaningful distance (after scaling), low-to-moderate dimensionality, quick baselines, and problems where similar inputs truly have similar outputs.
- **Similarity search** over learned embeddings (images, text, products): kNN is the retrieval step of RAG and recommendations.
- **Avoid** for very high-dimensional raw features, huge datasets without an ANN index, or when you need fast predictions on modest hardware.`,
    pitfalls: `- Forgetting to scale features.
- Even k for binary classification causes ties: prefer odd k (or weight votes by distance).
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
      { type: 'mcq', q: 'kNN with k = 1 achieves 100% training accuracy. Why is that not impressive?', options: ['Because k = 1 is always optimal', 'Each training point\'s nearest neighbour is itself: it memorised the data', 'Because accuracy is the wrong metric', 'Because distances were not scaled'], answer: 1, explain: 'Evaluate on held-out data; 1-NN memorises and is very sensitive to noise.' },
      { type: 'mcq', q: 'Why prefer an odd k for binary classification?', options: ['It is faster', 'It avoids tied votes', 'It reduces the dimensionality', 'Even k is not allowed'], answer: 1, explain: 'With two classes and odd k, the vote cannot tie.' },
      { type: 'tf', q: 'For kNN regression, the prediction is typically the average target of the k nearest neighbours.', answer: true, explain: 'Optionally weighted by inverse distance.' },
      { type: 'mcq', q: 'What is the main problem with kNN on 1,000 raw features?', options: ['It cannot compute distances in 1,000 dimensions', 'Distances become nearly equal, so neighbours carry little information', 'It needs a GPU', 'It always underfits'], answer: 1, explain: 'This is the curse of dimensionality: distances concentrate.' },
      { type: 'mcq', q: 'Which technology makes nearest-neighbour search fast over billions of embedding vectors?', options: ['Gradient boosting', 'Approximate nearest-neighbour indexes (e.g. HNSW) in vector databases', 'One-hot encoding', 'Dropout'], answer: 1, explain: 'ANN indexes trade a little accuracy for orders-of-magnitude speedups.' },
      {"type":"mcq","q":"In k-NN, which choice gives the **most complex** model (the most jagged decision boundary)?","options":["k = 1","k = 10","k equal to the number of training points","It does not depend on k"],"answer":0,"explain":"With one neighbour the boundary bends around every single training point. Larger k averages over more points and smooths the boundary; with k = n every input gets the majority class."},
      {"type":"mcq","q":"How does k-NN regression predict a value for a new input?","options":["It fits a line through the k nearest points and evaluates it","It averages the target values of the k nearest training points","It returns the target of the farthest training point","It predicts the overall mean of the training targets"],"answer":1,"explain":"k-NN regression averages the targets of the k nearest neighbours (optionally weighted by distance)."}
    ],
    problems: [
      {
        id: 'knn-classify', title: 'k-NN Classifier', difficulty: 'medium',
        prompt: `Implement \`knn_predict(X_train, y_train, X_test, k)\`:

For each test row:
1. compute the Euclidean distance to every training row;
2. take the indices of the k smallest distances: use \`np.argsort(d, kind="stable")[:k]\` so equal distances keep training order;
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
        explain: `The loop handles one query at a time: compute distances, sort, vote. Sorted labels from \`np.unique\` plus \`argmax\` (first maximum) gives deterministic tie-breaking. A fully vectorized version computes the whole (n_test, n_train) distance matrix at once, as in the lesson: faster, but memory grows with n_test × n_train.`,
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
      { title: 'scikit-learn, Nearest Neighbors', url: 'https://scikit-learn.org/stable/modules/neighbors.html' },
      { title: "Stanford CS231n, Image classification and nearest neighbours", url: "https://cs231n.github.io/classification/" },
      { title: "Andreas Müller & Sarah Guido, Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
    ]
  },

  // =====================================================================================
  {
    id: 'decision-trees', branch: 'ml', title: 'Decision Trees', level: 2, minutes: 40, prereqs: ['supervised-learning', 'information-theory'],
    tags: ['tree', 'gini', 'entropy', 'information gain', 'cart', 'pruning'],
    summary: 'A flowchart of yes/no questions learned from data: how trees choose splits with Gini impurity or entropy, how they overfit, and why they are the foundation of the strongest tabular models.',
    why: `Decision trees are among the most interpretable models (you can print one and follow its reasoning) and they handle mixed feature types, non-linear interactions and unscaled data with no fuss. On their own they overfit easily, but combined into **random forests** and **gradient-boosted trees** (XGBoost, LightGBM) they are still the state of the art for spreadsheet-style tabular data, often beating deep learning there. Understanding how a single tree picks its splits is the key to all of them.`,
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
      { type: 'text', title: 'Measuring impurity', md: `A good split produces children that are **purer** (more dominated by a single class) than the parent. Two standard impurity measures for a node whose class proportions are p₁, …, p_K:

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
      { type: 'chart', title: "Impurity of a two-class node", after: `All three measures are 0 for a pure node (p = 0 or 1) and largest for a 50/50 node. Entropy and Gini are smooth and strictly curved, which rewards splits that make children purer even when the majority class does not change; the misclassification rate is not, which is why trees are grown with Gini or entropy.`, chart: {"kind":"line","x":[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1],"points":false,"series":[{"name":"entropy (bits)","y":[0,0.081,0.141,0.194,0.242,0.286,0.327,0.366,0.402,0.436,0.469,0.5,0.529,0.557,0.584,0.61,0.634,0.658,0.68,0.701,0.722,0.741,0.76,0.778,0.795,0.811,0.827,0.841,0.855,0.869,0.881,0.893,0.904,0.915,0.925,0.934,0.943,0.951,0.958,0.965,0.971,0.977,0.981,0.986,0.99,0.993,0.995,0.997,0.999,1,1,1,0.999,0.997,0.995,0.993,0.99,0.986,0.981,0.977,0.971,0.965,0.958,0.951,0.943,0.934,0.925,0.915,0.904,0.893,0.881,0.869,0.855,0.841,0.827,0.811,0.795,0.778,0.76,0.741,0.722,0.701,0.68,0.658,0.634,0.61,0.584,0.557,0.529,0.5,0.469,0.436,0.402,0.366,0.327,0.286,0.242,0.194,0.141,0.081,0]},{"name":"Gini","y":[0,0.02,0.039,0.058,0.077,0.095,0.113,0.13,0.147,0.164,0.18,0.196,0.211,0.226,0.241,0.255,0.269,0.282,0.295,0.308,0.32,0.332,0.343,0.354,0.365,0.375,0.385,0.394,0.403,0.412,0.42,0.428,0.435,0.442,0.449,0.455,0.461,0.466,0.471,0.476,0.48,0.484,0.487,0.49,0.493,0.495,0.497,0.498,0.499,0.5,0.5,0.5,0.499,0.498,0.497,0.495,0.493,0.49,0.487,0.484,0.48,0.476,0.471,0.466,0.461,0.455,0.449,0.442,0.435,0.428,0.42,0.412,0.403,0.394,0.385,0.375,0.365,0.354,0.343,0.332,0.32,0.308,0.295,0.282,0.269,0.255,0.241,0.226,0.211,0.196,0.18,0.164,0.147,0.13,0.113,0.095,0.077,0.058,0.039,0.02,0]},{"name":"misclassification rate","y":[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.49,0.48,0.47,0.46,0.45,0.44,0.43,0.42,0.41,0.4,0.39,0.38,0.37,0.36,0.35,0.34,0.33,0.32,0.31,0.3,0.29,0.28,0.27,0.26,0.25,0.24,0.23,0.22,0.21,0.2,0.19,0.18,0.17,0.16,0.15,0.14,0.13,0.12,0.11,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0]}],"xLabel":"share p of the node's examples in class 1","yLabel":"impurity"} },
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
      { type: 'code', title: "Training and reading a tree", md: `A depth-2 tree for the iris flowers, using only the petal measurements. scikit-learn can print the tree as text (in a notebook you can also draw it with \`sklearn.tree.plot_tree\` or Graphviz). For each leaf, \`weights\` shows how many training flowers of each species ended up there.`, code: `from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X = iris.data[:, 2:]                     # petal length and petal width only
y = iris.target
print("petal length and width (cm) of one flower of each species:")
for row, label in zip(X[::50], y[::50]):
    print("  ", row.tolist(), "->", iris.target_names[label])
tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
print(export_text(tree, feature_names=["petal length", "petal width"], show_weights=True))

flower = [[5.0, 1.5]]                    # petals 5 cm long and 1.5 cm wide
p = tree.predict_proba(flower)[0]
print(f"P(setosa) = {p[0]:.3f}   P(versicolor) = {p[1]:.3f}   P(virginica) = {p[2]:.3f}")
print("prediction:", iris.target_names[tree.predict(flower)][0])
print("feature importances:", dict(zip(["petal length", "petal width"], tree.feature_importances_.round(3))))`, after: `Reading the printout from the top:

- **Root:** is the petal length at most 2.45 cm? All 50 setosas go left, into a **pure** leaf (Gini 0).
- **Right child:** is the petal width at most 1.75 cm? The left leaf holds 49 versicolors and 5 virginicas; the right leaf holds 1 versicolor and 45 virginicas.
- **Probabilities** come from the leaf's class proportions: our flower lands in the 49/5 leaf, so P(versicolor) = 49/54 ≈ 0.907. Every point in the same leaf gets exactly the same probabilities.

This transparency is why trees are called **white-box** models: you can trace, and explain, every decision.`, predict: {"type":"mcq","q":"The flower lands in a leaf that holds 49 versicolor and 5 virginica training flowers. What probability of versicolor will predict_proba report?","options":["1.0, because the leaf's majority is versicolor","About 0.91 (49 out of 54)","About 0.5","It cannot give probabilities"],"answer":1,"explain":"A tree's probability estimate is the share of each class among the training examples in the leaf: 49 / 54 ≈ 0.907.","expect":["P(versicolor) = 0.907"]} },
      { type: 'text', title: "The CART training algorithm", md: `scikit-learn grows trees with **CART** (classification and regression trees). At every node it searches all features k and all thresholds t_k, and picks the split with the lowest weighted impurity of the two children:

$$ J(k, t_k) = (m_left / m) · G_left + (m_right / m) · G_right $$
:::where
J(k, t_k) — the cost of splitting on feature k at threshold t_k; CART picks the pair with the smallest J
m — the number of training examples in the node being split
m_left, m_right — how many of them go to the left and right child
G_left, G_right — the impurity (Gini or entropy) of each child
:::

It then repeats the search inside each child, and stops when it reaches \`max_depth\`, when a node is pure, or when no split reduces the impurity (or another stopping rule applies).

CART is **greedy**: each split is the best one *now*, without looking ahead at whether a worse split now would allow much better splits later. Finding the truly optimal tree is computationally intractable (the problem is NP-complete), so every practical library uses a greedy search, which works well enough.

**Cost.** A prediction walks one path from the root to a leaf. A reasonably balanced tree on m examples has a depth of about log₂(m), so predictions are very fast: about 20 comparisons for a million training examples. Training compares every feature on every example at each level, roughly n · m · log₂(m) work for n features: fine for hundreds of thousands of rows.` },
      { type: 'demo', title: "Step through: how a tree chooses its splits", demo: 'walk-tree-split' },
      { type: 'text', title: "Gini impurity or entropy?", md: `Both measure how mixed a node is, and most of the time they lead to very similar trees. Gini is a little faster to compute, which is why it is scikit-learn's default. When they differ, Gini tends to isolate the most frequent class in its own branch, while entropy tends to produce slightly more balanced trees. Choose with cross-validation if you care, but it rarely matters much.` },
      { type: 'text', title: "Regularization hyperparameters", md: `An unrestricted tree keeps splitting until every leaf is pure, which means memorizing the training data: it is a **non-parametric** model, whose structure grows with the data instead of being fixed in advance. To stop it overfitting, restrict its growth (pre-pruning) or cut it back afterwards (post-pruning):

| Setting | Effect | Regularize by… |
| --- | --- | --- |
| \`max_depth\` | maximum number of questions on any path | lowering it |
| \`min_samples_split\` | a node needs at least this many examples to be split | raising it |
| \`min_samples_leaf\` | every leaf must keep at least this many examples | raising it |
| \`min_weight_fraction_leaf\` | the same, as a fraction of the (weighted) data | raising it |
| \`max_leaf_nodes\` | maximum number of leaves | lowering it |
| \`max_features\` | how many features are tried at each split | lowering it (adds randomness) |
| \`ccp_alpha\` | cost-complexity *post*-pruning: remove branches that add little | raising it |` },
      { type: 'code', title: "Regularizing a tree on noisy data", md: `Two interleaving half-moons with a lot of noise. The unrestricted tree is compared with several regularized versions; \`get_n_leaves()\` shows how big each tree became.`, code: `from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
print(f"X: {X.shape[0]} points with 2 coordinates; y says which moon (0 or 1). The first 3:")
for row, label in zip(X[:3], y[:3]):
    print("  ", row.round(2).tolist(), "->", label)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
settings = [{}, {"max_depth": 4}, {"min_samples_leaf": 5}, {"max_leaf_nodes": 8}, {"ccp_alpha": 0.01}]
for kw in settings:
    tree = DecisionTreeClassifier(random_state=0, **kw).fit(X_train, y_train)
    name = ", ".join(f"{k}={v}" for k, v in kw.items()) or "no limits"
    print(f"{name:<22} leaves {tree.get_n_leaves():3d}   train {tree.score(X_train, y_train):.3f}   test {tree.score(X_test, y_test):.3f}")`, after: `The unrestricted tree is perfect on the training data but among the worst on new data: its many leaves include little islands around single noisy points. Limiting the depth alone does not help on this data, while requiring larger leaves, capping the number of leaves or pruning raises the test score from 0.83 to about 0.91–0.92 with far fewer leaves. Which setting works best varies from dataset to dataset, so tune it with cross-validation.` },
      { type: 'code', title: "Regression trees, and why they cannot extrapolate", md: `For regression, each leaf predicts the **mean** target of its training examples, and CART chooses splits that minimise the weighted **mean squared error** of the children instead of the impurity. The result is a step function. Here a quantity grows exponentially (think of prices over time); a tree and a linear model on the logarithm are trained on the first 70% of the time range and asked to predict beyond it.`, code: `import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 10, 80))
y = 50 * np.exp(0.25 * x) + rng.normal(0, 20, 80)
past = x < 7                                           # train on the past only
tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(x[past, None], np.log(y[past]))
line = LinearRegression().fit(x[past, None], np.log(y[past]))

print("   x   true value   tree   linear model")
for q in [6.5, 8.0, 9.5]:
    t, l = np.exp(tree.predict([[q]])[0]), np.exp(line.predict([[q]])[0])
    print(f"{q:4.1f}   {50 * np.exp(0.25 * q):10.0f} {t:6.0f}   {l:12.0f}")
print("tree prediction is flat beyond the training data:", np.exp(tree.predict([[8.0]]))[0] == np.exp(tree.predict([[9.5]]))[0])`, after: `Inside the training range both models are close to the truth. Beyond it, the tree repeats the value of its last leaf forever: **trees (and forests, and boosted trees) cannot extrapolate** outside the range of their training data. For trends that continue (prices, growth, time), use a model that can, or model the trend separately.` },
      { type: 'text', title: "Feature importance", md: `\`feature_importances_\` summarises how much each feature reduced impurity across all the splits that used it, normalized so the importances add up to 1. It is a quick overview, with caveats:

- A feature with importance **0** was not used, which does not mean it is uninformative: another, correlated feature may have been chosen instead.
- Importance says nothing about **direction**: whether a large value pushes towards one class or the other.
- For a more trustworthy picture, average importances over a random forest, or use *permutation importance* (see [Interpretability](#/concept/interpretability)).` },
      { type: 'text', title: "Strengths, weaknesses and instability", md: `| Strengths | Weaknesses |
| --- | --- |
| easy to read, draw and explain | a single tree overfits easily and generalizes worse than most models |
| no scaling or normalization needed: each split looks at one feature at a time | cannot extrapolate beyond the training range |
| handles mixed feature types and makes few assumptions about the data | boundaries are always axis-aligned "staircases" |
| fast to train and very fast to predict | **unstable**: small changes in the data can produce a very different tree |

**Instability** deserves attention. Because every split is axis-aligned, simply rotating the data can turn one clean diagonal boundary into a long staircase (a PCA rotation sometimes helps). And because CART is greedy, removing one training point or changing \`random_state\` can change the very first split and therefore the whole tree. Averaging many trees, as random forests do, removes most of this instability.` },
      { type: 'check', question: { type: 'fill', q: 'A node has 6 examples of class A and 2 of class B. What is its Gini impurity? (decimal)', answer: ['0.375', '.375', '3/8'], explain: '1 − (0.75² + 0.25²) = 1 − (0.5625 + 0.0625) = 0.375.' } },
      { type: 'text', title: 'Growing and pruning', md: `Trees are grown **greedily** (the CART algorithm):
1. For every feature and every candidate threshold, compute the weighted impurity of the resulting split.
2. Apply the best split.
3. Recurse on each child until a stopping rule triggers.

Grown without limits, a tree keeps splitting until every leaf is pure: memorising the training data, noise included. Controls (hyperparameters):
- \`max_depth\`: limit the number of questions in a row;
- \`min_samples_leaf\` / \`min_samples_split\`: don't create tiny leaves;
- \`max_leaf_nodes\`;
- **cost-complexity pruning** (\`ccp_alpha\`): grow fully, then cut back branches that don't pay for their complexity.

Even well-tuned single trees are **unstable**: a small change in the data can change the top split and therefore the whole tree. That high variance is exactly what ensembles average away.` },
      { type: 'code', title: 'A tree in scikit-learn', md: `\`export_text\` prints the learned rules: interpretability for free.`, code: `
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
- Avoid a single deep tree when accuracy matters: use an ensemble.`,
    pitfalls: `- Unlimited depth → memorisation (overfitting).
- Instability: small data changes produce very different trees; don't over-interpret one tree's structure.
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
      { type: 'mcq', q: 'What do regression trees use as impurity?', options: ['Gini', 'Variance / MSE of the targets in the node', 'Accuracy', 'Cosine similarity'], answer: 1, explain: 'They split to reduce the spread of target values; leaves predict the mean.' },
      {"type":"mcq","q":"How does CART choose a split at a node?","options":["It picks a random feature and threshold","It tries all features and thresholds and keeps the one whose children have the lowest weighted impurity","It looks several levels ahead to find the globally best tree","It splits on the feature with the largest values"],"answer":1,"explain":"CART greedily minimises the weighted impurity (or MSE for regression) of the two children at each node, without looking ahead."},
      {"type":"mcq","q":"A regression tree is trained on house prices from 2015 to 2022. What will it predict for 2030?","options":["A continuation of the trend","Roughly the value of the leaf for the latest years: no trend beyond the data","Zero","An error"],"answer":1,"explain":"Trees cannot extrapolate: inputs beyond the training range fall into the outermost leaf, whose value is the mean of the most recent training examples."},
      {"type":"tf","q":"Increasing `min_samples_leaf` makes a decision tree less likely to overfit.","answer":true,"explain":"Requiring more examples in every leaf prevents tiny leaves built around single noisy points, which regularizes the tree."}
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
      { title: 'scikit-learn, Decision Trees', url: 'https://scikit-learn.org/stable/modules/tree.html' },
      { title: 'R2D3, A visual introduction to machine learning', url: 'http://www.r2d3.us/visual-intro-to-machine-learning-part-1/' },
      { title: "Andreas Müller & Sarah Guido, Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
    ]
  },

  // =====================================================================================
  {
    id: 'ensembles', branch: 'ml', title: 'Ensembles: Random Forests & Gradient Boosting', short: 'Ensembles & Boosting', level: 2, minutes: 45, prereqs: ['decision-trees', 'overfitting'],
    tags: ['random forest', 'bagging', 'boosting', 'gradient boosting', 'xgboost', 'lightgbm'],
    summary: 'Many weak models beat one strong one: bagging and random forests reduce variance; boosting builds models that fix each other\'s mistakes. The best tools for tabular data.',
    why: `For structured, spreadsheet-like data (customer records, transactions, sensor logs) **gradient-boosted trees** (XGBoost, LightGBM, CatBoost) and **random forests** remain the strongest practical models, regularly winning competitions and powering production systems in finance, retail and logistics. They need little preprocessing, handle non-linear interactions, and are robust. The ideas behind them (averaging to cut variance, and sequentially correcting errors) reappear throughout ML (dropout, model averaging, residual connections).`,
    objectives: [
      'Explain why averaging many models reduces variance',
      'Describe bagging, bootstrap samples, random forests and out-of-bag error',
      'Describe boosting and gradient boosting as fitting residuals step by step',
      'Choose between random forests and gradient boosting and tune their key knobs'
    ],
    sections: [
      { type: 'text', title: 'The wisdom of crowds', md: `Average the guesses of many people at the weight of an ox and the average is often closer than nearly any individual guess, *if* their errors are independent. The same holds for models: averaging B models whose errors are uncorrelated divides the variance by B.

Two ensemble families exploit this differently:
- **Bagging** trains many models **in parallel** on random variations of the data and averages them → reduces **variance**. (Best with low-bias, high-variance models like deep trees.)
- **Boosting** trains models **sequentially**, each focusing on the errors of the ensemble so far → reduces **bias** (and with care, variance).` },
      { type: 'text', title: 'Bagging and random forests', md: `**Bagging** (bootstrap aggregating):
1. Draw a **bootstrap sample**: n examples sampled *with replacement* from the n training examples (about 63% of distinct examples appear; some repeat).
2. Train a model (typically a deep, unpruned tree) on it.
3. Repeat B times; predict by majority vote (classification) or averaging (regression).

**Random forests** add one twist: at each split, a tree may only consider a **random subset of features** (e.g. √d of them). This *decorrelates* the trees (otherwise every tree would split on the same dominant feature first) and decorrelated errors average out much better.

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
   - compute the **residuals** rᵢ = yᵢ − F_{m−1}(xᵢ): what is still unexplained;
   - fit a small tree h_m (depth 1–6) to predict those residuals;
   - update F_m(x) = F_{m−1}(x) + η · h_m(x), with a small **learning rate** η (e.g. 0.05–0.1).

Why "gradient"? The residuals are exactly the negative gradient of the squared-error loss with respect to the predictions. Gradient boosting is **gradient descent in function space**: each tree is a step downhill, and other losses (log loss for classification, quantile loss, ranking losses) simply use their own gradients as targets.

Key hyperparameters: number of trees M, learning rate η (smaller needs more trees but generalizes better), tree depth, subsampling of rows/columns, and **early stopping** on validation loss. Modern libraries (XGBoost, LightGBM, CatBoost) add regularization, clever split finding and native categorical handling.` },
      { type: 'demo', title: "Step through: gradient boosting, tree by tree", demo: 'walk-boosting' },
      { type: 'check', question: { type: 'mcq', q: 'In gradient boosting with squared-error loss, what does each new tree try to predict?', options: ['The original targets y', 'The residuals of the current ensemble', 'A random subset of labels', 'The feature importances'], answer: 1, explain: 'Each tree fits what the ensemble still gets wrong: the residuals, i.e. the negative gradient.' } },
      { type: 'code', title: 'Random forest vs gradient boosting in scikit-learn', timeoutSec: 90, code: `
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
print(f"X: {X.shape[0]} tumours x {X.shape[1]} measurements. The first 3 rows (first 5 columns) and labels (0 = malignant, 1 = benign):")
for row, label in zip(X[:3], y[:3]):
    print("  ", row[:5].round(2).tolist(), "->", label)
models = {
    "single tree": DecisionTreeClassifier(random_state=0),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0),
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:<18} accuracy {scores.mean():.3f} ± {scores.std():.3f}")
` },
      { type: 'code', title: "Random forests: how many trees, and which features matter", code: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, random_state=0)
scores = {}
for n in [1, 10, 100, 300]:
    forest = RandomForestClassifier(n_estimators=n, random_state=0).fit(X_train, y_train)
    scores[n] = forest.score(X_test, y_test)
    print(f"{n:3d} trees   train {forest.score(X_train, y_train):.3f}   test {scores[n]:.3f}")
print("100 trees beat 1 tree:", scores[100] > scores[1])

forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
top = np.argsort(forest.feature_importances_)[::-1][:5]
print("five most important features:")
for i in top:
    print(f"  {data.feature_names[i]:<22} {forest.feature_importances_[i]:.3f}")`, after: `A single tree overfits; ten trees already fix most of that, and beyond about a hundred the gains flatten out. Unlike boosting, **adding more trees to a random forest never causes overfitting**: it only costs time and memory. The forest's feature importances are averaged over all its trees, which makes them far more stable than a single tree's.`, predict: {"type":"mcq","q":"A random forest with 1 tree and one with 100 trees are trained on the same data. Which has the higher test accuracy?","options":["1 tree: fewer trees means less overfitting","100 trees","They are always identical","It is impossible to say before training"],"answer":1,"explain":"Averaging many decorrelated trees cancels much of each tree's individual overfitting: 0.881 for one tree against 0.944 for a hundred here.","expect":["100 trees beat 1 tree: True"]} },
      { type: 'chart', title: "Test accuracy against the number of trees", after: `The forest improves quickly and then stays flat: extra trees can only reduce variance. Boosting with a high learning rate improves fast too, but every extra round fits the remaining errors more tightly; on noisier data this curve eventually turns down, which is why boosting needs early stopping or a tuned number of rounds.`, chart: {"kind":"line","x":[1,2,3,5,8,12,20,30,50,80,120,200,300,500],"xLog":true,"series":[{"name":"random forest","y":[0.881,0.902,0.951,0.958,0.958,0.958,0.951,0.944,0.944,0.944,0.944,0.944,0.944,0.944]},{"name":"gradient boosting (learning rate 0.3)","y":[0.916,0.937,0.93,0.93,0.944,0.944,0.951,0.951,0.951,0.951,0.93,0.944,0.944,0.944]}],"xLabel":"number of trees (log scale)","yLabel":"test accuracy"} },
      { type: 'text', title: "Random forests in detail", md: `| Setting | What it does | Guidance |
| --- | --- | --- |
| \`n_estimators\` | number of trees | more is better (with diminishing returns); 100–500 is typical |
| \`max_features\` | features considered at each split | the main source of randomness: \`"sqrt"\` (classification default); smaller values give more diverse trees, which helps against overfitting |
| \`max_depth\`, \`min_samples_leaf\` | size of each tree | usually left unrestricted; limit them to save memory and time |
| \`bootstrap\` | each tree sees a random sample of the rows, drawn with replacement | the other source of randomness; leaving it on also gives a free "out-of-bag" error estimate (\`oob_score=True\`) |
| \`n_jobs\` | number of CPU cores to use | \`-1\` uses all cores; trees are trained independently, so this parallelizes perfectly |
| \`random_state\` | seeds the randomness | fix it for reproducible results |

**Strengths:** among the most widely used models; very accurate with default settings; no scaling needed; robust to irrelevant features; parallel. **Weaknesses:** slower and larger than a linear model; hard to inspect a hundred trees; weaker on very high-dimensional sparse data such as text, where linear models shine.` },
      { type: 'code', title: "Gradient boosting: depth, learning rate and number of trees", timeoutSec: 90, md: `Gradient boosting builds **shallow trees one after another**, each correcting the errors of the ensemble so far. Its main settings interact: \`learning_rate\` scales how much each tree corrects, \`n_estimators\` sets how many trees are added, and \`max_depth\` (often 1 to 5) limits each tree.`, code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
for kw in [{}, {"max_depth": 1}, {"learning_rate": 0.01}, {"n_estimators": 500}]:
    gb = GradientBoostingClassifier(random_state=0, **kw).fit(X_train, y_train)
    name = ", ".join(f"{k}={v}" for k, v in kw.items()) or "defaults (100 trees, depth 3, rate 0.1)"
    print(f"{name:<42} train {gb.score(X_train, y_train):.3f}   test {gb.score(X_test, y_test):.3f}")`, after: `With default settings the ensemble fits the training data perfectly. Making each tree a single split (\`max_depth=1\`) regularizes it and helps here; a smaller learning rate also simplifies the model (it would need more trees to catch up); and **unlike a random forest, adding more boosting rounds can overfit**. A common recipe: fix \`max_depth\` small, set \`n_estimators\` as large as your time budget allows, then tune \`learning_rate\`, or let early stopping (\`n_iter_no_change\`) pick the number of trees.` },
      { type: 'text', title: "Gradient boosting in detail", md: `**Strengths:** frequently the most accurate model on tabular data, and the usual winner of data-science competitions; fast predictions and small models (shallow trees); no scaling needed. **Weaknesses:** needs more careful tuning than a random forest; trees are built one after another, so training is harder to parallelize; like all tree models, poor on very sparse high-dimensional data and cannot extrapolate.

For large datasets use \`HistGradientBoostingClassifier\`, which bins the features first and is often ten or more times faster, or the specialised libraries **XGBoost**, **LightGBM** and **CatBoost**, which follow the same ideas.` },
      { type: 'check', question: { type: 'tf', q: 'Random forests decorrelate their trees by letting each split consider only a random subset of features.', answer: true, explain: 'Feature subsampling (plus bootstrap samples) makes the trees\' errors less correlated, so averaging helps more.' } }
    ],
    when: `- **Tabular data**: start with gradient boosting (LightGBM/XGBoost/CatBoost or scikit-learn's \`HistGradientBoosting\`) and a random forest as a robust baseline.
- **Random forest** when you want strong results with almost no tuning, or parallel training.
- **Gradient boosting** when you want the last few points of accuracy and can tune (learning rate, depth, early stopping).
- **Prefer neural networks** for images, audio, text, and very large unstructured datasets.`,
    pitfalls: `- Boosting with too many trees and no early stopping overfits.
- A high learning rate in boosting needs fewer trees but generalizes worse.
- Interpreting impurity-based feature importance as causal: use permutation importance or SHAP, and still be careful.
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
      { type: 'mcq', q: 'For a dataset of customer records with 40 mixed numeric/categorical columns, a strong first choice is…', options: ['A convolutional neural network', 'Gradient-boosted trees', 'k-means', 'A transformer trained from scratch'], answer: 1, explain: 'Boosted trees dominate tabular data with minimal preprocessing.' },
      {"type":"mcq","q":"Which statement about adding more trees is correct?","options":["More trees always overfit, in both random forests and gradient boosting","More trees never hurt a random forest's accuracy, but more boosting rounds can overfit","More trees always help gradient boosting, but can overfit a random forest","The number of trees has no effect on either"],"answer":1,"explain":"A forest averages independent trees, so extra trees only reduce variance. Boosting keeps fitting the remaining errors, so after a point extra rounds start fitting noise."},
      {"type":"mcq","q":"In a random forest, what is the main purpose of `max_features`?","options":["To limit memory use","To make the trees different from each other by limiting the features tried at each split","To select the final features of the model","To control the learning rate"],"answer":1,"explain":"Restricting the candidate features at each split decorrelates the trees, so averaging them cancels more of their individual errors."}
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
        explain: `Each round fits the current residuals (the negative gradient of the squared error) with the best one-split function, then takes a small step (lr) in that direction. With many rounds, the sum of stumps approximates any 1-D curve as a staircase. Real gradient boosting uses deeper trees (capturing interactions), row/column subsampling and early stopping, but the loop is the same.`,
        tests: [
          { name: 'first stump on a step function', call: 'fit_stump(np.array([1.0, 2.0, 3.0, 4.0]), np.array([-1.0, -1.0, 1.0, 1.0]))', expect: '(2.5, -1.0, 1.0)' },
          { name: 'two rounds, exact', call: 'fit_boosting([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 3.0, 5.0], n_rounds=2, lr=0.5)' },
          { name: 'training error decreases', setup: 'rng = np.random.default_rng(0)\nx = np.sort(rng.uniform(0, 6, 60))\ny = np.sin(x) + rng.normal(0, 0.1, 60)', code: 'errs = []\nfor m in (1, 10, 50):\n    f0, st = fit_boosting(x, y, n_rounds=m, lr=0.3)\n    errs.append(np.mean((predict_boosting(f0, st, x, lr=0.3) - y) ** 2))\nassert errs[0] > errs[1] > errs[2], "training MSE should fall as rounds increase: %s" % errs', hidden: true },
          { name: 'predictions match the reference', setup: 'rng = np.random.default_rng(3)\nx = rng.uniform(0, 10, 40)\ny = 0.5 * x + np.sin(x) + rng.normal(0, 0.2, 40)\nxn = np.linspace(0, 10, 7)', call: 'predict_boosting(*fit_boosting(x, y, n_rounds=20, lr=0.2), xn, lr=0.2)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn, Ensemble methods', url: 'https://scikit-learn.org/stable/modules/ensemble.html' },
      { title: 'XGBoost, Introduction to boosted trees', url: 'https://xgboost.readthedocs.io/en/stable/tutorials/model.html' },
      { title: "Andreas Müller & Sarah Guido, Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
