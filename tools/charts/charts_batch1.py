"""Compute chart data for the deepened classical-ML lessons and write them as a lesson patch."""
import json
import sys
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression, make_moons, load_digits, load_iris
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
warnings.filterwarnings("ignore")
r = lambda v, d=3: [None if v_ is None else round(float(v_), d) for v_ in v]
out = []

def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"]
    if md:
        block.append(md)
    block += ["%% spec", json.dumps(spec)]
    if after:
        block += ["%% after", after]
    out.append("\n".join(block))

# 1. k-NN: accuracy against k
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=66)
s = StandardScaler().fit(Xtr); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
ks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]
tr, te = [], []
for k in ks:
    m = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr); tr.append(m.score(Xtr, ytr)); te.append(m.score(Xte, yte))
best = ks[int(np.argmax(te))]
chart("knn", {"before": "Model complexity in k-NN: training versus test accuracy"}, "Training and test accuracy as k grows",
      {"kind": "line", "x": ks, "xLog": True, "series": [{"name": "training", "y": r(tr)}, {"name": "test", "y": r(te)}],
       "xLabel": "k (number of neighbours, log scale)", "yLabel": "accuracy", "vlines": [{"x": best, "label": f"best test k = {best}"}]},
      md="The breast-cancer data again, with k from 1 to 150. Read the chart from left to right: from a very complex model to a very simple one.",
      after="On the left, k = 1 memorizes the training set (accuracy 1.0) but generalizes less well. As k grows the two curves approach each other; past the best k, both fall together because the model has become too simple. The gap between the curves is the overfitting.")

# 2. Overfitting: error against polynomial degree, and learning curves
rng = np.random.default_rng(1)
x = rng.uniform(0, 1, 40); yv = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 40)
xt = rng.uniform(0, 1, 400); yt = np.sin(2 * np.pi * xt) + rng.normal(0, 0.25, 400)
degs = list(range(0, 13)); etr, ete = [], []
for d in degs:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], yv)
    etr.append(np.mean((m.predict(x[:, None]) - yv) ** 2)); ete.append(np.mean((m.predict(xt[:, None]) - yt) ** 2))
chart("overfitting", {"before": "Bias and variance"}, "Training and test error against model complexity",
      {"kind": "line", "x": degs, "yLog": True, "series": [{"name": "training error", "y": r(etr, 4)}, {"name": "test error", "y": r(ete, 4)}],
       "xLabel": "polynomial degree (model complexity)", "yLabel": "mean squared error (log scale)", "hlines": [{"y": 0.0625, "label": "noise level"}],
       "vlines": [{"x": int(degs[int(np.argmin(ete))]), "label": "sweet spot"}]},
      md="Forty noisy points from a sine wave, fitted with polynomials of increasing degree. This is the most important picture in machine learning:",
      after="Training error only ever goes down as the model gets more flexible. Test error falls while the model is still underfitting (low degrees cannot bend like a sine), bottoms out at the sweet spot, then climbs as the polynomial starts fitting the noise. The dashed line is the noise level (σ² = 0.0625): no model can do better on new data.")
sizes = [10, 15, 20, 30, 45, 70, 100, 150, 250, 400]
lc = {"simple (degree 1) train": [], "simple (degree 1) test": [], "flexible (degree 9) train": [], "flexible (degree 9) test": []}
big_x = rng.uniform(0, 1, 400); big_y = np.sin(2 * np.pi * big_x) + rng.normal(0, 0.25, 400)
for n in sizes:
    for d, name in [(1, "simple (degree 1)"), (9, "flexible (degree 9)")]:
        m = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1e-6)).fit(big_x[:n, None], big_y[:n])
        lc[name + " train"].append(np.mean((m.predict(big_x[:n, None]) - big_y[:n]) ** 2))
        lc[name + " test"].append(np.mean((m.predict(xt[:, None]) - yt) ** 2))
chart("overfitting", {"after": "Learning curves"}, "Learning curves for a simple and a flexible model",
      {"kind": "line", "x": sizes, "xLog": True, "yLog": True, "yMax": 2,
       "series": [{"name": k, "y": r([min(v, 2) for v in vals], 4), "dash": k.endswith("train")} for k, vals in lc.items()],
       "xLabel": "number of training examples (log scale)", "yLabel": "mean squared error (log scale)"},
      after="Dashed lines are training error, solid lines test error. The **simple** model (a straight line) converges quickly to a high error: more data cannot fix high bias. The **flexible** model is terrible with little data (high variance) but keeps improving as data grows, until it beats the simple model by a wide margin. That is the practical meaning of \"more data lets you afford more complex models\".")

# 3. Regularization: test R2 against alpha
X, y = make_regression(n_samples=150, n_features=80, n_informative=10, noise=25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=70, random_state=0)
alphas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
rr, la, used = [], [], []
for a in alphas:
    rr.append(Ridge(alpha=a).fit(Xtr, ytr).score(Xte, yte))
    m = Lasso(alpha=a, max_iter=200000).fit(Xtr, ytr); la.append(m.score(Xte, yte)); used.append(int(np.sum(m.coef_ != 0)))
ols = LinearRegression().fit(Xtr, ytr).score(Xte, yte)
chart("regularization", {"after": "Ridge and lasso in scikit-learn"}, "Test R² against regularization strength",
      {"kind": "line", "x": alphas, "xLog": True, "series": [{"name": "ridge", "y": r(rr)}, {"name": "lasso", "y": r(la)}],
       "xLabel": "alpha (log scale)", "yLabel": "test R²", "hlines": [{"y": round(ols, 3), "label": "least squares"}], "yMin": 0},
      after="Both penalties have a sweet spot. Lasso's is much higher on this data because only 10 of the 80 features matter and it can switch the other 70 off; ridge can only shrink them. Too much of either penalty underfits and the score collapses.")
chart("regularization", {"after": "Test R² against regularization strength"}, "How many features lasso keeps",
      {"kind": "line", "x": alphas, "xLog": True, "series": [{"name": "features with a non-zero weight", "y": used}],
       "xLabel": "alpha (log scale)", "yLabel": "features used (of 80)", "hlines": [{"y": 10, "label": "truly informative features"}], "yMin": 0},
      after="As alpha grows, lasso sets more and more weights exactly to zero: automatic feature selection.")

# 4. Logistic regression: log loss, and accuracy against C
p = np.linspace(0.01, 0.99, 99)
chart("logistic-regression", {"after": "Training with log loss"}, "Log loss punishes confident mistakes",
      {"kind": "line", "x": r(p, 2), "points": False, "series": [{"name": "true label 1:  −log p", "y": r(-np.log(p))}, {"name": "true label 0:  −log(1 − p)", "y": r(-np.log(1 - p))}],
       "xLabel": "predicted probability p of class 1", "yLabel": "loss for one example"},
      after="When the true label is 1, predicting p = 0.9 costs only 0.105, but predicting p = 0.01 costs 4.6, and the loss grows without limit as p approaches 0. A model trained with log loss is therefore pushed hard to avoid confident wrong answers.")
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
s = StandardScaler().fit(Xtr); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
Cs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
tr = [LogisticRegression(C=c, max_iter=10000).fit(Xtr, ytr).score(Xtr, ytr) for c in Cs]
te = [LogisticRegression(C=c, max_iter=10000).fit(Xtr, ytr).score(Xte, yte) for c in Cs]
chart("logistic-regression", {"after": "C in action on the breast-cancer data"}, "Accuracy against C",
      {"kind": "line", "x": Cs, "xLog": True, "series": [{"name": "training", "y": r(tr)}, {"name": "test", "y": r(te)}],
       "xLabel": "C (inverse regularization strength, log scale)", "yLabel": "accuracy"},
      after="Moving right means less regularization. Training accuracy climbs steadily; test accuracy peaks around C = 1 and then falls: the same complexity trade-off as k in k-NN, seen through a different setting.")

# 5. Decision trees: impurity measures
p = np.linspace(0, 1, 101)
ent = np.where((p > 0) & (p < 1), -(p * np.log2(np.clip(p, 1e-12, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1))), 0)
chart("decision-trees", {"after": "Measuring impurity"}, "Impurity of a two-class node",
      {"kind": "line", "x": r(p, 2), "points": False, "series": [{"name": "entropy (bits)", "y": r(ent)}, {"name": "Gini", "y": r(2 * p * (1 - p))}, {"name": "misclassification rate", "y": r(1 - np.maximum(p, 1 - p))}],
       "xLabel": "share p of the node's examples in class 1", "yLabel": "impurity"},
      after="All three measures are 0 for a pure node (p = 0 or 1) and largest for a 50/50 node. Entropy and Gini are smooth and strictly curved, which rewards splits that make children purer even when the majority class does not change; the misclassification rate is not, which is why trees are grown with Gini or entropy.")

# 6. Ensembles: test accuracy against number of trees
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
ns = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500]
rf = [RandomForestClassifier(n_estimators=n, random_state=0).fit(Xtr, ytr).score(Xte, yte) for n in ns]
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.3, random_state=0).fit(Xtr, ytr)
staged = [np.mean(pred == yte) for pred in gb.staged_predict(Xte)]
gbv = [staged[n - 1] for n in ns]
chart("ensembles", {"after": "Random forests: how many trees, and which features matter"}, "Test accuracy against the number of trees",
      {"kind": "line", "x": ns, "xLog": True, "series": [{"name": "random forest", "y": r(rf)}, {"name": "gradient boosting (learning rate 0.3)", "y": r(gbv)}],
       "xLabel": "number of trees (log scale)", "yLabel": "test accuracy"},
      after="The forest improves quickly and then stays flat: extra trees can only reduce variance. Boosting with a high learning rate improves fast too, but every extra round fits the remaining errors more tightly; on noisier data this curve eventually turns down, which is why boosting needs early stopping or a tuned number of rounds.")

# 7. SVM: the losses compared
m = np.linspace(-2, 3, 101)
chart("svm", {"after": "Soft margins and the hinge loss"}, "Three losses as a function of the margin",
      {"kind": "line", "x": r(m, 2), "points": False,
       "series": [{"name": "0–1 loss (wrong or not)", "y": r((m < 0).astype(float))}, {"name": "hinge: max(0, 1 − t·h)", "y": r(np.maximum(0, 1 - m))}, {"name": "log loss (in bits)", "y": r(np.log2(1 + np.exp(-m)))}],
       "xLabel": "t · h(x): the signed score (positive = correct side)", "yLabel": "loss"},
      after="The 0–1 loss counts mistakes but gives no gradient. The hinge loss is zero only for points beyond the margin (t·h ≥ 1) and grows linearly for points inside it or on the wrong side, which is what makes SVM solutions depend only on the support vectors. The log loss of logistic regression is similar but never exactly zero, so every point keeps a small influence.")

# 8. Evaluation metrics: precision and recall against the threshold
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, stratify=d.target, random_state=0)
y5 = ytr == 5
scores = cross_val_predict(SGDClassifier(random_state=42), Xtr, y5, cv=3, method="decision_function")
prec, rec, th = precision_recall_curve(y5, scores)
grid = np.linspace(-300, 300, 25)
pv = [float(prec[min(np.searchsorted(th, g), len(th) - 1)]) for g in grid]
rv = [float(rec[min(np.searchsorted(th, g), len(th) - 1)]) for g in grid]
chart("evaluation-metrics", {"after": "Choosing a threshold for a goal"}, "Precision and recall against the threshold",
      {"kind": "line", "x": r(grid, 0), "points": False, "series": [{"name": "precision", "y": r(pv)}, {"name": "recall", "y": r(rv)}],
       "xLabel": "decision threshold on the classifier's score", "yLabel": "value", "vlines": [{"x": 0, "label": "default threshold 0"}]},
      after="Moving the threshold right makes the classifier more cautious: precision rises towards 1 while recall falls, because fewer real 5s clear the bar. Every threshold is a different trade-off; the default of 0 is just one of them.")
chart("evaluation-metrics", {"after": "Precision and recall against the threshold"}, "The precision–recall curve",
      {"kind": "line", "points": False, "series": [{"name": "linear classifier: is it a 5?", "x": r(rec[::4]), "y": r(prec[::4])}],
       "xLabel": "recall", "yLabel": "precision", "hlines": [{"y": round(float(y5.mean()), 3), "label": "random guessing (share of 5s)"}], "yMin": 0},
      after="The same trade-off as one curve: each point is one threshold. A perfect classifier would sit in the top-right corner. Random guessing gives a precision equal to the share of positives, about 0.10 here, which is why a precision–recall curve is so revealing when positives are rare.")

# 9. MLP: training loss with raw and scaled inputs
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
s = StandardScaler().fit(Xtr)
raw = MLPClassifier(max_iter=200, random_state=42).fit(Xtr, ytr).loss_curve_
sc = MLPClassifier(max_iter=200, random_state=42).fit(s.transform(Xtr), ytr).loss_curve_
n = min(len(raw), len(sc), 200)
ep = list(range(1, n + 1, 4))
chart("mlp", {"after": "Scaling and regularization decide the result"}, "Training loss per epoch, raw versus scaled inputs",
      {"kind": "line", "x": ep, "points": False, "yLog": True, "series": [{"name": "raw features", "y": r([raw[i - 1] for i in ep], 4)}, {"name": "standardized features", "y": r([sc[i - 1] for i in ep], 4)}],
       "xLabel": "epoch (pass over the training data)", "yLabel": "training log loss (log scale)"},
      after="With raw features the loss falls slowly and erratically: a few huge features dominate the gradients. With standardized features it falls smoothly and much lower. Watching this curve is the first thing to do when a network trains badly.")

# 10. Cross-validation: CV accuracy against C for several gammas
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
Cs = [0.001, 0.01, 0.1, 1, 10, 100]
gs = GridSearchCV(SVC(), {"C": Cs, "gamma": [0.01, 0.1, 1, 10]}, cv=5).fit(Xtr, ytr)
res = gs.cv_results_
series = []
for g in [0.01, 0.1, 1, 10]:
    series.append({"name": f"gamma = {g}", "y": r([res["mean_test_score"][i] for i in range(len(res["params"])) if res["params"][i]["gamma"] == g])})
chart("cross-validation", {"after": "Reading the results: is the grid in the right place?"}, "Mean cross-validation accuracy across the grid",
      {"kind": "line", "x": Cs, "xLog": True, "series": series, "xLabel": "C (log scale)", "yLabel": "mean CV accuracy", "hlines": [{"y": 0.37, "label": "share of the largest class"}]},
      after="Each line is one value of gamma. Small C underfits for every gamma; gamma = 10 never gets close to the best because its bells are too narrow for this data. The best region (C between 1 and 100 with gamma between 0.01 and 1) is surrounded by worse values, so the grid covers it well.")

# 11. Supervised learning: the model comparison as bars
names = ["baseline", "k-NN", "logistic", "naive Bayes", "tree", "forest", "boosting", "SVM", "MLP"]
test = [0.629, 0.951, 0.958, 0.923, 0.902, 0.944, 0.958, 0.958, 0.958]
chart("supervised-learning", {"after": "A first look at many model families on one dataset"}, "Test accuracy of each model family",
      {"kind": "bar", "x": names, "series": [{"name": "test accuracy", "y": test}], "xLabel": "model (default settings)", "yLabel": "test accuracy", "yMin": 0.5, "yMax": 1.0},
      after="Every real model clears the baseline easily, and the best ones are within about one test example of each other. Note the value axis starts at 0.5 here to make small differences visible; with bars that is a deliberate exception, flagged so it does not mislead.")

# 12. Pipelines: the leak, as bars
chart("sklearn-workflow", {"after": "The leak you cannot see"}, "Cross-validated R² on pure noise",
      {"kind": "bar", "x": ["features selected before CV (leaky)", "selection inside a pipeline (honest)"], "series": [{"name": "mean cross-validation R²", "y": [0.898, -0.252]}],
       "xLabel": "procedure", "yLabel": "R² (0 = no better than the mean)", "hlines": [{"y": 0, "label": "no skill"}]},
      after="The same data and the same model. Only where the feature selection happens differs, and it turns \"no signal at all\" into an apparently excellent model.")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print(len(out), "charts written")
