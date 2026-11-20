"""Chart data for batch C2. Apply AFTER patches/c2-text.txt."""
import json, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
sig = lambda v, floor=1e-12: [None if x is None else float(f"{max(float(x), floor):.4g}") for x in v]
out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- learning theory: bounds and double descent
ns = [100, 300, 1000, 3000, 10000, 30000, 100000]
delta = 0.05
series = [{"name": "one fixed model (Hoeffding)", "y": r([math.sqrt(math.log(2 / delta) / (2 * n)) for n in ns])}]
for H in (1e3, 1e6):
    series.append({"name": f"best of {int(H):,} models (union bound)", "y": r([math.sqrt((math.log(H) + math.log(2 / delta)) / (2 * n)) for n in ns])})
chart("learning-theory", {"after": "Many models: the union bound and PAC learning"}, "How far can the training error be from the true error?",
      {"kind": "line", "x": ns, "xLog": True, "yLog": True, "series": series, "xLabel": "number of training examples n (log scale)", "yLabel": "guaranteed gap ε (95% confidence)"},
      after="All three bounds shrink like 1 / √n: ten times more data cuts the gap by about 3. Choosing the best of many models costs extra, but only logarithmically: picking from a million models instead of one needs roughly four times as much data for the same guarantee, not a million times.")
rng = np.random.default_rng(0)
n, d = 50, 10
beta = rng.normal(size=d) / np.sqrt(d)
target = lambda X: X @ beta + 0.5 * np.tanh(X[:, 0] * X[:, 1])
X_train, X_test = rng.normal(size=(n, d)), rng.normal(size=(2000, d))
y_train = target(X_train) + 0.5 * rng.normal(size=n); y_test = target(X_test)
W = rng.normal(size=(d, 1000)) / np.sqrt(d)
ps = [5, 10, 15, 20, 25, 30, 35, 40, 44, 47, 49, 50, 51, 53, 56, 60, 70, 80, 100, 150, 200, 300, 500, 1000]
te, tr = [], []
for p in ps:
    Ftr, Fte = np.maximum(0, X_train @ W[:, :p]), np.maximum(0, X_test @ W[:, :p])
    coef = np.linalg.pinv(Ftr) @ y_train
    te.append(np.mean((Fte @ coef - y_test) ** 2)); tr.append(np.mean((Ftr @ coef - y_train) ** 2))
chart("learning-theory", {"after": "Double descent with random features"}, "Double descent: test error against the number of random features",
      {"kind": "line", "x": ps, "xLog": True, "yLog": True, "series": [{"name": "test MSE", "y": sig(te)}, {"name": "training MSE", "y": sig(tr, 1e-6)}],
       "xLabel": "number of random features p (log scale)", "yLabel": "mean squared error (log scale)", "vlines": [{"x": 50, "label": "p = n = 50"}], "yMin": 1e-6},
      md="The experiment above, with more feature counts:",
      after=f"Left of the line, more features lower the training error and the test error first improves, then worsens: the classical U. At p = n the training error hits zero (it is plotted at the floor of the chart from there on) and the test error peaks at about {max(te):.0f}. Beyond it the test error comes down again as the minimum-norm fit gets smoother: the second descent.")

# ---------------- calibration: reliability diagram for the lesson's model, before and after temperature scaling
rng = np.random.default_rng(0)
n, k = 6000, 4
true_logits = rng.normal(size=(n, k)) * 1.5
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
p_true = softmax(true_logits)
labels = (rng.random(n)[:, None] > np.cumsum(p_true, axis=1)).sum(axis=1)
model_logits = 2.5 * true_logits
def nll(T): P = softmax(model_logits / T); return -np.mean(np.log(P[np.arange(n), labels]))
Ts = np.linspace(0.5, 5, 451); T = float(Ts[np.argmin([nll(t) for t in Ts])])
def rel(P, bins=10):
    conf, correct = P.max(axis=1), P.argmax(axis=1) == labels
    idx = np.minimum((conf * bins).astype(int), bins - 1); xs, ys = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20: xs.append(conf[m].mean()); ys.append(correct[m].mean())
    return xs, ys
x1, y1 = rel(softmax(model_logits)); x2, y2 = rel(softmax(model_logits / T))
chart("calibration-uncertainty", {"after": "Temperature scaling fixes it with one number"}, "Reliability diagram: before and after temperature scaling",
      {"kind": "line", "series": [
          {"name": "perfect calibration", "x": [0.25, 1], "y": [0.25, 1], "dash": True},
          {"name": "original model", "x": r(x1, 3), "y": r(y1, 3)},
          {"name": f"after temperature scaling (T = {T:.2f})", "x": r(x2, 3), "y": r(y2, 3)}],
       "xLabel": "mean confidence in the bin", "yLabel": "accuracy in the bin", "xMin": 0.25, "xMax": 1, "yMin": 0.25, "yMax": 1},
      md="The 4-class model from the reliability table above, binned by confidence:",
      after=f"The original model's curve sags far below the diagonal: in every bin it is more confident than it is accurate. Dividing all logits by a single temperature T = {T:.2f} (fitted by minimising the log loss) brings every bin onto the diagonal, without changing a single prediction.")

# ---------------- data-centric: label noise
X, yy = make_classification(3000, n_features=10, n_informative=5, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=1000, random_state=0)
fracs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
models = [("1-nearest neighbour", lambda: KNeighborsClassifier(1)), ("25-nearest neighbours", lambda: KNeighborsClassifier(25)), ("logistic regression", lambda: LogisticRegression(max_iter=1000))]
acc = {m[0]: [] for m in models}
for frac in fracs:
    rr = np.random.default_rng(1); yn = ytr.copy(); flip = rr.random(len(ytr)) < frac; yn[flip] = 1 - yn[flip]
    for name, mk in models: acc[name].append(mk().fit(Xtr, yn).score(Xte, yte))
chart("data-centric", {"after": "Noisy labels"}, "Test accuracy as training labels get noisier",
      {"kind": "line", "x": [round(100 * f) for f in fracs], "series": [{"name": k, "y": r(v, 3)} for k, v in acc.items()], "xLabel": "share of training labels flipped at random (%)", "yLabel": "accuracy on clean test labels"},
      md="A synthetic two-class problem; a growing share of the training labels is flipped, and every model is tested on clean labels:",
      after="The 1-nearest-neighbour classifier copies the label of a single training point, so every flipped label becomes a wrong prediction around it: its accuracy falls almost in step with the noise. Averaging over 25 neighbours outvotes isolated errors until about 30% noise, and the simple logistic regression barely moves. Flexible models memorise noise, which is why label quality matters most exactly where models are most powerful.")

# ---------------- linear regression: residuals
rng = np.random.default_rng(3)
x = np.sort(rng.uniform(0, 10, 25)); y = 1.5 + 0.8 * x + rng.normal(0, 1.2, 25)
w1, w0 = np.polyfit(x, y, 1); yhat = w0 + w1 * x
segx, segy = [], []
for xi, yi, fi in zip(x, y, yhat): segx += [xi, xi, None]; segy += [yi, fi, None]
chart("linear-regression", {"after": "Solution 1: the closed form (least squares)"}, "Least squares minimises the squared vertical distances",
      {"kind": "line", "points": False, "series": [
          {"name": "data", "x": r(x, 3), "y": r(y, 3), "scatter": True},
          {"name": f"least-squares line ŷ = {w0:.2f} + {w1:.2f}x", "x": [0, 10], "y": r([w0, w0 + 10 * w1], 3)},
          {"name": "residuals yᵢ − ŷᵢ", "x": r(segx, 3), "y": r(segy, 3), "dash": True}],
       "xLabel": "feature x", "yLabel": "target y"},
      after="Each dashed segment is one residual. Least squares picks the line that makes the sum of their squares as small as possible; any other line, tilted or shifted, would make that sum larger. Note that the distances are vertical (errors in y), not perpendicular to the line: the model predicts y from x, it does not treat the two symmetrically.")
xc = np.sort(rng.uniform(-3, 3, 80)); yc = 1 + xc + 0.6 * xc ** 2 + rng.normal(0, 0.6, 80)
A1 = np.column_stack([np.ones_like(xc), xc]); A2 = np.column_stack([np.ones_like(xc), xc, xc ** 2])
f1 = A1 @ np.linalg.lstsq(A1, yc, rcond=None)[0]; f2 = A2 @ np.linalg.lstsq(A2, yc, rcond=None)[0]
chart("linear-regression", {"after": "Checking the assumptions: residual plots"}, "Residuals against fitted values",
      {"kind": "scatter", "series": [{"name": "straight-line model", "x": r(f1, 3), "y": r(yc - f1, 3)}, {"name": "model with an x² term", "x": r(f2, 3), "y": r(yc - f2, 3)}],
       "xLabel": "fitted value ŷ", "yLabel": "residual y − ŷ"},
      md="The data follows a curve. Two linear models, one with a straight line and one with an added x² feature:",
      after="The straight-line model's residuals form a U: positive at both ends and negative in the middle, so the errors are predictable from the fitted value and the model is missing structure. With the x² feature the residuals are a shapeless band around 0, which is what a well-specified model leaves behind.")

# ---------------- naive Bayes vs logistic regression
Xb, yb = load_breast_cancer(return_X_y=True)
sizes = [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320]
nb, lr = [], []
for s in sizes:
    a1, a2 = [], []
    for seed in range(30):
        Xtr2, Xte2, ytr2, yte2 = train_test_split(Xb, yb, train_size=s, stratify=yb, random_state=seed)
        a1.append(GaussianNB().fit(Xtr2, ytr2).score(Xte2, yte2))
        a2.append(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr2, ytr2).score(Xte2, yte2))
    nb.append(np.mean(a1)); lr.append(np.mean(a2))
chart("naive-bayes", {"after": "Generative or discriminative?"}, "Naive Bayes against logistic regression as the training set grows",
      {"kind": "line", "x": sizes, "xLog": True, "series": [{"name": "Gaussian naive Bayes", "y": r(nb, 3)}, {"name": "logistic regression", "y": r(lr, 3)}],
       "xLabel": "training examples (log scale)", "yLabel": "test accuracy (average of 30 splits)"},
      md="The breast-cancer dataset (30 numeric features), averaged over 30 random splits per size:",
      after=f"With 10 examples naive Bayes is slightly ahead ({nb[0]:.3f} against {lr[0]:.3f}): it only has to estimate a mean and a variance per feature and class. But it levels off around {nb[-1]:.2f}, limited by its wrong independence assumption, while logistic regression keeps improving to {lr[-1]:.2f}. The textbook pattern: generative models reach their (lower) ceiling faster.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
print(len(out), "charts")
