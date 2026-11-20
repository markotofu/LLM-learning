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
      { type: 'chart', title: "Three losses as a function of the margin", after: `The 0–1 loss counts mistakes but gives no gradient. The hinge loss is zero only for points beyond the margin (t·h ≥ 1) and grows linearly for points inside it or on the wrong side, which is what makes SVM solutions depend only on the support vectors. The log loss of logistic regression is similar but never exactly zero, so every point keeps a small influence.`, chart: {"kind":"line","x":[-2,-1.95,-1.9,-1.85,-1.8,-1.75,-1.7,-1.65,-1.6,-1.55,-1.5,-1.45,-1.4,-1.35,-1.3,-1.25,-1.2,-1.15,-1.1,-1.05,-1,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,2.55,2.6,2.65,2.7,2.75,2.8,2.85,2.9,2.95,3],"points":false,"series":[{"name":"0–1 loss (wrong or not)","y":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"name":"hinge: max(0, 1 − t·h)","y":[3,2.95,2.9,2.85,2.8,2.75,2.7,2.65,2.6,2.55,2.5,2.45,2.4,2.35,2.3,2.25,2.2,2.15,2.1,2.05,2,1.95,1.9,1.85,1.8,1.75,1.7,1.65,1.6,1.55,1.5,1.45,1.4,1.35,1.3,1.25,1.2,1.15,1.1,1.05,1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"name":"log loss (in bits)","y":[3.069,3.005,2.942,2.88,2.818,2.756,2.695,2.634,2.574,2.514,2.455,2.396,2.338,2.28,2.223,2.167,2.111,2.056,2.002,1.948,1.895,1.842,1.791,1.74,1.69,1.64,1.592,1.544,1.497,1.451,1.405,1.361,1.317,1.274,1.233,1.192,1.151,1.112,1.074,1.037,1,0.964,0.93,0.896,0.863,0.831,0.8,0.77,0.74,0.712,0.684,0.657,0.631,0.606,0.582,0.558,0.535,0.513,0.492,0.472,0.452,0.433,0.415,0.397,0.38,0.363,0.348,0.333,0.318,0.304,0.291,0.278,0.265,0.253,0.242,0.231,0.221,0.211,0.201,0.192,0.183,0.175,0.167,0.159,0.152,0.145,0.138,0.131,0.125,0.119,0.114,0.108,0.103,0.098,0.094,0.089,0.085,0.081,0.077,0.074,0.07]}],"xLabel":"t · h(x): the signed score (positive = correct side)","yLabel":"loss"} },
      { type: 'text', title: "Support vectors, scaling and C in practice", md: `Three practical facts follow from the "widest street" picture:

- **Only the support vectors matter.** The training points that lie on the edge of the street, or inside it, fix the boundary. Every other point could move (staying off the street) or be deleted without changing the model. That is where the name comes from, and why a trained SVM only needs to store those points.
- **Scale the features.** The width of the street is measured in the units of the features. If one feature ranges from 0 to 1,000 and another from 0 to 1, the street is effectively measured along the large feature only. Always standardize the features first (\`StandardScaler\`).
- **C trades street width against violations.** A *hard margin* (no point allowed inside the street) only exists when the classes are perfectly separable, and a single outlier can ruin it. The *soft margin* allows violations, weighted by C: a **small C** gives a wide street with more violations (a simpler, more regularized model); a **large C** gives a narrow street with few violations. If an SVM overfits, lower C.` },
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
      { type: 'code', title: "Polynomial features versus the polynomial kernel", md: `Two ways to give a linear SVM curved boundaries on the two-moons data. The first adds all polynomial features up to degree 3 explicitly and fits a linear SVM on them. The second uses the **polynomial kernel**, which computes the same kind of decision function *without ever creating those features*.`, code: `import warnings
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC
warnings.filterwarnings("ignore")

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

explicit = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearSVC(C=10, max_iter=100_000))
explicit.fit(X_train, y_train)
print("features created explicitly:", explicit[0].n_output_features_)
print(f"polynomial features + linear SVM   test accuracy {explicit.score(X_test, y_test):.3f}")

kernel = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, coef0=1, C=5))
kernel.fit(X_train, y_train)
print(f"polynomial kernel SVM              test accuracy {kernel.score(X_test, y_test):.3f}")`, after: `Both reach a similar accuracy. With 2 features and degree 3 the explicit route creates only 10 features, but with 100 features and degree 3 it would create about 177,000, and with higher degrees the count explodes. The kernel trick computes the dot products those features would produce directly from the original inputs, so its cost does not depend on the number of implicit features. The \`coef0\` setting controls how much high-degree terms count compared with low-degree ones.` },
      { type: 'text', title: "Similarity features and the Gaussian RBF kernel", md: `Another way to bend the boundary is to describe each point by its **similarity** to a few reference points called *landmarks*. The Gaussian radial basis function (RBF) measures similarity as a bell curve:

$$ φ(x, ℓ) = exp( −γ · ‖x − ℓ‖² ) $$
:::where
x — an input point
ℓ — "ell": a landmark, a reference point in the same space
‖x − ℓ‖ — the distance between x and the landmark
γ — "gamma": how quickly the similarity falls off with distance (larger γ means a narrower bell)
φ(x, ℓ) — the similarity: 1 when x sits on the landmark, falling towards 0 far away
:::

Replacing each point by its similarities to a few landmarks can turn a curved problem into a linearly separable one. The simplest choice is a landmark at **every** training point, which would create as many features as there are examples. The **RBF kernel** gets the result of that construction without building the features, which is what \`SVC(kernel="rbf")\` (the default) does.

The two settings work together:

- **gamma** acts like an inverse range of influence. A large γ makes each training point's bell narrow, so the boundary wiggles around individual points (overfitting). A small γ makes broad bells and a smooth boundary (possibly underfitting).
- **C** limits violations as before. If the model overfits, reduce γ, C or both.` },
      { type: 'code', title: "gamma and C on noisy moons", code: `from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("gamma      C    train   test   support vectors")
for gamma in [0.1, 1, 10]:
    for C in [0.1, 1, 1000]:
        svm = SVC(kernel="rbf", gamma=gamma, C=C).fit(X_train, y_train)
        print(f"{gamma:5}  {C:5}    {svm.score(X_train, y_train):.3f}  {svm.score(X_test, y_test):.3f}   {len(svm.support_):4d}")`, after: `Small gamma and small C give a very smooth boundary that underfits, with many support vectors (a wide street touches many points). Large gamma and large C fit the training data almost perfectly, but the test score drops: the boundary now curls around individual noisy points. The best models sit in between, and the right combination depends on the data, so choose it with a [grid search](#/concept/cross-validation).` },
      { type: 'text', title: "Which kernel, and how fast?", md: `**Start with a linear kernel.** \`LinearSVC\` is much faster than \`SVC(kernel="linear")\`, especially with many examples or many features, as in text classification. If the data is not too large, try the **RBF kernel** next; it works well in most cases. Specialised kernels exist for particular data structures (string kernels for text or DNA sequences), and cross-validation plus a grid search can compare them.

| Class | Time to train | Kernel trick | Scaling required | Works out of core |
| --- | --- | --- | --- | --- |
| \`LinearSVC\` | grows like m × n | no | yes | no |
| \`SGDClassifier(loss="hinge")\` | grows like m × n | no | yes | yes (\`partial_fit\`) |
| \`SVC\` | grows like m² × n to m³ × n | yes | yes | no |

Here m is the number of training examples and n the number of features. \`SVC\` is a good choice for small and medium datasets (up to a few tens of thousands of examples) but becomes painfully slow on hundreds of thousands.` },
      { type: 'code', title: 'Linear vs RBF kernel on rings (scikit-learn)', code: `
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ["linear", "rbf"]:
    clf = SVC(kernel=kernel, C=1.0).fit(X_tr, y_tr)
    print(f"{kernel:>6} kernel: test accuracy {clf.score(X_te, y_te):.3f}, support vectors {len(clf.support_)}")
`, predict: {"type":"mcq","q":"The two classes form concentric rings. Roughly what test accuracy does a linear-kernel SVM reach?","options":["About 50%","About 75%","About 90%","100%"],"answer":0,"explain":"0.525: no straight line separates an inner ring from an outer one. The RBF kernel works in a space where they are separable and scores 1.000, with far fewer support vectors.","expect":["linear kernel: test accuracy 0.5"]} },
      { type: 'check', question: { type: 'tf', q: 'Increasing C in scikit-learn\'s SVC makes margin violations more expensive, which tends to narrow the margin and fit the training data more tightly.', answer: true, explain: 'C acts like 1/λ: less regularization, more risk of overfitting.' } },
      { type: 'code', title: "SVM regression", md: `SVMs also do regression, by reversing the goal: instead of the widest street *between* the classes, fit as many points as possible *on* a street of width 2ε around the prediction, and limit the points that fall outside it. Points inside the street cost nothing, so the model is called **ε-insensitive**. \`LinearSVR\` handles linear regression; \`SVR\` supports kernels.`, code: `import numpy as np
from sklearn.svm import SVR

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 6, 120))[:, None]
y = np.sin(x[:, 0]) + rng.normal(0, 0.15, 120)
counts = []
for eps in [0.05, 0.2, 0.5]:
    svr = SVR(kernel="rbf", C=10, epsilon=eps).fit(x, y)
    counts.append(len(svr.support_))
    print(f"epsilon={eps:<5} support vectors {counts[-1]:3d}   R2 {svr.score(x, y):.3f}")
print("fewer support vectors as epsilon grows:", counts[0] > counts[1] > counts[2])`, after: `Only the points **outside or on the edge of** the ε-street become support vectors, so a wider street needs far fewer of them. Too wide a street, though, ignores real structure: with ε = 0.5 the model barely notices the noise-free shape and the fit worsens.`, predict: {"type":"mcq","q":"In SVM regression, ε sets the width of the street inside which errors cost nothing. As ε grows from 0.05 to 0.5, what happens to the number of support vectors?","options":["It grows, because the street holds more points","It shrinks, because fewer points fall outside the street","It stays the same","It becomes zero immediately"],"answer":1,"explain":"Only points on or outside the street are support vectors. A wider street contains more points at no cost, so fewer remain outside: 83, then 25, then just 2 here.","expect":["fewer support vectors as epsilon grows: True"]} },
      { type: 'text', title: "Under the hood: the training objective", md: `For a linear SVM the decision function is h = w·x + b; the predicted class is positive when h ≥ 0. The street's edges are where h = 1 and h = −1, so the street's width is 2 / ‖w‖: **a smaller weight vector means a wider street.** Training a hard-margin SVM is therefore a constrained minimisation:

$$ minimise  ½ ‖w‖²   subject to   tᵢ · (w·xᵢ + b) ≥ 1   for every training example i $$
:::where
w, b — the weight vector and the bias of the decision function
‖w‖² — the squared length of w; minimising it maximises the street's width 2 / ‖w‖
tᵢ — the label of example i, written as −1 or +1
xᵢ — the features of example i
tᵢ · (w·xᵢ + b) ≥ 1 — every example is on the correct side of the street, at least one unit away from the boundary
:::

The soft margin adds a **slack variable** ζᵢ ≥ 0 for each example, measuring how far it is allowed to violate the margin, and pays for the total slack:

$$ minimise  ½ ‖w‖² + C · Σᵢ ζᵢ   subject to   tᵢ · (w·xᵢ + b) ≥ 1 − ζᵢ  and  ζᵢ ≥ 0 $$
:::where
ζᵢ — "zeta": the slack of example i, 0 when it is off the street on the correct side
C — the trade-off between a wide street (small ‖w‖) and few, small violations
Σᵢ — the sum over all training examples
:::

Both are **quadratic programs** (a quadratic objective with linear constraints), a well-studied class of problems with efficient solvers.` },
      { type: 'text', title: "The dual problem and the kernel trick", md: `A constrained problem can be rewritten as a related **dual problem** with one variable αᵢ ≥ 0 per training example. For SVMs the solution of the dual gives the solution of the original problem, and it has two useful properties:

- αᵢ is non-zero only for the **support vectors**, so the model is a weighted sum over a few training points.
- The training examples appear in the dual **only through dot products** xᵢ·xⱼ. Replace every dot product by a kernel K(xᵢ, xⱼ) and you train a linear SVM in the (possibly enormous) feature space of that kernel, without ever computing the features. Predictions use the same trick:

$$ h(x) = Σᵢ αᵢ · tᵢ · K(xᵢ, x) + b $$
:::where
h(x) — the decision score of a new point x
αᵢ — the dual weight of training example i (zero except for the support vectors)
tᵢ — the label of example i (−1 or +1)
K(xᵢ, x) — the kernel: a similarity between training example i and x, equal to a dot product in some feature space
b — the bias
:::

A function can serve as a kernel only if it really is a dot product in some space (Mercer's condition); the linear, polynomial and RBF kernels all qualify. Solving the dual is also faster than the original form when there are fewer training examples than features.` },
      { type: 'text', title: "Online SVMs and the hinge loss", md: `A linear SVM can also be trained by gradient descent on the **hinge loss** with an L2 penalty, which lets it learn from data that arrives in a stream or does not fit in memory:

$$ L(w, b) = (λ / 2) · ‖w‖² + (1/m) · Σᵢ max(0, 1 − tᵢ · (w·xᵢ + b)) $$
:::where
L(w, b) — the training objective
λ — "lambda": the regularization strength (it plays the role of 1/C)
m — the number of training examples
max(0, 1 − tᵢ · (w·xᵢ + b)) — the hinge loss: 0 for points off the street on the correct side, growing linearly with the violation
:::

In scikit-learn this is \`SGDClassifier(loss="hinge")\`, which updates the model one batch at a time with \`partial_fit\`. It converges more slowly than \`LinearSVC\`, but it never needs all the data at once.` },
      { type: 'text', title: "Strengths and weaknesses", md: `| Strengths | Weaknesses |
| --- | --- |
| powerful on small and medium datasets, with few or many features | \`SVC\` scales badly: slow beyond tens of thousands of examples |
| a kernel can capture complex boundaries even with few features | needs careful scaling and tuning of C and gamma |
| the model depends only on the support vectors | hard to inspect or explain why a prediction was made |
| works well when all features are measurements of the same kind (pixels, sensor readings) | no probabilities by default (\`probability=True\` adds a slow, cross-validated calibration) |` }
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
      { type: 'tf', q: 'A wider margin corresponds to a smaller ‖w‖.', answer: true, explain: 'The margin width is 2/‖w‖.' },
      {"type":"mcq","q":"An RBF-kernel SVM scores 100% on its training data and 82% on the test set. Which change is most likely to help?","options":["Increase gamma","Increase C","Decrease gamma and/or C","Remove feature scaling"],"answer":2,"explain":"Large gamma (narrow bells) and large C (few allowed violations) both let the boundary curl around individual points. Reducing them smooths the boundary."},
      {"type":"mcq","q":"Why does the kernel trick make the polynomial kernel practical for high degrees?","options":["It drops most of the features at random","It computes the dot products of the high-dimensional features directly from the original inputs, without creating the features","It only works with two features","It replaces the SVM by a decision tree"],"answer":1,"explain":"K(a, b) equals the dot product the explicit features would give, computed in the original space, so the cost does not grow with the number of implicit features."},
      {"type":"tf","q":"In SVM regression, training points that fall inside the ε-street do not affect the model.","answer":true,"explain":"Errors inside the street cost nothing (the ε-insensitive loss), so only points on or outside it become support vectors."}
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
      { title: "Stanford CS229 — Machine Learning lecture notes (SVMs and kernels)", url: "https://cs229.stanford.edu/main_notes.pdf" },
      { title: "Andreas Müller & Sarah Guido — Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" },
      { title: "Aurélien Géron — Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow (book; code and notebooks)", url: "https://github.com/ageron/handson-ml3" }
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
      { type: 'text', title: "Three flavours of naive Bayes in scikit-learn", md: `All three make the same "naive" independence assumption and differ only in how they model each feature within a class:

| Class | Features | What training stores per class |
| --- | --- | --- |
| \`GaussianNB\` | continuous measurements | the mean and variance of every feature |
| \`BernoulliNB\` | binary (present / absent) | how often each feature is present |
| \`MultinomialNB\` | counts, such as word counts | the average count of each feature |

For \`GaussianNB\`, the likelihood of a feature value is a bell curve centred on that class's mean:

$$ P(x_j | c) = (1 / √(2π σ²_cj)) · exp( −(x_j − μ_cj)² / (2 σ²_cj) ) $$
:::where
x_j — the value of feature j for the example being classified
c — a candidate class
μ_cj — the mean of feature j among the training examples of class c
σ²_cj — the variance of feature j among the training examples of class c
π — the constant 3.14159…
exp — the exponential function
:::

\`BernoulliNB\` and \`MultinomialNB\` have one setting, **alpha**, the smoothing strength (alpha = 1 is the add-one smoothing from the spam filter above). Larger alpha makes the model smoother and less confident; results are usually not very sensitive to it.` },
      { type: 'code', title: "Training is just counting", md: `Bernoulli naive Bayes on four tiny examples with four binary features. Training only needs, for each class, how often every feature is present:`, code: `import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
for label in np.unique(y):
    print(f"class {label}: feature counts {X[y == label].sum(axis=0)}")

model = BernoulliNB(alpha=1.0).fit(X, y)
print("P(class | [1, 0, 1, 1]) =", model.predict_proba([[1, 0, 1, 1]]).round(3))

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
gnb = GaussianNB().fit(X_train, y_train)
print(f"Gaussian naive Bayes on breast cancer: test accuracy {gnb.score(X_test, y_test):.3f}")
print("learned means of the first 3 features, per class:")
print(gnb.theta_[:, :3].round(2))`, after: `Class 1 contains features 0 and 2 in both of its examples, so a new example that has them is classified as class 1 with high confidence. On the breast-cancer data, Gaussian naive Bayes is a few points less accurate than logistic regression, but it trains in a fraction of the time, because "training" is computing means and variances.` },
      { type: 'text', title: "Strengths and weaknesses", md: `| Strengths | Weaknesses |
| --- | --- |
| extremely fast to train (one pass of counting) and to predict | usually a bit less accurate than linear models |
| works with very high-dimensional, sparse data such as text | its probabilities are poorly calibrated: often too close to 0 or 1, because evidence from correlated features is counted twice |
| almost no tuning: alpha rarely matters much | Gaussian NB assumes bell-shaped features within each class |
| a sensible baseline for huge datasets where even linear models are slow | classification only |` },
      { type: 'text', title: "Generative or discriminative?", md: `Naive Bayes is a **generative** classifier: it models how each class produces its features, P(x | y), and how common each class is, P(y), then applies Bayes' rule. Logistic regression is its **discriminative** counterpart: it models P(y | x) directly. For word counts the two even share the same form, a linear score per class, but they fit it differently: naive Bayes by counting, logistic regression by minimising the log loss.

That difference shapes how they learn, as a classic comparison by Ng and Jordan showed:
- naive Bayes has fewer and simpler estimates to make, so it approaches its best accuracy with very little data;
- its best accuracy is limited by the independence assumption, which is almost always wrong;
- logistic regression needs more data but keeps improving, because it never assumes how the features relate to each other.` },
      { type: 'chart', title: "Naive Bayes against logistic regression as the training set grows", md: `The breast-cancer dataset (30 numeric features), averaged over 30 random splits per size:`, after: `With 10 examples naive Bayes is slightly ahead (0.909 against 0.904): it only has to estimate a mean and a variance per feature and class. But it levels off around 0.94, limited by its wrong independence assumption, while logistic regression keeps improving to 0.98. The textbook pattern: generative models reach their (lower) ceiling faster.`, chart: {"kind":"line","x":[10,15,20,30,40,60,80,120,160,240,320],"xLog":true,"series":[{"name":"Gaussian naive Bayes","y":[0.909,0.915,0.927,0.937,0.937,0.939,0.937,0.936,0.935,0.935,0.936]},{"name":"logistic regression","y":[0.904,0.919,0.928,0.942,0.95,0.959,0.964,0.967,0.968,0.974,0.976]}],"xLabel":"training examples (log scale)","yLabel":"test accuracy (average of 30 splits)"} },
      { type: 'text', title: "Why naive Bayes is overconfident", md: `The independence assumption also distorts the probabilities. If "free", "free!" and "FREE" all appear in an email, naive Bayes counts them as three independent pieces of evidence, although they say almost the same thing. Correlated features are counted again and again, so the predicted probabilities are pushed towards 0 and 1: naive Bayes is typically very **overconfident**.

Its *ranking* of examples is often still good, which is why it works well as a classifier (only the side of 0.5 matters) and as a fast baseline. If you need trustworthy probabilities, calibrate it on held-out data (see [Calibration & Uncertainty](#/concept/calibration-uncertainty)).` },
      { type: 'code', title: "Duplicated evidence, inflated confidence", code: `import numpy as np
from sklearn.naive_bayes import MultinomialNB
docs = np.array([[3, 0, 1, 0],          # counts of 4 words in 6 short documents
                 [2, 1, 0, 0],
                 [0, 2, 0, 3],
                 [0, 3, 1, 2],
                 [1, 0, 2, 0],
                 [0, 1, 0, 2]])
labels = np.array([1, 1, 0, 0, 1, 0])
new_doc = np.array([[1, 1, 1, 1]])

once = MultinomialNB().fit(docs, labels)
thrice = MultinomialNB().fit(np.hstack([docs] * 3), labels)      # every word column copied three times
p1 = once.predict_proba(new_doc)[0, 1]
p3 = thrice.predict_proba(np.hstack([new_doc] * 3))[0, 1]
print(f"P(class 1) with each word once:          {p1:.3f}")
print(f"P(class 1) with each word three times:   {p3:.3f}")
print("same prediction:", once.predict(new_doc)[0] == thrice.predict(np.hstack([new_doc] * 3))[0])
print("duplicating features makes it more confident:", bool(p3 > p1 > 0.5))`, after: `No information was added, only copies, yet the probability moved from about 0.58 to about 0.72; the predicted class did not change. Real text is full of near-duplicates (word forms, synonyms, set phrases), so naive Bayes probabilities close to 0 or 1 should not be taken literally.`, predict: {"type":"mcq","q":"Every word column is duplicated three times, which adds no new information. What happens to naive Bayes' probability for the more likely class?","options":["It stays the same","It moves closer to 0.5","It moves further from 0.5, towards 1","The predicted class flips"],"answer":2,"explain":"Naive Bayes treats each copy as independent evidence, so the evidence in the log-odds is tripled and the probability becomes more extreme, while the predicted class stays the same.","expect":["duplicating features makes it more confident: True"]} },
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
      { type: 'mcq', q: 'How should Naive Bayes treat a word that never appeared in the training data?', options: ['Give it probability 0', 'Skip it (it is not in the vocabulary)', 'Reject the document', 'Retrain the model'], answer: 1, explain: 'Out-of-vocabulary words carry no learned information, so they are ignored.' },
      {"type":"mcq","q":"Which naive Bayes variant fits word counts in documents?","options":["GaussianNB","BernoulliNB","MultinomialNB","None: naive Bayes cannot use counts"],"answer":2,"explain":"MultinomialNB models counts (how many times each word appears). BernoulliNB only uses presence or absence; GaussianNB assumes continuous, bell-shaped features."},
      {"type":"mcq","q":"Compared with logistic regression, naive Bayes typically…","options":["Needs more data and reaches a higher final accuracy","Reaches its best accuracy with less data, but that best accuracy is often lower","Is always more accurate","Cannot handle text"],"answer":1,"explain":"As a generative model with simple estimates it learns quickly, but its independence assumption caps its accuracy; logistic regression keeps improving with more data."},
      {"type":"tf","q":"Naive Bayes' predicted probabilities are usually well calibrated.","answer":false,"explain":"Correlated features are counted as independent evidence, which pushes its probabilities towards 0 and 1. Its rankings are often good, but its probabilities need calibration."}
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
      { title: 'Paul Graham — A Plan for Spam (2002)', url: 'http://www.paulgraham.com/spam.html' },
      { title: "Andreas Müller & Sarah Guido — Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
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
      { type: 'chart', title: "The elbow: inertia against k", md: `The three-blob data from the code above, clustered with k = 1 to 8:`, after: `Inertia always falls as k grows, but it falls steeply only until k reaches the true number of groups. After the elbow each extra cluster merely splits a real group in two and buys little.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8],"series":[{"name":"inertia (best of 10 starts)","y":[901,485.4,108.8,92.7,79.1,67,57,49.2]}],"xLabel":"number of clusters k","yLabel":"inertia","vlines":[{"x":3,"label":"elbow at k = 3"}],"yMin":0} },
      { type: 'chart', title: "The silhouette score against k", after: `Unlike inertia, the silhouette does not automatically improve with more clusters, so its peak can be read directly: k = 3 scores clearly best (about 0.71).`, chart: {"kind":"bar","x":["2","3","4","5","6","7","8"],"series":[{"name":"average silhouette (higher is better)","y":[0.498,0.709,0.582,0.469,0.335,0.361,0.371]}],"xLabel":"number of clusters k","yLabel":"silhouette score","yMin":0,"yMax":1} },
      { type: 'text', title: "When k-means fails, and what to use instead", md: `k-means draws straight boundaries: every point goes to its nearest centroid, so the space is cut into convex cells. Whenever the true groups are not round blobs of similar size, it cuts in the wrong places:

| Data | What k-means does | Better choice |
| --- | --- | --- |
| Curved or nested shapes (two moons, rings) | slices across them with straight lines | DBSCAN, spectral clustering, single linkage |
| Elongated or tilted clusters | splits them along their long axis | a Gaussian mixture with full covariances |
| Clusters of very different sizes | steals points from the big cluster for the small one | a Gaussian mixture, DBSCAN |
| Features on very different scales | clusters only along the largest-scale feature | standardise the features first |
| Outliers | drags centroids towards them | k-medoids, DBSCAN, or remove the outliers |

The last two rows are the most common problems in practice, and the easiest to fix.` },
      { type: 'chart', title: "k-means on two interleaved moons", after: `Each moon is one true group, but k-means can only split the plane with the straight line halfway between two centroids, so each "cluster" takes the tip of the other moon. No initialisation fixes this: the shape violates k-means' assumptions. DBSCAN or single-linkage clustering separates the moons cleanly.`, chart: {"kind":"scatter","series":[{"name":"k-means cluster 1","x":[0.235,-1.023,0.508,-0.711,0.225,-0.056,-0.399,0.075,-0.436,-0.886,-0.846,-0.769,-0.362,0.314,-0.87,0.009,0.111,0.134,-0.275,-0.175,0.381,-0.909,0.251,0.624,-0.012,0.044,-0.652,0.205,0.053,-0.037,-0.921,-1.118,0.233,-0.734,0.557,0.014,-0.954,-0.465,-0.98,-0.796,0.018,-0.041,0.559,-0.026,-0.008,-0.495,0.373,0.288,-0.843,0.152,0.073,0.042,0.006,0.666,-0.314,0.242,0.162,-0.702,-0.91,0.426,-0.117,-0.616,0.125,0.008,-0.416,0.148,0.337,0.032,-0.976,0.717,-0.585,0.063,-0.847,-0.025,-0.173,0.338,-0.478,0.26,-0.172,0.393,0.74,-0.056,0.56,-0.915,0.326,0.268,-0.992,0.577,-0.71,-0.956,-0.985,-0.655,-0.61,-0.457,0.02,0.155,-0.749,-0.329,-0.362,0.236,0.524,0.486,-0.198,-0.959,-1.008,-1.109,-0.142,0.16,-0.525,-0.896,0.044,-0.283,0.081,-0.824,-1.087,-0.976,-0.3,-0.171,-0.81,-0.589,0.333,0.024,0.194,-0.892,0.238,0.65,-0.581,-0.923,-0.099,0.669,-0.945,0.318,-1.02,0.176,0.123,-0.035,-0.874,0.17,-0.987,0.11,-0.48,-0.796,0.132,0.111,0.136,0.625,-0.311,0.064,-0.111,0.258,-0.735,0.575,0.105],"y":[-0.244,0.178,0.851,0.802,-0.24,1.003,0.963,0.199,0.905,0.489,0.449,0.597,0.955,0.981,0.568,0.514,-0.035,-0.154,0.994,0.967,0.816,0.274,0.941,0.854,0.916,0.003,0.784,0.958,0.135,0.413,0.304,0.208,1.032,0.689,0.893,0.409,0.078,0.954,0.297,0.591,0.284,1.135,0.9,0.441,0.342,0.869,0.822,-0.149,0.466,0.092,1.007,0.322,0.364,0.664,0.923,0.865,0.159,0.636,0.114,0.946,1.021,0.75,-0.051,0.256,0.908,0.272,0.995,0.995,0.256,0.764,0.779,0.346,0.427,1.015,0.946,0.895,0.872,-0.127,1.001,0.781,0.794,0.374,0.947,0.516,0.951,-0.238,0.468,0.798,0.736,0.334,-0.053,0.788,0.904,0.89,0.243,0.034,0.662,0.879,0.892,0.991,0.752,0.86,1.013,0.058,0.089,0.087,1.044,-0.119,0.886,0.397,1.016,0.945,-0.009,0.534,0.11,0.167,0.917,0.969,0.591,0.83,0.957,0.125,1.062,0.556,-0.095,0.635,0.831,0.378,0.916,0.679,0.396,0.922,0.374,-0.08,0.253,0.546,0.504,-0.279,0.107,0.027,0.997,0.698,0.993,0.156,-0.008,0.807,0.911,0.932,0.831,-0.256,0.525,0.938,0.102]},{"name":"k-means cluster 2","x":[0.727,0.938,0.983,1.227,1.187,0.675,1.934,0.923,1.437,0.932,1.774,1.092,0.381,0.986,0.417,2.017,1.246,1.94,0.928,0.792,1.967,0.925,1.058,0.746,1.191,0.998,0.665,1.597,0.877,2.068,1.95,1.974,0.927,1.589,1.051,1.05,1.775,0.835,1.585,0.752,0.283,1.697,0.797,1.211,0.697,1.055,1.646,1.101,1.831,1.906,0.278,0.89,1.818,1.483,1.041,1.403,0.617,2.024,0.804,1.496,1.955,0.886,0.87,0.908,0.438,1.1,0.526,1.677,1.884,1.736,1.012,0.882,0.993,0.828,0.743,2.006,0.671,0.933,1.552,1.315,1.737,0.878,1.203,1.024,0.433,0.954,2.151,1.139,0.529,0.934,0.733,1.601,1.981,0.763,1.315,0.916,0.806,0.738,1.065,1.885,1.957,1.554,1.24,2,0.642,0.92,1.39,0.558,1.768,2.021,0.406,0.936,0.505,1.653,1.856,1.71,1.921,0.984,0.768,1.186,1.062,1.868,0.914,1.67,1.185,1.616,0.844,1.445,0.504,0.921,0.779,1.872,1.81,1.798,0.508,1.456,1.712,0.816,1.945,1.882,0.432,1.395,1.888,0.649,1.087,0.371,1.531],"y":[-0.505,0.366,-0.492,-0.403,-0.474,-0.406,0.27,0.312,-0.513,0.351,0.019,-0.566,-0.197,-0.596,-0.306,0.248,-0.508,0.391,0.147,-0.489,0.525,0.152,-0.624,0.639,-0.559,0.128,-0.434,-0.32,0.355,0.212,0.466,0.246,-0.427,-0.258,0.346,-0.475,0.078,0.498,-0.328,0.694,-0.305,-0.222,-0.529,-0.479,-0.409,-0.571,-0.371,-0.429,0.176,-0.066,-0.306,0.339,0.101,-0.322,-0.522,-0.476,-0.476,0.255,-0.45,-0.469,0.097,0.383,0.643,0.264,-0.332,0.125,-0.512,-0.187,0.006,-0.089,0.271,-0.565,0.075,0.478,0.64,0.453,-0.586,0.387,-0.371,-0.429,0.004,0.01,-0.518,0.271,-0.384,0.304,0.349,-0.515,-0.372,-0.594,-0.365,-0.351,0.34,-0.544,-0.338,-0.484,-0.495,0.616,-0.154,0.021,0.515,-0.217,-0.493,0.489,-0.387,0.444,-0.466,-0.591,-0.19,0.502,-0.338,0.115,-0.385,-0.234,0.021,-0.365,0.095,0.33,0.651,-0.409,-0.039,-0.096,0.294,-0.264,-0.482,-0.374,0.58,-0.365,-0.56,0.511,0.768,0.137,-0.104,-0.039,-0.339,-0.412,-0.174,0.475,0.219,0.237,-0.252,-0.51,-0.007,-0.528,0.181,-0.279,-0.305]},{"name":"centroids","x":[-0.196,1.21],"y":[0.563,-0.089]}],"xLabel":"x₁","yLabel":"x₂"} },
      { type: 'code', title: "Scaling decides what k-means sees", md: `Customers come in two age groups. Their income varies a lot but has nothing to do with the groups. Watch what k-means finds with and without scaling.`, code: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
rng = np.random.default_rng(0)
n = 300
group = rng.integers(0, 2, n)                                    # the hidden truth: younger or older
age = np.where(group == 0, rng.normal(28, 4, n), rng.normal(58, 5, n))
income = rng.normal(60_000, 25_000, n)                           # large numbers, unrelated to the groups
X = np.column_stack([age, income])

ari = {}
for name, data in [("raw features", X), ("standardised", StandardScaler().fit_transform(X))]:
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(data)
    ari[name] = adjusted_rand_score(group, labels)
    print(f"{name:<13} agreement with the age groups (adjusted Rand index): {ari[name]:.2f}")
print("scaling recovers the age groups:", ari["standardised"] > 0.9 > ari["raw features"])`, after: `On the raw features, distances are almost entirely income differences (tens of thousands) rather than age differences (tens of years), so k-means splits customers into lower and higher earners, which has nothing to do with the real groups (an adjusted Rand index near 0). After standardisation both features count equally and the age groups are recovered perfectly.`, predict: {"type":"mcq","q":"Without scaling, what will k-means split these customers by?","options":["Age","Income","Both equally","Nothing: it fails to run"],"answer":1,"explain":"Euclidean distance is dominated by the feature with the largest numbers, here income in the tens of thousands, so the clusters follow income.","expect":["scaling recovers the age groups: True"]} },
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
      { type: 'mcq', q: 'What does k-means++ improve?', options: ['The choice of k', 'The initial centroids, spreading them out for better and more consistent results', 'The distance metric', 'Training speed on GPUs'], answer: 1, explain: 'Smart seeding avoids starting several centroids in the same cluster.' },
      {"type":"mcq","q":"The silhouette score for k = 2, 3, 4 and 5 is 0.50, 0.71, 0.58 and 0.47. Which k does it suggest?","options":["2","3","4","5"],"answer":1,"explain":"Unlike inertia, the silhouette does not automatically improve with more clusters, so you pick its maximum: k = 3."},
      {"type":"mcq","q":"Two features are age (20 to 70) and income (20,000 to 150,000). Without scaling, k-means clusters will mostly reflect…","options":["Age","Income","Both equally","Neither"],"answer":1,"explain":"Squared Euclidean distances are dominated by the feature with the largest spread, so standardise first."}
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
      { type: 'demo', title: 'Interactive: find the direction of greatest variance', demo: 'pca' },
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
      { type: 'text', title: "Reading the components: scores, loadings and whitening", md: `After fitting, PCA gives you two things to look at:

- **Scores** are the coordinates of each example along the components (\`pca.transform(X)\`). They are new, uncorrelated features, and a scatter plot of the first two scores is the classic way to look at high-dimensional data.
- **Loadings** are the weights of each component on the original features (the rows of \`pca.components_\`). A component with large weights on rooms, area and bathrooms is a "size" axis; one that contrasts price with distance to the centre is a "location" axis. Reading loadings is how PCA becomes a tool for understanding data, not only for compressing it.

$$ score_ik = Σ_j ( x_ij − x̄_j ) · v_kj $$
:::where
x_ij — feature j of example i; x̄_j — the mean of feature j
v_kj — the loading of feature j on component k (component k is a unit vector v_k)
score_ik — where example i lands along component k
:::

**Whitening** (\`PCA(whiten=True)\`) also divides each score by its component's standard deviation, so that every component has variance 1. That suits algorithms that expect equally scaled, uncorrelated inputs, but it also inflates low-variance components that are mostly noise.

**How many components?** Look at the cumulative explained variance: \`PCA(n_components=0.90)\` keeps just enough components for 90% of the variance. The handwritten digits below show what that means on real data.` },
      { type: 'code', title: "PCA on handwritten digits", md: `Each 8 × 8 image of a digit is a vector of 64 pixel intensities. How many directions does the data really use?`, code: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)                 # 1,797 images, 64 pixels each
X = X / 16.0                                        # pixel values from 0 to 1
pca = PCA().fit(X)
cumulative = np.cumsum(pca.explained_variance_ratio_)
k90 = int(np.searchsorted(cumulative, 0.90) + 1)
print("variance explained by the first 5 components:", pca.explained_variance_ratio_[:5].round(3))
print("components needed for 90% of the variance:", k90)
print("about 20 components suffice:", abs(k90 - 20) <= 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
for k in (2, 10, 21, 64):
    p = PCA(n_components=k).fit(X_train)            # fit on the training data only
    clf = LogisticRegression(max_iter=2000).fit(p.transform(X_train), y_train)
    rebuilt = p.inverse_transform(p.transform(X_test))
    print(f"{k:2d} components: test accuracy {clf.score(p.transform(X_test), y_test):.3f}   "
          f"reconstruction error {np.mean((X_test - rebuilt) ** 2):.4f}")`, after: `Two components keep less than 30% of the variance, and a classifier on them reaches only about 58% accuracy. 21 components keep 90% of the variance and come within one point of the accuracy on all 64 pixels (about 96%), with a third of the features. The images are far simpler than their 64 numbers suggest.`, predict: {"type":"mcq","q":"How many of the 64 principal components do the digit images need to keep 90% of the variance?","options":["2","About 20","About 50","All 64"],"answer":1,"explain":"Neighbouring pixels are strongly correlated and many border pixels are always blank, so about 21 directions carry 90% of the variation.","expect":["about 20 components suffice: True"]} },
      { type: 'chart', title: "Cumulative explained variance of the digit images", after: `The curve rises steeply and then flattens: the first 10 components keep 74% of the variance, 21 keep 90%, and the last 20 or so add almost nothing (some pixels at the image border are always blank). Reading the elbow of this curve is the usual way to choose k.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64],"points":false,"series":[{"name":"share of the variance kept by the first k components","y":[0.1489,0.2851,0.403,0.4871,0.545,0.5941,0.6373,0.6739,0.7074,0.7382,0.762,0.7847,0.8029,0.8206,0.8353,0.8494,0.8626,0.8751,0.8852,0.8943,0.9032,0.9112,0.9188,0.9261,0.933,0.939,0.9447,0.9499,0.9548,0.9591,0.9628,0.9664,0.9697,0.973,0.9761,0.979,0.9816,0.9839,0.9861,0.9882,0.9901,0.9917,0.9932,0.9946,0.9958,0.9968,0.9978,0.9986,0.9991,0.9995,0.9998,0.9998,0.9999,0.9999,1,1,1,1,1,1,1,1,1,1]}],"xLabel":"number of components k (of 64)","yLabel":"cumulative explained variance","yMin":0,"yMax":1,"hlines":[{"y":0.9,"label":"90%"}],"vlines":[{"x":21,"label":"k = 21"}]} },
      { type: 'chart', title: "The digits projected onto the first two components", md: `Each dot is one 64-pixel image, placed by its scores on the first two components (four of the ten digits shown):`, after: `Even two directions, chosen without ever looking at the labels, separate the zeros, ones, fours and sevens into recognisable regions. Where clouds overlap, two components are not enough, which matches the low accuracy of the 2-component classifier above.`, chart: {"kind":"scatter","series":[{"name":"digit 0","x":[-0.079,0.701,0.402,0.015,0.383,0.314,0.246,0.645,0.394,-0.214,-0.079,0.535,0.478,0.126,0.223,0.505,0.376,0.268,0.176,0.674,0.666,0.357,0.925,0.707,0.009,0.497,0.353,0.197,0.04,0.364,-0.198,0.229,-0.039,0.07,0.124,0.215,0.033,0.074,0.155,-0.33,-0.05,0.284,0.469,-0.167,0.175,0.335,-0.338,0.136,-0.057,0.006,0.179,0.038,-0.22,0.008,-0.404,0.013,-0.39,-0.157,-0.311,0.228,0.107,-0.539,-0.108,-0.545,-0.184,0.195,-0.057,0.472,0.309,-0.004,0.481,-0.002,0.91,0.526,0.36,0.242,0.264,0.21,-0.204,-0.256,0.267,0.246,0.223,0.113,0.335,0.229,0.421,0.728,-0.321,-0.179],"y":[-1.33,-1.057,-1.217,-1.662,-0.994,-1.298,-1.305,-0.874,-1.256,-1.428,-1.488,-1.47,-1.318,-1.508,-1.346,-1.392,-1.302,-1.411,-1.517,-1.011,-1.341,-1.527,-1.018,-1.29,-1.568,-1.451,-1.175,-1.302,-1.301,-1.156,-1.186,-1.284,-1.553,-1.333,-1.295,-1.536,-1.125,-1.27,-1.504,-1.563,-1.559,-1.502,-1.062,-1.41,-1.658,-1.342,-1.635,-1.178,-1.491,-1.692,-1.402,-1.396,-1.5,-1.372,-1.144,-1.387,-1.448,-1.543,-1.824,-1.326,-1.182,-1.128,-1.261,-1.201,-1.541,-1.463,-0.72,-1.32,-1.209,-1.563,-1.401,-0.825,-0.588,-1.253,-1.113,-1.448,-1.413,-1.08,-1.232,-1.658,-1.129,-1.58,-1.472,-1.113,-1.08,-1.358,-1.401,-0.821,-1.328,-1.422]},{"name":"digit 1","x":[0.497,0.188,0.002,0.177,0.103,0.203,0.495,0.012,0.469,0.196,0.785,1.245,-0.049,-0.244,0.175,-0.139,-0.145,-0.12,0.129,0.45,0.09,0.067,-0.102,0.474,0.245,0.596,0.678,0.548,0.839,0.886,0.607,0.35,0.712,0.517,0.43,0.683,0.633,0.217,0.95,-0.379,0.157,0.281,-0.082,0.094,-0.016,-0.03,0.402,0.791,-0.021,0.088,0.082,0.142,-0.055,-0.014,0.101,0.463,0.748,0.605,0.756,0.516,0.31,0.559,0.027,0.495,-0.013,0.644,0.856,1.134,0.641,0.648,0.035,0.947,0.041,0.503,0.618,0.865,0.925,0.671,0.157,-0.157,0.262,0.775,0.471,-0.002,0.052,0.688,0.583,0.646,0.563,0.358],"y":[1.298,0.75,0.646,0.58,0.671,0.6,0.678,0.973,1.304,0.881,1.265,1.2,0.577,-0.239,0.454,0.157,0.213,-0.228,0.462,0.617,0.341,0.667,0.015,0.681,0.688,0.61,1.329,1.162,1.126,1.176,1.053,0.723,1.276,1.584,1.289,1.444,1.296,1.578,0.788,-0.152,0.654,0.529,0.214,0.487,0.21,0.505,0.796,0.755,0.663,0.701,0.365,0.227,0.353,0.25,0.403,0.227,0.176,0.183,0.069,0.332,0.296,0.11,0.581,0.721,0.391,1.631,1,0.696,1.388,1.221,1.513,0.886,1.644,0.933,1.056,1.155,1.096,1.39,1.674,1.278,1.642,0.592,1.184,0.189,1.344,0.8,1.231,1.581,1.407,1.627]},{"name":"digit 4","x":[1.457,1.529,1.096,1.565,1.761,1.215,1.138,1.411,1.423,1.052,1.172,1.044,1.532,1.151,1.401,1.228,1.584,1.274,1.319,1.641,1.553,1.3,1.391,1.097,1.419,1.307,1.653,1.462,1.723,1.806,1.623,1.651,1.9,1.293,1.842,1.756,1.803,1.677,1.111,1.374,1.381,1.199,1.411,1.264,1.084,1.331,1.299,0.934,1.349,1.27,1.188,1.21,0.74,1.395,1.198,1.546,0.917,1.053,1.159,1.239,0.979,1.525,0.776,0.941,1.454,1.698,1.067,1.148,0.983,1.397,1.574,1.22,0.485,0.935,1,1.463,0.863,1.764,1.863,1.609,1.81,1.686,1.804,1.581,1.588,1.605,1.704,1.797,1.847,1.652],"y":[-0.267,-0.166,-0.289,-0.048,-0.464,0.276,0.826,-0.043,-0.312,0.087,-0.168,0.757,-0.077,0.935,0.68,0.776,0.787,0.653,0.799,0.738,0.819,0.804,0.999,0.514,0.83,0.1,0.246,0.063,-0.258,-0.171,-0.161,0.26,-0.488,0.399,-0.586,0.044,-0.105,0.077,0.236,0.608,0.507,0.398,0.301,0.384,-0.066,0.279,0.446,0.211,0.256,0.116,0.593,0.402,0.478,0.083,0.176,0.158,0.572,0.405,0.128,0.223,-0.015,0.052,0.391,0.271,0.653,0.58,0.232,0.664,0.369,0.418,0.422,0.35,0.738,0.042,0.617,-0.16,0.708,-0.195,0.261,-0.155,-0.196,0.057,0.045,-0.853,-0.147,-0.273,0.035,0.023,-0.246,-0.594]},{"name":"digit 7","x":[-0.185,0.351,0.298,0.354,-0.223,0.431,0.298,0.084,0.408,-0.01,0.23,0.148,0.018,0.067,0.305,0.241,0.008,-0.139,0.048,0.557,0.071,0.357,-0.103,-0.013,0.345,-0.356,0.067,0.469,-0.264,0.085,-0.328,0.118,0.165,0.06,-0.123,0.273,0.009,0.324,0.16,0.211,0.391,0.45,-0.272,0.158,0.386,0.552,-0.33,-0.097,0.302,0.18,-0.087,-0.365,-0.451,-0.154,0.236,-0.27,-0.359,-0.087,-0.343,-0.131,0.168,-0.29,-0.185,-0.399,-0.099,0.133,0.251,0.913,0.499,0.258,0.035,0.236,0.611,0.483,0.357,0.794,-0.149,-0.146,-0.18,-0.427,-0.51,-0.196,0.175,-0.462,-0.328,0.05,-0.075,0,-0.145,-0.49],"y":[1.317,0.808,0.324,0.648,1.183,0.833,0.734,1.226,0.232,1.092,0.709,1.104,0.997,1.123,1.275,1.227,1.25,1.221,1.387,1.318,1.476,1.235,1.36,1.305,1.279,1.397,1.111,0.899,1.108,0.858,1.222,1.041,0.891,0.888,1.088,1.064,1,0.745,0.535,0.342,0.294,0.12,1.037,0.444,0.04,-0.043,0.883,0.143,0.389,0.145,0.701,1.154,1.389,1,1.038,1.262,1.04,1.207,1.308,1.306,1.21,1.276,1.092,1.389,1.104,1.182,0.804,0.452,0.523,1.013,1.065,0.846,0.619,0.912,0.781,0.514,1.242,1.013,1.331,1.424,1.13,1.066,1.029,1.288,1.169,1.085,1.329,1.162,1.202,1.26]}],"xLabel":"score on principal component 1","yLabel":"score on principal component 2"} },
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
      { type: 'mcq', q: 'Which method would better reveal curved, non-linear structure for visualization?', options: ['PCA with more components', 'UMAP or t-SNE', 'Linear regression', 'One-hot encoding'], answer: 1, explain: 'They are non-linear embedding methods designed for visualization.' },
      {"type":"mcq","q":"What are the loadings of a principal component?","options":["The coordinates of each example along the component","The weights of the original features that define the component's direction","The variance explained by the component","The class labels"],"answer":1,"explain":"Loadings (the rows of components_) say how much each original feature contributes to the direction; scores are the examples' coordinates along it."},
      {"type":"tf","q":"PCA(n_components=0.9) keeps the smallest number of components that together explain at least 90% of the variance.","answer":true,"explain":"A fraction between 0 and 1 tells scikit-learn to choose the number of components from the cumulative explained-variance ratio."}
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
      { type: 'code', title: "The leak you cannot see", md: `Here the features are **pure random noise**: 10,000 of them for 100 examples, and a random target. No model can predict anything. First, the classic mistake: select the 5% of features most correlated with the target using *all* the data, then cross-validate. Then the correct way: put the selection inside a pipeline, so it is redone on the training part of every fold.`, code: `import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 10_000))       # random features
y = rng.normal(size=100)                 # a random target: nothing to learn

selected = SelectPercentile(f_regression, percentile=5).fit(X, y).transform(X)   # looks at ALL labels
leaky = cross_val_score(Ridge(), selected, y, cv=5).mean()
honest = cross_val_score(make_pipeline(SelectPercentile(f_regression, percentile=5), Ridge()), X, y, cv=5).mean()
print(f"selection outside cross-validation: R2 = {leaky:.3f}")
print(f"selection inside a pipeline:        R2 = {honest:.3f}")
print("the leaky estimate is wildly optimistic:", leaky > 0.5 > honest)`, after: `The leaky version reports an excellent R² on data that contains no signal at all. Among 10,000 random features, some correlate with the random target **by chance**, and choosing them with knowledge of every label, including the labels of the validation folds, carries that chance correlation into every fold. Inside the pipeline, the selection only sees each fold's training part, and the honest result appears: no predictive power (a negative R² means worse than predicting the mean).

The same leak, in milder form, happens with scaling, imputation, encoding or anything else that is *fitted* on data. The rule: **every step that learns from data must be inside the cross-validation loop**, which is exactly what a pipeline guarantees.`, predict: {"type":"mcq","q":"The features and the target are pure random noise. You pick the 500 features most correlated with the target using all 100 examples, then cross-validate a model on them. What R² will cross-validation report?","options":["About 0 or negative: there is nothing to learn","A high value, around 0.9","Exactly 1","Cross-validation will refuse to run"],"answer":1,"explain":"The selection already saw every validation label, so the chosen features fit them by chance, and cross-validation reports around 0.9. Inside a pipeline the honest answer appears: about −0.25.","expect":["the leaky estimate is wildly optimistic: True"]} },
      { type: 'chart', title: "Cross-validated R² on pure noise", after: `The same data and the same model. Only where the feature selection happens differs, and it turns "no signal at all" into an apparently excellent model.`, chart: {"kind":"bar","x":["features selected before CV (leaky)","selection inside a pipeline (honest)"],"series":[{"name":"mean cross-validation R²","y":[0.898,-0.252]}],"xLabel":"procedure","yLabel":"R² (0 = no better than the mean)","hlines":[{"y":0,"label":"no skill"}]} },
      { type: 'text', title: "The general pipeline interface", md: `A pipeline is a list of steps. Every step except the last must be a **transformer** (it has \`fit\` and \`transform\`); the last step can be anything with \`fit\`: a classifier, a regressor, or another transformer.

- \`pipe.fit(X, y)\` calls \`fit_transform\` on each step in turn, passing the transformed data along, then \`fit\` on the last step.
- \`pipe.predict(X)\` calls \`transform\` on each step, then \`predict\` on the last step. \`score\` and \`predict_proba\` work the same way.
- The pipeline as a whole has the methods of its last step, so it can be used anywhere a model can: in \`cross_val_score\`, in \`GridSearchCV\`, in another pipeline.

\`make_pipeline(StandardScaler(), LogisticRegression())\` names the steps automatically after their classes, in lower case (\`"standardscaler"\`, \`"logisticregression"\`; repeated classes get numbers). With \`Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])\` you choose the names yourself.` },
      { type: 'code', title: "Building a pipeline and looking inside it", code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
pipe = make_pipeline(StandardScaler(), PCA(n_components=5), LogisticRegression())
print("steps:", [name for name, _ in pipe.steps])
pipe.fit(X_train, y_train)
print(f"test accuracy: {pipe.score(X_test, y_test):.3f}")

pca = pipe.named_steps["pca"]                 # access a step by name...
print("PCA components:", pca.components_.shape)
print("logistic regression weights:", pipe[-1].coef_.round(2))   # ...or by position`, after: `Thirty features are scaled, compressed to five principal components, and classified, all with one \`fit\` and one \`score\`. Fitted steps keep their learned attributes, so you can inspect them through \`named_steps\` or by index.` },
      { type: 'code', title: "Grid-searching preprocessing and model settings together", md: `Preprocessing has settings too, and the best ones depend on the model's settings. In a pipeline, a parameter is addressed as **step name, two underscores, parameter name**: \`polynomialfeatures__degree\`, \`ridge__alpha\`. Here the degree of the polynomial features and ridge's alpha are tuned together on the diabetes data (442 patients, a disease-progression score as the target).`, code: `import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
grid = {"polynomialfeatures__degree": [1, 2, 3], "ridge__alpha": [0.1, 1, 10, 100, 1000]}
search = GridSearchCV(pipe, grid, cv=5).fit(X_train, y_train)

table = pd.DataFrame(search.cv_results_).pivot_table(
    index="param_polynomialfeatures__degree", columns="param_ridge__alpha", values="mean_test_score")
print("mean cross-validation R2 (rows: degree, columns: alpha)")
print(table.round(2).to_string())
print("best:", search.best_params_)
print(f"test R2: {search.score(X_test, y_test):.3f}")`, after: `The interaction is the point. Degree-3 features (285 of them) with weak regularization are a disaster (a negative R²: far worse than predicting the mean), but with strong regularization they become usable. Searching the two settings separately would miss that. On this small, noisy dataset the plain linear model still wins, which is a useful result in itself.` },
      { type: 'code', title: "Letting the search choose the model", md: `The steps of a pipeline are parameters too, so a grid search can choose **which** preprocessing and **which** model to use. Each dictionary in the list is one family of candidates; \`None\` skips a step.`, code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])
param_grid = [
    {"classifier": [SVC()], "preprocessing": [StandardScaler()],
     "classifier__gamma": [0.001, 0.01, 0.1, 1], "classifier__C": [0.1, 1, 10, 100]},
    {"classifier": [RandomForestClassifier(n_estimators=100, random_state=0)], "preprocessing": [None],
     "classifier__max_features": [1, 3, 5]},
]
search = GridSearchCV(pipe, param_grid, cv=5).fit(X_train, y_train)
print("winner:", search.best_params_["classifier"])
print("its settings:", {k: v for k, v in search.best_params_.items() if k.startswith("classifier__")})
print(f"cross-validation accuracy {search.best_score_:.3f}   test accuracy {search.score(X_test, y_test):.3f}")`, after: `The random forest candidates skip scaling (trees do not need it), while the SVM candidates always get it. The search compared 16 SVMs and 3 forests on equal terms and picked a scaled RBF-kernel SVM. Use this sparingly: comparing many model families inflates the selection bias discussed in [cross-validation](#/concept/cross-validation), so confirm the winner on the test set.` },
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
      { type: 'mcq', q: 'What should you save to deploy a model trained with scaling?', options: ['Only the classifier', 'The whole fitted pipeline (scaler + model)', 'Only the training data', 'The test set predictions'], answer: 1, explain: 'New data must be preprocessed exactly like the training data.' },
      {"type":"mcq","q":"In a pipeline built with `make_pipeline(StandardScaler(), SVC())`, how do you refer to the SVM's C parameter in a grid search?","options":["`C`","`svc.C`","`svc__C`","`SVC_C`"],"answer":2,"explain":"Pipeline parameters are named step__parameter, with two underscores; make_pipeline names the SVC step \"svc\"."},
      {"type":"mcq","q":"Which steps must be inside the cross-validation loop to avoid leakage?","options":["Only the model","Only scaling","Every step that learns anything from the data: scaling, imputation, encoding, feature selection, the model","None: leakage only matters for the test set"],"answer":2,"explain":"Any fitted step that sees validation data leaks information into the evaluation. A pipeline refits every step on each training fold."},
      {"type":"tf","q":"Every step of a scikit-learn pipeline except the last must have a `transform` method.","answer":true,"explain":"Intermediate steps pass transformed data along, so they must be transformers; the final step only needs `fit` (and usually `predict`)."}
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
      { title: 'scikit-learn — Pipelines and composite estimators', url: 'https://scikit-learn.org/stable/modules/compose.html' },
      { title: "Andreas Müller & Sarah Guido — Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
