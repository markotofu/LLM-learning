"""Chart data for the maths batch (M1). Apply AFTER patches/m1-text.txt (some anchors are new sections)."""
import json, sys, math
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
sig = lambda v, floor=1e-12: [float(f"{max(float(x), floor):.4g}") for x in v]     # significant figures, for log axes
out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))
pdf = lambda x, m=0, s=1: np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))
def beta_pdf(t, a, b):
    return np.exp((a - 1) * np.log(t) + (b - 1) * np.log(1 - t) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))

# ---------------- statistics
x = np.linspace(-4, 4, 161)
chart("statistics", {"after": "Spread: variance and standard deviation"}, "The normal distribution and the 68–95–99.7 rule",
      {"kind": "line", "x": r(x, 2), "points": False, "series": [{"name": "normal density, mean 0, standard deviation 1", "y": r(pdf(x))}],
       "xLabel": "distance from the mean, in standard deviations", "yLabel": "density",
       "vlines": [{"x": -2, "label": "−2σ"}, {"x": -1, "label": "−1σ"}, {"x": 1, "label": "+1σ"}, {"x": 2, "label": "+2σ"}], "yMin": 0},
      after="For bell-shaped data, about **68%** of values lie within one standard deviation of the mean, **95%** within two and **99.7%** within three. That is what makes the standard deviation interpretable, and why a value 3 standard deviations away is unusual. The rule only holds for roughly normal data; skewed data breaks it.")
rng = np.random.default_rng(0)
incomes = rng.lognormal(mean=10.6, sigma=0.6, size=20000) / 1000
edges = np.arange(0, 141, 10); counts, _ = np.histogram(incomes, edges)
chart("statistics", {"after": "Shape: skew, heavy tails and log transforms"}, "A right-skewed distribution: incomes",
      {"kind": "bar", "x": [str(a + 5) for a in edges[:-1]], "series": [{"name": "number of people (of 20,000)", "y": [int(c) for c in counts]}],
       "xLabel": "income in thousands (centre of each 10k-wide bin)", "yLabel": "people"},
      after=f"Simulated incomes: most people earn a moderate amount and a few earn a lot (about {np.mean(incomes > 140):.0%} earn more than 140k and are off the chart). The mean ({incomes.mean():.0f}k) sits to the right of the median ({np.median(incomes):.0f}k), pulled by the long tail. For skewed data like this, report the median, and consider a log transform before modelling.")
draws = rng.exponential(1.0, size=(200000, 30))
grid = np.linspace(0, 3, 61); series = []
for n in [1, 5, 30]:
    means = draws[:, :n].mean(axis=1)
    h, e = np.histogram(means, bins=grid, density=True)
    series.append({"name": f"mean of {n} value{'s' if n > 1 else ''}", "y": r(h, 3)})
chart("statistics", {"after": "Samples, noise and the standard error"}, "The central limit theorem in action",
      {"kind": "line", "x": r((grid[:-1] + grid[1:]) / 2, 3), "points": False, "series": series, "xLabel": "value of the sample mean", "yLabel": "density"},
      md="Start from a strongly skewed distribution (exponential, mean 1), draw samples of size n, and plot the distribution of their **means**:",
      after="A single value (n = 1) follows the skewed original. Averages of 5 values are already more symmetric, and averages of 30 are close to a bell curve centred on the true mean, with a spread of σ / √n. This is the **central limit theorem**: averages of many independent values are approximately normal, whatever the original distribution. It is why confidence intervals for averages and accuracies work.")

# ---------------- stat-inference
ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
width = [2 * 1.96 * math.sqrt(0.8 * 0.2 / n) for n in ns]
chart("stat-inference", {"after": "Confidence intervals"}, "How wide is a 95% confidence interval for an accuracy of 80%?",
      {"kind": "line", "x": ns, "xLog": True, "series": [{"name": "full width of the 95% interval", "y": r(width)}], "xLabel": "number of test examples (log scale)", "yLabel": "interval width (accuracy points)"},
      after="With 100 test examples, an accuracy of 0.80 is really \"somewhere between 0.72 and 0.88\": two models that differ by 3 points cannot be told apart. The width shrinks like 1 / √n, so halving it needs four times as many examples.")
from math import erf
Phi = lambda z: 0.5 * (1 + erf(z / math.sqrt(2)))
n_per = [500, 1000, 2000, 5000, 10000, 20000, 40000]
def power(n, p1=0.10, p2=0.11):
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    return Phi(abs(p2 - p1) / se - 1.96)
chart("stat-inference", {"after": "Designing A/B tests: power and sample size"}, "Power of an A/B test against sample size",
      {"kind": "line", "x": n_per, "xLog": True, "series": [{"name": "chance of detecting a lift from 10% to 11%", "y": r([power(n) for n in n_per], 3)}],
       "xLabel": "users per group (log scale)", "yLabel": "power", "hlines": [{"y": 0.8, "label": "the usual target: 80%"}], "yMin": 0, "yMax": 1},
      after="A one-point lift on a 10% conversion rate is a big business win but a small statistical effect: with 2,000 users per group the test would miss it most of the time. Reaching 80% power needs about 15,000 users per group. Run this calculation **before** the experiment, not after.")

# ---------------- mle-map
th = np.linspace(0.01, 0.99, 99)
def lik(h, n):
    l = h * np.log(th) + (n - h) * np.log(1 - th); l = np.exp(l - l.max()); return l
chart("mle-map", {"after": "Example: estimating a coin"}, "The likelihood of a coin's bias, after 10 and after 100 flips",
      {"kind": "line", "x": r(th, 2), "points": False, "series": [{"name": "7 heads in 10 flips", "y": r(lik(7, 10), 3)}, {"name": "70 heads in 100 flips", "y": r(lik(70, 100), 3)}],
       "xLabel": "θ, the probability of heads", "yLabel": "likelihood (scaled so the peak is 1)", "vlines": [{"x": 0.7, "label": "MLE θ = 0.7"}], "yMin": 0},
      after="Both curves peak at the same place, θ = 0.7: the maximum-likelihood estimate is the observed share of heads. But with 100 flips the curve is far narrower: many values that were plausible after 10 flips (0.5 or 0.9) are now very unlikely. The width of the likelihood is a measure of uncertainty, which is the bridge to confidence intervals and to Bayesian posteriors.")
t = np.linspace(0.005, 0.995, 199)
chart("mle-map", {"after": "MAP: adding prior beliefs"}, "Prior × likelihood = posterior, for a coin",
      {"kind": "line", "x": r(t, 3), "points": False, "series": [
          {"name": "prior Beta(5, 5): probably fair", "y": r(beta_pdf(t, 5, 5), 3), "dash": True},
          {"name": "likelihood of 9 heads in 10 (normalised)", "y": r(beta_pdf(t, 10, 2), 3)},
          {"name": "posterior Beta(14, 6)", "y": r(beta_pdf(t, 14, 6), 3)}],
       "xLabel": "θ, the probability of heads", "yLabel": "density", "yMin": 0,
       "vlines": [{"x": 0.9, "label": "MLE 0.9"}, {"x": round(13 / 18, 3), "label": "MAP 0.72"}]},
      md="A prior that says \"coins are usually close to fair\" meets data with 9 heads in 10 flips:",
      after="The posterior is a compromise: it sits between the prior's centre (0.5) and the data's verdict (0.9), and it is narrower than either because it combines both sources of information. Its peak, the MAP estimate, is (9 + 5 − 1) / (10 + 5 + 5 − 2) = 13/18 ≈ 0.72. With 1,000 flips the likelihood would be so narrow that the prior would hardly move it.")

# ---------------- eigen-svd
def angle_err(A, x0, top, steps):
    x, errs = x0 / np.linalg.norm(x0), []
    for k in range(steps):
        errs.append(abs(math.sin(math.acos(min(1.0, abs(float(x @ top)))))))
        x = A @ x; x = x / np.linalg.norm(x)
    return errs
def rot(deg):
    a = math.radians(deg); return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
steps = 41
A1 = rot(30) @ np.diag([3.0, 1.0]) @ rot(30).T
A2 = rot(30) @ np.diag([1.0, 0.9]) @ rot(30).T
top = rot(30) @ np.array([1.0, 0.0])
x0 = rot(30 + 60) @ np.array([1.0, 0.0])                   # start 60° away from the top eigenvector
chart("eigen-svd", {"after": "Eigenvectors with NumPy and power iteration"}, "How fast power iteration converges",
      {"kind": "line", "x": list(range(steps)), "yLog": True, "points": False, "series": [
          {"name": "eigenvalues 3 and 1 (ratio 0.33)", "y": sig(angle_err(A1, x0, top, steps), 1e-12)},
          {"name": "eigenvalues 1 and 0.9 (ratio 0.9)", "y": sig(angle_err(A2, x0, top, steps), 1e-12)}],
       "xLabel": "iteration", "yLabel": "error: sin(angle to the top eigenvector)", "yMin": 1e-12},
      after="Each iteration shrinks the error by the ratio |λ₂ / λ₁| of the two largest eigenvalues. With a ratio of 1/3 the error falls by a factor of 10 every two steps; with 0.9 it takes about 22 steps per factor of 10. A large **eigengap** makes power iteration (and PageRank, and PCA solvers built on it) fast.")
rng = np.random.default_rng(3)
M = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 40)) + 0.3 * rng.normal(size=(60, 40))
U, S, Vt = np.linalg.svd(M, full_matrices=False)
ks = list(range(1, 21))
err = [np.linalg.norm(M - (U[:, :k] * S[:k]) @ Vt[:k]) / np.linalg.norm(M) for k in ks]
chart("eigen-svd", {"after": "Rank and low-rank approximation"}, "Singular values of a nearly low-rank matrix",
      {"kind": "line", "x": ks, "yLog": True, "series": [{"name": "singular value σₖ", "y": r(S[:20], 3)}], "xLabel": "k (singular values in decreasing order)", "yLabel": "σₖ (log scale)"},
      md="A 60 × 40 matrix built from 4 hidden factors plus a little noise. Plotting its singular values in order (a *scree plot*) reveals the hidden rank:",
      after="Four large singular values, then a sudden drop to a flat floor of small ones that only describe the noise. The elbow tells you how many components are worth keeping, which is exactly how you choose the number of components in [PCA](#/concept/pca).")
chart("eigen-svd", {"after": "Singular values of a nearly low-rank matrix"}, "Reconstruction error against rank",
      {"kind": "line", "x": ks, "series": [{"name": "relative error of the rank-k approximation", "y": r(err, 3)}], "xLabel": "rank k of the approximation", "yLabel": "‖M − Mₖ‖ / ‖M‖", "vlines": [{"x": 4, "label": "true rank 4"}], "yMin": 0},
      after="Keeping 4 components already captures almost everything; every extra component only shaves off a little noise. Storing the rank-4 version takes 4 · (60 + 40 + 1) = 404 numbers instead of 2,400.")

# ---------------- probability
from math import comb
k = list(range(0, 21)); pmf = [comb(20, i) * 0.3 ** i * 0.7 ** (20 - i) for i in k]
chart("probability", {"after": "A field guide to common distributions"}, "A binomial distribution: heads in 20 tosses of a coin with P(heads) = 0.3",
      {"kind": "bar", "x": [str(i) for i in k], "series": [{"name": "probability", "y": r(pmf)}], "xLabel": "number of heads", "yLabel": "probability"},
      after="The most likely count is 6 (= 20 × 0.3), but 4, 5, 7 and 8 are all common: a single experiment rarely lands exactly on its expected value. The bars add up to 1.")
rng = np.random.default_rng(5); flips = rng.random(100000) < 0.5
ns = np.unique(np.round(np.logspace(0, 5, 60)).astype(int)); running = np.cumsum(flips)[ns - 1] / ns
chart("probability", {"after": "Simulating randomness with NumPy"}, "The law of large numbers: the share of heads settles down",
      {"kind": "line", "x": [int(v) for v in ns], "xLog": True, "points": False, "series": [{"name": "share of heads so far", "y": r(running, 4)}], "xLabel": "number of tosses (log scale)", "yLabel": "share of heads", "hlines": [{"y": 0.5, "label": "true probability 0.5"}], "yMin": 0, "yMax": 1},
      after="After a few tosses the share of heads swings wildly; after thousands it hugs 0.5. Frequencies converge to probabilities, slowly: the typical deviation shrinks like 1 / √n.")

# ---------------- information theory
p = np.linspace(0.001, 0.999, 100)
H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
chart("information-theory", {"after": "Entropy: average surprise"}, "Entropy of a coin with P(heads) = p",
      {"kind": "line", "x": r(p, 3), "points": False, "series": [{"name": "entropy H(p) in bits", "y": r(H, 4)}], "xLabel": "p, the probability of heads", "yLabel": "entropy (bits)", "yMin": 0},
      after="A fair coin (p = 0.5) is the most uncertain two-outcome variable: exactly 1 bit. As the coin becomes predictable, entropy falls towards 0; a coin that always lands heads carries no information at all.")
q = np.linspace(0.01, 1, 100)
chart("information-theory", {"after": "Cross-entropy: the classification loss"}, "The loss −log q for the probability q given to the correct answer",
      {"kind": "line", "x": r(q, 2), "points": False, "series": [{"name": "cross-entropy loss −ln q (nats)", "y": r(-np.log(q), 4)}],
       "xLabel": "q, the probability the model gave the correct class", "yLabel": "loss (nats)", "yMin": 0, "hlines": [{"y": round(math.log(2), 3), "label": "a 50/50 guess: 0.69"}]},
      after="Near q = 1 the curve is almost flat: making a confident correct answer even more confident gains little. Towards q = 0 it shoots up: going from 0.1 to 0.01 adds as much loss (2.3 nats) as going from 1.0 to 0.1. This steepness is what makes confident mistakes so expensive.")

# ---------------- gradient descent
steps = list(range(0, 31)); series = []
for lr, name in [(0.01, "η = 0.01 (too small)"), (0.1, "η = 0.1 (good)"), (0.45, "η = 0.45 (fast)"), (0.95, "η = 0.95 (oscillates)"), (1.05, "η = 1.05 (diverges)")]:
    w, ls = 0.0, []
    for s_ in steps:
        ls.append(min((w - 3) ** 2, 1e3)); w -= lr * 2 * (w - 3)
    series.append({"name": name, "y": sig(ls, 1e-8)})
chart("gradient-descent", {"after": "Learning rates on a simple bowl"}, "Loss per step for different learning rates",
      {"kind": "line", "x": steps, "yLog": True, "points": False, "series": series, "xLabel": "step", "yLabel": "loss (w − 3)² (log scale)", "yMin": 1e-8, "yMax": 1000},
      md="The loss (w − 3)², starting from w = 0, minimised with five different learning rates:",
      after="On a log scale, steady progress is a straight downward line. Too small a step crawls; a well-chosen one converges in a few steps (0.45 reaches the bottom of the chart in four); 0.95 overshoots back and forth but still converges slowly; past the critical value (here 1.0) every step overshoots by more than the last and the loss explodes. In real training you watch exactly this curve.")
# zig-zag on an elongated bowl: contours (one series with gaps) plus the path
cx, cy = [], []
for L in [1, 5, 20, 50, 100]:
    tt = np.linspace(0, 2 * math.pi, 61)
    cx += list(np.sqrt(L) * np.cos(tt)) + [None]; cy += list(np.sqrt(L / 10) * np.sin(tt)) + [None]
w = np.array([-9.0, 1.5]); px, py = [], []
for s_ in range(16):
    px.append(w[0]); py.append(w[1]); w = w - 0.09 * np.array([2 * w[0], 20 * w[1]])
chart("gradient-descent", {"after": "Gradients in many dimensions"}, "Gradient descent zig-zags on an elongated bowl",
      {"kind": "line", "series": [
          {"name": "contours of w₁² + 10·w₂² (loss 1, 5, 20, 50, 100)", "x": r(cx, 3), "y": r(cy, 3), "dash": True},
          {"name": "15 steps of gradient descent, η = 0.09", "x": r(px, 3), "y": r(py, 3)}],
       "xLabel": "w₁ (the shallow direction)", "yLabel": "w₂ (the steep direction)", "xMin": -10.5, "xMax": 10.5, "yMin": -3.3, "yMax": 3.3},
      md="Fifteen steps from w = (−9, 1.5) with η = 0.09, just under the stability limit of 0.1:",
      after="Each step is perpendicular to the contour it starts on, so instead of heading for the centre the path bounces between the steep walls (w₂ flips sign every step) while creeping along the valley floor in w₁. After 15 steps w₁ is still at −0.46. Rescaling w₂, or an optimizer with momentum, would remove the zig-zag.")
# batch vs mini-batch progress, measured per pass over the data
rng = np.random.default_rng(0)
n = 1000
X = rng.uniform(-1, 1, size=(n, 2)); yv = X @ np.array([3.0, -2.0]) + 0.5 + rng.normal(0, 0.1, size=n)
def run(batch, every):
    w, b, xs, ys, step = np.zeros(2), 0.0, [0.0], [float(np.mean((X @ np.zeros(2) - yv) ** 2))], 0
    r2 = np.random.default_rng(1)
    for epoch in range(5):
        perm = r2.permutation(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]; err_ = X[idx] @ w + b - yv[idx]
            w = w - 0.1 * 2 * X[idx].T @ err_ / len(idx); b -= 0.1 * 2 * err_.mean(); step += 1
            if step % every == 0:
                xs.append(round(epoch + (start + len(idx)) / n, 4)); ys.append(float(np.mean((X @ w + b - yv) ** 2)))
    return xs, ys
series = []
for batch, every, name in [(1000, 1, "full batch (1 update per pass)"), (32, 1, "mini-batches of 32"), (1, 25, "single examples (SGD)")]:
    xs, ys = run(batch, every); series.append({"name": name, "x": xs, "y": sig(ys)})
chart("gradient-descent", {"after": "Mini-batch SGD for linear regression"}, "Progress per pass over the data: full batch vs mini-batches",
      {"kind": "line", "yLog": True, "points": False, "series": series, "xLabel": "epochs (passes over the 1,000 training examples)", "yLabel": "full-data MSE (log scale)",
       "hlines": [{"y": 0.01, "label": "noise floor (σ² = 0.01)"}]},
      after="With the same learning rate, full-batch gradient descent has made only 5 updates after 5 passes and is still far from the minimum. Mini-batches of 32 make 32 updates per pass and reach the noise floor within about one epoch. Single-example SGD gets there sooner still, but its loss never settles: each step follows one noisy example, which is why its learning rate is usually decayed over time.")

# ---------------- loss functions
e = np.linspace(-3, 3, 121)
hub = np.where(np.abs(e) <= 1, 0.5 * e ** 2, np.abs(e) - 0.5)
chart("loss-functions", {"after": "Regression losses"}, "Regression losses as a function of the error",
      {"kind": "line", "x": r(e, 2), "points": False, "series": [{"name": "squared error e²", "y": r(e ** 2, 4)}, {"name": "absolute error |e|", "y": r(np.abs(e), 4)}, {"name": "Huber (δ = 1)", "y": r(hub, 4)}],
       "xLabel": "error e = prediction − target", "yLabel": "loss", "yMax": 9},
      after="Squared error grows quadratically, so a single large error dominates the total: great when errors are Gaussian, bad with outliers. Absolute error grows linearly and is robust, but has a kink at 0. Huber is quadratic near zero and linear far away: the best of both.")
m = np.linspace(-2, 3, 101)
expo = [float(v) if v <= 4 else None for v in np.exp(-m)]
chart("loss-functions", {"after": "Classification losses"}, "Classification losses as a function of the margin",
      {"kind": "line", "x": r(m, 2), "points": False, "series": [
          {"name": "0-1 loss (error)", "y": [1.0 if v < 0 else 0.0 for v in m]},
          {"name": "hinge max(0, 1 − m)", "y": r(np.maximum(0, 1 - m), 4)},
          {"name": "log loss log₂(1 + e⁻ᵐ)", "y": r(np.log2(1 + np.exp(-m)), 4)},
          {"name": "exponential e⁻ᵐ (AdaBoost)", "y": r(expo, 4)}],
       "xLabel": "margin m = y · score (positive = correct side)", "yLabel": "loss", "yMin": 0, "vlines": [{"x": 0, "label": "decision boundary"}]},
      md="For a label y ∈ {−1, +1} and a raw score s, the **margin** m = y · s is positive when the prediction is on the correct side and large when it is confidently correct. Every classification loss is a function of it:",
      after="The 0-1 loss is a step: flat everywhere except at the boundary, so it has no useful gradient. The three surrogates are convex and lie on or above it, so pushing them down also pushes the error rate down. Hinge loss stops caring once the margin passes 1; log loss keeps rewarding larger margins, ever more weakly; the exponential loss punishes badly wrong points so hard that it is sensitive to mislabelled data.")
pt = np.linspace(0.01, 1, 100)
chart("loss-functions", {"after": "Losses for imbalanced data and overconfident models"}, "Focal loss against cross-entropy",
      {"kind": "line", "x": r(pt, 2), "points": False, "series": [
          {"name": "cross-entropy (γ = 0)", "y": r(-np.log(pt), 4)},
          {"name": "focal, γ = 1", "y": r(-(1 - pt) * np.log(pt), 4)},
          {"name": "focal, γ = 2", "y": r(-(1 - pt) ** 2 * np.log(pt), 4)}],
       "xLabel": "p_t, the probability given to the true class", "yLabel": "loss", "yMin": 0},
      after="For an easy example (p_t = 0.9) the γ = 2 focal loss is 100 times smaller than cross-entropy, while for a badly misclassified one (p_t = 0.1) it is only about 20% smaller. Thousands of easy examples no longer drown out the few hard ones.")
tau = [0.1, 0.5, 0.9]
chart("loss-functions", {"after": "The quantile loss: predicting a range, not a point"}, "The quantile (pinball) loss",
      {"kind": "line", "x": r(e, 2), "points": False, "series": [{"name": f"τ = {t}", "y": r(np.where(e >= 0, (1 - t) * e, -t * e), 4)} for t in tau],
       "xLabel": "error e = prediction − target", "yLabel": "loss"},
      after="Read the τ = 0.9 line: predicting too low (left) costs 9 times more per unit than predicting too high (right), so the best prediction is high enough that 90% of outcomes fall below it. τ = 0.5 is symmetric: half the absolute error, which targets the median.")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print(len(out), "charts")
