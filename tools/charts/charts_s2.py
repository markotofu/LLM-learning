"""Chart data for batch S2 (RL, deep RL, MLOps, responsible AI, ML security). Apply AFTER patches/s2-text.txt."""
import json, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
out, diag = [], []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- RL: epsilon-greedy learning curves on the lesson's 5-armed bandit
true_means = np.array([0.2, 0.5, 0.9, 0.4, 0.1]); runs, steps = 2000, 1000
rng = np.random.default_rng(0); curves = {}
for eps in (0.0, 0.01, 0.1, 0.3):
    Q = np.zeros((runs, 5)); N = np.zeros((runs, 5)); best = np.zeros(steps); rewards = np.zeros(steps)
    for t in range(steps):
        explore = rng.random(runs) < eps
        greedy = np.argmax(Q + 1e-6 * rng.random((runs, 5)), axis=1)
        a = np.where(explore, rng.integers(0, 5, runs), greedy)
        rwd = rng.normal(true_means[a], 1.0)
        N[np.arange(runs), a] += 1; Q[np.arange(runs), a] += (rwd - Q[np.arange(runs), a]) / N[np.arange(runs), a]
        best[t] = np.mean(a == 2); rewards[t] = rwd.mean()
    curves[eps] = best
diag.append(("bandit", {e: round(float(c[-100:].mean()), 3) for e, c in curves.items()}))
xs = list(range(0, steps, 10))
chart("reinforcement-learning", {"after": "ε-greedy on a 5-armed bandit"}, "How often each strategy picks the best arm",
      {"kind": "line", "x": [x + 1 for x in xs], "points": False, "series": [{"name": f"ε = {e}", "y": r([c[x:x + 10].mean() for x in xs], 3)} for e, c in curves.items()],
       "xLabel": "step", "yLabel": "share of 2,000 runs choosing the best arm", "yMin": 0, "yMax": 1},
      md="The 5-armed bandit above (rewards are noisy: standard deviation 1), averaged over 2,000 independent runs:",
      after=f"The purely greedy agent (ε = 0) locks onto whichever arm looked good early and finds the best arm in only about {curves[0.0][-100:].mean():.0%} of runs. Exploring 10% of the time finds it far more reliably and soon picks it about {curves[0.1][-100:].mean():.0%} of the time, the most it can while still exploring 10% of the time. ε = 0.01 learns slowly but would end up higher in the long run, and ε = 0.3 explores too much. Annealing ε from high to low, or exploring by uncertainty (UCB, Thompson sampling), gets the best of both.")

# ---------------- deep RL: overestimation from max over noisy estimates
rng = np.random.default_rng(1); ks = [1, 2, 3, 5, 10, 20, 50]; single, double = [], []
for k in ks:
    est_a = rng.normal(0, 1, (20000, k)); est_b = rng.normal(0, 1, (20000, k))     # two independent noisy estimates, true value 0
    single.append(est_a.max(axis=1).mean())
    pick = est_a.argmax(axis=1); double.append(est_b[np.arange(20000), pick].mean())
diag.append(("overest", r(single, 3), r(double, 3)))
chart("deep-rl", {"after": "Why max-based targets overestimate"}, "The bias of taking the maximum of noisy estimates",
      {"kind": "line", "x": ks, "xLog": True, "series": [{"name": "max of the estimates (Q-learning / DQN target)", "y": r(single, 3)}, {"name": "pick with one estimate, evaluate with another (double Q-learning)", "y": r(double, 3)}],
       "xLabel": "number of actions (log scale)", "yLabel": "estimated value of the best action (true value 0)", "hlines": [{"y": 0, "label": ""}]},
      md="Every action is truly worth 0, but each value is estimated with noise of standard deviation 1:",
      after=f"Taking the maximum of noisy estimates systematically picks whichever estimate happened to be too high: with 10 actions the target is about {single[4]:.1f} too high, and the error grows with the number of actions. In Q-learning these inflated targets feed back into later updates. Double Q-learning chooses the action with one estimate and evaluates it with an independent one, which removes the bias (the second line stays at 0); Double DQN applies this with the online and target networks.")

# ---------------- MLOps: PSI against the size of a shift
rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 20000); edges = np.quantile(ref, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
def psi(cur):
    e = np.histogram(ref, edges)[0] / len(ref); a = np.histogram(cur, edges)[0] / len(cur)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None); return float(np.sum((a - e) * np.log(a / e)))
shifts = np.round(np.arange(0, 1.01, 0.1), 2); vals = [psi(rng.normal(s_, 1, 5000)) for s_ in shifts]
diag.append(("psi", r(vals, 3)))
chart("mlops", {"after": "Detecting drift with PSI"}, "PSI as an input distribution shifts",
      {"kind": "line", "x": shifts.tolist(), "series": [{"name": "PSI of the shifted data against the reference", "y": r(vals, 3)}], "xLabel": "shift of the mean, in standard deviations", "yLabel": "population stability index",
       "hlines": [{"y": 0.1, "label": "0.1: watch"}, {"y": 0.25, "label": "0.25: investigate"}], "yMin": 0},
      md="A feature whose distribution shifts by a growing amount, compared with its training-time reference (10 quantile bins, 5,000 new values):",
      after=f"PSI grows roughly with the square of the shift. By the usual rules of thumb, a shift of about 0.4 standard deviations crosses the \"watch\" line (PSI ≈ {vals[4]:.2f}) and one of about half a standard deviation reaches the \"investigate\" line (PSI ≈ {vals[5]:.2f}). The thresholds are conventions, not laws: tune them per feature, and remember that drift in an unimportant feature may not hurt the model at all, while a small drift in a key feature might.")

# ---------------- responsible AI: one threshold, two groups
rng = np.random.default_rng(0)
score_a = 1 / (1 + np.exp(-rng.normal(0.6, 1.0, 20000))); score_b = 1 / (1 + np.exp(-rng.normal(-0.2, 1.0, 20000)))
ths = np.round(np.arange(0.2, 0.81, 0.05), 2)
chart("responsible-ai", {"after": "Group fairness metrics"}, "Selection rates of two groups under one threshold",
      {"kind": "line", "x": ths.tolist(), "series": [{"name": "group A approved", "y": r([np.mean(score_a >= t) for t in ths], 3)}, {"name": "group B approved", "y": r([np.mean(score_b >= t) for t in ths], 3)}],
       "xLabel": "approval threshold on the model's score", "yLabel": "share approved", "yMin": 0, "yMax": 1, "vlines": [{"x": 0.5, "label": "a typical threshold"}]},
      md="Model scores whose distribution differs between two groups, approved whenever the score passes a single threshold:",
      after=f"At any single threshold the two groups are approved at different rates: at 0.5, {np.mean(score_a >= 0.5):.0%} of group A against {np.mean(score_b >= 0.5):.0%} of group B. Whether that gap is unfair depends on why the scores differ, which is exactly what the fairness criteria disagree about: demographic parity would demand equal approval rates (different thresholds per group), while equal opportunity looks only at people who would have repaid.")

# ---------------- ML security: FGSM against random noise of the same size
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
X, y = load_digits(return_X_y=True); X = X / 16.0
sel = (y == 3) | (y == 8); X, y = X[sel], (y[sel] == 8).astype(int)
rng = np.random.default_rng(0); idx = rng.permutation(len(X)); tr, te = idx[:250], idx[250:]
clf = LogisticRegression(max_iter=2000, C=10).fit(X[tr], y[tr]); w = clf.coef_[0]
epss = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]; adv, noise = [], []
for e in epss:
    s = np.where(y[te] == 1, -1, 1)[:, None]                      # push each example towards the wrong class
    Xa = np.clip(X[te] + e * s * np.sign(w), 0, 1); adv.append(clf.score(Xa, y[te]))
    Xn = np.clip(X[te] + e * rng.choice([-1, 1], X[te].shape), 0, 1); noise.append(clf.score(Xn, y[te]))
diag.append(("fgsm", r(adv, 3), r(noise, 3)))
chart("ml-security", {"after": "FGSM against a linear classifier"}, "Accuracy under adversarial and random perturbations of the same size",
      {"kind": "line", "x": epss, "series": [{"name": "random ±ε noise on every pixel", "y": r(noise, 3)}, {"name": "FGSM: ε in the worst direction", "y": r(adv, 3)}], "xLabel": "ε: maximum change per pixel (pixels range from 0 to 1)", "yLabel": "accuracy (3 versus 8)", "yMin": 0, "yMax": 1},
      md="A logistic-regression classifier telling handwritten 3s from 8s, with every pixel changed by at most ε:",
      after=f"Random changes of size ε barely matter: they mostly cancel out in the weighted sum. The same budget spent in the direction of the gradient's sign moves every pixel against the classifier at once, and accuracy falls from {adv[0]:.0%} to {adv[3]:.0%} at ε = 0.1 and to {adv[5]:.0%} at ε = 0.2, where random changes of the same size still leave it at {noise[5]:.0%}. Adversarial examples are not about large noise but about many small changes that all point the same way, which is why high-dimensional inputs are so vulnerable.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts")
