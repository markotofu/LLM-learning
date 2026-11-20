"""Chart data for batch C1 (PCA, k-means, GMM, GPs, Bayesian). Apply AFTER patches/c1b-text.txt."""
import json, sys, math
import numpy as np
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- PCA on digits
X, y = load_digits(return_X_y=True); X = X / 16.0
pca = PCA().fit(X); cum = np.cumsum(pca.explained_variance_ratio_); k90 = int(np.searchsorted(cum, 0.90) + 1)
chart("pca", {"after": "PCA on handwritten digits"}, "Cumulative explained variance of the digit images",
      {"kind": "line", "x": list(range(1, 65)), "points": False, "series": [{"name": "share of the variance kept by the first k components", "y": r(cum, 4)}],
       "xLabel": "number of components k (of 64)", "yLabel": "cumulative explained variance", "yMin": 0, "yMax": 1,
       "hlines": [{"y": 0.9, "label": "90%"}], "vlines": [{"x": k90, "label": f"k = {k90}"}]},
      after=f"The curve rises steeply and then flattens: the first 10 components keep {cum[9]:.0%} of the variance, {k90} keep 90%, and the last 20 or so add almost nothing (some pixels at the image border are always blank). Reading the elbow of this curve is the usual way to choose k.")
Z = PCA(n_components=2).fit_transform(X)
series = []
for digit in (0, 1, 4, 7):
    idx = np.where(y == digit)[0][:90]
    series.append({"name": f"digit {digit}", "x": r(Z[idx, 0], 3), "y": r(Z[idx, 1], 3)})
chart("pca", {"after": "Cumulative explained variance of the digit images"}, "The digits projected onto the first two components",
      {"kind": "scatter", "series": series, "xLabel": "score on principal component 1", "yLabel": "score on principal component 2"},
      md="Each dot is one 64-pixel image, placed by its scores on the first two components (four of the ten digits shown):",
      after="Even two directions, chosen without ever looking at the labels, separate the zeros, ones, fours and sevens into recognisable regions. Where clouds overlap, two components are not enough, which matches the low accuracy of the 2-component classifier above.")

# ---------------- k-means: elbow, silhouette, moons
rng = np.random.default_rng(0)
Xb = np.vstack([rng.normal(c, 0.6, (50, 2)) for c in ([0, 0], [4, 0], [2, 3.5])])
ks = list(range(1, 9)); inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(Xb); inertia.append(km.inertia_)
    if k >= 2: sil.append(silhouette_score(Xb, km.labels_))
chart("kmeans", {"after": "Initialization, choosing k, and limitations"}, "The elbow: inertia against k",
      {"kind": "line", "x": ks, "series": [{"name": "inertia (best of 10 starts)", "y": r(inertia, 1)}], "xLabel": "number of clusters k", "yLabel": "inertia", "vlines": [{"x": 3, "label": "elbow at k = 3"}], "yMin": 0},
      md="The three-blob data from the code above, clustered with k = 1 to 8:",
      after="Inertia always falls as k grows, but it falls steeply only until k reaches the true number of groups. After the elbow each extra cluster merely splits a real group in two and buys little.")
chart("kmeans", {"after": "The elbow: inertia against k"}, "The silhouette score against k",
      {"kind": "bar", "x": [str(k) for k in ks[1:]], "series": [{"name": "average silhouette (higher is better)", "y": r(sil, 3)}], "xLabel": "number of clusters k", "yLabel": "silhouette score", "yMin": 0, "yMax": 1},
      after="Unlike inertia, the silhouette does not automatically improve with more clusters, so its peak can be read directly: k = 3 scores clearly best (about 0.71).")
Xm, ym = make_moons(300, noise=0.06, random_state=0)
km = KMeans(2, n_init=10, random_state=0).fit(Xm)
series = [{"name": f"k-means cluster {j + 1}", "x": r(Xm[km.labels_ == j, 0], 3), "y": r(Xm[km.labels_ == j, 1], 3)} for j in range(2)]
series.append({"name": "centroids", "x": r(km.cluster_centers_[:, 0], 3), "y": r(km.cluster_centers_[:, 1], 3)})
chart("kmeans", {"after": "When k-means fails, and what to use instead"}, "k-means on two interleaved moons",
      {"kind": "scatter", "series": series, "xLabel": "x₁", "yLabel": "x₂"},
      after="Each moon is one true group, but k-means can only split the plane with the straight line halfway between two centroids, so each \"cluster\" takes the tip of the other moon. No initialisation fixes this: the shape violates k-means' assumptions. DBSCAN or single-linkage clustering separates the moons cleanly.")

# ---------------- GMM: 1-D fit and BIC
rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(-2, 0.8, 300), rng.normal(3, 1.5, 200)])
def npdf(x, mu, s): return np.exp(-(x - mu) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
pi, mu, sigma = np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])
for it in range(40):
    dens = pi * npdf(x[:, None], mu, sigma); rr = dens / dens.sum(axis=1, keepdims=True); Nk = rr.sum(axis=0)
    pi, mu = Nk / len(x), (rr * x[:, None]).sum(axis=0) / Nk; sigma = np.sqrt((rr * (x[:, None] - mu) ** 2).sum(axis=0) / Nk)
edges = np.arange(-5, 9.01, 0.5); h, _ = np.histogram(x, edges, density=True)
sx, sy = [], []
for i in range(len(h)): sx += [edges[i], edges[i + 1]]; sy += [h[i], h[i]]
grid = np.linspace(-5, 9, 141)
c1, c2 = pi[0] * npdf(grid, mu[0], sigma[0]), pi[1] * npdf(grid, mu[1], sigma[1])
chart("gmm-em", {"after": "EM for a one-dimensional mixture"}, "The fitted mixture against the data",
      {"kind": "line", "points": False, "series": [
          {"name": "histogram of the data (as a density)", "x": r(sx, 2), "y": r(sy, 4)},
          {"name": "fitted mixture p(x)", "x": r(grid, 2), "y": r(c1 + c2, 4)},
          {"name": "component 1 × π₁", "x": r(grid, 2), "y": r(c1, 4), "dash": True},
          {"name": "component 2 × π₂", "x": r(grid, 2), "y": r(c2, 4), "dash": True}],
       "xLabel": "x", "yLabel": "density", "yMin": 0},
      after=f"The mixture density (the sum of the two dashed bells) follows the histogram closely. EM found weights {pi[0]:.2f} and {pi[1]:.2f}, means {mu[0]:.2f} and {mu[1]:.2f}, and spreads {sigma[0]:.2f} and {sigma[1]:.2f}, close to the 0.6 / 0.4, −2 / 3 and 0.8 / 1.5 used to generate the data. Where the dashed curves overlap, around x = 0.5, a point gets split responsibilities.")
Xg, _ = make_blobs(600, centers=[[0, 0], [5, 1], [2, 5]], cluster_std=[0.8, 1.2, 0.6], random_state=1)
bics = [GaussianMixture(k, random_state=0).fit(Xg).bic(Xg) for k in range(1, 7)]
chart("gmm-em", {"after": "How many components? BIC"}, "BIC against the number of components",
      {"kind": "line", "x": list(range(1, 7)), "series": [{"name": "BIC (lower is better)", "y": r(bics, 1)}], "xLabel": "number of components K", "yLabel": "BIC", "vlines": [{"x": 3, "label": "minimum: K = 3"}]},
      md="Data drawn from three Gaussian blobs, fitted with 1 to 6 components:",
      after="The likelihood keeps improving with every component, but after K = 3 the improvement no longer pays for the extra parameters, so the BIC turns upwards and picks the true number of blobs.")

# ---------------- Gaussian processes
def rbf(a, b, ell=1.0, sf=1.0): return sf ** 2 * np.exp(-(a[:, None] - b[None, :]) ** 2 / (2 * ell ** 2))
xt = np.array([-4.0, -3.0, -1.5, 0.0, 0.5, 2.5, 3.0, 3.5]); rng = np.random.default_rng(2); yt = np.sin(xt) + rng.normal(0, 0.1, len(xt))
grid = np.linspace(-5, 5, 101)
def posterior(ell, sn=0.1):
    K = rbf(xt, xt, ell) + sn ** 2 * np.eye(len(xt)); Ks = rbf(xt, grid, ell); Kss = rbf(grid, grid, ell)
    m = Ks.T @ np.linalg.solve(K, yt); v = np.diag(Kss - Ks.T @ np.linalg.solve(K, Ks)); return m, np.sqrt(np.maximum(v, 0))
m, s = posterior(1.0)
chart("gaussian-processes", {"after": "Conditioning on data: the GP posterior"}, "A GP posterior: mean and 95% band",
      {"kind": "line", "points": False, "series": [
          {"name": "posterior mean", "x": r(grid, 2), "y": r(m, 3)},
          {"name": "95% band (mean ± 2σ)", "x": r(list(grid) + [None] + list(grid), 2), "y": r(list(m + 2 * s) + [None] + list(m - 2 * s), 3), "dash": True},
          {"name": "true function sin(x)", "x": r(grid, 2), "y": r(np.sin(grid), 3), "dash": True},
          {"name": "8 noisy observations", "x": r(xt, 2), "y": r(yt, 3), "scatter": True}],
       "xLabel": "x", "yLabel": "f(x)"},
      md="RBF kernel with length scale ℓ = 1, noise σ_n = 0.1, eight observations of sin(x):",
      after="Near the observations the band is narrow and the mean passes close to the data. In the gap between x = 0.5 and 2.5 it widens, and beyond the last points (x > 3.5 or x < −4) it opens up to the prior's ±2 while the mean falls back to the prior mean, 0. The model knows where it has no evidence.")
series = []
for ell in (0.3, 1.0, 3.0):
    m, s = posterior(ell); series.append({"name": f"posterior mean, ℓ = {ell}", "x": r(grid, 2), "y": r(m, 3)})
series.append({"name": "observations", "x": r(xt, 2), "y": r(yt, 3), "scatter": True})
chart("gaussian-processes", {"after": "Choosing the kernel: the marginal likelihood"}, "The same data under three length scales",
      {"kind": "line", "points": False, "series": series, "xLabel": "x", "yLabel": "f(x)"},
      after="With ℓ = 0.3 the function is allowed to wiggle quickly, so the mean chases every point and snaps back to 0 between them (overfitting). With ℓ = 3 it must be very smooth and cannot follow the sine at all (underfitting). ℓ = 1 is about right. Maximising the marginal likelihood finds such a value automatically.")

# ---------------- Bayesian methods
t = np.linspace(0.002, 0.998, 250)
def beta_pdf(t, a, b): return np.exp((a - 1) * np.log(t) + (b - 1) * np.log(1 - t) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))
series = [{"name": "prior Beta(1, 1)", "y": r(np.ones_like(t), 3), "dash": True}]
for k, n in [(2, 5), (6, 20), (31, 100)]:
    series.append({"name": f"after {k} clicks in {n} views", "y": r(beta_pdf(t, 1 + k, 1 + n - k), 3)})
chart("bayesian-methods", {"after": "Updating beliefs about a click-through rate"}, "The posterior of a click rate as views accumulate",
      {"kind": "line", "x": r(t, 3), "points": False, "series": series, "xLabel": "click-through rate θ", "yLabel": "posterior density", "yMin": 0},
      after="Starting from a flat prior, 5 views leave almost every rate between 0.05 and 0.75 plausible. After 20 views the curve has a clear peak near 0.3, and after 100 views the rate is pinned down to roughly 0.22 to 0.40. The posterior always concentrates, at a speed of about 1/√n, around the rate that generated the data.")
rng = np.random.default_rng(1)
def logpost(th): return -np.inf if not 0 < th < 1 else 7 * math.log(th) + 3 * math.log(1 - th)
th, trace = 0.02, []
for i in range(600):
    prop = th + rng.normal(0, 0.1)
    if math.log(rng.random()) < logpost(prop) - logpost(th): th = prop
    trace.append(th)
chart("bayesian-methods", {"after": "A Metropolis sampler for a coin's bias"}, "A Metropolis trace: the sampler's path",
      {"kind": "line", "x": list(range(600)), "points": False, "series": [{"name": "θ at each iteration (7 heads in 10, uniform prior)", "y": r(trace, 3)}],
       "xLabel": "iteration", "yLabel": "θ", "yMin": 0, "yMax": 1, "hlines": [{"y": round(8 / 12, 3), "label": "posterior mean 0.67"}]},
      md="One chain started deliberately badly, at θ = 0.02:",
      after="The first few iterations climb out of the unlikely region (the burn-in, usually discarded). After that the chain wanders around the posterior, flat stretches are rejected proposals, and a histogram of the visited values approximates the posterior. A healthy trace looks like this \"hairy caterpillar\"; a trace that drifts slowly or stays stuck for long stretches signals poor mixing.")
rng = np.random.default_rng(0)
lift = rng.beta(146, 856, 100_000) - rng.beta(121, 881, 100_000)
edges = np.linspace(-0.04, 0.09, 66); h, _ = np.histogram(lift, edges, density=True)
chart("bayesian-methods", {"after": "A Bayesian A/B test"}, "The posterior distribution of the lift θ_B − θ_A",
      {"kind": "line", "x": r((edges[:-1] + edges[1:]) / 2, 4), "points": False, "series": [{"name": "posterior density of the lift", "y": r(h, 2)}],
       "xLabel": "lift in conversion rate (B minus A)", "yLabel": "density", "yMin": 0, "vlines": [{"x": 0, "label": "no difference"}]},
      after=f"About {np.mean(lift > 0):.0%} of the area lies to the right of 0: that is P(B is better). The left tail, B being worse, is small and close to 0, which is why the expected loss of shipping B is tiny. The whole distribution, not a single yes/no verdict, is what the decision should rest on.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
print(len(out), "charts")
