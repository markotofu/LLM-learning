"""Chart data for batch C3 (applied classical lessons). Apply AFTER patches/c3-text.txt."""
import json, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
r = lambda v, d=4: [None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(float(x), d) for x in v]
out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- time series
rng = np.random.default_rng(4)
t = np.arange(48)
y = 100 + 1.5 * t + 12 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 48)
s = pd.Series(y)
trend = s.rolling(12).mean().rolling(2).mean().shift(-6)          # a centred 2x12 moving average
chart("time-series", {"after": "Components of a time series"}, "Four years of monthly sales and their trend",
      {"kind": "line", "x": list(range(1, 49)), "points": False, "series": [{"name": "observed", "y": r(y, 2)}, {"name": "trend: centred 12-month moving average", "y": r(trend.tolist(), 2)}],
       "xLabel": "month", "yLabel": "sales"},
      after="Averaging over exactly one full year cancels the seasonal swing and leaves the trend (the moving average is undefined for the first and last six months). What remains after subtracting the trend is the seasonal pattern plus noise: here a yearly wave of about ±12 around a steady rise of 1.5 per month.")
rng = np.random.default_rng(0)
weekly = np.array([0, 2, 3, 3, 5, 12, 9])
days = np.arange(-27, 15)
series = 100 + 0.1 * (days + 300) + weekly[(days + 27) % 7] + rng.normal(0, 2, len(days))
hist, fut = series[days <= 0], series[days > 0]
h = np.arange(1, 15)
naive = np.full(14, hist[-1]); seas = np.array([hist[-7 + (k - 1) % 7] for k in h]); mean = np.full(14, hist.mean())
mae = lambda f: np.mean(np.abs(f - fut))
chart("time-series", {"after": "Baselines you must beat"}, "Three baseline forecasts for the next two weeks",
      {"kind": "line", "points": False, "series": [
          {"name": "history", "x": list(range(-27, 1)), "y": r(hist, 2)},
          {"name": "what actually happened", "x": h.tolist(), "y": r(fut, 2), "dash": True},
          {"name": f"naive (MAE {mae(naive):.1f})", "x": h.tolist(), "y": r(naive, 2)},
          {"name": f"seasonal naive (MAE {mae(seas):.1f})", "x": h.tolist(), "y": r(seas, 2)},
          {"name": f"mean of 4 weeks (MAE {mae(mean):.1f})", "x": h.tolist(), "y": r(mean, 2)}],
       "xLabel": "day (0 = today)", "yLabel": "daily sales", "vlines": [{"x": 0, "label": "forecast starts"}]},
      after="The naive forecast freezes today's value and the mean flattens everything; only the seasonal naive forecast, which repeats last week, follows the weekly rhythm, and its error is about half that of the others. A model that cannot beat this one-line baseline is not worth deploying.")
rng = np.random.default_rng(0)
tt = np.arange(364)
sales = pd.Series(100 + 0.1 * tt + weekly[tt % 7] + rng.normal(0, 2, 364))
lags = list(range(1, 15))
raw = [sales.autocorr(k) for k in lags]; diff = [sales.diff().dropna().autocorr(k) for k in lags]
chart("time-series", {"after": "Seeing the weekly pattern in the autocorrelation"}, "Autocorrelation before and after differencing",
      {"kind": "bar", "x": [str(k) for k in lags], "series": [{"name": "raw series", "y": r(raw, 3)}, {"name": "after differencing", "y": r(diff, 3)}],
       "xLabel": "lag (days)", "yLabel": "autocorrelation", "yMin": -0.6, "yMax": 1},
      after="The trend makes every raw lag look strongly correlated. After differencing, the ordinary lags drop to around zero or below, and only lags 7 and 14 stay high: the signature of a weekly cycle.")

# ---------------- anomaly detection
rng = np.random.default_rng(0)
Xn = rng.normal(0, 1, (2000, 2)); Xa = rng.uniform(-6, 6, (20, 2)); Xa = Xa[np.linalg.norm(Xa, axis=1) > 3.5][:20]
X = np.vstack([Xn, Xa]); lab = np.r_[np.zeros(len(Xn)), np.ones(len(Xa))]
iso = IsolationForest(n_estimators=200, random_state=0).fit(X)
sc = -iso.score_samples(X)
edges = np.arange(0.35, 0.851, 0.025)
hn, _ = np.histogram(sc[lab == 0], edges); ha, _ = np.histogram(sc[lab == 1], edges)
chart("anomaly-detection", {"after": "Isolation forests"}, "Isolation-forest scores: normal points against anomalies",
      {"kind": "bar", "x": [f"{(a + b) / 2:.2f}" for a, b in zip(edges[:-1], edges[1:])], "series": [{"name": f"{len(Xn):,} normal points (% of group)", "y": r(100 * hn / hn.sum(), 2)}, {"name": f"{len(Xa)} anomalies (% of group)", "y": r(100 * ha / ha.sum(), 2)}],
       "xLabel": "anomaly score (higher = isolated sooner)", "yLabel": "% of the group"},
      md=f"{len(Xn):,} points from a 2-D normal distribution plus {len(Xa)} scattered far away; each point's score is based on its average path length over 200 random trees:",
      after="Normal points sit in the dense cloud and need many random splits to be isolated, so their scores bunch at the low end. The anomalies are cut off after a few splits and score clearly higher. The groups overlap only at the edge: a threshold just below the lowest anomaly score (about 0.66) would catch all 15 anomalies at the cost of 3 false alarms among 2,000 normal points, while a threshold of 0.6 would add over 50. Real data is rarely this clean.")
order = np.argsort(-sc); ks = list(range(1, 61))
prec = [lab[order[:k]].mean() for k in ks]; rec = [lab[order[:k]].sum() / lab.sum() for k in ks]
chart("anomaly-detection", {"after": "Thresholds and evaluation"}, "Precision and recall within a review budget",
      {"kind": "line", "x": ks, "points": False, "series": [{"name": "precision@k: share of flagged points that are anomalies", "y": r(prec, 3)}, {"name": "recall@k: share of all anomalies found", "y": r(rec, 3)}],
       "xLabel": "k, the number of top-scored points reviewed", "yLabel": "share", "yMin": 0, "yMax": 1, "vlines": [{"x": len(Xa), "label": f"k = {len(Xa)} (the true number)"}]},
      md="The isolation forest above, reviewing its top-scored points one by one:",
      after=f"Reviewing only the top few points, almost every flag is a real anomaly (high precision) but most anomalies are still unfound. As the budget grows, recall climbs to 100% at around k = {int(np.argmax(np.array(rec) >= 1) + 1)}, and from then on every extra review is a false alarm, so precision falls. The right k is a business decision: how much analyst time is a missed anomaly worth?")

# ---------------- learning to rank
pos = list(range(1, 11))
chart("learning-to-rank", {"after": "Measuring a ranking"}, "How much each position counts",
      {"kind": "line", "x": pos, "series": [{"name": "DCG discount 1 / log₂(i + 1)", "y": r([1 / math.log2(i + 1) for i in pos], 3)}, {"name": "reciprocal rank 1 / i (MRR)", "y": r([1 / i for i in pos], 3)}],
       "xLabel": "position i in the ranking", "yLabel": "weight", "yMin": 0, "yMax": 1.05},
      after="Both metrics care most about the top of the list, but differently. MRR's weight drops by half from position 1 to 2 and is almost nothing by position 10: only the first relevant result matters. DCG's logarithmic discount is gentler (position 10 still counts 0.29), so it rewards getting several relevant items into the top positions.")

# ---------------- recommender systems
rng = np.random.default_rng(0)
n_items = 2000
p = 1 / np.arange(1, n_items + 1) ** 1.1; p /= p.sum()
counts = np.sort(np.bincount(rng.choice(n_items, 200_000, p=p), minlength=n_items))[::-1]
top1, top20 = counts[:20].sum() / counts.sum(), counts[:400].sum() / counts.sum()
ranks = np.unique(np.round(np.logspace(0, math.log10(n_items), 60)).astype(int))
chart("recommender-systems", {"after": "The problem"}, "The long tail of item popularity",
      {"kind": "line", "x": ranks.tolist(), "xLog": True, "yLog": True, "points": False, "series": [{"name": "interactions per item", "y": [int(max(counts[i - 1], 1)) for i in ranks]}],
       "xLabel": "item rank by popularity (log scale)", "yLabel": "interactions (log scale)"},
      md="200,000 simulated interactions spread over 2,000 items with a typical power-law popularity:",
      after=f"The top 1% of items (20 of them) collect {top1:.0%} of all interactions and the top 20% collect {top20:.0%}, while the items in the long tail have only a handful each. Collaborative filtering learns little about tail items, and a recommender that ranks by learned relevance tends to push the head even harder.")

# ---------------- causal inference
chart("causal-inference", {"after": "Simpson's paradox"}, "Simpson's paradox: success rates by stone size",
      {"kind": "bar", "x": ["small stones", "large stones", "all patients"], "series": [{"name": "treatment A", "y": [round(100 * 81 / 87, 1), round(100 * 192 / 263, 1), round(100 * 273 / 350, 1)]}, {"name": "treatment B", "y": [round(100 * 234 / 270, 1), round(100 * 55 / 80, 1), round(100 * 289 / 350, 1)]}],
       "xLabel": "patient group", "yLabel": "success rate (%)", "yMin": 0, "yMax": 100},
      after="A wins in both groups, yet B wins overall, because A was mostly given to the hard cases: 263 of A's 350 patients had large stones, against 80 of B's. Stone size drives both the choice of treatment and the outcome, so the overall comparison is confounded. Comparing within groups (adjusting for the confounder) gives the right answer: A.")
months = np.arange(-6, 7)
rng = np.random.default_rng(2)
control = 80 + 1.2 * months + rng.normal(0, 0.8, len(months))
treated = 100 + 1.2 * months + 10 * (months >= 0) + rng.normal(0, 0.8, len(months))
cf = 100 + 1.2 * months
chart("causal-inference", {"after": "Difference-in-differences and natural experiments"}, "Difference-in-differences: the control group supplies the trend",
      {"kind": "line", "series": [
          {"name": "treated city", "x": months.tolist(), "y": r(treated, 2)},
          {"name": "control city", "x": months.tolist(), "y": r(control, 2)},
          {"name": "treated city without the programme (parallel to the control)", "x": months[months >= 0].tolist(), "y": r(cf[months >= 0], 2), "dash": True}],
       "xLabel": "month relative to the programme start", "yLabel": "average sales", "vlines": [{"x": 0, "label": "programme starts"}]},
      after="Before the programme both cities rise in parallel. Afterwards the treated city jumps above the dashed line, which continues its old path by borrowing the control city's trend. The gap between the solid and dashed lines is the effect, about 10; the treated city's raw before-and-after change is larger because it also contains the common trend.")

# ---------------- interpretability: superposition geometry
Ns = [64, 128, 256, 512, 1024, 2048, 4096]
series = []
for d in (64, 256, 1024):
    rng = np.random.default_rng(0); vals = []
    V_all = rng.normal(size=(max(Ns), d)); V_all /= np.linalg.norm(V_all, axis=1, keepdims=True)
    for N in Ns:                                   # nested sets: the first N of the same 4,096 directions
        V = V_all[:N]; G = np.abs(V @ V.T); np.fill_diagonal(G, 0); vals.append(G.max())
    series.append({"name": f"d = {d} dimensions", "y": r(vals, 3)})
chart("interpretability", {"after": "Superposition and sparse autoencoders"}, "How many directions fit almost perpendicularly?",
      {"kind": "line", "x": Ns, "xLog": True, "series": series, "xLabel": "number of random feature directions N (log scale)", "yLabel": "worst |cosine| between any two", "yMin": 0, "yMax": 0.7},
      md="Pick N random unit vectors in d dimensions and measure the largest overlap (|cosine|) between any two of them:",
      after="In 1,024 dimensions, even 4,096 random directions (four times the dimension) overlap by at most about 0.16, and a typical pair by only about 0.025. Interference grows very slowly with N, so a model can store many more sparse features than it has neurons, at the price of a little crosstalk. That is the geometry behind superposition, and why one neuron rarely means one thing.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
print(len(out), "charts")
