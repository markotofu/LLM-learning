"""Chart data for batch X2 (Capstone: An End-to-End Tabular ML Project). Apply AFTER patches/x2-text.txt."""
import json, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

def make_data(seed=42, n=4000, mix=(0.55, 0.25, 0.20), calls_effect=0.45):     # the lesson's generator
    rng = np.random.default_rng(seed)
    contract = rng.choice(["monthly", "one-year", "two-year"], n, p=list(mix))
    tenure = np.round(rng.gamma(2.0, 12.0, n)).astype(int)
    charges = np.round(rng.normal(60, 20, n).clip(15, 140), 2)
    calls = rng.poisson(1.2, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    logit = (-1.2 + 1.3 * (contract == "monthly") - 0.9 * (contract == "two-year")
             - 0.035 * tenure + 0.012 * (charges - 60) + calls_effect * calls)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"contract": contract, "tenure": tenure, "monthly_charges": charges, "support_calls": calls,
                       "region": region, "refund_issued": churned * (rng.random(n) < 0.9).astype(int), "churned": churned})
    df.loc[rng.random(n) < 0.06, "monthly_charges"] = np.nan
    return df

df = make_data()
num, cat = ["tenure", "monthly_charges", "support_calls"], ["contract", "region"]
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=["churned", "refund_issued"]), df["churned"], test_size=0.25,
                                          stratify=df["churned"], random_state=0)
pre = lambda: ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
pipe = make_pipeline(pre(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
y = y_tr.values

# ---- reliability diagram of the out-of-fold probabilities
bins = np.minimum((p_oof * 10).astype(int), 9)
mx, my, counts = [], [], []
for b in range(10):
    m = bins == b
    if m.sum() >= 20:
        mx.append(round(float(p_oof[m].mean()), 3)); my.append(round(float(y[m].mean()), 3)); counts.append(int(m.sum()))
print("reliability bins:", list(zip(mx, my, counts)))
chart("capstone-tabular", {"after": "Step 4 — check calibration and choose a threshold by cost"}, "Are the out-of-fold probabilities calibrated?",
      {"kind": "line", "x": mx, "series": [{"name": "out-of-fold predictions, in bins of 0.1", "y": my},
                                          {"name": "perfect calibration", "x": [0, 1], "y": [0, 1], "dash": True}],
       "xLabel": "average predicted churn probability in the bin", "yLabel": "share of those customers who churned", "xMin": 0, "xMax": 1, "yMin": 0, "yMax": 1},
      md="A reliability diagram: customers grouped by their out-of-fold predicted probability, against how many of them actually churned (bins with at least 20 customers):",
      after="The points lie close to the diagonal: among customers given about 35%, about 35% churned. The two highest bins hold few customers (80 and 22), so their larger gaps are mostly noise. That is what an expected calibration error of 0.016 looks like, and it is what makes the probabilities usable in a cost calculation. A model whose points bend away from the diagonal would need recalibrating first (see [Calibration & Uncertainty](#/concept/calibration-uncertainty)).")

# ---- expected cost against the threshold
def expected_cost(p, y, t, offer=20.0, lost=200.0, save_rate=0.5):
    flag = p >= t
    return (offer * flag.sum() + lost * ((y == 1) & ~flag).sum() + lost * (1 - save_rate) * ((y == 1) & flag).sum()) / len(y)
grid = [round(t, 2) for t in np.linspace(0.02, 0.90, 89)]
costs = [round(expected_cost(p_oof, y, t), 2) for t in grid]
t_best = grid[int(np.argmin(costs))]
none, everyone = expected_cost(p_oof, y, 1.01), expected_cost(p_oof, y, 0.0)
print("best", t_best, min(costs), "none", round(none, 1), "everyone", round(everyone, 1), "at 0.5", costs[grid.index(0.5)])
chart("capstone-tabular", {"after": "Choosing the threshold by expected cost"}, "Expected cost per customer against the threshold",
      {"kind": "line", "x": grid, "points": False, "series": [{"name": "expected cost per customer (out-of-fold predictions)", "y": costs}],
       "xLabel": "decision threshold t (flag customers whose churn probability is at least t)", "yLabel": "expected cost per customer",
       "vlines": [{"x": t_best, "label": f"lowest cost, t = {t_best}"}],
       "hlines": [{"y": round(none, 1), "label": "no offers at all"}, {"y": round(everyone, 1), "label": "offers to everyone"}]},
      after=f"The curve is lowest near t = {t_best}, close to the break-even probability of 0.2, and fairly flat around it: every threshold from about 0.11 to 0.28 costs within 1 per customer of the minimum. To the right, the model flags fewer and fewer customers and the cost climbs towards that of making no offers at all (64 per customer). The default threshold of 0.5 costs more than simply making an offer to everyone.")

# ---- learning curves
sizes = [100, 200, 400, 800, 1600, 2400]
curves = {}
for name, model in [("logistic regression", LogisticRegression(max_iter=1000)),
                    ("gradient boosting", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0))]:
    n, tr, va = learning_curve(make_pipeline(pre(), model), X_tr, y_tr, train_sizes=sizes, cv=cv, scoring="average_precision")
    curves[name] = ([round(float(v), 3) for v in tr.mean(axis=1)], [round(float(v), 3) for v in va.mean(axis=1)])
    print(name, curves[name])
chart("capstone-tabular", {"after": "Step 5 — would more data help?"}, "Learning curves: training and validation PR-AUC",
      {"kind": "line", "x": sizes, "xLog": True, "series": [
          {"name": "logistic regression, validation", "y": curves["logistic regression"][1]},
          {"name": "logistic regression, training", "y": curves["logistic regression"][0], "dash": True},
          {"name": "gradient boosting, validation", "y": curves["gradient boosting"][1]},
          {"name": "gradient boosting, training", "y": curves["gradient boosting"][0], "dash": True}],
       "xLabel": "training rows (log scale)", "yLabel": "PR-AUC (5-fold cross-validation)", "hlines": [{"y": 0.322, "label": "always 'no churn' (the churn rate)"}]},
      md="The same pipeline trained on growing subsets of the training data, with gradient boosting from step 3 for comparison:",
      after="The logistic regression's training and validation scores meet at about 0.60 by 800 rows: it has learned what these five columns can tell it. Gradient boosting fits its training data far better than it generalizes and is still catching up at 2,400 rows, but its validation score is lower than the logistic regression's along the whole curve. Neither curve points to more rows of the same data as the cheapest improvement; new information about the customers would be.")

# ---- data drift against concept drift
pipe.fit(X_tr, y_tr)
months = [("test set", X_te.assign(churned=y_te.values)),
          ("data drift: 75% monthly", make_data(seed=7, n=2000, mix=(0.75, 0.14, 0.11)).drop(columns="refund_issued")),
          ("concept drift: calls hurt more", make_data(seed=8, n=2000, calls_effect=0.9).drop(columns="refund_issued"))]
pred, obs = [], []
for name, m in months:
    p = pipe.predict_proba(m.drop(columns="churned"))[:, 1]
    pred.append(round(float(p.mean()) * 100, 1)); obs.append(round(float(m["churned"].mean()) * 100, 1))
print("drift", pred, obs)
chart("capstone-tabular", {"after": "Step 6 — monitoring after launch: data drift and concept drift"}, "Predicted and observed churn after launch",
      {"kind": "bar", "x": [n for n, _ in months], "series": [{"name": "average predicted churn probability", "y": pred},
                                                              {"name": "share of customers who churned", "y": obs}],
       "xLabel": "month", "yLabel": "churn (%)", "yMin": 0},
      after="Under data drift the predictions rise with the observed churn rate, because the model already knows how each contract type behaves. Under concept drift the predictions stay where they were while 9 more customers in every 100 leave. Watching the inputs alone would miss this, since their distribution has not changed: only comparing predictions with real outcomes, once they arrive, reveals it.")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print(len(out), "charts")
