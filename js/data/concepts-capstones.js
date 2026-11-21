/* ML Atlas – content: capstone projects (end-to-end tabular ML, a RAG system with evaluation, LoRA fine-tuning) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'capstone-tabular', branch: 'ml', title: 'Capstone: An End-to-End Tabular ML Project', short: 'Tabular ML Capstone', level: 3, minutes: 75, prereqs: ['sklearn-workflow', 'ensembles', 'calibration-uncertainty', 'data-centric', 'stat-inference'],
    tags: ['capstone', 'project', 'churn prediction', 'leakage', 'pipeline', 'cross-validation', 'model selection', 'calibration', 'decision threshold', 'expected cost', 'bootstrap', 'slice analysis', 'error analysis', 'model card', 'monitoring'],
    summary: 'A complete machine-learning project on tabular data, from raw table to a decision you can defend: exploring the data, framing the problem and catching a leaky column, a leak-free pipeline with honest baselines and cross-validated model selection, checking calibration and choosing a threshold by expected business cost, evaluating with bootstrap confidence intervals and per-segment slices, and planning documentation and monitoring.',
    why: `Individual lessons teach individual tools; real projects need all of them at once, in the right order, with judgement at every step. This capstone walks through the workflow used for countless business problems: here, predicting which customers will cancel so a retention team can contact them. Every step uses something from earlier lessons: pandas exploration, leakage, pipelines, cross-validation, ensembles, calibration, statistical inference and responsible deployment. The goal is not the best possible score but a model whose value, limits and risks you can explain.`,
    objectives: [
      'Frame a prediction problem: target, prediction time, available features and the cost of each kind of mistake',
      'Find and remove leakage, then compare honest baselines and models with cross-validation',
      'Turn probabilities into decisions with a cost-based threshold after checking calibration',
      'Report test performance with confidence intervals and per-segment slices, and plan monitoring'
    ],
    sections: [
      { type: 'text', title: 'The project', md: `**Scenario.** A subscription company wants to predict which customers will cancel ("churn") in the next month. A retention team can offer a discount: an offer costs **20** per customer, a customer who leaves costs **200** in lost revenue, and an offer saves about **half** of the customers who would otherwise leave.

**Deliverables.** A trained model, an evaluation report with uncertainty, a decision threshold justified by cost, a list of known weaknesses, and a monitoring plan.

**Steps.** 1 · explore the data → 2 · frame the problem and find leakage → 3 · a leak-free pipeline, baselines and model selection → 4 · calibration and a cost-based threshold → 5 · honest evaluation with confidence intervals and slices → 6 · ship it responsibly.

The data is synthetic (generated in each code cell, so every cell runs on its own) but has the usual problems: missing values, class imbalance, a skewed numeric feature and one column that should not be there.` },
      { type: 'code', title: 'Step 1: load and explore the data', code: `
import numpy as np, pandas as pd

def make_data(seed=42, n=4000):
    rng = np.random.default_rng(seed)
    contract = rng.choice(["monthly", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.round(rng.gamma(2.0, 12.0, n)).astype(int)                 # months as a customer
    charges = np.round(rng.normal(60, 20, n).clip(15, 140), 2)             # monthly bill
    calls = rng.poisson(1.2, n)                                            # support calls last quarter
    region = rng.choice(["north", "south", "east", "west"], n)
    logit = (-1.2 + 1.3 * (contract == "monthly") - 0.9 * (contract == "two-year")
             - 0.035 * tenure + 0.012 * (charges - 60) + 0.45 * calls)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"contract": contract, "tenure": tenure, "monthly_charges": charges,
                       "support_calls": calls, "region": region,
                       "refund_issued": churned * (rng.random(n) < 0.9).astype(int),   # recorded AFTER a cancellation
                       "churned": churned})
    df.loc[rng.random(n) < 0.06, "monthly_charges"] = np.nan                         # some missing values
    return df

df = make_data()
print(df.head().to_string())                         # to_string() keeps every column on one line
print("rows, columns:", df.shape, "  churn rate:", round(df["churned"].mean(), 3))
print("missing values per column:", {k: round(v, 3) for k, v in df.isna().mean().items() if v > 0})
print(df.groupby("contract")["churned"].agg(["mean", "size"]).round(3))
print("correlation of refund_issued with churned:", round(df[["refund_issued", "churned"]].corr().iloc[0, 1], 3))
`, after: `About a third of customers churn, monthly contracts churn four times as often as two-year ones, 5.5% of bills are missing, and one column, refund_issued, is almost perfectly correlated with the target. That is the first red flag of the project.` },
      { type: 'text', title: 'Step 2: frame the problem and find leakage', md: `Before any modelling, write down:

- **Target**: churned within the next month (1) or not (0).
- **Prediction time**: the start of the month, when the retention team acts. Every feature must be known at that moment.
- **Features available then**: contract, tenure, monthly charges, support calls, region.
- **refund_issued** is recorded *after* a customer cancels. It is a consequence of the target, not a cause: a textbook leak (see *Data Preparation*). It would make offline results look superb and be useless in production, where it is always 0 for customers who haven't left yet. Drop it.
- **Metric**: with a third of customers churning, accuracy is misleading; use the precision–recall curve (PR-AUC) for model comparison and expected **cost** for the final decision.
- **Baseline**: always predicting "no churn". Any model must beat it clearly to be worth deploying.` },
      { type: 'code', title: 'Step 3: a leak-free pipeline, baselines and model selection', timeoutSec: 90, code: `
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
warnings.filterwarnings("ignore")

def make_data(seed=42, n=4000):                                       # the same generator as in step 1
    rng = np.random.default_rng(seed)
    contract = rng.choice(["monthly", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.round(rng.gamma(2.0, 12.0, n)).astype(int)
    charges = np.round(rng.normal(60, 20, n).clip(15, 140), 2)
    calls = rng.poisson(1.2, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    logit = (-1.2 + 1.3 * (contract == "monthly") - 0.9 * (contract == "two-year")
             - 0.035 * tenure + 0.012 * (charges - 60) + 0.45 * calls)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"contract": contract, "tenure": tenure, "monthly_charges": charges, "support_calls": calls,
                       "region": region, "refund_issued": churned * (rng.random(n) < 0.9).astype(int), "churned": churned})
    df.loc[rng.random(n) < 0.06, "monthly_charges"] = np.nan
    return df

df = make_data()
num, cat = ["tenure", "monthly_charges", "support_calls"], ["contract", "region"]
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="churned"), df["churned"], test_size=0.25,
                                          stratify=df["churned"], random_state=0)   # the test set is locked away now

def preprocess(numeric):                                              # fitted inside every CV fold: no leakage
    return ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, model in [("always 'no churn'", DummyClassifier()), ("logistic regression", LogisticRegression(max_iter=1000)),
                    ("gradient boosting", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0))]:
    pipe = make_pipeline(preprocess(num), model)
    ap = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="average_precision")
    roc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    print(f"{name:20s} PR-AUC {ap.mean():.3f} +/- {ap.std():.3f}   ROC-AUC {roc.mean():.3f}")

leaky = make_pipeline(preprocess(num + ["refund_issued"]), LogisticRegression(max_iter=1000))
print(f"with the leaky column: ROC-AUC {cross_val_score(leaky, X_tr, y_tr, cv=cv, scoring='roc_auc').mean():.3f}  <- too good to be true")
`, after: `Both real models beat the baseline by a wide margin, and the simple logistic regression edges out gradient boosting: here the underlying relationship is close to linear, and with few features the simpler model wins. Always let cross-validation decide rather than assuming the fanciest model is best. The leaky column would have reported a near-perfect score for a model that cannot work in production.` },
      { type: 'check', question: { type: 'mcq', q: 'Why is refund_issued removed even though it is the most predictive column?', options: ['It has missing values', 'It is recorded after the customer cancels, so it is not available at prediction time', 'It is categorical', 'It is correlated with tenure'], answer: 1, explain: 'Using information from after the prediction time is leakage.' } },
      { type: 'code', title: "Step 4: check calibration and choose a threshold by cost", code: `import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

def make_data(seed=42, n=4000):
    rng = np.random.default_rng(seed)
    contract = rng.choice(["monthly", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.round(rng.gamma(2.0, 12.0, n)).astype(int)
    charges = np.round(rng.normal(60, 20, n).clip(15, 140), 2)
    calls = rng.poisson(1.2, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    logit = (-1.2 + 1.3 * (contract == "monthly") - 0.9 * (contract == "two-year")
             - 0.035 * tenure + 0.012 * (charges - 60) + 0.45 * calls)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"contract": contract, "tenure": tenure, "monthly_charges": charges, "support_calls": calls,
                       "region": region, "refund_issued": churned * (rng.random(n) < 0.9).astype(int), "churned": churned})
    df.loc[rng.random(n) < 0.06, "monthly_charges"] = np.nan
    return df

df = make_data()
num, cat = ["tenure", "monthly_charges", "support_calls"], ["contract", "region"]
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=["churned", "refund_issued"]), df["churned"], test_size=0.25,
                                          stratify=df["churned"], random_state=0)
pipe = make_pipeline(ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]),
                     LogisticRegression(max_iter=1000))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]   # honest out-of-fold probabilities
y = y_tr.values

bins = np.minimum((p_oof * 10).astype(int), 9)                      # calibration check (see Calibration & Uncertainty)
ece = sum(np.mean(bins == b) * abs(y[bins == b].mean() - p_oof[bins == b].mean()) for b in range(10) if np.any(bins == b))
print(f"expected calibration error of the out-of-fold probabilities: {ece:.3f}")

def expected_cost(p, y, t, offer=20.0, lost=200.0, save_rate=0.5):     # average cost per customer
    flag = p >= t
    return (offer * flag.sum() + lost * ((y == 1) & ~flag).sum() + lost * (1 - save_rate) * ((y == 1) & flag).sum()) / len(y)

grid = np.round(np.linspace(0.02, 0.90, 89), 2)
costs = np.array([expected_cost(p_oof, y, t) for t in grid])
t_best = grid[costs.argmin()]
print(f"cost per customer: no offers {expected_cost(p_oof, y, 1.01):.1f}   offers to everyone {expected_cost(p_oof, y, 0.0):.1f}   "
      f"threshold 0.5 {expected_cost(p_oof, y, 0.5):.1f}   best threshold {t_best} -> {costs.min():.1f}")
print("the best threshold is far below 0.5:", bool(t_best < 0.3))

pipe.fit(X_tr, y_tr)                                                 # final model on all training data
p_te = pipe.predict_proba(X_te)[:, 1]
print(f"on the untouched test set, threshold {t_best}: {expected_cost(p_te, y_te.values, t_best):.1f} per customer "
      f"(no offers: {expected_cost(p_te, y_te.values, 1.01):.1f})")`, after: `The probabilities are well calibrated, so they can be trusted as probabilities. Because a missed churner costs ten times more than an offer, the cost-minimizing threshold is far below 0.5: the default threshold would actually cost more than simply making offers to everyone. The threshold was chosen on out-of-fold predictions and confirmed on the untouched test set.`, predict: {"type":"mcq","q":"A missed churner costs 200, an offer costs 20, and an offer keeps half of the customers who would otherwise leave. Where will the cost-minimizing threshold be?","options":["Around 0.5, the usual default","Around 0.8: flag only the customers we are sure about","Around 0.2, far below 0.5","At the churn rate, about 0.32"],"answer":2,"explain":"An offer pays for itself when its expected saving beats its cost: p × 0.5 × 200 > 20, so p > 0.2. With calibrated probabilities the best threshold lands close to that break-even point (0.17 here).","expect":["the best threshold is far below 0.5: True"]} },
      { type: 'chart', title: "Are the out-of-fold probabilities calibrated?", md: `A reliability diagram: customers grouped by their out-of-fold predicted probability, against how many of them actually churned (bins with at least 20 customers):`, after: `The points lie close to the diagonal: among customers given about 35%, about 35% churned. The two highest bins hold few customers (80 and 22), so their larger gaps are mostly noise. That is what an expected calibration error of 0.016 looks like, and it is what makes the probabilities usable in a cost calculation. A model whose points bend away from the diagonal would need recalibrating first (see [Calibration & Uncertainty](#/concept/calibration-uncertainty)).`, chart: {"kind":"line","x":[0.06,0.147,0.249,0.352,0.449,0.546,0.644,0.738,0.834],"series":[{"name":"out-of-fold predictions, in bins of 0.1","y":[0.058,0.163,0.265,0.345,0.42,0.527,0.636,0.8,0.955]},{"name":"perfect calibration","x":[0,1],"y":[0,1],"dash":true}],"xLabel":"average predicted churn probability in the bin","yLabel":"share of those customers who churned","xMin":0,"xMax":1,"yMin":0,"yMax":1} },
      { type: 'text', title: "Choosing the threshold by expected cost", md: `For each customer flagged (probability ≥ t) we pay for an offer; for each churner we miss we lose their value; flagged churners are saved with some probability. Averaged over n customers:

$$ cost(t) = [ c_offer · #flagged + c_lost · #missed churners + c_lost · (1 − s) · #flagged churners ] / n $$
:::where
t — the decision threshold on the predicted probability
c_offer — the cost of one retention offer (20)
c_lost — the value lost when a customer leaves (200)
s — the fraction of contacted churners the offer saves (0.5)
#flagged, #missed churners, #flagged churners — counts at threshold t
:::

Minimize it over t on held-out (out-of-fold) predictions. The ratio of the costs, not the class balance, decides where the threshold should be; when the costs change, re-optimize the threshold: no retraining needed.

With calibrated probabilities you can also work the answer out in advance. An offer to a customer whose churn probability is p costs c_offer and saves, on average, p · s · c_lost, so it pays for itself when

$$ p > t* = c_offer / (s · c_lost) $$
:::where
p — the customer's predicted churn probability
t* — the break-even probability: above it, an offer saves more than it costs, on average
c_offer, s, c_lost — the offer cost, the save rate and the value of a customer, as above; here t* = 20 / (0.5 × 200) = 0.2
:::

The threshold found on the out-of-fold predictions, 0.17, is close to this value. A large disagreement between the two would be a sign that the probabilities are not calibrated.` },
      { type: 'chart', title: "Expected cost per customer against the threshold", after: `The curve is lowest near t = 0.17, close to the break-even probability of 0.2, and fairly flat around it: every threshold from about 0.11 to 0.28 costs within 1 per customer of the minimum. To the right, the model flags fewer and fewer customers and the cost climbs towards that of making no offers at all (64 per customer). The default threshold of 0.5 costs more than simply making an offer to everyone.`, chart: {"kind":"line","x":[0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9],"points":false,"series":[{"name":"expected cost per customer (out-of-fold predictions)","y":[52.05,51.85,51.55,51.23,50.97,50.66,50.49,50.21,49.91,49.58,49.39,49.31,49.14,49.19,49.04,48.85,48.87,49.15,49.2,48.98,49.12,49.23,49.29,49.26,49.43,49.56,49.65,49.92,50.17,50.25,50.33,50.69,50.61,50.87,51.03,51.26,51.67,51.94,52.29,52.42,52.69,52.84,53.04,53.47,53.57,53.77,54.13,54.58,55.05,55.57,56.02,56.36,56.82,57.19,57.55,58.07,58.36,58.85,59.11,59.55,60.07,60.29,60.46,60.86,61.06,61.3,61.47,61.79,62.1,62.49,62.63,62.77,62.89,63.15,63.34,63.53,63.55,63.65,63.7,63.88,63.93,63.99,64.01,64.04,64.12,64.17,64.2,64.25,64.25]}],"xLabel":"decision threshold t (flag customers whose churn probability is at least t)","yLabel":"expected cost per customer","vlines":[{"x":0.17,"label":"lowest cost, t = 0.17"}],"hlines":[{"y":64.3,"label":"no offers at all"},{"y":52.2,"label":"offers to everyone"}]} },
      { type: 'code', title: 'Step 5: evaluate honestly (uncertainty and slices)', code: `
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")

def make_data(seed=42, n=4000):
    rng = np.random.default_rng(seed)
    contract = rng.choice(["monthly", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.round(rng.gamma(2.0, 12.0, n)).astype(int)
    charges = np.round(rng.normal(60, 20, n).clip(15, 140), 2)
    calls = rng.poisson(1.2, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    logit = (-1.2 + 1.3 * (contract == "monthly") - 0.9 * (contract == "two-year")
             - 0.035 * tenure + 0.012 * (charges - 60) + 0.45 * calls)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"contract": contract, "tenure": tenure, "monthly_charges": charges, "support_calls": calls,
                       "region": region, "refund_issued": churned * (rng.random(n) < 0.9).astype(int), "churned": churned})
    df.loc[rng.random(n) < 0.06, "monthly_charges"] = np.nan
    return df

df = make_data()
num, cat = ["tenure", "monthly_charges", "support_calls"], ["contract", "region"]
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=["churned", "refund_issued"]), df["churned"], test_size=0.25,
                                          stratify=df["churned"], random_state=0)
pipe = make_pipeline(ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]),
                     LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
p, y = pipe.predict_proba(X_te)[:, 1], y_te.values

rng = np.random.default_rng(0)                                       # bootstrap the test set for an error bar
boots = []
for _ in range(1000):
    idx = rng.integers(0, len(y), len(y))
    boots.append(average_precision_score(y[idx], p[idx]))
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"test PR-AUC {average_precision_score(y, p):.3f}  (95% bootstrap interval {lo:.3f} to {hi:.3f})")

flag = p >= 0.17                                                     # the threshold chosen in step 4
print("segment     customers  churn rate  recall  precision")
for c in ["monthly", "one-year", "two-year"]:
    m = (X_te["contract"] == c).values
    tp, fn, fp = np.sum(flag[m] & (y[m] == 1)), np.sum(~flag[m] & (y[m] == 1)), np.sum(flag[m] & (y[m] == 0))
    print(f"{c:10s} {m.sum():10d} {y[m].mean():11.2f} {tp / max(tp + fn, 1):7.2f} {tp / max(tp + fp, 1):10.2f}")

missed = X_te[(~flag) & (y == 1)].assign(probability=p[(~flag) & (y == 1)]).sort_values("probability")
print("churners the model was most sure would stay:")
print(missed[["contract", "tenure", "monthly_charges", "support_calls", "probability"]].head(3).round(3).to_string())
`, after: `The single test score comes with a wide error bar (about ±0.06) which is what one test set of 1,000 customers can tell you. The slices reveal what the overall number hides: almost every monthly-contract churner is caught, but most two-year churners are missed. The missed customers look loyal on every feature we have; to catch them we would need new signals (usage drops, competitor offers), not a fancier model.` },
      { type: 'code', title: "Step 5: which features drive the model?", md: `**Permutation importance** (see [Interpretability](#/concept/interpretability)) measures how much worse the model gets when one column is shuffled, which breaks that column's link to the target while keeping its distribution. It is computed on the test set after training, so it shows what the fitted model relies on:`, code: `import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")

def make_data(seed=42, n=4000):                                       # the same generator as in step 1
    rng = np.random.default_rng(seed)
    contract = rng.choice(["monthly", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.round(rng.gamma(2.0, 12.0, n)).astype(int)
    charges = np.round(rng.normal(60, 20, n).clip(15, 140), 2)
    calls = rng.poisson(1.2, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    logit = (-1.2 + 1.3 * (contract == "monthly") - 0.9 * (contract == "two-year")
             - 0.035 * tenure + 0.012 * (charges - 60) + 0.45 * calls)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"contract": contract, "tenure": tenure, "monthly_charges": charges, "support_calls": calls,
                       "region": region, "refund_issued": churned * (rng.random(n) < 0.9).astype(int), "churned": churned})
    df.loc[rng.random(n) < 0.06, "monthly_charges"] = np.nan
    return df

df = make_data()
num, cat = ["tenure", "monthly_charges", "support_calls"], ["contract", "region"]
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=["churned", "refund_issued"]), df["churned"], test_size=0.25,
                                          stratify=df["churned"], random_state=0)
pipe = make_pipeline(ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]),
                     LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)

result = permutation_importance(pipe, X_te, y_te, scoring="average_precision", n_repeats=10, random_state=0)
order = np.argsort(-result.importances_mean)
print("feature           drop in test PR-AUC when the column is shuffled")
for i in order:
    print(f"{X_te.columns[i]:16s} {result.importances_mean[i]:6.3f} +/- {result.importances_std[i]:.3f}")
print("most important feature:", X_te.columns[order[0]])
print("region adds nothing:", bool(abs(result.importances_mean[list(X_te.columns).index("region")]) < 0.01))`, after: `Contract type matters most, followed by tenure and support calls, almost tied, with monthly charges a distant fourth. Region scores about zero, which is correct: the data has no regional effect. The same check catches bugs, because an identifier or a timestamp with a high score is usually a leak. Two cautions. Importance describes the **model**, not the world: customers who call support often are more likely to leave, but hiding the support number would not keep them (see [Causal Inference](#/concept/causal-inference)). And correlated features share their importance, so a feature can look unimportant only because another one carries the same information.`, predict: {"type":"mcq","q":"Which column will the fitted model rely on most?","options":["Contract type","Tenure","Monthly charges","Region"],"answer":0,"explain":"Shuffling the contract column costs the most PR-AUC, more than 0.13: monthly contracts churn four times as often as two-year ones. Region scores about zero, as it should, because it has no effect in this data.","expect":["most important feature: contract","region adds nothing: True"]} },
      { type: 'code', title: "Step 5: would more data help?", md: `Before asking for more data, check whether it would help. A **learning curve** trains the same pipeline on growing subsets of the training data and scores each with cross-validation:`, code: `import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

def make_data(seed=42, n=4000):                                       # the same generator as in step 1
    rng = np.random.default_rng(seed)
    contract = rng.choice(["monthly", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.round(rng.gamma(2.0, 12.0, n)).astype(int)
    charges = np.round(rng.normal(60, 20, n).clip(15, 140), 2)
    calls = rng.poisson(1.2, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    logit = (-1.2 + 1.3 * (contract == "monthly") - 0.9 * (contract == "two-year")
             - 0.035 * tenure + 0.012 * (charges - 60) + 0.45 * calls)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"contract": contract, "tenure": tenure, "monthly_charges": charges, "support_calls": calls,
                       "region": region, "refund_issued": churned * (rng.random(n) < 0.9).astype(int), "churned": churned})
    df.loc[rng.random(n) < 0.06, "monthly_charges"] = np.nan
    return df

df = make_data()
num, cat = ["tenure", "monthly_charges", "support_calls"], ["contract", "region"]
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=["churned", "refund_issued"]), df["churned"], test_size=0.25,
                                          stratify=df["churned"], random_state=0)
pipe = make_pipeline(ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]),
                     LogisticRegression(max_iter=1000))

sizes, train_scores, val_scores = learning_curve(pipe, X_tr, y_tr, train_sizes=[100, 200, 400, 800, 1600, 2400],
                                                 cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="average_precision")
print("training rows   train PR-AUC   validation PR-AUC")
for n, tr, va in zip(sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"{n:13d} {tr:14.3f} {va:19.3f}")
gain = val_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-3]
print(f"gain from the last threefold increase in data: {gain:+.3f}")
print("more rows would barely help:", bool(gain < 0.02))`, after: `Validation PR-AUC rises from about 0.56 with 100 rows to about 0.60 with 800, then stops: three times as many rows add half a point. Training and validation scores have met, so the model is limited by what the five columns can tell it (bias), not by the amount of data (variance). The way forward is new information about the customers, such as usage trends, complaints or competitor offers, which is also what the slices suggested. If instead the training score sat far above the validation score with the gap still closing, more data or stronger regularization would be the levers (see [the bias–variance trade-off](#/concept/overfitting)).`, predict: {"type":"mcq","q":"With 800 training rows the validation PR-AUC is about 0.60. What will three times as many rows (2,400) give?","options":["About 0.60: the curve has flattened","About 0.70","About 0.80","About 0.90"],"answer":0,"explain":"About 0.604 against 0.599. Training and validation scores have met, so the model is limited by what the features can tell it, not by the amount of data. New signals would help more than more rows of the same columns.","expect":["more rows would barely help: True"]} },
      { type: 'chart', title: "Learning curves: training and validation PR-AUC", md: `The same pipeline trained on growing subsets of the training data, with gradient boosting from step 3 for comparison:`, after: `The logistic regression's training and validation scores meet at about 0.60 by 800 rows: it has learned what these five columns can tell it. Gradient boosting fits its training data far better than it generalizes and is still catching up at 2,400 rows, but its validation score is lower than the logistic regression's along the whole curve. Neither curve points to more rows of the same data as the cheapest improvement; new information about the customers would be.`, chart: {"kind":"line","x":[100,200,400,800,1600,2400],"xLog":true,"series":[{"name":"logistic regression, validation","y":[0.556,0.569,0.585,0.599,0.602,0.604]},{"name":"logistic regression, training","y":[0.669,0.541,0.542,0.594,0.598,0.604],"dash":true},{"name":"gradient boosting, validation","y":[0.504,0.486,0.499,0.555,0.579,0.583]},{"name":"gradient boosting, training","y":[0.874,0.847,0.792,0.743,0.704,0.676],"dash":true}],"xLabel":"training rows (log scale)","yLabel":"PR-AUC (5-fold cross-validation)","hlines":[{"y":0.322,"label":"always 'no churn' (the churn rate)"}]} },
      { type: 'text', title: 'Step 6: ship it responsibly', md: `- **Model card** (see *Responsible AI*): purpose, training data and period, features, metrics with intervals, per-segment results, known weaknesses (two-year customers), and the threshold with its cost assumptions.
- **Fairness**: if features or outcomes relate to protected groups (age, for example), compare error rates across them before deployment.
- **Monitoring** (see *MLOps*): track feature drift (PSI on tenure and charges), the share of customers flagged, offer acceptance, and (once labels arrive a month later) precision, recall and calibration. Alert when they move.
- **Retraining and review**: a schedule (e.g. quarterly) plus triggers (drift, business changes), always re-validated on a fresh hold-out set and compared with the current model.
- **Decision log**: record why refund_issued was dropped, why logistic regression was chosen, and how the threshold was set, future you (and auditors) will need it.` },
      { type: 'code', title: "Step 6: monitoring after launch (data drift and concept drift)", md: `Two different things can go wrong after launch. In **data drift** the mix of customers changes (a promotion brings in many more monthly contracts) but each kind of customer behaves as before. In **concept drift** the relationship itself changes: here a product problem makes every support call twice as damaging as it was. This cell scores two future months, each generated with one change, and compares them with the test set:`, code: `import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

def make_data(seed=42, n=4000, mix=(0.55, 0.25, 0.20), calls_effect=0.45):    # step 1's generator, with two knobs
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
pipe = make_pipeline(ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]),
                     LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)

def psi(expected, actual):                                    # population stability index (see MLOps)
    e, a = np.clip(expected, 1e-4, None), np.clip(actual, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

def shares(column, categories):
    return column.value_counts(normalize=True).reindex(categories, fill_value=0).values

months = {"test set (no drift)": X_te.assign(churned=y_te.values),
          "data drift: 75% monthly": make_data(seed=7, n=2000, mix=(0.75, 0.14, 0.11)).drop(columns="refund_issued"),
          "concept drift: calls hurt more": make_data(seed=8, n=2000, calls_effect=0.9).drop(columns="refund_issued")}
contracts = ["monthly", "one-year", "two-year"]
calibrated = {}
print("month                            PSI contract  flagged  predicted  observed")
for name, m in months.items():
    p = pipe.predict_proba(m.drop(columns="churned"))[:, 1]
    print(f"{name:32s} {psi(shares(X_tr['contract'], contracts), shares(m['contract'], contracts)):12.3f} "
          f"{np.mean(p >= 0.17):8.0%} {p.mean():10.3f} {m['churned'].mean():9.3f}")
    calibrated[name] = bool(abs(p.mean() - m["churned"].mean()) < 0.03)
print("calibrated after data drift:", calibrated["data drift: 75% monthly"])
print("calibrated after concept drift:", calibrated["concept drift: calls hurt more"])`, after: `The data-drift month moves the contract mix enough to push PSI to about 0.16, past the usual "watch" line of 0.1 (see [MLOps](#/concept/mlops)), and the share of customers flagged for an offer rises from 69% to 80%, so the retention budget has to grow. Yet the model stays calibrated, predicting 37.5% churn when 37.5% happens, because it learned how each kind of customer behaves and that has not changed. The concept-drift month is the dangerous one: every input looks as before (PSI 0.002), the model keeps predicting about 32%, and 41% of the customers leave. Input monitoring cannot see this; only comparing predictions with outcomes, once the labels arrive a month later, can. That delay is why the monitoring plan also tracks faster signals, such as offer acceptance and the number of support tickets.`, predict: {"type":"mcq","q":"In the second month only the mix of contracts changes (75% monthly instead of 55%); how each kind of customer behaves stays the same. Will the model's average prediction still match the share of customers who leave?","options":["No: its predictions will be far too low","Yes: the model knows how each contract type behaves, so its predictions follow the new mix","No: its predictions will be far too high","Only after it is retrained on the new month"],"answer":1,"explain":"The model learned how churn depends on the features, and that has not changed, so more monthly customers simply means more high predictions: 37.5% predicted against 37.5% observed. Concept drift, the third month, is different: the inputs look the same but more customers leave.","expect":["calibrated after data drift: True","calibrated after concept drift: False"]} },
      { type: 'chart', title: "Predicted and observed churn after launch", after: `Under data drift the predictions rise with the observed churn rate, because the model already knows how each contract type behaves. Under concept drift the predictions stay where they were while 9 more customers in every 100 leave. Watching the inputs alone would miss this, since their distribution has not changed: only comparing predictions with real outcomes, once they arrive, reveals it.`, chart: {"kind":"bar","x":["test set","data drift: 75% monthly","concept drift: calls hurt more"],"series":[{"name":"average predicted churn probability","y":[31.5,37.5,32.4]},{"name":"share of customers who churned","y":[32.2,37.5,41.4]}],"xLabel":"month","yLabel":"churn (%)","yMin":0} },
      { type: 'check', question: {"type":"mcq","q":"In the concept-drift month the inputs look unchanged (PSI 0.002), but churn rises from 32% to 41%. Which monitoring signal catches it?","options":["PSI on the input features","The share of customers flagged for an offer","Comparing the predictions with actual outcomes once the labels arrive","The model's training loss"],"answer":2,"explain":"Only the outcomes reveal that the same inputs now lead to more churn. The inputs and the share flagged look normal, and the training loss never changes after training."} },
      { type: 'check', question: { type: 'tf', q: 'The decision threshold should be chosen on the test set to get the best test-set cost.', answer: false, explain: 'Choose it on validation or out-of-fold predictions; the test set is only for the final, unbiased check.' } }
    ],
    when: `- Any tabular prediction problem: churn, fraud, credit risk, demand, lead scoring, maintenance.
- As a checklist for reviewing someone else's project: leakage, baselines, honest validation, decision threshold, uncertainty, slices, monitoring.
- When stakeholders need a decision and its business impact, not just a metric.`,
    pitfalls: `- Features that are only known after the prediction time (leakage).
- Tuning anything: preprocessing, model, threshold (on the test set).
- Reporting one overall score without uncertainty or segment breakdowns.
- Using the default 0.5 threshold when mistakes have very different costs.
- Shipping without a monitoring plan; the world drifts and labels arrive late.`,
    takeaways: [
      'Frame first: target, prediction time, available features, metric and baseline.',
      'Leak-free pipelines fit all preprocessing inside cross-validation folds; lock the test set away.',
      'Let cross-validation choose the model; simple models often win on simple relationships.',
      'Choose the threshold by expected cost on out-of-fold predictions after checking calibration.',
      'Report intervals and slices, document decisions, and monitor drift and performance after launch.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the first thing to establish in a churn-prediction project?', options: ['Which deep network to use', 'The target, the prediction time and which features are available at that time', 'The GPU budget', 'The colour scheme of the dashboard'], answer: 1, explain: 'Everything else (leakage, metrics, validation) follows from the framing.' },
      { type: 'mcq', q: 'A missed churner costs 200 and an offer costs 20. Compared with a symmetric-cost problem, the best threshold will usually be…', options: ['Higher than 0.5', 'Lower than 0.5', 'Exactly 0.5', 'Unrelated to the costs'], answer: 1, explain: 'Expensive misses push the threshold down to flag more customers.' },
      { type: 'tf', q: 'Fitting the imputer and scaler on the full dataset before cross-validation can leak information into the validation folds.', answer: true, explain: 'Statistics such as medians and means must be computed on the training folds only.' },
      { type: 'mcq', q: 'The model has high overall recall but misses most churners in one segment. What is the best next step?', options: ['Ignore it: the average is good', 'Investigate that segment: look at the missed cases and seek new features or a segment-specific strategy', 'Lower the threshold until everyone is flagged', 'Delete the segment from the data'], answer: 1, explain: 'Slice analysis exists to find exactly these blind spots.' },
      { type: 'fill', q: 'A bootstrap of the test set gives a 95% interval of 0.52 to 0.63 for PR-AUC. A new model scores 0.58 on the same test set. Can you conclude it is better than the old model\'s 0.57? (yes or no)', answer: ['no'], explain: 'The difference is far smaller than the uncertainty; compare both models on the same bootstrap resamples, or collect more data.' },
      {"type":"fill","q":"An offer costs 30, a lost customer costs 300, and an offer saves 40% of the customers who would leave. What is the break-even probability t* = c_offer / (s · c_lost)? (decimal, 2 places)","answer":["0.25",".25"],"explain":"30 / (0.4 × 300) = 30 / 120 = 0.25. Offers pay for themselves, on average, for customers whose calibrated churn probability is above 0.25."},
      {"type":"mcq","q":"In a reliability diagram, the bin of customers given about a 30% churn probability sits on the diagonal. What does that mean?","options":["30% of all customers churn","About 30% of the customers in that bin actually churned","The model is 30% accurate","The threshold should be 0.3"],"answer":1,"explain":"Calibration compares predicted probabilities with observed frequencies, bin by bin: a point on the diagonal means the probability can be taken at face value."},
      {"type":"mcq","q":"As the training set grows, training and validation scores meet at a modest level. What is the most promising next step?","options":["Collect many more rows of the same columns","Find new, informative features","Use a larger test set","Lower the decision threshold"],"answer":1,"explain":"Converged curves mean the model is limited by bias: what the features can tell it. More rows of the same columns will not move it; new information will."},
      {"type":"mcq","q":"Shuffling the region column leaves the model's test PR-AUC unchanged. What does that tell you?","options":["Region causes churn","The model does not rely on region for its predictions","The model is broken","Region must be removed for fairness reasons"],"answer":1,"explain":"Permutation importance describes what the fitted model uses. A score of about zero means the model's predictions do not depend on that column."},
      {"type":"tf","q":"Under data drift, where only the mix of customers changes, a model that has learned well how churn depends on the features keeps its probabilities calibrated.","answer":true,"explain":"The relationship between features and churn has not changed, so the predictions follow the new mix. Concept drift, where that relationship changes, is what breaks calibration."}
    ],
    problems: [
      {
        id: 'cost-threshold', title: 'Cost-Based Decision Threshold', difficulty: 'medium',
        prompt: `Implement:

- \`expected_cost(probs, y, t, offer=20.0, lost=200.0, save_rate=0.5)\` → the average cost per customer when everyone with probability ≥ t is flagged: (offer · #flagged + lost · #missed positives + lost · (1 − save_rate) · #flagged positives) / n.
- \`best_threshold(probs, y, thresholds, **costs)\` → the threshold from \`thresholds\` with the lowest expected cost (the first one if tied). Pass any extra keyword arguments on to \`expected_cost\`.

\`probs\` and \`y\` are arrays (y holds 0/1 labels).`,
        starter: `
import numpy as np

def expected_cost(probs, y, t, offer=20.0, lost=200.0, save_rate=0.5):
    # TODO
    pass

def best_threshold(probs, y, thresholds, **costs):
    # TODO
    pass
`,
        solution: `
import numpy as np

def expected_cost(probs, y, t, offer=20.0, lost=200.0, save_rate=0.5):
    p, y = np.asarray(probs, dtype=float), np.asarray(y)
    flag = p >= t
    missed = np.sum((y == 1) & ~flag)
    caught = np.sum((y == 1) & flag)
    return float((offer * flag.sum() + lost * missed + lost * (1 - save_rate) * caught) / len(y))

def best_threshold(probs, y, thresholds, **costs):
    best, best_cost = None, np.inf
    for t in thresholds:
        c = expected_cost(probs, y, t, **costs)
        if c < best_cost:
            best, best_cost = t, c
    return best
`,
        hints: ['Build a boolean array of flagged customers once, then count with `np.sum`.', 'Keep a threshold only when its cost is strictly lower, so the first best one wins ties.'],
        explain: `Separating the probability model from the decision rule is what makes ML useful to a business: when the cost of an offer or the value of a customer changes, you re-run this optimization on held-out predictions instead of retraining.`,
        tests: [
          { call: 'expected_cost(np.array([0.9, 0.2, 0.6]), np.array([1, 1, 0]), 0.5)', expect: '(20 * 2 + 200 * 1 + 100 * 1) / 3' },
          { call: 'best_threshold(np.array([0.9, 0.2, 0.6]), np.array([1, 1, 0]), [0.1, 0.5, 1.0])', expect: '0.1' },
          { name: 'cheap misses: never make offers', call: 'best_threshold(np.array([0.9, 0.2, 0.6]), np.array([1, 1, 0]), [0.1, 0.5, 1.0], offer=50.0, lost=10.0)', expect: '1.0' },
          { name: 'random data vs reference', setup: 'rng = np.random.default_rng(0)\np = rng.random(200); y = (rng.random(200) < p).astype(int)', call: '[expected_cost(p, y, 0.3), best_threshold(p, y, np.linspace(0, 1, 21), offer=10.0, lost=100.0, save_rate=0.4)]', hidden: true }
        ]
      },
      {
        id: 'bootstrap-slices', title: 'Bootstrap Intervals and Slice Metrics', difficulty: 'medium',
        prompt: `Implement:

- \`bootstrap_ci(y, p, metric, n_boot=1000, alpha=0.05, seed=0)\` → \`(low, high)\`: with \`rng = np.random.default_rng(seed)\`, repeat n_boot times: \`idx = rng.integers(0, n, n)\` and compute \`metric(y[idx], p[idx])\`; return the alpha/2 and 1 − alpha/2 percentiles (\`np.percentile\` with 100·alpha/2 and 100·(1 − alpha/2)).
- \`slice_metric(y, pred, groups, metric)\` → a dict {group: metric(y[mask], pred[mask])} for every distinct value in \`groups\` (use the values as they are as keys).`,
        starter: `
import numpy as np

def bootstrap_ci(y, p, metric, n_boot=1000, alpha=0.05, seed=0):
    # TODO
    pass

def slice_metric(y, pred, groups, metric):
    # TODO
    pass
`,
        solution: `
import numpy as np

def bootstrap_ci(y, p, metric, n_boot=1000, alpha=0.05, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

def slice_metric(y, pred, groups, metric):
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    return {g.item() if hasattr(g, "item") else g: metric(y[groups == g], pred[groups == g]) for g in np.unique(groups)}
`,
        hints: ['Draw all indices for one resample with a single `rng.integers` call.', 'Boolean masks such as `groups == g` select one slice.'],
        explain: `The bootstrap turns one test set into an estimate of how much the score would vary on another test set of the same size. Slices find the groups where the model fails, often more actionable than any overall number.`,
        tests: [
          { name: 'a constant metric has a zero-width interval', call: 'bootstrap_ci(np.array([0, 1, 1, 0]), np.array([0.2, 0.8, 0.6, 0.1]), lambda a, b: 1.0, n_boot=50)', expect: '(1.0, 1.0)' },
          { call: 'slice_metric(np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1]), np.array(["a", "a", "b", "b"]), lambda t, q: float(np.mean(t == q)))', expect: '{"a": 1.0, "b": 0.5}' },
          { name: 'accuracy interval vs reference', setup: 'rng = np.random.default_rng(3)\ny = rng.integers(0, 2, 300); pred = np.where(rng.random(300) < 0.8, y, 1 - y)', call: 'bootstrap_ci(y, pred, lambda a, b: float(np.mean(a == b)), n_boot=300, seed=7)', hidden: true },
          { name: 'the interval contains the full-data value', setup: 'rng = np.random.default_rng(4)\ny = rng.integers(0, 2, 400); pred = np.where(rng.random(400) < 0.7, y, 1 - y)\nacc = lambda a, b: float(np.mean(a == b))', code: 'lo, hi = bootstrap_ci(y, pred, acc, n_boot=500)\nassert lo < acc(y, pred) < hi and hi - lo < 0.15', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Google, Rules of Machine Learning: Best Practices for ML Engineering', url: 'https://developers.google.com/machine-learning/guides/rules-of-ml' },
      { title: 'scikit-learn, Common pitfalls and recommended practices', url: 'https://scikit-learn.org/stable/common_pitfalls.html' },
      { title: 'Kapoor & Narayanan, Leakage and the Reproducibility Crisis in ML-based Science', url: 'https://arxiv.org/abs/2207.07048' },
      { title: 'Mitchell et al., Model Cards for Model Reporting', url: 'https://arxiv.org/abs/1810.03993' }
    ]
  },

  // =====================================================================================
  {
    id: 'capstone-rag', branch: 'apps', title: 'Capstone: Build and Evaluate a RAG System', short: 'RAG Capstone', level: 3, minutes: 75, prereqs: ['rag', 'vector-search', 'learning-to-rank', 'llm-evaluation'],
    tags: ['capstone', 'project', 'rag', 'retrieval', 'bm25', 'embeddings', 'hybrid retrieval', 'chunking', 'context recall', 'evaluation set', 'error analysis', 'iteration', 'citations', 'faithfulness'],
    summary: 'Build a retrieval-augmented question-answering system for a product help centre and improve it the way professionals do, by measuring: an evaluation set with question types, BM25 and embedding retrieval, combining retrievers, fixing chunking, tracking context recall and cost version by version, then generating cited answers with an LLM and checking them.',
    why: `Most RAG demos stop at "it answered my question". Real systems are judged on hundreds of questions, many phrased in ways the documents never use, and they fail quietly: the right passage is not retrieved, or the answer is split across chunks, or the model answers without support. This capstone builds a small help-centre assistant and improves it **one measured change at a time**: the habit that separates working RAG systems from impressive demos. It ties together *Retrieval-Augmented Generation*, *Vector Search*, *Learning to Rank* and *Evaluating LLM Applications*.`,
    objectives: [
      'Write an evaluation set with gold documents, gold answers and question types before building',
      'Implement and compare lexical (BM25) and embedding retrieval, and combine them',
      'Diagnose failures by type and fix them with changes such as better chunking, measuring each change',
      'Generate grounded, cited answers with an LLM and check citations and faithfulness'
    ],
    sections: [
      { type: 'text', title: 'The project', md: `**Scenario.** "Nimbus Notes", a note-taking app, wants an assistant that answers customer questions from its eight help-centre articles.

**Deliverables.** A retrieval pipeline, an evaluation set, a version-by-version results table (quality and cost), an error analysis, and a generation step that cites its sources.

**Metric.** For this capstone the main metric is **context recall@k**: does the retrieved context contain the information needed to answer? If it does not, no LLM can answer correctly without guessing; if it does, answer quality becomes a generation problem (step 4). We report it separately for two question types:

- **keyword** questions use the documents' own words or identifiers ("What does E42 mean?");
- **paraphrase** questions say the same thing in other words ("Can I work on my notes when I have no internet?").

Write the evaluation set **first**, from real user questions when you have them, and never tune on the questions you report.` },
      { type: 'code', title: "Steps 1–3: build, measure, improve", code: `import re
import numpy as np

DOCS = {
    "pricing":  "Nimbus Notes has three plans. Free costs nothing and syncs two devices. Plus costs 4 dollars per month and syncs unlimited devices. Teams costs 8 dollars per user per month and adds shared workspaces.",
    "offline":  "Every plan includes offline mode. Notes you open are stored on the device, so you can read and edit them without a connection. Changes upload automatically when the device is back online.",
    "export":   "You can export a notebook as Markdown, PDF or HTML from the notebook menu. Exports include images and attachments. Bulk export of all notebooks is available on Plus and Teams.",
    "security": "Notes are encrypted in transit with TLS and at rest with AES-256. Teams workspaces can require single sign-on. Nimbus staff cannot read private notebooks.",
    "refunds":  "Annual subscriptions can be refunded within 14 days of purchase. Monthly subscriptions are not refunded, but you can cancel at any time and keep access until the period ends.",
    "delete":   "To delete your account, open Settings, then Account, then Delete account. Deletion erases all notes after a 30 day grace period. During the grace period you can restore the account.",
    "errors":   "Error E42 means the sync token expired; sign out and sign in again to fix it. Error E17 means the note is larger than the 25 MB limit.",
    "sharing":  "Share a note with a link or invite people by email. Links can be view only or allow editing. Teams admins can disable public links.",
}
QUESTIONS = [   # (question, gold document, text the context must contain, type)
    ("What does Plus cost?", "pricing", "4 dollars", "keyword"),
    ("What does E42 mean?", "errors", "sync token expired", "keyword"),
    ("Can I export a notebook as PDF?", "export", "Markdown, PDF or HTML", "keyword"),
    ("How long is the grace period after deleting my account?", "delete", "30 day", "keyword"),
    ("Is AES-256 used?", "security", "AES-256", "keyword"),
    ("Can I work on my notes when I have no internet?", "offline", "without a connection", "paraphrase"),
    ("How do I get my money back for a yearly plan?", "refunds", "within 14 days", "paraphrase"),
    ("How can I remove my profile permanently?", "delete", "Settings, then Account", "paraphrase"),
    ("Can colleagues edit a page I send them?", "sharing", "allow editing", "paraphrase"),
    ("Is my data protected from employees?", "security", "cannot read private notebooks", "paraphrase"),
    ("What is the largest note I can store?", "errors", "25 MB", "paraphrase"),
    ("Which subscription lets me use as many phones and laptops as I want?", "pricing", "unlimited devices", "paraphrase"),
]
STOP = set("a an the and or of to is are be can i my me you your it its in on for with as at by do does how what which when from this that there".split())

def tokenize(text):
    return re.findall(r"[a-z0-9-]+", text.lower())

def make_chunks(window):                                   # sliding windows of whole sentences inside each document
    chunks = []
    for doc, text in DOCS.items():
        sents = [s if s.endswith(".") else s + "." for s in text.split(". ")]
        for i in range(max(1, len(sents) - window + 1)):
            chunks.append((doc, " ".join(sents[i:i + window])))
    return chunks

def bm25_ranker(chunks, k1=1.5, b=0.75):                   # lexical retrieval (see Learning to Rank)
    toks = [[w for w in tokenize(t) if w not in STOP] for _, t in chunks]
    N, avgdl = len(toks), np.mean([len(t) for t in toks])
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    def rank(query):
        s = np.zeros(N)
        for w in [w for w in tokenize(query) if w not in STOP]:
            for i, doc in enumerate(toks):
                f = doc.count(w)
                if f:
                    idf = np.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1)
                    s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / avgdl))
        return list(np.argsort(-s, kind="stable"))
    return rank

CONCEPTS = {   # a toy embedding model: related words share a dimension, like nearby vectors in a real one
    "price": "cost costs price pricing money pay paid dollars subscription plan plans",
    "device": "device devices phones phone laptops laptop computers unlimited many",
    "offline": "offline internet connection online without wifi network",
    "export": "export download pdf markdown html save",
    "security": "encrypted encryption secure protected protection private staff employees read",
    "refund": "refund refunded money back return cancel",
    "delete": "delete deletion remove erase permanently account profile",
    "share": "share sharing send invite link links colleagues people editing edit",
    "error": "error errors problem fix wrong",
    "size": "largest larger limit size mb big",
    "time": "days day period grace long when within",
}
WORD2DIM = {w: i for i, words in enumerate(CONCEPTS.values()) for w in words.split()}

def embed(text):
    v = np.zeros(len(CONCEPTS))
    for w in tokenize(text):
        if w in WORD2DIM:
            v[WORD2DIM[w]] += 1
    n = np.linalg.norm(v)
    return v / n if n else v

def dense_ranker(chunks):
    E = np.array([embed(t) for _, t in chunks])
    return lambda query: list(np.argsort(-(E @ embed(query)), kind="stable"))

def context_recall(chunks, retrieve, qtype):
    sel = [x for x in QUESTIONS if x[3] == qtype]
    return np.mean([need.lower() in " ".join(chunks[i][1] for i in retrieve(q)).lower() for q, _, need, _ in sel])

print("version                               keyword  paraphrase  context words")
for name, window, use_dense in [("v1  BM25, one-sentence chunks", 1, False), ("v2  + embeddings, one-sentence chunks", 1, True),
                                ("v3  + embeddings, two-sentence chunks", 2, True), ("v4  + embeddings, three-sentence chunks", 3, True)]:
    chunks = make_chunks(window)
    bm25 = bm25_ranker(chunks)
    dense = dense_ranker(chunks) if use_dense else None
    def retrieve(q):                                       # top 3 from each retriever, without duplicates
        ids = bm25(q)[:3] + (dense(q)[:3] if dense else [])
        return list(dict.fromkeys(ids))
    words = np.mean([sum(len(chunks[i][1].split()) for i in retrieve(q)) for q, *_ in QUESTIONS])
    print(f"{name:38s} {context_recall(chunks, retrieve, 'keyword'):6.0%} {context_recall(chunks, retrieve, 'paraphrase'):10.0%} {words:12.0f}")`, after: `**v1** finds every keyword question but only 2 of 7 paraphrases: BM25 needs shared words. **v2** adds an embedding retriever (top 3 from each, merged), which rescues most paraphrases. The remaining failures are chunking failures: the right document was found, but the one-sentence chunk retrieved did not contain the answer. **v3** uses overlapping two-sentence chunks and reaches 100% on both types, at the price of about twice as much context per question. **v4** adds more context without improving anything, so v3 is the version to keep.`, predict: {"type":"mcq","q":"Version v1 retrieves with BM25 alone, from one-sentence chunks. For what share of the 7 paraphrase questions will its top 3 chunks contain the answer?","options":["All of them (100%)","About 70%","About 30% (2 of 7)","None of them"],"answer":2,"explain":"Only 2 of 7. BM25 scores chunks by shared words, and paraphrases (\"money back\", \"yearly plan\") use different words from the documents (\"refunded\", \"annual\"). The keyword questions, which reuse the documents' words, all succeed.","expect":["29%"]} },
      { type: 'check', question: { type: 'mcq', q: 'Retrieval finds the right document but the retrieved chunk does not contain the answer. Which change is most likely to help?', options: ['A bigger LLM', 'Larger or overlapping chunks so the answer stays together with the words that match the question', 'Removing the embedding retriever', 'A lower temperature'], answer: 1, explain: 'That is a chunk-boundary failure, fixed in the chunking step.' } },
      { type: 'text', title: 'Reading the results like an engineer', md: `- **Measure by question type.** The overall average hid the real story: BM25 was perfect on keyword questions and poor on paraphrases. Embeddings were the opposite on identifier questions like "What does E42 mean?": rare codes and product names are exactly what embedding models represent poorly.
- **Combine retrievers carefully.** Merging the top results of both retrievers worked here. A naive rank fusion (such as reciprocal rank fusion with equal weights) can do *worse* than the best single retriever when one of them is useless for a question type: always compare on your evaluation set.
- **Change one thing at a time** and keep a results table with the version, the change, quality per type and cost (context size, latency). Without it, improvements and regressions are anecdotes.
- **Watch for overfitting to the eval set.** Twelve questions is a toy; real evaluation sets have hundreds, cover every question type and difficulty, and include questions the documents cannot answer (the system should say so).
- **Next levers**: a reranker over a larger candidate set (a cross-encoder or an LLM), query rewriting, metadata filters, and better documents, often the cheapest fix of all.` },
      { type: 'chart', title: "Recall as more chunks are retrieved, with and without fusion", md: `Instead of concatenating two result lists, **reciprocal rank fusion** (RRF) merges them into one ranking, so a fixed number of chunks can be kept:

$$ RRF(d) = Σ_r 1 / (60 + rank_r(d)) $$
:::where
d — a chunk
r — each retriever (here BM25 and the embedding retriever)
rank_r(d) — d's position in retriever r's results (1 for the best); a retriever that did not return d adds nothing
60 — a constant that keeps the very top positions from dominating the sum
:::

Here it is on the two-sentence chunks of version v3:`, after: `Fusion reaches 100% with only three chunks (about 51 words of context, against about 80 for v3's two merged lists), while BM25 alone stalls at 67% and the embedding retriever needs eight chunks to reach 92%. The average hides a weakness, though: on the paraphrase questions, where BM25 returns plausible but wrong chunks, fusion is *worse* than the embedding retriever alone when only one or two chunks are kept (43% and 71% against 86% and 100%). Whether to fuse, and how many chunks to keep, is a measurement, not a default.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8],"series":[{"name":"BM25 alone","y":[33.3,58.3,58.3,66.7,66.7,66.7,66.7,66.7],"dash":false},{"name":"embeddings alone","y":[66.7,83.3,83.3,83.3,83.3,83.3,83.3,91.7],"dash":false},{"name":"reciprocal rank fusion of both","y":[66.7,83.3,100,100,100,100,100,100],"dash":false}],"xLabel":"chunks in the context (k)","yLabel":"context recall@k, all 12 questions (%)","yMin":0,"yMax":100} },
      { type: 'text', title: "How big must the evaluation set be?", md: `A measured recall is only an estimate, and its uncertainty shrinks with the square root of the number of questions:

$$ half-width ≈ 1.96 · √( p · (1 − p) / n ) $$
:::where
p — the measured share, for example a context recall of 0.8
n — the number of questions in the evaluation set
1.96 — the multiplier for 95% confidence under the normal approximation
half-width — how far the true value may plausibly lie from the measured one, in either direction
:::

The size of the set therefore decides which differences you can see at all. Twelve questions cannot tell 80% from 95%; a few hundred resolve differences of about five points. Two habits stretch a small set further:

- **Compare versions on the same questions**, and count the questions on which they differ (a *paired* comparison). A change that fixes eight questions and breaks one is convincing even when the overall scores look close; one that fixes four and breaks three is noise.
- **Grow the set from failures.** Every production question the system gets wrong becomes, once reviewed, a new test case, so the set keeps covering the cases that matter.

Keep a **development** set for tuning and a separate **test** set that you only use to report results. Tuning chunk sizes and prompts on the questions you report overfits them, just as tuning hyperparameters on a test set does.` },
      { type: 'code', title: "How many questions are enough?", code: `import math
z = 1.96                                               # 95% confidence
print("questions   a measured recall of 80% means, with 95% confidence…")
for n in (12, 50, 100, 300, 1000):
    half = z * math.sqrt(0.8 * 0.2 / n)
    print(f"{n:9d}   80% +/- {100 * half:4.1f} points  ({100 * (0.8 - half):3.0f}% to {100 * min(1.0, 0.8 + half):3.0f}%)")
need = math.ceil((z / 0.05) ** 2 * 0.8 * 0.2)
print(f"questions needed for +/-5 points: {need}")
print("with 12 questions the interval is wider than +/-20 points:", z * math.sqrt(0.8 * 0.2 / 12) > 0.20)`, after: `Twelve questions put the true recall anywhere between about 57% and 100%. Measuring it to within ±5 points takes about 250 questions, and ±2.5 points about 1,000: halving the uncertainty takes four times as many questions. Near 0% or 100% the normal approximation used here becomes unreliable; the Wilson interval from [Statistical Inference](#/concept/stat-inference) handles those cases.`, predict: {"type":"mcq","q":"A context recall of 80% is measured on 12 questions. Roughly how wide is its 95% confidence interval?","options":["±2 points","±5 points","±10 points","±23 points"],"answer":3,"explain":"1.96 × √(0.8 × 0.2 / 12) ≈ 0.23, so the true recall could be anywhere from about 57% to 100%. Twelve questions are enough to debug a pipeline, not to compare versions.","expect":["with 12 questions the interval is wider than +/-20 points: True"]} },
      { type: 'chart', title: "Uncertainty of a measured recall against the size of the evaluation set", after: `Each tenfold increase in questions shrinks the uncertainty by a factor of about three (the square root of ten). The curves are highest for a recall near 50% and lower near the extremes, but even at 95% recall, 50 questions leave about ±6 points of doubt. Seeing a five-point improvement needs a few hundred questions, or a paired comparison on the same questions.`, chart: {"kind":"line","x":[10,20,50,100,200,500,1000,2000],"xLog":true,"series":[{"name":"measured recall 50%","y":[31,21.9,13.9,9.8,6.9,4.4,3.1,2.2]},{"name":"measured recall 80%","y":[24.8,17.5,11.1,7.8,5.5,3.5,2.5,1.8]},{"name":"measured recall 95%","y":[13.5,9.6,6,4.3,3,1.9,1.4,1]}],"xLabel":"questions in the evaluation set (log scale)","yLabel":"95% interval half-width (percentage points)","yMin":0,"hlines":[{"y":5,"label":"±5 points"}]} },
      { type: 'text', title: 'Step 4: generation with citations, and checking it', md: `With good context, generation is about **faithfulness**: the answer must be supported by the retrieved passages and say which ones. Give each passage an id, instruct the model to answer only from them, cite them, and say when the answer isn't there. Then check automatically:

1. **Citation validity**: every cited id exists, and the cited passage contains the key fact (string match for short facts, an LLM judge for longer ones).
2. **Answer correctness**: compare with the gold answer (exact match or token F1 for short answers; see *Evaluating LLM Applications*).
3. **Abstention**: for unanswerable questions, the system should decline rather than invent.
4. **Faithfulness**: an LLM judge asks "is every claim in the answer supported by the passages?", calibrated against a few human labels.` },
      { type: 'code', title: "Step 4: grading answers automatically", md: `Most of these checks need no model at all: compare each answer with a gold answer and with the passages it cites. Six answers to three questions show what each check catches:`, code: `import re
from collections import Counter

PASSAGES = {1: "Annual subscriptions can be refunded within 14 days of purchase.",
            2: "Monthly subscriptions are not refunded, but you can cancel at any time and keep access until the period ends.",
            3: "Error E42 means the sync token expired; sign out and sign in again to fix it."}
CASES = [   # (question, gold answer, the fact a cited passage must contain, the system's answer); None = unanswerable
    ("How do I get my money back for a yearly plan?", "within 14 days of purchase", "14 days",
     "Annual plans can be refunded within 14 days of purchase [1]."),
    ("How do I get my money back for a yearly plan?", "within 14 days of purchase", "14 days",
     "You can get a refund within 30 days of purchase [1]."),
    ("What does E42 mean?", "the sync token expired", "sync token expired",
     "It means your sync token has expired; sign out and in again [3]."),
    ("What does E42 mean?", "the sync token expired", "sync token expired",
     "It means the sync token expired [2]."),
    ("Is there a dark mode?", None, None, "Yes, you can turn on dark mode in Settings [2]."),
    ("Is there a dark mode?", None, None, "The documents do not say, so I don't know."),
]

def words(text):
    return re.findall(r"[a-z0-9]+", re.sub(r"\\[\\d+\\]", " ", text.lower()))

def token_f1(answer, gold):                            # overlap of words with the gold answer
    a, g = words(answer), words(gold)
    common = sum((Counter(a) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(a), common / len(g)
    return 2 * precision * recall / (precision + recall)

def check(gold, fact, answer):
    cited = [int(n) for n in re.findall(r"\\[(\\d+)\\]", answer)]
    cited_text = " ".join(PASSAGES.get(n, "") for n in cited).lower()
    abstains = bool(re.search(r"don't know|do not say|not in the documents", answer.lower()))
    if gold is None:                                   # unanswerable: the only right answer is to decline
        return {"f1": None, "cites the fact": None, "numbers supported": None, "abstains": abstains, "pass": abstains}
    numbers_ok = all(num in cited_text for num in re.findall(r"\\d+", re.sub(r"\\[\\d+\\]", " ", answer)))
    result = {"f1": token_f1(answer, gold), "cites the fact": fact in cited_text, "numbers supported": numbers_ok, "abstains": abstains}
    result["pass"] = result["f1"] >= 0.5 and result["cites the fact"] and numbers_ok and not abstains
    return result

show = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
print("answer   F1   cites the fact   numbers supported   abstains   pass")
passing = []
for i, (q, gold, fact, answer) in enumerate(CASES, 1):
    r = check(gold, fact, answer)
    print(f"{i:6d} {show(r['f1']):>5s} {show(r['cites the fact']):>16s} {show(r['numbers supported']):>19s} {show(r['abstains']):>10s} {show(r['pass']):>6s}")
    if r["pass"]:
        passing.append(i)
print("passing answers:", passing)`, after: `Each check catches a different failure. Answer 2 has the wrong number but still shares many words with the gold answer (F1 0.53); only the check that every number appears in a cited passage catches it. Answer 4 says the right thing but cites the wrong passage, and answer 5 invents a feature for a question the documents cannot answer. Answer 3, however, is **correct** and still fails: its extra, correct advice dilutes its word overlap with the short gold answer (F1 0.38). Word-overlap metrics punish rewording, so for anything longer than a short fact, teams use an LLM judge whose verdicts are checked against a sample of human labels (see [Evaluating LLM Applications](#/concept/llm-evaluation)).`, predict: {"type":"mcq","q":"Answer 3, \"It means your sync token has expired; sign out and in again [3].\", is correct. Will it pass the automatic checks?","options":["Yes, it passes every check","No: its word overlap with the short gold answer is too low","No: it cites the wrong passage","No: it contains a number that is not in the passage"],"answer":1,"explain":"It shares only 3 of its 12 words with the gold answer \"the sync token expired\", so its token F1 is 0.38, below the 0.5 threshold. Word-overlap metrics punish correct rewording, which is why longer answers need an LLM judge calibrated against human labels.","expect":["passing answers: [1, 6]"]} },
      { type: 'code', title: 'Step 4: generating cited answers with the Claude API', run: false, code: `
import re
import anthropic

passages = [
    "Annual subscriptions can be refunded within 14 days of purchase.",
    "Monthly subscriptions are not refunded, but you can cancel at any time and keep access until the period ends.",
]                                                    # the context retrieved by version v3
question = "How do I get my money back for a yearly plan?"
context = "\\n".join(f'<doc id="{i + 1}">{p}</doc>' for i, p in enumerate(passages))

client = anthropic.Anthropic()
response = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
    system="Answer only from the documents provided and cite them like [1]. If the answer is not there, say that you don't know.",
    messages=[{"role": "user", "content": context + "\\n\\nQuestion: " + question}],
)
if response.stop_reason == "refusal":
    print("The model declined this request.")
else:
    answer = next(b.text for b in response.content if b.type == "text")
    cited = [int(n) for n in re.findall(r"\\[(\\d+)\\]", answer)]
    valid = all(1 <= n <= len(passages) for n in cited)
    supported = "14 days" in " ".join(passages[n - 1] for n in cited if 1 <= n <= len(passages))
    print(answer)
    print("citations:", cited, "| all valid:", valid, "| key fact in a cited passage:", supported)
` },
      { type: 'text', title: "Step 5: error analysis (from failures to fixes)", md: `Read every failed question and label where the pipeline broke. The label points to the fix:

| Where it broke | What you see | Typical fixes |
| --- | --- | --- |
| The documents | No document contains the answer, or it is out of date | Write or update the article: often the cheapest fix of all |
| Retrieval | The right document is not among the candidates | Hybrid search, query rewriting, better embeddings, metadata filters |
| Chunking | The right document is found, but the chunk lacks the answer | Overlapping or larger chunks, splitting on headings, the document title added to every chunk |
| Ranking | The answer is among the candidates but not in the top k | A reranker over a larger candidate set, or a larger k if the context budget allows |
| Generation | The context contains the answer, but the answer is wrong or unsupported | Clearer instructions, required citations, a stronger model, automatic answer checks |
| Abstention | The system answers a question it should decline, or declines one it could answer | Unanswerable questions in the evaluation set, explicit instructions, a confidence threshold |

Count the labels before fixing anything. If most failures are in retrieval, a better prompt for the generator will not help, and the counts show which fix pays off first. Then re-run the whole evaluation after each fix, because a change that repairs one category can break another.` },
      { type: 'check', question: { type: 'tf', q: 'If the retrieved context does not contain the answer, a better prompt can reliably make the LLM answer correctly.', answer: false, explain: 'Without the information it can only guess; fix retrieval first.' } }
    ],
    when: `- Building or reviewing any RAG system: help centres, internal documentation, legal or medical references, code search.
- Before switching models or vector databases: check whether the evaluation results justify it.
- As a template for evaluation-driven development of other LLM features.`,
    pitfalls: `- Building before writing the evaluation set, then every change is judged by gut feeling.
- Reporting one average that hides failing question types.
- Changing several things at once, so nobody knows what helped.
- Assuming hybrid or reranking always helps: measure on your data.
- Tuning on the questions you report, or never testing unanswerable questions.`,
    takeaways: [
      'Write the evaluation set first: gold documents, needed facts and question types.',
      'Context recall@k tells you whether generation even has a chance; measure it per question type.',
      'Lexical and embedding retrievers fail differently; combine them and verify the combination.',
      'Chunk boundaries cause silent failures; overlapping windows fix many at a cost in context size.',
      'Generate with ids and citations, then check citation validity, correctness, abstention and faithfulness.'
    ],
    quiz: [
      { type: 'mcq', q: 'Which question would BM25 most likely miss?', options: ['"What does E42 mean?"', '"Can I work on my notes when I have no internet?" when the document says "without a connection"', '"Is AES-256 used?"', '"Can I export as PDF?"'], answer: 1, explain: 'No shared words: a paraphrase.' },
      { type: 'mcq', q: 'Why do embedding retrievers often struggle with questions like "What does E42 mean?"', options: ['They cannot read numbers at all', 'Rare identifiers carry little meaning in the embedding space, so exact-match signals are weak', 'The question is too short', 'They only work in English'], answer: 1, explain: 'Exact codes and names are BM25\'s strength.' },
      { type: 'fill', q: 'A version change raises context recall from 71% to 100% but doubles the context per question. Name the trade-off you must also report besides quality (one word).', answer: ['cost', 'latency', 'tokens'], explain: 'More context means more tokens, cost and latency per query.' },
      { type: 'tf', q: 'An evaluation set for RAG should include some questions the documents cannot answer.', answer: true, explain: 'The system should abstain instead of inventing an answer.' },
      { type: 'mcq', q: 'What does a citation-validity check verify?', options: ['That the answer is long enough', 'That every cited passage exists and actually contains the claimed fact', 'That the user is satisfied', 'That the model used the latest version'], answer: 1, explain: 'It catches invented or mismatched citations.' },
      {"type":"fill","q":"A context recall of 50% is measured on 100 questions. What is the 95% half-width 1.96 · √(p(1 − p)/n), in percentage points? (one decimal)","answer":["9.8"],"explain":"1.96 × √(0.25 / 100) = 1.96 × 0.05 = 0.098, about ±9.8 points."},
      {"type":"mcq","q":"On the same 100 questions, version B fixes 8 questions that version A got wrong and breaks 1 that A got right. What does this paired view suggest?","options":["Nothing: the 7-point difference is within the noise","B is very likely better: 8 fixes against 1 break is unlikely to happen by chance","A is better, because B broke a question","The evaluation set must be replaced"],"answer":1,"explain":"Only the 9 questions on which the versions differ carry information about which is better, and an 8-to-1 split is unlikely if they are equally good (a sign test gives p ≈ 0.04)."},
      {"type":"tf","q":"Reciprocal rank fusion of two retrievers always beats the better of the two on its own.","answer":false,"explain":"Fusion favours chunks that both retrievers rank well. When one retriever returns plausible but wrong chunks for a type of question, fusion can rank the right chunk lower than the good retriever alone would."},
      {"type":"mcq","q":"Why can a correct answer get a low token F1?","options":["Token F1 ignores the gold answer","It uses different or extra words compared with the short gold answer","Its citations are wrong","F1 only measures recall"],"answer":1,"explain":"Token F1 counts shared words. Rewording lowers recall, and extra (even correct) words lower precision."},
      {"type":"mcq","q":"A production question was answered wrongly, and a reviewer has confirmed the right answer. What should happen to it?","options":["Nothing: production questions are not part of evaluation","Add it to the evaluation set as a new test case","Add it to the system prompt as an example","Remove the documents it came from"],"answer":1,"explain":"Growing the evaluation set from real failures keeps it focused on the cases that matter, and protects against the same failure coming back later."}
    ],
    problems: [
      {
        id: 'bm25-scores', title: 'BM25 Scores', difficulty: 'medium',
        prompt: `Implement \`bm25_scores(query_tokens, docs_tokens, k1=1.5, b=0.75)\` → a NumPy array with one BM25 score per document. \`docs_tokens\` is a list of token lists; \`query_tokens\` a list of tokens (a token repeated in the query counts every time).

For each query token t and document d: IDF(t) = ln((N − n_t + 0.5) / (n_t + 0.5) + 1), where N is the number of documents and n_t the number of documents containing t; the contribution is IDF(t) · f · (k1 + 1) / (f + k1 · (1 − b + b · |d| / avgdl)), with f the count of t in d, |d| the length of d and avgdl the average length. Tokens that appear in no document contribute nothing.`,
        starter: `
import math
import numpy as np

def bm25_scores(query_tokens, docs_tokens, k1=1.5, b=0.75):
    # TODO
    pass
`,
        solution: `
import math
import numpy as np

def bm25_scores(query_tokens, docs_tokens, k1=1.5, b=0.75):
    N = len(docs_tokens)
    avgdl = sum(len(d) for d in docs_tokens) / N
    df = {}
    for d in docs_tokens:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = np.zeros(N)
    for t in query_tokens:
        if t not in df:
            continue
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
        for i, d in enumerate(docs_tokens):
            f = d.count(t)
            if f:
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
    return scores
`,
        hints: ['Count document frequencies with a set per document.', 'Skip query tokens that never occur: their n_t is 0 and every f is 0 anyway.'],
        explain: `BM25 rewards rare terms (IDF), saturates repeated terms (k1) and normalizes for length (b). It remains a strong first-stage retriever and a valuable signal next to embeddings, especially for names, codes and exact phrases.`,
        tests: [
          { name: 'a single matching document', call: 'bm25_scores(["cat"], [["cat"], ["dog"]])', expect: 'np.array([math.log(1.5 / 1.5 + 1) * 1 * 2.5 / (1 + 1.5), 0.0])' },
          { name: 'unknown words score nothing', call: 'bm25_scores(["zebra"], [["cat", "sat"], ["dog"]])', expect: 'np.zeros(2)' },
          { name: 'rarer terms count more', code: 'docs = [["a", "x"], ["a", "y"], ["a", "z"]]\ns_common = bm25_scores(["a"], docs)[0]\ns_rare = bm25_scores(["x"], docs)[0]\nassert s_rare > s_common > 0', hidden: true },
          { name: 'random corpus vs reference', setup: 'rng = np.random.default_rng(0)\nvocab = ["w%d" % i for i in range(12)]\ndocs = [list(rng.choice(vocab, size=rng.integers(3, 9))) for _ in range(6)]\nq = list(rng.choice(vocab, size=4))', call: 'bm25_scores(q, docs, k1=1.2, b=0.6)', hidden: true }
        ]
      },
      {
        id: 'rag-pipeline-eval', title: 'Chunks, Merged Retrieval and Context Recall', difficulty: 'easy',
        prompt: `Implement the pieces of the capstone's evaluation loop:

- \`sliding_chunks(sentences, window)\` → the list of strings made by joining \`window\` consecutive sentences with a space, moving one sentence at a time; if there are fewer sentences than the window, return one chunk with all of them.
- \`merge_rankings(rank_a, rank_b, k)\` → the first k ids of rank_a followed by the first k ids of rank_b, without duplicates, keeping the first occurrence.
- \`context_recall(contexts, needed)\` → the fraction of questions i whose needed text (case-insensitive) appears in \`contexts[i]\`.

\`\`\`text
sliding_chunks(["A.", "B.", "C."], 2)  ->  ["A. B.", "B. C."]
merge_rankings([3, 1, 2], [1, 5, 3], 2)  ->  [3, 1, 5]
\`\`\``,
        starter: `
def sliding_chunks(sentences, window):
    # TODO
    pass

def merge_rankings(rank_a, rank_b, k):
    # TODO
    pass

def context_recall(contexts, needed):
    # TODO
    pass
`,
        solution: `
def sliding_chunks(sentences, window):
    if len(sentences) <= window:
        return [" ".join(sentences)]
    return [" ".join(sentences[i:i + window]) for i in range(len(sentences) - window + 1)]

def merge_rankings(rank_a, rank_b, k):
    out = []
    for i in list(rank_a[:k]) + list(rank_b[:k]):
        if i not in out:
            out.append(i)
    return out

def context_recall(contexts, needed):
    hits = [n.lower() in c.lower() for c, n in zip(contexts, needed)]
    return sum(hits) / len(hits)
`,
        hints: ['A window of w over n sentences gives n − w + 1 chunks.', 'Check membership before appending to keep the first occurrence only.'],
        explain: `These three functions are the skeleton of evaluation-driven RAG development: change the chunking or the retrievers, recompute context recall per question type, and keep only the changes that help.`,
        tests: [
          { call: 'sliding_chunks(["A.", "B.", "C."], 2)', expect: '["A. B.", "B. C."]' },
          { call: 'merge_rankings([3, 1, 2], [1, 5, 3], 2)', expect: '[3, 1, 5]' },
          { call: 'context_recall(["The fee is 4 dollars.", "No answer here."], ["4 DOLLARS", "30 days"])', expect: '0.5' },
          { name: 'short documents', call: 'sliding_chunks(["Only one."], 3)', expect: '["Only one."]', hidden: true },
          { name: 'more cases vs reference', call: '[sliding_chunks(["a", "b", "c", "d", "e"], 3), merge_rankings([9, 8, 7, 6], [6, 7, 1, 2], 3)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Es et al., RAGAS: Automated Evaluation of Retrieval Augmented Generation', url: 'https://arxiv.org/abs/2309.15217' },
      { title: 'Anthropic, Introducing Contextual Retrieval', url: 'https://www.anthropic.com/news/contextual-retrieval' },
      { title: 'Thakur et al., BEIR: A Heterogeneous Benchmark for Information Retrieval', url: 'https://arxiv.org/abs/2104.08663' }
    ]
  },

  // =====================================================================================
  {
    id: 'capstone-lora', branch: 'llm', title: 'Capstone: Fine-Tune a Small Language Model with LoRA', short: 'LoRA Capstone', level: 3, minutes: 75, prereqs: ['lora', 'mini-gpt', 'quantization', 'llm-evaluation'],
    tags: ['capstone', 'project', 'fine-tuning', 'lora', 'rank', 'catastrophic forgetting', 'replay', 'adapter merging', 'quantization', 'evaluation', 'perplexity', 'peft'],
    summary: 'Adapt a pretrained language model to a new style and measure the result: pretrain a small character-level model in the browser, fine-tune it fully and with LoRA at two ranks, measure adaptation and forgetting, prevent forgetting with replay, merge the adapter, quantize to int8 and generate, then see the same workflow with Hugging Face PEFT on a real model.',
    why: `Fine-tuning is how a general model becomes your model, for a format, a domain, a tone or a task. Doing it well means answering questions the tutorials skip: How much does the model learn of the new task? How much does it forget of the old? How many parameters must be trained? Does merging and quantizing for deployment change anything? This capstone answers all of them with measurements on a model small enough to train in your browser, then maps every step onto the real-world toolchain.`,
    objectives: [
      'Run a fine-tuning experiment that measures both adaptation (new data) and forgetting (old data)',
      'Compare full fine-tuning with LoRA at different ranks in quality and trainable parameters',
      'Use replay to prevent catastrophic forgetting, then merge and quantize the adapted model',
      'Translate the experiment into a real LoRA fine-tune with Hugging Face PEFT'
    ],
    sections: [
      { type: 'text', title: 'The project', md: `**Scenario.** We have a small language model trained on one kind of text (short stories about animals: the "old skill") and want it to write a new kind (recipe steps: the "new style"), without losing the old skill.

**Deliverables.** A results table (trainable parameters, loss on the new style, loss on the old style) for each fine-tuning method; a merged and int8-quantized model; sample generations; and a recommendation.

**Why two test sets?** Fine-tuning reports usually show only improvement on the target task. The cost (**catastrophic forgetting** of what the model could do before) is invisible unless you measure it on held-out data from the original distribution.

**Method.** A character-level model (8 characters of context → embedding → hidden layer → next character), the same family as in *Capstone: A Tiny GPT in NumPy*, small enough to pretrain in about a second on a laptop. In the browser each of the experiments below takes several seconds, and up to a minute on a slow computer, so they are allowed a longer time limit than other samples. LoRA adapts the two weight matrices with low-rank updates W + A·B, exactly as in *LoRA & Parameter-Efficient Fine-Tuning*.` },
      { type: 'code', title: "The whole experiment: pretrain, fine-tune four ways, merge, quantize, generate", timeoutSec: 120, code: `import numpy as np
rng = np.random.default_rng(0)

# ---------- Step 1: two kinds of text ---------------------------------------------------------------
animals, places = ["cat", "dog", "fox", "owl", "hen", "cow", "pig", "bat"], ["mat", "log", "hill", "barn", "pond", "road", "tree", "box"]
verbs = ["sat on", "ran to", "hid in", "slept by", "looked at"]
units, foods = ["spoon", "pinch", "dash"], ["salt", "rice", "milk", "oats", "dates", "beans"]
acts = ["stir", "mix", "bake", "boil", "chop", "whisk"]
story = lambda n: " ".join(f"the {rng.choice(animals)} {rng.choice(verbs)} the {rng.choice(places)}." for _ in range(n))
recipe = lambda n: " ".join(f"add one {rng.choice(units)} of {rng.choice(foods)} and {rng.choice(acts)}." for _ in range(n))
texts = {"A_train": story(400), "A_test": story(60), "B_train": recipe(40), "B_test": recipe(60)}  # A = old skill, B = new style
chars = sorted(set("".join(texts.values())))
stoi, ctx = {c: i for i, c in enumerate(chars)}, 8
V, E, H = len(chars), 12, 128

def windows(text):                                       # (8 previous characters -> next character) pairs
    ids = np.array([stoi[c] for c in text])
    return np.stack([ids[i:len(ids) - ctx + i] for i in range(ctx)], axis=1), ids[ctx:]
data = {k: windows(v) for k, v in texts.items()}

# ---------- Step 2: a small character-level language model -----------------------------------------
def forward(P, X, lora=None):
    W1, W2 = P["W1"], P["W2"]
    if lora:                                             # LoRA: W + A @ B, where A @ B has low rank
        W1, W2 = W1 + lora["A1"] @ lora["B1"], W2 + lora["A2"] @ lora["B2"]
    emb = P["C"][X].reshape(len(X), -1)
    h = np.tanh(emb @ W1 + P["b1"])
    return emb, h, h @ W2 + P["b2"], W1, W2

def loss(P, split, lora=None):                           # mean cross-entropy (nats per character)
    X, y = data[split]
    z = forward(P, X, lora)[2]
    z = z - z.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))

def train(P, steps, lr=0.5, lora=None, replay=0.0, split="B_train", seed=5):
    r = np.random.default_rng(seed)
    (XB, yB), (XA, yA) = data[split], data["A_train"]
    for _ in range(steps):
        n_old = int(64 * replay)                          # replay: mix some old-style examples into every batch
        iB, iA = r.integers(0, len(yB), 64 - n_old), r.integers(0, len(yA), n_old)
        X, y = np.concatenate([XB[iB], XA[iA]]), np.concatenate([yB[iB], yA[iA]])
        emb, h, z, W1, W2 = forward(P, X, lora)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        p /= len(y)                                       # gradient of the mean cross-entropy w.r.t. the logits
        dh = p @ W2.T * (1 - h ** 2)
        g = {"W2": h.T @ p, "b2": p.sum(0), "W1": emb.T @ dh, "b1": dh.sum(0)}
        if lora is None:                                  # full fine-tuning: every weight moves
            gC = np.zeros_like(P["C"])
            np.add.at(gC, X.ravel(), (dh @ W1.T).reshape(-1, E))
            g["C"] = gC
            for k in P:
                P[k] -= lr * g[k]
        else:                                             # LoRA: only the low-rank factors move
            for n in "12":
                gW = g["W" + n]
                gA, gB = gW @ lora["B" + n].T, lora["A" + n].T @ gW
                lora["A" + n] -= lr * gA
                lora["B" + n] -= lr * gB

r1 = np.random.default_rng(1)
base = {"C": r1.normal(0, 0.3, (V, E)), "W1": r1.normal(0, 1 / np.sqrt(ctx * E), (ctx * E, H)), "b1": np.zeros(H),
        "W2": r1.normal(0, 1 / np.sqrt(H), (H, V)), "b2": np.zeros(V)}
train(base, 3000, split="A_train", seed=0)                # "pretraining" on the old style only
n_base = sum(w.size for w in base.values())

# ---------- Step 3: fine-tune on the new style ----------------------------------------------------
def new_lora(rank):
    r = np.random.default_rng(2)
    return {"A1": r.normal(0, 0.1, (ctx * E, rank)), "B1": np.zeros((rank, H)),
            "A2": r.normal(0, 0.1, (H, rank)), "B2": np.zeros((rank, V))}   # B = 0: training starts exactly at the base model

runs = {"base model, no fine-tuning": (base, None, 0)}
for name, rank, replay in [("full fine-tuning", None, 0.0), ("LoRA rank 2", 2, 0.0), ("LoRA rank 8", 8, 0.0),
                           ("full fine-tuning + replay", None, 0.5), ("LoRA rank 8 + replay", 8, 0.5)]:
    if rank is None:
        P = {k: v.copy() for k, v in base.items()}
        train(P, 300, replay=replay)
        runs[name] = (P, None, n_base)
    else:
        lora = new_lora(rank)
        train(base, 300, lora=lora, replay=replay)
        runs[name] = (base, lora, sum(v.size for v in lora.values()))

print(f"{'':28s} trainable   loss on NEW style   loss on OLD style")
for name, (P, lora, n) in runs.items():
    print(f"{name:28s} {n:9d} {loss(P, 'B_test', lora):19.3f} {loss(P, 'A_test', lora):19.3f}")
print("full fine-tuning forgets: old-style loss up more than fivefold:",
      loss(runs["full fine-tuning"][0], "A_test") > 5 * loss(base, "A_test"))

# ---------- Step 4: merge the adapter, quantize, generate -----------------------------------------
P, lora, _ = runs["LoRA rank 8 + replay"]
merged = dict(P, W1=P["W1"] + lora["A1"] @ lora["B1"], W2=P["W2"] + lora["A2"] @ lora["B2"])
print()
print(f"merged model: loss on new style {loss(merged, 'B_test'):.4f} (adapter kept separate: {loss(P, 'B_test', lora):.4f})")

def int8(W):                                              # per-row absmax quantization, then back to floats
    scale = np.abs(W).max(axis=1, keepdims=True) / 127
    return np.round(W / scale).clip(-127, 127) * scale
quantized = dict(merged, W1=int8(merged["W1"]), W2=int8(merged["W2"]))
print(f"int8 weights: loss on new style {loss(quantized, 'B_test'):.4f}")

def generate(P, prompt, n=40):
    out = prompt
    for _ in range(n):
        X = np.array([[stoi[c] for c in out[-ctx:]]])
        out += chars[int(forward(P, X)[2][0].argmax())]   # greedy decoding
    return out
print("base model:     ", repr(generate(base, "add one ")))
print("after LoRA:     ", repr(generate(quantized, "add one ")))
print("old skill kept: ", repr(generate(quantized, "the cat ")))`, predict: {"type":"mcq","q":"The base model was trained only on animal stories, and its loss on held-out stories is 0.25. After full fine-tuning on recipes alone, what will its loss on the stories be?","options":["About the same, 0.25: what a model knows is not lost","A little higher, about 0.3","Several times higher, around 1.8","Lower, because the model is now better at language in general"],"answer":2,"explain":"About 1.85, more than seven times the base model’s loss. Nothing in the recipe data asks the model to keep predicting stories, so its weights move wherever the new data pulls them: catastrophic forgetting. Replay prevents most of it.","expect":["full fine-tuning forgets: old-style loss up more than fivefold: True"]} },
      { type: 'chart', title: "Each method on the two skills", md: `The results table as a picture: the goal is the bottom-left corner, a low loss on both kinds of text.`, after: `The base model sits far to the right: it cannot write recipes at all. Fine-tuning without replay moves every method to the left but also far up, because the old skill is lost; LoRA, with far fewer trainable parameters, ends up even higher than full fine-tuning in this tiny model. Only the two runs with replay reach the bottom-left corner, close to the base model's old-style loss and close to full fine-tuning's new-style loss.`, chart: {"kind":"scatter","series":[{"name":"base model, no fine-tuning","x":[7.089],"y":[0.25]},{"name":"full fine-tuning","x":[0.21],"y":[1.845]},{"name":"LoRA rank 2","x":[0.542],"y":[4.664]},{"name":"LoRA rank 8","x":[0.228],"y":[4.339]},{"name":"full fine-tuning + replay","x":[0.217],"y":[0.301]},{"name":"LoRA rank 8 + replay","x":[0.28],"y":[0.334]}],"xLog":true,"yLog":true,"xLabel":"loss on the NEW style, recipes (log scale)","yLabel":"loss on the OLD style, stories (log scale)"} },
      { type: 'text', title: 'Reading the results', md: `The losses are average cross-entropy in nats per character (lower is better; e^loss is the per-character perplexity).

- **The base model** is good at the old style and terrible at the new one: it confidently predicts animal stories.
- **Full fine-tuning** learns the new style but its loss on the old style jumps several-fold: **catastrophic forgetting**.
- **LoRA rank 8** learns the new style about as well as full fine-tuning while training only about a fifth as many parameters. **Rank 2** is too small to capture the new style fully: rank is a capacity knob.
- In this tiny model LoRA does **not** protect the old skill: it forgets as much or more. Studies of large models find LoRA tends to learn somewhat less and forget somewhat less than full fine-tuning, but it is not a guarantee.
- **Replay**: mixing 50% old-style examples into every fine-tuning batch, nearly eliminates forgetting for both methods, at a small cost in new-style loss. If the old skill matters, keep some of its data in the mix (and measure it).
- **Merging** the adapter into the weights gives exactly the same model (no extra latency at inference); **int8 quantization** of the merged weights barely changes the loss.

**Recommendation from this experiment**: LoRA rank 8 with replay, merged and quantized, nearly the best of both styles with a fifth of the trainable parameters.` },
      { type: 'check', question: { type: 'mcq', q: 'Why must a fine-tuning report include the loss on held-out data from the original task?', options: ['To compute the learning rate', 'To detect catastrophic forgetting, which the new-task metrics cannot show', 'Because the old data is larger', 'It is not necessary'], answer: 1, explain: 'Improvement on the target task can hide loss of earlier skills.' } },
      { type: 'code', title: "Watching forgetting happen", timeoutSec: 120, md: `A final results table hides *when* the damage is done. This run fine-tunes in blocks of 20 steps and measures both losses after every block, for full fine-tuning, LoRA rank 8, and LoRA rank 8 with replay (like every sample in this lesson, it first pretrains the small model again):`, code: `import numpy as np
rng = np.random.default_rng(0)

# ---------- Step 1: two kinds of text ---------------------------------------------------------------
animals, places = ["cat", "dog", "fox", "owl", "hen", "cow", "pig", "bat"], ["mat", "log", "hill", "barn", "pond", "road", "tree", "box"]
verbs = ["sat on", "ran to", "hid in", "slept by", "looked at"]
units, foods = ["spoon", "pinch", "dash"], ["salt", "rice", "milk", "oats", "dates", "beans"]
acts = ["stir", "mix", "bake", "boil", "chop", "whisk"]
story = lambda n: " ".join(f"the {rng.choice(animals)} {rng.choice(verbs)} the {rng.choice(places)}." for _ in range(n))
recipe = lambda n: " ".join(f"add one {rng.choice(units)} of {rng.choice(foods)} and {rng.choice(acts)}." for _ in range(n))
texts = {"A_train": story(400), "A_test": story(60), "B_train": recipe(40), "B_test": recipe(60)}  # A = old skill, B = new style
chars = sorted(set("".join(texts.values())))
stoi, ctx = {c: i for i, c in enumerate(chars)}, 8
V, E, H = len(chars), 12, 128

def windows(text):                                       # (8 previous characters -> next character) pairs
    ids = np.array([stoi[c] for c in text])
    return np.stack([ids[i:len(ids) - ctx + i] for i in range(ctx)], axis=1), ids[ctx:]
data = {k: windows(v) for k, v in texts.items()}

# ---------- Step 2: a small character-level language model -----------------------------------------
def forward(P, X, lora=None):
    W1, W2 = P["W1"], P["W2"]
    if lora:                                             # LoRA: W + A @ B, where A @ B has low rank
        W1, W2 = W1 + lora["A1"] @ lora["B1"], W2 + lora["A2"] @ lora["B2"]
    emb = P["C"][X].reshape(len(X), -1)
    h = np.tanh(emb @ W1 + P["b1"])
    return emb, h, h @ W2 + P["b2"], W1, W2

def loss(P, split, lora=None):                           # mean cross-entropy (nats per character)
    X, y = data[split]
    z = forward(P, X, lora)[2]
    z = z - z.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))

def train(P, steps, lr=0.5, lora=None, replay=0.0, split="B_train", seed=5):
    r = np.random.default_rng(seed)
    (XB, yB), (XA, yA) = data[split], data["A_train"]
    for _ in range(steps):
        n_old = int(64 * replay)                          # replay: mix some old-style examples into every batch
        iB, iA = r.integers(0, len(yB), 64 - n_old), r.integers(0, len(yA), n_old)
        X, y = np.concatenate([XB[iB], XA[iA]]), np.concatenate([yB[iB], yA[iA]])
        emb, h, z, W1, W2 = forward(P, X, lora)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        p /= len(y)                                       # gradient of the mean cross-entropy w.r.t. the logits
        dh = p @ W2.T * (1 - h ** 2)
        g = {"W2": h.T @ p, "b2": p.sum(0), "W1": emb.T @ dh, "b1": dh.sum(0)}
        if lora is None:                                  # full fine-tuning: every weight moves
            gC = np.zeros_like(P["C"])
            np.add.at(gC, X.ravel(), (dh @ W1.T).reshape(-1, E))
            g["C"] = gC
            for k in P:
                P[k] -= lr * g[k]
        else:                                             # LoRA: only the low-rank factors move
            for n in "12":
                gW = g["W" + n]
                gA, gB = gW @ lora["B" + n].T, lora["A" + n].T @ gW
                lora["A" + n] -= lr * gA
                lora["B" + n] -= lr * gB

r1 = np.random.default_rng(1)
base = {"C": r1.normal(0, 0.3, (V, E)), "W1": r1.normal(0, 1 / np.sqrt(ctx * E), (ctx * E, H)), "b1": np.zeros(H),
        "W2": r1.normal(0, 1 / np.sqrt(H), (H, V)), "b2": np.zeros(V)}
train(base, 3000, split="A_train", seed=0)                # "pretraining" on the old style only
n_base = sum(w.size for w in base.values())

# ---------- Fine-tune in blocks of 20 steps and measure both losses after each block --------------
def new_lora(rank):
    r = np.random.default_rng(2)
    return {"A1": r.normal(0, 0.1, (ctx * E, rank)), "B1": np.zeros((rank, H)),
            "A2": r.normal(0, 0.1, (H, rank)), "B2": np.zeros((rank, V))}   # B = 0: training starts exactly at the base model

def curve(rank, replay, steps=300, every=20):
    P, lora = ({k: v.copy() for k, v in base.items()}, None) if rank is None else (base, new_lora(rank))
    rows = [(0, loss(P, "B_test", lora), loss(P, "A_test", lora))]
    for done in range(every, steps + 1, every):
        train(P, every, lora=lora, replay=replay, seed=done)
        rows.append((done, loss(P, "B_test", lora), loss(P, "A_test", lora)))
    return rows

curves = {"full fine-tuning": curve(None, 0.0), "LoRA rank 8": curve(8, 0.0), "LoRA rank 8 + replay": curve(8, 0.5)}
show = (0, 20, 40, 60, 100, 200, 300)
print("loss on the OLD style after this many fine-tuning steps")
print(f"{'':22s}" + "".join(f"{k:>7d}" for k in show))
for name, rows in curves.items():
    old = {s: a for s, b, a in rows}
    print(f"{name:22s}" + "".join(f"{old[k]:7.2f}" for k in show))
print("loss on the NEW style after 300 steps:", {name: round(rows[-1][1], 2) for name, rows in curves.items()})
full = {s: a for s, b, a in curves["full fine-tuning"]}
share = (full[60] - full[0]) / (full[300] - full[0])
print(f"share of full fine-tuning's forgetting that happens in the first 60 steps: {share:.0%}")
print("most of the forgetting happens early:", bool(share > 0.5))`, after: `Full fine-tuning about triples the old-style loss within 20 steps and does about 59% of its final damage in the first 60; after that the loss keeps creeping up. LoRA without replay forgets even faster and further, as the results table showed. With replay the old-style loss rises a little at first, to about 0.48 after 40 steps, then recovers to 0.30 as the mixed batches pull the model back, while the new style is still learned. In a real project, evaluate the old skills early and often during training, not only at the end, and keep the checkpoint that balances both.`, predict: {"type":"mcq","q":"Full fine-tuning runs for 300 steps. When does most of the damage to the old skill happen?","options":["Evenly, a little at every step","Mostly in the first fifth of training","Mostly at the very end","It never happens with full fine-tuning"],"answer":1,"explain":"About 59% of the final rise in old-style loss happens in the first 60 of the 300 steps: the first updates move the weights furthest from where pretraining left them. Checking the old skill only at the end tells you that it was lost, not that it went almost at once.","expect":["most of the forgetting happens early: True"]} },
      { type: 'chart', title: "Loss on the old style during fine-tuning", after: `Without replay, the old-style loss shoots up in the first few dozen steps and keeps rising more slowly afterwards, for full fine-tuning and even more for LoRA. With replay it wobbles briefly and settles back close to the base model's level. Measured every 20 steps, the curves show what a single final number cannot: when the damage happens, and which checkpoint would have been the best compromise.`, chart: {"kind":"line","x":[0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],"series":[{"name":"full fine-tuning: old style","y":[0.25,0.727,1.086,1.168,1.3,1.404,1.423,1.47,1.531,1.599,1.634,1.678,1.711,1.725,1.797,1.808]},{"name":"LoRA rank 8: old style","y":[0.25,1.31,2.518,3.136,3.655,4.002,4.026,4.219,4.251,4.599,4.683,4.8,4.822,4.837,4.984,5.068]},{"name":"LoRA rank 8 + replay: old style","y":[0.25,0.377,0.478,0.4,0.435,0.388,0.384,0.342,0.371,0.326,0.316,0.312,0.299,0.307,0.307,0.297]}],"xLabel":"fine-tuning steps","yLabel":"loss on the OLD style, stories","hlines":[{"y":0.25,"label":"base model"}]} },
      { type: 'code', title: "How big should the rank be?", timeoutSec: 120, md: `The rank r decides how many independent directions of change each adapted matrix can learn. Sweeping it shows how much capacity this change of style needs:`, code: `import numpy as np
rng = np.random.default_rng(0)

# ---------- Step 1: two kinds of text ---------------------------------------------------------------
animals, places = ["cat", "dog", "fox", "owl", "hen", "cow", "pig", "bat"], ["mat", "log", "hill", "barn", "pond", "road", "tree", "box"]
verbs = ["sat on", "ran to", "hid in", "slept by", "looked at"]
units, foods = ["spoon", "pinch", "dash"], ["salt", "rice", "milk", "oats", "dates", "beans"]
acts = ["stir", "mix", "bake", "boil", "chop", "whisk"]
story = lambda n: " ".join(f"the {rng.choice(animals)} {rng.choice(verbs)} the {rng.choice(places)}." for _ in range(n))
recipe = lambda n: " ".join(f"add one {rng.choice(units)} of {rng.choice(foods)} and {rng.choice(acts)}." for _ in range(n))
texts = {"A_train": story(400), "A_test": story(60), "B_train": recipe(40), "B_test": recipe(60)}  # A = old skill, B = new style
chars = sorted(set("".join(texts.values())))
stoi, ctx = {c: i for i, c in enumerate(chars)}, 8
V, E, H = len(chars), 12, 128

def windows(text):                                       # (8 previous characters -> next character) pairs
    ids = np.array([stoi[c] for c in text])
    return np.stack([ids[i:len(ids) - ctx + i] for i in range(ctx)], axis=1), ids[ctx:]
data = {k: windows(v) for k, v in texts.items()}

# ---------- Step 2: a small character-level language model -----------------------------------------
def forward(P, X, lora=None):
    W1, W2 = P["W1"], P["W2"]
    if lora:                                             # LoRA: W + A @ B, where A @ B has low rank
        W1, W2 = W1 + lora["A1"] @ lora["B1"], W2 + lora["A2"] @ lora["B2"]
    emb = P["C"][X].reshape(len(X), -1)
    h = np.tanh(emb @ W1 + P["b1"])
    return emb, h, h @ W2 + P["b2"], W1, W2

def loss(P, split, lora=None):                           # mean cross-entropy (nats per character)
    X, y = data[split]
    z = forward(P, X, lora)[2]
    z = z - z.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))

def train(P, steps, lr=0.5, lora=None, replay=0.0, split="B_train", seed=5):
    r = np.random.default_rng(seed)
    (XB, yB), (XA, yA) = data[split], data["A_train"]
    for _ in range(steps):
        n_old = int(64 * replay)                          # replay: mix some old-style examples into every batch
        iB, iA = r.integers(0, len(yB), 64 - n_old), r.integers(0, len(yA), n_old)
        X, y = np.concatenate([XB[iB], XA[iA]]), np.concatenate([yB[iB], yA[iA]])
        emb, h, z, W1, W2 = forward(P, X, lora)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        p /= len(y)                                       # gradient of the mean cross-entropy w.r.t. the logits
        dh = p @ W2.T * (1 - h ** 2)
        g = {"W2": h.T @ p, "b2": p.sum(0), "W1": emb.T @ dh, "b1": dh.sum(0)}
        if lora is None:                                  # full fine-tuning: every weight moves
            gC = np.zeros_like(P["C"])
            np.add.at(gC, X.ravel(), (dh @ W1.T).reshape(-1, E))
            g["C"] = gC
            for k in P:
                P[k] -= lr * g[k]
        else:                                             # LoRA: only the low-rank factors move
            for n in "12":
                gW = g["W" + n]
                gA, gB = gW @ lora["B" + n].T, lora["A" + n].T @ gW
                lora["A" + n] -= lr * gA
                lora["B" + n] -= lr * gB

r1 = np.random.default_rng(1)
base = {"C": r1.normal(0, 0.3, (V, E)), "W1": r1.normal(0, 1 / np.sqrt(ctx * E), (ctx * E, H)), "b1": np.zeros(H),
        "W2": r1.normal(0, 1 / np.sqrt(H), (H, V)), "b2": np.zeros(V)}
train(base, 3000, split="A_train", seed=0)                # "pretraining" on the old style only
n_base = sum(w.size for w in base.values())

# ---------- Fine-tune with LoRA at several ranks ----------------------------------------------------
def new_lora(rank):
    r = np.random.default_rng(2)
    return {"A1": r.normal(0, 0.1, (ctx * E, rank)), "B1": np.zeros((rank, H)),
            "A2": r.normal(0, 0.1, (H, rank)), "B2": np.zeros((rank, V))}   # B = 0: training starts exactly at the base model

full = {k: v.copy() for k, v in base.items()}
train(full, 300)
print(f"full fine-tuning: {n_base} trainable parameters, loss on the new style {loss(full, 'B_test'):.3f}")
print("rank   trainable   share of full   loss on NEW style   loss on OLD style")
results = {}
for rank in (2, 4, 8, 16, 32):
    lora = new_lora(rank)
    train(base, 300, lora=lora)
    n = sum(v.size for v in lora.values())
    results[rank] = loss(base, "B_test", lora)
    print(f"{rank:4d} {n:11d} {n / n_base:14.1%} {results[rank]:19.3f} {loss(base, 'A_test', lora):19.3f}")
print("rank 32 is barely better than rank 8:", bool(results[8] - results[32] < 0.02))`, after: `Rank 2 is clearly too small (0.54), but from rank 4 the loss is within a few hundredths of full fine-tuning's 0.21: rank 8 reaches 0.23 with a fifth of the trainable parameters, and ranks 16 and 32 add almost nothing. Every rank forgets the old style badly here (losses of about 3.6 to 5.0), so the rank is a capacity knob, not a protection against forgetting; replay is. Real models behave similarly when the change is one of style or format, where small ranks suffice. Teaching a lot of new knowledge needs more capacity, and there larger ranks, or full fine-tuning, pull ahead.`, predict: {"type":"mcq","q":"Rank 8 reaches a loss of about 0.23 on the new style, and full fine-tuning about 0.21. What will rank 32, with four times as many trainable parameters as rank 8, reach?","options":["About 0.06: four times the capacity, four times better","About 0.22: barely better than rank 8","About 0.5: worse, because it overfits","Far below full fine-tuning"],"answer":1,"explain":"About 0.22. A change of style needs only a few directions in each weight matrix; beyond rank 8 the extra capacity has little left to learn, so the loss stays close to full fine-tuning’s 0.21.","expect":["rank 32 is barely better than rank 8: True"]} },
      { type: 'chart', title: "Loss on the new style against the LoRA rank", after: `The loss drops steeply from rank 2 to rank 4 and then flattens just above full fine-tuning: past the capacity this change needs, extra rank buys almost nothing. Choose the smallest rank on the flat part, since it trains, stores and serves most cheaply.`, chart: {"kind":"line","x":[2,4,8,16,32],"xLog":true,"series":[{"name":"LoRA, 300 steps","y":[0.542,0.254,0.228,0.212,0.219]}],"xLabel":"rank r (log scale)","yLabel":"loss on the NEW style, recipes","yMin":0,"hlines":[{"y":0.21,"label":"full fine-tuning"}]} },
      { type: 'check', question: {"type":"mcq","q":"In the sweep, rank 2 reaches a new-style loss of 0.54 while rank 4 reaches 0.25 and rank 32 0.22. What does that suggest?","options":["Rank 2 had a bug","The new style needs more than two directions of change per matrix, but little beyond four to eight","Higher ranks always win, so use the largest one that fits","The rank does not matter"],"answer":1,"explain":"The big jump between ranks 2 and 4, followed by a plateau, shows where the needed capacity lies. Pick the smallest rank on the plateau: it trains and serves more cheaply for the same quality."} },
      { type: 'text', title: "Hyperparameters that matter most", md: `- **Learning rate** matters most. LoRA usually needs a higher one than full fine-tuning, because only the small adapter matrices move: around 1e-4 to 2e-4 for LoRA against about 1e-5 to 2e-5 for full fine-tuning of a language model. A 2025 study by Thinking Machines Lab ("LoRA Without Regret") found the best LoRA learning rate to be consistently about ten times the best full fine-tuning rate. Too high a rate makes training diverge: in this lesson's model, rank 1 at the learning rate used above blows up within 300 steps, which is why the sweep starts at rank 2.
- **Rank and α.** Real implementations multiply the adapter's output by α / r, so that changing the rank does not change the effective size of the updates and the learning rate need not be tuned again. Common choices are r between 8 and 64, with α = r or α = 2r. Start small, and raise the rank only if the held-out loss on the new task stays clearly worse than full fine-tuning's.
- **Which layers.** Adapting every linear layer, including the MLP blocks, usually works better than adapting only the attention projections, for little extra cost.
- **Epochs.** With a few thousand examples, one to three epochs is typical. Watch the held-out loss: if it rises while the training loss keeps falling, the model is memorizing its examples.
- **Data.** A few hundred clean, consistent examples in the model's own chat format often beat thousands of noisy ones. Duplicates, contradictions and formatting slips are learned too.
- **Forgetting.** Measure the old skills throughout training, as above, and mix replay data into the batches (general instruction data, or samples of the original task) if they slip.
- **Memory.** QLoRA keeps the frozen base model in 4-bit precision and trains the adapters in 16-bit, which fits far larger models on one GPU with little loss in quality (see [Quantization](#/concept/quantization)).` },
      { type: 'text', title: 'The real-world version', md: `The same workflow on a real model, with Hugging Face \`transformers\` and \`peft\`:

1. **Data**: a few hundred to a few thousand high-quality examples in the chat format the model expects (see *Fine-Tuning & Instruction Tuning*), plus a held-out set for the new task and one for general abilities.
2. **Setup**: load the base model (in bf16, or 4-bit for QLoRA), attach LoRA adapters to the attention and MLP projections, usually with r = 8–64 and α = 2r.
3. **Train** with the loss masked to the assistant's tokens, a small learning rate (around 1e-4 to 2e-4 for LoRA), and evaluation every few hundred steps.
4. **Evaluate** on the new task (task metrics, LLM-as-judge win rates) *and* on general benchmarks or your own regression suite: forgetting shows up there.
5. **Ship**: merge the adapter (or serve several adapters on one base model), quantize, and run the deployment checks in *LLM Inference & Serving*.` },
      { type: 'code', title: 'A LoRA fine-tune with Hugging Face PEFT (run locally on a GPU)', lang: 'pytorch', code: `
# pip install torch transformers peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-0.5B-Instruct"                     # any small chat model you are allowed to fine-tune
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, config)
model.print_trainable_parameters()                            # typically well under 1% of all parameters

examples = [{"prompt": "Give me a recipe step with oats.", "answer": "Add one spoon of oats and stir."}]   # your data

def encode(ex):
    prompt = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + a_ids                      # loss only on the answer tokens
    return torch.tensor([p_ids + a_ids]), torch.tensor([labels])

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train()
for epoch in range(3):
    for ex in examples:
        input_ids, labels = encode(ex)
        out = model(input_ids=input_ids.to(model.device), labels=labels.to(model.device))
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# evaluate on held-out NEW-task data and on OLD-task / general data here, before shipping

merged = model.merge_and_unload()                              # fold the adapter into the base weights
merged.save_pretrained("my-merged-model")
tok.save_pretrained("my-merged-model")
` }
    ],
    when: `- Adapting an open model to a format, domain, tone or task where prompting is not enough.
- When many task-specific variants are needed: one base model with several small LoRA adapters.
- Before fine-tuning, try prompting and retrieval; fine-tune for behaviour and style, retrieve for facts.`,
    pitfalls: `- Measuring only the new task and missing catastrophic forgetting.
- Too small a rank for a large change (underfitting), or a learning rate that wipes out general abilities.
- Training on low-quality or inconsistent examples; a few hundred excellent ones beat thousands of sloppy ones.
- Forgetting to mask the prompt tokens in the loss.
- Skipping evaluation after merging and quantization.`,
    takeaways: [
      'Always measure adaptation (new task) and forgetting (old task) on held-out data.',
      'LoRA trains a small fraction of the parameters; its rank is a capacity knob.',
      'LoRA alone does not guarantee less forgetting; replay of old data is the reliable protection.',
      'Merged adapters add no inference cost; int8 quantization usually changes quality very little: check it.',
      'Real fine-tunes follow the same steps: data, LoRA config, masked training, two-sided evaluation, merge and ship.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is catastrophic forgetting?', options: ['Forgetting to save the model', 'Losing previously learned abilities when training on new data', 'A failing GPU', 'Overfitting the tokenizer'], answer: 1, explain: 'It shows up as worse performance on the original distribution.' },
      { type: 'mcq', q: 'LoRA rank 2 learns the new style worse than rank 8. Why?', options: ['Lower ranks train more slowly but reach the same result', 'A rank-2 update can only change the weights in two directions: too little capacity for this change', 'Rank 2 is not valid', 'The learning rate is different'], answer: 1, explain: 'The rank limits how expressive the update A·B can be.' },
      { type: 'tf', q: 'Merging a LoRA adapter into the base weights changes the model\'s outputs.', answer: false, explain: 'W + A·B computed once gives exactly the same function.' },
      { type: 'mcq', q: 'Which technique most directly prevents forgetting during fine-tuning?', options: ['A larger learning rate', 'Mixing examples from the original data into the fine-tuning batches (replay)', 'Removing the LoRA adapters', 'Training for more epochs'], answer: 1, explain: 'The model keeps practising the old skill.' },
      { type: 'fill', q: 'A LoRA adapter on a 4096 × 4096 weight matrix with rank 8 has how many trainable parameters for that matrix?', answer: ['65536', '65,536'], explain: '4096 × 8 + 8 × 4096 = 65,536: versus 16.8 million for the full matrix.' },
      {"type":"fill","q":"A rank-16 LoRA adapter on a 2048 × 2048 weight matrix trains r · (d_in + d_out) parameters. What percentage of the matrix's own 4,194,304 weights is that? (one decimal)","answer":["1.6","1.56","1.5625"],"explain":"16 × (2048 + 2048) = 65,536 parameters, and 65,536 / 4,194,304 ≈ 1.6% of the full matrix."},
      {"type":"mcq","q":"Why should the old skill be evaluated during fine-tuning, not only at the end?","options":["It makes training faster","Much of the damage can happen early, and the best trade-off may be an intermediate checkpoint","The final model is always the worst one","Evaluation during training changes the weights"],"answer":1,"explain":"In the training curves, more than half of full fine-tuning's forgetting happened in the first fifth of training. Regular measurements show when it happens and which checkpoint balances both skills."},
      {"type":"tf","q":"Doubling the LoRA rank roughly halves the loss on the new task.","answer":false,"explain":"The loss falls quickly up to the capacity the task needs (here about rank 4 to 8) and then hardly moves. Beyond that, extra rank only costs memory and compute."},
      {"type":"mcq","q":"Every rank in the sweep forgot the old style badly. What does that show?","options":["LoRA does not work","The rank controls capacity for the new task, not forgetting; replay is what protects the old skill","Higher ranks forget less, so the rank should be maximized","The base model was not pretrained"],"answer":1,"explain":"Old-style losses stayed between about 3.6 and 5.0 at every rank, while replay kept them near 0.3. Forgetting is driven by what the training data asks for, not by the size of the adapter."},
      {"type":"mcq","q":"Why do real LoRA implementations scale the adapter's output by α / r?","options":["To make the adapter smaller","So that changing the rank does not change the effective size of the updates, and the learning rate need not be re-tuned","To quantize the adapter","To make merging possible"],"answer":1,"explain":"With the α / r factor, α acts like a learning-rate multiplier that stays meaningful when r changes, which is why many recipes simply set α to r or 2r."}
    ],
    problems: [
      {
        id: 'lora-merge-quantize', title: 'Merging and Quantizing an Adapter', difficulty: 'easy',
        prompt: `Implement:

- \`lora_delta(A, B, alpha, r)\` → (alpha / r) · A @ B.
- \`merge(W, A, B, alpha, r)\` → W + lora_delta(A, B, alpha, r).
- \`quantize_int8(W)\` → \`(q, scale)\`: per row, scale = max |W_row| / 127 (use 1.0 for an all-zero row) and q = round(W / scale) clipped to [−127, 127], as a NumPy int8 array; scale has shape (rows, 1).
- \`dequantize(q, scale)\` → q · scale as floats.`,
        starter: `
import numpy as np

def lora_delta(A, B, alpha, r):
    # TODO
    pass

def merge(W, A, B, alpha, r):
    # TODO
    pass

def quantize_int8(W):
    # TODO
    pass

def dequantize(q, scale):
    # TODO
    pass
`,
        solution: `
import numpy as np

def lora_delta(A, B, alpha, r):
    return (alpha / r) * (np.asarray(A, dtype=float) @ np.asarray(B, dtype=float))

def merge(W, A, B, alpha, r):
    return np.asarray(W, dtype=float) + lora_delta(A, B, alpha, r)

def quantize_int8(W):
    W = np.asarray(W, dtype=float)
    scale = np.abs(W).max(axis=1, keepdims=True) / 127
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale

def dequantize(q, scale):
    return q.astype(float) * scale
`,
        hints: ['`np.abs(W).max(axis=1, keepdims=True)` gives one maximum per row.', 'Convert to int8 only after rounding and clipping.'],
        explain: `Merging removes the adapter's runtime cost; per-row scales keep the quantization error proportional to each row's own range, so one large row cannot ruin the precision of the others.`,
        tests: [
          { call: 'merge(np.zeros((2, 2)), np.array([[1.0], [2.0]]), np.array([[1.0, -1.0]]), 2.0, 1)', expect: 'np.array([[2.0, -2.0], [4.0, -4.0]])' },
          { call: 'quantize_int8(np.array([[1.0, -0.5], [0.0, 0.0]]))[0].tolist()', expect: '[[127, -64], [0, 0]]' },
          { name: 'round trip error is small', setup: 'rng = np.random.default_rng(0)\nW = rng.normal(size=(8, 16))', code: 'q, s = quantize_int8(W)\nassert q.dtype == np.int8 and s.shape == (8, 1)\nassert np.max(np.abs(dequantize(q, s) - W)) <= np.max(s) / 2 + 1e-12', hidden: true },
          { name: 'merge vs reference', setup: 'rng = np.random.default_rng(1)\nW = rng.normal(size=(5, 4)); A = rng.normal(size=(5, 2)); B = rng.normal(size=(2, 4))', call: 'merge(W, A, B, 16, 2)', hidden: true }
        ]
      },
      {
        id: 'lora-gradients', title: 'Gradients of a LoRA Layer', difficulty: 'medium',
        prompt: `A LoRA layer computes H = X (W + s · A B), where X is (n, d_in), W (d_in, d_out) is frozen, A (d_in, r) and B (r, d_out) are trained, and s = alpha / r is a scale. Given the upstream gradient G = ∂L/∂H (n, d_out), implement \`lora_grads(X, A, B, G, s)\` → \`(dA, dB)\`:

dA = s · Xᵀ G Bᵀ   (shape d_in × r)        dB = s · Aᵀ Xᵀ G   (shape r × d_out)

(The gradient with respect to the effective weight is Xᵀ G; the chain rule through the product A B gives these two factors.)`,
        starter: `
import numpy as np

def lora_grads(X, A, B, G, s):
    # TODO
    pass
`,
        solution: `
import numpy as np

def lora_grads(X, A, B, G, s):
    X, A, B, G = (np.asarray(v, dtype=float) for v in (X, A, B, G))
    gW = X.T @ G
    return s * gW @ B.T, s * A.T @ gW
`,
        hints: ['Compute the effective-weight gradient Xᵀ G once and reuse it.', 'Check shapes: (d_in × d_out)(d_out × r) for dA and (r × d_in)(d_in × d_out) for dB.'],
        explain: `At initialization B = 0, so dA is zero and only B receives gradient at first: the adapter starts exactly at the base model and grows from there. This is also why LoRA's memory savings come from the optimizer: only A and B need gradient statistics.`,
        tests: [
          { call: 'lora_grads(np.array([[1.0, 2.0]]), np.array([[1.0], [0.0]]), np.array([[3.0]]), np.array([[2.0]]), 1.0)', expect: '(np.array([[6.0], [12.0]]), np.array([[2.0]]))' },
          { name: 'zero B means zero gradient for A', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(4, 3)); A = rng.normal(size=(3, 2)); G = rng.normal(size=(4, 5))', code: 'dA, dB = lora_grads(X, A, np.zeros((2, 5)), G, 2.0)\nassert np.allclose(dA, 0) and dB.shape == (2, 5) and not np.allclose(dB, 0)', hidden: true },
          { name: 'matches finite differences', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(6, 4)); W = rng.normal(size=(4, 3)); A = rng.normal(size=(4, 2)); B = rng.normal(size=(2, 3)); T = rng.normal(size=(6, 3)); s = 0.5\ndef L(A_, B_):\n    return 0.5 * np.sum((X @ (W + s * A_ @ B_) - T) ** 2)', code: 'G = X @ (W + s * A @ B) - T\ndA, dB = lora_grads(X, A, B, G, s)\nnum = np.zeros_like(A)\nfor i in range(A.size):\n    d = np.zeros_like(A); d.flat[i] = 1e-6\n    num.flat[i] = (L(A + d, B) - L(A - d, B)) / 2e-6\nassert np.allclose(dA, num, atol=1e-5)', hidden: true },
          { name: 'random layer vs reference', setup: 'rng = np.random.default_rng(2)\nX = rng.normal(size=(5, 4)); A = rng.normal(size=(4, 3)); B = rng.normal(size=(3, 2)); G = rng.normal(size=(5, 2))', call: 'lora_grads(X, A, B, G, 4.0)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Hu et al., LoRA: Low-Rank Adaptation of Large Language Models', url: 'https://arxiv.org/abs/2106.09685' },
      { title: 'Biderman et al., LoRA Learns Less and Forgets Less', url: 'https://arxiv.org/abs/2405.09673' },
      { title: 'Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs', url: 'https://arxiv.org/abs/2305.14314' },
      { title: 'Hugging Face, PEFT documentation', url: 'https://huggingface.co/docs/peft' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
