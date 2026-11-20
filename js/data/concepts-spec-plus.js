/* ML Atlas – content: more specializations (interpretability, time series, anomaly detection, detection & segmentation, GNNs, speech & audio, causal inference) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'interpretability', branch: 'spec', title: 'Mechanistic Interpretability', short: 'Interpretability', level: 3, minutes: 45, prereqs: ['transformer-block', 'logistic-regression'],
    tags: ['interpretability', 'mechanistic interpretability', 'linear probe', 'logit lens', 'activation patching', 'superposition', 'sparse autoencoder', 'features', 'circuits', 'induction heads', 'steering'],
    summary: 'Looking inside neural networks to understand how they compute: linear probes, the logit lens, activation patching to find the components that matter, superposition and polysemantic neurons, sparse autoencoders that extract interpretable features, circuits such as induction heads, and steering a model by editing its activations.',
    why: `We train models whose inner workings nobody designed. **Mechanistic interpretability** tries to reverse-engineer them: which internal directions represent "this text is French" or "this code has a bug", which attention heads copy earlier tokens, how a fact is looked up. It matters for safety (detecting deception or dangerous capabilities), for debugging, and for trust — and it is a fascinating application of almost everything in this atlas: linear algebra, logistic regression, autoencoders and transformers.`,
    objectives: [
      'Train a linear probe on hidden activations and interpret its accuracy carefully',
      'Use the logit lens and activation patching to locate where a computation happens',
      'Explain superposition and polysemantic neurons, and how sparse autoencoders recover interpretable features',
      'Describe circuits such as induction heads, and steering with activation directions'
    ],
    sections: [
      { type: 'text', title: 'Features as directions', md: `A transformer's residual stream (see *The Transformer Block*) holds one vector per token. A useful working hypothesis — the **linear representation hypothesis** — says concepts correspond to **directions** in that space: moving along one direction makes the text "more French", along another "more negative", and so on.

A **linear probe** tests whether a property can be read from the activations: collect activations h for many inputs, label each input with the property, and train a logistic regression on them:

$$ p( property | h ) = σ( w · h + b ) $$
:::where
h — the activation vector at some layer and token position
w, b — the probe's learned weights and bias; w is a candidate direction for the property
σ — the sigmoid, turning the score into a probability
:::

High held-out accuracy means the information is **linearly decodable** — not that the model *uses* it. Always compare against a control (for example the same probe trained on shuffled labels) and confirm with interventions.` },
      { type: 'code', title: 'A linear probe on synthetic activations', code: `
import numpy as np
rng = np.random.default_rng(0)
n, d = 600, 64
concept = rng.normal(size=d)
concept /= np.linalg.norm(concept)                                   # hidden direction for, say, "is French"
labels = rng.integers(0, 2, n)
H = rng.normal(size=(n, d)) + np.outer(2 * labels - 1, concept) * 1.5   # activations: noise ± a shift along it

def fit_probe(X, y, lr=0.5, steps=300):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * X.T @ (p - y) / len(y)
        b -= lr * np.mean(p - y)
    return w, b

def accuracy(X, y, w, b):
    return float(np.mean(((X @ w + b) > 0) == y))

w, b = fit_probe(H[:400], labels[:400])
print(f"probe accuracy on held-out activations: {accuracy(H[400:], labels[400:], w, b):.2f}")
print(f"cosine between the probe and the true direction: {w @ concept / np.linalg.norm(w):.2f}")
w0, b0 = fit_probe(H[:400], rng.permutation(labels[:400]))          # control: shuffled labels
print(f"control probe (shuffled labels) on held-out data: {accuracy(H[400:], labels[400:], w0, b0):.2f}")
`, after: `The probe recovers the hidden direction and predicts the property well, while the control stays near chance — evidence that the information is really there, not something any probe could fit.` },
      { type: 'check', question: { type: 'mcq', q: 'A linear probe predicts a property from layer-10 activations with 95% accuracy. What can you conclude?', options: ['The model uses this property to make its predictions', 'The property is linearly decodable from those activations; whether the model uses it needs interventions', 'The probe is overfitting', 'Layer 10 computes the property from scratch'], answer: 1, explain: 'Decodable is not the same as used; causal tests are needed.' } },
      { type: 'text', title: 'The logit lens and activation patching', md: `- **Logit lens**: apply the model's final normalization and unembedding to the residual stream at an *intermediate* layer to see which token it would predict "so far". Predictions typically sharpen layer by layer, showing where an answer emerges.
- **Activation patching** (causal tracing): run the model on a clean prompt ("The Eiffel Tower is in" → Paris) and on a corrupted one (with the subject replaced). Then rerun the corrupted prompt while copying in *one* component's activation from the clean run. If the correct answer comes back, that component carries the crucial information. Systematic patching located factual recall in middle-layer MLPs at the subject's last token, and makes "which heads matter for this behaviour?" an experimental question.` },
      { type: 'text', title: 'Superposition and sparse autoencoders', md: `Individual neurons are often **polysemantic**: one neuron fires for academic citations, Korean text and HTTP headers. One explanation is **superposition** — the model represents far more features than it has dimensions by using nearly perpendicular directions, which works because most features are rarely active at the same time.

**Sparse autoencoders (SAEs)** — a form of dictionary learning — try to undo this. They map an activation into a much wider, mostly-zero feature vector and back:

$$ f = ReLU( W_enc (h − b_dec) + b_enc )          ĥ = W_dec f + b_dec $$
$$ L = ‖ h − ĥ ‖² + λ · ‖ f ‖₁ $$
:::where
h — a model activation vector (length d)
f — the feature activations: a much longer vector (m ≫ d) in which most entries are zero
W_enc, b_enc — the encoder weights and bias
W_dec, b_dec — the decoder weights and bias
columns of W_dec — the learned feature directions (the "dictionary")
λ — the strength of the sparsity penalty
‖ f ‖₁ — the sum of absolute values (the L1 norm), which pushes most features to exactly zero
:::

Trained on a large model's activations, SAEs have found millions of interpretable features — from specific landmarks to code bugs to deception-related concepts — and clamping a feature on or off measurably changes the model's behaviour.` },
      { type: 'text', title: 'Circuits and steering', md: `- **Circuits** are small subnetworks that implement an algorithm. The best known is the **induction head** pair: a "previous-token" head writes "the token before me was A" into each position, and an induction head searches for an earlier position preceded by the current token and copies what came next ("…A B … A → B"). Induction heads appear suddenly during training, together with a jump in in-context learning ability.
- **Steering**: add a direction to the residual stream to push behaviour. A simple way to find one is the **difference of means** between activations on two contrasting prompt sets (e.g. positive versus negative reviews). Researchers have found directions that control refusals, sentiment, truthfulness and topic.
- **Limits**: explanations can be partial or misleading; probes find correlates; interventions can have side effects; scaling these methods to frontier models is an open research problem.` },
      { type: 'code', title: 'Steering with a difference-of-means direction', code: `
import numpy as np
rng = np.random.default_rng(1)
d = 64
direction = rng.normal(size=d)
direction /= np.linalg.norm(direction)                          # the concept, unknown to us
positive = rng.normal(size=(200, d)) + 2 * direction             # activations on "positive" prompts
negative = rng.normal(size=(200, d)) - 2 * direction             # activations on "negative" prompts
steer = positive.mean(axis=0) - negative.mean(axis=0)            # difference of means
steer /= np.linalg.norm(steer)
print("cosine(steering vector, true direction):", round(float(steer @ direction), 3))

h = rng.normal(size=d)                                           # a neutral activation
for alpha in (0, 2, 4, -4):
    print(f"alpha = {alpha:+d}: position along the concept = {(h + alpha * steer) @ direction:+.2f}")
`, after: `Averaging over many examples cancels the noise and leaves the concept direction; adding a multiple of it moves any activation along the concept, which is how steering shifts a model's behaviour.` },
      { type: 'check', question: { type: 'tf', q: 'A sparse autoencoder\'s L1 penalty encourages each activation to be explained by only a few active features.', answer: true, explain: 'Sparsity is what makes the learned features individually interpretable.' } }
    ],
    when: `- Debugging surprising model behaviour and validating that a model relies on the right signals.
- Safety research: monitoring internal features, detecting harmful capabilities, checking for deceptive reasoning.
- Building controllable models (steering) and understanding failures such as hallucinations.
- Teaching and research: small models (or toy models) are the best place to start.`,
    pitfalls: `- Treating probe accuracy as proof that the model uses the information.
- Forgetting controls (shuffled labels, random directions, baseline layers).
- Over-interpreting a single neuron or a few cherry-picked examples.
- Assuming a feature's label from its top activations without checking its effect when intervened on.
- Steering vectors that are too strong, which break fluency or have side effects.`,
    takeaways: [
      'Concepts often correspond to directions in activation space; linear probes test decodability.',
      'Activation patching and the logit lens locate where computations happen.',
      'Superposition packs more features than dimensions, making neurons polysemantic.',
      'Sparse autoencoders learn an overcomplete, sparse dictionary of more interpretable features.',
      'Circuits (e.g. induction heads) implement algorithms; difference-of-means directions can steer behaviour.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is activation patching used for?', options: ['Compressing activations', 'Finding which component carries the information responsible for a behaviour, by copying activations between runs', 'Fine-tuning a model', 'Removing biases'], answer: 1, explain: 'It is a causal test.' },
      { type: 'mcq', q: 'What does superposition mean in interpretability?', options: ['Two models trained together', 'Representing more features than dimensions using nearly perpendicular directions', 'Adding residual connections', 'Running attention twice'], answer: 1, explain: 'It explains polysemantic neurons.' },
      { type: 'tf', q: 'A sparse autoencoder\'s hidden layer is usually much wider than the activation it encodes.', answer: true, explain: 'An overcomplete dictionary with sparse codes.' },
      { type: 'mcq', q: 'What does an induction head do?', options: ['Normalizes activations', 'Finds an earlier occurrence of the current token and copies the token that followed it', 'Predicts the first token', 'Masks future tokens'], answer: 1, explain: 'A key mechanism behind in-context learning.' },
      { type: 'fill', q: 'A steering vector is computed as the mean activation on set A minus the mean on set B. What is this method called? (three words)', answer: ['difference of means', 'difference-of-means'], explain: 'Averaging cancels noise and leaves the contrasting direction.' }
    ],
    problems: [
      {
        id: 'linear-probe', title: 'Train a Linear Probe', difficulty: 'medium',
        prompt: `Implement:

- \`train_probe(H, y, lr=0.5, steps=300)\` → \`(w, b)\`: logistic regression trained by full-batch gradient descent, starting from w = 0 and b = 0. Each step: p = σ(H w + b); w −= lr · Hᵀ(p − y) / n; b −= lr · mean(p − y).
- \`probe_accuracy(H, y, w, b)\` → the fraction of rows where (H w + b > 0) equals the label.

\`H\` is an (n, d) array of activations and \`y\` holds 0/1 labels.`,
        starter: `
import numpy as np

def train_probe(H, y, lr=0.5, steps=300):
    # TODO
    pass

def probe_accuracy(H, y, w, b):
    # TODO
    pass
`,
        solution: `
import numpy as np

def train_probe(H, y, lr=0.5, steps=300):
    H = np.asarray(H, dtype=float)
    y = np.asarray(y, dtype=float)
    w, b = np.zeros(H.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(H @ w + b)))
        w = w - lr * H.T @ (p - y) / len(y)
        b = b - lr * float(np.mean(p - y))
    return w, b

def probe_accuracy(H, y, w, b):
    return float(np.mean(((np.asarray(H) @ w + b) > 0) == np.asarray(y)))
`,
        hints: ['The gradient of the log loss with respect to the scores is p − y.', 'Update w and b together from the same p in each step.'],
        explain: `A probe is ordinary logistic regression — the interpretability is in what you train it on. Its weight vector is a candidate direction for the property, which you can then test causally by steering or patching.`,
        tests: [
          { name: 'zero steps returns the zero probe', call: 'train_probe(np.ones((3, 2)), [0, 1, 1], steps=0)', expect: '(np.zeros(2), 0.0)' },
          { name: 'learns a planted direction', setup: 'rng = np.random.default_rng(0)\nd = 16\ndirection = rng.normal(size=d); direction /= np.linalg.norm(direction)\ny = rng.integers(0, 2, 300)\nH = rng.normal(size=(300, d)) + np.outer(2 * y - 1, direction) * 2', code: 'w, b = train_probe(H[:200], y[:200])\nassert probe_accuracy(H[200:], y[200:], w, b) > 0.9\nassert w @ direction / np.linalg.norm(w) > 0.7' },
          { name: 'reference run', setup: 'rng = np.random.default_rng(1)\nH = rng.normal(size=(40, 5)); y = (H[:, 0] + 0.5 * H[:, 1] > 0).astype(int)', call: 'train_probe(H, y, lr=0.3, steps=50)', hidden: true },
          { name: 'accuracy', call: 'probe_accuracy(np.array([[1.0], [-1.0], [2.0]]), [1, 0, 0], np.array([1.0]), 0.0)', expect: '2 / 3', hidden: true }
        ]
      },
      {
        id: 'sae-forward', title: 'A Sparse Autoencoder Forward Pass', difficulty: 'medium',
        prompt: `For a batch of activations h (shape (B, d)) implement:

- \`sae_forward(h, W_enc, b_enc, W_dec, b_dec)\` → \`(f, h_hat)\` with W_enc (m, d), b_enc (m,), W_dec (d, m) and b_dec (d,):
  f = ReLU((h − b_dec) @ W_encᵀ + b_enc), shape (B, m); h_hat = f @ W_decᵀ + b_dec, shape (B, d).
- \`sae_loss(h, f, h_hat, lam)\` → the batch mean of ‖h − ĥ‖² + λ‖f‖₁ (sum of squared errors plus λ times the sum of absolute feature values, per example).
- \`active_features(f)\` → the average number of non-zero features per example.`,
        starter: `
import numpy as np

def sae_forward(h, W_enc, b_enc, W_dec, b_dec):
    # TODO
    pass

def sae_loss(h, f, h_hat, lam):
    # TODO
    pass

def active_features(f):
    # TODO
    pass
`,
        solution: `
import numpy as np

def sae_forward(h, W_enc, b_enc, W_dec, b_dec):
    h = np.asarray(h, dtype=float)
    f = np.maximum(0, (h - b_dec) @ W_enc.T + b_enc)
    h_hat = f @ W_dec.T + b_dec
    return f, h_hat

def sae_loss(h, f, h_hat, lam):
    h, f, h_hat = (np.asarray(a, dtype=float) for a in (h, f, h_hat))
    recon = np.sum((h - h_hat) ** 2, axis=1)
    sparsity = np.sum(np.abs(f), axis=1)
    return float(np.mean(recon + lam * sparsity))

def active_features(f):
    return float(np.mean(np.sum(np.asarray(f) != 0, axis=1)))
`,
        hints: ['Subtract b_dec before encoding, add it back after decoding.', 'Sum over features for each example, then average over the batch.'],
        explain: `The decoder bias is subtracted before encoding so that features describe deviations from a typical activation. Training balances the two loss terms: a larger λ gives sparser, more interpretable features at the cost of worse reconstruction.`,
        tests: [
          { name: 'identity dictionary', call: 'sae_forward(np.array([[1.0, -2.0]]), np.eye(2), np.zeros(2), np.eye(2), np.zeros(2))', expect: '(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]))' },
          { call: 'sae_loss(np.array([[1.0, -2.0]]), np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), 0.5)', expect: '4.5' },
          { name: 'random SAE vs reference', setup: 'rng = np.random.default_rng(0)\nh = rng.normal(size=(5, 4)); W_enc = rng.normal(size=(12, 4)); b_enc = -0.5 * np.ones(12); W_dec = rng.normal(size=(4, 12)); b_dec = rng.normal(size=4)', call: '[sae_forward(h, W_enc, b_enc, W_dec, b_dec), active_features(sae_forward(h, W_enc, b_enc, W_dec, b_dec)[0])]', hidden: true },
          { name: 'loss on random data vs reference', setup: 'rng = np.random.default_rng(1)\nh = rng.normal(size=(6, 3)); f = np.maximum(0, rng.normal(size=(6, 10))); hh = rng.normal(size=(6, 3))', call: 'sae_loss(h, f, hh, 0.1)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Anthropic — Toy Models of Superposition', url: 'https://transformer-circuits.pub/2022/toy_model/index.html' },
      { title: 'Anthropic — Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet', url: 'https://transformer-circuits.pub/2024/scaling-monosemanticity/' },
      { title: 'Olsson et al. — In-context Learning and Induction Heads', url: 'https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html' },
      { title: 'Meng et al. — Locating and Editing Factual Associations in GPT (ROME)', url: 'https://arxiv.org/abs/2202.05262' }
    ]
  },

  // =====================================================================================
  {
    id: 'time-series', branch: 'spec', title: 'Time Series & Forecasting', short: 'Time Series', level: 2, minutes: 45, prereqs: ['linear-regression', 'cross-validation'],
    tags: ['time series', 'forecasting', 'trend', 'seasonality', 'autocorrelation', 'stationarity', 'lag features', 'exponential smoothing', 'arima', 'walk-forward validation', 'mase'],
    summary: 'Predicting the future from the past: trend, seasonality and noise, autocorrelation, baselines you must beat (naive, seasonal naive, moving average, exponential smoothing), autoregressive models with lag features, and honest evaluation with walk-forward validation and scale-free metrics.',
    why: `Sales, energy demand, website traffic, stock levels and sensor readings are all sequences over time, and forecasting them drives inventory, staffing and capacity planning. Time series break the core assumption of ordinary machine learning — that examples are independent — so a random train/test split leaks the future and produces fantasy results. A few simple methods and an honest validation scheme go a very long way.`,
    objectives: [
      'Decompose a series into trend, seasonality and noise, and read autocorrelation',
      'Build strong baselines: naive, seasonal naive, moving average and exponential smoothing',
      'Turn forecasting into supervised learning with lag features and fit an autoregressive model',
      'Evaluate with walk-forward validation and scale-free metrics such as MASE'
    ],
    sections: [
      { type: 'text', title: 'Components of a time series', md: `Most series mix three ingredients:

- **Trend** — a long-term rise or fall;
- **Seasonality** — patterns that repeat with a fixed period (hours of the day, days of the week, months of the year);
- **Noise** — what's left.

They combine **additively** (y = trend + season + noise) or **multiplicatively** (the seasonal swing grows with the level). **Autocorrelation** measures how much a series resembles itself shifted by k steps:

$$ ACF(k) = Σ_t (y_t − ȳ)(y_{t+k} − ȳ) / Σ_t (y_t − ȳ)² $$
:::where
y_t — the value at time t
ȳ — the mean of the series
k — the lag: how many steps the series is shifted
ACF(k) — the autocorrelation at lag k, between −1 and 1
:::

Daily data with a weekly pattern shows ACF peaks at lags 7, 14, 21, … A series is **stationary** when its statistics don't change over time; differencing (y_t − y_{t−1}) removes a trend and seasonal differencing (y_t − y_{t−7}) removes a weekly pattern.` },
      { type: 'text', title: 'Baselines you must beat', md: `Always compare a model with simple forecasts — surprisingly often they win:

- **Naive**: tomorrow equals today, ŷ_{t+h} = y_t.
- **Seasonal naive**: the same day last season, ŷ_{t+h} = y_{t+h−s} for season length s.
- **Moving average**: the mean of the last few values.
- **Simple exponential smoothing**: a weighted average that forgets old values gradually:

$$ ŷ_{t+1} = α · y_t + (1 − α) · ŷ_t $$
:::where
ŷ_{t+1} — the forecast for the next step
y_t — the latest observed value
ŷ_t — the previous forecast
α — "alpha": the smoothing factor between 0 and 1; a large α reacts quickly, a small α smooths heavily
:::

**Holt–Winters** extends exponential smoothing with a trend and a seasonal component and is a strong classic method.` },
      { type: 'code', title: 'Baselines on a seasonal series', code: `
import numpy as np
rng = np.random.default_rng(0)
n = 365
t = np.arange(n)
y = 100 + 0.1 * t + 12 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, n)   # trend + weekly season + noise
train, test = y[:-28], y[-28:]                                               # forecast the last 4 weeks

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

naive = np.full(28, train[-1])
seasonal_naive = np.array([train[-7 + (h % 7)] for h in range(28)])
moving_avg = np.full(28, train[-7:].mean())
level = train[0]
for v in train[1:]:
    level = 0.3 * v + 0.7 * level                                            # exponential smoothing
ses = np.full(28, level)
for name, forecast in [("naive", naive), ("seasonal naive", seasonal_naive), ("7-day moving average", moving_avg), ("exponential smoothing", ses)]:
    print(f"{name:22s} MAE {mae(forecast, test):6.2f}")
`, after: `Only the seasonal naive forecast knows about the weekly cycle, so it beats the others by a wide margin — a good reminder to understand the structure of a series before reaching for complex models.` },
      { type: 'check', question: { type: 'mcq', q: 'Daily sales have a strong weekly pattern. What does the seasonal naive forecast predict for next Monday?', options: ['Yesterday\'s sales', 'Last Monday\'s sales', 'The average of the whole history', 'Zero'], answer: 1, explain: 'The value one season (7 days) earlier.' } },
      { type: 'text', title: 'Forecasting as supervised learning', md: `Turn the series into a table: each row's target is y_t, and its features are **lags** (y_{t−1}, y_{t−7}, y_{t−14}), rolling statistics (the mean of the last 7 days), and calendar features (day of week, month, holidays). Any regression model can then learn from it — linear regression gives an **autoregressive** (AR) model:

$$ ŷ_t = c + φ₁ y_{t−1} + φ₂ y_{t−2} + … + φ_p y_{t−p} $$
:::where
c — a constant
φ_i — "phi i": the learned weight of the value i steps back
p — the order: how many past values the model uses
:::

**ARIMA** adds differencing (the "I") and terms for past forecast errors (the "MA"). Gradient-boosted trees on lag features are a strong practical choice, and neural forecasters (RNNs, temporal convolutions, transformers, pretrained time-series foundation models) shine with many related series.

For several steps ahead, either forecast **recursively** (feed predictions back in as lags) or train a **direct** model for each horizon.` },
      { type: 'code', title: 'An autoregressive model with lag features', code: `
import numpy as np
rng = np.random.default_rng(0)
n = 365
t = np.arange(n)
y = 100 + 0.1 * t + 12 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, n)
train = y[:-28]
lags = [1, 7, 14]

def make_lags(series, lags):
    m = max(lags)
    X = np.column_stack([series[m - L:len(series) - L] for L in lags])
    return X, series[m:]

X, target = make_lags(train, lags)
coef = np.linalg.lstsq(np.c_[np.ones(len(X)), X], target, rcond=None)[0]
print("intercept and lag weights (lags 1, 7, 14):", coef.round(3))

history, preds = list(train), []
for h in range(28):                                         # recursive forecast: predictions become lags
    p = float(np.r_[1.0, [history[-L] for L in lags]] @ coef)
    preds.append(p)
    history.append(p)
seasonal_naive = np.array([train[-7 + (h % 7)] for h in range(28)])
print(f"AR model MAE {np.mean(np.abs(np.array(preds) - y[-28:])):.2f}    seasonal naive MAE {np.mean(np.abs(seasonal_naive - y[-28:])):.2f}")
`, after: `Almost all the weight sits on lags 7 and 14 — the model discovered the weekly cycle by itself — and averaging the same weekday from two past weeks cancels part of the noise, which is why it beats the seasonal naive forecast.` },
      { type: 'text', title: 'Honest evaluation: walk-forward validation', md: `Never shuffle time series. **Walk-forward** (rolling-origin) validation trains on the past and tests on the period right after it, then moves forward and repeats:

\`\`\`text
fold 1:  train [######        ]  test [###]
fold 2:  train [#########     ]  test [###]
fold 3:  train [############  ]  test [###]
\`\`\`

Metrics: MAE and RMSE in the data's units; MAPE (percentage error) is intuitive but breaks down near zero. **MASE** compares a model with the naive forecast and works across series of different scales:

$$ MASE = MAE_model / MAE_naive( in-sample ) $$
:::where
MAE_model — the model's mean absolute error on the test period
MAE_naive( in-sample ) — the mean absolute error of the (seasonal) naive forecast on the training data
MASE below 1 means the model beats the naive method.
:::

Watch for leakage: features computed with future values (centered rolling windows, statistics over the whole series), and data that is revised or arrives late in reality.` },
      { type: 'check', question: { type: 'tf', q: 'Randomly shuffling a time series before splitting into train and test gives an honest estimate of forecasting accuracy.', answer: false, explain: 'It lets the model learn from the future; use walk-forward splits.' } }
    ],
    when: `- Demand, sales, traffic, energy and capacity forecasting; sensor monitoring.
- Start with seasonal naive and exponential smoothing; move to lag-feature models or ARIMA when they beat the baselines in walk-forward validation.
- Many related series (thousands of products): global models (one model across all series) with gradient boosting or neural networks.`,
    pitfalls: `- Random splits or features that peek into the future.
- Ignoring seasonality, holidays and calendar effects.
- Evaluating only one-step-ahead when the business needs forecasts weeks ahead.
- MAPE on series with zeros or tiny values.
- Not re-checking models when the world changes (concept drift).`,
    takeaways: [
      'Series = trend + seasonality + noise; autocorrelation reveals the season length.',
      'Baselines (naive, seasonal naive, exponential smoothing) are strong — always beat them first.',
      'Lag features turn forecasting into supervised learning; AR/ARIMA are the classical models.',
      'Forecast multiple steps recursively or with one model per horizon.',
      'Validate walk-forward, never with random splits; MASE compares against the naive method.'
    ],
    quiz: [
      { type: 'mcq', q: 'Hourly electricity demand shows ACF peaks at lags 24, 48 and 168. What does that suggest?', options: ['No pattern', 'Daily and weekly seasonality', 'A trend only', 'Pure noise'], answer: 1, explain: '24 hours and 168 hours = one week.' },
      { type: 'fill', q: 'Exponential smoothing with α = 0.5: the previous forecast was 10 and the new observation is 20. What is the next forecast?', answer: ['15', '15.0'], explain: '0.5 · 20 + 0.5 · 10 = 15.' },
      { type: 'mcq', q: 'What is walk-forward validation?', options: ['Shuffling and splitting k times', 'Training on the past and testing on the following period, repeatedly moving forward in time', 'Testing on the first part of the series', 'Validating on random days'], answer: 1, explain: 'It mirrors how forecasts are used in reality.' },
      { type: 'tf', q: 'A MASE below 1 means the model beats the naive forecast.', answer: true, explain: 'MASE is the model\'s error divided by the naive method\'s error.' },
      { type: 'mcq', q: 'Why might differencing (y_t − y_{t−1}) help?', options: ['It removes the trend and makes the series closer to stationary', 'It adds seasonality', 'It increases the data size', 'It removes noise completely'], answer: 0, explain: 'Many models assume stable statistics over time.' }
    ],
    problems: [
      {
        id: 'lag-features', title: 'Lag Features and Seasonal Naive', difficulty: 'easy',
        prompt: `Implement:

- \`make_lags(y, lags)\` → \`(X, target)\`: with m = max(lags), the target is y[m:] and column j of X holds y shifted by lags[j], i.e. X[i, j] = y[m + i − lags[j]].
- \`seasonal_naive(y, season, horizon)\` → a forecast of length \`horizon\` that repeats the last \`season\` values: forecast[h] = y[len(y) − season + (h mod season)].

\`\`\`text
make_lags([1, 2, 3, 4, 5], [1, 2])  ->  X = [[2, 1], [3, 2], [4, 3]], target = [3, 4, 5]
seasonal_naive([1, 2, 3, 4, 5, 6], 3, 5)  ->  [4, 5, 6, 4, 5]
\`\`\``,
        starter: `
import numpy as np

def make_lags(y, lags):
    # TODO
    pass

def seasonal_naive(y, season, horizon):
    # TODO
    pass
`,
        solution: `
import numpy as np

def make_lags(y, lags):
    y = np.asarray(y, dtype=float)
    m = max(lags)
    X = np.column_stack([y[m - L:len(y) - L] for L in lags])
    return X, y[m:]

def seasonal_naive(y, season, horizon):
    y = np.asarray(y, dtype=float)
    return np.array([y[len(y) - season + (h % season)] for h in range(horizon)])
`,
        hints: ['For lag L, the column is the slice y[m − L : len(y) − L].', '`np.column_stack` puts the columns side by side.'],
        explain: `Aligning every lag column with the target row is the whole trick of turning a series into a supervised-learning table. Rows before max(lags) are dropped because their lags would reach before the start of the data.`,
        tests: [
          { call: 'make_lags([1, 2, 3, 4, 5], [1, 2])', expect: '(np.array([[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]]), np.array([3.0, 4.0, 5.0]))' },
          { call: 'seasonal_naive([1, 2, 3, 4, 5, 6], 3, 5)', expect: 'np.array([4.0, 5.0, 6.0, 4.0, 5.0])' },
          { name: 'columns are true lags', setup: 'rng = np.random.default_rng(0)\ny = rng.normal(size=50)', code: 'X, t = make_lags(y, [1, 7, 14])\nassert X.shape == (36, 3)\nfor i in range(36):\n    assert X[i, 0] == y[14 + i - 1] and X[i, 2] == y[i] and t[i] == y[14 + i]', hidden: true },
          { name: 'random series vs reference', setup: 'rng = np.random.default_rng(1)\ny = rng.normal(size=30)', call: '[make_lags(y, [2, 3]), seasonal_naive(y, 7, 10)]', hidden: true }
        ]
      },
      {
        id: 'smoothing-backtest', title: 'Exponential Smoothing and Walk-Forward Splits', difficulty: 'medium',
        prompt: `Implement:

- \`exp_smooth(y, alpha)\` → one-step-ahead forecasts f of the same length as y: f[0] = y[0], and f[t] = α·y[t−1] + (1 − α)·f[t−1] for t ≥ 1.
- \`walk_forward_splits(n, initial, horizon)\` → a list of \`(train_end, test_end)\` pairs: training uses indices [0, train_end), testing [train_end, test_end). Start with train_end = initial, step forward by \`horizon\`, and only include folds whose test window fits (train_end + horizon ≤ n).
- \`mase(y_true, y_pred, y_train, season=1)\` → MAE(y_true, y_pred) divided by the mean of |y_train[t] − y_train[t − season]| over the training data.

\`\`\`text
walk_forward_splits(10, 4, 3)  ->  [(4, 7), (7, 10)]
\`\`\``,
        starter: `
import numpy as np

def exp_smooth(y, alpha):
    # TODO
    pass

def walk_forward_splits(n, initial, horizon):
    # TODO
    pass

def mase(y_true, y_pred, y_train, season=1):
    # TODO
    pass
`,
        solution: `
import numpy as np

def exp_smooth(y, alpha):
    y = np.asarray(y, dtype=float)
    f = np.empty_like(y)
    f[0] = y[0]
    for t in range(1, len(y)):
        f[t] = alpha * y[t - 1] + (1 - alpha) * f[t - 1]
    return f

def walk_forward_splits(n, initial, horizon):
    splits = []
    end = initial
    while end + horizon <= n:
        splits.append((end, end + horizon))
        end += horizon
    return splits

def mase(y_true, y_pred, y_train, season=1):
    y_true, y_pred, y_train = (np.asarray(a, dtype=float) for a in (y_true, y_pred, y_train))
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return float(np.mean(np.abs(y_true - y_pred)) / scale)
`,
        hints: ['Loop forward in time: each forecast uses the previous forecast and the previous observation.', 'The naive in-sample error compares y_train[season:] with y_train[:-season].'],
        explain: `Exponential smoothing needs only the last forecast and the last value, so it updates in constant time — ideal for streaming data. Walk-forward folds always test on data that comes after their training window, just as a real deployment would.`,
        tests: [
          { call: 'exp_smooth([10.0, 20.0, 20.0], 0.5)', expect: 'np.array([10.0, 10.0, 15.0])' },
          { call: 'walk_forward_splits(10, 4, 3)', expect: '[(4, 7), (7, 10)]' },
          { name: 'MASE by hand', call: 'mase([4.0, 6.0], [3.0, 4.0], [1.0, 2.0, 3.0], 1)', expect: '1.5' },
          { name: 'alpha = 1 is the naive forecast', setup: 'rng = np.random.default_rng(0)\ny = rng.normal(size=20)', code: 'f = exp_smooth(y, 1.0)\nassert np.allclose(f[1:], y[:-1])', hidden: true },
          { name: 'random series vs reference', setup: 'rng = np.random.default_rng(1)\ny = np.cumsum(rng.normal(size=40))', call: '[exp_smooth(y, 0.3), walk_forward_splits(40, 25, 4), mase(y[30:], np.full(10, y[29]), y[:30], 2)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Hyndman & Athanasopoulos — Forecasting: Principles and Practice (free book)', url: 'https://otexts.com/fpp3/' },
      { title: 'statsmodels — Time series analysis', url: 'https://www.statsmodels.org/stable/tsa.html' },
      { title: 'Makridakis et al. — The M5 forecasting competition', url: 'https://doi.org/10.1016/j.ijforecast.2021.11.013' }
    ]
  },

  // =====================================================================================
  {
    id: 'anomaly-detection', branch: 'spec', title: 'Anomaly Detection', level: 2, minutes: 40, prereqs: ['statistics', 'decision-trees'],
    tags: ['anomaly detection', 'outliers', 'z-score', 'mad', 'iqr', 'isolation forest', 'local outlier factor', 'reconstruction error', 'novelty detection', 'fraud detection'],
    summary: 'Finding the unusual: statistical rules (z-scores, robust MAD scores, IQR fences), isolation forests that isolate outliers with a few random splits, distance-, density- and reconstruction-based methods, and how to set thresholds and evaluate when anomalies are rare and labels are scarce.',
    why: `Fraud, intrusions, failing machines, data-pipeline bugs and medical emergencies are rare by definition — there are too few labelled examples to train an ordinary classifier. **Anomaly detection** learns what "normal" looks like and flags what doesn't fit. It is also a data-quality tool: catching broken records before they poison a model or a dashboard.`,
    objectives: [
      'Distinguish outlier detection, novelty detection and supervised rare-event classification',
      'Apply z-score, robust (MAD) and IQR rules, and know when each fails',
      'Explain isolation forests and reconstruction-error methods',
      'Set thresholds and evaluate with precision@k and precision–recall when anomalies are rare'
    ],
    sections: [
      { type: 'text', title: 'What counts as an anomaly?', md: `- **Point anomalies**: a single value far from the rest (a $9,000 coffee).
- **Contextual anomalies**: normal in general but not in context (30 °C is fine in July, alarming in January).
- **Collective anomalies**: a sequence that is unusual as a whole (a burst of failed logins).

Settings differ by the labels you have: **unsupervised** (no labels — find the odd ones out), **novelty detection** (train on known-normal data only, flag anything new) and **supervised** (some labelled anomalies — a heavily imbalanced classification problem).` },
      { type: 'text', title: 'Statistical rules', md: `The classic **z-score** measures how many standard deviations a value is from the mean:

$$ z = ( x − μ ) / σ $$
:::where
x — the value being checked
μ, σ — the mean and the standard deviation of the data
A common rule flags |z| > 3.
:::

The weakness: outliers inflate μ and σ themselves, which can hide them (**masking**). The **robust z-score** uses the median and the median absolute deviation instead, which outliers barely move:

$$ robust z = 0.6745 · ( x − median ) / MAD          MAD = median( |x − median| ) $$
:::where
median — the middle value of the data
MAD — the median absolute deviation from the median
0.6745 — a constant that makes the robust score comparable to an ordinary z-score for normal data
A common rule flags |robust z| > 3.5.
:::

The **IQR rule** (behind box plots) flags values below Q1 − 1.5·IQR or above Q3 + 1.5·IQR, where Q1 and Q3 are the 25th and 75th percentiles and IQR = Q3 − Q1.` },
      { type: 'code', title: 'z-score versus robust rules', code: `
import numpy as np
rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(50, 5, 200), rng.normal(100, 10, 30)])     # 200 normal values, then 30 anomalies
is_anomaly = np.r_[np.zeros(200, dtype=bool), np.ones(30, dtype=bool)]
z = (x - x.mean()) / x.std()
med = np.median(x)
mad = np.median(np.abs(x - med))
robust = 0.6745 * (x - med) / mad
q1, q3 = np.percentile(x, [25, 75])
iqr_flag = (x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))
print(f"mean {x.mean():.1f}, std {x.std():.1f}    median {med:.1f}, MAD {mad:.1f}    (normal data: mean 50, std 5)")
for name, flag in [("|z| > 3", np.abs(z) > 3), ("|robust z| > 3.5", np.abs(robust) > 3.5), ("IQR rule", iqr_flag)]:
    print(f"{name:17s} catches {np.sum(flag & is_anomaly):2d} of 30 anomalies, with {np.sum(flag & ~is_anomaly)} false alarms")
`, after: `The anomalies drag the mean up and inflate the standard deviation to more than three times its normal size, so the plain z-score catches only a handful of them — that is masking. The median, the MAD and the quartiles barely move, so both robust rules catch every anomaly without a single false alarm.` },
      { type: 'check', question: { type: 'mcq', q: 'Why can the ordinary z-score miss outliers?', options: ['It only works on integers', 'The outliers inflate the mean and standard deviation used to compute it (masking)', 'It ignores large values', 'It needs labels'], answer: 1, explain: 'Robust statistics such as the median and MAD avoid this.' } },
      { type: 'text', title: 'Isolation forests', md: `**Isolation forests** turn the question around: instead of modelling normal data, they measure how **easy a point is to isolate**. Build many random trees: at each node pick a random feature and a random split value between its minimum and maximum, until points are separated. Anomalies — few and different — get isolated after very few splits; normal points sit in dense regions and need many.

$$ s(x) = 2^( −E[h(x)] / c(n) ) $$
:::where
h(x) — the path length: how many splits it took to isolate x in one tree
E[h(x)] — its average over all the trees
c(n) — the average path length expected for n points, used to normalize: c(n) = 2(ln(n − 1) + 0.5772) − 2(n − 1)/n
s(x) — the anomaly score: close to 1 for anomalies, around 0.5 or below for normal points
:::

They are fast, scale to many features, need no distance computations, and work well as a first method on tabular data.` },
      { type: 'text', title: 'Distance, density and reconstruction', md: `- **Distance-based**: the distance to the k-th nearest neighbour; far from everyone = suspicious.
- **Density-based**: the **Local Outlier Factor** compares a point's local density with its neighbours' — it finds points that are sparse *relative to their surroundings*.
- **One-class SVM**: learns a boundary around normal data.
- **Reconstruction error**: train PCA or an autoencoder on (mostly) normal data; anomalies don't fit the learned structure and reconstruct poorly.
- **Forecast residuals** for time series: flag values far from what a forecasting model expected.` },
      { type: 'code', title: 'Anomalies that only show in combination', code: `
import numpy as np
from sklearn.ensemble import IsolationForest
rng = np.random.default_rng(0)
x1 = rng.normal(size=500)
normal = np.column_stack([x1, 0.8 * x1 + 0.6 * rng.normal(size=500)])      # two features that rise together
size = rng.uniform(1.5, 2.5, (10, 1))
sign = rng.choice([-1.0, 1.0], (10, 1))
odd = sign * size * np.array([1.0, -1.0])                                   # anomalies like (2, -2) break the pattern
X = np.vstack([normal, odd])
is_anomaly = np.r_[np.zeros(500), np.ones(10)]

per_feature = np.max(np.abs((X - X.mean(axis=0)) / X.std(axis=0)), axis=1)  # each feature on its own
iso_score = -IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X)  # higher = more anomalous

Xc = X - X.mean(axis=0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
recon = (Xc @ Vt[:1].T) @ Vt[:1]                                            # keep only the main direction
pca_score = np.sum((Xc - recon) ** 2, axis=1)                               # squared distance from that line

for name, score in [("per-feature |z|", per_feature), ("isolation forest", iso_score), ("PCA reconstruction error", pca_score)]:
    top = np.argsort(-score)[:10]
    print(f"{name:26s} precision@10 = {is_anomaly[top].mean():.1f}")
`, after: `Each anomaly looks ordinary feature by feature — only the combination is odd, because the two features normally rise together. So per-feature z-scores put none of them in their top ten, while PCA reconstruction error, which models the correlation directly, ranks all ten at the top. The isolation forest lands in between: its splits test one feature at a time (axis-aligned), a known weakness for this kind of anomaly that the extended isolation forest addresses with random slanted splits.` },
      { type: 'text', title: 'Thresholds and evaluation', md: `Anomaly scores still need a **threshold**:
- by an expected rate ("flag the top 0.5%");
- by a review **budget** ("analysts can check 50 cases a day" — then measure **precision@k**);
- with a small labelled set, by choosing a point on the **precision–recall curve** (ROC curves look deceptively good when positives are rare).

In production, the analysts' verdicts become labels: over time you can move to semi-supervised or supervised models. And remember that "normal" drifts — retrain and re-check thresholds regularly.` },
      { type: 'check', question: { type: 'tf', q: 'In an isolation forest, anomalies tend to have shorter average path lengths than normal points.', answer: true, explain: 'Few, different points are separated by few random splits.' } }
    ],
    when: `- Fraud, intrusion and abuse detection; predictive maintenance; monitoring business metrics.
- Data validation: catching broken records and pipeline bugs before training or reporting.
- Robust rules for a single variable; isolation forests for tabular data; reconstruction or forecasting residuals for images, signals and time series.
- Switch to supervised classification once you have enough labelled anomalies.`,
    pitfalls: `- Mean/standard-deviation rules on data that already contains outliers (masking).
- Evaluating with accuracy or ROC AUC alone when anomalies are rare — use precision@k and PR curves.
- Unscaled features dominating distance-based methods.
- Checking features one at a time when anomalies only show in combinations of values (a normal height with an impossible weight) — use multivariate methods.
- Thresholds that never adapt as normal behaviour drifts.
- Treating every statistical outlier as an error — some are the most interesting cases.`,
    takeaways: [
      'Anomalies can be point, contextual or collective; label availability decides the approach.',
      'z-scores suffer from masking; median/MAD and IQR rules are robust.',
      'Isolation forests score points by how quickly random splits isolate them.',
      'Reconstruction error (PCA, autoencoders) flags points that don\'t fit normal structure.',
      'Pick thresholds by rate, budget or precision–recall; evaluate with precision@k.'
    ],
    quiz: [
      { type: 'fill', q: 'Data has median 100 and MAD 10. What is the robust z-score of x = 160? (2 decimals)', answer: ['4.05', '4.047', '4.0'], explain: '0.6745 × (160 − 100) / 10 ≈ 4.05.' },
      { type: 'mcq', q: 'Q1 = 20 and Q3 = 30. Which value is flagged by the IQR rule?', options: ['14', '38', '46', '25'], answer: 2, explain: 'Fences: 20 − 15 = 5 and 30 + 15 = 45; only 46 is outside.' },
      { type: 'mcq', q: 'What does an isolation forest measure?', options: ['Distance to the class centre', 'How few random splits it takes to isolate a point', 'Reconstruction error', 'The number of neighbours'], answer: 1, explain: 'Short paths mean anomalous.' },
      { type: 'tf', q: 'With 0.1% anomalies, a detector that flags nothing has 99.9% accuracy — so accuracy is a poor metric here.', answer: true, explain: 'Use precision, recall and precision@k instead.' },
      { type: 'mcq', q: 'Novelty detection means…', options: ['Training on data that contains anomalies', 'Training only on known-normal data and flagging anything that looks new', 'Detecting new features', 'Supervised classification'], answer: 1, explain: 'A semi-supervised setting.' }
    ],
    problems: [
      {
        id: 'robust-outliers', title: 'Robust Outlier Rules', difficulty: 'easy',
        prompt: `Implement:

- \`zscores(x)\` → (x − mean) / std (population std, NumPy's default)
- \`robust_zscores(x)\` → 0.6745 · (x − median) / MAD with MAD = median(|x − median|); if MAD is 0, return all zeros
- \`iqr_outliers(x)\` → a boolean array marking values below Q1 − 1.5·IQR or above Q3 + 1.5·IQR (\`np.percentile(x, [25, 75])\`)

\`\`\`text
robust_zscores([1, 2, 3, 4, 100])  ->  [-1.349, -0.6745, 0.0, 0.6745, 65.43]   (median 3, MAD 1)
\`\`\``,
        starter: `
import numpy as np

def zscores(x):
    # TODO
    pass

def robust_zscores(x):
    # TODO
    pass

def iqr_outliers(x):
    # TODO
    pass
`,
        solution: `
import numpy as np

def zscores(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()

def robust_zscores(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

def iqr_outliers(x):
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)
`,
        hints: ['`np.median(np.abs(x - med))` is the MAD.', 'Combine the two fences with `|`.'],
        explain: `The median and the quartiles depend only on the middle of the data, so a handful of extreme values can't drag them around — that's what "robust" means here.`,
        tests: [
          { call: 'robust_zscores([1, 2, 3, 4, 100])', expect: '0.6745 * (np.array([1, 2, 3, 4, 100]) - 3) / 1.0' },
          { call: 'iqr_outliers([1, 2, 3, 4, 100])', expect: 'np.array([False, False, False, False, True])' },
          { name: 'z-scores', call: 'zscores([2.0, 4.0, 6.0])', expect: 'np.array([-1.0, 0.0, 1.0]) / np.sqrt(8 / 3) * 2' },
          { name: 'constant data', call: 'robust_zscores([5, 5, 5, 5])', expect: 'np.zeros(4)', hidden: true },
          { name: 'random data vs reference', setup: 'rng = np.random.default_rng(0)\nx = np.r_[rng.normal(0, 1, 50), [8.0, -9.0]]', call: '[zscores(x), robust_zscores(x), iqr_outliers(x)]', hidden: true }
        ]
      },
      {
        id: 'isolation-depth', title: 'Isolation Path Lengths', difficulty: 'medium',
        prompt: `Implement the pieces of an isolation forest:

- \`c_factor(n)\` → 0 for n ≤ 1, 1 for n = 2, otherwise 2·(ln(n − 1) + 0.5772156649) − 2·(n − 1)/n.
- \`anomaly_score(mean_path, n)\` → 2^(−mean_path / c_factor(n)).
- \`isolation_depth(point, data, rng, depth=0, max_depth=10)\` → the path length of \`point\` in one random isolation tree built on \`data\` (an (n, d) array):
  - if len(data) ≤ 1 or depth ≥ max_depth: return depth + c_factor(len(data));
  - pick a feature \`j = int(rng.integers(data.shape[1]))\`; if all values of feature j are equal, return depth + c_factor(len(data));
  - draw \`split = rng.uniform(lo, hi)\` between that feature's min and max;
  - keep the rows on the same side as the point (\`data[:, j] < split\` if point[j] < split, otherwise the others) and recurse with depth + 1.`,
        starter: `
import math
import numpy as np

def c_factor(n):
    # TODO
    pass

def anomaly_score(mean_path, n):
    # TODO
    pass

def isolation_depth(point, data, rng, depth=0, max_depth=10):
    # TODO
    pass
`,
        solution: `
import math
import numpy as np

def c_factor(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

def anomaly_score(mean_path, n):
    return 2 ** (-mean_path / c_factor(n))

def isolation_depth(point, data, rng, depth=0, max_depth=10):
    data = np.asarray(data, dtype=float)
    if len(data) <= 1 or depth >= max_depth:
        return depth + c_factor(len(data))
    j = int(rng.integers(data.shape[1]))
    lo, hi = data[:, j].min(), data[:, j].max()
    if lo == hi:
        return depth + c_factor(len(data))
    split = rng.uniform(lo, hi)
    left = data[:, j] < split
    if point[j] < split:
        return isolation_depth(point, data[left], rng, depth + 1, max_depth)
    return isolation_depth(point, data[~left], rng, depth + 1, max_depth)
`,
        hints: ['Use the random generator in exactly the order described so results are reproducible.', 'c_factor(len(data)) estimates the remaining depth when the recursion stops early.'],
        explain: `Averaging isolation_depth over many trees gives E[h(x)]; dividing by c(n) puts scores from datasets of different sizes on one scale, so 0.5 means "as hard to isolate as a typical point".`,
        tests: [
          { call: '[c_factor(1), c_factor(2), round(c_factor(256), 4)]', expect: '[0.0, 1.0, 10.2448]' },
          { name: 'average depth gives a score of 0.5', call: 'anomaly_score(c_factor(100), 100)', expect: '0.5' },
          { name: 'reference tree', setup: 'rng0 = np.random.default_rng(0)\ndata = rng0.normal(size=(64, 3))', call: 'isolation_depth(data[5], data, np.random.default_rng(1))', hidden: true },
          { name: 'outliers are isolated faster', setup: 'rng0 = np.random.default_rng(2)\ndata = np.vstack([rng0.normal(size=(200, 2)), [[6.0, 6.0]]])', code: 'rng = np.random.default_rng(3)\nout = np.mean([isolation_depth(data[-1], data, rng) for _ in range(100)])\ninner = np.mean([isolation_depth(data[0], data, rng) for _ in range(100)])\nassert out < inner, (out, inner)\nassert anomaly_score(out, len(data)) > 0.6', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Liu, Ting & Zhou — Isolation Forest', url: 'https://doi.org/10.1109/ICDM.2008.17' },
      { title: 'scikit-learn — Novelty and outlier detection', url: 'https://scikit-learn.org/stable/modules/outlier_detection.html' },
      { title: 'Chandola, Banerjee & Kumar — Anomaly Detection: A Survey', url: 'https://doi.org/10.1145/1541880.1541882' }
    ]
  },

  // =====================================================================================
  {
    id: 'detection-segmentation', branch: 'spec', title: 'Object Detection & Segmentation', short: 'Detection & Segmentation', level: 3, minutes: 45, prereqs: ['cnn', 'evaluation-metrics'],
    tags: ['object detection', 'bounding box', 'iou', 'non-maximum suppression', 'yolo', 'anchors', 'average precision', 'map', 'semantic segmentation', 'instance segmentation', 'u-net', 'dice'],
    summary: 'Beyond "what is in this image?" to "where?": bounding boxes and intersection over union, one-stage detectors such as YOLO, non-maximum suppression, average precision, and pixel-level segmentation with encoder–decoder networks such as U-Net, evaluated with IoU and the Dice score.',
    why: `Self-driving cars, medical imaging, retail shelf analysis, document layout parsing and photo editing all need to know **where** things are, not just whether they're present. Detection draws boxes around objects; segmentation labels every pixel. The building blocks — IoU, non-maximum suppression, encoder–decoder CNNs with skip connections — recur throughout computer vision.`,
    objectives: [
      'Represent boxes and compute intersection over union (IoU)',
      'Explain how one-stage detectors predict boxes and why non-maximum suppression is needed',
      'Compute precision, recall and average precision for detections',
      'Describe semantic and instance segmentation, U-Net and the Dice score'
    ],
    sections: [
      { type: 'text', title: 'Vision tasks, from coarse to fine', md: `| Task | Output |
| --- | --- |
| Classification | one label for the whole image |
| Localization | one label and one box |
| Object detection | a box, a label and a confidence for **every** object |
| Semantic segmentation | a class for every pixel (all cars share one label) |
| Instance segmentation | a separate mask for each object (car #1, car #2) |
| Keypoint / pose estimation | positions of joints or landmarks |` },
      { type: 'text', title: 'Boxes and IoU', md: `A box is usually stored as its corners (x1, y1, x2, y2). How well does a predicted box match a true one? **Intersection over union**:

$$ IoU(A, B) = area( A ∩ B ) / area( A ∪ B ) $$
:::where
A, B — the two boxes (or masks)
A ∩ B — their overlap (intersection)
A ∪ B — the region covered by either (union) = area(A) + area(B) − area(A ∩ B)
IoU — from 0 (no overlap) to 1 (identical)
:::

A detection usually counts as correct when its IoU with an unmatched ground-truth box of the same class is at least 0.5.` },
      { type: 'text', title: 'How detectors work', md: `- **Two-stage** detectors (Faster R-CNN) first propose regions likely to contain objects, then classify and refine each one.
- **One-stage** detectors (YOLO, SSD, RetinaNet) run a CNN once over the image. For every cell of a coarse grid (and several anchor shapes per cell) they predict an **objectness** score, class probabilities and box offsets — fast enough for real-time video.
- **Transformer detectors** (DETR) predict a fixed set of boxes directly and learn to avoid duplicates.

Training combines a classification loss, a box-regression loss (L1 or an IoU-based loss) and an objectness loss. Because neighbouring cells and anchors all fire on the same object, one-stage detectors output many overlapping boxes, which **non-maximum suppression (NMS)** cleans up: sort the boxes by score; keep the best; remove every other box that overlaps it with IoU above a threshold; repeat with the remaining boxes.` },
      { type: 'code', title: 'IoU and non-maximum suppression', code: `
import numpy as np

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)

def nms(boxes, scores, threshold=0.5):
    order = list(np.argsort(-np.asarray(scores), kind="stable"))
    keep = []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        order = [i for i in order if iou(boxes[best], boxes[i]) <= threshold]
    return keep

boxes = [(10, 10, 60, 60), (12, 8, 62, 58), (15, 15, 65, 65), (100, 100, 150, 140), (102, 98, 148, 142)]
scores = [0.90, 0.85, 0.60, 0.80, 0.75]
print("IoU of boxes 0 and 1:", round(iou(boxes[0], boxes[1]), 3))
print("IoU of boxes 0 and 3:", iou(boxes[0], boxes[3]))
print("kept after NMS:", nms(boxes, scores))
`, after: `Three boxes cover the first object and two cover the second; NMS keeps the highest-scoring box of each group.` },
      { type: 'check', question: { type: 'fill', q: 'Box A has area 100, box B has area 100, and they overlap in an area of 50. What is their IoU? (2 decimals)', answer: ['0.33', '0.333', '1/3'], explain: '50 / (100 + 100 − 50) = 50 / 150 ≈ 0.33.' } },
      { type: 'text', title: 'Evaluating detectors: average precision', md: `For one class: sort all detections by confidence. Walk down the list: a detection is a **true positive** if it matches a not-yet-matched ground-truth box with IoU ≥ 0.5, otherwise a **false positive**. After each detection, compute precision and recall; **average precision** summarizes the resulting precision–recall curve:

$$ AP = Σ_k ( R_k − R_{k−1} ) · P_k $$
:::where
k — the position in the confidence-sorted list of detections
P_k — the precision of the top k detections: true positives so far divided by k
R_k — the recall of the top k: true positives so far divided by the number of ground-truth objects (R_0 = 0)
:::

**mAP** is the mean AP over classes. The COCO benchmark also averages over IoU thresholds from 0.5 to 0.95, rewarding tight boxes.` },
      { type: 'text', title: 'Segmentation', md: `**Semantic segmentation** predicts a class for every pixel with an **encoder–decoder** network. The encoder shrinks the image into rich, low-resolution features (what is there); the decoder upsamples back to full resolution (where it is). **U-Net** adds **skip connections** that copy high-resolution encoder features directly to the matching decoder level, so fine boundaries survive — it remains a workhorse in medical imaging and is the backbone of many diffusion models.

The loss is per-pixel cross-entropy, often combined with the **Dice** loss, which handles small objects better. Both IoU and Dice measure the overlap of masks:

$$ Dice = 2 · |P ∩ G| / ( |P| + |G| ) $$
:::where
P — the set of pixels predicted as the object
G — the set of ground-truth pixels
|…| — the number of pixels in a set
Dice ranges from 0 (no overlap) to 1 (perfect) and is always at least as large as IoU.
:::

**Instance segmentation** (Mask R-CNN) adds a mask head to a detector. Promptable models such as SAM segment whatever you click on or describe.` },
      { type: 'code', title: 'Dice and IoU for masks', code: `
import numpy as np
gt = np.zeros((8, 8), dtype=bool)
gt[2:6, 2:6] = True                                   # the true object: a 4 x 4 square
pred = np.zeros((8, 8), dtype=bool)
pred[3:7, 2:6] = True                                 # the prediction is shifted down by one row
inter, union = int(np.sum(gt & pred)), int(np.sum(gt | pred))
print("intersection", inter, "  union", union)
print(f"IoU  = {inter / union:.3f}")
print(f"Dice = {2 * inter / (gt.sum() + pred.sum()):.3f}")
print(pred.astype(int) + 2 * gt.astype(int))          # 3 = correct, 2 = missed, 1 = false alarm
` },
      { type: 'check', question: { type: 'tf', q: 'Non-maximum suppression removes boxes that overlap a higher-scoring box by more than a threshold.', answer: true, explain: 'It keeps one box per object.' } }
    ],
    when: `- Detection for counting and locating objects: vehicles, people, products, defects, text regions.
- Semantic segmentation for per-pixel understanding: medical scans, satellite imagery, road scenes, background removal.
- Instance segmentation when objects of the same class must be separated.
- Start from pretrained detectors and segmenters and fine-tune on your images.`,
    pitfalls: `- Mixing box formats (corners versus centre + width/height, pixels versus fractions).
- An NMS threshold that is too low (merges neighbouring objects) or too high (duplicates).
- Reporting accuracy instead of AP or IoU/Dice.
- Tiny objects that disappear in coarse feature maps — use higher resolution or multi-scale features.
- Annotation inconsistencies; a few hundred carefully labelled images beat thousands of sloppy ones.`,
    takeaways: [
      'IoU = overlap / union; a detection usually counts when IoU ≥ 0.5.',
      'One-stage detectors predict objectness, classes and box offsets on a grid in one pass.',
      'Non-maximum suppression removes duplicate boxes for the same object.',
      'AP summarizes the precision–recall curve; mAP averages over classes (and IoU thresholds).',
      'Segmentation labels every pixel; U-Net\'s skip connections keep fine detail; Dice and IoU measure overlap.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why is non-maximum suppression needed?', options: ['To resize images', 'Detectors output many overlapping boxes for the same object', 'To compute IoU', 'To train faster'], answer: 1, explain: 'NMS keeps one box per object.' },
      { type: 'mcq', q: 'What do U-Net\'s skip connections do?', options: ['Skip training steps', 'Copy high-resolution encoder features to the decoder so fine detail is preserved', 'Remove layers', 'Connect two different images'], answer: 1, explain: 'They restore detail lost by downsampling.' },
      { type: 'fill', q: 'A predicted mask has 30 pixels, the true mask 50, and they share 20. What is the Dice score?', answer: ['0.5', '0.50'], explain: '2 · 20 / (30 + 50) = 0.5.' },
      { type: 'tf', q: 'Semantic segmentation gives two overlapping cars different labels.', answer: false, explain: 'That is instance segmentation; semantic segmentation labels both pixels "car".' },
      { type: 'mcq', q: 'What does mAP stand for in detection benchmarks?', options: ['Mean absolute precision', 'Mean average precision over classes', 'Maximum area of prediction', 'Mapped accuracy percentage'], answer: 1, explain: 'AP per class, averaged.' }
    ],
    problems: [
      {
        id: 'iou-nms', title: 'IoU and Non-Maximum Suppression', difficulty: 'medium',
        prompt: `Boxes are tuples (x1, y1, x2, y2) with x1 < x2 and y1 < y2. Implement:

- \`iou(a, b)\` → intersection area / union area (0.0 when they don't overlap).
- \`nms(boxes, scores, threshold=0.5)\` → the indices of the kept boxes, in order of decreasing score (ties: lower index first, \`np.argsort(-scores, kind="stable")\`). Repeatedly keep the best remaining box and discard every remaining box whose IoU with it is **greater** than the threshold.

\`\`\`text
iou((0, 0, 2, 2), (1, 1, 3, 3))  ->  1 / 7
\`\`\``,
        starter: `
import numpy as np

def iou(a, b):
    # TODO
    pass

def nms(boxes, scores, threshold=0.5):
    # TODO
    pass
`,
        solution: `
import numpy as np

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

def nms(boxes, scores, threshold=0.5):
    order = [int(i) for i in np.argsort(-np.asarray(scores, dtype=float), kind="stable")]
    keep = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order if iou(boxes[best], boxes[i]) <= threshold]
    return keep
`,
        hints: ['The overlap rectangle runs from the larger of the left/top edges to the smaller of the right/bottom edges.', 'Clamp negative widths and heights to zero.', 'After keeping a box, filter the remaining list in one pass.'],
        explain: `NMS is greedy: the most confident box claims its object, and anything that overlaps it heavily is assumed to be a duplicate. Batched GPU versions do the same with vectorized IoU matrices.`,
        tests: [
          { call: 'iou((0, 0, 2, 2), (1, 1, 3, 3))', expect: '1 / 7' },
          { call: 'iou((0, 0, 1, 1), (2, 2, 3, 3))', expect: '0.0' },
          { name: 'two objects', call: 'nms([(10, 10, 60, 60), (12, 8, 62, 58), (15, 15, 65, 65), (100, 100, 150, 140), (102, 98, 148, 142)], [0.9, 0.85, 0.6, 0.8, 0.75])', expect: '[0, 3]' },
          { name: 'threshold matters', call: '[nms([(0, 0, 10, 10), (5, 0, 15, 10)], [0.9, 0.8], 0.5), nms([(0, 0, 10, 10), (5, 0, 15, 10)], [0.9, 0.8], 0.3)]', expect: '[[0, 1], [0]]', hidden: true },
          { name: 'random boxes vs reference', setup: 'rng = np.random.default_rng(0)\nxy = rng.uniform(0, 50, (12, 2)); wh = rng.uniform(5, 20, (12, 2))\nboxes = [tuple(map(float, (x, y, x + w, y + h))) for (x, y), (w, h) in zip(xy, wh)]\nscores = rng.random(12)', call: 'nms(boxes, scores, 0.3)', hidden: true }
        ]
      },
      {
        id: 'average-precision', title: 'Average Precision and Dice', difficulty: 'medium',
        prompt: `Implement:

- \`average_precision(scores, is_tp, n_gt)\` → AP = Σ_k (R_k − R_{k−1}) · P_k over the detections sorted by decreasing score (stable sort), where P_k = TP so far / k and R_k = TP so far / n_gt, with R_0 = 0. \`is_tp[i]\` says whether detection i was matched to a ground-truth box.
- \`dice(pred, gt)\` → 2·|P ∩ G| / (|P| + |G|) for two boolean masks; return 1.0 if both masks are empty.

\`\`\`text
average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 1], 4)  ->  0.25·1 + 0.25·(2/3) + 0.25·0.75 ≈ 0.604
\`\`\``,
        starter: `
import numpy as np

def average_precision(scores, is_tp, n_gt):
    # TODO
    pass

def dice(pred, gt):
    # TODO
    pass
`,
        solution: `
import numpy as np

def average_precision(scores, is_tp, n_gt):
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    tp = np.asarray(is_tp, dtype=float)[order]
    cum_tp = np.cumsum(tp)
    precision = cum_tp / np.arange(1, len(tp) + 1)
    recall = cum_tp / n_gt
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))

def dice(pred, gt):
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2 * np.sum(pred & gt) / total)
`,
        hints: ['`np.cumsum` gives true positives so far at every rank.', 'Recall only increases at true positives, so only those ranks add area.'],
        explain: `Only true positives move recall, so AP effectively averages the precision at each correctly found object — and divides by n_gt, so missed objects lower the score.`,
        tests: [
          { call: 'average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 1], 4)', expect: '0.25 * 1 + 0.25 * (2 / 3) + 0.25 * 0.75' },
          { name: 'perfect ranking', call: 'average_precision([0.3, 0.9, 0.5], [1, 1, 1], 3)', expect: '1.0' },
          { call: 'dice(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]))', expect: '0.5' },
          { name: 'empty masks', call: 'dice(np.zeros((3, 3)), np.zeros((3, 3)))', expect: '1.0', hidden: true },
          { name: 'random detections vs reference', setup: 'rng = np.random.default_rng(0)\nscores = rng.random(30); is_tp = (rng.random(30) < 0.6).astype(int)', call: 'average_precision(scores, is_tp, 25)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Redmon et al. — You Only Look Once: Unified, Real-Time Object Detection', url: 'https://arxiv.org/abs/1506.02640' },
      { title: 'Ronneberger et al. — U-Net: Convolutional Networks for Biomedical Image Segmentation', url: 'https://arxiv.org/abs/1505.04597' },
      { title: 'COCO — Detection evaluation', url: 'https://cocodataset.org/#detection-eval' },
      { title: 'Kirillov et al. — Segment Anything', url: 'https://arxiv.org/abs/2304.02643' }
    ]
  },

  // =====================================================================================
  {
    id: 'gnn', branch: 'spec', title: 'Graph Neural Networks', short: 'Graph Neural Networks', level: 3, minutes: 40, prereqs: ['mlp', 'embeddings'],
    tags: ['graph neural network', 'gnn', 'message passing', 'adjacency matrix', 'gcn', 'graph attention', 'node classification', 'link prediction', 'graph classification', 'over-smoothing'],
    summary: 'Learning on networks — molecules, social graphs, road maps, citation graphs: graphs as adjacency matrices, message passing where every node aggregates its neighbours, graph convolutional networks, node-, edge- and graph-level tasks, and pitfalls such as over-smoothing.',
    why: `Much data is relational: atoms bonded in a molecule, users following users, papers citing papers, roads connecting junctions. Flattening such data into a table throws the structure away. **Graph neural networks** learn directly on graphs and power drug and materials discovery, traffic and arrival-time prediction, fraud-ring detection and recommendations.`,
    objectives: [
      'Represent a graph with adjacency and feature matrices',
      'Explain message passing: aggregate the neighbours, update, repeat',
      'Implement a graph convolutional layer with symmetric normalization',
      'Choose node-, edge- or graph-level heads and recognize over-smoothing'
    ],
    sections: [
      { type: 'text', title: 'Graphs as matrices', md: `A graph has **nodes** (atoms, users, intersections) and **edges** (bonds, friendships, roads). In matrix form:

- the **feature matrix** X (n × d) — one row of features per node;
- the **adjacency matrix** A (n × n) — A[i, j] = 1 if nodes i and j are connected (symmetric for undirected graphs);
- the **degree** of a node — its number of neighbours (the row sums of A).

Real graphs are huge and sparse, so libraries store edge lists rather than full matrices. Unlike images (a regular grid) or text (a sequence), nodes have no natural order and a varying number of neighbours — any GNN operation must give the same result however the nodes are numbered.` },
      { type: 'text', title: 'Message passing', md: `Each GNN layer lets every node collect information from its neighbours and update its own vector:

$$ h_v^(l+1) = σ( W_self h_v^(l) + W_neigh · mean_{u ∈ N(v)} h_u^(l) ) $$
:::where
h_v^(l) — the vector of node v after l layers (h^(0) are the input features)
N(v) — the neighbours of node v
mean — an order-independent aggregation (sum or max also work)
W_self, W_neigh — learned weight matrices, shared by all nodes
σ — a nonlinearity such as ReLU
:::

After L layers, each node's vector summarizes its **L-hop neighbourhood**. Because the same weights are used everywhere and the aggregation ignores order, a GNN works on graphs of any size and shape.` },
      { type: 'text', title: 'Graph convolutional networks', md: `The **GCN** layer (Kipf & Welling, 2017) writes message passing compactly with a normalized adjacency matrix:

$$ H′ = σ( D̃^(−1/2) Ã D̃^(−1/2) H W ) $$
:::where
Ã = A + I — the adjacency matrix with self-loops, so each node keeps its own information
D̃ — the diagonal degree matrix of Ã
D̃^(−1/2) Ã D̃^(−1/2) — the symmetric normalization: high-degree nodes don't dominate and the scale stays stable
H — the node vectors (n × d)
W — the layer's learned weights (d × d′)
:::

**Graph attention networks (GAT)** replace the fixed weights with learned attention over neighbours — and a transformer is, in fact, a graph network on a fully connected graph. **GraphSAGE** samples a fixed number of neighbours, which makes training on graphs with billions of edges possible.` },
      { type: 'code', title: 'GCN layers and over-smoothing on a small graph', code: `
import numpy as np
rng = np.random.default_rng(0)
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)]     # two triangles joined by the edge 2-3
n = 6
A = np.zeros((n, n))
for i, j in edges:
    A[i, j] = A[j, i] = 1
A_tilde = A + np.eye(n)                                               # self-loops
deg = A_tilde.sum(axis=1)
A_hat = A_tilde / np.sqrt(np.outer(deg, deg))                         # D^-1/2 (A + I) D^-1/2

H = rng.normal(size=(n, 4))                                           # random node features
W1, W2 = rng.normal(0, 0.7, (4, 8)), rng.normal(0, 0.7, (8, 3))
H2 = A_hat @ np.maximum(0, A_hat @ H @ W1) @ W2                       # two GCN layers
print("node vectors after two GCN layers:")
print(H2.round(2))

def mean_cosine(Z):
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    S = Zn @ Zn.T
    return (S.sum() - n) / (n * (n - 1))

for layers in (0, 1, 2, 4, 8, 16):
    Z = np.linalg.matrix_power(A_hat, layers) @ H                     # repeated neighbour averaging
    print(f"{layers:2d} propagation steps: average cosine similarity between nodes {mean_cosine(Z):+.2f}")
`, after: `Nodes 0 and 1 come out identical: with self-loops, both see exactly the same nodes {0, 1, 2} with the same weights, so they average the same inputs (likewise nodes 4 and 5). More generally, each propagation step mixes every node with its neighbours. A few steps share useful information; many steps make all node vectors nearly identical — over-smoothing, the reason most GNNs use only 2–4 layers.` },
      { type: 'check', question: { type: 'mcq', q: 'After 3 message-passing layers, which nodes can influence a node\'s vector?', options: ['Only its direct neighbours', 'All nodes within 3 hops', 'Every node in the graph', 'Only itself'], answer: 1, explain: 'Each layer extends the reach by one hop.' } },
      { type: 'text', title: 'Tasks and readouts', md: `| Task | Example | Head |
| --- | --- | --- |
| **Node classification** | label users as bots, papers by topic | a classifier on each node's final vector |
| **Link prediction** | recommend friends, predict drug–target interactions | a score for a pair, e.g. σ(h_u · h_v) |
| **Graph classification / regression** | is this molecule toxic? its solubility? | pool all node vectors (sum, mean) into one graph vector, then an MLP |

Edge features (bond types, road lengths) can enter the messages, and many chemistry models add 3-D coordinates with symmetry-aware layers.` },
      { type: 'text', title: 'Pitfalls', md: `- **Over-smoothing**: too many layers make all node vectors alike (see the code above).
- **Over-squashing**: information from far away must squeeze through a few bottleneck nodes.
- **Scalability**: full-graph training does not fit in memory for huge graphs — sample neighbourhoods and train on mini-batches of subgraphs.
- **Leakage in link prediction**: the edges you test on must be removed from the graph the model sees during training.
- **Expressiveness**: standard message passing cannot distinguish some differently structured graphs (it is bounded by the Weisfeiler–Lehman test).` },
      { type: 'check', question: { type: 'tf', q: 'A GNN\'s output for a node changes if you renumber the nodes of the graph.', answer: false, explain: 'Shared weights and order-independent aggregation make it permutation-equivariant.' } }
    ],
    when: `- Molecules and materials (property prediction, generation), proteins.
- Social, citation and transaction networks: classification, fraud rings, recommendation.
- Traffic and logistics on road networks; physical simulation on meshes.
- If relations are weak or few, a tabular model with a couple of graph features (degree, neighbour averages) may be enough.`,
    pitfalls: `- Using many layers and getting over-smoothed, indistinguishable node vectors.
- Leaking test edges into training for link prediction.
- Ignoring node and edge features that carry most of the signal.
- Full-batch training on graphs that don't fit in memory.`,
    takeaways: [
      'Graphs = feature matrix X + adjacency matrix A; nodes have no natural order.',
      'Message passing: each layer aggregates neighbours and applies shared weights; L layers reach L hops.',
      'GCN: H′ = σ(D̃^(−1/2) Ã D̃^(−1/2) H W).',
      'Heads for node, edge (link) and graph tasks; pooling builds graph-level vectors.',
      'Beware over-smoothing, over-squashing, scalability limits and link-prediction leakage.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why must the aggregation in message passing be order-independent (sum, mean, max)?', options: ['For speed', 'Neighbours have no natural order, so the result must not depend on how they are listed', 'To reduce memory', 'It does not need to be'], answer: 1, explain: 'Renumbering the graph must not change the outcome.' },
      { type: 'fill', q: 'In a GCN, a node has 2 neighbours. Including its self-loop, what is its degree in D̃?', answer: ['3'], explain: 'Ã = A + I adds the node itself.' },
      { type: 'mcq', q: 'Which head would you use to predict whether a molecule is toxic?', options: ['Node classification', 'Graph-level pooling followed by a classifier', 'Link prediction', 'No head'], answer: 1, explain: 'One prediction per whole graph.' },
      { type: 'tf', q: 'Stacking many GNN layers can make all node vectors very similar (over-smoothing).', answer: true, explain: 'Repeated averaging washes out differences.' },
      { type: 'mcq', q: 'How is a transformer related to graph neural networks?', options: ['It is unrelated', 'Self-attention is message passing on a fully connected graph with learned weights', 'Transformers can only process trees', 'GNNs are a kind of tokenizer'], answer: 1, explain: 'Every token attends to every other token.' }
    ],
    problems: [
      {
        id: 'gcn-layer', title: 'A Graph Convolutional Layer', difficulty: 'medium',
        prompt: `Implement:

- \`normalized_adjacency(A)\` → D̃^(−1/2) (A + I) D̃^(−1/2), where D̃ holds the row sums of A + I. Element-wise this is (A + I)[i, j] / √(deg_i · deg_j).
- \`gcn_layer(A, H, W)\` → ReLU(Â H W) with Â the normalized adjacency.

\`\`\`text
normalized_adjacency([[0, 1], [1, 0]])  ->  [[0.5, 0.5], [0.5, 0.5]]
\`\`\``,
        starter: `
import numpy as np

def normalized_adjacency(A):
    # TODO
    pass

def gcn_layer(A, H, W):
    # TODO
    pass
`,
        solution: `
import numpy as np

def normalized_adjacency(A):
    A = np.asarray(A, dtype=float)
    A_tilde = A + np.eye(len(A))
    deg = A_tilde.sum(axis=1)
    return A_tilde / np.sqrt(np.outer(deg, deg))

def gcn_layer(A, H, W):
    return np.maximum(0, normalized_adjacency(A) @ np.asarray(H, dtype=float) @ np.asarray(W, dtype=float))
`,
        hints: ['Add the identity matrix for self-loops.', '`np.outer(deg, deg)` holds deg_i · deg_j for every pair.'],
        explain: `Dividing by √(deg_i·deg_j) averages neighbours while preventing high-degree hubs from blowing up the scale — the same reason we normalize activations elsewhere in deep learning.`,
        tests: [
          { call: 'normalized_adjacency([[0, 1], [1, 0]])', expect: 'np.array([[0.5, 0.5], [0.5, 0.5]])' },
          { name: 'a path of three nodes', call: 'normalized_adjacency([[0, 1, 0], [1, 0, 1], [0, 1, 0]])', expect: 'np.array([[1 / 2, 1 / np.sqrt(6), 0], [1 / np.sqrt(6), 1 / 3, 1 / np.sqrt(6)], [0, 1 / np.sqrt(6), 1 / 2]])' },
          { name: 'random layer vs reference', setup: 'rng = np.random.default_rng(0)\nA = (rng.random((6, 6)) < 0.4).astype(float); A = np.triu(A, 1); A = A + A.T\nH = rng.normal(size=(6, 3)); W = rng.normal(size=(3, 4))', call: 'gcn_layer(A, H, W)', hidden: true },
          { name: 'relabelling nodes permutes the output', setup: 'rng = np.random.default_rng(1)\nA = (rng.random((5, 5)) < 0.5).astype(float); A = np.triu(A, 1); A = A + A.T\nH = rng.normal(size=(5, 2)); W = rng.normal(size=(2, 2)); P = np.eye(5)[[3, 0, 4, 1, 2]]', code: 'out = gcn_layer(A, H, W)\nout_p = gcn_layer(P @ A @ P.T, P @ H, W)\nassert np.allclose(out_p, P @ out)', hidden: true }
        ]
      },
      {
        id: 'neighbor-aggregation', title: 'Neighbour Aggregation and Reach', difficulty: 'easy',
        prompt: `Implement:

- \`mean_neighbors(A, H)\` → for each node, the mean of its **neighbours'** feature vectors (not including itself); a node without neighbours gets a zero vector.
- \`k_hop_reachable(A, k)\` → a boolean (n × n) matrix whose entry [i, j] is True when j can be reached from i in at most k steps (every node reaches itself in 0 steps). Hint: the non-zero pattern of (I + A)^k.

\`\`\`text
A path 0 - 1 - 2: k_hop_reachable(A, 1)[0] -> [True, True, False]
\`\`\``,
        starter: `
import numpy as np

def mean_neighbors(A, H):
    # TODO
    pass

def k_hop_reachable(A, k):
    # TODO
    pass
`,
        solution: `
import numpy as np

def mean_neighbors(A, H):
    A = np.asarray(A, dtype=float)
    H = np.asarray(H, dtype=float)
    deg = A.sum(axis=1, keepdims=True)
    return (A @ H) / np.maximum(deg, 1)

def k_hop_reachable(A, k):
    A = np.asarray(A, dtype=float)
    M = np.linalg.matrix_power(np.eye(len(A)) + A, k)
    return M > 0
`,
        hints: ['`A @ H` sums each node\'s neighbours\' rows; divide by the degree.', 'Use `np.linalg.matrix_power` and test for positive entries.'],
        explain: `\`A @ H\` is message passing in one line: row i sums the vectors of i's neighbours. Powers of (I + A) show how far information can spread after k layers — the receptive field of a k-layer GNN.`,
        tests: [
          { call: 'mean_neighbors([[0, 1, 1], [1, 0, 0], [1, 0, 0]], [[1.0, 0.0], [2.0, 2.0], [4.0, 0.0]])', expect: 'np.array([[3.0, 1.0], [1.0, 0.0], [1.0, 0.0]])' },
          { call: 'k_hop_reachable([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 1)', expect: 'np.array([[True, True, False], [True, True, True], [False, True, True]])' },
          { name: 'isolated node', call: 'mean_neighbors([[0, 0], [0, 0]], [[5.0], [7.0]])', expect: 'np.array([[0.0], [0.0]])', hidden: true },
          { name: 'reach on a longer path', setup: 'A = np.diag(np.ones(5), 1) + np.diag(np.ones(5), -1)', call: '[k_hop_reachable(A, 2).sum(), k_hop_reachable(A, 5).all()]', expect: '[24, True]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Kipf & Welling — Semi-Supervised Classification with Graph Convolutional Networks', url: 'https://arxiv.org/abs/1609.02907' },
      { title: 'Sanchez-Lengeling et al. — A Gentle Introduction to Graph Neural Networks (Distill)', url: 'https://distill.pub/2021/gnn-intro/' },
      { title: 'PyTorch Geometric documentation', url: 'https://pytorch-geometric.readthedocs.io/' }
    ]
  },

  // =====================================================================================
  {
    id: 'speech-audio', branch: 'spec', title: 'Speech & Audio Models', short: 'Speech & Audio', level: 3, minutes: 40, prereqs: ['cnn', 'transformer-block'],
    tags: ['audio', 'speech recognition', 'asr', 'sampling rate', 'fourier transform', 'fft', 'spectrogram', 'mel scale', 'ctc', 'whisper', 'word error rate', 'text-to-speech', 'audio tokens'],
    summary: 'How machines hear and speak: sound as a sampled waveform, the Fourier transform and spectrograms, the mel scale, speech recognition with CTC and with encoder–decoder transformers such as Whisper, the word error rate, text-to-speech, and audio tokens for speech-native LLMs.',
    why: `Voice assistants, meeting transcription, subtitles, call-centre analytics and voice-to-voice assistants all start from raw audio — tens of thousands of numbers per second. Turning that into features a network can use, and turning predictions back into speech, follows a few key ideas: frequency analysis, spectrograms, and the same CNNs and transformers you already know.`,
    objectives: [
      'Describe sampling, waveforms and frequencies, and find the frequencies in a signal with the FFT',
      'Build a (mel) spectrogram with the short-time Fourier transform',
      'Explain speech recognition with CTC and with encoder–decoder transformers, and compute the word error rate',
      'Outline text-to-speech and audio tokenization for multimodal LLMs'
    ],
    sections: [
      { type: 'text', title: 'Sound as numbers', md: `A microphone measures air pressure many times per second. Speech is typically recorded at a **sample rate** of 16,000 samples per second (16 kHz); music at 44.1 or 48 kHz. Each sample is one number — the waveform.

- **Pitch** corresponds to frequency (Hz); **loudness** to amplitude.
- A sampled signal can only represent frequencies up to half the sample rate (the **Nyquist frequency**) — 8 kHz for 16 kHz audio, enough for intelligible speech.
- One second of speech is 16,000 numbers: far too long and too detailed to feed to a model directly, so audio models start by analysing frequencies.` },
      { type: 'text', title: 'From waveform to frequencies', md: `Any signal can be written as a sum of sine waves of different frequencies. The **discrete Fourier transform** measures how much of each frequency a window of N samples contains:

$$ X_k = Σ_{n=0}^{N−1} x_n · e^(−2πi·k·n / N) $$
:::where
x_n — the n-th sample of the window
N — the number of samples in the window
k — the frequency bin; it corresponds to the frequency k · sample_rate / N in Hz
X_k — a complex number whose magnitude |X_k| says how strongly that frequency is present
e^(−2πi·k·n / N) — a rotating complex sinusoid at frequency k that the signal is compared against
:::

The **fast Fourier transform (FFT)** computes all N bins in O(N log N) operations instead of O(N²).` },
      { type: 'code', title: 'Finding the notes in a signal with the FFT', code: `
import numpy as np
sr = 8000                                             # samples per second
t = np.arange(0, 1.0, 1 / sr)                         # one second of audio
rng = np.random.default_rng(0)
signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 660 * t) + 0.3 * rng.normal(size=t.size)

spectrum = np.abs(np.fft.rfft(signal))                # magnitudes for frequencies 0 .. sr/2
freqs = np.fft.rfftfreq(signal.size, 1 / sr)
top = np.argsort(-spectrum)[:2]
print("strongest frequencies (Hz):", sorted(freqs[top].round(1).tolist()))
print("frequency resolution:", freqs[1], "Hz per bin   highest representable frequency:", freqs[-1], "Hz")
`, after: `Hidden in the noisy waveform are two tones, 440 Hz (the note A) and 660 Hz, and the FFT finds them immediately.` },
      { type: 'text', title: 'Spectrograms and the mel scale', md: `Speech changes from moment to moment, so instead of one FFT over the whole recording we use the **short-time Fourier transform**: slide a short window (≈ 25 ms) along the signal in steps (≈ 10 ms) and take an FFT of each window. Stacking the results gives a **spectrogram** — an image with time on one axis and frequency on the other.

Human hearing distinguishes low frequencies much better than high ones. The **mel scale** spaces frequencies the way we perceive them:

$$ mel(f) = 2595 · log₁₀( 1 + f / 700 ) $$
:::where
f — a frequency in Hz
mel(f) — the perceived pitch on the mel scale: roughly linear below 1 kHz and logarithmic above
:::

Pooling the spectrogram into about 80 mel bands and taking the logarithm gives the **log-mel spectrogram**, the standard input to modern speech models. Because it is an image, CNNs and transformers can process it directly.` },
      { type: 'code', title: 'A spectrogram from scratch', code: `
import numpy as np
sr = 8000
t = np.arange(0, 2.0, 1 / sr)
chirp = np.sin(2 * np.pi * (200 * t + 400 * t ** 2))      # a tone sweeping from 200 Hz up to 1800 Hz

win, hop = 256, 128                                        # 32 ms windows every 16 ms
window = np.hanning(win)                                   # tapering reduces spectral leakage
frames = np.stack([chirp[i:i + win] * window for i in range(0, len(chirp) - win, hop)])
spec = np.abs(np.fft.rfft(frames, axis=1))                 # (time frames, frequency bins)
freqs = np.fft.rfftfreq(win, 1 / sr)
print("spectrogram shape (time frames, frequency bins):", spec.shape)
for f in range(0, len(frames), 20):
    print(f"t = {f * hop / sr:4.2f} s   loudest frequency {freqs[np.argmax(spec[f])]:6.0f} Hz")

def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)
print("mel values:", {hz: round(float(hz_to_mel(hz))) for hz in (100, 1000, 4000, 8000)})
`, after: `The loudest frequency climbs over time, exactly as the sweep was built. On the mel scale, 1,000 → 8,000 Hz spans less than twice the range of 0 → 1,000 Hz, mirroring human pitch perception.` },
      { type: 'check', question: { type: 'mcq', q: 'Audio is sampled at 16 kHz. What is the highest frequency it can represent?', options: ['16 kHz', '8 kHz', '32 kHz', '1.6 kHz'], answer: 1, explain: 'The Nyquist frequency is half the sample rate.' } },
      { type: 'text', title: 'Speech recognition', md: `**Acoustic encoders** (CNN front-ends followed by transformer or Conformer layers) turn the log-mel spectrogram into one vector per frame (about 50–100 per second). The difficulty: there are far more frames than characters, and nobody labels which frame belongs to which letter.

- **CTC** (connectionist temporal classification) lets the model output a character *or a blank* for every frame. Decoding collapses repeated characters and then removes blanks ("hh_e_ll_lo" → "hello"; the blank between the l's keeps the double letter). Training sums the probabilities of every frame alignment that collapses to the right transcript.
- **Encoder–decoder** models such as **Whisper** encode 30-second log-mel spectrograms and generate text tokens with a decoder that cross-attends to the encoder — exactly the sequence-to-sequence pattern. Trained on hundreds of thousands of hours of weakly labelled audio, Whisper handles many languages, accents and noise, and can also translate.

Accuracy is measured with the **word error rate**:

$$ WER = ( S + D + I ) / N $$
:::where
S — substituted words (wrong word)
D — deleted words (missing)
I — inserted words (extra)
N — the number of words in the reference transcript
These are counted from the minimum edit (Levenshtein) alignment between the hypothesis and the reference; WER can exceed 1.
:::` },
      { type: 'text', title: 'Speaking and listening LLMs', md: `- **Text-to-speech (TTS)**: a model predicts acoustic features (often a mel spectrogram) from text, and a **vocoder** network turns them into a waveform. Modern systems sound natural and can clone a voice from seconds of audio.
- **Neural audio codecs** compress audio into sequences of discrete **tokens** (using residual vector quantization), so a language model can read and generate audio just like text.
- **Speech-native LLMs** take audio tokens or features in and produce audio out directly — lower latency than chaining speech-to-text, an LLM and text-to-speech, and they keep tone and emotion.
- **Responsibility**: voice cloning enables impersonation and fraud; use consent, disclosure and audio watermarking.` },
      { type: 'check', question: { type: 'tf', q: 'In CTC decoding, repeated characters are collapsed first and blanks are removed afterwards.', answer: true, explain: 'That order lets a blank separate genuine double letters.' } }
    ],
    when: `- Transcription, subtitles, voice commands and call analytics (speech recognition).
- Voice interfaces and accessibility (text-to-speech), speaker identification, sound-event detection (alarms, machinery faults).
- Start from pretrained models (Whisper and similar) and fine-tune on in-domain audio for accents, jargon or noise.`,
    pitfalls: `- Mismatched sample rates between training and inference audio — resample first.
- Evaluating WER without normalizing text (casing, punctuation, numbers written as words versus digits).
- Long recordings: chunking without overlap cuts words in half.
- Background noise, overlapping speakers and far-field microphones that differ from the training data.
- Ignoring consent and privacy for voice data.`,
    takeaways: [
      'Audio is a sampled waveform; frequencies above half the sample rate are lost.',
      'The FFT reveals frequencies; the short-time FFT over sliding windows gives a spectrogram.',
      'Log-mel spectrograms, spaced like human hearing, are the standard input to speech models.',
      'Speech recognition uses CTC (collapse repeats, drop blanks) or encoder–decoders like Whisper; WER = (S + D + I) / N.',
      'Audio tokens let language models listen and speak directly.'
    ],
    quiz: [
      { type: 'fill', q: 'An FFT of 512 samples at a 16,000 Hz sample rate: what frequency (in Hz) does each bin step represent?', answer: ['31.25'], explain: 'sample_rate / N = 16000 / 512 = 31.25 Hz.' },
      { type: 'mcq', q: 'Why use the mel scale for speech features?', options: ['It is faster to compute', 'It spaces frequencies the way humans perceive pitch, with more detail at low frequencies', 'It removes noise', 'It doubles the resolution'], answer: 1, explain: 'Low frequencies carry much of speech information.' },
      { type: 'mcq', q: 'CTC frame outputs "h h _ e e _ l l _ l o o" (_ = blank). What is decoded?', options: ['hello', 'helo', 'hhello', 'hel lo'], answer: 0, explain: 'Collapse repeats: h _ e _ l _ l o; remove blanks: hello.' },
      { type: 'fill', q: 'A 10-word reference is transcribed with 1 substitution, 1 deletion and 1 insertion. What is the WER?', answer: ['0.3', '30%', '0.30'], explain: '(1 + 1 + 1) / 10 = 0.3.' },
      { type: 'tf', q: 'Whisper is an encoder–decoder transformer that generates text tokens while attending to an encoded spectrogram.', answer: true, explain: 'A sequence-to-sequence model with cross-attention.' }
    ],
    problems: [
      {
        id: 'ctc-decode', title: 'CTC Greedy Decoding', difficulty: 'easy',
        prompt: `A CTC model outputs one id per audio frame. Implement:

- \`ctc_collapse(frame_ids, blank=0)\` → merge consecutive repeated ids into one, then remove every \`blank\`.
- \`ctc_to_text(frame_ids, alphabet, blank=0)\` → the collapsed ids mapped to characters with \`alphabet[id]\` and joined into a string.

\`\`\`text
ctc_collapse([1, 1, 0, 2, 2, 0, 3, 3, 0, 3, 4])  ->  [1, 2, 3, 3, 4]
alphabet = "_helo";  ctc_to_text([1, 1, 0, 2, 0, 3, 3, 0, 3, 4, 4], alphabet)  ->  "hello"
\`\`\``,
        starter: `
def ctc_collapse(frame_ids, blank=0):
    # TODO
    pass

def ctc_to_text(frame_ids, alphabet, blank=0):
    # TODO
    pass
`,
        solution: `
def ctc_collapse(frame_ids, blank=0):
    out, prev = [], None
    for i in frame_ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out

def ctc_to_text(frame_ids, alphabet, blank=0):
    return "".join(alphabet[i] for i in ctc_collapse(frame_ids, blank))
`,
        hints: ['Keep track of the previous id while scanning.', 'An id is emitted only if it differs from the previous frame and is not the blank.'],
        explain: `Doing the comparison with the previous *frame* (including blanks) implements "collapse first, then remove blanks" in one pass: a blank between two identical letters makes the second one count again.`,
        tests: [
          { call: 'ctc_collapse([1, 1, 0, 2, 2, 0, 3, 3, 0, 3, 4])', expect: '[1, 2, 3, 3, 4]' },
          { call: 'ctc_to_text([1, 1, 0, 2, 0, 3, 3, 0, 3, 4, 4], "_helo")', expect: '"hello"' },
          { name: 'all blanks', call: 'ctc_collapse([0, 0, 0])', expect: '[]', hidden: true },
          { name: 'another blank id', call: 'ctc_to_text([5, 5, 1, 1, 5, 1, 2], "_ab???", blank=5)', expect: '"aab"', hidden: true }
        ]
      },
      {
        id: 'word-error-rate', title: 'Word Error Rate', difficulty: 'medium',
        prompt: `Implement \`wer(reference, hypothesis)\` → (substitutions + deletions + insertions) / (number of reference words), using the **minimum** edit distance between the two word sequences (split both strings on whitespace). The reference has at least one word.

Use dynamic programming: D[i][j] = the edit distance between the first i reference words and the first j hypothesis words, with D[i][0] = i, D[0][j] = j and

D[i][j] = min(D[i−1][j] + 1, D[i][j−1] + 1, D[i−1][j−1] + (0 if the words match else 1)).

\`\`\`text
wer("the cat sat on the mat", "the cat sat on mat")    ->  1/6   (one deletion)
wer("hello world", "hello there world")                ->  0.5   (one insertion)
\`\`\``,
        starter: `
def wer(reference, hypothesis):
    # TODO
    pass
`,
        solution: `
def wer(reference, hypothesis):
    r, h = reference.split(), hypothesis.split()
    D = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        D[i][0] = i
    for j in range(len(h) + 1):
        D[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost)
    return D[len(r)][len(h)] / len(r)
`,
        hints: ['The first row and column count pure insertions and deletions.', 'Each cell looks at its left, upper and upper-left neighbours.'],
        explain: `This is the classic Levenshtein algorithm applied to words instead of characters. Real WER tools first normalize the text (lowercase, strip punctuation, standardize numbers) so formatting differences don't count as errors.`,
        tests: [
          { call: 'wer("the cat sat on the mat", "the cat sat on mat")', expect: '1 / 6' },
          { call: 'wer("hello world", "hello there world")', expect: '0.5' },
          { name: 'substitution', call: 'wer("a b c", "a x c")', expect: '1 / 3', hidden: true },
          { name: 'WER can exceed one', call: 'wer("yes", "no no no")', expect: '3.0', hidden: true },
          { name: 'perfect transcript', call: 'wer("speech is hard", "speech is hard")', expect: '0.0', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Radford et al. — Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)', url: 'https://arxiv.org/abs/2212.04356' },
      { title: 'Graves et al. — Connectionist Temporal Classification', url: 'https://www.cs.toronto.edu/~graves/icml_2006.pdf' },
      { title: 'Awni Hannun — Sequence Modeling with CTC (Distill)', url: 'https://distill.pub/2017/ctc/' }
    ]
  },

  // =====================================================================================
  {
    id: 'causal-inference', branch: 'spec', title: 'Causal Inference', level: 2, minutes: 45, prereqs: ['stat-inference', 'linear-regression'],
    tags: ['causal inference', 'causation', 'confounding', 'potential outcomes', 'average treatment effect', 'randomized experiment', 'simpsons paradox', 'causal graph', 'propensity score', 'ipw', 'uplift'],
    summary: 'From "what predicts" to "what causes": correlation versus causation, potential outcomes and the average treatment effect, why randomized experiments work, confounders and Simpson\'s paradox, causal graphs and what to adjust for, and estimating effects from observational data with regression adjustment and inverse propensity weighting.',
    why: `Machine-learning models are superb at prediction, but many important questions are **causal**: "Will this discount increase sales?", "Does this drug work?", "Did the new feature cause the drop in sign-ups?". Correlations can point the wrong way — ice-cream sales predict drownings, and hospital patients die more often than people at home. Causal inference provides the tools and the discipline to answer "what happens if we act?".`,
    objectives: [
      'Distinguish prediction from causal questions and define the average treatment effect with potential outcomes',
      'Explain why randomization removes confounding, and recognize Simpson\'s paradox',
      'Use a causal graph to decide what to adjust for: confounders, but not mediators or colliders',
      'Estimate effects from observational data with regression adjustment and inverse propensity weighting'
    ],
    sections: [
      { type: 'text', title: 'Correlation is not causation', md: `- **Confounding**: ice-cream sales and drownings rise together because hot weather drives both.
- **Selection**: people in hospitals die more often than people at home — because sick people go to hospitals.
- **Reverse causation**: more firefighters at a fire are associated with more damage; big fires cause both.

A predictive model happily uses all of these associations, which is fine for *prediction*. But if you *act* — send fewer firefighters — the association breaks. Causal questions are about **interventions**, and they need different methods.` },
      { type: 'text', title: 'Potential outcomes', md: `For each unit (a person, a store), imagine two **potential outcomes**: Y(1) if treated and Y(0) if not. The individual effect is Y(1) − Y(0) — but we only ever observe one of the two. This is the **fundamental problem of causal inference**. What we can estimate is an average:

$$ ATE = E[ Y(1) − Y(0) ] $$
:::where
Y(1) — the outcome a unit would have with the treatment
Y(0) — the outcome the same unit would have without it
E[ … ] — the average over the population
ATE — the average treatment effect
:::

The naive comparison of treated and untreated groups equals the ATE **plus selection bias** whenever the groups differed to begin with. **Randomization** fixes this: when a coin decides who is treated, treatment is independent of everything else, so the two groups are comparable and the simple difference in means is an unbiased estimate of the ATE. That is why randomized experiments (A/B tests, clinical trials) are the gold standard.` },
      { type: 'code', title: 'Confounding in a simulation', code: `
import numpy as np
rng = np.random.default_rng(0)
n = 20000
age = rng.uniform(20, 80, n)
p_treat = 1 / (1 + np.exp(-(age - 50) / 8))                   # older people take the treatment more often...
t_obs = (rng.random(n) < p_treat).astype(float)

def outcome(t, age):
    return 2.0 * t - 0.1 * age + rng.normal(0, 1, len(age))   # ...and age lowers the outcome; true effect +2

y_obs = outcome(t_obs, age)
naive = y_obs[t_obs == 1].mean() - y_obs[t_obs == 0].mean()
print(f"observational, naive difference:   {naive:+.2f}   (true effect +2.00)")

t_rct = (rng.random(n) < 0.5).astype(float)                   # a randomized experiment: a coin decides
y_rct = outcome(t_rct, age)
print(f"randomized experiment difference:  {y_rct[t_rct == 1].mean() - y_rct[t_rct == 0].mean():+.2f}")

coef = np.linalg.lstsq(np.c_[np.ones(n), t_obs, age], y_obs, rcond=None)[0]
print(f"observational, adjusted for age:   {coef[1]:+.2f}")
`, after: `The treated group is older, and age lowers the outcome, so the naive comparison makes a helpful treatment look slightly harmful. Randomization, or adjusting for the confounder, recovers the true effect.` },
      { type: 'check', question: { type: 'mcq', q: 'Why does randomly assigning treatment remove confounding?', options: ['It increases the sample size', 'Treatment becomes independent of every other characteristic, so the groups are comparable on average', 'It removes noise from the outcome', 'It makes the effect larger'], answer: 1, explain: 'Nothing about a unit can influence the coin flip.' } },
      { type: 'text', title: 'Simpson\'s paradox', md: `A famous study of kidney-stone treatments:

| | Treatment A | Treatment B |
| --- | --- | --- |
| Small stones | 81 / 87 = **93%** | 234 / 270 = 87% |
| Large stones | 192 / 263 = **73%** | 55 / 80 = 69% |
| All patients | 273 / 350 = 78% | 289 / 350 = **83%** |

A is better for small stones *and* for large stones, yet B looks better overall. The reason: doctors gave A mostly for the harder, large stones. Stone size is a **confounder** — the comparison within each size group is the fair one here. Which comparison is right depends on the causal structure, not on the numbers alone.` },
      { type: 'text', title: 'Causal graphs: what to adjust for', md: `Draw the causal story as a **directed acyclic graph** (DAG), with arrows from causes to effects. Then:

- **Confounder** (a common cause, T ← C → Y): adjust for it, or it biases the estimate (age in the simulation, stone size in the table).
- **Mediator** (on the causal path, T → M → Y): don't adjust if you want the *total* effect — you would block part of it.
- **Collider** (a common effect, T → C ← Y): don't adjust — conditioning on it *creates* a spurious association. Example: among celebrities (selected for talent *or* looks), talent and looks appear negatively related.

The **backdoor criterion** formalizes this: adjust for a set of variables that blocks every non-causal path from treatment to outcome without opening new ones through colliders.` },
      { type: 'text', title: 'Estimating effects from observational data', md: `When you can't experiment, you need assumptions: all confounders are measured (**no unmeasured confounding**), and every kind of unit had some chance of each treatment (**overlap**).

- **Regression adjustment**: include the confounders as features; the coefficient on the treatment estimates its effect (if the model is right).
- **Propensity score** e(x) = P(T = 1 | x), the probability of treatment given the confounders, usually fitted with logistic regression. **Inverse propensity weighting** reweights the data so the groups look as if treatment were random:

$$ ATE_IPW = (1/n) Σ_i [ T_i Y_i / e(x_i) − (1 − T_i) Y_i / (1 − e(x_i)) ] $$
:::where
T_i — 1 if unit i was treated, 0 otherwise
Y_i — the observed outcome of unit i
e(x_i) — the propensity score: the estimated probability that unit i, with confounders x_i, is treated
1 / e(x_i) — the weight of a treated unit: rarely treated kinds of units count more
:::

Other designs: **matching** (compare each treated unit with a similar untreated one), **difference-in-differences** (compare changes over time against a control group), **instrumental variables** and **regression discontinuity**. **Uplift modelling** goes further and predicts the effect for each individual — who will respond to a coupon — so you can target treatment.` },
      { type: 'code', title: 'Inverse propensity weighting', code: `
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
n = 20000
age = rng.uniform(20, 80, n)
t = (rng.random(n) < 1 / (1 + np.exp(-(age - 50) / 8))).astype(float)
y = 2.0 * t - 0.1 * age + rng.normal(0, 1, n)                      # true effect +2

e = LogisticRegression().fit(age.reshape(-1, 1), t).predict_proba(age.reshape(-1, 1))[:, 1]
e = np.clip(e, 0.01, 0.99)                                         # avoid extreme weights
ipw = np.mean(t * y / e - (1 - t) * y / (1 - e))
w1, w0 = t / e, (1 - t) / (1 - e)
hajek = np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0)   # normalized weights: more stable
print(f"naive difference {y[t == 1].mean() - y[t == 0].mean():+.2f}")
print(f"IPW estimate     {ipw:+.2f}")
print(f"normalized IPW   {hajek:+.2f}   (true +2.00)")
`, after: `Weighting each person by the inverse of their probability of the treatment they actually got rebalances the groups by age. The normalized version divides by the total weight instead of n, which usually makes it more stable when some weights are large.` },
      { type: 'check', question: { type: 'tf', q: 'Adjusting for a collider (a common effect of treatment and outcome) can create a spurious association.', answer: true, explain: 'Conditioning on a common effect links its causes.' } }
    ],
    when: `- Decisions about actions: pricing, marketing, product changes, medical treatments, policies.
- Run a randomized experiment whenever you can; use observational methods only when you can't, and state the assumptions.
- Uplift models to target interventions at the people they help most.
- Use plain predictive models when you only need forecasts, not decisions about interventions.`,
    pitfalls: `- Reading predictive feature importance as causal effect.
- Adjusting for everything available, including mediators and colliders.
- Unmeasured confounding — no algorithm can fix variables you didn't record.
- Extreme propensity weights from poor overlap; check the distribution of e(x).
- Ignoring Simpson's paradox when aggregating across groups.`,
    takeaways: [
      'Causal questions ask what happens under an intervention; prediction alone can mislead.',
      'ATE = E[Y(1) − Y(0)]; we never observe both outcomes for the same unit.',
      'Randomization makes groups comparable, so the difference in means is unbiased.',
      'Adjust for confounders, not mediators or colliders — a causal graph tells you which is which.',
      'Without experiments: regression adjustment, propensity weighting, matching, difference-in-differences — under explicit assumptions.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the fundamental problem of causal inference?', options: ['Data is too large', 'For each unit we observe only one of its potential outcomes', 'Models cannot handle categories', 'Causes are always hidden'], answer: 1, explain: 'We never see the same person both treated and untreated at the same time.' },
      { type: 'mcq', q: 'Treatment A is better than B for small stones and for large stones, but worse overall. What is this called?', options: ['Overfitting', 'Simpson\'s paradox', 'Regression to the mean', 'Data leakage'], answer: 1, explain: 'A confounder (stone size) reverses the aggregate comparison.' },
      { type: 'mcq', q: 'Exercise → lower blood pressure → fewer heart attacks. To estimate exercise\'s total effect on heart attacks, should you adjust for blood pressure?', options: ['Yes, always', 'No — it is a mediator on the causal path', 'Only if it is measured', 'Only in randomized trials'], answer: 1, explain: 'Adjusting would remove the part of the effect that flows through it.' },
      { type: 'fill', q: 'A treated unit has propensity score 0.2. What is its inverse propensity weight?', answer: ['5', '5.0'], explain: '1 / 0.2 = 5: only about one in five similar units gets treated, so this one stands in for all five.' },
      { type: 'tf', q: 'A randomized A/B test is not affected by confounding.', answer: true, explain: 'Randomization makes treatment independent of other characteristics (up to chance imbalance).' }
    ],
    problems: [
      {
        id: 'ate-estimators', title: 'Treatment-Effect Estimators', difficulty: 'medium',
        prompt: `Implement three estimators of the average treatment effect (\`y\` outcomes, \`t\` 0/1 treatment indicators):

- \`naive_difference(y, t)\` → mean of y for t = 1 minus mean of y for t = 0.
- \`regression_ate(y, t, X)\` → the coefficient on t in an ordinary least-squares fit of y on [1, t, X] (\`np.linalg.lstsq\`); X is an (n, k) array of confounders.
- \`ipw_ate(y, t, e)\` → mean of t·y/e − (1 − t)·y/(1 − e), after clipping the propensity scores e to [0.01, 0.99].

\`\`\`text
naive_difference([5, 7, 1, 3], [1, 1, 0, 0])  ->  4.0
\`\`\``,
        starter: `
import numpy as np

def naive_difference(y, t):
    # TODO
    pass

def regression_ate(y, t, X):
    # TODO
    pass

def ipw_ate(y, t, e):
    # TODO
    pass
`,
        solution: `
import numpy as np

def naive_difference(y, t):
    y, t = np.asarray(y, dtype=float), np.asarray(t)
    return float(y[t == 1].mean() - y[t == 0].mean())

def regression_ate(y, t, X):
    y, t, X = np.asarray(y, dtype=float), np.asarray(t, dtype=float), np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    design = np.c_[np.ones(len(y)), t, X]
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coef[1])

def ipw_ate(y, t, e):
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    e = np.clip(np.asarray(e, dtype=float), 0.01, 0.99)
    return float(np.mean(t * y / e - (1 - t) * y / (1 - e)))
`,
        hints: ['Boolean masks select the treated and control outcomes.', '`np.c_[np.ones(n), t, X]` builds the design matrix; the coefficient on t is index 1.', 'Clip e with `np.clip(e, 0.01, 0.99)`.'],
        explain: `All three answer the same question under different assumptions: the naive difference assumes no confounding; regression assumes the outcome model is right; IPW assumes the propensity model is right. When they disagree, the causal assumptions deserve a closer look.`,
        tests: [
          { call: 'naive_difference([5, 7, 1, 3], [1, 1, 0, 0])', expect: '4.0' },
          { name: 'regression recovers the effect when the model is exact', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(50, 2)); t = (rng.random(50) < 0.5).astype(float)\ny = 1.0 + 3.0 * t + X @ np.array([2.0, -1.0])', call: 'regression_ate(y, t, X)', expect: '3.0' },
          { name: 'IPW with known propensities', call: 'ipw_ate([4.0, 2.0], [1, 0], [0.5, 0.5])', expect: '(4.0 / 0.5 - 2.0 / 0.5) / 2' },
          { name: 'confounded simulation', setup: 'rng = np.random.default_rng(1)\nn = 20000\nx = rng.normal(size=n)\ne = 1 / (1 + np.exp(-1.5 * x))\nt = (rng.random(n) < e).astype(float)\ny = 2.0 * t + 3.0 * x + rng.normal(0, 1, n)', code: 'assert naive_difference(y, t) > 3.5, "confounding should inflate the naive estimate"\nassert abs(regression_ate(y, t, x) - 2.0) < 0.1\nassert abs(ipw_ate(y, t, e) - 2.0) < 0.4', hidden: true },
          { name: 'clipping extreme propensities', call: 'ipw_ate([1.0, 3.0], [1, 0], [0.0, 1.0])', expect: '(1.0 / 0.01 - 3.0 / 0.01) / 2', hidden: true }
        ]
      },
      {
        id: 'simpsons-paradox', title: 'Detecting Simpson\'s Paradox', difficulty: 'easy',
        prompt: `\`table\` maps a group to a dict of treatments, each a pair \`(successes, total)\`, e.g.
\`{"small": {"A": (81, 87), "B": (234, 270)}, "large": {"A": (192, 263), "B": (55, 80)}}\`. Implement:

- \`group_rates(table)\` → \`{group: {treatment: successes / total}}\`
- \`overall_rates(table)\` → \`{treatment: (sum of successes) / (sum of totals)}\` across all groups
- \`is_simpsons_paradox(table, a, b)\` → True when treatment a has a higher rate than b in **every** group but a lower overall rate — or the reverse (b better in every group, a better overall).`,
        starter: `
def group_rates(table):
    # TODO
    pass

def overall_rates(table):
    # TODO
    pass

def is_simpsons_paradox(table, a, b):
    # TODO
    pass
`,
        solution: `
def group_rates(table):
    return {g: {tr: s / n for tr, (s, n) in arms.items()} for g, arms in table.items()}

def overall_rates(table):
    totals = {}
    for arms in table.values():
        for tr, (s, n) in arms.items():
            succ, cnt = totals.get(tr, (0, 0))
            totals[tr] = (succ + s, cnt + n)
    return {tr: s / n for tr, (s, n) in totals.items()}

def is_simpsons_paradox(table, a, b):
    rates = group_rates(table)
    overall = overall_rates(table)
    a_wins_everywhere = all(r[a] > r[b] for r in rates.values())
    b_wins_everywhere = all(r[b] > r[a] for r in rates.values())
    return (a_wins_everywhere and overall[a] < overall[b]) or (b_wins_everywhere and overall[b] < overall[a])
`,
        hints: ['Sum successes and totals per treatment before dividing — averaging the group rates is not the same.', '`all(...)` checks a condition across every group.'],
        explain: `The overall rate weights each group by how many patients received each treatment there. When treatment choice depends on the group (doctors used A for the hard cases), those weights differ between treatments and the aggregate comparison can flip.`,
        tests: [
          { setup: 'kidney = {"small": {"A": (81, 87), "B": (234, 270)}, "large": {"A": (192, 263), "B": (55, 80)}}', call: '[overall_rates(kidney), is_simpsons_paradox(kidney, "A", "B")]', expect: '[{"A": 273 / 350, "B": 289 / 350}, True]' },
          { setup: 'kidney = {"small": {"A": (81, 87), "B": (234, 270)}, "large": {"A": (192, 263), "B": (55, 80)}}', call: 'group_rates(kidney)["small"]', expect: '{"A": 81 / 87, "B": 234 / 270}' },
          { name: 'no paradox', call: 'is_simpsons_paradox({"x": {"A": (9, 10), "B": (5, 10)}, "y": {"A": (8, 10), "B": (6, 10)}}, "A", "B")', expect: 'False', hidden: true },
          { name: 'reverse direction', call: 'is_simpsons_paradox({"small": {"B": (81, 87), "A": (234, 270)}, "large": {"B": (192, 263), "A": (55, 80)}}, "A", "B")', expect: 'True', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Hernán & Robins — Causal Inference: What If (free book)', url: 'https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/' },
      { title: 'Pearl & Mackenzie — The Book of Why', url: 'http://bayes.cs.ucla.edu/WHY/' },
      { title: 'Brady Neal — Introduction to Causal Inference (free course)', url: 'https://www.bradyneal.com/causal-inference-course' },
      { title: 'Charig et al. — Comparison of treatment of renal calculi (the kidney-stone data)', url: 'https://doi.org/10.1136/bmj.292.6524.879' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
