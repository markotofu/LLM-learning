"""Chart data for batch N1 (perceptron, activations, backprop, autograd, training loop). Apply AFTER patches/n1-text.txt."""
import json, sys, math
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
sig = lambda v, floor=1e-30: [None if x is None else float(f"{max(float(x), floor):.4g}") for x in v]
out, diag = [], []
def sci(v):
    m, e = f"{v:.0e}".split("e")
    return f"{m} × 10" + str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- perceptron: mistakes per epoch (the lesson's code), and three different separators
def train_perceptron(X, y, lr=1.0, epochs=25, order=None):
    w, b, hist = np.zeros(X.shape[1]), 0.0, []
    for epoch in range(epochs):
        mistakes = 0
        idx = range(len(X)) if order is None else order(epoch)
        for i in idx:
            xi, yi = X[i], y[i]
            pred = 1 if xi @ w + b >= 0 else 0
            if pred != yi:
                w += lr * (yi - pred) * xi; b += lr * (yi - pred); mistakes += 1
        hist.append(mistakes)
    return w, b, hist
X4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
hists = {name: train_perceptron(X4, np.array(y), epochs=12)[2] for name, y in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]}
diag.append(("perceptron mistakes", hists))
chart("perceptron", {"after": "Training a perceptron on AND, OR and XOR"}, "Mistakes per epoch",
      {"kind": "line", "x": list(range(1, 13)), "series": [{"name": k, "y": v} for k, v in hists.items()], "xLabel": "epoch (one pass over the 4 examples)", "yLabel": "mistakes in the epoch", "yMin": 0, "yMax": 4},
      after="AND and OR are linearly separable, so the mistakes drop to zero within a few epochs and stay there: the perceptron has converged. XOR never gets there: every epoch still contains mistakes, and the weights cycle forever, exactly as the convergence theorem's condition predicts.")
rng = np.random.default_rng(0)
A = rng.normal([-1.2, -1.0], 0.55, (25, 2)); B = rng.normal([1.2, 1.0], 0.55, (25, 2))
Xs = np.vstack([A, B]); ys = np.r_[np.zeros(25, int), np.ones(25, int)]
series = [{"name": "class 0", "x": r(A[:, 0], 3), "y": r(A[:, 1], 3), "scatter": True}, {"name": "class 1", "x": r(B[:, 0], 3), "y": r(B[:, 1], 3), "scatter": True}]
grid = np.array([-3.0, 3.0]); lines = []
for seed in (1, 2, 3):
    rr = np.random.default_rng(seed)
    w, b, hist = train_perceptron(Xs, ys, epochs=100, order=lambda e: rr.permutation(len(Xs)))
    errors = int(np.sum(((Xs @ w + b) >= 0).astype(int) != ys))
    lines.append((seed, w.copy(), b, errors, next((i + 1 for i, m in enumerate(hist) if m == 0), None)))
    series.append({"name": f"boundary, data order {seed}", "x": [-3.0, 3.0], "y": r([(-b - w[0] * gx) / w[1] for gx in grid], 3)})
diag.append(("perceptron runs", [(s, w.round(2).tolist(), round(b, 2), e, conv) for s, w, b, e, conv in lines]))
chart("perceptron", {"after": "The perceptron finds a separator, not the best one"}, "Three runs, three different boundaries",
      {"kind": "line", "points": False, "series": series, "xLabel": "x₁", "yLabel": "x₂", "xMin": -3, "xMax": 3, "yMin": -3, "yMax": 3},
      md="The same separable data, presented in three different random orders:",
      after="Every run stops with zero training mistakes, yet each stops at a different line: the perceptron halts at the first separator it reaches. Some lines pass uncomfortably close to points of one class. Choosing the line with the widest gap to both classes is exactly what a [support vector machine](#/concept/svm) does.")

# ---------------- activation functions: gradient flow through depth
def grad_profile(act, init, depth=20, width=100, n=200, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, width)); Ws, zs, hs = [], [], [h]
    for l in range(depth):
        W = rng.normal(0, init(width), (width, width)); z = hs[-1] @ W
        Ws.append(W); zs.append(z); hs.append(act[0](z))
    g = np.ones_like(hs[-1]) / n; norms = []
    for l in reversed(range(depth)):
        gz = g * act[1](zs[l], hs[l + 1]); norms.append(float(np.sqrt(np.mean((gz @ Ws[l].T) ** 2)) * n)); g = gz @ Ws[l].T
    return norms[::-1]
sigm = (lambda z: 1 / (1 + np.exp(-z)), lambda z, h: h * (1 - h))
tanh = (np.tanh, lambda z, h: 1 - h ** 2)
relu = (lambda z: np.maximum(0, z), lambda z, h: (z > 0).astype(float))
xav = lambda n: math.sqrt(1 / n); he = lambda n: math.sqrt(2 / n)
prof = {"sigmoid (Xavier init)": grad_profile(sigm, xav), "tanh (Xavier init)": grad_profile(tanh, xav), "ReLU (He init)": grad_profile(relu, he)}
diag.append(("grad profile first/last", {k: (round(v[0], 6), round(v[-1], 4)) for k, v in prof.items()}))
chart("activation-functions", {"after": "Vanishing gradients"}, "How the gradient's size changes on its way back through 20 layers",
      {"kind": "line", "x": list(range(1, 21)), "yLog": True, "points": False, "series": [{"name": k, "y": sig(v)} for k, v in prof.items()],
       "xLabel": "layer (1 = closest to the input; the gradient arrives from layer 20)", "yLabel": "typical gradient size (log scale)"},
      md="A 20-layer network, 100 units per layer, random weights; the same gradient is sent back from the output and measured at every layer:",
      after=f"With sigmoid activations the gradient shrinks by a factor of about {(prof['sigmoid (Xavier init)'][-1] / prof['sigmoid (Xavier init)'][0]) ** (1 / 19):.0f} at every layer (the sigmoid's slope is at most 0.25), so at layer 1 it is about {sci(prof['sigmoid (Xavier init)'][0] / prof['sigmoid (Xavier init)'][-1])} of its size at layer 20: the early layers get almost no learning signal. tanh with Xavier initialisation loses only a factor of about {prof['tanh (Xavier init)'][-1] / prof['tanh (Xavier init)'][0]:.0f} over all 20 layers, and ReLU with He initialisation keeps the gradient at roughly the same size throughout. Better activations and initialisation are what made deep networks trainable.")

# ---------------- backprop: gradient flow with and without residual connections
def resid_profile(residual, depth=50, width=64, n=128, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, width)); Ws, zs, hs = [], [], [h]
    for l in range(depth):
        W = rng.normal(0, 0.25 * math.sqrt(1 / width), (width, width)); z = hs[-1] @ W          # the same weights in both networks
        Ws.append(W); zs.append(z); hs.append((hs[-1] if residual else 0) + np.tanh(z))
    g = rng.normal(size=hs[-1].shape); norms = [float(np.linalg.norm(g))]
    for l in reversed(range(depth)):
        local = (g * (1 - np.tanh(zs[l]) ** 2)) @ Ws[l].T
        g = (g if residual else 0) + local; norms.append(float(np.linalg.norm(g)))
    return norms[::-1]
plain, res = resid_profile(False), resid_profile(True)
diag.append(("residual profile", (round(plain[0] / plain[-1], 8), round(res[0] / res[-1], 3))))
chart("backpropagation", {"after": "Why it is efficient — and where it goes wrong"}, "Gradient size by layer, with and without residual connections",
      {"kind": "line", "x": list(range(0, 51)), "yLog": True, "points": False, "series": [{"name": "plain 50-layer tanh network", "y": sig([v / plain[-1] for v in plain], 1e-33)}, {"name": "same network with residual connections (h + f(h))", "y": sig([v / res[-1] for v in res])}],
       "xLabel": "layer (0 = input; the gradient starts at layer 50)", "yLabel": "gradient norm relative to the output (log scale)"},
      after=f"Without shortcuts, every layer multiplies the gradient by a weight matrix and a tanh derivative, here shrinking it about {(plain[-1] / plain[0]) ** (1 / 50):.0f}-fold per layer, so the signal reaching the first layer is about {sci(plain[0] / plain[-1])} of the original: those layers cannot learn. With the very same weights plus residual connections, each layer computes h + f(h), whose local derivative is the identity plus a small term. The gradient keeps a direct path back and arrives at the first layer at roughly its original size ({res[0] / res[-1]:.1f} times the output gradient). This is why ResNets and transformers can be hundreds of layers deep.")

# ---------------- autograd: activation memory with and without checkpointing
Ls = [10, 20, 50, 100, 200, 500, 1000]
chart("autograd", {"after": "Memory: what the backward pass needs"}, "Activation memory: store everything or checkpoint",
      {"kind": "line", "x": Ls, "xLog": True, "yLog": True, "series": [{"name": "store every layer's activations", "y": Ls}, {"name": "checkpoint every √L layers", "y": r([2 * math.sqrt(L) for L in Ls], 1)}],
       "xLabel": "number of layers L (log scale)", "yLabel": "activations held in memory (in layers, log scale)"},
      after="Storing everything costs memory proportional to the depth. Keeping only every √L-th layer's activations and recomputing each segment during the backward pass needs about 2√L layers' worth: 63 instead of 1,000 for a 1,000-layer stack, for roughly one extra forward pass (about 30% more compute). Large-model training uses this trade constantly.")

# ---------------- training loop: overfitting curves and an LR range test
def moons(n, noise, seed):
    rng = np.random.default_rng(seed); t = rng.uniform(0, math.pi, n); lab = rng.integers(0, 2, n)
    x = np.where(lab == 0, np.cos(t), 1 - np.cos(t)); y = np.where(lab == 0, np.sin(t), 0.5 - np.sin(t))
    return np.column_stack([x, y]) + rng.normal(0, noise, (n, 2)), lab
Xtr, ytr = moons(60, 0.35, 1); Xva, yva = moons(1000, 0.35, 2)
def mlp_run(epochs, lr, hidden=64, seed=0, record=True, X=Xtr, y=ytr):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, math.sqrt(2 / 2), (2, hidden)); b1 = np.zeros(hidden); W2 = rng.normal(0, math.sqrt(1 / hidden), (hidden, 1)); b2 = np.zeros(1)
    def loss(X, y, W1, b1, W2, b2):
        p = 1 / (1 + np.exp(-(np.maximum(0, X @ W1 + b1) @ W2 + b2)[:, 0])); p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    tr, va = [], []
    for ep in range(epochs):
        z1 = X @ W1 + b1; h = np.maximum(0, z1); p = 1 / (1 + np.exp(-(h @ W2 + b2)[:, 0]))
        d = (p - y)[:, None] / len(y)
        gW2 = h.T @ d; gb2 = d.sum(0); dh = d @ W2.T * (z1 > 0); gW1 = X.T @ dh; gb1 = dh.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
        if record: tr.append(loss(Xtr, ytr, W1, b1, W2, b2)); va.append(loss(Xva, yva, W1, b1, W2, b2))
    return tr, va
tr, va = mlp_run(3000, 0.5)
ep = list(range(1, 3001)); keep = [e for e in ep if e <= 20 or e % 25 == 0]
best = int(np.argmin(va)) + 1
diag.append(("overfit", best, round(min(va), 3), round(va[-1], 3), round(tr[-1], 3)))
chart("training-loop", {"after": "Reading loss curves"}, "Training and validation loss of a small network that overfits",
      {"kind": "line", "x": keep, "xLog": True, "points": False, "series": [{"name": "training loss (60 examples)", "y": r([tr[e - 1] for e in keep], 4)}, {"name": "validation loss (1,000 examples)", "y": r([va[e - 1] for e in keep], 4)}],
       "xLabel": "epoch (log scale)", "yLabel": "cross-entropy loss", "vlines": [{"x": best, "label": f"best validation: epoch {best}"}]},
      md="A 64-unit network trained with full-batch gradient descent on only 60 noisy examples of the two-moons data:",
      after=f"Both losses fall together at first. Around epoch {best} the validation loss bottoms out at {min(va):.2f}; after that the training loss keeps falling (to {tr[-1]:.2f}) while the validation loss climbs back to {va[-1]:.2f}: the network is memorising the noise of its 60 examples. Early stopping would keep the weights from the marked epoch.")
lrs = np.logspace(-3, 1.6, 24); final = []
for lr in lrs:
    t_, v_ = mlp_run(40, lr, record=True)
    final.append(t_[-1] if np.isfinite(t_[-1]) else None)
diag.append(("lr range", [(round(float(a), 4), None if b is None else round(b, 3)) for a, b in zip(lrs, final)]))
chart("training-loop", {"after": "Going further: a debugging checklist"}, "A learning-rate range test",
      {"kind": "line", "x": [float(f"{v:.3g}") for v in lrs], "xLog": True, "series": [{"name": "training loss after 40 steps", "y": r(final, 4)}], "xLabel": "learning rate (log scale)", "yLabel": "loss after 40 steps"},
      md="The same network trained for just 40 steps at each of 24 learning rates:",
      after="Too small a rate barely moves the loss; it then falls faster and faster as the rate grows, until the steps start to overshoot and the loss turns back up. A good choice is somewhat below the bottom of the valley (often a few times smaller than the rate where the loss starts to rise), which leaves a safety margin for the rest of training.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d in diag: print(d)
print(len(out), "charts")
