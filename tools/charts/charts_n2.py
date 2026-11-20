"""Chart data for batch N2 (optimizers, normalization, CNN, RNN). Apply AFTER patches/n2-text.txt."""
import json, sys, math
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
sig = lambda v, floor=1e-30: [None if x is None else float(f"{max(float(x), floor):.4g}") for x in v]
out, diag = [], []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- optimizers: paths on the lesson's ravine
grad = lambda p: np.array([p[0], 50 * p[1]])
loss = lambda p: 0.5 * (p[0] ** 2 + 50 * p[1] ** 2)
def path(update, steps=100):
    p, s, pts = np.array([-10.0, 1.0]), {}, []
    pts.append(p.copy())
    for t in range(1, steps + 1):
        p = update(p, grad(p), s, t); pts.append(p.copy())
    return np.array(pts)
def sgd(p, g, s, t, lr=0.03): return p - lr * g
def momentum(p, g, s, t, lr=0.03, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g; return p - lr * s["v"]
def adam(p, g, s, t, lr=0.5, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g; s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    return p - lr * (s["m"] / (1 - b1 ** t)) / (np.sqrt(s["v"] / (1 - b2 ** t)) + eps)
cx, cy = [], []
for c in [0.5, 5, 20, 50]:
    tt = np.linspace(0, 2 * math.pi, 73); cx += list(np.sqrt(2 * c) * np.cos(tt)) + [None]; cy += list(np.sqrt(2 * c / 50) * np.sin(tt)) + [None]
series = [{"name": "loss contours", "x": r(cx, 3), "y": r(cy, 3), "dash": True}]
for name, fn in [("SGD", sgd), ("momentum", momentum), ("Adam", adam)]:
    P = path(fn); series.append({"name": f"{name} (loss after 100 steps {loss(P[-1]):.2g})", "x": r(P[:, 0], 3), "y": r(P[:, 1], 3)})
    diag.append((name, loss(P[-1]), P[-1].round(3).tolist()))
chart("optimizers", {"after": "SGD vs momentum vs Adam on a ravine"}, "The three optimizers' paths through the ravine",
      {"kind": "line", "points": False, "series": series, "xLabel": "x (the gentle direction)", "yLabel": "y (the steep direction)", "xMin": -11, "xMax": 3, "yMin": -1.3, "yMax": 1.3},
      md="The same 100 steps, drawn in parameter space, starting from (−10, 1):",
      after="SGD drops to the valley floor almost at once, then shrinks x by only 3% per step: after 100 steps it is still at x ≈ −0.48, with a loss about 300 times higher than momentum's. Momentum builds up speed along the valley and overshoots a little before settling. Adam takes steps of roughly the same size in both coordinates, so it marches along x at about 0.5 per step instead of crawling.")

# ---------------- normalization: activation scale through 30 layers, with and without LayerNorm
rng = np.random.default_rng(0)
x = rng.normal(size=(512, 256))
def profile(std, norm, seed=0):
    rr = np.random.default_rng(seed); h, out_ = x, []
    for _ in range(30):
        h = np.maximum(0, h @ rr.normal(0, std, (256, 256)))
        if norm: h = (h - h.mean(axis=1, keepdims=True)) / np.sqrt(h.var(axis=1, keepdims=True) + 1e-5)
        out_.append(float(np.abs(h).mean()))
    return out_
profs = {"init std 0.01, no normalisation": profile(0.01, False), "init std 0.01, LayerNorm after every layer": profile(0.01, True), "He init, no normalisation": profile(math.sqrt(2 / 256), False)}
diag.append(("norm last", {k: v[-1] for k, v in profs.items()}))
chart("normalization", {"after": "Layer normalization and RMSNorm"}, "Activation size through 30 ReLU layers",
      {"kind": "line", "x": list(range(1, 31)), "yLog": True, "points": False, "series": [{"name": k, "y": sig(v)} for k, v in profs.items()],
       "xLabel": "layer", "yLabel": "average |activation| (log scale)"},
      md="The badly scaled network from the first code example, with and without a LayerNorm after every layer:",
      after="With weights that are too small, the activations shrink about tenfold per layer and are numerically dead long before layer 30. Adding a LayerNorm after every layer resets each token's activations to mean 0 and variance 1, so the same bad weights keep a healthy scale all the way down. Normalisation makes training far less sensitive to the initial scale, although a good initialisation (the He line) achieves the same without it.")

# ---------------- CNN: receptive field and parameter counts
def rf_curve(pool_every):
    rf, jump, out_, convs = 1, 1, [], 0
    while convs < 12:
        rf += 2 * jump; convs += 1; out_.append(rf)
        if pool_every and convs % pool_every == 0: rf += jump; jump *= 2
    return out_
chart("cnn", {"after": "Receptive fields and parameter counts"}, "Receptive field against depth",
      {"kind": "line", "x": list(range(1, 13)), "series": [{"name": "3×3 convolutions only", "y": rf_curve(0)}, {"name": "3×3 convolutions, 2×2 pooling after every two", "y": rf_curve(2)}],
       "xLabel": "number of 3×3 convolution layers", "yLabel": "receptive field (pixels, one side)", "hlines": [{"y": 224, "label": "a 224-pixel image"}]},
      after="Stacking 3×3 convolutions alone grows the receptive field by only 2 pixels per layer: 12 layers see a 25×25 patch. Pooling after every second layer doubles the step size each time, so 12 layers see more than the whole 224×224 image. That is how deep layers come to respond to whole objects while each individual filter stays tiny.")
labels = ["conv 3×3, 3→64", "conv 3×3, 64→128", "dense 7×7×512→4,096", "dense 224×224×3→1,000"]
counts = [3 * 3 * 3 * 64 + 64, 3 * 3 * 64 * 128 + 128, 7 * 7 * 512 * 4096 + 4096, 224 * 224 * 3 * 1000 + 1000]
chart("cnn", {"after": "Receptive field against depth"}, "Parameters of convolutional and dense layers",
      {"kind": "bar", "x": labels, "yLog": True, "series": [{"name": "parameters (weights + biases)", "y": counts}], "xLabel": "layer", "yLabel": "parameters (log scale)", "yMin": 1000},
      after="The two convolutions have 1,792 and 73,856 parameters, independent of the image size. VGG-16's first dense layer alone has 103 million, more than all of its 13 convolutional layers together, and a dense layer connected straight to a 224×224 colour image would need 150 million for just 1,000 units. That gap is why modern CNNs end with global average pooling instead of large dense layers.")

# ---------------- RNN: the lesson's gradient experiment, over all 50 steps
rng = np.random.default_rng(1)
d = 32; series = []
for scale in [0.5, 1.0, 1.5]:
    Wh = rng.normal(0, scale / np.sqrt(d), (d, d)); g = np.ones(d); norms = []
    for t in range(50): g = Wh.T @ g; norms.append(np.linalg.norm(g))
    series.append({"name": f"recurrent weight scale {scale}", "y": sig(norms)}); diag.append(("rnn", scale, norms[9], norms[49]))
chart("rnn", {"after": "Watching a gradient vanish (or explode)"}, "The gradient's size as it travels back in time",
      {"kind": "line", "x": list(range(1, 51)), "yLog": True, "points": False, "series": series, "xLabel": "time steps back", "yLabel": "gradient norm (log scale)"},
      after="The experiment above, step by step. Each step back multiplies the gradient by the same recurrent matrix, so its size changes geometrically: at scale 0.5 it halves at every step and is negligible after a couple of dozen; at 1.5 it grows by half at every step and explodes; near 1 it changes slowly (here it grows about fifteenfold over 50 steps), a knife-edge that training cannot rely on. (The tanh derivatives, left out here, only shrink it further.)")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts")
