"""Chart data for batch S1 (GNN, detection, speech, diffusion). Apply AFTER patches/s1-text.txt."""
import json, sys, math
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
sig = lambda v, floor=1e-12: [None if x is None else float(f"{max(float(x), floor):.4g}") for x in v]
out, diag = [], []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- GNN: over-smoothing, plain propagation versus residual
rng = np.random.default_rng(0)
n = 40; A = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        same = (i < 20) == (j < 20)
        if rng.random() < (0.25 if same else 0.08): A[i, j] = A[j, i] = 1
At = A + np.eye(n); d = At.sum(1); P = At / np.sqrt(np.outer(d, d))
H0 = rng.normal(size=(n, 8))
def spread(H):
    U = H / np.linalg.norm(H, axis=1, keepdims=True); return float(np.mean(np.linalg.norm(U - U.mean(0), axis=1)))
layers = list(range(0, 49)); plain, resid = [], []
Hp, Hr = H0.copy(), H0.copy()
for l in layers:
    plain.append(spread(Hp)); resid.append(spread(Hr))
    Hp = P @ Hp; Hr = 0.5 * Hr + 0.5 * (P @ Hr)
diag.append(("gnn", r(plain[::8], 4), r(resid[::8], 4)))
chart("gnn", {"after": "Pitfalls"}, "Over-smoothing: node vectors become indistinguishable",
      {"kind": "line", "x": layers, "yLog": True, "points": False, "series": [{"name": "plain propagation: H ← ÂH", "y": sig(plain)}, {"name": "half residual: H ← ½H + ½ÂH", "y": sig(resid)}],
       "xLabel": "number of message-passing layers", "yLabel": "spread of node-vector directions (log scale)"},
      md="Random features on a 40-node graph with two communities, propagated with the normalised adjacency Â of a GCN (no learned weights), measuring how much the directions of the node vectors still differ:",
      after="Each propagation step averages every node with its neighbours, so the differences shrink geometrically: after a few dozen layers every node's vector points in the same direction (only its length still depends on its degree), and a classifier on top can no longer tell the nodes apart. Keeping half of each node's own vector at every step (a residual-style update) slows the collapse considerably. This is why most GNNs use only 2 to 4 layers, plus residual connections, normalisation or jumping-knowledge readouts when they need more.")

# ---------------- detection: a precision-recall curve and AP
rng = np.random.default_rng(3)
n_gt, n_det = 60, 90
conf = rng.uniform(0, 1, n_det)
tp = rng.random(n_det) < (0.15 + 0.8 * conf)            # higher confidence, more likely correct
order = np.argsort(-conf); tp_sorted = tp[order]
cum_tp = np.cumsum(tp_sorted); cum_fp = np.cumsum(~tp_sorted)
precision = cum_tp / (cum_tp + cum_fp); recall = cum_tp / n_gt
interp = np.maximum.accumulate(precision[::-1])[::-1]     # the precision envelope used by AP
ap = float(np.sum(np.diff(np.r_[0, recall]) * interp))
diag.append(("ap", round(ap, 3), int(cum_tp[-1]), float(recall[-1])))
chart("detection-segmentation", {"after": "Evaluating detectors: average precision"}, "A precision–recall curve and its average precision",
      {"kind": "line", "points": False, "series": [{"name": "precision at each confidence threshold", "x": r(recall, 3), "y": r(precision, 3)}, {"name": "interpolated envelope (used for AP)", "x": r(recall, 3), "y": r(interp, 3), "dash": True}],
       "xLabel": "recall: share of the 60 objects found", "yLabel": "precision", "xMin": 0, "xMax": 1, "yMin": 0, "yMax": 1.02},
      md="90 detections of one class, sorted by confidence, scored against 60 ground-truth objects:",
      after=f"Walking down the list, every true positive moves the curve right and up a little, and every false positive pulls precision down. Even at the lowest threshold the detector finds only {recall[-1]:.0%} of the 60 objects, so recall never reaches 1. Average precision is the area under the dashed envelope, the best precision achievable at each recall level or beyond: here AP ≈ {ap:.2f}. mAP averages it over classes, and COCO's metric also averages over IoU thresholds from 0.5 to 0.95.")

# ---------------- speech: the mel scale
f = np.linspace(0, 8000, 81)
mel = 2595 * np.log10(1 + f / 700)
chart("speech-audio", {"after": "Spectrograms and the mel scale"}, "The mel scale compresses high frequencies",
      {"kind": "line", "x": r(f, 0), "points": False, "series": [{"name": "mel(f) = 2595 · log₁₀(1 + f / 700)", "y": r(mel, 1)}, {"name": "a linear scale, for comparison", "y": r(f * mel[-1] / 8000, 1), "dash": True}],
       "xLabel": "frequency f (Hz)", "yLabel": "mels"},
      after=f"Below about 1 kHz the mel scale is nearly linear; above it the curve flattens, so the 4,000 Hz between 4 and 8 kHz span only about {mel[-1] - mel[40]:.0f} mels, while the first 1,000 Hz span {2595 * math.log10(1 + 1000 / 700):.0f}. Spacing filters evenly in mels mirrors human hearing, which separates low pitches finely and high pitches coarsely, and gives speech models more detail where speech carries most of its information.")

# ---------------- diffusion: noise schedules
T = 1000; t = np.arange(1, T + 1)
beta = np.linspace(1e-4, 0.02, T); abar_lin = np.cumprod(1 - beta)
s = 0.008; fcos = np.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2; abar_cos = fcos / (math.cos(s / (1 + s) * math.pi / 2) ** 2)
keep = list(range(0, T, 20)) + [T - 1]
chart("diffusion-models", {"after": "The forward process"}, "How much signal survives at step t",
      {"kind": "line", "x": [int(t[i]) for i in keep], "points": False, "series": [{"name": "linear β schedule (DDPM)", "y": r(np.sqrt(abar_lin[keep]), 4)}, {"name": "cosine schedule", "y": r(np.sqrt(abar_cos[keep]), 4)}],
       "xLabel": "diffusion step t (of 1,000)", "yLabel": "signal scale √ᾱ_t", "yMin": 0, "yMax": 1},
      md="x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε: the curves show how much of the original data each noisy version still contains:",
      after=f"With the original linear schedule the signal is already mostly gone halfway through (√ᾱ ≈ {math.sqrt(abar_lin[499]):.2f} at t = 500), so many of the late steps are spent on almost pure noise. The cosine schedule destroys information more evenly ({math.sqrt(abar_cos[499]):.2f} at t = 500), which improved sample quality, especially for small images. Choosing how fast to add noise is one of the main design decisions of a diffusion model.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts")
