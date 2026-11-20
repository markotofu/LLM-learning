"""Chart data for batch N3 (embeddings, distillation & pruning, self-supervised). Apply AFTER patches/n3-text.txt."""
import json, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
sig = lambda v, floor=1e-12: [None if x is None else float(f"{max(float(x), floor):.4g}") for x in v]
out, diag = [], []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- embeddings: the lesson's co-occurrence vectors
corpus = """the cat drinks milk . the dog drinks water . a cat eats fish . a dog eats meat .
the kitten drinks milk . the puppy drinks water . a kitten eats fish . a puppy eats meat .
the car needs fuel . a truck needs fuel . the car drives fast . a truck drives slow .""".split()
vocab = sorted(set(corpus)); idx = {w: i for i, w in enumerate(vocab)}
C = np.zeros((len(vocab), len(vocab)))
for i, w in enumerate(corpus):
    for j in range(max(0, i - 2), min(len(corpus), i + 3)):
        if j != i: C[idx[w], idx[corpus[j]]] += 1
U, S, Vt = np.linalg.svd(np.log1p(C)); E = U[:, :4] * S[:4]; E = E / np.linalg.norm(E, axis=1, keepdims=True)
words = ["kitten", "dog", "puppy", "milk", "fish", "drinks", "car", "truck", "fuel"]
sims = [float(E[idx["cat"]] @ E[idx[w]]) for w in words]
Ec = E - E.mean(axis=0); Ec = Ec / np.linalg.norm(Ec, axis=1, keepdims=True)      # centred: remove the shared direction
sims_c = [float(Ec[idx["cat"]] @ Ec[idx[w]]) for w in words]
diag.append(("cat sims centred", dict(zip(words, [round(x, 3) for x in sims_c]))))
diag.append(("cat sims", dict(zip(words, [round(s, 3) for s in sims]))))
an = E[idx["puppy"]] - E[idx["dog"]] + E[idx["cat"]]; an = an / np.linalg.norm(an)
ranking = [vocab[i] for i in np.argsort(-(E @ an)) if vocab[i] not in ("puppy", "dog", "cat")][:3]
diag.append(("analogy puppy-dog+cat", ranking))
chart("embeddings", {"after": "Learning embeddings from co-occurrence counts"}, "How similar is each word to “cat”?",
      {"kind": "bar", "x": words, "series": [{"name": "raw vectors", "y": r(sims, 3)}, {"name": "after subtracting the mean vector", "y": r(sims_c, 3)}], "xLabel": "word", "yLabel": "cosine similarity with “cat”", "yMin": -1, "yMax": 1},
      md="The 4-dimensional vectors learned above from 12 tiny sentences, compared with “cat”:",
      after="With the raw vectors every word looks similar to “cat” (even “car” scores about 0.8), because all words share the same frequent neighbours (the, a, and the full stop), and that shared component dominates every vector. Subtracting the mean vector removes it, and the structure appears: the other animals stay close (about 0.8), while the vehicle words drop to clearly negative similarity. With only 4 dimensions and 12 sentences the picture is crude (fish ends up far from cat too), but real embedding spaces show the same shared-direction effect, called anisotropy, which is why centring or whitening embeddings often improves similarity search.")

# ---------------- distillation: soft targets and pruning
logits = np.array([9.0, 6.5, 5.4, 0.0]); classes = ["cat", "tiger", "dog", "car"]
def softmax(z, T=1.0):
    e = np.exp((z - z.max()) / T); return e / e.sum()
chart("distillation-pruning", {"after": "Soft targets reveal what the teacher knows"}, "The teacher's probabilities at three temperatures",
      {"kind": "bar", "x": classes, "series": [{"name": f"T = {T}", "y": r(softmax(logits, T), 4)} for T in (1, 2, 5)], "xLabel": "class", "yLabel": "probability", "yMin": 0, "yMax": 1},
      after="At T = 1 the teacher's answer is almost a hard label: 90% cat, and the other classes are hard to see. Raising the temperature flattens the distribution without changing the order, so the student can see that tiger and dog are plausible confusions while car is not. That relative structure is the extra signal distillation passes on.")
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
digits = load_digits(); X, y = digits.data / 16.0, digits.target
rng = np.random.default_rng(0); order = rng.permutation(len(X)); train, test = order[:1300], order[1300:]
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=0).fit(X[train], y[train])
original = [w.copy() for w in mlp.coefs_]; magnitudes = np.concatenate([np.abs(w).ravel() for w in original])
sparsities = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
acc_mag, acc_rand = [], []
rng2 = np.random.default_rng(1)
for s_ in sparsities:
    thr = np.quantile(magnitudes, s_) if s_ > 0 else -1
    mlp.coefs_ = [np.where(np.abs(w) >= thr, w, 0.0) for w in original]; acc_mag.append(mlp.score(X[test], y[test]))
    accs = []
    for _ in range(10):                                   # average over 10 random masks
        mlp.coefs_ = [np.where(rng2.random(w.shape) >= s_, w, 0.0) for w in original]; accs.append(mlp.score(X[test], y[test]))
    acc_rand.append(float(np.mean(accs)))
mlp.coefs_ = original
diag.append(("pruning", list(zip(sparsities, [round(a, 3) for a in acc_mag], [round(a, 3) for a in acc_rand]))))
chart("distillation-pruning", {"after": "Magnitude pruning a trained network"}, "Accuracy as weights are removed (no retraining)",
      {"kind": "line", "x": [round(100 * s_) for s_ in sparsities], "series": [{"name": "remove the smallest weights first", "y": r(acc_mag, 3)}, {"name": "remove weights at random (average of 10 tries)", "y": r(acc_rand, 3)}],
       "xLabel": "share of weights set to zero (%)", "yLabel": "test accuracy", "yMin": 0, "yMax": 1, "hlines": [{"y": 0.1, "label": "chance (10 classes)"}]},
      md="The digit classifier above, pruned to increasing sparsity in two ways:",
      after="Removing the smallest weights first costs almost nothing up to about 70% sparsity, then accuracy falls steeply. Removing the same number of weights at random hurts much sooner, because it also hits the large weights that carry most of the computation. Magnitude really is a decent proxy for importance, and fine-tuning after each round of pruning would recover much of the remaining loss.")

# ---------------- self-supervised: eigenvalue spectra of healthy and collapsed embeddings
rng = np.random.default_rng(0); B, d = 2048, 64
healthy = rng.normal(size=(B, d)) @ np.diag(np.linspace(1.0, 0.5, d))
partial = rng.normal(size=(B, 6)) @ rng.normal(size=(6, d)) + 0.02 * rng.normal(size=(B, d))
complete = np.tile(rng.normal(size=(1, d)), (B, 1)) + 1e-3 * rng.normal(size=(B, d))
def spectrum(Z):
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); Z = Z - Z.mean(axis=0)
    ev = np.sort(np.linalg.eigvalsh(Z.T @ Z / len(Z)))[::-1]; return np.maximum(ev, 1e-12)
specs = {"healthy: variance spread over all 64 dimensions": spectrum(healthy), "dimensional collapse: only 6 directions used": spectrum(partial), "complete collapse: every input gives the same vector": spectrum(complete)}
diag.append(("spectra", {k: (float(v[0]), float(v[5]), float(v[6]), float(v[-1])) for k, v in specs.items()}))
chart("self-supervised", {"after": "Diagnosing collapse: the eigenvalue spectrum"}, "Eigenvalue spectra of three sets of embeddings",
      {"kind": "line", "x": list(range(1, d + 1)), "yLog": True, "points": False, "series": [{"name": k, "y": sig(v)} for k, v in specs.items()], "xLabel": "eigenvalue rank (largest first)", "yLabel": "variance along the direction (log scale)", "yMin": 1e-9},
      after="A healthy representation spreads its variance over many directions, so the spectrum declines gently. Dimensional collapse shows up as a cliff: here everything after the sixth eigenvalue is noise, so the embeddings live in a 6-dimensional subspace of the 64 available, however good the training loss looks. Complete collapse leaves no variance at all. Plotting this spectrum during training is a cheap, label-free health check.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts")
