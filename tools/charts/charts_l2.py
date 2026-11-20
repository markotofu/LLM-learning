"""Chart data for batch L2 (transformer block, GPT, mini-GPT, pretraining, decoding). Apply AFTER patches/l2-text.txt."""
import json, sys, math
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
out, diag = [], []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- transformer block: FLOPs per token per layer against context length
d = 4096; Ts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
lin = [24 * d * d / 1e6 for _ in Ts]; att = [4 * T * d / 1e6 for T in Ts]
chart("transformer-block", {"after": "Counting parameters"}, "Where a token's compute goes in one block",
      {"kind": "line", "x": Ts, "xLog": True, "yLog": True, "series": [{"name": "weight matrices: projections and FFN (24d²)", "y": r(lin, 1)}, {"name": "attention scores and weighted sum (4·T·d)", "y": r(att, 1)}],
       "xLabel": "context length T (log scale)", "yLabel": "million FLOPs per token per layer (log scale)", "vlines": [{"x": 6 * d, "label": "T = 6d"}]},
      md="Forward-pass FLOPs for one new token in one block of a model with d = 4,096 (each multiply-add counts as 2 FLOPs):",
      after=f"The matrix multiplications with the weights cost the same for every token, about {lin[0]:.0f} million FLOPs here, whatever the context. The attention computation grows with the number of tokens attended to and only overtakes the weights at T ≈ 6d, about 24,600 tokens for this width. For ordinary prompts a transformer's compute is dominated by its weights; for very long contexts, attention takes over, which is where efficient-attention methods pay off.")

# ---------------- GPT-2 parameter breakdown
sizes = [("small", 768, 12), ("medium", 1024, 24), ("large", 1280, 36), ("XL", 1600, 48)]
V, Tm = 50257, 1024
emb = [(V * dd + Tm * dd) / 1e6 for _, dd, L in sizes]; blocks = [L * (12 * dd * dd + 13 * dd) / 1e6 for _, dd, L in sizes]
diag.append(("gpt2", [round(e + b + 2 * dd / 1e6, 1) for e, b, (_, dd, _) in zip(emb, blocks, sizes)]))
chart("gpt", {"after": "Counting parameters exactly"}, "Where GPT-2's parameters live",
      {"kind": "bar", "x": [f"GPT-2 {n}" for n, _, _ in sizes], "series": [{"name": "embeddings (tokens and positions)", "y": r(emb, 1)}, {"name": "transformer blocks", "y": r(blocks, 1)}], "xLabel": "model", "yLabel": "parameters (millions)", "yMin": 0},
      after=f"In GPT-2 small, nearly a third of the {emb[0] + blocks[0]:.0f} million parameters are the token-embedding table. The embeddings grow only linearly with the width, while the blocks grow with width² × depth, so in GPT-2 XL they are just {emb[3] / (emb[3] + blocks[3]):.0%} of the total. That is why parameter counts are often quoted without embeddings, and why vocabulary size matters most for small models.")

# ---------------- mini-GPT: the loss curve of the in-browser model (the lesson's exact code)
text = ("the model reads the text one character at a time. it learns which letters tend to follow which, "
        "then which words follow which. with more data and more layers it learns grammar, facts and style. "
        "the network is small, the data is tiny and the training is short, so the samples will look like "
        "words but will not make much sense. bigger models trained on more text write better text. ")
chars = sorted(set(text)); Vc = len(chars); stoi = {c: i for i, c in enumerate(chars)}; ids = np.array([stoi[c] for c in text])
ctx, embd, hidden, B = 4, 12, 128, 64
X = np.stack([ids[i:i + ctx] for i in range(len(ids) - ctx)]); Y = ids[ctx:]
rng = np.random.default_rng(0)
C = rng.normal(0, 0.5, (Vc, embd)); W1 = rng.normal(0, 1 / np.sqrt(ctx * embd), (ctx * embd, hidden)); b1 = np.zeros(hidden)
W2 = rng.normal(0, 0.01, (hidden, Vc)); b2 = np.zeros(Vc)
steps, losses, full = [], [], []
def full_loss():
    e = C[X].reshape(len(X), -1); h = np.tanh(e @ W1 + b1); lg = h @ W2 + b2
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); return float(-np.log(p[np.arange(len(X)), Y]).mean())
for step in range(2001):
    bi = rng.integers(0, len(X), B); e = C[X[bi]].reshape(B, -1); h = np.tanh(e @ W1 + b1); logits = h @ W2 + b2
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(B), Y[bi]]).mean()
    d_ = p.copy(); d_[np.arange(B), Y[bi]] -= 1; d_ /= B
    gW2, gb2 = h.T @ d_, d_.sum(0); dh = d_ @ W2.T * (1 - h ** 2); gW1, gb1 = e.T @ dh, dh.sum(0)
    gC = np.zeros_like(C); np.add.at(gC, X[bi], (dh @ W1.T).reshape(B, ctx, embd))
    if step % 25 == 0: steps.append(step); losses.append(float(loss)); full.append(full_loss())
    for param, grad in ((C, gC), (W1, gW1), (b1, gb1), (W2, gW2), (b2, gb2)): param -= 0.5 * grad
counts = np.bincount(ids, minlength=Vc) / len(ids); unigram = float(-(counts * np.log(counts)).sum())
pairs = np.zeros((Vc, Vc))
for a_, b_ in zip(ids, ids[1:]): pairs[a_, b_] += 1
cond = pairs / pairs.sum(1, keepdims=True); bigram = float(-np.mean([math.log(cond[a_, b_]) for a_, b_ in zip(ids, ids[1:])]))
diag.append(("mini", Vc, round(math.log(Vc), 3), round(unigram, 3), round(bigram, 3), round(full[-1], 3)))
chart("mini-gpt", {"after": "Step 2 — train a neural language model in your browser"}, "The loss curve you should see",
      {"kind": "line", "x": steps, "points": False, "series": [{"name": "mini-batch loss", "y": r(losses, 3)}, {"name": "loss on the whole text", "y": r(full, 3)}], "xLabel": "training step", "yLabel": "cross-entropy (nats per character)", "yMin": 0,
       "hlines": [{"y": round(math.log(Vc), 3), "label": f"uniform guess: ln {Vc}"}, {"y": round(unigram, 3), "label": "letter frequencies only"}, {"y": round(bigram, 3), "label": "best bigram model"}]},
      md="The model above, step by step (the mini-batch loss is noisy; the loss on the whole text is smooth):",
      after=f"The loss starts at ln {Vc} ≈ {math.log(Vc):.2f}, a uniform guess over the {Vc} characters. It quickly drops below what letter frequencies alone achieve ({unigram:.2f}), then below the best possible bigram model ({bigram:.2f}), because four characters of context say more than one. By the end it reaches about {full[-1]:.2f}: on such a tiny text the network has largely memorised which character follows each 4-character window.")

# ---------------- pretraining: IsoFLOP curves with the lesson's constants
E, A, Bc, alpha, beta = 1.8172, 482.01, 2085.43, 0.3478, 0.3658
loss = lambda N, D: E + A / N ** alpha + Bc / D ** beta
Ns = np.logspace(8, 12, 41); series = []
for Cb in (1e20, 1e21, 1e22, 1e23):
    ys = [loss(N, Cb / (6 * N)) if Cb / (6 * N) >= 1e9 else None for N in Ns]
    series.append({"name": f"budget {Cb:.0e} FLOPs".replace("e+", "e"), "y": r(ys, 4)})
    vals = [(loss(N, Cb / (6 * N)), N) for N in Ns]; diag.append(("iso", Cb, min(vals)[1]))
chart("pretraining", {"after": "The compute-optimal trade-off"}, "IsoFLOP curves: the best model size for each budget",
      {"kind": "line", "x": [float(f"{n:.3g}") for n in Ns], "xLog": True, "points": False, "series": series, "xLabel": "parameters N (log scale); data D = C / (6N)", "yLabel": "predicted loss (nats per token)"},
      md="For each fixed compute budget C, every model size N implies a number of training tokens D = C / (6N). The fitted scaling law then predicts the loss:",
      after="Each curve is a U: models that are too small cannot use their data, and models that are too big see too few tokens. The bottom of each U is the compute-optimal size, and it moves right roughly in proportion to √C, so ten times more compute buys a model about three times larger trained on three times more data. The minima are shallow, which is why practitioners deliberately choose smaller models trained on more tokens: they cost less to serve.")

# ---------------- decoding: how many tokens top-p keeps
ranks = np.arange(1, 1001)
dists = {"confident model (steep distribution)": 1 / ranks ** 2.0, "uncertain model (flat distribution)": 1 / ranks ** 0.8}
ps = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]; series = []
for name, w in dists.items():
    pr = w / w.sum(); cum = np.cumsum(pr); series.append({"name": name, "y": [int(np.searchsorted(cum, p_) + 1) for p_ in ps]})
diag.append(("topp", [s_["y"] for s_ in series]))
chart("decoding", {"after": "Top-k and top-p (nucleus) sampling"}, "How many tokens top-p keeps",
      {"kind": "line", "x": ps, "yLog": True, "series": series, "xLabel": "top-p threshold", "yLabel": "tokens kept (of 1,000, log scale)"},
      md="Two next-token distributions over a 1,000-token vocabulary, one peaked and one flat:",
      after=f"With p = 0.9, top-p keeps only {series[0]['y'][5]} tokens when the model is confident but {series[1]['y'][5]} when it is uncertain: the candidate set adapts to the situation. A fixed top-k cannot do this; k = 40 would either cut off plausible options in the flat case or let unlikely tokens through in the peaked one.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts")
