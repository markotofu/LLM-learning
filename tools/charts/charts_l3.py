"""Chart data for batch L3 (fine-tuning, LoRA, alignment, quantization, efficient attention). Apply AFTER patches/l3-text.txt."""
import json, sys, math, warnings, time
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

# ---------------- fine-tuning: catastrophic forgetting (digits 0-4, then 5-9)
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
X, y = load_digits(return_X_y=True); X = X / 16.0
rng = np.random.default_rng(0); order = rng.permutation(len(X)); X, y = X[order], y[order]
tr, te = slice(0, 1300), slice(1300, None)
A_tr, B_tr = y[tr] < 5, y[tr] >= 5
A_te, B_te = y[te] < 5, y[te] >= 5
def run(replay):
    m = MLPClassifier(hidden_layer_sizes=(64,), learning_rate_init=0.01, random_state=0)
    Xa, ya = X[tr][A_tr], y[tr][A_tr]
    for _ in range(30): m.partial_fit(Xa, ya, classes=np.arange(10))
    accA, accB = [m.score(X[te][A_te], y[te][A_te])], [m.score(X[te][B_te], y[te][B_te])]
    Xb, yb = X[tr][B_tr], y[tr][B_tr]
    if replay:
        k = len(Xb) // 5; idx = rng.choice(len(Xa), k, replace=False)
        Xb, yb = np.vstack([Xb, Xa[idx]]), np.r_[yb, ya[idx]]
    for _ in range(30):
        m.partial_fit(Xb, yb); accA.append(m.score(X[te][A_te], y[te][A_te])); accB.append(m.score(X[te][B_te], y[te][B_te]))
    return accA, accB
t0 = time.time(); a0, b0 = run(False); a1, b1 = run(True)
diag.append(("forget", round(time.time() - t0, 1), r(a0[::6], 3), r(b0[::6], 3), r(a1[::6], 3), r(b1[::6], 3)))
ep = list(range(0, 31))
chart("fine-tuning", {"after": "When to fine-tune — and when not to"}, "Catastrophic forgetting, and how replay prevents it",
      {"kind": "line", "x": ep, "points": False, "series": [
          {"name": "old task (digits 0–4), fine-tuned on new data only", "y": r(a0, 3)},
          {"name": "old task, with 20% old examples mixed in", "y": r(a1, 3), "dash": True},
          {"name": "new task (digits 5–9), new data only", "y": r(b0, 3)},
          {"name": "new task, with replay", "y": r(b1, 3), "dash": True}],
       "xLabel": "fine-tuning epochs on the new task", "yLabel": "test accuracy", "yMin": 0, "yMax": 1},
      md="A small network first learns digits 0 to 4, then is fine-tuned on digits 5 to 9:",
      after=f"Trained only on the new digits, the network learns them within a few epochs and almost completely forgets the old ones (accuracy on 0 to 4 drops from {a0[0]:.0%} to {a0[-1]:.0%}): nothing in the new data asks it to keep answering 0 to 4. With a fifth as many old examples mixed into the fine-tuning data, the old task still collapses at first, while the new classes take over, but it recovers to about {a1[-1]:.0%} within about a dozen epochs, and the new task is learned just as well. LLM fine-tuning uses the same remedies: replay some general data, keep the learning rate small, train few epochs, or change few weights (LoRA).")

# ---------------- LoRA: memory to fine-tune a 7B model
N = 6.74e9; lora_params = 2_498_560 * 16
labels = ["full fine-tuning (Adam, mixed precision)", "LoRA, rank 16 (bf16 base)", "QLoRA, rank 16 (4-bit base)"]
vals = [16 * N / 1e9, (2 * N + 16 * lora_params) / 1e9, (0.5 * N + 16 * lora_params) / 1e9]
diag.append(("lora mem", [round(v, 1) for v in vals], lora_params))
chart("lora", {"after": "Using LoRA in practice"}, "Memory to fine-tune a 7-billion-parameter model",
      {"kind": "bar", "x": ["full fine-tuning", "LoRA (r = 16)", "QLoRA (r = 16)"], "series": [{"name": "weights + gradients + optimizer state (GB, before activations)", "y": r(vals, 1)}], "xLabel": "method", "yLabel": "GPU memory (GB)", "yMin": 0},
      md="A Llama-style 7B model (32 layers), with LoRA on all its linear layers; activations, which depend on batch size and sequence length, are not included:",
      after=f"Full fine-tuning needs about 16 bytes per parameter, roughly {vals[0]:.0f} GB, so it takes several large GPUs. LoRA freezes the base weights (2 bytes each, no gradients or optimizer state) and trains only {lora_params / 1e6:.0f} million adapter parameters, about {lora_params / N:.1%} of the model: about {vals[1]:.0f} GB. QLoRA stores the frozen weights in 4 bits and fits in about {vals[2]:.1f} GB, within reach of a single consumer GPU.")

# ---------------- alignment: best-of-n against a flawed reward model (Goodhart)
rng = np.random.default_rng(0)
ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]; trials = 3000
true_m, proxy_m = [], []
for n in ns:
    t = rng.normal(size=(trials, n)); err = rng.standard_t(2, size=(trials, n)) * 0.6
    proxy = t + err; pick = np.argmax(proxy, axis=1)
    true_m.append(float(t[np.arange(trials), pick].mean())); proxy_m.append(float(proxy[np.arange(trials), pick].mean()))
diag.append(("bon", r(true_m, 3), r(proxy_m, 2)))
chart("alignment", {"after": "Goodhart's law: optimising a proxy"}, "Best-of-n: the proxy reward keeps rising, the true quality does not",
      {"kind": "line", "x": ns, "xLog": True, "series": [{"name": "reward-model score of the chosen answer", "y": r(proxy_m, 3)}, {"name": "true quality of the chosen answer", "y": r(true_m, 3)}],
       "xLabel": "n, the number of candidate answers to choose from (log scale)", "yLabel": "average score (true quality has mean 0, sd 1)"},
      md="Sample n answers, keep the one the reward model likes best. Each answer's true quality is random; the reward model sees it plus an error that is usually small but occasionally huge (heavy-tailed):",
      after=f"Choosing from more candidates always raises the reward model's score of the winner. True quality improves at first, peaking at about {max(true_m):.2f} standard deviations with n ≈ {ns[int(np.argmax(true_m))]}, and then declines: with many candidates the winner is increasingly an answer the reward model merely over-rates. That is reward hacking in its simplest form, and why RLHF limits how far the policy may move (the KL penalty) and why reward models are re-trained on the policy's new outputs.")

# ---------------- quantization: error against bits, per-tensor vs per-block
rng = np.random.default_rng(0)
w = rng.normal(0, 0.02, 4096 * 64); w[rng.choice(len(w), 20, replace=False)] = rng.choice([-1, 1], 20) * 0.6   # a few outliers
def absmax(x, bits, block=None):
    qmax = 2 ** (bits - 1) - 1
    if block is None:
        s = np.abs(x).max() / qmax; return np.round(x / s) * s
    xb = x.reshape(-1, block); s = np.abs(xb).max(axis=1, keepdims=True) / qmax
    return (np.round(xb / s) * s).reshape(-1)
bits = [2, 3, 4, 5, 6, 8]
rel = lambda q: float(np.sqrt(np.mean((q - w) ** 2)) / np.sqrt(np.mean(w ** 2)))
pt = [rel(absmax(w, b)) for b in bits]; pb = [rel(absmax(w, b, 64)) for b in bits]
diag.append(("quant", r(pt, 3), r(pb, 3)))
chart("quantization", {"after": "Quantize a weight vector"}, "Quantisation error: one scale per tensor or one per block",
      {"kind": "line", "x": bits, "yLog": True, "series": [{"name": "one scale for the whole tensor", "y": sig(pt)}, {"name": "one scale per block of 64 weights", "y": sig(pb)}],
       "xLabel": "bits per weight", "yLabel": "relative error (RMS error ÷ RMS weight, log scale)"},
      md="262,144 weights drawn from a normal distribution, plus 20 large outliers, quantized with absmax rounding:",
      after=f"With one scale per tensor, the outliers stretch the range so much that at 4 bits almost every ordinary weight rounds to zero: the error is {pt[2]:.0%} of the weights' size. Giving every block of 64 weights its own scale confines each outlier's damage to its own block, cutting the 4-bit error to {pb[2]:.0%}. That is why practical 4-bit formats (GPTQ, AWQ, NF4) all use small groups of weights per scale.")

# ---------------- efficient attention: score-matrix memory against context length
Ts = [1024, 4096, 16384, 65536, 131072]
naive = [T * T * 32 * 2 / 1e9 for T in Ts]; flash = [T * 32 * 128 * 2 * 3 / 1e9 for T in Ts]
chart("efficient-attention", {"after": "FlashAttention: same result, far less memory traffic"}, "Extra memory for attention in one layer",
      {"kind": "line", "x": Ts, "xLog": True, "yLog": True, "series": [{"name": "materialise the T × T score matrix (32 heads)", "y": sig(naive)}, {"name": "FlashAttention: Q, K, V and output only", "y": sig(flash)}],
       "xLabel": "context length T (log scale)", "yLabel": "GB per layer, 16-bit (log scale)"},
      md="One layer with 32 heads of width 128, for a single sequence:",
      after=f"Writing out the full score matrix needs T² numbers per head: about {naive[2]:.0f} GB per layer at 16k tokens and over a terabyte at 128k, impossible on any GPU. FlashAttention never stores it: it streams tiles of keys and values through fast on-chip memory and keeps only running softmax statistics, so its memory grows linearly with T. The arithmetic is the same; what changes is how much data moves.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts")
