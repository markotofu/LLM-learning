"""Chart data for batch L1 (language models, tokenization, attention, MHA, positions). Apply AFTER patches/l1-text.txt.
usage: python charts_l1.py out.txt corpus_ml.txt corpus_dl.txt"""
import json, sys, math, re, time
from collections import Counter
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
out, diag = [], []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- language models: n-gram perplexity on a synthetic language with a known order (trigram)
rng = np.random.default_rng(0)
V = 30
trans = rng.dirichlet(np.full(V, 0.15), size=(V, V))          # P(next | previous two tokens)
def generate(n):
    s = list(rng.integers(0, V, 2))
    for _ in range(n - 2):
        s.append(int(rng.choice(V, p=trans[s[-2], s[-1]])))
    return s
train, held = generate(20000), generate(5000)
def ngram_ppl(n, data, k=0.1):
    ctx, full = Counter(), Counter()
    for i in range(n - 1, len(train)):
        h = tuple(train[i - n + 1:i]); ctx[h] += 1; full[h + (train[i],)] += 1
    lp, cnt = 0.0, 0
    for i in range(n - 1, len(data)):
        h = tuple(data[i - n + 1:i]); p = (full[h + (data[i],)] + k) / (ctx[h] + k * V); lp += math.log(p); cnt += 1
    return math.exp(-lp / cnt)
ns = [1, 2, 3, 4, 5, 6]
tr = [ngram_ppl(n, train) for n in ns]; he = [ngram_ppl(n, held) for n in ns]
diag.append(("ngram", [round(x, 2) for x in tr], [round(x, 2) for x in he]))
chart("language-models", {"after": "Smoothing: giving unseen n-grams a chance"}, "Perplexity of n-gram models on training and held-out text",
      {"kind": "line", "x": ns, "series": [{"name": "training text", "y": r(tr, 2)}, {"name": "held-out text", "y": r(he, 2)}], "xLabel": "n (the model conditions on the previous n − 1 tokens)", "yLabel": "perplexity", "yMin": 0},
      md="A made-up language of 30 tokens in which each token truly depends on the previous two, so the ideal model is a trigram (n = 3). n-gram models with add-0.1 smoothing, trained on 20,000 tokens:",
      after=f"On the training text, perplexity keeps falling with every extra token of context (from 30 to under 4), as the counts increasingly memorise the text. On held-out text it falls until n = 3, the true order (about {he[2]:.1f}), then rises again: longer contexts are seen so rarely that their counts are mostly noise. That is the n-gram dilemma: more context is more informative but impossible to estimate by counting, which is exactly what neural models solve by sharing statistical strength between similar contexts.")

# ---------------- tokenization: BPE on the app's own lesson text
def words(path):
    return re.findall(r"[a-z]+", open(path, encoding="utf-8").read().lower())
train_words, held_words = words(sys.argv[2]), words(sys.argv[3])
freq = Counter(train_words)
vocab = {tuple(w) + ("</w>",): c for w, c in freq.items()}
def apply_merges(word, merges_rank):
    sym = list(word) + ["</w>"]
    while len(sym) > 1:
        pairs = [(merges_rank.get((sym[i], sym[i + 1]), 1e18), i) for i in range(len(sym) - 1)]
        best, i = min(pairs)
        if best == 1e18: break
        sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
    return sym
merges, checkpoints, results = [], [0, 50, 100, 200, 400, 800, 1600, 3200], []
t0 = time.time()
for step in range(1, 3201):
    pairs = Counter()
    for w, c in vocab.items():
        for a, b in zip(w, w[1:]): pairs[(a, b)] += c
    if not pairs: break
    best = max(pairs, key=pairs.get); merges.append(best)
    new = {}
    for w, c in vocab.items():
        out_w, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == best: out_w.append(w[i] + w[i + 1]); i += 2
            else: out_w.append(w[i]); i += 1
        new[tuple(out_w)] = c
    vocab = new
def tokens_per_word(ws, nmerge, cache={}):
    rank = {m: i for i, m in enumerate(merges[:nmerge])}
    tot = 0
    for w, c in Counter(ws).items(): tot += c * len(apply_merges(w, rank))
    return tot / len(ws)
tpw_tr = [tokens_per_word(train_words, k) for k in checkpoints]; tpw_he = [tokens_per_word(held_words, k) for k in checkpoints]
diag.append(("bpe", round(time.time() - t0, 1), "s", [round(x, 2) for x in tpw_tr], [round(x, 2) for x in tpw_he], len(train_words), len(held_words)))
chart("tokenization", {"after": "Training BPE from scratch"}, "Tokens per word as the BPE vocabulary grows",
      {"kind": "line", "x": [k + 27 for k in checkpoints], "xLog": True, "series": [{"name": "text the merges were learned on", "y": r(tpw_tr, 3)}, {"name": "new text (other lessons)", "y": r(tpw_he, 3)}],
       "xLabel": "vocabulary size: 27 symbols + number of merges (log scale)", "yLabel": "tokens per word", "yMin": 0},
      md=f"BPE trained on the words of this app's classical-ML lessons (about {len(train_words) // 1000},000 words), then used to tokenize the neural-network lessons:",
      after=f"With only characters, a word costs about {tpw_tr[0]:.1f} tokens (its letters plus the end-of-word marker). Every merge creates a token for a frequent fragment, and a few thousand merges bring common words down to one or two tokens. New text needs slightly more tokens than the training text because its words and fragments are less familiar, the same reason languages and domains under-represented in a tokenizer's training data cost more tokens.")

# ---------------- attention: why divide by sqrt(d)
dims = [4, 16, 64, 256, 1024]; T = 64
rng = np.random.default_rng(1)
def max_weight(d, scaled):
    m = []
    for _ in range(200):
        q = rng.normal(size=d); K = rng.normal(size=(T, d)); s = K @ q / (math.sqrt(d) if scaled else 1.0)
        w = np.exp(s - s.max()); w /= w.sum(); m.append(w.max())
    return float(np.mean(m))
mu, ms = [max_weight(d, False) for d in dims], [max_weight(d, True) for d in dims]
diag.append(("attn", [round(x, 3) for x in mu], [round(x, 3) for x in ms]))
chart("attention", {"after": "Self-attention in NumPy"}, "Why the scores are divided by √d",
      {"kind": "line", "x": dims, "xLog": True, "series": [{"name": "softmax(q·k)", "y": r(mu, 3)}, {"name": "softmax(q·k / √d)", "y": r(ms, 3)}], "xLabel": "dimension d of queries and keys (log scale)", "yLabel": "largest attention weight (of 64 keys)", "yMin": 0, "yMax": 1, "hlines": [{"y": round(1 / 64, 4), "label": "uniform: 1/64"}]},
      md="Random queries and keys with unit-variance entries, 64 keys, averaged over 200 draws:",
      after="A dot product of two d-dimensional vectors with unit-variance entries has a standard deviation of √d. Without scaling, the scores spread out as d grows, the softmax saturates and almost all the weight lands on a single key, where the softmax's gradient is nearly zero and learning stalls. Dividing by √d keeps the scores' spread, and therefore the softness of the attention, the same at every width.")

# ---------------- multi-head attention: KV-cache size
Ts = [1024, 4096, 16384, 65536, 131072]
def kv_gb(T, kv_heads, layers=32, d_head=128, bytes_=2): return 2 * layers * kv_heads * d_head * T * bytes_ / 1e9
series = [{"name": f"{name} ({h} key/value heads)", "y": r([kv_gb(T, h) for T in Ts], 2)} for name, h in [("multi-head", 32), ("grouped-query", 8), ("multi-query", 1)]]
chart("multi-head-attention", {"after": "Multi-query and grouped-query attention"}, "KV-cache memory for one sequence",
      {"kind": "line", "x": Ts, "xLog": True, "yLog": True, "series": series, "xLabel": "context length in tokens (log scale)", "yLabel": "KV cache in GB (log scale)"},
      md="A 7-billion-parameter-class model: 32 layers, 32 query heads of width 128, 16-bit numbers:",
      after=f"The cache grows linearly with the context. With full multi-head attention a single 128k-token sequence needs about {kv_gb(131072, 32):.0f} GB, more than the model's own weights. Grouped-query attention with 8 key/value heads cuts that by 4 (about {kv_gb(131072, 8):.0f} GB) and multi-query attention by 32, which is why long-context models use GQA almost universally.")

# ---------------- positional encoding: RoPE score against relative distance
d = 64; theta = 10000.0 ** (-np.arange(0, d, 2) / d)
dist = np.arange(0, 513, 4)
rng = np.random.default_rng(2); scores = np.zeros(len(dist))
for _ in range(500):
    q = rng.normal(size=d); k = q + 0.5 * rng.normal(size=d)       # a key similar to the query
    qa, qb, ka, kb = q[0::2], q[1::2], k[0::2], k[1::2]
    dot_same = qa * ka + qb * kb; cross = qa * kb - qb * ka
    scores += np.array([np.sum(dot_same * np.cos(D * theta) + cross * np.sin(D * theta)) for D in dist])
scores /= scores[0]
chart("positional-encoding", {"after": "Learned, relative and rotary positions"}, "RoPE: how the same query–key match decays with distance",
      {"kind": "line", "x": dist.tolist(), "points": False, "series": [{"name": "average attention score, relative to distance 0", "y": r(scores, 3)}], "xLabel": "distance between the query's and the key's positions (tokens)", "yLabel": "relative score", "hlines": [{"y": 0, "label": ""}]},
      md="A query and a similar key, rotated by RoPE to positions that are a given distance apart (d = 64, base 10,000, averaged over 500 random pairs):",
      after="At distance 0 the rotations cancel and the score is the plain dot product. As the distance grows, the fast-rotating dimensions fall out of phase first, and the average score drops quickly over the first few dozen tokens, then decays more slowly with ripples. The score depends only on the distance, never on the absolute positions, and nearby matches are naturally favoured over distant ones with the same content.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts")
