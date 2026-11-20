"""Chart data for batch L4 (serving, MoE, reasoning, text embeddings). Apply AFTER patches/l4-text.txt.
usage: python charts_l4.py out.txt lessons.json"""
import json, sys, math, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
out, diag = [], []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- serving: speculative decoding, expected tokens per target pass
alphas = np.round(np.arange(0.3, 0.96, 0.05), 2)
series = [{"name": f"draft proposes k = {k} tokens", "y": r([(1 - a ** (k + 1)) / (1 - a) for a in alphas], 3)} for k in (2, 4, 8)]
chart("llm-serving", {"after": "How much does speculative decoding help?"}, "Tokens produced per pass of the big model",
      {"kind": "line", "x": alphas.tolist(), "series": series, "xLabel": "acceptance rate α (chance the target model accepts each drafted token)", "yLabel": "expected tokens per target forward pass", "yMin": 0},
      md="If each drafted token is accepted independently with probability α, one verification pass yields (1 − α^(k+1)) / (1 − α) tokens on average:",
      after="At α = 0.5 longer drafts barely help: about 2 tokens per pass whatever k is, because the chain of acceptances usually breaks early. Only a draft model that agrees with the target most of the time (α ≥ 0.8) makes long drafts pay, reaching 4 to 5 tokens per pass at k = 8. The real speed-up is smaller than these numbers, since drafting itself costs time, which is why drafts come from much smaller models or from extra prediction heads on the target model.")

# ---------------- MoE: total versus active parameters
models = [("dense 70B", 70, 70), ("Mixtral 8x7B", 47, 13), ("DeepSeek-V3", 671, 37)]
chart("mixture-of-experts", {"after": "Total versus active parameters"}, "Parameters stored versus parameters used per token",
      {"kind": "bar", "x": [m[0] for m in models], "yLog": True, "series": [{"name": "total (must be in memory)", "y": [m[1] for m in models]}, {"name": "active per token (sets the compute)", "y": [m[2] for m in models]}], "xLabel": "model", "yLabel": "billions of parameters (log scale)", "yMin": 1},
      after="A dense model uses every parameter for every token. Mixtral stores 47 billion parameters but runs only about 13 billion per token, and DeepSeek-V3 stores 671 billion but runs 37 billion: about as much compute per token as a mid-sized dense model, with the knowledge capacity of a much larger one. The catch is memory: all experts must be loaded, because any token may need any of them.")

# ---------------- reasoning: accuracy against the number of samples (the lesson's simulation, more k)
rng = np.random.default_rng(0); p_correct, n_wrong, questions = 0.4, 5, 2000
ks = [1, 2, 3, 5, 8, 11, 16, 21, 32, 48, 64]; vote, verify = [], []
for k in ks:
    right = rng.random((questions, k)) < p_correct; wrong = rng.integers(1, n_wrong + 1, (questions, k)); ans = np.where(right, 0, wrong)
    counts = np.stack([(ans == a).sum(axis=1) for a in range(n_wrong + 1)], axis=1)
    maj = np.argmax(counts + 0.1 * rng.random(counts.shape), axis=1); vote.append(np.mean(maj == 0)); verify.append(np.mean((ans == 0).any(axis=1)))
diag.append(("reason", r(vote, 3), r(verify, 3)))
chart("reasoning-models", {"after": "Self-consistency and best-of-n, simulated"}, "Spending more samples: majority vote versus a verifier",
      {"kind": "line", "x": ks, "xLog": True, "series": [{"name": "best-of-k with a perfect verifier (pass@k)", "y": r(verify, 3)}, {"name": "majority vote over k samples", "y": r(vote, 3)}, {"name": "a single sample", "y": [0.4] * len(ks), "dash": True}],
       "xLabel": "number of sampled answers k (log scale)", "yLabel": "accuracy", "yMin": 0, "yMax": 1},
      md="The simulation above, extended: each sample is right with probability 0.4, otherwise one of 5 wrong answers:",
      after="Both methods turn extra samples into accuracy, with diminishing returns on a log scale. Majority voting works because the right answer is more common than any single wrong one; it levels off near 100% here only because the wrong answers are spread out. With a verifier the curve climbs much faster, since one correct sample is enough, which is why reasoning systems invest in checkers: unit tests, proof checkers and trained verifiers.")

# ---------------- text embeddings: retrieval on this app's own lessons
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
rows = json.load(open(sys.argv[2], encoding="utf-8"))
docs = [re.sub(r"[`*$#|_]", " ", r_["body"]) for r_ in rows]; queries = [r_["summary"] for r_ in rows]
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(docs + queries)
D, Q = vec.transform(docs), vec.transform(queries)
def recall_at(Dm, Qm, k):
    Dn = Dm / (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12); Qn = Qm / (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-12)
    S = Qn @ Dn.T; ranks = np.argsort(-S, axis=1)[:, :k]
    return float(np.mean([i in ranks[i] for i in range(len(rows))]))
tfidf1 = recall_at(D.toarray(), Q.toarray(), 1)
dims = [8, 16, 32, 64, 96]; lsa1 = []
for dd in dims:
    svd = TruncatedSVD(dd, random_state=0).fit(D); lsa1.append(recall_at(svd.transform(D), svd.transform(Q), 1))
diag.append(("retrieval", len(rows), round(tfidf1, 3), r(lsa1, 3), D.shape))
chart("text-embeddings", {"after": "Vector size, precision and the cost of an index"}, "Finding the right lesson from its one-line summary",
      {"kind": "line", "x": dims, "xLog": True, "series": [{"name": "dense vectors from a truncated SVD of TF-IDF (LSA)", "y": r(lsa1, 3)}], "xLabel": "vector dimension (log scale)", "yLabel": "top-1 accuracy (106 lessons)", "yMin": 0, "yMax": 1,
       "hlines": [{"y": round(tfidf1, 3), "label": f"sparse TF-IDF, {D.shape[1]:,} dimensions"}]},
      md="A real retrieval test on this app: each lesson's one-sentence summary is the query, and the system must rank that lesson's full text first among all 106 lessons:",
      after=f"Sparse word-count vectors find the right lesson for {tfidf1:.0%} of the summaries, using {D.shape[1]:,} dimensions. Compressing them into dense vectors with an SVD costs a lot at 8 dimensions ({lsa1[0]:.0%}) but reaches the sparse vectors' accuracy by 64: a few dozen directions capture most of what distinguishes the lessons. Modern embedding models work the same way in spirit, with learned instead of SVD directions, and models trained with nested (Matryoshka) objectives let you truncate their vectors to trade accuracy for index size.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts")
