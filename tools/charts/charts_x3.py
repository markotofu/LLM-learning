"""Chart data for batch X3 (Capstone: Build and Evaluate a RAG System). Apply AFTER patches/x3-text.txt.
Runs the lesson's own "Steps 1–3" code (read from js/data with Node), so the charts match what the sample prints.
Usage, from anywhere: python tools/charts/charts_x3.py <output patch>"""
import json, sys, io, os, math, subprocess, contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def lesson_code(concept, title):
    js = ("const MLA = require('./tests/_load.js').load(); MLA.content.init();"
          f"const s = MLA.content.byId[{json.dumps(concept)}].sections.find(s => s.title === {json.dumps(title)});"
          "process.stdout.write(s.code);")
    return subprocess.run(["node", "-e", js], cwd=ROOT, capture_output=True, encoding="utf-8", check=True).stdout

out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

ns = {}                                    # run the lesson's "Steps 1–3" code to get its corpus, questions and retrievers
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(lesson_code("capstone-rag", "Steps 1–3 — build, measure, improve"), "rag_main", "exec"), ns)
np, QUESTIONS, tokenize, STOP, embed = ns["np"], ns["QUESTIONS"], ns["tokenize"], ns["STOP"], ns["embed"]

# ---- recall@k for BM25, the toy embedding retriever and reciprocal rank fusion, on two-sentence chunks (version v3)
chunks = ns["make_chunks"](2)
bm25, dense = ns["bm25_ranker"](chunks), ns["dense_ranker"](chunks)
toks = [set(w for w in tokenize(t) if w not in STOP) for _, t in chunks]
E = np.array([embed(t) for _, t in chunks])
def bm25_hits(q):                          # a search engine returns only chunks that share a query word
    qw = set(w for w in tokenize(q) if w not in STOP)
    return [i for i in bm25(q) if toks[i] & qw]
def dense_hits(q):
    s = E @ embed(q)
    return [i for i in dense(q) if s[i] > 0]
def rrf(q, k0=60):
    scores = {}
    for ranking in (bm25_hits(q), dense_hits(q)):
        for r, i in enumerate(ranking):
            scores[i] = scores.get(i, 0) + 1 / (k0 + r + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))
ks = list(range(1, 9))
def recall(fn, qtype=None):
    sel = [x for x in QUESTIONS if qtype is None or x[3] == qtype]
    return [round(100 * float(np.mean([need.lower() in " ".join(chunks[i][1] for i in fn(q)[:k]).lower() for q, _, need, _ in sel])), 1) for k in ks]
series = [("BM25 alone", bm25, False), ("embeddings alone", dense, False), ("reciprocal rank fusion of both", rrf, False)]
for name, fn, _ in series:
    print(name, recall(fn), "keyword", recall(fn, "keyword"), "paraphrase", recall(fn, "paraphrase"))
words3 = np.mean([sum(len(chunks[i][1].split()) for i in rrf(q)[:3]) for q, *_ in QUESTIONS])
print("RRF@3 context words", round(float(words3)))
chart("capstone-rag", {"after": "Reading the results like an engineer"}, "Recall as more chunks are retrieved, with and without fusion",
      {"kind": "line", "x": ks, "series": [{"name": name, "y": recall(fn), "dash": dash} for name, fn, dash in series],
       "xLabel": "chunks in the context (k)", "yLabel": "context recall@k, all 12 questions (%)", "yMin": 0, "yMax": 100},
      md="""Instead of concatenating two result lists, **reciprocal rank fusion** (RRF) merges them into one ranking, so a fixed number of chunks can be kept:

$$ RRF(d) = Σ_r 1 / (60 + rank_r(d)) $$
:::where
d — a chunk
r — each retriever (here BM25 and the embedding retriever)
rank_r(d) — d's position in retriever r's results (1 for the best); a retriever that did not return d adds nothing
60 — a constant that keeps the very top positions from dominating the sum
:::

Here it is on the two-sentence chunks of version v3:""",
      after=f"Fusion reaches 100% with only three chunks (about {round(float(words3))} words of context, against about 80 for v3's two merged lists), while BM25 alone stalls at 67% and the embedding retriever needs eight chunks to reach 92%. The average hides a weakness, though: on the paraphrase questions, where BM25 returns plausible but wrong chunks, fusion is *worse* than the embedding retriever alone when only one or two chunks are kept (43% and 71% against 86% and 100%). Whether to fuse, and how many chunks to keep, is a measurement, not a default.")

# ---- uncertainty of a measured share against the number of questions (normal approximation)
ns_ = [10, 20, 50, 100, 200, 500, 1000, 2000]
half = lambda p, n: round(196 * math.sqrt(p * (1 - p) / n), 1)
print("half widths", [half(0.8, n) for n in ns_])
chart("capstone-rag", {"after": "How many questions are enough?"}, "Uncertainty of a measured recall against the size of the evaluation set",
      {"kind": "line", "x": ns_, "xLog": True, "series": [{"name": f"measured recall {int(p * 100)}%", "y": [half(p, n) for n in ns_]} for p in (0.5, 0.8, 0.95)],
       "xLabel": "questions in the evaluation set (log scale)", "yLabel": "95% interval half-width (percentage points)", "yMin": 0,
       "hlines": [{"y": 5, "label": "±5 points"}]},
      after="Each tenfold increase in questions shrinks the uncertainty by a factor of about three (the square root of ten). The curves are highest for a recall near 50% and lower near the extremes, but even at 95% recall, 50 questions leave about ±6 points of doubt. Seeing a five-point improvement needs a few hundred questions, or a paired comparison on the same questions.")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print(len(out), "charts")
