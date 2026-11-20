"""Chart data for batch A1 (prompting, RAG, vector search, LLM evaluation, agents). Apply AFTER patches/a1x-text.txt.
usage: python charts_a1.py out.txt lessons.json"""
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

# ---------------- prompting: cumulative input tokens of a conversation, with and without prefix caching
turns = list(range(1, 41)); system, per_turn = 2000, 400
plain = np.cumsum([system + per_turn * t for t in turns])
cached = np.cumsum([(system + per_turn * (t - 1)) * 0.1 + per_turn for t in turns])   # earlier prefix billed at 10%, new turn at full price
chart("prompting", {"after": "Long conversations and prompt caching"}, "Input tokens billed over a conversation",
      {"kind": "line", "x": turns, "series": [{"name": "full price for the whole history every turn", "y": r(plain / 1000, 1)}, {"name": "with prefix caching (repeated prefix at 10%)", "y": r(cached / 1000, 1)}], "xLabel": "turn", "yLabel": "cumulative input tokens billed (thousands)", "yMin": 0},
      md="A 2,000-token system prompt and 400 new tokens per turn (question plus answer), resent in full at every turn:",
      after=f"Because every request resends the whole history, the total grows quadratically: after 40 turns about {plain[-1] / 1e3:.0f} thousand input tokens have been billed, although the conversation itself is only {(system + per_turn * 40) / 1e3:.0f} thousand tokens long. Caching the repeated prefix (billed here at an illustrative 10% of the normal price) flattens the curve to about {cached[-1] / 1e3:.0f} thousand. Long-running chats and agents also trim or summarise old turns for the same reason.")

# ---------------- RAG: chunk size against recall@k on this app's lessons
from sklearn.feature_extraction.text import TfidfVectorizer
rows = json.load(open(sys.argv[2], encoding="utf-8"))
def clean(t): return re.sub(r"[`*$#|_>]", " ", t)
def chunks(text, size):
    w = clean(text).split(); return [" ".join(w[i:i + size]) for i in range(0, max(len(w), 1), size)]
ks = [1, 2, 3, 5, 10]; series = []
for size in (50, 150, 400):
    cs, owner = [], []
    for i, r_ in enumerate(rows):
        for c in chunks(r_["body"], size): cs.append(c); owner.append(i)
    owner = np.array(owner)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(cs)
    C = vec.transform(cs); Q = vec.transform([r_["summary"] for r_ in rows])
    S = (Q @ C.T).toarray(); order = np.argsort(-S, axis=1)
    rec = [float(np.mean([i in owner[order[i, :k]] for i in range(len(rows))])) for k in ks]
    series.append({"name": f"chunks of {size} words ({len(cs):,} chunks)", "y": r(rec, 3)}); diag.append(("rag", size, len(cs), r(rec, 3)))
chart("rag", {"after": "Evaluating a RAG system"}, "Retrieval recall for three chunk sizes",
      {"kind": "line", "x": ks, "series": series, "xLabel": "k, the number of chunks retrieved", "yLabel": "recall@k: right lesson among the chunks", "yMin": 0, "yMax": 1},
      md="Retrieval on this app's 106 lessons: each lesson's one-sentence summary is the query, and a hit means at least one of the top k chunks comes from the right lesson (keyword-based TF-IDF retrieval):",
      after="Small chunks are precise but each holds little vocabulary, so the top result is more often a chunk of a neighbouring lesson; large chunks match more of the query's words but blur several topics together and would pass more irrelevant text to the model. Retrieving several chunks closes much of the gap. The best size depends on the documents and the queries, which is why chunking is tuned with a retrieval evaluation like this one, not chosen by habit.")

# ---------------- vector search: IVF recall against the number of probed lists
rng = np.random.default_rng(1)
n, d, n_list = 20000, 32, 100
centres = rng.normal(size=(60, d)) * 2
X = centres[rng.integers(0, 60, n)] + rng.normal(size=(n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
Qs = centres[rng.integers(0, 60, 200)] + rng.normal(size=(200, d)); Qs /= np.linalg.norm(Qs, axis=1, keepdims=True)
C = X[rng.choice(n, n_list, replace=False)]
for _ in range(10):
    assign = np.argmax(X @ C.T, axis=1)
    C = np.array([X[assign == j].mean(0) if np.any(assign == j) else C[j] for j in range(n_list)]); C /= np.linalg.norm(C, axis=1, keepdims=True)
assign = np.argmax(X @ C.T, axis=1)
exact = np.argsort(-(Qs @ X.T), axis=1)[:, :10]
probes = [1, 2, 4, 8, 16, 32, 64]; recall, scanned = [], []
for p_ in probes:
    near = np.argsort(-(Qs @ C.T), axis=1)[:, :p_]; rs, sc = [], []
    for qi in range(len(Qs)):
        cand = np.where(np.isin(assign, near[qi]))[0]; top = cand[np.argsort(-(X[cand] @ Qs[qi]))[:10]]
        rs.append(len(set(top) & set(exact[qi])) / 10); sc.append(len(cand) / n)
    recall.append(np.mean(rs)); scanned.append(np.mean(sc))
diag.append(("ivf", r(recall, 3), r(scanned, 3)))
chart("vector-search", {"after": "An IVF index from scratch"}, "IVF: recall against the share of vectors scanned",
      {"kind": "line", "x": probes, "xLog": True, "series": [{"name": "recall@10 (versus exact search)", "y": r(recall, 3)}, {"name": "share of the database scanned", "y": r(scanned, 3)}], "xLabel": "n_probe: lists searched (of 100, log scale)", "yLabel": "share", "yMin": 0, "yMax": 1},
      md="20,000 clustered vectors in an IVF index with 100 lists, 200 queries:",
      after=f"Searching a single list scans about {scanned[0]:.0%} of the database and finds about {recall[0]:.0%} of the true top 10: true neighbours often sit just across a list boundary. Each doubling of n_probe recovers more of them: with {probes[2]} lists recall is essentially {recall[2]:.0%} while scanning only about {scanned[2]:.0%} of the vectors. That steep early rise is why ANN search is so effective: most of the recall comes from a small part of the data.")

# ---------------- LLM evaluation: CI half-width against the number of cases
ns = [10, 20, 50, 100, 200, 500, 1000, 2000]
chart("llm-evaluation", {"after": "Comparing two systems: win rate and uncertainty"}, "How precise is a pass rate measured on n cases?",
      {"kind": "line", "x": ns, "xLog": True, "series": [{"name": "true rate 50%", "y": r([1.96 * math.sqrt(0.25 / n) for n in ns], 4)}, {"name": "true rate 90%", "y": r([1.96 * math.sqrt(0.09 / n) for n in ns], 4)}],
       "xLabel": "number of test cases n (log scale)", "yLabel": "± half-width of the 95% interval", "yMin": 0},
      after="With 20 cases a measured pass rate of 50% really means anything from about 28% to 72%: far too vague to tell two prompts apart. 200 cases narrow it to about ±7 points and 1,000 to about ±3. Rates near 0 or 100% are measured more precisely, which is why small eval sets can still catch a regression from 95% to 70% but not from 80% to 75%.")

# ---------------- agents: compounding reliability
steps = list(range(1, 51))
chart("agents", {"after": "Checks and retries change the arithmetic"}, "Probability of finishing a task without a single failed step",
      {"kind": "line", "x": steps, "points": False, "series": [{"name": f"each step succeeds {int(p * 100)}% of the time", "y": r([p ** s for s in steps], 4)} for p in (0.99, 0.95, 0.9)], "xLabel": "number of steps in the task", "yLabel": "chance that every step succeeds", "yMin": 0, "yMax": 1},
      after="Per-step reliability that sounds excellent compounds badly: at 95% per step, a 20-step task succeeds only about 36% of the time, and at 90% fewer than one in eight 20-step runs goes through without an error. This is why agent design focuses on detecting and recovering from failed steps, not just on making each step slightly better.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
for d_ in diag: print(d_)
print(len(out), "charts", round(plain[-1]), round(cached[-1]))
