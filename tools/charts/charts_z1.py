"""Chart data for batch Z1: the ten topic lessons (graphical models, GLMs and survival analysis, 3D vision, AI for
science, audio generation, AI governance, contextual bandits, geometric deep learning, interpretability in practice,
AI hardware). Runs the lessons' own code samples (read from js/data with Node) where a chart shows their results, and
replays their random streams so the curves are the ones the samples print.
Usage, from anywhere: python tools/charts/charts_z1.py <output patch>"""
import json, sys, io, os, contextlib, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def lesson_code(concept, title):
    js = ("const MLA = require('./tests/_load.js').load(); MLA.content.init();"
          f"const s = MLA.content.byId[{json.dumps(concept)}].sections.find(s => s.title === {json.dumps(title)});"
          "process.stdout.write(s.code);")
    return subprocess.run(["node", "-e", js], cwd=ROOT, capture_output=True, encoding="utf-8", check=True).stdout
def run(code):
    ns = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "<lesson>", "exec"), ns)
    return ns

out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))
sig = lambda v: float(f"{v:.4g}")

# ---- graphical models: work needed to sum over hidden paths, brute force against the forward algorithm
days = list(range(1, 41))
chart("graphical-models", {"after": "The forward algorithm and the Viterbi algorithm"}, "Summing over hidden paths: brute force against the forward algorithm",
      {"kind": "line", "x": days, "yLog": True, "points": False, "series": [
          {"name": "every path, 2 states", "y": [2 * T * 2 ** T for T in days]},
          {"name": "forward algorithm, 2 states", "y": [T * (2 * 2 + 2) for T in days]},
          {"name": "forward algorithm, 10 states", "y": [T * (10 * 10 + 10) for T in days]}],
       "xLabel": "length of the observed sequence (days)", "yLabel": "multiplications (log scale)"},
      md="Multiplications needed to compute P(observations) for a hidden Markov model as the sequence gets longer: about 2T for each of the Kᵀ paths when summing them one by one, against K² + K per step for the forward algorithm:",
      after="The brute-force sum doubles with every extra day, so 40 days of two-state weather need about 10¹⁴ multiplications. The forward algorithm grows in a straight line: even with 10 hidden states, 40 days take 4,400. Paths that share a past share its sum, and dynamic programming computes that sum once.")

# ---- GLMs and survival: the Kaplan-Meier curve against the two shortcuts and the truth
ns = run(lesson_code("glm-survival", "Kaplan–Meier, and what goes wrong without it"))
def at(curve, t):                                     # value of a step curve at time t
    s = 1.0
    for ti, si in curve:
        if ti > t:
            break
        s = si
    return s
months = [round(0.5 * k, 1) for k in range(37)]
curves = {name: [round(at(ns[name], t), 4) for t in months] for name in ("km", "drop", "as_event")}
print("survival at 6, 12, 18 months:", {k: (v[12], v[24], v[36]) for k, v in curves.items()}, "true", [round(float(np.exp(-t / 12)), 3) for t in (6, 12, 18)])
chart("glm-survival", {"after": "Kaplan–Meier, and what goes wrong without it"}, "Survival curves with and without handling censoring",
      {"kind": "line", "x": months, "points": False, "series": [
          {"name": "Kaplan–Meier", "y": curves["km"]},
          {"name": "dropping censored customers", "y": curves["drop"]},
          {"name": "censored counted as cancelled", "y": curves["as_event"]},
          {"name": "true survival", "y": [round(float(np.exp(-t / 12.0)), 4) for t in months], "dash": True}],
       "xLabel": "months since joining", "yLabel": "share still subscribed", "yMin": 0, "yMax": 1},
      md="The three estimates from the code above over the whole 18 months of follow-up, with the true survival curve of the simulation:",
      after="The Kaplan–Meier curve stays within five percentage points of the true curve over the whole follow-up: 36% still subscribed after a year, against a true 37%. Dropping the censored customers keeps only those who cancelled during the follow-up, so that curve falls to 4% after a year and to zero by 18 months, and counting the censored customers as cancellations is not much better. Neither shortcut can tell a customer who left from one who has not been watched long enough.")

# ---- 3D vision: depth error caused by a one-pixel matching mistake
f = 800.0
Z = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50]
err = lambda B, z: f * B / (f * B / z - 1) - z
chart("vision-3d", {"after": "Projection and stereo depth"}, "Stereo depth error against distance",
      {"kind": "line", "x": Z, "yLog": True, "series": [
          {"name": "cameras 12 cm apart", "y": [sig(err(0.12, z)) for z in Z]},
          {"name": "cameras 50 cm apart", "y": [sig(err(0.5, z)) for z in Z]}],
       "xLabel": "distance to the point (m)", "yLabel": "error from a 1-pixel mistake (m, log scale)"},
      md="The formula from the code above (focal length 800 pixels) for the 12 cm rig and for a wider 50 cm rig:",
      after="The error grows roughly with the square of the distance and explodes as the disparity approaches one pixel: at 50 m the 12 cm rig can be off by more than 50 m. Four times the baseline cuts the error by four times or more at every distance (1.16 m against 0.26 m at 10 m), which is why long-range stereo uses wide rigs and why cars add LiDAR or radar.")
print("stereo error at 10 and 50 m:", [round(err(B, z), 2) for B in (0.12, 0.5) for z in (10, 50)])

# ---- AI for science: forecast error against lead time, one forecast and an ensemble
ns = run(lesson_code("ai-for-science", "Chaos and ensembles"))
step = ns["step"]
rng = np.random.default_rng(0)                        # replay the sample's random stream
cases, members = 100, 50
truth = rng.normal(size=(cases, 3)) + [1.0, 1.0, 20.0]
for _ in range(1000):
    truth = step(truth)
start = truth + rng.normal(0, 0.1, truth.shape)
single = start.copy()
ensemble = start[:, None, :] + rng.normal(0, 0.1, (cases, members, 3))
lead, e_single, e_mean, spread = [], [], [], []
for k in range(1, 2001):
    truth, single, ensemble = step(truth), step(single), step(ensemble)
    if k % 25 == 0:
        lead.append(round(k * 0.01, 2))
        e_single.append(sig(np.linalg.norm(single - truth, axis=1).mean()))
        e_mean.append(sig(np.linalg.norm(ensemble.mean(axis=1) - truth, axis=1).mean()))
        spread.append(sig(np.linalg.norm(ensemble.std(axis=1), axis=1).mean()))
print("lorenz at 2.5, 5, 10, 20:", [(lead[i], e_single[i], e_mean[i], spread[i]) for i in (9, 19, 39, 79)])
chart("ai-for-science", {"after": "Chaos and ensembles"}, "Forecast error against lead time",
      {"kind": "line", "x": lead, "points": False, "series": [
          {"name": "one forecast", "y": e_single},
          {"name": "ensemble mean", "y": e_mean},
          {"name": "ensemble spread", "y": spread, "dash": True}],
       "xLabel": "lead time (model time units)", "yLabel": "average error", "yMin": 0},
      md="The forecasts from the code above, measured every 25 steps instead of every 250:",
      after="Errors that start at about 0.2 grow quickly, and by a lead time of about 5 a single forecast is as far from the truth as a random point on the attractor and stops getting worse. The ensemble mean levels off 20 to 30% lower. The spread follows the ensemble mean's error at every lead time, which is what makes an ensemble useful: it says how far to trust each forecast.")

# ---- audio: residual vector quantization error against bits per frame, for two codebook sizes
ns = run(lesson_code("audio-generation", "Residual vector quantization, step by step"))
frames, kmeans = ns["frames"], ns["kmeans"]
def rvq_errors(K, levels=8):
    residual, errs = frames.copy(), []
    for _ in range(levels):
        codebook = kmeans(residual, K)
        index = ((residual[:, None, :] - codebook[None]) ** 2).sum(-1).argmin(1)
        residual = residual - codebook[index]
        errs.append(sig(np.linalg.norm(residual) / np.linalg.norm(frames)))
    return errs
e16 = [sig(v) for v in ns["errors"]]
ns["rng"] = np.random.default_rng(1)
e64 = rvq_errors(64)
print("rvq 16:", e16, "rvq 64:", e64)
chart("audio-generation", {"after": "Residual vector quantization, step by step"}, "Remaining error against bits per frame",
      {"kind": "line", "series": [
          {"name": "codebooks of 16 codes (4 bits each)", "x": [4 * k for k in range(1, 9)], "y": e16},
          {"name": "codebooks of 64 codes (6 bits each)", "x": [6 * k for k in range(1, 9)], "y": e64}],
       "xLabel": "bits per frame (one point per codebook added)", "yLabel": "remaining error (share of the signal)", "yMin": 0},
      md="The stages from the code above, and the same procedure with codebooks of 64 codes:",
      after="Each codebook removes a similar share of what is left, about a quarter, so the error falls quickly at first and then more slowly. At the same number of bits, codebooks of 64 codes leave less error than codebooks of 16 (0.10 against 0.13 at 24 bits) and need fewer stages, so a language model has fewer tokens to generate per frame. Real codecs use about 1,024 codes per codebook for the same reasons; the price is a larger codebook to learn and search.")

# ---- governance: training compute against model size, with the EU AI Act threshold
params = [sig(v) for v in np.logspace(8, 12.3, 30)]
named = [(1e9, 1e12), (8e9, 15e12), (70e9, 15e12), (405e9, 15.6e12), (1e12, 30e12)]
chart("ai-governance", {"after": "Training compute and the systemic-risk threshold"}, "Training compute against model size",
      {"kind": "line", "x": params, "xLog": True, "yLog": True, "points": False, "series": [
          {"name": "20 tokens per parameter (compute-optimal)", "y": [sig(6 * N * 20 * N) for N in params]},
          {"name": "15 trillion tokens", "y": [sig(6 * N * 15e12) for N in params]},
          {"name": "the five models in the code", "x": [N for N, _ in named], "y": [sig(6 * N * D) for N, D in named], "scatter": True}],
       "xLabel": "parameters (log scale)", "yLabel": "training FLOPs (log scale)",
       "hlines": [{"y": 1e25, "label": "EU AI Act threshold, 10²⁵ FLOPs"}]},
      md="Compute from the 6 × parameters × tokens rule for two ways of choosing the amount of training data:",
      after="With a fixed budget of 15 trillion tokens, which recent open models have used, the line is crossed at about 110 billion parameters. At the compute-optimal 20 tokens per parameter, a model needs about 290 billion parameters and 6 trillion tokens to reach it. As training sets keep growing, smaller models cross the line, which is one reason the Act lets the European Commission update the threshold.")

# ---- contextual bandits: cumulative regret of the three strategies
ns = run(lesson_code("contextual-bandits", "Epsilon-greedy, UCB and Thompson sampling"))
result = ns["result"]
idx = [0] + list(range(99, 5000, 100))
print("regret at 1000, 5000:", {k: (round(float(v[999]), 1), round(float(v[-1]), 1)) for k, v in result.items()})
chart("contextual-bandits", {"after": "Epsilon-greedy, UCB and Thompson sampling"}, "Clicks lost as visitors arrive",
      {"kind": "line", "x": [i + 1 for i in idx], "points": False, "series": [
          {"name": name, "y": [sig(float(r[i])) for i in idx]} for name, r in result.items()],
       "xLabel": "visitors so far", "yLabel": "clicks lost (cumulative regret)", "yMin": 0},
      md="The cumulative regret from the code above, averaged over its 200 runs:",
      after="Epsilon-greedy's losses settle into a straight line, because one visitor in ten always sees a random headline. Thompson sampling's curve keeps bending over as its posteriors narrow. UCB's rises steeply for thousands of visitors, because with click rates this low its exploration bonus stays large compared with the gaps between headlines; in theory its losses eventually grow only with the logarithm of the number of visitors, but that takes far longer than 5,000 visitors here.")

# ---- interpretability: length of each feature's direction in the toy model, dense against sparse
ns = run(lesson_code("interpretability-practice", "Superposition appears as features become sparse"))
ns["rng"] = np.random.default_rng(0)                  # replay the sample: same seed, same order of training runs
W = {s: ns["train"](s) for s in (0.0, 0.7, 0.9, 0.97, 0.99)}
norms = {s: [round(float(v), 3) for v in np.linalg.norm(W[s], axis=0)] for s in W}
print("represented:", {s: sum(v > 0.5 for v in norms[s]) for s in norms})
chart("interpretability-practice", {"after": "Superposition appears as features become sparse"}, "How long is each feature's direction?",
      {"kind": "bar", "x": [str(i) for i in range(1, 21)], "series": [
          {"name": "dense features (sparsity 0)", "y": norms[0.0]},
          {"name": "sparse features (sparsity 0.99)", "y": norms[0.99]}],
       "xLabel": "feature (1 is the most important)", "yLabel": "length of its column in W", "yMin": 0,
       "hlines": [{"y": 0.5, "label": "counted as represented"}]},
      md="The lengths of the 20 columns of W in two of the models trained above:",
      after="With dense features, the five most important features get directions of length 1 and the other fifteen stay close to 0: the model ignores them. At 99% sparsity, the 16 most important features have full-length directions and two more have about half, all within five dimensions. In both cases, the features left out are the least important ones.")

# ---- hardware: arithmetic intensity against tile size, with the ridge point
ns = run(lesson_code("ai-hardware", "How tiling raises arithmetic intensity"))
n, moved, flops, ridge = ns["n"], ns["moved"], ns["flops"], ns["ridge"]
tiles = [2 ** k for k in range(13)]
intensity = [sig(flops / (2 * moved(t))) for t in tiles]
cross = next(t for t in range(1, n + 1) if flops / (2 * moved(t)) >= ridge)
print("intensity:", dict(zip(tiles, intensity)), "crosses the ridge at", cross)
chart("ai-hardware", {"after": "How tiling raises arithmetic intensity"}, "Arithmetic intensity against tile size",
      {"kind": "line", "x": tiles, "xLog": True, "yLog": True, "series": [
          {"name": "4096 × 4096 matrix product, 16-bit values", "y": intensity}],
       "xLabel": "tile size t (log scale)", "yLabel": "FLOPs per byte moved (log scale)",
       "hlines": [{"y": round(ridge), "label": f"ridge point of an H100, about {round(ridge)}"}]},
      md="The simple count from the code above, from no reuse up to tiles as large as the matrices:",
      after=f"Intensity grows in proportion to the tile size and crosses the ridge point at tiles about {round(cross, -1)} values wide. It cannot go past about {round(n / 3):,} FLOPs per byte for these matrices, the point where every input value is read only once. Real kernels use tiles of 64 to 256 values in shared memory and get the rest of the reuse from the L2 cache.")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print(len(out), "charts")
