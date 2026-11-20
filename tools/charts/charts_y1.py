"""Chart data for batch Y1: the lessons added to cover the roadmap.sh AI Engineer and Machine Learning roadmaps
(video models, world models, robotics & VLAs, AI engineering, LLMs in production, discrete maths). Apply AFTER the
y*-text patches. Runs the lessons' own code samples (read from js/data with Node) where a chart shows their results.
Usage, from anywhere: python tools/charts/charts_y1.py <output patch>"""
import json, sys, io, os, math, subprocess, contextlib
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

# ---- video: tokens against clip length (the lesson's formula), and drift of frame-by-frame generation
def video_tokens(frames, h, w, t_comp=4, s_comp=8, patch=2):
    return math.ceil(frames / t_comp) * (h // (s_comp * patch)) * (w // (s_comp * patch))
secs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
chart("video-generation", {"after": "How many tokens is a video clip?"}, "Tokens in a clip as it gets longer",
      {"kind": "line", "x": secs, "yLog": True, "series": [
          {"name": f"{name} at 24 frames per second", "y": [video_tokens(24 * s, h, w) for s in secs]}
          for name, h, w in (("480p", 480, 848), ("720p", 720, 1280), ("1080p", 1088, 1920))],
       "xLabel": "clip length (seconds)", "yLabel": "tokens (log scale)", "hlines": [{"y": 1024, "label": "one 512 × 512 image"}]},
      md="The token formula above with a 4 × 8 × 8 video VAE and 2 × 2 patches:",
      after="Tokens grow in a straight line with length and in proportion to the number of pixels per frame, so a 20-second 1080p clip has close to a million tokens, about a thousand images' worth. Full attention cost grows with the square of these numbers, which is why most models generate short clips and extend them.")
frames = list(range(0, 241, 8))
drift = lambda sigma: [round(sigma * math.sqrt(2 * t / math.pi), 3) for t in frames]     # mean |N(0, σ² t)|
chart("video-generation", {"after": "Why frames must be generated together"}, "How frame-by-frame generation drifts",
      {"kind": "line", "x": frames, "points": False, "series": [
          {"name": "frame by frame, small error per frame (σ = 0.3 px)", "y": drift(0.3)},
          {"name": "frame by frame, larger error per frame (σ = 0.6 px)", "y": drift(0.6)},
          {"name": "all frames generated together", "y": [0.0] * len(frames), "dash": True}],
       "xLabel": "frame (240 frames = 10 seconds at 24 fps)", "yLabel": "average position error (pixels)", "yMin": 0},
      md="The blob example extended to ten seconds: the average distance between where frame-by-frame generation puts the blob and where the prompt's motion says it should be:",
      after="The error grows like the square root of the number of frames, the signature of a random walk: small, independent mistakes that are never corrected. Doubling the per-frame error doubles the drift at every length. Joint generation has no such accumulation within the clip it generates, which is why models generate as long a clip as they can afford in one go.")

# ---- world models: open-loop prediction error against the horizon
ns = run(lesson_code("world-models", "Errors compound over long rollouts"))
ns["rng"] = np.random.default_rng(1)            # replay the sample's random stream, so the curves are the ones it printed
H = 50
series = []
for n in (20, 200):
    A, B = ns["fit"](n)
    e = ns["rollout_error"](A, B, H)
    series.append({"name": f"model fitted from {n} transitions", "y": [sig(v) for v in e]})
    print("world", n, round(e[0], 4), round(e[9], 4), round(e[-1], 4))
chart("world-models", {"after": "Errors compound over long rollouts"}, "Open-loop prediction error against the horizon",
      {"kind": "line", "x": list(range(1, H + 1)), "yLog": True, "points": False, "series": series,
       "xLabel": "steps predicted ahead without seeing the real state", "yLabel": "average position error (log scale)"},
      md="The two models from the code above, step by step:",
      after="On a log scale the error rises steeply over the first steps and keeps growing, by two orders of magnitude over 50 steps for both models. Ten times more data lowers the whole curve but hardly changes its slope: better models delay the problem, and replanning is what solves it.")

# ---- robotics: survival of behaviour cloning and DAgger, and action-token rounding error
ns = run(lesson_code("robotics-vla", "Behaviour cloning drifts, DAgger recovers"))
steps = 100
alive = {}
for name in ("expert", "bc", "dagger"):
    lens = [len(ns["run"](ns[name])) for _ in range(400)]
    alive[name] = [round(100 * float(np.mean([L >= t for L in lens])), 1) for t in range(1, steps + 1)]
print("robot alive at 100", {k: v[-1] for k, v in alive.items()})
chart("robotics-vla", {"after": "Behaviour cloning drifts, DAgger recovers"}, "Share of runs still in the safe zone",
      {"kind": "line", "x": list(range(1, steps + 1)), "points": False, "series": [
          {"name": "expert", "y": alive["expert"], "dash": True},
          {"name": "behaviour cloning (20 calm demonstrations)", "y": alive["bc"]},
          {"name": "DAgger (after 4 rounds)", "y": alive["dagger"]}],
       "xLabel": "step", "yLabel": "runs still in the safe zone (%)", "yMin": 0, "yMax": 100},
      md="The policies from the code above, each run 400 more times:",
      after=f"Behaviour cloning loses runs steadily from the first few dozen steps on, each time a bump pushes the robot beyond the states its demonstrations covered, and only about {round(alive['bc'][-1])}% last all 100 steps. The DAgger policy stays with the expert throughout. A longer task would lose even more of the cloned runs, which is the compounding of errors in practice.")
bins = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
chart("robotics-vla", {"after": "Turning actions into tokens"}, "Rounding error against the number of action bins",
      {"kind": "line", "x": bins, "xLog": True, "yLog": True, "series": [
          {"name": "largest error: half a bin", "y": [sig(2 / (2 * (k - 1))) for k in bins]},
          {"name": "typical (RMS) error for uniform actions", "y": [sig(2 / (k - 1) / math.sqrt(12)) for k in bins]}],
       "xLabel": "bins per action dimension (log scale)", "yLabel": "error on an action range of 2 (log scale)", "vlines": [{"x": 256, "label": "RT-2: 256"}]},
      md="Exact values for actions spread evenly over [−1, 1], from the formula above:",
      after="Both errors halve every time the number of bins doubles: a straight line on log-log axes. At 256 bins the largest error is about 0.004, small next to the noise of most robot motions, so the real costs of action tokens are speed and vocabulary size, not precision.")

# ---- AI engineering: total cost per 1,000 requests as errors get more expensive
models = {"large": (3.00, 15.00, 0.92), "medium": (0.80, 4.00, 0.85), "small": (0.15, 0.60, 0.70)}
ecost = [round(0.01 * i, 2) for i in range(0, 31)]
per = {n: (1500 * pi + 300 * po) / 1e6 for n, (pi, po, a) in models.items()}
chart("ai-engineering", {"after": "Cost per outcome, not per token"}, "Total cost per 1,000 requests as errors get more expensive",
      {"kind": "line", "x": ecost, "points": False, "series": [
          {"name": f"{n} model ({int(a * 100)}% correct)", "y": [round(1000 * (per[n] + (1 - a) * e), 2) for e in ecost]}
          for n, (pi, po, a) in models.items()],
       "xLabel": "cost of one wrong answer (dollars)", "yLabel": "total cost per 1,000 requests (dollars)", "yMin": 0},
      md="The three models from the code, for error costs from nothing to 30 cents:",
      after="When errors cost nothing, the small model wins on price alone. From about 1.3 cents per error the medium model is cheapest, and from about 9 cents the large one: its higher price per token is repaid by its fewer mistakes. Real products often sit far to the right of this chart, with errors that cost minutes of someone's time.")

# ---- LLMs in production: the latency distribution with its percentiles
ns = run(lesson_code("llm-observability", "Percentiles, not averages"))
lat = ns["latency"]
edges = np.arange(0, 8.01, 0.2)
hist, _ = np.histogram(lat, bins=edges)
centres = [round(float(c), 2) for c in (edges[:-1] + 0.1)]
p50, p95, p99 = np.percentile(lat, [50, 95, 99])
print("latency", round(float(lat.mean()), 2), round(float(p50), 2), round(float(p95), 2), round(float(p99), 2))
chart("llm-observability", {"after": "Percentiles, not averages"}, "How the latencies are distributed",
      {"kind": "line", "x": centres, "points": False, "series": [{"name": "share of requests in each 0.2-second bin", "y": [round(100 * h / len(lat), 2) for h in hist]}],
       "xLabel": "latency (seconds)", "yLabel": "requests (%)", "yMin": 0,
       "vlines": [{"x": round(float(p50), 2), "label": "p50"}, {"x": round(float(p95), 2), "label": "p95"}, {"x": round(float(p99), 2), "label": "p99"}]},
      md="The 10,000 simulated latencies from the code above:",
      after="Most requests cluster around 1 to 2 seconds, but the distribution has a long right tail. The median sits left of the mean, and p95 and p99 lie far out in the tail, where the users who complain are. A dashboard that shows only the average would look the same whether that tail were short or long.")

# ---- discrete maths: how counts grow
n = list(range(2, 21))
chart("discrete-math", {"after": "How fast do counts explode?"}, "How counts grow with n",
      {"kind": "line", "x": n, "yLog": True, "series": [
          {"name": "n log₂ n (sorting)", "y": [sig(k * math.log2(k)) for k in n]},
          {"name": "n² (all pairs)", "y": [k * k for k in n]},
          {"name": "2ⁿ (all subsets)", "y": [2 ** k for k in n]},
          {"name": "n! (all orderings)", "y": [math.factorial(k) for k in n]}],
       "xLabel": "n", "yLabel": "count (log scale)"},
      after="On a log scale polynomial growth flattens out while exponential growth is a straight line and factorial growth curves upwards. By n = 20 there are about a million subsets and over 10¹⁸ orderings, against only 400 pairs: algorithms that enumerate subsets or orderings stop being usable long before the input gets large.")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print(len(out), "charts")
