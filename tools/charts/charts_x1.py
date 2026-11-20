"""Chart data for batch X1 (Advanced Agents). Apply AFTER patches/x1-text.txt."""
import json, sys
import numpy as np

out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---- pass@k and pass^k for the simulated agents of the evaluation harness (exact expectation over task difficulty)
d = np.linspace(0, 1, 200_001)                     # task difficulty ~ Uniform(0, 1), as in the harness
ks = list(range(1, 11))
series = []
for name, skill, dash in (("v1", 0.55, False), ("v2", 0.62, True)):
    p = 1 / (1 + np.exp(-(skill - d) * 6))
    at = [round(float(np.mean(1 - (1 - p) ** k)) * 100, 1) for k in ks]
    every = [round(float(np.mean(p ** k)) * 100, 1) for k in ks]
    series += [{"name": f"agent {name}: pass@k (solved at least once)", "y": at, "dash": dash},
               {"name": f"agent {name}: pass^k (solved every time)", "y": every, "dash": dash}]
    print(name, "pass@1", at[0], "pass@5", at[4], "pass^5", every[4], "pass@10", at[9], "pass^10", every[9])
chart("advanced-agents", {"after": "An agent evaluation harness"}, "Solved at least once against solved every time",
      {"kind": "line", "x": ks, "series": series, "xLabel": "number of attempts k", "yLabel": "share of tasks (%)", "yMin": 0, "yMax": 100},
      md="The two simulated agents of the harness, averaged over the whole range of task difficulty rather than over 30 sampled tasks:",
      after="At k = 1 both metrics are the plain success rate. With more attempts they move in opposite directions: pass@k climbs towards 100% because almost every task is solved *sometimes*, while pass^k falls because few tasks are solved *every* time. The better agent's lead is small on pass@k and much larger, in relative terms, on pass^k: consistency is where extra skill shows. Report the one that matches how the agent is used: pass@k when a person or a test picks the best of several tries, pass^k when people rely on a single run.")

# ---- input tokens read over a run: full history, compaction, prompt caching (same model as the code sample)
def cumulative(steps, system=5_000, per_step=2_000, budget=None, keep=10_000, cached_price=None):
    context, total, rows = system, 0.0, [0.0]
    for i in range(steps):
        if cached_price is None:
            total += context
        else:
            new = system if i == 0 else per_step
            total += cached_price * (context - new) + new
        context += per_step
        if budget and context > budget:
            context = keep
        rows.append(total)
    return rows
steps = list(range(0, 201, 5))
full, comp, cache = cumulative(200), cumulative(200, budget=60_000), cumulative(200, cached_price=0.1)
pick = lambda rows: [round(rows[s] / 1e6, 2) for s in steps]
print("200 steps:", round(full[200] / 1e6, 1), round(comp[200] / 1e6, 1), round(cache[200] / 1e6, 1))
chart("advanced-agents", {"after": "Why long agent runs get expensive"}, "Tokens read as an agent run gets longer",
      {"kind": "line", "x": steps, "points": False, "series": [
          {"name": "full history, read in full at every step", "y": pick(full)},
          {"name": "compacted whenever the context passes 60k tokens", "y": pick(comp)},
          {"name": "full history with a cached prefix (cost in uncached tokens)", "y": pick(cache), "dash": True}],
       "xLabel": "steps so far", "yLabel": "input tokens read in total (millions)"},
      md="The run from the code sample: a 5,000-token system prompt, and 2,000 new tokens of actions and tool results per step:",
      after="Re-reading a growing history makes the total grow with the square of the run length: 200 steps read about 41 million tokens. Compaction keeps the context bounded, so the total grows in a straight line once it starts (about 7 million). Prompt caching leaves the tokens the same but makes re-reading the unchanged prefix cheap, which brings the cost of the full history down to the equivalent of about 4.4 million uncached tokens. Real agents combine both, and the dashed line shows why a stable, unchanging prefix (system prompt and tool definitions first) is worth protecting.")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print(len(out), "charts")
