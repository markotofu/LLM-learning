"""Chart data for batch X4 (Capstone: Fine-Tune a Small Language Model with LoRA). Apply AFTER patches/x4-text.txt.
Runs the lesson's own code samples (read from js/data with Node), so the charts always match what the samples print.
Usage, from anywhere: python tools/charts/charts_x4.py <output patch>"""
import json, sys, io, os, subprocess, contextlib

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

MAIN = "The whole experiment: pretrain, fine-tune four ways, merge, quantize, generate"
ns = run(lesson_code("capstone-lora", MAIN))
points = []
for name, (P, lora, n) in ns["runs"].items():
    points.append((name, round(ns["loss"](P, "B_test", lora), 3), round(ns["loss"](P, "A_test", lora), 3)))
print(points)
chart("capstone-lora", {"after": MAIN}, "Each method on the two skills",
      {"kind": "scatter", "series": [{"name": name, "x": [new], "y": [old]} for name, new, old in points],
       "xLog": True, "yLog": True, "xLabel": "loss on the NEW style, recipes (log scale)", "yLabel": "loss on the OLD style, stories (log scale)"},
      md="The results table as a picture: the goal is the bottom-left corner, a low loss on both kinds of text.",
      after="The base model sits far to the right: it cannot write recipes at all. Fine-tuning without replay moves every method to the left but also far up, because the old skill is lost; LoRA, with far fewer trainable parameters, ends up even higher than full fine-tuning in this tiny model. Only the two runs with replay reach the bottom-left corner, close to the base model's old-style loss and close to full fine-tuning's new-style loss.")

ns = run(lesson_code("capstone-lora", "Watching forgetting happen"))
curves = ns["curves"]
steps = [s for s, _, _ in curves["full fine-tuning"]]
series = [{"name": name + ": old style", "y": [round(a, 3) for _, _, a in rows]} for name, rows in curves.items()]
print({name: [round(a, 2) for _, _, a in rows][::5] for name, rows in curves.items()})
chart("capstone-lora", {"after": "Watching forgetting happen"}, "Loss on the old style during fine-tuning",
      {"kind": "line", "x": steps, "series": series, "xLabel": "fine-tuning steps", "yLabel": "loss on the OLD style, stories",
       "hlines": [{"y": round(curves["full fine-tuning"][0][2], 2), "label": "base model"}]},
      after="Without replay, the old-style loss shoots up in the first few dozen steps and keeps rising more slowly afterwards, for full fine-tuning and even more for LoRA. With replay it wobbles briefly and settles back close to the base model's level. Measured every 20 steps, the curves show what a single final number cannot: when the damage happens, and which checkpoint would have been the best compromise.")

ns = run(lesson_code("capstone-lora", "How big should the rank be?"))
ranks = sorted(ns["results"])
full_new = round(ns["loss"](ns["full"], "B_test"), 3)
lossr = [round(ns["results"][r], 3) for r in ranks]
print("rank sweep", ranks, lossr, "full", full_new)
chart("capstone-lora", {"after": "How big should the rank be?"}, "Loss on the new style against the LoRA rank",
      {"kind": "line", "x": ranks, "xLog": True, "series": [{"name": "LoRA, 300 steps", "y": lossr}],
       "xLabel": "rank r (log scale)", "yLabel": "loss on the NEW style, recipes", "yMin": 0,
       "hlines": [{"y": full_new, "label": "full fine-tuning"}]},
      after="The loss drops steeply from rank 2 to rank 4 and then flattens just above full fine-tuning: past the capacity this change needs, extra rank buys almost nothing. Choose the smallest rank on the flat part, since it trains, stores and serves most cheaply.")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n\n".join(out) + "\n")
print(len(out), "charts")
