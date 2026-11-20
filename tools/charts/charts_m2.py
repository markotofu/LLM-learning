"""Chart data for batch M2 (rest of maths + Python). Apply AFTER patches/m2-text.txt."""
import json, sys, math
import numpy as np
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
sig = lambda v, floor=1e-17: [None if x is None else float(f"{max(float(x), floor):.4g}") for x in v]
out = []
def sci(v):
    m, e = f"{v:.0e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m} × 10{sup}"
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- derivatives
x = np.linspace(-2.2, 2.2, 89)
chart("derivatives", {"after": "The rules you actually need"}, "A function and its derivative",
      {"kind": "line", "x": r(x, 2), "points": False, "series": [{"name": "f(x) = x³ − 3x", "y": r(x ** 3 - 3 * x)}, {"name": "f′(x) = 3x² − 3", "y": r(3 * x ** 2 - 3), "dash": True}],
       "xLabel": "x", "yLabel": "value", "vlines": [{"x": -1, "label": "f′ = 0: local maximum"}, {"x": 1, "label": "f′ = 0: local minimum"}]},
      after="Wherever the dashed derivative is positive, f rises; wherever it is negative, f falls. Where the derivative crosses zero, f has a flat spot: a local maximum at x = −1 and a local minimum at x = 1. Optimisers hunt for exactly these points, and the derivative's sign tells them which way to go.")
f, fp = np.sin, np.cos
hs = np.logspace(-1, -13, 25)
fwd = [abs((f(1 + h) - f(1)) / h - fp(1)) for h in hs]
cen = [abs((f(1 + h) - f(1 - h)) / (2 * h) - fp(1)) for h in hs]
chart("derivatives", {"after": "Estimating derivatives numerically"}, "Error of a numerical derivative against the step size h",
      {"kind": "line", "x": [float(f"{h:.3g}") for h in hs], "xLog": True, "yLog": True, "series": [{"name": "forward difference", "y": sig(fwd)}, {"name": "central difference", "y": sig(cen)}],
       "xLabel": "step size h (log scale; smaller to the right is not better!)", "yLabel": "error (log scale)", "points": False},
      md="Estimating the derivative of sin(x) at x = 1 (exactly cos 1) with ever smaller steps:",
      after=f"Read from left to right as h shrinks. At first the error falls, by 10× per decade of h for the forward difference and by 100× for the central one. Then floating-point rounding takes over: f(x + h) and f(x) agree in almost every digit, and their difference is mostly rounding noise divided by a tiny h. The best forward step is about 10⁻⁸ (error about {sci(min(fwd))}); the best central step about 10⁻⁵ to 10⁻⁶ (error about {sci(min(cen))}).")
x = np.linspace(-1, 3, 81); x0 = 1.0
tang = np.cos(x0) - np.sin(x0) * (x - x0)
quad = tang - 0.5 * np.cos(x0) * (x - x0) ** 2
clip = lambda v: [float(t) if -1.6 <= t <= 1.6 else None for t in v]
chart("derivatives", {"after": "Curvature and Taylor approximations"}, "Taylor approximations of cos(x) around x₀ = 1",
      {"kind": "line", "x": r(x, 2), "points": False, "series": [{"name": "cos(x)", "y": r(np.cos(x))}, {"name": "first order: the tangent line", "y": r(clip(tang)), "dash": True}, {"name": "second order: a parabola", "y": r(clip(quad)), "dash": True}],
       "xLabel": "x", "yLabel": "value", "vlines": [{"x": 1, "label": "x₀ = 1"}], "yMin": -1.6, "yMax": 1.6},
      after="Near x₀ both approximations hug the curve; further away the tangent line drifts off quickly, and the parabola, which also knows the curvature, stays close for longer. Gradient descent trusts the tangent line (so it must take small steps); Newton's method jumps straight to the bottom of the parabola.")

# ---------------- vectors-matrices
t = np.linspace(0, 2 * math.pi, 121)
chart("vectors-matrices", {"after": "Norms: three ways to measure size"}, "All vectors of length 1, under three norms",
      {"kind": "line", "points": False, "series": [
          {"name": "L1: |x₁| + |x₂| = 1", "x": [1, 0, -1, 0, 1], "y": [0, 1, 0, -1, 0]},
          {"name": "L2: √(x₁² + x₂²) = 1", "x": r(np.cos(t), 4), "y": r(np.sin(t), 4)},
          {"name": "L∞: max(|x₁|, |x₂|) = 1", "x": [1, -1, -1, 1, 1], "y": [1, 1, -1, -1, 1]}],
       "xLabel": "x₁", "yLabel": "x₂", "xMin": -2.82, "xMax": 2.82, "yMin": -1.2, "yMax": 1.2},
      after="The L1 \"circle\" is a diamond with its corners on the axes, where one coordinate is exactly 0. That shape is why [L1 regularization](#/concept/regularization) (lasso) tends to set weights exactly to zero: the loss contours usually first touch the diamond at a corner. L2 is the familiar round circle, and L∞ only cares about the largest coordinate.")

# ---------------- matrix-calculus
ns = [10 ** k for k in range(1, 10)]
chart("matrix-calculus", {"after": "Forward mode, reverse mode and the cost of gradients"}, "Cost of computing every partial derivative of one loss",
      {"kind": "line", "x": ns, "xLog": True, "yLog": True, "series": [
          {"name": "finite differences: 2n + 1 loss evaluations", "y": [2 * n + 1 for n in ns]},
          {"name": "forward mode: one pass per parameter", "y": ns},
          {"name": "reverse mode (backpropagation): about 3 passes", "y": [3 for n in ns]}],
       "xLabel": "number of parameters n (log scale)", "yLabel": "cost in forward passes (log scale)"},
      after="For a model with a billion parameters, finite differences would need two billion evaluations of the loss for a single gradient. Reverse mode gets all billion partial derivatives for the price of a few forward passes, which is the only reason training large networks is possible at all.")

# ---------------- optimization-theory: GD (two conditionings) vs Newton on L2-regularised logistic regression
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2)); yv = (rng.random(200) < 1 / (1 + np.exp(-(X @ np.array([1.5, -1.0]))))).astype(float)
lam = 0.01
def make(Xs):
    def f(w):
        z = Xs @ w; return float(np.mean(np.logaddexp(0, z) - yv * z) + lam / 2 * w @ w)
    def g(w):
        p = 1 / (1 + np.exp(-(Xs @ w))); return Xs.T @ (p - yv) / len(yv) + lam * w
    def H(w):
        p = 1 / (1 + np.exp(-(Xs @ w))); return (Xs.T * (p * (1 - p))) @ Xs / len(yv) + lam * np.eye(2)
    return f, g, H
def optimum(f, g, H):
    w = np.zeros(2)
    for _ in range(50): w = w - np.linalg.solve(H(w), g(w))
    return w, f(w)
runs = []
for name, Xs in [("gradient descent", X), ("gradient descent, one feature ×10", X * np.array([1.0, 10.0]))]:
    f_, g_, H_ = make(Xs); ws, fs = optimum(f_, g_, H_)
    L = 0.25 * np.linalg.eigvalsh(Xs.T @ Xs / len(yv)).max() + lam
    w, errs = np.zeros(2), []
    for k in range(41):
        errs.append(f_(w) - fs); w = w - (1 / L) * g_(w)
    runs.append({"name": name + f" (step 1/L)", "y": sig(errs, 1e-16)})
f_, g_, H_ = make(X); ws, fs = optimum(f_, g_, H_)
w, errs = np.zeros(2), []
for k in range(41):
    errs.append(f_(w) - fs); w = w - np.linalg.solve(H_(w), g_(w))
runs.insert(1, {"name": "Newton's method", "y": sig(errs, 1e-16)})
chart("optimization-theory", {"after": "Convergence rates: how fast is fast?"}, "Gradient descent against Newton's method on logistic regression",
      {"kind": "line", "x": list(range(41)), "yLog": True, "points": False, "series": runs, "xLabel": "iteration", "yLabel": "loss − best possible loss (log scale)", "yMin": 1e-16},
      md="L2-regularised logistic regression with two features, minimised from w = 0:",
      after="Gradient descent improves by a constant factor per step (a straight line on this log scale: linear convergence). Newton's method needs a few steps to get close, then the number of correct digits roughly doubles each step (quadratic convergence) until it hits the limits of floating-point precision. Stretching one feature by 10 multiplies that direction's curvature by 100, and the same gradient descent slows to a crawl; Newton's method is unaffected by such rescaling.")

# ---------------- numpy: spacing between neighbouring floats
mags = np.logspace(-3, 6, 37)
def spacing(dt):
    o = []
    for v in mags:
        if dt == "bfloat16": o.append(2.0 ** (math.floor(math.log2(v)) - 7)); continue
        if dt == "float16" and v > 65504: o.append(None); continue
        o.append(float(np.spacing(np.array(v, dtype=dt))))
    return o
chart("numpy", {"after": "dtypes, precision and memory"}, "The gap between neighbouring floating-point numbers",
      {"kind": "line", "x": [float(f"{m:.3g}") for m in mags], "xLog": True, "yLog": True, "points": False, "series": [
          {"name": "float16", "y": sig(spacing("float16"), 1e-30)}, {"name": "bfloat16", "y": sig(spacing("bfloat16"), 1e-30)},
          {"name": "float32", "y": sig(spacing("float32"), 1e-30)}, {"name": "float64", "y": sig(spacing("float64"), 1e-30)}],
       "xLabel": "size of the number (log scale)", "yLabel": "gap to the next representable number"},
      after="Every format keeps a fixed number of significant bits, so the gap grows with the size of the number. Near 1,000, float16 can only step in units of 0.5 and bfloat16 in units of 4; float32 resolves about 0.00006, and float64 about 10⁻¹³. float16 also stops at 65,504 (larger values become inf), while bfloat16 keeps float32's huge range by sacrificing precision. That trade-off is why it is the favourite format for training large models.")

# ---------------- feature-engineering: binning and polynomial counts
rng = np.random.default_rng(0)
xd = rng.uniform(-3, 3, 100); yd = np.sin(2 * xd) + 0.3 * xd + rng.normal(0, 0.3, 100)
grid = np.linspace(-3, 3, 301)
coef = np.polyfit(xd, yd, 1)
edges = np.linspace(-3, 3, 11); which = np.clip(np.digitize(xd, edges) - 1, 0, 9)
means = np.array([yd[which == b].mean() for b in range(10)])
gb = np.clip(np.digitize(grid, edges) - 1, 0, 9)
chart("feature-engineering", {"after": "Binning: letting a linear model bend"}, "A linear model before and after binning its input",
      {"kind": "line", "series": [
          {"name": "data", "x": r(xd, 3), "y": r(yd, 3), "scatter": True},
          {"name": "linear regression on x", "x": r(grid, 3), "y": r(np.polyval(coef, grid), 3)},
          {"name": "linear regression on 10 one-hot bins", "x": r(grid, 3), "y": r(means[gb], 3)}],
       "xLabel": "input feature x", "yLabel": "target y"},
      after="On the raw feature a linear model can only draw one straight line through a wavy pattern. After binning x into 10 intervals and one-hot encoding them, the same linear model learns one level per bin: a step function that follows the wave. The price: the steps are crude, the model knows nothing about how neighbouring bins relate, and with too many bins each level rests on very few points.")
from math import comb
degs = [1, 2, 3, 4, 5]
chart("feature-engineering", {"after": "Polynomial features and interactions"}, "How many polynomial features?",
      {"kind": "line", "x": degs, "yLog": True, "series": [{"name": f"{d} input features", "y": [comb(d + k, k) - 1 for k in degs]} for d in (10, 100, 1000)],
       "xLabel": "polynomial degree", "yLabel": "number of features (log scale)"},
      after="With 10 inputs, degree 3 gives 285 features: manageable. With 100 inputs it gives 176,850, and with 1,000 inputs over 167 million. Full polynomial expansions only make sense for a handful of inputs; otherwise add the few interactions you have a reason to believe in, or use a model (trees, kernels, neural networks) that finds them for you.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
print(len(out), "charts")
