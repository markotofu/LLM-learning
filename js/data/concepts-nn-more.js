/* ML Atlas – content: Neural Networks (optimizers, normalization, PyTorch, CNNs, RNNs, embeddings) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'optimizers', branch: 'dl', title: 'Optimizers: Momentum, RMSProp & Adam', short: 'Optimizers', level: 2, minutes: 40, prereqs: ['training-loop'],
    tags: ['sgd', 'momentum', 'rmsprop', 'adam', 'adamw', 'learning rate schedule', 'warmup', 'gradient clipping'],
    summary: 'Smarter ways to follow the gradient: momentum to roll through ravines, adaptive per-parameter step sizes, Adam/AdamW, learning-rate warmup and cosine decay, and gradient clipping.',
    why: `Plain gradient descent struggles on the long, narrow valleys and noisy gradients typical of neural networks. Optimizers like **Adam** and **AdamW** — used to train virtually every transformer and LLM — fix this with momentum and per-parameter adaptive step sizes. Combined with a **learning-rate schedule** (warmup then cosine decay) and **gradient clipping**, they are what makes training large models stable. The optimizer state is also a big part of the memory cost of training (Adam stores two extra numbers per parameter).`,
    objectives: [
      'Explain the problems of plain SGD: ravines, noise and a single global learning rate',
      'Implement momentum and explain it as a running average of gradients',
      'Implement Adam with bias correction and explain AdamW\'s decoupled weight decay',
      'Implement a warmup + cosine learning-rate schedule and gradient clipping'
    ],
    sections: [
      { type: 'text', title: 'Where plain SGD struggles', md: `1. **Ravines.** When the loss curves steeply in one direction and gently in another (an elongated bowl), the gradient points mostly across the ravine. A learning rate small enough to avoid bouncing between the steep walls crawls painfully along the valley floor.
2. **Noise.** Mini-batch gradients are noisy, so the path zig-zags.
3. **One learning rate for everyone.** Some parameters get large, frequent gradients; others (e.g. embeddings of rare words) get tiny, rare ones. A single global step size can't suit both.` },
      { type: 'text', title: 'Momentum', md: `Keep a **velocity** — a running (exponentially weighted) sum of past gradients — and step along it:

$$ v ← β v + g          θ ← θ − η v $$
:::where
g — the gradient of the loss on the current mini-batch
v — the velocity: a running, exponentially weighted sum of past gradients (starts at 0)
β — "beta": how much of the old velocity is kept at each step
θ — the parameters
η — the learning rate
:::

with β ≈ 0.9. Gradient components that consistently point the same way **build up speed**; components that flip sign every step (bouncing across a ravine) **cancel out**. Like a heavy ball rolling downhill, it smooths the noise and accelerates along the valley.

**Nesterov momentum** evaluates the gradient at the look-ahead point θ − ηβv, which slightly improves the correction.` },
      { type: 'text', title: 'Adaptive learning rates: RMSProp and Adam', md: `**RMSProp** divides each parameter's step by a running average of its squared gradients:

$$ s ← ρ s + (1 − ρ) g²          θ ← θ − η g / (√s + ε) $$
:::where
g — the current gradient; g² squares it element by element
s — a running average of squared gradients, one value per parameter (starts at 0)
ρ — "rho": the decay rate of that average (≈ 0.9 or 0.99)
√s — roughly the typical gradient size of each parameter; dividing by it evens out the step sizes
ε — "epsilon": a tiny constant (≈ 1e-8) that prevents division by zero
η — the learning rate
:::

Parameters with large gradients take smaller steps; parameters with small gradients take relatively larger ones.

**Adam** (adaptive moment estimation) combines momentum (first moment m) with RMSProp (second moment v), plus **bias correction** because both start at zero:

$$ m ← β₁ m + (1 − β₁) g          v ← β₂ v + (1 − β₂) g² $$
$$ m̂ = m / (1 − β₁ᵗ)          v̂ = v / (1 − β₂ᵗ) $$
$$ θ ← θ − η m̂ / (√v̂ + ε) $$
:::where
g — the current gradient, per parameter
m — the first moment: a running average of gradients (momentum)
v — the second moment: a running average of squared gradients
β₁, β₂ — the decay rates of m and v (defaults 0.9 and 0.999)
t — the step number, starting at 1
β₁ᵗ — β₁ raised to the power t (here the superscript really is a power)
m̂, v̂ — "m hat, v hat": bias-corrected moments; dividing by (1 − βᵗ) removes the pull towards their zero starting values
ε — a tiny constant that prevents division by zero
η — the learning rate
:::

Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e-8, and a learning rate around 1e-3 (3e-4 is a popular "safe" value for transformers). Adam's step size is roughly η per parameter regardless of the gradient's scale, which makes it robust and the default choice in deep learning.

**AdamW** applies weight decay *directly* to the weights (θ ← θ − η·λ·θ) instead of adding λθ to the gradient, where Adam's adaptive scaling would distort it. AdamW is the standard optimizer for training transformers and LLMs.` },
      { type: 'code', title: 'SGD vs momentum vs Adam on a ravine', code: `
import numpy as np

def loss(p):  return 0.5 * (p[0] ** 2 + 50 * p[1] ** 2)     # steep in y, gentle in x
def grad(p):  return np.array([p[0], 50 * p[1]])

def run(update, steps=100):
    p, state = np.array([-10.0, 1.0]), {}
    for t in range(1, steps + 1):
        p = update(p, grad(p), state, t)
    return loss(p)

def sgd(p, g, s, t, lr=0.03):
    return p - lr * g

def momentum(p, g, s, t, lr=0.03, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g
    return p - lr * s["v"]

def adam(p, g, s, t, lr=0.5, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    m_hat, v_hat = s["m"] / (1 - b1 ** t), s["v"] / (1 - b2 ** t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps)

for name, fn in [("SGD", sgd), ("momentum", momentum), ("Adam", adam)]:
    print(f"{name:<9} loss after 100 steps: {run(fn):.6f}")
`, after: `SGD's learning rate is capped by the steep direction (above 0.04 it diverges), so it barely moves along the gentle one. Momentum accumulates speed along the valley; Adam rescales each coordinate.` },
      { type: 'chart', title: "The three optimizers' paths through the ravine", md: `The same 100 steps, drawn in parameter space, starting from (−10, 1):`, after: `SGD bounces across the ravine for a few steps, settles on the valley floor, then shrinks x by only 3% per step: after 100 steps it is still at x ≈ −0.48, with a loss about 300 times higher than momentum's. Momentum races along the valley but keeps swinging from wall to wall, and overshoots the minimum by more than 2 before settling: the velocity it builds up is both its strength and its weakness. Adam takes steps of roughly equal size in both coordinates, so it marches along x at a steady pace, swaying gently across the valley, and reaches the minimum's neighbourhood in about 20 steps.`, chart: {"kind":"line","points":false,"series":[{"name":"loss contours","x":[1,0.996,0.985,0.966,0.94,0.906,0.866,0.819,0.766,0.707,0.643,0.574,0.5,0.423,0.342,0.259,0.174,0.087,0,-0.087,-0.174,-0.259,-0.342,-0.423,-0.5,-0.574,-0.643,-0.707,-0.766,-0.819,-0.866,-0.906,-0.94,-0.966,-0.985,-0.996,-1,-0.996,-0.985,-0.966,-0.94,-0.906,-0.866,-0.819,-0.766,-0.707,-0.643,-0.574,-0.5,-0.423,-0.342,-0.259,-0.174,-0.087,0,0.087,0.174,0.259,0.342,0.423,0.5,0.574,0.643,0.707,0.766,0.819,0.866,0.906,0.94,0.966,0.985,0.996,1,null,3.162,3.15,3.114,3.055,2.972,2.866,2.739,2.59,2.422,2.236,2.033,1.814,1.581,1.336,1.082,0.818,0.549,0.276,0,-0.276,-0.549,-0.818,-1.082,-1.336,-1.581,-1.814,-2.033,-2.236,-2.422,-2.59,-2.739,-2.866,-2.972,-3.055,-3.114,-3.15,-3.162,-3.15,-3.114,-3.055,-2.972,-2.866,-2.739,-2.59,-2.422,-2.236,-2.033,-1.814,-1.581,-1.336,-1.082,-0.818,-0.549,-0.276,0,0.276,0.549,0.818,1.082,1.336,1.581,1.814,2.033,2.236,2.422,2.59,2.739,2.866,2.972,3.055,3.114,3.15,3.162,null,6.325,6.3,6.228,6.109,5.943,5.732,5.477,5.181,4.845,4.472,4.065,3.628,3.162,2.673,2.163,1.637,1.098,0.551,0,-0.551,-1.098,-1.637,-2.163,-2.673,-3.162,-3.628,-4.065,-4.472,-4.845,-5.181,-5.477,-5.732,-5.943,-6.109,-6.228,-6.3,-6.325,-6.3,-6.228,-6.109,-5.943,-5.732,-5.477,-5.181,-4.845,-4.472,-4.065,-3.628,-3.162,-2.673,-2.163,-1.637,-1.098,-0.551,0,0.551,1.098,1.637,2.163,2.673,3.162,3.628,4.065,4.472,4.845,5.181,5.477,5.732,5.943,6.109,6.228,6.3,6.325,null,10,9.962,9.848,9.659,9.397,9.063,8.66,8.192,7.66,7.071,6.428,5.736,5,4.226,3.42,2.588,1.736,0.872,0,-0.872,-1.736,-2.588,-3.42,-4.226,-5,-5.736,-6.428,-7.071,-7.66,-8.192,-8.66,-9.063,-9.397,-9.659,-9.848,-9.962,-10,-9.962,-9.848,-9.659,-9.397,-9.063,-8.66,-8.192,-7.66,-7.071,-6.428,-5.736,-5,-4.226,-3.42,-2.588,-1.736,-0.872,0,0.872,1.736,2.588,3.42,4.226,5,5.736,6.428,7.071,7.66,8.192,8.66,9.063,9.397,9.659,9.848,9.962,10,null],"y":[0,0.012,0.025,0.037,0.048,0.06,0.071,0.081,0.091,0.1,0.108,0.116,0.122,0.128,0.133,0.137,0.139,0.141,0.141,0.141,0.139,0.137,0.133,0.128,0.122,0.116,0.108,0.1,0.091,0.081,0.071,0.06,0.048,0.037,0.025,0.012,0,-0.012,-0.025,-0.037,-0.048,-0.06,-0.071,-0.081,-0.091,-0.1,-0.108,-0.116,-0.122,-0.128,-0.133,-0.137,-0.139,-0.141,-0.141,-0.141,-0.139,-0.137,-0.133,-0.128,-0.122,-0.116,-0.108,-0.1,-0.091,-0.081,-0.071,-0.06,-0.048,-0.037,-0.025,-0.012,0,null,0,0.039,0.078,0.116,0.153,0.189,0.224,0.257,0.287,0.316,0.343,0.366,0.387,0.405,0.42,0.432,0.44,0.446,0.447,0.446,0.44,0.432,0.42,0.405,0.387,0.366,0.343,0.316,0.287,0.257,0.224,0.189,0.153,0.116,0.078,0.039,0,-0.039,-0.078,-0.116,-0.153,-0.189,-0.224,-0.257,-0.287,-0.316,-0.343,-0.366,-0.387,-0.405,-0.42,-0.432,-0.44,-0.446,-0.447,-0.446,-0.44,-0.432,-0.42,-0.405,-0.387,-0.366,-0.343,-0.316,-0.287,-0.257,-0.224,-0.189,-0.153,-0.116,-0.078,-0.039,0,null,0,0.078,0.155,0.231,0.306,0.378,0.447,0.513,0.575,0.632,0.685,0.733,0.775,0.811,0.84,0.864,0.881,0.891,0.894,0.891,0.881,0.864,0.84,0.811,0.775,0.733,0.685,0.632,0.575,0.513,0.447,0.378,0.306,0.231,0.155,0.078,0,-0.078,-0.155,-0.231,-0.306,-0.378,-0.447,-0.513,-0.575,-0.632,-0.685,-0.733,-0.775,-0.811,-0.84,-0.864,-0.881,-0.891,-0.894,-0.891,-0.881,-0.864,-0.84,-0.811,-0.775,-0.733,-0.685,-0.632,-0.575,-0.513,-0.447,-0.378,-0.306,-0.231,-0.155,-0.078,0,null,0,0.123,0.246,0.366,0.484,0.598,0.707,0.811,0.909,1,1.083,1.158,1.225,1.282,1.329,1.366,1.393,1.409,1.414,1.409,1.393,1.366,1.329,1.282,1.225,1.158,1.083,1,0.909,0.811,0.707,0.598,0.484,0.366,0.246,0.123,0,-0.123,-0.246,-0.366,-0.484,-0.598,-0.707,-0.811,-0.909,-1,-1.083,-1.158,-1.225,-1.282,-1.329,-1.366,-1.393,-1.409,-1.414,-1.409,-1.393,-1.366,-1.329,-1.282,-1.225,-1.158,-1.083,-1,-0.909,-0.811,-0.707,-0.598,-0.484,-0.366,-0.246,-0.123,0,null],"dash":true},{"name":"SGD (loss after 100 steps 0.11)","x":[-10,-9.7,-9.409,-9.127,-8.853,-8.587,-8.33,-8.08,-7.837,-7.602,-7.374,-7.153,-6.938,-6.73,-6.528,-6.333,-6.143,-5.958,-5.78,-5.606,-5.438,-5.275,-5.117,-4.963,-4.814,-4.67,-4.53,-4.394,-4.262,-4.134,-4.01,-3.89,-3.773,-3.66,-3.55,-3.444,-3.34,-3.24,-3.143,-3.049,-2.957,-2.868,-2.782,-2.699,-2.618,-2.539,-2.463,-2.389,-2.318,-2.248,-2.181,-2.115,-2.052,-1.99,-1.931,-1.873,-1.816,-1.762,-1.709,-1.658,-1.608,-1.56,-1.513,-1.468,-1.424,-1.381,-1.339,-1.299,-1.26,-1.223,-1.186,-1.15,-1.116,-1.082,-1.05,-1.018,-0.988,-0.958,-0.929,-0.902,-0.874,-0.848,-0.823,-0.798,-0.774,-0.751,-0.728,-0.707,-0.685,-0.665,-0.645,-0.626,-0.607,-0.589,-0.571,-0.554,-0.537,-0.521,-0.505,-0.49,-0.476],"y":[1,-0.5,0.25,-0.125,0.062,-0.031,0.016,-0.008,0.004,-0.002,0.001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"name":"momentum (loss after 100 steps 0.00034)","x":[-10,-9.7,-9.139,-8.36,-7.408,-6.329,-5.168,-3.968,-2.769,-1.607,-0.513,0.487,1.373,2.129,2.745,3.217,3.546,3.735,3.793,3.732,3.565,3.308,2.977,2.59,2.164,1.715,1.26,0.813,0.386,-0.01,-0.366,-0.675,-0.933,-1.138,-1.288,-1.384,-1.429,-1.427,-1.382,-1.3,-1.187,-1.05,-0.895,-0.729,-0.558,-0.387,-0.221,-0.066,0.076,0.202,0.309,0.396,0.462,0.508,0.534,0.542,0.532,0.507,0.47,0.422,0.367,0.306,0.242,0.177,0.113,0.052,-0.004,-0.055,-0.098,-0.135,-0.164,-0.185,-0.198,-0.204,-0.203,-0.197,-0.185,-0.169,-0.149,-0.127,-0.103,-0.078,-0.054,-0.03,-0.008,0.012,0.03,0.045,0.057,0.066,0.073,0.076,0.077,0.076,0.072,0.067,0.06,0.052,0.043,0.034,0.025],"y":[1,-0.5,-1.1,0.01,0.994,0.389,-0.739,-0.645,0.407,0.744,-0.069,-0.697,-0.217,0.54,0.411,-0.322,-0.499,0.09,0.485,0.113,-0.391,-0.258,0.249,0.332,-0.091,-0.335,-0.052,0.281,0.159,-0.189,-0.219,0.083,0.23,0.018,-0.2,-0.096,0.142,0.143,-0.07,-0.157,0.001,0.141,0.056,-0.105,-0.092,0.057,0.106,-0.009,-0.099,-0.031,0.077,0.059,-0.045,-0.071,0.012,0.069,0.016,-0.056,-0.037,0.035,0.047,-0.013,-0.048,-0.008,0.04,0.023,-0.027,-0.031,0.012,0.033,0.003,-0.028,-0.014,0.02,0.02,-0.01,-0.022,0,0.02,0.008,-0.015,-0.013,0.008,0.015,-0.001,-0.014,-0.005,0.011,0.008,-0.006,-0.01,0.002,0.01,0.002,-0.008,-0.005,0.005,0.007,-0.002,-0.007,-0.001]},{"name":"Adam (loss after 100 steps 0.0014)","x":[-10,-9.5,-9.001,-8.503,-8.008,-7.515,-7.026,-6.541,-6.062,-5.589,-5.123,-4.666,-4.218,-3.78,-3.354,-2.942,-2.543,-2.16,-1.793,-1.443,-1.112,-0.801,-0.51,-0.24,0.008,0.234,0.437,0.618,0.776,0.913,1.027,1.12,1.193,1.246,1.281,1.298,1.299,1.286,1.258,1.219,1.169,1.11,1.042,0.969,0.89,0.807,0.722,0.635,0.549,0.463,0.379,0.298,0.22,0.146,0.077,0.014,-0.044,-0.096,-0.142,-0.181,-0.215,-0.242,-0.263,-0.279,-0.288,-0.293,-0.293,-0.289,-0.28,-0.268,-0.254,-0.236,-0.217,-0.196,-0.174,-0.151,-0.128,-0.105,-0.082,-0.061,-0.04,-0.021,-0.003,0.014,0.028,0.041,0.052,0.06,0.067,0.073,0.076,0.078,0.078,0.077,0.074,0.071,0.066,0.061,0.055,0.049,0.042],"y":[1,0.5,0.034,-0.336,-0.542,-0.585,-0.511,-0.364,-0.179,0.012,0.18,0.303,0.365,0.365,0.313,0.222,0.108,-0.01,-0.116,-0.196,-0.239,-0.242,-0.21,-0.15,-0.073,0.008,0.081,0.134,0.162,0.162,0.136,0.091,0.035,-0.022,-0.07,-0.101,-0.113,-0.105,-0.079,-0.042,-0.001,0.037,0.065,0.078,0.076,0.06,0.034,0.004,-0.024,-0.045,-0.055,-0.053,-0.041,-0.022,0,0.019,0.034,0.039,0.036,0.026,0.011,-0.005,-0.018,-0.026,-0.028,-0.023,-0.014,-0.002,0.009,0.017,0.02,0.019,0.013,0.005,-0.004,-0.011,-0.015,-0.014,-0.011,-0.005,0.002,0.007,0.01,0.011,0.008,0.004,-0.001,-0.005,-0.008,-0.008,-0.006,-0.003,0.001,0.004,0.006,0.006,0.004,0.002,-0.001,-0.003,-0.004]}],"xLabel":"x (the gentle direction)","yLabel":"y (the steep direction)","xMin":-11,"xMax":3,"yMin":-1.3,"yMax":1.3} },
      { type: 'check', question: { type: 'mcq', q: 'What does the momentum term mainly achieve?', options: ['It makes every step smaller', 'It averages past gradients so consistent directions accelerate and oscillating ones cancel', 'It adds weight decay', 'It removes the need for a learning rate'], answer: 1, explain: 'The velocity is an exponentially weighted sum of gradients.' } },
      { type: 'text', title: "What optimizers cost, and what came after Adam", md: `Optimizers keep **state**, and in large models that state dominates memory. Per parameter, mixed-precision training with Adam typically stores the weight, its gradient, a full-precision master copy of the weight, and Adam's two moving averages:

$$ bytes per parameter ≈ 2 (bf16 weight) + 2 (bf16 gradient) + 4 (fp32 master weight) + 4 + 4 (Adam's m and v) = 16 $$
:::where
bf16 — 16-bit "brain float" numbers, 2 bytes each, used in the forward and backward passes
fp32 master weight — a 4-byte copy of each weight that receives the updates, so that small updates are not rounded away
m, v — Adam's moving averages of the gradient and of the squared gradient, 4 bytes each
:::

A 7-billion-parameter model therefore needs about 112 GB for weights, gradients and optimizer state before a single activation is stored, which is why large models are trained on many GPUs with this state split between them (see [Training at Scale](#/concept/training-at-scale)).

The search for better optimizers continues:
- **Adafactor** stores factored second moments (one average per row and per column of each weight matrix) to save memory;
- **Lion** keeps only a momentum and steps by its *sign*, so every parameter moves by the same amount;
- **Shampoo** and **Muon** precondition each weight matrix as a whole, using its structure instead of treating every parameter independently; Muon has been used to train recent large models with reported savings in compute.

For most projects, AdamW with a warm-up and a cosine or linear decay remains the sensible default.` },
      { type: 'code', title: "Adam ignores the scale of the gradient", code: `import numpy as np
g = np.array([0.002, -0.5, 3.0])                   # three parameters with very different gradient sizes

def adam_step(g, steps=3, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    m, v = np.zeros_like(g), np.zeros_like(g)
    for t in range(1, steps + 1):                   # a constant gradient for a few steps
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return step

print("SGD step (lr 0.01):          ", [f"{v:+.5f}" for v in 0.01 * g])
print("Adam step:                   ", [f"{v:+.5f}" for v in adam_step(g)])
print("Adam step, gradients x 1000: ", [f"{v:+.5f}" for v in adam_step(1000 * g)])
print("Adam is scale invariant:", np.allclose(adam_step(g), adam_step(1000 * g)))`, after: `With a steady gradient, Adam moves every parameter by about the learning rate whatever the size of its gradient: the 0.002 gradient and the 3.0 gradient both give a step of 0.01, while SGD's steps differ by a factor of 1,500. Multiplying all gradients by 1,000, as rescaling the loss would, leaves Adam's steps unchanged: what matters to Adam is the direction and consistency of each gradient, not its size.`, predict: {"type":"mcq","q":"Every gradient is multiplied by 1,000 (for example because the loss was scaled up). How do Adam's updates change?","options":["They become 1,000 times larger","They become about 31 times larger","They stay essentially the same","They become 1,000 times smaller"],"answer":2,"explain":"Adam divides the averaged gradient by the square root of the averaged squared gradient, so a common scale factor cancels out (apart from the tiny ε).","expect":["Adam is scale invariant: True"]} },
      { type: 'text', title: 'Learning-rate schedules', md: `The best learning rate changes during training: early on, large updates make fast progress; later, smaller ones settle into a good minimum. Common schedules:

- **Step decay** — divide the rate by 10 at fixed epochs (classic for CNNs).
- **Cosine decay** — smoothly lower the rate along half a cosine wave from the peak to a minimum.
- **Warmup** — start from a tiny rate and increase it linearly over the first few hundred or thousand steps. Adam's statistics are unreliable at the start and transformers are fragile early on; warmup prevents early divergence.

**Warmup + cosine decay** is the standard recipe for training transformers and LLMs.

$$ warmup: lr = lr_max · (step + 1) / warmup_steps $$
$$ cosine: lr = lr_min + ½ (lr_max − lr_min) (1 + cos(π · progress)) $$
:::where
lr — the learning rate used at the current step
lr_max — the peak learning rate, reached at the end of warmup
step — the current training step, counting from 0
warmup_steps — how many steps the warmup lasts
lr_min — the final, smallest learning rate
progress — the fraction of the post-warmup training completed, from 0 to 1
cos(π · progress) — goes from 1 down to −1, so the learning rate glides smoothly from lr_max down to lr_min
:::` },
      { type: 'demo', title: 'Interactive: shape a learning-rate schedule', demo: 'lr-schedule' },
      { type: 'text', title: 'Gradient clipping', md: `A single bad batch can produce a huge gradient that throws the parameters far away (the loss spikes or becomes NaN). **Gradient clipping** rescales the whole gradient vector whenever its norm exceeds a threshold c:

$$ if ‖g‖ > c:  g ← g · c / ‖g‖ $$
:::where
g — the gradient of all the parameters together, viewed as one long vector
‖g‖ — its length (L2 norm)
c — the maximum allowed norm, for example 1.0
g · c / ‖g‖ — the same direction, rescaled so that its length is exactly c
:::

The direction is kept; only the size is capped. It is standard for RNNs and transformers (a max norm of 1.0 is a common default). In PyTorch: \`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\` right before \`optimizer.step()\`.` },
      { type: 'check', question: { type: 'tf', q: 'Warmup means starting training with a small learning rate and increasing it over the first steps.', answer: true, explain: 'It stabilizes the early phase, when optimizer statistics are poor and the model is fragile.' } }
    ],
    when: `- **AdamW** with warmup + cosine decay: the default for transformers and most deep learning.
- **SGD with momentum** (often with step or cosine schedules): still excellent for CNNs, sometimes generalizing slightly better than Adam.
- Always consider **gradient clipping** for RNNs, transformers and any training with occasional loss spikes.
- Tune the **learning rate** first — it matters more than the choice between reasonable optimizers.`,
    pitfalls: `- Using Adam's default learning rate (1e-3) for fine-tuning large pretrained models — usually far too high (1e-5 to 1e-4 is typical).
- Adding L2 regularization to the loss with Adam and expecting weight-decay behaviour — use AdamW.
- Forgetting warmup for transformers (early divergence).
- Forgetting the optimizer state when saving/resuming checkpoints.
- Memory: Adam keeps two extra tensors the size of the model.`,
    takeaways: [
      'Momentum: v ← βv + g; θ ← θ − ηv. Accelerates consistent directions, damps oscillations.',
      'RMSProp/Adam scale each parameter\'s step by the root of its average squared gradient.',
      'Adam = momentum + RMSProp + bias correction; AdamW decouples weight decay — the LLM standard.',
      'Warmup + cosine decay is the standard schedule for transformers.',
      'Gradient clipping caps the gradient norm to prevent destructive updates.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why does Adam use bias correction (dividing by 1 − βᵗ)?', options: ['To add regularization', 'Because m and v start at zero and are biased toward zero in early steps', 'To make the learning rate decay', 'To clip gradients'], answer: 1, explain: 'Without correction the first moment estimates would be far too small at the start.' },
      { type: 'mcq', q: 'Which optimizer is the de-facto standard for training large transformers?', options: ['Plain SGD', 'AdamW', 'Newton\'s method', 'k-means'], answer: 1, explain: 'AdamW with warmup and cosine decay is used by most LLM training recipes.' },
      { type: 'fill', q: 'With momentum β = 0.9, v starts at 0 and the gradient is a constant 1. What is v after two steps?', answer: ['1.9'], explain: 'Step 1: v = 0.9·0 + 1 = 1. Step 2: v = 0.9·1 + 1 = 1.9.' },
      { type: 'mcq', q: 'What does gradient clipping by norm preserve?', options: ['The gradient\'s magnitude', 'The gradient\'s direction', 'The learning rate', 'The loss value'], answer: 1, explain: 'It rescales the vector to the threshold length but keeps its direction.' },
      { type: 'tf', q: 'In AdamW, weight decay is applied directly to the parameters rather than being added to the gradient.', answer: true, explain: 'Decoupling prevents Adam\'s per-parameter scaling from distorting the decay.' },
      {"type":"fill","q":"Mixed-precision Adam training needs about 16 bytes per parameter for weights, gradients and optimizer state. About how many gigabytes is that for a 1-billion-parameter model?","answer":["16","16 GB","16GB"],"explain":"1 billion × 16 bytes = 16 GB, before counting activations."},
      {"type":"mcq","q":"Why does plain SGD crawl along the floor of a narrow ravine?","options":["The gradient is zero there","One global learning rate is capped by the steep direction, which makes the steps along the gentle direction tiny","SGD cannot handle negative gradients","The ravine has no minimum"],"answer":1,"explain":"A learning rate large enough to move quickly along the gentle direction would make it diverge across the steep one. Momentum and per-parameter scaling (Adam) fix this."}
    ],
    problems: [
      {
        id: 'momentum-step', title: 'SGD with Momentum', difficulty: 'easy',
        prompt: `Implement one step of SGD with momentum:

\`momentum_step(w, v, grad, lr=0.01, beta=0.9)\` → \`(w_new, v_new)\` where

$$ v_new = β·v + grad          w_new = w − lr·v_new $$
:::where
w — the current weights
v — the current velocity (the running sum of past gradients)
grad — the gradient at w
β — beta, how much of the velocity is kept (0.9 by default)
lr — the learning rate
:::

All arrays are NumPy arrays of the same shape. Return new arrays (don't modify the inputs).

\`\`\`text
momentum_step(np.array([1.0]), np.array([0.0]), np.array([2.0]), lr=0.1, beta=0.9)  ->  ([0.8], [2.0])
\`\`\``,
        starter: `
import numpy as np

def momentum_step(w, v, grad, lr=0.01, beta=0.9):
    # TODO
    pass
`,
        solution: `
import numpy as np

def momentum_step(w, v, grad, lr=0.01, beta=0.9):
    v_new = beta * np.asarray(v, dtype=float) + np.asarray(grad, dtype=float)
    w_new = np.asarray(w, dtype=float) - lr * v_new
    return w_new, v_new
`,
        hints: ['Update the velocity first, then use the new velocity for the weights.'],
        explain: `The velocity is an exponentially weighted sum of past gradients: with β = 0.9 it remembers roughly the last 10 steps. A constant gradient g makes v approach g/(1 − β) = 10g — momentum effectively multiplies the learning rate along consistent directions.`,
        tests: [
          { call: 'momentum_step(np.array([1.0]), np.array([0.0]), np.array([2.0]), lr=0.1, beta=0.9)', expect: '(np.array([0.8]), np.array([2.0]))' },
          { name: 'second step uses the velocity', call: 'momentum_step(np.array([0.8]), np.array([2.0]), np.array([2.0]), lr=0.1, beta=0.9)', expect: '(np.array([0.42]), np.array([3.8]))' },
          { name: 'random vectors', setup: 'rng = np.random.default_rng(1)\nw = rng.normal(size=5); v = rng.normal(size=5); g = rng.normal(size=5)', call: 'momentum_step(w, v, g, lr=0.05, beta=0.8)', hidden: true },
          { name: 'inputs unchanged', code: 'w = np.array([1.0, 2.0]); v = np.array([0.5, 0.5])\nmomentum_step(w, v, np.array([1.0, 1.0]))\nassert np.allclose(w, [1.0, 2.0]) and np.allclose(v, [0.5, 0.5])', hidden: true }
        ]
      },
      {
        id: 'adam-step', title: 'The Adam Update', difficulty: 'medium',
        prompt: `Implement one Adam step:

\`adam_step(w, m, v, grad, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)\` → \`(w_new, m_new, v_new)\`

\`\`\`text
m_new = beta1 * m + (1 - beta1) * grad
v_new = beta2 * v + (1 - beta2) * grad**2
m_hat = m_new / (1 - beta1**t)
v_hat = v_new / (1 - beta2**t)
w_new = w - lr * m_hat / (sqrt(v_hat) + eps)
\`\`\`

\`t\` is the step number, starting at **1**. A remarkable property you can check: on the first step, every parameter moves by almost exactly \`lr\`, whatever the size of its gradient.`,
        starter: `
import numpy as np

def adam_step(w, m, v, grad, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # TODO
    pass
`,
        solution: `
import numpy as np

def adam_step(w, m, v, grad, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    grad = np.asarray(grad, dtype=float)
    m_new = beta1 * np.asarray(m, dtype=float) + (1 - beta1) * grad
    v_new = beta2 * np.asarray(v, dtype=float) + (1 - beta2) * grad ** 2
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    w_new = np.asarray(w, dtype=float) - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w_new, m_new, v_new
`,
        hints: ['Follow the five lines in order; `t` appears only in the bias corrections.', 'Use `np.sqrt` for the element-wise square root.'],
        explain: `On step 1, m̂ = g and v̂ = g², so the update is lr·g/(|g| + ε) ≈ lr·sign(g): the step size no longer depends on the gradient's scale. That scale-invariance is why Adam is so forgiving to tune — and why its learning rate is directly "how far each weight moves per step".`,
        tests: [
          { name: 'first step', call: 'adam_step(np.array([1.0, -2.0]), np.zeros(2), np.zeros(2), np.array([0.5, -30.0]), t=1)' },
          { name: 'first step moves each weight by about lr', code: 'w = np.array([0.0, 0.0, 0.0])\nw2, m, v = adam_step(w, np.zeros(3), np.zeros(3), np.array([1e-4, 1.0, 1e4]), t=1, lr=0.01)\nassert np.allclose(np.abs(w2 - w), 0.01, rtol=1e-3), np.abs(w2 - w)' },
          { name: 'several steps', setup: 'rng = np.random.default_rng(0)\ndef run():\n    w = rng.normal(size=4); m = np.zeros(4); v = np.zeros(4)\n    for t in range(1, 6):\n        g = 2 * w\n        w, m, v = adam_step(w, m, v, g, t, lr=0.1)\n    return w, m, v', call: 'run()', hidden: true }
        ]
      },
      {
        id: 'warmup-cosine', title: 'Warmup + Cosine Learning-Rate Schedule', difficulty: 'easy',
        prompt: `Implement \`lr_at(step, max_lr, warmup_steps, total_steps, min_lr=0.0)\` (steps start at 0):

- **Warmup** — if \`step < warmup_steps\`: \`max_lr * (step + 1) / warmup_steps\`.
- **Cosine decay** — otherwise: \`progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)\`, capped at 1, and
  \`min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))\`.

\`\`\`text
lr_at(0, 1.0, 10, 100)    ->  0.1
lr_at(10, 1.0, 10, 100)   ->  1.0
lr_at(55, 1.0, 10, 100)   ->  0.5
lr_at(100, 1.0, 10, 100)  ->  0.0
\`\`\``,
        starter: `
import math

def lr_at(step, max_lr, warmup_steps, total_steps, min_lr=0.0):
    # TODO
    pass
`,
        solution: `
import math

def lr_at(step, max_lr, warmup_steps, total_steps, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
`,
        hints: ['Two branches: linear ramp, then half a cosine.', 'cos(0) = 1 gives max_lr at the start of decay; cos(π) = −1 gives min_lr at the end.'],
        explain: `This is the schedule used by most LLM training runs (e.g. warmup for a few thousand steps, cosine decay to ~10% of the peak). Capping progress at 1 keeps the rate at min_lr if training continues past total_steps.`,
        tests: [
          { call: 'lr_at(0, 1.0, 10, 100)', expect: '0.1' },
          { call: 'lr_at(10, 1.0, 10, 100)', expect: '1.0' },
          { call: 'lr_at(55, 1.0, 10, 100)', expect: '0.5' },
          { call: 'lr_at(100, 1.0, 10, 100)', expect: '0.0' },
          { name: 'with a floor', call: '[round(lr_at(s, 3e-4, 100, 1000, 3e-5), 12) for s in (0, 99, 100, 550, 1000, 5000)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Kingma & Ba — Adam: A Method for Stochastic Optimization', url: 'https://arxiv.org/abs/1412.6980' },
      { title: 'Loshchilov & Hutter — Decoupled Weight Decay Regularization (AdamW)', url: 'https://arxiv.org/abs/1711.05101' },
      { title: 'Distill — Why Momentum Really Works', url: 'https://distill.pub/2017/momentum/' }
    ]
  },

  // =====================================================================================
  {
    id: 'normalization', branch: 'dl', title: 'Initialization, Normalization & Dropout', short: 'Init, Norm & Dropout', level: 2, minutes: 40, prereqs: ['training-loop', 'statistics'],
    tags: ['xavier', 'he init', 'batch norm', 'layer norm', 'rmsnorm', 'dropout', 'residual'],
    summary: 'Keeping signals healthy in deep networks: variance-preserving weight initialization, batch and layer normalization, dropout, and residual connections.',
    why: `A deep network is a long chain of multiplications: if each layer slightly shrinks or grows the signal, after 50 layers it vanishes or explodes — and so do the gradients. **Initialization** schemes keep the signal's scale steady at the start; **normalization layers** (BatchNorm, LayerNorm, RMSNorm) keep it steady during training; **residual connections** give gradients a highway; and **dropout** regularizes. Every transformer block contains LayerNorm (or RMSNorm), residual connections and, often, dropout — these are not optional details.`,
    objectives: [
      'Explain why weight scale matters and use Xavier and He initialization',
      'Compute batch normalization and layer normalization, and contrast them',
      'Implement inverted dropout and explain train vs eval behaviour',
      'Explain how residual connections help very deep networks'
    ],
    sections: [
      { type: 'text', title: 'Why initialization matters', md: `Each layer computes a sum of n inputs times weights. If the weights have variance σ², the output's variance is roughly n·σ² times the input's. Stack many layers:
- σ² too small → activations shrink toward 0 layer after layer (and so do gradients);
- σ² too large → activations blow up (or saturate tanh/sigmoid units).

**Variance-preserving initializations** choose σ² so the scale stays constant:

| Scheme | Weight variance | For |
| --- | --- | --- |
| **Xavier / Glorot** | 1/fan_in (or 2/(fan_in + fan_out)) | tanh, sigmoid, linear |
| **He / Kaiming** | 2/fan_in | ReLU (which zeroes half the units, hence the 2) |

where fan_in is the number of inputs to a neuron. Biases are usually initialized to 0. Frameworks apply sensible defaults, but custom layers and very deep models need care.` },
      { type: 'code', title: 'Signal scale through 30 ReLU layers', code: `
import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=(512, 256))                     # a batch of inputs

for name, std in [("too small (0.01)", 0.01), ("too large (0.2)", 0.2), ("He: sqrt(2/fan_in)", np.sqrt(2 / 256))]:
    h = x
    for layer in range(30):
        W = rng.normal(0, std, size=(256, 256))
        h = np.maximum(0, h @ W)                    # ReLU layer
    print(f"{name:<22} activation std after 30 layers: {h.std():.3e}")
`, predict: {"type":"mcq","q":"Weights are drawn with a standard deviation of 0.01 in each of 30 ReLU layers of width 256. What happens to the standard deviation of the activations by the last layer?","options":["It stays about the same","It shrinks to almost exactly zero","It grows to billions","It becomes nan"],"answer":1,"explain":"About 5e-29: each layer multiplies the scale by roughly 0.01 × √(256 / 2) ≈ 0.11, and thirty such factors vanish. With 0.2 it explodes (3e10); He initialization, √(2 / fan_in), keeps it near 1.","expect":["too small (0.01)       activation std after 30 layers: 4.916e-29"]} },
      { type: 'text', title: 'Batch normalization', md: `**BatchNorm** (2015) normalizes each feature using statistics of the current mini-batch:

$$ x̂ = (x − μ_batch) / √(σ²_batch + ε)          y = γ · x̂ + β $$
:::where
x — one feature's value for one example in the mini-batch
μ_batch — the mean of that feature over the current mini-batch
σ²_batch — the variance of that feature over the mini-batch
ε — a tiny constant (e.g. 1e-5) that avoids dividing by zero
x̂ — "x hat": the normalized value, with mean 0 and variance 1 across the batch
γ, β — "gamma, beta": a learned scale and shift for each feature (this β is unrelated to momentum's β)
y — the layer's output
:::

γ and β are learned so the layer can still represent any scale and shift. Benefits: faster, more stable training, higher usable learning rates, mild regularization.

Caveats: it depends on the batch — tiny batches give noisy statistics; at inference time it must switch to **running averages** collected during training (forgetting \`model.eval()\` is a classic bug); and it is awkward for variable-length sequences. That is why it is common in CNNs but not in transformers.` },
      { type: 'text', title: 'Layer normalization and RMSNorm', md: `**LayerNorm** normalizes across the **features of each example** instead of across the batch:

$$ μ = mean of x over its d features,   σ² = variance over its d features $$
$$ y = γ · (x − μ) / √(σ² + ε) + β $$
:::where
x — the vector of d features of one example (for instance one token's hidden vector)
μ, σ² — that vector's own mean and variance
ε — a tiny constant for numerical safety
γ, β — a learned scale and shift, each a vector of length d
y — the normalized output, with the same shape as x
:::

It behaves identically in training and inference, works with any batch size (even 1) and any sequence length — so it is used in **every transformer**. Modern LLMs place it *before* each sub-layer ("pre-norm"), which trains more stably.

**RMSNorm** (used by LLaMA and many recent LLMs) drops the mean subtraction and β: y = γ · x / √(mean(x²) + ε) — slightly cheaper, works just as well.` },
      { type: 'chart', title: "Activation size through 30 ReLU layers", md: `The badly scaled network from the first code example, with and without a LayerNorm after every layer:`, after: `With weights that are too small, the activations shrink about tenfold per layer and are numerically dead long before layer 30. Adding a LayerNorm after every layer resets each token's activations to mean 0 and variance 1, so the same bad weights keep a healthy scale all the way down. Normalisation makes training far less sensitive to the initial scale, although a good initialisation (the He line) achieves the same without it.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"yLog":true,"points":false,"series":[{"name":"init std 0.01, no normalisation","y":[0.06362,0.006954,0.0008181,0.00008922,0.00001083,0.000001223,1.579e-7,1.9e-8,2.285e-9,2.675e-10,2.546e-11,2.62e-12,3.113e-13,3.797e-14,3.739e-15,4.53e-16,4.482e-17,5.578e-18,6.688e-19,7.05e-20,8.503e-21,1.002e-21,1.177e-22,1.415e-23,1.449e-24,1.841e-25,1.827e-26,2.362e-27,2.846e-28,3.445e-29]},{"name":"init std 0.01, LayerNorm after every layer","y":[0.7912,0.7921,0.7907,0.7915,0.7902,0.7938,0.7907,0.7919,0.7909,0.7911,0.7931,0.7907,0.7914,0.7905,0.793,0.7919,0.7908,0.7918,0.7916,0.7919,0.7907,0.7919,0.7894,0.7919,0.7908,0.792,0.7923,0.7892,0.7903,0.7913]},{"name":"He init, no normalisation","y":[0.5623,0.5432,0.5649,0.5446,0.5843,0.5831,0.6653,0.7077,0.7524,0.7785,0.655,0.5957,0.6255,0.6746,0.5871,0.6286,0.5498,0.6048,0.641,0.5972,0.6366,0.663,0.6886,0.7316,0.6619,0.7437,0.6523,0.7454,0.7937,0.8492]}],"xLabel":"layer","yLabel":"average |activation| (log scale)"} },
      { type: 'check', question: { type: 'mcq', q: 'Why do transformers use LayerNorm rather than BatchNorm?', options: ['LayerNorm has more parameters', 'LayerNorm normalizes each token\'s features independently of the batch, so it works for any batch size and sequence length and behaves the same at inference', 'BatchNorm cannot be backpropagated', 'LayerNorm is a type of dropout'], answer: 1, explain: 'Per-example statistics avoid batch dependence — ideal for variable-length text and small or single-example batches.' } },
      { type: 'text', title: 'Dropout', md: `**Dropout** randomly sets each activation to zero with probability p during training (p = 0.1–0.5). The network can't rely on any single unit, so it learns redundant, more robust features — like training an ensemble of thinned networks that share weights.

**Inverted dropout** (what frameworks implement) scales the surviving activations by 1/(1 − p) during training, so their expected value is unchanged; at evaluation time dropout is simply turned **off**.

\`\`\`text
training:   mask = random(shape) >= p;   out = x * mask / (1 - p)
evaluation: out = x
\`\`\`

Large LLMs trained on huge datasets often use little or no dropout (each example is seen about once, so overfitting is less of an issue); smaller models and fine-tuning commonly use 0.1.` },
      { type: 'text', title: "BatchNorm at training time and at inference time", md: `During training, BatchNorm normalises with the mean and variance of the **current mini-batch**, and it also keeps **running averages** of those statistics. At inference there may be a single example, so it switches to the running averages instead. That switch is what \`model.train()\` and \`model.eval()\` control, and it is behind three classic problems:

- **Forgetting \`model.eval()\`**: predictions then depend on whatever else is in the batch, and a batch of one gives nonsense.
- **Tiny batches**: with 2 or 4 examples per batch the statistics are so noisy that training suffers; **GroupNorm** or **LayerNorm**, which normalise within each example, avoid the problem.
- **Distribution shift**: running averages computed on the training data may not match new data; recomputing them on a sample of the new data is a cheap fix.

$$ running_mean ← (1 − m) · running_mean + m · μ_batch $$
:::where
running_mean — the exponential moving average of the batch means, used at inference (the variance is tracked the same way)
μ_batch — the mean of the current training batch
m — the momentum of the average, 0.1 by default in PyTorch (the opposite convention to an optimizer's momentum)
:::` },
      { type: 'code', title: "BatchNorm in train and eval mode", code: `import numpy as np
rng = np.random.default_rng(0)

class BatchNorm1d:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.mean, self.var = np.zeros(d), np.ones(d)          # running statistics
        self.m, self.eps, self.training = momentum, eps, True

    def __call__(self, x):
        if self.training:
            mu, var = x.mean(axis=0), x.var(axis=0)            # statistics of this batch
            self.mean = (1 - self.m) * self.mean + self.m * mu
            self.var = (1 - self.m) * self.var + self.m * var
        else:
            mu, var = self.mean, self.var                       # statistics remembered from training
        return (x - mu) / np.sqrt(var + self.eps)

bn = BatchNorm1d(3)
for _ in range(200):                                            # training: batches of 32 with mean 5 and std 2
    bn(rng.normal(5, 2, size=(32, 3)))
print("running mean:", bn.mean.round(2), "  running std:", np.sqrt(bn.var).round(2))

single = np.array([[7.0, 5.0, 3.0]])                            # one example at inference time
saved = bn.mean.copy(), bn.var.copy()
bn.training = True
print("train mode, batch of one:", bn(single).round(2))         # normalised by its own statistics
bn.mean, bn.var = saved                                         # train mode also updated the running averages: undo that
bn.training = False
print("eval mode, batch of one: ", bn(single).round(2))
print("eval uses running statistics:", np.allclose(bn(single), (single - bn.mean) / np.sqrt(bn.var + 1e-5)))`, after: `The running averages settle near the training data's mean (5) and standard deviation (2). In training mode a batch of one is normalised by its own statistics, so every value becomes 0: a useless output. In evaluation mode the same example is normalised with the remembered statistics, and 7 becomes about +1, one standard deviation above the training mean.`, predict: {"type":"mcq","q":"A BatchNorm layer in training mode receives a batch containing a single example. What does it output?","options":["The example, unchanged","All zeros","The example normalised with the running averages","An error"],"answer":1,"explain":"In training mode it uses the batch's own mean and variance. With one example the mean is the example itself, so every value becomes (x − x) / √ε = 0.","expect":["eval uses running statistics: True"]} },
      { type: 'text', title: 'Residual connections', md: `A **residual** (skip) connection adds a block's input to its output:

$$ y = x + F(x) $$
:::where
x — the block's input
F(x) — what the block computes (e.g. two layers, or attention): the learned correction
y — the output, the input plus the correction (so x and F(x) must have the same shape)
:::

Instead of learning a whole transformation, each block learns a *correction* to the identity. In the backward pass, the gradient flows straight through the "+ x" path unchanged, so even networks with hundreds of layers train well. Introduced in **ResNet** (2015), residual connections are now in every transformer: \`x = x + attention(norm(x))\`, then \`x = x + mlp(norm(x))\`.` },
      { type: 'code', title: 'LayerNorm and dropout in NumPy', code: `
import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(5, 3, size=(2, 6))                         # 2 tokens, 6 features each

mu = x.mean(axis=-1, keepdims=True)
var = x.var(axis=-1, keepdims=True)
x_hat = (x - mu) / np.sqrt(var + 1e-5)                    # gamma = 1, beta = 0
print("per-token mean after LayerNorm:", x_hat.mean(axis=-1).round(6))
print("per-token std after LayerNorm: ", x_hat.std(axis=-1).round(4))

p = 0.5
mask = rng.random(x.shape) >= p
dropped = x * mask / (1 - p)
print("kept fraction:", mask.mean(), " mean before/after:", x.mean().round(3), dropped.mean().round(3))
` },
      { type: 'check', question: { type: 'tf', q: 'With inverted dropout, you must remember to multiply activations by (1 − p) at inference time.', answer: false, explain: 'Inverted dropout already scales by 1/(1 − p) during training, so inference uses the activations unchanged.' } }
    ],
    when: `- **He initialization** for ReLU-family networks, **Xavier** for tanh/sigmoid (framework defaults are usually right).
- **BatchNorm** in CNNs with reasonably large batches; **LayerNorm / RMSNorm** in transformers and RNNs.
- **Dropout** (0.1–0.5) when a model overfits; often 0.1 in transformers, less or none for huge pretraining runs.
- **Residual connections** whenever you stack many layers.`,
    pitfalls: `- Forgetting \`model.eval()\` — BatchNorm keeps using batch statistics and dropout keeps dropping at inference.
- All-zero (or identical) weight initialization — every neuron computes the same thing forever (symmetry is never broken).
- Tiny batches with BatchNorm.
- Applying dropout to the inputs of the final classifier too aggressively, or using it at evaluation.`,
    takeaways: [
      'Initialize weights with variance ≈ 1/fan_in (Xavier) or 2/fan_in (He, for ReLU) to keep signals stable.',
      'BatchNorm normalizes each feature over the batch; LayerNorm normalizes each example over its features.',
      'Transformers use LayerNorm or RMSNorm (pre-norm) plus residual connections.',
      'Inverted dropout: zero with probability p, scale by 1/(1 − p) in training, do nothing at eval.',
      'Residual connections y = x + F(x) give gradients a direct path through deep networks.'
    ],
    quiz: [
      { type: 'fill', q: 'He initialization for a layer with fan_in = 512 uses weights with what standard deviation? (decimal, 4 places)', answer: ['0.0625'], explain: '√(2/512) = √(1/256) = 1/16 = 0.0625.' },
      { type: 'mcq', q: 'LayerNorm computes its mean and variance over…', options: ['the whole batch for each feature', 'the features of each individual example', 'the entire dataset', 'the weights'], answer: 1, explain: 'Each example (token) is normalized using its own features.' },
      { type: 'mcq', q: 'What goes wrong if all weights of a hidden layer are initialized to the same value?', options: ['Training is faster', 'All neurons compute identical outputs and receive identical gradients, so they never differentiate', 'Nothing', 'The loss becomes NaN'], answer: 1, explain: 'Random initialization breaks the symmetry between neurons.' },
      { type: 'tf', q: 'Dropout is active only during training.', answer: true, explain: 'At evaluation every unit is used (and with inverted dropout no rescaling is needed).' },
      { type: 'mcq', q: 'How do residual connections help optimization?', options: ['They reduce the number of parameters', 'The identity path passes gradients backward unchanged, mitigating vanishing gradients', 'They replace normalization', 'They add noise'], answer: 1, explain: 'Gradients reach early layers through the skip connections.' },
      {"type":"mcq","q":"What does model.eval() change for a BatchNorm layer?","options":["It freezes only the learned scale γ","It normalises with the running averages collected during training instead of the current batch's statistics","It removes the layer","It doubles the batch size"],"answer":1,"explain":"In evaluation mode BatchNorm uses its running mean and variance, so predictions no longer depend on the other examples in the batch."},
      {"type":"mcq","q":"You can only train with batches of 2 images. Which normalisation is the safer choice?","options":["BatchNorm","GroupNorm or LayerNorm","No normalisation is possible","Dropout"],"answer":1,"explain":"Statistics over 2 examples are extremely noisy. GroupNorm and LayerNorm normalise within each example, so they do not depend on the batch size."}
    ],
    problems: [
      {
        id: 'layer-norm', title: 'Layer Normalization', difficulty: 'medium',
        prompt: `Implement \`layer_norm(x, gamma, beta, eps=1e-5)\`, normalizing over the **last axis** (each row/token independently):

\`\`\`text
mu  = mean of x over the last axis   (keepdims)
var = variance (population) over the last axis   (keepdims)
out = gamma * (x - mu) / sqrt(var + eps) + beta
\`\`\`

\`gamma\` and \`beta\` have the size of the last axis and broadcast. x may be 1-D, 2-D or 3-D (batch, tokens, features).`,
        starter: `
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return np.asarray(gamma) * (x - mu) / np.sqrt(var + eps) + np.asarray(beta)
`,
        hints: ['Use `axis=-1, keepdims=True` for both the mean and the variance.', '`x.var()` is the population variance (ddof=0), which is what LayerNorm uses.'],
        explain: `\`axis=-1\` with \`keepdims=True\` makes the same code work for any number of leading dimensions — exactly how LayerNorm is applied to a (batch, sequence, features) tensor in a transformer.`,
        tests: [
          { call: 'layer_norm(np.array([[1.0, 2.0, 3.0]]), np.ones(3), np.zeros(3))', expect: 'np.array([[-1.0, 0.0, 1.0]]) / np.sqrt(2 / 3 + 1e-5)' },
          { name: 'gamma and beta', call: 'layer_norm(np.array([2.0, 4.0]), np.array([2.0, 3.0]), np.array([0.5, -1.0]))', expect: 'np.array([2.0 * -1 / np.sqrt(1 + 1e-5) + 0.5, 3.0 * 1 / np.sqrt(1 + 1e-5) - 1.0])' },
          { name: '3-D input: every token normalized', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(3, 5, size=(2, 4, 8))', code: 'y = layer_norm(x, np.ones(8), np.zeros(8))\nassert y.shape == x.shape\nassert np.allclose(y.mean(axis=-1), 0, atol=1e-9)\nassert np.allclose(y.std(axis=-1), 1, atol=1e-4)', hidden: true },
          { name: 'random gamma/beta', setup: 'rng = np.random.default_rng(1)\nx = rng.normal(size=(3, 5)); g = rng.normal(size=5); b = rng.normal(size=5)', call: 'layer_norm(x, g, b)', hidden: true }
        ]
      },
      {
        id: 'inverted-dropout', title: 'Inverted Dropout', difficulty: 'easy',
        prompt: `Implement \`dropout(x, p, seed=0, training=True)\`:

- If \`training\` is False **or** \`p == 0\`: return \`x\` unchanged (as a float array).
- Otherwise: \`rng = np.random.default_rng(seed)\`, \`mask = rng.random(x.shape) >= p\`, and return \`x * mask / (1 - p)\`.

\`\`\`text
dropout(np.ones(4), 0.5, seed=0)  ->  values are 0.0 or 2.0
dropout(np.ones(4), 0.5, training=False)  ->  [1., 1., 1., 1.]
\`\`\``,
        starter: `
import numpy as np

def dropout(x, p, seed=0, training=True):
    x = np.asarray(x, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def dropout(x, p, seed=0, training=True):
    x = np.asarray(x, dtype=float)
    if not training or p == 0:
        return x
    rng = np.random.default_rng(seed)
    mask = rng.random(x.shape) >= p
    return x * mask / (1 - p)
`,
        hints: ['Handle the "no dropout" cases first.', 'Scale the kept values by 1/(1 − p) so the expected value is unchanged.'],
        explain: `Because each unit survives with probability 1 − p and is scaled by 1/(1 − p), its expected value is unchanged — so the network sees activations of the same average size in training and evaluation, and evaluation needs no adjustment.`,
        tests: [
          { name: 'seeded mask', call: 'dropout(np.arange(1.0, 9.0), 0.5, seed=0)' },
          { call: 'dropout(np.ones(4), 0.5, training=False)', expect: 'np.ones(4)' },
          { name: 'expected value preserved', code: 'out = dropout(np.ones((200, 200)), 0.3, seed=1)\nassert set(np.unique(out).round(6).tolist()) <= {0.0, round(1 / 0.7, 6)}\nassert abs(out.mean() - 1.0) < 0.02, out.mean()', hidden: true },
          { name: 'p = 0 returns the input', call: 'dropout(np.array([[1.0, -2.0]]), 0.0, seed=5)', expect: 'np.array([[1.0, -2.0]])', hidden: true }
        ]
      },
      {
        id: 'he-init', title: 'He Initialization', difficulty: 'easy',
        prompt: `Implement \`he_init(fan_in, fan_out, seed=0)\` returning a weight matrix of shape (fan_in, fan_out) drawn from a normal distribution with mean 0 and standard deviation √(2 / fan_in), using \`np.random.default_rng(seed).normal(...)\`.

Also implement \`xavier_init(fan_in, fan_out, seed=0)\` with standard deviation √(2 / (fan_in + fan_out)).`,
        starter: `
import numpy as np

def he_init(fan_in, fan_out, seed=0):
    # TODO
    pass

def xavier_init(fan_in, fan_out, seed=0):
    # TODO
    pass
`,
        solution: `
import numpy as np

def he_init(fan_in, fan_out, seed=0):
    return np.random.default_rng(seed).normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))

def xavier_init(fan_in, fan_out, seed=0):
    return np.random.default_rng(seed).normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_in, fan_out))
`,
        hints: ['`rng.normal(loc, scale, size=(rows, cols))` — scale is the standard deviation, not the variance.'],
        explain: `Scaling the standard deviation by 1/√fan_in keeps each neuron's summed input at a similar scale no matter how many inputs it has; the factor 2 compensates for ReLU zeroing half of its inputs.`,
        tests: [
          { name: 'seeded He matrix', call: 'he_init(4, 3, seed=1)' },
          { name: 'statistics', code: 'W = he_init(400, 300, seed=2)\nassert W.shape == (400, 300)\nassert abs(W.std() - np.sqrt(2 / 400)) < 0.002, W.std()\nassert abs(W.mean()) < 0.002', hidden: true },
          { name: 'seeded Xavier matrix', call: 'xavier_init(5, 5, seed=3)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'He et al. — Delving Deep into Rectifiers (He initialization)', url: 'https://arxiv.org/abs/1502.01852' },
      { title: 'Ba et al. — Layer Normalization', url: 'https://arxiv.org/abs/1607.06450' },
      { title: 'He et al. — Deep Residual Learning (ResNet)', url: 'https://arxiv.org/abs/1512.03385' }
    ]
  },

  // =====================================================================================
  {
    id: 'pytorch', branch: 'dl', title: 'PyTorch Fundamentals', short: 'PyTorch', level: 2, minutes: 45, prereqs: ['training-loop', 'optimizers'],
    tags: ['pytorch', 'tensor', 'nn.Module', 'autograd', 'dataloader', 'gpu'],
    summary: 'The framework most modern AI is built in: tensors, autograd, nn.Module, losses and optimizers, the PyTorch training loop, devices, and saving models — mapped onto what you already built in NumPy.',
    why: `Nearly every recent model you have heard of — Llama, Stable Diffusion, Whisper, and most research code — is written in **PyTorch**. Good news: you already understand everything it does. A tensor is a NumPy array that can live on a GPU and remember its computation; \`loss.backward()\` is the autograd engine you built; \`nn.Module\` is the class pattern from the Python lesson; \`optimizer.step()\` is the update rule. This lesson maps your knowledge onto PyTorch's vocabulary so you can read and write real model code.`,
    objectives: [
      'Create and manipulate tensors and move them between CPU and GPU',
      'Use autograd: requires_grad, backward, .grad, no_grad',
      'Define models with nn.Module and use built-in layers, losses and optimizers',
      'Write the standard PyTorch training and evaluation loops, and save/load models'
    ],
    sections: [
      { type: 'text', title: 'From NumPy to PyTorch', md: `| NumPy / what you built | PyTorch |
| --- | --- |
| \`np.array\` | \`torch.tensor\` (same indexing, broadcasting, \`@\`) |
| arrays on CPU | tensors on CPU **or GPU** (\`.to("cuda")\`, \`.to("mps")\` on Apple) |
| hand-written backward | **autograd**: \`loss.backward()\` fills \`.grad\` |
| a class holding W, b | \`nn.Module\` subclass; layers like \`nn.Linear\` |
| softmax + cross-entropy by hand | \`nn.CrossEntropyLoss()\` (takes raw logits) |
| \`w -= lr * grad\` | \`torch.optim.AdamW(...)\`, \`optimizer.step()\` |
| index batches yourself | \`DataLoader(dataset, batch_size, shuffle=True)\` |

PyTorch isn't available in the browser runtime, so the PyTorch examples here are for reading — run them locally (\`pip install torch\`) or in Google Colab. The practice problems build the same pieces in NumPy.` },
      { type: 'code', title: 'Tensors and autograd', lang: 'pytorch', code: `
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape, x.dtype)                 # torch.Size([2, 2]) torch.float32
print(x @ x.T)                          # matrix multiply, just like NumPy
print(x.mean(dim=0))                    # 'dim' instead of 'axis'

device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)                        # move data to the GPU if there is one

w = torch.tensor([0.5, -1.0], requires_grad=True)   # track operations on w
loss = ((x.cpu() @ w) ** 2).mean()
loss.backward()                         # autograd: d(loss)/dw
print(w.grad)                           # gradients accumulate here

with torch.no_grad():                   # turn tracking off (inference, manual updates)
    w -= 0.1 * w.grad
w.grad.zero_()                          # reset before the next backward
` },
      { type: 'text', title: 'nn.Module: models as classes', md: `A model subclasses \`nn.Module\`: create layers in \`__init__\`, describe the computation in \`forward\`. Every parameter of every sub-layer is registered automatically, so \`model.parameters()\` hands them all to the optimizer.

Building blocks you will use constantly: \`nn.Linear(in, out)\`, \`nn.ReLU()\`/\`nn.GELU()\`, \`nn.LayerNorm(d)\`, \`nn.Dropout(p)\`, \`nn.Embedding(vocab, d)\`, \`nn.Conv2d\`, \`nn.Sequential(...)\` to chain layers, and losses such as \`nn.CrossEntropyLoss\`, \`nn.MSELoss\`, \`nn.BCEWithLogitsLoss\`.` },
      { type: 'code', title: 'A model and the standard training loop', lang: 'pytorch', code: `
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, n_classes),           # raw logits, no softmax here
        )

    def forward(self, x):
        return self.net(x)

X = torch.randn(1000, 20); y = (X[:, 0] + X[:, 1] > 0).long()
train_loader = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=64, shuffle=True)
val_X, val_y = X[800:], y[800:]

model = MLP(20, 64, 2)
loss_fn = nn.CrossEntropyLoss()                       # applies log-softmax internally
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(10):
    model.train()                                     # enable dropout (and BatchNorm updates)
    for xb, yb in train_loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()                         # clear old gradients
        loss.backward()                               # backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                              # update parameters
    model.eval()                                      # disable dropout
    with torch.no_grad():
        acc = (model(val_X).argmax(dim=1) == val_y).float().mean().item()
    print(f"epoch {epoch}: last batch loss {loss.item():.3f}, val acc {acc:.3f}")

torch.save(model.state_dict(), "mlp.pt")              # save the weights
model.load_state_dict(torch.load("mlp.pt"))           # ...and load them back
` },
      { type: 'check', question: { type: 'mcq', q: 'Why must you call optimizer.zero_grad() each iteration?', options: ['To reset the learning rate', 'Because PyTorch accumulates gradients in .grad across backward calls', 'To move data to the GPU', 'To enable dropout'], answer: 1, explain: 'Without zeroing, each step would use the sum of all previous gradients.' } },
      { type: 'text', title: 'The same structure in NumPy', md: `Under the hood, a PyTorch layer is an object with parameters, a forward that caches what backward needs, and a backward that stores parameter gradients and returns the gradient for its input. The practice problems ask you to build exactly that:

\`\`\`text
class Linear:            forward(x): cache x, return x @ W + b
                         backward(g): dW = x.T @ g, db = g.sum(0), return g @ W.T
class ReLU:              forward(x): cache mask x > 0
                         backward(g): return g * mask
class Sequential:        forward: run layers in order; backward: run them in reverse
\`\`\`

When that clicks, PyTorch stops being a black box: it is this design plus automatic differentiation, GPU kernels and a huge library of layers.` },
      { type: 'check', question: { type: 'tf', q: 'nn.CrossEntropyLoss expects raw logits, so you should not apply softmax in the model\'s forward before it.', answer: true, explain: 'It applies log-softmax internally in a numerically stable way; adding your own softmax distorts the loss.' } },
      { type: 'text', title: "Shape bugs that do not crash", md: `The most dangerous bugs in deep-learning code raise no error: broadcasting quietly turns a mistake into a valid computation. The classic one: a regression model outputs predictions of shape (N, 1), the targets have shape (N,), and

\`\`\`python
loss = ((pred - target) ** 2).mean()      # (N, 1) - (N,) broadcasts to (N, N)!
\`\`\`

averages over all N × N pairs of predictions and targets. Training runs, the loss goes down a little, and the model learns to predict roughly the average. (PyTorch's \`nn.MSELoss\` warns about this mismatch; a hand-written loss does not.)

Habits that catch shape bugs early:
- **assert shapes** at the boundaries of your modules: \`assert pred.shape == target.shape\`;
- **name the dimensions** in comments, such as \`# (batch, seq, d_model)\`;
- **squeeze deliberately**: \`pred.squeeze(-1)\` or \`target.unsqueeze(-1)\`, never a bare \`squeeze()\`, which also removes a batch dimension of size 1;
- **check the loss at initialisation** and **overfit one batch** (see [Training Neural Networks](#/concept/training-loop)): a shape bug usually makes overfitting impossible.

Other silent killers: \`reshape\` where a \`transpose\` was needed (it regroups the numbers instead of swapping the axes), mixing up \`dim=0\` and \`dim=1\` in a softmax or a mean, and evaluating without \`model.eval()\`.` },
      { type: 'code', title: "The (N, 1) versus (N,) bug", code: `import numpy as np
rng = np.random.default_rng(0)
target = rng.normal(size=100)                    # shape (100,)
pred = target.reshape(-1, 1)                     # a perfect model, but its output has shape (100, 1)

diff = pred - target
print("shape of pred - target:", diff.shape)
print("buggy loss:  ", round(float((diff ** 2).mean()), 3))
print("correct loss:", round(float(((pred.squeeze(-1) - target) ** 2).mean()), 3))`, after: `Even a perfect model gets a large loss, because the broadcast compares every prediction with every target. The buggy loss is about twice the variance of the targets, and minimising it would push every prediction towards their mean.`, predict: {"type":"mcq","q":"pred has shape (100, 1) and target has shape (100,). What is the shape of pred − target?","options":["(100,)","(100, 1)","(100, 100)","An error is raised"],"answer":2,"explain":"Broadcasting aligns the shapes from the right: (100, 1) and (1, 100) stretch to (100, 100).","expect":["shape of pred - target: (100, 100)"]} },
      { type: 'text', title: "Saving, loading and reproducibility", md: `Save **state dicts** (dictionaries of tensors) rather than whole pickled models, which break when the code changes:

\`\`\`python
torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),     # Adam's moving averages: needed to resume training
            "epoch": epoch}, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt", map_location="cpu")  # load onto the CPU, then move where needed
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
model.eval()                                             # before evaluating or serving
\`\`\`

For reproducible runs, seed every source of randomness (\`random.seed\`, NumPy's generator, \`torch.manual_seed\`), seed the data loader's workers, and record the library versions. Bit-for-bit reproducibility on GPUs also needs \`torch.use_deterministic_algorithms(True)\`, at some cost in speed. Even then, results can differ across hardware and versions, so compare runs by their metrics over several seeds rather than by identical numbers.` },
      { type: 'text', title: 'Going further: data loading, devices and debugging', md: `- **Datasets and DataLoaders**: a Dataset defines \`__len__\` and \`__getitem__\`; a DataLoader shuffles, batches and can load in parallel worker processes.
- **Devices and precision**: model and batch must live on the same device (\`.to("cuda")\`); \`torch.autocast\` runs most operations in 16-bit for speed (see *Training at Scale*).
- **Modes**: \`model.train()\` and \`model.eval()\` switch dropout and batch normalization between training and inference behaviour; wrap evaluation in \`torch.no_grad()\` to save memory.
- **Saving**: save \`model.state_dict()\` (just the weights) and load it with \`load_state_dict\`, rather than pickling the whole model object.
- **Debugging**: print shapes on a tiny batch, overfit one batch first, beware of silent broadcasting, and check custom operations with \`torch.autograd.gradcheck\`.
- **Speed**: \`torch.compile(model)\` compiles the model into optimized kernels and often speeds up training and inference.` }
    ],
    when: `- Use PyTorch for research, custom architectures, fine-tuning open models (Hugging Face transformers is built on it) and most deep-learning work.
- Keep NumPy for data wrangling and small experiments; JAX is the main alternative for some research and TPU work.
- Use \`torch.no_grad()\` / \`torch.inference_mode()\` for inference to save memory and time.`,
    pitfalls: `- Forgetting \`optimizer.zero_grad()\`.
- Forgetting \`model.train()\` / \`model.eval()\`.
- Adding softmax before \`CrossEntropyLoss\`.
- Tensors on different devices ("expected all tensors to be on the same device").
- Integer label tensors must be \`long\` for CrossEntropyLoss; inputs \`float32\`.
- Keeping references to losses (e.g. appending \`loss\` instead of \`loss.item()\`) keeps whole graphs in memory.`,
    takeaways: [
      'Tensors = NumPy arrays + GPU support + autograd.',
      'loss.backward() fills .grad; gradients accumulate, so zero them each step.',
      'nn.Module: layers in __init__, computation in forward; parameters registered automatically.',
      'Loop: model.train(); for batches: forward, loss, zero_grad, backward, (clip), step; then model.eval() + no_grad for validation.',
      'Save and load state_dict(); move model and data to the same device.'
    ],
    quiz: [
      { type: 'order', q: 'Order the core lines of a PyTorch training step.', items: ['logits = model(xb)', 'loss = loss_fn(logits, yb)', 'optimizer.zero_grad()', 'loss.backward()', 'optimizer.step()'], explain: 'Forward, loss, clear old gradients, backprop, update. (zero_grad may also come first — it just has to precede backward.)' },
      { type: 'mcq', q: 'What does `model.eval()` change?', options: ['It freezes the weights permanently', 'Layers like dropout and BatchNorm switch to inference behaviour', 'It moves the model to the GPU', 'It computes the validation loss'], answer: 1, explain: 'Dropout turns off and BatchNorm uses running statistics.' },
      { type: 'mcq', q: 'In PyTorch, what is `dim=1` in `x.mean(dim=1)` equivalent to in NumPy?', options: ['keepdims=True', 'axis=1', 'ddof=1', 'out=1'], answer: 1, explain: 'PyTorch calls the axis argument dim.' },
      { type: 'tf', q: 'Wrapping inference code in `torch.no_grad()` saves memory because no computational graph is recorded.', answer: true, explain: 'Without gradient tracking, intermediate values needn\'t be kept for backward.' },
      { type: 'mcq', q: 'Where are the gradients stored after `loss.backward()`?', options: ['In loss.grad', 'In each parameter\'s .grad attribute', 'In the optimizer', 'They are discarded'], answer: 1, explain: 'Each leaf tensor with requires_grad=True gets a .grad.' },
      {"type":"mcq","q":"pred has shape (N, 1) and target has shape (N,). What does ((pred − target) ** 2).mean() compute?","options":["The correct mean squared error","The mean over all N × N pairs of predictions and targets","An error is raised","The squared error of the first example only"],"answer":1,"explain":"Broadcasting stretches the difference to (N, N), so every prediction is compared with every target. Squeeze or reshape so that the shapes match."},
      {"type":"mcq","q":"Why save the optimizer's state dict along with the model's?","options":["It is needed to make predictions","To resume training exactly: Adam's moving averages and step count are part of the training state","It makes the file smaller","It stores the training data"],"answer":1,"explain":"Restarting Adam with empty moving averages changes the effective step sizes, so resumed training would not continue where it left off."}
    ],
    problems: [
      {
        id: 'linear-layer-class', title: 'Build nn.Linear and nn.ReLU in NumPy', difficulty: 'medium',
        prompt: `Build the two most common layers as classes with \`forward\` and \`backward\`:

**\`Linear(in_features, out_features, seed=0)\`**
- \`__init__\`: \`rng = np.random.default_rng(seed)\`; \`self.W = rng.normal(0, np.sqrt(2 / in_features), (in_features, out_features))\`; \`self.b = np.zeros(out_features)\`.
- \`forward(x)\`: store x as \`self.x\`; return \`x @ self.W + self.b\`.
- \`backward(grad_out)\`: set \`self.dW = self.x.T @ grad_out\` and \`self.db = grad_out.sum(axis=0)\`; return \`grad_out @ self.W.T\`.

**\`ReLU()\`**
- \`forward(x)\`: store the mask \`x > 0\`; return \`np.maximum(0, x)\`.
- \`backward(grad_out)\`: return \`grad_out * mask\`.`,
        starter: `
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, seed=0):
        # TODO
        pass

    def forward(self, x):
        # TODO
        pass

    def backward(self, grad_out):
        # TODO
        pass


class ReLU:
    def forward(self, x):
        # TODO
        pass

    def backward(self, grad_out):
        # TODO
        pass
`,
        solution: `
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2 / in_features), (in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = np.asarray(x, dtype=float)
        return self.x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T


class ReLU:
    def forward(self, x):
        self.mask = np.asarray(x) > 0
        return np.maximum(0, x)

    def backward(self, grad_out):
        return grad_out * self.mask
`,
        hints: ['Cache exactly what backward needs: the input for Linear, the mask for ReLU.', 'Shapes: dW is (in, out) = x.T (in, n) @ grad_out (n, out).'],
        explain: `This is the module design of every deep-learning framework: forward caches, backward uses the cache, parameter gradients live on the layer. The hidden test checks your backward against numerical gradients of a scalar loss.`,
        tests: [
          { name: 'initial weights', call: 'Linear(3, 2, seed=4).W', expect: 'np.random.default_rng(4).normal(0, np.sqrt(2 / 3), (3, 2))' },
          { name: 'forward', setup: 'lin = Linear(2, 2, seed=0)\nlin.W = np.array([[1.0, 2.0], [3.0, 4.0]]); lin.b = np.array([0.5, -0.5])', call: 'lin.forward(np.array([[1.0, 1.0]]))', expect: 'np.array([[4.5, 5.5]])' },
          { name: 'backward of Linear', setup: 'lin = Linear(2, 2, seed=0)\nlin.W = np.array([[1.0, 2.0], [3.0, 4.0]])\nlin.forward(np.array([[1.0, 2.0], [0.0, 1.0]]))\ng_in = lin.backward(np.array([[1.0, 0.0], [0.0, 1.0]]))', call: '(g_in, lin.dW, lin.db)', expect: '(np.array([[1.0, 3.0], [2.0, 4.0]]), np.array([[1.0, 0.0], [2.0, 1.0]]), np.array([1.0, 1.0]))' },
          { name: 'ReLU forward/backward', setup: 'r = ReLU()\nout = r.forward(np.array([[-1.0, 2.0, 0.0]]))\ng = r.backward(np.array([[5.0, 5.0, 5.0]]))', call: '(out, g)', expect: '(np.array([[0.0, 2.0, 0.0]]), np.array([[0.0, 5.0, 0.0]]))' },
          { name: 'gradients match numerical ones', setup: 'rng = np.random.default_rng(1)\nx = rng.normal(size=(4, 3))\nlin = Linear(3, 2, seed=7)\nt = rng.normal(size=(4, 2))\ndef loss_of(W):\n    return np.sum((x @ W + lin.b - t) ** 2)', code: 'out = lin.forward(x)\nlin.backward(2 * (out - t))\nnum = np.zeros_like(lin.W)\nfor i in np.ndindex(lin.W.shape):\n    Wp = lin.W.copy(); Wp[i] += 1e-6\n    Wm = lin.W.copy(); Wm[i] -= 1e-6\n    num[i] = (loss_of(Wp) - loss_of(Wm)) / 2e-6\nassert np.allclose(lin.dW, num, atol=1e-5)', hidden: true }
        ]
      },
      {
        id: 'sequential-model', title: 'A Tiny Sequential Framework', difficulty: 'medium',
        prompt: `Using the provided \`Linear\` and \`ReLU\` layers, finish a mini framework:

1. \`Sequential(layers)\`: \`forward(x)\` runs the layers in order; \`backward(grad)\` runs them in **reverse**, passing each layer's returned gradient to the previous one.
2. \`mse_loss(pred, target)\` → \`(loss, grad)\` with \`loss = mean((pred - target)**2)\` and \`grad = 2 * (pred - target) / pred.size\`.
3. \`sgd_step(model, lr)\`: for every \`Linear\` layer in \`model.layers\`: \`W -= lr * dW\`, \`b -= lr * db\`.

The test trains \`Sequential([Linear(1, 16, seed=0), ReLU(), Linear(16, 1, seed=1)])\` on y = 3x − 1 and checks the loss drops a lot.`,
        starter: `
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2 / in_features), (in_features, out_features))
        self.b = np.zeros(out_features)
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, g):
        self.dW, self.db = self.x.T @ g, g.sum(axis=0)
        return g @ self.W.T

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)
    def backward(self, g):
        return g * self.mask

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        # TODO
        pass
    def backward(self, grad):
        # TODO
        pass

def mse_loss(pred, target):
    # TODO
    pass

def sgd_step(model, lr):
    # TODO
    pass
`,
        solution: `
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2 / in_features), (in_features, out_features))
        self.b = np.zeros(out_features)
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, g):
        self.dW, self.db = self.x.T @ g, g.sum(axis=0)
        return g @ self.W.T

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)
    def backward(self, g):
        return g * self.mask

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

def mse_loss(pred, target):
    diff = pred - target
    return float(np.mean(diff ** 2)), 2 * diff / diff.size

def sgd_step(model, lr):
    for layer in model.layers:
        if isinstance(layer, Linear):
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db
`,
        hints: ['Forward: `for layer in self.layers: x = layer.forward(x)`.', 'Backward: iterate `reversed(self.layers)` and keep replacing `grad` with the returned value.', 'Only Linear layers have parameters to update.'],
        explain: `\`Sequential\` composes the chain rule: each backward call turns "gradient w.r.t. my output" into "gradient w.r.t. my input". That's \`nn.Sequential\` + \`loss.backward()\` + \`optimizer.step()\` in about twenty lines.`,
        tests: [
          { name: 'mse_loss', call: 'mse_loss(np.array([[1.0], [3.0]]), np.array([[0.0], [1.0]]))', expect: '(2.5, np.array([[1.0], [2.0]]))' },
          { name: 'training reduces the loss', setup: 'rng = np.random.default_rng(0)\nX = rng.uniform(-1, 1, (64, 1))\ny = 3 * X - 1\ndef run():\n    model = Sequential([Linear(1, 16, seed=0), ReLU(), Linear(16, 1, seed=1)])\n    first = None\n    for step in range(300):\n        loss, g = mse_loss(model.forward(X), y)\n        first = loss if first is None else first\n        model.backward(g)\n        sgd_step(model, 0.05)\n    return first, loss', code: 'first, last = run()\nassert last < 0.05 * first, "loss went from %.4f to %.4f" % (first, last)' },
          { name: 'exact final loss', setup: 'rng = np.random.default_rng(2)\nX = rng.uniform(-1, 1, (32, 2))\ny = (X[:, :1] ** 2 - X[:, 1:])\ndef run():\n    model = Sequential([Linear(2, 8, seed=3), ReLU(), Linear(8, 1, seed=4)])\n    for step in range(50):\n        loss, g = mse_loss(model.forward(X), y)\n        model.backward(g)\n        sgd_step(model, 0.1)\n    return loss', call: 'run()', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'PyTorch — Learn the Basics', url: 'https://pytorch.org/tutorials/beginner/basics/intro.html' },
      { title: "PyTorch documentation", url: "https://pytorch.org/docs/stable/index.html" }
    ]
  },

  // =====================================================================================
  {
    id: 'cnn', branch: 'dl', title: 'Convolutional Neural Networks', short: 'CNNs', level: 2, minutes: 45, prereqs: ['mlp'],
    tags: ['convolution', 'cnn', 'filters', 'pooling', 'stride', 'padding', 'resnet', 'computer vision'],
    summary: 'How networks see: sliding filters that detect local patterns anywhere in an image, stride and padding, pooling, feature hierarchies, and the architectures that launched the deep-learning era.',
    why: `In 2012 a convolutional network (AlexNet) crushed the ImageNet competition and kicked off the modern deep-learning boom. CNNs encode two powerful assumptions about images — **local patterns matter** and **the same pattern can appear anywhere** — which lets them learn from images with far fewer parameters than an MLP. They power photo tagging, medical imaging, self-driving perception and document scanning, and their ideas (weight sharing, hierarchical features, residual connections) carried straight into transformers and diffusion models.`,
    objectives: [
      'Compute a 2-D convolution by hand and explain filters and feature maps',
      'Calculate output sizes from input size, kernel, stride and padding',
      'Explain pooling, channels, receptive fields and the feature hierarchy',
      'Describe a typical CNN architecture and why CNNs need far fewer parameters than MLPs'
    ],
    sections: [
      { type: 'text', title: 'Why not just use an MLP?', md: `A 224×224 colour image has 150,528 numbers. A single dense layer with 1,000 hidden units would need **150 million** weights — and it would have to learn "a cat in the top-left corner" separately from "a cat in the bottom-right".

Images have structure an MLP ignores:
1. **Locality** — nearby pixels are related; useful patterns (edges, corners, textures) are small.
2. **Translation equivariance** — a pattern means the same thing wherever it appears.

A **convolutional layer** bakes both in: small filters look at local patches, and the **same filter weights are reused at every position** (weight sharing).` },
      { type: 'text', title: 'The convolution operation', md: `A **filter** (kernel) is a small grid of weights, e.g. 3×3. Slide it over the image; at each position multiply the overlapping numbers element-wise and add them up (plus a bias). The grid of results is a **feature map** — high where the image locally matches the filter's pattern.

\`\`\`text
image patch      kernel (vertical edge)       output
[0 0 1]          [-1 0 1]
[0 0 1]    ⊙     [-1 0 1]     → sum = 3      (strong edge here)
[0 0 1]          [-1 0 1]
\`\`\`

(Strictly speaking, deep-learning libraries compute *cross-correlation* — the kernel isn't flipped — but everyone calls it convolution.)

**Channels.** A colour image has 3 input channels; a filter spans all of them (3×3×3 weights). A layer usually has many filters — say 64 — producing 64 output channels (feature maps). Its weights form a tensor of shape (64, 3, 3, 3).

**Learned, not designed.** Classical computer vision hand-crafted edge filters; a CNN learns its filters by backpropagation.` },
      { type: 'demo', title: 'Interactive: convolution with different kernels', md: `Toggle pixels and switch kernels. Hover an output cell to see its receptive field and the multiply-and-add behind it.`, demo: 'convolution' },
      { type: 'text', title: 'Stride, padding and output size', md: `- **Stride** s: how far the filter moves each step. Stride 2 halves the resolution.
- **Padding** p: zeros added around the border so the filter can be centred on edge pixels. "Same" padding (p = 1 for a 3×3 kernel with stride 1) keeps the spatial size.

For an input of size n, kernel k, padding p and stride s:

$$ output size = ⌊ (n + 2p − k) / s ⌋ + 1 $$
:::where
n — the input size along one dimension (e.g. the width in pixels)
k — the kernel (filter) size
p — the padding added on each side
s — the stride: how many pixels the kernel moves at each step
⌊ … ⌋ — "floor": round down to a whole number
:::

Examples: 32×32 input, 3×3 kernel, p = 1, s = 1 → 32×32. 32×32, 3×3, p = 0, s = 1 → 30×30. 32×32, 2×2, p = 0, s = 2 → 16×16.` },
      { type: 'check', question: { type: 'fill', q: 'Input 28×28, kernel 5×5, no padding, stride 1. What is the output size (one side)?', answer: ['24', '24x24', '24×24'], explain: '(28 + 0 − 5)/1 + 1 = 24.' } },
      { type: 'text', title: 'Pooling and the feature hierarchy', md: `**Max pooling** (typically 2×2, stride 2) keeps the largest value in each window: it halves the resolution, keeps the strongest responses, and adds a little tolerance to small shifts. Average pooling keeps the mean; **global average pooling** at the end turns each feature map into one number.

Stacking conv → ReLU → (pool) blocks builds a **hierarchy**: early layers detect edges and colours; middle layers, textures and parts (eyes, wheels); late layers, whole objects. Each layer sees a larger region of the original image — its **receptive field** grows with depth.

A classic CNN classifier:

\`\`\`text
[conv 3x3, 32] → ReLU → [conv 3x3, 32] → ReLU → maxpool
[conv 3x3, 64] → ReLU → [conv 3x3, 64] → ReLU → maxpool
global average pool → dense → softmax
\`\`\`` },
      { type: 'code', title: 'Convolution and pooling in NumPy', code: `
import numpy as np

img = np.zeros((6, 6)); img[:, 3:] = 1.0          # a vertical edge: dark left, bright right
kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)

def conv2d(x, k):
    kh, kw = k.shape
    out = np.zeros((x.shape[0] - kh + 1, x.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * k)   # multiply the patch and add up
    return out

fmap = conv2d(img, kernel)
print("feature map (high = edge found):")
print(fmap)
pooled = fmap.reshape(2, 2, 2, 2).max(axis=(1, 3))       # 2x2 max pooling of the 4x4 map
print("after 2x2 max pooling:")
print(pooled)

mlp_params = (224 * 224 * 3) * 1000
conv_params = 64 * 3 * 3 * 3 + 64
print(f"dense layer on a 224x224 RGB image with 1000 units: {mlp_params:,} weights")
print(f"conv layer with 64 filters of 3x3x3:              {conv_params:,} weights")
` },
      { type: 'code', title: 'A small CNN in PyTorch', lang: 'pytorch', code: `
import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 16x16 -> 8x8
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_classes))

    def forward(self, x):                                      # x: (batch, 3, 32, 32)
        return self.head(self.features(x))

model = SmallCNN()
print(model(torch.randn(8, 3, 32, 32)).shape)                 # torch.Size([8, 10])
print(sum(p.numel() for p in model.parameters()), "parameters")
` },
      { type: 'text', title: "Receptive fields and parameter counts", md: `The **receptive field** of a unit is the patch of the input image that can influence it. A 3×3 convolution sees 3×3 pixels; a unit in the second layer sees 3×3 units of the first, which together cover 5×5 pixels; and so on. Downsampling (pooling, or a stride of 2) makes every later step count double:

$$ r_l = r_{l−1} + (k_l − 1) · j_{l−1}          j_l = j_{l−1} · s_l $$
:::where
r_l — the receptive field, in input pixels, after layer l, starting from r₀ = 1
k_l — the kernel size of layer l (3 for a 3×3 convolution, 2 for 2×2 pooling)
s_l — the stride of layer l
j_l — the "jump": how many input pixels one step in layer l's output covers, the product of all strides so far (j₀ = 1)
:::

The number of weights in a convolutional layer does not depend on the image size at all, only on the kernel and the channels:

$$ parameters = k · k · C_in · C_out + C_out $$
:::where
k — the kernel size; C_in — the number of input channels; C_out — the number of filters (output channels)
+ C_out — one bias per filter
:::

That is the payoff of weight sharing: a 3×3 convolution from 64 to 128 channels has 73,856 parameters whether the image is 32×32 or 1,024×1,024, while a dense layer on a 224×224 colour image needs 150,528 weights for every single hidden unit.` },
      { type: 'chart', title: "Receptive field against depth", after: `Stacking 3×3 convolutions alone grows the receptive field by only 2 pixels per layer: 12 layers see a 25×25 patch. Pooling after every second layer doubles the step size each time, so 12 layers see more than the whole 224×224 image. That is how deep layers come to respond to whole objects while each individual filter stays tiny.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12],"series":[{"name":"3×3 convolutions only","y":[3,5,7,9,11,13,15,17,19,21,23,25]},{"name":"3×3 convolutions, 2×2 pooling after every two","y":[3,5,10,14,24,32,52,68,108,140,220,284]}],"xLabel":"number of 3×3 convolution layers","yLabel":"receptive field (pixels, one side)","hlines":[{"y":224,"label":"a 224-pixel image"}]} },
      { type: 'chart', title: "Parameters of convolutional and dense layers", after: `The two convolutions have 1,792 and 73,856 parameters, independent of the image size. VGG-16's first dense layer alone has 103 million, more than all of its 13 convolutional layers together, and a dense layer connected straight to a 224×224 colour image would need 150 million for just 1,000 units. That gap is why modern CNNs end with global average pooling instead of large dense layers.`, chart: {"kind":"bar","x":["conv 3×3, 3→64","conv 3×3, 64→128","dense 7×7×512→4,096","dense 224×224×3→1,000"],"yLog":true,"series":[{"name":"parameters (weights + biases)","y":[1792,73856,102764544,150529000]}],"xLabel":"layer","yLabel":"parameters (log scale)","yMin":1000} },
      { type: 'code', title: "Output sizes, receptive fields and parameters, layer by layer", code: `layers = [("conv 3x3, 3->32", 3, 1, 3, 32), ("conv 3x3, 32->32", 3, 1, 32, 32), ("max-pool 2x2", 2, 2, None, None),
          ("conv 3x3, 32->64", 3, 1, 32, 64), ("conv 3x3, 64->64", 3, 1, 64, 64), ("max-pool 2x2", 2, 2, None, None)]
size, rf, jump, total = 32, 1, 1, 0                         # a 32x32 colour image
print(f"{'layer':<18}{'output':>8}{'receptive field':>17}{'parameters':>12}")
for name, k, s, c_in, c_out in layers:
    pad = (k - 1) // 2 if name.startswith("conv") else 0     # 'same' padding for the convolutions
    size = (size + 2 * pad - k) // s + 1
    rf, jump = rf + (k - 1) * jump, jump * s
    params = k * k * c_in * c_out + c_out if c_in else 0
    total += params
    print(f"{name:<18}{f'{size}x{size}':>8}{f'{rf}x{rf}':>17}{params:>12,}")
print(f"total parameters: {total:,}")
print("weights of a 3x3 conv from 64 to 128 channels (no biases):", 3 * 3 * 64 * 128)`, after: `Each pooling layer halves the resolution and doubles the jump, so the receptive field grows faster after it: from 5×5 after two convolutions to 16×16 after the second pooling layer. The whole network has about 66,000 parameters, fewer than a single dense layer with 22 units on the raw 32×32×3 image.`, predict: {"type":"mcq","q":"A 3×3 convolution maps 64 channels to 128 channels. How many weights does it have, not counting biases?","options":["1,152","8,192","73,728","9,437,184"],"answer":2,"explain":"3 × 3 × 64 × 128 = 73,728, whatever the size of the image.","expect":["(no biases): 73728"]} },
      { type: 'text', title: "Transfer learning in practice", md: `Training a CNN from scratch takes a lot of labelled images. With a few hundred or a few thousand, start from a network **pretrained** on a large dataset such as ImageNet: its early layers already detect edges, textures and shapes that are useful almost everywhere.

| Your data | Strategy |
| --- | --- |
| small, and similar to the pretraining images | **feature extraction**: freeze the backbone and train only a new final layer |
| small, and quite different (X-rays, satellite images) | freeze the early layers and fine-tune the later ones with a small learning rate |
| large | fine-tune everything, still starting from the pretrained weights |

Practical tips: replace the final layer to match your classes; give the pretrained layers a learning rate 10 to 100 times smaller than the new head; use the same preprocessing (resizing and normalisation) the backbone was trained with; and add **data augmentation** (flips, crops, colour changes) matching the variations you expect, so the model learns invariances instead of memorising the few images it has (see [Data-Centric ML](#/concept/data-centric)).` },
      { type: 'text', title: 'Landmark architectures', md: `- **LeNet-5** (1998) — digit recognition for cheques; the original CNN design.
- **AlexNet** (2012) — deep CNN on GPUs with ReLU and dropout; won ImageNet by a huge margin.
- **VGG** (2014) — simple stacks of 3×3 convolutions.
- **ResNet** (2015) — residual connections allowed 50–150+ layers; still a standard backbone.
- **EfficientNet / ConvNeXt** — carefully scaled and modernised CNNs.
- **Vision Transformers** (2020) — split images into patches and use attention instead of convolution; competitive at large scale (see *Vision Transformers*).

In practice you rarely train a CNN from scratch: start from a **pretrained** backbone (on ImageNet) and fine-tune it on your images — **transfer learning** works remarkably well even with a few hundred labelled images.` },
      { type: 'check', question: { type: 'mcq', q: 'What does weight sharing in a convolutional layer mean?', options: ['All layers share one set of weights', 'The same filter weights are applied at every position of the input', 'Weights are copied between models', 'Filters share biases with dense layers'], answer: 1, explain: 'One small filter slides everywhere, so it detects its pattern anywhere with very few parameters.' } }
    ],
    when: `- Images, video frames, spectrograms (audio as images), and other grid-like data with local structure.
- Small/medium image datasets: fine-tune a pretrained CNN (ResNet, EfficientNet, ConvNeXt).
- Very large datasets or multimodal models: vision transformers are competitive or better.
- 1-D convolutions also work well for some time series and sequences.`,
    pitfalls: `- Training from scratch on a small dataset instead of fine-tuning a pretrained network.
- Mismatched preprocessing (normalization statistics, channel order, image size) between training and deployment.
- Ignoring data augmentation (flips, crops, colour jitter), the cheapest regularizer for images.
- Shape confusion: PyTorch uses (batch, channels, height, width).`,
    takeaways: [
      'Convolution slides a small filter over the input, computing local weighted sums → feature maps.',
      'Weight sharing + locality = far fewer parameters and translation equivariance.',
      'Output size = ⌊(n + 2p − k)/s⌋ + 1.',
      'Pooling downsamples; stacked layers build a hierarchy from edges to objects.',
      'Fine-tune pretrained CNNs; residual connections (ResNet) enabled very deep networks.'
    ],
    quiz: [
      { type: 'fill', q: 'Input 64×64, kernel 3×3, padding 1, stride 2. What is the output size (one side)?', answer: ['32', '32x32', '32×32'], explain: '⌊(64 + 2 − 3)/2⌋ + 1 = ⌊63/2⌋ + 1 = 31 + 1 = 32.' },
      { type: 'mcq', q: 'How many weights (excluding biases) does a conv layer with 16 filters of size 3×3 on a 3-channel input have?', options: ['48', '144', '432', '9'], answer: 2, explain: '16 filters × 3 channels × 3 × 3 = 432.' },
      { type: 'mcq', q: 'What does 2×2 max pooling with stride 2 do to a 32×32 feature map?', options: ['Makes it 64×64', 'Makes it 16×16, keeping the max of each 2×2 block', 'Averages all values', 'Nothing'], answer: 1, explain: 'Each non-overlapping 2×2 block becomes one value.' },
      { type: 'tf', q: 'Deeper layers in a CNN have larger receptive fields and tend to detect more complex patterns.', answer: true, explain: 'Each layer combines features from a wider area of the input.' },
      { type: 'mcq', q: 'You have 500 labelled X-ray images. The best starting point is usually…', options: ['Train a big CNN from scratch', 'Fine-tune a CNN pretrained on a large image dataset', 'Use k-means', 'Use an MLP on raw pixels'], answer: 1, explain: 'Transfer learning reuses general visual features learned from millions of images.' },
      {"type":"fill","q":"Two 3×3 convolutions with stride 1 are stacked. What is the receptive field of a unit in the second layer (one side, in pixels)?","answer":["5","5x5","5×5"],"explain":"r₂ = 1 + (3 − 1) + (3 − 1) = 5: two 3×3 layers see a 5×5 patch, like one 5×5 layer but with fewer parameters."},
      {"type":"mcq","q":"Why does the number of parameters of a convolutional layer not depend on the image size?","options":["Images are always resized to 224×224","The same small kernel is reused at every position (weight sharing)","Convolutions have no weights","Pooling removes the extra parameters"],"answer":1,"explain":"A filter's k × k × C_in weights are shared across all positions, so a bigger image only means more positions, not more weights."}
    ],
    problems: [
      {
        id: 'conv-output-size', title: 'Convolution Output Size', difficulty: 'easy',
        prompt: `Implement \`conv_output_size(n, k, stride=1, padding=0)\` = ⌊(n + 2·padding − k) / stride⌋ + 1 (an int).

Then implement \`trace_shapes(n, layers)\`: \`layers\` is a list of \`(k, stride, padding)\` tuples applied in order to a square input of side n; return the list of output sides after each layer.

\`\`\`text
conv_output_size(32, 3, 1, 1)   ->  32
trace_shapes(32, [(3, 1, 1), (2, 2, 0), (3, 1, 0)])  ->  [32, 16, 14]
\`\`\``,
        starter: `
def conv_output_size(n, k, stride=1, padding=0):
    # TODO
    pass

def trace_shapes(n, layers):
    # TODO
    pass
`,
        solution: `
def conv_output_size(n, k, stride=1, padding=0):
    return (n + 2 * padding - k) // stride + 1

def trace_shapes(n, layers):
    sizes = []
    for k, s, p in layers:
        n = conv_output_size(n, k, s, p)
        sizes.append(n)
    return sizes
`,
        hints: ['Use integer division `//` for the floor.', 'Feed each layer\'s output size into the next layer.'],
        explain: `Tracing sizes layer by layer is how you design CNNs and debug "size mismatch" errors when a convolutional stack is flattened into a dense layer.`,
        tests: [
          { call: 'conv_output_size(32, 3, 1, 1)', expect: '32' },
          { call: 'conv_output_size(28, 5)', expect: '24' },
          { call: 'trace_shapes(32, [(3, 1, 1), (2, 2, 0), (3, 1, 0)])', expect: '[32, 16, 14]' },
          { name: 'stride with floor', call: 'conv_output_size(64, 3, 2, 1)', expect: '32', hidden: true },
          { name: 'AlexNet-style first layer', call: 'trace_shapes(227, [(11, 4, 0), (3, 2, 0), (5, 1, 2)])', expect: '[55, 27, 27]', hidden: true }
        ]
      },
      {
        id: 'conv2d', title: '2-D Convolution from Scratch', difficulty: 'medium',
        prompt: `Implement \`conv2d(image, kernel, stride=1)\` — a single-channel "valid" convolution (cross-correlation, no kernel flip, no padding):

- image: (H, W), kernel: (kh, kw).
- Output shape: ((H − kh)//stride + 1, (W − kw)//stride + 1).
- \`out[i, j] = sum(image[i*stride : i*stride + kh, j*stride : j*stride + kw] * kernel)\`.

\`\`\`text
conv2d(np.arange(16.0).reshape(4, 4), np.ones((2, 2)))  ->  3x3 array of 2x2 window sums
\`\`\``,
        starter: `
import numpy as np

def conv2d(image, kernel, stride=1):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def conv2d(image, kernel, stride=1):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    oh = (image.shape[0] - kh) // stride + 1
    ow = (image.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            out[i, j] = np.sum(image[r:r + kh, c:c + kw] * kernel)
    return out
`,
        hints: ['Compute the output shape first with the size formula.', 'Each output cell is the sum of an element-wise product of a window and the kernel.'],
        explain: `Two loops over output positions, one element-wise multiply-and-sum each. Real libraries rearrange the windows into a big matrix ("im2col") and use one matrix multiplication — or specialised GPU kernels — to do all positions at once.`,
        tests: [
          { call: 'conv2d(np.arange(16.0).reshape(4, 4), np.ones((2, 2)))', expect: 'np.array([[10.0, 14.0, 18.0], [26.0, 30.0, 34.0], [42.0, 46.0, 50.0]])' },
          { name: 'vertical edge detector', setup: 'img = np.zeros((5, 5)); img[:, 2:] = 1.0\nk = np.array([[-1.0, 0.0, 1.0]] * 3)', call: 'conv2d(img, k)', expect: 'np.array([[3.0, 3.0, 0.0], [3.0, 3.0, 0.0], [3.0, 3.0, 0.0]])' },
          { name: 'stride 2', call: 'conv2d(np.arange(25.0).reshape(5, 5), np.array([[1.0, 0.0], [0.0, -1.0]]), stride=2)', expect: 'np.array([[-6.0, -6.0], [-6.0, -6.0]])', hidden: true },
          { name: 'random image vs reference', setup: 'rng = np.random.default_rng(0)\nimg = rng.normal(size=(7, 9)); k = rng.normal(size=(3, 2))', call: 'conv2d(img, k)', hidden: true }
        ]
      },
      {
        id: 'max-pool', title: 'Max Pooling', difficulty: 'easy',
        prompt: `Implement \`max_pool2d(x, size=2)\` for a 2-D array: non-overlapping windows of size×size (stride = size), taking the max of each. Ignore leftover rows/columns that don't fill a whole window.

\`\`\`text
max_pool2d(np.array([[1, 3, 2, 0],
                     [4, 2, 1, 5],
                     [0, 1, 7, 2],
                     [3, 2, 1, 1]]))   ->  [[4, 5], [3, 7]]
\`\`\``,
        starter: `
import numpy as np

def max_pool2d(x, size=2):
    x = np.asarray(x, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def max_pool2d(x, size=2):
    x = np.asarray(x, dtype=float)
    H, W = x.shape[0] // size, x.shape[1] // size
    x = x[:H * size, :W * size]
    return x.reshape(H, size, W, size).max(axis=(1, 3))
`,
        hints: ['Crop to a multiple of size first.', 'A neat vectorized trick: `x.reshape(H, size, W, size).max(axis=(1, 3))`.'],
        explain: `Reshaping to (H, size, W, size) groups each window's rows and columns into their own axes, so one \`max\` over axes 1 and 3 pools every window at once — no loops.`,
        tests: [
          { call: 'max_pool2d(np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 7, 2], [3, 2, 1, 1]]))', expect: 'np.array([[4.0, 5.0], [3.0, 7.0]])' },
          { name: 'odd size is cropped', call: 'max_pool2d(np.arange(25.0).reshape(5, 5))', expect: 'np.array([[6.0, 8.0], [16.0, 18.0]])' },
          { name: 'pool size 3', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(size=(9, 6))', call: 'max_pool2d(x, 3)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'CS231n — Convolutional Neural Networks for Visual Recognition', url: 'https://cs231n.github.io/convolutional-networks/' },
      { title: 'Krizhevsky et al. — ImageNet Classification with Deep CNNs (AlexNet)', url: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'rnn', branch: 'dl', title: 'Recurrent Networks & LSTMs', short: 'RNNs & LSTMs', level: 2, minutes: 45, prereqs: ['mlp', 'backpropagation'],
    tags: ['rnn', 'lstm', 'gru', 'sequence', 'bptt', 'vanishing gradient'],
    summary: 'Networks with memory for sequences: the recurrent hidden state, backpropagation through time, vanishing gradients, LSTM gates — and why attention replaced them.',
    why: `Text, speech, sensor readings and prices are **sequences**, where order and context matter. Recurrent neural networks process a sequence one step at a time while carrying a memory (the hidden state). From about 2014 to 2017, LSTMs powered machine translation, speech recognition and text generation. Their limitations — slow sequential processing and difficulty remembering far back — are precisely the problems **attention** and **transformers** solved. Understanding RNNs makes it clear why transformers are designed the way they are, and RNN-like ideas are returning in efficient "state space" models.`,
    objectives: [
      'Write the RNN recurrence and unroll it over time',
      'Explain backpropagation through time and why gradients vanish or explode',
      'Describe the LSTM cell: cell state and forget, input and output gates',
      'Explain the limitations that motivated attention and transformers'
    ],
    sections: [
      { type: 'text', title: 'A network with memory', md: `An RNN reads a sequence x₁, x₂, …, x_T one element at a time and maintains a **hidden state** h_t that summarises everything seen so far:

$$ h_t = tanh( x_t W_x + h_{t−1} W_h + b ) $$
:::where
x_t — the input at time step t (e.g. one word's embedding)
h_t — the hidden state after step t: the network's memory of everything seen so far
h_{t−1} — the previous hidden state (h₀ is usually all zeros)
W_x — the input-to-hidden weights
W_h — the hidden-to-hidden (recurrent) weights
b — the bias
tanh — the hyperbolic tangent, which squashes values into the range (−1, 1)
:::

The **same weights** (W_x, W_h, b) are used at every time step — weight sharing across time, just as a CNN shares weights across space. The hidden state can feed an output at each step (tagging every word) or only at the end (classifying a whole review).

**Unrolling** draws the recurrence as a deep feed-forward network with one layer per time step, all sharing weights:

\`\`\`text
x1 → [cell] → h1 → [cell] → h2 → [cell] → h3 ...
                ↑ x2           ↑ x3
\`\`\`` },
      { type: 'code', title: 'An RNN forward pass in NumPy', code: `
import numpy as np
rng = np.random.default_rng(0)

T, d_in, d_h = 6, 2, 4
X = rng.normal(size=(T, d_in))                 # a sequence of 6 input vectors
Wx = rng.normal(0, 0.5, (d_in, d_h))
Wh = rng.normal(0, 0.5, (d_h, d_h))
b = np.zeros(d_h)

h = np.zeros(d_h)                              # empty memory at the start
for t in range(T):
    h = np.tanh(X[t] @ Wx + h @ Wh + b)        # new memory = f(current input, old memory)
    print(f"t={t}: h = {h.round(3)}")
` },
      { type: 'text', title: 'Backpropagation through time and vanishing gradients', md: `Training unrolls the network over the sequence and runs ordinary backpropagation — **backpropagation through time (BPTT)**. The gradient from step T back to step 1 passes through the recurrence T times, multiplying by W_h (and by tanh′ ≤ 1) at every step.

- If those factors are mostly **smaller than 1**, the gradient shrinks exponentially: the network can't learn dependencies more than ~10–20 steps apart (**vanishing gradients**). In "The cat, which already ate a whole fish this morning, *was* full", connecting *was* to *cat* is hard.
- If they are **larger than 1**, it grows exponentially (**exploding gradients**) — fixed with gradient clipping.

Long sequences are often trained with **truncated BPTT**: backpropagate through only the last k steps.` },
      { type: 'code', title: 'Watching a gradient vanish (or explode)', code: `
import numpy as np
rng = np.random.default_rng(1)
d = 32
for scale in [0.5, 1.0, 1.5]:
    Wh = rng.normal(0, scale / np.sqrt(d), (d, d))
    g = np.ones(d)
    norms = []
    for t in range(50):
        g = Wh.T @ g                     # gradient flowing one step further back in time
        norms.append(np.linalg.norm(g))
    print(f"weight scale {scale}: gradient norm after 10 steps {norms[9]:.2e}, after 50 steps {norms[49]:.2e}")
` },
      { type: 'chart', title: "The gradient's size as it travels back in time", after: `The experiment above, step by step. Each step back multiplies the gradient by the same recurrent matrix, so its size changes geometrically: at scale 0.5 it halves at every step and is negligible after a couple of dozen; at 1.5 it grows by half at every step and explodes; near 1 it changes slowly (here it grows about fifteenfold over 50 steps), a knife-edge that training cannot rely on. (The tanh derivatives, left out here, only shrink it further.)`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],"yLog":true,"points":false,"series":[{"name":"recurrent weight scale 0.5","y":[2.369,0.9765,0.5033,0.2326,0.1144,0.06013,0.02953,0.01366,0.007905,0.004145,0.001917,0.0009345,0.0004775,0.0002225,0.0001214,0.00006427,0.00003124,0.00001398,0.00000732,0.000003702,0.000001928,9.365e-7,4.399e-7,2.289e-7,9.989e-8,5.727e-8,3.01e-8,1.459e-8,6.765e-9,3.404e-9,1.71e-9,9.167e-10,4.564e-10,2.198e-10,1.145e-10,4.814e-11,2.877e-11,1.485e-11,7.411e-12,3.646e-12,1.709e-12,8.985e-13,4.769e-13,2.401e-13,1.226e-13,6.203e-14,2.674e-14,1.597e-14,7.926e-15,4.175e-15]},{"name":"recurrent weight scale 1.0","y":[5.396,6.768,7.737,8.582,10.9,9.931,10.19,9.776,10.97,11,10.8,12.94,11.97,13.44,13.77,15.96,17.35,17.88,16.14,14.63,19.07,19.21,21.08,23.22,24.71,25.36,26.6,27.17,23.52,27.81,31.12,32.51,36.12,38.27,36.57,36.01,42.38,41.33,42.53,49.42,51.19,52.52,57.27,59.43,54.25,62.1,69.84,68.66,75.44,82.46]},{"name":"recurrent weight scale 1.5","y":[7.823,11.41,12.95,16.2,18.24,25.97,36.51,43.26,48.27,64.55,100.6,127.4,173.1,239.9,284.7,371.9,507.2,772.9,1122,1318,1644,1979,2719,3515,4825,5266,7534,9623,14060,19760,27610,37800,57040,68700,91680,118400,162400,227000,339100,486800,745400,913000,1212000,1494000,1989000,2666000,3540000,5071000,7650000,10430000]}],"xLabel":"time steps back","yLabel":"gradient norm (log scale)"} },
      { type: 'check', question: { type: 'mcq', q: 'Why do plain RNNs struggle with long-range dependencies?', options: ['They have too many parameters', 'Gradients are multiplied through every time step and tend to vanish (or explode)', 'They cannot process numbers', 'They forget their weights'], answer: 1, explain: 'Repeated multiplication by the recurrent Jacobian shrinks or grows the gradient exponentially with distance.' } },
      { type: 'text', title: 'LSTMs: gated memory', md: `The **Long Short-Term Memory** cell (1997) adds a separate **cell state** c_t — a conveyor belt of memory updated mostly by *addition* — and three **gates** (sigmoids between 0 and 1) that decide what flows:

- **forget gate** f_t — how much of the old cell state to keep;
- **input gate** i_t — how much of the new candidate g_t to write;
- **output gate** o_t — how much of the cell state to expose as h_t.

$$ i, f, o = σ(·),   g = tanh(·)   (each from x_t W + h_{t−1} U + b) $$
$$ c_t = f ⊙ c_{t−1} + i ⊙ g          h_t = o ⊙ tanh(c_t) $$
:::where
x_t — the input at step t
h_{t−1} — the previous hidden state
W, U, b — each gate's own input weights, recurrent weights and bias
σ(·) — the sigmoid, giving gate values between 0 (closed) and 1 (open)
i, f, o — the input, forget and output gates
g — the candidate values to write, from tanh (between −1 and 1)
c_t — the cell state, the long-term memory: f decides what to keep and i decides what to write
⊙ — element-wise multiplication
h_t — the new hidden state, a gated view of the cell state
:::

Because c_t is updated additively and the forget gate can stay near 1, gradients flow back through many steps — the same "highway" idea as residual connections. The **GRU** is a simpler two-gate variant with similar performance.` },
      { type: 'demo', title: 'Diagram: inside an LSTM cell', demo: 'dg-lstm' },
      { type: 'text', title: "GRUs, bidirectional RNNs and sequence tasks", md: `The **GRU** (gated recurrent unit, 2014) is a lighter cousin of the LSTM: it merges the cell and hidden states and uses two gates instead of three:

$$ z_t = σ(x_t W_z + h_{t−1} U_z)          r_t = σ(x_t W_r + h_{t−1} U_r) $$
$$ h̃_t = tanh(x_t W + (r_t ⊙ h_{t−1}) U)          h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t $$
:::where
z_t — the update gate: how much of the state to replace with new content (biases are left out for brevity)
r_t — the reset gate: how much of the old state to use when proposing new content
h̃_t — the candidate state
⊙ — element-wise multiplication
h_t — the new state, a gated mix of the old state and the candidate; when z_t ≈ 0 the state is copied forward unchanged, which is what lets gradients survive long spans
:::

GRUs often match LSTMs with fewer parameters. Two other variations are common:
- **Bidirectional** RNNs run one RNN left to right and another right to left and combine their states, so every position sees its whole context. That suits tagging and classification, but not generation, where the future is unknown.
- **Stacked** RNNs feed one layer's hidden states into the next for more capacity.

| Pattern | Input → output | Example |
| --- | --- | --- |
| many-to-one | a sequence → one label | the sentiment of a review |
| many-to-many, aligned | one output per input step | part-of-speech tagging |
| sequence-to-sequence | a sequence → a different sequence | translation (see [Sequence-to-Sequence Models](#/concept/seq2seq)) |
| one-to-many | one input → a sequence | image captioning |

Long sequences are trained with **truncated BPTT**: backpropagate through only the last k steps (say 100), while carrying the hidden state forward from one chunk to the next.` },
      { type: 'code', title: "An LSTM cell keeps what a plain RNN forgets", md: `A memory test: store a value in the state, feed 50 steps of unrelated input, and measure how much of the value is left.`, code: `import numpy as np
rng = np.random.default_rng(0)
d, steps = 16, 50
Wh = rng.normal(0, 0.9 / np.sqrt(d), (d, d))      # recurrent weights of a plain RNN
Wx = rng.normal(0, 0.3, (d, d))
noise = rng.normal(size=(steps, d))               # 50 steps of unrelated input
sigmoid = lambda z: 1 / (1 + np.exp(-z))

def plain_rnn(start):
    h = start.copy()
    for x in noise:
        h = np.tanh(x @ Wx + h @ Wh)               # the whole state is rewritten at every step
    return h

def lstm_cell(start):
    c = start.copy()
    f, i = sigmoid(5.0), sigmoid(-5.0)            # forget gate near 1 (keep), input gate near 0 (write little)
    for x in noise:
        c = f * c + i * np.tanh(x @ Wx)           # the conveyor belt: mostly addition
    return c

memory = np.zeros(d)
memory[0] = 1.0                                   # the value we want to remember
empty = np.zeros(d)
rnn_left = np.linalg.norm(plain_rnn(memory) - plain_rnn(empty))
lstm_left = np.linalg.norm(lstm_cell(memory) - lstm_cell(empty))
print(f"what is left of the stored value after {steps} steps:  plain RNN {rnn_left:.2e}   LSTM cell {lstm_left:.3f}")
print("cell state preserved:", bool(lstm_left > 0.6 and rnn_left < 0.1))`, after: `Comparing runs with and without the stored value isolates what the state still knows about it. The plain RNN pushes its whole state through a weight matrix and tanh at every step, and after 50 steps the difference is about 10⁻¹⁵: forgotten. The LSTM's cell is only multiplied by the forget gate, σ(5) ≈ 0.993, at each step, so 0.993⁵⁰ ≈ 0.71 of the value survives. A trained LSTM learns when to hold its gates like this and when to overwrite.`, predict: {"type":"mcq","q":"A value is stored, then 50 steps of unrelated input follow. With the forget gate held near 1 and the input gate near 0, how much of the value survives in the LSTM's cell state?","options":["Almost none","About 70%","Exactly 100%","It grows"],"answer":1,"explain":"Each step multiplies the cell by the forget gate, about 0.993, so 0.993⁵⁰ ≈ 0.71 remains. The plain RNN rewrites its whole state every step and keeps almost nothing.","expect":["cell state preserved: True"]} },
      { type: 'text', title: 'Why transformers took over', md: `RNNs (even LSTMs) have two structural limits:
1. **Sequential computation** — step t needs step t−1, so you can't parallelise across the sequence. Training on huge datasets with GPUs is slow.
2. **A fixed-size memory bottleneck** — everything the model knows about the past must squeeze into one hidden vector; long-range information still degrades.

**Attention** (2014) first helped by letting a decoder look directly at all encoder states. The **Transformer** (2017) went all-in: no recurrence at all, every token attends to every other token in parallel. That made training massively parallel and long-range dependencies one step away — and enabled today's LLMs. (Recent "state-space" models such as Mamba revisit recurrence with efficient parallel training — recurrence isn't dead.)` },
      { type: 'check', question: { type: 'tf', q: 'In an LSTM, the forget gate decides how much of the previous cell state to keep.', answer: true, explain: 'c_t = f ⊙ c_{t−1} + i ⊙ g: f near 1 keeps the memory, near 0 erases it.' } }
    ],
    when: `- Small sequence problems, streaming/online settings where you process one step at a time with constant memory, and some time-series tasks.
- Most NLP and many sequence tasks today use transformers; for long time series, consider temporal CNNs, transformers or state-space models too.
- Use LSTM/GRU rather than a vanilla RNN whenever dependencies span more than a few steps.`,
    pitfalls: `- Expecting a vanilla RNN to remember things from far back.
- Forgetting gradient clipping (exploding gradients).
- Leaking the future: in forecasting, the model must only see past values at each step.
- Batching variable-length sequences without padding masks.`,
    takeaways: [
      'RNN: h_t = tanh(x_t W_x + h_{t−1} W_h + b), same weights every step.',
      'BPTT backpropagates through the unrolled network; gradients vanish or explode over long spans.',
      'LSTMs add a cell state with forget/input/output gates for long-term memory.',
      'Sequential computation and a fixed-size memory limit RNNs.',
      'Transformers replace recurrence with parallel attention over all positions.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the hidden state h_t of an RNN represent?', options: ['The next input', 'A learned summary of the sequence up to step t', 'The loss at step t', 'The learning rate'], answer: 1, explain: 'It is the network\'s running memory.' },
      { type: 'mcq', q: 'Which mechanism lets LSTM gradients flow over long spans?', options: ['Larger learning rates', 'The additively updated cell state controlled by gates', 'Dropout', 'Max pooling'], answer: 1, explain: 'With the forget gate near 1, the cell state passes information (and gradients) almost unchanged.' },
      { type: 'tf', q: 'RNNs can process all time steps of a sequence in parallel during training.', answer: false, explain: 'Each step depends on the previous hidden state, which forces sequential computation.' },
      { type: 'mcq', q: 'What is the standard fix for exploding gradients in RNNs?', options: ['Gradient clipping', 'Removing the bias', 'Smaller batch size', 'Max pooling'], answer: 0, explain: 'Clip the gradient norm before the update.' },
      { type: 'mcq', q: 'What key advantage did transformers have over RNNs for training on huge datasets?', options: ['Fewer parameters', 'Parallel processing of all positions and direct access to any earlier token', 'No need for data', 'No activation functions'], answer: 1, explain: 'Attention connects any two positions in one step and parallelises across the sequence.' },
      {"type":"mcq","q":"Why can't a bidirectional RNN generate text one token at a time?","options":["It is too slow","Its backward direction needs the future tokens, which do not exist yet during generation","It has no hidden state","It cannot use embeddings"],"answer":1,"explain":"The right-to-left pass reads the sequence from its end, so it is only usable when the whole input is available, as in tagging or classification."},
      {"type":"mcq","q":"In a GRU, what happens when the update gate z_t is close to 0?","options":["The state is reset to zero","The previous state is copied forward almost unchanged","The input is ignored and the output is zero","The gradient explodes"],"answer":1,"explain":"h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t, so z_t ≈ 0 keeps the old state, which also gives gradients a direct path back in time."}
    ],
    problems: [
      {
        id: 'rnn-forward', title: 'RNN Forward Pass', difficulty: 'medium',
        prompt: `Implement \`rnn_forward(X, h0, Wx, Wh, b)\` → \`H\`:

- X: (T, d_in) — one input vector per time step; h0: (d_h,) initial state.
- For t = 0 … T−1: \`h = tanh(X[t] @ Wx + h @ Wh + b)\`.
- Return H: (T, d_h), the hidden state after every step.

\`\`\`text
With Wx = [[1]], Wh = [[0]], b = [0] the RNN has no memory: H = tanh(X).
\`\`\``,
        starter: `
import numpy as np

def rnn_forward(X, h0, Wx, Wh, b):
    X = np.asarray(X, dtype=float)
    h = np.asarray(h0, dtype=float)
    states = []
    # TODO
    return np.array(states)
`,
        solution: `
import numpy as np

def rnn_forward(X, h0, Wx, Wh, b):
    X = np.asarray(X, dtype=float)
    h = np.asarray(h0, dtype=float)
    states = []
    for t in range(len(X)):
        h = np.tanh(X[t] @ Wx + h @ Wh + b)
        states.append(h)
    return np.array(states)
`,
        hints: ['Loop over time; the new h depends on the old h.', 'Append each new state, then stack with `np.array(states)`.'],
        explain: `The loop is the essence of recurrence — and also why RNNs are slow on GPUs: step t can't start until step t−1 has finished. Notice how the same three weight arrays are used at every step.`,
        tests: [
          { name: 'no memory: H = tanh(X)', call: 'rnn_forward(np.array([[0.5], [-1.0], [2.0]]), np.zeros(1), np.array([[1.0]]), np.array([[0.0]]), np.zeros(1))', expect: 'np.tanh(np.array([[0.5], [-1.0], [2.0]]))' },
          { name: 'with memory', call: 'rnn_forward(np.array([[1.0], [0.0]]), np.zeros(1), np.array([[1.0]]), np.array([[1.0]]), np.zeros(1))', expect: 'np.array([[np.tanh(1.0)], [np.tanh(np.tanh(1.0))]])' },
          { name: 'random sequence', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(7, 3)); h0 = rng.normal(size=4)\nWx = rng.normal(size=(3, 4)) * 0.5; Wh = rng.normal(size=(4, 4)) * 0.5; b = rng.normal(size=4) * 0.1', call: 'rnn_forward(X, h0, Wx, Wh, b)', hidden: true }
        ]
      },
      {
        id: 'lstm-cell', title: 'An LSTM Cell', difficulty: 'hard',
        prompt: `Implement one LSTM step: \`lstm_step(x, h, c, W, U, b)\` → \`(h_new, c_new)\`.

- x: (d_in,), h and c: (H,); W: (d_in, 4H), U: (H, 4H), b: (4H,).
- \`z = x @ W + h @ U + b\` (length 4H), split into four consecutive chunks **in this order**: i, f, g, o.
- \`i = σ(z_i)\`, \`f = σ(z_f)\`, \`g = tanh(z_g)\`, \`o = σ(z_o)\`.
- \`c_new = f * c + i * g\`, \`h_new = o * tanh(c_new)\`.

\`\`\`text
With every weight and bias 0: i = f = o = 0.5, g = 0
-> c_new = 0.5 * c,  h_new = 0.5 * tanh(0.5 * c)
\`\`\``,
        starter: `
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def lstm_step(x, h, c, W, U, b):
    H = len(h)
    # TODO
    pass
`,
        solution: `
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def lstm_step(x, h, c, W, U, b):
    H = len(h)
    z = np.asarray(x) @ W + np.asarray(h) @ U + b
    i = sigmoid(z[:H])
    f = sigmoid(z[H:2 * H])
    g = np.tanh(z[2 * H:3 * H])
    o = sigmoid(z[3 * H:])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new
`,
        hints: ['Compute all four pre-activations in one matrix product, then slice with H.', 'Only g uses tanh; the gates use sigmoid.'],
        explain: `Computing all gates with one stacked matrix multiply is exactly how real implementations work (it's much faster than four separate products). The update c_new = f·c + i·g is the additive memory path that keeps gradients alive across many steps.`,
        tests: [
          { name: 'zero weights', setup: 'H = 2\nW = np.zeros((3, 8)); U = np.zeros((2, 8)); b = np.zeros(8)\nc = np.array([1.0, -2.0])', call: 'lstm_step(np.ones(3), np.zeros(2), c, W, U, b)', expect: '(0.5 * np.tanh(0.5 * c), 0.5 * c)' },
          { name: 'forget gate wide open, input gate shut', setup: 'W = np.zeros((1, 4)); U = np.zeros((1, 4))\nb = np.array([-100.0, 100.0, 0.0, 100.0])', call: 'lstm_step(np.array([3.0]), np.array([0.0]), np.array([0.7]), W, U, b)', expect: '(np.array([np.tanh(0.7)]), np.array([0.7]))' },
          { name: 'random cell vs reference', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(size=3); h = rng.normal(size=4); c = rng.normal(size=4)\nW = rng.normal(size=(3, 16)) * 0.5; U = rng.normal(size=(4, 16)) * 0.5; b = rng.normal(size=16) * 0.1', call: 'lstm_step(x, h, c, W, U, b)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Chris Olah — Understanding LSTM Networks', url: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/' },
      { title: 'Andrej Karpathy — The Unreasonable Effectiveness of RNNs', url: 'https://karpathy.github.io/2015/05/21/rnn-effectiveness/' }
    ]
  },

  // =====================================================================================
  {
    id: 'embeddings', branch: 'dl', title: 'Embeddings: Meaning as Vectors', short: 'Embeddings', level: 2, minutes: 40, prereqs: ['mlp', 'vectors-matrices'],
    tags: ['embeddings', 'word2vec', 'lookup table', 'semantic similarity', 'vector search'],
    summary: 'Representing words, tokens, users or products as dense learned vectors where similar things are close: lookup tables, how embeddings are learned, similarity and analogies.',
    why: `Embeddings are how neural networks represent **discrete things** — words, tokens, products, users, songs — as points in a continuous space where **distance means similarity**. Every LLM starts by looking up an embedding vector for each input token; recommendation systems embed users and items; semantic search and RAG embed documents and queries and find nearest neighbours. Understanding embeddings is understanding how meaning becomes geometry.`,
    objectives: [
      'Explain why one-hot vectors are a poor representation and what dense embeddings fix',
      'Implement an embedding layer as a lookup table and relate it to a one-hot matrix product',
      'Explain how embeddings are learned (word2vec idea, end-to-end training)',
      'Find nearest neighbours and solve analogies with cosine similarity'
    ],
    sections: [
      { type: 'text', title: 'From one-hot to dense vectors', md: `How do we feed the word "cat" to a network? One option is **one-hot**: a vector the size of the vocabulary (say 50,000) with a single 1. Problems:
- **Huge and sparse** — 50,000 numbers per word, all but one zero.
- **No notion of similarity** — "cat" and "kitten" are exactly as different as "cat" and "carburettor" (all one-hot vectors are orthogonal).

An **embedding** maps each item to a short **dense** vector of, say, 64–4096 learned numbers. Training arranges the space so that items used in similar ways end up **close together**. Individual dimensions usually aren't interpretable, but directions often are (e.g. a rough "gender" or "plural" direction).` },
      { type: 'text', title: 'An embedding layer is a lookup table', md: `An embedding layer is just a matrix **E** of shape (vocabulary size V, dimension d). Row i is the vector for token i:

$$ embedding(token i) = E[i] $$
:::where
E — the embedding matrix, shape (V, d): one learned row per token
V — the vocabulary size
d — the embedding dimension
E[i] — row i of E, the vector for token i
:::

Mathematically that equals multiplying a one-hot vector by E (one_hot(i) @ E picks row i), but the lookup is far cheaper. In PyTorch: \`nn.Embedding(V, d)\`.

E is learned by backpropagation like any other weights: the gradient only touches the rows of the tokens that appeared in the batch. In an LLM, the embedding matrix is often among the largest single weight tensors (e.g. 128,000 tokens × 4,096 dimensions ≈ 524M parameters), and many models **tie** it to the output layer.` },
      { type: 'demo', title: 'Interactive: a toy embedding space', md: `Click words to see their nearest neighbours, then try vector arithmetic: king − man + woman.`, demo: 'embeddings' },
      { type: 'text', title: 'How embeddings are learned', md: `**The distributional hypothesis:** "you shall know a word by the company it keeps". Words appearing in similar contexts have similar meanings.

- **word2vec** (2013) trains a tiny network to predict a word from its neighbours (CBOW) or the neighbours from the word (skip-gram). The learned input weights *are* the embeddings. Famous result: vector("king") − vector("man") + vector("woman") ≈ vector("queen").
- **GloVe** factorizes a global word co-occurrence matrix.
- **End-to-end**: in modern models, embeddings are simply the first layer, trained jointly with everything else for the task (e.g. next-token prediction).
- **Contextual embeddings**: in a transformer, each token's vector is updated by attention through the layers, so "bank" gets different representations in "river bank" and "bank loan". Static word2vec vectors can't do that.
- **Sentence/document embeddings** (from models trained with contrastive objectives) power semantic search and RAG.` },
      { type: 'text', title: "How word2vec trains: negative sampling", md: `Skip-gram predicts the words around each word. A full softmax over a 100,000-word vocabulary at every position would be far too slow, so word2vec turns the task into many small yes-or-no questions. For a real (word, context) pair from the text, the model should answer "yes, these occur together"; for a few randomly drawn **negative** context words, it should answer "no":

$$ ℓ = −log σ( v_w · u_c ) − Σ_{n=1..k} log σ( −v_w · u_n ) $$
:::where
v_w — the embedding of the centre word w (the vectors kept at the end)
u_c — the context vector of a word c that really appeared near w
u_n — the context vectors of k randomly sampled negative words (k is typically 5 to 20)
σ — the sigmoid; σ(v · u) is the model's probability that the two words occur together
:::

Each update pulls the centre word towards its real neighbours and pushes it away from random words. Words that keep the same company receive the same pulls, so their vectors end up close: the distributional hypothesis, written as a loss. Very frequent words such as "the" are down-sampled, and negatives are drawn in proportion to frequency^0.75, which gives rarer words a little more weight.

The same recipe, real pairs against sampled negatives, reappears in contrastive learning ([Self-Supervised Representation Learning](#/concept/self-supervised)), in CLIP, and in the two-tower retrieval models of [Recommender Systems](#/concept/recommender-systems).` },
      { type: 'code', title: 'Learning embeddings from co-occurrence counts', md: `A tiny, classic way to get word vectors: count which words appear near each other, then compress the count matrix with SVD (as in PCA). Words with similar neighbours get similar vectors.`, code: `
import numpy as np

corpus = """the cat drinks milk . the dog drinks water . a cat eats fish . a dog eats meat .
the kitten drinks milk . the puppy drinks water . a kitten eats fish . a puppy eats meat .
the car needs fuel . a truck needs fuel . the car drives fast . a truck drives slow .""".split()
vocab = sorted(set(corpus)); idx = {w: i for i, w in enumerate(vocab)}

C = np.zeros((len(vocab), len(vocab)))
for i, w in enumerate(corpus):                         # count neighbours within a window of 2
    for j in range(max(0, i - 2), min(len(corpus), i + 3)):
        if j != i:
            C[idx[w], idx[corpus[j]]] += 1

U, S, Vt = np.linalg.svd(np.log1p(C))                  # compress counts to dense vectors
E = U[:, :4] * S[:4]                                   # 4-dimensional embeddings
E = E / np.linalg.norm(E, axis=1, keepdims=True)

def neighbours(word, k=3):
    sims = E @ E[idx[word]]
    order = [i for i in np.argsort(-sims) if vocab[i] != word][:k]
    return [(vocab[i], round(float(sims[i]), 2)) for i in order]

for w in ["cat", "puppy", "car", "milk"]:
    print(f"{w:>6}: {neighbours(w)}")
` },
      { type: 'chart', title: "How similar is each word to “cat”?", md: `The 4-dimensional vectors learned above from 12 tiny sentences, compared with “cat”:`, after: `With the raw vectors every word looks similar to “cat” (even “car” scores about 0.8), because all words share the same frequent neighbours (the, a, and the full stop), and that shared component dominates every vector. Subtracting the mean vector removes it, and the structure appears: the other animals stay close (about 0.8), while the vehicle words drop to clearly negative similarity. With only 4 dimensions and 12 sentences the picture is crude (fish ends up far from cat too), but real embedding spaces show the same shared-direction effect, called anisotropy, which is why centring or whitening embeddings often improves similarity search.`, chart: {"kind":"bar","x":["kitten","dog","puppy","milk","fish","drinks","car","truck","fuel"],"series":[{"name":"raw vectors","y":[0.979,0.975,0.975,0.869,0.811,0.826,0.795,0.634,0.802]},{"name":"after subtracting the mean vector","y":[0.836,0.792,0.792,0.548,-0.83,0.41,-0.342,-0.664,-0.813]}],"xLabel":"word","yLabel":"cosine similarity with “cat”","yMin":-1,"yMax":1} },
      { type: 'code', title: "Nearest neighbours and an analogy", code: `import numpy as np
corpus = """the cat drinks milk . the dog drinks water . a cat eats fish . a dog eats meat .
the kitten drinks milk . the puppy drinks water . a kitten eats fish . a puppy eats meat .
the car needs fuel . a truck needs fuel . the car drives fast . a truck drives slow .""".split()
vocab = sorted(set(corpus)); idx = {w: i for i, w in enumerate(vocab)}
C = np.zeros((len(vocab), len(vocab)))
for i, w in enumerate(corpus):
    for j in range(max(0, i - 2), min(len(corpus), i + 3)):
        if j != i:
            C[idx[w], idx[corpus[j]]] += 1
U, S, Vt = np.linalg.svd(np.log1p(C))
E = U[:, :4] * S[:4]
E = E / np.linalg.norm(E, axis=1, keepdims=True)

def closest(vector, exclude=(), k=3):
    vector = vector / np.linalg.norm(vector)
    order = [i for i in np.argsort(-(E @ vector)) if vocab[i] not in exclude]
    return [vocab[i] for i in order[:k]]

query = E[idx["puppy"]] - E[idx["dog"]] + E[idx["cat"]]     # "dog is to puppy as cat is to ?"
answer = closest(query, exclude=("puppy", "dog", "cat"))
print("puppy - dog + cat ->", answer)
print("the analogy finds kitten:", answer[0] == "kitten")`, after: `The offset from dog to puppy (adult to young), added to cat, lands nearest to kitten. Analogies like this make a striking demonstration rather than a reliable tool: they work best for frequent words and simple relations, and published examples exclude the three input words from the answers, as done here.`, predict: {"type":"mcq","q":"With these word vectors, which word is nearest to vector(puppy) − vector(dog) + vector(cat)?","options":["kitten","milk","truck","dog"],"answer":0,"explain":"The difference puppy − dog captures 'young version of', and adding it to cat points towards kitten, which appears in the same contexts as puppy does relative to dog.","expect":["the analogy finds kitten: True"]} },
      { type: 'check', question: { type: 'mcq', q: 'How is an embedding layer computed for token id 42?', options: ['A matrix multiplication with random noise', 'Row 42 of the embedding matrix is looked up', 'The id is divided by the vocabulary size', 'A softmax over the vocabulary'], answer: 1, explain: 'It is a lookup — equivalent to one_hot(42) @ E, but far cheaper.' } },
      { type: 'text', title: 'Embeddings everywhere', md: `- **LLMs**: token embeddings (+ positional information) are the input to every transformer.
- **Recommenders**: user and item embeddings whose dot product predicts a rating or click.
- **Search / RAG**: embed documents and queries with the same model; retrieve nearest neighbours by cosine similarity.
- **Categorical features** in tabular models: an embedding per category instead of huge one-hot vectors.
- **Images, audio, multimodal**: CLIP embeds images and text into one shared space, so "a photo of a dog" lands near dog photos.

Cautions: embeddings absorb **biases** from their training data (word embeddings famously associated professions with genders), and similarity in embedding space reflects co-occurrence statistics — not verified truth.` },
      { type: 'check', question: { type: 'tf', q: 'In static word2vec embeddings, the word "bank" has the same vector whether it refers to a river bank or a financial bank.', answer: true, explain: 'Static embeddings give one vector per word type; contextual embeddings from transformers fix this.' } }
    ],
    when: `- Any time you feed discrete items (tokens, IDs, categories) to a neural network.
- Similarity search, clustering, recommendation and retrieval: compare embeddings with cosine similarity.
- Use pretrained embedding models (sentence-transformers, API embedding endpoints) for search and RAG instead of training your own.`,
    pitfalls: `- Comparing embeddings from **different** models — their spaces are unrelated.
- Forgetting to normalise before using dot products as cosine similarity.
- Treating embedding similarity as factual or unbiased.
- One-hot encoding a huge vocabulary when an embedding layer would do.`,
    takeaways: [
      'Embeddings map discrete items to dense vectors where closeness means similarity.',
      'An embedding layer is a learned (V, d) lookup table: E[token_id].',
      'Learned from context (word2vec, GloVe) or end-to-end with the model.',
      'Cosine similarity finds neighbours; vector arithmetic can capture relationships.',
      'They power LLM inputs, recommendation, semantic search and RAG.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the main advantage of dense embeddings over one-hot vectors?', options: ['They are always exactly orthogonal', 'They are compact and place similar items close together', 'They need no training', 'They are interpretable dimension by dimension'], answer: 1, explain: 'One-hot vectors are huge and encode no similarity.' },
      { type: 'fill', q: 'A vocabulary has 32,000 tokens and the embedding dimension is 512. How many parameters does the embedding matrix have?', answer: ['16384000', '16,384,000', '16.384M', '16.4M'], explain: '32,000 × 512 = 16,384,000.' },
      { type: 'mcq', q: 'Which idea underlies word2vec?', options: ['Words are defined by their spelling', 'Words that appear in similar contexts have similar meanings', 'Words are independent of each other', 'Longer words are more important'], answer: 1, explain: 'The distributional hypothesis.' },
      { type: 'mcq', q: 'For semantic search over documents, you compare…', options: ['token ids', 'embedding vectors of the query and documents with cosine similarity', 'document lengths', 'one-hot vectors'], answer: 1, explain: 'Nearest neighbours in embedding space are the most semantically similar documents.' },
      { type: 'tf', q: 'Embeddings produced by two different models can be compared directly with cosine similarity.', answer: false, explain: 'Each model has its own coordinate system; vectors are only comparable within one model.' },
      {"type":"mcq","q":"In word2vec with negative sampling, what is a negative example?","options":["A word with a negative embedding","A randomly sampled word paired with the centre word, which the model should score as not occurring together","A misspelled word","A word that appears only once"],"answer":1,"explain":"Instead of a softmax over the whole vocabulary, the model learns to tell real (word, context) pairs from a few random ones."},
      {"type":"tf","q":"Subtracting the mean vector from a set of embeddings can make their cosine similarities more informative.","answer":true,"explain":"A shared component (from very frequent contexts) makes all vectors point roughly the same way; removing it reveals the differences that carry meaning."}
    ],
    problems: [
      {
        id: 'embedding-lookup', title: 'Embedding Lookup = One-Hot × Matrix', difficulty: 'easy',
        prompt: `Implement two equivalent ways to embed a list of token ids with an embedding matrix \`E\` of shape (V, d):

1. \`embed(ids, E)\` → \`E[ids]\`, shape (len(ids), d) — the fast lookup.
2. \`embed_one_hot(ids, E)\` → build the one-hot matrix of shape (len(ids), V) and multiply it by E.

Both must give the same result.

\`\`\`text
E = [[0, 1], [2, 3], [4, 5]]
embed([2, 0, 2], E)  ->  [[4, 5], [0, 1], [4, 5]]
\`\`\``,
        starter: `
import numpy as np

def embed(ids, E):
    # TODO
    pass

def embed_one_hot(ids, E):
    # TODO
    pass
`,
        solution: `
import numpy as np

def embed(ids, E):
    return np.asarray(E)[np.asarray(ids)]

def embed_one_hot(ids, E):
    E = np.asarray(E, dtype=float)
    ids = np.asarray(ids)
    one_hot = np.zeros((len(ids), E.shape[0]))
    one_hot[np.arange(len(ids)), ids] = 1.0
    return one_hot @ E
`,
        hints: ['Fancy indexing: `E[ids]` with an array of ids returns those rows.', 'Set `one_hot[np.arange(n), ids] = 1` in one go.'],
        explain: `A one-hot row times E selects one row of E — so an embedding layer is mathematically a linear layer applied to one-hot inputs, implemented as a cheap lookup. That's also why its gradient only updates the rows of tokens present in the batch.`,
        tests: [
          { call: 'embed([2, 0, 2], np.array([[0, 1], [2, 3], [4, 5]]))', expect: 'np.array([[4, 5], [0, 1], [4, 5]])' },
          { call: 'embed_one_hot([2, 0, 2], np.array([[0, 1], [2, 3], [4, 5]]))', expect: 'np.array([[4.0, 5.0], [0.0, 1.0], [4.0, 5.0]])' },
          { name: 'both agree on random data', setup: 'rng = np.random.default_rng(0)\nE = rng.normal(size=(50, 8))\nids = rng.integers(0, 50, size=20)', code: 'assert np.allclose(embed(ids, E), embed_one_hot(ids, E))\nassert embed(ids, E).shape == (20, 8)', hidden: true }
        ]
      },
      {
        id: 'most-similar', title: 'Nearest Neighbours by Cosine Similarity', difficulty: 'medium',
        prompt: `Implement \`most_similar(word, vocab, E, k=3)\`:

- \`vocab\` is a list of words; row i of E is the embedding of \`vocab[i]\`.
- Compute the cosine similarity between \`word\`'s vector and every word's vector.
- Return a list of the k most similar \`(other_word, similarity)\` pairs, **excluding the word itself**, sorted by similarity (highest first); ties keep vocabulary order. Similarities are Python floats.

\`\`\`text
vocab = ["cat", "dog", "car"]; E = [[1, 0], [0.9, 0.1], [0, 1]]
most_similar("cat", vocab, E, 1)  ->  [("dog", 0.9939...)]
\`\`\``,
        starter: `
import numpy as np

def most_similar(word, vocab, E, k=3):
    E = np.asarray(E, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def most_similar(word, vocab, E, k=3):
    E = np.asarray(E, dtype=float)
    unit = E / np.linalg.norm(E, axis=1, keepdims=True)
    i = vocab.index(word)
    sims = unit @ unit[i]
    order = [j for j in np.argsort(-sims, kind="stable") if j != i][:k]
    return [(vocab[j], float(sims[j])) for j in order]
`,
        hints: ['Normalise every row once; then cosine similarity is a dot product.', '`np.argsort(-sims, kind="stable")` sorts descending while keeping vocabulary order on ties.', 'Skip the word\'s own index.'],
        explain: `Normalising all vectors once turns cosine similarity into a single matrix-vector product. Vector databases do the same thing — at the scale of billions of vectors, with approximate nearest-neighbour indexes instead of a full sort.`,
        tests: [
          { call: 'most_similar("cat", ["cat", "dog", "car"], [[1, 0], [0.9, 0.1], [0, 1]], 1)', expect: '[("dog", 0.9 / np.sqrt(0.82))]' },
          { name: 'order and exclusion', call: 'most_similar("a", ["a", "b", "c", "d"], [[1, 0], [0, 1], [1, 1], [-1, 0]], 3)', expect: '[("c", 1 / np.sqrt(2)), ("b", 0.0), ("d", -1.0)]' },
          { name: 'random embeddings', setup: 'rng = np.random.default_rng(0)\nvocab = ["w%d" % i for i in range(30)]\nE = rng.normal(size=(30, 6))', call: 'most_similar("w7", vocab, E, 5)', hidden: true }
        ]
      },
      {
        id: 'word-analogy', title: 'Solve Word Analogies', difficulty: 'medium',
        prompt: `"a is to b as c is to ?" — e.g. man : king :: woman : **queen**.

Implement \`analogy(a, b, c, vocab, E)\`:
1. target = E[b] − E[a] + E[c]
2. return the word whose embedding has the **highest cosine similarity** with target, **excluding** a, b and c.

\`\`\`text
analogy("man", "king", "woman", vocab, E)  ->  "queen"   (with a suitable E)
\`\`\``,
        starter: `
import numpy as np

def analogy(a, b, c, vocab, E):
    E = np.asarray(E, dtype=float)
    idx = {w: i for i, w in enumerate(vocab)}
    # TODO
    pass
`,
        solution: `
import numpy as np

def analogy(a, b, c, vocab, E):
    E = np.asarray(E, dtype=float)
    idx = {w: i for i, w in enumerate(vocab)}
    target = E[idx[b]] - E[idx[a]] + E[idx[c]]
    unit = E / np.linalg.norm(E, axis=1, keepdims=True)
    sims = unit @ (target / np.linalg.norm(target))
    for w in (a, b, c):
        sims[idx[w]] = -np.inf
    return vocab[int(np.argmax(sims))]
`,
        hints: ['Build the target vector from the three input words.', 'Set the similarity of a, b and c to −inf so they can\'t win.'],
        explain: `If the embedding space encodes a consistent "royalty" direction and a "gender" direction, then king − man + woman lands near queen. Excluding the input words matters: the nearest vector to the target is often b itself.`,
        tests: [
          { name: 'royalty', setup: 'vocab = ["man", "woman", "king", "queen", "apple"]\nE = np.array([[1.0, 0.0, 0.1], [1.0, 1.0, 0.1], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.2, -1.0]])', call: 'analogy("man", "king", "woman", vocab, E)', expect: '"queen"' },
          { name: 'plural', setup: 'vocab = ["cat", "cats", "dog", "dogs", "car"]\nE = np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 0.0, -2.0]])', call: 'analogy("cat", "cats", "dog", vocab, E)', expect: '"dogs"' },
          { name: 'random space vs reference', setup: 'rng = np.random.default_rng(3)\nvocab = ["w%d" % i for i in range(40)]\nE = rng.normal(size=(40, 10))', call: 'analogy("w1", "w2", "w3", vocab, E)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Mikolov et al. — Efficient Estimation of Word Representations (word2vec)', url: 'https://arxiv.org/abs/1301.3781' },
      { title: 'Jay Alammar — The Illustrated Word2vec', url: 'https://jalammar.github.io/illustrated-word2vec/' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
