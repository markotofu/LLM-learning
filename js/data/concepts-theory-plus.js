/* ML Atlas – content: theory additions (matrix calculus, optimization theory, generalization theory, Bayesian inference, Gaussian processes) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'matrix-calculus', branch: 'math', title: 'Matrix Calculus: Gradients of Vectors & Matrices', short: 'Matrix Calculus', level: 2, minutes: 45, prereqs: ['derivatives', 'vectors-matrices'],
    tags: ['matrix calculus', 'gradient', 'jacobian', 'hessian', 'chain rule', 'vector-jacobian product', 'backpropagation', 'gradient check', 'shapes', 'softmax cross-entropy'],
    summary: 'Derivatives when inputs and outputs are vectors and matrices: gradients, Jacobians and Hessians and their shapes, a small table of identities, the chain rule as vector–Jacobian products (exactly what backpropagation computes), the gradients of a linear layer and of softmax cross-entropy, and checking any of them numerically.',
    why: `Every model is trained with the gradient of one number — the loss — with respect to millions of parameters arranged in vectors and matrices. Differentiating one entry at a time is hopeless; **matrix calculus** lets you write the whole gradient of a layer in one line, such as ∂L/∂W = Xᵀ G, check it by its shape, and understand exactly what an autograd engine does for you. It is also the language of almost every machine-learning paper.`,
    objectives: [
      'Tell gradients, Jacobians and Hessians apart and predict their shapes',
      'Differentiate linear and quadratic expressions with a few identities',
      'Apply the chain rule as vector–Jacobian products, the way backpropagation does',
      'Derive the gradients of a linear layer and of softmax cross-entropy, and verify them numerically'
    ],
    sections: [
      { type: 'text', title: 'Three kinds of derivatives', md: `In *Derivatives, Gradients & the Chain Rule* you met the derivative of a function of one number and the gradient of a function of several. Machine learning needs three objects:

- a **gradient** when a vector goes in and a single number comes out (a loss of the parameters);
- a **Jacobian** when a vector goes in and a vector comes out (a layer);
- a **Hessian** for the second derivatives of a single-number function (curvature).

$$ ∇f(x) = [ ∂f/∂x₁, …, ∂f/∂x_n ]          J_ij = ∂f_i / ∂x_j          H_ij = ∂²f / ∂x_i ∂x_j $$
:::where
f — the function being differentiated
x — the input vector (length n)
∇f(x) — "nabla f": the gradient, one partial derivative per input, with the same shape as x
J — the Jacobian of a function with m outputs: an m × n matrix whose row i is the gradient of output i
H — the Hessian of a single-number function: an n × n symmetric matrix of second derivatives
:::

**The shape rule.** In machine learning, the derivative of a scalar loss L with respect to any array has **the same shape as that array**: if W is 3 × 5, then ∂L/∂W is 3 × 5, holding one partial derivative per weight. Whenever you derive a gradient, check its shape first — most mistakes show up as a wrong shape or a missing transpose.` },
      { type: 'check', question: { type: 'mcq', q: 'A function maps vectors of length 5 to vectors of length 3. What is the shape of its Jacobian?', options: ['5 × 3', '3 × 5', '3', '5'], answer: 1, explain: 'One row per output (3), one column per input (5).' } },
      { type: 'text', title: 'Identities you will use constantly', md: `Most gradients in machine learning are built from a handful of rules. For a vector x:

| Function f(x) | Gradient ∇ₓf |
| --- | --- |
| aᵀx (a dot product) | a |
| xᵀx = ‖x‖² | 2x |
| xᵀA x | (A + Aᵀ) x, which is 2A x when A is symmetric |
| ‖A x − b‖² | 2 Aᵀ (A x − b) |
| A x (a vector) | Jacobian A |
:::where Symbols in the table
x — the variable we differentiate with respect to (a vector)
a, b — constant vectors
A — a constant matrix; Aᵀ is its transpose
‖v‖² — the squared length of v, the sum of its squared entries
:::

And for a matrix variable W, the gradient of aᵀ W b is the outer product a bᵀ — a matrix with the same shape as W. You can always rederive these by writing the expression as a sum over indices, differentiating one entry, and recognizing the result as a matrix product.` },
      { type: 'code', title: 'Checking gradient identities numerically', code: `
import numpy as np
rng = np.random.default_rng(0)

def numerical_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step.flat[i] = eps
        g.flat[i] = (f(x + step) - f(x - step)) / (2 * eps)   # central difference
    return g

n = 4
a, b, x = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
A = rng.normal(size=(n, n))

checks = {
    "a.x       -> a":            (lambda v: a @ v,                     a),
    "x.A.x     -> (A + A^T) x":  (lambda v: v @ A @ v,                 (A + A.T) @ x),
    "|Ax - b|^2 -> 2 A^T(Ax - b)": (lambda v: np.sum((A @ v - b) ** 2), 2 * A.T @ (A @ x - b)),
}
for name, (f, analytic) in checks.items():
    err = np.max(np.abs(numerical_grad(f, x) - analytic))
    print(f"{name:30s} largest difference {err:.1e}")
`, after: `Each formula matches the numerical estimate to many decimal places. Comparing a hand-derived gradient with finite differences — a **gradient check** — catches most mistakes in custom layers and losses.` },
      { type: 'text', title: 'The chain rule with matrices', md: `If a computation goes x → y = f(x) → L, the multivariable chain rule says the gradient flowing back to x is the transposed Jacobian times the gradient arriving at y:

$$ ∂L/∂x = Jᵀ · ∂L/∂y $$
:::where
x — the input of one step of the computation (length n)
y = f(x) — its output (length m)
J — the m × n Jacobian of f at x
∂L/∂y — the gradient arriving from the later steps (length m)
Jᵀ · ∂L/∂y — a vector–Jacobian product: the gradient passed further back (length n)
:::

Backpropagation is a chain of these **vector–Jacobian products**. It never builds a Jacobian explicitly — a layer with a million inputs and outputs would have a trillion entries — it only computes what the Jacobian does to the incoming gradient.

For a **linear layer** applied to a batch, written the way code does it:

$$ Y = X W + b          ∂L/∂W = Xᵀ G          ∂L/∂b = Σ_rows G          ∂L/∂X = G Wᵀ $$
:::where
X — a batch of inputs, one row per example (B × n)
W — the weight matrix (n × m)
b — the bias vector (length m), added to every row
Y — the outputs (B × m)
G — the upstream gradient ∂L/∂Y, the same shape as Y (B × m)
Σ_rows G — G summed over the batch, because every example uses the same bias
:::

Check the shapes: Xᵀ G is (n × B)(B × m) = n × m, the shape of W, and G Wᵀ is (B × m)(m × n) = B × n, the shape of X. The weight gradient sums the contributions of every example in the batch automatically through the matrix product.` },
      { type: 'code', title: 'A linear layer\'s backward pass, derived and checked', code: `
import numpy as np
rng = np.random.default_rng(1)
B, n, m = 5, 4, 3                        # batch size, inputs, outputs
X = rng.normal(size=(B, n))
W = rng.normal(size=(n, m))
b = rng.normal(size=m)
T = rng.normal(size=(B, m))              # targets

def loss(W, b, X):
    Y = X @ W + b
    return 0.5 * np.sum((Y - T) ** 2)

G = X @ W + b - T                        # dL/dY for this loss
dW, db, dX = X.T @ G, G.sum(axis=0), G @ W.T
print("shapes:  W", W.shape, "dW", dW.shape, "| b", b.shape, "db", db.shape, "| X", X.shape, "dX", dX.shape)

def num_grad(f, P, eps=1e-6):
    g = np.zeros_like(P)
    for i in range(P.size):
        d = np.zeros_like(P)
        d.flat[i] = eps
        g.flat[i] = (f(P + d) - f(P - d)) / (2 * eps)
    return g

print("dW error:", np.abs(dW - num_grad(lambda P: loss(P, b, X), W)).max())
print("db error:", np.abs(db - num_grad(lambda P: loss(W, P, X), b)).max())
print("dX error:", np.abs(dX - num_grad(lambda P: loss(W, b, P), X)).max())
` },
      { type: 'check', question: { type: 'tf', q: 'The gradient of the bias of a linear layer is the upstream gradient summed over the examples in the batch.', answer: true, explain: 'Every example adds the same bias, so every example contributes to its gradient.' } },
      { type: 'text', title: 'Two derivations worth knowing by heart', md: `**Least squares.** For linear regression with mean squared error:

$$ ∇_w ( ‖X w − y‖² / n ) = (2/n) Xᵀ ( X w − y ) $$
:::where
X — the data matrix, one row per example (n × d)
y — the targets (length n)
w — the weights (length d)
X w − y — the vector of residuals (prediction minus target)
:::

Setting it to zero gives the normal equations XᵀX w = Xᵀy from *Linear Regression*.

**Softmax followed by cross-entropy.** With logits z, probabilities p = softmax(z) and true class y, the loss is −log p_y, and its gradient is strikingly simple:

$$ ∂/∂z ( −log softmax(z)_y ) = softmax(z) − e_y $$
:::where
z — the vector of logits (one per class)
softmax(z) — the predicted probabilities
e_y — the one-hot vector with a 1 at the true class y
:::

"Predicted minus true" — the same p − y you saw in logistic regression. It is why classifiers learn quickly when they are confidently wrong, and why frameworks fuse softmax and cross-entropy into one numerically stable operation.` },
      { type: 'text', title: 'The Hessian and curvature', md: `The Hessian describes how the gradient itself changes. Near a point x, a second-order Taylor expansion approximates the function:

$$ f(x + Δ) ≈ f(x) + ∇f(x)ᵀ Δ + ½ Δᵀ H Δ $$
:::where
Δ — "delta": a small step away from x
∇f(x)ᵀ Δ — the first-order (slope) term
½ Δᵀ H Δ — the second-order (curvature) term, with H the Hessian at x
:::

At a point where the gradient is zero, the Hessian's eigenvalues (see *Eigenvectors & the SVD*) classify it: all positive means a local minimum, all negative a local maximum, mixed signs a **saddle point** — the most common kind of critical point in high-dimensional neural-network losses. The ratio of the largest to the smallest eigenvalue, the condition number, determines how hard the problem is for gradient descent (see *Optimization Theory*). A network with a billion parameters has a Hessian with 10¹⁸ entries, so it is never formed; autograd can still compute **Hessian–vector products** with two backward passes.` }
    ],
    when: `- Deriving gradients for a custom layer, loss or model — and checking them before trusting them.
- Reading papers, where results are written in matrix form.
- Writing efficient code: a derived matrix expression replaces slow Python loops.
- Understanding optimizers and training difficulties through the Hessian (curvature, saddle points, conditioning).`,
    pitfalls: `- Mixing layout conventions from different textbooks (some write gradients as rows, some as columns); stick to "the gradient has the shape of the variable".
- Forgetting a transpose or the sum over the batch for bias gradients — check every shape.
- Building full Jacobians when a vector–Jacobian product is all you need (memory explodes).
- Skipping the gradient check: finite differences with a step around 1e-6 and float64 catch most errors.
- Computing softmax and log separately (overflow and log(0)); use the combined, stable form.`,
    takeaways: [
      'Gradient: vector → number; Jacobian: vector → vector; Hessian: second derivatives of a number.',
      'The gradient with respect to an array has the same shape as the array.',
      'Backpropagation is a chain of vector–Jacobian products: ∂L/∂x = Jᵀ ∂L/∂y.',
      'Linear layer Y = XW + b: dW = XᵀG, db = column sums of G, dX = GWᵀ.',
      'Softmax cross-entropy has gradient p − e_y; always verify derivations with a gradient check.'
    ],
    quiz: [
      { type: 'mcq', q: 'W is a 3 × 5 matrix and L is a single number. What is the shape of ∂L/∂W?', options: ['5 × 3', '3 × 5', '15 × 1', '1 × 1'], answer: 1, explain: 'The gradient has the same shape as the variable.' },
      { type: 'mcq', q: 'What is the gradient of xᵀA x when A is symmetric?', options: ['A x', '2A x', 'xᵀA', 'A'], answer: 1, explain: 'In general (A + Aᵀ)x, which is 2Ax for symmetric A.' },
      { type: 'tf', q: 'Backpropagation computes the full Jacobian matrix of every layer.', answer: false, explain: 'It only computes vector–Jacobian products, which is far cheaper.' },
      { type: 'fill', q: 'Softmax probabilities are [0.7, 0.2, 0.1] and the true class is the first one. What is the gradient of the cross-entropy loss with respect to the first logit?', answer: ['-0.3', '−0.3'], explain: 'p − e_y: 0.7 − 1 = −0.3.' },
      { type: 'mcq', q: 'At a point with zero gradient, the Hessian has eigenvalues 3 and −2. What kind of point is it?', options: ['A local minimum', 'A local maximum', 'A saddle point', 'It cannot be determined'], answer: 2, explain: 'Curving up in one direction and down in another.' }
    ],
    problems: [
      {
        id: 'matrix-gradient-check', title: 'Gradient Check for Matrix Formulas', difficulty: 'easy',
        prompt: `Implement:

- \`numerical_grad(f, x, eps=1e-6)\` → the central-difference gradient of a function f that takes an array and returns a number. x may have any shape; return an array of the same shape whose entry i is (f(x + eps·e_i) − f(x − eps·e_i)) / (2·eps), where e_i is 1 at position i and 0 elsewhere.
- \`quad_grad(A, x)\` → the gradient of xᵀA x, which is (A + Aᵀ) x.
- \`lstsq_grad(A, x, b)\` → the gradient of ‖A x − b‖², which is 2 Aᵀ (A x − b).

Use float arrays and do not modify x in place.`,
        starter: `
import numpy as np

def numerical_grad(f, x, eps=1e-6):
    # TODO
    pass

def quad_grad(A, x):
    # TODO
    pass

def lstsq_grad(A, x, b):
    # TODO
    pass
`,
        solution: `
import numpy as np

def numerical_grad(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step.flat[i] = eps
        g.flat[i] = (f(x + step) - f(x - step)) / (2 * eps)
    return g

def quad_grad(A, x):
    A, x = np.asarray(A, dtype=float), np.asarray(x, dtype=float)
    return (A + A.T) @ x

def lstsq_grad(A, x, b):
    A, x, b = (np.asarray(v, dtype=float) for v in (A, x, b))
    return 2 * A.T @ (A @ x - b)
`,
        hints: ['Use `x.flat[i]` to address entry i of an array of any shape.', 'Build a fresh step array for each entry so x itself is never changed.'],
        explain: `Central differences have an error proportional to eps², far smaller than the one-sided (f(x + eps) − f(x)) / eps. In float64, eps around 1e-6 balances that error against rounding error. Frameworks offer the same check, e.g. \`torch.autograd.gradcheck\`.`,
        tests: [
          { call: 'quad_grad(np.array([[1.0, 2.0], [0.0, 3.0]]), np.array([1.0, -1.0]))', expect: 'np.array([0.0, -4.0])' },
          { call: 'numerical_grad(lambda v: np.sum(v ** 2), np.array([1.0, -2.0, 3.0]))', expect: 'np.array([2.0, -4.0, 6.0])' },
          { name: 'matrix-shaped input', setup: 'M = np.array([[1.0, 2.0], [-1.0, 0.5]])', code: 'g = numerical_grad(lambda P: np.sum(P ** 3), M)\nassert g.shape == M.shape\nassert np.allclose(g, 3 * M ** 2, atol=1e-5)', hidden: true },
          { name: 'least squares vs reference', setup: 'rng = np.random.default_rng(0)\nA = rng.normal(size=(5, 3)); x = rng.normal(size=3); b = rng.normal(size=5)', call: 'lstsq_grad(A, x, b)', hidden: true },
          { name: 'numerical matches analytic', setup: 'rng = np.random.default_rng(1)\nA = rng.normal(size=(4, 4)); x = rng.normal(size=4)', code: 'assert np.allclose(numerical_grad(lambda v: v @ A @ v, x), quad_grad(A, x), atol=1e-5)', hidden: true }
        ]
      },
      {
        id: 'linear-layer-backward', title: 'Backward Pass of a Linear Layer', difficulty: 'medium',
        prompt: `Implement the gradients you derived in this lesson:

- \`linear_backward(X, W, G)\` → \`(dW, db, dX)\` for the layer Y = X W + b, where X is (B, n), W is (n, m) and G = ∂L/∂Y is (B, m).
- \`softmax_ce_grad(Z, y)\` → the gradient of the **mean** cross-entropy loss over the batch with respect to the logits Z (B, k), where y holds the true class index of each row: (softmax(Z) − onehot(y)) / B.

Compute softmax stably by subtracting each row's maximum.`,
        starter: `
import numpy as np

def linear_backward(X, W, G):
    # TODO
    pass

def softmax_ce_grad(Z, y):
    # TODO
    pass
`,
        solution: `
import numpy as np

def linear_backward(X, W, G):
    X, W, G = (np.asarray(v, dtype=float) for v in (X, W, G))
    return X.T @ G, G.sum(axis=0), G @ W.T

def softmax_ce_grad(Z, y):
    Z = np.asarray(Z, dtype=float)
    P = np.exp(Z - Z.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(Z)), np.asarray(y)] -= 1.0
    return P / len(Z)
`,
        hints: ['Check every result\'s shape against the variable it belongs to.', 'Subtract 1 at the true class of each row: `P[np.arange(B), y] -= 1`.'],
        explain: `These two formulas are the heart of every classifier's backward pass. Dividing by B comes from averaging the loss over the batch; summing instead would make the gradient (and so the effective learning rate) grow with the batch size.`,
        tests: [
          { call: 'linear_backward(np.array([[1.0, 2.0]]), np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]), np.array([[1.0, -1.0, 0.5]]))', expect: '(np.array([[1.0, -1.0, 0.5], [2.0, -2.0, 1.0]]), np.array([1.0, -1.0, 0.5]), np.array([[0.5, 1.0]]))' },
          { call: 'softmax_ce_grad(np.zeros((1, 3)), [2])', expect: 'np.array([[1 / 3, 1 / 3, -2 / 3]])' },
          { name: 'random layer vs reference', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(6, 4)); W = rng.normal(size=(4, 3)); G = rng.normal(size=(6, 3))', call: 'linear_backward(X, W, G)', hidden: true },
          { name: 'matches finite differences', setup: 'rng = np.random.default_rng(1)\nZ = rng.normal(size=(4, 5)); y = np.array([0, 3, 1, 4])\ndef ce(Zv):\n    P = np.exp(Zv - Zv.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)\n    return -np.mean(np.log(P[np.arange(4), y]))', code: 'num = np.zeros_like(Z)\nfor i in range(Z.size):\n    d = np.zeros_like(Z); d.flat[i] = 1e-6\n    num.flat[i] = (ce(Z + d) - ce(Z - d)) / 2e-6\nassert np.allclose(softmax_ce_grad(Z, y), num, atol=1e-6)', hidden: true },
          { name: 'rows sum to zero', setup: 'rng = np.random.default_rng(2)\nZ = rng.normal(size=(5, 4)) * 10', code: 'assert np.allclose(softmax_ce_grad(Z, [0, 1, 2, 3, 0]).sum(axis=1), 0)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Petersen & Pedersen — The Matrix Cookbook', url: 'https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf' },
      { title: 'Parr & Howard — The Matrix Calculus You Need For Deep Learning', url: 'https://explained.ai/matrix-calculus/' },
      { title: 'Stanford CS231n — Derivatives, backpropagation and vectorization', url: 'https://cs231n.stanford.edu/handouts/derivatives.pdf' }
    ]
  },

  // =====================================================================================
  {
    id: 'optimization-theory', branch: 'math', title: 'Optimization Theory: Convexity, Constraints & Second-Order Methods', short: 'Optimization Theory', level: 3, minutes: 45, prereqs: ['gradient-descent', 'matrix-calculus'],
    tags: ['optimization', 'convexity', 'convex function', 'global minimum', 'condition number', 'newton\'s method', 'quasi-newton', 'l-bfgs', 'lagrange multipliers', 'kkt conditions', 'duality', 'constrained optimization'],
    summary: 'Why some problems are easy and others hard: convex functions and why convexity guarantees a global minimum, how curvature and the condition number set the speed of gradient descent, Newton\'s method and quasi-Newton methods such as L-BFGS, and constrained optimization with Lagrange multipliers and the KKT conditions — the tools behind PCA, SVMs and KL-constrained reinforcement learning.',
    why: `Training a model is solving an optimization problem. Knowing which problems are **convex** (linear and logistic regression, SVMs) tells you that any solution you reach is the best one; knowing why neural networks are not — and still train well — explains much of deep-learning practice. Conditioning explains why feature scaling, normalization layers and Adam help. And **Lagrange multipliers** appear in derivations everywhere: PCA, SVMs, maximum-entropy models and the KL penalty in RLHF.`,
    objectives: [
      'Define convex functions and explain why every local minimum of one is global',
      'Relate the condition number of the Hessian to the speed of gradient descent',
      'Explain Newton\'s method and why quasi-Newton or first-order methods are used instead at scale',
      'Solve constrained problems with Lagrange multipliers and interpret the KKT conditions'
    ],
    sections: [
      { type: 'text', title: 'Convex sets and functions', md: `A function is **convex** if the straight line between any two points on its graph never dips below the graph:

$$ f( θx + (1 − θ)y ) ≤ θ f(x) + (1 − θ) f(y)     for all x, y and 0 ≤ θ ≤ 1 $$
:::where
x, y — any two points (for us, two parameter vectors)
θ — "theta": a mixing weight; as it goes from 0 to 1, θx + (1 − θ)y moves along the segment from y to x
The right-hand side is the chord, the straight line between the points (x, f(x)) and (y, f(y)).
:::

For smooth functions this is equivalent to the Hessian being positive semi-definite everywhere — curving upward (or staying flat) in every direction. The crucial consequence: **every local minimum of a convex function is a global minimum**, so gradient descent with a sensible step size reaches the best possible solution.

- **Convex in the weights**: linear regression with squared error, logistic regression with log loss, SVMs with the hinge loss — and adding L1 or L2 penalties keeps them convex.
- **Not convex**: neural networks. Swapping two hidden units gives a different weight vector with exactly the same loss, so there are many separate minima, plus countless saddle points. That they still train well — overparameterized networks have many good minima reachable by SGD — is an empirical fact that theory is still catching up with.` },
      { type: 'code', title: 'Convex versus non-convex: where gradient descent ends up', code: `
import numpy as np

def descend(grad, x0, lr, steps=1000):
    x = x0
    for _ in range(steps):
        x -= lr * grad(x)
    return x

bowl = lambda x: (x - 2) ** 2 + 1              # convex
bowl_grad = lambda x: 2 * (x - 2)
bumpy = lambda x: x ** 4 - 3 * x ** 2 + x       # not convex: two valleys
bumpy_grad = lambda x: 4 * x ** 3 - 6 * x + 1

for x0 in (-2.0, -0.5, 0.5, 2.0):
    a = descend(bowl_grad, x0, lr=0.1)
    b = descend(bumpy_grad, x0, lr=0.01)
    print(f"start {x0:+.1f}:  convex -> x = {a:.3f} (f = {bowl(a):.3f})    non-convex -> x = {b:+.3f} (f = {bumpy(b):+.3f})")
`, after: `On the bowl every start reaches the same answer. On the bumpy function the starting point decides which valley you end in, and the valley at x ≈ +1.13 (f ≈ −1.07) is only a local minimum — the global one is at x ≈ −1.30 (f ≈ −3.51).` },
      { type: 'check', question: { type: 'tf', q: 'For a convex loss, any local minimum that gradient descent finds is also a global minimum.', answer: true, explain: 'That is the defining benefit of convexity.' } },
      { type: 'text', title: 'Curvature and the condition number', md: `Near a minimum, most losses look like a quadratic bowl ½ xᵀA x whose Hessian A has eigenvalues from λ_min to λ_max. Gradient descent is only stable if the step size η < 2 / λ_max — the steepest direction sets the speed limit — while progress along the flattest direction is proportional to η · λ_min. Their ratio decides how slow it is:

$$ κ = λ_max / λ_min          error after k steps ≤ ( (κ − 1) / (κ + 1) )^k · initial error $$
:::where
κ — "kappa": the condition number of the Hessian
λ_max, λ_min — its largest and smallest eigenvalues: the steepest and the flattest curvature
k — the number of gradient-descent steps, taken with the best fixed step size η = 2 / (λ_max + λ_min)
:::

With κ = 1 (a perfectly round bowl) a single step is enough; with κ = 100 each step removes only about 2% of the remaining error. In a long, narrow valley gradient descent zigzags across the steep direction while crawling along the flat one. Remedies: **standardize the features** (which reduces κ for linear models enormously), normalization layers, momentum, adaptive methods such as Adam (which rescale each parameter — a cheap, diagonal approximation of curvature) and second-order methods.` },
      { type: 'demo', title: 'Interactive: gradient descent in a narrow valley', demo: 'conditioning' },
      { type: 'code', title: 'Iterations needed on round and narrow bowls', code: `
import numpy as np
for lam_min, lam_max in [(1, 1), (1, 10), (1, 100), (1, 1000)]:
    A = np.diag([lam_min, lam_max]).astype(float)        # Hessian of f(x) = 0.5 * x^T A x
    lr = 2 / (lam_min + lam_max)                          # the best fixed step size
    x, steps = np.array([1.0, 1.0]), 0
    while np.linalg.norm(x) > 1e-6:
        x = x - lr * (A @ x)                              # the gradient of f is A x
        steps += 1
    x0 = np.array([1.0, 1.0])
    newton = x0 - np.linalg.solve(A, A @ x0)              # one Newton step: divide by the curvature
    print(f"kappa = {lam_max / lam_min:5.0f}:  gradient descent needs {steps:5d} steps;  one Newton step reaches {newton}")
`, after: `The number of gradient-descent steps grows in proportion to κ, while Newton's method — which divides each direction by its own curvature — solves any quadratic in a single step.` },
      { type: 'text', title: 'Newton\'s method and quasi-Newton methods', md: `Newton's method uses the Hessian to rescale the gradient:

$$ x_{k+1} = x_k − H(x_k)⁻¹ ∇f(x_k) $$
:::where
x_k — the current point
∇f(x_k) — the gradient there
H(x_k) — the Hessian there; multiplying by its inverse divides each direction by its curvature
:::

Near a minimum it converges extremely fast (the number of correct digits roughly doubles every step) and the condition number no longer matters. But:

- for d parameters the Hessian has d² entries and solving with it costs about d³ operations — impossible for a network with billions of weights;
- far from a minimum the Hessian may not be positive definite, and a raw Newton step can head for a maximum or a saddle, so practical versions add line searches or damping.

**Quasi-Newton** methods (BFGS) build an approximation of H⁻¹ from how the gradient changes between steps. **L-BFGS** keeps only the last few (≈ 10) pairs of changes, so its memory is linear in d; it is the standard solver for small and medium smooth problems — scikit-learn's \`LogisticRegression\` uses it by default. Deep learning relies on first-order methods with cheap preconditioning: Adam's per-parameter scaling, and matrix-aware optimizers such as Shampoo and Muon that precondition each weight matrix as a whole.` },
      { type: 'text', title: 'Constrained optimization and Lagrange multipliers', md: `Many problems come with constraints: a direction of unit length, probabilities that sum to one, a policy that must stay close to a reference model. To minimize f(x) subject to g(x) = 0, notice that at the optimum you cannot decrease f without leaving the constraint — the gradients of f and g must be parallel. A **Lagrange multiplier** λ turns this into an equation:

$$ ℒ(x, λ) = f(x) + λ · g(x)          ∇ₓℒ = ∇f(x) + λ ∇g(x) = 0,     g(x) = 0 $$
:::where
f — the objective to minimize
g(x) = 0 — the constraint
λ — "lambda": the Lagrange multiplier; at the optimum it says how much the best objective would change if the constraint were loosened slightly
ℒ — the Lagrangian, which folds the objective and the constraint into one function
:::

**Example — PCA.** Find the unit direction w that maximizes the variance of the data projected onto it:

$$ maximize wᵀΣw subject to wᵀw = 1     ⇒     ∇_w [ wᵀΣw − λ(wᵀw − 1) ] = 2Σw − 2λw = 0     ⇒     Σw = λw $$
:::where
Σ — "Sigma": the covariance matrix of the centred data
w — a unit-length direction
wᵀΣw — the variance of the data projected onto w
Σw = λw — the eigenvector equation, so the best direction is an eigenvector, and the variance it captures, wᵀΣw = λ, is its eigenvalue
:::

With **inequality** constraints g_i(x) ≤ 0, the optimum satisfies the **KKT conditions**: the gradient condition, feasibility, multipliers λ_i ≥ 0, and **complementary slackness** λ_i · g_i(x) = 0 — a multiplier can be non-zero only for a constraint that is active (exactly tight). In an SVM the training points with non-zero multipliers are the ones on or inside the margin: the **support vectors**. Every Lagrangian also defines a **dual problem**; for convex problems it usually has the same optimal value, and the SVM's dual depends on the data only through dot products — which is what makes the kernel trick possible.

**Penalties are constraints in disguise.** Minimizing loss + λ‖w‖² gives the same solution as minimizing the loss subject to ‖w‖² ≤ t for a matching t. The KL penalty in RLHF is, in the same way, the Lagrangian of "maximize reward while staying within a KL budget of the reference model" — the trust-region idea behind TRPO and PPO.` },
      { type: 'code', title: 'Lagrange multipliers find the principal component', code: `
import numpy as np
rng = np.random.default_rng(0)
x1 = rng.normal(size=500)
data = np.column_stack([x1, 0.6 * x1 + 0.4 * rng.normal(size=500)])   # two correlated features
S = np.cov(data, rowvar=False)                                          # covariance matrix

angles = np.linspace(0, np.pi, 10001)                                   # every unit direction (up to sign)
W = np.column_stack([np.cos(angles), np.sin(angles)])
variance = np.einsum("ij,jk,ik->i", W, S, W)                             # w^T S w for each direction
best = W[np.argmax(variance)]

eigvals, eigvecs = np.linalg.eigh(S)
top = eigvecs[:, -1] * np.sign(eigvecs[0, -1])
print("best direction by brute force:  ", best.round(3), "  variance captured:", variance.max().round(4))
print("top eigenvector of the covariance:", top.round(3), "  eigenvalue:       ", eigvals[-1].round(4))
`, after: `Searching every direction on the unit circle and solving Σw = λw give the same answer: the constrained maximum is the top eigenvector, and the largest variance equals its eigenvalue — exactly what the Lagrange condition predicted.` },
      { type: 'check', question: { type: 'mcq', q: 'In a trained SVM, which training points have non-zero Lagrange multipliers?', options: ['All of them', 'Only misclassified points', 'The support vectors: points on or inside the margin', 'None of them'], answer: 2, explain: 'Complementary slackness: only active constraints get non-zero multipliers.' } }
    ],
    when: `- Recognizing convex problems (linear and logistic regression, lasso, SVMs): a solver that reaches zero gradient is done.
- Diagnosing slow or unstable training through conditioning: scale features, normalize, lower the learning rate.
- L-BFGS for small and medium smooth problems; momentum or Adam for deep networks.
- Lagrange multipliers to derive PCA, SVMs, maximum-entropy models and constrained or KL-penalized policies.`,
    pitfalls: `- Assuming deep-learning losses are convex — initialization and randomness change where you end up.
- Unscaled features that create huge condition numbers and painfully slow training.
- Learning rates above 2 / λ_max, which make training diverge.
- Raw Newton steps far from a minimum, without a line search or damping.
- Forgetting that a penalty weight implicitly fixes a constraint level — changing λ changes the problem.`,
    takeaways: [
      'Convex functions have no bad local minima; linear models with standard losses are convex, neural networks are not.',
      'The condition number κ = λ_max / λ_min sets how slowly gradient descent converges.',
      'Newton\'s method rescales by curvature and is fast, but impossible at deep-learning scale; L-BFGS approximates it.',
      'Lagrange multipliers solve constrained problems; PCA falls out as an eigenvector problem.',
      'KKT complementary slackness explains support vectors; penalties are constraints in disguise.'
    ],
    quiz: [
      { type: 'mcq', q: 'Which of these losses is convex in the model\'s weights?', options: ['A two-layer neural network with MSE', 'Logistic regression with log loss', 'A transformer language model', 'The k-means objective over centroids and assignments'], answer: 1, explain: 'Linear models with standard losses are convex; networks are not.' },
      { type: 'fill', q: 'The Hessian at a minimum has eigenvalues 2 and 200. What is the condition number?', answer: ['100'], explain: 'κ = 200 / 2 = 100.' },
      { type: 'mcq', q: 'Why isn\'t Newton\'s method used to train large language models?', options: ['It needs labelled data', 'The Hessian has (number of parameters)² entries — impossible to store or invert', 'It only works for linear models', 'It always diverges'], answer: 1, explain: 'A billion parameters means 10¹⁸ Hessian entries.' },
      { type: 'tf', q: 'At a constrained optimum, the gradient of the objective is parallel to the gradient of an active constraint.', answer: true, explain: 'Otherwise you could move along the constraint and improve the objective.' },
      { type: 'mcq', q: 'What does complementary slackness say?', options: ['Every multiplier is positive', 'A multiplier can be non-zero only if its constraint is active', 'The objective equals the constraint', 'The dual problem is always easier'], answer: 1, explain: 'λ_i · g_i(x) = 0 at the optimum.' }
    ],
    problems: [
      {
        id: 'conditioning-gd', title: 'Conditioning and Gradient Descent', difficulty: 'medium',
        prompt: `For the quadratic f(x) = ½·xᵀA x with a symmetric positive-definite matrix A (its gradient is A x), implement:

- \`condition_number(A)\` → λ_max / λ_min (use \`np.linalg.eigvalsh\`).
- \`best_step(A)\` → the best fixed step size 2 / (λ_max + λ_min).
- \`gd_steps(A, x0, tol=1e-6, lr=None, max_steps=100000)\` → how many updates x ← x − lr·A x are needed until ‖x‖ ≤ tol. Use \`best_step(A)\` when lr is None. Return 0 if ‖x0‖ ≤ tol already, and \`max_steps\` if the tolerance is never reached within that many steps.`,
        starter: `
import numpy as np

def condition_number(A):
    # TODO
    pass

def best_step(A):
    # TODO
    pass

def gd_steps(A, x0, tol=1e-6, lr=None, max_steps=100000):
    # TODO
    pass
`,
        solution: `
import numpy as np

def condition_number(A):
    lam = np.linalg.eigvalsh(np.asarray(A, dtype=float))
    return float(lam[-1] / lam[0])

def best_step(A):
    lam = np.linalg.eigvalsh(np.asarray(A, dtype=float))
    return float(2 / (lam[-1] + lam[0]))

def gd_steps(A, x0, tol=1e-6, lr=None, max_steps=100000):
    A = np.asarray(A, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    if lr is None:
        lr = best_step(A)
    steps = 0
    while np.linalg.norm(x) > tol:
        if steps >= max_steps:
            return max_steps
        x = x - lr * (A @ x)
        steps += 1
    return steps
`,
        hints: ['`np.linalg.eigvalsh` returns the eigenvalues of a symmetric matrix in ascending order.', 'Count updates in a while loop and stop at max_steps.'],
        explain: `With the best step size, each update shrinks the error by at least (κ − 1)/(κ + 1), so the step count grows roughly in proportion to κ. A learning rate above 2/λ_max makes the steepest direction grow instead of shrink — the run never reaches the tolerance.`,
        tests: [
          { call: 'condition_number(np.diag([1.0, 100.0]))', expect: '100.0' },
          { call: 'best_step(np.diag([1.0, 3.0]))', expect: '0.5' },
          { name: 'a round bowl takes one step', call: 'gd_steps(np.eye(2), np.array([1.0, 1.0]))', expect: '1' },
          { name: 'narrow bowl vs reference', call: 'gd_steps(np.diag([1.0, 100.0]), np.array([1.0, 1.0]))', hidden: true },
          { name: 'a too-large learning rate never converges', call: 'gd_steps(np.diag([1.0, 10.0]), np.array([1.0, 1.0]), lr=0.25, max_steps=500)', expect: '500', hidden: true }
        ]
      },
      {
        id: 'newton-lagrange', title: 'Newton Steps and the Constrained Maximum', difficulty: 'medium',
        prompt: `Implement:

- \`newton(grad, hess, x0, steps=10)\` → apply the update x ← x − H(x)⁻¹ ∇f(x) \`steps\` times and return x. \`grad(x)\` returns a vector and \`hess(x)\` a matrix; use \`np.linalg.solve\` rather than an explicit inverse.
- \`top_component(S)\` → \`(lam, w)\`: the solution of "maximize wᵀS w subject to wᵀw = 1" for a symmetric matrix S — the largest eigenvalue and its unit eigenvector. Choose the sign of w so that its entry with the largest absolute value is positive.`,
        starter: `
import numpy as np

def newton(grad, hess, x0, steps=10):
    # TODO
    pass

def top_component(S):
    # TODO
    pass
`,
        solution: `
import numpy as np

def newton(grad, hess, x0, steps=10):
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(steps):
        x = x - np.linalg.solve(np.asarray(hess(x), dtype=float), np.asarray(grad(x), dtype=float))
    return x

def top_component(S):
    lam, vecs = np.linalg.eigh(np.asarray(S, dtype=float))
    w = vecs[:, -1]
    if w[np.argmax(np.abs(w))] < 0:
        w = -w
    return float(lam[-1]), w
`,
        hints: ['Solve H·step = gradient instead of inverting H.', '`np.linalg.eigh` returns eigenvalues in ascending order; the last column of the eigenvector matrix belongs to the largest one.'],
        explain: `On a quadratic, one Newton step lands exactly on the minimum. For the constrained problem, the Lagrange condition Sw = λw says the candidates are eigenvectors, and the objective wᵀSw = λ picks the one with the largest eigenvalue.`,
        tests: [
          { name: 'one step solves a quadratic', call: 'newton(lambda x: np.array([[2.0, 0.0], [0.0, 8.0]]) @ x - np.array([2.0, 8.0]), lambda x: np.array([[2.0, 0.0], [0.0, 8.0]]), np.array([5.0, -3.0]), steps=1)', expect: 'np.array([1.0, 1.0])' },
          { call: 'top_component(np.array([[2.0, 0.0], [0.0, 5.0]]))', expect: '(5.0, np.array([0.0, 1.0]))' },
          { name: 'a non-quadratic function', call: 'newton(lambda x: x ** 3 - 1, lambda x: np.atleast_2d(3 * x ** 2), np.array([2.0]), steps=10)', expect: 'np.array([1.0])', hidden: true },
          { name: 'random covariance vs reference', setup: 'rng = np.random.default_rng(0)\nZ = rng.normal(size=(50, 3)) @ rng.normal(size=(3, 3))\nS = np.cov(Z, rowvar=False)', call: 'top_component(S)', hidden: true },
          { name: 'satisfies the Lagrange condition', setup: 'rng = np.random.default_rng(1)\nM = rng.normal(size=(4, 4)); S = M @ M.T', code: 'lam, w = top_component(S)\nassert np.isclose(np.linalg.norm(w), 1)\nassert np.allclose(S @ w, lam * w)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Boyd & Vandenberghe — Convex Optimization (free book)', url: 'https://web.stanford.edu/~boyd/cvxbook/' },
      { title: 'Nocedal & Wright — Numerical Optimization', url: 'https://link.springer.com/book/10.1007/978-0-387-40065-5' },
      { title: 'Goh — Why Momentum Really Works (Distill)', url: 'https://distill.pub/2017/momentum/' }
    ]
  },

  // =====================================================================================
  {
    id: 'learning-theory', branch: 'ml', title: 'Generalization Theory: PAC Learning, VC Dimension & Double Descent', short: 'Generalization Theory', level: 3, minutes: 45, prereqs: ['overfitting', 'stat-inference'],
    tags: ['generalization', 'learning theory', 'empirical risk', 'hoeffding inequality', 'union bound', 'pac learning', 'sample complexity', 'vc dimension', 'shattering', 'double descent', 'implicit regularization', 'benign overfitting', 'grokking'],
    summary: 'When does a model that fits its training data also do well on new data? True versus empirical risk, Hoeffding\'s inequality for a single model, the union bound and PAC learning for a finite set of models, VC dimension for infinite ones — and what classical theory misses about deep learning: double descent, implicit regularization, benign overfitting and grokking.',
    why: `Every time you trust a validation score, choose between models or wonder how much data you need, you are making a generalization argument. **Learning theory** makes those arguments precise: it explains why testing many models on the same validation set makes the winner look better than it is, why more data shrinks the gap like 1/√n, and why model capacity matters. It also shows where classical intuition breaks: huge neural networks generalize even though they can memorize random labels.`,
    objectives: [
      'Define true and empirical risk and explain why the training error of a selected model is optimistic',
      'Use Hoeffding\'s inequality and the union bound to compute generalization gaps and sample sizes',
      'Explain shattering and the VC dimension with simple examples',
      'Describe double descent, implicit regularization and why classical bounds fail for deep networks'
    ],
    sections: [
      { type: 'text', title: 'Training error is an optimistic estimate', md: `A model h has a **true risk** — its expected loss on new data from the same distribution — and an **empirical risk**, its average loss on the n training examples:

$$ R(h) = E_(x,y)~D [ ℓ(h(x), y) ]          R̂(h) = (1/n) Σ_i ℓ(h(x_i), y_i) $$
:::where
h — a hypothesis: one particular model (one setting of the parameters)
D — the unknown distribution the data comes from
ℓ — the loss of one prediction, e.g. 0 when correct and 1 when wrong
R(h) — the true risk: the expected loss on new examples
R̂(h) — "R hat": the empirical risk, the average loss on the training set
:::

Learning usually picks the hypothesis with the smallest R̂ (**empirical risk minimization**). For one *fixed* h, R̂ is an unbiased estimate of R. But we chose h *because* its R̂ was small, so for the chosen model R̂ is optimistic — and the more models we choose among, the more optimistic it is. **Generalization theory** bounds the gap R(h) − R̂(h).` },
      { type: 'text', title: 'One model: Hoeffding\'s inequality', md: `For a single model fixed before seeing the data, the empirical risk is an average of n independent losses, and averages concentrate around their mean:

$$ P( |R̂(h) − R(h)| > ε ) ≤ 2 · exp( −2 n ε² ) $$
:::where
ε — "epsilon": the size of deviation we worry about
n — the number of examples
exp — the exponential function
The losses must lie between 0 and 1 (such as the 0/1 classification error), and h must not depend on the sample.
:::

Solving for ε: with probability at least 1 − δ, the gap is at most √( ln(2/δ) / 2n ). The gap shrinks like 1/√n — quadrupling the data halves it. This is also the honest error bar on a **test set**, which is exactly a fixed model evaluated on fresh data.` },
      { type: 'code', title: 'Hoeffding\'s bound holds — loosely', code: `
import numpy as np
rng = np.random.default_rng(0)
true_error, eps = 0.3, 0.05
print("    n   P(estimate off by more than 0.05)   Hoeffding bound")
for n in (50, 200, 500, 1000, 2000):
    estimates = rng.binomial(n, true_error, size=20000) / n      # 20,000 test sets of size n
    observed = np.mean(np.abs(estimates - true_error) > eps)
    bound = min(1.0, 2 * np.exp(-2 * n * eps ** 2))
    print(f"{n:5d}   {observed:33.4f}   {bound:15.4f}")
`, after: `The bound is never violated, but it is loose: real deviations are rarer than it allows, and for small n it says nothing at all (a bound of 1). Bounds like this hold for every possible distribution, so they are guarantees rather than predictions.` },
      { type: 'text', title: 'Many models: the union bound and PAC learning', md: `If we choose among a finite set of |H| models, the chance that *any* of them has a large gap is at most |H| times the chance for one (the **union bound**). So with probability at least 1 − δ, every model in H — including the one we pick — satisfies:

$$ R(h) ≤ R̂(h) + √( ( ln|H| + ln(2/δ) ) / (2n) ) $$
:::where
|H| — the number of models we choose among
δ — "delta": the allowed probability that the guarantee fails, e.g. 0.05
ln — the natural logarithm; the number of models enters only logarithmically
:::

This is the core of **PAC learning** ("probably approximately correct"): with enough data, the chosen model is *approximately* as good as it looks (within ε), *probably* (with probability 1 − δ). Rearranged, it gives a **sample complexity**: n ≥ (ln|H| + ln(2/δ)) / (2ε²).

Practical lesson: every configuration you try on a validation set enlarges H. Trying a thousand configurations costs only ln 1000 ≈ 7 in the numerator — but it is not free, which is why a heavily tuned validation score is optimistic and a final, untouched test set matters.` },
      { type: 'text', title: 'Infinite classes: the VC dimension', md: `A linear classifier has infinitely many settings, so ln|H| is useless. The **Vapnik–Chervonenkis (VC) dimension** measures capacity instead: the largest number of points that the class can **shatter** — label in all 2^m possible ways.

- Thresholds on a line (predict 1 when x ≥ t): VC dimension 1.
- Intervals on a line (predict 1 when a ≤ x ≤ b): 2.
- Linear classifiers in d dimensions: d + 1. In the plane, three points can be split every way, but no four points can (the XOR labelling fails).

A typical bound replaces ln|H| with roughly the VC dimension:

$$ R(h) ≤ R̂(h) + O( √( ( d_VC · ln(n / d_VC) + ln(1/δ) ) / n ) ) $$
:::where
d_VC — the VC dimension of the model class
O( … ) — "on the order of": constant factors are left out
:::

The rule of thumb that follows: you need several times more examples than the model's effective number of degrees of freedom.` },
      { type: 'code', title: 'Shattering points with a line', code: `
import numpy as np
from itertools import product

def linearly_separable(points, labels, epochs=1000):
    X = np.c_[points, np.ones(len(points))]          # add a bias input
    s = 2 * np.array(labels) - 1                      # labels as -1 / +1
    w = np.zeros(X.shape[1])
    for _ in range(epochs):                           # the perceptron converges whenever a separating line exists
        mistakes = 0
        for xi, si in zip(X, s):
            if si * (w @ xi) <= 0:
                w += si * xi
                mistakes += 1
        if mistakes == 0:
            return True
    return False

three = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
four = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
for name, pts in [("3 points", three), ("4 points", four)]:
    ok = [lab for lab in product([0, 1], repeat=len(pts)) if linearly_separable(pts, lab)]
    print(f"{name}: a line can produce {len(ok)} of the {2 ** len(pts)} labellings")
fails = [lab for lab in product([0, 1], repeat=4) if not linearly_separable(four, lab)]
print("labellings of the square that no line can produce:", fails)
`, after: `Three points in general position can be split every possible way, but the two XOR labellings of the four corners cannot — lines in the plane have VC dimension 3 = d + 1.`, predict: {"type":"mcq","q":"Four points sit at the corners of a square. Of the 16 ways to label them positive or negative, how many can a straight line separate?","options":["All 16","14","8","4"],"answer":1,"explain":"14: the two diagonal (XOR-style) labellings are impossible. A line can shatter 3 points but no set of 4, so its VC dimension is 3.","expect":["4 points: a line can produce 14 of the 16 labellings"]} },
      { type: 'check', question: { type: 'mcq', q: 'What is the VC dimension of intervals on the real line (predict 1 inside [a, b])?', options: ['1', '2', '3', 'Infinite'], answer: 1, explain: 'Any 2 points can be labelled every way, but the labelling 1, 0, 1 of three points cannot.' } },
      { type: 'text', title: 'What classical theory misses', md: `Modern networks have far more parameters than training examples. They can even memorize completely **random** labels (Zhang et al., 2017), so their capacity is enormous and the classical bounds above say nothing — yet the same networks generalize well on real labels. Several ideas help explain this:

- **Implicit regularization**: the training algorithm prefers some solutions over others. Gradient descent on linear least squares, started at zero, converges to the **minimum-norm** solution that fits the data; the noise in SGD favours flat minima.
- **Double descent**: as models grow, test error first follows the classical U-shape, peaks where the model can *just barely* fit the training data exactly (the **interpolation threshold**), then *decreases again* as the model keeps growing.
- **Benign overfitting**: in high dimensions, fitting noisy training labels exactly can do surprisingly little harm.
- **Grokking**: on some tasks, test accuracy jumps long after training accuracy reached 100%, especially with weight decay.
- Bounds based on **norms, margins, compression and PAC-Bayes** can give non-vacuous guarantees for real networks, where parameter counting cannot.` },
      { type: 'code', title: 'Double descent with random features', code: `
import numpy as np
rng = np.random.default_rng(0)
n, d = 50, 10                                            # 50 training examples, 10 input features
beta = rng.normal(size=d) / np.sqrt(d)
target = lambda X: X @ beta + 0.5 * np.tanh(X[:, 0] * X[:, 1])
X_train, X_test = rng.normal(size=(n, d)), rng.normal(size=(2000, d))
y_train = target(X_train) + 0.5 * rng.normal(size=n)     # noisy training labels
y_test = target(X_test)
W = rng.normal(size=(d, 1000)) / np.sqrt(d)               # a random hidden layer that is never trained

print(" features   train MSE   test MSE   weight norm")
for p in (10, 25, 40, 45, 50, 55, 60, 100, 200, 1000):
    F_train, F_test = np.maximum(0, X_train @ W[:, :p]), np.maximum(0, X_test @ W[:, :p])
    coef = np.linalg.pinv(F_train) @ y_train              # least squares; the minimum-norm exact fit once p >= n
    print(f"{p:9d} {np.mean((F_train @ coef - y_train) ** 2):11.4f} {np.mean((F_test @ coef - y_test) ** 2):10.3f} {np.linalg.norm(coef):13.1f}")
`, after: `Training error reaches zero once there are as many features as examples (p = n = 50). Exactly there the test error explodes — the only exact fit needs huge, noise-chasing weights — and then it falls again as features are added, because the minimum-norm fit among the many exact fits gets smoother (watch the weight norm). Classical theory predicts the first half of this curve, not the second. In this small, noisy example the smallest model is still the best one.` }
    ],
    when: `- Judging how far to trust a validation or test score, and how much it is inflated by model selection.
- Estimating how much data a task needs, and why more data shrinks the generalization gap like 1/√n.
- Reasoning about model capacity: simple models on small data, flexible models on large data.
- Interpreting surprising deep-learning behaviour: double descent, interpolation, grokking.`,
    pitfalls: `- Reporting the validation score of the configuration you selected as if it were an unbiased estimate.
- Re-using the test set for decisions, which turns it into another validation set.
- Treating classical capacity bounds as predictions for deep networks — they are usually vacuous.
- Concluding "bigger is always better" from double descent: near the interpolation threshold, a slightly bigger model can be much worse, and regularization still matters.`,
    takeaways: [
      'Generalization gap = true risk − empirical risk; selection on the same data makes it optimistic.',
      'Hoeffding: for one fixed model the gap shrinks like √(ln(2/δ) / 2n).',
      'Union bound / PAC: choosing among |H| models adds ln|H| — validation tuning is not free.',
      'The VC dimension (largest shattered set) measures capacity; lines in d dimensions have d + 1.',
      'Deep networks defy classical bounds: implicit regularization, double descent and benign overfitting.'
    ],
    quiz: [
      { type: 'mcq', q: 'You evaluate 200 model configurations on one validation set and report the best score. What is true?', options: ['The score is an unbiased estimate of future performance', 'The score is optimistic because of selection; confirm on an untouched test set', 'The score is pessimistic', 'Selection does not affect the score'], answer: 1, explain: 'Choosing the maximum of many noisy estimates biases it upwards.' },
      { type: 'fill', q: 'By what factor must the number of test examples grow to halve the Hoeffding error bar?', answer: ['4', 'four', '4x'], explain: 'The bar shrinks like 1/√n.' },
      { type: 'mcq', q: 'What is the VC dimension of linear classifiers in 2-D (lines with a bias)?', options: ['2', '3', '4', 'Infinite'], answer: 1, explain: 'd + 1 = 3: three points can be shattered, four cannot.' },
      { type: 'tf', q: 'In double descent, test error peaks near the point where the model can first fit the training data exactly.', answer: true, explain: 'That is the interpolation threshold.' },
      { type: 'mcq', q: 'Which fact shows that classical capacity bounds cannot explain deep-learning generalization?', options: ['Networks train slowly', 'Networks can memorize random labels, yet generalize on real ones', 'Networks use GPUs', 'Networks are convex'], answer: 1, explain: 'Their capacity is huge, so capacity-based bounds are vacuous.' }
    ],
    problems: [
      {
        id: 'generalization-bounds', title: 'Generalization Bounds and Sample Sizes', difficulty: 'easy',
        prompt: `Implement:

- \`hoeffding(n, eps)\` → the bound min(1, 2·exp(−2·n·eps²)) on the probability that an empirical error is off by more than eps.
- \`generalization_gap(n, delta, n_models=1)\` → √( (ln(n_models) + ln(2/delta)) / (2n) ).
- \`samples_needed(eps, delta, n_models=1)\` → the smallest integer n for which the gap is at most eps: ⌈(ln(n_models) + ln(2/delta)) / (2·eps²)⌉.`,
        starter: `
import math

def hoeffding(n, eps):
    # TODO
    pass

def generalization_gap(n, delta, n_models=1):
    # TODO
    pass

def samples_needed(eps, delta, n_models=1):
    # TODO
    pass
`,
        solution: `
import math

def hoeffding(n, eps):
    return min(1.0, 2 * math.exp(-2 * n * eps ** 2))

def generalization_gap(n, delta, n_models=1):
    return math.sqrt((math.log(n_models) + math.log(2 / delta)) / (2 * n))

def samples_needed(eps, delta, n_models=1):
    return math.ceil((math.log(n_models) + math.log(2 / delta)) / (2 * eps ** 2))
`,
        hints: ['`math.log` is the natural logarithm.', 'Rearrange gap ≤ eps for n, then round up with `math.ceil`.'],
        explain: `The number of models enters only through its logarithm, while the precision enters squared: halving eps quadruples the data you need. That asymmetry is why fine-grained comparisons between models need surprisingly large test sets.`,
        tests: [
          { call: 'hoeffding(100, 0.1)', expect: '2 * math.exp(-2)' },
          { call: 'generalization_gap(1000, 0.05)', expect: 'math.sqrt(math.log(40) / 2000)' },
          { call: 'samples_needed(0.05, 0.05, 1000)', expect: 'math.ceil((math.log(1000) + math.log(40)) / (2 * 0.05 ** 2))' },
          { name: 'the bound is capped at 1', call: 'hoeffding(1, 0.1)', expect: '1.0', hidden: true },
          { name: 'the sample size is the smallest that works', code: 'n = samples_needed(0.02, 0.01, 50)\nassert generalization_gap(n, 0.01, 50) <= 0.02 + 1e-12\nassert generalization_gap(n - 1, 0.01, 50) > 0.02', hidden: true }
        ]
      },
      {
        id: 'shatter-intervals', title: 'Shattering with Rays and Intervals', difficulty: 'medium',
        prompt: `For distinct points on a line, collect the labellings that two simple model classes can produce. Labellings are tuples of 0/1 labels in the **same order as the input points**.

- \`ray_labelings(xs)\` → the set of labellings produced by h_t(x) = 1 if x ≥ t, over every threshold t (including thresholds below and above all points).
- \`interval_labelings(xs)\` → the set produced by h(x) = 1 if a ≤ x ≤ b, over every interval (including empty ones).
- \`shatters(labelings, m)\` → True if the set contains all 2^m labellings of m points.

\`\`\`text
ray_labelings([1, 2])  ->  {(0, 0), (0, 1), (1, 1)}
\`\`\``,
        starter: `
def ray_labelings(xs):
    # TODO
    pass

def interval_labelings(xs):
    # TODO
    pass

def shatters(labelings, m):
    # TODO
    pass
`,
        solution: `
def ray_labelings(xs):
    xs = list(xs)
    thresholds = sorted(xs) + [max(xs) + 1]
    return {tuple(1 if x >= t else 0 for x in xs) for t in thresholds}

def interval_labelings(xs):
    xs = list(xs)
    s = sorted(xs)
    out = {tuple(0 for _ in xs)}
    for i in range(len(s)):
        for j in range(i, len(s)):
            out.add(tuple(1 if s[i] <= x <= s[j] else 0 for x in xs))
    return out

def shatters(labelings, m):
    return len(set(labelings)) == 2 ** m
`,
        hints: ['Only thresholds at the data points (plus one beyond the largest) give different labellings.', 'An interval is determined by which consecutive run of sorted points it covers.'],
        explain: `n points produce n + 1 ray labellings and n(n + 1)/2 + 1 interval labellings — polynomially many, far fewer than 2^n. That polynomial growth is exactly what the VC dimension captures: rays shatter 1 point, intervals 2.`,
        tests: [
          { call: 'ray_labelings([1, 2])', expect: '{(0, 0), (0, 1), (1, 1)}' },
          { call: 'len(interval_labelings([3, 1, 2]))', expect: '7' },
          { call: '[shatters(ray_labelings([5.0]), 1), shatters(ray_labelings([1.0, 2.0]), 2)]', expect: '[True, False]' },
          { name: 'intervals shatter 2 points but not 3', call: '[shatters(interval_labelings([1.0, 2.0]), 2), shatters(interval_labelings([1.0, 2.0, 3.0]), 3)]', expect: '[True, False]', hidden: true },
          { name: 'counts on six points', call: '[len(ray_labelings([0.3, -1.0, 2.5, 7.0, 4.0, 1.1])), len(interval_labelings([0.3, -1.0, 2.5, 7.0, 4.0, 1.1]))]', expect: '[7, 22]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Shalev-Shwartz & Ben-David — Understanding Machine Learning: From Theory to Algorithms (free book)', url: 'https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/' },
      { title: 'Zhang et al. — Understanding deep learning requires rethinking generalization', url: 'https://arxiv.org/abs/1611.03530' },
      { title: 'Belkin et al. — Reconciling modern machine learning practice and the bias–variance trade-off', url: 'https://arxiv.org/abs/1812.11118' },
      { title: 'Nakkiran et al. — Deep Double Descent', url: 'https://arxiv.org/abs/1912.02292' }
    ]
  },

  // =====================================================================================
  {
    id: 'bayesian-methods', branch: 'ml', title: 'Bayesian Inference: Priors, Posteriors, MCMC & Variational Inference', short: 'Bayesian Inference', level: 3, minutes: 50, prereqs: ['mle-map'],
    tags: ['bayesian inference', 'prior', 'posterior', 'conjugate prior', 'beta distribution', 'credible interval', 'posterior predictive', 'bayesian linear regression', 'mcmc', 'metropolis-hastings', 'variational inference', 'elbo', 'probabilistic programming'],
    summary: 'Treating unknown parameters as random variables: from MAP\'s single best guess to the whole posterior distribution, conjugate priors (Beta–Binomial), credible intervals and the posterior predictive, Bayesian linear regression with honest error bars, sampling posteriors with Markov chain Monte Carlo (Metropolis–Hastings), and approximating them with variational inference and the ELBO — the idea behind VAEs.',
    why: `MLE and MAP give one best guess for the parameters. But with little data, many parameter values explain the data almost equally well, and a single guess hides that uncertainty. **Bayesian inference** keeps the whole distribution of plausible parameters, so predictions come with honest error bars, prior knowledge enters explicitly, and decisions can weigh risks ("there is a 17% chance the click rate is above 5%"). It powers A/B-test analysis, Bayesian optimization, uncertainty estimates for models, and — through variational inference — VAEs.`,
    objectives: [
      'Compute posteriors with conjugate priors and summarize them with means and credible intervals',
      'Derive Bayesian linear regression and explain why its error bars grow away from the data',
      'Sample from an unnormalized posterior with the Metropolis algorithm and check the result',
      'Explain variational inference and the ELBO, and compare it with MCMC'
    ],
    sections: [
      { type: 'text', title: 'From a point estimate to a distribution', md: `Bayes' rule turns a prior belief about parameters θ into a posterior after seeing data D:

$$ p(θ | D) = p(D | θ) · p(θ) / p(D) $$
$$ p(x_new | D) = ∫ p(x_new | θ) · p(θ | D) dθ $$
:::where
θ — "theta": the unknown parameters
p(θ) — the prior: what we believe before seeing the data
p(D | θ) — the likelihood: how probable the data is for each θ
p(θ | D) — the posterior: what we believe after seeing the data
p(D) — the evidence: a normalizing constant that makes the posterior integrate to 1
p(x_new | D) — the posterior predictive: a prediction that averages over all plausible parameters instead of trusting one
:::

MLE maximizes the likelihood and MAP maximizes the posterior; both keep a single point. The Bayesian answer keeps the whole distribution. The catch is computational: the evidence p(D) is an integral over all parameters, which rarely has a formula. Three strategies follow — conjugate priors (exact), MCMC (sampling) and variational inference (optimization).` },
      { type: 'text', title: 'Conjugate priors: the Beta–Binomial model', md: `For a success probability θ (a click rate, a coin's bias), the Beta distribution is a natural prior. With a Beta(a, b) prior and k successes in n trials, the posterior is again a Beta — the prior is **conjugate** to the likelihood:

$$ prior Beta(a, b)  +  k successes, n − k failures   ⇒   posterior Beta(a + k, b + n − k)          mean = (a + k) / (a + b + n) $$
:::where
a, b — the prior's parameters; they act like pseudo-counts of earlier successes and failures
k — the observed number of successes
n — the number of trials
Beta(1, 1) is the uniform prior, saying that every rate between 0 and 1 is equally plausible.
:::

Updating is just adding counts. A **95% credible interval** is the range that contains the parameter with 95% posterior probability — the statement people usually (wrongly) read into a confidence interval. Other conjugate pairs: Normal–Normal for means, Gamma–Poisson for rates, Dirichlet–Multinomial for category probabilities (the basis of Laplace smoothing in Naive Bayes).` },
      { type: 'demo', title: 'Interactive: prior + data = posterior', demo: 'beta' },
      { type: 'code', title: 'Updating beliefs about a click-through rate', code: `
import numpy as np
rng = np.random.default_rng(0)
a, b = 1, 1                                         # Beta(1, 1): every click rate equally plausible
batches = [(3, 50), (9, 150), (31, 800)]            # (clicks, views) arriving over time
for clicks, views in batches:
    a, b = a + clicks, b + views - clicks           # conjugate update: just add the counts
    draws = rng.beta(a, b, 200_000)                  # samples from the posterior
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"after {a + b - 2:4d} views: Beta({a}, {b})  mean {a / (a + b):.4f}  "
          f"95% credible interval [{lo:.4f}, {hi:.4f}]  P(rate > 5%) = {np.mean(draws > 0.05):.3f}")
`, after: `Early on the interval is wide: the data leans towards a rate above 5% (about a 75–80% probability) but cannot rule out much lower rates. With 1,000 views the posterior concentrates around 4.4%, and the probability that the true rate is above 5% falls to about 17% — a direct answer to the business question that a p-value cannot give.` },
      { type: 'check', question: { type: 'fill', q: 'Prior Beta(2, 2); you observe 8 successes in 10 trials. What is the posterior mean? (fraction or decimal)', answer: ['10/14', '5/7', '0.714', '0.71'], explain: 'Posterior Beta(10, 4): mean 10 / 14 ≈ 0.714.' } },
      { type: 'text', title: 'Bayesian linear regression', md: `With a Gaussian prior w ~ N(0, τ²I) on the weights and Gaussian noise of variance σ², the posterior over the weights is also Gaussian, with a closed form:

$$ Σ_post = ( XᵀX / σ² + I / τ² )⁻¹          μ_post = Σ_post Xᵀy / σ²          Var[y_new] = σ² + x_newᵀ Σ_post x_new $$
:::where
X — the feature matrix (one row per example); y — the targets
σ² — "sigma squared": the variance of the observation noise
τ² — "tau squared": the prior variance of each weight; smaller means a stronger pull towards 0
μ_post, Σ_post — the mean and covariance of the posterior over the weights
x_new — the features of a new input
Var[y_new] — the predictive variance: irreducible noise σ² plus the uncertainty about the weights
:::

The posterior mean is exactly the **ridge regression** solution with λ = σ²/τ² — the same link you saw between MAP and L2 regularization. What the Bayesian view adds is the second term of the predictive variance: it is small near the training data and grows as x_new moves away, so the model says when it is extrapolating.` },
      { type: 'code', title: 'Bayesian linear regression knows what it doesn\'t know', code: `
import numpy as np
rng = np.random.default_rng(1)
x = rng.uniform(-1, 1, 20)
y = 0.5 + 2.0 * x + rng.normal(0, 0.3, 20)
X = np.c_[np.ones_like(x), x]                       # features: intercept and slope
sigma2, tau2 = 0.3 ** 2, 10.0                       # noise variance, prior variance of each weight

Sigma_post = np.linalg.inv(X.T @ X / sigma2 + np.eye(2) / tau2)
mu_post = Sigma_post @ X.T @ y / sigma2
ridge = np.linalg.solve(X.T @ X + (sigma2 / tau2) * np.eye(2), X.T @ y)
print("posterior mean weights:", mu_post.round(4), "  ridge with lambda = sigma2 / tau2:", ridge.round(4))
for x_new in (0.0, 1.0, 3.0, 10.0):
    phi = np.array([1.0, x_new])
    sd = np.sqrt(sigma2 + phi @ Sigma_post @ phi)    # predictive standard deviation
    print(f"x = {x_new:5.1f}: prediction {phi @ mu_post:6.2f} ± {2 * sd:.2f}")
`, after: `The posterior mean equals the ridge solution. The ±2-standard-deviation band is narrow inside the training range (−1 to 1) and widens steadily as we move away from it.` },
      { type: 'text', title: 'When there is no formula: Markov chain Monte Carlo', md: `Most posteriors have no closed form, but we can always compute the **unnormalized** posterior p̃(θ) = p(D | θ)·p(θ). **MCMC** builds a random walk whose long-run distribution is the posterior, using only p̃. The simplest version, **Metropolis–Hastings** with a symmetric proposal:

1. From the current θ, propose θ′ = θ + small random noise.
2. Accept the move with probability

$$ A = min( 1, p̃(θ′) / p̃(θ) ) $$
:::where
p̃ — the unnormalized posterior (likelihood × prior); the unknown constant p(D) cancels in the ratio
θ′ — "theta prime": the proposed new value
A — the acceptance probability: uphill moves are always accepted, downhill moves sometimes
:::

3. Otherwise stay at θ (and record θ again).

After a **burn-in**, the recorded values are (correlated) samples from the posterior; averages over them estimate posterior means, intervals and predictions. Check convergence with trace plots, several chains started apart (the R̂ statistic) and the effective sample size. Modern samplers such as **Hamiltonian Monte Carlo** and NUTS (used by Stan, PyMC and NumPyro) follow gradients to make long, efficient moves.` },
      { type: 'code', title: 'A Metropolis sampler for a coin\'s bias', code: `
import numpy as np
rng = np.random.default_rng(0)
heads, flips = 7, 10

def log_post(theta):                                 # log(prior x likelihood), up to a constant
    if not 0 < theta < 1:
        return -np.inf                               # zero probability outside (0, 1)
    return heads * np.log(theta) + (flips - heads) * np.log(1 - theta)   # flat Beta(1, 1) prior

theta, samples, accepted = 0.5, [], 0
for step in range(20_000):
    proposal = theta + 0.2 * rng.normal()            # random-walk proposal
    if np.log(rng.random()) < log_post(proposal) - log_post(theta):
        theta, accepted = proposal, accepted + 1     # accept; otherwise stay where we are
    samples.append(theta)
samples = np.array(samples[2000:])                   # drop the burn-in
a, b = 1 + heads, 1 + flips - heads                  # the exact posterior is Beta(8, 4)
print(f"acceptance rate {accepted / 20000:.2f}")
print(f"MCMC:  mean {samples.mean():.3f}  sd {samples.std():.3f}")
print(f"exact: mean {a / (a + b):.3f}  sd {np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))):.3f}")
`, after: `Without ever normalizing the posterior, the chain reproduces its mean and spread. The same few lines work for posteriors that have no formula at all — which is the whole point of MCMC.` },
      { type: 'text', title: 'Variational inference', md: `MCMC can be slow for big models and big data. **Variational inference (VI)** turns integration into optimization: choose a simple family of distributions q(θ) (for example independent Gaussians with their own means and variances) and tune it to be as close as possible to the posterior. Since the posterior is unknown, VI maximizes a quantity it can compute, the **evidence lower bound**:

$$ log p(D) = ELBO(q) + KL( q(θ) ‖ p(θ | D) )          ELBO(q) = E_q[ log p(D, θ) ] − E_q[ log q(θ) ] $$
:::where
q(θ) — the simple approximating distribution we tune
p(D, θ) — the joint probability of data and parameters, likelihood × prior
E_q[ … ] — an average over θ drawn from q
KL( q ‖ p(θ | D) ) — how far q is from the true posterior; it is never negative
ELBO — the evidence lower bound; because log p(D) is fixed, raising the ELBO lowers the KL divergence
:::

VI scales to huge models with stochastic gradients, and **amortized** VI — one network that outputs q for every input — is exactly the encoder of a VAE (see *Generative Models*). The trade-off: VI is fast but biased (it typically underestimates uncertainty), while MCMC is asymptotically exact but slower. In deep learning, practical approximations to Bayesian uncertainty include deep ensembles and MC dropout (see *Calibration & Uncertainty*).` },
      { type: 'check', question: { type: 'tf', q: 'Metropolis–Hastings needs the normalizing constant p(D) of the posterior.', answer: false, explain: 'It only uses ratios of the unnormalized posterior, in which p(D) cancels.' } }
    ],
    when: `- Small or expensive data where uncertainty matters: A/B tests, medical studies, early-stage metrics.
- When prior knowledge is real and should be used (physical constraints, earlier experiments).
- Decisions that need probabilities of outcomes, not just point estimates.
- Hierarchical models that share strength across groups (many stores, many products) — with probabilistic programming tools such as PyMC, Stan or NumPyro.`,
    pitfalls: `- Priors chosen carelessly — with little data they dominate; always check sensitivity to the prior.
- Reading MCMC output without convergence checks (one chain, no burn-in, no effective sample size).
- Random-walk proposals that are far too small (everything accepted, chain barely moves) or too large (nothing accepted).
- Trusting variational posteriors' variances too much — VI tends to be overconfident.
- Calling something Bayesian while still reporting only a single point estimate.`,
    takeaways: [
      'Posterior ∝ likelihood × prior; predictions average over the whole posterior.',
      'Conjugate priors give exact updates: Beta(a, b) + k of n successes → Beta(a + k, b + n − k).',
      'Bayesian linear regression: posterior mean = ridge; error bars grow away from the data.',
      'MCMC (Metropolis–Hastings, HMC) samples posteriors using only the unnormalized density.',
      'Variational inference maximizes the ELBO: fast and scalable, but usually overconfident.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the posterior predictive distribution do?', options: ['Uses the single most likely parameter value', 'Averages predictions over all parameter values, weighted by their posterior probability', 'Ignores the prior', 'Predicts the prior'], answer: 1, explain: 'p(x_new | D) = ∫ p(x_new | θ) p(θ | D) dθ.' },
      { type: 'mcq', q: 'With prior Beta(1, 1), you observe 3 successes and 1 failure. What is the posterior?', options: ['Beta(3, 1)', 'Beta(4, 2)', 'Beta(2, 4)', 'Beta(1, 1)'], answer: 1, explain: 'Add the counts: Beta(1 + 3, 1 + 1).' },
      { type: 'tf', q: 'The posterior mean of Bayesian linear regression with a Gaussian prior equals the ridge-regression solution.', answer: true, explain: 'With λ = σ² / τ².' },
      { type: 'mcq', q: 'In Metropolis–Hastings, a proposal with lower posterior density than the current point is…', options: ['Always rejected', 'Accepted with probability equal to the density ratio', 'Always accepted', 'Accepted only during burn-in'], answer: 1, explain: 'A = min(1, p̃(θ′)/p̃(θ)).' },
      { type: 'mcq', q: 'What does variational inference optimize?', options: ['The likelihood only', 'The evidence lower bound (ELBO)', 'The acceptance rate', 'The number of samples'], answer: 1, explain: 'Maximizing the ELBO minimizes KL(q ‖ posterior).' }
    ],
    problems: [
      {
        id: 'conjugate-updates', title: 'Conjugate Updates and Bayesian Regression', difficulty: 'easy',
        prompt: `Implement:

- \`beta_update(a, b, successes, failures)\` → the posterior parameters \`(a + successes, b + failures)\`.
- \`beta_mean_var(a, b)\` → \`(mean, variance)\` of Beta(a, b): mean a/(a + b) and variance ab / ((a + b)²(a + b + 1)).
- \`bayes_linreg(X, y, sigma2, tau2)\` → \`(mu, Sigma)\`, the posterior over the weights for the prior N(0, tau2·I) and noise variance sigma2: Sigma = (XᵀX/sigma2 + I/tau2)⁻¹ and mu = Sigma·Xᵀy/sigma2.`,
        starter: `
import numpy as np

def beta_update(a, b, successes, failures):
    # TODO
    pass

def beta_mean_var(a, b):
    # TODO
    pass

def bayes_linreg(X, y, sigma2, tau2):
    # TODO
    pass
`,
        solution: `
import numpy as np

def beta_update(a, b, successes, failures):
    return a + successes, b + failures

def beta_mean_var(a, b):
    s = a + b
    return a / s, a * b / (s ** 2 * (s + 1))

def bayes_linreg(X, y, sigma2, tau2):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    Sigma = np.linalg.inv(X.T @ X / sigma2 + np.eye(X.shape[1]) / tau2)
    mu = Sigma @ X.T @ y / sigma2
    return mu, Sigma
`,
        hints: ['The Beta posterior just adds the observed counts to a and b.', 'The identity matrix must match the number of features: `np.eye(X.shape[1])`.'],
        explain: `Conjugate updates are sequential and order-free: updating batch by batch gives the same posterior as one update with all the data. In Bayesian regression, a larger tau2 means a weaker prior and the posterior mean approaches ordinary least squares.`,
        tests: [
          { call: 'beta_update(1, 1, 7, 3)', expect: '(8, 4)' },
          { call: 'beta_mean_var(8, 4)', expect: '(2 / 3, 32 / (144 * 13))' },
          { name: 'regression vs reference', setup: 'rng = np.random.default_rng(0)\nX = np.c_[np.ones(15), rng.normal(size=(15, 2))]; y = rng.normal(size=15)', call: 'bayes_linreg(X, y, 0.5, 2.0)', hidden: true },
          { name: 'posterior mean equals ridge', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(30, 3)); y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=30)', code: 'mu, Sigma = bayes_linreg(X, y, 0.25, 4.0)\nridge = np.linalg.solve(X.T @ X + (0.25 / 4.0) * np.eye(3), X.T @ y)\nassert np.allclose(mu, ridge)\nassert np.allclose(Sigma, Sigma.T)', hidden: true }
        ]
      },
      {
        id: 'metropolis-sampler', title: 'A Metropolis Sampler', difficulty: 'medium',
        prompt: `Implement \`metropolis(log_p, x0, n_steps, step, seed=0)\` → \`(samples, acceptance_rate)\` for a one-dimensional target known only up to a constant through \`log_p(x)\`.

Use \`rng = np.random.default_rng(seed)\` and, for each of the n_steps iterations, **in this order**:

1. \`proposal = x + step * rng.normal()\`
2. \`u = rng.random()\`
3. accept (set x = proposal) if \`np.log(u) < log_p(proposal) - log_p(x)\`
4. append the current x (whether or not the move was accepted)

Return the samples as a NumPy array of length n_steps and the fraction of accepted proposals.`,
        starter: `
import numpy as np

def metropolis(log_p, x0, n_steps, step, seed=0):
    # TODO
    pass
`,
        solution: `
import numpy as np

def metropolis(log_p, x0, n_steps, step, seed=0):
    rng = np.random.default_rng(seed)
    x = float(x0)
    lp = log_p(x)
    samples = np.empty(n_steps)
    accepted = 0
    for i in range(n_steps):
        proposal = x + step * rng.normal()
        u = rng.random()
        lp_new = log_p(proposal)
        if np.log(u) < lp_new - lp:
            x, lp = proposal, lp_new
            accepted += 1
        samples[i] = x
    return samples, accepted / n_steps
`,
        hints: ['Remember log_p of the current point so you do not recompute it every step.', 'Proposals outside the support have log_p = −inf and are rejected automatically.'],
        explain: `Working with log-densities avoids underflow, and comparing log(u) with the log ratio is the same as comparing u with the ratio. Because rejected steps repeat the current value, the chain spends more time in high-probability regions — exactly in proportion to the target density.`,
        tests: [
          { name: 'reference run', call: 'metropolis(lambda x: -0.5 * x ** 2, 0.0, 50, 1.0, seed=3)' },
          { name: 'samples a standard normal', code: 's, rate = metropolis(lambda x: -0.5 * x ** 2, 0.0, 20000, 2.4, seed=1)\nassert abs(s[1000:].mean()) < 0.1 and abs(s[1000:].std() - 1) < 0.1, (s.mean(), s.std())\nassert 0.2 < rate < 0.7', hidden: false },
          { name: 'stays inside the support', code: 'lp = lambda x: 0.0 if 0 <= x <= 1 else -np.inf\ns, rate = metropolis(lp, 0.5, 2000, 0.5, seed=0)\nassert s.min() >= 0 and s.max() <= 1', hidden: true },
          { name: 'bigger steps are accepted less often', code: 'r1 = metropolis(lambda x: -0.5 * x ** 2, 0.0, 5000, 0.5, seed=2)[1]\nr2 = metropolis(lambda x: -0.5 * x ** 2, 0.0, 5000, 5.0, seed=2)[1]\nassert r1 > r2', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Gelman et al. — Bayesian Data Analysis (free PDF)', url: 'http://www.stat.columbia.edu/~gelman/book/' },
      { title: 'McElreath — Statistical Rethinking', url: 'https://xcelab.net/rm/' },
      { title: 'Blei, Kucukelbir & McAuliffe — Variational Inference: A Review for Statisticians', url: 'https://arxiv.org/abs/1601.00670' },
      { title: 'PyMC documentation', url: 'https://www.pymc.io/' }
    ]
  },

  // =====================================================================================
  {
    id: 'gaussian-processes', branch: 'ml', title: 'Gaussian Processes & Bayesian Optimization', short: 'Gaussian Processes', level: 3, minutes: 45, prereqs: ['bayesian-methods', 'svm'],
    tags: ['gaussian process', 'kernel', 'rbf kernel', 'length scale', 'posterior mean', 'posterior variance', 'marginal likelihood', 'bayesian optimization', 'acquisition function', 'expected improvement', 'upper confidence bound', 'hyperparameter tuning'],
    summary: 'A distribution over functions: Gaussian processes defined by a kernel, exact predictions with error bars, how the length scale and noise shape the fit, choosing kernel hyperparameters by marginal likelihood — and Bayesian optimization, which uses a GP to find the best hyperparameters (or any expensive setting) in a handful of trials.',
    why: `Some functions are expensive to evaluate: one training run for one hyperparameter setting can take hours. **Gaussian processes** model an unknown function from a few observations and say how uncertain they are everywhere else; **Bayesian optimization** uses that uncertainty to decide what to try next. GPs are also the cleanest example of a model with calibrated uncertainty, a workhorse for small-data regression in science and engineering, and the place where kernels — first met in SVMs — become a prior over functions.`,
    objectives: [
      'Explain a GP as a prior over functions defined by a mean and a kernel',
      'Compute the GP posterior mean and variance, and interpret the length scale and noise',
      'Describe how kernel hyperparameters are chosen by maximizing the marginal likelihood',
      'Run Bayesian optimization with an acquisition function such as UCB or expected improvement'
    ],
    sections: [
      { type: 'text', title: 'A prior over functions', md: `A **Gaussian process** says: for any finite set of inputs, the function values there are jointly Gaussian, with covariances given by a **kernel** k(x, x′) — a measure of how similar two inputs are. Similar inputs get strongly correlated outputs, which is what makes the functions smooth. The most common kernel is the RBF (squared-exponential) kernel:

$$ k(x, x′) = σ_f² · exp( −‖x − x′‖² / (2ℓ²) ) $$
:::where
x, x′ — two inputs
‖x − x′‖ — the distance between them
ℓ — "ell": the length scale, how far apart inputs must be before their function values become unrelated
σ_f² — the signal variance, how far the function typically strays from its mean
:::

Other kernels encode other assumptions: Matérn kernels for rougher functions, periodic kernels for repeating patterns, linear kernels for trends. Kernels can be added (either pattern) or multiplied (both at once).` },
      { type: 'code', title: 'Functions drawn from a GP prior', code: `
import numpy as np
rng = np.random.default_rng(0)
x = np.linspace(0, 10, 200)

def rbf(a, b, length, var=1.0):
    return var * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / length ** 2)

for length in (0.3, 1.0, 3.0):
    K = rbf(x, x, length) + 1e-6 * np.eye(len(x))          # a tiny "jitter" keeps the matrix numerically positive definite
    L = np.linalg.cholesky(K)
    f = L @ rng.normal(size=(len(x), 3))                    # three random functions with covariance K
    crossings = np.mean(np.sum(np.diff(np.sign(f), axis=0) != 0, axis=0))
    print(f"length scale {length:3.1f}: values between {f.min():+.2f} and {f.max():+.2f}; each curve crosses zero {crossings:4.1f} times on average")
`, after: `Multiplying a matrix square root of K (the Cholesky factor) by independent normal noise produces values with covariance K. Short length scales give wiggly functions that cross zero often; long ones give smooth, slowly varying curves — the kernel is the GP's entire set of assumptions about the function.` },
      { type: 'text', title: 'Conditioning on data: the GP posterior', md: `Given noisy observations y at training inputs X, the function values at test inputs X* are still Gaussian, with a mean and covariance that have closed forms:

$$ μ* = K*ᵀ ( K + σ_n² I )⁻¹ y          Σ* = K** − K*ᵀ ( K + σ_n² I )⁻¹ K* $$
:::where
K — the kernel matrix between the n training inputs (n × n)
K* — the kernel values between training and test inputs (n × m)
K** — the kernel matrix between the test inputs (m × m)
σ_n² — the noise variance of the observations
y — the observed training targets
μ*, Σ* — the posterior mean and covariance of the function at the test inputs
:::

The mean is a weighted combination of the training targets, with weights decided by kernel similarity. The variance starts at the prior variance and is reduced by every nearby observation — so it is small near the data and returns to the prior far away. Exact GPs cost O(n³) time (solving with an n × n matrix, done via a Cholesky factorization), which limits them to roughly ten thousand points; sparse "inducing point" approximations go further.` },
      { type: 'code', title: 'GP regression from scratch (and with scikit-learn)', code: `
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

rng = np.random.default_rng(2)
X = np.array([0.5, 1.5, 2.0, 3.5, 4.0, 6.0])
y = np.sin(X) + rng.normal(0, 0.1, len(X))
Xs = np.array([1.0, 3.0, 5.0, 8.0, 15.0])                 # test inputs, some far from the data
length, var, noise = 1.0, 1.0, 0.1 ** 2

def rbf(a, b):
    return var * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / length ** 2)

K = rbf(X, X) + noise * np.eye(len(X))
Ks, Kss = rbf(X, Xs), rbf(Xs, Xs)
mean = Ks.T @ np.linalg.solve(K, y)
std = np.sqrt(np.diag(Kss - Ks.T @ np.linalg.solve(K, Ks)))

kernel = ConstantKernel(var, "fixed") * RBF(length, "fixed") + WhiteKernel(noise, "fixed")
sk_mean = GaussianProcessRegressor(kernel, optimizer=None).fit(X[:, None], y).predict(Xs[:, None])
for x, m, s, sm in zip(Xs, mean, std, sk_mean):
    print(f"x = {x:4.1f}: mean {m:+.3f} ± {2 * s:.3f}   (sin x = {np.sin(x):+.3f})   scikit-learn mean {sm:+.3f}")
`, after: `Close to the training points (x = 1) the mean is near sin(x) and the band is tight; in the gaps between points (x = 3 and 5) the band widens, and the true value stays inside it. At x = 8 and 15, far from all data, the mean falls back to the prior mean of 0 and the band widens to the prior's ±2 — the GP knows it is guessing. scikit-learn computes the same mean.` },
      { type: 'text', title: 'Choosing the kernel: the marginal likelihood', md: `How should ℓ, σ_f and σ_n be chosen? A GP can score each setting by how probable it makes the observed data, integrating over all functions:

$$ log p(y | X) = −½ yᵀ(K + σ_n² I)⁻¹ y − ½ log det(K + σ_n² I) − (n/2) log 2π $$
:::where
the first term — data fit, high when the kernel explains the targets well
the second term — a complexity penalty: flexible kernels (short length scales) spread probability over many possible functions and pay for it
det — the determinant of a matrix
n — the number of training points
:::

Maximizing it with gradient-based optimizers (what scikit-learn does by default) gives an automatic Occam's razor: a kernel that is too flexible is penalized even though it fits the training data better.` },
      { type: 'demo', title: 'Interactive: fit a Gaussian process by clicking', demo: 'gp' },
      { type: 'check', question: { type: 'mcq', q: 'Far away from all training points, what does a GP with an RBF kernel and zero prior mean predict?', options: ['The value of the nearest training point', 'The prior mean (0), with the prior\'s full uncertainty', 'A straight-line extrapolation', 'An error'], answer: 1, explain: 'The kernel correlation vanishes, so the data no longer informs the prediction.' } },
      { type: 'text', title: 'Bayesian optimization', md: `Suppose f(x) is a validation score as a function of hyperparameters, and each evaluation costs a full training run. **Bayesian optimization** loops:

1. Fit a GP (or another probabilistic surrogate) to all evaluations so far.
2. Choose the next x by maximizing an **acquisition function** that balances exploitation (a high predicted mean) and exploration (high uncertainty).
3. Evaluate f there and repeat.

Two common acquisition functions (for maximization): the **upper confidence bound** UCB(x) = μ(x) + κ·σ(x), and **expected improvement**:

$$ EI(x) = ( μ(x) − f* ) · Φ(Z) + σ(x) · φ(Z)          Z = ( μ(x) − f* ) / σ(x) $$
:::where
μ(x), σ(x) — the surrogate's predicted mean and standard deviation at x
f* — the best value observed so far
Φ, φ — the cumulative distribution and density functions of the standard normal distribution
EI(x) — the expected amount by which x would beat f*, large when the mean is promising or the uncertainty is large
:::

In practice: Optuna's TPE, scikit-optimize and Ax implement such loops; **random search** is a strong baseline in high dimensions; and **successive halving / Hyperband** save compute by stopping unpromising runs early.` },
      { type: 'code', title: 'Bayesian optimization versus random search', code: `
import numpy as np
rng = np.random.default_rng(0)

def objective(x):                          # pretend each call is an expensive training run
    return -(x - 2.0) ** 2 + np.sin(5 * x)

grid = np.linspace(0, 5, 501)

def gp_posterior(X, y, Xs, length=0.5, var=4.0, noise=1e-4):
    k = lambda a, b: var * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / length ** 2)
    K = k(X, X) + noise * np.eye(len(X))
    Ks = k(X, Xs)
    m = y.mean()                                           # use the data mean as the prior mean
    mean = m + Ks.T @ np.linalg.solve(K, y - m)
    var_s = var - np.sum(Ks * np.linalg.solve(K, Ks), axis=0)
    return mean, np.sqrt(np.maximum(var_s, 1e-12))

X = rng.uniform(0, 5, 3)                                   # three random starting points
y = objective(X)
for it in range(9):
    mean, sd = gp_posterior(X, y, grid)
    x_next = grid[np.argmax(mean + 2.0 * sd)]              # upper confidence bound: optimism under uncertainty
    X, y = np.append(X, x_next), np.append(y, objective(x_next))
    print(f"evaluation {len(X):2d}: tried x = {x_next:.2f}, best so far {y.max():.3f}")

random_x = rng.uniform(0, 5, 12)
print(f"Bayesian optimization, 12 evaluations: best {y.max():.3f} at x = {X[np.argmax(y)]:.2f}")
print(f"random search, 12 evaluations:         best {objective(random_x).max():.3f}")
print(f"true maximum on the grid:              {objective(grid).max():.3f} at x = {grid[np.argmax(objective(grid))]:.2f}")
`, after: `After a few exploratory evaluations — including the edge x = 5, where the uncertainty was largest — the GP homes in on the peak near x = 1.6 and finds the true maximum by the 9th evaluation, while random search with the same budget gets nowhere near it. Once the optimum is found, UCB keeps proposing the same point; real implementations notice repeats and explore elsewhere instead.` }
    ],
    when: `- Expensive black-box optimization: hyperparameters, experimental design, A/B-test parameters, chemistry and materials searches.
- Small-data regression where calibrated error bars matter (science, engineering, sensor data).
- Interpolating smooth functions and modelling spatial or temporal correlation (kriging in geostatistics).
- Not for large datasets without approximations (exact GPs scale as n³), or for very high-dimensional inputs.`,
    pitfalls: `- Forgetting to scale inputs and outputs — length scales and variances then mean nothing.
- Numerical failures in the Cholesky factorization; add a small jitter to the diagonal.
- Trusting a GP's uncertainty when the kernel is wrong for the function (e.g. RBF for a function with jumps).
- Running Bayesian optimization in dozens of dimensions, where random search or Hyperband often do as well.
- Overfitting the kernel hyperparameters with too few points.`,
    takeaways: [
      'A GP is a prior over functions: any finite set of values is jointly Gaussian with kernel covariances.',
      'The posterior mean interpolates the data; the variance shrinks near data and returns to the prior far away.',
      'The length scale controls smoothness; the marginal likelihood picks kernel hyperparameters with a built-in complexity penalty.',
      'Exact GPs cost O(n³), limiting them to modest datasets.',
      'Bayesian optimization uses the GP\'s uncertainty (UCB, expected improvement) to choose the next expensive evaluation.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does a short RBF length scale imply about the functions a GP expects?', options: ['Very smooth, slowly varying functions', 'Wiggly functions whose values decorrelate quickly', 'Linear functions', 'Periodic functions'], answer: 1, explain: 'Inputs need only be a little apart before their values become unrelated.' },
      { type: 'tf', q: 'The GP posterior variance at a test point depends on the observed target values y.', answer: false, explain: 'Σ* depends only on the inputs and the kernel; only the mean uses y.' },
      { type: 'mcq', q: 'Why does exact GP regression not scale to millions of points?', options: ['Kernels are slow to evaluate', 'Solving with the n × n kernel matrix costs O(n³)', 'GPs need labels for every point', 'GPs cannot use GPUs'], answer: 1, explain: 'Cubic cost and quadratic memory in n.' },
      { type: 'mcq', q: 'In Bayesian optimization, what is the acquisition function for?', options: ['Training the final model', 'Choosing the next point to evaluate by balancing high predicted value and high uncertainty', 'Measuring test accuracy', 'Normalizing the data'], answer: 1, explain: 'It trades exploitation against exploration.' },
      { type: 'fill', q: 'UCB = μ + κσ with κ = 2. At a point with μ = 0.5 and σ = 0.3, what is the UCB value?', answer: ['1.1', '1.10'], explain: '0.5 + 2 × 0.3 = 1.1.' }
    ],
    problems: [
      {
        id: 'gp-regression', title: 'Gaussian Process Regression', difficulty: 'medium',
        prompt: `Implement exact GP regression with an RBF kernel and zero prior mean:

- \`rbf_kernel(A, B, length=1.0, var=1.0)\` → the (n, m) matrix var·exp(−‖a_i − b_j‖² / (2·length²)) for row vectors A (n, d) and B (m, d).
- \`gp_posterior(X, y, Xs, length=1.0, var=1.0, noise=1e-2)\` → \`(mean, std)\` at the test inputs Xs: mean = K*ᵀ(K + noise·I)⁻¹y and std = the square root of the diagonal of K** − K*ᵀ(K + noise·I)⁻¹K* (the uncertainty of the function itself; clip tiny negative variances to 0).

Use \`np.linalg.solve\` instead of an explicit inverse.`,
        starter: `
import numpy as np

def rbf_kernel(A, B, length=1.0, var=1.0):
    # TODO
    pass

def gp_posterior(X, y, Xs, length=1.0, var=1.0, noise=1e-2):
    # TODO
    pass
`,
        solution: `
import numpy as np

def rbf_kernel(A, B, length=1.0, var=1.0):
    A, B = np.atleast_2d(np.asarray(A, dtype=float)), np.atleast_2d(np.asarray(B, dtype=float))
    d2 = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * A @ B.T
    return var * np.exp(-0.5 * np.maximum(d2, 0) / length ** 2)

def gp_posterior(X, y, Xs, length=1.0, var=1.0, noise=1e-2):
    K = rbf_kernel(X, X, length, var) + noise * np.eye(len(X))
    Ks = rbf_kernel(X, Xs, length, var)
    Kss = rbf_kernel(Xs, Xs, length, var)
    mean = Ks.T @ np.linalg.solve(K, np.asarray(y, dtype=float))
    cov = Kss - Ks.T @ np.linalg.solve(K, Ks)
    return mean, np.sqrt(np.maximum(np.diag(cov), 0))
`,
        hints: ['Squared distances: ‖a‖² + ‖b‖² − 2a·b, computed for all pairs with broadcasting.', 'Only the diagonal of the posterior covariance is needed for the standard deviations.'],
        explain: `Solving the linear system once for many right-hand sides is the standard trick; production code factors K + noise·I with a Cholesky decomposition and reuses the factor for the mean, the variance and the marginal likelihood.`,
        tests: [
          { call: 'rbf_kernel(np.array([[0.0], [1.0]]), np.array([[0.0], [2.0]]))', expect: 'np.array([[1.0, np.exp(-2.0)], [np.exp(-0.5), np.exp(-0.5)]])' },
          { name: 'far from the data: prior mean and prior uncertainty', call: 'gp_posterior(np.array([[0.0], [1.0]]), np.array([1.0, -1.0]), np.array([[100.0]]), length=1.0, var=4.0)', expect: '(np.array([0.0]), np.array([2.0]))' },
          { name: 'random data vs reference', setup: 'rng = np.random.default_rng(0)\nX = rng.uniform(-3, 3, (8, 2)); y = np.sin(X[:, 0]) + X[:, 1]; Xs = rng.uniform(-3, 3, (5, 2))', call: 'gp_posterior(X, y, Xs, length=1.5, var=2.0, noise=0.05)', hidden: true },
          { name: 'interpolates with tiny noise', setup: 'X = np.array([[0.0], [1.0], [2.5]]); y = np.array([0.3, -0.2, 1.0])', code: 'm, s = gp_posterior(X, y, X, noise=1e-8)\nassert np.allclose(m, y, atol=1e-4)\nassert np.all(s < 1e-3)', hidden: true }
        ]
      },
      {
        id: 'acquisition-functions', title: 'Acquisition Functions', difficulty: 'medium',
        prompt: `Implement the acquisition functions of Bayesian optimization (for **maximization**). Inputs are NumPy arrays of the same length.

- \`expected_improvement(mu, sigma, best)\` → (mu − best)·Φ(z) + sigma·φ(z) with z = (mu − best)/sigma. Where sigma is 0, return max(mu − best, 0). Φ is the standard normal CDF, 0.5·(1 + erf(z/√2)) (use \`math.erf\`), and φ(z) = exp(−z²/2)/√(2π).
- \`ucb(mu, sigma, kappa=2.0)\` → mu + kappa·sigma.
- \`propose(xs, mu, sigma, best)\` → the element of xs with the highest expected improvement (the first one if tied).`,
        starter: `
import math
import numpy as np

def expected_improvement(mu, sigma, best):
    # TODO
    pass

def ucb(mu, sigma, kappa=2.0):
    # TODO
    pass

def propose(xs, mu, sigma, best):
    # TODO
    pass
`,
        solution: `
import math
import numpy as np

def expected_improvement(mu, sigma, best):
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    out = np.maximum(mu - best, 0.0)
    for i in range(len(mu)):
        if sigma[i] > 0:
            z = (mu[i] - best) / sigma[i]
            cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            out[i] = (mu[i] - best) * cdf + sigma[i] * pdf
    return out

def ucb(mu, sigma, kappa=2.0):
    return np.asarray(mu, dtype=float) + kappa * np.asarray(sigma, dtype=float)

def propose(xs, mu, sigma, best):
    return xs[int(np.argmax(expected_improvement(mu, sigma, best)))]
`,
        hints: ['`math.erf` works on single numbers; loop over the entries (or use `np.vectorize`).', 'With sigma = 0 there is no uncertainty: the improvement is simply max(mu − best, 0).'],
        explain: `Expected improvement is never negative, rewards points whose mean beats the incumbent, and also rewards uncertain points — even one with a mediocre mean can have a decent chance of a large improvement. UCB makes the same trade-off with an explicit knob, kappa.`,
        tests: [
          { call: 'expected_improvement(np.array([0.0]), np.array([1.0]), 0.0)', expect: 'np.array([1 / np.sqrt(2 * np.pi)])' },
          { call: 'expected_improvement(np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.5)', expect: 'np.array([0.5, 0.0])' },
          { call: 'ucb(np.array([1.0, 2.0]), np.array([0.5, 0.1]))', expect: 'np.array([2.0, 2.2])' },
          { name: 'random inputs vs reference', setup: 'rng = np.random.default_rng(0)\nmu = rng.normal(size=6); sigma = rng.uniform(0.1, 1.0, 6)', call: 'expected_improvement(mu, sigma, 0.4)', hidden: true },
          { name: 'uncertainty wins when means are equal', call: 'propose(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.8, 0.3]), 0.6)', expect: '2.0', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Rasmussen & Williams — Gaussian Processes for Machine Learning (free book)', url: 'https://gaussianprocess.org/gpml/' },
      { title: 'Görtler, Kehlbeck & Deussen — A Visual Exploration of Gaussian Processes (Distill)', url: 'https://distill.pub/2019/visual-exploration-gaussian-processes/' },
      { title: 'Snoek, Larochelle & Adams — Practical Bayesian Optimization of Machine Learning Algorithms', url: 'https://arxiv.org/abs/1206.2944' },
      { title: 'Optuna documentation', url: 'https://optuna.org/' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
