/* ML Atlas – content: extra foundations (eigenvectors & SVD, MLE & MAP, statistical inference, pandas & EDA, Gaussian mixtures & EM) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'eigen-svd', branch: 'math', title: 'Eigenvectors, Eigenvalues & the SVD', short: 'Eigenvectors & SVD', level: 2, minutes: 45, prereqs: ['vectors-matrices'],
    tags: ['eigenvector', 'eigenvalue', 'svd', 'singular value decomposition', 'rank', 'low-rank approximation', 'power iteration', 'linear algebra'],
    summary: 'What a matrix does to space, the special directions it only stretches (eigenvectors), and the singular value decomposition that writes any matrix as rotate–stretch–rotate — the mathematics behind PCA, low-rank approximation, LoRA and recommender systems.',
    why: `Many ML ideas are really statements about the structure of a matrix. PCA looks for the directions in which data varies most; LoRA assumes a weight update is "low rank"; matrix factorization compresses a ratings table; RNN gradients explode or vanish depending on how much a weight matrix stretches vectors. **Eigenvectors** and the **singular value decomposition (SVD)** are the tools that make "important directions" and "rank" precise, so these lessons stop being recipes and start making sense.`,
    objectives: [
      'See a matrix as a transformation that rotates, stretches and projects vectors',
      'Find and interpret eigenvectors and eigenvalues, and compute the top one with power iteration',
      'Decompose any matrix with the SVD and interpret U, S and V',
      'Build the best rank-k approximation and explain why "low rank" means "compressible"'
    ],
    sections: [
      { type: 'text', title: 'A matrix is a transformation', md: `Multiplying a vector by a matrix produces a new vector: the matrix **moves space around** — it can rotate, stretch, squash, shear or project it. The columns of the matrix show where the basis vectors [1, 0] and [0, 1] end up.

| Matrix | What it does |
| --- | --- |
| [[2, 0], [0, 0.5]] | stretches x by 2, squashes y to half |
| [[0, −1], [1, 0]] | rotates by 90° |
| [[1, 1], [0, 1]] | shears: slides the top of the plane sideways |
| [[1, 0], [0, 0]] | projects onto the x-axis (throws y away) |

Most vectors change **direction** when you multiply them by a matrix. A few special directions don't — and they reveal what the matrix really does.` },
      { type: 'text', title: 'Eigenvectors: directions that only get stretched', md: `An **eigenvector** of a square matrix is a direction that the matrix only stretches (or flips), never turns:

$$ A v = λ v $$
:::where
A — a square matrix (n × n)
v — an eigenvector: a non-zero vector whose direction A leaves unchanged
λ — "lambda": the eigenvalue, the factor by which A stretches v (a negative λ flips it, |λ| < 1 shrinks it)
:::

Example: for A = [[2, 1], [1, 2]], the vector [1, 1] becomes [3, 3] = 3·[1, 1], so it is an eigenvector with λ = 3; [1, −1] becomes [1, −1], an eigenvector with λ = 1.

Why it matters:
- **Symmetric matrices** (such as covariance matrices) have real eigenvalues and mutually perpendicular eigenvectors. For a covariance matrix those eigenvectors are exactly the **principal axes** of the data — PCA.
- **Repeated multiplication**: A^k v = λ^k v. The direction with the largest |λ| eventually dominates. That is the idea behind **power iteration** (and PageRank), and why recurrent networks explode when |λ| > 1 and vanish when |λ| < 1.` },
      { type: 'code', title: 'Eigenvectors with NumPy and power iteration', code: `
import numpy as np
A = np.array([[2.0, 1.0], [1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)              # eigh: for symmetric matrices; eigenvalues in ascending order
print("eigenvalues:", vals)
print("eigenvectors (one per column):")
print(vecs.round(3))
v = vecs[:, 1]
print("A v =", (A @ v).round(3), "   λ v =", (vals[1] * v).round(3))

# power iteration: multiply and normalize again and again -> converges to the top eigenvector
x = np.array([1.0, 0.0])
for step in range(20):
    x = A @ x
    x = x / np.linalg.norm(x)
print("power iteration finds:", x.round(4), "  eigenvalue estimate x·Ax:", round(float(x @ A @ x), 4))
`, after: `Power iteration only needs matrix–vector products, which is why it scales to enormous sparse matrices such as the web graph.` },
      { type: 'check', question: { type: 'mcq', q: 'If A v = 3v, what does the matrix A do to the vector v?', options: ['Rotates it by 3 degrees', 'Stretches it by a factor of 3 without changing its direction', 'Adds 3 to every element', 'Nothing'], answer: 1, explain: 'That is the definition of an eigenvector with eigenvalue 3.' } },
      { type: 'text', title: 'The singular value decomposition', md: `Eigenvectors only exist for square matrices, and only symmetric ones behave nicely. The **SVD** works for **every** matrix, of any shape:

$$ A = U S Vᵀ $$
:::where
A — any m × n matrix
U — an m × r matrix with orthonormal columns: the output directions
S — the singular values σ₁ ≥ σ₂ ≥ … ≥ 0 on a diagonal: the stretch factors
Vᵀ — an r × n matrix with orthonormal rows: the input directions
r — min(m, n) for the "thin" SVD
:::

Read it right to left: **rotate** the input (Vᵀ), **stretch** along the axes (S), **rotate** into the output space (U). Every linear map is just that.

Connections: the singular values are the square roots of the eigenvalues of AᵀA, and the columns of V are its eigenvectors. For centred data X, AᵀA is (proportional to) the covariance matrix — so the SVD of the data *is* PCA.` },
      { type: 'text', title: 'Rank and low-rank approximation', md: `The **rank** of a matrix is its number of non-zero singular values: how many independent directions it really uses. Writing the SVD as a sum of rank-1 pieces and keeping only the largest k gives the **best possible** rank-k approximation (the Eckart–Young theorem):

$$ A_k = Σ_{i=1..k} σ_i · u_i v_iᵀ $$
$$ ‖ A − A_k ‖²_F = Σ_{i>k} σ_i² $$
:::where
σ_i — the i-th singular value
u_i — the i-th column of U; v_i — the i-th column of V
u_i v_iᵀ — an outer product: a rank-1 matrix
A_k — the best rank-k approximation of A
‖ … ‖_F — the Frobenius norm: the square root of the sum of all squared entries
Σ_{i>k} σ_i² — the squared error equals the energy in the singular values you dropped
:::

Storing U_k, the k singular values and V_k takes k·(m + n + 1) numbers instead of m·n. When the singular values drop off quickly, the matrix is **compressible**. That is the assumption behind PCA (keep the top components), LoRA (weight updates are well approximated with small rank), recommender systems (ratings ≈ user factors × item factors), image compression and denoising.` },
      { type: 'code', title: 'Compressing a matrix with the SVD', code: `
import numpy as np
rng = np.random.default_rng(0)
x, y = np.linspace(0, 3, 40), np.linspace(0, 2, 60)
img = np.outer(np.sin(2 * y), np.cos(x)) + 0.5 * np.outer(y, np.ones_like(x))   # a rank-2 pattern
img = img + 0.05 * rng.normal(size=img.shape)                                     # plus a little noise

U, S, Vt = np.linalg.svd(img, full_matrices=False)
print("largest singular values:", S[:6].round(2))
for k in (1, 2, 3, 10):
    approx = (U[:, :k] * S[:k]) @ Vt[:k]
    err = np.linalg.norm(img - approx) / np.linalg.norm(img)
    print(f"rank {k:2d}: relative error {err:.3f}   numbers stored {k * (img.shape[0] + img.shape[1] + 1):5d} of {img.size}")
`, after: `Two singular values are large and the rest are noise, so a rank-2 approximation already captures the pattern almost perfectly with a fraction of the numbers.` },
      { type: 'check', question: { type: 'tf', q: 'Every matrix, even a non-square one, has a singular value decomposition.', answer: true, explain: 'Unlike eigen-decomposition, the SVD exists for any m × n matrix.' } }
    ],
    when: `- **PCA** and dimensionality reduction (via the SVD of centred data).
- Compressing or analysing weights: low-rank layers, LoRA, model pruning.
- Latent-factor models: recommender systems, latent semantic analysis of documents.
- Solving least squares stably (the pseudo-inverse) and diagnosing ill-conditioned problems.
- Stability analysis: eigenvalues of recurrent weight matrices, of Hessians in optimization, of graph Laplacians.`,
    pitfalls: `- Using \`np.linalg.eig\` on symmetric matrices — \`eigh\` is faster, more accurate and returns sorted real eigenvalues.
- Comparing signs of eigenvectors or singular vectors: −v is just as valid as v.
- Forgetting to centre the data before using the SVD for PCA.
- Computing a full SVD of a huge matrix when a truncated or randomized SVD of the top k components would do.
- Mixing up V and Vᵀ: NumPy returns Vᵀ (rows are the directions).`,
    takeaways: [
      'A matrix is a transformation; eigenvectors are the directions it only stretches: A v = λ v.',
      'Symmetric matrices have perpendicular eigenvectors — for a covariance matrix they are the principal axes.',
      'Every matrix has an SVD A = U S Vᵀ: rotate, stretch, rotate.',
      'The rank is the number of non-zero singular values; keeping the top k gives the best rank-k approximation.',
      'Fast-decaying singular values mean the matrix is compressible — the idea behind PCA, LoRA and matrix factorization.'
    ],
    quiz: [
      { type: 'mcq', q: 'For A = [[3, 0], [0, 1]], which vector is an eigenvector with eigenvalue 3?', options: ['[0, 1]', '[1, 0]', '[1, 1]', '[3, 1]'], answer: 1, explain: 'A·[1, 0] = [3, 0] = 3·[1, 0].' },
      { type: 'fill', q: 'A 1000 × 500 matrix is stored as a rank-20 approximation (U_k, the singular values and V_k). How many numbers is that?', answer: ['30020', '30,020'], explain: '20 × (1000 + 500 + 1) = 30,020 instead of 500,000.' },
      { type: 'mcq', q: 'What does the number of non-zero singular values tell you?', options: ['The size of the matrix', 'Its rank: how many independent directions it really uses', 'Its largest entry', 'Its trace'], answer: 1, explain: 'Zero singular values mean collapsed directions.' },
      { type: 'tf', q: 'If a matrix\'s singular values decay quickly, a low-rank approximation of it will be accurate.', answer: true, explain: 'The error equals the energy in the dropped singular values.' },
      { type: 'mcq', q: 'Why does LoRA relate to the SVD?', options: ['It computes an SVD at every training step', 'It assumes the weight update has low rank — only a few significant singular values', 'It uses eigenvalues as learning rates', 'It does not relate'], answer: 1, explain: 'B·A with small inner dimension r is a rank-r matrix.' }
    ],
    problems: [
      {
        id: 'power-iteration', title: 'Power Iteration', difficulty: 'medium',
        prompt: `Implement \`power_iteration(A, steps=100, seed=0)\` → \`(eigenvalue, eigenvector)\` for a square matrix whose largest-magnitude eigenvalue is unique:

1. \`rng = np.random.default_rng(seed)\`; start from \`x = rng.normal(size=n)\` and normalize it to length 1.
2. Repeat \`steps\` times: \`x = A @ x\`, then normalize again.
3. Make the result's sign deterministic: if the component with the largest absolute value is negative, flip x.
4. The eigenvalue estimate is the Rayleigh quotient x · A x. Return it as a float, together with x.

\`\`\`text
power_iteration(np.array([[2.0, 1.0], [1.0, 2.0]]))  ->  (3.0, [0.7071, 0.7071])
\`\`\``,
        starter: `
import numpy as np

def power_iteration(A, steps=100, seed=0):
    A = np.asarray(A, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def power_iteration(A, steps=100, seed=0):
    A = np.asarray(A, dtype=float)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=A.shape[0])
    x = x / np.linalg.norm(x)
    for _ in range(steps):
        x = A @ x
        x = x / np.linalg.norm(x)
    if x[np.argmax(np.abs(x))] < 0:
        x = -x
    return float(x @ A @ x), x
`,
        hints: ['Normalize with `x / np.linalg.norm(x)` after every multiplication.', '`np.argmax(np.abs(x))` finds the largest-magnitude component.', 'Since x has length 1, x · A x is the eigenvalue estimate.'],
        explain: `Every multiplication amplifies the component along the top eigenvector by |λ₁| and the others by smaller factors, so their ratio shrinks like (|λ₂|/|λ₁|)^steps. A small gap between the top two eigenvalues means slow convergence.`,
        tests: [
          { call: 'power_iteration(np.array([[2.0, 1.0], [1.0, 2.0]]))', expect: '(3.0, np.array([1.0, 1.0]) / np.sqrt(2))' },
          { name: 'diagonal matrix', call: 'power_iteration(np.diag([1.0, 5.0, 2.0]))', expect: '(5.0, np.array([0.0, 1.0, 0.0]))' },
          { name: 'matches numpy on a random symmetric matrix', setup: 'rng = np.random.default_rng(0)\nM = rng.normal(size=(6, 6))\nA = M @ M.T', code: 'lam, v = power_iteration(A, steps=500)\nvals, vecs = np.linalg.eigh(A)\nassert abs(lam - vals[-1]) < 1e-6 * vals[-1]\nassert abs(abs(v @ vecs[:, -1]) - 1) < 1e-6', hidden: true },
          { name: 'reference run with another seed', setup: 'rng = np.random.default_rng(2)\nM = rng.normal(size=(4, 4))\nA = M + M.T + 8 * np.eye(4)', call: 'power_iteration(A, steps=60, seed=3)', hidden: true }
        ]
      },
      {
        id: 'low-rank-approx', title: 'Low-Rank Approximation', difficulty: 'medium',
        prompt: `Using \`np.linalg.svd(A, full_matrices=False)\`, implement:

- \`low_rank(A, k)\` → the best rank-k approximation U_k · diag(S_k) · V_kᵀ
- \`relative_error(A, k)\` → ‖A − A_k‖_F / ‖A‖_F as a float (\`np.linalg.norm\` of a matrix is the Frobenius norm)
- \`rank_for_energy(A, energy=0.9)\` → the smallest k such that the top k singular values hold at least the fraction \`energy\` of Σσ²

\`\`\`text
relative_error(np.diag([3.0, 4.0]), 1)      ->  0.6     (keeps 4, drops 3: 3/5)
rank_for_energy(np.diag([3.0, 4.0]), 0.6)   ->  1       (16/25 = 0.64 ≥ 0.6)
\`\`\``,
        starter: `
import numpy as np

def low_rank(A, k):
    # TODO
    pass

def relative_error(A, k):
    # TODO
    pass

def rank_for_energy(A, energy=0.9):
    # TODO
    pass
`,
        solution: `
import numpy as np

def low_rank(A, k):
    U, S, Vt = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]

def relative_error(A, k):
    A = np.asarray(A, dtype=float)
    return float(np.linalg.norm(A - low_rank(A, k)) / np.linalg.norm(A))

def rank_for_energy(A, energy=0.9):
    S = np.linalg.svd(np.asarray(A, dtype=float), compute_uv=False)
    fraction = np.cumsum(S ** 2) / np.sum(S ** 2)
    return int(np.searchsorted(fraction, energy - 1e-12) + 1)
`,
        hints: ['`(U[:, :k] * S[:k]) @ Vt[:k]` scales the first k columns of U by the singular values.', '`np.cumsum(S ** 2) / np.sum(S ** 2)` is the captured energy for k = 1, 2, …', '`np.searchsorted` finds the first position that reaches the target.'],
        explain: `The SVD orders directions by importance, so truncating it is optimal — no other rank-k matrix is closer in squared error. "Energy" (Σσ²) is the same quantity PCA reports as explained variance.`,
        tests: [
          { name: 'a rank-1 matrix is reproduced exactly', call: 'low_rank(np.outer([1.0, 2.0, 3.0], [1.0, -1.0]), 1)', expect: 'np.outer([1.0, 2.0, 3.0], [1.0, -1.0])' },
          { call: 'relative_error(np.diag([3.0, 4.0]), 1)', expect: '0.6' },
          { call: '[rank_for_energy(np.diag([3.0, 4.0]), 0.6), rank_for_energy(np.diag([3.0, 4.0]), 0.9)]', expect: '[1, 2]' },
          { name: 'error equals the dropped energy', setup: 'rng = np.random.default_rng(0)\nA = rng.normal(size=(12, 7))', code: 'S = np.linalg.svd(A, compute_uv=False)\nfor k in range(1, 7):\n    assert abs(relative_error(A, k) - np.sqrt(np.sum(S[k:] ** 2) / np.sum(S ** 2))) < 1e-9', hidden: true },
          { name: 'random matrix vs reference', setup: 'rng = np.random.default_rng(1)\nA = rng.normal(size=(8, 5)) @ rng.normal(size=(5, 6))', call: '[low_rank(A, 2), rank_for_energy(A, 0.95)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: '3Blue1Brown — Essence of Linear Algebra (videos)', url: 'https://www.3blue1brown.com/topics/linear-algebra' },
      { title: 'Gilbert Strang — Linear Algebra and Learning from Data', url: 'https://math.mit.edu/~gs/learningfromdata/' },
      { title: 'Steve Brunton — Singular Value Decomposition (video series)', url: 'https://www.youtube.com/watch?v=gXbThCXjZFM' }
    ]
  },

  // =====================================================================================
  {
    id: 'mle-map', branch: 'ml', title: 'Maximum Likelihood & MAP Estimation', short: 'MLE & MAP', level: 2, minutes: 45, prereqs: ['probability', 'loss-functions'],
    tags: ['maximum likelihood', 'mle', 'map', 'likelihood', 'log-likelihood', 'negative log-likelihood', 'prior', 'posterior', 'regularization', 'bayesian'],
    summary: 'Where loss functions come from: the likelihood of parameters given data, maximum-likelihood estimation, why squared error and cross-entropy are negative log-likelihoods, and how adding a prior turns MLE into MAP estimation — which is exactly L2 or L1 regularization.',
    why: `Loss functions can look like arbitrary choices. They aren't: almost every standard loss is the **negative log-likelihood** of a probabilistic model of the data. Squared error assumes Gaussian noise; cross-entropy assumes a categorical distribution; training a language model is maximum-likelihood estimation on text. Add a prior belief about the parameters and you get **MAP** estimation — which turns out to be regularization. This one idea explains a large part of the course.`,
    objectives: [
      'Define the likelihood of parameters given data and compute maximum-likelihood estimates',
      'Show that MSE is the negative log-likelihood under Gaussian noise and cross-entropy under a categorical model',
      'Explain MAP estimation and show that a Gaussian prior gives L2 regularization and a Laplace prior gives L1',
      'Recognize when MLE overfits and why priors help with little data'
    ],
    sections: [
      { type: 'text', title: 'Likelihood: how well do parameters explain the data?', md: `A probabilistic model with parameters θ assigns a probability (or density) p(x | θ) to every possible observation. Given observed data, the **likelihood** asks: how probable is *this* data under *these* parameters? For independent observations:

$$ L(θ) = p(x₁, …, xₙ | θ) = Π_i p(x_i | θ) $$
$$ log L(θ) = Σ_i log p(x_i | θ) $$
:::where
θ — "theta": the model's parameters
x_i — the i-th observed data point (n of them)
p(x_i | θ) — the probability (or density) the model with parameters θ gives to x_i
Π_i — the product over all data points (allowed because they are independent)
log L(θ) — the log-likelihood: a sum instead of a product, which is numerically stable
:::

The likelihood is a function of θ with the data held fixed — it is *not* a probability distribution over θ.

**Maximum-likelihood estimation (MLE)** picks the parameters that make the observed data most probable. Equivalently, it minimizes the **negative log-likelihood (NLL)** — which is why "training" so often means "minimizing an NLL".` },
      { type: 'text', title: 'Example: estimating a coin', md: `Flip a coin n times and see h heads. With θ = P(heads):

$$ log L(θ) = h · log θ + (n − h) · log(1 − θ)          θ̂_MLE = h / n $$
:::where
n — the number of flips
h — the number of heads
θ — the probability of heads we want to estimate
θ̂_MLE — "theta hat": the maximum-likelihood estimate, found by setting the derivative of log L to zero
:::

The MLE is simply the observed fraction — sensible. But with 3 flips and 3 heads it says θ̂ = 1: the coin can *never* land tails. With little data, MLE trusts the sample completely and **overfits**.` },
      { type: 'code', title: 'Maximum likelihood by brute force', code: `
import numpy as np
flips = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])                 # 7 heads in 10 flips
thetas = np.linspace(0.01, 0.99, 99)
h, n = flips.sum(), len(flips)
loglik = h * np.log(thetas) + (n - h) * np.log(1 - thetas)
print("grid-search MLE:", thetas[np.argmax(loglik)], "   closed form h/n:", flips.mean())

# a Gaussian: the MLE of the mean and standard deviation are the sample mean and (biased) std
rng = np.random.default_rng(0)
x = rng.normal(5.0, 2.0, 1000)
M, S = np.meshgrid(np.linspace(4, 6, 41), np.linspace(1.5, 2.5, 21))
nll = ((x[:, None, None] - M) ** 2 / (2 * S ** 2) + np.log(S * np.sqrt(2 * np.pi))).sum(axis=0)
i, j = np.unravel_index(np.argmin(nll), nll.shape)
print(f"grid MLE: mu = {M[i, j]:.2f}, sigma = {S[i, j]:.2f}    closed form: mean = {x.mean():.2f}, std = {x.std():.2f}")
` },
      { type: 'text', title: 'Where squared error and cross-entropy come from', md: `Assume a regression target is the model's prediction plus Gaussian noise: y = f(x; θ) + ε with ε ~ N(0, σ²). The negative log-likelihood of one example is

$$ −log p(y | x, θ) = ( y − f(x; θ) )² / (2σ²) + log( σ √(2π) ) $$
:::where
f(x; θ) — the model's prediction for input x
y — the observed target
σ — the standard deviation of the noise (a constant here)
log( σ √(2π) ) — a constant that does not depend on θ
:::

Minimizing the NLL over θ is therefore exactly minimizing the **squared error** — MSE *is* maximum likelihood under Gaussian noise. The same reasoning gives the other standard losses:

| Assumed distribution of the target | Negative log-likelihood = |
| --- | --- |
| Gaussian noise | mean squared error |
| Laplace noise (heavier tails) | mean absolute error |
| Bernoulli (yes/no) | binary cross-entropy |
| Categorical (one of K classes) | cross-entropy |
| Poisson (counts) | Poisson loss |

A language model maximizes Σ log p(next token | previous tokens) over its training text: pretraining is maximum likelihood.` },
      { type: 'check', question: { type: 'mcq', q: 'Minimizing mean squared error is equivalent to maximum likelihood under which assumption?', options: ['The targets are counts', 'The prediction errors are Gaussian noise with constant variance', 'The classes are balanced', 'The features are independent'], answer: 1, explain: 'The Gaussian log-density is a negative squared distance plus constants.' } },
      { type: 'text', title: 'MAP: adding prior beliefs', md: `Bayes' rule turns the likelihood into a distribution over parameters by multiplying by a **prior**:

$$ p(θ | data) ∝ p(data | θ) · p(θ) $$
$$ θ̂_MAP = argmin_θ [ −log p(data | θ) − log p(θ) ] $$
:::where
p(θ) — the prior: what you believe about the parameters before seeing data
p(θ | data) — the posterior: the updated belief after seeing the data
∝ — "proportional to" (the normalizing constant p(data) doesn't depend on θ)
θ̂_MAP — the maximum a posteriori estimate: the peak of the posterior
−log p(θ) — an extra penalty term on the parameters
:::

Plug in common priors:
- a **Gaussian prior** w ~ N(0, τ²) gives −log p(w) = Σ w² / (2τ²) + const — that is **L2 regularization** (ridge, weight decay), with a stronger penalty for a narrower prior;
- a **Laplace prior** gives Σ |w| / b + const — **L1 regularization** (lasso);
- for the coin, a Beta(a, b) prior gives θ̂_MAP = (h + a − 1) / (n + a + b − 2). With a = b = 2 that is (h + 1) / (n + 2): **Laplace (add-one) smoothing** is a prior too.

With more data the likelihood dominates and MAP approaches MLE. Fully **Bayesian** methods keep the whole posterior, not just its peak, which also gives uncertainty estimates.` },
      { type: 'code', title: 'MAP shrinks small-data estimates', code: `
import numpy as np
def coin_map(h, n, a, b):
    return (h + a - 1) / (n + a + b - 2)

for h, n in [(3, 3), (7, 10), (300, 300), (700, 1000)]:
    print(f"{h:3d}/{n:<4d} heads   MLE {h / n:.3f}   MAP, Beta(2,2) prior {coin_map(h, n, 2, 2):.3f}   MAP, Beta(20,20) prior {coin_map(h, n, 20, 20):.3f}")

# ridge regression is MAP with a Gaussian prior on the weights
rng = np.random.default_rng(1)
X = rng.normal(size=(8, 5))
w_true = np.array([1.0, 0.0, -1.0, 0.0, 0.5])
y = X @ w_true + rng.normal(0, 0.5, 8)
w_mle = np.linalg.lstsq(X, y, rcond=None)[0]
w_map = np.linalg.solve(X.T @ X + 2.0 * np.eye(5), X.T @ y)
print("true weights:         ", w_true)
print("MLE (least squares):  ", w_mle.round(2), "  length", round(float(np.linalg.norm(w_mle)), 2))
print("MAP (ridge, lambda=2):", w_map.round(2), "  length", round(float(np.linalg.norm(w_map)), 2))
`, after: `With 3 flips the prior pulls the estimate away from the extreme; with 300 flips it barely matters. With only 8 examples for 5 weights, the MAP (ridge) weights are shrunk towards zero — a smaller, more cautious model.` },
      { type: 'check', question: { type: 'tf', q: 'With more and more data, the MAP estimate approaches the maximum-likelihood estimate.', answer: true, explain: 'The likelihood term grows with n while the prior stays fixed.' } }
    ],
    when: `- Designing a loss for a new kind of target: counts → Poisson NLL, heavy-tailed noise → Laplace or Student-t, probabilities → cross-entropy.
- Choosing and interpreting regularization (it is a prior).
- Small datasets, where priors and smoothing prevent extreme estimates.
- Understanding probabilistic outputs and model calibration.`,
    pitfalls: `- Treating the likelihood as the probability that the parameters are correct.
- Multiplying many probabilities (underflow) instead of summing log-probabilities.
- Forgetting that the noise assumption shapes the loss: MSE (Gaussian) is very sensitive to outliers.
- Trusting MLE on tiny samples (3 heads in 3 flips → "always heads").
- The MLE of a variance divides by n and is slightly biased; use n − 1 for an unbiased sample variance.`,
    takeaways: [
      'Likelihood: how probable the observed data is under parameters θ; MLE maximizes it.',
      'Training usually minimizes a negative log-likelihood.',
      'MSE = Gaussian NLL; cross-entropy = categorical NLL; LM pretraining = MLE on text.',
      'MAP adds −log prior: a Gaussian prior is L2 regularization, a Laplace prior is L1.',
      'Priors matter most with little data; with lots of data MAP ≈ MLE.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the likelihood L(θ)?', options: ['The probability that θ is correct', 'The probability of the observed data under parameters θ, viewed as a function of θ', 'The prior on θ', 'The loss gradient'], answer: 1, explain: 'Data fixed, parameters varying.' },
      { type: 'fill', q: 'A coin lands heads 7 times in 10 flips. What is the maximum-likelihood estimate of P(heads)?', answer: ['0.7', '7/10', '70%'], explain: 'h / n = 7 / 10.' },
      { type: 'mcq', q: 'Which prior on the weights corresponds to L2 regularization?', options: ['A uniform prior', 'A Gaussian prior centred at zero', 'A Laplace prior', 'A Poisson prior'], answer: 1, explain: '−log of a Gaussian is a squared penalty.' },
      { type: 'tf', q: 'Cross-entropy loss is the negative log-likelihood of a categorical model.', answer: true, explain: '−log q(correct class).' },
      { type: 'fill', q: 'With a Beta(2, 2) prior, a coin shows 3 heads in 3 flips. What is the MAP estimate (h + 1) / (n + 2)?', answer: ['0.8', '4/5'], explain: '(3 + 1) / (3 + 2) = 0.8.' }
    ],
    problems: [
      {
        id: 'gaussian-mle', title: 'Gaussian Likelihood and MLE', difficulty: 'easy',
        prompt: `Implement:

- \`gaussian_nll(x, mu, sigma)\` → the total negative log-likelihood of the data under N(μ, σ²), as a float:

$$ NLL = Σ_i [ (x_i − μ)² / (2σ²) + log( σ √(2π) ) ] $$
:::where
x_i — the i-th data point
μ, σ — the mean and standard deviation of the Gaussian
:::

- \`gaussian_mle(x)\` → \`(mean, std)\`: the maximum-likelihood estimates, i.e. the sample mean and the standard deviation that divides by n (NumPy's default \`np.std\`).

\`\`\`text
gaussian_mle([1.0, 2.0, 3.0, 4.0])  ->  (2.5, 1.118…)
\`\`\``,
        starter: `
import numpy as np

def gaussian_nll(x, mu, sigma):
    # TODO
    pass

def gaussian_mle(x):
    # TODO
    pass
`,
        solution: `
import numpy as np

def gaussian_nll(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - mu) ** 2 / (2 * sigma ** 2) + np.log(sigma * np.sqrt(2 * np.pi))))

def gaussian_mle(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std())
`,
        hints: ['Translate the formula term by term with NumPy and sum.', '`np.std` divides by n by default, which is the MLE.'],
        explain: `The hidden test checks the defining property of the MLE: moving μ or σ away from the estimates can only increase the negative log-likelihood.`,
        tests: [
          { call: 'gaussian_nll([0.0], 0.0, 1.0)', expect: 'np.log(np.sqrt(2 * np.pi))' },
          { call: 'gaussian_mle([1.0, 2.0, 3.0, 4.0])', expect: '(2.5, np.sqrt(1.25))' },
          { name: 'the MLE minimizes the NLL', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(3.0, 1.5, 200)', code: 'mu, s = gaussian_mle(x)\nbest = gaussian_nll(x, mu, s)\nfor dm, ds in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:\n    assert gaussian_nll(x, mu + dm, s + ds) > best', hidden: true },
          { name: 'random data vs reference', setup: 'rng = np.random.default_rng(1)\nx = rng.normal(-2.0, 0.7, 50)', call: '[gaussian_nll(x, -2.0, 0.7), gaussian_mle(x)]', hidden: true }
        ]
      },
      {
        id: 'map-estimates', title: 'MAP Estimates: Coin and Ridge', difficulty: 'medium',
        prompt: `Implement:

- \`coin_map(heads, n, a, b)\` → the MAP estimate of P(heads) under a Beta(a, b) prior: (heads + a − 1) / (n + a + b − 2)
- \`ridge_map(X, y, lam)\` → the MAP weights of linear regression (no intercept) under a Gaussian prior, i.e. the ridge solution of (XᵀX + λI) w = Xᵀy. Use \`np.linalg.solve\`, not an explicit inverse.

\`\`\`text
coin_map(3, 3, 2, 2)  ->  0.8        (add-one smoothing)
coin_map(3, 3, 1, 1)  ->  1.0        (a flat prior gives back the MLE)
\`\`\``,
        starter: `
import numpy as np

def coin_map(heads, n, a, b):
    # TODO
    pass

def ridge_map(X, y, lam):
    # TODO
    pass
`,
        solution: `
import numpy as np

def coin_map(heads, n, a, b):
    return (heads + a - 1) / (n + a + b - 2)

def ridge_map(X, y, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
`,
        hints: ['A Beta(1, 1) prior is flat, so the MAP equals the MLE.', 'Build the matrix XᵀX + λI with `np.eye(d)`.'],
        explain: `Both functions are "MLE plus pseudo-data": the Beta prior adds a − 1 imaginary heads and b − 1 tails, and the Gaussian prior adds λ to the diagonal, as if extra examples pulled each weight towards zero.`,
        tests: [
          { call: '[coin_map(3, 3, 2, 2), coin_map(3, 3, 1, 1)]', expect: '[0.8, 1.0]' },
          { name: 'lambda = 0 gives least squares', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(30, 4)); y = rng.normal(size=30)', call: 'ridge_map(X, y, 0.0)', expect: 'np.linalg.lstsq(X, y, rcond=None)[0]' },
          { name: 'the prior shrinks the weights', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(10, 6)); y = rng.normal(size=10)', code: 'norms = [np.linalg.norm(ridge_map(X, y, lam)) for lam in (0.0, 1.0, 10.0, 100.0)]\nassert all(a > b for a, b in zip(norms, norms[1:]))', hidden: true },
          { name: 'random problem vs reference', setup: 'rng = np.random.default_rng(2)\nX = rng.normal(size=(12, 3)); y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 12)', call: '[ridge_map(X, y, 3.0), coin_map(17, 40, 5, 3)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Bishop — Pattern Recognition and Machine Learning, chapters 1–3 (free PDF)', url: 'https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/' },
      { title: 'StatQuest — Maximum Likelihood, clearly explained (video)', url: 'https://www.youtube.com/watch?v=XepXtl9YKwc' },
      { title: 'Murphy — Probabilistic Machine Learning: An Introduction (free book)', url: 'https://probml.github.io/pml-book/book1.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'stat-inference', branch: 'math', title: 'Statistical Inference: Confidence Intervals, Tests & A/B Testing', short: 'Statistical Inference', level: 2, minutes: 45, prereqs: ['statistics', 'probability'],
    tags: ['confidence interval', 'standard error', 'bootstrap', 'hypothesis test', 'p-value', 'permutation test', 'a/b test', 'statistical power', 'sample size', 'multiple comparisons'],
    summary: 'How to tell a real effect from noise: sampling variability and standard errors, confidence intervals (analytic and bootstrap), hypothesis tests and p-values, A/B tests for comparing models or products, statistical power and sample size, and traps such as peeking and multiple comparisons.',
    why: `"Model B scored 0.83 and model A 0.81 — is B better?" Every metric is computed from a finite sample, so every number is noisy. **Statistical inference** tells you how large that noise is and whether a difference is likely to be real. It is essential for comparing models, running A/B tests, evaluating LLMs and reading research results critically — and it protects you from shipping changes that only looked better by luck.`,
    objectives: [
      'Explain sampling variability and compute the standard error of a mean or proportion',
      'Build confidence intervals analytically and with the bootstrap',
      'Run and interpret a two-proportion z-test and a permutation test, and say what a p-value is (and is not)',
      'Design an A/B test: sample size, power, and pitfalls such as peeking and multiple comparisons'
    ],
    sections: [
      { type: 'text', title: 'Every metric is a random variable', md: `Accuracy measured on one set of 500 test examples would come out slightly different on another 500. The value you report is one draw from a **sampling distribution**. Its spread is the **standard error**; for a proportion (accuracy, conversion rate):

$$ SE( p̂ ) = √( p̂ (1 − p̂) / n ) $$
:::where
p̂ — "p hat": the measured proportion, e.g. 0.83 accuracy
n — the number of independent examples it was measured on
SE — the standard error: the typical size of the gap between p̂ and the true value
:::

By the **central limit theorem**, averages of many independent values are approximately normally distributed, whatever the shape of the individual values. Note the √n: four times the data halves the noise.` },
      { type: 'text', title: 'Confidence intervals', md: `A 95% **confidence interval** is a range built by a procedure that captures the true value in 95% of repeated experiments. For an approximately normal estimate:

$$ CI_95 = estimate ± 1.96 · SE $$
:::where
estimate — the measured value, such as an accuracy or a mean
SE — its standard error
1.96 — the number of standard errors that covers 95% of a normal distribution
:::

Example: 0.83 accuracy on 500 examples: SE = √(0.83 · 0.17 / 500) ≈ 0.017, so the interval is about [0.80, 0.86]. A rival model at 0.81 sits comfortably inside it — on this test set alone you cannot claim one is better.

**The bootstrap** gives intervals for *any* statistic (median, F1, AUC, a ratio) without formulas: resample the data **with replacement** many times, recompute the statistic each time, and take the 2.5th and 97.5th percentiles of the results.` },
      { type: 'code', title: 'Analytic and bootstrap confidence intervals', code: `
import numpy as np
rng = np.random.default_rng(0)
correct = rng.random(500) < 0.83                     # was the model right on each of 500 test items?
acc = correct.mean()
se = np.sqrt(acc * (1 - acc) / len(correct))
print(f"accuracy {acc:.3f}   analytic 95% CI [{acc - 1.96 * se:.3f}, {acc + 1.96 * se:.3f}]")

boot = [correct[rng.integers(0, 500, 500)].mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")

# the bootstrap works for any statistic, e.g. a median response time
times = rng.lognormal(5, 0.6, 300)
meds = [np.median(times[rng.integers(0, 300, 300)]) for _ in range(2000)]
print(f"median response time {np.median(times):.0f} ms, 95% CI [{np.percentile(meds, 2.5):.0f}, {np.percentile(meds, 97.5):.0f}]")
`, after: `For accuracy both methods agree; for the median, where no simple formula exists, the bootstrap still works.` },
      { type: 'text', title: 'Hypothesis tests and p-values', md: `A hypothesis test asks: *if there were really no difference, how surprising would my data be?*

1. State the **null hypothesis** H₀ (e.g. "A and B convert equally often").
2. Compute a **test statistic** from the data.
3. The **p-value** is the probability of a result at least as extreme as the one observed, *assuming H₀ is true*. A small p-value (conventionally < 0.05) is evidence against H₀.

For two conversion rates, the **two-proportion z-test**:

$$ z = ( p̂_B − p̂_A ) / √( p̂ (1 − p̂) (1/n_A + 1/n_B) ) $$
:::where
p̂_A, p̂_B — the measured rates in groups A and B
n_A, n_B — the group sizes
p̂ — the pooled rate: all successes divided by all trials (the best estimate if H₀ is true)
z — how many standard errors apart the two rates are
The two-sided p-value is 2 · (1 − Φ(|z|)), where Φ is the standard normal cumulative distribution.
:::

A **permutation test** needs no formula: shuffle the group labels many times and count how often the shuffled difference is at least as large as the real one.

What a p-value is **not**: the probability that H₀ is true, or a measure of how big or important the effect is. A tiny, useless difference becomes "significant" with enough data; always report the effect size with its confidence interval.` },
      { type: 'code', title: 'An A/B test: z-test and permutation test', code: `
import math
import numpy as np
rng = np.random.default_rng(1)
a = rng.random(5000) < 0.100                 # control converts at 10%
b = rng.random(5000) < 0.115                 # variant converts at 11.5%
pa, pb = a.mean(), b.mean()
pooled = (a.sum() + b.sum()) / (len(a) + len(b))
se = math.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
z = (pb - pa) / se
p_value = math.erfc(abs(z) / math.sqrt(2))   # two-sided p-value
print(f"A {pa:.2%}   B {pb:.2%}   difference {pb - pa:+.2%}   z = {z:.2f}   p = {p_value:.4f}")

both, observed, count = np.concatenate([a, b]), pb - pa, 0
for _ in range(2000):
    perm = rng.permutation(both)             # pretend the labels don't matter
    count += abs(perm[5000:].mean() - perm[:5000].mean()) >= abs(observed)
print(f"permutation test p ≈ {count / 2000:.4f}")
` },
      { type: 'check', question: { type: 'mcq', q: 'A test gives p = 0.03. What does that mean?', options: ['There is a 3% chance that H₀ is true', 'If H₀ were true, a difference at least this large would occur about 3% of the time', 'The effect is 3% large', 'The test is 97% accurate'], answer: 1, explain: 'A p-value is computed assuming H₀; it is not the probability of H₀.' } },
      { type: 'text', title: 'Designing A/B tests: power and sample size', md: `Two kinds of mistakes: a **false positive** (declaring a difference that isn't there; rate α, usually 5%) and a **false negative** (missing a real one; rate β). **Power** = 1 − β, commonly 80%. To detect a difference δ between two proportions near p you need roughly, per group:

$$ n ≈ ( z_α + z_β )² · 2 p (1 − p) / δ² $$
:::where
n — the number of users (or examples) in each group
p — the baseline rate, e.g. 0.10
δ — "delta": the smallest difference worth detecting, e.g. 0.01
z_α — 1.96 for a two-sided test at α = 0.05
z_β — 0.84 for 80% power
:::

Detecting 10% → 11% needs about (2.8)² · 2 · 0.09 / 0.0001 ≈ 14,100 users per group. Small effects need big samples.

Common traps:
- **Peeking**: checking the p-value every day and stopping when it dips below 0.05 inflates false positives dramatically. Fix the sample size in advance (or use sequential methods designed for peeking).
- **Multiple comparisons**: test 20 metrics and one will look "significant" by chance. Pick a primary metric, or correct (Bonferroni: use α / m for m tests).
- **Novelty effects**, seasonality, and users who aren't independent (the same person in both groups).
- **Simpson's paradox** and confounding when groups weren't randomized (see *Causal Inference*).` },
      { type: 'check', question: { type: 'tf', q: 'Stopping an A/B test as soon as the p-value first drops below 0.05 keeps the false-positive rate at 5%.', answer: false, explain: 'Repeated peeking gives many chances to cross the threshold by luck.' } }
    ],
    when: `- Comparing two models on a test set or an LLM evaluation set.
- Product A/B tests: features, ranking changes, prompts.
- Reporting any metric — always with an interval, not a bare number.
- Planning how much data (or how many users) an experiment needs.`,
    pitfalls: `- Reporting point estimates without uncertainty.
- Reading p > 0.05 as "no difference" — it may just mean too little data.
- Peeking, multiple comparisons and changing the metric after seeing the results (p-hacking).
- Treating correlated samples (same user, same document) as independent.
- Confusing statistical significance with practical importance.`,
    takeaways: [
      'Every metric is noisy: SE = √(p(1 − p)/n) for a proportion, shrinking with √n.',
      'A 95% CI ≈ estimate ± 1.96·SE; the bootstrap gives intervals for any statistic.',
      'A p-value is the probability of data at least this extreme if there were no effect — not the probability of no effect.',
      'Plan sample sizes with a power calculation; small effects need large samples.',
      'Avoid peeking and multiple comparisons; report effect sizes with intervals.'
    ],
    quiz: [
      { type: 'fill', q: 'A model is 90% accurate on 400 examples. What is the standard error? (3 decimals)', answer: ['0.015'], explain: '√(0.9 · 0.1 / 400) = √0.000225 = 0.015.' },
      { type: 'mcq', q: 'How does the bootstrap build a confidence interval?', options: ['From a formula for the standard error', 'By resampling the data with replacement many times and taking percentiles of the recomputed statistic', 'By training many models', 'By splitting the data into folds'], answer: 1, explain: 'The spread of the resampled statistics estimates its sampling variability.' },
      { type: 'mcq', q: 'You test 20 independent metrics at α = 0.05 and none truly changed. About how many will look significant by chance?', options: ['0', '1', '5', '20'], answer: 1, explain: '20 × 0.05 = 1 expected false positive.' },
      { type: 'tf', q: 'To halve the standard error of an estimate you need about four times as much data.', answer: true, explain: 'SE scales with 1/√n.' },
      { type: 'mcq', q: 'Why use a permutation test?', options: ['It is always more powerful', 'It needs no distributional formula: shuffling labels shows what differences arise by chance', 'It removes confounding', 'It increases the sample size'], answer: 1, explain: 'It builds the null distribution directly from the data.' }
    ],
    problems: [
      {
        id: 'bootstrap-ci', title: 'Bootstrap Confidence Interval', difficulty: 'medium',
        prompt: `Implement \`bootstrap_ci(data, stat, n_boot=1000, alpha=0.05, seed=0)\` → \`(low, high)\` as floats:

1. \`rng = np.random.default_rng(seed)\`, n = len(data).
2. Repeat \`n_boot\` times: \`idx = rng.integers(0, n, n)\` (exactly one call per resample) and record \`stat(data[idx])\`.
3. Return the percentiles 100·α/2 and 100·(1 − α/2) of the recorded values (\`np.percentile\`).

\`stat\` is any function of an array, e.g. \`np.mean\` or \`np.median\`.`,
        starter: `
import numpy as np

def bootstrap_ci(data, stat, n_boot=1000, alpha=0.05, seed=0):
    data = np.asarray(data)
    # TODO
    pass
`,
        solution: `
import numpy as np

def bootstrap_ci(data, stat, n_boot=1000, alpha=0.05, seed=0):
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    n = len(data)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(stat(data[idx]))
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))
`,
        hints: ['`rng.integers(0, n, n)` draws n random indices with replacement.', '`np.percentile(values, [2.5, 97.5])` gives both ends at once for α = 0.05.'],
        explain: `Resampling with replacement mimics drawing a fresh dataset from the same population. The variability of the statistic across resamples approximates its true sampling variability — no formula needed.`,
        tests: [
          { name: 'mean of a small sample vs reference', call: 'bootstrap_ci([2.0, 4.0, 4.0, 5.0, 7.0, 9.0], np.mean, n_boot=500, seed=1)' },
          { name: 'constant data has a zero-width interval', call: 'bootstrap_ci([3.0] * 20, np.median)', expect: '(3.0, 3.0)' },
          { name: 'interval contains the sample mean', setup: 'rng = np.random.default_rng(0)\ndata = rng.normal(10, 2, 200)', code: 'lo, hi = bootstrap_ci(data, np.mean)\nassert lo < data.mean() < hi\nassert 0.3 < hi - lo < 0.8', hidden: true },
          { name: 'median with another alpha vs reference', setup: 'rng = np.random.default_rng(4)\ndata = rng.exponential(2.0, 80)', call: 'bootstrap_ci(data, np.median, n_boot=300, alpha=0.1, seed=7)', hidden: true }
        ]
      },
      {
        id: 'two-proportion-test', title: 'A/B Test Statistics', difficulty: 'medium',
        prompt: `Implement:

- \`z_test(success_a, n_a, success_b, n_b)\` → \`(z, p_value)\`: the two-proportion z-statistic (B minus A, using the pooled rate) and the two-sided p-value \`math.erfc(abs(z) / math.sqrt(2))\`.
- \`sample_size(p, delta, z_alpha=1.96, z_beta=0.84)\` → the users needed per group, rounded **up** with \`math.ceil\`:

$$ n = ( z_α + z_β )² · 2 p (1 − p) / δ² $$
:::where
p — the baseline conversion rate
δ — the smallest difference you want to detect
z_α, z_β — the normal quantiles for the significance level and the power
:::

\`\`\`text
z_test(50, 100, 50, 100)  ->  (0.0, 1.0)
\`\`\``,
        starter: `
import math

def z_test(success_a, n_a, success_b, n_b):
    # TODO
    pass

def sample_size(p, delta, z_alpha=1.96, z_beta=0.84):
    # TODO
    pass
`,
        solution: `
import math

def z_test(success_a, n_a, success_b, n_b):
    pa, pb = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se
    return z, math.erfc(abs(z) / math.sqrt(2))

def sample_size(p, delta, z_alpha=1.96, z_beta=0.84):
    return math.ceil((z_alpha + z_beta) ** 2 * 2 * p * (1 - p) / delta ** 2)
`,
        hints: ['The pooled rate is (success_a + success_b) / (n_a + n_b).', 'erfc(|z|/√2) equals 2·(1 − Φ(|z|)), the two-sided p-value.'],
        explain: `Under the null hypothesis both groups share one rate, so the pooled rate gives the best estimate of the standard error. The sample-size formula inverts the same logic: how large must n be for a difference δ to sit z_α + z_β standard errors away from zero?`,
        tests: [
          { call: 'z_test(50, 100, 50, 100)', expect: '(0.0, 1.0)' },
          { name: 'a real difference', call: 'z_test(100, 1000, 130, 1000)' },
          { name: 'sample sizes', call: '[sample_size(0.2, 0.05), sample_size(0.15, 0.03)]', expect: '[1004, 2222]' },
          { name: 'more data, smaller p-value', code: 'z1, p1 = z_test(100, 1000, 120, 1000)\nz2, p2 = z_test(1000, 10000, 1200, 10000)\nassert z2 > z1 and p2 < p1', hidden: true },
          { name: 'sign and symmetry', call: '[z_test(130, 1000, 100, 1000), z_test(20, 400, 31, 380)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'OpenIntro Statistics (free textbook)', url: 'https://www.openintro.org/book/os/' },
      { title: 'Kohavi, Tang & Xu — Trustworthy Online Controlled Experiments', url: 'https://experimentguide.com/' },
      { title: 'Evan Miller — How Not To Run an A/B Test (on peeking)', url: 'https://www.evanmiller.org/how-not-to-run-an-ab-test.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'pandas-eda', branch: 'code', title: 'pandas & Exploratory Data Analysis', short: 'pandas & EDA', level: 1, minutes: 45, prereqs: ['python-essentials', 'numpy'],
    tags: ['pandas', 'dataframe', 'series', 'groupby', 'merge', 'pivot table', 'missing values', 'eda', 'visualization', 'matplotlib', 'histogram'],
    summary: 'Working with real tables: pandas Series and DataFrames, selecting and filtering, cleaning types, missing values and duplicates, groupby aggregations, pivot tables and joins, and an exploratory-data-analysis routine with summary statistics and plots — the first step of every ML project.',
    why: `Real data arrives as messy tables — CSV exports, database queries, logs — not as clean NumPy arrays. **pandas** is the standard Python tool for loading, cleaning, reshaping and summarizing tables, and **exploratory data analysis (EDA)** is how you find out what is actually in the data before modelling: missing values, outliers, skewed distributions, leaky columns, and how features relate to the target. Skipping EDA is the most common way to build a confident model on broken data.`,
    objectives: [
      'Create, inspect and select from pandas DataFrames',
      'Clean data: types, missing values, duplicates and derived columns',
      'Summarize with groupby, pivot tables and joins',
      'Run an EDA checklist and make basic plots'
    ],
    sections: [
      { type: 'text', title: 'Series and DataFrames', md: `A **DataFrame** is a table: named columns, each a **Series** with its own type (dtype), plus an **index** that labels the rows. It is built on NumPy, so column operations are vectorized and fast.

| Task | Code |
| --- | --- |
| Load a CSV | \`df = pd.read_csv("houses.csv")\` |
| First rows, size, types | \`df.head()\`, \`df.shape\`, \`df.dtypes\`, \`df.info()\` |
| Summary statistics | \`df.describe()\` |
| One column / several | \`df["price"]\`, \`df[["price", "sqm"]]\` |
| Rows by condition | \`df[df["price"] > 500]\`, \`df[(df.city == "Rome") & (df.rooms >= 3)]\` |
| By label / by position | \`df.loc[row_labels, cols]\`, \`df.iloc[0:5, 1:3]\` |
| New column | \`df["ppsqm"] = df["price"] / df["sqm"]\` |
| Sort | \`df.sort_values("price", ascending=False)\` |` },
      { type: 'code', title: 'First steps with a DataFrame', code: `
import pandas as pd
df = pd.DataFrame({
    "city":  ["Paris", "Rome", "Rome", "Berlin", "Paris", "Berlin", "Rome", "Paris"],
    "sqm":   [55, 80, 80, 120, 35, 64, None, 92],
    "rooms": [2, 3, 3, 4, 1, 2, 3, 4],
    "price": [610, 420, 420, 510, 450, 280, 390, 980],      # thousands of euros
})
print(df.head(3))
print("shape:", df.shape, "| missing values per column:", df.isna().sum().to_dict())
print(df.describe().round(1))

df = df.drop_duplicates()                                  # rows 1 and 2 were identical
df["price_per_sqm"] = (df["price"] * 1000 / df["sqm"]).round(0)
print(df[df["city"] == "Paris"].sort_values("price", ascending=False))
` },
      { type: 'text', title: 'Cleaning data', md: `- **Missing values**: count them with \`df.isna().sum()\`; drop rows (\`dropna\`) or fill (\`fillna\`) — with statistics computed on the **training** data only, or you leak information from the test set.
- **Wrong types**: numbers stored as text ("1,200", "n/a") → \`pd.to_numeric(col, errors="coerce")\` turns unparseable values into NaN; dates → \`pd.to_datetime\`.
- **Duplicates**: \`df.drop_duplicates()\`; check whether IDs should be unique.
- **Inconsistent categories**: " Paris", "paris", "PARIS" → \`df["city"].str.strip().str.lower()\`.
- **Impossible values**: negative ages, prices of 0, dates in the future — investigate, then fix or drop.

Keep cleaning in a function (or a scikit-learn pipeline) so exactly the same steps run on new data.` },
      { type: 'text', title: 'Groupby, pivot tables and joins', md: `**Split–apply–combine** with \`groupby\`: split rows into groups, apply an aggregation to each, combine the results.

\`\`\`text
df.groupby("city")["price"].agg(["count", "mean", "median"])     one row per city
df.groupby(["city", "rooms"])["price"].mean()                    one row per (city, rooms) pair
df.pivot_table(index="city", columns="month", values="revenue", aggfunc="sum")
\`\`\`

**Joins** combine tables on a key, like SQL: \`orders.merge(customers, on="customer_id", how="left")\` keeps every order and adds customer columns where a match exists (missing ones become NaN); \`how="inner"\` keeps only matches. Always check the row count after a join: duplicate keys multiply rows.` },
      { type: 'code', title: 'Summaries and joins', code: `
import pandas as pd
sales = pd.DataFrame({"city": ["Paris", "Rome", "Paris", "Berlin", "Rome", "Paris"],
                      "month": ["Jan", "Jan", "Feb", "Feb", "Feb", "Mar"],
                      "revenue": [120, 80, 150, 95, 70, 160]})
cities = pd.DataFrame({"city": ["Paris", "Rome", "Madrid"], "population_m": [2.1, 2.8, 3.3]})

print(sales.groupby("city")["revenue"].agg(["count", "sum", "mean"]))
print(sales.pivot_table(index="city", columns="month", values="revenue", aggfunc="sum", fill_value=0))

merged = sales.merge(cities, on="city", how="left")        # Berlin has no population row -> NaN
merged["revenue_per_million"] = (merged["revenue"] / merged["population_m"]).round(1)
print(merged)
print("rows before and after the join:", len(sales), len(merged))
` },
      { type: 'check', question: { type: 'mcq', q: 'What does df.groupby("city")["price"].mean() return?', options: ['The overall mean price', 'The average price for each city', 'The cities sorted by price', 'The number of rows per city'], answer: 1, explain: 'One aggregated value per group.' } },
      { type: 'text', title: 'An EDA checklist', md: `1. **Shape and types**: rows, columns, dtypes, a few sample rows.
2. **Missing values** per column — and whether "missing" itself carries information.
3. **Distributions** of numeric columns (histograms): skewed data (incomes, prices) often benefits from a log transform.
4. **Outliers and impossible values**.
5. **The target**: its distribution, class balance, how it was measured.
6. **Relationships**: correlations, target versus each feature, \`groupby\` means by category.
7. **Leakage suspects**: columns that would not exist at prediction time, IDs, timestamps, "refund_issued" when predicting returns.
8. **Time and grouping structure**: does the split need to respect time or users?

| Question | Plot |
| --- | --- |
| How is one numeric column distributed? | histogram |
| How does a numeric column differ across categories? | box plot |
| How do two numeric columns relate? | scatter plot |
| How common is each category? | bar chart |
| How does a value change over time? | line plot |
| Which numeric columns move together? | correlation heatmap |` },
      { type: 'code', title: 'Quick plots for EDA', code: `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
df = pd.DataFrame({"sqm": rng.gamma(4, 18, 300)})
df["price"] = 3.2 * df["sqm"] + rng.normal(0, 40, 300)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].hist(df["price"], bins=30)
axes[0].set_title("Distribution of price (right-skewed)")
axes[1].scatter(df["sqm"], df["price"], s=8, alpha=0.6)
axes[1].set_title("Price vs size")
axes[1].set_xlabel("sqm")
axes[1].set_ylabel("price (k€)")
plt.tight_layout()
plt.show()
print(df.corr().round(2))
` },
      { type: 'check', question: { type: 'tf', q: 'Filling missing values with the mean of the whole dataset before splitting into train and test is safe.', answer: false, explain: 'It leaks test-set information into training; compute fill values on the training split.' } }
    ],
    when: `- Every new dataset: load, clean and explore before any modelling.
- Feature engineering with group statistics, joins and date parts.
- Reporting and quick analyses that don't need a model at all.
- For data that doesn't fit in memory, the same ideas carry over to Polars, DuckDB, SQL or Spark.`,
    pitfalls: `- Chained assignment (\`df[df.a > 0]["b"] = 1\`) that silently modifies a copy — use \`df.loc[mask, "b"] = 1\`.
- Joins that multiply rows because of duplicate keys.
- Computing fill values or encodings on the full dataset (leakage).
- Looping over rows with Python instead of using vectorized column operations.
- Skipping plots: summary statistics alone can hide very different data (Anscombe's quartet).`,
    takeaways: [
      'A DataFrame is a table of typed columns (Series) with an index; selection uses [], .loc and .iloc.',
      'Clean types, missing values, duplicates and categories in a reusable function.',
      'groupby (split–apply–combine), pivot_table and merge summarize and combine tables.',
      'EDA = shape, missingness, distributions, outliers, target, relationships, leakage, structure.',
      'Plot before you model: histograms, scatter plots, box plots.'
    ],
    quiz: [
      { type: 'mcq', q: 'Which expression selects the rows where age is over 30?', options: ['df["age" > 30]', 'df[df["age"] > 30]', 'df.age(30)', 'df.loc[30]'], answer: 1, explain: 'A boolean Series inside [] filters rows.' },
      { type: 'mcq', q: 'What is the difference between .loc and .iloc?', options: ['There is none', '.loc selects by labels, .iloc by integer positions', '.loc is for columns, .iloc for rows', '.iloc is faster but deprecated'], answer: 1, explain: 'Labels versus positions.' },
      { type: 'tf', q: 'df.describe() summarizes numeric columns with count, mean, standard deviation, minimum, quartiles and maximum.', answer: true, explain: 'A quick first look at every numeric column.' },
      { type: 'mcq', q: 'A left join of orders with customers on customer_id keeps…', options: ['only orders with a matching customer', 'every order, adding customer columns where a match exists', 'every customer', 'only customers without orders'], answer: 1, explain: 'Non-matching orders get NaN in the customer columns.' },
      { type: 'mcq', q: 'Which plot best shows the distribution of a single numeric column?', options: ['A histogram', 'A pie chart', 'A line plot', 'A scatter plot'], answer: 0, explain: 'It shows the shape: skew, peaks and outliers.' }
    ],
    problems: [
      {
        id: 'clean-dataframe', title: 'Clean a Messy Table', difficulty: 'medium',
        prompt: `Implement \`clean(df)\` that returns a **new** cleaned DataFrame (don't modify the input), applying these steps in order:

1. drop exact duplicate rows;
2. convert "price" to numbers with \`pd.to_numeric(..., errors="coerce")\`;
3. drop rows whose price is missing;
4. strip spaces and lowercase "city";
5. fill missing "rooms" with the **median** of the remaining rooms values;
6. reset the index (\`reset_index(drop=True)\`).

\`\`\`text
city      price   rooms              city    price   rooms
" Paris"  "300"   3                  paris   300.0   3.0
"Rome "   "250"   NaN       ->       rome    250.0   3.5
"Rome "   "250"   NaN                paris   410.0   4.0
"berlin"  "n/a"   2
"PARIS"   "410"   4
\`\`\``,
        starter: `
import pandas as pd

def clean(df):
    out = df.copy()
    # TODO
    return out
`,
        solution: `
import pandas as pd

def clean(df):
    out = df.drop_duplicates().copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"]).copy()
    out["city"] = out["city"].str.strip().str.lower()
    out["rooms"] = out["rooms"].fillna(out["rooms"].median())
    return out.reset_index(drop=True)
`,
        hints: ['`df.drop_duplicates()` and `df.dropna(subset=["price"])` each return new DataFrames.', 'String methods live under `.str`: `.str.strip().str.lower()`.', 'Compute the median after dropping rows, so it uses only the rows you keep.'],
        explain: `Order matters: dropping the unparseable price before computing the median means the fill value comes from the data you actually keep. Packaging the steps in one function lets you apply exactly the same cleaning to new data later.`,
        tests: [
          { setup: 'import pandas as pd\ndf = pd.DataFrame({"city": [" Paris", "Rome ", "Rome ", "berlin", "PARIS"], "price": ["300", "250", "250", "n/a", "410"], "rooms": [3, None, None, 2, 4]})', call: 'clean(df).to_dict("list")', expect: '{"city": ["paris", "rome", "paris"], "price": [300.0, 250.0, 410.0], "rooms": [3.0, 3.5, 4.0]}' },
          { name: 'the input is not modified', setup: 'import pandas as pd\ndf = pd.DataFrame({"city": [" A", "b"], "price": ["1", "x"], "rooms": [None, 2.0]})', code: 'before = df.copy()\nclean(df)\nassert df.equals(before), "clean must not change its input"' },
          { name: 'a larger table vs reference', setup: 'import pandas as pd\nimport numpy as np\nrng = np.random.default_rng(0)\ncities = [" Oslo", "oslo ", "LIMA", "Lima", " pune "]\nrows = [{"city": cities[int(rng.integers(5))], "price": str(int(rng.integers(100, 900))) if rng.random() > 0.15 else "?", "rooms": float(rng.integers(1, 6)) if rng.random() > 0.2 else None} for _ in range(30)]\ndf = pd.DataFrame(rows + rows[:4])', call: 'clean(df).to_dict("list")', hidden: true }
        ]
      },
      {
        id: 'groupby-summary', title: 'Summaries and Joins', difficulty: 'medium',
        prompt: `\`df\` has the columns "city", "sqm" and "price". Implement:

- \`city_summary(df)\` → a DataFrame indexed by city with the columns **n** (number of rows), **mean_price**, **median_price** and **price_per_sqm** (the mean of price / sqm over the city's rows), sorted by mean_price from high to low. Named aggregation helps: \`.agg(n=("price", "size"), mean_price=("price", "mean"), …)\`.
- \`add_population(df, cities)\` → a **left** join of df with the table \`cities\` (columns "city" and "population") on "city".

The tests compare \`city_summary(df).round(4).reset_index().to_dict("list")\`.`,
        starter: `
import pandas as pd

def city_summary(df):
    # TODO
    pass

def add_population(df, cities):
    # TODO
    pass
`,
        solution: `
import pandas as pd

def city_summary(df):
    out = df.assign(ppsqm=df["price"] / df["sqm"]).groupby("city").agg(
        n=("price", "size"),
        mean_price=("price", "mean"),
        median_price=("price", "median"),
        price_per_sqm=("ppsqm", "mean"),
    )
    return out.sort_values("mean_price", ascending=False)

def add_population(df, cities):
    return df.merge(cities, on="city", how="left")
`,
        hints: ['`df.assign(ppsqm=df["price"] / df["sqm"])` adds a helper column without modifying df.', 'Named aggregation: `.agg(new_name=("column", "function"))`.', '`merge(..., on="city", how="left")` keeps every row of df.'],
        explain: `Named aggregations produce tidy, clearly labelled summary tables in one step. The left join keeps every original row — cities missing from the lookup table simply get NaN, which is easy to detect afterwards.`,
        tests: [
          { setup: 'import pandas as pd\ndf = pd.DataFrame({"city": ["a", "b", "a", "c", "b", "a"], "sqm": [50, 100, 70, 40, 80, 60], "price": [300, 400, 350, 100, 480, 310]})', call: 'city_summary(df).round(4).reset_index().to_dict("list")' },
          { name: 'left join keeps every row', setup: 'import pandas as pd\ndf = pd.DataFrame({"city": ["a", "b", "z"], "sqm": [1, 2, 3], "price": [10, 20, 30]})\ncities = pd.DataFrame({"city": ["a", "b"], "population": [5, 7]})', call: 'add_population(df, cities)["population"].tolist()', expect: '[5.0, 7.0, float("nan")]' },
          { name: 'random data vs reference', setup: 'import pandas as pd\nimport numpy as np\nrng = np.random.default_rng(1)\ndf = pd.DataFrame({"city": rng.choice(["x", "y", "z", "w"], 40), "sqm": rng.integers(30, 150, 40), "price": rng.integers(100, 1000, 40)})', call: 'city_summary(df).round(4).reset_index().to_dict("list")', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'pandas — 10 minutes to pandas', url: 'https://pandas.pydata.org/docs/user_guide/10min.html' },
      { title: 'Wes McKinney — Python for Data Analysis (free online edition)', url: 'https://wesmckinney.com/book/' },
      { title: 'matplotlib — Quick start guide', url: 'https://matplotlib.org/stable/users/explain/quick_start.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'gmm-em', branch: 'ml', title: 'Gaussian Mixtures, EM & Other Clustering', short: 'GMMs & EM', level: 3, minutes: 45, prereqs: ['kmeans', 'mle-map'],
    tags: ['gaussian mixture model', 'em algorithm', 'soft clustering', 'responsibilities', 'bic', 'dbscan', 'hierarchical clustering', 'density-based clustering'],
    summary: 'Clustering beyond k-means: Gaussian mixture models give every point a probability of belonging to each cluster and are fitted with the expectation–maximization (EM) algorithm; DBSCAN finds clusters of any shape and flags noise; hierarchical clustering builds a tree of merges.',
    why: `k-means makes hard assignments and assumes round clusters of similar size. Real groups overlap, stretch and differ in density. **Gaussian mixture models** fix this with soft, probabilistic clusters, and they are the textbook example of the **EM algorithm** — a general recipe for maximum likelihood when some variables are hidden. **DBSCAN** and **hierarchical clustering** handle the cases where you don't know how many clusters there are or where clusters have odd shapes.`,
    objectives: [
      'Describe a Gaussian mixture as a generative model with hidden cluster labels',
      'Run EM: compute responsibilities (E-step) and re-estimate the parameters (M-step)',
      'Choose the number of components with BIC and compare GMMs with k-means',
      'Explain DBSCAN and agglomerative clustering, and when to use each method'
    ],
    sections: [
      { type: 'text', title: 'A mixture of Gaussians', md: `A **Gaussian mixture model (GMM)** tells a generative story: to produce a point, first pick a component k with probability π_k, then draw the point from that component's Gaussian. The density of the data is a weighted sum of bells:

$$ p(x) = Σ_k π_k · N( x | μ_k, Σ_k ) $$
:::where
K — the number of components (clusters)
π_k — the mixing weight of component k: how common it is (the weights sum to 1)
μ_k — the centre (mean) of component k
Σ_k — its covariance matrix: the cluster's spread, shape and orientation
N( x | μ, Σ ) — the Gaussian density at x
:::

Instead of a hard label, each point gets a **responsibility** for each component — the posterior probability that the component generated it:

$$ r_ik = π_k N( x_i | μ_k, Σ_k ) / Σ_j π_j N( x_i | μ_j, Σ_j ) $$
:::where
r_ik — the probability that point i belongs to component k (each row sums to 1)
x_i — the i-th data point
Σ_j — the sum over all K components, which normalizes the probabilities
:::` },
      { type: 'text', title: 'The EM algorithm', md: `If we knew which component produced each point, fitting would be easy (a mean and variance per group). If we knew the parameters, assigning points would be easy (responsibilities). **Expectation–maximization** alternates between the two:

- **E-step**: compute the responsibilities r_ik with the current parameters.
- **M-step**: re-estimate each component from all points, weighted by responsibility (shown for one dimension):

$$ N_k = Σ_i r_ik          π_k = N_k / n          μ_k = (1/N_k) Σ_i r_ik x_i          σ_k² = (1/N_k) Σ_i r_ik (x_i − μ_k)² $$
:::where
N_k — the effective number of points assigned to component k
n — the total number of points
μ_k, σ_k² — the responsibility-weighted mean and variance of component k
:::

Each iteration never decreases the likelihood, so EM converges — but to a **local** maximum that depends on the starting point. Use several restarts (or initialize from k-means). k-means is the special case of EM with hard assignments and identical round clusters.` },
      { type: 'code', title: 'EM for a one-dimensional mixture', code: `
import numpy as np
rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(-2, 0.8, 300), rng.normal(3, 1.5, 200)])   # two hidden groups

def pdf(x, mu, s):
    return np.exp(-(x - mu) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))

pi, mu, sigma = np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])   # rough start
for it in range(40):
    dens = pi * pdf(x[:, None], mu, sigma)            # (n, 2): weighted density of each component
    r = dens / dens.sum(axis=1, keepdims=True)         # E-step: responsibilities
    Nk = r.sum(axis=0)                                 # M-step: weighted statistics
    pi, mu = Nk / len(x), (r * x[:, None]).sum(axis=0) / Nk
    sigma = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk)
    if it % 10 == 0 or it == 39:
        print(f"iter {it:2d}  log-likelihood {np.log(dens.sum(axis=1)).sum():9.2f}   pi {pi.round(2)}   mu {mu.round(2)}   sigma {sigma.round(2)}")

w = pi * pdf(0.5, mu, sigma)
print("a point at x = 0.5 belongs to the two components with probabilities", (w / w.sum()).round(3))
`, after: `EM recovers the hidden groups (about 60/40, means near −2 and 3, spreads 0.8 and 1.5) and gives soft answers for points in between.` },
      { type: 'check', question: { type: 'mcq', q: 'What does the E-step of EM compute for a Gaussian mixture?', options: ['New means and variances', 'The responsibilities: the probability that each component generated each point', 'The number of clusters', 'A random restart'], answer: 1, explain: 'The M-step then re-estimates the parameters from those soft assignments.' } },
      { type: 'text', title: 'How many components? BIC', md: `More components always fit the training data at least as well, so the likelihood alone always prefers more. The **Bayesian information criterion** adds a complexity penalty; choose the K with the lowest value:

$$ BIC = −2 · log L + p · log n $$
:::where
log L — the maximized log-likelihood of the model
p — the number of free parameters (weights, means and covariance entries)
n — the number of data points
:::` },
      { type: 'text', title: 'DBSCAN and hierarchical clustering', md: `**DBSCAN** (density-based clustering) has two settings: a radius eps and a minimum number of neighbours. Points with enough neighbours within eps are **core points**; clusters are chains of connected core points plus their neighbours; everything else is **noise**. It finds clusters of arbitrary shape, needs no K, and labels outliers — but struggles when clusters differ in density, and in high dimensions.

**Hierarchical (agglomerative) clustering** starts with every point as its own cluster and repeatedly merges the two closest clusters, producing a tree (a **dendrogram**). Cutting the tree at some height gives the clusters. The **linkage** defines "closest": single (nearest points — follows chains), complete (farthest points), average, or Ward (smallest increase in variance). Great for small data and nested structure, but its memory grows with n².

| Method | Cluster shapes | Needs K? | Soft assignments | Handles noise |
| --- | --- | --- | --- | --- |
| k-means | round, similar size | yes | no | no |
| Gaussian mixture | elliptical, different sizes | yes | yes | no |
| DBSCAN | any shape | no | no | yes |
| Agglomerative | depends on linkage | cut the tree | no | no |` },
      { type: 'code', title: 'Comparing clustering methods', code: `
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

X1, y1 = make_moons(300, noise=0.06, random_state=0)
X2, y2 = make_blobs(300, centers=[[0, 0], [4, 4]], cluster_std=[0.4, 1.6], random_state=0)
models = {
    "k-means": lambda: KMeans(2, n_init=10, random_state=0),
    "Gaussian mixture": lambda: GaussianMixture(2, random_state=0),
    "DBSCAN": lambda: DBSCAN(eps=0.2, min_samples=5),
    "agglomerative (single)": lambda: AgglomerativeClustering(2, linkage="single"),
}
print("adjusted Rand index (1 = perfect, 0 = random):")
for name, make in models.items():
    s1 = adjusted_rand_score(y1, make().fit_predict(X1))
    s2 = adjusted_rand_score(y2, make().fit_predict(X2))
    print(f"  {name:24s} two moons {s1:5.2f}    blobs of different spread {s2:5.2f}")
`, after: `No method wins everywhere. Density and single-linkage methods follow the curved moons; the Gaussian mixture handles clusters with different spreads; k-means assumes similar round clusters.` },
      { type: 'check', question: { type: 'tf', q: 'DBSCAN needs you to specify the number of clusters in advance.', answer: false, explain: 'It needs a radius and a minimum neighbour count; the number of clusters falls out.' } }
    ],
    when: `- **GMM**: overlapping clusters, elliptical shapes, or when you need probabilities (soft assignments) or a density estimate (anomaly scores).
- **EM** in general: any maximum-likelihood problem with hidden variables (mixtures, missing data, hidden Markov models).
- **DBSCAN**: arbitrary shapes, unknown number of clusters, outliers to discard (e.g. GPS points).
- **Hierarchical**: small datasets, nested groupings (taxonomies), when you want to inspect the dendrogram.`,
    pitfalls: `- EM converging to a poor local maximum — use multiple restarts.
- Components collapsing onto single points (variance → 0) — add a small variance floor or regularization.
- Choosing K by likelihood alone — use BIC or held-out likelihood.
- Unscaled features distorting distances and covariances.
- DBSCAN with one eps for clusters of very different density.`,
    takeaways: [
      'A GMM models the data as a weighted sum of Gaussians, with hidden cluster labels.',
      'Responsibilities are soft assignments: the posterior probability of each component.',
      'EM alternates E (responsibilities) and M (weighted re-estimates); likelihood never decreases.',
      'Choose K with BIC; k-means is EM with hard, round clusters.',
      'DBSCAN finds arbitrary shapes and noise; agglomerative clustering builds a dendrogram.'
    ],
    quiz: [
      { type: 'mcq', q: 'How does a Gaussian mixture assign a point that sits between two clusters?', options: ['To the nearest centre only', 'With probabilities for each component (soft assignment)', 'It discards it', 'Randomly'], answer: 1, explain: 'Responsibilities express the uncertainty.' },
      { type: 'mcq', q: 'Why does EM need several random restarts?', options: ['It is slow', 'It converges to a local maximum of the likelihood that depends on the start', 'Each restart adds a cluster', 'To estimate BIC'], answer: 1, explain: 'Different starts can end in different solutions; keep the best likelihood.' },
      { type: 'fill', q: 'After an E-step, three points have responsibilities 0.9, 0.5 and 0.1 for component 1. What is N_1, its effective number of points?', answer: ['1.5'], explain: '0.9 + 0.5 + 0.1 = 1.5.' },
      { type: 'tf', q: 'Adding more mixture components can never decrease the maximized training likelihood.', answer: true, explain: 'That is why a penalty such as BIC is needed.' },
      { type: 'mcq', q: 'Which method naturally labels some points as noise?', options: ['k-means', 'Gaussian mixture', 'DBSCAN', 'PCA'], answer: 2, explain: 'Points that are neither core points nor near one are noise.' }
    ],
    problems: [
      {
        id: 'gmm-e-step', title: 'GMM Responsibilities (E-step)', difficulty: 'medium',
        prompt: `For a one-dimensional Gaussian mixture, implement:

- \`normal_pdf(x, mu, sigma)\` → the Gaussian density exp(−(x − μ)² / (2σ²)) / (σ√(2π)), working element-wise on arrays.
- \`e_step(x, pi, mu, sigma)\` → the responsibilities as an (n, K) array: x has n points, and pi, mu, sigma each have K entries. Row i is π_k·N(x_i | μ_k, σ_k) normalized to sum to 1.

\`\`\`text
e_step([0.0], [0.5, 0.5], [-1.0, 1.0], [1.0, 1.0])  ->  [[0.5, 0.5]]
\`\`\``,
        starter: `
import numpy as np

def normal_pdf(x, mu, sigma):
    # TODO
    pass

def e_step(x, pi, mu, sigma):
    # TODO
    pass
`,
        solution: `
import numpy as np

def normal_pdf(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def e_step(x, pi, mu, sigma):
    x = np.asarray(x, dtype=float)[:, None]
    pi, mu, sigma = (np.asarray(a, dtype=float) for a in (pi, mu, sigma))
    dens = pi * normal_pdf(x, mu, sigma)
    return dens / dens.sum(axis=1, keepdims=True)
`,
        hints: ['`x[:, None]` makes x a column so it broadcasts against the K component parameters.', 'Normalize each row with `dens / dens.sum(axis=1, keepdims=True)`.'],
        explain: `Broadcasting an (n, 1) column against K parameters evaluates every point under every component in one expression. Normalizing each row is just Bayes' rule.`,
        tests: [
          { call: 'e_step([0.0], [0.5, 0.5], [-1.0, 1.0], [1.0, 1.0])', expect: 'np.array([[0.5, 0.5]])' },
          { name: 'a point far to the right', code: 'r = e_step([10.0], [0.5, 0.5], [-1.0, 1.0], [1.0, 1.0])\nassert r[0, 1] > 0.999999', hidden: false },
          { name: 'rows sum to one', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(0, 3, 50)', code: 'r = e_step(x, [0.2, 0.5, 0.3], [-3.0, 0.0, 4.0], [1.0, 0.5, 2.0])\nassert r.shape == (50, 3)\nassert np.allclose(r.sum(axis=1), 1)', hidden: true },
          { name: 'random mixture vs reference', setup: 'rng = np.random.default_rng(1)\nx = rng.normal(0, 2, 20)', call: 'e_step(x, [0.3, 0.7], [-1.0, 2.0], [0.8, 1.5])', hidden: true }
        ]
      },
      {
        id: 'gmm-em-fit', title: 'Fit a Mixture with EM', difficulty: 'hard',
        prompt: `Complete EM for a one-dimensional mixture (\`e_step\` is provided):

- \`m_step(x, r)\` → \`(pi, mu, sigma)\` from responsibilities r (n × K): N_k = Σ_i r_ik, π_k = N_k / n, μ_k = Σ_i r_ik x_i / N_k, σ_k = √(Σ_i r_ik (x_i − μ_k)² / N_k).
- \`log_likelihood(x, pi, mu, sigma)\` → Σ_i log Σ_k π_k N(x_i | μ_k, σ_k), as a float.
- \`em(x, pi, mu, sigma, iters=50)\` → \`(pi, mu, sigma, lls)\`: each iteration runs the E-step, then the M-step, then appends the log-likelihood of the **new** parameters to \`lls\`.`,
        starter: `
import numpy as np

def normal_pdf(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def e_step(x, pi, mu, sigma):
    dens = pi * normal_pdf(np.asarray(x, dtype=float)[:, None], mu, sigma)
    return dens / dens.sum(axis=1, keepdims=True)

def m_step(x, r):
    # TODO
    pass

def log_likelihood(x, pi, mu, sigma):
    # TODO
    pass

def em(x, pi, mu, sigma, iters=50):
    pi, mu, sigma = (np.asarray(a, dtype=float) for a in (pi, mu, sigma))
    lls = []
    # TODO
    return pi, mu, sigma, lls
`,
        solution: `
import numpy as np

def normal_pdf(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def e_step(x, pi, mu, sigma):
    dens = pi * normal_pdf(np.asarray(x, dtype=float)[:, None], mu, sigma)
    return dens / dens.sum(axis=1, keepdims=True)

def m_step(x, r):
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    Nk = r.sum(axis=0)
    pi = Nk / len(x)
    mu = (r * x[:, None]).sum(axis=0) / Nk
    sigma = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk)
    return pi, mu, sigma

def log_likelihood(x, pi, mu, sigma):
    dens = pi * normal_pdf(np.asarray(x, dtype=float)[:, None], mu, sigma)
    return float(np.log(dens.sum(axis=1)).sum())

def em(x, pi, mu, sigma, iters=50):
    pi, mu, sigma = (np.asarray(a, dtype=float) for a in (pi, mu, sigma))
    lls = []
    for _ in range(iters):
        r = e_step(x, pi, mu, sigma)
        pi, mu, sigma = m_step(x, r)
        lls.append(log_likelihood(x, pi, mu, sigma))
    return pi, mu, sigma, lls
`,
        hints: ['Sum the responsibilities down each column to get N_k.', 'Weighted means: `(r * x[:, None]).sum(axis=0) / Nk`.', 'Compute the variance around the new means.'],
        explain: `The hidden tests check EM's guarantee — the log-likelihood never goes down — and that it recovers the true parameters of well-separated data from a rough start.`,
        tests: [
          { name: 'M-step with hard assignments', call: 'm_step([0.0, 2.0, 10.0, 12.0], [[1, 0], [1, 0], [0, 1], [0, 1]])', expect: '(np.array([0.5, 0.5]), np.array([1.0, 11.0]), np.array([1.0, 1.0]))' },
          { name: 'log-likelihood of one point', call: 'log_likelihood([0.0], np.array([1.0]), np.array([0.0]), np.array([1.0]))', expect: '-np.log(np.sqrt(2 * np.pi))' },
          { name: 'recovers well-separated components', setup: 'rng = np.random.default_rng(0)\nx = np.concatenate([rng.normal(-4, 1.0, 400), rng.normal(5, 2.0, 600)])', code: 'pi, mu, sigma, lls = em(x, [0.5, 0.5], [-1.0, 1.0], [1.0, 1.0], 60)\nassert np.allclose(np.sort(mu), [-4, 5], atol=0.25)\nassert abs(min(pi) - 0.4) < 0.05\nassert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:])), "the log-likelihood must never decrease"', hidden: true },
          { name: 'reference run', setup: 'rng = np.random.default_rng(3)\nx = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 0.5, 30)])', call: 'em(x, [0.5, 0.5], [-0.5, 2.0], [1.0, 1.0], 10)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Gaussian mixture models', url: 'https://scikit-learn.org/stable/modules/mixture.html' },
      { title: 'scikit-learn — Clustering (overview and comparison)', url: 'https://scikit-learn.org/stable/modules/clustering.html' },
      { title: 'Ester et al. — A Density-Based Algorithm for Discovering Clusters (DBSCAN)', url: 'https://cdn.aaai.org/KDD/1996/KDD96-037.pdf' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
