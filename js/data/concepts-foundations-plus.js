/* ML Atlas – content: extra foundations (eigenvectors & SVD, MLE & MAP, statistical inference, pandas & EDA, Gaussian mixtures & EM) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'eigen-svd', branch: 'math', title: 'Eigenvectors, Eigenvalues & the SVD', short: 'Eigenvectors & SVD', level: 2, minutes: 45, prereqs: ['vectors-matrices'],
    tags: ['eigenvector', 'eigenvalue', 'svd', 'singular value decomposition', 'rank', 'low-rank approximation', 'power iteration', 'linear algebra'],
    summary: 'What a matrix does to space, the special directions it only stretches (eigenvectors), and the singular value decomposition that writes any matrix as rotate–stretch–rotate: the mathematics behind PCA, low-rank approximation, LoRA and recommender systems.',
    why: `Many ML ideas are really statements about the structure of a matrix. PCA looks for the directions in which data varies most; LoRA assumes a weight update is "low rank"; matrix factorization compresses a ratings table; RNN gradients explode or vanish depending on how much a weight matrix stretches vectors. **Eigenvectors** and the **singular value decomposition (SVD)** are the tools that make "important directions" and "rank" precise, so these lessons stop being recipes and start making sense.`,
    objectives: [
      'See a matrix as a transformation that rotates, stretches and projects vectors',
      'Find and interpret eigenvectors and eigenvalues, and compute the top one with power iteration',
      'Decompose any matrix with the SVD and interpret U, S and V',
      'Build the best rank-k approximation and explain why "low rank" means "compressible"'
    ],
    sections: [
      { type: 'text', title: 'A matrix is a transformation', md: `Multiplying a vector by a matrix produces a new vector: the matrix **moves space around** (it can rotate, stretch, squash, shear or project it). The columns of the matrix show where the basis vectors [1, 0] and [0, 1] end up.

| Matrix | What it does |
| --- | --- |
| [[2, 0], [0, 0.5]] | stretches x by 2, squashes y to half |
| [[0, −1], [1, 0]] | rotates by 90° |
| [[1, 1], [0, 1]] | shears: slides the top of the plane sideways |
| [[1, 0], [0, 0]] | projects onto the x-axis (throws y away) |

Most vectors change **direction** when you multiply them by a matrix. A few special directions don't, and they reveal what the matrix really does.` },
      { type: 'text', title: 'Eigenvectors: directions that only get stretched', md: `An **eigenvector** of a square matrix is a direction that the matrix only stretches (or flips), never turns:

$$ A v = λ v $$
:::where
A — a square matrix (n × n)
v — an eigenvector: a non-zero vector whose direction A leaves unchanged
λ — "lambda": the eigenvalue, the factor by which A stretches v (a negative λ flips it, |λ| < 1 shrinks it)
:::

Example: for A = [[2, 1], [1, 2]], the vector [1, 1] becomes [3, 3] = 3·[1, 1], so it is an eigenvector with λ = 3; [1, −1] becomes [1, −1], an eigenvector with λ = 1.

Why it matters:
- **Symmetric matrices** (such as covariance matrices) have real eigenvalues and mutually perpendicular eigenvectors. For a covariance matrix those eigenvectors are exactly the **principal axes** of the data: PCA.
- **Repeated multiplication**: A^k v = λ^k v. The direction with the largest |λ| eventually dominates. That is the idea behind **power iteration** (and PageRank), and why recurrent networks explode when |λ| > 1 and vanish when |λ| < 1.` },
      { type: 'text', title: "Finding eigenvalues: the characteristic equation", md: `Rewrite A v = λ v as (A − λI) v = 0. A non-zero v can only solve this if A − λI flattens some direction to zero, which happens exactly when its determinant is zero:

$$ det(A − λI) = 0          for a 2 × 2 matrix:   λ² − tr(A) · λ + det(A) = 0 $$
:::where
I — the identity matrix (1s on the diagonal, 0 elsewhere)
det(…) — the determinant: the factor by which a matrix scales areas (volumes in higher dimensions); 0 means some direction is flattened
tr(A) — the trace, the sum of the diagonal entries; it always equals the sum of the eigenvalues
det(A) — the determinant of A; it always equals the product of the eigenvalues
:::

Worked example, A = [[2, 1], [1, 2]]: tr(A) = 4 and det(A) = 2·2 − 1·1 = 3, so λ² − 4λ + 3 = (λ − 3)(λ − 1) = 0, giving λ = 3 and λ = 1, the values found above. Substituting λ = 3 into (A − 3I) v = 0 gives v ∝ [1, 1].

For a **symmetric** matrix the eigenvectors can be chosen perpendicular and of length 1. Collecting them gives the **eigendecomposition**:

$$ A = Q Λ Qᵀ          A^k = Q Λ^k Qᵀ $$
:::where
Q — a matrix whose columns are the unit eigenvectors; it is orthogonal (Qᵀ Q = I), a pure rotation or reflection
Λ — "Lambda": a diagonal matrix holding the eigenvalues
Qᵀ — the transpose of Q, which undoes it
A^k — A multiplied by itself k times: only the eigenvalues get raised to the power k
:::

Read Q Λ Qᵀ right to left: rotate into the eigenvector axes, stretch each axis by its eigenvalue, rotate back. Powers of A only power the stretches, which is why the largest eigenvalue dominates after many multiplications. For big matrices, computers never solve the characteristic polynomial (that is numerically unstable); \`np.linalg.eigh\` and friends use iterative algorithms instead.` },
      { type: 'check', question: {"type":"mcq","q":"A 2 × 2 matrix has eigenvalues 4 and 1. What are its trace and determinant?","options":["Trace 5, determinant 4","Trace 4, determinant 5","Trace 5, determinant 5","Trace 3, determinant 4"],"answer":0,"explain":"The trace is the sum of the eigenvalues (4 + 1 = 5) and the determinant their product (4 × 1 = 4)."} },
      { type: 'demo', title: 'Interactive: what a 2×2 matrix does to directions', demo: 'eigen' },
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
      { type: 'chart', title: "How fast power iteration converges", after: `Each iteration shrinks the error by the ratio |λ₂ / λ₁| of the two largest eigenvalues. With a ratio of 1/3 the error falls by a factor of 10 every two steps; with 0.9 it takes about 22 steps per factor of 10. A large **eigengap** makes power iteration (and PageRank, and PCA solvers built on it) fast.`, chart: {"kind":"line","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],"yLog":true,"points":false,"series":[{"name":"eigenvalues 3 and 1 (ratio 0.33)","y":[0.866,0.5,0.189,0.06402,0.02138,0.007128,0.002376,0.000792,0.000264,0.000088,0.00002933,0.000009777,0.000003259,0.000001086,3.623e-7,1.192e-7,3.942e-8,2.107e-8,2.107e-8,1.49e-8,1e-12,1.49e-8,1.49e-8,1e-12,1e-12,1.49e-8,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12]},{"name":"eigenvalues 1 and 0.9 (ratio 0.9)","y":[0.866,0.8417,0.8143,0.7839,0.7507,0.715,0.6772,0.638,0.5977,0.5572,0.517,0.4776,0.4394,0.4029,0.3684,0.3359,0.3056,0.2775,0.2516,0.2278,0.2061,0.1862,0.1681,0.1517,0.1369,0.1234,0.1112,0.1002,0.09028,0.08131,0.07323,0.06594,0.05937,0.05345,0.04812,0.04332,0.03899,0.0351,0.03159,0.02843,0.02559]}],"xLabel":"iteration","yLabel":"error: sin(angle to the top eigenvector)","yMin":1e-12} },
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

Connections: the singular values are the square roots of the eigenvalues of AᵀA, and the columns of V are its eigenvectors. For centred data X, AᵀA is (proportional to) the covariance matrix, so the SVD of the data *is* PCA.` },
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
      { type: 'chart', title: "Singular values of a nearly low-rank matrix", md: `A 60 × 40 matrix built from 4 hidden factors plus a little noise. Plotting its singular values in order (a *scree plot*) reveals the hidden rank:`, after: `Four large singular values, then a sudden drop to a flat floor of small ones that only describe the noise. The elbow tells you how many components are worth keeping, which is exactly how you choose the number of components in [PCA](#/concept/pca).`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"yLog":true,"series":[{"name":"singular value σₖ","y":[65.354,56.885,37.394,36.02,3.945,3.705,3.641,3.346,3.115,3.049,3.016,2.848,2.79,2.7,2.613,2.529,2.468,2.453,2.32,2.233]}],"xLabel":"k (singular values in decreasing order)","yLabel":"σₖ (log scale)"} },
      { type: 'chart', title: "Reconstruction error against rank", after: `Keeping 4 components already captures almost everything; every extra component only shaves off a little noise. Storing the rank-4 version takes 4 · (60 + 40 + 1) = 404 numbers instead of 2,400.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"series":[{"name":"relative error of the rank-k approximation","y":[0.767,0.526,0.377,0.132,0.126,0.121,0.115,0.11,0.106,0.102,0.097,0.093,0.089,0.085,0.081,0.077,0.073,0.069,0.065,0.062]}],"xLabel":"rank k of the approximation","yLabel":"‖M − Mₖ‖ / ‖M‖","vlines":[{"x":4,"label":"true rank 4"}],"yMin":0} },
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
      { type: 'text', title: "The SVD at work: least squares and conditioning", md: `The SVD also tells you when solving a linear system is safe. Least squares (fitting a linear regression) solves X w ≈ y; with the SVD X = U S Vᵀ the best solution is

$$ ŵ = V S⁻¹ Uᵀ y = X⁺ y          κ(X) = σ_max / σ_min $$
:::where
X — the n × d data matrix; y — the targets
X⁺ — the pseudo-inverse of X: invert the non-zero singular values and transpose the rotations
S⁻¹ — the diagonal matrix of the values 1/σᵢ: small singular values become huge
κ(X) — "kappa": the condition number, the ratio of the largest to the smallest singular value
:::

Because the solution divides by every singular value, a tiny σ, caused for example by two nearly identical features such as height in cm and height in inches, multiplies whatever noise lies in that direction by an enormous factor. The **condition number** measures this: close to 1 is perfectly stable, and the larger it is, the more the individual weights swing with small changes to the data.

**Ridge regression** fixes it gently. It replaces each 1/σᵢ by σᵢ / (σᵢ² + λ), which is almost 1/σᵢ for large singular values but stays small for tiny ones: the unstable directions are damped instead of amplified. That is the linear-algebra view of [regularization](#/concept/regularization).

Other places where the SVD does the work: [PCA](#/concept/pca) (the right singular vectors of the centred data), latent semantic analysis (an SVD of a term–document matrix finds "topics"), [recommender systems](#/concept/recommender-systems) (a low-rank user × item matrix) and LoRA fine-tuning (low-rank weight updates).` },
      { type: 'code', title: "A nearly singular regression", code: `import numpy as np
rng = np.random.default_rng(5)
n = 100
height_cm = rng.normal(170, 10, n)
height_in = height_cm / 2.54 + rng.normal(0, 0.005, n)      # the same measurement in inches, tiny noise
X = np.column_stack([height_cm, height_in])
X = X - X.mean(axis=0)
y = 0.5 * height_cm + rng.normal(0, 5, n)                   # the truth: 0.5 per cm
y = y - y.mean()

s = np.linalg.svd(X, compute_uv=False)
print("singular values:", s.round(3), "  condition number:", round(float(s[0] / s[-1])))
print("ill-conditioned:", s[0] / s[-1] > 1000)

for trial in range(3):                                       # refit on resampled data
    idx = rng.choice(n, n)
    w_ls = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    w_ridge = np.linalg.solve(X[idx].T @ X[idx] + 10 * np.eye(2), X[idx].T @ y[idx])
    print(f"resample {trial + 1}: least squares w = {w_ls.round(1)}, effect per cm {w_ls[0] + w_ls[1] / 2.54:.2f}"
          f"  |  ridge w = {w_ridge.round(2)}, effect per cm {w_ridge[0] + w_ridge[1] / 2.54:.2f}")`, after: `The second singular value is tiny, so the condition number is in the thousands. Least squares splits the effect between the two copies of height almost arbitrarily: large weights of opposite sign that change from one resample to the next. Yet the combined effect per cm, w₁ + w₂ / 2.54, barely changes (0.55 to 0.61, near the true 0.5 given the noise), so the predictions are fine and only the individual weights are meaningless. Ridge returns small weights that change far less, with the same combined effect.`, predict: {"type":"mcq","q":"Two features are the same measurement in different units, plus tiny noise. What will the condition number of the data matrix be?","options":["About 1","About 2.54","Very large (thousands or more)","Exactly 0"],"answer":2,"explain":"The two columns point in almost the same direction, so one singular value is tiny and the ratio of the largest to the smallest is huge.","expect":["ill-conditioned: True"]} },
      { type: 'check', question: { type: 'tf', q: 'Every matrix, even a non-square one, has a singular value decomposition.', answer: true, explain: 'Unlike eigen-decomposition, the SVD exists for any m × n matrix.' } }
    ],
    when: `- **PCA** and dimensionality reduction (via the SVD of centred data).
- Compressing or analysing weights: low-rank layers, LoRA, model pruning.
- Latent-factor models: recommender systems, latent semantic analysis of documents.
- Solving least squares stably (the pseudo-inverse) and diagnosing ill-conditioned problems.
- Stability analysis: eigenvalues of recurrent weight matrices, of Hessians in optimization, of graph Laplacians.`,
    pitfalls: `- Using \`np.linalg.eig\` on symmetric matrices: \`eigh\` is faster, more accurate and returns sorted real eigenvalues.
- Comparing signs of eigenvectors or singular vectors: −v is just as valid as v.
- Forgetting to centre the data before using the SVD for PCA.
- Computing a full SVD of a huge matrix when a truncated or randomized SVD of the top k components would do.
- Mixing up V and Vᵀ: NumPy returns Vᵀ (rows are the directions).`,
    takeaways: [
      'A matrix is a transformation; eigenvectors are the directions it only stretches: A v = λ v.',
      'Symmetric matrices have perpendicular eigenvectors, for a covariance matrix they are the principal axes.',
      'Every matrix has an SVD A = U S Vᵀ: rotate, stretch, rotate.',
      'The rank is the number of non-zero singular values; keeping the top k gives the best rank-k approximation.',
      'Fast-decaying singular values mean the matrix is compressible: the idea behind PCA, LoRA and matrix factorization.'
    ],
    quiz: [
      { type: 'mcq', q: 'For A = [[3, 0], [0, 1]], which vector is an eigenvector with eigenvalue 3?', options: ['[0, 1]', '[1, 0]', '[1, 1]', '[3, 1]'], answer: 1, explain: 'A·[1, 0] = [3, 0] = 3·[1, 0].' },
      { type: 'fill', q: 'A 1000 × 500 matrix is stored as a rank-20 approximation (U_k, the singular values and V_k). How many numbers is that?', answer: ['30020', '30,020'], explain: '20 × (1000 + 500 + 1) = 30,020 instead of 500,000.' },
      { type: 'mcq', q: 'What does the number of non-zero singular values tell you?', options: ['The size of the matrix', 'Its rank: how many independent directions it really uses', 'Its largest entry', 'Its trace'], answer: 1, explain: 'Zero singular values mean collapsed directions.' },
      { type: 'tf', q: 'If a matrix\'s singular values decay quickly, a low-rank approximation of it will be accurate.', answer: true, explain: 'The error equals the energy in the dropped singular values.' },
      { type: 'mcq', q: 'Why does LoRA relate to the SVD?', options: ['It computes an SVD at every training step', 'It assumes the weight update has low rank, only a few significant singular values', 'It uses eigenvalues as learning rates', 'It does not relate'], answer: 1, explain: 'B·A with small inner dimension r is a rank-r matrix.' },
      {"type":"mcq","q":"A 2 × 2 matrix has trace 5 and determinant 6. What are its eigenvalues?","options":["1 and 6","2 and 3","5 and 6","−2 and −3"],"answer":1,"explain":"The eigenvalues sum to the trace (5) and multiply to the determinant (6): λ² − 5λ + 6 = (λ − 2)(λ − 3)."},
      {"type":"mcq","q":"The condition number of a data matrix is 10⁸. What does that suggest for least-squares regression?","options":["The fit will be perfect","Some features are nearly linearly dependent, so the individual weights are unstable; remove redundant features or use ridge","Only more rows will help","Nothing: condition numbers do not matter for regression"],"answer":1,"explain":"A huge ratio of singular values means some direction is almost flat; dividing by its tiny singular value amplifies noise. Ridge damps exactly those directions."},
      {"type":"tf","q":"Power iteration converges fastest when the two largest eigenvalues are almost equal in size.","answer":false,"explain":"The error shrinks by |λ₂ / λ₁| per step, so a ratio close to 1 (almost equal eigenvalues) means very slow convergence. A large gap is fast."}
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
        explain: `The SVD orders directions by importance, so truncating it is optimal: no other rank-k matrix is closer in squared error. "Energy" (Σσ²) is the same quantity PCA reports as explained variance.`,
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
      { title: '3Blue1Brown, Essence of Linear Algebra (videos)', url: 'https://www.3blue1brown.com/topics/linear-algebra' },
      { title: 'Gilbert Strang, Linear Algebra and Learning from Data', url: 'https://math.mit.edu/~gs/learningfromdata/' },
      { title: 'Steve Brunton, Singular Value Decomposition (video series)', url: 'https://www.youtube.com/watch?v=gXbThCXjZFM' }
    ]
  },

  // =====================================================================================
  {
    id: 'mle-map', branch: 'ml', title: 'Maximum Likelihood & MAP Estimation', short: 'MLE & MAP', level: 2, minutes: 45, prereqs: ['probability', 'loss-functions'],
    tags: ['maximum likelihood', 'mle', 'map', 'likelihood', 'log-likelihood', 'negative log-likelihood', 'prior', 'posterior', 'regularization', 'bayesian'],
    summary: 'Where loss functions come from: the likelihood of parameters given data, maximum-likelihood estimation, why squared error and cross-entropy are negative log-likelihoods, and how adding a prior turns MLE into MAP estimation, which is exactly L2 or L1 regularization.',
    why: `Loss functions can look like arbitrary choices. They aren't: almost every standard loss is the **negative log-likelihood** of a probabilistic model of the data. Squared error assumes Gaussian noise; cross-entropy assumes a categorical distribution; training a language model is maximum-likelihood estimation on text. Add a prior belief about the parameters and you get **MAP** estimation, which turns out to be regularization. This one idea explains a large part of the course.`,
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

The likelihood is a function of θ with the data held fixed: it is *not* a probability distribution over θ.

**Maximum-likelihood estimation (MLE)** picks the parameters that make the observed data most probable. Equivalently, it minimizes the **negative log-likelihood (NLL)**, which is why "training" so often means "minimizing an NLL".` },
      { type: 'text', title: 'Example: estimating a coin', md: `Flip a coin n times and see h heads. With θ = P(heads):

$$ log L(θ) = h · log θ + (n − h) · log(1 − θ)          θ̂_MLE = h / n $$
:::where
n — the number of flips
h — the number of heads
θ — the probability of heads we want to estimate
θ̂_MLE — "theta hat": the maximum-likelihood estimate, found by setting the derivative of log L to zero
:::

The MLE is simply the observed fraction: sensible. But with 3 flips and 3 heads it says θ̂ = 1: the coin can *never* land tails. With little data, MLE trusts the sample completely and **overfits**.` },
      { type: 'chart', title: "The likelihood of a coin's bias, after 10 and after 100 flips", after: `Both curves peak at the same place, θ = 0.7: the maximum-likelihood estimate is the observed share of heads. But with 100 flips the curve is far narrower: many values that were plausible after 10 flips (0.5 or 0.9) are now very unlikely. The width of the likelihood is a measure of uncertainty, which is the bridge to confidence intervals and to Bayesian posteriors.`, chart: {"kind":"line","x":[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99],"points":false,"series":[{"name":"7 heads in 10 flips","y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001,0.001,0.002,0.002,0.003,0.004,0.005,0.007,0.009,0.012,0.015,0.018,0.023,0.028,0.034,0.041,0.049,0.058,0.068,0.079,0.092,0.107,0.123,0.14,0.159,0.18,0.202,0.226,0.252,0.28,0.309,0.339,0.371,0.405,0.439,0.475,0.511,0.548,0.586,0.624,0.662,0.699,0.736,0.771,0.806,0.838,0.869,0.897,0.923,0.945,0.964,0.98,0.991,0.998,1,0.998,0.99,0.978,0.96,0.938,0.91,0.878,0.841,0.8,0.755,0.706,0.654,0.6,0.544,0.487,0.429,0.373,0.318,0.265,0.215,0.169,0.128,0.093,0.063,0.039,0.022,0.01,0.003,0]},{"name":"70 heads in 100 flips","y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001,0.001,0.002,0.005,0.009,0.016,0.028,0.046,0.075,0.115,0.172,0.246,0.338,0.448,0.569,0.695,0.813,0.911,0.977,1,0.976,0.907,0.8,0.668,0.527,0.392,0.273,0.177,0.107,0.06,0.031,0.014,0.006,0.002,0.001,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}],"xLabel":"θ, the probability of heads","yLabel":"likelihood (scaled so the peak is 1)","vlines":[{"x":0.7,"label":"MLE θ = 0.7"}],"yMin":0} },
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
      { type: 'text', title: "Finding the maximum: calculus, then gradient descent", md: `For simple models you can find the MLE on paper: set the derivative of the log-likelihood to zero and solve. For the coin:

$$ d/dθ log L(θ) = h/θ − (n − h)/(1 − θ) = 0     ⇒     θ̂ = h / n $$
:::where
d/dθ log L(θ) — the slope of the log-likelihood as θ changes; at the peak it is zero
h/θ — the pull towards larger θ from the h heads
(n − h)/(1 − θ) — the pull towards smaller θ from the n − h tails
θ̂ — the value where the two pulls balance: the observed share of heads
:::

The same recipe gives the familiar estimators of other distributions:

| Model | Parameters | Maximum-likelihood estimate |
| --- | --- | --- |
| Bernoulli (coin) | θ = P(1) | the share of 1s |
| Gaussian | mean μ, variance σ² | the sample mean, and the average squared deviation (dividing by n, not n − 1) |
| Poisson (counts per hour) | rate λ | the average count |
| Categorical (K classes) | class probabilities | each class's share of the data |

Most models are not that simple: there is no closed form for logistic regression or a neural network. Then the negative log-likelihood is minimised numerically with [gradient descent](#/concept/gradient-descent). That is all "training a classifier" means: maximum likelihood, solved by an optimiser.

The *shape* of the log-likelihood carries information too. A sharply curved peak means the data pins the parameter down; a flat peak means many values are almost equally plausible. For the coin, the curvature at the peak gives back the familiar standard error √(θ̂(1 − θ̂)/n) used for confidence intervals in [Statistical Inference](#/concept/stat-inference).` },
      { type: 'code', title: "Maximum likelihood for counts, and by gradient descent", code: `import numpy as np
rng = np.random.default_rng(4)

# 1) Poisson: support tickets per hour. log L(lam) = sum(k * log(lam) - lam) + a constant
tickets = np.array([2, 0, 3, 1, 4, 2, 5, 3, 2, 3])
lams = np.linspace(0.1, 6, 591)
loglik = np.array([np.sum(tickets * np.log(l) - l) for l in lams])
best = lams[np.argmax(loglik)]
print("grid MLE of the rate:", round(float(best), 2), "   sample mean:", tickets.mean())
print("MLE equals the sample mean:", abs(best - tickets.mean()) < 0.01)

# 2) Logistic regression has no closed form: minimise the negative log-likelihood by gradient descent
X = rng.normal(size=(500, 2))
true_w, true_b = np.array([2.0, -1.0]), 0.5
y = (rng.random(500) < 1 / (1 + np.exp(-(X @ true_w + true_b)))).astype(float)
w, b = np.zeros(2), 0.0
for step in range(2001):
    p = 1 / (1 + np.exp(-(X @ w + b)))
    nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    if step in (0, 10, 100, 2000):
        print(f"step {step:4d}: NLL {nll:.4f}   w = {w.round(2)}   b = {b:.2f}")
    w -= 0.5 * X.T @ (p - y) / len(y)            # gradient of the mean NLL
    b -= 0.5 * np.mean(p - y)
print("true parameters: w =", true_w, "  b =", true_b)`, after: `The grid search over λ lands on the sample mean, as the calculus predicts. For logistic regression the NLL falls quickly and the estimated weights end up close to the true ones; with more data they would get closer still. scikit-learn's \`LogisticRegression\` minimises the same NLL plus, by default, an L2 penalty, which makes it a MAP estimate, as the next sections explain.`, predict: {"type":"mcq","q":"Support tickets per hour were [2, 0, 3, 1, 4, 2, 5, 3, 2, 3]. What is the maximum-likelihood estimate of the Poisson rate?","options":["2","2.5","3","5"],"answer":1,"explain":"For a Poisson model the MLE of the rate is the average count: 25 / 10 = 2.5.","expect":["MLE equals the sample mean: True"]} },
      { type: 'text', title: 'Where squared error and cross-entropy come from', md: `Assume a regression target is the model's prediction plus Gaussian noise: y = f(x; θ) + ε with ε ~ N(0, σ²). The negative log-likelihood of one example is

$$ −log p(y | x, θ) = ( y − f(x; θ) )² / (2σ²) + log( σ √(2π) ) $$
:::where
f(x; θ) — the model's prediction for input x
y — the observed target
σ — the standard deviation of the noise (a constant here)
log( σ √(2π) ) — a constant that does not depend on θ
:::

Minimizing the NLL over θ is therefore exactly minimizing the **squared error**: MSE *is* maximum likelihood under Gaussian noise. The same reasoning gives the other standard losses:

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
- a **Gaussian prior** w ~ N(0, τ²) gives −log p(w) = Σ w² / (2τ²) + const: that is **L2 regularization** (ridge, weight decay), with a stronger penalty for a narrower prior;
- a **Laplace prior** gives Σ |w| / b + const: **L1 regularization** (lasso);
- for the coin, a Beta(a, b) prior gives θ̂_MAP = (h + a − 1) / (n + a + b − 2). With a = b = 2 that is (h + 1) / (n + 2): **Laplace (add-one) smoothing** is a prior too.

With more data the likelihood dominates and MAP approaches MLE. Fully **Bayesian** methods keep the whole posterior, not just its peak, which also gives uncertainty estimates.` },
      { type: 'chart', title: "Prior × likelihood = posterior, for a coin", md: `A prior that says "coins are usually close to fair" meets data with 9 heads in 10 flips:`, after: `The posterior is a compromise: it sits between the prior's centre (0.5) and the data's verdict (0.9), and it is narrower than either because it combines both sources of information. Its peak, the MAP estimate, is (9 + 5 − 1) / (10 + 5 + 5 − 2) = 13/18 ≈ 0.72. With 1,000 flips the likelihood would be so narrow that the prior would hardly move it.`, chart: {"kind":"line","x":[0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.125,0.13,0.135,0.14,0.145,0.15,0.155,0.16,0.165,0.17,0.175,0.18,0.185,0.19,0.195,0.2,0.205,0.21,0.215,0.22,0.225,0.23,0.235,0.24,0.245,0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.285,0.29,0.295,0.3,0.305,0.31,0.315,0.32,0.325,0.33,0.335,0.34,0.345,0.35,0.355,0.36,0.365,0.37,0.375,0.38,0.385,0.39,0.395,0.4,0.405,0.41,0.415,0.42,0.425,0.43,0.435,0.44,0.445,0.45,0.455,0.46,0.465,0.47,0.475,0.48,0.485,0.49,0.495,0.5,0.505,0.51,0.515,0.52,0.525,0.53,0.535,0.54,0.545,0.55,0.555,0.56,0.565,0.57,0.575,0.58,0.585,0.59,0.595,0.6,0.605,0.61,0.615,0.62,0.625,0.63,0.635,0.64,0.645,0.65,0.655,0.66,0.665,0.67,0.675,0.68,0.685,0.69,0.695,0.7,0.705,0.71,0.715,0.72,0.725,0.73,0.735,0.74,0.745,0.75,0.755,0.76,0.765,0.77,0.775,0.78,0.785,0.79,0.795,0.8,0.805,0.81,0.815,0.82,0.825,0.83,0.835,0.84,0.845,0.85,0.855,0.86,0.865,0.87,0.875,0.88,0.885,0.89,0.895,0.9,0.905,0.91,0.915,0.92,0.925,0.93,0.935,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.975,0.98,0.985,0.99,0.995],"points":false,"series":[{"name":"prior Beta(5, 5): probably fair","y":[0,0,0,0,0,0,0.001,0.001,0.002,0.003,0.005,0.006,0.009,0.011,0.015,0.018,0.023,0.028,0.034,0.041,0.049,0.058,0.068,0.078,0.09,0.103,0.117,0.132,0.149,0.166,0.185,0.206,0.227,0.25,0.274,0.299,0.326,0.353,0.383,0.413,0.444,0.477,0.511,0.546,0.582,0.62,0.658,0.697,0.738,0.779,0.821,0.863,0.907,0.951,0.995,1.041,1.086,1.132,1.179,1.225,1.272,1.319,1.366,1.412,1.459,1.506,1.552,1.597,1.643,1.688,1.732,1.775,1.818,1.86,1.901,1.941,1.98,2.018,2.055,2.09,2.124,2.157,2.189,2.218,2.247,2.274,2.299,2.322,2.344,2.364,2.382,2.399,2.413,2.426,2.436,2.445,2.452,2.457,2.46,2.461,2.46,2.457,2.452,2.445,2.436,2.426,2.413,2.399,2.382,2.364,2.344,2.322,2.299,2.274,2.247,2.218,2.189,2.157,2.124,2.09,2.055,2.018,1.98,1.941,1.901,1.86,1.818,1.775,1.732,1.688,1.643,1.597,1.552,1.506,1.459,1.412,1.366,1.319,1.272,1.225,1.179,1.132,1.086,1.041,0.995,0.951,0.907,0.863,0.821,0.779,0.738,0.697,0.658,0.62,0.582,0.546,0.511,0.477,0.444,0.413,0.383,0.353,0.326,0.299,0.274,0.25,0.227,0.206,0.185,0.166,0.149,0.132,0.117,0.103,0.09,0.078,0.068,0.058,0.049,0.041,0.034,0.028,0.023,0.018,0.015,0.011,0.009,0.006,0.005,0.003,0.002,0.001,0.001,0,0,0,0,0,0],"dash":true},{"name":"likelihood of 9 heads in 10 (normalised)","y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.002,0.002,0.002,0.002,0.003,0.003,0.003,0.004,0.004,0.005,0.006,0.006,0.007,0.008,0.009,0.01,0.011,0.013,0.014,0.016,0.017,0.019,0.021,0.023,0.026,0.029,0.032,0.035,0.038,0.042,0.046,0.05,0.055,0.06,0.065,0.071,0.077,0.084,0.091,0.099,0.107,0.116,0.126,0.136,0.147,0.158,0.171,0.184,0.198,0.212,0.228,0.245,0.262,0.281,0.3,0.321,0.343,0.366,0.391,0.416,0.443,0.472,0.502,0.533,0.566,0.6,0.636,0.674,0.713,0.754,0.797,0.842,0.889,0.937,0.988,1.04,1.094,1.151,1.209,1.269,1.332,1.396,1.463,1.531,1.602,1.674,1.748,1.825,1.903,1.983,2.065,2.148,2.233,2.32,2.407,2.496,2.586,2.677,2.769,2.861,2.953,3.045,3.137,3.228,3.319,3.408,3.496,3.581,3.665,3.745,3.822,3.895,3.963,4.026,4.083,4.134,4.178,4.213,4.239,4.256,4.262,4.256,4.237,4.203,4.155,4.09,4.007,3.905,3.782,3.636,3.466,3.271,3.047,2.794,2.509,2.19,1.834,1.44,1.005,0.526]},{"name":"posterior Beta(14, 6)","y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001,0.001,0.001,0.001,0.001,0.002,0.002,0.002,0.003,0.004,0.004,0.005,0.006,0.007,0.009,0.01,0.012,0.014,0.017,0.019,0.022,0.026,0.03,0.034,0.039,0.045,0.051,0.058,0.066,0.075,0.085,0.096,0.108,0.121,0.135,0.151,0.168,0.187,0.208,0.23,0.254,0.28,0.309,0.339,0.372,0.407,0.444,0.485,0.527,0.573,0.621,0.672,0.726,0.783,0.843,0.906,0.972,1.041,1.113,1.188,1.266,1.347,1.43,1.516,1.605,1.695,1.788,1.883,1.98,2.078,2.177,2.277,2.378,2.479,2.58,2.681,2.78,2.878,2.975,3.069,3.161,3.25,3.335,3.416,3.493,3.565,3.631,3.691,3.745,3.793,3.833,3.865,3.89,3.907,3.915,3.915,3.905,3.887,3.859,3.823,3.777,3.722,3.658,3.586,3.505,3.416,3.319,3.215,3.104,2.987,2.864,2.736,2.604,2.469,2.331,2.191,2.051,1.91,1.77,1.631,1.495,1.362,1.232,1.108,0.989,0.876,0.769,0.669,0.576,0.491,0.414,0.344,0.282,0.228,0.18,0.14,0.107,0.079,0.057,0.039,0.026,0.017,0.01,0.005,0.003,0.001,0,0,0,0]}],"xLabel":"θ, the probability of heads","yLabel":"density","yMin":0,"vlines":[{"x":0.9,"label":"MLE 0.9"},{"x":0.722,"label":"MAP 0.72"}]} },
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
`, after: `With 3 flips the prior pulls the estimate away from the extreme; with 300 flips it barely matters. With only 8 examples for 5 weights, the MAP (ridge) weights are shrunk towards zero: a smaller, more cautious model.`, predict: {"type":"mcq","q":"A coin lands heads 3 times in 3 flips, so the maximum-likelihood estimate of P(heads) is 1.0. What is the MAP estimate with a Beta(2, 2) prior?","options":["1.0","0.8","0.6","0.5"],"answer":1,"explain":"(h + a − 1) / (n + a + b − 2) = (3 + 1) / (3 + 2) = 0.8. The prior acts like one extra head and one extra tail, pulling the estimate away from the extreme; with 300 heads in 300 flips it barely matters (0.997).","expect":["MAP, Beta(2,2) prior 0.800"]} },
      { type: 'text', title: "MLE, MAP or fully Bayesian?", md: `| | MLE | MAP | Full Bayesian |
| --- | --- | --- | --- |
| Result | one best parameter value | one best value, pulled towards the prior | a whole distribution over parameter values |
| Objective | maximise p(data given θ) | maximise p(data given θ) · p(θ) | compute p(θ given data) |
| In practice | least squares, unregularised logistic regression, LLM pretraining | ridge, lasso, weight decay, smoothed counts | Gaussian processes, Bayesian A/B tests, probabilistic programming |
| Uncertainty | from the curvature of the likelihood, or the bootstrap | the same | built in: a credible interval for every parameter |
| Cost | cheap | cheap | often expensive (sampling or approximation) |

Two subtleties worth knowing:
- **MAP depends on how you parametrise.** The peak of a density moves when you change variables (say from θ to log θ); the full posterior does not have this problem. For point estimates it rarely matters in practice.
- **Priors are assumptions you can test.** Choosing the regularisation strength (the width of the prior) by [cross-validation](#/concept/cross-validation) is MAP with the prior tuned on held-out data: the most common workflow in applied machine learning.

For the full Bayesian treatment, continue with [Bayesian Methods](#/concept/bayesian-methods).` },
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
      { type: 'fill', q: 'With a Beta(2, 2) prior, a coin shows 3 heads in 3 flips. What is the MAP estimate (h + 1) / (n + 2)?', answer: ['0.8', '4/5'], explain: '(3 + 1) / (3 + 2) = 0.8.' },
      {"type":"mcq","q":"Why do we usually maximise the log-likelihood rather than the likelihood itself?","options":["The log-likelihood has a different maximum","Products of many small probabilities underflow, while sums of logs are stable and easier to differentiate","Logs make the model more accurate","The likelihood cannot be computed"],"answer":1,"explain":"The log is increasing, so the maximum is in the same place; it turns a product into a sum, which avoids underflow and simplifies the derivatives."},
      {"type":"fill","q":"Support tickets per hour were [1, 3, 2, 4, 0]. What is the maximum-likelihood estimate of the Poisson rate?","answer":["2","2.0"],"explain":"For a Poisson model the MLE of the rate is the average count: 10 / 5 = 2."},
      {"type":"tf","q":"With the same share of heads, the likelihood curve after 100 flips is narrower than after 10 flips.","answer":true,"explain":"More data pins the parameter down: the log-likelihood becomes more sharply curved, and the standard error shrinks like 1/√n."}
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
      { title: 'Bishop, Pattern Recognition and Machine Learning, chapters 1–3 (free PDF)', url: 'https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/' },
      { title: 'StatQuest, Maximum Likelihood, clearly explained (video)', url: 'https://www.youtube.com/watch?v=XepXtl9YKwc' },
      { title: 'Murphy, Probabilistic Machine Learning: An Introduction (free book)', url: 'https://probml.github.io/pml-book/book1.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'stat-inference', branch: 'math', title: 'Statistical Inference: Confidence Intervals, Tests & A/B Testing', short: 'Statistical Inference', level: 2, minutes: 45, prereqs: ['statistics', 'probability'],
    tags: ['confidence interval', 'standard error', 'bootstrap', 'hypothesis test', 'p-value', 'permutation test', 'a/b test', 'statistical power', 'sample size', 'multiple comparisons'],
    summary: 'How to tell a real effect from noise: sampling variability and standard errors, confidence intervals (analytic and bootstrap), hypothesis tests and p-values, A/B tests for comparing models or products, statistical power and sample size, and traps such as peeking and multiple comparisons.',
    why: `"Model B scored 0.83 and model A 0.81: is B better?" Every metric is computed from a finite sample, so every number is noisy. **Statistical inference** tells you how large that noise is and whether a difference is likely to be real. It is essential for comparing models, running A/B tests, evaluating LLMs and reading research results critically, and it protects you from shipping changes that only looked better by luck.`,
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

Example: 0.83 accuracy on 500 examples: SE = √(0.83 · 0.17 / 500) ≈ 0.017, so the interval is about [0.80, 0.86]. A rival model at 0.81 sits comfortably inside it: on this test set alone you cannot claim one is better.

**The bootstrap** gives intervals for *any* statistic (median, F1, AUC, a ratio) without formulas: resample the data **with replacement** many times, recompute the statistic each time, and take the 2.5th and 97.5th percentiles of the results.` },
      { type: 'chart', title: "How wide is a 95% confidence interval for an accuracy of 80%?", after: `With 100 test examples, an accuracy of 0.80 is really "somewhere between 0.72 and 0.88": two models that differ by 3 points cannot be told apart. The width shrinks like 1 / √n, so halving it needs four times as many examples.`, chart: {"kind":"line","x":[50,100,200,500,1000,2000,5000,10000],"xLog":true,"series":[{"name":"full width of the 95% interval","y":[0.2217,0.1568,0.1109,0.0701,0.0496,0.0351,0.0222,0.0157]}],"xLabel":"number of test examples (log scale)","yLabel":"interval width (accuracy points)"} },
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
      { type: 'text', title: "What \"95% confident\" really means", md: `The true accuracy of a model is a fixed number; it is the **interval** that is random, because it is computed from a random test set. "95%" describes the procedure: if you repeated the whole experiment many times, about 95% of the intervals you built would contain the true value. It does *not* mean there is a 95% probability that the truth lies inside the one interval you happened to get. That statement needs a [Bayesian](#/concept/bayesian-methods) credible interval.

Practical consequences:
- **Report intervals, not just point estimates.** "0.83 (95% CI 0.80 to 0.86)" says far more than "0.83".
- **Compare models through their difference.** Overlapping intervals do not prove two models are equal, and requiring non-overlap is a stricter test than you need. Build an interval or a test for the *difference* instead (see below).
- **Mind the independence assumption.** The formulas assume independent examples. If the test set contains near-duplicates, or several rows per user, the effective n is smaller and the real uncertainty larger; bootstrap whole users or documents instead of rows.` },
      { type: 'code', title: "Checking the coverage by simulation", md: `Simulate 1,000 independent test sets for a model whose true accuracy is 0.83, build the usual 95% interval on each, and count how many intervals contain 0.83.`, code: `import numpy as np
rng = np.random.default_rng(0)
true_acc, n, trials = 0.83, 500, 1000

covered = 0
for t in range(trials):
    correct = rng.random(n) < true_acc               # a fresh test set
    acc = correct.mean()
    se = np.sqrt(acc * (1 - acc) / n)
    lo, hi = acc - 1.96 * se, acc + 1.96 * se
    hit = bool(lo <= true_acc <= hi)
    covered += hit
    if t < 5:
        print(f"test set {t + 1}: accuracy {acc:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   contains 0.83: {hit}")

print(f"{covered} of {trials} intervals contain the true accuracy ({covered / trials:.1%})")
print("coverage close to 95%:", abs(covered / trials - 0.95) < 0.02)`, after: `Each test set gives a different interval, and about 1 in 20 of them misses the true value. Nothing about an individual interval tells you whether it is one of the misses.`, predict: {"type":"mcq","q":"Out of 1,000 simulated test sets, roughly how many 95% intervals will contain the true accuracy?","options":["About 500","About 950","All 1,000","About 50"],"answer":1,"explain":"That is what the 95% means: the procedure captures the true value in about 95% of repeated experiments, so about 50 intervals miss it.","expect":["coverage close to 95%: True"]} },
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
      { type: 'text', title: "Comparing two models on the same test set", md: `Model A scores 0.856 and model B 0.873 on the same 1,000 test examples. Treating the two scores as independent samples throws information away: both models saw the **same** examples, and most examples are easy (both right) or hard (both wrong) for both. Only the examples on which the models **disagree** carry evidence about which one is better.

Count the disagreements: b = examples A got right and B got wrong, c = examples A got wrong and B got right. If the models were equally good, each disagreement would be a coin flip, so b and c should be similar. **McNemar's test** checks exactly that:

$$ z = ( c − b ) / √( b + c ) $$
:::where
b — the number of examples model A classified correctly and model B incorrectly
c — the number of examples model B classified correctly and model A incorrectly
b + c — the total number of disagreements; examples both got right or both got wrong are ignored
z — approximately standard normal if the models are equally accurate; |z| > 1.96 means p < 0.05
:::

Example: on 1,000 examples A is right 850 times and B 870 times. If they disagree on 60 examples (b = 20, c = 40), z = 20 / √60 ≈ 2.58 and p ≈ 0.01: B really is better. If they disagree on 400 examples (b = 190, c = 210), the same 2-point gap gives z = 20 / √400 = 1.0, and the difference is noise.

For any other metric (F1, AUC, BLEU) use the **paired bootstrap**: resample test examples with replacement, compute *both* models' metric on each resample, and look at the distribution of the difference. Each resample uses the same examples for both models, so their shared difficulty cancels out.` },
      { type: 'code', title: "McNemar's test and the paired bootstrap", code: `import math
import numpy as np
rng = np.random.default_rng(6)
n = 1000
ease = rng.random(n)                                   # how easy each test example is (the same for both models)
a_right = ease + rng.normal(0, 0.04, n) > 0.15         # model A
b_right = ease + rng.normal(0, 0.04, n) > 0.13         # model B, slightly better
pa, pb = a_right.mean(), b_right.mean()
print(f"accuracy A {pa:.3f}   B {pb:.3f}")

# unpaired: treats the two scores as independent samples, ignoring that the examples are shared
pooled = (pa + pb) / 2
z_unpaired = (pb - pa) / math.sqrt(pooled * (1 - pooled) * 2 / n)
p_unpaired = math.erfc(abs(z_unpaired) / math.sqrt(2))

# paired: McNemar's test on the disagreements only
b = int(np.sum(a_right & ~b_right))                    # A right, B wrong
c = int(np.sum(~a_right & b_right))                    # A wrong, B right
z = (c - b) / math.sqrt(b + c)
p_paired = math.erfc(abs(z) / math.sqrt(2))
print(f"unpaired two-proportion test: p = {p_unpaired:.3f}")
print(f"McNemar: A-only {b}, B-only {c}, z = {z:.2f}, p = {p_paired:.3f}")
print("paired test more sensitive:", p_paired < p_unpaired)

diffs = []
for _ in range(2000):
    idx = rng.integers(0, n, n)                        # the same resampled examples for both models
    diffs.append(b_right[idx].mean() - a_right[idx].mean())
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"paired bootstrap 95% CI for B - A: [{lo:+.3f}, {hi:+.3f}]")`, after: `Judged as two independent samples, the 1.7-point gap looks like noise (p ≈ 0.27). But the models agree on 955 of the 1,000 examples, and B wins 31 of the 45 disagreements: McNemar's test gives p ≈ 0.01, and the paired bootstrap interval for the difference stays above 0. Same data, a sharper question.`, predict: {"type":"mcq","q":"Model B beats model A by about 2 points on the same 1,000 test examples. Which test is more likely to show that the difference is real?","options":["The unpaired two-proportion test","McNemar's paired test","Both give the same p-value","Neither can be used on a shared test set"],"answer":1,"explain":"Both models saw the same examples, so a paired test can ignore the many examples they agree on; only the disagreements count, which removes most of the noise.","expect":["paired test more sensitive: True"]} },
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
      { type: 'chart', title: "Power of an A/B test against sample size", after: `A one-point lift on a 10% conversion rate is a big business win but a small statistical effect: with 2,000 users per group the test would miss it most of the time. Reaching 80% power needs about 15,000 users per group. Run this calculation **before** the experiment, not after.`, chart: {"kind":"line","x":[500,1000,2000,5000,10000,20000,40000],"xLog":true,"series":[{"name":"chance of detecting a lift from 10% to 11%","y":[0.074,0.109,0.177,0.371,0.636,0.904,0.996]}],"xLabel":"users per group (log scale)","yLabel":"power","hlines":[{"y":0.8,"label":"the usual target: 80%"}],"yMin":0,"yMax":1} },
      { type: 'text', title: "Rare events and small samples", md: `The ± 1.96 · SE formula relies on the normal approximation, which breaks down when n is small or the rate is close to 0 or 1. The tell-tale sign is an interval that leaves [0, 1]: 2 errors in 40 examples gives 0.05 ± 0.068, a negative lower bound.

Two tools for these cases:
- **The rule of three.** If an event never happened in n independent trials, a 95% upper bound for its rate is about 3 / n. A model that produced no harmful output in 300 red-team prompts may still do so up to 1% of the time; to claim "below 0.1%" you need about 3,000 clean trials.
- **Better intervals.** The Wilson score interval (in Python, \`proportion_confint(..., method="wilson")\` from statsmodels) stays inside [0, 1] and keeps close to its promised coverage even for small n. The bootstrap also works, unless n is tiny.

$$ upper bound ≈ 3 / n          (0 events in n trials, 95% confidence) $$
:::where
n — the number of independent trials in which the event was never observed
3 / n — an approximate 95% upper bound for the event's true rate: it solves (1 − p)ⁿ = 0.05, because ln(0.05) ≈ −3
:::

Safety and reliability claims ("the model never leaks personal data") are exactly this situation: a clean result on a small test is only weak evidence that the problem is absent.` },
      { type: 'check', question: { type: 'tf', q: 'Stopping an A/B test as soon as the p-value first drops below 0.05 keeps the false-positive rate at 5%.', answer: false, explain: 'Repeated peeking gives many chances to cross the threshold by luck.' } }
    ],
    when: `- Comparing two models on a test set or an LLM evaluation set.
- Product A/B tests: features, ranking changes, prompts.
- Reporting any metric: always with an interval, not a bare number.
- Planning how much data (or how many users) an experiment needs.`,
    pitfalls: `- Reporting point estimates without uncertainty.
- Reading p > 0.05 as "no difference": it may just mean too little data.
- Peeking, multiple comparisons and changing the metric after seeing the results (p-hacking).
- Treating correlated samples (same user, same document) as independent.
- Confusing statistical significance with practical importance.`,
    takeaways: [
      'Every metric is noisy: SE = √(p(1 − p)/n) for a proportion, shrinking with √n.',
      'A 95% CI ≈ estimate ± 1.96·SE; the bootstrap gives intervals for any statistic.',
      'A p-value is the probability of data at least this extreme if there were no effect, not the probability of no effect.',
      'Plan sample sizes with a power calculation; small effects need large samples.',
      'Avoid peeking and multiple comparisons; report effect sizes with intervals.'
    ],
    quiz: [
      { type: 'fill', q: 'A model is 90% accurate on 400 examples. What is the standard error? (3 decimals)', answer: ['0.015'], explain: '√(0.9 · 0.1 / 400) = √0.000225 = 0.015.' },
      { type: 'mcq', q: 'How does the bootstrap build a confidence interval?', options: ['From a formula for the standard error', 'By resampling the data with replacement many times and taking percentiles of the recomputed statistic', 'By training many models', 'By splitting the data into folds'], answer: 1, explain: 'The spread of the resampled statistics estimates its sampling variability.' },
      { type: 'mcq', q: 'You test 20 independent metrics at α = 0.05 and none truly changed. About how many will look significant by chance?', options: ['0', '1', '5', '20'], answer: 1, explain: '20 × 0.05 = 1 expected false positive.' },
      { type: 'tf', q: 'To halve the standard error of an estimate you need about four times as much data.', answer: true, explain: 'SE scales with 1/√n.' },
      { type: 'mcq', q: 'Why use a permutation test?', options: ['It is always more powerful', 'It needs no distributional formula: shuffling labels shows what differences arise by chance', 'It removes confounding', 'It increases the sample size'], answer: 1, explain: 'It builds the null distribution directly from the data.' },
      {"type":"mcq","q":"A 95% confidence interval for an accuracy is [0.80, 0.86]. Which statement is correct?","options":["There is a 95% probability that the true accuracy lies in [0.80, 0.86]","95% of the test examples have accuracy between 0.80 and 0.86","The procedure that built the interval captures the true value in about 95% of repeated experiments","The model is 95% accurate"],"answer":2,"explain":"The truth is fixed and the interval is random. The 95% is a property of the procedure; a probability statement about the truth itself needs a Bayesian credible interval."},
      {"type":"mcq","q":"Two models are evaluated on the same test set. Which examples does McNemar's test use?","options":["All examples","Only the examples both models got right","Only the examples on which the two models disagree","Only the examples both models got wrong"],"answer":2,"explain":"Examples both models get right, or both get wrong, say nothing about which model is better; only the disagreements carry evidence."},
      {"type":"fill","q":"A chatbot gave no unsafe answers in 600 independent test prompts. By the rule of three, what is the approximate 95% upper bound on its unsafe-answer rate? (as a decimal)","answer":["0.005",".005","0.5%","1/200"],"explain":"3 / 600 = 0.005: up to 0.5% of answers could still be unsafe."}
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
        explain: `Resampling with replacement mimics drawing a fresh dataset from the same population. The variability of the statistic across resamples approximates its true sampling variability: no formula needed.`,
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
      { title: 'Kohavi, Tang & Xu, Trustworthy Online Controlled Experiments', url: 'https://experimentguide.com/' },
      { title: 'Evan Miller, How Not To Run an A/B Test (on peeking)', url: 'https://www.evanmiller.org/how-not-to-run-an-ab-test.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'pandas-eda', branch: 'code', title: 'pandas & Exploratory Data Analysis', short: 'pandas & EDA', level: 1, minutes: 45, prereqs: ['python-essentials', 'numpy'],
    tags: ['pandas', 'dataframe', 'series', 'groupby', 'merge', 'pivot table', 'missing values', 'eda', 'visualization', 'matplotlib', 'histogram'],
    summary: 'Working with real tables: pandas Series and DataFrames, selecting and filtering, cleaning types, missing values and duplicates, groupby aggregations, pivot tables and joins, and an exploratory-data-analysis routine with summary statistics and plots (the first step of every ML project).',
    why: `Real data arrives as messy tables (CSV exports, database queries, logs) not as clean NumPy arrays. **pandas** is the standard Python tool for loading, cleaning, reshaping and summarizing tables, and **exploratory data analysis (EDA)** is how you find out what is actually in the data before modelling: missing values, outliers, skewed distributions, leaky columns, and how features relate to the target. Skipping EDA is the most common way to build a confident model on broken data.`,
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
      { type: 'text', title: 'Cleaning data', md: `- **Missing values**: count them with \`df.isna().sum()\`; drop rows (\`dropna\`) or fill (\`fillna\`), with statistics computed on the **training** data only, or you leak information from the test set.
- **Wrong types**: numbers stored as text ("1,200", "n/a") → \`pd.to_numeric(col, errors="coerce")\` turns unparseable values into NaN; dates → \`pd.to_datetime\`.
- **Duplicates**: \`df.drop_duplicates()\`; check whether IDs should be unique.
- **Inconsistent categories**: " Paris", "paris", "PARIS" → \`df["city"].str.strip().str.lower()\`.
- **Impossible values**: negative ages, prices of 0, dates in the future (investigate, then fix or drop).

Keep cleaning in a function (or a scikit-learn pipeline) so exactly the same steps run on new data.` },
      { type: 'text', title: 'Groupby, pivot tables and joins', md: `**Split–apply–combine** with \`groupby\`: split rows into groups, apply an aggregation to each, combine the results.

\`\`\`text
df.groupby("city")["price"].agg(["count", "mean", "median"])     one row per city
df.groupby(["city", "rooms"])["price"].mean()                    one row per (city, rooms) pair
df.pivot_table(index="city", columns="month", values="revenue", aggfunc="sum")
\`\`\`

**Joins** combine tables on a key, like SQL: \`orders.merge(customers, on="customer_id", how="left")\` keeps every order and adds customer columns where a match exists (missing ones become NaN); \`how="inner"\` keeps only matches. Always check the row count after a join: duplicate keys multiply rows.` },
      { type: 'demo', title: "Diagram: split, apply, combine", demo: 'dg-groupby' },
      { type: 'code', title: 'Summaries and joins', code: `
import pandas as pd
sales = pd.DataFrame({"city": ["Paris", "Rome", "Paris", "Berlin", "Rome", "Paris"],
                      "month": ["Jan", "Jan", "Feb", "Feb", "Feb", "Mar"],
                      "revenue": [120, 80, 150, 95, 70, 160]})
cities = pd.DataFrame({"city": ["Paris", "Rome", "Madrid"], "population_m": [2.1, 2.8, 3.3]})

print(sales.head(), "\\n")
print(sales.groupby("city")["revenue"].agg(["count", "sum", "mean"]))
print(sales.pivot_table(index="city", columns="month", values="revenue", aggfunc="sum", fill_value=0))

merged = sales.merge(cities, on="city", how="left")        # Berlin has no population row -> NaN
merged["revenue_per_million"] = (merged["revenue"] / merged["population_m"]).round(1)
print(merged)
print("rows before and after the join:", len(sales), len(merged))
` },
      { type: 'code', title: "Hierarchical indexing", md: `An index can have several **levels**. That lets a Series or DataFrame hold higher-dimensional data in a 2-D shape, and it is what \`groupby\` with several keys returns.`, code: `import numpy as np
import pandas as pd

sales = pd.Series([10, 12, 7, 8, 15, 9],
                  index=[["north", "north", "north", "south", "south", "south"], [2023, 2024, 2025, 2023, 2024, 2025]])
sales.index.names = ["region", "year"]
print(sales, "\\n")
print("one region:", sales["south"].to_dict())
print("one year across regions:", sales.loc[:, 2024].to_dict())

table = sales.unstack()                     # inner level (year) becomes the columns
print("\\nunstacked:")
print(table)
print("\\ntotal per region:", sales.groupby(level="region").sum().to_dict())
print("swapped levels, sorted:")
print(sales.swaplevel().sort_index().head(4))`, after: `- Select with the outer label (\`sales["south"]\`), or with a tuple or a slice per level (\`sales.loc[:, 2024]\`).
- **\`unstack\`** moves the innermost row level into the columns, giving a 2-D table; **\`stack\`** does the reverse. Missing combinations appear as NaN.
- \`swaplevel\` exchanges two levels, and \`sort_index(level=…)\` sorts by them; many operations are faster on a sorted index.
- \`groupby(level="region")\` aggregates over one level, and \`set_index(["a", "b"])\` / \`reset_index()\` move columns into and out of the index.` },
      { type: 'code', title: "Database-style joins with merge", md: `\`merge\` combines two tables on shared **key** columns, like a SQL join. The \`how\` argument decides which keys survive:

| how | keeps |
| --- | --- |
| \`"inner"\` (default) | keys present in **both** tables |
| \`"left"\` | every key of the left table |
| \`"right"\` | every key of the right table |
| \`"outer"\` | keys present in **either** table |`, code: `import pandas as pd

orders = pd.DataFrame({"customer": ["ana", "ben", "ana", "dan", "ben", "ana"], "amount": [30, 12, 55, 8, 40, 21]})
customers = pd.DataFrame({"customer": ["ana", "ben", "cara"], "city": ["Lyon", "Nice", "Paris"]})

print("inner:", len(orders.merge(customers, on="customer")), "rows")
left = orders.merge(customers, on="customer", how="left", indicator=True)
print(left)
print("\\nouter keys:", sorted(orders.merge(customers, on="customer", how="outer")["customer"].unique()))

# many-to-many: every matching pair of rows is combined
a = pd.DataFrame({"key": ["x", "x", "y"], "left_val": [1, 2, 3]})
b = pd.DataFrame({"key": ["x", "x", "x", "y"], "right_val": [10, 20, 30, 40]})
print("\\nmany-to-many result rows:", len(a.merge(b, on="key")), "(2 x 3 for x, plus 1 x 1 for y)")
try:
    orders.merge(a.rename(columns={"key": "customer"}), on="customer", validate="many_to_one")
    print("validation passed")
except Exception as err:
    print("validate caught a problem:", type(err).__name__)`, after: `- \`indicator=True\` adds a \`_merge\` column saying whether each row matched in \`both\` tables or only one: a quick way to find orphan keys ("dan" has no customer record).
- **Many-to-many** joins produce the *Cartesian product* of matching rows, which can silently multiply your data. Pass \`validate="one_to_one"\` or \`"many_to_one"\` to make pandas check the relationship you expect.
- When the key columns have different names, use \`left_on\` and \`right_on\`; overlapping non-key columns get the \`suffixes\` \`_x\` and \`_y\` unless you choose others.` },
      { type: 'code', title: "Concatenating, joining on the index, and patching gaps", code: `import numpy as np
import pandas as pd

jan = pd.DataFrame({"sales": [5, 7]}, index=["ana", "ben"])
feb = pd.DataFrame({"sales": [6, 3, 9]}, index=["ana", "ben", "cara"])
print("stacked rows, labelled by source:")
print(pd.concat([jan, feb], keys=["jan", "feb"]))
print("\\nside by side (columns), aligned on the index:")
print(pd.concat([jan, feb], axis=1, keys=["jan", "feb"]))

profile = pd.DataFrame({"city": ["Lyon", "Nice"]}, index=["ana", "ben"])
print("\\njoin on the index:")
print(feb.join(profile))

primary = pd.Series([1.0, np.nan, 3.0, np.nan], index=list("abcd"))
backup = pd.Series([10.0, 20.0, 30.0, np.nan], index=list("abcd"))
print("\\npatched with a backup source:", primary.combine_first(backup).to_dict())`, after: `- **\`pd.concat\`** glues objects along an axis: rows (\`axis=0\`, the default) or columns (\`axis=1\`). \`keys\` labels where each piece came from, creating a hierarchical index; \`ignore_index=True\` renumbers the rows instead. Labels that do not exist in every piece produce NaN.
- **\`join\`** is a convenient merge on the index.
- **\`combine_first\`** fills the gaps of one object with values from another, aligned by label: useful when two sources cover the same entities with different holes.` },
      { type: 'code', title: "Reshaping: long and wide formats", md: `The same data can be laid out **long** (one row per observation: entity, variable, value) or **wide** (one row per entity, one column per variable). Databases and logs produce long data; models and spreadsheets usually want wide data.`, code: `import pandas as pd

long = pd.DataFrame({
    "date":     ["2026-01", "2026-01", "2026-01", "2026-02", "2026-02", "2026-02"],
    "variable": ["sales", "visits", "returns", "sales", "visits", "returns"],
    "value":    [120, 4000, 6, 135, 4300, 9],
})
wide = long.pivot(index="date", columns="variable", values="value")
print("long -> wide (pivot):")
print(wide)

back = wide.reset_index().melt(id_vars="date", var_name="variable", value_name="value")
print("\\nwide -> long (melt):", len(back), "rows")
print(back.head(3))

stacked = wide.stack()                  # columns -> inner row level
print("\\nwide -> stacked Series:", stacked.shape, " first:", stacked.index[0], "=", stacked.iloc[0])`, after: `| From | To | Tool |
| --- | --- | --- |
| long | wide | \`pivot(index=, columns=, values=)\`, or \`pivot_table\` when you must aggregate duplicates |
| wide | long | \`melt(id_vars=, value_vars=)\` |
| columns | inner row level | \`stack()\` |
| inner row level | columns | \`unstack()\` |

\`pivot\` fails if an (index, column) pair occurs twice; \`pivot_table\` resolves that with an aggregation such as \`aggfunc="mean"\`. Tidy data, one observation per row and one variable per column, is the best starting point for most analyses.` },
      { type: 'check', question: { type: 'mcq', q: 'What does df.groupby("city")["price"].mean() return?', options: ['The overall mean price', 'The average price for each city', 'The cities sorted by price', 'The number of rows per city'], answer: 1, explain: 'One aggregated value per group.' } },
      { type: 'text', title: 'An EDA checklist', md: `1. **Shape and types**: rows, columns, dtypes, a few sample rows.
2. **Missing values** per column, and whether "missing" itself carries information.
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
print(df.head().round(1))

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
    pitfalls: `- Chained assignment (\`df[df.a > 0]["b"] = 1\`) that silently modifies a copy: use \`df.loc[mask, "b"] = 1\`.
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
      { type: 'mcq', q: 'Which plot best shows the distribution of a single numeric column?', options: ['A histogram', 'A pie chart', 'A line plot', 'A scatter plot'], answer: 0, explain: 'It shows the shape: skew, peaks and outliers.' },
      {"type":"mcq","q":"Table A has 3 rows with key \"x\" and table B has 4 rows with key \"x\". How many rows does an inner merge on the key produce?","options":["3","4","7","12"],"answer":3,"explain":"A many-to-many merge combines every matching pair: 3 × 4 = 12 rows. Use `validate=` to catch unexpected many-to-many joins."},
      {"type":"mcq","q":"You need to keep every order, adding customer details where they exist. Which merge?","options":["how=\"inner\"","how=\"left\" with orders on the left","how=\"right\" with orders on the left","pd.concat"],"answer":1,"explain":"A left join keeps every row of the left table (orders) and fills customer columns with NaN where no match exists."},
      {"type":"mcq","q":"Which operation turns a long table (date, variable, value) into a wide table with one column per variable?","options":["melt","pivot","concat","stack"],"answer":1,"explain":"pivot spreads the values of one column into new columns; melt does the reverse."}
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
        explain: `Named aggregations produce tidy, clearly labelled summary tables in one step. The left join keeps every original row: cities missing from the lookup table simply get NaN, which is easy to detect afterwards.`,
        tests: [
          { setup: 'import pandas as pd\ndf = pd.DataFrame({"city": ["a", "b", "a", "c", "b", "a"], "sqm": [50, 100, 70, 40, 80, 60], "price": [300, 400, 350, 100, 480, 310]})', call: 'city_summary(df).round(4).reset_index().to_dict("list")' },
          { name: 'left join keeps every row', setup: 'import pandas as pd\ndf = pd.DataFrame({"city": ["a", "b", "z"], "sqm": [1, 2, 3], "price": [10, 20, 30]})\ncities = pd.DataFrame({"city": ["a", "b"], "population": [5, 7]})', call: 'add_population(df, cities)["population"].tolist()', expect: '[5.0, 7.0, float("nan")]' },
          { name: 'random data vs reference', setup: 'import pandas as pd\nimport numpy as np\nrng = np.random.default_rng(1)\ndf = pd.DataFrame({"city": rng.choice(["x", "y", "z", "w"], 40), "sqm": rng.integers(30, 150, 40), "price": rng.integers(100, 1000, 40)})', call: 'city_summary(df).round(4).reset_index().to_dict("list")', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'pandas, 10 minutes to pandas', url: 'https://pandas.pydata.org/docs/user_guide/10min.html' },
      { title: 'Wes McKinney, Python for Data Analysis (free online edition)', url: 'https://wesmckinney.com/book/' },
      { title: 'matplotlib, Quick start guide', url: 'https://matplotlib.org/stable/users/explain/quick_start.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'gmm-em', branch: 'ml', title: 'Gaussian Mixtures, EM & Other Clustering', short: 'GMMs & EM', level: 3, minutes: 45, prereqs: ['kmeans', 'mle-map'],
    tags: ['gaussian mixture model', 'em algorithm', 'soft clustering', 'responsibilities', 'bic', 'dbscan', 'hierarchical clustering', 'density-based clustering'],
    summary: 'Clustering beyond k-means: Gaussian mixture models give every point a probability of belonging to each cluster and are fitted with the expectation–maximization (EM) algorithm; DBSCAN finds clusters of any shape and flags noise; hierarchical clustering builds a tree of merges.',
    why: `k-means makes hard assignments and assumes round clusters of similar size. Real groups overlap, stretch and differ in density. **Gaussian mixture models** fix this with soft, probabilistic clusters, and they are the textbook example of the **EM algorithm**: a general recipe for maximum likelihood when some variables are hidden. **DBSCAN** and **hierarchical clustering** handle the cases where you don't know how many clusters there are or where clusters have odd shapes.`,
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

Instead of a hard label, each point gets a **responsibility** for each component, the posterior probability that the component generated it:

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

Each iteration never decreases the likelihood, so EM converges, but to a **local** maximum that depends on the starting point. Use several restarts (or initialize from k-means). k-means is the special case of EM with hard assignments and identical round clusters.` },
      { type: 'demo', title: "Step through: EM, round by round", demo: 'walk-em-gmm' },
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
      { type: 'chart', title: "The fitted mixture against the data", after: `The mixture density (the sum of the two dashed bells) follows the histogram closely. EM found weights 0.60 and 0.40, means −2.05 and 3.01, and spreads 0.79 and 1.42, close to the 0.6 / 0.4, −2 / 3 and 0.8 / 1.5 used to generate the data. Where the dashed curves overlap, around x = 0.5, a point gets split responsibilities.`, chart: {"kind":"line","points":false,"series":[{"name":"histogram of the data (as a density)","x":[-5,-4.5,-4.5,-4,-4,-3.5,-3.5,-3,-3,-2.5,-2.5,-2,-2,-1.5,-1.5,-1,-1,-0.5,-0.5,0,0,0.5,0.5,1,1,1.5,1.5,2,2,2.5,2.5,3,3,3.5,3.5,4,4,4.5,4.5,5,5,5.5,5.5,6,6,6.5,6.5,7,7,7.5,7.5,8,8,8.5,8.5,9],"y":[0,0,0.008,0.008,0.036,0.036,0.1,0.1,0.212,0.212,0.292,0.292,0.236,0.236,0.188,0.188,0.116,0.116,0.016,0.016,0.024,0.024,0.04,0.04,0.052,0.052,0.08,0.08,0.104,0.104,0.104,0.104,0.1,0.1,0.084,0.084,0.068,0.068,0.072,0.072,0.036,0.036,0.02,0.02,0.012,0.012,0,0,0,0,0,0,0,0,0,0]},{"name":"fitted mixture p(x)","x":[-5,-4.9,-4.8,-4.7,-4.6,-4.5,-4.4,-4.3,-4.2,-4.1,-4,-3.9,-3.8,-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,-3,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9],"y":[0.0003,0.0005,0.0008,0.0012,0.0017,0.0026,0.0038,0.0054,0.0077,0.0108,0.0148,0.02,0.0266,0.0348,0.0448,0.0569,0.071,0.0873,0.1056,0.1257,0.1473,0.1699,0.1929,0.2156,0.2371,0.2567,0.2735,0.2869,0.2962,0.301,0.3011,0.2965,0.2873,0.2741,0.2575,0.2381,0.2167,0.1943,0.1716,0.1493,0.128,0.1083,0.0905,0.0748,0.0614,0.0502,0.0411,0.034,0.0286,0.0249,0.0225,0.0213,0.0211,0.0218,0.0231,0.025,0.0275,0.0304,0.0337,0.0373,0.0412,0.0453,0.0497,0.0542,0.0589,0.0636,0.0684,0.0733,0.078,0.0827,0.0872,0.0915,0.0955,0.0992,0.1025,0.1055,0.1079,0.1099,0.1114,0.1123,0.1126,0.1124,0.1117,0.1104,0.1085,0.1062,0.1034,0.1001,0.0965,0.0926,0.0884,0.0839,0.0793,0.0745,0.0697,0.0649,0.0601,0.0554,0.0508,0.0464,0.0421,0.038,0.0342,0.0306,0.0272,0.0241,0.0212,0.0186,0.0162,0.0141,0.0122,0.0105,0.009,0.0076,0.0065,0.0054,0.0046,0.0038,0.0032,0.0026,0.0021,0.0017,0.0014,0.0012,0.0009,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.0002,0.0001,0.0001,0.0001,0.0001,0,0,0,0,0]},{"name":"component 1 × π₁","x":[-5,-4.9,-4.8,-4.7,-4.6,-4.5,-4.4,-4.3,-4.2,-4.1,-4,-3.9,-3.8,-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,-3,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9],"y":[0.0003,0.0005,0.0008,0.0012,0.0017,0.0026,0.0038,0.0054,0.0077,0.0108,0.0148,0.02,0.0266,0.0348,0.0448,0.0569,0.071,0.0873,0.1056,0.1257,0.1473,0.1699,0.1929,0.2155,0.2371,0.2566,0.2735,0.2868,0.2961,0.3008,0.3009,0.2962,0.287,0.2737,0.2569,0.2374,0.2158,0.1932,0.1702,0.1476,0.126,0.1058,0.0875,0.0712,0.0571,0.045,0.0349,0.0267,0.0201,0.0148,0.0108,0.0078,0.0055,0.0038,0.0026,0.0017,0.0012,0.0008,0.0005,0.0003,0.0002,0.0001,0.0001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"dash":true},{"name":"component 2 × π₂","x":[-5,-4.9,-4.8,-4.7,-4.6,-4.5,-4.4,-4.3,-4.2,-4.1,-4,-3.9,-3.8,-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,-3,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9],"y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0001,0.0001,0.0001,0.0001,0.0002,0.0002,0.0003,0.0003,0.0004,0.0006,0.0007,0.0009,0.0011,0.0013,0.0017,0.002,0.0025,0.003,0.0036,0.0043,0.0052,0.0062,0.0073,0.0086,0.01,0.0117,0.0136,0.0156,0.018,0.0205,0.0233,0.0263,0.0296,0.0332,0.037,0.041,0.0452,0.0496,0.0542,0.0588,0.0636,0.0684,0.0732,0.078,0.0827,0.0872,0.0915,0.0955,0.0992,0.1025,0.1055,0.1079,0.1099,0.1114,0.1123,0.1126,0.1124,0.1117,0.1104,0.1085,0.1062,0.1034,0.1001,0.0965,0.0926,0.0884,0.0839,0.0793,0.0745,0.0697,0.0649,0.0601,0.0554,0.0508,0.0464,0.0421,0.038,0.0342,0.0306,0.0272,0.0241,0.0212,0.0186,0.0162,0.0141,0.0122,0.0105,0.009,0.0076,0.0065,0.0054,0.0046,0.0038,0.0032,0.0026,0.0021,0.0017,0.0014,0.0012,0.0009,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.0002,0.0001,0.0001,0.0001,0.0001,0,0,0,0,0],"dash":true}],"xLabel":"x","yLabel":"density","yMin":0} },
      { type: 'check', question: { type: 'mcq', q: 'What does the E-step of EM compute for a Gaussian mixture?', options: ['New means and variances', 'The responsibilities: the probability that each component generated each point', 'The number of clusters', 'A random restart'], answer: 1, explain: 'The M-step then re-estimates the parameters from those soft assignments.' } },
      { type: 'text', title: 'How many components? BIC', md: `More components always fit the training data at least as well, so the likelihood alone always prefers more. The **Bayesian information criterion** adds a complexity penalty; choose the K with the lowest value:

$$ BIC = −2 · log L + p · log n $$
:::where
log L — the maximized log-likelihood of the model
p — the number of free parameters (weights, means and covariance entries)
n — the number of data points
:::` },
      { type: 'chart', title: "BIC against the number of components", md: `Data drawn from three Gaussian blobs, fitted with 1 to 6 components:`, after: `The likelihood keeps improving with every component, but after K = 3 the improvement no longer pays for the extra parameters, so the BIC turns upwards and picks the true number of blobs.`, chart: {"kind":"line","x":[1,2,3,4,5,6],"series":[{"name":"BIC (lower is better)","y":[5408,4710.7,4307.7,4328.7,4363.7,4395.3]}],"xLabel":"number of components K","yLabel":"BIC","vlines":[{"x":3,"label":"minimum: K = 3"}]} },
      { type: 'text', title: "Covariance types: how flexible is each cluster?", md: `Each component's covariance matrix decides its shape. scikit-learn's \`GaussianMixture(covariance_type=...)\` offers four choices, trading flexibility against the number of parameters to estimate:

| covariance_type | Cluster shape | Covariance parameters (d features) |
| --- | --- | --- |
| \`"spherical"\` | round, one variance per cluster | 1 per component |
| \`"diag"\` | ellipses aligned with the axes | d per component |
| \`"tied"\` | one tilted ellipse shared by all clusters | d(d + 1)/2 in total |
| \`"full"\` (the default) | any tilted ellipse, different for each cluster | d(d + 1)/2 per component |

With 50 features, one full covariance has 1,275 numbers, which takes a lot of data to estimate; diagonal covariances are the usual fallback in high dimensions. The BIC can compare covariance types as well as numbers of components, because it charges for every parameter.

A fitted mixture is more than a clustering: it is a **density model**. \`score_samples\` gives the log-density of any point, so points in low-density regions can be flagged as anomalies, and \`sample\` generates new synthetic points that look like the data.` },
      { type: 'code', title: "Choosing K with BIC, then flagging unusual points", code: `import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
X, _ = make_blobs(600, centers=[[0, 0], [5, 1], [2, 5]], cluster_std=[0.8, 1.2, 0.6], random_state=1)

bic = {k: GaussianMixture(n_components=k, random_state=0).fit(X).bic(X) for k in range(1, 7)}
for k, v in bic.items():
    print(f"K = {k}: BIC {v:8.1f}")
best = min(bic, key=bic.get)
print("BIC chooses", best, "components:", best == 3)

gmm = GaussianMixture(n_components=best, random_state=0).fit(X)
log_density = gmm.score_samples(X)
threshold = np.percentile(log_density, 1)                    # the least likely 1% of the training data
print(f"anomaly threshold (log-density): {threshold:.2f}")
for point in ([5.0, 1.0], [10.0, 10.0]):
    d = gmm.score_samples([point])[0]
    print(f"point {point}: log-density {d:6.1f} -> {'anomaly' if d < threshold else 'normal'}")
new_points, components = gmm.sample(3)                       # a density model can also generate data
print("three generated points:", new_points.round(2).tolist())`, after: `The BIC drops sharply up to 3 components and then rises slowly: each extra component costs more parameters than it gains in likelihood. A point at the centre of a cluster has a high log-density, while the point at (10, 10) lies far below the threshold set by the least likely 1% of the training data.`, predict: {"type":"mcq","q":"The data comes from 3 Gaussian blobs. Which number of components will the BIC choose?","options":["1","2","3","6"],"answer":2,"explain":"More components always raise the likelihood, but after 3 the gain no longer pays for the extra parameters, so the BIC is lowest at the true number of blobs.","expect":["BIC chooses 3 components: True"]} },
      { type: 'text', title: 'DBSCAN and hierarchical clustering', md: `**DBSCAN** (density-based clustering) has two settings: a radius eps and a minimum number of neighbours. Points with enough neighbours within eps are **core points**; clusters are chains of connected core points plus their neighbours; everything else is **noise**. It finds clusters of arbitrary shape, needs no K, and labels outliers, but struggles when clusters differ in density, and in high dimensions.

**Hierarchical (agglomerative) clustering** starts with every point as its own cluster and repeatedly merges the two closest clusters, producing a tree (a **dendrogram**). Cutting the tree at some height gives the clusters. The **linkage** defines "closest": single (nearest points, follows chains), complete (farthest points), average, or Ward (smallest increase in variance). Great for small data and nested structure, but its memory grows with n².

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
    pitfalls: `- EM converging to a poor local maximum: use multiple restarts.
- Components collapsing onto single points (variance → 0): add a small variance floor or regularization.
- Choosing K by likelihood alone: use BIC or held-out likelihood.
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
      { type: 'mcq', q: 'Which method naturally labels some points as noise?', options: ['k-means', 'Gaussian mixture', 'DBSCAN', 'PCA'], answer: 2, explain: 'Points that are neither core points nor near one are noise.' },
      {"type":"mcq","q":"With d = 10 features, how many free numbers does one full covariance matrix have?","options":["10","20","55","100"],"answer":2,"explain":"A symmetric d × d matrix has d(d + 1)/2 = 10 × 11 / 2 = 55 free entries."},
      {"type":"tf","q":"A fitted Gaussian mixture can generate new data points as well as cluster existing ones.","answer":true,"explain":"It is a generative density model: sample() picks a component by its weight and draws from that Gaussian."}
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
        explain: `The hidden tests check EM's guarantee (the log-likelihood never goes down) and that it recovers the true parameters of well-separated data from a rough start.`,
        tests: [
          { name: 'M-step with hard assignments', call: 'm_step([0.0, 2.0, 10.0, 12.0], [[1, 0], [1, 0], [0, 1], [0, 1]])', expect: '(np.array([0.5, 0.5]), np.array([1.0, 11.0]), np.array([1.0, 1.0]))' },
          { name: 'log-likelihood of one point', call: 'log_likelihood([0.0], np.array([1.0]), np.array([0.0]), np.array([1.0]))', expect: '-np.log(np.sqrt(2 * np.pi))' },
          { name: 'recovers well-separated components', setup: 'rng = np.random.default_rng(0)\nx = np.concatenate([rng.normal(-4, 1.0, 400), rng.normal(5, 2.0, 600)])', code: 'pi, mu, sigma, lls = em(x, [0.5, 0.5], [-1.0, 1.0], [1.0, 1.0], 60)\nassert np.allclose(np.sort(mu), [-4, 5], atol=0.25)\nassert abs(min(pi) - 0.4) < 0.05\nassert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:])), "the log-likelihood must never decrease"', hidden: true },
          { name: 'reference run', setup: 'rng = np.random.default_rng(3)\nx = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 0.5, 30)])', call: 'em(x, [0.5, 0.5], [-0.5, 2.0], [1.0, 1.0], 10)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn, Gaussian mixture models', url: 'https://scikit-learn.org/stable/modules/mixture.html' },
      { title: 'scikit-learn, Clustering (overview and comparison)', url: 'https://scikit-learn.org/stable/modules/clustering.html' },
      { title: 'Ester et al., A Density-Based Algorithm for Discovering Clusters (DBSCAN)', url: 'https://cdn.aaai.org/KDD/1996/KDD96-037.pdf' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
