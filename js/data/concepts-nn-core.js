/* ML Atlas – content: Neural Networks (core) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'perceptron', branch: 'dl', title: 'Neurons & the Perceptron', short: 'The Perceptron', level: 1, minutes: 30, prereqs: ['logistic-regression'],
    tags: ['neuron', 'perceptron', 'xor', 'linear separability', 'history'],
    summary: 'The artificial neuron — a weighted sum, a bias and an activation — the perceptron learning rule, and the XOR problem that showed why we need layers.',
    why: `Every neural network, from a digit recognizer to GPT, is built from one tiny unit: the **artificial neuron**. It takes a weighted sum of its inputs, adds a bias and passes the result through an activation function. The perceptron (1958) was the first learning machine built from it — and its famous failure on XOR explains exactly why networks need hidden layers. Understanding one neuron makes the rest of deep learning a matter of connecting many of them.`,
    objectives: [
      'Compute the output of an artificial neuron and relate it to logistic regression',
      'Apply the perceptron learning rule and explain when it converges',
      'Explain linear separability and why a single neuron cannot learn XOR',
      'Describe how stacking neurons into layers overcomes that limit'
    ],
    sections: [
      { type: 'text', title: 'An artificial neuron', md: `Loosely inspired by biological neurons (which fire when their combined input exceeds a threshold), an artificial neuron computes:

$$ z = w₁x₁ + w₂x₂ + … + w_n x_n + b = w · x + b          output = φ(z) $$
:::where
x₁ … x_n — the n inputs to the neuron (x is the vector of all of them)
w₁ … w_n — the **weights**: how much each input matters (positive = excitatory, negative = inhibitory)
b — the **bias**: shifts the threshold, making the neuron easier or harder to activate
z — the weighted sum, also called the pre-activation
φ — "phi": the **activation function**, which decides the output: a step (perceptron), a sigmoid (≈ logistic regression), a ReLU (modern networks)
:::

A neuron with a sigmoid activation *is* logistic regression. The power of neural networks comes not from a smarter neuron but from **connecting many** of them in layers and training all weights together.

:::warn Brains are not this simple
Biological neurons are far more complex (spike timing, dendritic computation, chemistry). The "neural" in neural networks is inspiration, not a faithful model.
:::` },
      { type: 'demo', title: 'Interactive: a single neuron on logic gates', md: `Adjust the weights and bias until the neuron computes AND, OR or NAND. Then try XOR.`, demo: 'neuron' },
      { type: 'text', title: 'The perceptron learning rule', md: `Rosenblatt's perceptron uses a **step** activation: output 1 if z ≥ 0, else 0. It learns one example at a time, and only from mistakes:

$$ w ← w + η (y − ŷ) x          b ← b + η (y − ŷ) $$
:::where
x — the input of the current training example
y — its true label (0 or 1)
ŷ — the neuron's prediction for it (0 or 1)
y − ŷ — the error: 0 when the prediction is right, +1 or −1 when it is wrong
η — "eta": the learning rate, the size of each correction
← — "becomes": update the value on the left
:::

- If the prediction is right, y − ŷ = 0: nothing changes.
- If it said 0 but the answer is 1, add the input to the weights (making z larger next time).
- If it said 1 but the answer is 0, subtract it.

**Perceptron convergence theorem:** if the classes are **linearly separable** (a straight line/hyperplane can split them perfectly), this rule finds a separating boundary in a finite number of steps. If they are not, it never settles.` },
      { type: 'code', title: 'Training a perceptron on AND, OR and XOR', code: `
import numpy as np

def train_perceptron(X, y, lr=1.0, epochs=25):
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(epochs):
        mistakes = 0
        for xi, yi in zip(X, y):
            pred = 1 if xi @ w + b >= 0 else 0
            if pred != yi:                       # only learn from mistakes
                w += lr * (yi - pred) * xi
                b += lr * (yi - pred)
                mistakes += 1
        if mistakes == 0:
            return w, b, epoch + 1
    return w, b, None

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
for name, y in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]:
    w, b, epochs = train_perceptron(X, np.array(y))
    preds = (X @ w + b >= 0).astype(int).tolist()
    status = f"converged in {epochs} epochs" if epochs else "never converged"
    print(f"{name}: predictions {preds} target {y}  -> {status}")
` },
      { type: 'check', question: { type: 'mcq', q: 'A perceptron predicts 1 for an example whose label is 0. How does the update change the weights?', options: ['Adds the input x (times the learning rate)', 'Subtracts the input x (times the learning rate)', 'Leaves them unchanged', 'Resets them to zero'], answer: 1, explain: 'y − ŷ = 0 − 1 = −1, so w ← w − η·x, lowering z for similar inputs.' } },
      { type: 'text', title: 'XOR and the need for layers', md: `XOR outputs 1 when exactly one input is 1: (0,1) and (1,0) are positive, (0,0) and (1,1) are negative. Plot those four points: no single straight line separates them. A single neuron draws exactly one line, so **no choice of weights works**.

In 1969 Minsky and Papert's book *Perceptrons* highlighted this limitation, and funding for neural network research dried up for years (an "AI winter").

The fix is to **combine neurons**: one hidden neuron computes OR, another computes AND, and an output neuron computes "OR and not AND" — which is XOR. A hidden layer re-represents the input so that the problem *becomes* linearly separable. The missing piece in 1969 was a way to **train** hidden layers — which arrived with backpropagation (popularised in 1986).` },
      { type: 'check', question: { type: 'tf', q: 'A single perceptron can learn any boolean function of two inputs.', answer: false, explain: 'It can only learn linearly separable functions; XOR and XNOR are not.' } }
    ],
    when: `- Think in single neurons to understand what each unit of a network computes.
- The classic perceptron itself is mostly of historical and educational value; in practice use logistic regression (probabilistic, trained by gradient descent) or a multi-layer network.`,
    pitfalls: `- Expecting a single neuron to learn non-linear patterns.
- Confusing the perceptron's step function (no useful gradient) with sigmoid/ReLU neurons trained by gradient descent.
- Taking the brain analogy literally.`,
    takeaways: [
      'An artificial neuron computes φ(w·x + b).',
      'The perceptron updates only on mistakes: w ← w + η(y − ŷ)x.',
      'It converges if and only if the data is linearly separable.',
      'XOR is not linearly separable, so one neuron cannot learn it.',
      'Hidden layers re-represent inputs so that hard problems become separable.'
    ],
    quiz: [
      { type: 'fill', q: 'A neuron has weights [2, −1], bias 0.5 and a step activation (1 if z ≥ 0). What does it output for input [1, 3]?', answer: ['0'], explain: 'z = 2·1 − 1·3 + 0.5 = −0.5 < 0 → output 0.' },
      { type: 'mcq', q: 'A neuron with a sigmoid activation is equivalent to…', options: ['k-means', 'logistic regression', 'a decision tree', 'PCA'], answer: 1, explain: 'σ(w·x + b) is exactly logistic regression\'s prediction.' },
      { type: 'mcq', q: 'Which boolean function can a single perceptron NOT learn?', options: ['AND', 'OR', 'NAND', 'XOR'], answer: 3, explain: 'XOR is not linearly separable.' },
      { type: 'tf', q: 'The perceptron learning rule only changes the weights when the prediction is wrong.', answer: true, explain: 'When ŷ = y, the factor (y − ŷ) is zero.' },
      { type: 'mcq', q: 'How do multi-layer networks solve XOR?', options: ['By using a larger learning rate', 'Hidden units re-represent the inputs so the output unit\'s problem becomes linearly separable', 'By memorising the four cases in the bias', 'They cannot'], answer: 1, explain: 'E.g. hidden units for OR and AND, then output = OR AND NOT AND.' }
    ],
    problems: [
      {
        id: 'neuron-forward', title: 'One Neuron, Three Activations', difficulty: 'easy',
        prompt: `Implement \`neuron(x, w, b, activation)\` returning the neuron's output for a single input vector:

- z = w · x + b
- \`activation\` is one of \`"step"\` (1.0 if z ≥ 0 else 0.0), \`"sigmoid"\` (1 / (1 + e^−z)) or \`"relu"\` (max(0, z)).
- Return a Python float.

\`\`\`text
neuron([1, 3], [2, -1], 0.5, "step")   ->  0.0     (z = -0.5)
neuron([1, 3], [2, -1], 0.5, "relu")   ->  0.0
neuron([2, 1], [1, 1], -1, "relu")     ->  2.0
\`\`\``,
        starter: `
import math

def neuron(x, w, b, activation):
    # TODO: weighted sum plus bias, then the chosen activation
    pass
`,
        solution: `
import math

def neuron(x, w, b, activation):
    z = sum(xi * wi for xi, wi in zip(x, w)) + b
    if activation == "step":
        return 1.0 if z >= 0 else 0.0
    if activation == "sigmoid":
        return 1.0 / (1.0 + math.exp(-z))
    if activation == "relu":
        return float(max(0.0, z))
    raise ValueError("unknown activation: " + str(activation))
`,
        hints: ['z = sum of x_i * w_i, plus b.', 'Branch on the activation name.'],
        explain: `The same weighted sum feeds every activation; only φ changes. The step function was the original perceptron's choice, the sigmoid makes the neuron logistic regression, and ReLU is what most modern hidden layers use.`,
        tests: [
          { call: 'neuron([1, 3], [2, -1], 0.5, "step")', expect: '0.0' },
          { call: 'neuron([2, 1], [1, 1], -1, "relu")', expect: '2.0' },
          { call: 'neuron([0, 0], [5, 5], 0.0, "sigmoid")', expect: '0.5' },
          { name: 'sigmoid value', call: 'neuron([1, 2], [0.5, -0.25], 0.1, "sigmoid")', expect: '1 / (1 + math.exp(-0.1))', hidden: true },
          { name: 'step at exactly zero', call: 'neuron([1, 1], [1, -1], 0.0, "step")', expect: '1.0', hidden: true }
        ]
      },
      {
        id: 'perceptron-train', title: 'Train a Perceptron', difficulty: 'medium',
        prompt: `Implement \`train_perceptron(X, y, lr=1.0, epochs=10)\` → \`(w, b)\` with labels 0/1:

- start with \`w = zeros(d)\`, \`b = 0.0\`;
- for each epoch, go through the examples **in order**; predict \`1 if x·w + b >= 0 else 0\`; update
  \`w += lr * (y_i - pred) * x_i\` and \`b += lr * (y_i - pred)\`;
- run all \`epochs\` (no early exit needed) and return \`w\` (array) and \`b\` (float).

The tests check the exact weights on AND and OR, and that XOR is never fully learned.`,
        starter: `
import numpy as np

def train_perceptron(X, y, lr=1.0, epochs=10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    w, b = np.zeros(X.shape[1]), 0.0
    # TODO
    return w, float(b)
`,
        solution: `
import numpy as np

def train_perceptron(X, y, lr=1.0, epochs=10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            pred = 1 if xi @ w + b >= 0 else 0
            w = w + lr * (yi - pred) * xi
            b = b + lr * (yi - pred)
    return w, float(b)
`,
        hints: ['Two nested loops: epochs, then examples with `zip(X, y)`.', 'The update is zero when the prediction is correct, so you can apply it unconditionally.'],
        explain: `Because (y − ŷ) is 0 for correct predictions, the update formula only "fires" on mistakes. On AND and OR the weights stop changing once every example is classified correctly; on XOR some example is always misclassified, so the weights keep cycling.`,
        tests: [
          { name: 'AND gate', call: 'train_perceptron([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])' },
          { name: 'OR gate', call: 'train_perceptron([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1])' },
          { name: 'learned AND is correct', code: 'X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)\nw, b = train_perceptron(X, [0, 0, 0, 1])\nassert ((X @ w + b >= 0).astype(int) == [0, 0, 0, 1]).all()', hidden: true },
          { name: 'XOR cannot be learned', code: 'X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)\nw, b = train_perceptron(X, [0, 1, 1, 0], epochs=50)\nassert not ((X @ w + b >= 0).astype(int) == [0, 1, 1, 0]).all(), "a single perceptron should not solve XOR"', hidden: true },
          { name: 'separable random data', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(30, 2))\ny = (X[:, 0] + 2 * X[:, 1] > 0).astype(int)', call: 'train_perceptron(X, y, lr=0.5, epochs=20)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: '3Blue1Brown — But what is a neural network?', url: 'https://www.3blue1brown.com/lessons/neural-networks' }
    ]
  },

  // =====================================================================================
  {
    id: 'activation-functions', branch: 'dl', title: 'Activation Functions', level: 1, minutes: 35, prereqs: ['perceptron', 'derivatives'],
    tags: ['relu', 'sigmoid', 'tanh', 'gelu', 'softmax', 'vanishing gradient'],
    summary: 'Why networks need non-linearities, how sigmoid, tanh, ReLU and GELU differ, the vanishing-gradient problem, and the numerically stable softmax.',
    why: `Without activation functions, a 100-layer network collapses into a single linear model — all that depth for nothing. The choice of activation also decides whether gradients can flow through deep networks: the switch from sigmoids to **ReLU** was one of the key reasons deep learning took off around 2012, and transformers use smooth relatives like **GELU** and **SwiGLU**. At the output, **softmax** turns scores into the probabilities every classifier and LLM produces.`,
    objectives: [
      'Prove that stacking linear layers without activations is still linear',
      'Compare sigmoid, tanh, ReLU, Leaky ReLU and GELU: shapes, ranges and gradients',
      'Explain the vanishing-gradient problem and why ReLU helps',
      'Implement a numerically stable softmax and pair output activations with losses'
    ],
    sections: [
      { type: 'text', title: 'Why non-linearity is essential', md: `Two linear layers in a row compute (XW₁)W₂ = X(W₁W₂) — which is just **one** linear layer with weight matrix W₁W₂. However many linear layers you stack, the result is a linear function, and a linear function can't learn XOR, curved boundaries or anything interesting.

Inserting a non-linear function φ between layers breaks this collapse: φ(XW₁)W₂ cannot be rewritten as a single matrix product. With non-linearities, networks become **universal approximators** — given enough hidden units they can approximate any continuous function.` },
      { type: 'code', title: 'Linear layers collapse; non-linear ones don\'t', code: `
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))
W1, W2 = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))

two_linear_layers = (X @ W1) @ W2
one_linear_layer = X @ (W1 @ W2)            # a single 3x2 matrix does the same job
print("two linear layers == one linear layer:", np.allclose(two_linear_layers, one_linear_layer))

with_relu = np.maximum(0, X @ W1) @ W2      # a non-linearity in between breaks the collapse
print("with ReLU in between, still equal?   ", np.allclose(with_relu, one_linear_layer))
` },
      { type: 'demo', title: 'Interactive: activation functions and their gradients', md: `Toggle functions and show their derivatives. Watch where each derivative becomes (nearly) zero.`, demo: 'activations' },
      { type: 'text', title: 'The main activation functions', md: `| Function | Formula | Range | Derivative | Used in |
| --- | --- | --- | --- | --- |
| **Sigmoid** | 1/(1+e⁻ᶻ) | (0, 1) | σ(1−σ), max 0.25 | binary output layers, gates (LSTM) |
| **Tanh** | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (−1, 1) | 1 − tanh², max 1 | RNNs; zero-centred |
| **ReLU** | max(0, z) | [0, ∞) | 1 if z > 0 else 0 | default for hidden layers (CNNs, MLPs) |
| **Leaky ReLU** | z if z > 0 else αz | (−∞, ∞) | 1 or α | fixes "dying ReLU" |
| **GELU** | z·Φ(z) (Φ = Gaussian CDF) | ≈ (−0.17, ∞) | smooth | BERT, GPT |
| **SiLU / Swish** | z·σ(z) | ≈ (−0.28, ∞) | smooth | LLaMA's SwiGLU, EfficientNet |

**ReLU's advantages**: trivially cheap; gradient exactly 1 for positive inputs (no shrinking); produces sparse activations. **Its weakness**: a neuron whose input is always negative outputs 0 and gets gradient 0 forever — a "dead" ReLU (Leaky ReLU, GELU and careful initialization help).` },
      { type: 'text', title: 'Vanishing gradients', md: `Backpropagation multiplies the activation's derivative at **every layer**. The sigmoid's derivative is at most 0.25, and nearly 0 when |z| is large (the function is "saturated", flat). Through 10 sigmoid layers the gradient can shrink by a factor of 0.25¹⁰ ≈ 10⁻⁶ — the early layers effectively stop learning. This **vanishing gradient problem** made deep networks hard to train before ~2010.

Remedies that made deep learning work:
- **ReLU-family activations** (derivative 1 on the positive side);
- **good initialization** (He/Xavier) and **normalization layers** (BatchNorm, LayerNorm);
- **residual connections** (the gradient has a shortcut path — key to ResNets and transformers);
- **gated architectures** (LSTMs) for recurrent nets.

The opposite, **exploding gradients**, happens when the factors are larger than 1; gradient clipping is the usual fix.` },
      { type: 'check', question: { type: 'mcq', q: 'Why did ReLU help train deeper networks than sigmoid?', options: ['It is bounded between 0 and 1', 'Its derivative is 1 for positive inputs, so gradients don\'t shrink layer after layer', 'It is smoother', 'It outputs probabilities'], answer: 1, explain: 'Sigmoid derivatives (≤ 0.25) multiply into vanishing gradients; ReLU passes gradients through unchanged where active.' } },
      { type: 'text', title: 'Softmax for outputs', md: `For K-class classification the output layer produces K scores (**logits**) z₁…z_K, and **softmax** turns them into probabilities:

$$ softmax(z)_k = e^(z_k) / Σⱼ e^(z_j) $$
:::where
z — the vector of K logits (raw scores), one per class
z_k — the logit of class k
e^(z_k) — e (≈ 2.718) raised to that logit: always positive, and larger logits become exponentially larger
Σⱼ e^(z_j) — the sum over all K classes, which makes the outputs add up to 1
softmax(z)_k — the predicted probability of class k
:::

**Numerical stability:** e^1000 overflows to infinity. Since softmax is unchanged by subtracting the same constant from every logit, compute e^(z_k − max(z)) instead: the largest exponent becomes e⁰ = 1, nothing overflows.

**Pairing output activations with losses:**

| Task | Output activation | Loss |
| --- | --- | --- |
| Regression | none (linear) | MSE / MAE / Huber |
| Binary classification | sigmoid | binary cross-entropy |
| Multi-class (one label) | softmax | categorical cross-entropy |
| Multi-label | sigmoid per label | binary cross-entropy per label |

In PyTorch you normally feed raw logits to \`CrossEntropyLoss\` / \`BCEWithLogitsLoss\`, which apply softmax/sigmoid internally in a numerically stable way.` },
      { type: 'code', title: 'Stable softmax', code: `
import numpy as np

def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()

def softmax_stable(z):
    e = np.exp(z - np.max(z))           # subtracting the max changes nothing mathematically
    return e / e.sum()

z = np.array([2.0, 1.0, 0.1])
print("small logits:", softmax_naive(z).round(4), softmax_stable(z).round(4))

big = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("naive, big logits: ", softmax_naive(big))       # inf / inf = nan
print("stable, big logits:", softmax_stable(big).round(4))
` },
      { type: 'check', question: { type: 'tf', q: 'Adding the same constant to every logit does not change the softmax output.', answer: true, explain: 'e^(z+c) = e^c·e^z; the e^c cancels between numerator and denominator — that is why subtracting the max is safe.' } }
    ],
    when: `- **Hidden layers**: ReLU by default for MLPs and CNNs; GELU/SiLU (or SwiGLU) in transformers; tanh inside classic RNNs.
- **Outputs**: linear for regression, sigmoid for binary/multi-label, softmax for multi-class.
- Avoid sigmoid/tanh in deep stacks of hidden layers (vanishing gradients) unless the architecture needs them (gates).`,
    pitfalls: `- Forgetting the activation between layers (the network becomes linear).
- Applying softmax *and* then using a loss that applies it again (e.g. PyTorch's CrossEntropyLoss expects raw logits).
- Naive softmax/sigmoid overflow — use stable formulas.
- Dead ReLUs from large learning rates or bad initialization.`,
    takeaways: [
      'Without non-linear activations, any stack of layers is a single linear map.',
      'ReLU is the default hidden activation; GELU/SiLU in transformers; sigmoid/softmax at outputs.',
      'Small derivatives multiplied across layers cause vanishing gradients.',
      'Stable softmax subtracts the max logit before exponentiating.',
      'Match the output activation to the loss: linear+MSE, sigmoid+BCE, softmax+CE.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the maximum value of the sigmoid\'s derivative?', options: ['1', '0.5', '0.25', 'unbounded'], answer: 2, explain: 'σ(1 − σ) peaks at σ = 0.5, giving 0.25.' },
      { type: 'fill', q: 'What is ReLU(−3.2)?', answer: ['0', '0.0'], explain: 'max(0, −3.2) = 0.' },
      { type: 'mcq', q: 'Which activation is typical in the feed-forward layers of GPT-style transformers?', options: ['Step function', 'GELU (or SwiGLU variants)', 'Sigmoid', 'Softmax'], answer: 1, explain: 'GPT-2 used GELU; LLaMA-style models use SwiGLU, built on SiLU.' },
      { type: 'tf', q: 'A network with 50 layers but no activation functions can represent the XOR function.', answer: false, explain: 'It is equivalent to a single linear layer, which cannot represent XOR.' },
      { type: 'mcq', q: 'For multi-label classification (an image can have several tags), which output activation fits?', options: ['Softmax over all tags', 'An independent sigmoid per tag', 'ReLU', 'No activation'], answer: 1, explain: 'Softmax forces probabilities to compete and sum to 1; independent sigmoids allow several tags at once.' }
    ],
    problems: [
      {
        id: 'relu-leaky', title: 'ReLU and Leaky ReLU', difficulty: 'easy',
        prompt: `Implement two element-wise activations for NumPy arrays (return new arrays of the same shape):

- \`relu(x)\` = max(0, x)
- \`leaky_relu(x, alpha=0.01)\` = x where x > 0, otherwise alpha·x

\`\`\`text
relu(np.array([-2.0, 0.0, 3.0]))        ->  [0., 0., 3.]
leaky_relu(np.array([-2.0, 0.0, 3.0]))  ->  [-0.02, 0., 3.]
\`\`\``,
        starter: `
import numpy as np

def relu(x):
    # TODO
    pass

def leaky_relu(x, alpha=0.01):
    # TODO
    pass
`,
        solution: `
import numpy as np

def relu(x):
    return np.maximum(0, np.asarray(x, dtype=float))

def leaky_relu(x, alpha=0.01):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)
`,
        hints: ['`np.maximum(0, x)` is element-wise (not `np.max`, which reduces).', '`np.where(condition, a, b)` picks element-wise.'],
        explain: `\`np.maximum\` compares element by element with 0; \`np.max\` would instead return the single largest element — a classic slip. Leaky ReLU keeps a small slope for negative inputs so the neuron still receives gradient.`,
        tests: [
          { call: 'relu(np.array([-2.0, 0.0, 3.0]))', expect: 'np.array([0.0, 0.0, 3.0])' },
          { call: 'leaky_relu(np.array([-2.0, 0.0, 3.0]))', expect: 'np.array([-0.02, 0.0, 3.0])' },
          { name: 'custom alpha, 2-D input', call: 'leaky_relu(np.array([[-1.0, 2.0], [4.0, -5.0]]), alpha=0.2)', expect: 'np.array([[-0.2, 2.0], [4.0, -1.0]])', hidden: true },
          { name: 'random matrix', setup: 'rng = np.random.default_rng(1)\nZ = rng.normal(size=(5, 4))', call: '(relu(Z), leaky_relu(Z, 0.1))', hidden: true }
        ]
      },
      {
        id: 'softmax-stable', title: 'Numerically Stable Softmax', difficulty: 'medium',
        prompt: `Implement \`softmax(z, axis=-1)\` for NumPy arrays of any shape, normalising along \`axis\`:

- subtract the max along that axis (with \`keepdims=True\`) before exponentiating;
- divide by the sum along the same axis.

The hidden tests use logits like 1000 with NumPy set to **raise** on overflow, and a 2-D batch where every row must sum to 1.

\`\`\`text
softmax(np.array([1.0, 2.0, 3.0]))  ->  [0.0900, 0.2447, 0.6652]
\`\`\``,
        starter: `
import numpy as np

def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)
`,
        hints: ['`np.max(z, axis=axis, keepdims=True)` keeps a size-1 dimension so it broadcasts back.', 'Do the same `keepdims=True` for the sum.'],
        explain: `\`keepdims=True\` is what makes this work for any shape and axis: the max and the sum keep a size-1 dimension that broadcasts against z. Every LLM computes exactly this over its vocabulary at every step (with temperature applied to z first).`,
        tests: [
          { call: 'softmax(np.array([1.0, 2.0, 3.0]))', expect: 'np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum()' },
          { name: 'rows of a batch sum to 1', setup: 'rng = np.random.default_rng(0)\nZ = rng.normal(size=(4, 6))', code: 'P = softmax(Z)\nassert P.shape == (4, 6)\nassert np.allclose(P.sum(axis=1), 1)\nassert np.allclose(P, np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True))' },
          { name: 'huge logits do not overflow', code: 'with np.errstate(over="raise", invalid="raise"):\n    p = softmax(np.array([1000.0, 999.0, 998.0]))\nassert np.allclose(p, softmax(np.array([2.0, 1.0, 0.0])))', hidden: true },
          { name: 'softmax along axis 0', setup: 'rng = np.random.default_rng(3)\nZ = rng.normal(size=(3, 5))', call: 'softmax(Z, axis=0)', hidden: true }
        ]
      },
      {
        id: 'activation-derivatives', title: 'Derivatives of Activations', difficulty: 'medium',
        prompt: `Backpropagation needs each activation's derivative. Implement, element-wise for NumPy arrays:

- \`sigmoid_grad(z)\` = σ(z)·(1 − σ(z))
- \`tanh_grad(z)\` = 1 − tanh(z)²
- \`relu_grad(z)\` = 1.0 where z > 0, else 0.0 (use 0 at exactly z = 0)

\`\`\`text
sigmoid_grad(np.array([0.0]))  ->  [0.25]
relu_grad(np.array([-1.0, 0.0, 2.0]))  ->  [0., 0., 1.]
\`\`\``,
        starter: `
import numpy as np

def sigmoid_grad(z):
    # TODO
    pass

def tanh_grad(z):
    # TODO
    pass

def relu_grad(z):
    # TODO
    pass
`,
        solution: `
import numpy as np

def sigmoid_grad(z):
    s = 1 / (1 + np.exp(-np.asarray(z, dtype=float)))
    return s * (1 - s)

def tanh_grad(z):
    t = np.tanh(np.asarray(z, dtype=float))
    return 1 - t ** 2

def relu_grad(z):
    return (np.asarray(z, dtype=float) > 0).astype(float)
`,
        hints: ['Compute σ(z) once and reuse it.', 'For ReLU, convert the boolean mask `z > 0` to floats.'],
        explain: `These derivatives are cheap because they reuse the forward value: σ′ = σ(1 − σ), tanh′ = 1 − tanh². The hidden test compares them with numerical derivatives — the same gradient-checking idea you'll use for whole networks.`,
        tests: [
          { call: 'sigmoid_grad(np.array([0.0]))', expect: 'np.array([0.25])' },
          { call: 'relu_grad(np.array([-1.0, 0.0, 2.0]))', expect: 'np.array([0.0, 0.0, 1.0])' },
          { call: 'tanh_grad(np.array([0.0, 1.0]))', expect: 'np.array([1.0, 1 - np.tanh(1.0) ** 2])' },
          { name: 'matches numerical derivatives', setup: 'z = np.linspace(-4, 4, 17) + 0.05\nh = 1e-6', code: 'num_s = (1 / (1 + np.exp(-(z + h))) - 1 / (1 + np.exp(-(z - h)))) / (2 * h)\nnum_t = (np.tanh(z + h) - np.tanh(z - h)) / (2 * h)\nassert np.allclose(sigmoid_grad(z), num_s, atol=1e-7)\nassert np.allclose(tanh_grad(z), num_t, atol=1e-7)\nassert np.allclose(relu_grad(z), (z > 0).astype(float))', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Hendrycks & Gimpel — Gaussian Error Linear Units (GELU)', url: 'https://arxiv.org/abs/1606.08415' }
    ]
  },

  // =====================================================================================
  {
    id: 'mlp', branch: 'dl', title: 'Multi-Layer Networks & the Forward Pass', short: 'Multi-Layer Perceptrons', level: 1, minutes: 40, prereqs: ['activation-functions', 'numpy'],
    tags: ['mlp', 'dense layer', 'hidden layer', 'forward pass', 'universal approximation', 'parameters'],
    summary: 'Stacking layers of neurons: dense layers as matrix multiplications, tracking shapes through a network, hidden representations, depth vs width, and counting parameters.',
    why: `The multi-layer perceptron (MLP) is the basic neural network — and it is everywhere: as a complete model for tabular data, as the classification head on top of CNNs, and as the **feed-forward block inside every transformer layer** (where most of an LLM's parameters live). Being able to write its forward pass in a few lines of NumPy, and to reason about shapes and parameter counts, is a core deep-learning skill.`,
    objectives: [
      'Write a dense (fully connected) layer as act(XW + b) and track shapes through a network',
      'Explain what hidden layers learn and why depth helps',
      'Count the parameters of an MLP',
      'Choose output layers and losses for regression and classification'
    ],
    sections: [
      { type: 'text', title: 'Layers of neurons', md: `A **dense** (fully connected) layer contains many neurons, each connected to every input. For a batch of examples X (shape n × d_in), a layer with d_out neurons computes

$$ H = φ(X W + b)          X: (n, d_in)   W: (d_in, d_out)   b: (d_out,)   H: (n, d_out) $$
:::where
X — the input batch: n examples, each with d_in features
W — the weight matrix; column j holds the weights of neuron j
b — the bias vector, one bias per neuron, added to every row
φ — the activation function, applied element by element (for example ReLU)
H — the layer's output: d_out activations for each of the n examples
n — the batch size
:::

Column j of W holds neuron j's weights; the bias broadcasts across the batch. An **MLP** stacks such layers:

$$ input → [dense + ReLU] → [dense + ReLU] → [dense] → output $$
:::where
→ — data flows left to right; each layer's output is the next layer's input
[dense + ReLU] — a dense layer XW + b followed by the ReLU activation max(0, z)
[dense] — the last layer has no activation here: it outputs raw scores (logits) or regression values
:::

The layers between input and output are **hidden layers**; their outputs are **hidden representations** (activations). The whole computation from input to prediction is the **forward pass**.` },
      { type: 'code', title: 'A forward pass with shapes', code: `
import numpy as np
rng = np.random.default_rng(0)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

batch, d_in, d_hidden, n_classes = 4, 3, 5, 2
X = rng.normal(size=(batch, d_in))
W1, b1 = rng.normal(0, 0.5, (d_in, d_hidden)), np.zeros(d_hidden)
W2, b2 = rng.normal(0, 0.5, (d_hidden, n_classes)), np.zeros(n_classes)

H = relu(X @ W1 + b1)          # (4, 3) @ (3, 5) -> (4, 5)
logits = H @ W2 + b2            # (4, 5) @ (5, 2) -> (4, 2)
probs = softmax(logits)
print("X", X.shape, "-> H", H.shape, "-> logits", logits.shape)
print("probabilities (each row sums to 1):")
print(probs.round(3))
print("number of parameters:", W1.size + b1.size + W2.size + b2.size)
` },
      { type: 'check', question: { type: 'fill', q: 'A dense layer maps 784 inputs to 128 outputs. How many parameters does it have (weights + biases)?', answer: ['100480', '100,480'], explain: '784·128 weights + 128 biases = 100,352 + 128 = 100,480.' } },
      { type: 'text', title: 'What hidden layers learn', md: `Each hidden layer **re-represents** its input. In the XOR example, two hidden neurons turn the inputs into "OR" and "AND" features, after which the output neuron's job is a simple linear one.

In deep networks trained on images, early layers learn edges and colour blobs, middle layers combine them into textures and parts, and later layers respond to whole objects. In language models, layers progressively build representations of syntax, meaning and context. **The network learns its own features** — that's the big difference from classical ML, where you engineer features by hand.

**Universal approximation theorem:** an MLP with one hidden layer and enough neurons can approximate any continuous function on a bounded domain arbitrarily well. It says nothing about how many neurons that takes or whether training will find them — in practice **depth** (several layers) represents complex functions with far fewer parameters than one enormous layer, because each layer reuses the features built by the previous one.` },
      { type: 'code', title: 'Solving XOR with a hidden layer (hand-set weights)', code: `
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

# hidden neuron 1 computes OR, hidden neuron 2 computes AND (step activations)
W1 = np.array([[1.0, 1.0],
               [1.0, 1.0]])
b1 = np.array([-0.5, -1.5])
H = (X @ W1 + b1 > 0).astype(float)

# output: OR and not AND  ==  XOR
W2 = np.array([1.0, -2.0]); b2 = -0.5
print("hidden features (OR, AND):", H.tolist())
print("XOR:", (H @ W2 + b2 > 0).astype(int).tolist())
`, after: `Training finds weights like these automatically — that is what backpropagation is for (next lesson).` },
      { type: 'text', title: 'Outputs, losses and sizes', md: `The last layer matches the task:
- **Regression**: one linear output per target, MSE loss.
- **Binary classification**: one output + sigmoid (or two + softmax), binary cross-entropy.
- **K classes**: K outputs + softmax, cross-entropy.

**Sizing an MLP** (rules of thumb): start with 1–3 hidden layers of 64–512 units; widths often stay constant or shrink towards the output; more data supports bigger networks; use validation curves to decide.

**Parameter counting** — for layer sizes [d₀, d₁, …, d_L]:

$$ params = Σ_l ( d_{l−1} · d_l + d_l ) $$
:::where
d₀ — the number of input features
d_l — the width (number of neurons) of layer l; d_L is the output size
d_{l−1} · d_l — the weights of layer l, one per (input, neuron) pair
+ d_l — the biases of layer l, one per neuron
Σ_l — add up over the layers l = 1 … L
:::

A transformer's feed-forward block is exactly an MLP: d_model → 4·d_model → d_model. For GPT-2 small (d_model = 768) that is about 4.7 million parameters per layer — roughly two thirds of each layer's weights.` },
      { type: 'check', question: { type: 'mcq', q: 'What is the main role of hidden layers?', options: ['To store the training data', 'To learn intermediate representations (features) that make the final prediction easier', 'To speed up training', 'To normalize the inputs'], answer: 1, explain: 'Hidden layers transform the input into features the next layer can use — learned feature engineering.' } }
    ],
    when: `- **Tabular data**: an MLP is a reasonable alternative to boosted trees, especially with lots of data or when combined with embeddings for categorical features (trees often still win on small tables).
- **As a component**: classification heads, transformer feed-forward blocks, projection layers.
- For images, audio and sequences, use architectures with the right inductive bias (CNNs, transformers), which contain MLP pieces.`,
    pitfalls: `- Shape mismatches: write down (n, d_in) @ (d_in, d_out) at every layer.
- Forgetting the activation between layers.
- Applying softmax in the model *and* in the loss.
- Making the network huge for a small dataset (overfitting) — use validation curves and regularization.`,
    takeaways: [
      'A dense layer computes φ(XW + b); an MLP stacks several of them.',
      'Shapes: (n, d_in) @ (d_in, d_out) + (d_out,) → (n, d_out).',
      'Hidden layers learn their own features; depth reuses features efficiently.',
      'Parameters per layer = d_in·d_out + d_out.',
      'Transformer feed-forward blocks are MLPs.'
    ],
    quiz: [
      { type: 'mcq', q: 'X has shape (32, 20) and a layer has W with shape (20, 64). What is the shape of the layer\'s output?', options: ['(20, 64)', '(32, 64)', '(64, 32)', '(32, 20)'], answer: 1, explain: '(32, 20) @ (20, 64) → (32, 64).' },
      { type: 'fill', q: 'How many parameters does an MLP with layer sizes [10, 20, 3] have?', answer: ['283'], explain: '(10·20 + 20) + (20·3 + 3) = 220 + 63 = 283.' },
      { type: 'tf', q: 'The universal approximation theorem guarantees that gradient descent will find a good network.', answer: false, explain: 'It only says such a network exists; it says nothing about finding it or how large it must be.' },
      { type: 'mcq', q: 'For a 10-class image classifier, the final layer usually has…', options: ['1 output with ReLU', '10 outputs followed by softmax', '10 outputs with ReLU', '2 outputs with sigmoid'], answer: 1, explain: 'One logit per class, softmax to probabilities, cross-entropy loss.' },
      { type: 'mcq', q: 'Inside a transformer layer, the feed-forward block is…', options: ['a convolution', 'an MLP applied to each token\'s vector', 'a decision tree', 'an RNN'], answer: 1, explain: 'Typically d_model → 4·d_model → d_model with a GELU/SwiGLU non-linearity.' }
    ],
    problems: [
      {
        id: 'dense-forward', title: 'A Dense Layer', difficulty: 'easy',
        prompt: `Implement \`dense(X, W, b, activation="relu")\` returning φ(X @ W + b) for a batch:

- \`activation\` is \`"relu"\`, \`"sigmoid"\` or \`"linear"\` (no activation).
- X: (n, d_in), W: (d_in, d_out), b: (d_out,) → output (n, d_out).

Also implement \`count_params(sizes)\` for an MLP with layer sizes \`[d0, d1, ..., dL]\`: the total number of weights and biases.

\`\`\`text
count_params([784, 128, 10])  ->  101770
\`\`\``,
        starter: `
import numpy as np

def dense(X, W, b, activation="relu"):
    # TODO
    pass

def count_params(sizes):
    # TODO
    pass
`,
        solution: `
import numpy as np

def dense(X, W, b, activation="relu"):
    z = np.asarray(X, dtype=float) @ np.asarray(W, dtype=float) + np.asarray(b, dtype=float)
    if activation == "relu":
        return np.maximum(0, z)
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    if activation == "linear":
        return z
    raise ValueError("unknown activation " + str(activation))

def count_params(sizes):
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))
`,
        hints: ['`X @ W + b` — the bias broadcasts over the rows.', 'For parameters, pair consecutive sizes with `zip(sizes[:-1], sizes[1:])`.'],
        explain: `A dense layer is one matrix multiply, one broadcast add and one element-wise function — the entire computation of a fully connected neuron layer, for the whole batch at once.`,
        tests: [
          { call: 'dense([[1.0, 2.0]], [[1.0, -1.0], [0.5, 0.5]], [0.0, 1.0])', expect: 'np.array([[2.0, 1.0]])' },
          { call: 'dense([[1.0, 2.0]], [[1.0, -1.0], [0.5, 0.5]], [0.0, -2.0], "linear")', expect: 'np.array([[2.0, -2.0]])' },
          { call: 'count_params([784, 128, 10])', expect: '101770' },
          { name: 'sigmoid layer on a random batch', setup: 'rng = np.random.default_rng(2)\nX = rng.normal(size=(5, 3)); W = rng.normal(size=(3, 4)); b = rng.normal(size=4)', call: 'dense(X, W, b, "sigmoid")', hidden: true },
          { name: 'deeper MLP', call: 'count_params([10, 20, 20, 3])', expect: '703', hidden: true }
        ]
      },
      {
        id: 'mlp-forward', title: 'Forward Pass of an MLP Classifier', difficulty: 'medium',
        prompt: `Implement \`mlp_forward(X, layers)\` for a classifier:

- \`layers\` is a list of \`(W, b)\` pairs.
- Apply **ReLU** after every layer **except the last**.
- Apply a **numerically stable softmax** (row-wise) after the last layer.
- Return the (n, n_classes) array of probabilities.

\`\`\`text
layers = [(np.eye(2), np.zeros(2)), (np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2))]
mlp_forward(np.array([[1.0, -1.0]]), layers)   ->  softmax([1, 0]) = [[0.731, 0.269]]
\`\`\``,
        starter: `
import numpy as np

def mlp_forward(X, layers):
    h = np.asarray(X, dtype=float)
    # TODO: hidden layers with ReLU, final layer + softmax
    pass
`,
        solution: `
import numpy as np

def mlp_forward(X, layers):
    h = np.asarray(X, dtype=float)
    for W, b in layers[:-1]:
        h = np.maximum(0, h @ W + b)
    W, b = layers[-1]
    z = h @ W + b
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
`,
        hints: ['Loop over `layers[:-1]` with ReLU, then handle `layers[-1]` separately.', 'Subtract the row-wise max before `np.exp`.'],
        explain: `The loop is the whole network: each iteration is one hidden layer. Treating the last layer differently (no ReLU, softmax instead) matches the output to the classification task. Frameworks usually stop at the logits and let the loss apply softmax.`,
        tests: [
          { call: 'mlp_forward(np.array([[1.0, -1.0]]), [(np.eye(2), np.zeros(2)), (np.eye(2), np.zeros(2))])', expect: 'np.array([[np.exp(1) / (np.exp(1) + 1), 1 / (np.exp(1) + 1)]])' },
          { name: 'rows sum to one', setup: 'rng = np.random.default_rng(0)\nlayers = [(rng.normal(size=(4, 8)), rng.normal(size=8)), (rng.normal(size=(8, 8)), rng.normal(size=8)), (rng.normal(size=(8, 3)), rng.normal(size=3))]\nX = rng.normal(size=(6, 4))', code: 'P = mlp_forward(X, layers)\nassert P.shape == (6, 3)\nassert np.allclose(P.sum(axis=1), 1)' },
          { name: 'random network vs reference', setup: 'rng = np.random.default_rng(5)\nlayers = [(rng.normal(size=(3, 6)), rng.normal(size=6)), (rng.normal(size=(6, 4)), rng.normal(size=4))]\nX = rng.normal(size=(5, 3))', call: 'mlp_forward(X, layers)', hidden: true },
          { name: 'large logits stay finite', setup: 'layers = [(np.array([[500.0, 0.0]]), np.zeros(2))]', code: 'with np.errstate(over="raise"):\n    P = mlp_forward(np.array([[3.0]]), layers)\nassert np.allclose(P, [[1.0, 0.0]])', hidden: true }
        ]
      }
    ],
    refs: [
      { title: '3Blue1Brown — Neural networks (series)', url: 'https://www.3blue1brown.com/topics/neural-networks' },
      { title: 'Dive into Deep Learning — Multilayer Perceptrons', url: 'https://d2l.ai/chapter_multilayer-perceptrons/index.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'backpropagation', branch: 'dl', title: 'Backpropagation', level: 2, minutes: 50, prereqs: ['mlp', 'derivatives', 'loss-functions'],
    tags: ['backprop', 'chain rule', 'computational graph', 'gradients', 'gradient checking'],
    summary: 'How a network computes the gradient of the loss for every weight in one backward sweep: computational graphs, local gradients, the chain rule in matrix form, and gradient checking.',
    why: `Backpropagation is the algorithm that makes deep learning possible. A modern network has millions to trillions of parameters; computing each gradient separately (as with numerical derivatives) would be hopeless. Backprop computes **all** of them with roughly the cost of one extra forward pass, by applying the chain rule backwards through the network and reusing intermediate results. Every framework — PyTorch, TensorFlow, JAX — automates it, but understanding it lets you debug training, design layers, and reason about vanishing gradients.`,
    objectives: [
      'Represent a computation as a graph and propagate gradients backwards with the chain rule',
      'Derive the backward pass of a dense layer, ReLU, and softmax + cross-entropy',
      'Implement forward and backward passes for a 2-layer network in NumPy',
      'Verify gradients numerically (gradient checking)'
    ],
    sections: [
      { type: 'text', title: 'Computational graphs', md: `Any network computation is a graph of simple operations. For one example with a single hidden layer:

\`\`\`text
x ─► z1 = x·W1 + b1 ─► h1 = relu(z1) ─► z2 = h1·W2 + b2 ─► p = softmax(z2) ─► L = −log p[y]
\`\`\`

**Forward pass:** compute every node from left to right, and **remember** the intermediate values (z1, h1, p…).

**Backward pass:** start from ∂L/∂L = 1 and move right to left. At each node, the chain rule says:

$$ (gradient w.r.t. a node's input) = (gradient w.r.t. its output) × (local derivative) $$
:::where
gradient w.r.t. x — ∂L/∂x: how much the final loss L changes when x changes a little ("w.r.t." means "with respect to")
gradient w.r.t. its output — the "upstream" gradient, passed back from the node after it
local derivative — how this node's own output changes with its input (e.g. 2x for x², 1 for an addition)
:::

Each operation only needs to know its **local** derivative — how its output changes with its inputs — and receives the "upstream" gradient from the node after it. When a value feeds into several places, the gradients from all paths **add up**.` },
      { type: 'text', title: 'The backward rules you need', md: `For a batch, with δ denoting "gradient of the loss with respect to this quantity" (same shape as the quantity):

| Forward | Backward |
| --- | --- |
| Z = X W + b | dW = Xᵀ δZ, db = Σ_rows δZ, dX = δZ Wᵀ |
| H = relu(Z) | δZ = δH ⊙ (Z > 0) |
| H = σ(Z) | δZ = δH ⊙ H ⊙ (1 − H) |
| P = softmax(Z), L = mean cross-entropy | δZ = (P − Y_onehot) / n |

:::where Symbols in this table
X — a dense layer's input batch, shape (n, d_in)
W, b — the layer's weights (d_in, d_out) and bias (d_out,)
Z — the pre-activations X W + b
H — the activations after the nonlinearity (relu or σ)
δZ, δH — the gradients of the loss with respect to Z and H; each has the same shape as the thing it belongs to
dW, db, dX — the gradients with respect to W, b and X
Σ_rows — a sum over the batch dimension (axis 0)
⊙ — element-wise multiplication (NumPy's * operator)
(Z > 0) — a 0/1 mask that is 1 wherever Z was positive
P — the softmax probabilities, one row per example
Y_onehot — the true labels as one-hot rows (1 in the correct class's column)
n — the batch size
:::

The softmax + cross-entropy result is famously simple: the gradient of the logits is just **predicted probabilities minus the one-hot truth** (divided by the batch size for a mean loss).

**Shape rule of thumb:** a gradient always has the same shape as the thing it is the gradient of. If dW must be (d_in, d_out) and you have X (n, d_in) and δZ (n, d_out), the only product that fits is Xᵀ δZ. When in doubt, let the shapes tell you the formula.` },
      { type: 'code', title: 'Manual backprop for a 2-layer network, with a gradient check', code: `
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(6, 3)); y = np.array([0, 1, 2, 1, 0, 2])
P = {"W1": rng.normal(0, 0.5, (3, 4)), "b1": np.zeros(4), "W2": rng.normal(0, 0.5, (4, 3)), "b2": np.zeros(3)}

def loss_and_grads(P):
    z1 = X @ P["W1"] + P["b1"]; h1 = np.maximum(0, z1)          # forward (keep what backward needs)
    z2 = h1 @ P["W2"] + P["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)
    p = np.exp(z2) / np.exp(z2).sum(axis=1, keepdims=True)
    n = len(X)
    loss = -np.mean(np.log(p[np.arange(n), y]))
    d_z2 = p.copy(); d_z2[np.arange(n), y] -= 1; d_z2 /= n       # softmax + cross-entropy
    g = {"W2": h1.T @ d_z2, "b2": d_z2.sum(axis=0)}
    d_h1 = d_z2 @ P["W2"].T
    d_z1 = d_h1 * (z1 > 0)                                        # ReLU lets gradient through where z1 > 0
    g["W1"] = X.T @ d_z1; g["b1"] = d_z1.sum(axis=0)
    return loss, g

loss, g = loss_and_grads(P)
print("loss:", round(float(loss), 4))
for name in P:                                                     # numerical check of every gradient entry
    num = np.zeros_like(P[name])
    for i in np.ndindex(P[name].shape):
        old = P[name][i]
        P[name][i] = old + 1e-5; lp = loss_and_grads(P)[0]
        P[name][i] = old - 1e-5; lm = loss_and_grads(P)[0]
        P[name][i] = old; num[i] = (lp - lm) / 2e-5
    rel = np.max(np.abs(num - g[name]) / np.maximum(1e-8, np.abs(num) + np.abs(g[name])))
    print(f"{name}: shape {g[name].shape}  max relative error {rel:.1e}")
` },
      { type: 'check', question: { type: 'mcq', q: 'For a dense layer Z = XW + b with X (n, d_in) and upstream gradient δZ (n, d_out), what is dW?', options: ['δZ Xᵀ', 'Xᵀ δZ', 'X δZ', 'W δZ'], answer: 1, explain: 'dW must have shape (d_in, d_out): only Xᵀ (d_in, n) @ δZ (n, d_out) fits.' } },
      { type: 'text', title: 'Why it is efficient — and where it goes wrong', md: `**Efficiency.** Backprop visits each operation once going forward and once going backward, so computing the gradient of *all* parameters costs only a small constant times one forward pass — independent of the number of parameters. Numerical differentiation would need two forward passes **per parameter**. This is called **reverse-mode automatic differentiation**.

**Memory.** The forward pass must store intermediate activations for the backward pass, so training uses much more memory than inference. Tricks like *gradient checkpointing* recompute some activations to save memory.

**Vanishing and exploding gradients.** The backward pass multiplies by each layer's local derivatives and weight matrices. Many factors below 1 shrink the gradient towards zero (early layers stop learning); factors above 1 blow it up. Good activations, initialization, normalization, residual connections and gradient clipping keep it in range.

**Gradient checking.** When you write backward passes by hand, compare them with central-difference numerical gradients on a tiny network; relative errors around 1e-7 or below mean you got it right.` },
      { type: 'check', question: { type: 'tf', q: 'With softmax + cross-entropy (mean over a batch of n), the gradient of the logits is (P − Y_onehot)/n.', answer: true, explain: 'The softmax Jacobian and the log combine into this simple form.' } }
    ],
    when: `- You use backprop every time you train a neural network — automatically, through autograd.
- Write backward passes by hand only to learn, or for custom operations; always gradient-check them.
- Think about the backward pass when diagnosing vanishing/exploding gradients or memory use.`,
    pitfalls: `- Shape errors — derive formulas from shapes.
- Forgetting to accumulate (+=) gradients when a value is used in several places.
- Forgetting the 1/n of a mean loss (gradients off by the batch size).
- Mutating cached forward values before the backward pass uses them.
- Not gradient-checking hand-written gradients.`,
    takeaways: [
      'Backprop = chain rule applied from the loss backwards through the computational graph.',
      'Each operation needs only its local derivative; gradients from multiple paths add.',
      'Dense layer: dW = Xᵀ δZ, db = Σ δZ, dX = δZ Wᵀ. ReLU: mask by Z > 0.',
      'Softmax + cross-entropy: δZ = (P − Y)/n.',
      'All gradients for about the cost of one extra pass; verify hand-written ones numerically.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the forward pass need to store for backpropagation?', options: ['Nothing', 'Intermediate values (activations) used by the local derivatives', 'Only the final loss', 'The gradients'], answer: 1, explain: 'E.g. the ReLU backward needs z1 > 0; the dense backward needs the layer input X.' },
      { type: 'mcq', q: 'A value feeds into two different later operations. Its gradient is…', options: ['the gradient from the first path only', 'the product of the two path gradients', 'the sum of the gradients from both paths', 'zero'], answer: 2, explain: 'The multivariable chain rule sums contributions from every path.' },
      { type: 'fill', q: 'In the ReLU backward pass, the upstream gradient is multiplied by 1 where z > 0 and by what value elsewhere?', answer: ['0', '0.0', 'zero'], explain: 'ReLU\'s derivative is 0 for z < 0, blocking the gradient.' },
      { type: 'mcq', q: 'Why is backprop vastly faster than computing each gradient numerically?', options: ['It uses a GPU', 'It reuses intermediate results to get every gradient in one backward pass instead of two forward passes per parameter', 'It skips half the parameters', 'It approximates the gradients'], answer: 1, explain: 'Reverse-mode differentiation shares work across all parameters.' },
      { type: 'tf', q: 'A relative error of about 1e-2 between analytic and numerical gradients means the backward pass is correct.', answer: false, explain: 'Correct implementations typically reach 1e-7 or better in double precision; 1e-2 indicates a bug.' }
    ],
    problems: [
      {
        id: 'backprop-2layer', title: 'Backprop Through a 2-Layer Network', difficulty: 'hard',
        prompt: `Implement \`forward_backward(X, y, params)\` for a 2-layer classifier:

\`\`\`text
z1 = X @ W1 + b1          h1 = relu(z1)
z2 = h1 @ W2 + b2         p  = softmax(z2)   (row-wise, numerically stable)
loss = mean over examples of  -log p[i, y[i]]
\`\`\`

- \`params\` is a dict with keys \`"W1"\` (d, h), \`"b1"\` (h,), \`"W2"\` (h, k), \`"b2"\` (k,); \`y\` holds integer class labels.
- Return \`(loss, grads)\` where \`grads\` is a dict with the **same keys** and the gradients of the loss.

Hints on the table in the lesson: δz2 = (p − onehot)/n, then the dense and ReLU backward rules. The hidden tests compare with **numerical gradients**, so you can't pass without correct calculus.`,
        starter: `
import numpy as np

def forward_backward(X, y, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    n = len(X)
    # forward
    # TODO
    # backward
    # TODO
    return loss, {"W1": None, "b1": None, "W2": None, "b2": None}
`,
        solution: `
import numpy as np

def forward_backward(X, y, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    z1 = X @ W1 + b1
    h1 = np.maximum(0, z1)
    z2 = h1 @ W2 + b2
    z2 = z2 - z2.max(axis=1, keepdims=True)
    p = np.exp(z2) / np.exp(z2).sum(axis=1, keepdims=True)
    loss = float(-np.mean(np.log(p[np.arange(n), y])))
    d_z2 = p.copy()
    d_z2[np.arange(n), y] -= 1
    d_z2 /= n
    dW2 = h1.T @ d_z2
    db2 = d_z2.sum(axis=0)
    d_h1 = d_z2 @ W2.T
    d_z1 = d_h1 * (z1 > 0)
    dW1 = X.T @ d_z1
    db1 = d_z1.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
`,
        hints: ['`p[np.arange(n), y]` picks each example\'s probability for its true class.', 'Start backward with `d_z2 = p.copy(); d_z2[np.arange(n), y] -= 1; d_z2 /= n`.', 'Dense backward: `dW = input.T @ d_out`, `db = d_out.sum(axis=0)`, `d_input = d_out @ W.T`; ReLU backward: multiply by `(z1 > 0)`.'],
        explain: `Two dense layers, one ReLU and the softmax-cross-entropy pair — each step backward applies one rule from the table. The same pattern, repeated, is backprop for any MLP; autograd frameworks simply record the forward operations and replay these local rules in reverse.`,
        tests: [
          { name: 'loss and gradients on a fixed example', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(5, 3))\ny = np.array([0, 1, 2, 1, 0])\nparams = {"W1": rng.normal(0, 0.5, (3, 4)), "b1": rng.normal(0, 0.1, 4), "W2": rng.normal(0, 0.5, (4, 3)), "b2": rng.normal(0, 0.1, 3)}', call: 'forward_backward(X, y, params)' },
          { name: 'gradient shapes', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(7, 2))\ny = rng.integers(0, 4, 7)\nparams = {"W1": rng.normal(size=(2, 5)), "b1": np.zeros(5), "W2": rng.normal(size=(5, 4)), "b2": np.zeros(4)}', code: 'loss, g = forward_backward(X, y, params)\nfor k in params:\n    assert g[k] is not None and np.shape(g[k]) == params[k].shape, "grads[%s] has shape %s, expected %s" % (k, np.shape(g[k]), params[k].shape)' },
          { name: 'matches numerical gradients', setup: 'rng = np.random.default_rng(3)\nX = rng.normal(size=(4, 3))\ny = np.array([2, 0, 1, 2])\nparams = {"W1": rng.normal(0, 0.7, (3, 5)), "b1": rng.normal(0, 0.2, 5), "W2": rng.normal(0, 0.7, (5, 3)), "b2": rng.normal(0, 0.2, 3)}', code: 'loss, g = forward_backward(X, y, params)\nfor k in params:\n    num = np.zeros_like(params[k])\n    for i in np.ndindex(params[k].shape):\n        old = params[k][i]\n        params[k][i] = old + 1e-5; lp = forward_backward(X, y, params)[0]\n        params[k][i] = old - 1e-5; lm = forward_backward(X, y, params)[0]\n        params[k][i] = old\n        num[i] = (lp - lm) / 2e-5\n    rel = np.max(np.abs(num - g[k]) / np.maximum(1e-8, np.abs(num) + np.abs(g[k])))\n    assert rel < 1e-5, "gradient for %s is off (max relative error %.2e)" % (k, rel)', hidden: true }
        ]
      },
      {
        id: 'gradient-check', title: 'Gradient Checking', difficulty: 'medium',
        prompt: `Write the tool you use to test backward passes. Implement \`grad_check(f, x, analytic, h=1e-5)\`:

- \`f(x)\` returns a scalar; \`x\` is a float NumPy array (any shape); \`analytic\` is your claimed gradient (same shape).
- Compute the numerical gradient with central differences, perturbing **one entry at a time** (use \`np.ndindex(x.shape)\`) and restoring it afterwards.
- Return the **maximum relative error** over all entries: |a − n| / max(1e-8, |a| + |n|).

\`\`\`text
grad_check(lambda v: np.sum(v ** 2), np.array([1.0, 2.0]), np.array([2.0, 4.0]))  ->  ~1e-11 (tiny)
grad_check(lambda v: np.sum(v ** 2), np.array([1.0, 2.0]), np.array([2.0, 5.0]))  ->  ~0.111
\`\`\``,
        starter: `
import numpy as np

def grad_check(f, x, analytic, h=1e-5):
    x = np.asarray(x, dtype=float)
    analytic = np.asarray(analytic, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def grad_check(f, x, analytic, h=1e-5):
    x = np.asarray(x, dtype=float)
    analytic = np.asarray(analytic, dtype=float)
    num = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + h
        fp = f(x)
        x[i] = old - h
        fm = f(x)
        x[i] = old
        num[i] = (fp - fm) / (2 * h)
    rel = np.abs(analytic - num) / np.maximum(1e-8, np.abs(analytic) + np.abs(num))
    return float(np.max(rel))
`,
        hints: ['`for i in np.ndindex(x.shape):` loops over every index of an array of any shape.', 'Restore `x[i]` after the two evaluations.', 'Relative error divides by the sum of magnitudes, guarded by 1e-8.'],
        explain: `Relative (not absolute) error matters because gradients can be tiny or huge; dividing by their magnitudes makes "1e-7" mean "agrees to 7 digits" everywhere. This little function catches the vast majority of backprop bugs.`,
        tests: [
          { name: 'correct gradient gives a tiny error', code: 'e = grad_check(lambda v: np.sum(v ** 2), np.array([1.0, 2.0]), np.array([2.0, 4.0]))\nassert e < 1e-7, e' },
          { name: 'wrong gradient is detected', call: 'grad_check(lambda v: np.sum(v ** 2), np.array([1.0, 2.0]), np.array([2.0, 5.0]))', expect: '1 / 9', tol: 1e-5 },
          { name: '2-D input', setup: 'rng = np.random.default_rng(0)\nA = rng.normal(size=(3, 3))\nx0 = rng.normal(size=(3, 3))', code: 'e = grad_check(lambda M: np.sum(np.tanh(M * A)), x0.copy(), A * (1 - np.tanh(x0 * A) ** 2))\nassert e < 1e-6, e', hidden: true },
          { name: 'x is restored', code: 'x = np.array([0.3, -0.7])\ngrad_check(lambda v: np.sum(np.sin(v)), x, np.cos(x))\nassert np.allclose(x, [0.3, -0.7])', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'CS231n — Backpropagation, intuitions', url: 'https://cs231n.github.io/optimization-2/' },
      { title: '3Blue1Brown — Backpropagation calculus', url: 'https://www.3blue1brown.com/lessons/backpropagation-calculus' }
    ]
  },

  // =====================================================================================
  {
    id: 'autograd', branch: 'dl', title: 'Build a Tiny Autograd Engine', short: 'Autograd from Scratch', level: 3, minutes: 50, prereqs: ['backpropagation'],
    tags: ['autograd', 'micrograd', 'computational graph', 'topological sort', 'operator overloading'],
    summary: 'Recreate the heart of PyTorch in ~50 lines: Value objects that record how they were computed, local backward functions, and a topological sort that runs backprop automatically.',
    why: `When you call \`loss.backward()\` in PyTorch, an **autograd** engine walks backwards through every operation you performed and fills in \`.grad\` for every parameter. It feels like magic until you build one yourself — and then it is just bookkeeping plus the chain rule. This exercise (inspired by Andrej Karpathy's *micrograd*) removes the last bit of mystery from training neural networks and makes framework errors ("trying to backward through the graph a second time", leaf tensors, accumulated gradients) suddenly understandable.`,
    objectives: [
      'Represent scalars as Value objects that remember their parents and the operation that created them',
      'Write the local backward function for addition, multiplication, power, tanh and ReLU',
      'Run backpropagation via a topological ordering of the graph',
      'Explain gradient accumulation and why frameworks require zero_grad()'
    ],
    sections: [
      { type: 'text', title: 'The idea', md: `Wrap every number in an object that records **how it was made**:

- \`data\` — the value itself;
- \`grad\` — ∂(final output)/∂(this value), filled in by backward (starts at 0);
- \`_prev\` — the Value objects it was computed from;
- \`_backward\` — a small function that pushes this node's gradient to its parents using the **local derivative**.

Python's operator overloading (\`__add__\`, \`__mul__\`, …) lets us build the graph just by writing ordinary expressions: \`c = a * b + a\` creates new Value nodes that point back to \`a\` and \`b\`.` },
      { type: 'text', title: 'Local backward rules', md: `For out = a + b: ∂out/∂a = 1 and ∂out/∂b = 1, so

\`\`\`text
a.grad += out.grad
b.grad += out.grad
\`\`\`

For out = a · b: ∂out/∂a = b and ∂out/∂b = a:

\`\`\`text
a.grad += b.data * out.grad
b.grad += a.data * out.grad
\`\`\`

For out = aᵏ: a.grad += k·aᵏ⁻¹·out.grad. For out = tanh(a): a.grad += (1 − out²)·out.grad. For out = relu(a): a.grad += (out > 0)·out.grad.

Why **+=**? A value used in several places receives gradient from every path (the multivariable chain rule sums them). It also explains why PyTorch **accumulates** gradients across backward calls — and why every training step starts with \`optimizer.zero_grad()\`.` },
      { type: 'text', title: 'Running backward in the right order', md: `A node may only push its gradient to its parents once its own gradient is complete — i.e. after **every** node that uses it has been processed. A **topological sort** (children after parents) reversed gives exactly that order:

\`\`\`text
def backward(self):
    order, seen = [], set()
    def build(v):
        if v not in seen:
            seen.add(v)
            for parent in v._prev:
                build(parent)
            order.append(v)        # appended after all its inputs
    build(self)
    self.grad = 1.0                # d(output)/d(output)
    for v in reversed(order):
        v._backward()
\`\`\`

That's all of autograd. PyTorch does the same thing with tensors instead of scalars, hundreds of operation types, and C++/CUDA kernels.` },
      { type: 'code', title: 'A minimal engine in action', code: `
import math

class Value:
    def __init__(self, data, _prev=()):
        self.data, self.grad = float(data), 0.0
        self._prev, self._backward = _prev, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev:
                    build(p)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

# a single neuron: out = tanh(w1*x1 + w2*x2 + b)
x1, x2, w1, w2, b = Value(2.0), Value(0.0), Value(-3.0), Value(1.0), Value(6.8814)
out = (x1 * w1 + x2 * w2 + b).tanh()
out.backward()
print("output:", round(out.data, 4))
print("grads  w1:", round(w1.grad, 4), " w2:", round(w2.grad, 4), " b:", round(b.grad, 4), " x1:", round(x1.grad, 4))
` },
      { type: 'check', question: { type: 'mcq', q: 'x = Value(3); y = x * x. After y.backward(), what is x.grad?', options: ['3', '6', '9', '1'], answer: 1, explain: 'x is used twice. Each use contributes x.data = 3 to the other operand\'s gradient: 3 + 3 = 6 = d(x²)/dx.' } },
      { type: 'text', title: 'From scalars to neural networks', md: `With Value objects you can build a neuron (\`sum(wi*xi) + b\` then \`.tanh()\`), a layer (a list of neurons), and an MLP (a list of layers). A training step is then:

\`\`\`text
loss = sum((model(x) - y) ** 2 for x, y in data)
for p in model.parameters(): p.grad = 0.0      # zero_grad
loss.backward()
for p in model.parameters(): p.data -= lr * p.grad
\`\`\`

The only thing PyTorch adds conceptually is **tensors**: each node holds a whole array, so one node's backward handles millions of numbers in one optimized kernel. Same algorithm, enormously faster.` }
    ],
    when: `- Build this once to understand frameworks deeply; then use PyTorch/JAX autograd for real work.
- Knowing the mechanics helps when you implement custom layers, debug "None" gradients, or reason about memory (the graph keeps references to intermediate values).`,
    pitfalls: `- Using \`=\` instead of \`+=\` in backward (breaks values used multiple times).
- Calling backward twice without zeroing gradients (they accumulate).
- Running _backward in the wrong order (a node pushing its gradient before it is complete).
- Forgetting to wrap plain numbers into Value objects in operators.`,
    takeaways: [
      'Each Value stores data, grad, its parents and a local backward function.',
      'Operator overloading builds the computational graph automatically.',
      'Local rules: add passes the gradient through; mul swaps in the other operand; tanh uses 1 − t².',
      'Backward = topological sort, then call _backward in reverse order, starting with grad = 1.',
      'Gradients accumulate (+=), which is why training loops zero them every step.'
    ],
    quiz: [
      { type: 'mcq', q: 'In an autograd engine, what does each node\'s _backward function do?', options: ['Recomputes the forward value', 'Pushes the node\'s gradient to its inputs using local derivatives', 'Updates the weights', 'Deletes the graph'], answer: 1, explain: 'It applies the chain rule locally: parent.grad += local_derivative × out.grad.' },
      { type: 'mcq', q: 'Why does backward process nodes in reverse topological order?', options: ['It is faster in Python', 'So each node\'s gradient is complete before it is pushed further back', 'To save memory', 'Any order works'], answer: 1, explain: 'All users of a value must contribute their gradients first.' },
      { type: 'fill', q: 'a = Value(2), b = Value(5), c = a * b. After c.backward(), what is a.grad?', answer: ['5', '5.0'], explain: '∂(ab)/∂a = b = 5.' },
      { type: 'tf', q: 'PyTorch accumulates gradients into .grad across backward calls unless you reset them.', answer: true, explain: 'That is why optimizer.zero_grad() is called each iteration.' },
      { type: 'mcq', q: 'out = tanh(a) with out.data = 0.5. If out.grad = 2, what does a.grad receive?', options: ['2', '1.5', '0.75', '0.5'], answer: 1, explain: '(1 − 0.5²)·2 = 0.75·2 = 1.5.' }
    ],
    problems: [
      {
        id: 'value-autograd', title: 'Your Own Autograd Value', difficulty: 'hard',
        prompt: `Complete the \`Value\` class so that expressions build a graph and \`backward()\` fills in every \`.grad\`.

Implement:
- \`__add__\`, \`__mul__\` (other may be a plain number — wrap it),
- \`__pow__(k)\` for a plain int/float exponent k,
- \`relu()\` and \`tanh()\`,
- \`backward()\` — topological sort, set this node's grad to 1.0, call every \`_backward\` in reverse order.

Already provided (built from your methods): \`__neg__\`, \`__sub__\`, \`__radd__\`, \`__rmul__\`, \`__rsub__\`, \`__truediv__\`.

\`\`\`text
a, b = Value(2.0), Value(-3.0)
c = a * b + a ** 2        # -6 + 4 = -2
c.backward()
a.grad -> 1.0 (= b + 2a),  b.grad -> 2.0 (= a)
\`\`\``,
        starter: `
import math

class Value:
    def __init__(self, data, _prev=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None

    def __repr__(self):
        return "Value(data=%g, grad=%g)" % (self.data, self.grad)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # TODO: create out, define _backward, attach it, return out
        pass

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # TODO
        pass

    def __pow__(self, k):
        # TODO (k is a plain number)
        pass

    def relu(self):
        # TODO
        pass

    def tanh(self):
        # TODO
        pass

    def backward(self):
        # TODO: topological order, self.grad = 1.0, call _backward in reverse
        pass

    # provided helpers (they use your methods above)
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)
`,
        solution: `
import math

class Value:
    def __init__(self, data, _prev=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None

    def __repr__(self):
        return "Value(data=%g, grad=%g)" % (self.data, self.grad)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,))
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def build(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._prev:
                    build(p)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)
`,
        hints: ['Each operation: compute `out`, define an inner `_backward()` that does `parent.grad += local * out.grad`, set `out._backward = _backward`, return out.', 'Use `+=` in every backward function — values can be used more than once.', 'In `backward`, build the order with a recursive DFS that appends a node after visiting its parents; then iterate `reversed(order)`.'],
        explain: `Closures make this elegant: each \`_backward\` remembers its own \`self\`, \`other\` and \`out\`. Tracking visited nodes by \`id(v)\` avoids relying on hashing. With these few operations you can already express and train a small MLP — exactly what micrograd demonstrates.`,
        tests: [
          { name: 'forward values', setup: 'def run():\n    return (Value(2.0) * Value(3.0) + Value(1.0)).data', call: 'run()', expect: '7.0' },
          { name: 'gradients of c = a*b + a**2', setup: 'def run():\n    a, b = Value(2.0), Value(-3.0)\n    c = a * b + a ** 2\n    c.backward()\n    return (c.data, a.grad, b.grad)', call: 'run()', expect: '(-2.0, 1.0, 2.0)' },
          { name: 'gradient accumulation (x used twice)', setup: 'def run():\n    x = Value(3.0)\n    y = x * x + x\n    y.backward()\n    return x.grad', call: 'run()', expect: '7.0' },
          { name: 'neuron with tanh', setup: 'def run():\n    x1, x2, w1, w2, b = Value(2.0), Value(0.0), Value(-3.0), Value(1.0), Value(6.8814)\n    out = (x1 * w1 + x2 * w2 + b).tanh()\n    out.backward()\n    return (round(out.data, 6), round(w1.grad, 6), round(w2.grad, 6), round(b.grad, 6), round(x1.grad, 6))', call: 'run()', hidden: true },
          { name: 'relu, division, subtraction vs numerical derivative', setup: 'def f(v):\n    a = Value(v)\n    return ((a * 2 - 1).relu() / (a ** 2 + 1) + 3 - a).data\ndef run():\n    a = Value(0.8)\n    out = (a * 2 - 1).relu() / (a ** 2 + 1) + 3 - a\n    out.backward()\n    num = (f(0.8 + 1e-6) - f(0.8 - 1e-6)) / 2e-6\n    return abs(a.grad - num) < 1e-6', call: 'run()', expect: 'True', hidden: true },
          { name: 'shared sub-expression', setup: 'def run():\n    a = Value(1.5)\n    b = a * a\n    c = b * b + b\n    c.backward()\n    return a.grad', call: 'run()', expect: '4 * 1.5 ** 3 + 2 * 1.5', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Andrej Karpathy — micrograd', url: 'https://github.com/karpathy/micrograd', note: 'the inspiration for this lesson' },
      { title: 'Karpathy — Neural Networks: Zero to Hero (video course)', url: 'https://karpathy.ai/zero-to-hero.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'training-loop', branch: 'dl', title: 'Training Neural Networks: The Loop', short: 'The Training Loop', level: 2, minutes: 45, prereqs: ['backpropagation', 'gradient-descent'],
    tags: ['training loop', 'epochs', 'batches', 'early stopping', 'debugging', 'learning curves'],
    summary: 'Putting it together: the canonical training loop, reading loss curves, sanity checks, early stopping and a practical debugging checklist.',
    why: `Every neural network — from a toy classifier to a frontier LLM — is trained by the same loop: forward, loss, backward, update, repeat. What separates a working model from a frustrating week is knowing how to **watch** that loop: reading training and validation curves, catching NaNs and divergence early, verifying the pipeline on a tiny batch, and stopping at the right time. These habits transfer directly to PyTorch and to fine-tuning large models.`,
    objectives: [
      'Write the canonical mini-batch training loop',
      'Interpret training/validation loss curves and spot common failure modes',
      'Use sanity checks: initial loss, overfitting a tiny batch, gradient norms',
      'Implement early stopping with patience and checkpointing'
    ],
    sections: [
      { type: 'text', title: 'The canonical loop', md: `\`\`\`text
initialize parameters
for epoch in range(num_epochs):
    shuffle the training data
    for each mini-batch (X_b, y_b):
        predictions = model(X_b)                 # forward pass
        loss = loss_fn(predictions, y_b)
        gradients = backward(loss)               # backprop
        parameters -= learning_rate * gradients  # (or optimizer.step())
    evaluate on the validation set               # no gradient updates here!
    log training & validation loss / metrics
    save a checkpoint if validation improved; stop early if it hasn't for a while
\`\`\`

That's it — every framework's training script is an elaboration of this. In PyTorch the update block reads \`optimizer.zero_grad(); loss.backward(); optimizer.step()\`.` },
      { type: 'code', title: 'A complete NumPy training loop', md: `A 2-16-2 network learning two interleaved half-moons, with mini-batches, a validation split and per-epoch logging.`, code: `
import numpy as np
rng = np.random.default_rng(0)

# two interleaving half-moons
n = 400
t = rng.uniform(0, np.pi, n); cls = rng.integers(0, 2, n)
X = np.where(cls[:, None] == 0, np.column_stack([np.cos(t), np.sin(t)]), np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)]))
X = X + rng.normal(0, 0.12, X.shape); y = cls
Xtr, ytr, Xva, yva = X[:300], y[:300], X[300:], y[300:]

H = 16
P = {"W1": rng.normal(0, np.sqrt(2 / 2), (2, H)), "b1": np.zeros(H), "W2": rng.normal(0, np.sqrt(2 / H), (H, 2)), "b2": np.zeros(2)}

def forward(X, P):
    z1 = X @ P["W1"] + P["b1"]; h1 = np.maximum(0, z1)
    z2 = h1 @ P["W2"] + P["b2"]; z2 = z2 - z2.max(1, keepdims=True)
    p = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
    return p, (X, z1, h1)

def loss_acc(X, y, P):
    p, _ = forward(X, P)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)), np.mean(p.argmax(1) == y)

lr, batch = 0.5, 32
for epoch in range(1, 151):
    perm = rng.permutation(len(Xtr))
    for s in range(0, len(Xtr), batch):
        idx = perm[s:s + batch]; Xb, yb = Xtr[idx], ytr[idx]
        p, (Xc, z1, h1) = forward(Xb, P)
        d2 = p.copy(); d2[np.arange(len(yb)), yb] -= 1; d2 /= len(yb)
        g = {"W2": h1.T @ d2, "b2": d2.sum(0)}
        d1 = (d2 @ P["W2"].T) * (z1 > 0)
        g["W1"] = Xc.T @ d1; g["b1"] = d1.sum(0)
        for k in P:
            P[k] -= lr * g[k]
    if epoch % 30 == 0:
        tl, ta = loss_acc(Xtr, ytr, P); vl, va = loss_acc(Xva, yva, P)
        print(f"epoch {epoch:3d}  train loss {tl:.3f} acc {ta:.3f}  |  val loss {vl:.3f} acc {va:.3f}")
` },
      { type: 'text', title: 'Reading loss curves', md: `Plot training and validation loss every epoch (or every N steps). Typical pictures:

| What you see | Likely cause | What to try |
| --- | --- | --- |
| Loss doesn't move | learning rate too small, bug (no gradient flow), dead ReLUs | raise LR; check gradients are non-zero |
| Loss explodes / NaN | learning rate too large, bad data, log(0) | lower LR; gradient clipping; check inputs |
| Both losses high and flat | underfitting | bigger model, better features, train longer |
| Train ↓, val ↓ then ↑ | overfitting | early stopping, regularization, more data |
| Val loss very noisy | tiny validation set or batch | bigger validation set; smoothing |
| Loss jumps periodically | data not shuffled, or LR schedule restarts | shuffle; check scheduler |` },
      { type: 'check', question: { type: 'mcq', q: 'A 10-class classifier starts training with a loss around 2.30. Is that suspicious?', options: ['Yes — it should start near 0', 'No — ln(10) ≈ 2.30 is exactly the loss of uniform random guessing', 'Yes — it should be negative', 'No — the starting loss is meaningless'], answer: 1, explain: 'With small random weights the model predicts ~1/10 for each class: −ln(0.1) ≈ 2.30. A much higher initial loss suggests a bug or bad initialization.' } },
      { type: 'text', title: 'Sanity checks before long runs', md: `1. **Check the initial loss** — for K balanced classes it should be about ln K.
2. **Overfit one small batch** — train on, say, 20 examples. A correct pipeline drives the loss to ~0 quickly. If it can't, there is a bug (wrong labels, broken gradients, frozen parameters).
3. **Look at the data** right before it enters the model (after augmentation and preprocessing) — mismatched images and labels are a classic bug.
4. **Monitor gradient norms** — all zeros means no learning; huge values signal explosion.
5. **Start simple** — a smaller model and a known-good learning rate; scale up once it works.
6. **Fix random seeds** for reproducibility while debugging.` },
      { type: 'text', title: 'Early stopping and checkpoints', md: `Keep track of the best validation loss. After each epoch:
- if it improved: save a **checkpoint** (the parameters, optimizer state and epoch) and reset a counter;
- otherwise increment the counter; when it reaches the **patience** (e.g. 5–10 epochs), stop and restore the best checkpoint.

Early stopping is free regularization: training as long as it helps, and not a step longer. Checkpoints also protect long runs against crashes — large-model training saves them regularly and resumes from the latest one.` },
      { type: 'check', question: { type: 'tf', q: 'If a model cannot overfit a single small batch of 20 examples, the pipeline probably has a bug.', answer: true, explain: 'Any reasonable network can memorise 20 examples; failing to do so points to broken data, labels or gradients.' } }
    ],
    when: `- Use this loop (or the framework equivalent) for every neural network.
- Validate every epoch (or every N steps for long epochs); use early stopping by default.
- Run the sanity checks whenever you build a new model or data pipeline.`,
    pitfalls: `- Updating parameters during validation, or using training-mode behaviour (dropout) when evaluating.
- Not shuffling the training data.
- Comparing losses computed with different reductions (sum vs mean).
- Watching only the training loss.
- Running a multi-day job without first overfitting a tiny batch.`,
    takeaways: [
      'Loop: shuffle → for each batch: forward, loss, backward, update → validate → log → checkpoint.',
      'Read loss curves: flat = LR too small or bug; NaN = LR too large; gap = overfitting.',
      'Initial loss ≈ ln(K) for K classes; overfit a tiny batch before long runs.',
      'Early stopping with patience and best-checkpoint restore is cheap, effective regularization.',
      'Fix seeds and inspect the actual data entering the model when debugging.'
    ],
    quiz: [
      { type: 'order', q: 'Order the steps inside one training iteration.', items: ['Take the next mini-batch', 'Run the forward pass and compute the loss', 'Backpropagate to get gradients', 'Update the parameters'], explain: 'Batch → forward + loss → backward → update.' },
      { type: 'mcq', q: 'Training loss keeps decreasing while validation loss started increasing 5 epochs ago. What should you do?', options: ['Train much longer', 'Stop and keep the checkpoint with the best validation loss', 'Increase the learning rate', 'Remove the validation set'], answer: 1, explain: 'That is overfitting; early stopping restores the best point.' },
      { type: 'fill', q: 'What initial cross-entropy loss (natural log) do you expect for a 4-class problem with random predictions? (2 decimals)', answer: ['1.39', '1.386', 'ln4', 'ln 4', 'ln(4)'], display: 'ln 4 ≈ 1.39', explain: '−ln(1/4) = ln 4 ≈ 1.386.' },
      { type: 'mcq', q: 'The loss becomes NaN after a few hundred steps. The first thing to try is…', options: ['a bigger model', 'a lower learning rate (and check for log(0) / bad inputs)', 'more epochs', 'removing the bias terms'], answer: 1, explain: 'Divergence from too-large steps is the most common cause.' },
      { type: 'tf', q: 'You should apply gradient updates while evaluating on the validation set.', answer: false, explain: 'Validation must never change the parameters.' }
    ],
    problems: [
      {
        id: 'early-stopping-epoch', title: 'Early Stopping with Patience', difficulty: 'easy',
        prompt: `Implement \`early_stopping(val_losses, patience)\` → \`(best_epoch, stop_epoch)\`:

- Walk through the validation losses epoch by epoch (epochs are 0-based indices).
- Track the best (lowest) loss so far — it must be **strictly lower** to count as an improvement.
- Count epochs since the last improvement; when that count reaches \`patience\`, training stops **at that epoch**.
- If training never stops early, \`stop_epoch\` is the last epoch.

\`\`\`text
early_stopping([1.0, 0.8, 0.7, 0.75, 0.72, 0.71, 0.9], patience=3)  ->  (2, 5)
\`\`\``,
        starter: `
def early_stopping(val_losses, patience):
    # TODO
    pass
`,
        solution: `
def early_stopping(val_losses, patience):
    best, best_epoch, wait = float("inf"), -1, 0
    for epoch, loss in enumerate(val_losses):
        if loss < best:
            best, best_epoch, wait = loss, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                return best_epoch, epoch
    return best_epoch, len(val_losses) - 1
`,
        hints: ['Keep `best`, `best_epoch` and a `wait` counter.', 'Reset `wait` on improvement; otherwise increment it and stop when it reaches patience.'],
        explain: `This is the logic inside every EarlyStopping callback. In a real loop you would also restore the parameters saved at \`best_epoch\` — the stop epoch is merely when you noticed things weren't improving.`,
        tests: [
          { call: 'early_stopping([1.0, 0.8, 0.7, 0.75, 0.72, 0.71, 0.9], 3)', expect: '(2, 5)' },
          { name: 'never stops early', call: 'early_stopping([3.0, 2.0, 1.5, 1.2], 2)', expect: '(3, 3)' },
          { name: 'equal loss is not an improvement', call: 'early_stopping([1.0, 1.0, 1.0], 2)', expect: '(0, 2)', hidden: true },
          { name: 'patience 1', call: 'early_stopping([0.5, 0.4, 0.45, 0.3], 1)', expect: '(1, 2)', hidden: true }
        ]
      },
      {
        id: 'train-xor', title: 'Train a Network to Learn XOR', difficulty: 'medium',
        prompt: `The perceptron couldn't learn XOR — a hidden layer can. The starter gives you \`init_params\`, \`forward\` and \`backward\` for a 2-8-1 network (tanh hidden layer, sigmoid output, binary cross-entropy).

Implement \`train(X, y, lr=1.0, epochs=2000, seed=0)\` → \`(params, losses)\`:
- \`params = init_params(seed)\`;
- each epoch (full batch): \`out, cache = forward(X, params)\`; append \`bce(out, y)\` to \`losses\`; \`grads = backward(y, cache, params)\`; update **every** parameter: \`params[k] = params[k] - lr * grads[k]\`.

The tests check that the loss falls, that all four XOR cases are predicted correctly, and the exact final loss.`,
        starter: `
import numpy as np

def init_params(seed=0, hidden=8):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 1.0, (2, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1.0, (hidden, 1)), "b2": np.zeros(1)}

def forward(X, p):
    h1 = np.tanh(X @ p["W1"] + p["b1"])
    out = 1 / (1 + np.exp(-(h1 @ p["W2"] + p["b2"])))
    return out, (X, h1, out)

def backward(y, cache, p):
    X, h1, out = cache
    dz2 = (out - y.reshape(-1, 1)) / len(X)          # sigmoid + binary cross-entropy
    grads = {"W2": h1.T @ dz2, "b2": dz2.sum(axis=0)}
    dz1 = (dz2 @ p["W2"].T) * (1 - h1 ** 2)          # tanh derivative
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads

def bce(out, y):
    out = np.clip(out.ravel(), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(out) + (1 - y) * np.log(1 - out)))

def train(X, y, lr=1.0, epochs=2000, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    params = init_params(seed)
    losses = []
    # TODO: the training loop
    return params, losses
`,
        solution: `
import numpy as np

def init_params(seed=0, hidden=8):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 1.0, (2, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1.0, (hidden, 1)), "b2": np.zeros(1)}

def forward(X, p):
    h1 = np.tanh(X @ p["W1"] + p["b1"])
    out = 1 / (1 + np.exp(-(h1 @ p["W2"] + p["b2"])))
    return out, (X, h1, out)

def backward(y, cache, p):
    X, h1, out = cache
    dz2 = (out - y.reshape(-1, 1)) / len(X)
    grads = {"W2": h1.T @ dz2, "b2": dz2.sum(axis=0)}
    dz1 = (dz2 @ p["W2"].T) * (1 - h1 ** 2)
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads

def bce(out, y):
    out = np.clip(out.ravel(), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(out) + (1 - y) * np.log(1 - out)))

def train(X, y, lr=1.0, epochs=2000, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    params = init_params(seed)
    losses = []
    for _ in range(epochs):
        out, cache = forward(X, params)
        losses.append(bce(out, y))
        grads = backward(y, cache, params)
        for k in params:
            params[k] = params[k] - lr * grads[k]
    return params, losses
`,
        hints: ['Inside the loop: forward → record the loss → backward → update.', 'Update every key: `for k in params: params[k] = params[k] - lr * grads[k]`.'],
        explain: `Four lines inside a loop are all it takes. Watch the recorded losses: they typically plateau for a while near ln 2 ≈ 0.69 (the network hasn't found a useful hidden representation yet), then drop sharply once the hidden units split the problem into OR/AND-like features.`,
        tests: [
          { name: 'learns XOR', setup: 'X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)\ny = np.array([0, 1, 1, 0], dtype=float)', code: 'params, losses = train(X, y)\nout, _ = forward(X, params)\npred = (out.ravel() >= 0.5).astype(int)\nassert pred.tolist() == [0, 1, 1, 0], "predictions %s" % pred.tolist()\nassert len(losses) == 2000 and losses[-1] < losses[0]' },
          { name: 'final loss is small', setup: 'X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)\ny = np.array([0, 1, 1, 0], dtype=float)', code: 'params, losses = train(X, y)\nassert losses[-1] < 0.05, "final loss %.4f" % losses[-1]', hidden: true },
          { name: 'exact final loss (another seed)', setup: 'X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)\ny = np.array([0, 1, 1, 0], dtype=float)', call: 'train(X, y, lr=0.8, epochs=500, seed=3)[1][-1]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Andrej Karpathy — A Recipe for Training Neural Networks', url: 'https://karpathy.github.io/2019/04/25/recipe/', note: 'the classic practical guide' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
