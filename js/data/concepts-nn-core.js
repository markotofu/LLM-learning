/* ML Atlas – content: Neural Networks (core) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'perceptron', branch: 'dl', title: 'Neurons & the Perceptron', short: 'The Perceptron', level: 1, minutes: 30, prereqs: ['logistic-regression'],
    tags: ['neuron', 'perceptron', 'xor', 'linear separability', 'history'],
    summary: 'The artificial neuron (a weighted sum, a bias and an activation) the perceptron learning rule, and the XOR problem that showed why we need layers.',
    why: `Every neural network, from a digit recognizer to GPT, is built from one tiny unit: the **artificial neuron**. It takes a weighted sum of its inputs, adds a bias and passes the result through an activation function. The perceptron (1958) was the first learning machine built from it, and its famous failure on XOR explains exactly why networks need hidden layers. Understanding one neuron makes the rest of deep learning a matter of connecting many of them.`,
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
      { type: 'demo', title: "Step through: the perceptron learning rule", demo: 'walk-perceptron' },
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
      { type: 'chart', title: "Mistakes per epoch", after: `AND and OR are linearly separable, so the mistakes drop to zero within a few epochs and stay there: the perceptron has converged. XOR never gets there: every epoch still contains mistakes, and the weights cycle forever, exactly as the convergence theorem's condition predicts.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12],"series":[{"name":"AND","y":[2,3,3,2,1,0,0,0,0,0,0,0]},{"name":"OR","y":[2,2,1,0,0,0,0,0,0,0,0,0]},{"name":"XOR","y":[3,3,4,4,4,4,4,4,4,4,4,4]}],"xLabel":"epoch (one pass over the 4 examples)","yLabel":"mistakes in the epoch","yMin":0,"yMax":4} },
      { type: 'check', question: { type: 'mcq', q: 'A perceptron predicts 1 for an example whose label is 0. How does the update change the weights?', options: ['Adds the input x (times the learning rate)', 'Subtracts the input x (times the learning rate)', 'Leaves them unchanged', 'Resets them to zero'], answer: 1, explain: 'y − ŷ = 0 − 1 = −1, so w ← w − η·x, lowering z for similar inputs.' } },
      { type: 'text', title: 'XOR and the need for layers', md: `XOR outputs 1 when exactly one input is 1: (0,1) and (1,0) are positive, (0,0) and (1,1) are negative. Plot those four points: no single straight line separates them. A single neuron draws exactly one line, so **no choice of weights works**.

In 1969 Minsky and Papert's book *Perceptrons* highlighted this limitation, and funding for neural network research dried up for years (an "AI winter").

The fix is to **combine neurons**: one hidden neuron computes OR, another computes AND, and an output neuron computes "OR and not AND", which is XOR. A hidden layer re-represents the input so that the problem *becomes* linearly separable. The missing piece in 1969 was a way to **train** hidden layers, which arrived with backpropagation (popularised in 1986).` },
      { type: 'text', title: "The perceptron finds a separator, not the best one", md: `The weight vector has a geometric meaning: the boundary w · x + b = 0 is a line (a hyperplane in more dimensions) **perpendicular to w**, and the signed distance of a point from it is

$$ distance(x) = ( w · x + b ) / ‖w‖ $$
:::where
w, b — the perceptron's weights and bias
w · x + b — the pre-activation; its sign says which side of the boundary x lies on
‖w‖ — the length of w; dividing by it turns the score into a distance
:::

How long does learning take? **Novikoff's theorem** bounds the number of mistakes on separable data: if every point lies within a radius R of the origin, and some boundary separates the classes with a gap (a margin) of γ on each side, the perceptron makes at most (R / γ)² mistakes, however many examples there are and in whatever order they arrive. Well-separated classes are learned in a few updates; classes that almost touch take many.

What the perceptron does *not* do is pick a good boundary. It stops at the first line that classifies every training point correctly, and which line that is depends on the order of the examples. Two runs on the same data can end with very different boundaries, some of them hugging one class.` },
      { type: 'chart', title: "Three runs, three different boundaries", md: `The same separable data, presented in three different random orders:`, after: `Every run stops with zero training mistakes, yet each stops at a different line: the perceptron halts at the first separator it reaches. Some lines pass uncomfortably close to points of one class. Choosing the line with the widest gap to both classes is exactly what a [support vector machine](#/concept/svm) does.`, chart: {"kind":"line","points":false,"series":[{"name":"class 0","x":[-1.131,-0.848,-1.495,-0.483,-1.587,-1.543,-2.479,-1.885,-1.499,-0.974,-1.271,-1.566,-0.703,-1.609,-1.452,-1.755,-1.288,-1.082,-1.56,-0.769,-1.892,-0.46,-1.055,-0.398,-0.209],"y":[-1.073,-0.942,-0.801,-0.479,-1.696,-0.977,-1.12,-1.403,-1.174,-0.427,-0.248,-0.807,-0.948,-1.507,-0.879,-1.115,-0.703,-0.805,-1.071,-0.179,-0.167,-0.57,-1.173,0.078,-0.277],"scatter":true},{"name":"class 1","x":[1.397,1.198,0.491,1.436,0.549,0.96,2.157,1.381,2.071,1.548,1.229,1.752,2.202,0.836,1.227,1.304,0.992,0.497,1.52,0.785,1.042,0.962,1.337,1.289,0.462],"y":[0.335,1.361,1.217,1.383,0.636,0.357,0.727,0.858,1.726,-0.212,1.376,0.66,0.274,1.514,2.101,0.652,0.4,1.347,1.712,1.929,1.866,0.595,1.567,0.678,0.229],"scatter":true},{"name":"boundary, data order 1","x":[-3,3],"y":[5.619,-3.826]},{"name":"boundary, data order 2","x":[-3,3],"y":[7.53,-5.745]},{"name":"boundary, data order 3","x":[-3,3],"y":[5.698,-5.698]}],"xLabel":"x₁","yLabel":"x₂","xMin":-3,"xMax":3,"yMin":-3,"yMax":3} },
      { type: 'text', title: "From the perceptron to gradient descent", md: `The perceptron rule is a special case of something general: it is **stochastic gradient descent** on the perceptron loss, which is zero for a correctly classified point and grows linearly with how far a misclassified point is on the wrong side:

$$ ℓ(w, b) = max( 0, −y′ · (w · x + b) ) $$
:::where
y′ — the label written as −1 or +1 instead of 0 or 1
w · x + b — the score; y′ · (w · x + b) is positive when the point is on the correct side
max(0, …) — no loss for correct points; for a mistake, the loss grows with how far the point lies on the wrong side
:::

For a misclassified point the gradient is −y′ · x, so a gradient step adds y′ · x to the weights: exactly the perceptron update. Three small changes turn this into the models used today:
- demand a **margin**, so that the loss is zero only when y′ · score ≥ 1: the hinge loss of the [SVM](#/concept/svm);
- replace the step by a smooth **sigmoid** and the loss by cross-entropy: [logistic regression](#/concept/logistic-regression), which gives probabilities and learns from every example, not only from mistakes;
- stack many such units with non-linear activations and train them all with the same gradient idea: the [multi-layer perceptron](#/concept/mlp).` },
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
      { type: 'mcq', q: 'How do multi-layer networks solve XOR?', options: ['By using a larger learning rate', 'Hidden units re-represent the inputs so the output unit\'s problem becomes linearly separable', 'By memorising the four cases in the bias', 'They cannot'], answer: 1, explain: 'E.g. hidden units for OR and AND, then output = OR AND NOT AND.' },
      {"type":"mcq","q":"Data is separable with a margin γ = 0.5, and every point lies within radius R = 2 of the origin. At most how many mistakes can the perceptron make?","options":["4","8","16","There is no bound"],"answer":2,"explain":"Novikoff's bound is (R/γ)² = (2 / 0.5)² = 16 mistakes, whatever the number or order of the examples."},
      {"type":"tf","q":"On separable data, the perceptron always finds the boundary with the largest margin.","answer":false,"explain":"It stops at the first separating boundary it reaches, which depends on the order of the examples. Maximising the margin is what an SVM does."}
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
      { title: '3Blue1Brown, But what is a neural network?', url: 'https://www.3blue1brown.com/lessons/neural-networks' },
      { title: "Nielsen, Neural Networks and Deep Learning (free book)", url: "http://neuralnetworksanddeeplearning.com/" }
    ]
  },

  // =====================================================================================
  {
    id: 'activation-functions', branch: 'dl', title: 'Activation Functions', level: 1, minutes: 35, prereqs: ['perceptron', 'derivatives'],
    tags: ['relu', 'sigmoid', 'tanh', 'gelu', 'softmax', 'vanishing gradient'],
    summary: 'Why networks need non-linearities, how sigmoid, tanh, ReLU and GELU differ, the vanishing-gradient problem, and the numerically stable softmax.',
    why: `Without activation functions, a 100-layer network collapses into a single linear model: all that depth for nothing. The choice of activation also decides whether gradients can flow through deep networks: the switch from sigmoids to **ReLU** was one of the key reasons deep learning took off around 2012, and transformers use smooth relatives like **GELU** and **SwiGLU**. At the output, **softmax** turns scores into the probabilities every classifier and LLM produces.`,
    objectives: [
      'Prove that stacking linear layers without activations is still linear',
      'Compare sigmoid, tanh, ReLU, Leaky ReLU and GELU: shapes, ranges and gradients',
      'Explain the vanishing-gradient problem and why ReLU helps',
      'Implement a numerically stable softmax and pair output activations with losses'
    ],
    sections: [
      { type: 'text', title: 'Why non-linearity is essential', md: `Two linear layers in a row compute (XW₁)W₂ = X(W₁W₂), which is just **one** linear layer with weight matrix W₁W₂. However many linear layers you stack, the result is a linear function, and a linear function can't learn XOR, curved boundaries or anything interesting.

Inserting a non-linear function φ between layers breaks this collapse: φ(XW₁)W₂ cannot be rewritten as a single matrix product. With non-linearities, networks become **universal approximators**: given enough hidden units they can approximate any continuous function.` },
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
`, predict: {"type":"mcq","q":"Is applying two linear layers in a row, (X @ W1) @ W2, equivalent to a single linear layer?","options":["Yes, always","No, two layers are always more expressive","Only if W1 and W2 are square","Only if the layers have biases"],"answer":0,"explain":"Yes: (X W1) W2 = X (W1 W2), and W1 W2 is just another matrix. Without a non-linearity between them, extra linear layers add nothing; with a ReLU in between, the equality breaks.","expect":["two linear layers == one linear layer: True"]} },
      { type: 'demo', title: 'Interactive: activation functions and their gradients', md: `Toggle functions and show their derivatives. Watch where each derivative becomes (nearly) zero.`, demo: 'activations' },
      { type: 'text', title: 'The main activation functions', md: `| Function | Formula | Range | Derivative | Used in |
| --- | --- | --- | --- | --- |
| **Sigmoid** | 1/(1+e⁻ᶻ) | (0, 1) | σ(1−σ), max 0.25 | binary output layers, gates (LSTM) |
| **Tanh** | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (−1, 1) | 1 − tanh², max 1 | RNNs; zero-centred |
| **ReLU** | max(0, z) | [0, ∞) | 1 if z > 0 else 0 | default for hidden layers (CNNs, MLPs) |
| **Leaky ReLU** | z if z > 0 else αz | (−∞, ∞) | 1 or α | fixes "dying ReLU" |
| **GELU** | z·Φ(z) (Φ = Gaussian CDF) | ≈ (−0.17, ∞) | smooth | BERT, GPT |
| **SiLU / Swish** | z·σ(z) | ≈ (−0.28, ∞) | smooth | LLaMA's SwiGLU, EfficientNet |

**ReLU's advantages**: trivially cheap; gradient exactly 1 for positive inputs (no shrinking); produces sparse activations. **Its weakness**: a neuron whose input is always negative outputs 0 and gets gradient 0 forever, a "dead" ReLU (Leaky ReLU, GELU and careful initialization help).` },
      { type: 'text', title: 'Vanishing gradients', md: `Backpropagation multiplies the activation's derivative at **every layer**. The sigmoid's derivative is at most 0.25, and nearly 0 when |z| is large (the function is "saturated", flat). Through 10 sigmoid layers the gradient can shrink by a factor of 0.25¹⁰ ≈ 10⁻⁶: the early layers effectively stop learning. This **vanishing gradient problem** made deep networks hard to train before ~2010.

Remedies that made deep learning work:
- **ReLU-family activations** (derivative 1 on the positive side);
- **good initialization** (He/Xavier) and **normalization layers** (BatchNorm, LayerNorm);
- **residual connections** (the gradient has a shortcut path: key to ResNets and transformers);
- **gated architectures** (LSTMs) for recurrent nets.

The opposite, **exploding gradients**, happens when the factors are larger than 1; gradient clipping is the usual fix.` },
      { type: 'chart', title: "How the gradient's size changes on its way back through 20 layers", md: `A 20-layer network, 100 units per layer, random weights; the same gradient is sent back from the output and measured at every layer:`, after: `With sigmoid activations the gradient shrinks by a factor of about 4 at every layer (the sigmoid's slope is at most 0.25), so at layer 1 it is about 8 × 10⁻¹³ of its size at layer 20: the early layers get almost no learning signal. tanh with Xavier initialisation loses only a factor of about 5 over all 20 layers, and ReLU with He initialisation keeps the gradient at roughly the same size throughout. Better activations and initialisation are what made deep networks trainable.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"yLog":true,"points":false,"series":[{"name":"sigmoid (Xavier init)","y":[1.97e-13,9.794e-13,3.826e-12,1.806e-11,8.299e-11,3.497e-10,1.624e-9,6.711e-9,2.986e-8,1.239e-7,4.927e-7,0.000002103,0.000009547,0.00004358,0.0001968,0.0008265,0.003599,0.01518,0.06578,0.26]},{"name":"tanh (Xavier init)","y":[0.1954,0.2826,0.3487,0.4384,0.5331,0.5621,0.6684,0.6512,0.7192,0.7269,0.7035,0.7223,0.7867,0.9306,0.988,0.9737,1.021,1.057,1.121,1.06]},{"name":"ReLU (He init)","y":[3.68,3.68,3.667,3.399,3.307,3.206,3.027,3.077,3.063,2.935,2.612,2.418,2.209,2.191,1.972,1.674,1.482,1.41,1.398,1.066]}],"xLabel":"layer (1 = closest to the input; the gradient arrives from layer 20)","yLabel":"typical gradient size (log scale)"} },
      { type: 'text', title: "Dead ReLUs and how to avoid them", md: `ReLU's gradient is exactly 0 for negative inputs. Usually that is harmless, because a unit is negative for some examples and positive for others. But if a unit's input becomes negative for **every** example, for instance after one huge update pushed its bias far below zero, it outputs 0 everywhere, receives a gradient of 0 everywhere, and never changes again: a **dead ReLU**. In a badly tuned network a large share of units can die this way, silently wasting capacity.

Ways to avoid it:
- a **sensible learning rate** (with warm-up) and **gradient clipping**, so that no single step throws the biases far away;
- **good initialisation** and **normalisation** layers, which keep pre-activations centred around 0 (see [Initialization, Normalization & Dropout](#/concept/normalization));
- activations with a non-zero slope for negative inputs: **Leaky ReLU** (a small slope α such as 0.01) or **ELU**, or the smooth **GELU** and **SiLU**, whose gradient is small but not exactly zero there.` },
      { type: 'code', title: "Counting dead units", code: `import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 64))                       # 1,000 inputs
W = rng.normal(0, np.sqrt(2 / 64), (64, 256))         # a ReLU layer with 256 units, He initialisation
b = np.zeros(256)

def dead_share(b):
    z = X @ W + b
    return np.mean((z <= 0).all(axis=0))               # units that are negative for every input

print(f"healthy biases:       {dead_share(b):.0%} of units dead")
b_bad = b.copy()
b_bad[:128] -= 6.0                                     # one bad update pushes half the biases far down
print(f"after the bad update: {dead_share(b_bad):.0%} of units dead")

z = X @ W + b_bad
relu_grad = (z > 0).mean(axis=0)                       # share of inputs that let a gradient through, per unit
leaky_grad = np.where(z > 0, 1.0, 0.01).mean(axis=0)   # Leaky ReLU with slope 0.01 on the negative side
r, l = relu_grad[:128].mean(), leaky_grad[:128].mean()
print(f"gradient reaching the pushed-down units: ReLU {r:.4f}, Leaky ReLU {l:.4f}")
print("only Leaky ReLU keeps a gradient:", bool(r < 0.001 and l > 0.005))`, after: `About half the units now output zero for every one of the 1,000 inputs, so plain ReLU passes almost no gradient back to them and they cannot recover. Leaky ReLU still passes 1% of the gradient, which is enough for later updates to pull those biases back.`, predict: {"type":"mcq","q":"Half of a ReLU layer's biases are pushed down by 6, while the pre-activations vary by only about 1.4. About what share of the layer's units is now dead?","options":["About 0%","About 10%","About 50%","100%"],"answer":2,"explain":"For the pushed-down units the input is about 4 standard deviations below zero for practically every example, so almost all of them, half the layer, are dead.","expect":["only Leaky ReLU keeps a gradient: True"]} },
      { type: 'check', question: { type: 'mcq', q: 'Why did ReLU help train deeper networks than sigmoid?', options: ['It is bounded between 0 and 1', 'Its derivative is 1 for positive inputs, so gradients don\'t shrink layer after layer', 'It is smoother', 'It outputs probabilities'], answer: 1, explain: 'Sigmoid derivatives (≤ 0.25) multiply into vanishing gradients; ReLU passes gradients through unchanged where active.' } },
      { type: 'text', title: "Gated units: GLU and SwiGLU", md: `Modern transformers rarely use a plain activation in their feed-forward blocks. They use a **gated linear unit**: two linear projections of the same input, one passed through an activation and used as a gate that multiplies the other, element by element:

$$ SwiGLU(x) = ( SiLU(x W) ⊙ (x V) ) W₂ $$
:::where
x — the token's input vector
W, V — two separate weight matrices that project x up to the hidden width
SiLU(z) — z · σ(z), a smooth ReLU-like curve (also called Swish)
⊙ — element-wise multiplication: the SiLU branch decides how much of each hidden unit of x V passes through
W₂ — the projection back down to the model width
:::

The gate lets the network switch features on and off depending on the input, and gated units train to a lower loss for the same compute; LLaMA, PaLM and most recent open models use SwiGLU. Because it has three weight matrices instead of two, the hidden width is usually reduced to about two thirds (for example 8/3 of the model width instead of 4 times) to keep the number of parameters the same.` },
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
`, predict: {"type":"mcq","q":"What does the naive softmax, exp(z) / sum(exp(z)), return for the logits [1000, 999, 998]?","options":["About [0.67, 0.24, 0.09]","[1, 0, 0]","[nan, nan, nan]","[0.33, 0.33, 0.33]"],"answer":2,"explain":"exp(1000) overflows to infinity, and inf / inf is nan. Subtracting the largest logit first gives the same mathematical result without overflow: [0.665, 0.245, 0.09].","expect":["naive, big logits:  [nan nan nan]"]} },
      { type: 'check', question: { type: 'tf', q: 'Adding the same constant to every logit does not change the softmax output.', answer: true, explain: 'e^(z+c) = e^c·e^z; the e^c cancels between numerator and denominator: that is why subtracting the max is safe.' } }
    ],
    when: `- **Hidden layers**: ReLU by default for MLPs and CNNs; GELU/SiLU (or SwiGLU) in transformers; tanh inside classic RNNs.
- **Outputs**: linear for regression, sigmoid for binary/multi-label, softmax for multi-class.
- Avoid sigmoid/tanh in deep stacks of hidden layers (vanishing gradients) unless the architecture needs them (gates).`,
    pitfalls: `- Forgetting the activation between layers (the network becomes linear).
- Applying softmax *and* then using a loss that applies it again (e.g. PyTorch's CrossEntropyLoss expects raw logits).
- Naive softmax/sigmoid overflow: use stable formulas.
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
      { type: 'mcq', q: 'For multi-label classification (an image can have several tags), which output activation fits?', options: ['Softmax over all tags', 'An independent sigmoid per tag', 'ReLU', 'No activation'], answer: 1, explain: 'Softmax forces probabilities to compete and sum to 1; independent sigmoids allow several tags at once.' },
      {"type":"tf","q":"A ReLU unit whose input is negative for every example receives no gradient and can stay dead for the rest of training.","answer":true,"explain":"ReLU's gradient is 0 for negative inputs, so if a unit is negative for all inputs, no update can reach its weights or bias. Leaky ReLU, GELU or SiLU keep a small gradient."},
      {"type":"mcq","q":"In a SwiGLU feed-forward block, what does the SiLU branch do?","options":["It normalises the input","It acts as a gate that scales each hidden unit of the other projection","It replaces attention","It computes the softmax over the vocabulary"],"answer":1,"explain":"SwiGLU multiplies SiLU(xW) element-wise with xV, so the first projection decides how much of each hidden unit of the second passes through."}
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
        explain: `\`np.maximum\` compares element by element with 0; \`np.max\` would instead return the single largest element: a classic slip. Leaky ReLU keeps a small slope for negative inputs so the neuron still receives gradient.`,
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
        explain: `These derivatives are cheap because they reuse the forward value: σ′ = σ(1 − σ), tanh′ = 1 − tanh². The hidden test compares them with numerical derivatives: the same gradient-checking idea you'll use for whole networks.`,
        tests: [
          { call: 'sigmoid_grad(np.array([0.0]))', expect: 'np.array([0.25])' },
          { call: 'relu_grad(np.array([-1.0, 0.0, 2.0]))', expect: 'np.array([0.0, 0.0, 1.0])' },
          { call: 'tanh_grad(np.array([0.0, 1.0]))', expect: 'np.array([1.0, 1 - np.tanh(1.0) ** 2])' },
          { name: 'matches numerical derivatives', setup: 'z = np.linspace(-4, 4, 17) + 0.05\nh = 1e-6', code: 'num_s = (1 / (1 + np.exp(-(z + h))) - 1 / (1 + np.exp(-(z - h)))) / (2 * h)\nnum_t = (np.tanh(z + h) - np.tanh(z - h)) / (2 * h)\nassert np.allclose(sigmoid_grad(z), num_s, atol=1e-7)\nassert np.allclose(tanh_grad(z), num_t, atol=1e-7)\nassert np.allclose(relu_grad(z), (z > 0).astype(float))', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Hendrycks & Gimpel, Gaussian Error Linear Units (GELU)', url: 'https://arxiv.org/abs/1606.08415' },
      { title: "Zhang, Lipton, Li & Smola, Dive into Deep Learning (free book)", url: "https://d2l.ai/" }
    ]
  },

  // =====================================================================================
  {
    id: 'mlp', branch: 'dl', title: 'Multi-Layer Networks & the Forward Pass', short: 'Multi-Layer Perceptrons', level: 1, minutes: 40, prereqs: ['activation-functions', 'numpy'],
    tags: ['mlp', 'dense layer', 'hidden layer', 'forward pass', 'universal approximation', 'parameters'],
    summary: 'Stacking layers of neurons: dense layers as matrix multiplications, tracking shapes through a network, hidden representations, depth vs width, and counting parameters.',
    why: `The multi-layer perceptron (MLP) is the basic neural network, and it is everywhere: as a complete model for tabular data, as the classification head on top of CNNs, and as the **feed-forward block inside every transformer layer** (where most of an LLM's parameters live). Being able to write its forward pass in a few lines of NumPy, and to reason about shapes and parameter counts, is a core deep-learning skill.`,
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

In deep networks trained on images, early layers learn edges and colour blobs, middle layers combine them into textures and parts, and later layers respond to whole objects. In language models, layers progressively build representations of syntax, meaning and context. **The network learns its own features**: that's the big difference from classical ML, where you engineer features by hand.

**Universal approximation theorem:** an MLP with one hidden layer and enough neurons can approximate any continuous function on a bounded domain arbitrarily well. It says nothing about how many neurons that takes or whether training will find them: in practice **depth** (several layers) represents complex functions with far fewer parameters than one enormous layer, because each layer reuses the features built by the previous one.` },
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
`, after: `Training finds weights like these automatically: that is what backpropagation is for (next lesson).` },
      { type: 'text', title: 'Outputs, losses and sizes', md: `The last layer matches the task:
- **Regression**: one linear output per target, MSE loss.
- **Binary classification**: one output + sigmoid (or two + softmax), binary cross-entropy.
- **K classes**: K outputs + softmax, cross-entropy.

**Sizing an MLP** (rules of thumb): start with 1–3 hidden layers of 64–512 units; widths often stay constant or shrink towards the output; more data supports bigger networks; use validation curves to decide.

**Parameter counting**: for layer sizes [d₀, d₁, …, d_L]:

$$ params = Σ_l ( d_{l−1} · d_l + d_l ) $$
:::where
d₀ — the number of input features
d_l — the width (number of neurons) of layer l; d_L is the output size
d_{l−1} · d_l — the weights of layer l, one per (input, neuron) pair
+ d_l — the biases of layer l, one per neuron
Σ_l — add up over the layers l = 1 … L
:::

A transformer's feed-forward block is exactly an MLP: d_model → 4·d_model → d_model. For GPT-2 small (d_model = 768) that is about 4.7 million parameters per layer, roughly two thirds of each layer's weights.` },
      { type: 'code', title: "Neural networks in scikit-learn: MLPClassifier", timeoutSec: 90, md: `\`MLPClassifier\` builds the networks of this lesson for you. \`hidden_layer_sizes=(100,)\` means one hidden layer of 100 units; \`(10, 10)\` means two hidden layers of 10 units each. The data are two interleaving half-moons, which no straight line can separate.`, code: `import warnings
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

X, y = make_moons(n_samples=200, noise=0.25, random_state=3)
print(f"X: {X.shape[0]} points with 2 coordinates; y says which moon (0 or 1). The first 3:")
for row, label in zip(X[:3], y[:3]):
    print("  ", row.round(2).tolist(), "->", label)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

for hidden in [(10,), (100,), (10, 10)]:
    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=3000, random_state=0).fit(X_train, y_train)
    print(f"hidden layers {str(hidden):<9} train {mlp.score(X_train, y_train):.3f}   test {mlp.score(X_test, y_test):.3f}")

print("same network, different random starting weights:")
for seed in range(4):
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000, random_state=seed).fit(X_train, y_train)
    print(f"  random_state={seed}: test {mlp.score(X_test, y_test):.3f}")`, after: `The 100-unit network draws a curved boundary that follows the moons; the 10-unit networks are too small for this shape. Notice also that **the same architecture gives different results with different random starting weights**: training a network is a non-convex optimization, so different starting points end in different solutions. Small networks are especially sensitive. Report results over several seeds, or average several networks.` },
      { type: 'code', title: "Scaling and regularization decide the result", timeoutSec: 90, md: `The breast-cancer features range from about 0.001 to 2,500. Neural networks assume inputs of similar size, typically mean 0 and standard deviation 1.`, code: `import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

raw = MLPClassifier(max_iter=1000, random_state=42).fit(X_train, y_train)
print(f"raw features      train {raw.score(X_train, y_train):.3f}   test {raw.score(X_test, y_test):.3f}")
scaled = MLPClassifier(max_iter=1000, random_state=42).fit(Xs_train, y_train)
print(f"scaled            train {scaled.score(Xs_train, y_train):.3f}   test {scaled.score(Xs_test, y_test):.3f}")
reg = MLPClassifier(max_iter=1000, alpha=1, random_state=42).fit(Xs_train, y_train)
print(f"scaled, alpha=1   train {reg.score(Xs_train, y_train):.3f}   test {reg.score(Xs_test, y_test):.3f}")
print("scaling helped:", scaled.score(Xs_test, y_test) > raw.score(X_test, y_test))`, after: `Scaling alone lifts the test accuracy by several points. The scaled network then fits the training data almost perfectly, a hint of overfitting; increasing **alpha** (the L2 penalty on the weights, as in ridge regression) trades a little training accuracy for better test accuracy.`, predict: {"type":"mcq","q":"The same network is trained on raw features (ranging from 0.001 to 2,500) and on standardized features. Which reaches the higher test accuracy?","options":["Raw features: scaling throws information away","Standardized features","They are identical: neural networks are scale-invariant","Neither learns anything"],"answer":1,"explain":"Standardized features train much better (0.965 against 0.916 here). Gradient-based training struggles when inputs differ in size by factors of a million: the large features dominate the first layer.","expect":["scaling helped: True"]} },
      { type: 'chart', title: "Training loss per epoch, raw versus scaled inputs", after: `With raw features the loss falls slowly and erratically: a few huge features dominate the gradients. With standardized features it falls smoothly and much lower. Watching this curve is the first thing to do when a network trains badly.`, chart: {"kind":"line","x":[1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93,97,101,105,109,113,117,121,125,129,133,137,141,145],"points":false,"yLog":true,"series":[{"name":"raw features","y":[22.5908,9.2735,2.2284,0.5973,0.5817,0.425,0.3355,0.3046,0.2881,0.3169,0.2697,0.2718,0.2478,0.2367,0.2315,0.2238,0.2238,0.225,0.2208,0.1954,0.1885,0.1905,0.179,0.1763,0.1787,0.1784,0.184,0.1682,0.23,0.1716,0.1657,0.1736,0.1608,0.1574,0.1581,0.19,0.1605]},{"name":"standardized features","y":[0.5021,0.263,0.1812,0.1423,0.1198,0.1049,0.0945,0.0866,0.08,0.0748,0.0703,0.0664,0.0628,0.0595,0.0569,0.0544,0.0518,0.0495,0.0472,0.0454,0.0436,0.0419,0.0404,0.0389,0.0376,0.0363,0.0353,0.0344,0.0327,0.0316,0.0304,0.0295,0.0286,0.0275,0.0266,0.0258,0.0249]}],"xLabel":"epoch (pass over the training data)","yLabel":"training log loss (log scale)"} },
      { type: 'text', title: "Tuning a neural network", md: `The settings that matter most in \`MLPClassifier\`:

| Setting | What it controls | Typical values |
| --- | --- | --- |
| \`hidden_layer_sizes\` | number of layers and units per layer: the capacity | one or two layers of 10–1000 units |
| \`alpha\` | L2 penalty on the weights | 0.0001 (default) up to about 10 |
| \`activation\` | the non-linearity | \`"relu"\` (default) or \`"tanh"\` |
| \`solver\` | the optimizer | \`"adam"\` (default, needs scaled data), \`"lbfgs"\` (robust on small data) |
| \`max_iter\`, \`early_stopping\` | how long to train | raise \`max_iter\` if you see convergence warnings |

A dependable recipe: first make the network **large enough to overfit** (training accuracy near 100%), which proves it can learn the task. Then **shrink it or regularize it** (larger alpha, fewer units, early stopping) until the validation score is as high as possible.

A rough measure of a network's complexity is its number of weights. With 30 inputs, one hidden layer of 100 units and 1 output, there are 30·100 + 100 = 3,100 weights in the first layer and 100 + 1 = 101 in the second: about 3,200 in all. A second layer of 100 units adds another 10,100.` },
      { type: 'text', title: "Strengths and weaknesses", md: `| Strengths | Weaknesses |
| --- | --- |
| can capture very complex relationships in large amounts of data | large networks take a long time to train |
| with enough data, time and tuning, often the most accurate model | need scaled, preprocessed inputs; best when all features mean similar things (pixels, audio samples) |
| one framework covers images, sound, text and tables | many settings, and results vary with the random start |
| the basis of everything in the deep-learning branches of this map | on heterogeneous tables (mixed units and categories), tree ensembles are often better |

scikit-learn's \`MLPClassifier\` is fine for small problems. Real deep learning, with GPUs, convolutional layers and transformers, uses [PyTorch](#/concept/pytorch).` },
      { type: 'check', question: { type: 'mcq', q: 'What is the main role of hidden layers?', options: ['To store the training data', 'To learn intermediate representations (features) that make the final prediction easier', 'To speed up training', 'To normalize the inputs'], answer: 1, explain: 'Hidden layers transform the input into features the next layer can use: learned feature engineering.' } }
    ],
    when: `- **Tabular data**: an MLP is a reasonable alternative to boosted trees, especially with lots of data or when combined with embeddings for categorical features (trees often still win on small tables).
- **As a component**: classification heads, transformer feed-forward blocks, projection layers.
- For images, audio and sequences, use architectures with the right inductive bias (CNNs, transformers), which contain MLP pieces.`,
    pitfalls: `- Shape mismatches: write down (n, d_in) @ (d_in, d_out) at every layer.
- Forgetting the activation between layers.
- Applying softmax in the model *and* in the loss.
- Making the network huge for a small dataset (overfitting): use validation curves and regularization.`,
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
      { type: 'mcq', q: 'Inside a transformer layer, the feed-forward block is…', options: ['a convolution', 'an MLP applied to each token\'s vector', 'a decision tree', 'an RNN'], answer: 1, explain: 'Typically d_model → 4·d_model → d_model with a GELU/SwiGLU non-linearity.' },
      {"type":"mcq","q":"A network reaches 100% training accuracy but only 85% on validation data. Which change is most likely to help?","options":["Increase alpha (stronger L2 regularization)","Remove the feature scaling","Use more hidden units","Train for even longer"],"answer":0,"explain":"The gap shows overfitting. A stronger penalty on the weights, a smaller network or early stopping reduce it; more units or longer training make it worse."}
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
        hints: ['`X @ W + b`: the bias broadcasts over the rows.', 'For parameters, pair consecutive sizes with `zip(sizes[:-1], sizes[1:])`.'],
        explain: `A dense layer is one matrix multiply, one broadcast add and one element-wise function: the entire computation of a fully connected neuron layer, for the whole batch at once.`,
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
      { title: '3Blue1Brown, Neural networks (series)', url: 'https://www.3blue1brown.com/topics/neural-networks' },
      { title: 'Dive into Deep Learning, Multilayer Perceptrons', url: 'https://d2l.ai/chapter_multilayer-perceptrons/index.html' },
      { title: "Andreas Müller & Sarah Guido, Introduction to Machine Learning with Python (book; code and notebooks)", url: "https://github.com/amueller/introduction_to_ml_with_python" }
    ]
  },

  // =====================================================================================
  {
    id: 'backpropagation', branch: 'dl', title: 'Backpropagation', level: 2, minutes: 50, prereqs: ['mlp', 'derivatives', 'loss-functions', 'matrix-calculus'],
    tags: ['backprop', 'chain rule', 'computational graph', 'gradients', 'gradient checking'],
    summary: 'How a network computes the gradient of the loss for every weight in one backward sweep: computational graphs, local gradients, the chain rule in matrix form, and gradient checking.',
    why: `Backpropagation is the algorithm that makes deep learning possible. A modern network has millions to trillions of parameters; computing each gradient separately (as with numerical derivatives) would be hopeless. Backprop computes **all** of them with roughly the cost of one extra forward pass, by applying the chain rule backwards through the network and reusing intermediate results. Every framework (PyTorch, TensorFlow, JAX) automates it, but understanding it lets you debug training, design layers, and reason about vanishing gradients.`,
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

Each operation only needs to know its **local** derivative (how its output changes with its inputs) and receives the "upstream" gradient from the node after it. When a value feeds into several places, the gradients from all paths **add up**.` },
      { type: 'text', title: 'The backward rules you need', md: `For a batch, with δ denoting "gradient of the loss with respect to this quantity" (same shape as the quantity):

| Forward | Backward |
| --- | --- |
| Z = X W + b | dW = Xᵀ δZ, db = Σ_rows δZ, dX = δZ Wᵀ |
| H = relu(Z) | δZ = δH ⊙ (Z > 0) |
| H = σ(Z) | δZ = δH ⊙ H ⊙ (1 − H) |
| P = softmax(Z), L = mean cross-entropy | δZ = (P − Y_onehot) / n |

:::where Symbols in this table
X: a dense layer's input batch, shape (n, d_in)
W, b: the layer's weights (d_in, d_out) and bias (d_out,)
Z: the pre-activations X W + b
H: the activations after the nonlinearity (relu or σ)
δZ, δH: the gradients of the loss with respect to Z and H; each has the same shape as the thing it belongs to
dW, db, dX: the gradients with respect to W, b and X
Σ_rows: a sum over the batch dimension (axis 0)
⊙: element-wise multiplication (NumPy's * operator)
(Z > 0): a 0/1 mask that is 1 wherever Z was positive
P: the softmax probabilities, one row per example
Y_onehot: the true labels as one-hot rows (1 in the correct class's column)
n: the batch size
:::

The softmax + cross-entropy result is famously simple: the gradient of the logits is just **predicted probabilities minus the one-hot truth** (divided by the batch size for a mean loss).

**Shape rule of thumb:** a gradient always has the same shape as the thing it is the gradient of. If dW must be (d_in, d_out) and you have X (n, d_in) and δZ (n, d_out), the only product that fits is Xᵀ δZ. When in doubt, let the shapes tell you the formula.` },
      { type: 'text', title: "A worked example by hand", md: `One neuron, one example, every number visible. The input is x = 1.5 with target y = 0; the parameters are w = 0.8 and b = −0.2; the neuron applies a sigmoid and the loss is the squared error.

| Step | Forward value | Local derivative | Gradient arriving from above |
| --- | --- | --- | --- |
| L = (p − y)² | L = 0.534 | ∂L/∂p = 2(p − y) = 1.462 | ∂L/∂L = 1 |
| p = σ(z) | p = 0.731 | ∂p/∂z = p(1 − p) = 0.197 | ∂L/∂p = 1.462 |
| z = w·x + b | z = 1.0 | ∂z/∂w = x = 1.5, ∂z/∂b = 1, ∂z/∂x = w = 0.8 | ∂L/∂z = 1.462 × 0.197 = 0.287 |

The parameter gradients multiply the chain together:

$$ ∂L/∂w = ∂L/∂p · ∂p/∂z · ∂z/∂w = 1.462 × 0.197 × 1.5 = 0.431 $$
:::where
∂L/∂p — how the loss changes with the prediction (the error signal)
∂p/∂z — the sigmoid's slope at z = 1
∂z/∂w — how the pre-activation changes with the weight: the input x
∂L/∂w — the product of the three: the gradient used to update w
:::

Likewise ∂L/∂b = 0.287 × 1 = 0.287, and ∂L/∂x = 0.287 × 0.8 = 0.230 (needed only when x comes from an earlier layer). The prediction is too high and both gradients are positive, so a gradient step lowers w and b; w moves more, because its effect on z is amplified by x = 1.5.` },
      { type: 'code', title: "The worked example in code", code: `import math
x, y = 1.5, 0.0                  # one training example
w, b = 0.8, -0.2                 # the current parameters

def forward(w, b):
    z = w * x + b
    p = 1 / (1 + math.exp(-z))
    return z, p, (p - y) ** 2

z, p, L = forward(w, b)
print(f"forward:  z = {z:.4f}   p = {p:.4f}   L = {L:.4f}")

dL_dp = 2 * (p - y)              # backward, one local derivative at a time
dp_dz = p * (1 - p)
dL_dz = dL_dp * dp_dz
dL_dw, dL_db = dL_dz * x, dL_dz * 1.0
print(f"backward: dL/dp = {dL_dp:.4f}   dp/dz = {dp_dz:.4f}   dL/dz = {dL_dz:.4f}   dL/dw = {dL_dw:.4f}   dL/db = {dL_db:.4f}")

h = 1e-6                         # check dL/dw numerically
numeric = (forward(w + h, b)[2] - forward(w - h, b)[2]) / (2 * h)
print(f"numerical dL/dw = {numeric:.4f}")

lr = 0.5
w_new, b_new = w - lr * dL_dw, b - lr * dL_db
L_new = forward(w_new, b_new)[2]
print(f"after one step: w = {w_new:.4f}, b = {b_new:.4f}, loss {L_new:.4f} (was {L:.4f})")
print("w decreases and the loss falls:", w_new < w and L_new < L)`, after: `The hand-computed gradient matches the numerical check. One step with learning rate 0.5 moves w from 0.8 to 0.584 and b from −0.2 to −0.344, and the loss falls from 0.534 to 0.397. A real network does exactly this, for millions of parameters at once.`, predict: {"type":"mcq","q":"The prediction (0.73) is above the target (0) and the input x = 1.5 is positive. After one gradient-descent step, will w go up or down?","options":["Up","Down","It stays the same","It depends on the learning rate's sign"],"answer":1,"explain":"∂L/∂w = (positive error signal) × (positive sigmoid slope) × (positive x) > 0, and gradient descent moves against the gradient, so w decreases.","expect":["w decreases and the loss falls: True"]} },
      { type: 'demo', title: 'Interactive: backpropagation, one step at a time', demo: 'backprop' },
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
      { type: 'text', title: 'Why it is efficient, and where it goes wrong', md: `**Efficiency.** Backprop visits each operation once going forward and once going backward, so computing the gradient of *all* parameters costs only a small constant times one forward pass: independent of the number of parameters. Numerical differentiation would need two forward passes **per parameter**. This is called **reverse-mode automatic differentiation**.

**Memory.** The forward pass must store intermediate activations for the backward pass, so training uses much more memory than inference. Tricks like *gradient checkpointing* recompute some activations to save memory.

**Vanishing and exploding gradients.** The backward pass multiplies by each layer's local derivatives and weight matrices. Many factors below 1 shrink the gradient towards zero (early layers stop learning); factors above 1 blow it up. Good activations, initialization, normalization, residual connections and gradient clipping keep it in range.

**Gradient checking.** When you write backward passes by hand, compare them with central-difference numerical gradients on a tiny network; relative errors around 1e-7 or below mean you got it right.` },
      { type: 'chart', title: "Gradient size by layer, with and without residual connections", after: `Without shortcuts, every layer multiplies the gradient by a weight matrix and a tanh derivative, here shrinking it about 4-fold per layer, so the signal reaching the first layer is about 7 × 10⁻³¹ of the original: those layers cannot learn. With the very same weights plus residual connections, each layer computes h + f(h), whose local derivative is the identity plus a small term. The gradient keeps a direct path back and arrives at the first layer at roughly its original size (3.1 times the output gradient). This is why ResNets and transformers can be hundreds of layers deep.`, chart: {"kind":"line","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],"yLog":true,"points":false,"series":[{"name":"plain 50-layer tanh network","y":[7.172e-31,2.793e-30,1.087e-29,4.271e-29,1.783e-28,6.911e-28,2.619e-27,1.044e-26,4.079e-26,1.747e-25,7.55e-25,3.057e-24,1.202e-23,5.437e-23,2.131e-22,7.958e-22,3.345e-21,1.261e-20,5.32e-20,1.988e-19,7.603e-19,3.103e-18,1.222e-17,4.93e-17,1.989e-16,7.81e-16,3.121e-15,1.271e-14,5.084e-14,1.981e-13,8.431e-13,3.4e-12,1.37e-11,5.256e-11,2.038e-10,8.022e-10,3.461e-9,1.412e-8,5.474e-8,2.278e-7,9.33e-7,0.000003739,0.00001514,0.00005864,0.0002322,0.0009146,0.003727,0.01501,0.06112,0.2517,1]},{"name":"same network with residual connections (h + f(h))","y":[3.078,2.98,2.913,2.831,2.762,2.715,2.648,2.581,2.491,2.43,2.357,2.28,2.214,2.17,2.136,2.064,2.011,1.979,1.966,1.924,1.873,1.824,1.772,1.727,1.708,1.684,1.645,1.608,1.567,1.537,1.494,1.457,1.419,1.386,1.359,1.329,1.316,1.295,1.268,1.241,1.211,1.186,1.158,1.148,1.126,1.104,1.085,1.065,1.038,1.018,1]}],"xLabel":"layer (0 = input; the gradient starts at layer 50)","yLabel":"gradient norm relative to the output (log scale)"} },
      { type: 'check', question: { type: 'tf', q: 'With softmax + cross-entropy (mean over a batch of n), the gradient of the logits is (P − Y_onehot)/n.', answer: true, explain: 'The softmax Jacobian and the log combine into this simple form.' } }
    ],
    when: `- You use backprop every time you train a neural network: automatically, through autograd.
- Write backward passes by hand only to learn, or for custom operations; always gradient-check them.
- Think about the backward pass when diagnosing vanishing/exploding gradients or memory use.`,
    pitfalls: `- Shape errors: derive formulas from shapes.
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
      { type: 'tf', q: 'A relative error of about 1e-2 between analytic and numerical gradients means the backward pass is correct.', answer: false, explain: 'Correct implementations typically reach 1e-7 or better in double precision; 1e-2 indicates a bug.' },
      {"type":"fill","q":"For p = σ(z) with p = 0.8, what is the local derivative ∂p/∂z? (decimal)","answer":["0.16",".16"],"explain":"∂p/∂z = p(1 − p) = 0.8 × 0.2 = 0.16."},
      {"type":"mcq","q":"Why do residual connections help gradients reach the early layers?","options":["They remove the non-linearity","Each block computes h + f(h), so its local derivative includes the identity and the gradient has a direct path back","They normalise the gradients","They reduce the number of parameters"],"answer":1,"explain":"The derivative of h + f(h) is I + ∂f/∂h: even if the second term is small, the identity carries the gradient through unchanged."}
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
        explain: `Two dense layers, one ReLU and the softmax-cross-entropy pair, each step backward applies one rule from the table. The same pattern, repeated, is backprop for any MLP; autograd frameworks simply record the forward operations and replay these local rules in reverse.`,
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
      { title: 'CS231n, Backpropagation, intuitions', url: 'https://cs231n.github.io/optimization-2/' },
      { title: '3Blue1Brown, Backpropagation calculus', url: 'https://www.3blue1brown.com/lessons/backpropagation-calculus' }
    ]
  },

  // =====================================================================================
  {
    id: 'autograd', branch: 'dl', title: 'Build a Tiny Autograd Engine', short: 'Autograd from Scratch', level: 3, minutes: 50, prereqs: ['backpropagation'],
    tags: ['autograd', 'micrograd', 'computational graph', 'topological sort', 'operator overloading'],
    summary: 'Recreate the heart of PyTorch in ~50 lines: Value objects that record how they were computed, local backward functions, and a topological sort that runs backprop automatically.',
    why: `When you call \`loss.backward()\` in PyTorch, an **autograd** engine walks backwards through every operation you performed and fills in \`.grad\` for every parameter. It feels like magic until you build one yourself, and then it is just bookkeeping plus the chain rule. This exercise (inspired by Andrej Karpathy's *micrograd*) removes the last bit of mystery from training neural networks and makes framework errors ("trying to backward through the graph a second time", leaf tensors, accumulated gradients) suddenly understandable.`,
    objectives: [
      'Represent scalars as Value objects that remember their parents and the operation that created them',
      'Write the local backward function for addition, multiplication, power, tanh and ReLU',
      'Run backpropagation via a topological ordering of the graph',
      'Explain gradient accumulation and why frameworks require zero_grad()'
    ],
    sections: [
      { type: 'text', title: 'The idea', md: `Wrap every number in an object that records **how it was made**:

- \`data\`: the value itself;
- \`grad\`: ∂(final output)/∂(this value), filled in by backward (starts at 0);
- \`_prev\`: the Value objects it was computed from;
- \`_backward\`: a small function that pushes this node's gradient to its parents using the **local derivative**.

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

Why **+=**? A value used in several places receives gradient from every path (the multivariable chain rule sums them). It also explains why PyTorch **accumulates** gradients across backward calls, and why every training step starts with \`optimizer.zero_grad()\`.` },
      { type: 'text', title: 'Running backward in the right order', md: `A node may only push its gradient to its parents once its own gradient is complete, i.e. after **every** node that uses it has been processed. A **topological sort** (children after parents) reversed gives exactly that order:

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

The only thing PyTorch adds conceptually is **tensors**: each node holds a whole array, so one node's backward handles millions of numbers in one optimized kernel. Same algorithm, enormously faster.` },
      { type: 'text', title: 'Going further: from scalars to real frameworks', md: `- **Tensors, not scalars**: real engines apply the same idea to whole arrays using vector–Jacobian products (see *Matrix Calculus*), so one graph node handles millions of numbers at once.
- **Reverse versus forward mode**: reverse mode (backpropagation) gives the gradient of one output with respect to every input in a single backward pass (ideal for a scalar loss). Forward mode carries derivatives along with the values and is efficient when there are few inputs.
- **Memory**: reverse mode must keep the forward activations until the backward pass, which is why long sequences and big batches run out of memory, and why activation checkpointing exists (see *Training at Scale*).
- **Dynamic and traced graphs**: PyTorch records the graph as the code runs; JAX traces a function once and transforms it (grad, jit, vmap).
- **Higher-order derivatives**: differentiating the backward pass again gives Hessian–vector products, used by some optimizers and in meta-learning.` },
      { type: 'text', title: "Memory: what the backward pass needs", md: `Every \`_backward\` closure refers to values computed in the forward pass: a multiplication needs both inputs, tanh needs its output, ReLU needs to know which inputs were positive. So the whole graph, with its intermediate values, stays in memory until backward has run. For a network those are the **activations** of every layer for every example in the batch, and in large models they take far more memory than the weights themselves.

Two standard ways to cut that memory:
- **Gradient checkpointing**: keep the activations of only some layers, and during the backward pass recompute each missing segment from the nearest checkpoint. Memory drops sharply, for the price of roughly one extra forward pass.
- **Not building a graph at all** when no gradients are needed, during evaluation and inference: \`torch.no_grad()\` (or \`torch.inference_mode()\`) tells the engine not to record operations, which saves both memory and time.` },
      { type: 'chart', title: "Activation memory: store everything or checkpoint", after: `Storing everything costs memory proportional to the depth. Keeping only every √L-th layer's activations and recomputing each segment during the backward pass needs about 2√L layers' worth: 63 instead of 1,000 for a 1,000-layer stack, for roughly one extra forward pass (about 30% more compute). Large-model training uses this trade constantly.`, chart: {"kind":"line","x":[10,20,50,100,200,500,1000],"xLog":true,"yLog":true,"series":[{"name":"store every layer's activations","y":[10,20,50,100,200,500,1000]},{"name":"checkpoint every √L layers","y":[6.3,8.9,14.1,20,28.3,44.7,63.2]}],"xLabel":"number of layers L (log scale)","yLabel":"activations held in memory (in layers, log scale)"} },
      { type: 'text', title: "Practical autograd: accumulation, no_grad and detach", md: `Three behaviours of real engines follow directly from the design you just built:

| Behaviour | Why it exists | The trap |
| --- | --- | --- |
| \`.grad\` **accumulates** (+=) | a value used in several places must add up its gradients | forgetting \`optimizer.zero_grad()\` adds this step's gradients to the previous step's |
| \`torch.no_grad()\` | evaluation needs no graph | forgetting it during validation wastes memory and can exhaust it |
| \`.detach()\` | treat a value as a constant: gradients stop there | detaching by accident (for example by converting to NumPy and back) silently cuts part of a model out of training |

Accumulation is also a feature. To train with batches of 256 on hardware that fits only 64, run four forward and backward passes of 64 and call the optimiser once: the gradients add up exactly as for a batch of 256 (see [Training Neural Networks](#/concept/training-loop)).` },
      { type: 'code', title: "Accumulation and detach in a tiny engine", code: `class Value:
    def __init__(self, data, _prev=()):
        self.data, self.grad, self._prev, self._backward = float(data), 0.0, _prev, lambda: None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def detach(self):
        return Value(self.data)                     # the same number, with no link to the graph

    def backward(self):
        order, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev:
                    build(p)
                order.append(v)
        build(self)
        for v in order:
            if v._prev:
                v.grad = 0.0                        # intermediate results start fresh; leaves keep accumulating
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

w, x = Value(3.0), Value(2.0)
loss = w * x + w                                    # dloss/dw = x + 1 = 3
loss.backward()
print("w.grad after one backward:", w.grad)
loss.backward()                                     # forgot to zero the gradient
print("w.grad after a second backward:", w.grad)
print("accumulated twice:", w.grad == 2 * 3.0)

w.grad = 0.0                                        # this is what zero_grad() does
loss2 = w.detach() * x + w                          # the first term no longer depends on w
loss2.backward()
print("w.grad with the first term detached:", w.grad)`, after: `Leaf gradients add up across backward calls, which is exactly what accumulation over micro-batches relies on, and exactly why training loops zero them before every step. After detaching, the first term is a constant as far as the engine knows, so only the "+ w" term still sends a gradient to w.`, predict: {"type":"mcq","q":"loss = w·x + w with w = 3 and x = 2, so dloss/dw = 3. You call loss.backward() twice without zeroing the gradient. What is w.grad?","options":["3","6","9","0"],"answer":1,"explain":"Leaf gradients are added (+=), so the second backward adds another 3 to the first.","expect":["accumulated twice: True"]} }
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
      { type: 'mcq', q: 'out = tanh(a) with out.data = 0.5. If out.grad = 2, what does a.grad receive?', options: ['2', '1.5', '0.75', '0.5'], answer: 1, explain: '(1 − 0.5²)·2 = 0.75·2 = 1.5.' },
      {"type":"mcq","q":"Why does training need more memory than inference with the same model?","options":["The weights are copied","The forward pass must keep intermediate activations for the backward pass","Gradients are stored as text","Inference always uses a smaller batch"],"answer":1,"explain":"Every local backward rule needs values from the forward pass, so the activations of every layer stay in memory until backward has run."},
      {"type":"tf","q":"Wrapping validation in torch.no_grad() changes the model's predictions.","answer":false,"explain":"It only stops the engine from recording a graph, which saves memory and time; the forward computation is identical."}
    ],
    problems: [
      {
        id: 'value-autograd', title: 'Your Own Autograd Value', difficulty: 'hard',
        prompt: `Complete the \`Value\` class so that expressions build a graph and \`backward()\` fills in every \`.grad\`.

Implement:
- \`__add__\`, \`__mul__\` (other may be a plain number: wrap it),
- \`__pow__(k)\` for a plain int/float exponent k,
- \`relu()\` and \`tanh()\`,
- \`backward()\`: topological sort, set this node's grad to 1.0, call every \`_backward\` in reverse order.

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
        hints: ['Each operation: compute `out`, define an inner `_backward()` that does `parent.grad += local * out.grad`, set `out._backward = _backward`, return out.', 'Use `+=` in every backward function: values can be used more than once.', 'In `backward`, build the order with a recursive DFS that appends a node after visiting its parents; then iterate `reversed(order)`.'],
        explain: `Closures make this elegant: each \`_backward\` remembers its own \`self\`, \`other\` and \`out\`. Tracking visited nodes by \`id(v)\` avoids relying on hashing. With these few operations you can already express and train a small MLP, exactly what micrograd demonstrates.`,
        tests: [
          { name: 'forward values', setup: 'def run():\n    return (Value(2.0) * Value(3.0) + Value(1.0)).data', call: 'run()', expect: '7.0' },
          { name: 'gradients of c = a*b + a**2', setup: 'def run():\n    a, b = Value(2.0), Value(-3.0)\n    c = a * b + a ** 2\n    c.backward()\n    return (c.data, a.grad, b.grad)', call: 'run()', expect: '(-2.0, 1.0, 2.0)' },
          { name: 'gradient accumulation (x used twice)', setup: 'def run():\n    x = Value(3.0)\n    y = x * x + x\n    y.backward()\n    return x.grad', call: 'run()', expect: '7.0' },
          { name: 'neuron with tanh', setup: 'def run():\n    x1, x2, w1, w2, b = Value(2.0), Value(0.0), Value(-3.0), Value(1.0), Value(6.8814)\n    out = (x1 * w1 + x2 * w2 + b).tanh()\n    out.backward()\n    return (round(out.data, 6), round(w1.grad, 6), round(w2.grad, 6), round(b.grad, 6), round(x1.grad, 6))', call: 'run()', hidden: true },
          { name: 'relu, division, subtraction vs numerical derivative', setup: 'def f(v):\n    a = Value(v)\n    return ((a * 2 - 1).relu() / (a ** 2 + 1) + 3 - a).data\ndef run():\n    a = Value(0.8)\n    out = (a * 2 - 1).relu() / (a ** 2 + 1) + 3 - a\n    out.backward()\n    num = (f(0.8 + 1e-6) - f(0.8 - 1e-6)) / 2e-6\n    return abs(a.grad - num) < 1e-6', call: 'run()', expect: 'True', hidden: true },
          { name: 'shared sub-expression', setup: 'def run():\n    a = Value(1.5)\n    b = a * a\n    c = b * b + b\n    c.backward()\n    return a.grad', call: 'run()', expect: '4 * 1.5 ** 3 + 2 * 1.5', hidden: true }
        ]
      },
      {
        id: 'local-backward-rules', title: 'Local Backward Rules for Arrays', difficulty: 'medium',
        prompt: `An autograd engine is a collection of small backward rules, each mapping the gradient of an operation's output to the gradients of its inputs. Implement them for NumPy arrays (g is the upstream gradient, shaped like the output):

1. \`mul_backward(a, b, g)\` → \`(g * b, g * a)\` for out = a * b (element-wise).
2. \`tanh_backward(out, g)\` → g · (1 − out²), where out = tanh(x) is the saved forward result.
3. \`matmul_backward(A, B, G)\` → \`(G @ Bᵀ, Aᵀ @ G)\` for out = A @ B.
4. \`sum_backward(shape, g)\` → an array of the given shape filled with the scalar g, for out = x.sum().
5. \`broadcast_add_backward(x_shape, g)\` → for out = x + b where a bias vector of shape (n,) was broadcast over the rows of a (m, n) array: return the gradient for the bias, \`g.sum(axis=0)\`, if \`x_shape\` has one dimension, otherwise g itself.`,
        starter: `
import numpy as np

def mul_backward(a, b, g):
    # TODO
    pass

def tanh_backward(out, g):
    # TODO
    pass

def matmul_backward(A, B, G):
    # TODO
    pass

def sum_backward(shape, g):
    # TODO
    pass

def broadcast_add_backward(x_shape, g):
    # TODO
    pass
`,
        solution: `
import numpy as np

def mul_backward(a, b, g):
    return g * b, g * a

def tanh_backward(out, g):
    return g * (1 - np.asarray(out) ** 2)

def matmul_backward(A, B, G):
    return G @ B.T, A.T @ G

def sum_backward(shape, g):
    return np.full(shape, float(g))

def broadcast_add_backward(x_shape, g):
    g = np.asarray(g)
    return g.sum(axis=0) if len(x_shape) == 1 else g
`,
        hints: ['Each rule is the chain rule for one operation: local derivative times upstream gradient.', 'A broadcast value was used once per row, so its gradient adds up the rows.', 'Check shapes: the gradient of an input always has that input\'s shape.'],
        explain: `Real frameworks implement exactly these rules (for hundreds of operations) and chain them in reverse topological order. Broadcasting is the classic source of shape bugs: whatever was broadcast in the forward pass must be summed in the backward pass.`,
        tests: [
          { call: 'mul_backward(np.array([2.0, 3.0]), np.array([4.0, 5.0]), np.array([1.0, 10.0]))', expect: '(np.array([4.0, 50.0]), np.array([2.0, 30.0]))' },
          { call: '[sum_backward((2, 2), 3.0), broadcast_add_backward((2,), np.array([[1.0, 2.0], [3.0, 4.0]]))]', expect: '[np.full((2, 2), 3.0), np.array([4.0, 6.0])]' },
          { name: 'tanh rule matches finite differences', setup: 'x = np.array([-1.0, 0.2, 2.0]); g = np.array([1.0, 2.0, -1.0])', code: 'num = (np.tanh(x + 1e-6) - np.tanh(x - 1e-6)) / 2e-6 * g\nassert np.allclose(tanh_backward(np.tanh(x), g), num, atol=1e-6)', hidden: false },
          { name: 'matmul rule vs finite differences', setup: 'rng = np.random.default_rng(0)\nA = rng.normal(size=(3, 4)); B = rng.normal(size=(4, 2)); G = rng.normal(size=(3, 2))', code: 'dA, dB = matmul_backward(A, B, G)\nnum = np.zeros_like(A)\nfor i in range(A.size):\n    d = np.zeros_like(A); d.flat[i] = 1e-6\n    num.flat[i] = (np.sum(((A + d) @ B) * G) - np.sum(((A - d) @ B) * G)) / 2e-6\nassert np.allclose(dA, num, atol=1e-6) and dB.shape == B.shape', hidden: true },
          { name: 'random values vs reference', setup: 'rng = np.random.default_rng(1)\na = rng.normal(size=5); b = rng.normal(size=5); g = rng.normal(size=5)', call: '[mul_backward(a, b, g), tanh_backward(np.tanh(a), g)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Andrej Karpathy, micrograd', url: 'https://github.com/karpathy/micrograd', note: 'the inspiration for this lesson' },
      { title: 'Karpathy, Neural Networks: Zero to Hero (video course)', url: 'https://karpathy.ai/zero-to-hero.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'training-loop', branch: 'dl', title: 'Training Neural Networks: The Loop', short: 'The Training Loop', level: 2, minutes: 45, prereqs: ['backpropagation', 'gradient-descent'],
    tags: ['training loop', 'epochs', 'batches', 'early stopping', 'debugging', 'learning curves'],
    summary: 'Putting it together: the canonical training loop, reading loss curves, sanity checks, early stopping and a practical debugging checklist.',
    why: `Every neural network (from a toy classifier to a frontier LLM) is trained by the same loop: forward, loss, backward, update, repeat. What separates a working model from a frustrating week is knowing how to **watch** that loop: reading training and validation curves, catching NaNs and divergence early, verifying the pipeline on a tiny batch, and stopping at the right time. These habits transfer directly to PyTorch and to fine-tuning large models.`,
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
    evaluate on the validation set               # no gradient updates here
    log training & validation loss / metrics
    save a checkpoint if validation improved; stop early if it hasn't for a while
\`\`\`

That's it, every framework's training script is an elaboration of this. In PyTorch the update block reads \`optimizer.zero_grad(); loss.backward(); optimizer.step()\`.` },
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
      { type: 'chart', title: "Training and validation loss of a small network that overfits", md: `A 64-unit network trained with full-batch gradient descent on only 60 noisy examples of the two-moons data:`, after: `Both losses fall together at first. Around epoch 22 the validation loss bottoms out at 0.38; after that the training loss keeps falling (to 0.05) while the validation loss climbs back to 1.01: the network is memorising the noise of its 60 examples. Early stopping would keep the weights from the marked epoch.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075,1100,1125,1150,1175,1200,1225,1250,1275,1300,1325,1350,1375,1400,1425,1450,1475,1500,1525,1550,1575,1600,1625,1650,1675,1700,1725,1750,1775,1800,1825,1850,1875,1900,1925,1950,1975,2000,2025,2050,2075,2100,2125,2150,2175,2200,2225,2250,2275,2300,2325,2350,2375,2400,2425,2450,2475,2500,2525,2550,2575,2600,2625,2650,2675,2700,2725,2750,2775,2800,2825,2850,2875,2900,2925,2950,2975,3000],"xLog":true,"points":false,"series":[{"name":"training loss (60 examples)","y":[0.5291,0.3444,0.3161,0.3085,0.3054,0.3033,0.3015,0.2999,0.2983,0.2969,0.2955,0.2942,0.2929,0.2918,0.2906,0.2895,0.2885,0.2875,0.2865,0.2856,0.2814,0.2668,0.2569,0.2483,0.2406,0.2335,0.2271,0.222,0.2174,0.2131,0.209,0.2049,0.201,0.1972,0.1933,0.1895,0.1856,0.1818,0.1785,0.1751,0.1719,0.1688,0.1659,0.1631,0.1603,0.1576,0.1551,0.1526,0.1501,0.1477,0.1454,0.1432,0.1411,0.1389,0.1369,0.135,0.1329,0.13,0.1273,0.125,0.123,0.1213,0.1196,0.1179,0.1161,0.1146,0.1131,0.1116,0.1102,0.1089,0.1075,0.1062,0.1049,0.1036,0.1024,0.1014,0.1001,0.099,0.0978,0.0966,0.0955,0.0944,0.0933,0.0922,0.0913,0.0904,0.0894,0.0884,0.0873,0.0864,0.0855,0.0843,0.0835,0.0827,0.0821,0.0811,0.0803,0.0793,0.0785,0.0778,0.0771,0.0763,0.0757,0.075,0.074,0.073,0.0722,0.0717,0.0711,0.0704,0.07,0.0693,0.0687,0.0681,0.0676,0.0668,0.066,0.0656,0.0648,0.0643,0.0636,0.0632,0.0626,0.0621,0.0617,0.0612,0.0609,0.0602,0.0595,0.0589,0.0585,0.0581,0.0575,0.0572,0.0569,0.0563,0.0561,0.0557,0.0552,0.0547]},{"name":"validation loss (1,000 examples)","y":[0.7927,0.4953,0.4245,0.3997,0.3905,0.3868,0.3851,0.384,0.3833,0.3827,0.3822,0.3817,0.3813,0.381,0.3807,0.3805,0.3803,0.3802,0.38,0.38,0.38,0.3851,0.3909,0.3962,0.4003,0.4047,0.4088,0.4136,0.4176,0.4214,0.4249,0.4279,0.4306,0.4329,0.4351,0.4372,0.4392,0.4412,0.4444,0.448,0.452,0.456,0.4609,0.4664,0.4719,0.4769,0.4826,0.4887,0.4956,0.5022,0.5096,0.5158,0.5229,0.5296,0.538,0.5447,0.5523,0.5571,0.5628,0.5694,0.5766,0.5848,0.5918,0.5992,0.6066,0.6143,0.62,0.6264,0.6334,0.64,0.6471,0.6538,0.6601,0.6667,0.6746,0.6842,0.6887,0.696,0.7031,0.7088,0.7156,0.7225,0.7286,0.7332,0.7414,0.7482,0.7562,0.7621,0.768,0.7733,0.7786,0.7829,0.7904,0.7961,0.8048,0.8065,0.8144,0.8176,0.8225,0.8294,0.8352,0.8392,0.8479,0.8496,0.8519,0.8586,0.8621,0.8693,0.8771,0.8798,0.8844,0.8937,0.8944,0.9001,0.9026,0.9104,0.9144,0.9196,0.9233,0.9232,0.9298,0.9329,0.9362,0.9445,0.948,0.9512,0.955,0.9656,0.9659,0.9737,0.9716,0.9825,0.985,0.9872,0.9992,0.9952,1.009,1.0025,1.0155,1.0148]}],"xLabel":"epoch (log scale)","yLabel":"cross-entropy loss","vlines":[{"x":22,"label":"best validation: epoch 22"}]} },
      { type: 'check', question: { type: 'mcq', q: 'A 10-class classifier starts training with a loss around 2.30. Is that suspicious?', options: ['Yes: it should start near 0', 'No: ln(10) ≈ 2.30 is exactly the loss of uniform random guessing', 'Yes: it should be negative', 'No: the starting loss is meaningless'], answer: 1, explain: 'With small random weights the model predicts ~1/10 for each class: −ln(0.1) ≈ 2.30. A much higher initial loss suggests a bug or bad initialization.' } },
      { type: 'text', title: 'Sanity checks before long runs', md: `1. **Check the initial loss**: for K balanced classes it should be about ln K.
2. **Overfit one small batch**: train on, say, 20 examples. A correct pipeline drives the loss to ~0 quickly. If it can't, there is a bug (wrong labels, broken gradients, frozen parameters).
3. **Look at the data** right before it enters the model (after augmentation and preprocessing): mismatched images and labels are a classic bug.
4. **Monitor gradient norms**: all zeros means no learning; huge values signal explosion.
5. **Start simple**: a smaller model and a known-good learning rate; scale up once it works.
6. **Fix random seeds** for reproducibility while debugging.` },
      { type: 'text', title: 'Early stopping and checkpoints', md: `Keep track of the best validation loss. After each epoch:
- if it improved: save a **checkpoint** (the parameters, optimizer state and epoch) and reset a counter;
- otherwise increment the counter; when it reaches the **patience** (e.g. 5–10 epochs), stop and restore the best checkpoint.

Early stopping is free regularization: training as long as it helps, and not a step longer. Checkpoints also protect long runs against crashes: large-model training saves them regularly and resumes from the latest one.` },
      { type: 'text', title: "Batch size, gradient accumulation and the learning rate", md: `The batch size trades noise against throughput. Larger batches give smoother gradient estimates and use the hardware better, but each epoch contains fewer updates, and beyond some size bigger batches stop paying off. Smaller batches give noisier steps, which often generalise slightly better.

Batch size and learning rate interact. A common rule of thumb scales the learning rate with the batch size (the **linear scaling rule**: double the batch, double the rate), usually together with a warm-up period at the start of training. It holds only up to a point, and adaptive optimizers such as Adam are less sensitive to it.

When the batch you want does not fit in memory, use **gradient accumulation**: split it into micro-batches, run forward and backward on each, add up their gradients, and update once:

$$ effective batch = micro-batch size × accumulation steps × number of devices $$
:::where
micro-batch size — the examples processed in one forward and backward pass on one device
accumulation steps — how many micro-batches are added up before each optimiser step
number of devices — with data parallelism, each GPU processes its own micro-batches and the gradients are averaged across GPUs
:::

Because the gradient of an average is the average of the gradients, the accumulated update equals one big batch, provided each micro-batch's loss is divided by the number of accumulation steps. Layers that compute statistics over the batch, such as batch normalisation, are the exception: they only ever see a micro-batch.` },
      { type: 'code', title: "Gradient accumulation gives the same update", code: `import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(32, 5))
y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.normal(0, 0.1, 32)
w = rng.normal(size=5)

def grad_mse(Xb, yb, w):                             # gradient of the mean squared error
    return 2 * Xb.T @ (Xb @ w - yb) / len(yb)

full = grad_mse(X, y, w)                             # one batch of 32
accumulated = np.zeros(5)
for k in range(4):                                   # four micro-batches of 8
    part = slice(8 * k, 8 * (k + 1))
    accumulated += grad_mse(X[part], y[part], w) / 4  # divide by the number of accumulation steps
print("full batch: ", full.round(4))
print("accumulated:", accumulated.round(4))
print("same as one batch of 32:", np.allclose(full, accumulated))`, predict: {"type":"mcq","q":"The gradients of 4 micro-batches of 8 examples are each divided by 4 and added up. They match the gradient of one batch of how many examples?","options":["8","4","32","They never match exactly"],"answer":2,"explain":"The mean over 32 examples equals the average of the four means over 8, so accumulation reproduces the full-batch gradient (up to floating-point rounding).","expect":["same as one batch of 32: True"]} },
      { type: 'check', question: { type: 'tf', q: 'If a model cannot overfit a single small batch of 20 examples, the pipeline probably has a bug.', answer: true, explain: 'Any reasonable network can memorise 20 examples; failing to do so points to broken data, labels or gradients.' } },
      { type: 'text', title: 'Going further: a debugging checklist', md: `1. **Check the loss at initialization**: with k balanced classes it should be close to ln k. Much higher means the initialization or the loss is wrong.
2. **Overfit one small batch** to nearly zero loss. If the model cannot memorize a handful of examples, there is a bug.
3. **Find a workable learning rate** with a short sweep or a range test: increase it exponentially every step until the loss blows up, then use a value about ten times smaller.
4. **Watch gradient norms and update sizes**: NaNs, exploding norms, or updates that are tiny relative to the weights all signal problems.
5. **Look at the data right before the model**: decode a batch back into images or text and check the labels (data bugs are the most common bugs).
6. **Change one thing at a time** and log every run (seed, configuration, metrics); always compare with a simple baseline.
7. **Unstable training?** Lower the learning rate, add warmup, clip gradients, and check normalization and mixed-precision settings.` },
      { type: 'chart', title: "A learning-rate range test", md: `The same network trained for just 40 steps at each of 24 learning rates:`, after: `Too small a rate barely moves the loss; it then falls faster and faster as the rate grows, until the steps start to overshoot and the loss turns back up. A good choice is somewhat below the bottom of the valley (often a few times smaller than the rate where the loss starts to rise), which leaves a safety margin for the rest of training.`, chart: {"kind":"line","x":[0.001,0.00158,0.00251,0.00398,0.00631,0.01,0.0158,0.0251,0.0398,0.0631,0.1,0.158,0.251,0.398,0.631,1,1.58,2.51,3.98,6.31,10,15.8,25.1,39.8],"xLog":true,"series":[{"name":"training loss after 40 steps","y":[0.6343,0.585,0.5311,0.4795,0.4344,0.3963,0.3651,0.3412,0.3242,0.3125,0.3035,0.2953,0.2866,0.2771,0.2656,0.2491,0.2266,0.1937,0.2011,0.236,0.3249,0.3758,1.3439,0.558]}],"xLabel":"learning rate (log scale)","yLabel":"loss after 40 steps"} }
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
      { type: 'tf', q: 'You should apply gradient updates while evaluating on the validation set.', answer: false, explain: 'Validation must never change the parameters.' },
      {"type":"fill","q":"Each GPU processes micro-batches of 16, gradients are accumulated over 8 steps, and there are 4 GPUs. What is the effective batch size?","answer":["512"],"explain":"16 × 8 × 4 = 512 examples contribute to each optimiser step."},
      {"type":"mcq","q":"In a learning-rate range test, where should you pick the learning rate?","options":["At the largest rate tried","Somewhat below the rate where the loss is lowest, well before it starts to rise","At the smallest rate tried","Exactly where the loss explodes"],"answer":1,"explain":"The bottom of the curve is already close to instability; a rate a few times smaller leaves a safety margin for the rest of training."}
    ],
    problems: [
      {
        id: 'early-stopping-epoch', title: 'Early Stopping with Patience', difficulty: 'easy',
        prompt: `Implement \`early_stopping(val_losses, patience)\` → \`(best_epoch, stop_epoch)\`:

- Walk through the validation losses epoch by epoch (epochs are 0-based indices).
- Track the best (lowest) loss so far: it must be **strictly lower** to count as an improvement.
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
        explain: `This is the logic inside every EarlyStopping callback. In a real loop you would also restore the parameters saved at \`best_epoch\`: the stop epoch is merely when you noticed things weren't improving.`,
        tests: [
          { call: 'early_stopping([1.0, 0.8, 0.7, 0.75, 0.72, 0.71, 0.9], 3)', expect: '(2, 5)' },
          { name: 'never stops early', call: 'early_stopping([3.0, 2.0, 1.5, 1.2], 2)', expect: '(3, 3)' },
          { name: 'equal loss is not an improvement', call: 'early_stopping([1.0, 1.0, 1.0], 2)', expect: '(0, 2)', hidden: true },
          { name: 'patience 1', call: 'early_stopping([0.5, 0.4, 0.45, 0.3], 1)', expect: '(1, 2)', hidden: true }
        ]
      },
      {
        id: 'train-xor', title: 'Train a Network to Learn XOR', difficulty: 'medium',
        prompt: `The perceptron couldn't learn XOR: a hidden layer can. The starter gives you \`init_params\`, \`forward\` and \`backward\` for a 2-8-1 network (tanh hidden layer, sigmoid output, binary cross-entropy).

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
      { title: 'Andrej Karpathy, A Recipe for Training Neural Networks', url: 'https://karpathy.github.io/2019/04/25/recipe/', note: 'the classic practical guide' },
      { title: "Google Research, Deep Learning Tuning Playbook", url: "https://github.com/google-research/tuning_playbook" }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
