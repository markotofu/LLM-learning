/* ML Atlas – extra quiz questions for the Deep Learning lessons. They appear only in custom quizzes (#/quiz), never in a
   lesson's own quiz or the review queue. Keyed by lesson id; the format is the lesson quiz format (docs/CONTENT_GUIDE.md). */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  MLA.data.quizBank = MLA.data.quizBank || {};
  Object.assign(MLA.data.quizBank, {
    'perceptron': [
      { type: 'mcq', q: "A neuron computes z = w·x + b with w = [1, 2], b = −1 and x = [3, 1]. What is z?",
        options: ["4", "5", "6", "3"], answer: 0,
        explain: "1 · 3 + 2 · 1 − 1 = 4. The activation function is then applied to this score." },
      { type: 'mcq', q: "A perceptron with w = [1, 1] outputs 1 when w·x + b > 0. Which bias makes it compute AND on inputs of 0 and 1?",
        options: ["−1.5", "−0.5", "0.5", "1.5"], answer: 0,
        explain: "Only (1, 1) gives 1 + 1 − 1.5 = 0.5 > 0. With a bias of −0.5 the same neuron computes OR instead." },
      { type: 'tf', q: "Two classes are linearly separable when a single line, plane or hyperplane puts all of one class on one side and all of the other class on the other.", answer: true,
        explain: "This is exactly when the perceptron learning rule is guaranteed to converge. XOR is the classic example of data that is not linearly separable." },
      { type: 'mcq', q: "A perceptron misclassifies a positive example x (label +1). With learning rate 1, how does the rule update w?",
        options: ["w ← w + x", "w ← w − x", "w ← 0", "w ← w · x"], answer: 0,
        explain: "Adding x to w raises the score w·x for that example the next time it is seen. A misclassified negative example is subtracted instead." }
    ],
    'activation-functions': [
      { type: 'mcq', q: "What is the range of tanh?",
        options: ["(−1, 1)", "(0, 1)", "[0, ∞)", "All real numbers"], answer: 0,
        explain: "tanh squashes any input into (−1, 1) and is centred at 0, unlike the sigmoid, whose outputs lie in (0, 1)." },
      { type: 'mcq', q: "A Leaky ReLU has slope 0.01 for negative inputs. What is its output for x = −5?",
        options: ["−0.05", "0", "−5", "0.05"], answer: 0,
        explain: "Negative inputs are multiplied by 0.01, so −5 becomes −0.05. The small slope keeps a gradient flowing where a plain ReLU would give 0." },
      { type: 'tf', q: "Computing softmax as exp(z − max z) / Σ exp(z − max z) gives the same probabilities as the plain formula while avoiding overflow.", answer: true,
        explain: "Subtracting the same constant from every logit cancels in the ratio, and the largest exponent becomes e⁰ = 1, so nothing overflows." },
      { type: 'mcq', q: "Which output layer and loss suit predicting a house price?",
        options: ["A linear output (no activation) with MSE", "A sigmoid with binary cross-entropy", "A softmax with cross-entropy", "A step function with hinge loss"], answer: 0,
        explain: "A price is an unbounded number, so the last layer should output any real value. Sigmoid and softmax outputs are probabilities." }
    ],
    'mlp': [
      { type: 'mcq', q: "A layer's output has shape (64, 128) for a batch of 64 examples. What does 128 stand for?",
        options: ["The number of units (neurons) in the layer", "The batch size", "The number of input features", "The number of layers"], answer: 0,
        explain: "A dense layer turns each example's inputs into one value per unit, so the second dimension is the layer's width." },
      { type: 'tf', q: "An MLP with no hidden layers and a sigmoid output is the same model as logistic regression.", answer: true,
        explain: "It computes σ(w·x + b), which is exactly logistic regression. Hidden layers with non-linear activations are what add flexibility." },
      { type: 'mcq', q: "A binary classifier ends in a single unit. Which output activation and loss belong together?",
        options: ["A sigmoid with binary cross-entropy", "A softmax over that one unit with MSE", "A ReLU with cross-entropy", "No activation with accuracy as the loss"], answer: 0,
        explain: "The sigmoid turns the score into the probability of the positive class. A softmax over one unit always outputs 1. PyTorch's BCEWithLogitsLoss combines sigmoid and loss stably." },
      { type: 'mcq', q: "What does depth (more layers) offer compared with one very wide hidden layer?",
        options: ["Features built from features, which often represent complex functions with far fewer units", "Nothing at all", "No need for training data", "A guaranteed global minimum"], answer: 0,
        explain: "Early layers learn simple patterns and later layers combine them. A single hidden layer can approximate any function in principle, but may need enormously many units." }
    ],
    'backpropagation': [
      { type: 'mcq', q: "For z = x · y with x = 3 and y = 4, the upstream gradient ∂L/∂z is 2. What is ∂L/∂x?",
        options: ["8", "6", "2", "24"], answer: 0,
        explain: "The local derivative of a product with respect to x is the other factor, y = 4, so ∂L/∂x = 2 · 4 = 8. Likewise ∂L/∂y = 2 · 3 = 6." },
      { type: 'mcq', q: "For softmax followed by cross-entropy, what is the gradient with respect to the logits?",
        options: ["p − y: the predicted probabilities minus the one-hot target", "The square of y − p", "log p", "The softmax Jacobian alone"], answer: 0,
        explain: "The two Jacobians combine into this simple difference, which is one reason the pair is fused in frameworks." },
      { type: 'tf', q: "Gradient checking compares backprop's gradients with finite-difference estimates, and it is used for debugging, not for training.", answer: true,
        explain: "Numerical gradients need extra forward passes for every parameter, which is far too slow for training but fine for checking a few entries." },
      { type: 'mcq', q: "How does the gradient flow back through an addition node z = x + y?",
        options: ["Both x and y receive the upstream gradient unchanged", "It is split in half between them", "x gets all of it and y gets nothing", "It is multiplied by z"], answer: 0,
        explain: "∂z/∂x = ∂z/∂y = 1, so addition passes the gradient through to both inputs. That is why residual connections carry gradients so well." }
    ],
    'autograd': [
      { type: 'mcq', q: "For out = a ** 3 with a.data = 2 and out.grad = 1, what does the power node add to a.grad?",
        options: ["12", "8", "6", "3"], answer: 0,
        explain: "The local derivative is 3a² = 12, multiplied by out.grad = 1." },
      { type: 'mcq', q: "d = a + a with a = Value(3). After d.backward(), what is a.grad?",
        options: ["2", "1", "3", "6"], answer: 0,
        explain: "a is used twice, and each use passes back a gradient of 1. Accumulating with += gives 2; overwriting with = would wrongly give 1." },
      { type: 'tf', q: "Before backpropagation starts, the gradient of the final output with respect to itself is set to 1.", answer: true,
        explain: "∂L/∂L = 1 is where the chain rule starts. Every other gradient is built from it by multiplying local derivatives along the paths." },
      { type: 'mcq', q: "For out = relu(a) with a.data = −2 and out.grad = 5, what gradient does a receive?",
        options: ["0", "5", "−10", "−2"], answer: 0,
        explain: "ReLU's slope is 0 for negative inputs, so no gradient flows back to a." }
    ],
    'training-loop': [
      { type: 'mcq', q: "As a sanity check, you train on one small batch for many steps. What should happen?",
        options: ["The loss falls close to zero, showing that the model and loop can fit data", "The loss stays constant", "Validation accuracy reaches 100%", "The loss rises"], answer: 0,
        explain: "A working setup can memorise a handful of examples. If it cannot, look for a bug in the data, the loss or the update before training for real." },
      { type: 'mcq', q: "What does early stopping with a patience of 5 do?",
        options: ["Stops after 5 evaluations in a row without a new best validation loss, and keeps the best checkpoint", "Stops after exactly 5 epochs", "Waits 5 seconds between steps", "Lowers the learning rate 5 times"], answer: 0,
        explain: "Patience allows for noisy validation curves. Restoring the best checkpoint means the extra epochs cost nothing." },
      { type: 'tf', q: "If a PyTorch loop never calls `optimizer.zero_grad()`, the gradients of earlier steps keep adding up.", answer: true,
        explain: "PyTorch accumulates into .grad, so each update would use the sum of all previous gradients, and training would behave erratically." },
      { type: 'mcq', q: "Training and validation loss both stay high and nearly flat from the start. What is a likely cause?",
        options: ["A learning rate far too small, a model too small, or a bug such as an unused gradient", "Overfitting", "A validation set that is too large", "Too much training data"], answer: 0,
        explain: "Overfitting needs the training loss to fall. A flat training loss means the model is not learning at all." }
    ],
    'optimizers': [
      { type: 'mcq', q: "What does Adam's second-moment estimate v do?",
        options: ["Scales each parameter's step by the running average of its squared gradients", "Adds momentum", "Clips the gradients", "Decays the weights"], answer: 0,
        explain: "Dividing by √v gives each parameter its own effective learning rate: parameters with large, noisy gradients take smaller steps." },
      { type: 'mcq', q: "During warmup, how does the learning rate change?",
        options: ["It rises from near 0 to its peak over the first steps", "It falls from its peak to 0", "It stays constant", "It oscillates"], answer: 0,
        explain: "Early on, the weights are random and Adam's statistics are noisy, so large steps can destabilise training. A cosine schedule then lowers the rate after the peak." },
      { type: 'fill', q: "A gradient has norm 10 and is clipped to a maximum norm of 1. By what factor is it multiplied?", answer: ["0.1", "1/10"],
        explain: "Clipping by norm rescales the whole gradient by max_norm / norm = 1 / 10, keeping its direction." },
      { type: 'tf', q: "Momentum builds up speed along directions where successive gradients agree, and damps directions where they keep flipping sign.", answer: true,
        explain: "The running average adds consistent components and cancels alternating ones, which is what helps in long, narrow ravines." }
    ],
    'normalization': [
      { type: 'mcq', q: "Inverted dropout with p = 0.5 drops half of the activations during training. What happens to the kept ones?",
        options: ["They are multiplied by 2, so the expected activation stays the same", "Nothing", "They are halved", "They are set to 1"], answer: 0,
        explain: "Scaling by 1 / (1 − p) at training time means the network can be used unchanged at evaluation time, when nothing is dropped." },
      { type: 'mcq', q: "Batch normalization computes its mean and variance over…",
        options: ["Each feature, across the examples in the batch", "Each example, across its features", "The whole dataset at every step", "The weights"], answer: 0,
        explain: "That ties each example's output to the rest of its batch. LayerNorm instead normalizes each example on its own, across its features." },
      { type: 'tf', q: "He initialization uses twice the variance of Xavier initialization (2 / fan_in instead of 1 / fan_in) to suit ReLU layers.", answer: true,
        explain: "ReLU zeroes about half of its inputs, which halves the signal's variance at each layer. Doubling the weight variance keeps the scale steady through depth." },
      { type: 'mcq', q: "Why do BatchNorm and LayerNorm include a learnable scale γ and shift β?",
        options: ["So the network can adjust, or even undo, the normalization where that helps", "To normalize faster", "To replace the layer's weights", "To store the batch statistics"], answer: 0,
        explain: "Forcing every feature to mean 0 and variance 1 could limit what a layer can represent. γ and β give that freedom back." }
    ],
    'pytorch': [
      { type: 'mcq', q: "How do you move a model and a batch to the GPU in PyTorch?",
        options: ["`model.to('cuda')` and `x = x.to('cuda')`", "`model.cuda = True`", "`torch.gpu(model)`", "They move there automatically"], answer: 0,
        explain: "An operation needs all of its inputs on the same device. Note that `x.to(...)` returns a new tensor, so assign the result." },
      { type: 'mcq', q: "What does `x.view(4, -1)` do to a tensor with 12 elements?",
        options: ["Reshapes it to (4, 3), with the −1 worked out", "Raises an error", "Gives shape (4, 12)", "Drops 4 elements"], answer: 0,
        explain: "One dimension may be −1, and PyTorch fills in whatever size makes the element count match: 12 / 4 = 3." },
      { type: 'tf', q: "A tensor created with requires_grad=True records the operations applied to it, so backward can later compute gradients.", answer: true,
        explain: "PyTorch builds the computation graph as the forward pass runs, then walks it in reverse when you call backward." },
      { type: 'mcq', q: "Which PyTorch loss expects raw logits and integer class labels for multi-class classification?",
        options: ["`nn.CrossEntropyLoss`", "`nn.MSELoss`", "`nn.BCELoss`", "`nn.NLLLoss` applied to probabilities"], answer: 0,
        explain: "CrossEntropyLoss applies log-softmax itself, so the model's last layer should output logits, not probabilities." }
    ],
    'cnn': [
      { type: 'fill', q: "How much padding keeps a 3 × 3 convolution with stride 1 from shrinking its input?", answer: ["1", "one"],
        explain: "Output size = (n + 2p − k) / s + 1. With k = 3, s = 1 and p = 1 this is n, which is why 3 × 3 layers usually use padding 1." },
      { type: 'mcq', q: "The 3 × 3 patch [[1, 0, 0], [0, 1, 0], [0, 0, 1]] meets a 3 × 3 filter of all ones. What value does the convolution output there?",
        options: ["3", "9", "1", "0"], answer: 0,
        explain: "A CNN's convolution multiplies the patch and the filter elementwise and sums the products: the three ones give 3." },
      { type: 'mcq', q: "What does weight sharing in a convolutional layer mean?",
        options: ["The same filter weights are applied at every position of the image", "Every layer uses the same weights", "Weights are copied between networks", "All channels share one bias"], answer: 0,
        explain: "A filter detects its pattern wherever the pattern appears, and the layer needs only the filter's few weights." },
      { type: 'tf', q: "A max-pooling layer has no learnable parameters.", answer: true,
        explain: "It just keeps the largest value in each window. It shrinks the feature map and makes it a little tolerant of small shifts." }
    ],
    'rnn': [
      { type: 'mcq', q: "What does the forget gate of an LSTM do?",
        options: ["Decides how much of the previous cell state to keep", "Chooses the output token", "Resets the weights", "Adds noise to the input"], answer: 0,
        explain: "Values near 1 keep the memory and values near 0 erase it. The input gate decides what new information is written." },
      { type: 'tf', q: "An RNN uses the same weights at every time step.", answer: true,
        explain: "Sharing weights across time lets it handle sequences of any length with a fixed number of parameters. It is also why gradients get multiplied by the same matrix again and again." },
      { type: 'mcq', q: "During backpropagation through time, the gradient is repeatedly multiplied by roughly the same matrix. If its largest singular value is below 1, what happens over many steps?",
        options: ["The gradient shrinks toward zero", "The gradient explodes", "The gradient stays the same", "The gradient changes sign every step"], answer: 0,
        explain: "Repeated multiplication by something smaller than 1 shrinks the gradient exponentially, so early steps barely learn. Above 1 it explodes instead." },
      { type: 'mcq', q: "What does truncated backpropagation through time do?",
        options: ["Backpropagates through only the most recent k steps of a long sequence", "Removes the hidden state", "Clips the gradients", "Trains only on short sentences"], answer: 0,
        explain: "It saves memory and time on long sequences. The hidden state still carries forward, but no gradient flows back further than k steps." }
    ],
    'embeddings': [
      { type: 'mcq', q: "An embedding layer returns row i of its matrix E for token i. Which matrix product gives the same result?",
        options: ["The one-hot vector for token i times E", "E times Eᵀ", "E times the number i", "The sum of all rows of E"], answer: 0,
        explain: "A one-hot row vector selects a single row of E. The lookup gives the same result without all the multiplications by zero." },
      { type: 'tf', q: "In a language model, the token embeddings are learned by backpropagation together with the rest of the network.", answer: true,
        explain: "The embedding matrix is just another set of parameters, updated by the same gradients as every other layer." },
      { type: 'mcq', q: "In the classic word-vector analogy, which word is closest to king − man + woman?",
        options: ["queen", "king", "man", "prince"], answer: 0,
        explain: "The difference woman − man roughly captures a gender direction, and adding it to king lands near queen. The input words are usually excluded from the search." },
      { type: 'mcq', q: "An embedding table has 50,000 rows of dimension 256. A batch holds 8 sequences of 20 tokens. What is the output shape?",
        options: ["(8, 20, 256)", "(8, 256)", "(50000, 256)", "(8, 20)"], answer: 0,
        explain: "Each of the 8 × 20 token ids is replaced by its 256-dimensional row." }
    ],
    'seq2seq': [
      { type: 'mcq', q: "What is teacher forcing?",
        options: ["Feeding the decoder the true previous token during training, instead of its own prediction", "Training with a larger teacher model", "Forcing the output to a fixed length", "Freezing the encoder"], answer: 0,
        explain: "It makes training fast and stable, but at inference the model sees its own predictions, which causes exposure bias." },
      { type: 'mcq', q: "The attention scores for one decoder step are [1, 1, 1] over three encoder states. What are the attention weights?",
        options: ["1/3 each, so the context vector is the average of the encoder states", "[1, 1, 1]", "[1, 0, 0]", "They cannot be computed"], answer: 0,
        explain: "Softmax of equal scores is uniform. With unequal scores, the context vector leans toward the states with higher scores." },
      { type: 'tf', q: "An encoder–decoder without attention has to squeeze the whole source sentence into one fixed-size vector.", answer: true,
        explain: "This bottleneck made long sentences translate badly, which is what attention was introduced to fix." },
      { type: 'mcq', q: "How does additive (Bahdanau) attention score an encoder state?",
        options: ["It passes the decoder state and the encoder state through a small feed-forward network", "It takes their dot product only", "It counts the shared words", "It concatenates all the encoder states"], answer: 0,
        explain: "The score is vᵀ tanh(W s + U h). Dot-product attention, later used in transformers, replaces this network with a dot product." }
    ],
    'distillation-pruning': [
      { type: 'mcq', q: "Softmax with temperature T divides the logits by T. What happens to the probabilities as T grows large?",
        options: ["They approach a uniform distribution", "They approach a one-hot distribution", "They become negative", "Nothing changes"], answer: 0,
        explain: "Large T shrinks the differences between logits, which reveals how the teacher ranks the wrong classes. The top class stays the same." },
      { type: 'mcq', q: "How are small language models often distilled from a large one?",
        options: ["The student trains on text the teacher generates, or on the teacher's next-token probabilities", "The teacher's weights are copied at random", "The teacher's vocabulary is pruned", "Only on hard labels written by people"], answer: 0,
        explain: "Generated data transfers the teacher's behaviour, and token-level probabilities transfer its full distribution at every position." },
      { type: 'tf', q: "After pruning, models are usually fine-tuned for a while to recover lost accuracy.", answer: true,
        explain: "The remaining weights adjust to make up for the removed ones. Iterative pruning alternates small pruning steps with fine-tuning." },
      { type: 'fill', q: "The teacher's probabilities are [0.7, 0.2, 0.1], and the student's are exactly the same. What is the KL divergence from teacher to student?", answer: "0",
        explain: "KL divergence is 0 exactly when the two distributions are equal, so the distillation term has nothing left to teach." }
    ],
    'self-supervised': [
      { type: 'mcq', q: "What does lowering the temperature in the NT-Xent contrastive loss do?",
        options: ["Sharpens the softmax, so the most similar negatives get more weight", "Removes the negatives", "Makes all similarities equal", "Speeds up the augmentations"], answer: 0,
        explain: "Dividing similarities by a small temperature exaggerates their differences, which focuses the loss on hard negatives. Too low a value makes training unstable." },
      { type: 'tf', q: "In SimCLR the augmentations matter a great deal, because they decide which differences the representation learns to ignore.", answer: true,
        explain: "Two augmented views of one image must map close together, so the features become insensitive to cropping, colour changes and similar edits." },
      { type: 'mcq', q: "How do BERT and masked autoencoders create their training signal?",
        options: ["They hide parts of the input and predict them from the rest", "They use human labels", "They predict each image's class", "They compare two unrelated images"], answer: 0,
        explain: "The hidden parts are the labels, so any amount of unlabelled text or images becomes training data." },
      { type: 'mcq', q: "What is a k-NN evaluation of self-supervised features?",
        options: ["Classifying each test image by the labels of its nearest training images in feature space, with nothing trained on top", "Training a deep network on the features", "Counting clusters", "Measuring the training loss"], answer: 0,
        explain: "Like a linear probe, it checks how much class information the frozen features already hold, without fine-tuning the encoder." }
    ]
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
