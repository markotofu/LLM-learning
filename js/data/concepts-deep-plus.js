/* ML Atlas – content: deep-learning additions (distillation & pruning, self-supervised learning, diffusion models, deep RL, ML security & privacy, LLM serving) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'distillation-pruning', branch: 'dl', title: 'Knowledge Distillation & Pruning', short: 'Distillation & Pruning', level: 3, minutes: 40, prereqs: ['information-theory', 'training-loop'],
    tags: ['knowledge distillation', 'teacher', 'student', 'soft targets', 'temperature', 'kl divergence', 'dark knowledge', 'sequence-level distillation', 'synthetic data', 'pruning', 'magnitude pruning', 'structured pruning', 'n:m sparsity', 'lottery ticket hypothesis', 'model compression'],
    summary: 'Making models smaller and cheaper while keeping what they know: training a small student on a large teacher\'s soft targets with temperature and KL divergence, distilling language models through generated data and token-level targets, and pruning — unstructured magnitude pruning, structured pruning, N:M sparsity and the lottery ticket hypothesis.',
    why: `The best models are often too big, slow or expensive to deploy everywhere. **Knowledge distillation** trains a small "student" to imitate a large "teacher", and it is one of the main reasons small modern models are so capable — many are trained on the outputs of much larger ones. **Pruning** removes weights or whole neurons the network doesn't need. Together with quantization (see *Quantization & Efficient Inference*), these are the standard tools of **model compression**.`,
    objectives: [
      'Explain soft targets, temperature and why they carry more information than hard labels',
      'Write the distillation loss and explain the T² factor',
      'Describe how language models are distilled with generated data and token-level targets',
      'Compare unstructured, structured and N:M pruning, and state the lottery ticket hypothesis'
    ],
    sections: [
      { type: 'text', title: 'Big teacher, small student', md: `A trained teacher knows more than its top-1 predictions reveal. For a picture of a cat it might give 90% cat, 7.4% tiger, 2.5% dog and 0.01% car — telling us cats resemble tigers far more than cars. That similarity structure, sometimes called **dark knowledge**, is lost when we train only on the hard label "cat".

**Knowledge distillation** (Hinton et al., 2015) trains a student on the teacher's full probability distribution, usually softened with a **temperature** T so that the small probabilities become visible:

$$ L = α · CE( y, softmax(z_s) ) + (1 − α) · T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) ) $$
:::where
z_s, z_t — the student's and the teacher's logits
y — the true (hard) label
T — the temperature; T > 1 flattens both distributions and exposes the teacher's secondary preferences
CE — the usual cross-entropy with the hard label
KL — the Kullback–Leibler divergence between the softened teacher and student distributions (see *Entropy, Cross-Entropy & KL*)
α — the balance between the two terms
T² — rescales the soft term, whose gradients shrink like 1/T², so both terms keep comparable weight
:::` },
      { type: 'code', title: 'Soft targets reveal what the teacher knows', code: `
import numpy as np
classes = ["cat", "tiger", "dog", "car"]
logits = np.array([9.0, 6.5, 5.4, 0.0])                 # a teacher looking at a photo of a cat

def softmax(z, T=1.0):
    e = np.exp((z - z.max()) / T)
    return e / e.sum()

for T in (1, 2, 5):
    p = softmax(logits, T)
    print(f"T = {T}: " + "   ".join(f"{c} {q:.3f}" for c, q in zip(classes, p)))
` , after: `At T = 1 almost everything goes to "cat". Raising the temperature reveals the teacher's view that a tiger is the next most likely class and a car is out of the question — information a hard label throws away.` },
      { type: 'code', title: 'Distillation beats hard labels when data is scarce', code: `
import numpy as np
rng = np.random.default_rng(0)
d, k = 20, 5

def softmax(z, T=1.0):
    e = np.exp((z - z.max(axis=1, keepdims=True)) / T)
    return e / e.sum(axis=1, keepdims=True)

W_teacher = rng.normal(size=(d, k)) * 1.2                 # a trained teacher (here we simply know its weights)
X_train, X_test = rng.normal(size=(60, d)), rng.normal(size=(5000, d))
P_true = softmax(X_train @ W_teacher)
y_train = np.array([rng.choice(k, p=p) for p in P_true])  # observed labels: one draw per example
y_test = (X_test @ W_teacher).argmax(axis=1)              # what the teacher would predict

def train_student(targets, T=1.0, steps=2000, lr=0.5):
    W = np.zeros((d, k))
    for _ in range(steps):
        P = softmax(X_train @ W, T)
        W -= lr * X_train.T @ (P - targets) / len(X_train) / T
    return W

accuracy = lambda W: np.mean((X_test @ W).argmax(axis=1) == y_test)
W_hard = train_student(np.eye(k)[y_train])                          # trained on the 60 hard labels
W_soft = train_student(softmax(X_train @ W_teacher, 2.0), T=2.0)    # trained on the teacher's soft targets, T = 2
print(f"student trained on hard labels:   agreement with the teacher {accuracy(W_hard):.3f}")
print(f"student trained on soft targets:  agreement with the teacher {accuracy(W_soft):.3f}")
`, after: `With only 60 examples, each hard label says just "this one class". The soft targets transmit the teacher's full judgement about every example, so the student learns far more from the same inputs and agrees with the teacher almost everywhere.` },
      { type: 'check', question: { type: 'mcq', q: 'Why use a temperature T > 1 for the teacher\'s targets?', options: ['To speed up training', 'To flatten the distribution so the small probabilities — the similarity structure — become visible', 'To make the teacher more confident', 'To reduce the number of classes'], answer: 1, explain: 'At T = 1 the secondary probabilities are often too tiny to matter.' } },
      { type: 'text', title: 'Distilling language models', md: `For LLMs, distillation comes in several forms:

- **Sequence-level distillation (synthetic data)**: the teacher generates responses — answers, explanations, reasoning traces — and the student is fine-tuned on them with ordinary next-token training. This is how many small instruction-following and reasoning models are made; for example, the DeepSeek-R1 release included smaller Qwen- and Llama-based models fine-tuned on the large model's reasoning traces.
- **Token-level (logit) distillation**: at each position the student matches the teacher's whole next-token distribution with a KL loss — far more signal per token than the single correct token, but it requires access to the teacher's probabilities (and the same tokenizer, or an alignment between tokenizers).
- **On-policy distillation**: the student generates text and the teacher provides targets on the student's own outputs, which avoids the mismatch between training on the teacher's text and generating one's own.

Data quality matters as much as quantity: filter generated data for correctness and diversity. Check model licences and terms of service before distilling from a model you don't own.` },
      { type: 'text', title: 'Pruning: removing what isn\'t needed', md: `Trained networks are over-parameterized: many weights can be removed with little loss.

- **Unstructured (magnitude) pruning** zeroes the weights with the smallest absolute values. Easy and effective, but the zeros are scattered, so speed-ups need specialized sparse kernels; the main win is smaller compressed models.
- **Structured pruning** removes whole neurons, channels, attention heads or layers. The remaining model is simply smaller and dense — real speed-ups on any hardware — but it usually needs fine-tuning to recover accuracy.
- **N:M sparsity** keeps N non-zero weights in every group of M (for example 2:4), a compromise that NVIDIA GPUs since the Ampere generation accelerate in hardware.
- **Iterative pruning**: prune a little, fine-tune, repeat — much better than one big cut.
- **The lottery ticket hypothesis** (Frankle & Carbin, 2019): dense networks contain sparse subnetworks that, trained in isolation from their original initialization, reach the full network's accuracy — suggesting that much of a big network's value lies in giving training many "tickets" to find a good one.` },
      { type: 'code', title: 'Magnitude pruning a trained network', code: `
import numpy as np, warnings
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

digits = load_digits()
X, y = digits.data / 16.0, digits.target
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
train, test = order[:1300], order[1300:]
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=0).fit(X[train], y[train])
print(f"dense network: {sum(w.size for w in mlp.coefs_)} weights, test accuracy {mlp.score(X[test], y[test]):.3f}")

original = [w.copy() for w in mlp.coefs_]
magnitudes = np.concatenate([np.abs(w).ravel() for w in original])
for sparsity in (0.5, 0.8, 0.9, 0.95):
    threshold = np.quantile(magnitudes, sparsity)                      # global magnitude threshold
    mlp.coefs_ = [np.where(np.abs(w) >= threshold, w, 0.0) for w in original]
    print(f"{sparsity:.0%} of weights removed: test accuracy {mlp.score(X[test], y[test]):.3f}   (no retraining)")
mlp.coefs_ = original
`, after: `Half the weights can be removed without any loss at all. Beyond that, accuracy drops quickly when nothing is retrained — which is why practical pruning removes weights gradually and fine-tunes in between.` }
    ],
    when: `- Deploying to phones, browsers, edge devices or high-traffic services where latency and cost matter.
- Creating small task-specific models from a large general one (distillation with generated data).
- Combining with quantization for maximum compression.
- Structured pruning when you need real speed-ups on ordinary hardware.`,
    pitfalls: `- Pruning in one big step without fine-tuning.
- Expecting unstructured sparsity to speed up inference on standard dense kernels.
- Distilling from a teacher's mistakes — filter generated data for correctness.
- Forgetting the T² factor (or using wildly different temperatures) so the soft term is too weak.
- Evaluating compressed models only on average accuracy — compression can hurt rare classes and long-tail inputs first.`,
    takeaways: [
      'Distillation trains a student on the teacher\'s soft probability distribution, which carries similarity information.',
      'Loss: α·CE(hard) + (1 − α)·T²·KL(teacher_T ‖ student_T).',
      'LLMs are distilled with generated data, token-level KL targets or on-policy training.',
      'Magnitude pruning removes small weights; structured pruning removes neurons, heads or channels for real speed-ups.',
      'Prune gradually with fine-tuning; N:M sparsity is hardware-friendly; the lottery ticket hypothesis explains why it works.'
    ],
    quiz: [
      { type: 'mcq', q: 'What extra information do soft targets give a student compared with hard labels?', options: ['None', 'How similar the teacher considers each wrong class to the right one', 'The teacher\'s architecture', 'The training data size'], answer: 1, explain: 'The relative probabilities of the wrong classes encode similarity.' },
      { type: 'tf', q: 'In the distillation loss, the soft term is multiplied by T² because its gradients shrink as the temperature grows.', answer: true, explain: 'Gradients scale like 1/T²; the factor restores their weight.' },
      { type: 'mcq', q: 'Which pruning method gives speed-ups on ordinary dense hardware without special kernels?', options: ['Unstructured magnitude pruning', 'Structured pruning of whole neurons, channels or heads', 'Random pruning', 'None of them'], answer: 1, explain: 'The result is a smaller dense model.' },
      { type: 'fill', q: 'In 2:4 sparsity, what fraction of the weights in each group are zero? (decimal or fraction)', answer: ['0.5', '1/2', '50%'], explain: 'Two of every four weights are kept.' },
      { type: 'mcq', q: 'What does the lottery ticket hypothesis claim?', options: ['Random weights are as good as trained ones', 'Dense networks contain sparse subnetworks that can be trained from their original initialization to full accuracy', 'Pruning always improves accuracy', 'Only the largest weights matter during training'], answer: 1, explain: 'Frankle & Carbin, 2019.' }
    ],
    problems: [
      {
        id: 'distillation-loss', title: 'The Distillation Loss', difficulty: 'medium',
        prompt: `Implement:

- \`softmax_t(z, T=1.0)\` → softmax(z / T) along the last axis (z may be 1-D or 2-D); subtract the maximum for stability.
- \`kd_loss(student_logits, teacher_logits, labels, T, alpha)\` → for a batch (rows are examples):
  α · mean_i CE(labels_i, softmax(s_i)) + (1 − α) · T² · mean_i KL(softmax(t_i / T) ‖ softmax(s_i / T)),
  where CE(y, p) = −log p[y] and KL(p ‖ q) = Σ p · (log p − log q).`,
        starter: `
import numpy as np

def softmax_t(z, T=1.0):
    # TODO
    pass

def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    # TODO
    pass
`,
        solution: `
import numpy as np

def softmax_t(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    S = np.asarray(student_logits, dtype=float)
    Tl = np.asarray(teacher_logits, dtype=float)
    y = np.asarray(labels)
    p = softmax_t(S)
    ce = -np.mean(np.log(p[np.arange(len(S)), y]))
    pt, ps = softmax_t(Tl, T), softmax_t(S, T)
    kl = np.mean(np.sum(pt * (np.log(pt) - np.log(ps)), axis=1))
    return float(alpha * ce + (1 - alpha) * T ** 2 * kl)
`,
        hints: ['`axis=-1` makes softmax work for both 1-D and 2-D inputs.', 'The hard-label term uses temperature 1; only the soft term uses T.'],
        explain: `When the student's softened distribution equals the teacher's, the KL term is zero and only the hard-label term remains. The T² factor keeps the soft term's gradients from vanishing as T grows.`,
        tests: [
          { call: 'softmax_t(np.array([2.0, 0.0]), 2.0)', expect: 'np.array([np.e / (np.e + 1), 1 / (np.e + 1)])' },
          { name: 'identical logits: only cross-entropy remains', call: 'kd_loss(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), [0], 2.0, 0.5)', expect: '0.5 * np.log(1 + np.exp(-1.0))' },
          { name: 'pure soft term', call: 'kd_loss(np.array([[0.0, 0.0]]), np.array([[2.0, 0.0]]), [1], 1.0, 0.0)', expect: '(np.e ** 2 / (np.e ** 2 + 1)) * np.log(2 * np.e ** 2 / (np.e ** 2 + 1)) + (1 / (np.e ** 2 + 1)) * np.log(2 / (np.e ** 2 + 1))' },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(0)\nS = rng.normal(size=(6, 4)); Tt = rng.normal(size=(6, 4)) * 3; y = rng.integers(0, 4, 6)', call: 'kd_loss(S, Tt, y, 3.0, 0.3)', hidden: true },
          { name: 'KL part is never negative', setup: 'rng = np.random.default_rng(1)\nS = rng.normal(size=(10, 5)); Tt = rng.normal(size=(10, 5))', code: 'assert kd_loss(S, Tt, np.zeros(10, dtype=int), 2.0, 0.0) >= -1e-12', hidden: true }
        ]
      },
      {
        id: 'magnitude-pruning', title: 'Magnitude and N:M Pruning Masks', difficulty: 'medium',
        prompt: `Implement pruning masks (True = keep the weight):

- \`prune_mask(W, sparsity)\` → remove the k = int(sparsity · W.size) weights with the smallest absolute values (ties: the earlier flat index is removed first — use a stable argsort of |W|.ravel()); return a boolean array shaped like W.
- \`nm_mask(W, n=2, m=4)\` → along the last axis, in each consecutive group of m weights keep the n with the largest absolute values (ties: keep the earlier one). The last axis length is a multiple of m.
- \`sparsity(W)\` → the fraction of entries that are exactly zero.`,
        starter: `
import numpy as np

def prune_mask(W, sparsity):
    # TODO
    pass

def nm_mask(W, n=2, m=4):
    # TODO
    pass

def sparsity(W):
    # TODO
    pass
`,
        solution: `
import numpy as np

def prune_mask(W, sparsity):
    W = np.asarray(W, dtype=float)
    k = int(sparsity * W.size)
    keep = np.ones(W.size, dtype=bool)
    keep[np.argsort(np.abs(W).ravel(), kind="stable")[:k]] = False
    return keep.reshape(W.shape)

def nm_mask(W, n=2, m=4):
    W = np.asarray(W, dtype=float)
    groups = np.abs(W).reshape(-1, m)
    order = np.argsort(-groups, axis=1, kind="stable")[:, :n]
    keep = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(keep, order, True, axis=1)
    return keep.reshape(W.shape)

def sparsity(W):
    return float(np.mean(np.asarray(W) == 0))
`,
        hints: ['A stable argsort of the flattened magnitudes lists the weights to remove first.', 'Reshape into rows of m weights, pick the top n per row, then reshape back.', '`np.put_along_axis` sets entries at per-row indices.'],
        explain: `Global magnitude pruning is a strong simple baseline; N:M patterns trade a little flexibility for hardware support. Applying a mask is just W * mask, and during fine-tuning the mask is re-applied after every update so pruned weights stay zero.`,
        tests: [
          { call: 'prune_mask(np.array([[0.5, -0.1], [2.0, 0.05]]), 0.5)', expect: 'np.array([[True, False], [True, False]])' },
          { call: 'nm_mask(np.array([[0.1, -0.9, 0.3, 0.2, 1.0, 1.0, -1.0, 0.0]]))', expect: 'np.array([[False, True, True, False, True, True, False, False]])' },
          { call: 'sparsity(np.array([0.0, 1.0, 0.0, 3.0]))', expect: '0.5' },
          { name: 'random matrix vs reference', setup: 'rng = np.random.default_rng(0)\nW = rng.normal(size=(6, 8))', call: '[prune_mask(W, 0.7), nm_mask(W, 2, 4)]', hidden: true },
          { name: 'masks have the requested density', setup: 'rng = np.random.default_rng(1)\nW = rng.normal(size=(16, 32))', code: 'assert prune_mask(W, 0.75).sum() == W.size - int(0.75 * W.size)\nassert nm_mask(W, 1, 4).sum() == W.size // 4', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Hinton, Vinyals & Dean — Distilling the Knowledge in a Neural Network', url: 'https://arxiv.org/abs/1503.02531' },
      { title: 'Frankle & Carbin — The Lottery Ticket Hypothesis', url: 'https://arxiv.org/abs/1803.03635' },
      { title: 'Kim & Rush — Sequence-Level Knowledge Distillation', url: 'https://arxiv.org/abs/1606.07947' },
      { title: 'Han et al. — Deep Compression: pruning, quantization and Huffman coding', url: 'https://arxiv.org/abs/1510.00149' }
    ]
  },

  // =====================================================================================
  {
    id: 'self-supervised', branch: 'dl', title: 'Self-Supervised Representation Learning', short: 'Self-Supervised Learning', level: 3, minutes: 45, prereqs: ['vision-transformers', 'text-embeddings'],
    tags: ['self-supervised learning', 'representation learning', 'pretext task', 'contrastive learning', 'simclr', 'nt-xent', 'augmentations', 'byol', 'dino', 'collapse', 'masked autoencoder', 'mae', 'jepa', 'linear probe', 'foundation models'],
    summary: 'Learning useful representations without labels: pretext tasks, contrastive learning with SimCLR and the NT-Xent loss, why augmentations define what is learned, non-contrastive methods such as BYOL and DINO and the collapse problem they must avoid, masked autoencoders, and how self-supervised features are evaluated and used.',
    why: `Labels are scarce and expensive; raw images, audio and text are nearly unlimited. **Self-supervised learning** creates its own training signal from the data — predict a hidden part from the rest, or recognize two views of the same thing — and the representations it learns transfer to many downstream tasks with few labels. It is the principle behind every foundation model: next-token prediction in GPT, masked tokens in BERT, image–text pairs in CLIP, and, for images and audio without text, methods such as SimCLR, DINO and masked autoencoders.`,
    objectives: [
      'Explain pretext tasks and how self-supervised features are evaluated (linear probes, k-NN, fine-tuning)',
      'Implement the NT-Xent contrastive loss and explain the role of augmentations, negatives and temperature',
      'Describe representation collapse and how BYOL, DINO and VICReg avoid it',
      'Explain masked autoencoders and why masking most of an image works'
    ],
    sections: [
      { type: 'text', title: 'Labels from the data itself', md: `A **pretext task** is a made-up prediction problem whose labels come for free:

- **Predict the hidden part**: the next token (GPT), masked tokens (BERT), masked image patches (MAE), masked stretches of audio (wav2vec 2.0, HuBERT).
- **Recognize the same thing**: two augmented views of one image (SimCLR, DINO), or an image and its caption (CLIP).

Nobody cares about the pretext task itself; the goal is the **representation** the network builds to solve it. It is evaluated by how useful it is downstream: a **linear probe** (freeze the network, train only a linear classifier on its features), k-nearest-neighbour classification on the features, or full fine-tuning on a small labelled set.` },
      { type: 'text', title: 'Contrastive learning: SimCLR', md: `**SimCLR** takes a batch of N images and creates two random augmentations of each (random crops, colour jitter, blur). An encoder plus a small projection head maps all 2N views to vectors z. Each view must identify its partner among the other 2N − 1 views — the **NT-Xent** loss, a softmax cross-entropy over cosine similarities:

$$ ℓ_i = −log [ exp( sim(z_i, z_p(i)) / τ ) / Σ_{k ≠ i} exp( sim(z_i, z_k) / τ ) ] $$
:::where
z_i — the projected vector of view i
p(i) — the index of the other view of the same image (the positive)
k — runs over all other 2N − 1 views in the batch; all but one of them are negatives
sim — cosine similarity
τ — "tau": the temperature; small values focus the loss on the hardest negatives
:::

The augmentations decide what the representation ignores: random crops teach that object identity doesn't depend on position or scale, colour jitter that it doesn't depend on exact colours. Contrastive methods need many negatives, hence large batches (or a memory bank of past embeddings, as in MoCo). The same loss family trains text embedding models (see *Text Embeddings*) and CLIP.` },
      { type: 'code', title: 'The NT-Xent loss for a batch of views', code: `
import numpy as np
rng = np.random.default_rng(0)
N, d = 64, 32

def nt_xent(z, tau=0.5):
    z = z / np.linalg.norm(z, axis=1, keepdims=True)          # cosine similarity = dot product of unit vectors
    n = len(z) // 2                                           # rows i and i + n are two views of one image
    sim = z @ z.T / tau
    np.fill_diagonal(sim, -np.inf)                            # a view is never its own candidate
    positive = np.r_[np.arange(n, 2 * n), np.arange(0, n)]
    m = sim.max(axis=1, keepdims=True)
    log_denominator = np.log(np.exp(sim - m).sum(axis=1)) + m[:, 0]
    return np.mean(log_denominator - sim[np.arange(2 * n), positive])

h = rng.normal(size=(N, d))                                   # what each image "really is"
aligned = np.vstack([h + 0.1 * rng.normal(size=(N, d)), h + 0.1 * rng.normal(size=(N, d))])
unrelated = rng.normal(size=(2 * N, d))
collapsed = np.ones((2 * N, d)) + 0.001 * rng.normal(size=(2 * N, d))
print(f"views agree, images differ:  loss {nt_xent(aligned):.3f}")
print(f"random embeddings:           loss {nt_xent(unrelated):.3f}")
print(f"everything collapsed:        loss {nt_xent(collapsed):.3f}   (log(2N - 1) = {np.log(2 * N - 1):.3f})")
`, after: `The loss is lowest when the two views of each image match *and* different images stay apart. If every embedding collapses to the same vector, each view sees 2N − 1 equally similar candidates and the loss is stuck at log(2N − 1): the negatives are what make collapse costly.` },
      { type: 'text', title: 'Without negatives: BYOL, DINO and collapse', md: `If a method only pulls the two views of an image together, a trivial solution exists: map every input to the same vector. This **collapse** makes the loss perfect and the representation useless. Non-contrastive methods avoid it without negatives:

- **BYOL** and **DINO** use a **student** network and a **teacher** whose weights are an exponential moving average (EMA) of the student's; the student predicts the teacher's output for another view, and gradients flow only through the student (stop-gradient):

$$ θ_teacher ← m · θ_teacher + (1 − m) · θ_student $$
:::where
θ_teacher, θ_student — the weights of the teacher and the student networks
m — the momentum, close to 1 (e.g. 0.996), so the teacher changes slowly and provides stable targets
:::

- **DINO** additionally **centres** and **sharpens** the teacher's outputs so that no single dimension can dominate; DINOv2 features are among the strongest general-purpose image features.
- **VICReg** adds explicit terms that keep the variance of every embedding dimension up and decorrelate the dimensions.

Collapse can also be partial — **dimensional collapse**, where embeddings use only a few directions. Monitor the per-dimension standard deviation and the effective rank of a batch of embeddings during training.` },
      { type: 'code', title: 'Detecting collapse', code: `
import numpy as np
rng = np.random.default_rng(1)
B, d = 512, 64

def health(Z):
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    std = Z.std(axis=0).mean()                                  # average spread of each dimension across the batch
    s = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    p = s / s.sum()
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-12)))           # exp(entropy of the singular values)
    return std, eff_rank

healthy = rng.normal(size=(B, d))
dimensional = rng.normal(size=(B, 3)) @ rng.normal(size=(3, d))  # all embeddings in a 3-D subspace
collapsed = np.ones((B, d)) + 0.01 * rng.normal(size=(B, d))
for name, Z in [("healthy", healthy), ("dimensional collapse", dimensional), ("complete collapse", collapsed)]:
    std, er = health(Z)
    print(f"{name:22s} mean per-dimension std {std:.3f}   effective rank {er:5.1f} of {d}")
`, after: `A healthy batch spreads over almost all 64 directions. Dimensional collapse keeps a reasonable spread but an effective rank of about 3; complete collapse shows near-zero spread. The effective rank of the remaining noise is high in that case, so always look at both numbers.` },
      { type: 'check', question: { type: 'tf', q: 'Without negatives or some other safeguard, a model trained only to make two views of an image agree can map every input to the same vector.', answer: true, explain: 'That is representation collapse.' } },
      { type: 'text', title: 'Masked image modelling: MAE', md: `**Masked autoencoders** (He et al., 2021) bring BERT's idea to images. Cut the image into patches, hide a random **75%** of them, feed only the visible 25% to a Vision Transformer encoder, and let a small decoder reconstruct the pixels of the hidden patches. A high masking ratio makes the task hard enough to require understanding (neighbouring patches are too redundant for a low ratio to teach much), and because the encoder sees only a quarter of the patches, pretraining is several times cheaper. MAE features fine-tune very well.

**JEPA** (joint-embedding predictive architectures) predict the *representations* of masked regions rather than their pixels, so the model need not spend capacity on unpredictable low-level detail.` },
      { type: 'code', title: 'Masking patches like an MAE', code: `
import numpy as np
rng = np.random.default_rng(0)
num_patches, mask_ratio, width = 196, 0.75, 768            # a 224 x 224 image cut into 16 x 16 patches
keep = int(num_patches * (1 - mask_ratio))

ids_shuffle = np.argsort(rng.random(num_patches))           # a random permutation of the patches
ids_restore = np.argsort(ids_shuffle)                       # its inverse, to put patches back in order
ids_keep = ids_shuffle[:keep]
mask = np.ones(num_patches)
mask[:keep] = 0
mask = mask[ids_restore]                                    # 1 = hidden, in the original patch order

patches = rng.normal(size=(num_patches, width))             # stand-in patch embeddings
visible = patches[ids_keep]                                 # only these go through the encoder
print(f"patches: {num_patches}, visible to the encoder: {len(visible)}, hidden: {int(mask.sum())}")
print(f"attention cost relative to the full image: {(len(visible) / num_patches) ** 2:.4f}  (it grows with the square of the token count)")
print(f"MLP cost relative to the full image:       {len(visible) / num_patches:.2f}")
print("first 20 patches (1 = hidden):", mask[:20].astype(int))
` },
      { type: 'text', title: 'Using self-supervised features', md: `- **Frozen features + a small head** — a linear probe or k-NN on the embeddings — for classification, retrieval and clustering with few labels.
- **Fine-tuning** the whole network on the downstream task, usually the most accurate option.
- **Dense features** from methods such as DINO also support segmentation and correspondence without labels.
- **Beyond images**: wav2vec 2.0 and HuBERT learn speech representations by predicting masked audio; the same ideas apply to video, time series, graphs and medical data.

Self-supervised pretraining followed by light supervision is now the default recipe whenever unlabelled data is abundant and labels are not.` }
    ],
    when: `- Plenty of unlabelled data and few labels: pretrain (or start from a self-supervised checkpoint), then probe or fine-tune.
- Domains poorly covered by public pretrained models — medical, satellite, industrial images, sensor data.
- Retrieval, clustering and deduplication using embedding similarity.
- Prefer existing checkpoints (DINOv2, MAE, CLIP) over pretraining from scratch unless your domain is very different.`,
    pitfalls: `- Augmentations that destroy the information you need downstream (e.g. colour jitter when colour is the label).
- Small batches for contrastive methods, which then see too few negatives.
- Undetected collapse — monitor embedding spread and effective rank.
- Judging features only by the pretext loss instead of downstream probes.
- Comparing linear-probe numbers across papers with different evaluation protocols.`,
    takeaways: [
      'Self-supervised learning invents labels from the data: predict hidden parts or match different views.',
      'SimCLR\'s NT-Xent loss pulls two views of an image together and pushes other images away; augmentations define the invariances.',
      'Without negatives, models can collapse to a constant; BYOL/DINO use an EMA teacher and stop-gradients, VICReg variance terms.',
      'MAE hides 75% of image patches and reconstructs them, which is both a hard task and cheap to train.',
      'Evaluate with linear probes, k-NN or fine-tuning; foundation models in every modality rest on these ideas.'
    ],
    quiz: [
      { type: 'mcq', q: 'In SimCLR, what counts as the positive example for a view of an image?', options: ['Another image of the same class', 'The other augmented view of the same image', 'The image\'s label', 'A random image from the batch'], answer: 1, explain: 'No labels are used; the pair comes from the same image.' },
      { type: 'mcq', q: 'What is representation collapse?', options: ['The loss becoming negative', 'The encoder mapping every input to (nearly) the same vector', 'The batch size shrinking', 'Overfitting to labels'], answer: 1, explain: 'It satisfies "views agree" trivially and carries no information.' },
      { type: 'fill', q: 'A ViT image has 196 patches and an MAE masks 75% of them. How many patches does the encoder process?', answer: ['49'], explain: '196 × 0.25 = 49.' },
      { type: 'tf', q: 'In BYOL and DINO, the teacher network is trained by gradient descent like the student.', answer: false, explain: 'The teacher is an exponential moving average of the student; gradients are stopped.' },
      { type: 'mcq', q: 'What is a linear probe?', options: ['Fine-tuning every layer', 'Training only a linear classifier on frozen features to measure their quality', 'A pruning method', 'A data augmentation'], answer: 1, explain: 'It measures how linearly separable the learned features are.' }
    ],
    problems: [
      {
        id: 'nt-xent-loss', title: 'The NT-Xent Loss', difficulty: 'medium',
        prompt: `Implement \`nt_xent(z, tau=0.5)\` for an array z of shape (2N, d): rows i and i + N are the two views of the same example.

1. Normalize each row to unit length; similarities are dot products divided by tau.
2. For every row i (all 2N of them), the loss is −log( exp(sim(i, p)) / Σ_{k ≠ i} exp(sim(i, k)) ), where p is i's partner (i + N or i − N).
3. Return the mean over all 2N rows. Use a numerically stable log-sum-exp.`,
        starter: `
import numpy as np

def nt_xent(z, tau=0.5):
    # TODO
    pass
`,
        solution: `
import numpy as np

def nt_xent(z, tau=0.5):
    z = np.asarray(z, dtype=float)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    n2 = len(z)
    n = n2 // 2
    sim = z @ z.T / tau
    np.fill_diagonal(sim, -np.inf)
    pos = np.r_[np.arange(n, n2), np.arange(0, n)]
    m = sim.max(axis=1, keepdims=True)
    lse = np.log(np.exp(sim - m).sum(axis=1)) + m[:, 0]
    return float(np.mean(lse - sim[np.arange(n2), pos]))
`,
        hints: ['Set the diagonal of the similarity matrix to −inf so a view never counts itself.', 'log Σ exp(x) = m + log Σ exp(x − m) with m = max(x).'],
        explain: `Each row is a (2N − 1)-way classification problem whose correct answer is the partner view. With identical directions everywhere the loss equals log(2N − 1); perfectly separated pairs drive it towards 0.`,
        tests: [
          { name: 'collapsed embeddings give log(2N − 1)', call: 'nt_xent(np.ones((6, 4)))', expect: 'np.log(5)' },
          { name: 'orthogonal pairs', call: 'nt_xent(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]), tau=1.0)', expect: '-1.0 + np.log(np.e + 2.0)' },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(0)\nZ = rng.normal(size=(8, 5))', call: 'nt_xent(Z, 0.2)', hidden: true },
          { name: 'aligned views beat random ones', setup: 'rng = np.random.default_rng(1)\nh = rng.normal(size=(16, 8))', code: 'good = nt_xent(np.vstack([h, h + 0.01 * rng.normal(size=h.shape)]))\nbad = nt_xent(rng.normal(size=(32, 8)))\nassert good < bad', hidden: true }
        ]
      },
      {
        id: 'ema-collapse', title: 'EMA Teachers and Collapse Diagnostics', difficulty: 'easy',
        prompt: `Implement:

- \`ema_update(teacher, student, m)\` → the new teacher weights m·teacher + (1 − m)·student (NumPy arrays).
- \`embedding_std(Z)\` → normalize each row of Z to unit length, then return the mean over dimensions of the standard deviation across the batch (NumPy's default, ddof = 0).
- \`effective_rank(Z)\` → centre Z by subtracting the column means, compute its singular values s, set p = s / Σs (ignore zero singular values) and return exp(−Σ p·ln p).`,
        starter: `
import numpy as np

def ema_update(teacher, student, m):
    # TODO
    pass

def embedding_std(Z):
    # TODO
    pass

def effective_rank(Z):
    # TODO
    pass
`,
        solution: `
import numpy as np

def ema_update(teacher, student, m):
    return m * np.asarray(teacher, dtype=float) + (1 - m) * np.asarray(student, dtype=float)

def embedding_std(Z):
    Z = np.asarray(Z, dtype=float)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return float(Z.std(axis=0).mean())

def effective_rank(Z):
    Z = np.asarray(Z, dtype=float)
    s = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))
`,
        hints: ['`np.linalg.svd(A, compute_uv=False)` returns only the singular values.', 'Drop (numerically) zero singular values before taking logs.'],
        explain: `A slowly moving EMA teacher gives the student stable targets, one ingredient that keeps BYOL and DINO from collapsing. The effective rank counts how many directions the embeddings really use: d for perfectly spread embeddings, 1 when they lie on a line.`,
        tests: [
          { call: 'ema_update(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.9)', expect: 'np.array([0.9, 0.1])' },
          { name: 'two orthogonal directions, equal spread', call: 'effective_rank(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]))', expect: '2.0' },
          { name: 'identical rows have zero spread', call: 'embedding_std(np.tile([3.0, 4.0], (5, 1)))', expect: '0.0' },
          { name: 'random embeddings vs reference', setup: 'rng = np.random.default_rng(0)\nZ = rng.normal(size=(50, 6)); Zlow = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 6))', call: '[embedding_std(Z), effective_rank(Z), effective_rank(Zlow)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Chen et al. — A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)', url: 'https://arxiv.org/abs/2002.05709' },
      { title: 'Grill et al. — Bootstrap Your Own Latent (BYOL)', url: 'https://arxiv.org/abs/2006.07733' },
      { title: 'Caron et al. — Emerging Properties in Self-Supervised Vision Transformers (DINO)', url: 'https://arxiv.org/abs/2104.14294' },
      { title: 'He et al. — Masked Autoencoders Are Scalable Vision Learners', url: 'https://arxiv.org/abs/2111.06377' }
    ]
  },

  // =====================================================================================
  {
    id: 'diffusion-models', branch: 'spec', title: 'Diffusion Models & Flow Matching', short: 'Diffusion Models', level: 3, minutes: 50, prereqs: ['generative-models', 'transformer-block'],
    tags: ['diffusion models', 'ddpm', 'noise schedule', 'forward process', 'reverse process', 'denoising', 'noise prediction', 'score matching', 'ddim', 'classifier-free guidance', 'latent diffusion', 'dit', 'flow matching', 'rectified flow', 'fid'],
    summary: 'How modern image, video and audio generators work: gradually noising data in closed form, training a network to predict the noise, sampling by running the process backwards (DDPM and faster samplers), conditioning on text with classifier-free guidance, latent diffusion and diffusion transformers, flow matching, and how generated samples are evaluated.',
    why: `Diffusion models produced the leap in image generation (Stable Diffusion, DALL·E, Imagen, Midjourney) and now drive video, audio, 3-D and even robot-control generation. The core idea is surprisingly simple — learn to remove a little noise — and it connects to concepts across this atlas: Gaussians, MSE regression, U-Nets, transformers, cross-attention and VAEs. **Flow matching**, a close relative, powers many of the newest models.`,
    objectives: [
      'Write the forward noising process and sample x_t directly from x_0',
      'Explain the noise-prediction training objective and its link to score matching',
      'Run the reverse (sampling) process and explain faster samplers',
      'Describe conditioning with classifier-free guidance, latent diffusion, DiT and flow matching'
    ],
    sections: [
      { type: 'text', title: 'Generating by denoising', md: `A diffusion model learns to reverse a process that slowly destroys data with noise. Training: take a real image, add a random amount of noise, and teach a network to predict the noise that was added. Generation: start from pure noise and repeatedly ask the network to remove a little of it, until an image appears.

Compared with GANs (see *Generative Models*), diffusion training is a plain, stable regression problem, and the many small steps make it easy to cover all modes of the data. The price is slow sampling — many network evaluations per sample — which much research has reduced.` },
      { type: 'text', title: 'The forward process', md: `Over T steps (typically 1,000), a little Gaussian noise is added at each step according to a **noise schedule** β₁ … β_T. The useful fact is that any step can be reached in one jump:

$$ x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε          ε ~ N(0, I)          ᾱ_t = Π_{s=1}^{t} (1 − β_s) $$
:::where
x₀ — a clean data example (e.g. an image)
x_t — the noisy version after t steps
ε — "epsilon": standard Gaussian noise of the same shape as x₀
β_s — the small amount of noise added at step s (the noise schedule)
ᾱ_t — "alpha bar": the fraction of signal power left after t steps; it falls from almost 1 to almost 0
:::

Early steps barely change the data; by the last step x_T is indistinguishable from pure noise. Schedules (linear, cosine) control how quickly the signal fades.` },
      { type: 'code', title: 'Noising data in closed form', code: `
import numpy as np
rng = np.random.default_rng(0)
x0 = np.concatenate([rng.normal([-2, 0], 0.3, (500, 2)), rng.normal([2, 0], 0.3, (500, 2))])   # two blobs of 2-D data
T = 1000
betas = np.linspace(1e-4, 0.02, T)                     # the linear schedule of the original DDPM paper
alpha_bar = np.cumprod(1 - betas)
for t in (0, 100, 250, 500, 750, 999):
    eps = rng.normal(size=x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    corr = np.corrcoef(x0[:, 0], xt[:, 0])[0, 1]
    print(f"t = {t:3d}: signal weight {np.sqrt(alpha_bar[t]):.3f}  noise weight {np.sqrt(1 - alpha_bar[t]):.3f}  "
          f"corr(x0, xt) {corr:+.3f}  std of xt {xt.std(axis=0).round(2)}")
`, after: `The signal weight decays from 1 to almost 0 while the noise weight grows to 1. By the last step the noisy points have unit spread and essentially no correlation with the original data: the two blobs have become a single Gaussian cloud.` },
      { type: 'demo', title: 'Interactive: watch data dissolve into noise', demo: 'diffusion' },
      { type: 'text', title: 'Training: predict the noise', md: `Pick a training example x₀, a random step t and random noise ε; build x_t with the formula above; train a network ε_θ to recover the noise:

$$ L = E_{x₀, t, ε} [ ‖ ε − ε_θ( x_t, t ) ‖² ] $$
:::where
ε_θ(x_t, t) — the network's prediction of the noise in x_t; it also receives the step t (as an embedding) so it knows how noisy its input is
‖ … ‖² — the squared error, so training is plain regression
E — the average over training examples, steps and noise draws
:::

Predicting the noise is equivalent to predicting the clean image, x̂₀ = (x_t − √(1 − ᾱ_t) ε_θ) / √ᾱ_t, and to estimating the **score** ∇ log p_t(x_t) = −ε / √(1 − ᾱ_t), the direction towards more probable data — which is why diffusion models are also called **score-based models**. The network is usually a U-Net (see *Object Detection & Segmentation*) or a transformer.` },
      { type: 'text', title: 'Sampling: running the process backwards', md: `Start from x_T ~ N(0, I) and step down to t = 0. The DDPM sampler removes the predicted noise and adds back a little fresh noise:

$$ x_{t−1} = ( x_t − β_t / √(1 − ᾱ_t) · ε_θ(x_t, t) ) / √(1 − β_t) + σ_t · z          z ~ N(0, I) $$
:::where
β_t — the noise schedule at step t
ε_θ(x_t, t) — the network's noise prediction
σ_t — the amount of fresh noise to add (σ_t² = β_t is a common choice; σ = 0 at the last step)
z — new standard Gaussian noise
:::

A thousand network calls per image is slow. **DDIM** makes the process deterministic and skips steps (20–50 suffice); higher-order ODE solvers do better still; **distillation** and **consistency models** produce good samples in 1–4 steps.` },
      { type: 'demo', title: 'Diagram: training and sampling side by side', demo: 'dg-diffusion' },
      { type: 'code', title: 'Sampling with a perfect denoiser', code: `
import numpy as np
rng = np.random.default_rng(0)
data = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])   # a tiny "training set" of three points
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas, alpha_bar = 1 - betas, np.cumprod(1 - betas)

def eps_hat(xt, t):
    # The best possible noise prediction for this training set: which training point most likely produced x_t?
    d2 = ((xt[:, None, :] - np.sqrt(alpha_bar[t]) * data[None]) ** 2).sum(-1)
    logw = -d2 / (2 * (1 - alpha_bar[t]))
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    x0_hat = w @ data                                     # expected clean point given x_t
    return (xt - np.sqrt(alpha_bar[t]) * x0_hat) / np.sqrt(1 - alpha_bar[t])

x = rng.normal(size=(2000, 2))                            # start from pure noise
for t in range(T - 1, -1, -1):                            # the DDPM reverse process
    mean = (x - betas[t] / np.sqrt(1 - alpha_bar[t]) * eps_hat(x, t)) / np.sqrt(alphas[t])
    x = mean + (np.sqrt(betas[t]) * rng.normal(size=x.shape) if t > 0 else 0)

nearest = ((x[:, None, :] - data[None]) ** 2).sum(-1).argmin(axis=1)
print("share of samples at each training point:", [round(float(np.mean(nearest == i)), 3) for i in range(3)])
print("largest distance from a sample to its nearest training point:", float(np.sqrt(((x - data[nearest]) ** 2).sum(1)).max()))
`, after: `Starting from pure noise, the reverse process delivers samples spread evenly over the data. But notice what the *perfect* denoiser for a finite training set does: it reproduces the training points exactly. Real networks cannot represent this ideal function and generalize to new samples instead — and when a model has too much capacity for too little data, it does memorize, which is a real privacy and copyright concern.` },
      { type: 'check', question: { type: 'mcq', q: 'What does the network learn to predict in standard DDPM training?', options: ['The class label', 'The noise ε that was added to produce x_t', 'The next pixel', 'The number of steps'], answer: 1, explain: 'An MSE regression onto the added noise.' } },
      { type: 'text', title: 'Conditioning and classifier-free guidance', md: `Text-to-image models give the denoiser a condition c — usually text embeddings from a language or CLIP encoder, injected through **cross-attention** layers. **Classifier-free guidance** strengthens the condition at sampling time. During training the condition is dropped (replaced by an empty one) about 10% of the time, so one network learns both the conditional and the unconditional prediction; at sampling time they are extrapolated:

$$ ε̃ = ε_θ(x_t, ∅) + w · ( ε_θ(x_t, c) − ε_θ(x_t, ∅) ) $$
:::where
ε_θ(x_t, c) — the noise prediction given the condition c (e.g. a prompt)
ε_θ(x_t, ∅) — the prediction with the empty condition
w — the guidance scale; w = 1 is plain conditional sampling, larger values (often 3–8) follow the prompt more closely at the cost of diversity
:::` },
      { type: 'text', title: 'Latent diffusion, DiT and flow matching', md: `- **Latent diffusion** (Stable Diffusion): first train an autoencoder (a VAE) that compresses images, e.g. 8× in each spatial dimension; run diffusion in that small latent space and decode at the end — far cheaper than diffusing raw pixels.
- **Diffusion transformers (DiT)** replace the U-Net with a transformer over latent patches, scaling cleanly with compute; many recent image and video models use them.
- **Flow matching / rectified flow** learns a velocity field that moves samples along (nearly) straight paths from noise to data, and generates by integrating an ordinary differential equation:

$$ x_t = (1 − t) · x₀ + t · ε          L = E ‖ v_θ(x_t, t) − (ε − x₀) ‖² $$
:::where
t — a time between 0 (data) and 1 (noise)
x₀, ε — a data example and a noise sample
v_θ(x_t, t) — the network's predicted velocity at the point x_t
ε − x₀ — the constant velocity of the straight path from data to noise, the regression target
:::

Straighter paths need fewer sampling steps; Stable Diffusion 3, for example, is a rectified-flow transformer. **Evaluation**: FID (the Fréchet distance between feature statistics of real and generated images), CLIP score for prompt alignment, and — still essential — human preference studies.` }
    ],
    when: `- Generating images, video, audio, 3-D shapes and molecules; inpainting, super-resolution and editing.
- Conditional generation (text-to-image, image-to-image) with guidance.
- Robotics and planning ("diffusion policies") and other problems with multi-modal outputs.
- Start from pretrained latent diffusion or flow models and fine-tune (e.g. with LoRA) rather than training from scratch.`,
    pitfalls: `- Too few sampling steps with a sampler not designed for it (blurry or broken outputs).
- Very high guidance scales: saturated, low-diversity images.
- Memorizing training data when models are large relative to their data — check for near-duplicates of training images.
- Judging quality by FID alone; it misses prompt faithfulness and many artefacts.
- Ignoring licensing, consent and misuse (deepfakes); use provenance and watermarking where possible.`,
    takeaways: [
      'Forward process: x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε, reachable in one jump for any t.',
      'Training is MSE regression onto the added noise, equivalent to learning the score of the noisy data.',
      'Sampling runs the process backwards from pure noise; DDIM, ODE solvers and distillation cut the number of steps.',
      'Classifier-free guidance extrapolates from unconditional to conditional predictions to follow prompts.',
      'Latent diffusion, diffusion transformers and flow matching (straight paths) power today\'s generators.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why can x_t be sampled directly from x₀ without simulating all intermediate steps?', options: ['Because the network is linear', 'Because adding Gaussian noise repeatedly is again Gaussian, with a closed-form mean and variance', 'Because T is small', 'Because the data is Gaussian'], answer: 1, explain: 'Sums of independent Gaussians are Gaussian.' },
      { type: 'fill', q: 'With ᾱ_t = 0.36, what is the weight √ᾱ_t of the clean signal in x_t?', answer: ['0.6', '0.60'], explain: '√0.36 = 0.6; the noise weight is √0.64 = 0.8.' },
      { type: 'mcq', q: 'What does classifier-free guidance with a scale w > 1 do?', options: ['Removes the text condition', 'Pushes predictions further in the direction the condition suggests, trading diversity for prompt adherence', 'Trains a separate classifier', 'Speeds up sampling by w times'], answer: 1, explain: 'It extrapolates from the unconditional to the conditional prediction.' },
      { type: 'tf', q: 'Latent diffusion models run the diffusion process in the compressed latent space of an autoencoder.', answer: true, explain: 'That is what makes them affordable at high resolution.' },
      { type: 'mcq', q: 'What does flow matching learn?', options: ['A discriminator', 'A velocity field that transports noise to data, integrated as an ODE at sampling time', 'A tokenizer', 'A ranking function'], answer: 1, explain: 'With straight-line paths the target velocity is ε − x₀.' }
    ],
    problems: [
      {
        id: 'diffusion-forward-process', title: 'The Forward Process', difficulty: 'easy',
        prompt: `Implement (step indices start at 0, so alpha_bar[t] = Π_{s=0}^{t} (1 − betas[s])):

- \`alpha_bars(betas)\` → the cumulative products ᾱ_t of (1 − β).
- \`q_sample(x0, t, eps, alpha_bar)\` → √ᾱ_t · x0 + √(1 − ᾱ_t) · eps.
- \`predict_x0(xt, t, eps_hat, alpha_bar)\` → the clean-data estimate (xt − √(1 − ᾱ_t) · eps_hat) / √ᾱ_t.`,
        starter: `
import numpy as np

def alpha_bars(betas):
    # TODO
    pass

def q_sample(x0, t, eps, alpha_bar):
    # TODO
    pass

def predict_x0(xt, t, eps_hat, alpha_bar):
    # TODO
    pass
`,
        solution: `
import numpy as np

def alpha_bars(betas):
    return np.cumprod(1 - np.asarray(betas, dtype=float))

def q_sample(x0, t, eps, alpha_bar):
    a = alpha_bar[t]
    return np.sqrt(a) * np.asarray(x0, dtype=float) + np.sqrt(1 - a) * np.asarray(eps, dtype=float)

def predict_x0(xt, t, eps_hat, alpha_bar):
    a = alpha_bar[t]
    return (np.asarray(xt, dtype=float) - np.sqrt(1 - a) * np.asarray(eps_hat, dtype=float)) / np.sqrt(a)
`,
        hints: ['`np.cumprod` computes running products.', 'predict_x0 simply solves the q_sample formula for x0.'],
        explain: `Because x_t is a known mix of signal and noise, a perfect noise prediction gives back the exact clean input — the relationship every diffusion sampler relies on.`,
        tests: [
          { call: 'alpha_bars([0.1, 0.5])', expect: 'np.array([0.9, 0.45])' },
          { call: 'q_sample(np.array([1.0, -1.0]), 0, np.array([0.5, 0.5]), np.array([0.64]))', expect: 'np.array([0.8 + 0.3, -0.8 + 0.3])' },
          { name: 'a perfect noise prediction recovers x0', setup: 'rng = np.random.default_rng(0)\nab = alpha_bars(np.linspace(1e-4, 0.02, 1000)); x0 = rng.normal(size=5); e = rng.normal(size=5)', code: 'xt = q_sample(x0, 600, e, ab)\nassert np.allclose(predict_x0(xt, 600, e, ab), x0)', hidden: true },
          { name: 'schedule vs reference', call: 'alpha_bars(np.linspace(1e-4, 0.02, 10))', hidden: true }
        ]
      },
      {
        id: 'guided-sampling-step', title: 'Guidance and a DDPM Step', difficulty: 'medium',
        prompt: `Implement:

- \`cfg(eps_uncond, eps_cond, w)\` → classifier-free guidance: eps_uncond + w · (eps_cond − eps_uncond).
- \`ddpm_step(xt, t, eps_hat, betas, z)\` → one reverse step with 0-based t:
  alpha_t = 1 − betas[t], alpha_bar_t = Π_{s ≤ t} (1 − betas[s]),
  mean = (xt − betas[t] / √(1 − alpha_bar_t) · eps_hat) / √alpha_t,
  return mean + √betas[t] · z, except at t = 0 where you return the mean (no noise).`,
        starter: `
import numpy as np

def cfg(eps_uncond, eps_cond, w):
    # TODO
    pass

def ddpm_step(xt, t, eps_hat, betas, z):
    # TODO
    pass
`,
        solution: `
import numpy as np

def cfg(eps_uncond, eps_cond, w):
    eu, ec = np.asarray(eps_uncond, dtype=float), np.asarray(eps_cond, dtype=float)
    return eu + w * (ec - eu)

def ddpm_step(xt, t, eps_hat, betas, z):
    betas = np.asarray(betas, dtype=float)
    alpha_bar = np.cumprod(1 - betas)[t]
    mean = (np.asarray(xt, dtype=float) - betas[t] / np.sqrt(1 - alpha_bar) * np.asarray(eps_hat, dtype=float)) / np.sqrt(1 - betas[t])
    if t == 0:
        return mean
    return mean + np.sqrt(betas[t]) * np.asarray(z, dtype=float)
`,
        hints: ['With w = 1 guidance returns the conditional prediction; with w = 0 the unconditional one.', 'Compute alpha_bar_t with `np.cumprod(1 - betas)[t]`.'],
        explain: `A sampler is just this step in a loop from t = T − 1 down to 0, with the network (and guidance) providing eps_hat at every step. The final step adds no noise so the output is the clean estimate.`,
        tests: [
          { call: 'cfg(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 3.0)', expect: 'np.array([3.0, 1.0])' },
          { name: 'last step adds no noise', call: 'ddpm_step(np.array([1.0]), 0, np.array([0.5]), np.array([0.19]), np.array([100.0]))', expect: 'np.array([(1.0 - 0.19 / np.sqrt(0.19) * 0.5) / np.sqrt(0.81)])' },
          { name: 'random step vs reference', setup: 'rng = np.random.default_rng(0)\nb = np.linspace(1e-4, 0.02, 50); x = rng.normal(size=4); e = rng.normal(size=4); z = rng.normal(size=4)', call: 'ddpm_step(x, 30, e, b, z)', hidden: true },
          { name: 'guidance with w = 1 is the conditional prediction', setup: 'rng = np.random.default_rng(1)\nu = rng.normal(size=6); c = rng.normal(size=6)', code: 'assert np.allclose(cfg(u, c, 1.0), c) and np.allclose(cfg(u, c, 0.0), u)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Ho, Jain & Abbeel — Denoising Diffusion Probabilistic Models', url: 'https://arxiv.org/abs/2006.11239' },
      { title: 'Rombach et al. — High-Resolution Image Synthesis with Latent Diffusion Models', url: 'https://arxiv.org/abs/2112.10752' },
      { title: 'Ho & Salimans — Classifier-Free Diffusion Guidance', url: 'https://arxiv.org/abs/2207.12598' },
      { title: 'Lipman et al. — Flow Matching for Generative Modeling', url: 'https://arxiv.org/abs/2210.02747' },
      { title: 'Weng — What are Diffusion Models?', url: 'https://lilianweng.github.io/posts/2021-07-11-diffusion-models/' }
    ]
  },

  // =====================================================================================
  {
    id: 'deep-rl', branch: 'spec', title: 'Deep Reinforcement Learning: DQN, Actor–Critic & PPO', short: 'Deep RL', level: 3, minutes: 50, prereqs: ['reinforcement-learning', 'mlp'],
    tags: ['deep reinforcement learning', 'dqn', 'experience replay', 'target network', 'double dqn', 'policy gradient', 'reinforce', 'baseline', 'advantage', 'actor-critic', 'gae', 'ppo', 'clipped objective', 'entropy bonus', 'model-based rl', 'offline rl'],
    summary: 'Reinforcement learning with neural networks: why function approximation is unstable, DQN with experience replay and target networks (and Double DQN against overestimation), the policy-gradient theorem with baselines and advantages, actor–critic methods with generalized advantage estimation, and PPO\'s clipped objective — the algorithm behind RLHF — plus a map of the wider landscape.',
    why: `Tabular Q-learning works for gridworlds, but real problems — games from pixels, robots, chip placement, data-centre cooling, and fine-tuning language models — have far too many states for a table. **Deep RL** replaces the table with neural networks, which brings powerful generalization and notorious instability. The tricks that make it work (replay buffers, target networks, baselines, advantage estimates, clipped updates) are exactly the ones you need to understand RLHF and RL-trained reasoning models.`,
    objectives: [
      'Explain why RL with function approximation is unstable and how DQN\'s replay buffer and target network help',
      'Show why max-based targets overestimate values and how Double DQN fixes it',
      'Derive REINFORCE with a baseline and explain advantages, actor–critic and GAE',
      'Explain PPO\'s clipped objective and connect it to RLHF and GRPO'
    ],
    sections: [
      { type: 'text', title: 'From tables to function approximation', md: `In *Reinforcement Learning* you stored one Q-value per (state, action). With images as states there are astronomically many states, and most are never seen twice. A neural network Q_θ(s, a) **generalizes** across similar states — but RL training data is unlike supervised data:

- consecutive experiences are strongly **correlated**;
- the **targets move**, because they are computed from the network being trained;
- the data distribution changes as the policy improves.

The combination of function approximation, bootstrapping (targets built from estimates) and off-policy learning is called the **deadly triad**, and it can make values diverge. Deep RL algorithms are largely collections of fixes for this.` },
      { type: 'text', title: 'DQN: Q-learning with a neural network', md: `**Deep Q-Networks** (Mnih et al., 2015) learned to play Atari games from pixels with two stabilizers:

- an **experience replay buffer** stores past transitions (s, a, r, s′, done); training samples random mini-batches from it, breaking correlations and reusing data;
- a **target network** — a periodically updated copy of the Q-network — computes the targets, so they don't shift with every gradient step.

$$ y = r + γ · (1 − done) · max_{a′} Q_target(s′, a′)          L = ( y − Q_θ(s, a) )² $$
:::where
r — the reward received after taking action a in state s
γ — "gamma": the discount factor
done — 1 if the episode ended at s′ (no future value), else 0
Q_target — the slowly updated target network
Q_θ — the network being trained
:::

Taking the max over *noisy* estimates is biased upwards: some action's error will be positive. **Double DQN** decouples the two roles — the online network chooses the action, the target network evaluates it: y = r + γ·Q_target(s′, argmax_a′ Q_θ(s′, a′)).` },
      { type: 'code', title: 'Why max-based targets overestimate', code: `
import numpy as np
rng = np.random.default_rng(0)
n_states, n_actions = 10000, 10
true_q = np.zeros((n_states, n_actions))                        # every action is truly worth exactly 0
online = true_q + rng.normal(0, 1, true_q.shape)                # two independently noisy estimates
target = true_q + rng.normal(0, 1, true_q.shape)

dqn = target.max(axis=1)                                        # DQN: max over the noisy target network
double = target[np.arange(n_states), online.argmax(axis=1)]     # Double DQN: online picks, target evaluates
print("true value of the best action:          0.000")
print(f"DQN target (max of noisy estimates):   {dqn.mean():+.3f}")
print(f"Double DQN target (decoupled):         {double.mean():+.3f}")
`, after: `Although every action is worth 0, the maximum of ten noisy estimates averages about +1.5, and bootstrapping feeds that optimism back into the next targets. Choosing with one estimate and evaluating with an independent one removes the bias.` },
      { type: 'text', title: 'Policy gradients: REINFORCE with a baseline', md: `Instead of learning values and acting greedily, **policy-gradient** methods parameterize the policy π_θ(a | s) directly (useful for continuous or stochastic actions) and climb the gradient of the expected return:

$$ ∇_θ J = E[ ∇_θ log π_θ(a_t | s_t) · ( G_t − b(s_t) ) ] $$
:::where
J — the expected return of the policy
π_θ(a | s) — the probability the policy gives action a in state s
∇_θ log π_θ — the direction that makes the taken action more likely
G_t — the return actually received from step t onward
b(s_t) — a baseline, for example an estimate of the state's value; subtracting it leaves the gradient unbiased but lowers its variance
:::

**REINFORCE** uses this estimate directly from complete episodes: actions followed by better-than-expected returns become more likely. G_t − b(s_t) estimates the **advantage** A(s, a) = Q(s, a) − V(s): how much better the action was than the policy's average in that state.` },
      { type: 'code', title: 'REINFORCE on a bandit, with and without a baseline', code: `
import numpy as np
true_means = np.array([1.0, 1.5, 2.0])                         # the expected reward of each of three arms

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def reinforce(use_baseline, seed, steps=2000, lr=0.05):
    rng = np.random.default_rng(seed)
    theta, b = np.zeros(3), 0.0
    rewards, grad_norms = [], []
    for t in range(steps):
        p = softmax(theta)
        a = rng.choice(3, p=p)
        r = true_means[a] + rng.normal(0, 1.0)
        grad_log_pi = -p
        grad_log_pi[a] += 1                                     # gradient of log softmax for the chosen arm
        advantage = r - (b if use_baseline else 0.0)
        theta += lr * grad_log_pi * advantage
        b += 0.05 * (r - b)                                     # running average reward as the baseline
        rewards.append(r)
        grad_norms.append(np.linalg.norm(grad_log_pi * advantage))
    return np.mean(rewards[-500:]), np.var(grad_norms[:500]), softmax(theta)[2]

for use_baseline in (False, True):
    runs = [reinforce(use_baseline, seed) for seed in range(5)]
    print(f"baseline={str(use_baseline):5s}  reward in the last 500 steps {np.mean([r[0] for r in runs]):.3f}   "
          f"variance of early gradient sizes {np.mean([r[1] for r in runs]):.3f}   P(best arm) {np.mean([r[2] for r in runs]):.3f}")
`, after: `Both learners find the best arm in this easy problem, but subtracting a baseline cuts the variance of the gradient estimates by roughly 40%. With many actions, long episodes and neural networks, that variance reduction is what makes policy gradients practical.` },
      { type: 'text', title: 'Actor–critic and generalized advantage estimation', md: `**Actor–critic** methods learn two networks: the **actor** (the policy) and the **critic** V_φ(s), which estimates state values and supplies the baseline. Instead of waiting for full returns, they use the **temporal-difference error**, and **GAE** blends errors over several steps:

$$ δ_t = r_t + γ V(s_{t+1}) − V(s_t)          Â_t = Σ_{l ≥ 0} (γλ)^l · δ_{t+l} $$
:::where
δ_t — "delta": the one-step surprise, how much better things went than the critic expected
V — the critic's value estimate
λ — "lambda" between 0 and 1: λ = 0 uses only the one-step error (low variance, more bias), λ = 1 the full return (unbiased, high variance)
Â_t — the advantage estimate used to weight the policy gradient
:::

Typical settings are γ = 0.99 and λ = 0.95. At the end of an episode the future terms are zero.` },
      { type: 'text', title: 'PPO: small, safe policy steps', md: `A policy-gradient step that is too large can wreck the policy, and the data it would need to recover was collected by the old policy. **Proximal Policy Optimization** (Schulman et al., 2017) reuses each batch of experience for several epochs of mini-batch updates, but clips the objective so the new policy cannot move far from the old one:

$$ r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)          L = E[ min( r_t · Â_t,  clip(r_t, 1 − ε, 1 + ε) · Â_t ) ] $$
:::where
r_t(θ) — the probability ratio between the new and the old policy for the action taken
Â_t — the advantage estimate (e.g. from GAE)
ε — the clipping range, typically 0.2
min — takes the more pessimistic of the clipped and unclipped objectives
:::

The full loss adds a value-function loss for the critic and an **entropy bonus** that keeps exploring. PPO is robust and simple, which is why it became the default — including for **RLHF**, where the "environment" is a prompt, the action is a generated response, the reward comes from a reward model, and a KL penalty keeps the policy near the reference model (see *RLHF, DPO & Alignment*). **GRPO** (see *Reasoning Models*) drops the critic and computes advantages relative to a group of responses to the same prompt.` },
      { type: 'code', title: 'What the clipped objective does', code: `
import numpy as np
eps = 0.2
ratios = np.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0])     # new / old probability of the action taken
for adv in (+1.0, -1.0):
    obj = np.minimum(ratios * adv, np.clip(ratios, 1 - eps, 1 + eps) * adv)
    print(f"advantage {adv:+.0f}:  " + "  ".join(f"r={r:.1f}: {o:+.2f}" for r, o in zip(ratios, obj)))
`, after: `For a good action (A > 0) the objective stops rising once the ratio exceeds 1.2, so there is no incentive to push its probability further in one update. For a bad action (A < 0) it is flat below 0.8, but taking the minimum keeps penalizing ratios above 1: PPO never ignores an update that would make a bad action more likely.` },
      { type: 'demo', title: 'Interactive: the PPO clipped objective', demo: 'ppo-clip' },
      { type: 'check', question: { type: 'mcq', q: 'What is the purpose of the target network in DQN?', options: ['To choose exploratory actions', 'To compute training targets with weights that change slowly, so the targets do not chase the network', 'To store the replay buffer', 'To replace the reward function'], answer: 1, explain: 'Stable targets reduce oscillation and divergence.' } },
      { type: 'text', title: 'The wider landscape', md: `- **Continuous control**: off-policy actor–critic methods such as **SAC** (with an entropy bonus) and **TD3** are sample-efficient on robots and simulators.
- **Model-based RL** learns a model of the environment and plans with it or trains inside it: MuZero (planning with a learned model), Dreamer (learning in imagination).
- **Offline RL** learns from a fixed log of past interactions without exploring, using conservative methods that avoid over-trusting actions the data never tried.
- **Exploration** bonuses (curiosity, novelty) help when rewards are sparse.
- **Sim-to-real** transfer trains robots in simulation with randomized physics before deploying them.
- **RL for language models**: RLHF with PPO, direct preference methods (DPO), and RL with verifiable rewards for reasoning.` }
    ],
    when: `- Sequential decision problems where actions affect future states: control, robotics, games, resource allocation, recommendation sequences.
- Fine-tuning generative models towards rewards that are easy to judge but hard to demonstrate (RLHF, RL from verifiable rewards).
- When a simulator exists (cheap experience); offline RL when only logs exist.
- Prefer simpler alternatives (supervised learning, bandits, planning) when they fit — deep RL is data-hungry and fragile.`,
    pitfalls: `- Tiny bugs (wrong done handling, unnormalized observations or advantages) that silently break learning.
- Reward hacking: agents optimize what you measure, not what you meant.
- Judging an algorithm on one random seed — results vary a lot between seeds.
- Learning rates and clip ranges set too aggressively, causing policy collapse.
- Forgetting that the value targets need (1 − done) at episode ends.`,
    takeaways: [
      'Neural function approximation + bootstrapping + off-policy data can diverge (the deadly triad).',
      'DQN stabilizes Q-learning with experience replay and a target network; Double DQN removes max-overestimation.',
      'Policy gradients: ∇J = E[∇log π · (G − b)]; a baseline cuts variance without bias.',
      'Actor–critic methods use a learned value as baseline and GAE to trade bias for variance.',
      'PPO clips the probability ratio for safe updates and is the workhorse of RLHF; GRPO drops the critic.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why does DQN sample random mini-batches from a replay buffer?', options: ['To save memory', 'To break the correlation between consecutive experiences and reuse data', 'To avoid using rewards', 'To make the policy deterministic'], answer: 1, explain: 'Gradient descent assumes roughly independent samples.' },
      { type: 'tf', q: 'Subtracting a baseline from the return makes the policy-gradient estimate biased.', answer: false, explain: 'Its expected contribution is zero; it only lowers variance.' },
      { type: 'fill', q: 'With γ = 0.9, reward 1, V(s) = 2 and V(s′) = 3, what is the TD error δ = r + γV(s′) − V(s)?', answer: ['1.7'], explain: '1 + 0.9 × 3 − 2 = 1.7.' },
      { type: 'mcq', q: 'With ε = 0.2 and a positive advantage, what is the PPO objective for a ratio of 1.5 (with A = 1)?', options: ['1.5', '1.2', '0.8', '1.0'], answer: 1, explain: 'min(1.5, clip(1.5, 0.8, 1.2) = 1.2) = 1.2.' },
      { type: 'mcq', q: 'How is PPO used in RLHF?', options: ['To train the tokenizer', 'To fine-tune the language model on reward-model scores, with a KL penalty to the reference model', 'To compress the model', 'To collect human labels'], answer: 1, explain: 'The response is the action and the reward model scores it.' }
    ],
    problems: [
      {
        id: 'gae-returns', title: 'Returns and Generalized Advantage Estimation', difficulty: 'medium',
        prompt: `For one trajectory of length T implement (dones[t] = 1 means the episode ended after step t, so nothing after it counts):

- \`discounted_returns(rewards, gamma, dones=None)\` → G_t = r_t + γ·(1 − done_t)·G_{t+1}, computed backwards, with G_T = 0.
- \`gae(rewards, values, gamma, lam, last_value=0.0, dones=None)\` → the advantages Â_t computed backwards: δ_t = r_t + γ·(1 − done_t)·V_{t+1} − V_t (with V_T = last_value) and Â_t = δ_t + γ·λ·(1 − done_t)·Â_{t+1} (with Â_T = 0).

When dones is None, treat every done as 0. Return NumPy arrays.`,
        starter: `
import numpy as np

def discounted_returns(rewards, gamma, dones=None):
    # TODO
    pass

def gae(rewards, values, gamma, lam, last_value=0.0, dones=None):
    # TODO
    pass
`,
        solution: `
import numpy as np

def discounted_returns(rewards, gamma, dones=None):
    r = np.asarray(rewards, dtype=float)
    d = np.zeros(len(r)) if dones is None else np.asarray(dones, dtype=float)
    G = np.zeros(len(r))
    running = 0.0
    for t in range(len(r) - 1, -1, -1):
        running = r[t] + gamma * (1 - d[t]) * running
        G[t] = running
    return G

def gae(rewards, values, gamma, lam, last_value=0.0, dones=None):
    r = np.asarray(rewards, dtype=float)
    v = np.asarray(values, dtype=float)
    d = np.zeros(len(r)) if dones is None else np.asarray(dones, dtype=float)
    adv = np.zeros(len(r))
    next_value, running = last_value, 0.0
    for t in range(len(r) - 1, -1, -1):
        delta = r[t] + gamma * (1 - d[t]) * next_value - v[t]
        running = delta + gamma * lam * (1 - d[t]) * running
        adv[t] = running
        next_value = v[t]
    return adv
`,
        hints: ['Loop backwards from the last step and carry the running sum.', 'Multiplying by (1 − done) cuts the chain at episode boundaries.'],
        explain: `With λ = 1 and a zero last value, GAE advantages equal returns minus values; with λ = 0 they are one-step TD errors. The done mask lets one array hold several episodes back to back, as rollout buffers do.`,
        tests: [
          { call: 'discounted_returns([1.0, 1.0, 1.0], 0.5)', expect: 'np.array([1.75, 1.5, 1.0])' },
          { name: 'an episode boundary', call: 'discounted_returns([1.0, 2.0, 3.0], 0.9, dones=[0, 1, 0])', expect: 'np.array([1.0 + 0.9 * 2.0, 2.0, 3.0])' },
          { call: 'gae([1.0, 0.0], [0.5, 0.5], 1.0, 0.0)', expect: 'np.array([1.0, -0.5])' },
          { name: 'lambda = 1 gives returns minus values', setup: 'rng = np.random.default_rng(0)\nr = rng.normal(size=6); v = rng.normal(size=6)', code: 'assert np.allclose(gae(r, v, 0.9, 1.0), discounted_returns(r, 0.9) - v)', hidden: true },
          { name: 'random rollout vs reference', setup: 'rng = np.random.default_rng(1)\nr = rng.normal(size=8); v = rng.normal(size=8); d = np.array([0, 0, 1, 0, 0, 0, 1, 0])', call: 'gae(r, v, 0.99, 0.95, last_value=0.7, dones=d)', hidden: true }
        ]
      },
      {
        id: 'ppo-dqn-targets', title: 'PPO Objective and DQN Targets', difficulty: 'medium',
        prompt: `Implement (arrays are batches):

- \`ppo_objective(ratio, adv, eps=0.2)\` → the mean of min(ratio·adv, clip(ratio, 1 − eps, 1 + eps)·adv).
- \`dqn_targets(rewards, next_q, dones, gamma)\` → r + γ·(1 − done)·max_a next_q[i, a], where next_q is (batch, actions).
- \`double_dqn_targets(rewards, next_q_online, next_q_target, dones, gamma)\` → r + γ·(1 − done)·next_q_target[i, argmax_a next_q_online[i, a]].`,
        starter: `
import numpy as np

def ppo_objective(ratio, adv, eps=0.2):
    # TODO
    pass

def dqn_targets(rewards, next_q, dones, gamma):
    # TODO
    pass

def double_dqn_targets(rewards, next_q_online, next_q_target, dones, gamma):
    # TODO
    pass
`,
        solution: `
import numpy as np

def ppo_objective(ratio, adv, eps=0.2):
    r, a = np.asarray(ratio, dtype=float), np.asarray(adv, dtype=float)
    return float(np.mean(np.minimum(r * a, np.clip(r, 1 - eps, 1 + eps) * a)))

def dqn_targets(rewards, next_q, dones, gamma):
    r, q, d = (np.asarray(v, dtype=float) for v in (rewards, next_q, dones))
    return r + gamma * (1 - d) * q.max(axis=1)

def double_dqn_targets(rewards, next_q_online, next_q_target, dones, gamma):
    r, qo, qt, d = (np.asarray(v, dtype=float) for v in (rewards, next_q_online, next_q_target, dones))
    best = qo.argmax(axis=1)
    return r + gamma * (1 - d) * qt[np.arange(len(r)), best]
`,
        hints: ['`np.clip(r, 1 - eps, 1 + eps)` and `np.minimum` give the clipped objective element-wise.', 'Index the target network\'s values with the online network\'s argmax.'],
        explain: `These few lines are the heart of two of the most used deep-RL algorithms. In real implementations the PPO objective is maximized (its negative minimized) together with a value loss and an entropy bonus.`,
        tests: [
          { call: 'ppo_objective(np.array([1.5, 0.5]), np.array([1.0, -1.0]))', expect: '(1.2 + (-0.8)) / 2' },
          { call: 'dqn_targets(np.array([1.0, 0.0]), np.array([[1.0, 3.0], [2.0, 0.0]]), np.array([0.0, 1.0]), 0.9)', expect: 'np.array([1.0 + 0.9 * 3.0, 0.0])' },
          { call: 'double_dqn_targets(np.array([0.0]), np.array([[5.0, 1.0]]), np.array([[0.5, 2.0]]), np.array([0.0]), 1.0)', expect: 'np.array([0.5])' },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(0)\nr = rng.normal(size=5); qo = rng.normal(size=(5, 3)); qt = rng.normal(size=(5, 3)); d = (rng.random(5) < 0.3).astype(float)', call: '[dqn_targets(r, qt, d, 0.99), double_dqn_targets(r, qo, qt, d, 0.99), ppo_objective(rng.uniform(0.5, 1.5, 5), r, 0.1)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Mnih et al. — Human-level control through deep reinforcement learning (DQN)', url: 'https://www.nature.com/articles/nature14236' },
      { title: 'Schulman et al. — Proximal Policy Optimization Algorithms', url: 'https://arxiv.org/abs/1707.06347' },
      { title: 'Schulman et al. — High-Dimensional Continuous Control Using Generalized Advantage Estimation', url: 'https://arxiv.org/abs/1506.02438' },
      { title: 'OpenAI — Spinning Up in Deep RL', url: 'https://spinningup.openai.com/' }
    ]
  },

  // =====================================================================================
  {
    id: 'ml-security', branch: 'spec', title: 'Adversarial ML, Security & Privacy', short: 'ML Security & Privacy', level: 3, minutes: 45, prereqs: ['responsible-ai', 'backpropagation'],
    tags: ['adversarial examples', 'fgsm', 'pgd', 'adversarial training', 'robustness', 'data poisoning', 'backdoor', 'model extraction', 'membership inference', 'differential privacy', 'dp-sgd', 'federated learning', 'prompt injection', 'jailbreak', 'threat model'],
    summary: 'Attacks on machine-learning systems and how to defend against them: threat models, adversarial examples (FGSM, PGD) and adversarial training, data poisoning and backdoors, membership inference and training-data extraction, differential privacy with DP-SGD, federated learning, and the security of LLM applications against prompt injection and jailbreaks.',
    why: `Models are software that learns from data — and both the software and the data can be attacked. Tiny, invisible changes can flip an image classifier's decision; a few poisoned training examples can plant a backdoor; a model can leak whether a person's record was in its training set; and an LLM agent can be hijacked by instructions hidden in a web page it reads. Anyone deploying ML needs to know these threats and the defences that actually work.`,
    objectives: [
      'Describe threat models by attacker goal, knowledge and access',
      'Create adversarial examples with FGSM and explain why linear models in high dimensions are vulnerable',
      'Explain poisoning, backdoors, membership inference and model extraction, and their defences',
      'Define differential privacy, describe DP-SGD and federated learning, and secure LLM applications'
    ],
    sections: [
      { type: 'text', title: 'Threat models', md: `Security starts by saying who the attacker is:

| Stage | Attack | Attacker wants |
| --- | --- | --- |
| Training data | **poisoning**, **backdoors** | to corrupt the model or plant a hidden trigger |
| Inference | **evasion** (adversarial examples), jailbreaks | wrong outputs for chosen inputs |
| Inference | **membership inference**, data extraction | private training data |
| Inference | **model extraction** | to copy the model through its API |
| Application | **prompt injection** | to make an LLM system follow the attacker's instructions |

Also specify the attacker's **knowledge** (white-box: weights and gradients; black-box: only queries) and **access** (one query or millions; control over a few training examples or none). A defence is only meaningful against a stated threat model.` },
      { type: 'text', title: 'Adversarial examples', md: `An adversarial example is an input changed slightly — often imperceptibly — so that the model gets it wrong. The **fast gradient sign method** takes one step that increases the loss as much as possible within a per-feature budget ε:

$$ x′ = x + ε · sign( ∇ₓ L(f(x), y) ) $$
:::where
x — the original input; x′ — the adversarial input
∇ₓ L — the gradient of the loss with respect to the **input** (not the weights)
sign — keeps only the direction (+1 or −1) of each component
ε — "epsilon": the largest change allowed to any single feature (an L∞ budget)
:::

**PGD** repeats small FGSM steps, projecting back into the ε-ball after each, and is the standard strong attack. Why are models so fragile? Even for a linear score wᵀx, the change ε·sign(w) shifts the score by ε·‖w‖₁ — a sum over every input dimension. With thousands of pixels, many imperceptible changes add up to a large effect.` },
      { type: 'code', title: 'FGSM against a linear classifier', code: `
import numpy as np, warnings
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

digits = load_digits()
X, y = digits.data / 16.0, digits.target                      # pixel values scaled to [0, 1]
keep = (y == 3) | (y == 8)                                     # a binary task: 3 versus 8
X, y = X[keep], (y[keep] == 8).astype(int)
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
train, test = order[:250], order[250:]
clf = LogisticRegression(max_iter=2000).fit(X[train], y[train])
w, b = clf.coef_[0], clf.intercept_[0]

def fgsm(Xs, ys, eps):
    p = 1 / (1 + np.exp(-(Xs @ w + b)))
    grad = (p - ys)[:, None] * w[None, :]                       # gradient of the log loss with respect to the pixels
    return np.clip(Xs + eps * np.sign(grad), 0, 1)              # one signed step, kept in the valid pixel range

for eps in (0.0, 0.05, 0.1, 0.2):
    Xa = fgsm(X[test], y[test], eps)
    print(f"eps = {eps:.2f}: accuracy {clf.score(Xa, y[test]):.3f}   average pixel change {np.abs(Xa - X[test]).mean():.3f}")
`, after: `The classifier is perfect on clean test digits, yet changing each pixel by at most 0.1 (on a 0–1 scale) pulls accuracy down to about 74%, and 0.2 to below 20% — worse than guessing.` },
      { type: 'text', title: 'Defences against evasion', md: `- **Adversarial training** — train on adversarial examples generated on the fly (usually with PGD) — is the most reliable empirical defence, at the cost of extra compute and some clean accuracy.
- **Certified defences** such as randomized smoothing give provable guarantees within a radius.
- **Input preprocessing** and "gradient masking" tricks have repeatedly been broken by adaptive attacks.
- Evaluate robustness with **adaptive** attacks designed against the specific defence (and with standard suites such as AutoAttack), never only with the attack the defence was tuned for.` },
      { type: 'text', title: 'Poisoning and backdoors', md: `If attackers can influence the training data, they can **poison** it: flip labels to degrade a model, or plant a **backdoor** — training examples with a small trigger (a pixel patch, a rare phrase) labelled with the attacker's target class. The model behaves normally until the trigger appears. Web-scale datasets are especially exposed: researchers showed it is practical to poison them, for example by buying expired domains that datasets still link to (Carlini et al., 2023).

Defences: data provenance and integrity checks (hashes of crawled content), deduplication and filtering, anomaly detection on training examples and on activations, robust training, and careful vetting of third-party models and datasets — a downloaded checkpoint can contain a backdoor, and pickle-based model files can execute code when loaded (prefer formats such as safetensors).` },
      { type: 'text', title: 'Privacy attacks and differential privacy', md: `Models can leak their training data:

- **Membership inference**: was this record in the training set? Overfit models have noticeably lower loss on their training examples, which an attacker can threshold.
- **Training-data extraction**: large language models can regurgitate memorized text, including personal data, when prompted the right way.
- **Model extraction**: enough API queries can train a functional copy of a model.

**Differential privacy (DP)** gives a mathematical guarantee: the output changes very little whether or not any one person's data is included:

$$ P( M(D) ∈ S ) ≤ e^ε · P( M(D′) ∈ S ) + δ $$
:::where
M — the training algorithm (it is randomized)
D, D′ — any two datasets that differ in one person's record
S — any set of possible outputs (trained models)
ε — "epsilon": the privacy budget; smaller means stronger privacy
δ — a tiny probability of the guarantee failing
:::

**DP-SGD** achieves it for neural networks: compute each example's gradient, **clip** it to a maximum norm C (so no individual can have a large influence), add Gaussian noise with standard deviation σ·C to the sum, then average and update. A privacy accountant tracks the total ε spent over training. The price is lower accuracy, especially on rare groups.

**Federated learning** trains across many devices that share model updates instead of raw data (FedAvg averages the clients' updates). It is not private on its own — updates can leak data — so it is combined with secure aggregation and DP.` },
      { type: 'code', title: 'Membership inference from loss values', code: `
import numpy as np, warnings
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

digits = load_digits()
X, y = digits.data / 16.0, digits.target
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
members, outsiders = order[:600], order[600:1200]                # trained on vs never seen

def attack(model):
    model.fit(X[members], y[members])
    def losses(idx):
        p = model.predict_proba(X[idx])[np.arange(len(idx)), y[idx]]
        return -np.log(np.clip(p, 1e-12, 1))
    lm, lo = losses(members), losses(outsiders)
    best = max(0.5 * (np.mean(lm <= t) + np.mean(lo > t)) for t in np.unique(np.r_[lm, lo]))   # best loss threshold
    return model.score(X[members], y[members]), model.score(X[outsiders], y[outsiders]), best

for name, model in [("fully grown decision tree", DecisionTreeClassifier(random_state=0)),
                    ("regularized logistic regression", LogisticRegression(C=0.1, max_iter=2000))]:
    train_acc, test_acc, attack_acc = attack(model)
    print(f"{name:32s} train acc {train_acc:.3f}  test acc {test_acc:.3f}  ->  membership attack accuracy {attack_acc:.3f} (0.5 = guessing)")
`, after: `The overfit tree is perfect on its training data and much worse elsewhere, and a simple loss threshold tells members from outsiders about 60% of the time. The regularized model generalizes and barely leaks. Overfitting is a privacy problem, not just an accuracy problem.` },
      { type: 'check', question: { type: 'tf', q: 'Federated learning on its own guarantees that no information about a user\'s data can leak.', answer: false, explain: 'Model updates can leak data; combine it with secure aggregation and differential privacy.' } },
      { type: 'text', title: 'Securing LLM applications', md: `LLM systems add new attack surfaces (see *LLM Agents & Tool Use*):

- **Prompt injection**: instructions hidden in user input or — worse — in content the model reads (web pages, emails, documents, tool results: *indirect* injection) that try to override the developer's instructions.
- **Jailbreaks**: prompts crafted to bypass a model's safety training.
- **Data exfiltration** through tools, links or images the model is tricked into producing; **insecure output handling** when model output is executed or rendered without validation.

Defences in depth: treat all model inputs from outside as untrusted data, never as instructions; give tools **least privilege** and require human confirmation for consequential actions; separate and label untrusted content; validate and sanitize outputs before executing or rendering them; monitor, rate-limit and log; and red-team the system regularly. The OWASP Top 10 for LLM Applications is a good checklist.` }
    ],
    when: `- Any model exposed to adversaries: fraud and spam detection, content moderation, malware detection, biometrics, autonomous systems.
- Training on personal or sensitive data: assess membership and extraction risk; consider DP.
- Using third-party data, checkpoints or web-scale crawls: provenance, scanning and safe file formats.
- LLM agents with tools or access to private data: injection-resistant design and least privilege.`,
    pitfalls: `- Claiming robustness from tests against weak or non-adaptive attacks.
- Treating federated learning or "anonymized" data as private by default.
- Loading untrusted pickle-based model files (they can execute code).
- Letting an LLM agent act on instructions found in documents or web pages.
- Ignoring the accuracy and fairness costs of DP and adversarial training.`,
    takeaways: [
      'Define the threat model: goal, knowledge and access of the attacker.',
      'FGSM: x′ = x + ε·sign(∇ₓL); many tiny changes add up in high dimensions. Adversarial training is the main defence.',
      'Poisoning and backdoors attack the training data; provenance, filtering and vetting are the defences.',
      'Overfit models leak membership; DP-SGD (clip per-example gradients, add noise) gives provable privacy.',
      'LLM apps: treat external content as data, not instructions; least-privilege tools; validate outputs.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does FGSM use to build an adversarial example?', options: ['The gradient of the loss with respect to the weights', 'The sign of the gradient of the loss with respect to the input', 'Random noise', 'The training labels only'], answer: 1, explain: 'One signed step on the input within an ε budget.' },
      { type: 'fill', q: 'For a linear score wᵀx, by how much does the change ε·sign(w) shift the score? Give the expression in terms of ε and w.', answer: ['ε‖w‖₁', 'eps * ||w||_1', 'ε·‖w‖₁', 'ε||w||1'], explain: 'Σ ε·|w_i| = ε‖w‖₁, a sum over all dimensions.' },
      { type: 'mcq', q: 'Which two operations does DP-SGD add to ordinary SGD?', options: ['Dropout and batch normalization', 'Clipping each example\'s gradient and adding Gaussian noise', 'Larger batches and more epochs', 'Pruning and quantization'], answer: 1, explain: 'Clipping bounds each person\'s influence; noise hides it.' },
      { type: 'mcq', q: 'What is indirect prompt injection?', options: ['A user typing a rude prompt', 'Malicious instructions hidden in content the LLM reads, such as a web page or email', 'A bug in the tokenizer', 'Overfitting to prompts'], answer: 1, explain: 'The attacker never talks to the model directly.' },
      { type: 'tf', q: 'A model that overfits its training data is more vulnerable to membership inference.', answer: true, explain: 'Its loss on training members is distinctly lower than on outsiders.' }
    ],
    problems: [
      {
        id: 'fgsm-attack', title: 'FGSM Against Logistic Regression', difficulty: 'easy',
        prompt: `For logistic regression p = σ(w·x + b) with the log loss, implement:

- \`input_gradient(x, y, w, b)\` → the gradient of the loss with respect to the **input** x: (σ(w·x + b) − y) · w (x is one example, y is 0 or 1).
- \`fgsm(x, grad, eps, lo=0.0, hi=1.0)\` → clip(x + eps · sign(grad), lo, hi).
- \`linf_norm(delta)\` → the largest absolute entry of delta.`,
        starter: `
import numpy as np

def input_gradient(x, y, w, b):
    # TODO
    pass

def fgsm(x, grad, eps, lo=0.0, hi=1.0):
    # TODO
    pass

def linf_norm(delta):
    # TODO
    pass
`,
        solution: `
import numpy as np

def input_gradient(x, y, w, b):
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    p = 1 / (1 + np.exp(-(w @ x + b)))
    return (p - y) * w

def fgsm(x, grad, eps, lo=0.0, hi=1.0):
    return np.clip(np.asarray(x, dtype=float) + eps * np.sign(grad), lo, hi)

def linf_norm(delta):
    return float(np.max(np.abs(delta)))
`,
        hints: ['The loss gradient with respect to the score is p − y; the score\'s gradient with respect to x is w.', '`np.sign` gives −1, 0 or +1 for each entry.'],
        explain: `The attack needs only the gradient's direction. Clipping keeps the result a valid input (e.g. pixel values in [0, 1]); the L∞ norm of the change is at most eps.`,
        tests: [
          { call: 'input_gradient(np.array([0.0, 0.0]), 1, np.array([2.0, -1.0]), 0.0)', expect: 'np.array([-1.0, 0.5])' },
          { call: 'fgsm(np.array([0.5, 0.95, 0.02]), np.array([1.0, 1.0, -3.0]), 0.1)', expect: 'np.array([0.6, 1.0, 0.0])' },
          { call: 'linf_norm(np.array([0.1, -0.3, 0.2]))', expect: '0.3' },
          { name: 'the attack increases the loss', setup: 'rng = np.random.default_rng(0)\nw = rng.normal(size=10); b = 0.1; x = rng.uniform(0.2, 0.8, 10); y = 1\nloss = lambda v: np.log(1 + np.exp(-(w @ v + b)))', code: 'xa = fgsm(x, input_gradient(x, y, w, b), 0.05)\nassert loss(xa) > loss(x)\nassert linf_norm(xa - x) <= 0.05 + 1e-12', hidden: true }
        ]
      },
      {
        id: 'dp-sgd-step', title: 'A DP-SGD Step and a Membership Attack', difficulty: 'medium',
        prompt: `Implement:

- \`clip_rows(G, C)\` → scale each row g of the per-example gradient matrix G (batch, params) by min(1, C / ‖g‖) so no row is longer than C (leave zero rows unchanged).
- \`dp_sgd_update(w, G, C, sigma, lr, noise)\` → w − lr · (Σ clipped rows + sigma · C · noise) / batch_size, where \`noise\` is a given standard-normal vector shaped like w (passed in so results are reproducible).
- \`membership_advantage(member_losses, outsider_losses)\` → the best balanced accuracy of the rule "member if loss ≤ t" over all thresholds t that appear in either list: max_t ½·(fraction of members with loss ≤ t + fraction of outsiders with loss > t); never less than 0.5.`,
        starter: `
import numpy as np

def clip_rows(G, C):
    # TODO
    pass

def dp_sgd_update(w, G, C, sigma, lr, noise):
    # TODO
    pass

def membership_advantage(member_losses, outsider_losses):
    # TODO
    pass
`,
        solution: `
import numpy as np

def clip_rows(G, C):
    G = np.asarray(G, dtype=float)
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    scale = np.minimum(1.0, C / np.where(norms > 0, norms, 1.0))
    return G * scale

def dp_sgd_update(w, G, C, sigma, lr, noise):
    G = clip_rows(G, C)
    total = G.sum(axis=0) + sigma * C * np.asarray(noise, dtype=float)
    return np.asarray(w, dtype=float) - lr * total / len(G)

def membership_advantage(member_losses, outsider_losses):
    lm, lo = np.asarray(member_losses, dtype=float), np.asarray(outsider_losses, dtype=float)
    best = 0.5
    for t in np.unique(np.r_[lm, lo]):
        best = max(best, 0.5 * (np.mean(lm <= t) + np.mean(lo > t)))
    return float(best)
`,
        hints: ['Row norms: `np.linalg.norm(G, axis=1, keepdims=True)`.', 'Add the noise to the summed gradient before dividing by the batch size.', 'Try every observed loss value as a threshold.'],
        explain: `Clipping bounds how much any single example can move the model, and the noise — scaled to that bound — hides whether a given example was present. The membership advantage measures exactly the kind of leakage DP is designed to limit.`,
        tests: [
          { call: 'clip_rows(np.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]]), 1.0)', expect: 'np.array([[0.6, 0.8], [0.3, 0.4], [0.0, 0.0]])' },
          { call: 'dp_sgd_update(np.zeros(2), np.array([[3.0, 4.0], [0.0, 1.0]]), 1.0, 0.5, 1.0, np.array([1.0, -1.0]))', expect: '-np.array([0.6 + 0.0 + 0.5, 0.8 + 1.0 - 0.5]) / 2' },
          { call: 'membership_advantage([0.1, 0.2], [0.5, 0.9])', expect: '1.0' },
          { name: 'identical distributions give no advantage', call: 'membership_advantage([0.3, 0.3], [0.3, 0.3])', expect: '0.5', hidden: true },
          { name: 'random values vs reference', setup: 'rng = np.random.default_rng(0)\nG = rng.normal(size=(5, 3)) * 3; w = rng.normal(size=3); z = rng.normal(size=3)\nlm = rng.exponential(0.5, 20); lo = rng.exponential(1.0, 20)', call: '[dp_sgd_update(w, G, 1.5, 1.1, 0.1, z), membership_advantage(lm, lo)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Goodfellow, Shlens & Szegedy — Explaining and Harnessing Adversarial Examples (FGSM)', url: 'https://arxiv.org/abs/1412.6572' },
      { title: 'Madry et al. — Towards Deep Learning Models Resistant to Adversarial Attacks (PGD)', url: 'https://arxiv.org/abs/1706.06083' },
      { title: 'Abadi et al. — Deep Learning with Differential Privacy (DP-SGD)', url: 'https://arxiv.org/abs/1607.00133' },
      { title: 'Shokri et al. — Membership Inference Attacks against Machine Learning Models', url: 'https://arxiv.org/abs/1610.05820' },
      { title: 'OWASP — Top 10 for Large Language Model Applications', url: 'https://owasp.org/www-project-top-10-for-large-language-model-applications/' }
    ]
  },

  // =====================================================================================
  {
    id: 'llm-serving', branch: 'llm', title: 'LLM Inference & Serving: Latency, Throughput & Cost', short: 'LLM Serving', level: 3, minutes: 45, prereqs: ['quantization', 'efficient-attention'],
    tags: ['inference', 'serving', 'prefill', 'decode', 'time to first token', 'inter-token latency', 'throughput', 'batching', 'continuous batching', 'arithmetic intensity', 'roofline', 'memory bandwidth', 'speculative decoding', 'prefix caching', 'vllm'],
    summary: 'What happens when an LLM answers many users at once: the prefill and decode phases, latency and throughput metrics, arithmetic intensity and the roofline model that explain why decoding is memory-bound, static versus continuous batching, speculative decoding and its expected speed-up, prefix caching and the other levers that set the cost per token.',
    why: `Training happens once; inference happens billions of times, so it dominates the cost of most LLM products. Why does generating text feel slow when GPUs are so fast? Why can serving 64 users cost little more than serving one? Why do providers discount cached prompts and batch jobs? The answers come from a little hardware arithmetic — FLOPs versus bytes moved — and a few serving techniques that every LLM engineer should know.`,
    objectives: [
      'Distinguish the prefill and decode phases and the metrics TTFT, inter-token latency and throughput',
      'Use arithmetic intensity and the roofline model to tell compute-bound from memory-bound work',
      'Explain continuous batching, PagedAttention and prefix caching',
      'Compute the expected speed-up of speculative decoding and estimate the cost per token'
    ],
    sections: [
      { type: 'text', title: 'Two phases: prefill and decode', md: `- **Prefill** processes the whole prompt in one parallel forward pass, filling the KV cache (see *Efficient Attention*) and producing the first output token. It is big matrix–matrix work: **compute-bound**.
- **Decode** then generates one token at a time. Every step reads all the model's weights — and the sequence's KV cache — from GPU memory to do comparatively little arithmetic: **memory-bandwidth-bound**.

User-facing metrics: **time to first token (TTFT)**, dominated by queueing and prefill; **inter-token latency (ITL)**, the time per generated token; **throughput**, total tokens per second across all users; and the **cost per million tokens**. For one request, latency ≈ TTFT + (output tokens − 1) × ITL.` },
      { type: 'text', title: 'Arithmetic intensity and the roofline', md: `A GPU has a peak compute rate (FLOP/s) and a memory bandwidth (bytes/s). Whether a computation is limited by one or the other depends on its **arithmetic intensity** — how much work it does per byte it moves:

$$ intensity = FLOPs / bytes moved          attainable FLOP/s = min( peak FLOP/s,  bandwidth × intensity ) $$
:::where
FLOPs — the floating-point operations a computation performs
bytes moved — the data it must read from and write to memory
peak FLOP/s — the hardware's maximum compute rate
bandwidth — the hardware's memory bandwidth in bytes per second
:::

The **ridge point** peak / bandwidth is the intensity needed to keep the compute units busy; for a modern data-centre GPU it is a few hundred FLOPs per byte. Decoding one token for one sequence multiplies each weight once (2 FLOPs per weight) while reading it (2 bytes in bf16): an intensity of about 1 — hundreds of times below the ridge. **Batching** B sequences reuses each weight read B times, raising the intensity by about B, which is why serving many users together is so much cheaper per token.` },
      { type: 'code', title: 'A roofline calculator for decoding', code: `
params = 8e9                        # an 8-billion-parameter model
bytes_per_param = 2                 # bf16 weights
peak_flops = 990e12                 # roughly an H100's dense bf16 peak, in FLOP/s
bandwidth = 3.35e12                 # roughly its memory bandwidth, in bytes/s
kv_bytes_per_token = 2 * 32 * 8 * 128 * 2   # K and V x 32 layers x 8 KV heads x 128 dims x 2 bytes = 128 KiB
context = 1000                      # tokens already in each sequence's KV cache

print(f"ridge point: {peak_flops / bandwidth:.0f} FLOPs per byte")
print(f"weights: {params * bytes_per_param / 1e9:.0f} GB,  KV cache per sequence: {context * kv_bytes_per_token / 1e6:.0f} MB")
for batch in (1, 8, 32, 128, 256):
    flops = 2 * params * batch                                          # one new token for every sequence
    bytes_moved = params * bytes_per_param + batch * context * kv_bytes_per_token
    t_compute, t_memory = flops / peak_flops, bytes_moved / bandwidth
    t = max(t_compute, t_memory)
    bound = "memory" if t_memory > t_compute else "compute"
    print(f"batch {batch:3d}: {t * 1000:5.2f} ms per step ({bound}-bound)  {1 / t:5.0f} tokens/s per user  {batch / t:7.0f} tokens/s in total")
`, after: `One sequence gets about 200 tokens per second and uses well under 1% of the GPU's compute. Batching multiplies total throughput by tens of times while each user slows down only gradually. With long contexts the KV cache grows with the batch, so even large batches stay memory-bound — one reason KV-cache size (grouped-query attention, KV quantization) matters so much.` },
      { type: 'demo', title: 'Interactive: batching on the roofline', demo: 'roofline' },
      { type: 'text', title: 'Batching: static versus continuous', md: `- **Static batching** waits to collect a batch and runs it until the *longest* response finishes; short requests sit idle and new ones wait.
- **Continuous (in-flight) batching** (Orca, vLLM) schedules at the level of single decode steps: finished sequences leave and new ones join at every step, keeping the GPU full.
- **PagedAttention** stores KV caches in fixed-size blocks, like virtual-memory pages, so memory isn't wasted on fragmentation and many more sequences fit.
- **Chunked prefill** splits long prompts into pieces interleaved with decode steps, so one huge prompt doesn't stall everyone's token stream; some systems go further and run prefill and decode on separate GPUs (**disaggregation**).

The core trade-off: bigger batches mean higher throughput and lower cost per token, but each user's tokens arrive more slowly. Serving systems tune batch limits against latency targets.` },
      { type: 'text', title: 'Speculative decoding', md: `Because decode steps are memory-bound, checking several tokens costs almost the same as generating one. **Speculative decoding** uses a small, fast **draft** model to propose k tokens; the large **target** model scores all of them in a single forward pass and accepts a prefix of them using a rejection-sampling rule that preserves the target model's exact output distribution. If each draft token is accepted with probability α (independently), one target pass yields on average:

$$ E[tokens per target pass] = ( 1 − α^{k+1} ) / ( 1 − α )          speed-up ≈ ( 1 − α^{k+1} ) / ( (1 − α)(1 + k·c) ) $$
:::where
α — "alpha": the probability that the target accepts a draft token
k — the number of tokens the draft proposes per round
c — the cost of one draft step relative to one target step
the +1 — even when a draft token is rejected, the target's pass still produces one token of its own
:::

Variants avoid a separate draft model: extra prediction heads (Medusa, EAGLE) or copying n-grams from the prompt ("prompt lookup") for tasks like editing and summarization.` },
      { type: 'code', title: 'How much does speculative decoding help?', code: `
import numpy as np
rng = np.random.default_rng(0)

def expected_tokens(alpha, k):
    return (1 - alpha ** (k + 1)) / (1 - alpha)

def simulate(alpha, k, rounds=20000):
    total = 0
    for _ in range(rounds):
        accepted = 0
        while accepted < k and rng.random() < alpha:     # draft tokens are accepted until the first rejection
            accepted += 1
        total += accepted + 1                             # plus one token from the target itself
    return total / rounds

c = 0.1                                                   # the draft model costs a tenth of the target per step
for alpha in (0.6, 0.8, 0.9):
    best_k = max(range(1, 17), key=lambda k: expected_tokens(alpha, k) / (1 + k * c))
    print(f"alpha = {alpha}: k = 4 gives {expected_tokens(alpha, 4):.2f} tokens per pass (simulated {simulate(alpha, 4):.2f}); "
          f"best k = {best_k}, speed-up {expected_tokens(alpha, best_k) / (1 + best_k * c):.2f}x")
`, after: `The simulation matches the formula. The better the draft agrees with the target, the more tokens each expensive pass yields and the longer the drafts worth proposing; with a weak draft, long proposals waste effort.` },
      { type: 'check', question: { type: 'mcq', q: 'Why is decoding a single sequence memory-bound?', options: ['Because tokens are large', 'Each step reads every weight from memory but performs only about 2 FLOPs per weight', 'Because attention is quadratic', 'Because GPUs have little memory'], answer: 1, explain: 'Arithmetic intensity ≈ 1 FLOP per byte, far below the ridge point.' } },
      { type: 'text', title: 'Other levers on cost and latency', md: `- **Quantization** (see *Quantization & Efficient Inference*): int8/int4 weights mean fewer bytes per decode step — a direct speed-up for memory-bound decoding — and KV-cache quantization fits more sequences.
- **Prefix (prompt) caching**: reuse the KV cache of a shared prefix — a long system prompt, tool definitions, a document — across requests. API providers expose this as prompt caching with discounted cached input tokens.
- **Batch APIs** for offline work trade latency for price (often around half the cost).
- **Routing**: send easy requests to smaller models and hard ones to larger models.
- **Streaming** output hides latency; capping output length and asking for concise formats cuts cost.
- **Serving engines**: vLLM, SGLang, TensorRT-LLM, llama.cpp and the managed APIs implement these techniques.

The cost per token follows from throughput:

$$ cost per 1M tokens = GPU cost per hour / ( tokens per second × 3600 / 1,000,000 ) $$
:::where
GPU cost per hour — what the hardware costs to rent or own, per hour
tokens per second — the sustained total throughput across all requests
:::` }
    ],
    when: `- Sizing hardware and estimating the cost of an LLM feature before launch.
- Choosing batch sizes, context limits and quantization levels against latency targets.
- Long shared prompts or repeated documents: use prefix caching.
- Latency-sensitive generation with a good draft model available: speculative decoding.`,
    pitfalls: `- Measuring throughput without latency (or the reverse) — always report both, under realistic load.
- Ignoring the KV cache when estimating memory; long contexts and big batches can need more memory than the weights.
- Benchmarking with short prompts and outputs that don't match production traffic.
- Expecting speculative decoding to help in compute-bound regimes (very large batches).
- Changing prompts in ways that break prefix-cache hits (put stable content first, variable content last).`,
    takeaways: [
      'Prefill is compute-bound and sets the time to first token; decode is memory-bound and sets the per-token latency.',
      'Roofline: attainable FLOP/s = min(peak, bandwidth × intensity); batching raises decode intensity about B-fold.',
      'Continuous batching, PagedAttention and chunked prefill keep GPUs busy without wasting memory.',
      'Speculative decoding yields (1 − α^{k+1})/(1 − α) tokens per target pass while preserving the output distribution.',
      'Quantization, prefix caching, batch APIs and routing are the main cost levers; cost per token = hardware cost / throughput.'
    ],
    quiz: [
      { type: 'mcq', q: 'Which phase mainly determines the time to first token for a long prompt?', options: ['Decode', 'Prefill', 'Tokenization', 'Sampling'], answer: 1, explain: 'The whole prompt must be processed before the first token appears.' },
      { type: 'fill', q: 'A GPU has a peak of 1,000 TFLOP/s and a bandwidth of 4 TB/s. What is its ridge point in FLOPs per byte?', answer: ['250'], explain: '1,000e12 / 4e12 = 250.' },
      { type: 'mcq', q: 'What does continuous batching do?', options: ['Waits until all requests in a batch finish', 'Adds and removes sequences at every decode step to keep the GPU full', 'Uses one GPU per request', 'Batches only the prefill'], answer: 1, explain: 'Scheduling happens per step, not per batch.' },
      { type: 'fill', q: 'Speculative decoding with acceptance rate α = 0.5 and k = 1 draft token: expected tokens per target pass? (decimal)', answer: ['1.5'], explain: '(1 − 0.5²) / (1 − 0.5) = 0.75 / 0.5 = 1.5.' },
      { type: 'tf', q: 'Speculative decoding changes the distribution of the text the target model produces.', answer: false, explain: 'The rejection-sampling rule preserves the target model\'s distribution exactly.' }
    ],
    problems: [
      {
        id: 'decode-roofline', title: 'A Decode Roofline Model', difficulty: 'medium',
        prompt: `Implement a simple performance model of one decode step (every sequence in the batch generates one token):

- \`decode_step_time(params, bytes_per_param, batch, kv_bytes_per_seq, peak_flops, bandwidth)\` → max(compute time, memory time), where compute time = 2·params·batch / peak_flops and memory time = (params·bytes_per_param + batch·kv_bytes_per_seq) / bandwidth.
- \`tokens_per_second(params, bytes_per_param, batch, kv_bytes_per_seq, peak_flops, bandwidth)\` → total throughput, batch / step time.
- \`cost_per_million(gpu_cost_per_hour, tokens_per_sec)\` → gpu_cost_per_hour / (tokens_per_sec · 3600 / 1e6).`,
        starter: `
def decode_step_time(params, bytes_per_param, batch, kv_bytes_per_seq, peak_flops, bandwidth):
    # TODO
    pass

def tokens_per_second(params, bytes_per_param, batch, kv_bytes_per_seq, peak_flops, bandwidth):
    # TODO
    pass

def cost_per_million(gpu_cost_per_hour, tokens_per_sec):
    # TODO
    pass
`,
        solution: `
def decode_step_time(params, bytes_per_param, batch, kv_bytes_per_seq, peak_flops, bandwidth):
    compute = 2 * params * batch / peak_flops
    memory = (params * bytes_per_param + batch * kv_bytes_per_seq) / bandwidth
    return max(compute, memory)

def tokens_per_second(params, bytes_per_param, batch, kv_bytes_per_seq, peak_flops, bandwidth):
    return batch / decode_step_time(params, bytes_per_param, batch, kv_bytes_per_seq, peak_flops, bandwidth)

def cost_per_million(gpu_cost_per_hour, tokens_per_sec):
    return gpu_cost_per_hour / (tokens_per_sec * 3600 / 1e6)
`,
        hints: ['A step takes as long as the slower of computing and moving the data.', 'Tokens per hour = tokens per second × 3600.'],
        explain: `This two-line roofline model explains most of LLM serving economics: at small batches the weights' memory traffic dominates, so extra users are nearly free until compute or KV-cache traffic catches up.`,
        tests: [
          { name: 'memory-bound single sequence', call: 'decode_step_time(1e9, 2, 1, 0, 1e15, 1e12)', expect: '2e-3' },
          { name: 'compute-bound at a huge batch', call: 'decode_step_time(1e9, 2, 10000, 0, 1e15, 1e12)', expect: '0.02' },
          { call: 'cost_per_million(3.6, 1000)', expect: '1.0' },
          { name: 'throughput vs reference', call: '[tokens_per_second(8e9, 2, b, 1.3e8, 990e12, 3.35e12) for b in (1, 16, 64, 256)]', hidden: true },
          { name: 'batching raises throughput', code: 'assert tokens_per_second(7e9, 2, 32, 5e7, 1e15, 3e12) > 10 * tokens_per_second(7e9, 2, 1, 5e7, 1e15, 3e12)', hidden: true }
        ]
      },
      {
        id: 'speculative-decoding', title: 'Speculative Decoding Arithmetic', difficulty: 'easy',
        prompt: `Implement:

- \`expected_tokens(alpha, k)\` → (1 − α^{k+1}) / (1 − α), the expected tokens per target pass; when α = 1 return k + 1.
- \`speedup(alpha, k, c)\` → expected_tokens(alpha, k) / (1 + k·c), where c is the draft's per-step cost relative to the target.
- \`best_k(alpha, c, k_max=16)\` → the k in 1..k_max with the highest speed-up (the smallest such k if tied).`,
        starter: `
def expected_tokens(alpha, k):
    # TODO
    pass

def speedup(alpha, k, c):
    # TODO
    pass

def best_k(alpha, c, k_max=16):
    # TODO
    pass
`,
        solution: `
def expected_tokens(alpha, k):
    if alpha == 1:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)

def speedup(alpha, k, c):
    return expected_tokens(alpha, k) / (1 + k * c)

def best_k(alpha, c, k_max=16):
    best, best_s = 1, speedup(alpha, 1, c)
    for k in range(2, k_max + 1):
        s = speedup(alpha, k, c)
        if s > best_s:
            best, best_s = k, s
    return best
`,
        hints: ['The expected count is a geometric series: 1 + α + α² + … + α^k.', 'Keep the first k that achieves the best speed-up.'],
        explain: `Each extra draft token adds α^j to the expected yield but costs c more drafting time, so the optimal k grows with the acceptance rate and shrinks as the draft gets more expensive.`,
        tests: [
          { call: 'expected_tokens(0.5, 1)', expect: '1.5' },
          { call: 'expected_tokens(1.0, 4)', expect: '5.0' },
          { call: 'speedup(0.8, 4, 0.1)', expect: '(1 - 0.8 ** 5) / 0.2 / 1.4' },
          { name: 'best k vs reference', call: '[best_k(0.6, 0.1), best_k(0.9, 0.05), best_k(0.3, 0.5)]', hidden: true },
          { name: 'a free, perfect draft wants the longest proposals', call: 'best_k(1.0, 0.0, 8)', expect: '8', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Kwon et al. — Efficient Memory Management for LLM Serving with PagedAttention (vLLM)', url: 'https://arxiv.org/abs/2309.06180' },
      { title: 'Leviathan, Kalman & Matias — Fast Inference from Transformers via Speculative Decoding', url: 'https://arxiv.org/abs/2211.17192' },
      { title: 'Yu et al. — Orca: A Distributed Serving System for Transformer-Based Generative Models', url: 'https://www.usenix.org/conference/osdi22/presentation/yu' },
      { title: 'Williams, Waterman & Patterson — Roofline: An Insightful Visual Performance Model', url: 'https://doi.org/10.1145/1498765.1498785' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
