/* ML Atlas – content: Transformers & LLMs (training, generation, adaptation, capstone) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'gpt', branch: 'llm', title: 'GPT: Decoder-Only Language Models', short: 'GPT', level: 3, minutes: 45, prereqs: ['transformer-block', 'language-models', 'tokenization'],
    tags: ['gpt', 'decoder-only', 'next token prediction', 'logits', 'weight tying', 'context window', 'kv cache'],
    summary: 'The complete GPT architecture: token and position embeddings, a stack of causal transformer blocks and a language-model head — trained with next-token cross-entropy on shifted sequences, and used to generate text one token at a time.',
    why: `GPT ("Generative Pre-trained Transformer") is the architecture behind ChatGPT, Claude, Llama, Gemini, Mistral and nearly every modern LLM. It is surprisingly simple: everything from the previous lessons wired end to end. Once you can trace the data flow (ids → embeddings → blocks → logits), understand the training trick (every position predicts its own next token, all in parallel) and count the parameters, you can read model cards, estimate memory, and build one yourself.`,
    objectives: [
      'Trace a batch of token ids through a GPT to logits, with the shape at every step',
      'Build input/target pairs by shifting sequences and compute the next-token loss',
      'Count the parameters of a GPT exactly',
      'Explain the context window, weight tying and the KV cache'
    ],
    sections: [
      { type: 'text', title: 'The whole model on one page', md: `With batch size B, context length T, model width d and vocabulary size V:

\`\`\`text
token ids          (B, T)        e.g. [[464, 3797, 3332, ...]]
token embedding    (B, T, d)     wte[ids]        a row lookup in a (V, d) table
+ positions        (B, T, d)     wpe[0..T-1]     (learned; RoPE models do this inside attention)
N x block          (B, T, d)     causal self-attention + MLP, with residuals (shape never changes)
final LayerNorm    (B, T, d)
LM head            (B, T, V)     x @ wte.T       one score (logit) per vocabulary token
softmax            (B, T, V)     P(next token | tokens up to this position)
\`\`\`

The output at position t is a prediction of the token at position t + 1, using only tokens 0…t (the causal mask guarantees it).

| Model | Layers | d_model | Heads | Context | Vocab | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 small | 12 | 768 | 12 | 1,024 | 50,257 | 124M |
| GPT-2 XL | 48 | 1,600 | 25 | 1,024 | 50,257 | 1.5B |
| GPT-3 | 96 | 12,288 | 96 | 2,048 | 50,257 | 175B |
| Llama 3 8B | 32 | 4,096 | 32 (8 KV) | 8,192 | 128,256 | 8B |` },
      { type: 'text', title: 'Training: every position is an example', md: `Take a chunk of T + 1 consecutive tokens. The inputs are the first T, the targets are the same tokens **shifted by one**:

\`\`\`text
text:     The   cat   sat   on    the   mat
x:        The   cat   sat   on    the
y:        cat   sat   on    the   mat
\`\`\`

Thanks to the causal mask, position t sees only x[0..t] and must predict y[t]. So one sequence of length T yields **T training examples at once**, all computed in a single parallel forward pass. The loss is the average cross-entropy over all B × T positions:

$$ L = −(1 / BT) · Σ_b Σ_t log softmax(logits[b, t])[ y[b, t] ] $$
:::where
B — the batch size, the number of sequences
T — the context length, the number of positions per sequence
logits[b, t] — the model's V scores for the next token at position t of sequence b
softmax(…)[ y[b, t] ] — the probability the model gave to the correct next token y[b, t]
Σ_b Σ_t — add up over every sequence and every position
log — the natural logarithm; the minus sign turns log-probabilities into a positive loss
:::

:::tip Sanity check
At initialization the model knows nothing, so its predictions are roughly uniform and the loss should be close to **ln V** — about 10.8 for GPT-2's 50,257 tokens. A much higher starting loss means a bug (or a bad initialization).
:::` },
      { type: 'code', title: 'A complete GPT forward pass and loss in NumPy', code: `
import numpy as np
rng = np.random.default_rng(0)
V, T, d, n_layers, n_heads = 20, 8, 16, 2, 2

def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def attention(x, Wqkv, Wo):
    B, T, d = x.shape; dh = d // n_heads
    q, k, v = np.split(x @ Wqkv, 3, axis=-1)                              # each (B, T, d)
    heads = lambda m: m.reshape(B, T, n_heads, dh).transpose(0, 2, 1, 3)  # (B, h, T, dh)
    q, k, v = heads(q), heads(k), heads(v)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)              # causal mask
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ Wo

params = {"wte": rng.normal(0, 0.02, (V, d)), "wpe": rng.normal(0, 0.02, (T, d)),
          "blocks": [{"Wqkv": rng.normal(0, 0.02, (d, 3 * d)), "Wo": rng.normal(0, 0.02, (d, d)),
                      "W1": rng.normal(0, 0.02, (d, 4 * d)), "W2": rng.normal(0, 0.02, (4 * d, d))}
                     for _ in range(n_layers)]}

def gpt(ids):
    x = params["wte"][ids] + params["wpe"][: ids.shape[1]]               # (B, T, d)
    for b in params["blocks"]:
        x = x + attention(layer_norm(x), b["Wqkv"], b["Wo"])
        x = x + np.maximum(0, layer_norm(x) @ b["W1"]) @ b["W2"]          # ReLU MLP for brevity
    return layer_norm(x) @ params["wte"].T                                 # tied LM head -> (B, T, V)

data = rng.integers(0, V, size=(2, T + 1))
x, y = data[:, :-1], data[:, 1:]                                           # shift by one
logits = gpt(x)
logp = logits - logits.max(-1, keepdims=True)
logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))                 # log-softmax
loss = -np.take_along_axis(logp, y[..., None], axis=-1).mean()
print("logits shape:", logits.shape)
print(f"loss at init: {loss:.3f}    ln(V) = {np.log(V):.3f}")
`, after: `The untrained model's loss is almost exactly ln V: it spreads probability evenly over the vocabulary. Training pushes it down.`, predict: {"type":"mcq","q":"This untrained GPT has a vocabulary of V = 20 tokens and tiny random weights. Roughly what cross-entropy loss does it report before any training?","options":["About 0","About 1","About 3 (that is, ln 20)","About 20"],"answer":2,"explain":"3.000, almost exactly ln 20 = 2.996. Tiny weights give nearly uniform predictions, and a uniform guess over V tokens has a loss of ln V. A first loss far from ln V usually means a bug.","expect":["loss at init: 3.000"]} },
      { type: 'check', question: { type: 'mcq', q: 'One training sequence has T = 1,024 input tokens (targets shifted by one). How many next-token predictions are trained in one forward pass?', options: ['1', '1,024 — one per position', '1,024²', '50,257'], answer: 1, explain: 'Every position predicts its own next token; the causal mask makes all of them valid examples simultaneously.' } },
      { type: 'text', title: 'Weight tying, the context window and the KV cache', md: `- **Weight tying.** The LM head reuses the token-embedding matrix: logits = x · wteᵀ. That saves V·d parameters (38.6M in GPT-2 small) and often helps quality, because "which token is this?" and "which token comes next?" use the same vector space. Many large models untie them.
- **Context window.** The maximum T the model was trained with (GPT-2: 1,024; GPT-3: 2,048; recent models: 128k or more). Learned position embeddings literally have no row beyond it. Longer inputs must be truncated, chunked or summarized.
- **Generation.** Run the model on the prompt, take the logits at the **last** position, choose a token (see *Decoding*), append it and repeat. Because of the causal mask, earlier positions never change when you append, so their keys and values can be stored — the **KV cache** — and each new token only computes its own query, key and value. That turns the cost of each step from re-processing the whole sequence into a single new position.` },
      { type: 'text', title: 'Counting parameters exactly', md: `For a GPT-2-style model (learned positions, biases, LayerNorm, tied head):

$$ embeddings:   V·d + T_max·d $$
$$ per block:    attention 4d² + 4d    MLP 8d² + 5d    two LayerNorms 4d    →  12d² + 13d $$
$$ final LayerNorm: 2d          LM head (tied): 0 $$
:::where
V — the vocabulary size
d — the model width, d_model
T_max — the context window (the number of learned position embeddings)
4d² + 4d — the Q, K, V and output projections: four d × d weight matrices plus their biases
8d² + 5d — the MLP: d × 4d and 4d × d weights, plus 4d + d biases
4d — two LayerNorms, each with a gain vector and a bias vector of length d
LM head (tied) — the output layer reuses the token-embedding matrix, so it adds no parameters
:::

GPT-2 small: 50,257·768 + 1,024·768 + 12·(12·768² + 13·768) + 2·768 = **124,439,808**. For small models the embedding table is a big share (31% here); for large models the blocks dominate, so "parameters ≈ 12·d²·layers" is a handy estimate.` },
      { type: 'chart', title: "Where GPT-2's parameters live", after: `In GPT-2 small, nearly a third of the 124 million parameters are the token-embedding table. The embeddings grow only linearly with the width, while the blocks grow with width² × depth, so in GPT-2 XL they are just 5% of the total. That is why parameter counts are often quoted without embeddings, and why vocabulary size matters most for small models.`, chart: {"kind":"bar","x":["GPT-2 small","GPT-2 medium","GPT-2 large","GPT-2 XL"],"series":[{"name":"embeddings (tokens and positions)","y":[39.4,52.5,65.6,82]},{"name":"transformer blocks","y":[85.1,302.3,708.4,1475.6]}],"xLabel":"model","yLabel":"parameters (millions)","yMin":0} },
      { type: 'code', title: "Counting GPT-2's parameters", code: `V, T_max = 50257, 1024                         # GPT-2's vocabulary and context length
sizes = {"small": (768, 12), "medium": (1024, 24), "large": (1280, 36), "XL": (1600, 48)}

def gpt2_params(d, layers):
    embeddings = V * d + T_max * d             # token and position tables (the LM head is tied: no extra cost)
    per_block = 12 * d * d + 13 * d            # attention 4d² + 4d, MLP 8d² + 5d, two LayerNorms 4d
    return embeddings, layers * per_block + 2 * d   # plus the final LayerNorm

for name, (d, layers) in sizes.items():
    emb, rest = gpt2_params(d, layers)
    print(f"GPT-2 {name:<6}: {(emb + rest) / 1e6:7.1f} M parameters ({emb / (emb + rest):.0%} in the embeddings)")
print("GPT-2 small has about 124 M parameters:", round(sum(gpt2_params(768, 12)) / 1e6) == 124)`, after: `The formula reproduces the published sizes: 124 million, 355 million, 774 million and 1.56 billion. The share held by the embeddings falls from about a third to about 5% as the models grow.`, predict: {"type":"mcq","q":"Roughly what share of GPT-2 small's 124 million parameters is in its embedding tables?","options":["About 3%","About 30%","About 60%","About 90%"],"answer":1,"explain":"The token table alone is 50,257 × 768 ≈ 38.6 million parameters, about 31% of the total.","expect":["about 124 M parameters: True"]} },
      { type: 'check', question: { type: 'tf', q: 'With weight tying, the output layer reuses the token embedding matrix.', answer: true, explain: 'logits = x · wteᵀ — the same (V, d) matrix is used for input lookup and output scoring.' } }
    ],
    when: `- Decoder-only GPT models are the default for generative LLMs: chat, code, reasoning, agents.
- Pick a model size from your memory and latency budget: parameters × bytes per parameter, plus the KV cache.
- For classification or retrieval without generation, an encoder or a dedicated embedding model is often cheaper.`,
    pitfalls: `- Off-by-one errors when shifting inputs and targets.
- Forgetting the causal mask in training: the loss drops to near zero because the model copies the answer, then generation fails.
- Feeding sequences longer than the context window.
- Using the wrong position's logits during generation (you want the last one).
- An initial loss far from ln V — a sign of a bug in the model or the loss.`,
    takeaways: [
      'A GPT = token + position embeddings → N causal transformer blocks → final norm → LM head (logits over the vocabulary).',
      'Training shifts the sequence by one; every position predicts its next token; loss = mean cross-entropy.',
      'At initialization the loss should be about ln(vocabulary size).',
      'Parameters ≈ 12·d²·layers + vocab·d; GPT-2 small has 124M.',
      'Generation feeds the model its own output; the KV cache avoids recomputing past tokens.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the LM head output at each position?', options: ['A single next-token id', 'A score (logit) for every token in the vocabulary', 'An embedding of size d', 'The attention weights'], answer: 1, explain: 'Softmax over those V logits gives the next-token distribution.' },
      { type: 'fill', q: 'A freshly initialized GPT has a 32,000-token vocabulary. Roughly what loss (natural log) should it start at? (1 decimal)', answer: ['10.4', '10.37', '10.373'], display: 'ln 32,000 ≈ 10.4', explain: 'A uniform guess over 32,000 tokens has cross-entropy ln 32,000 ≈ 10.37.' },
      { type: 'mcq', q: 'Why can a GPT be trained on all positions of a sequence in parallel?', options: ['Because of the residual stream', 'The causal mask stops each position from seeing its own target, so every position is a valid example at once', 'Because of weight tying', 'Because tokens are independent of each other'], answer: 1, explain: 'Without the mask, positions could peek at the answer.' },
      { type: 'tf', q: 'The KV cache stores the keys and values of previous tokens so generation does not recompute them.', answer: true, explain: 'Past positions never change under a causal mask, so their K and V can be reused.' },
      { type: 'mcq', q: 'The inputs are x = tokens[0:T]. What are the targets?', options: ['tokens[0:T]', 'tokens[1:T+1]', 'tokens[T] only', 'random tokens'], answer: 1, explain: 'Each input position is paired with the token that follows it.' },
      {"type":"mcq","q":"Doubling d_model (keeping the depth) multiplies a transformer block's parameter count by about…","options":["2","4","8","It does not change"],"answer":1,"explain":"A block has about 12·d² parameters, so doubling d multiplies them by 4."},
      {"type":"tf","q":"In small GPT models, the token-embedding table can hold a large share of all the parameters.","answer":true,"explain":"The table has V·d parameters, which grows only linearly with d, so it matters most when the blocks are small: about a third of GPT-2 small."}
    ],
    problems: [
      {
        id: 'shift-targets', title: 'Inputs and Targets', difficulty: 'easy',
        prompt: `Implement \`make_batch(ids, starts, block_size)\` → \`(x, y)\`. For each start index s in \`starts\`:

- the x row is \`ids[s : s + block_size]\`
- the y row is \`ids[s + 1 : s + block_size + 1]\` (the same window shifted by one)

Return two integer NumPy arrays of shape (len(starts), block_size).

\`\`\`text
make_batch([10, 11, 12, 13, 14, 15], [0, 2], 3)
-> x = [[10, 11, 12], [12, 13, 14]]
   y = [[11, 12, 13], [13, 14, 15]]
\`\`\``,
        starter: `
import numpy as np

def make_batch(ids, starts, block_size):
    ids = np.asarray(ids)
    # TODO: build x and y
    pass
`,
        solution: `
import numpy as np

def make_batch(ids, starts, block_size):
    ids = np.asarray(ids)
    x = np.stack([ids[s:s + block_size] for s in starts])
    y = np.stack([ids[s + 1:s + block_size + 1] for s in starts])
    return x, y
`,
        hints: ['Slice one window per start, then `np.stack` the windows into a matrix.', 'y uses the same slices moved one step to the right.'],
        explain: `This is the data loader of every GPT training script (nanoGPT's \`get_batch\` is exactly this, with random start positions). Each (x, y) row gives block_size next-token examples.`,
        tests: [
          { call: 'make_batch([10, 11, 12, 13, 14, 15], [0, 2], 3)', expect: '(np.array([[10, 11, 12], [12, 13, 14]]), np.array([[11, 12, 13], [13, 14, 15]]))' },
          { name: 'targets are inputs shifted by one', setup: 'rng = np.random.default_rng(0)\nids = rng.integers(0, 50, 200)\nstarts = rng.integers(0, 200 - 17, 8)', code: 'x, y = make_batch(ids, starts, 16)\nassert x.shape == (8, 16) and y.shape == (8, 16)\nassert np.array_equal(x[:, 1:], y[:, :-1])' },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(1)\nids = rng.integers(0, 100, 500)\nstarts = rng.integers(0, 500 - 33, 5)', call: 'make_batch(ids, starts, 32)', hidden: true }
        ]
      },
      {
        id: 'next-token-loss', title: 'The Next-Token Loss', difficulty: 'medium',
        prompt: `Implement:

1. \`log_softmax(z)\` — a numerically stable log-softmax over the **last** axis: subtract the max, then subtract the log of the sum of exponentials.
2. \`next_token_loss(logits, targets)\` — \`logits\` has shape (B, T, V) and \`targets\` (B, T) holds the correct next-token ids. Pick each target's log-probability (\`np.take_along_axis\`) and return the **mean negative log-probability** as a float.

\`\`\`text
All-zero logits are a uniform guess, so the loss is ln(V) for any targets.
\`\`\``,
        starter: `
import numpy as np

def log_softmax(z):
    # TODO: stable log-softmax over the last axis
    pass

def next_token_loss(logits, targets):
    # TODO
    pass
`,
        solution: `
import numpy as np

def log_softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def next_token_loss(logits, targets):
    logp = log_softmax(logits)
    picked = np.take_along_axis(logp, np.asarray(targets)[..., None], axis=-1)
    return float(-picked.mean())
`,
        hints: ['`targets[..., None]` adds a trailing axis so `take_along_axis` can index the last axis of logp.', 'Never compute `np.log(softmax(z))` directly: tiny probabilities underflow to 0.'],
        explain: `This is \`F.cross_entropy(logits.view(-1, V), targets.view(-1))\` in PyTorch. Working in log space with the max subtracted keeps it finite even for logits in the thousands.`,
        tests: [
          { name: 'uniform logits', call: 'next_token_loss(np.zeros((2, 3, 5)), np.array([[0, 1, 2], [3, 4, 0]]))', expect: 'np.log(5)' },
          { name: 'confident and correct', call: 'next_token_loss(np.array([[[10.0, 0.0, 0.0]]]), np.array([[0]]))', expect: '-np.log(np.exp(10) / (np.exp(10) + 2))' },
          { name: 'huge logits stay finite', call: 'next_token_loss(np.array([[[1000.0, 0.0], [0.0, 1000.0]]]), np.array([[1, 1]]))', expect: '500.0', hidden: true },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(0)\nlogits = rng.normal(size=(4, 6, 11)) * 3\ntargets = rng.integers(0, 11, (4, 6))', call: 'next_token_loss(logits, targets)', hidden: true }
        ]
      },
      {
        id: 'gpt-param-count', title: 'Count GPT-2 Parameters', difficulty: 'medium',
        prompt: `Implement \`gpt2_params(vocab_size, context, d_model, n_layers)\` → the exact parameter count of a GPT-2-style model:

- token embedding: V·d; learned position embedding: context·d
- each block: LayerNorm (2d) · attention QKV (d·3d weights + 3d biases) · attention output (d·d + d) · LayerNorm (2d) · MLP up (d·4d + 4d) · MLP down (4d·d + d)
- final LayerNorm (2d); the LM head is tied to the token embedding (adds nothing)

\`\`\`text
gpt2_params(50257, 1024, 768, 12)  ->  124439808     (GPT-2 small)
\`\`\``,
        starter: `
def gpt2_params(vocab_size, context, d_model, n_layers):
    # TODO
    pass
`,
        solution: `
def gpt2_params(vocab_size, context, d_model, n_layers):
    d = d_model
    embeddings = vocab_size * d + context * d
    attention = (d * 3 * d + 3 * d) + (d * d + d)
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)
    norms = 2 * (2 * d)
    return embeddings + n_layers * (attention + mlp + norms) + 2 * d
`,
        hints: ['Add up the pieces of one block, multiply by n_layers, then add the embeddings and the final LayerNorm.', 'A LayerNorm has a gain and a bias vector: 2d parameters.'],
        explain: `Per block this is 12d² + 13d. GPT-2 medium (d = 1024, 24 layers) gives 354,823,168 and XL (d = 1600, 48 layers) 1,557,611,200 — the numbers on the official model cards.`,
        tests: [
          { call: 'gpt2_params(50257, 1024, 768, 12)', expect: '124439808' },
          { name: 'GPT-2 medium', call: 'gpt2_params(50257, 1024, 1024, 24)', expect: '354823168' },
          { name: 'GPT-2 XL', call: 'gpt2_params(50257, 1024, 1600, 48)', expect: '1557611200', hidden: true },
          { name: 'tiny model', call: 'gpt2_params(65, 256, 64, 4)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Radford et al. — Language Models are Unsupervised Multitask Learners (GPT-2)', url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf' },
      { title: 'Brown et al. — Language Models are Few-Shot Learners (GPT-3)', url: 'https://arxiv.org/abs/2005.14165' },
      { title: 'Karpathy — nanoGPT', url: 'https://github.com/karpathy/nanoGPT', note: 'a clean, minimal GPT training codebase' }
    ]
  },

  // =====================================================================================
  {
    id: 'decoding', branch: 'llm', title: 'Decoding: Greedy, Sampling, Temperature, Top-k & Top-p', short: 'Decoding', level: 3, minutes: 35, prereqs: ['gpt'],
    tags: ['decoding', 'sampling', 'temperature', 'top-k', 'top-p', 'nucleus sampling', 'greedy', 'beam search', 'repetition penalty'],
    summary: 'How a language model turns next-token probabilities into text: greedy decoding, sampling, temperature, top-k and nucleus (top-p) filtering, beam search, stop conditions — and which settings suit which tasks.',
    why: `A language model only outputs a probability distribution over the next token; **decoding** decides what actually gets written. The same model can be dull and repetitive, creative, or incoherent depending on these settings. Every LLM API exposes temperature and top_p. Understanding them lets you choose deterministic settings for code and extraction, diverse ones for brainstorming, and reproducible ones for evaluation.`,
    objectives: [
      'Generate text autoregressively and explain stop conditions',
      'Apply temperature to logits and predict its effect',
      'Implement top-k and nucleus (top-p) filtering',
      'Choose decoding settings for different tasks; know beam search and repetition penalties'
    ],
    sections: [
      { type: 'text', title: 'The generation loop', md: `\`\`\`text
ids = prompt tokens
repeat until a stop token, a stop string, or max_new_tokens:
    logits = model(ids)[-1]          # scores for the next token (last position only)
    probs  = softmax(logits / T)     # temperature
    probs  = filter(probs)           # optional top-k / top-p
    next   = pick(probs)             # argmax (greedy) or a random draw (sampling)
    ids.append(next)
\`\`\`

Generation is **autoregressive**: each new token becomes input for the next step. That's why generation is slow (one forward pass per token), why the KV cache matters, and why an early mistake can snowball.` },
      { type: 'text', title: 'Greedy, sampling and beam search', md: `- **Greedy** — always take the most probable token. Deterministic and fine for short factual answers, but it produces bland text and easily falls into loops ("I think that I think that…").
- **Sampling** — draw the next token at random according to the probabilities. Diverse and human-like, but occasionally picks a low-probability token that derails the text.
- **Beam search** — keep the b most probable partial sequences, extend each, keep the best b again. Good when there is one "right" output (translation, speech recognition); rarely used for open-ended chat because it favours generic, repetitive text.` },
      { type: 'text', title: 'Temperature', md: `Divide the logits by a temperature T before the softmax:

$$ pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T) $$
:::where
zᵢ — the logit (raw score) of token i
T — the temperature, a positive number (here T is not the sequence length)
exp — e to the power of its argument
Σⱼ — a sum over every token j in the vocabulary
pᵢ — the resulting probability of choosing token i
:::

- **T = 1** — the model's own distribution.
- **T < 1** — sharper: likely tokens get even more probability; as T → 0 it becomes greedy.
- **T > 1** — flatter: more randomness, more mistakes.

Temperature never changes the *ranking* of tokens, only how peaked the distribution is.` },
      { type: 'demo', title: 'Interactive: temperature, top-k and top-p', md: `Move the sliders, then sample many tokens and watch how often each word is chosen. The entropy readout shows how uncertain the final distribution is.`, demo: 'softmax' },
      { type: 'text', title: 'Top-k and top-p (nucleus) sampling', md: `Even at a sensible temperature, the long tail of thousands of unlikely tokens holds real probability mass in total — sample long enough and you will hit garbage. Filters cut the tail before sampling:

- **top-k** — keep only the k most likely tokens and renormalize. Simple, but a fixed k is too many when the model is confident and too few when many continuations are plausible.
- **top-p (nucleus)** — keep the *smallest* set of top tokens whose cumulative probability reaches p (e.g. 0.9), then renormalize. The set adapts: one or two tokens when the model is sure, dozens when it isn't.
- **min-p** — keep tokens whose probability is at least min_p × the top probability; another adaptive rule.

Other knobs: **repetition / frequency / presence penalties** lower the scores of tokens that already appeared; **stop sequences** end generation; a fixed **seed** makes sampling reproducible.` },
      { type: 'chart', title: "How many tokens top-p keeps", md: `Two next-token distributions over a 1,000-token vocabulary, one peaked and one flat:`, after: `With p = 0.9, top-p keeps only 6 tokens when the model is confident but 668 when it is uncertain: the candidate set adapts to the situation. A fixed top-k cannot do this; k = 40 would either cut off plausible options in the flat case or let unlikely tokens through in the peaked one.`, chart: {"kind":"line","x":[0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.99],"yLog":true,"series":[{"name":"confident model (steep distribution)","y":[1,1,2,3,4,6,9,12,20,57]},{"name":"uncertain model (flat distribution)","y":[86,156,265,430,538,668,756,821,889,962]}],"xLabel":"top-p threshold","yLabel":"tokens kept (of 1,000, log scale)"} },
      { type: 'code', title: 'Decoding strategies on a toy model', code: `
import numpy as np
words = ["<s>", "the", "cat", "dog", "sat", "ran", "on", "mat", "away", "."]
idx = {w: i for i, w in enumerate(words)}
corpus = "<s> the cat sat on the mat . <s> the dog ran away . <s> the cat ran on the mat . <s> the dog sat on the mat ."
toks = [idx[w] for w in corpus.split()]
counts = np.full((len(words), len(words)), 0.1)                 # smoothed bigram counts
for a, b in zip(toks, toks[1:]):
    counts[a, b] += 1
logits_table = np.log(counts / counts.sum(1, keepdims=True))   # our "model": next-word log-probs

def generate(T=1.0, top_p=1.0, greedy=False, seed=0, max_new=12):
    rng = np.random.default_rng(seed)
    ids = [idx["<s>"]]
    for _ in range(max_new):
        z = logits_table[ids[-1]]
        if greedy:
            nxt = int(np.argmax(z))
        else:
            p = np.exp((z - z.max()) / T); p /= p.sum()                       # temperature
            order = np.argsort(-p)
            keep = order[: np.searchsorted(np.cumsum(p[order]), top_p) + 1]    # nucleus
            q = np.zeros_like(p); q[keep] = p[keep]; q /= q.sum()
            nxt = int(rng.choice(len(p), p=q))
        ids.append(nxt)
        if words[nxt] == ".":
            break
    return " ".join(words[i] for i in ids[1:])

print("greedy:", generate(greedy=True))
for label, kw in [("T=0.5", dict(T=0.5)), ("T=1.0", dict(T=1.0)), ("T=2.0", dict(T=2.0)),
                  ("T=2.0, top-p=0.5", dict(T=2.0, top_p=0.5))]:
    print(label)
    for seed in (1, 2, 3):
        print("   ", generate(seed=seed, **kw))
`, after: `Greedy decoding always writes the same sentence. Low temperature gives short, safe sentences; T = 2 produces word salad (even the start token shows up mid-sentence); the nucleus filter removes the unlikely tail and pulls high-temperature samples back towards sense.` },
      { type: 'check', question: { type: 'mcq', q: 'What happens to the distribution as the temperature goes from 1 toward 0?', options: ['It becomes uniform', 'It concentrates on the most likely token (approaching greedy)', 'The ranking of tokens reverses', 'Nothing changes'], answer: 1, explain: 'Dividing by a small T magnifies differences between logits.' } },
      { type: 'text', title: "Penalties, min-p and constrained decoding", md: `A few more controls appear in most LLM APIs and libraries. **Repetition penalties** lower the logits of tokens that already appeared in the output, to break loops:

$$ z′ᵢ = zᵢ − α_freq · countᵢ − α_pres · 1[ countᵢ > 0 ] $$
:::where
zᵢ — the original logit of token i
countᵢ — how many times token i has already appeared in the output
α_freq — the frequency penalty, which grows with every repetition
α_pres — the presence penalty, a flat penalty once the token has appeared at all
1[ … ] — 1 if the condition holds, otherwise 0
:::

- **Min-p sampling** keeps every token whose probability is at least a fraction (say 0.1) of the top token's probability. Like top-p it adapts to the model's confidence, and it holds up better at high temperatures.
- **Stop sequences and length limits** end generation at a chosen string or after a number of tokens.
- **Constrained (structured) decoding** masks every token that would break a required format, such as a JSON schema or a regular expression, before sampling, so the output is guaranteed to parse. Most providers offer this as "structured outputs".` },
      { type: 'code', title: "Top-p and min-p on confident and uncertain distributions", code: `import numpy as np
ranks = np.arange(1, 1001)                                 # a 1,000-token vocabulary
kept = {}
for name, w in [("confident", 1 / ranks ** 2.0), ("uncertain", 1 / ranks ** 0.8)]:
    p = w / w.sum()                                        # already sorted from most to least likely
    top_p = int(np.searchsorted(np.cumsum(p), 0.9) + 1)    # the smallest set holding 90% of the probability
    min_p = int(np.sum(p >= 0.1 * p[0]))                   # tokens at least 10% as likely as the best one
    kept[name] = top_p
    print(f"{name:<9}: top token {p[0]:.3f}   top-p 0.9 keeps {top_p:4d} tokens   min-p 0.1 keeps {min_p:3d} tokens")
print("top-p keeps far more tokens when uncertain:", kept["uncertain"] > 50 * kept["confident"])`, after: `In the confident case both filters keep a handful of tokens. In the uncertain case top-p keeps 668 of the 1,000 tokens, most of them individually unlikely, while min-p keeps 17: it only admits tokens that are reasonably competitive with the best one, which is why it degrades more gracefully at high temperatures.`, predict: {"type":"mcq","q":"With top-p = 0.9, roughly how many tokens are kept when the model is uncertain (a flat distribution over 1,000 tokens)?","options":["About 6","About 40","Hundreds","All 1,000"],"answer":2,"explain":"When probability is spread thinly, it takes hundreds of tokens to accumulate 90% of the mass.","expect":["top-p keeps far more tokens when uncertain: True"]} },
      { type: 'text', title: 'Choosing settings', md: `| Task | Typical settings |
| --- | --- |
| Code, math, extraction, classification | temperature 0–0.2 (near-greedy) |
| General chat / assistant | temperature ≈ 0.7, top-p ≈ 0.9–0.95 |
| Brainstorming, fiction | temperature 0.8–1.2, top-p ≈ 0.95 |
| Evaluations and tests | temperature 0 (or a fixed seed) for reproducibility |
| Translation, speech-to-text | beam search (4–5 beams) or greedy |

Two caveats: temperature 0 on a served model is *nearly* but not perfectly deterministic (batching and floating-point effects), and some reasoning models fix or restrict these parameters.` },
      { type: 'check', question: { type: 'tf', q: 'Top-p sampling always keeps the same number of tokens.', answer: false, explain: 'The nucleus grows or shrinks with how spread out the distribution is.' } }
    ],
    when: `- **Greedy / low temperature**: tasks with one correct answer (code, extraction, classification, math).
- **Temperature + top-p sampling**: open-ended generation; tune for diversity versus coherence.
- **Beam search**: short, constrained outputs where likelihood matters (translation, speech recognition).
- **Penalties and stop sequences**: control repetition and length.`,
    pitfalls: `- Raising temperature and top-p together until text becomes incoherent.
- Expecting temperature 0 to make a model *correct* — it only makes it consistent.
- Assuming every library applies temperature, top-k and top-p in the same order.
- Forgetting a stop condition and paying for tokens up to the limit.
- Comparing models that were evaluated with different decoding settings.`,
    takeaways: [
      'Generation is autoregressive: predict, pick, append, repeat.',
      'Greedy takes the argmax: deterministic, but bland and loop-prone.',
      'Temperature divides the logits: T < 1 sharpens, T > 1 flattens, ranking unchanged.',
      'Top-k keeps k tokens; top-p keeps the smallest set whose probability reaches p.',
      'Match settings to the task: near-zero temperature for code and facts, moderate for chat, higher for creativity.'
    ],
    quiz: [
      { type: 'mcq', q: 'Logits are [2, 1, 0]. Which statement is true?', options: ['T = 0.5 gives a flatter distribution than T = 2', 'T = 0.5 gives the first token more probability than T = 2 does', 'T = 0.5 and T = 2 give the same distribution', 'T = 2 changes which token is most likely'], answer: 1, explain: 'Lower temperature sharpens the distribution around the top token.' },
      { type: 'fill', q: 'Sorted probabilities are [0.5, 0.3, 0.15, 0.05]. With top-p = 0.75, how many tokens are kept?', answer: ['2', 'two'], explain: '0.5 < 0.75 but 0.5 + 0.3 = 0.8 ≥ 0.75, so the top 2 tokens form the nucleus.' },
      { type: 'mcq', q: 'Why is beam search rarely used for chatbots?', options: ['It is too random', 'It tends to produce generic, repetitive text in open-ended generation', 'It cannot use a KV cache', 'It needs a bigger vocabulary'], answer: 1, explain: 'Maximizing likelihood favours safe, dull continuations.' },
      { type: 'tf', q: 'Setting the temperature to 0 makes an LLM\'s answers factually correct.', answer: false, explain: 'It makes outputs (nearly) deterministic, not correct.' },
      { type: 'mcq', q: 'Which setting suits generating code that must compile?', options: ['Temperature 1.5, top-p 1.0', 'Temperature near 0', 'Top-k = 1,000 with temperature 2', 'Random tokens with a seed'], answer: 1, explain: 'Low temperature keeps the model on its most likely (usually correct) path.' },
      {"type":"mcq","q":"What does constrained (structured) decoding guarantee?","options":["That the answer is factually correct","That the output follows a required format, such as valid JSON, because tokens that would break it are masked","Faster generation","A lower temperature"],"answer":1,"explain":"It restricts which tokens can be sampled at each step, so the format is guaranteed; the content can still be wrong."},
      {"type":"mcq","q":"With min-p = 0.1 and a top-token probability of 0.6, which tokens are kept?","options":["Only the top token","Every token with a probability of at least 0.06","The tokens making up 10% of the probability","Every token"],"answer":1,"explain":"Min-p keeps tokens whose probability is at least 0.1 × 0.6 = 0.06."}
    ],
    problems: [
      {
        id: 'temperature-softmax', title: 'Softmax with Temperature', difficulty: 'easy',
        prompt: `Implement \`softmax_t(logits, T=1.0)\` → the probabilities of \`logits / T\`, using a numerically stable softmax (subtract the max before exponentiating). Assume T > 0.

\`\`\`text
softmax_t([1.0, 1.0])          -> [0.5, 0.5]
softmax_t([2.0, 1.0], T=0.5)   -> softmax([4.0, 2.0]) ≈ [0.881, 0.119]
\`\`\``,
        starter: `
import numpy as np

def softmax_t(logits, T=1.0):
    # TODO
    pass
`,
        solution: `
import numpy as np

def softmax_t(logits, T=1.0):
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
`,
        hints: ['Divide by T first, then do the usual stable softmax.'],
        explain: `Dividing by T before the softmax is all temperature is. Subtracting the max keeps \`np.exp\` from overflowing when logits are large or T is tiny.`,
        tests: [
          { call: 'softmax_t([1.0, 1.0])', expect: 'np.array([0.5, 0.5])' },
          { call: 'softmax_t([2.0, 1.0], T=0.5)', expect: 'np.exp([4.0, 2.0]) / np.exp([4.0, 2.0]).sum()' },
          { name: 'large logits', call: 'softmax_t([1000.0, 999.0, 0.0])', hidden: true },
          { name: 'high temperature is flatter', code: 'p1 = softmax_t([3.0, 1.0, 0.0], 1.0)\np2 = softmax_t([3.0, 1.0, 0.0], 5.0)\nassert p2.max() < p1.max() and np.isclose(p2.sum(), 1)', hidden: true },
          { name: 'low temperature approaches greedy', code: 'p = softmax_t([3.0, 2.9, 0.0], 0.01)\nassert p[0] > 0.99', hidden: true }
        ]
      },
      {
        id: 'top-k-top-p', title: 'Top-k and Nucleus Filtering', difficulty: 'medium',
        prompt: `Implement \`filter_probs(probs, k=None, p=None)\`:

1. Order token indices by probability, highest first: \`np.argsort(-probs, kind="stable")\`.
2. If \`k\` is given, keep only the first k of them.
3. If \`p\` is given, of those keep the **smallest prefix whose cumulative probability is ≥ p** (always at least one token; if the prefix never reaches p, keep them all).
4. Return a new array of the same length: kept tokens renormalized to sum to 1, every other token 0.

\`\`\`text
filter_probs([0.5, 0.2, 0.3], k=2)     -> [0.625, 0, 0.375]
filter_probs([0.5, 0.2, 0.3], p=0.75)  -> [0.625, 0, 0.375]    (0.5 + 0.3 = 0.8 ≥ 0.75)
filter_probs([0.5, 0.2, 0.3], p=0.4)   -> [1, 0, 0]
\`\`\``,
        starter: `
import numpy as np

def filter_probs(probs, k=None, p=None):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")
    # TODO
    pass
`,
        solution: `
import numpy as np

def filter_probs(probs, k=None, p=None):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")
    if k is not None:
        order = order[:k]
    if p is not None:
        cum = np.cumsum(probs[order])
        n = int(np.searchsorted(cum, p - 1e-12)) + 1
        order = order[:n]
    out = np.zeros_like(probs)
    out[order] = probs[order]
    return out / out.sum()
`,
        hints: ['`np.cumsum(probs[order])` gives the running total in sorted order.', '`np.searchsorted(cum, p)` finds the first position where the total reaches p; keep one more than that index.', 'Renormalize the kept probabilities at the end.'],
        explain: `Top-k and top-p compose naturally: k caps the candidate list, p trims it further to the "nucleus" that holds most of the probability. The tiny 1e-12 slack makes a prefix that sums to exactly p count as reaching it despite floating-point rounding.`,
        tests: [
          { call: 'filter_probs([0.5, 0.2, 0.3], k=2)', expect: 'np.array([0.625, 0.0, 0.375])' },
          { call: 'filter_probs([0.5, 0.2, 0.3], p=0.75)', expect: 'np.array([0.625, 0.0, 0.375])' },
          { call: 'filter_probs([0.5, 0.2, 0.3], p=0.4)', expect: 'np.array([1.0, 0.0, 0.0])' },
          { name: 'no filtering', call: 'filter_probs([0.1, 0.6, 0.3])', expect: 'np.array([0.1, 0.6, 0.3])', hidden: true },
          { name: 'k and p together', call: 'filter_probs([0.05, 0.4, 0.25, 0.2, 0.1], k=3, p=0.6)', expect: 'np.array([0.0, 0.4, 0.25, 0.0, 0.0]) / 0.65', hidden: true },
          { name: 'random distribution', setup: 'rng = np.random.default_rng(0)\nz = rng.normal(size=50) * 2\nprobs = np.exp(z) / np.exp(z).sum()', call: 'filter_probs(probs, k=20, p=0.9)', hidden: true }
        ]
      },
      {
        id: 'greedy-generate', title: 'Greedy Generation Loop', difficulty: 'medium',
        prompt: `Implement \`generate(next_logits, prompt, max_new, eos=None)\`:

- \`next_logits(ids)\` is the "model": a function that takes the list of ids so far and returns a NumPy array of next-token logits.
- Starting from a **copy** of \`prompt\` (a list of ints), repeat up to \`max_new\` times: get the logits, append the argmax (as a Python \`int\`), and stop right after appending \`eos\` if it is generated.
- Return the full list (prompt + generated tokens).

\`\`\`text
table = np.array([[0, 5, 1],     # after token 0, token 1 is most likely
                  [0, 0, 9],     # after 1 -> 2
                  [7, 0, 0]])    # after 2 -> 0
model = lambda ids: table[ids[-1]]
generate(model, [0], 4)          -> [0, 1, 2, 0, 1]
generate(model, [0], 4, eos=2)   -> [0, 1, 2]
\`\`\``,
        starter: `
import numpy as np

def generate(next_logits, prompt, max_new, eos=None):
    ids = list(prompt)
    # TODO
    return ids
`,
        solution: `
import numpy as np

def generate(next_logits, prompt, max_new, eos=None):
    ids = list(prompt)
    for _ in range(max_new):
        nxt = int(np.argmax(next_logits(ids)))
        ids.append(nxt)
        if eos is not None and nxt == eos:
            break
    return ids
`,
        hints: ['Loop `max_new` times; call `next_logits(ids)` each time with the growing list.', 'Check for `eos` after appending so it is included in the output.'],
        explain: `This loop is the core of every \`model.generate()\`. Swapping \`np.argmax\` for sampling from \`softmax_t\` + \`filter_probs\` gives you temperature, top-k and top-p generation.`,
        tests: [
          { setup: 'table = np.array([[0, 5, 1], [0, 0, 9], [7, 0, 0]])\nmodel = lambda ids: table[ids[-1]]', call: 'generate(model, [0], 4)', expect: '[0, 1, 2, 0, 1]' },
          { setup: 'table = np.array([[0, 5, 1], [0, 0, 9], [7, 0, 0]])\nmodel = lambda ids: table[ids[-1]]', call: 'generate(model, [0], 4, eos=2)', expect: '[0, 1, 2]' },
          { name: 'prompt is not modified', setup: 'table = np.array([[0, 5, 1], [0, 0, 9], [7, 0, 0]])\nmodel = lambda ids: table[ids[-1]]\nprompt = [2]', code: 'out = generate(model, prompt, 3)\nassert prompt == [2], "do not modify the prompt list"\nassert out == [2, 0, 1, 2]', hidden: true },
          { name: 'the model sees the whole context', setup: 'def model(ids):\n    z = np.zeros(10)\n    z[sum(ids) % 10] = 1.0\n    return z', call: 'generate(model, [3, 4], 6)', hidden: true },
          { name: 'max_new = 0', setup: 'model = lambda ids: np.array([0.0, 1.0])', call: 'generate(model, [1], 0)', expect: '[1]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Holtzman et al. — The Curious Case of Neural Text Degeneration (nucleus sampling)', url: 'https://arxiv.org/abs/1904.09751' },
      { title: 'Hugging Face — How to generate text: decoding methods', url: 'https://huggingface.co/blog/how-to-generate' }
    ]
  },

  // =====================================================================================
  {
    id: 'pretraining', branch: 'llm', title: 'Pretraining & Scaling Laws', short: 'Pretraining', level: 3, minutes: 40, prereqs: ['gpt'],
    tags: ['pretraining', 'scaling laws', 'chinchilla', 'compute', 'flops', 'data', 'deduplication', 'base model', 'emergent abilities'],
    summary: 'How base models are made: web-scale data pipelines, next-token training at scale, the C ≈ 6·N·D compute rule, scaling laws and compute-optimal (Chinchilla) training — and what a base model can and cannot do.',
    why: `Pretraining is where an LLM acquires nearly all of its knowledge and abilities — and where nearly all of the cost goes (millions of GPU-hours for frontier models). Scaling laws turned model building from guesswork into engineering: they predict how the loss improves with more parameters, data and compute, and how to split a budget between them. Even if you never pretrain a model, these ideas explain model sizes, the debates about training data, and why "a smaller model trained for longer" became the norm.`,
    objectives: [
      'Describe the pretraining pipeline: collection, filtering, deduplication, mixing, tokenization, training',
      'Estimate training compute with C ≈ 6·N·D and convert it to GPU time',
      'Explain scaling laws and compute-optimal training (≈ 20 tokens per parameter)',
      'Explain what a base model is and why it still needs fine-tuning to be an assistant'
    ],
    sections: [
      { type: 'text', title: 'The pretraining recipe', md: `1. **Collect** text: web crawls (Common Crawl), books, Wikipedia, code, scientific papers, forums — trillions of tokens.
2. **Filter**: drop spam, boilerplate, machine-generated junk and toxic content (heuristics plus small quality classifiers), remove personal information, keep the languages you want.
3. **Deduplicate**: remove exact and near-duplicate documents (e.g. MinHash over word n-grams). Duplicates waste compute and encourage memorization.
4. **Mix**: choose the proportion of each source; upweight high-quality text; code and math help reasoning.
5. **Tokenize** everything and pack it into fixed-length sequences.
6. **Train** the GPT with next-token cross-entropy: AdamW, warmup + cosine decay, bf16 mixed precision, on thousands of GPUs with data, tensor and pipeline parallelism, checkpointing frequently.

Data quality matters as much as quantity: carefully filtered datasets (for example FineWeb-Edu, which keeps educational web pages) give markedly better models for the same compute.` },
      { type: 'text', title: 'How much compute?', md: `For a transformer with N parameters trained on D tokens:

$$ training FLOPs   C ≈ 6 · N · D $$
:::where
C — the total training compute, counted in floating-point operations (FLOPs)
N — the number of model parameters
D — the number of training tokens
6 — about 2 operations per parameter per token for the forward pass, plus 4 for the backward pass
:::

About 2·N floating-point operations per token for the forward pass and 4·N for the backward pass. (Generating one token at inference costs about 2·N.)

**Example.** A 7B model on 2T tokens: 6 × 7·10⁹ × 2·10¹² = 8.4·10²² FLOPs. An H100 GPU peaks at roughly 10¹⁵ bf16 FLOP/s; at a realistic 40% utilization (MFU) that's 8.4·10²² / 4·10¹⁴ ≈ 2.1·10⁸ GPU-seconds ≈ **2,430 GPU-days** — about 2.5 days on 1,000 GPUs.` },
      { type: 'code', title: 'Compute and time for famous models', code: `
def train_flops(n_params, n_tokens):
    return 6 * n_params * n_tokens

def gpu_days(flops, peak_flops=1e15, mfu=0.4):
    return flops / (peak_flops * mfu) / 86400

models = [("GPT-2 small (repro)", 124e6, 10e9), ("GPT-3", 175e9, 300e9),
          ("Chinchilla", 70e9, 1.4e12), ("Llama 3 8B", 8e9, 15e12)]
for name, N, D in models:
    C = train_flops(N, D)
    print(f"{name:20s} N={N:8.3g}  D={D:8.3g}  tokens/param={D / N:7.1f}  C={C:.2e}  ~{gpu_days(C):10,.0f} H100-days")
`, after: `GPT-3 saw fewer than 2 tokens per parameter; Llama 3 8B saw almost 1,900. The next sections explain why that shift happened.` },
      { type: 'text', title: 'Scaling laws', md: `Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") found that loss falls as a smooth **power law** in model size N and data D, across many orders of magnitude:

$$ L(N, D) ≈ E + A / N^α + B / D^β $$
:::where
L(N, D) — the expected loss (nats per token) of a model with N parameters trained on D tokens
E — the irreducible loss: the inherent unpredictability of text, which no model can beat
A, B — fitted constants that set how much model size and data hold the loss back
α, β — "alpha, beta": fitted exponents (about 0.3) that set how fast each term shrinks
N^α — N raised to the power α
:::

E is the irreducible loss (the inherent unpredictability of text); the other two terms shrink as you add parameters or data, with diminishing returns. Because the curves are so smooth, labs fit them on small runs and *predict* the loss of a run 100× bigger before paying for it.

**Compute-optimal training.** For a fixed budget C = 6ND, the loss is lowest when N and D grow together — about **20 training tokens per parameter**. GPT-3 (175B parameters, 300B tokens) was heavily under-trained; Chinchilla (70B parameters, 1.4T tokens) used about the same compute as the 280B-parameter Gopher and beat it.

**Beyond compute-optimal.** Chinchilla-optimal minimizes *training* cost. A model that will answer billions of requests should be smaller and trained far longer (Llama 3 8B: ~1,900 tokens per parameter), because inference cost grows with N. Today, high-quality data — not compute — is often the bottleneck.` },
      { type: 'check', question: { type: 'fill', q: 'Using C ≈ 6·N·D, how many FLOPs does training a 1B-parameter model on 20B tokens take? (e.g. 3.4e19)', answer: ['1.2e20', '1.2e+20', '1.2 × 10^20', '1.2x10^20', '1.2*10^20', '120000000000000000000'], display: '1.2·10²⁰', explain: '6 × 10⁹ × 2·10¹⁰ = 1.2·10²⁰ FLOPs.' } },
      { type: 'code', title: 'The compute-optimal trade-off', code: `
import numpy as np
# L(N, D) = E + A / N^alpha + B / D^beta  with constants refit to the Chinchilla data (Besiroglu et al., 2024)
E, A, B, alpha, beta = 1.8172, 482.01, 2085.43, 0.3478, 0.3658
def loss(N, D):
    return E + A / N ** alpha + B / D ** beta

print("Gopher      280B params, 300B tokens:", round(loss(280e9, 300e9), 4))
print("Chinchilla   70B params, 1.4T tokens:", round(loss(70e9, 1.4e12), 4), "<- similar compute, lower loss")
print()
C = 6 * 70e9 * 1.4e12
print(f"Same budget C = {C:.2e} FLOPs, split differently:")
for N in [5e9, 20e9, 70e9, 150e9, 400e9, 1e12]:
    D = C / (6 * N)
    print(f"  N = {N:8.2e}  D = {D:8.2e}  tokens/param = {D / N:8.1f}  loss = {loss(N, D):.4f}")
`, after: `Too small a model can't use the data; too big a model sees too little data. The minimum sits near 70B parameters (~20 tokens per parameter), and it is shallow: being 2× off costs little.` },
      { type: 'chart', title: "IsoFLOP curves: the best model size for each budget", md: `For each fixed compute budget C, every model size N implies a number of training tokens D = C / (6N). The fitted scaling law then predicts the loss:`, after: `Each curve is a U: models that are too small cannot use their data, and models that are too big see too few tokens. The bottom of each U is the compute-optimal size, and it moves right roughly in proportion to √C, so ten times more compute buys a model about three times larger trained on three times more data. The minima are shallow, which is why practitioners deliberately choose smaller models trained on more tokens: they cost less to serve.`, chart: {"kind":"line","x":[100000000,126000000,158000000,200000000,251000000,316000000,398000000,501000000,631000000,794000000,1000000000,1260000000,1580000000,2000000000,2510000000,3160000000,3980000000,5010000000,6310000000,7940000000,10000000000,12600000000,15800000000,20000000000,25100000000,31600000000,39800000000,50100000000,63100000000,79400000000,100000000000,126000000000,158000000000,200000000000,251000000000,316000000000,398000000000,501000000000,631000000000,794000000000,1000000000000],"xLog":true,"points":false,"series":[{"name":"budget 1e20 FLOPs","y":[2.7765,2.7297,2.6888,2.6537,2.6241,2.5998,2.5807,2.5667,2.5577,2.5536,2.5546,2.5605,2.5715,2.5876,2.609,2.6359,2.6684,2.7067,2.7513,2.8024,2.8603,2.9255,2.9985,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]},{"name":"budget 1e21 FLOPs","y":[2.6833,2.6282,2.5785,2.5336,2.4935,2.4577,2.4261,2.3986,2.3748,2.3547,2.3381,2.325,2.3153,2.3089,2.3058,2.306,2.3096,2.3164,2.3267,2.3404,2.3578,2.3788,2.4038,2.4328,2.4661,2.504,2.5466,2.5944,2.6476,2.7066,2.7719,2.844,2.9233,null,null,null,null,null,null,null,null]},{"name":"budget 1e22 FLOPs","y":[2.6431,2.5846,2.531,2.4819,2.4372,2.3965,2.3596,2.3261,2.296,2.269,2.2449,2.2236,2.205,2.1889,2.1753,2.164,2.155,2.1483,2.1438,2.1415,2.1413,2.1434,2.1476,2.1542,2.163,2.1742,2.1878,2.2041,2.223,2.2447,2.2694,2.2973,2.3285,2.3633,2.402,2.4448,2.492,2.5439,2.601,2.6637,2.7323]},{"name":"budget 1e23 FLOPs","y":[2.6258,2.5657,2.5105,2.4597,2.413,2.3702,2.3309,2.2949,2.2621,2.2321,2.2047,2.1799,2.1575,2.1372,2.119,2.1028,2.0885,2.0759,2.065,2.0558,2.0481,2.0419,2.0373,2.0341,2.0324,2.0321,2.0333,2.0359,2.0401,2.0457,2.053,2.0618,2.0724,2.0847,2.0988,2.115,2.1332,2.1536,2.1764,2.2017,2.2297]}],"xLabel":"parameters N (log scale); data D = C / (6N)","yLabel":"predicted loss (nats per token)"} },
      { type: 'text', title: "Data: the other half of scaling", md: `Scaling laws treat all tokens as interchangeable, but the quality of the data changes the constants dramatically:

- **Filtering and deduplication.** Removing near-duplicate documents and low-quality pages lets a model reach the same loss with fewer tokens and reduces verbatim memorisation. Quality classifiers that favour text resembling textbooks and reference material have produced small models that punch far above their size.
- **Mixtures.** The shares of code, mathematics, multilingual text and dialogue shape what the model is good at; adding code, for example, tends to help reasoning as well.
- **Repetition.** When fresh data runs out, repeating it for up to about four epochs is almost as good as new data; beyond that the returns fall off quickly.
- **Synthetic data.** Text generated by stronger models (worked solutions, rewritten documents, reasoning traces) makes up a growing part of training sets, especially for small models and later training stages.
- **The data wall.** The stock of high-quality public human text is finite, estimated at a few hundred trillion tokens, and frontier training runs already use tens of trillions. This is why so much effort now goes into data quality, synthetic data and learning from interaction (reinforcement learning) rather than ever larger crawls.` },
      { type: 'text', title: 'Base models and emergent abilities', md: `Pretraining produces a **base model**: a superb document completer, not an assistant. Ask a base model "What is the capital of France?" and it may continue with more quiz questions, because that's what such text looks like on the web. Instruction tuning and preference tuning (the next lessons) turn it into a helpful assistant, but almost all of its knowledge and skills come from pretraining.

As models scale, some abilities — multi-step arithmetic, following few-shot examples, writing working code — improve sharply, which is often called **emergence**. Part of the apparent jump comes from all-or-nothing metrics (an answer is right or wrong) while the underlying loss improves smoothly, but the practical effect is real.` },
      { type: 'check', question: { type: 'tf', q: 'A base model that has only been pretrained reliably follows instructions like a chat assistant.', answer: false, explain: 'It continues documents; following instructions comes from fine-tuning.' } }
    ],
    when: `- Pretrain from scratch only with a large budget or a domain/language poorly covered by existing models (code, biology, a low-resource language).
- Otherwise start from an open base model and do **continued pretraining** (more domain text) and/or fine-tuning.
- Use scaling-law fits on small runs to choose model size and token count before a big run.`,
    pitfalls: `- Training on unfiltered, duplicated data and blaming the architecture.
- Contaminating training data with benchmark test sets (inflated scores).
- Picking the model size first and the data second — plan both from the compute budget.
- Forgetting that inference cost, not just training cost, matters for a deployed model.
- Mistaking a base model's document-style completions for a bug.`,
    takeaways: [
      'Pretraining = next-token prediction on trillions of filtered, deduplicated tokens.',
      'Training compute C ≈ 6·N·D FLOPs; inference ≈ 2·N FLOPs per token.',
      'Loss follows smooth power laws in N and D; fit small runs to predict big ones.',
      'Compute-optimal ≈ 20 tokens per parameter; deployed models are often trained far longer.',
      'The result is a base model: knowledgeable, but it needs fine-tuning to act as an assistant.'
    ],
    quiz: [
      { type: 'mcq', q: 'Roughly how many FLOPs per token does a training step cost for an N-parameter transformer?', options: ['N', '2N', '6N', 'N²'], answer: 2, explain: '≈ 2N for the forward pass and ≈ 4N for the backward pass.' },
      { type: 'mcq', q: 'What did the Chinchilla paper show?', options: ['Bigger models are always better at a fixed compute budget', 'For a fixed compute budget, model size and data should grow together (~20 tokens per parameter)', 'Data quality does not matter', 'Scaling laws do not hold for language'], answer: 1, explain: 'Many earlier models were too big for the amount of data they saw.' },
      { type: 'mcq', q: 'Why are popular open models trained on far more than 20 tokens per parameter?', options: ['To lower training cost', 'Smaller models are cheaper to run, so extra training compute pays off over billions of inference requests', 'Because scaling laws are wrong', 'Because they have more layers'], answer: 1, explain: 'Compute-optimal minimizes training cost, not total lifetime cost.' },
      { type: 'tf', q: 'Deduplicating pretraining data reduces wasted compute and memorization.', answer: true, explain: 'Repeated documents add little new information and encourage verbatim recall.' },
      { type: 'fill', q: 'Compute-optimal split: with C = 6·N·D and D = 20·N, what is N (in billions) for C = 1.2e21 FLOPs?', answer: ['3.16', '3.2', '3.162', 'sqrt(10)'], display: '≈ 3.16B', explain: 'C = 120·N² → N = √(1.2e21 / 120) = √(1e19) ≈ 3.16·10⁹.' },
      {"type":"mcq","q":"With ten times more compute, how does the compute-optimal model size roughly change?","options":["It stays the same","About 3 times larger (√10), trained on about 3 times more data","10 times larger, with the same data","100 times larger"],"answer":1,"explain":"The optimum scales roughly like N ∝ √C and D ∝ √C, so a factor of 10 in compute is split about evenly between model size and data."},
      {"type":"tf","q":"Repeating the same training data for a few epochs is almost as useful as fresh data, but the returns fall off quickly with many more repetitions.","answer":true,"explain":"Experiments on data-constrained scaling found up to about 4 epochs nearly as good as new data, with rapidly diminishing value after that."}
    ],
    problems: [
      {
        id: 'training-compute', title: 'Training Compute Estimates', difficulty: 'easy',
        prompt: `Implement:

- \`train_flops(n_params, n_tokens)\` → 6 · N · D.
- \`gpu_days(flops, n_gpus, peak_flops, mfu)\` → wall-clock **days**: flops / (n_gpus × peak_flops × mfu) / 86,400.
- \`inference_flops_per_token(n_params)\` → 2 · N.

\`\`\`text
train_flops(7e9, 2e12)              ->  8.4e22
gpu_days(8.4e22, 1000, 1e15, 0.4)   ->  ≈ 2.43 days
\`\`\``,
        starter: `
def train_flops(n_params, n_tokens):
    # TODO
    pass

def gpu_days(flops, n_gpus, peak_flops, mfu):
    # TODO
    pass

def inference_flops_per_token(n_params):
    # TODO
    pass
`,
        solution: `
def train_flops(n_params, n_tokens):
    return 6 * n_params * n_tokens

def gpu_days(flops, n_gpus, peak_flops, mfu):
    return flops / (n_gpus * peak_flops * mfu) / 86400

def inference_flops_per_token(n_params):
    return 2 * n_params
`,
        hints: ['There are 86,400 seconds in a day.', 'MFU (model FLOPs utilization) is the fraction of peak speed you actually achieve.'],
        explain: `Back-of-the-envelope compute math like this is how labs budget training runs and how you can sanity-check claims about model costs.`,
        tests: [
          { call: 'train_flops(7e9, 2e12)', expect: '8.4e22' },
          { call: 'gpu_days(8.4e22, 1000, 1e15, 0.4)', expect: '8.4e22 / (1000 * 1e15 * 0.4) / 86400' },
          { call: 'inference_flops_per_token(70e9)', expect: '1.4e11' },
          { name: 'other values', call: '[gpu_days(train_flops(1.3e9, 2.6e10), 8, 3.12e14, 0.5), inference_flops_per_token(1.3e9)]', hidden: true }
        ]
      },
      {
        id: 'chinchilla', title: 'Compute-Optimal Model Size', difficulty: 'medium',
        prompt: `Implement:

- \`scaling_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)\` → E + A / N^alpha + B / D^beta (the Chinchilla paper's original fit). It should also work when N and D are NumPy arrays.
- \`compute_optimal(C, tokens_per_param=20)\` → \`(N, D)\` such that 6·N·D = C and D = tokens_per_param · N. Solve: C = 6·r·N² ⇒ N = √(C / (6r)).

\`\`\`text
compute_optimal(6 * 20 * 1e18)  ->  (1e9, 2e10)      # 1B parameters, 20B tokens
\`\`\``,
        starter: `
import numpy as np

def scaling_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    # TODO
    pass

def compute_optimal(C, tokens_per_param=20):
    # TODO
    pass
`,
        solution: `
import numpy as np

def scaling_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    return E + A / N ** alpha + B / D ** beta

def compute_optimal(C, tokens_per_param=20):
    N = (C / (6 * tokens_per_param)) ** 0.5
    return N, tokens_per_param * N
`,
        hints: ['Substitute D = r·N into C = 6·N·D.', 'Use `** 0.5` or `math.sqrt` for the square root.'],
        explain: `Plug in Chinchilla's own budget (≈ 5.76e23 FLOPs) and you get ≈ 69B parameters and ≈ 1.4T tokens — the model they actually trained.`,
        tests: [
          { call: 'compute_optimal(6 * 20 * 1e18)', expect: '(1e9, 2e10)' },
          { call: 'scaling_loss(70e9, 1.4e12)', expect: '1.69 + 406.4 / 70e9 ** 0.34 + 410.7 / 1.4e12 ** 0.28' },
          { name: 'Chinchilla budget', call: 'compute_optimal(5.76e23)', hidden: true },
          { name: 'custom ratio', call: 'compute_optimal(1e22, tokens_per_param=100)', hidden: true },
          { name: 'vectorized loss', call: 'scaling_loss(np.array([1e8, 1e9, 1e10]), np.array([2e9, 2e10, 2e11]))', hidden: true },
          { name: 'more data lowers loss', code: 'assert scaling_loss(1e9, 1e11) < scaling_loss(1e9, 1e10)', hidden: true }
        ]
      },
      {
        id: 'near-dedup', title: 'Near-Duplicate Detection', difficulty: 'medium',
        prompt: `Web crawls are full of near-copies of the same page. Implement:

- \`shingles(text, n=3)\` → the **set** of word n-grams (tuples) of \`text.lower().split()\`. If the text has fewer than n words, return a set containing the single tuple of all its words.
- \`jaccard(a, b)\` → |a ∩ b| / |a ∪ b| for two sets (return 1.0 if both are empty).
- \`dedup(docs, threshold=0.5, n=3)\` → the list of documents to keep: go through \`docs\` in order and keep a document only if its Jaccard similarity to **every** already-kept document is below \`threshold\`.

\`\`\`text
dedup(["the cat sat on the mat today",
       "the cat sat on the mat today!",      # shares 4 of 6 shingles with doc 0 -> dropped
       "dogs bark loudly at night"])
-> ["the cat sat on the mat today", "dogs bark loudly at night"]
\`\`\``,
        starter: `
def shingles(text, n=3):
    # TODO
    pass

def jaccard(a, b):
    # TODO
    pass

def dedup(docs, threshold=0.5, n=3):
    # TODO
    pass
`,
        solution: `
def shingles(text, n=3):
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def dedup(docs, threshold=0.5, n=3):
    kept, kept_shingles = [], []
    for d in docs:
        s = shingles(d, n)
        if all(jaccard(s, k) < threshold for k in kept_shingles):
            kept.append(d)
            kept_shingles.append(s)
    return kept
`,
        hints: ['A set comprehension over `range(len(words) - n + 1)` builds the n-grams.', 'Python sets support `&` (intersection) and `|` (union).', 'Store the shingle sets of kept documents so you only compute them once.'],
        explain: `Comparing every pair is O(n²) — fine for thousands of documents, impossible for billions. Production pipelines estimate the same Jaccard similarity with **MinHash** signatures and locality-sensitive hashing, so only likely duplicates are ever compared.`,
        tests: [
          { call: 'shingles("A b c d", 3)', expect: '{("a", "b", "c"), ("b", "c", "d")}' },
          { call: 'jaccard({1, 2, 3}, {2, 3, 4})', expect: '0.5' },
          { call: 'dedup(["the cat sat on the mat today", "the cat sat on the mat today!", "dogs bark loudly at night"])', expect: '["the cat sat on the mat today", "dogs bark loudly at night"]' },
          { name: 'short texts', call: 'shingles("hi there", 3)', expect: '{("hi", "there")}', hidden: true },
          { name: 'two empty sets', call: 'jaccard(set(), set())', expect: '1.0', hidden: true },
          { name: 'corpus with planted near-copies', setup: 'rng = np.random.default_rng(0)\nvocab = ["data", "model", "train", "token", "loss", "layer", "learn", "graph", "web", "text"]\nbase = [" ".join(rng.choice(vocab, size=12)) for _ in range(6)]\ndocs = []\nfor b in base:\n    docs.append(b)\n    w = b.split()\n    w[int(rng.integers(0, 12))] = "zzz"\n    docs.append(" ".join(w))', call: 'dedup(docs, 0.6)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Kaplan et al. — Scaling Laws for Neural Language Models', url: 'https://arxiv.org/abs/2001.08361' },
      { title: 'Hoffmann et al. — Training Compute-Optimal Large Language Models (Chinchilla)', url: 'https://arxiv.org/abs/2203.15556' },
      { title: 'Besiroglu et al. — Chinchilla Scaling: A Replication Attempt', url: 'https://arxiv.org/abs/2404.10102' },
      { title: 'Hugging Face — FineWeb: decanting the web for the finest text data', url: 'https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1' }
    ]
  },

  // =====================================================================================
  {
    id: 'fine-tuning', branch: 'llm', title: 'Fine-Tuning & Instruction Tuning', short: 'Fine-Tuning', level: 3, minutes: 40, prereqs: ['pretraining'],
    tags: ['fine-tuning', 'sft', 'instruction tuning', 'chat template', 'loss masking', 'transfer learning', 'catastrophic forgetting'],
    summary: 'Adapting a pretrained model to follow instructions or do a task: supervised fine-tuning (SFT), chat templates and special tokens, masking the loss to the assistant\'s tokens, data quality, hyperparameters, and when fine-tuning beats prompting or RAG.',
    why: `A base model completes documents; an assistant follows instructions. The bridge is **fine-tuning**: continuing training on a smaller, curated dataset of examples of the behaviour you want. It's also how teams specialize open models (Llama, Mistral, Qwen, Gemma) for their domain, tone or output format. Knowing how SFT data is formatted and how the loss is masked is essential for doing it correctly — and knowing when *not* to fine-tune saves a lot of time and money.`,
    objectives: [
      'Explain transfer learning and supervised fine-tuning (SFT) of LLMs',
      'Format conversations with a chat template and special tokens',
      'Mask the loss so the model learns only from the assistant\'s tokens',
      'Decide between prompting, RAG and fine-tuning; avoid catastrophic forgetting and overfitting'
    ],
    sections: [
      { type: 'demo', title: 'From base model to assistant', demo: 'dg-pipeline', after: `Fine-tuning is **transfer learning**: the model reuses everything it learned, the same idea as fine-tuning an ImageNet CNN on X-ray images. Typical SFT settings: a low learning rate (≈ 1e-5 to 2e-5 for full fine-tuning, ≈ 1e-4 for LoRA), 1–3 epochs, and small but high-quality data. The LIMA paper showed that about 1,000 carefully curated examples can already produce a strong assistant from a good base model: **quality beats quantity**.` },
      { type: 'text', title: 'Chat templates and special tokens', md: `The model only ever sees one token sequence, so a conversation is serialized with special tokens that mark who is speaking:

\`\`\`text
<|system|>
You are a helpful assistant.<|end|>
<|user|>
What is 2 + 2?<|end|>
<|assistant|>
2 + 2 = 4.<|end|>
\`\`\`

Every model family has its own exact template (ChatML, Llama 3's header tokens, Gemma's turn markers…). Using a different template at inference than in training badly degrades a fine-tuned model, so libraries ship it with the tokenizer: in Hugging Face, \`tokenizer.apply_chat_template(messages)\`. At inference the prompt ends with the assistant header — the *generation prompt* — so the model's next tokens are the reply.` },
      { type: 'text', title: 'Masking the loss', md: `In SFT we usually train only on the **assistant's** tokens: the model should learn to *produce* answers, not to predict users' questions or the system prompt. Give every other token the label −100 (PyTorch's \`ignore_index\` for cross-entropy), so those positions contribute nothing to the loss or the gradient — the model still *reads* them through attention.

\`\`\`text
tokens:  <|user|>  What  is  2+2  ?  <|end|>  <|assistant|>  4    .    <|end|>
labels:  -100      -100  ... ...  .  -100     -100           4    .    <|end|>
\`\`\`

Include the end-of-turn token in the labels, or the model never learns when to stop. (Libraries shift labels by one position internally, so labels are written aligned with input_ids.)` },
      { type: 'code', title: 'Formatting and masking an SFT example', code: `
import numpy as np
vocab = {}
def tok(text):                                   # toy tokenizer: one id per whitespace-separated piece
    return [vocab.setdefault(w, len(vocab)) for w in text.split()]

conversation = [("system", "You are terse."), ("user", "Capital of France ?"), ("assistant", "Paris . <|end|>")]
input_ids, labels = [], []
for role, text in conversation:
    header = tok(f"<|{role}|>")
    body = tok(text) if role == "assistant" else tok(text + " <|end|>")
    input_ids += header + body
    labels += [-100] * len(header) + (body if role == "assistant" else [-100] * len(body))

inv = {i: w for w, i in vocab.items()}
for i, l in zip(input_ids, labels):
    print(f"{inv[i]:>14}   label: {'(ignored)' if l == -100 else inv[l]}")

token_losses = np.random.default_rng(0).uniform(0.5, 3.0, len(labels))   # pretend per-token losses
mask = np.array(labels) != -100
print("mean loss over all tokens:", round(token_losses.mean(), 3), "| SFT loss (assistant tokens only):", round(token_losses[mask].mean(), 3))
` },
      { type: 'check', question: { type: 'mcq', q: 'Why are user and system tokens usually given the label −100 during SFT?', options: ['To make training faster', 'So the loss only teaches the model to produce assistant replies, not to predict user messages', 'Because they are padding', 'To hide them from attention'], answer: 1, explain: 'They are still visible through attention; they just don\'t contribute to the loss.' } },
      { type: 'text', title: 'When to fine-tune — and when not to', md: `| Goal | Best first tool |
| --- | --- |
| Answer from private or changing knowledge (docs, tickets, policies) | **RAG** — facts stay updatable and answers can cite sources |
| A consistent format, tone or style | prompting with examples; **fine-tune** if prompts get long or unreliable |
| A narrow, high-volume skill (classify tickets, extract fields) at low cost | **fine-tune** a small model — it can rival a large prompted one |
| Make the model "know" new facts | usually RAG; fine-tuning on facts is unreliable and can increase hallucination |

Risks: **catastrophic forgetting** (losing general abilities — mix in general data, use a low learning rate or LoRA), **overfitting** small datasets (watch validation loss; 1–3 epochs), faithfully learning the flaws of noisy data, and safety regressions (fine-tuning can undo safety training).` },
      { type: 'text', title: "Data and settings that matter", md: `**Quality beats quantity.** A few thousand carefully written, diverse examples often teach a format or a behaviour better than a million scraped ones (the LIMA experiments used just 1,000). Good fine-tuning data is:
- **representative** of the real requests, including the hard and unusual ones;
- **consistent**: the same kind of question always gets the same kind of answer, otherwise the model learns the inconsistency;
- **clean**: deduplicated, free of personal data you have no right to use, and checked against your evaluation set so that test questions do not leak into training.

Synthetic data generated by a stronger model is common, but it inherits that model's mistakes and style, and the provider's terms may restrict using its outputs to train other models.

**Typical settings**, as starting points (they vary with the model and its size):

| Setting | Full fine-tuning | LoRA |
| --- | --- | --- |
| learning rate | about 1e-5 to 2e-5 | about 1e-4 to 2e-4 |
| epochs | 1 to 3 | 1 to 3 |
| batch size | as large as fits, often with gradient accumulation | the same |
| schedule | a short warm-up, then cosine or linear decay | the same |

Watch the validation loss **and** a few general benchmarks: fine-tuning that improves the target task while silently breaking general abilities is the most common failure, shown here in miniature.` },
      { type: 'code', title: "Catastrophic forgetting in miniature", md: `A small network learns digits 0 to 4, then is fine-tuned on digits 5 to 9: once on the new data only, and once with some old examples replayed.`, code: `import numpy as np, warnings
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")
X, y = load_digits(return_X_y=True)
X = X / 16.0
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
X, y = X[order], y[order]
Xtr, ytr, Xte, yte = X[:1300], y[:1300], X[1300:], y[1300:]
old_tr, new_tr, old_te, new_te = ytr < 5, ytr >= 5, yte < 5, yte >= 5

def experiment(replay):
    model = MLPClassifier(hidden_layer_sizes=(64,), learning_rate_init=0.01, random_state=0)
    Xa, ya = Xtr[old_tr], ytr[old_tr]
    for _ in range(30):                                   # stage 1: learn digits 0-4
        model.partial_fit(Xa, ya, classes=np.arange(10))
    Xb, yb = Xtr[new_tr], ytr[new_tr]
    if replay:                                            # mix in a fifth as many old examples
        keep = rng.choice(len(Xa), len(Xb) // 5, replace=False)
        Xb, yb = np.vstack([Xb, Xa[keep]]), np.concatenate([yb, ya[keep]])
    for _ in range(30):                                   # stage 2: fine-tune
        model.partial_fit(Xb, yb)
    return model.score(Xte[old_te], yte[old_te]), model.score(Xte[new_te], yte[new_te])

plain, replayed = experiment(False), experiment(True)
print(f"new data only:  old digits {plain[0]:.0%}, new digits {plain[1]:.0%}")
print(f"with replay:    old digits {replayed[0]:.0%}, new digits {replayed[1]:.0%}")
print("replay keeps most of the old task:", replayed[0] > 0.7 > plain[0])`, after: `Fine-tuned on new data only, the network forgets the old digits completely: nothing in its new training data asks it to output 0 to 4 any more, so it stops. Replaying a small share of old examples keeps about 85% of the old ability, at no cost to the new task.`, predict: {"type":"mcq","q":"After 30 epochs of fine-tuning on digits 5 to 9 only, what accuracy will the network have on digits 0 to 4?","options":["Still about 99%","About 70%","About 50%","About 0%"],"answer":3,"explain":"Every update now pushes towards the classes 5 to 9, and nothing pushes back, so the network stops predicting 0 to 4 at all.","expect":["replay keeps most of the old task: True"]} },
      { type: 'chart', title: "Catastrophic forgetting, and how replay prevents it", md: `A small network first learns digits 0 to 4, then is fine-tuned on digits 5 to 9:`, after: `Trained only on the new digits, the network learns them within a few epochs and almost completely forgets the old ones (accuracy on 0 to 4 drops from 100% to 0%): nothing in the new data asks it to keep answering 0 to 4. With a fifth as many old examples mixed into the fine-tuning data, the old task still collapses at first, while the new classes take over, but it recovers to about 85% within about a dozen epochs, and the new task is learned just as well. LLM fine-tuning uses the same remedies: replay some general data, keep the learning rate small, train few epochs, or change few weights (LoRA).`, chart: {"kind":"line","x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"points":false,"series":[{"name":"old task (digits 0–4), fine-tuned on new data only","y":[0.996,0.939,0.284,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"name":"old task, with 20% old examples mixed in","y":[0.996,0.943,0.349,0,0.004,0.065,0.264,0.421,0.471,0.54,0.632,0.736,0.793,0.828,0.854,0.862,0.858,0.862,0.858,0.854,0.854,0.854,0.854,0.858,0.858,0.862,0.858,0.847,0.854,0.851,0.854],"dash":true},{"name":"new task (digits 5–9), new data only","y":[0,0.008,0.487,0.589,0.661,0.847,0.852,0.936,0.941,0.949,0.966,0.966,0.97,0.97,0.966,0.97,0.975,0.97,0.975,0.975,0.979,0.975,0.979,0.983,0.983,0.983,0.983,0.983,0.983,0.983,0.983]},{"name":"new task, with replay","y":[0,0.004,0.504,0.551,0.657,0.754,0.835,0.869,0.907,0.928,0.941,0.953,0.962,0.962,0.97,0.97,0.97,0.97,0.97,0.975,0.975,0.979,0.979,0.979,0.979,0.983,0.983,0.983,0.983,0.983,0.983],"dash":true}],"xLabel":"fine-tuning epochs on the new task","yLabel":"test accuracy","yMin":0,"yMax":1} },
      { type: 'code', title: 'SFT with Hugging Face transformers (run locally)', lang: 'pytorch', code: `
# pip install torch transformers
# (In practice TRL's SFTTrainer does this formatting and masking for you.)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-0.5B-Instruct"                         # any small chat model
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)

messages = [{"role": "user", "content": "Summarize: the meeting moved to 3pm."},
            {"role": "assistant", "content": "Meeting now at 3pm."}]
prompt_ids = tok.apply_chat_template(messages[:1], add_generation_prompt=True)
full_ids = tok.apply_chat_template(messages)
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]   # learn only the reply

batch = {"input_ids": torch.tensor([full_ids]), "labels": torch.tensor([labels])}
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()
for step in range(3):
    out = model(**batch)                      # the model shifts the labels internally
    out.loss.backward()
    opt.step(); opt.zero_grad()
    print(step, out.loss.item())
` },
      { type: 'check', question: { type: 'tf', q: 'Fine-tuning on your company documents is the most reliable way to make a model answer questions about them accurately.', answer: false, explain: 'RAG is usually more reliable for knowledge: it stays current and can cite sources. Fine-tuning shines for behaviour, format and narrow skills.' } }
    ],
    when: `- A consistent output format, style or tone that prompting can't achieve reliably.
- Narrow, high-volume tasks where a small fine-tuned model is cheaper and faster than a large prompted one.
- Domain adaptation (medical or legal language) or a new language (continued pretraining + SFT).
- Try prompting first, RAG for knowledge, and fine-tuning when those plateau.`,
    pitfalls: `- Training on the prompt tokens too (no masking), or using a different chat template at inference.
- Too high a learning rate or too many epochs → forgetting and overfitting.
- Low-quality or inconsistent examples — the model learns their flaws faithfully.
- No held-out evaluation set and no comparison with a well-prompted baseline.
- Expecting fine-tuning to inject reliable factual knowledge.`,
    takeaways: [
      'SFT continues training a pretrained model on curated (instruction, response) examples.',
      'Conversations are serialized with a model-specific chat template and special tokens.',
      'Mask non-assistant tokens (label −100) so only replies are learned — including the end-of-turn token.',
      'Small, high-quality datasets with low learning rates and 1–3 epochs work best.',
      'Prompting → RAG → fine-tuning: fine-tune for behaviour and narrow skills, use RAG for knowledge.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is supervised fine-tuning (SFT) of an LLM?', options: ['Training from random weights on web text', 'Continuing training of a pretrained model on curated input → output examples', 'Adding a retrieval system', 'Compressing the weights to 4 bits'], answer: 1, explain: 'It adapts behaviour using demonstrations of the desired outputs.' },
      { type: 'mcq', q: 'At inference, a chat prompt should end with…', options: ['the end-of-sequence token', 'the assistant header (generation prompt) so the model writes the reply', 'the system prompt', 'a random token'], answer: 1, explain: 'The model then continues as the assistant.' },
      { type: 'tf', q: 'The end-of-turn token after the assistant\'s reply should usually be included in the training labels.', answer: true, explain: 'Otherwise the model never learns when to stop talking.' },
      { type: 'mcq', q: 'Which is a sign of catastrophic forgetting?', options: ['Training loss goes down', 'The fine-tuned model became worse at general tasks it used to do well', 'The model uses more memory', 'Validation loss equals training loss'], answer: 1, explain: 'Fine-tuning overwrote general abilities.' },
      { type: 'mcq', q: 'A typical learning rate for full fine-tuning of a 7B LLM is…', options: ['1.0', '0.1', 'around 1e-5', '1e-12'], answer: 2, explain: 'Much lower than pretraining peaks, to avoid destroying what the model knows.' },
      {"type":"mcq","q":"Which remedy helps against catastrophic forgetting during fine-tuning?","options":["A higher learning rate","Mixing some general or old-task data into the fine-tuning set","Training for many more epochs","Removing the validation set"],"answer":1,"explain":"Replaying old data keeps the old behaviour in the training signal; small learning rates, few epochs and parameter-efficient methods also help."},
      {"type":"tf","q":"A few thousand high-quality, diverse examples can teach a model a format or behaviour better than a much larger noisy dataset.","answer":true,"explain":"Fine-tuning mostly steers existing abilities, so consistency and quality of the examples matter more than sheer volume."}
    ],
    problems: [
      {
        id: 'chat-template', title: 'Build a Chat Template', difficulty: 'easy',
        prompt: `Implement \`format_chat(messages, add_generation_prompt=False)\`. \`messages\` is a list of dicts with "role" and "content". Each message becomes

\`\`\`text
<|ROLE|>\\nCONTENT<|end|>\\n
\`\`\`

(literally \`<|\` + role + \`|>\`, a newline, the content, \`<|end|>\`, a newline), and the messages are concatenated in order. If \`add_generation_prompt\` is True, append \`<|assistant|>\` followed by a newline.

Then implement \`extract_reply(generated)\` → the text before the first \`<|end|>\` (the whole string if there is none) — how a stop sequence cuts the model's output.

\`\`\`text
format_chat([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
-> "<|user|>\\nHi<|end|>\\n<|assistant|>\\n"
extract_reply("Hello!<|end|>\\n<|user|>\\n...")  ->  "Hello!"
\`\`\``,
        starter: `
def format_chat(messages, add_generation_prompt=False):
    # TODO
    pass

def extract_reply(generated):
    # TODO
    pass
`,
        solution: `
def format_chat(messages, add_generation_prompt=False):
    out = ""
    for m in messages:
        out += "<|" + m["role"] + "|>\\n" + m["content"] + "<|end|>\\n"
    if add_generation_prompt:
        out += "<|assistant|>\\n"
    return out

def extract_reply(generated):
    return generated.split("<|end|>")[0]
`,
        hints: ['Build the string with a loop, or with `"".join(...)` over the messages.', '`str.split("<|end|>")[0]` returns everything before the first marker.'],
        explain: `Real templates (ChatML, Llama 3) are exactly this kind of string formatting, usually written as Jinja templates shipped with the tokenizer. Getting one character wrong — a missing newline — measurably hurts a fine-tuned model.`,
        tests: [
          { call: 'format_chat([{"role": "user", "content": "Hi"}], add_generation_prompt=True)', expect: '"<|user|>\\nHi<|end|>\\n<|assistant|>\\n"' },
          { call: 'extract_reply("Hello!<|end|>\\n<|user|>\\n...")', expect: '"Hello!"' },
          { name: 'multi-turn conversation', call: 'format_chat([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}])', expect: '"<|system|>\\nBe brief.<|end|>\\n<|user|>\\n2+2?<|end|>\\n<|assistant|>\\n4<|end|>\\n"', hidden: true },
          { name: 'empty conversation', call: '[format_chat([]), format_chat([], add_generation_prompt=True)]', expect: '["", "<|assistant|>\\n"]', hidden: true },
          { name: 'no stop marker', call: 'extract_reply("unfinished reply")', expect: '"unfinished reply"', hidden: true }
        ]
      },
      {
        id: 'loss-mask', title: 'Mask the SFT Loss', difficulty: 'medium',
        prompt: `A tokenized conversation is a list of \`(role, token_ids)\` segments. Implement:

- \`build_labels(segments)\` → \`(input_ids, labels)\`: \`input_ids\` concatenates all token ids; \`labels\` copies the ids of **assistant** segments and uses \`-100\` for every token of any other role.
- \`masked_mean_loss(token_losses, labels)\` → the mean of \`token_losses\` over positions whose label is not −100 (return 0.0 if there are none).

\`\`\`text
build_labels([("user", [5, 6]), ("assistant", [7, 8, 9])])
-> ([5, 6, 7, 8, 9], [-100, -100, 7, 8, 9])
masked_mean_loss([1.0, 1.0, 2.0, 3.0, 4.0], [-100, -100, 7, 8, 9])  ->  3.0
\`\`\``,
        starter: `
import numpy as np

def build_labels(segments):
    input_ids, labels = [], []
    # TODO
    return input_ids, labels

def masked_mean_loss(token_losses, labels):
    # TODO
    pass
`,
        solution: `
import numpy as np

def build_labels(segments):
    input_ids, labels = [], []
    for role, ids in segments:
        input_ids += list(ids)
        labels += list(ids) if role == "assistant" else [-100] * len(ids)
    return input_ids, labels

def masked_mean_loss(token_losses, labels):
    losses = np.asarray(token_losses, dtype=float)
    mask = np.asarray(labels) != -100
    if not mask.any():
        return 0.0
    return float(losses[mask].mean())
`,
        hints: ['`[-100] * len(ids)` makes the ignored labels for a segment.', 'Build a boolean mask with `np.asarray(labels) != -100` and average only those losses.'],
        explain: `This is exactly what \`ignore_index=-100\` does inside PyTorch's cross-entropy: masked positions are excluded from both the sum and the count, so the loss is an average over the reply tokens only.`,
        tests: [
          { call: 'build_labels([("user", [5, 6]), ("assistant", [7, 8, 9])])', expect: '([5, 6, 7, 8, 9], [-100, -100, 7, 8, 9])' },
          { call: 'masked_mean_loss([1.0, 1.0, 2.0, 3.0, 4.0], [-100, -100, 7, 8, 9])', expect: '3.0' },
          { name: 'multi-turn', call: 'build_labels([("system", [1]), ("user", [2, 3]), ("assistant", [4]), ("user", [5]), ("assistant", [6, 7])])', expect: '([1, 2, 3, 4, 5, 6, 7], [-100, -100, -100, 4, -100, 6, 7])', hidden: true },
          { name: 'nothing to learn', call: 'masked_mean_loss([1.0, 2.0], [-100, -100])', expect: '0.0', hidden: true },
          { name: 'random conversation', setup: 'rng = np.random.default_rng(0)\nroles = ["system", "user", "assistant", "user", "assistant"]\nsegments = [(r, rng.integers(0, 100, int(rng.integers(1, 6))).tolist()) for r in roles]\nids, labs = build_labels(segments)\nlosses = rng.uniform(0, 5, len(ids))', call: '[build_labels(segments), masked_mean_loss(losses, labs)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Ouyang et al. — Training language models to follow instructions with human feedback (InstructGPT)', url: 'https://arxiv.org/abs/2203.02155' },
      { title: 'Zhou et al. — LIMA: Less Is More for Alignment', url: 'https://arxiv.org/abs/2305.11206' },
      { title: 'Hugging Face — Chat templates', url: 'https://huggingface.co/docs/transformers/main/en/chat_templating' }
    ]
  },

  // =====================================================================================
  {
    id: 'lora', branch: 'llm', title: 'LoRA & Parameter-Efficient Fine-Tuning', short: 'LoRA', level: 3, minutes: 40, prereqs: ['fine-tuning', 'eigen-svd'],
    tags: ['lora', 'qlora', 'peft', 'adapters', 'low-rank', 'fine-tuning', 'memory'],
    summary: 'Fine-tuning huge models cheaply: freeze the pretrained weights and learn a small low-rank update. How LoRA works, its hyperparameters (rank, alpha, target modules), merging, QLoRA, and other parameter-efficient methods.',
    why: `Full fine-tuning of a 7B model needs memory for the weights, gradients and Adam states of all 7 billion parameters — roughly 16 bytes each, over 100 GB. **LoRA** trains well under 1% of the parameters while matching full fine-tuning on most tasks, so a single consumer GPU can fine-tune a 7B model (and with QLoRA, much larger ones). It is the most common way open models are customized, and the resulting adapters are tiny files that can be swapped per task or per customer.`,
    objectives: [
      'Explain why full fine-tuning is memory-hungry',
      'Describe LoRA: a frozen W plus a trainable low-rank update (α/r)·B·A',
      'Count LoRA\'s trainable parameters and choose rank, alpha and target modules',
      'Merge adapters for inference and explain QLoRA'
    ],
    sections: [
      { type: 'text', title: 'The cost of full fine-tuning', md: `For every trainable parameter, mixed-precision training with Adam typically stores:

| Item | Bytes |
| --- | --- |
| bf16 weight | 2 |
| bf16 gradient | 2 |
| fp32 master copy of the weight | 4 |
| Adam first moment (m) | 4 |
| Adam second moment (v) | 4 |
| **Total** | **≈ 16** |

A 7B model therefore needs ≈ 112 GB before counting activations — several data-centre GPUs — and every fine-tuned variant is another full 14 GB copy of the weights.` },
      { type: 'text', title: 'Low-rank adaptation', md: `Hu et al. (2021) hypothesized that the weight *change* needed to adapt a model has a low "intrinsic rank". So instead of learning a full update ΔW, learn two thin matrices and keep W frozen:

$$ h = W x + (α / r) · B A x          W: d_out × d_in (frozen)     A: r × d_in     B: d_out × r $$
:::where
x — the layer's input vector, of length d_in
W — the pretrained weight matrix, d_out × d_in, kept frozen
A, B — the two small trainable matrices; their product B A has the same shape as W
r — the rank: the inner size shared by A and B, much smaller than d_in and d_out
α — "alpha": a scaling hyperparameter; α / r sets how strongly the update counts
h — the layer's output, of length d_out
:::

- The **rank** r is small: typically 4–64.
- A starts random and **B starts at zero**, so at step 0 BA = 0 and the model is exactly the pretrained one.
- α/r scales the update (α is often r or 2r), which keeps the effective learning rate stable when you change r.

**Parameter count**: a full d × d matrix has d² weights; LoRA trains r·(d_in + d_out). For d = 4096 and r = 8 that's 65,536 instead of 16.8M — 0.4%.

In code with row-vector inputs (x of shape (n, d_in), W stored as (d_in, d_out)), the same thing reads h = x W + (α/r)·(x A) B with A: (d_in, r) and B: (r, d_out).` },
      { type: 'demo', title: 'Diagram: a LoRA adapter', demo: 'dg-lora' },
      { type: 'code', title: 'LoRA in NumPy: only A and B are trained', code: `
import numpy as np
rng = np.random.default_rng(0)
d_in, d_out, r, alpha = 64, 64, 4, 8
W = rng.normal(0, 0.1, (d_in, d_out))                                  # frozen "pretrained" weight
delta_true = rng.normal(0, 0.3, (d_in, 2)) @ rng.normal(0, 0.3, (2, d_out))   # the task needs a rank-2 change
X = rng.normal(size=(512, d_in)); Y = X @ (W + delta_true)

A = rng.normal(0, 0.01, (d_in, r)); B = np.zeros((r, d_out))           # B = 0: starts as the base model
s, lr = alpha / r, 2.0
for step in range(301):
    H = X @ A                                                          # (n, r)
    err = X @ W + s * H @ B - Y
    loss = (err ** 2).mean()
    gB = s * H.T @ err * 2 / err.size                                  # gradients for A and B only
    gA = s * X.T @ (err @ B.T) * 2 / err.size
    A -= lr * gA; B -= lr * gB
    if step in (0, 25, 50, 100, 300):
        print(f"step {step:3d}   loss {loss:.8f}")
print(f"trainable parameters: {A.size + B.size} vs {W.size} in W ({(A.size + B.size) / W.size:.1%})")
`, after: `With just 12.5% of W's parameter count (and far less for real 4096-wide layers) the adapter recovers the rank-2 change almost perfectly, while W never changes.` },
      { type: 'check', question: { type: 'fill', q: 'How many trainable parameters does LoRA with rank r = 16 add to one 4096 × 4096 weight matrix?', answer: ['131072', '131,072'], explain: '16 × (4096 + 4096) = 131,072 — about 0.8% of the 16.8M weights.' } },
      { type: 'text', title: 'Using LoRA in practice', md: `- **Target modules**: the original paper adapted only the attention query and value projections; today all linear layers (q, k, v, o and the MLP) are common for the best quality.
- **Rank** 8–16 is a good default; raise it for big domain shifts. **Alpha** ≈ r or 2r. **Learning rate** ≈ 1e-4 to 2e-4 — higher than full fine-tuning.
- **Merging**: after training compute W' = W + (α/r)·B·A once, and inference costs exactly the same as the base model. Or keep adapters separate and hot-swap them, serving one base model with many adapters.
- **QLoRA** (Dettmers et al., 2023): store the frozen base model in **4-bit** (NF4) and train bf16 LoRA adapters on top — enough to fine-tune a 65B model on a single 48 GB GPU.
- **Other PEFT methods**: adapters (small bottleneck MLPs inserted in each block), prefix and prompt tuning (learned virtual tokens), IA³, DoRA.` },
      { type: 'chart', title: "Memory to fine-tune a 7-billion-parameter model", md: `A Llama-style 7B model (32 layers), with LoRA on all its linear layers; activations, which depend on batch size and sequence length, are not included:`, after: `Full fine-tuning needs about 16 bytes per parameter, roughly 108 GB, so it takes several large GPUs. LoRA freezes the base weights (2 bytes each, no gradients or optimizer state) and trains only 40 million adapter parameters, about 0.6% of the model: about 14 GB. QLoRA stores the frozen weights in 4 bits and fits in about 4.0 GB, within reach of a single consumer GPU.`, chart: {"kind":"bar","x":["full fine-tuning","LoRA (r = 16)","QLoRA (r = 16)"],"series":[{"name":"weights + gradients + optimizer state (GB, before activations)","y":[107.8,14.1,4]}],"xLabel":"method","yLabel":"GPU memory (GB)","yMin":0} },
      { type: 'text', title: "The parameter-efficient family", md: `LoRA is the most popular member of a family of **parameter-efficient fine-tuning** (PEFT) methods, which all freeze the pretrained weights and train something small:

| Method | What is trained | Notes |
| --- | --- | --- |
| LoRA | low-rank updates B·A added to chosen weight matrices | can be merged into the weights: no extra cost at inference |
| QLoRA | LoRA on top of a 4-bit quantized base model | fine-tunes large models on a single GPU |
| DoRA | separate updates of each weight vector's length and direction | often a little better than LoRA at the same rank |
| adapters | small bottleneck layers inserted into each block | an extra cost at inference unless fused |
| prefix and prompt tuning | learned "virtual tokens" in front of the input or of each layer | very few parameters, weaker on hard tasks |
| (IA)³ | learned scaling vectors for activations | tiny, but less expressive |

**Why low rank works.** Fine-tuning mostly *steers* abilities the model already has rather than teaching new ones, and that steering lives in a small subspace: experiments find that the weight changes made by full fine-tuning have a low "intrinsic dimension". When the new task is far from the pretraining data (a new language or modality), higher ranks or full fine-tuning help.

A practical bonus: adapters are small files, so one base model can serve many customers or tasks, loading a different adapter per request.` },
      { type: 'code', title: "Counting LoRA parameters for a 7B model", code: `layers, d, d_ff = 32, 4096, 11008                 # a Llama-style 7B model
total = 6.74e9
attention = 4 * (d + d)                           # q, k, v, o are d x d; LoRA adds r * (d_in + d_out) to each
mlp = 3 * (d + d_ff)                              # the gate, up and down projections
per_rank = layers * (attention + mlp)             # trainable parameters per unit of rank
for r in (4, 8, 16, 64):
    print(f"rank {r:3d}: {per_rank * r / 1e6:6.1f} M trainable parameters ({per_rank * r / total:.2%} of the model), "
          f"adapter file about {per_rank * r * 2 / 1e6:.0f} MB in bf16")
print("rank 16 trains under 1% of the parameters:", per_rank * 16 / total < 0.01)`, after: `Even at rank 64 the adapters hold under 2.5% of the parameters, and a rank-16 adapter file is about 80 MB, against roughly 13 GB for the model itself.`, predict: {"type":"mcq","q":"LoRA with rank 16 on every linear layer of a 7B model trains roughly what share of its parameters?","options":["About 0.6%","About 6%","About 25%","100%"],"answer":0,"explain":"Each matrix gets r·(d_in + d_out) parameters instead of d_in·d_out; summed over the model that is about 40 million of 6.7 billion.","expect":["rank 16 trains under 1% of the parameters: True"]} },
      { type: 'code', title: 'LoRA with Hugging Face PEFT (run locally)', lang: 'pytorch', code: `
# pip install torch transformers peft
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, config)
model.print_trainable_parameters()        # only a percent or two of the parameters are trainable
# ... train with your usual loop or the Hugging Face Trainer: only the LoRA matrices get gradients ...
model.save_pretrained("my-adapter")       # saves just the adapter (megabytes)
merged = model.merge_and_unload()         # fold (alpha/r)·B·A into W for fast inference
` },
      { type: 'check', question: { type: 'tf', q: 'Because B starts at zero, a LoRA model is identical to the pretrained model before training.', answer: true, explain: 'BA = 0, so the update contributes nothing until B is trained.' } }
    ],
    when: `- Fine-tuning large models on limited GPU memory (QLoRA for the largest).
- Serving many task- or customer-specific variants of one base model (swap small adapters).
- Most instruction and domain fine-tuning, where full fine-tuning brings little extra.
- Prefer full fine-tuning when you have the compute and need maximum quality on a big distribution shift (e.g. a new language).`,
    pitfalls: `- Initializing both A and B randomly — the model no longer starts from its pretrained behaviour.
- Forgetting the α/r scale when merging, so merged weights don't match the trained adapter.
- Too small a rank for a large domain shift, or a learning rate as low as full fine-tuning uses.
- Adapting only attention layers when the task needs MLP changes.
- Merging an adapter trained on a 4-bit base into a different-precision base without re-checking quality.`,
    takeaways: [
      'Full fine-tuning needs ≈ 16 bytes per parameter (weights, gradients, Adam states).',
      'LoRA freezes W and learns ΔW = (α/r)·B·A with a small rank r; B starts at zero.',
      'Trainable parameters per matrix: r·(d_in + d_out), usually well under 1% of the model.',
      'Adapters can be merged into W (no inference cost) or swapped per task.',
      'QLoRA = 4-bit frozen base + LoRA adapters → fine-tune very large models on one GPU.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does LoRA train?', options: ['All weights of the model', 'Only small low-rank matrices A and B added to frozen weights', 'Only the embedding layer', 'Only the biases'], answer: 1, explain: 'The pretrained weights stay frozen.' },
      { type: 'mcq', q: 'Why is B initialized to zero?', options: ['To save memory', 'So that BA = 0 and training starts from the exact pretrained model', 'Because quantization requires it', 'To make A unnecessary'], answer: 1, explain: 'The update starts at nothing and grows as needed.' },
      { type: 'fill', q: 'About how many bytes per parameter does full fine-tuning with Adam in mixed precision need (weights + gradients + optimizer states, no activations)?', answer: ['16'], explain: '2 (bf16 weight) + 2 (gradient) + 4 (fp32 master) + 4 + 4 (Adam m, v) = 16.' },
      { type: 'tf', q: 'After merging, a LoRA-fine-tuned model runs slower at inference than the base model.', answer: false, explain: 'The merged W\' has the same shape as W, so inference cost is unchanged.' },
      { type: 'mcq', q: 'What is QLoRA?', options: ['LoRA on a quantum computer', 'LoRA adapters trained on top of a 4-bit quantized, frozen base model', 'LoRA with rank 1', 'Quantizing only the adapters'], answer: 1, explain: 'It slashes the memory needed for the frozen weights.' },
      {"type":"mcq","q":"Why can one base model serve many LoRA fine-tunes efficiently?","options":["LoRA models are smaller than the base model","Each adapter is a small file, so different adapters can be loaded per task or customer on top of the same frozen weights","LoRA does not need GPUs","All tasks share one adapter"],"answer":1,"explain":"The large base weights are loaded once; only the small adapter changes between requests."},
      {"type":"fill","q":"LoRA adds r·(d_in + d_out) parameters to a weight matrix. How many does it add to a 4096 × 11008 matrix with r = 8?","answer":["120704","120,704"],"explain":"8 × (4,096 + 11,008) = 8 × 15,104 = 120,704, compared with 45 million in the full matrix."}
    ],
    problems: [
      {
        id: 'lora-params', title: 'Count LoRA Parameters', difficulty: 'easy',
        prompt: `Implement:

- \`lora_params(d_in, d_out, r)\` → trainable LoRA parameters for one weight matrix: r · (d_in + d_out).
- \`lora_model_params(n_layers, shapes, r)\` → the total when every layer has LoRA on each matrix listed in \`shapes\` (a list of \`(d_in, d_out)\` pairs).
- \`trainable_fraction(n_lora, n_total)\` → n_lora / (n_total + n_lora).

\`\`\`text
lora_params(4096, 4096, 8)                                -> 65536
lora_model_params(32, [(4096, 4096), (4096, 4096)], 8)    -> 4194304    # q and v in 32 layers
\`\`\``,
        starter: `
def lora_params(d_in, d_out, r):
    # TODO
    pass

def lora_model_params(n_layers, shapes, r):
    # TODO
    pass

def trainable_fraction(n_lora, n_total):
    # TODO
    pass
`,
        solution: `
def lora_params(d_in, d_out, r):
    return r * (d_in + d_out)

def lora_model_params(n_layers, shapes, r):
    return n_layers * sum(lora_params(a, b, r) for a, b in shapes)

def trainable_fraction(n_lora, n_total):
    return n_lora / (n_total + n_lora)
`,
        hints: ['A is r × d_in and B is d_out × r.', 'Sum over the listed matrices, then multiply by the number of layers.'],
        explain: `Even LoRA on every linear layer of a Llama-style 8B model at rank 16 is about 42M parameters — roughly half a percent — which is why adapters are megabytes, not gigabytes.`,
        tests: [
          { call: 'lora_params(4096, 4096, 8)', expect: '65536' },
          { call: 'lora_model_params(32, [(4096, 4096), (4096, 4096)], 8)', expect: '4194304' },
          { call: 'trainable_fraction(4194304, 7e9)', expect: '4194304 / (7e9 + 4194304)' },
          { name: 'non-square matrices', call: 'lora_params(4096, 11008, 16)', expect: '16 * (4096 + 11008)', hidden: true },
          { name: 'all linear layers of a Llama-style model', call: 'lora_model_params(32, [(4096, 4096), (4096, 1024), (4096, 1024), (4096, 4096), (4096, 14336), (4096, 14336), (14336, 4096)], 16)', hidden: true }
        ]
      },
      {
        id: 'lora-forward', title: 'LoRA Forward Pass and Merge', difficulty: 'medium',
        prompt: `With row-vector inputs x (n, d_in), a frozen weight W (d_in, d_out) and LoRA matrices A (d_in, r) and B (r, d_out), where r = A.shape[1]:

- \`lora_forward(x, W, A, B, alpha)\` → x @ W + (alpha / r) · (x @ A) @ B
- \`merge_lora(W, A, B, alpha)\` → a **new** matrix W + (alpha / r) · A @ B (do not modify W), so that \`x @ merge_lora(...)\` equals \`lora_forward(...)\`.

In the forward pass compute \`(x @ A) @ B\` — two thin matrix products — instead of forming the full A @ B.

\`\`\`text
With B = 0:  lora_forward(x, W, A, B, alpha) == x @ W
\`\`\``,
        starter: `
import numpy as np

def lora_forward(x, W, A, B, alpha):
    # TODO
    pass

def merge_lora(W, A, B, alpha):
    # TODO
    pass
`,
        solution: `
import numpy as np

def lora_forward(x, W, A, B, alpha):
    r = A.shape[1]
    return x @ W + (alpha / r) * ((x @ A) @ B)

def merge_lora(W, A, B, alpha):
    r = A.shape[1]
    return W + (alpha / r) * (A @ B)
`,
        hints: ['Read r from `A.shape[1]`.', '`W + ...` creates a new array; `W += ...` would modify the caller\'s weights.'],
        explain: `(x A) B costs n·r·(d_in + d_out) operations instead of d_in·d_out for forming A B — a big saving when r is small. Merging folds the update into W once, so the deployed model has no extra cost.`,
        tests: [
          { name: 'zero B gives the base model', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(size=(3, 6)); W = rng.normal(size=(6, 5)); A = rng.normal(size=(6, 2)); B = np.zeros((2, 5))', call: 'lora_forward(x, W, A, B, 4)', expect: 'x @ W' },
          { name: 'small example', call: 'lora_forward(np.array([[1.0, 2.0]]), np.eye(2), np.array([[1.0], [1.0]]), np.array([[1.0, -1.0]]), 2)', expect: 'np.array([[7.0, -4.0]])' },
          { name: 'merge matches forward', setup: 'rng = np.random.default_rng(1)\nx = rng.normal(size=(4, 8)); W = rng.normal(size=(8, 6)); A = rng.normal(size=(8, 3)); B = rng.normal(size=(3, 6))', code: 'assert np.allclose(x @ merge_lora(W, A, B, 16), lora_forward(x, W, A, B, 16))', hidden: true },
          { name: 'W is not modified', setup: 'rng = np.random.default_rng(2)\nW = rng.normal(size=(5, 5)); A = rng.normal(size=(5, 2)); B = rng.normal(size=(2, 5))', code: 'W0 = W.copy()\nmerge_lora(W, A, B, 8)\nassert np.array_equal(W, W0), "merge_lora must not change W in place"', hidden: true },
          { name: 'random forward vs reference', setup: 'rng = np.random.default_rng(3)\nx = rng.normal(size=(7, 10)); W = rng.normal(size=(10, 4)); A = rng.normal(size=(10, 2)); B = rng.normal(size=(2, 4))', call: 'lora_forward(x, W, A, B, 16)', hidden: true }
        ]
      },
      {
        id: 'lora-train', title: 'Train a LoRA Adapter', difficulty: 'hard',
        prompt: `A frozen weight W must be adapted so that X @ W' ≈ Y. Implement \`train_lora(X, Y, W, r, alpha, steps, lr, seed=0)\` with plain gradient descent on the mean squared error, updating **only** A and B:

- **init**: \`rng = np.random.default_rng(seed)\`, \`A = rng.normal(0, 0.01, (d_in, r))\`, \`B = np.zeros((r, d_out))\`; s = alpha / r.
- **each step**: H = X @ A; err = X @ W + s · H @ B − Y; loss = mean(err²) (record it *before* updating).
- **gradients** (N = err.size), both computed from the current A and B: dB = s · Hᵀ @ err · 2/N and dA = s · Xᵀ @ (err @ Bᵀ) · 2/N.
- **update**: A −= lr · dA, B −= lr · dB. Never modify W.

Return \`(A, B, losses)\` where \`losses\` is the list of per-step losses (Python floats).`,
        starter: `
import numpy as np

def train_lora(X, Y, W, r, alpha, steps, lr, seed=0):
    rng = np.random.default_rng(seed)
    d_in, d_out = W.shape
    # TODO: initialise A and B, then run gradient descent on A and B only
    pass
`,
        solution: `
import numpy as np

def train_lora(X, Y, W, r, alpha, steps, lr, seed=0):
    rng = np.random.default_rng(seed)
    d_in, d_out = W.shape
    A = rng.normal(0, 0.01, (d_in, r))
    B = np.zeros((r, d_out))
    s = alpha / r
    losses = []
    for _ in range(steps):
        H = X @ A
        err = X @ W + s * H @ B - Y
        losses.append(float((err ** 2).mean()))
        N = err.size
        dB = s * H.T @ err * 2 / N
        dA = s * X.T @ (err @ B.T) * 2 / N
        A = A - lr * dA
        B = B - lr * dB
    return A, B, losses
`,
        hints: ['Compute both gradients before updating either matrix.', 'At step 0, B = 0 so dA = 0: only B moves first, then A starts learning.', 'The first recorded loss is the base model\'s loss: mean((X @ W − Y)²).'],
        explain: `Notice that W never receives a gradient — that's the whole point. The chain rule through H = X A gives dA = Xᵀ (dpred Bᵀ), the same pattern as backprop through two stacked linear layers.`,
        tests: [
          { name: 'loss drops by over 1000x', setup: 'rng = np.random.default_rng(1)\nW = rng.normal(0, 0.1, (16, 12))\ndelta = rng.normal(0, 0.3, (16, 2)) @ rng.normal(0, 0.3, (2, 12))\nX = rng.normal(size=(200, 16)); Y = X @ (W + delta)', code: 'A, B, losses = train_lora(X, Y, W, 2, 4, 300, 0.5)\nassert len(losses) == 300\nassert losses[-1] < losses[0] / 1000, (losses[0], losses[-1])\nassert A.shape == (16, 2) and B.shape == (2, 12)' },
          { name: 'starts at the base model', setup: 'rng = np.random.default_rng(2)\nW = rng.normal(0, 0.1, (8, 8))\nX = rng.normal(size=(50, 8)); Y = X @ (W + 0.05)', code: 'A, B, losses = train_lora(X, Y, W, 2, 2, 5, 0.5)\nassert np.isclose(losses[0], np.mean((X @ W - Y) ** 2))' },
          { name: 'W is never modified', setup: 'rng = np.random.default_rng(3)\nW = rng.normal(0, 0.1, (8, 6)); X = rng.normal(size=(40, 8)); Y = rng.normal(size=(40, 6))', code: 'W0 = W.copy()\ntrain_lora(X, Y, W, 2, 4, 20, 0.5)\nassert np.array_equal(W, W0)', hidden: true },
          { name: 'matches the reference run', setup: 'rng = np.random.default_rng(4)\nW = rng.normal(0, 0.1, (10, 7))\ndelta = rng.normal(0, 0.3, (10, 1)) @ rng.normal(0, 0.3, (1, 7))\nX = rng.normal(size=(120, 10)); Y = X @ (W + delta)', call: 'train_lora(X, Y, W, 2, 4, 60, 0.5)[2]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Hu et al. — LoRA: Low-Rank Adaptation of Large Language Models', url: 'https://arxiv.org/abs/2106.09685' },
      { title: 'Dettmers et al. — QLoRA: Efficient Finetuning of Quantized LLMs', url: 'https://arxiv.org/abs/2305.14314' },
      { title: 'Hugging Face — PEFT documentation', url: 'https://huggingface.co/docs/peft' }
    ]
  },

  // =====================================================================================
  {
    id: 'alignment', branch: 'llm', title: 'RLHF, DPO & Alignment', short: 'RLHF & DPO', level: 3, minutes: 45, prereqs: ['fine-tuning', 'deep-rl'],
    tags: ['alignment', 'rlhf', 'dpo', 'reward model', 'ppo', 'preference data', 'constitutional ai', 'bradley-terry', 'kl penalty', 'reward hacking'],
    summary: 'Teaching models to be helpful, honest and harmless from preferences: preference data, reward models (Bradley-Terry), RLHF with PPO and a KL penalty, Direct Preference Optimization, AI feedback and verifiable rewards — and failure modes like reward hacking.',
    why: `SFT teaches the *format* of good answers, but many qualities — helpfulness, honesty, tone, declining harmful requests, not making things up — are much easier to **judge** than to **demonstrate**. Preference-based training (RLHF, DPO) uses comparisons ("answer A is better than B") to optimize directly for what people prefer. It is what turned GPT-3 into ChatGPT, it underpins every modern assistant, and it is where a lot of AI safety work happens.`,
    objectives: [
      'Explain the RLHF pipeline: SFT → preference data → reward model → RL with a KL penalty',
      'Train a reward model with the Bradley-Terry pairwise loss',
      'Compute the DPO loss from policy and reference log-probabilities',
      'Describe reward hacking and over-optimization, and alternatives such as Constitutional AI and verifiable rewards'
    ],
    sections: [
      { type: 'text', title: 'Why preferences?', md: `It's hard to write the perfect answer to "Explain quantum computing to a ten-year-old", but easy to say which of two answers is better. Preference data looks like:

\`\`\`text
prompt:    "Explain quantum computing to a ten-year-old."
chosen:    "Imagine a coin spinning in the air..."        (preferred by the labeler)
rejected:  "Quantum computing leverages superposition..."
\`\`\`

The responses are usually sampled from the model itself and compared by human labelers or by an AI judge. In the InstructGPT paper (2022), people preferred the outputs of a 1.3B-parameter model trained this way over the 175B-parameter GPT-3.` },
      { type: 'text', title: 'Step 1 — a reward model', md: `A reward model r(x, y) — typically the SFT model with its LM head replaced by a single scalar output — scores how good response y is for prompt x. It's trained on comparisons with the **Bradley-Terry** model of preferences:

$$ P(chosen ≻ rejected) = σ( r(x, y_c) − r(x, y_r) ) $$
$$ loss = −log σ( r_c − r_r ) $$
:::where
x — the prompt
y_c, y_r — the chosen (preferred) and the rejected response
r(x, y) — the reward model's score for response y to prompt x; r_c and r_r are short for the two scores
≻ — "is preferred to"
σ — the sigmoid, σ(z) = 1 / (1 + e^(−z)), which turns a score difference into a probability
loss — the negative log-probability that the reward model ranks the pair correctly
:::

Only the *difference* between the two scores matters: the loss pushes the chosen response's score above the rejected one's. A trained reward model is judged by its pairwise accuracy on held-out comparisons (often only 65–75% — people disagree too).` },
      { type: 'text', title: 'Step 2 — reinforcement learning with a KL leash', md: `Next, the policy (the LLM) is optimized with reinforcement learning — usually **PPO** — to maximize the reward on fresh prompts: each generated response is an episode, scored by the reward model. A **KL penalty** keeps the policy close to the SFT reference model:

$$ maximize   E[ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) ) $$
:::where
π(·|x) — "pi": the policy (the LLM being trained), seen as a probability distribution over responses to prompt x
π_ref — the frozen reference model (the SFT model)
E[ r(x, y) ] — the average reward of responses sampled from the policy
KL( π ‖ π_ref ) — the KL divergence: how far the policy has drifted from the reference
β — "beta": how strongly drifting is penalized
:::

In practice the reward for a sampled response is r(x, y) − β · Σₜ (log π(yₜ) − log π_ref(yₜ)).

Without the leash the policy finds **reward hacking** exploits: outputs the reward model scores highly that people don't actually want — excessive length, flattery, confident-sounding nonsense. This is Goodhart's law: *when a measure becomes a target, it ceases to be a good measure.*` },
      { type: 'code', title: 'Reward-model loss and a KL-penalized reward', code: `
import numpy as np
def log_sigmoid(x):
    return -np.logaddexp(0, -x)              # stable log(sigmoid(x))

# reward-model scores for 5 preference pairs
r_chosen = np.array([2.1, 0.3, 1.5, -0.2, 0.9])
r_rejected = np.array([0.4, 0.8, -1.0, -0.5, 0.9])
print("P(chosen preferred):", (1 / (1 + np.exp(-(r_chosen - r_rejected)))).round(3))
print("Bradley-Terry loss: ", round(float(-log_sigmoid(r_chosen - r_rejected).mean()), 4))
print("pairwise accuracy:  ", (r_chosen > r_rejected).mean())

# RLHF reward with a KL penalty for one sampled response (per-token log-probabilities)
logp_policy = np.array([-0.2, -1.1, -0.4, -0.9])
logp_ref = np.array([-0.5, -1.3, -0.6, -2.5])
beta, rm_score = 0.1, 1.8
kl = float((logp_policy - logp_ref).sum())
print(f"KL estimate {kl:.2f}  ->  penalized reward {rm_score - beta * kl:.3f} (raw reward-model score {rm_score})")
` },
      { type: 'check', question: { type: 'mcq', q: 'In the Bradley-Terry reward model, what determines P(chosen ≻ rejected)?', options: ['The lengths of the responses', 'The sigmoid of the difference between their reward scores', 'The product of their rewards', 'The KL divergence'], answer: 1, explain: 'σ(r_c − r_r): equal scores give 50%.' } },
      { type: 'text', title: 'DPO: preference tuning without RL', md: `**Direct Preference Optimization** (Rafailov et al., 2023) showed that the RLHF objective (reward minus β·KL) has a closed-form optimal policy. Rearranged, it lets you skip the reward model and PPO entirely and train the policy directly on preference pairs with a classification-style loss:

$$ margin = β · [ (log π(y_c|x) − log π_ref(y_c|x)) − (log π(y_r|x) − log π_ref(y_r|x)) ] $$
$$ loss = −log σ( margin ) $$
:::where
x — the prompt
y_c, y_r — the chosen and the rejected response
log π(y|x) — the summed log-probability of response y's tokens under the model being trained
log π_ref(y|x) — the same quantity under the frozen reference model
β — "beta": the strength of the leash that keeps the policy near the reference (typically 0.1)
σ — the sigmoid
margin — how much more the policy, relative to the reference, favours the chosen response over the rejected one
:::

Here log π(y|x) is the **summed** log-probability of the whole response (prompt tokens masked) under the model being trained, and π_ref is the frozen SFT model. The term β·(log π − log π_ref) acts as an **implicit reward**. DPO raises the likelihood of chosen responses relative to rejected ones, measured against the reference so the model can't drift arbitrarily far. It is stable and cheap — no sampling during training, just forward passes — and is widely used for open models; variants include IPO, KTO, ORPO and SimPO.` },
      { type: 'code', title: 'Computing the DPO loss', code: `
import numpy as np
def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    return float(np.logaddexp(0, -margin).mean()), margin

# summed log-probs of the chosen / rejected responses for 3 pairs under the frozen reference
ref_c, ref_r = np.array([-42.0, -30.5, -55.0]), np.array([-40.0, -31.0, -50.0])
for name, pi_c, pi_r in [("policy = reference", ref_c, ref_r),
                         ("after some DPO steps", ref_c + np.array([3.0, 1.0, 4.0]), ref_r - np.array([2.0, 1.5, 3.0]))]:
    loss, m = dpo_loss(pi_c, pi_r, ref_c, ref_r)
    print(f"{name:22s} loss {loss:.4f}   margins {m.round(2)}")
`, after: `At the start the policy equals the reference, every margin is 0 and the loss is ln 2 ≈ 0.693. Making chosen responses more likely and rejected ones less likely (relative to the reference) grows the margins and lowers the loss.` },
      { type: 'text', title: "Goodhart's law: optimising a proxy", md: `"When a measure becomes a target, it ceases to be a good measure." A reward model is a learned approximation of human preferences, trained on a limited set of comparisons. Optimise hard against it, and the policy finds the places where the approximation is wrong. Known patterns:

- **Length bias**: raters often prefer longer, more detailed answers, so the reward model learns "longer is better" and the policy pads its answers.
- **Sycophancy**: agreeing with and flattering the user earns approval, even when the user is wrong.
- **Style over substance**: answers that sound authoritative score well whether or not they are correct.
- **Exploits**: in reinforcement learning on code or tasks, the policy may game the checker (special-casing the tests) instead of solving the problem.

Mitigations: the KL penalty (stay close to the reference model), ensembles of reward models, refreshing the reward model with comparisons of the current policy's outputs, length-controlled evaluation, and, where possible, **verifiable rewards** such as unit tests or exact answers, which cannot be flattered.` },
      { type: 'code', title: "Best-of-n against a flawed reward model", md: `Best-of-n sampling is the simplest way to optimise against a reward model: generate n answers and keep the one it scores highest. Here each answer's true quality is random, and the reward model sees that quality plus an error that is usually small but occasionally huge.`, code: `import numpy as np
rng = np.random.default_rng(0)
trials, true_quality = 3000, {}
for n in (1, 4, 16, 64, 256, 1024):
    quality = rng.normal(size=(trials, n))                       # true quality of each candidate
    error = rng.standard_t(2, size=(trials, n)) * 0.6            # heavy-tailed reward-model error
    score = quality + error                                      # what the reward model reports
    best = np.argmax(score, axis=1)                              # keep the highest-scoring candidate
    rows = np.arange(trials)
    true_quality[n] = quality[rows, best].mean()
    print(f"n = {n:5d}: reward-model score {score[rows, best].mean():6.2f}   true quality {true_quality[n]:+.2f}")
print("true quality peaks before the largest n:", true_quality[1024] < max(true_quality.values()))`, after: `The reward model's score for the chosen answer climbs steadily, from 0 to about 25. True quality rises at first, to about +1 standard deviation at n = 16, then falls back towards 0 (about +0.2 at n = 1,024): the larger the pool, the more likely the winner is an answer whose reward-model error is enormous rather than one that is genuinely good. Light optimisation against a proxy helps; heavy optimisation hurts.`, predict: {"type":"mcq","q":"As n grows from 16 to 1,024, what happens to the true quality of the answer the reward model picks?","options":["It keeps rising","It falls","It stays exactly the same","It drops to random at once"],"answer":1,"explain":"With many candidates, the top score increasingly belongs to an answer whose reward-model error happens to be huge, not to a genuinely better answer.","expect":["true quality peaks before the largest n: True"]} },
      { type: 'chart', title: "Best-of-n: the proxy reward keeps rising, the true quality does not", md: `Sample n answers, keep the one the reward model likes best. Each answer's true quality is random; the reward model sees it plus an error that is usually small but occasionally huge (heavy-tailed):`, after: `Choosing from more candidates always raises the reward model's score of the winner. True quality improves at first, peaking at about 1.00 standard deviations with n ≈ 16, and then declines: with many candidates the winner is increasingly an answer the reward model merely over-rates. That is reward hacking in its simplest form, and why RLHF limits how far the policy may move (the KL penalty) and why reward models are re-trained on the policy's new outputs.`, chart: {"kind":"line","x":[1,2,4,8,16,32,64,128,256,512,1024],"xLog":true,"series":[{"name":"reward-model score of the chosen answer","y":[0.009,0.901,1.747,2.54,3.582,4.625,6.154,8.944,12.938,17.965,23.949]},{"name":"true quality of the chosen answer","y":[-0.034,0.414,0.732,0.947,1.004,0.972,0.836,0.617,0.444,0.271,0.177]}],"xLabel":"n, the number of candidate answers to choose from (log scale)","yLabel":"average score (true quality has mean 0, sd 1)"} },
      { type: 'text', title: 'Beyond human labels', md: `- **RLAIF and Constitutional AI** (Anthropic, 2022): an AI model critiques, revises and compares responses according to a written set of principles — a "constitution" — producing feedback at scale. People write and refine the principles instead of labelling every pair.
- **RL with verifiable rewards**: for math and code the reward can be checked automatically (is the final answer right? do the unit tests pass?). Large-scale RL on such tasks — for example with GRPO, a PPO variant without a value network — produced "reasoning" models that learn to work through problems step by step.
- **Open problems**: reward hacking, sycophancy (telling users what they want to hear), evaluating outputs that are hard for people to check, and keeping models honest about uncertainty. Alignment is an active research field, not a solved pipeline step.` },
      { type: 'check', question: { type: 'tf', q: 'DPO requires training a separate reward model and running PPO.', answer: false, explain: 'DPO optimizes the policy directly from preference pairs, with the reference model providing the implicit reward.' } }
    ],
    when: `- After SFT, when quality depends on judgments (helpfulness, tone, safety) more than on one correct answer.
- **DPO** (or a variant) as a simple, stable default for preference tuning of open models.
- **PPO / GRPO-style online RL** when you have a trustworthy reward signal (verifiable tasks or a strong reward model) and the infrastructure to sample during training.
- Your own users' comparisons (thumbs up/down, A/B choices) as preference data.`,
    pitfalls: `- Over-optimizing a reward model (reward hacking): track KL and human evaluations, not just reward.
- β too small → drift and degraded text; too large → hardly any change.
- Noisy or biased labels (labelers favouring longer or more confident answers).
- Summing DPO log-probs over prompt tokens too — mask the prompt.
- Evaluating only with the same reward model you optimized against.`,
    takeaways: [
      'Preference data = (prompt, chosen, rejected); easier to collect than perfect demonstrations.',
      'Reward models are trained with the Bradley-Terry loss −log σ(r_c − r_r).',
      'RLHF maximizes reward − β·KL(π ‖ π_ref) with PPO; the KL leash limits reward hacking.',
      'DPO skips the reward model: loss = −log σ(β·[(log π_c − log ref_c) − (log π_r − log ref_r)]).',
      'Modern pipelines add AI feedback (Constitutional AI) and verifiable rewards for reasoning.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the KL penalty in RLHF do?', options: ['Speeds up training', 'Keeps the policy close to the reference model, limiting reward hacking and preserving fluency', 'Normalizes the rewards', 'Removes the need for preference data'], answer: 1, explain: 'It charges the policy for drifting away from the SFT model.' },
      { type: 'fill', q: 'In DPO, if the policy equals the reference model, what is the loss for every pair? (natural log, 3 decimals)', answer: ['0.693', 'ln2', 'ln 2', '0.6931'], display: 'ln 2 ≈ 0.693', explain: 'The margin is 0, and −log σ(0) = log 2.' },
      { type: 'mcq', q: 'What is reward hacking?', options: ['Stealing reward-model weights', 'The policy exploiting flaws in the reward model to score highly without really being better', 'Using a higher learning rate', 'Labelling data with AI'], answer: 1, explain: 'Goodhart\'s law in action.' },
      { type: 'tf', q: 'In Constitutional AI, an AI model uses written principles to help generate feedback and preference labels.', answer: true, explain: 'People supply the principles; the model applies them at scale.' },
      { type: 'mcq', q: 'Which of these is a verifiable reward?', options: ['A human rating of politeness', 'Whether generated code passes unit tests', 'A reward model\'s score', 'The response length'], answer: 1, explain: 'It can be checked automatically and objectively.' },
      {"type":"mcq","q":"A reward model has learned that longer answers are usually preferred. What is the likely effect of optimising a policy hard against it?","options":["Shorter, crisper answers","Padded, overly long answers: a form of reward hacking","No effect","Better factual accuracy"],"answer":1,"explain":"The policy exploits whatever the reward model over-values; length is a classic example, which is why evaluations control for it."},
      {"type":"tf","q":"In best-of-n sampling against an imperfect reward model, choosing from more candidates always gives better answers.","answer":false,"explain":"Beyond some n, the winner is increasingly an answer the reward model over-rates: the proxy keeps rising while true quality falls."}
    ],
    problems: [
      {
        id: 'bradley-terry', title: 'Reward-Model Loss', difficulty: 'easy',
        prompt: `Implement, for scalars or NumPy arrays of reward scores:

- \`preference_prob(r_chosen, r_rejected)\` → σ(r_chosen − r_rejected).
- \`reward_loss(r_chosen, r_rejected)\` → the **mean** of −log σ(r_chosen − r_rejected), computed stably as \`np.logaddexp(0, -(r_chosen - r_rejected))\`.
- \`pairwise_accuracy(r_chosen, r_rejected)\` → the fraction of pairs where r_chosen > r_rejected.

\`\`\`text
preference_prob(1.0, 1.0)                        -> 0.5
reward_loss(np.array([0.0]), np.array([0.0]))    -> ln 2 ≈ 0.6931
\`\`\``,
        starter: `
import numpy as np

def preference_prob(r_chosen, r_rejected):
    # TODO
    pass

def reward_loss(r_chosen, r_rejected):
    # TODO
    pass

def pairwise_accuracy(r_chosen, r_rejected):
    # TODO
    pass
`,
        solution: `
import numpy as np

def preference_prob(r_chosen, r_rejected):
    return 1 / (1 + np.exp(-(np.asarray(r_chosen, dtype=float) - r_rejected)))

def reward_loss(r_chosen, r_rejected):
    diff = np.asarray(r_chosen, dtype=float) - np.asarray(r_rejected, dtype=float)
    return float(np.logaddexp(0, -diff).mean())

def pairwise_accuracy(r_chosen, r_rejected):
    return float(np.mean(np.asarray(r_chosen) > np.asarray(r_rejected)))
`,
        hints: ['−log σ(x) = log(1 + e^(−x)) = `np.logaddexp(0, -x)`.', 'The mean of a boolean array is the fraction of True values.'],
        explain: `\`np.logaddexp(0, -x)\` never overflows: for a huge negative margin it returns ≈ −x instead of computing log(1 + e^1000). The same stable form appears in logistic regression and in the DPO loss.`,
        tests: [
          { call: 'preference_prob(1.0, 1.0)', expect: '0.5' },
          { call: 'reward_loss(np.array([0.0]), np.array([0.0]))', expect: 'np.log(2)' },
          { call: 'pairwise_accuracy(np.array([2.0, 0.0, 1.0, 3.0]), np.array([1.0, 1.0, 0.5, 3.0]))', expect: '0.5' },
          { name: 'large margins stay finite', call: 'reward_loss(np.array([1000.0, -1000.0]), np.array([0.0, 0.0]))', expect: '500.0', hidden: true },
          { name: 'random scores vs reference', setup: 'rng = np.random.default_rng(0)\nrc = rng.normal(size=40) + 0.5\nrr = rng.normal(size=40)', call: '[preference_prob(rc, rr), reward_loss(rc, rr), pairwise_accuracy(rc, rr)]', hidden: true }
        ]
      },
      {
        id: 'dpo-loss', title: 'The DPO Loss', difficulty: 'medium',
        prompt: `The inputs are arrays (one entry per preference pair) of **summed** log-probabilities of the chosen and rejected responses under the policy (\`pi_c\`, \`pi_r\`) and under the frozen reference model (\`ref_c\`, \`ref_r\`). Implement \`dpo(pi_c, pi_r, ref_c, ref_r, beta=0.1)\` returning a dict of Python floats:

- \`"loss"\`: mean of −log σ(margin), with margin = β · ((pi_c − ref_c) − (pi_r − ref_r)) — use \`np.logaddexp(0, -margin)\`
- \`"chosen_reward"\`: mean of β · (pi_c − ref_c), and \`"rejected_reward"\`: mean of β · (pi_r − ref_r) — the implicit rewards
- \`"accuracy"\`: fraction of pairs with margin > 0

\`\`\`text
policy == reference  ->  loss = ln 2 ≈ 0.6931, both rewards 0, accuracy 0
\`\`\``,
        starter: `
import numpy as np

def dpo(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    # TODO
    pass
`,
        solution: `
import numpy as np

def dpo(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    pi_c, pi_r, ref_c, ref_r = (np.asarray(a, dtype=float) for a in (pi_c, pi_r, ref_c, ref_r))
    chosen = beta * (pi_c - ref_c)
    rejected = beta * (pi_r - ref_r)
    margin = chosen - rejected
    return {
        "loss": float(np.logaddexp(0, -margin).mean()),
        "chosen_reward": float(chosen.mean()),
        "rejected_reward": float(rejected.mean()),
        "accuracy": float((margin > 0).mean()),
    }
`,
        hints: ['Compute the two implicit rewards first; the margin is their difference.', 'Convert NumPy scalars with `float(...)`.'],
        explain: `These four numbers are exactly what DPO training dashboards (e.g. TRL's DPOTrainer) log. A healthy run shows the loss falling from 0.693, accuracy rising, and the chosen reward climbing above the rejected one.`,
        tests: [
          { name: 'policy equals reference', setup: 'ref_c = np.array([-10.0, -20.0]); ref_r = np.array([-12.0, -18.0])', call: 'dpo(ref_c, ref_r, ref_c, ref_r)', expect: '{"loss": np.log(2), "chosen_reward": 0.0, "rejected_reward": 0.0, "accuracy": 0.0}' },
          { name: 'one pair, beta 0.5', call: 'dpo(np.array([-5.0]), np.array([-9.0]), np.array([-6.0]), np.array([-8.0]), beta=0.5)', expect: '{"loss": np.logaddexp(0, -1.0), "chosen_reward": 0.5, "rejected_reward": -0.5, "accuracy": 1.0}' },
          { name: 'a more likely chosen response lowers the loss', code: 'base = dpo(np.array([-10.0]), np.array([-10.0]), np.array([-10.0]), np.array([-10.0]))["loss"]\nbetter = dpo(np.array([-8.0]), np.array([-10.0]), np.array([-10.0]), np.array([-10.0]))["loss"]\nassert better < base', hidden: true },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(0)\nref_c = rng.uniform(-80, -20, 16); ref_r = rng.uniform(-80, -20, 16)\npi_c = ref_c + rng.normal(1, 2, 16); pi_r = ref_r + rng.normal(-1, 2, 16)', call: 'dpo(pi_c, pi_r, ref_c, ref_r, beta=0.2)', hidden: true }
        ]
      },
      {
        id: 'kl-reward', title: 'KL-Penalized RLHF Reward', difficulty: 'easy',
        prompt: `In RLHF with PPO, the reward for one sampled response is the reward-model score minus a KL penalty that keeps the policy near the reference model. \`logp_policy\` and \`logp_ref\` hold the per-token log-probabilities of the response's tokens under each model.

- \`rlhf_reward(rm_score, logp_policy, logp_ref, beta)\` → rm_score − β · Σₜ (logp_policy[t] − logp_ref[t])
- \`per_token_rewards(rm_score, logp_policy, logp_ref, beta)\` → a NumPy array with −β · (logp_policy[t] − logp_ref[t]) at every token, plus rm_score added to the **last** token (the usual PPO setup)

\`\`\`text
rlhf_reward(2.0, [-1.0, -1.0], [-1.5, -2.0], 0.1)        ->  2.0 − 0.1 · 1.5 = 1.85
per_token_rewards(2.0, [-1.0, -1.0], [-1.5, -2.0], 0.1)  ->  [-0.05, 1.9]
\`\`\``,
        starter: `
import numpy as np

def rlhf_reward(rm_score, logp_policy, logp_ref, beta):
    # TODO
    pass

def per_token_rewards(rm_score, logp_policy, logp_ref, beta):
    # TODO
    pass
`,
        solution: `
import numpy as np

def rlhf_reward(rm_score, logp_policy, logp_ref, beta):
    kl = np.sum(np.asarray(logp_policy, dtype=float) - np.asarray(logp_ref, dtype=float))
    return float(rm_score - beta * kl)

def per_token_rewards(rm_score, logp_policy, logp_ref, beta):
    r = -beta * (np.asarray(logp_policy, dtype=float) - np.asarray(logp_ref, dtype=float))
    r[-1] += rm_score
    return r
`,
        hints: ['Convert the lists to arrays before subtracting.', 'The per-token rewards should sum to the same total as `rlhf_reward`.'],
        explain: `Σ (log π − log π_ref) over sampled tokens is a Monte-Carlo estimate of the KL divergence. Spreading the penalty over tokens and putting the reward-model score on the final token gives PPO a per-step reward signal.`,
        tests: [
          { call: 'rlhf_reward(2.0, [-1.0, -1.0], [-1.5, -2.0], 0.1)', expect: '1.85' },
          { call: 'per_token_rewards(2.0, [-1.0, -1.0], [-1.5, -2.0], 0.1)', expect: 'np.array([-0.05, 1.9])' },
          { name: 'per-token rewards sum to the total', setup: 'rng = np.random.default_rng(0)\na = -rng.uniform(0, 3, 12); b = -rng.uniform(0, 3, 12); s = 0.7', code: 'assert np.isclose(per_token_rewards(s, a, b, 0.05).sum(), rlhf_reward(s, a, b, 0.05))', hidden: true },
          { name: 'random response vs reference', setup: 'rng = np.random.default_rng(1)\na = -rng.uniform(0, 3, 9); b = -rng.uniform(0, 3, 9)', call: '[rlhf_reward(1.2, a, b, 0.2), per_token_rewards(1.2, a, b, 0.2)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Ouyang et al. — Training language models to follow instructions with human feedback', url: 'https://arxiv.org/abs/2203.02155' },
      { title: 'Rafailov et al. — Direct Preference Optimization', url: 'https://arxiv.org/abs/2305.18290' },
      { title: 'Bai et al. — Constitutional AI: Harmlessness from AI Feedback', url: 'https://arxiv.org/abs/2212.08073' }
    ]
  },

  // =====================================================================================
  {
    id: 'quantization', branch: 'llm', title: 'Quantization & Efficient Inference', short: 'Quantization', level: 3, minutes: 40, prereqs: ['gpt'],
    tags: ['quantization', 'int8', 'int4', 'bf16', 'fp16', 'kv cache', 'memory', 'inference', 'speculative decoding', 'distillation'],
    summary: 'Making LLMs fit and run fast: number formats (fp32, bf16, int8, 4-bit), absmax and block-wise quantization, the memory math for weights and the KV cache, and other inference tricks such as batching, speculative decoding and distillation.',
    why: `A 70B-parameter model needs 280 GB in fp32 — but only about 35–40 GB in 4-bit, small enough for a pair of consumer GPUs or a high-end laptop. Quantization is why you can run capable LLMs locally (llama.cpp, Ollama, MLX) and a big reason serving costs keep falling. The memory math (weights + KV cache) tells you whether a model fits on your hardware and how many users it can serve at once.`,
    objectives: [
      'Compare number formats and compute the memory needed for model weights',
      'Implement absmax and block-wise quantization and measure the error',
      'Compute KV-cache memory and explain why it limits context length and batch size',
      'Name other inference optimizations: batching, speculative decoding, distillation, MoE'
    ],
    sections: [
      { type: 'text', title: 'Number formats and memory', md: `| Format | Bytes per parameter | 7B model | 70B model | Typical use |
| --- | --- | --- | --- | --- |
| fp32 | 4 | 28 GB | 280 GB | master weights during training |
| bf16 / fp16 | 2 | 14 GB | 140 GB | standard training and inference |
| int8 / fp8 | 1 | 7 GB | 70 GB | near-lossless with good methods |
| 4-bit (int4, NF4) | 0.5 | 3.5 GB | 35 GB | local inference; small quality loss |

Add a little for scales, activations and the KV cache. **bf16** keeps fp32's exponent range with fewer mantissa bits, so it rarely overflows — that's why it replaced fp16 for training.

During single-stream generation the GPU spends most of its time **reading weights from memory**, not multiplying. Fewer bytes per weight therefore means proportionally faster tokens, not just a smaller download.` },
      { type: 'text', title: 'Absmax quantization and the outlier problem', md: `Map floats to integers with one scale per tensor (or per block):

$$ scale = max |w| / 127          q = round(w / scale) ∈ [−127, 127]          ŵ = q · scale $$
:::where
w — the original floating-point weights
max |w| — the largest absolute value among them
127 — the largest magnitude used for a signed 8-bit integer
q — the stored 8-bit integers
round — round to the nearest whole number
ŵ — "w hat": the reconstructed (dequantized) weights used at run time
:::

int8 q plus one float scale takes about ¼ of fp32's memory, and each weight's rounding error is at most scale / 2.

**Outliers** spoil this: a single large weight sets the scale for everyone, and all the small weights collapse onto a few integer levels. Remedies: **block-wise** quantization (a separate scale for every 32–128 weights — llama.cpp's GGUF formats and bitsandbytes' NF4 do this), keeping outlier features in higher precision (LLM.int8()), and error-aware methods that adjust the remaining weights to compensate for rounding (GPTQ, AWQ).` },
      { type: 'demo', title: 'Interactive: bits, scales and outliers', md: `Lower the bit width and switch on the outlier to see how the reconstruction error changes.`, demo: 'quantization' },
      { type: 'code', title: 'Quantize a weight vector', code: `
import numpy as np
rng = np.random.default_rng(0)
w = rng.normal(0, 0.02, 4096)
w[123] = 0.5                                              # a single outlier

def absmax(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(w).max() / qmax
    return np.clip(np.round(w / scale), -qmax, qmax), scale

def blockwise(w, bits=4, block=64):
    qmax = 2 ** (bits - 1) - 1
    blocks = w.reshape(-1, block)
    scales = np.abs(blocks).max(axis=1, keepdims=True) / qmax
    return (np.clip(np.round(blocks / scales), -qmax, qmax) * scales).ravel()

for bits in (8, 4):
    q, s = absmax(w, bits)
    print(f"per-tensor int{bits}:      mean abs error {np.abs(w - q * s).mean():.6f}   distinct levels used: {len(np.unique(q))}")
print(f"block-wise int4 (64):  mean abs error {np.abs(w - blockwise(w)).mean():.6f}")
`, after: `With one outlier, per-tensor 4-bit quantization rounds almost every weight to zero. Giving each block of 64 weights its own scale cuts the error several-fold at the same bit width.` },
      { type: 'text', title: "Post-training quantization methods", md: `Rounding every weight to the nearest grid point ("round-to-nearest") works well at 8 bits but loses noticeable quality at 4 bits and below. The standard methods do better without any retraining:

| Method | Idea |
| --- | --- |
| group-wise scales | one scale per small block of weights (32 to 128), so an outlier only spoils its own block |
| GPTQ | quantize the weights one column at a time and adjust the not-yet-quantized weights to compensate for each rounding error, using second-order information from a small calibration set |
| AWQ | find the few input channels with large activations, whose weights matter most, and rescale them so that they are quantized more precisely |
| SmoothQuant | move activation outliers into the weights with a per-channel rescaling, so that both can use 8 bits |
| NF4 | a 4-bit grid placed at the quantiles of a normal distribution, matching how weights are distributed (used by QLoRA) |

**Quantization-aware training** goes further: it simulates the rounding during training so the model learns to tolerate it, which recovers more quality at very low bit widths but costs a training run. Whatever the method, check quality on your own tasks: perplexity can hide losses in reasoning, long contexts and less common languages.` },
      { type: 'code', title: "Block-wise scales tame outliers", code: `import numpy as np
rng = np.random.default_rng(0)
w = rng.normal(0, 0.02, 4096 * 64)                       # ordinary weights
w[rng.choice(len(w), 20, replace=False)] = 0.6           # plus 20 large outliers

def quantize(x, bits, block=None):
    qmax = 2 ** (bits - 1) - 1                            # 7 for 4 bits, 127 for 8 bits
    if block is None:
        scale = np.abs(x).max() / qmax                    # one scale for the whole tensor
        return np.round(x / scale) * scale
    xb = x.reshape(-1, block)
    scale = np.abs(xb).max(axis=1, keepdims=True) / qmax  # one scale per block
    return (np.round(xb / scale) * scale).reshape(-1)

def relative_error(q):
    return np.sqrt(np.mean((q - w) ** 2)) / np.sqrt(np.mean(w ** 2))

results = {}
for bits in (8, 4):
    per_tensor, per_block = relative_error(quantize(w, bits)), relative_error(quantize(w, bits, 64))
    results[bits] = (per_tensor, per_block)
    print(f"{bits}-bit: one scale per tensor {per_tensor:6.1%} error   one scale per 64 weights {per_block:6.1%} error")
print(f"weights rounded to zero at 4 bits with one scale: {np.mean(quantize(w, 4) == 0):.0%}")
print("per-block scales cut the 4-bit error:", results[4][1] < results[4][0] / 4)`, after: `With one scale, the step between 4-bit levels is set by the outliers (0.6 / 7 ≈ 0.086), more than four times a typical weight, so about 97% of the weights round to zero and the error is about 92% of the weights' own size. Per-block scales let every block use its full range, cutting the 4-bit error to about 12%. At 8 bits even a single scale is tolerable (about 7%), and per-block scales make it nearly exact.`, predict: {"type":"mcq","q":"A tensor contains a few large outliers. At 4 bits with a single absmax scale, what happens to most ordinary weights?","options":["They are stored almost exactly","They round to zero","They become outliers","They are clipped to the maximum"],"answer":1,"explain":"The outliers set the step between levels to about 0.6 / 7 ≈ 0.086, several times larger than a typical weight, so most weights round to the zero level.","expect":["per-block scales cut the 4-bit error: True"]} },
      { type: 'chart', title: "Quantisation error: one scale per tensor or one per block", md: `262,144 weights drawn from a normal distribution, plus 20 large outliers, quantized with absmax rounding:`, after: `With one scale per tensor, the outliers stretch the range so much that at 4 bits almost every ordinary weight rounds to zero: the error is 92% of the weights' size. Giving every block of 64 weights its own scale confines each outlier's damage to its own block, cutting the 4-bit error to 12%. That is why practical 4-bit formats (GPTQ, AWQ, NF4) all use small groups of weights per scale.`, chart: {"kind":"line","x":[2,3,4,5,6,8],"yLog":true,"series":[{"name":"one scale for the whole tensor","y":[0.9674,0.9674,0.9175,0.5562,0.2701,0.06598]},{"name":"one scale per block of 64 weights","y":[0.6975,0.2526,0.1226,0.0619,0.03002,0.00731]}],"xLabel":"bits per weight","yLabel":"relative error (RMS error ÷ RMS weight, log scale)"} },
      { type: 'check', question: { type: 'mcq', q: 'Why does a single outlier weight hurt per-tensor absmax quantization?', options: ['It overflows int8', 'It makes the scale large, so small weights round onto just a few levels', 'It changes the data type', 'It removes the zero point'], answer: 1, explain: 'The scale is set by max|w|.' } },
      { type: 'text', title: 'The KV cache: the other memory hog', md: `During generation every layer keeps the keys and values of every previous token:

$$ KV bytes = 2 · layers · kv_heads · head_dim · tokens · batch · bytes_per_value $$
:::where
2 — one cached tensor for the keys and one for the values
layers — the number of transformer blocks
kv_heads — the number of key/value heads per layer (fewer than the query heads when GQA is used)
head_dim — the length of each head's vectors
tokens — how many tokens of context are cached
batch — how many sequences are served at once
bytes_per_value — 2 for fp16/bf16, 1 for an 8-bit cache
:::

Llama 2 7B (32 layers, 32 KV heads of 128 dims, fp16) at 4,096 tokens: 2 × 32 × 32 × 128 × 4,096 × 2 bytes = **2 GiB per sequence**. Serve 16 users at once and the cache (32 GiB) is more than twice the weights (14 GB)! That's why grouped-query attention (8 KV heads instead of 32: a 4× smaller cache), KV-cache quantization and memory managers such as vLLM's **PagedAttention** matter so much for long contexts and high throughput.` },
      { type: 'code', title: 'Will it fit? A memory calculator', code: `
def weights_gb(n_params, bits):
    return n_params * bits / 8 / 1e9

def kv_cache_gb(layers, kv_heads, head_dim, tokens, batch=1, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * tokens * batch * bytes_per / 1e9

print("Llama 3 8B weights:  bf16 %.1f GB | int8 %.1f GB | 4-bit %.1f GB" % (weights_gb(8e9, 16), weights_gb(8e9, 8), weights_gb(8e9, 4)))
# Llama 3 8B has 32 layers and 8 KV heads (grouped-query attention) of 128 dims
for ctx in (8_192, 32_768, 131_072):
    print(f"KV cache, {ctx:>7,} tokens: {kv_cache_gb(32, 8, 128, ctx):6.2f} GB per sequence   (with 32 KV heads: {kv_cache_gb(32, 32, 128, ctx):6.2f} GB)")
`, predict: {"type":"mcq","q":"How much memory do the weights of an 8-billion-parameter model take in bf16 (16 bits per weight)?","options":["8 GB","16 GB","32 GB","128 GB"],"answer":1,"explain":"8 × 10⁹ weights × 2 bytes = 16 GB. Halve it for int8 (8 GB) and again for 4-bit (4 GB). Rule of thumb: billions of parameters × 2 = gigabytes in bf16.","expect":["Llama 3 8B weights:  bf16 16.0 GB"]} },
      { type: 'text', title: 'Other inference optimizations', md: `- **Batching / continuous batching** — serve many requests together so each weight read is shared; the main lever for throughput.
- **Speculative decoding** — a small draft model proposes several tokens; the large model checks them all in one forward pass and keeps the accepted prefix. Often 2–3× faster, with the large model's output distribution preserved exactly.
- **Distillation** — train a small "student" to imitate a large "teacher's" output probabilities (DistilBERT, many small chat models).
- **Pruning and sparsity** — remove the least important weights, heads or layers.
- **FlashAttention and fused kernels** — compute attention block by block in fast on-chip memory.
- **Mixture of Experts (MoE)** — only a few expert MLPs run per token, so a model with many parameters costs like a much smaller one per token (Mixtral, DeepSeek-V3); all experts still have to fit in memory.` },
      { type: 'check', question: { type: 'tf', q: 'Speculative decoding changes the model\'s output distribution in exchange for speed.', answer: false, explain: 'With the standard acceptance rule the outputs follow exactly the large model\'s distribution.' } }
    ],
    when: `- Running models locally or on small GPUs: 4-bit (GGUF Q4_K_M, AWQ, GPTQ) or 8-bit weights.
- High-throughput serving: bf16/fp8 weights, continuous batching and a paged KV cache (vLLM, TGI, TensorRT-LLM).
- Long contexts: models with GQA, KV-cache quantization.
- Always re-check quality on your own task after quantizing.`,
    pitfalls: `- Forgetting the KV cache and activations when estimating memory.
- Per-tensor quantization of weights with outliers — use block-wise or outlier-aware methods.
- Going below ~4 bits without checking quality: 2–3-bit models degrade noticeably.
- Comparing speed numbers measured at different batch sizes or context lengths.
- Training large models in fp16 (overflow) — prefer bf16.`,
    takeaways: [
      'Weight memory = parameters × bytes per parameter (fp32 4, bf16 2, int8 1, 4-bit 0.5).',
      'Absmax: scale = max|w| / 127, q = round(w / scale); dequantize with q · scale.',
      'Outliers ruin per-tensor scales; block-wise scales over 32–128 weights fix most of it.',
      'KV cache = 2 · layers · kv_heads · head_dim · tokens · batch · bytes — it can exceed the weights.',
      'Batching, speculative decoding, distillation, MoE and FlashAttention make inference cheaper.'
    ],
    quiz: [
      { type: 'fill', q: 'How many GB do the weights of a 13B-parameter model take in 4-bit (ignore overhead)?', answer: ['6.5', '6.5gb', '6.5 gb'], explain: '13·10⁹ × 0.5 bytes = 6.5 GB.' },
      { type: 'mcq', q: 'Why is bf16 preferred over fp16 for training?', options: ['It uses less memory', 'It has fp32\'s exponent range, so values rarely overflow', 'It is more precise than fp32', 'GPUs cannot use fp16'], answer: 1, explain: 'bf16 trades mantissa precision for range.' },
      { type: 'mcq', q: 'What does grouped-query attention reduce?', options: ['The number of FFN parameters', 'The size of the KV cache', 'The vocabulary size', 'The amount of training data'], answer: 1, explain: 'Fewer distinct key/value heads means fewer cached values per token.' },
      { type: 'tf', q: 'For single-user generation, LLM speed is often limited by memory bandwidth (reading the weights) rather than by arithmetic.', answer: true, explain: 'Each token needs all the weights read once; there is little computation per byte.' },
      { type: 'mcq', q: 'In absmax int8 quantization, the scale is…', options: ['max|w| / 127', 'mean(w)', '1 / 127', 'max|w| × 127'], answer: 0, explain: 'The largest weight maps to ±127.' },
      {"type":"mcq","q":"What does GPTQ do differently from round-to-nearest quantization?","options":["It uses more bits","It adjusts the remaining weights to compensate for each rounding error, guided by calibration data","It deletes outlier weights","It retrains the model from scratch"],"answer":1,"explain":"By correcting later weights for the error made on earlier ones, GPTQ keeps each layer's output close to the original at 4 bits and below."},
      {"type":"tf","q":"Giving each small block of weights its own scale reduces the damage caused by outliers.","answer":true,"explain":"An outlier then only stretches the range of its own block, so the other blocks keep a fine grid."}
    ],
    problems: [
      {
        id: 'absmax-int8', title: 'Absmax int8 Quantization', difficulty: 'easy',
        prompt: `Implement:

- \`quantize(w)\` → \`(q, scale)\` with scale = max|w| / 127 and q = round(w / scale) stored as \`np.int8\`. If all weights are zero, use scale = 1.0.
- \`dequantize(q, scale)\` → q · scale as a float array.

\`\`\`text
quantize(np.array([0.5, -1.0, 0.25]))  ->  (array([ 64, -127,  32], dtype=int8), 1/127)
\`\`\``,
        starter: `
import numpy as np

def quantize(w):
    w = np.asarray(w, dtype=float)
    # TODO
    pass

def dequantize(q, scale):
    # TODO
    pass
`,
        solution: `
import numpy as np

def quantize(w):
    w = np.asarray(w, dtype=float)
    m = np.abs(w).max()
    scale = m / 127 if m > 0 else 1.0
    q = np.round(w / scale).astype(np.int8)
    return q, scale

def dequantize(q, scale):
    return np.asarray(q).astype(float) * scale
`,
        hints: ['`np.abs(w).max()` is the largest magnitude.', 'Round first, then `.astype(np.int8)` (casting alone truncates toward zero).'],
        explain: `Rounding to the nearest level guarantees an error of at most scale / 2 per weight. Casting with \`astype\` before rounding would truncate (e.g. 63.9 → 63) and double the worst-case error.`,
        tests: [
          { call: 'quantize(np.array([0.5, -1.0, 0.25]))', expect: '(np.array([64, -127, 32]), 1 / 127)' },
          { name: 'q is int8', code: 'q, s = quantize(np.array([0.3, -0.7, 0.05]))\nassert q.dtype == np.int8, "store q as np.int8"' },
          { name: 'round-trip error is at most scale / 2', setup: 'rng = np.random.default_rng(0)\nw = rng.normal(0, 0.05, 1000)', code: 'q, s = quantize(w)\nassert np.abs(dequantize(q, s) - w).max() <= s / 2 + 1e-12', hidden: true },
          { name: 'all zeros', call: 'quantize(np.zeros(4))', expect: '(np.zeros(4), 1.0)', hidden: true },
          { name: 'random weights vs reference', setup: 'rng = np.random.default_rng(1)\nw = rng.normal(0, 0.02, 64)', call: 'dequantize(*quantize(w))', hidden: true }
        ]
      },
      {
        id: 'blockwise-quant', title: 'Block-wise 4-bit Quantization', difficulty: 'medium',
        prompt: `Implement \`quantize_blocks(w, block=32, bits=4)\` for a 1-D array whose length is a multiple of \`block\`:

- qmax = 2^(bits − 1) − 1 (7 for 4 bits)
- split w into consecutive blocks; each block's scale = max|block| / qmax (use 1.0 for an all-zero block)
- q = clip(round(w / scale), −qmax, qmax)
- return \`(q, scales)\` with q of shape (n_blocks, block) and scales of shape (n_blocks,)

Also implement \`dequantize_blocks(q, scales)\` → the flat array of q × scale.

\`\`\`text
quantize_blocks(np.array([7.0, -3.0, 1.2, 0.0, 0.7, 0.2]), block=3, bits=4)
-> q = [[7, -3, 1], [0, 7, 2]],  scales = [1.0, 0.1]
\`\`\``,
        starter: `
import numpy as np

def quantize_blocks(w, block=32, bits=4):
    # TODO
    pass

def dequantize_blocks(q, scales):
    # TODO
    pass
`,
        solution: `
import numpy as np

def quantize_blocks(w, block=32, bits=4):
    qmax = 2 ** (bits - 1) - 1
    blocks = np.asarray(w, dtype=float).reshape(-1, block)
    scales = np.abs(blocks).max(axis=1) / qmax
    scales = np.where(scales == 0, 1.0, scales)
    q = np.clip(np.round(blocks / scales[:, None]), -qmax, qmax)
    return q, scales

def dequantize_blocks(q, scales):
    return (np.asarray(q) * np.asarray(scales)[:, None]).ravel()
`,
        hints: ['`w.reshape(-1, block)` gives one row per block.', 'Divide by `scales[:, None]` so each row uses its own scale.'],
        explain: `Real formats store the 4-bit integers packed two per byte plus one 16-bit scale per block: for 32-weight blocks that's 4.5 bits per weight on average. The hidden test shows how much this helps when there is an outlier.`,
        tests: [
          { name: 'two blocks', call: 'quantize_blocks(np.array([7.0, -3.0, 1.2, 0.0, 0.7, 0.2]), block=3, bits=4)', expect: '(np.array([[7.0, -3.0, 1.0], [0.0, 7.0, 2.0]]), np.array([1.0, 0.1]))' },
          { name: 'shapes', code: 'w = np.linspace(-1, 1, 64)\nq, s = quantize_blocks(w, 16)\nassert q.shape == (4, 16) and s.shape == (4,)\nassert dequantize_blocks(q, s).shape == (64,)' },
          { name: 'beats a single scale when there is an outlier', setup: 'rng = np.random.default_rng(0)\nw = rng.normal(0, 0.02, 1024)\nw[5] = 1.0', code: 'q, s = quantize_blocks(w, 32, 4)\nerr_block = np.abs(dequantize_blocks(q, s) - w).mean()\nscale = np.abs(w).max() / 7\nerr_global = np.abs(np.clip(np.round(w / scale), -7, 7) * scale - w).mean()\nassert err_block < err_global / 3, (err_block, err_global)', hidden: true },
          { name: 'all-zero block', call: 'quantize_blocks(np.zeros(8), 4)', expect: '(np.zeros((2, 4)), np.ones(2))', hidden: true },
          { name: 'random weights vs reference', setup: 'rng = np.random.default_rng(1)\nw = rng.normal(0, 0.05, 256)', call: 'dequantize_blocks(*quantize_blocks(w, 64, 4))', hidden: true }
        ]
      },
      {
        id: 'kv-cache-memory', title: 'KV-Cache and Weight Memory', difficulty: 'easy',
        prompt: `Implement:

- \`weight_bytes(n_params, bits)\` → n_params × bits / 8
- \`kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per_value=2)\` → 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_value
- \`max_batch(gpu_bytes, n_params, bits, n_layers, n_kv_heads, head_dim, seq_len, bytes_per_value=2)\` → the largest batch size whose weights plus KV cache fit in \`gpu_bytes\` (0 if not even one sequence fits)

\`\`\`text
kv_cache_bytes(32, 32, 128, 4096)  ->  2147483648     # 2 GiB: Llama 2 7B, one 4k-token sequence in fp16
\`\`\``,
        starter: `
def weight_bytes(n_params, bits):
    # TODO
    pass

def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per_value=2):
    # TODO
    pass

def max_batch(gpu_bytes, n_params, bits, n_layers, n_kv_heads, head_dim, seq_len, bytes_per_value=2):
    # TODO
    pass
`,
        solution: `
def weight_bytes(n_params, bits):
    return n_params * bits / 8

def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per_value=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_value

def max_batch(gpu_bytes, n_params, bits, n_layers, n_kv_heads, head_dim, seq_len, bytes_per_value=2):
    free = gpu_bytes - weight_bytes(n_params, bits)
    per_sequence = kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, 1, bytes_per_value)
    if free < per_sequence:
        return 0
    return int(free // per_sequence)
`,
        hints: ['The leading 2 counts keys and values.', 'Subtract the weights from the GPU memory, then divide by one sequence\'s cache with `//`.'],
        explain: `On an 80 GB GPU, Llama 3 8B in bf16 leaves room for about 59 sequences of 8k tokens — thanks to its 8 KV heads. With 32 KV heads it would be only 14. Real servers also reserve memory for activations, so actual numbers are a bit lower.`,
        tests: [
          { call: 'kv_cache_bytes(32, 32, 128, 4096)', expect: '2147483648' },
          { call: 'weight_bytes(7e9, 4)', expect: '3.5e9' },
          { name: 'Llama 3 8B on an 80 GB GPU', call: 'max_batch(80e9, 8e9, 16, 32, 8, 128, 8192)', expect: '59' },
          { name: 'does not fit', call: 'max_batch(16e9, 13e9, 16, 40, 40, 128, 4096)', expect: '0', hidden: true },
          { name: 'other configurations', call: '[max_batch(24e9, 7e9, 4, 32, 32, 128, 2048), kv_cache_bytes(80, 8, 128, 32768, 4, 1)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Dettmers et al. — LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale', url: 'https://arxiv.org/abs/2208.07339' },
      { title: 'Frantar et al. — GPTQ: Accurate Post-Training Quantization', url: 'https://arxiv.org/abs/2210.17323' },
      { title: 'Kwon et al. — Efficient Memory Management for LLM Serving with PagedAttention (vLLM)', url: 'https://arxiv.org/abs/2309.06180' },
      { title: 'Leviathan et al. — Fast Inference from Transformers via Speculative Decoding', url: 'https://arxiv.org/abs/2211.17192' }
    ]
  },

  // =====================================================================================
  {
    id: 'mini-gpt', branch: 'llm', title: 'Capstone: A Tiny GPT in NumPy', short: 'Mini-GPT Capstone', level: 3, minutes: 60, prereqs: ['decoding', 'gpt'],
    tags: ['capstone', 'gpt from scratch', 'character-level', 'nanogpt', 'training', 'generation', 'project'],
    summary: 'Put everything together: a character-level tokenizer, a neural language model trained from scratch in your browser, the complete GPT forward pass and generation loop in NumPy, and a full PyTorch script for training a real (small) GPT on a GPU.',
    why: `Building a GPT yourself is the best way to make every previous concept concrete: tokenization, embeddings, attention, residual blocks, the next-token loss, backpropagation, optimizers and decoding all have to work together. Projects like Andrej Karpathy's nanoGPT showed that a few hundred lines are enough to train a GPT-2-class model. Here you'll train a small neural language model in the browser, implement the GPT forward pass and generation, and get a complete PyTorch script to scale it up.`,
    objectives: [
      'Build a character-level tokenizer and (context, next-token) training pairs',
      'Train a small neural language model with manual backpropagation and sample from it',
      'Implement the complete GPT forward pass (embeddings → blocks → tied LM head) and generation with a context window',
      'Know how to scale up: the PyTorch training script, hyperparameters and what changes for bigger models'
    ],
    sections: [
      { type: 'text', title: 'The plan', md: `\`\`\`text
1. Data       text -> characters -> integer ids -> (context, next character) pairs
2. Model      ids -> embeddings -> [transformer blocks] -> logits over the vocabulary
3. Loss       cross-entropy between the logits and the true next character
4. Training   backpropagation + an optimizer, over many random mini-batches
5. Sampling   feed the model its own predictions, one character at a time
\`\`\`

**Character-level** models are perfect for learning: a tiny vocabulary (~65 symbols for English text), no tokenizer to train, and you can watch the samples go from random characters to word-like text as training progresses. Real LLMs use the same pipeline with BPE tokens instead of characters.` },
      { type: 'code', title: 'Step 1 — characters in, characters out', code: `
text = "hello world, hello transformers"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}          # string -> id
itos = {i: c for c, i in stoi.items()}              # id -> string
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
print("vocabulary:", "".join(chars), "| size", len(chars))
print("encode('hello') =", encode("hello"))
print("round trip:", decode(encode("hello world")))

ids = encode(text)
block_size = 8
x, y = ids[0:block_size], ids[1:block_size + 1]       # one window = 8 training examples
for t in range(block_size):
    print(f"context {decode(x[:t + 1])!r:12} -> next {decode([y[t]])!r}")
` },
      { type: 'code', title: 'Step 2 — train a neural language model in your browser', code: `
import numpy as np
text = ("the model reads the text one character at a time. it learns which letters tend to follow which, "
        "then which words follow which. with more data and more layers it learns grammar, facts and style. "
        "the network is small, the data is tiny and the training is short, so the samples will look like "
        "words but will not make much sense. bigger models trained on more text write better text. ")
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
ids = np.array([stoi[c] for c in text])

ctx, emb, hidden, B = 4, 12, 128, 64                 # 4 characters of context -> next character
X = np.stack([ids[i:i + ctx] for i in range(len(ids) - ctx)])
Y = ids[ctx:]
rng = np.random.default_rng(0)
C = rng.normal(0, 0.5, (V, emb))                                         # character embeddings
W1 = rng.normal(0, 1 / np.sqrt(ctx * emb), (ctx * emb, hidden)); b1 = np.zeros(hidden)
W2 = rng.normal(0, 0.01, (hidden, V)); b2 = np.zeros(V)

print(f"vocabulary {V}, examples {len(X)}, loss at init should be about ln V = {np.log(V):.2f}")
for step in range(2001):
    bi = rng.integers(0, len(X), B)                                      # random mini-batch
    e = C[X[bi]].reshape(B, -1)                                          # concatenated embeddings
    h = np.tanh(e @ W1 + b1)
    logits = h @ W2 + b2
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(B), Y[bi]]).mean()
    d = p.copy(); d[np.arange(B), Y[bi]] -= 1; d /= B                    # dLoss/dlogits
    gW2, gb2 = h.T @ d, d.sum(0)
    dh = d @ W2.T * (1 - h ** 2)                                         # back through tanh
    gW1, gb1 = e.T @ dh, dh.sum(0)
    gC = np.zeros_like(C); np.add.at(gC, X[bi], (dh @ W1.T).reshape(B, ctx, emb))
    for param, grad in ((C, gC), (W1, gW1), (b1, gb1), (W2, gW2), (b2, gb2)):
        param -= 0.5 * grad                                              # SGD, learning rate 0.5
    if step % 400 == 0:
        print(f"step {step:4d}   loss {loss:.3f}")

def sample(start, n=160, temperature=0.8):
    out = list(start)
    for _ in range(n):
        x = np.array([stoi[c] for c in out[-ctx:]])
        z = (np.tanh(C[x].reshape(-1) @ W1 + b1) @ W2 + b2) / temperature
        p = np.exp(z - z.max()); p /= p.sum()
        out.append(itos[int(rng.choice(V, p=p))])
    return "".join(out)
print(sample("the "))
`, after: `The loss falls from ln V to near zero: with only a few hundred characters the network memorizes the text, and the sample stitches memorized fragments together. Paste in a few hundred kilobytes of text (and keep a validation split) to watch it generalize instead. This is the classic Bengio et al. (2003) neural language model; a GPT replaces the fixed 4-character window and the MLP with attention over hundreds of positions.` },
      { type: 'chart', title: "The loss curve you should see", md: `The model above, step by step (the mini-batch loss is noisy; the loss on the whole text is smooth):`, after: `The loss starts at ln 25 ≈ 3.22, a uniform guess over the 25 characters. It quickly drops below what letter frequencies alone achieve (2.81), then below the best possible bigram model (1.84), because four characters of context say more than one. By the end it reaches about 0.15: on such a tiny text the network has largely memorised which character follows each 4-character window.`, chart: {"kind":"line","x":[0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075,1100,1125,1150,1175,1200,1225,1250,1275,1300,1325,1350,1375,1400,1425,1450,1475,1500,1525,1550,1575,1600,1625,1650,1675,1700,1725,1750,1775,1800,1825,1850,1875,1900,1925,1950,1975,2000],"points":false,"series":[{"name":"mini-batch loss","y":[3.224,1.988,1.634,1.306,1.169,0.958,0.609,0.41,0.378,0.575,0.407,0.217,0.261,0.138,0.155,0.204,0.131,0.144,0.198,0.283,0.072,0.203,0.123,0.179,0.057,0.23,0.225,0.115,0.167,0.251,0.143,0.171,0.176,0.148,0.161,0.333,0.233,0.2,0.217,0.226,0.135,0.224,0.108,0.197,0.166,0.204,0.117,0.134,0.095,0.205,0.092,0.122,0.153,0.239,0.144,0.151,0.283,0.12,0.217,0.139,0.293,0.158,0.25,0.11,0.167,0.119,0.134,0.104,0.15,0.146,0.191,0.215,0.17,0.212,0.137,0.128,0.283,0.233,0.215,0.097,0.05]},{"name":"loss on the whole text","y":[3.225,2.084,1.653,1.32,1.081,0.863,0.691,0.55,0.418,0.364,0.341,0.288,0.257,0.247,0.22,0.216,0.215,0.222,0.204,0.192,0.185,0.184,0.19,0.179,0.166,0.167,0.181,0.165,0.169,0.174,0.163,0.165,0.183,0.169,0.157,0.163,0.164,0.156,0.157,0.156,0.178,0.156,0.159,0.167,0.152,0.159,0.162,0.152,0.159,0.147,0.154,0.154,0.153,0.161,0.151,0.149,0.153,0.157,0.15,0.148,0.145,0.15,0.151,0.153,0.154,0.152,0.154,0.15,0.159,0.145,0.154,0.148,0.148,0.152,0.147,0.146,0.167,0.16,0.161,0.146,0.147]}],"xLabel":"training step","yLabel":"cross-entropy (nats per character)","yMin":0,"hlines":[{"y":3.219,"label":"uniform guess: ln 25"},{"y":2.81,"label":"letter frequencies only"},{"y":1.845,"label":"best bigram model"}]} },
      { type: 'text', title: "What the loss numbers mean", md: `A loss value means little without reference points. For a character model, three simple baselines frame it:

| Model | Loss (nats per character) | Perplexity |
| --- | --- | --- |
| a uniform guess over V characters | ln V | V |
| letter frequencies only (unigram) | the entropy of the character distribution | e raised to that entropy |
| the previous character only (bigram) | the entropy of each character given the one before | lower again |

Each extra piece of context lowers the achievable loss. A model that does not beat the unigram baseline has learned only which letters are common; one that does not beat the bigram baseline is not using its context window yet. And a training loss far below anything plausible for language (here it ends near 0.15 nats, a perplexity of about 1.16) means memorisation: always compare with a validation set the model never trained on.` },
      { type: 'code', title: "Computing the baselines", code: `import math
from collections import Counter
text = ("the model reads the text one character at a time. it learns which letters tend to follow which, "
        "then which words follow which. with more data and more layers it learns grammar, facts and style. "
        "the network is small, the data is tiny and the training is short, so the samples will look like "
        "words but will not make much sense. bigger models trained on more text write better text. ")
V = len(set(text))
freq = Counter(text)
unigram = -sum(c / len(text) * math.log(c / len(text)) for c in freq.values())
pairs = Counter(zip(text, text[1:]))
firsts = Counter(text[:-1])
bigram = -sum(c * math.log(c / firsts[a]) for (a, b), c in pairs.items()) / (len(text) - 1)
print(f"uniform: {math.log(V):.3f} nats   unigram: {unigram:.3f} nats   bigram: {bigram:.3f} nats")
print(f"perplexities: {V}, {math.exp(unigram):.1f}, {math.exp(bigram):.1f}")
print("more context, lower loss:", math.log(V) > unigram > bigram)`, after: `With 25 distinct characters a uniform guess costs 3.22 nats. Letter frequencies cut that to 2.81, and knowing the previous character to 1.85 (a perplexity of about 6: on average the model chooses among six characters). The 4-character network's final loss, about 0.15, is far below anything genuine language modelling achieves: it has memorised this short text.`, predict: {"type":"mcq","q":"Which baseline gives the lowest loss on this text?","options":["A uniform guess","Letter frequencies (unigram)","The previous character (bigram)","They are all equal"],"answer":2,"explain":"Each additional piece of context can only help: the previous character narrows down the next one far more than overall letter frequencies.","expect":["more context, lower loss: True"]} },
      { type: 'check', question: { type: 'mcq', q: 'In the model above, what does `np.add.at(gC, X[bi], ...)` do?', options: ['Adds noise to the embeddings', 'Accumulates gradient into each embedding row that was used, even if a character appears several times in the batch', 'Normalizes the embeddings', 'Computes the loss'], answer: 1, explain: 'Plain fancy-index assignment would keep only one update per repeated index; add.at sums them.' } },
      { type: 'text', title: 'Step 3 — from MLP to GPT', md: `To turn this into a GPT, replace "concatenate 4 embeddings → MLP" with the transformer stack you built in earlier lessons:

\`\`\`text
x = wte[ids] + wpe[0..T-1]                   (T, d)   token + position embeddings
for each block:                                       pre-norm residual blocks
    x = x + causal_multi_head_attention(layer_norm(x))
    x = x + mlp(layer_norm(x))
logits = layer_norm(x) @ wte.T               (T, V)   tied LM head: a prediction at every position
\`\`\`

What changes:
- the context grows from 4 characters to the **block size** (e.g. 256), and every position is trained at once;
- attention lets each position decide which earlier characters matter, instead of a fixed window;
- during generation you must **crop** the input to the last block_size tokens, because there are no position embeddings beyond that.

The practice problems have you implement exactly this forward pass and generation loop in NumPy. Training it needs backpropagation through attention and LayerNorm — which is where an autograd library earns its keep: the PyTorch version below is about 100 lines.` },
      { type: 'code', title: 'Step 4 — the complete GPT in PyTorch (run on a GPU or in Colab)', lang: 'pytorch', code: `
# A complete character-level GPT in the style of nanoGPT.
#   pip install torch     then put any text file next to this script as input.txt
import torch, torch.nn as nn, torch.nn.functional as F

block_size, batch_size, d_model, n_heads, n_layers = 256, 64, 384, 6, 6
lr, max_steps, dropout = 3e-4, 5000, 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, n_heads, C // n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, V, bias=False)
        self.head.weight = self.tok.weight                         # weight tying
    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=40):
        for _ in range(max_new):
            logits, _ = self(idx[:, -block_size:])                 # crop to the context window
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")            # top-k filtering
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

model = GPT().to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
for step in range(max_steps + 1):
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            val = sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20
        model.train()
        print(f"step {step}: train loss {loss.item():.3f}  val loss {val:.3f}")

model.eval()
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print("".join(itos[i] for i in model.generate(start, 500)[0].tolist()))
` },
      { type: 'text', title: 'Step 5 — scaling up', md: `The script above (~10M parameters) learns to write convincing pseudo-Shakespeare from a 1 MB text file in minutes on a single GPU. Turning it into a real LLM is "just" scaling and engineering:

| Change | Why |
| --- | --- |
| BPE tokenizer (e.g. GPT-2's 50k tokens via tiktoken) | ~4× shorter sequences, so more text per context window |
| Billions of tokens of filtered data (e.g. FineWeb-Edu) | the model learns facts and skills, not just one book's style |
| Bigger model (d_model 768+, 12+ layers) | capacity grows with parameters (scaling laws) |
| Learning-rate warmup + cosine decay | stable training at larger learning rates |
| bf16 autocast, \`torch.compile\`, FlashAttention | 2–4× faster training |
| Gradient accumulation and DistributedDataParallel | big effective batches across many GPUs |
| Checkpoints and evaluations (validation loss, HellaSwag) | track progress and recover from crashes |

Karpathy's nanoGPT and llm.c reproduce GPT-2 (124M) this way for the price of a few hours of rented GPUs. After pretraining, the next steps are the ones in this branch: supervised fine-tuning to follow instructions, preference tuning, quantization for deployment — and then building applications with prompting, RAG and agents.` },
      { type: 'check', question: { type: 'mcq', q: 'During generation, why is the input cropped to the last block_size tokens?', options: ['To save disk space', 'The model has position embeddings for — and was trained on — at most block_size positions', 'To make sampling deterministic', 'Because attention cannot handle short inputs'], answer: 1, explain: 'Position block_size + 1 has no embedding row.' } }
    ],
    when: `- To understand transformers deeply, debug real models, or teach.
- Small character- or byte-level models for toy tasks, research experiments or tiny devices.
- For real applications, start from a pretrained open model and fine-tune it — training from scratch is for research, huge budgets, or very specialized domains.`,
    pitfalls: `- Training on windows in a fixed order instead of random mini-batches.
- Watching only training loss — always keep a validation split to spot memorization.
- Forgetting to crop the context to block_size during generation.
- Too high a learning rate for transformers without warmup (loss spikes, NaNs).
- Expecting a tiny model trained on tiny data to produce meaningful text.`,
    takeaways: [
      'A GPT from scratch = tokenizer + embeddings + causal transformer blocks + LM head + cross-entropy + AdamW + a sampling loop.',
      'Character-level data and small models make every step visible and quick to iterate on.',
      'Manual backpropagation through a small network shows exactly what autograd does for you.',
      'Generation crops the context to the block size and samples with temperature and top-k.',
      'Scaling up = BPE tokens, much more data, a bigger model, mixed precision, LR schedules and distributed training.'
    ],
    quiz: [
      { type: 'mcq', q: 'A character-level model has 65 characters in its vocabulary. About what loss should it start at?', options: ['0', 'about 4.17 (ln 65)', '65', '1'], answer: 1, explain: 'A uniform guess over 65 symbols has cross-entropy ln 65 ≈ 4.17.' },
      { type: 'order', q: 'Order one training step.', items: ['Sample a random batch of (input, target) windows', 'Forward pass to get the logits', 'Compute the cross-entropy loss', 'Backward pass to get the gradients', 'Optimizer step to update the weights'], explain: 'Batch → forward → loss → backward → update, thousands of times.' },
      { type: 'tf', q: 'In the PyTorch GPT, weight tying makes the LM head share its weight matrix with the token embedding.', answer: true, explain: '`self.head.weight = self.tok.weight`.' },
      { type: 'mcq', q: 'Validation loss rises while training loss keeps falling. What is happening?', options: ['Underfitting', 'Overfitting: the model is memorizing the training text', 'A tokenizer bug', 'Perfect training'], answer: 1, explain: 'Stop early, add data, or regularize (dropout, weight decay).' },
      { type: 'mcq', q: 'What does the MLP language model\'s fixed context of 4 characters correspond to in a GPT?', options: ['The vocabulary size', 'The block size, except attention lets every position use all earlier positions in the window', 'The number of layers', 'The batch size'], answer: 1, explain: 'A GPT conditions on up to block_size previous tokens.' },
      {"type":"fill","q":"A character model's vocabulary has 25 characters. What loss (in nats, 2 decimals) does a uniform guess give?","answer":["3.22","3.219","3.2189"],"explain":"A uniform guess gives every character probability 1/25, so the loss is −ln(1/25) = ln 25 ≈ 3.22 nats."}
    ],
    problems: [
      {
        id: 'char-tokenizer', title: 'Character Tokenizer', difficulty: 'easy',
        prompt: `Implement:

- \`build_vocab(text)\` → \`(stoi, itos)\`: the sorted distinct characters of \`text\`; \`stoi\` maps character → id (0, 1, 2, … in sorted order) and \`itos\` maps id → character.
- \`encode(s, stoi)\` → list of ids; \`decode(ids, itos)\` → string.

\`\`\`text
stoi, itos = build_vocab("hello")     # stoi = {"e": 0, "h": 1, "l": 2, "o": 3}
encode("hell", stoi)                  -> [1, 0, 2, 2]
decode([3, 2], itos)                  -> "ol"
\`\`\``,
        starter: `
def build_vocab(text):
    # TODO
    pass

def encode(s, stoi):
    # TODO
    pass

def decode(ids, itos):
    # TODO
    pass
`,
        solution: `
def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos

def encode(s, stoi):
    return [stoi[c] for c in s]

def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)
`,
        hints: ['`sorted(set(text))` gives the vocabulary in order.', 'Use dict comprehensions with `enumerate`.', '`"".join(...)` turns a list of characters back into a string.'],
        explain: `This is the entire tokenizer of a character-level GPT. A BPE tokenizer has the same interface (encode/decode) — just with learned multi-character tokens.`,
        tests: [
          { call: 'build_vocab("hello")', expect: '({"e": 0, "h": 1, "l": 2, "o": 3}, {0: "e", 1: "h", 2: "l", 3: "o"})' },
          { setup: 'stoi, itos = build_vocab("hello")', call: 'encode("hell", stoi)', expect: '[1, 0, 2, 2]' },
          { setup: 'stoi, itos = build_vocab("hello")', call: 'decode([3, 2], itos)', expect: '"ol"' },
          { name: 'round trip', setup: 'text = "To be, or not to be: that is the question.\\n"\nstoi, itos = build_vocab(text)', code: 'assert decode(encode(text, stoi), itos) == text\nassert len(stoi) == len(set(text)) == len(itos)', hidden: true },
          { name: 'random text', setup: 'rng = np.random.default_rng(0)\ntext = "".join(rng.choice(list("abcdefgh .,!"), size=80))', call: 'build_vocab(text)', hidden: true }
        ]
      },
      {
        id: 'neural-bigram', title: 'A Neural Bigram Model', difficulty: 'medium',
        prompt: `Train the simplest neural language model: a single weight matrix W (V × V) whose row \`W[a]\` holds the logits for the character after \`a\`. Implement \`train_bigram(ids, V, steps=200, lr=10.0)\` with full-batch gradient descent:

- x = ids[:-1], y = ids[1:], N = len(x); start from W = zeros((V, V))
- each step: logits = W[x]; stable softmax per row → p; loss = mean(−log p[i, y[i]]) (record it before updating)
- gradient of the loss w.r.t. the logits: d = (p − onehot(y)) / N; accumulate it into the rows of W that were used with \`np.add.at(dW, x, d)\`
- W −= lr · dW

Return \`(W, losses)\`. Also implement \`count_bigram_loss(ids, V)\` → the loss of the count-based model, P(b | a) = count(a, b) / count(a, ·), on the same data. Gradient descent should approach this optimum.`,
        starter: `
import numpy as np

def train_bigram(ids, V, steps=200, lr=10.0):
    ids = np.asarray(ids)
    x, y = ids[:-1], ids[1:]
    N = len(x)
    W = np.zeros((V, V))
    losses = []
    # TODO
    return W, losses

def count_bigram_loss(ids, V):
    # TODO
    pass
`,
        solution: `
import numpy as np

def train_bigram(ids, V, steps=200, lr=10.0):
    ids = np.asarray(ids)
    x, y = ids[:-1], ids[1:]
    N = len(x)
    W = np.zeros((V, V))
    losses = []
    for _ in range(steps):
        z = W[x]
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        losses.append(float(-np.log(p[np.arange(N), y]).mean()))
        d = p.copy()
        d[np.arange(N), y] -= 1
        d /= N
        dW = np.zeros_like(W)
        np.add.at(dW, x, d)
        W -= lr * dW
    return W, losses

def count_bigram_loss(ids, V):
    ids = np.asarray(ids)
    C = np.zeros((V, V))
    np.add.at(C, (ids[:-1], ids[1:]), 1)
    P = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    return float(-np.log(P[ids[:-1], ids[1:]]).mean())
`,
        hints: ['`p[np.arange(N), y]` picks each row\'s probability of the true next character.', 'Softmax + cross-entropy has the gradient p − onehot(y) with respect to the logits.', 'Count pairs with `np.add.at(C, (ids[:-1], ids[1:]), 1)`.'],
        explain: `The trained network converges to the same probabilities you get by counting — a neural network with one-hot inputs and no hidden layer *is* a bigram table. Its advantage only appears when you add embeddings, hidden layers and longer contexts, which let it generalize to combinations it has never counted.`,
        tests: [
          { name: 'starts at ln V and approaches the counting optimum', setup: 'text = "the cat sat on the mat. the dog sat on the log. a cat and a dog are friends."\nchars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}\nids = [stoi[c] for c in text]; V = len(chars)', code: 'W, losses = train_bigram(ids, V, 200, 10.0)\nassert np.isclose(losses[0], np.log(V)), "all-zero logits give loss ln V"\nopt = count_bigram_loss(ids, V)\nassert abs(losses[-1] - opt) < 0.05, (losses[-1], opt)\nassert W.shape == (V, V)' },
          { name: 'count-based loss', call: 'count_bigram_loss([0, 1, 0, 1, 1], 2)', expect: '-(np.log(1.0) + np.log(0.5) + np.log(1.0) + np.log(0.5)) / 4' },
          { name: 'loss trajectory vs reference', setup: 'rng = np.random.default_rng(0)\nids = rng.integers(0, 6, 300)', call: 'train_bigram(ids, 6, 50, 10.0)[1]', hidden: true },
          { name: 'losses never increase on this data', setup: 'rng = np.random.default_rng(1)\nids = rng.integers(0, 4, 200)', code: 'W, losses = train_bigram(ids, 4, 100, 5.0)\nassert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))', hidden: true }
        ]
      },
      {
        id: 'tiny-gpt-forward', title: 'The Complete GPT Forward Pass', difficulty: 'hard',
        prompt: `Implement \`gpt_forward(ids, params, n_heads)\` → logits of shape (T, V) for a 1-D array of token ids (T ≤ the number of position embeddings):

\`\`\`text
x = params["wte"][ids] + params["wpe"][:T]            # token + position embeddings  (T, d)
for p in params["blocks"]:
    x = block(x, p, n_heads)                           # pre-norm transformer block (provided)
x = layer_norm(x, params["lnf_g"], params["lnf_b"])    # final LayerNorm (provided)
logits = x @ params["wte"].T                           # tied LM head  (T, V)
\`\`\`

Then implement \`generate_greedy(ids, params, n_heads, n_new, block_size)\`: start from a copy of \`ids\` (a list) and repeat \`n_new\` times — crop to the last \`block_size\` ids, run \`gpt_forward\`, append the argmax of the **last** row as an \`int\`. Return the full list.

\`layer_norm\` and \`block\` are provided in the starter. The tests build random parameters with a helper \`_make_params(seed)\` (vocabulary 11, width 8, 2 blocks, 6 positions).`,
        prelude: `
import numpy as np

def _make_params(seed, V=11, d=8, n_layers=2, block_size=6):
    r = np.random.default_rng(seed)
    def blk():
        return {"ln1_g": np.ones(d), "ln1_b": np.zeros(d), "ln2_g": np.ones(d), "ln2_b": np.zeros(d),
                "Wq": r.normal(0, 0.4, (d, d)), "Wk": r.normal(0, 0.4, (d, d)), "Wv": r.normal(0, 0.4, (d, d)),
                "Wo": r.normal(0, 0.4, (d, d)), "W1": r.normal(0, 0.4, (d, 4 * d)), "b1": np.zeros(4 * d),
                "W2": r.normal(0, 0.4, (4 * d, d)), "b2": np.zeros(d)}
    return {"wte": r.normal(0, 1.0, (V, d)), "wpe": r.normal(0, 0.5, (block_size, d)),
            "blocks": [blk() for _ in range(n_layers)], "lnf_g": np.ones(d), "lnf_b": np.zeros(d)}
`,
        starter: `
import numpy as np

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def block(x, p, n_heads):
    T, d = x.shape; dh = d // n_heads
    h = layer_norm(x, p["ln1_g"], p["ln1_b"])
    split = lambda M: M.reshape(T, n_heads, dh).transpose(1, 0, 2)
    Q, K, V = split(h @ p["Wq"]), split(h @ p["Wk"]), split(h @ p["Wv"])
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    x = x + (w @ V).transpose(1, 0, 2).reshape(T, d) @ p["Wo"]
    h = layer_norm(x, p["ln2_g"], p["ln2_b"])
    g = h @ p["W1"] + p["b1"]
    g = 0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g ** 3)))
    return x + g @ p["W2"] + p["b2"]

def gpt_forward(ids, params, n_heads):
    ids = np.asarray(ids)
    # TODO: embeddings -> blocks -> final LayerNorm -> tied LM head
    pass

def generate_greedy(ids, params, n_heads, n_new, block_size):
    ids = list(ids)
    # TODO
    return ids
`,
        solution: `
import numpy as np

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def block(x, p, n_heads):
    T, d = x.shape; dh = d // n_heads
    h = layer_norm(x, p["ln1_g"], p["ln1_b"])
    split = lambda M: M.reshape(T, n_heads, dh).transpose(1, 0, 2)
    Q, K, V = split(h @ p["Wq"]), split(h @ p["Wk"]), split(h @ p["Wv"])
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    x = x + (w @ V).transpose(1, 0, 2).reshape(T, d) @ p["Wo"]
    h = layer_norm(x, p["ln2_g"], p["ln2_b"])
    g = h @ p["W1"] + p["b1"]
    g = 0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g ** 3)))
    return x + g @ p["W2"] + p["b2"]

def gpt_forward(ids, params, n_heads):
    ids = np.asarray(ids)
    T = len(ids)
    x = params["wte"][ids] + params["wpe"][:T]
    for p in params["blocks"]:
        x = block(x, p, n_heads)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return x @ params["wte"].T

def generate_greedy(ids, params, n_heads, n_new, block_size):
    ids = list(ids)
    for _ in range(n_new):
        context = ids[-block_size:]
        logits = gpt_forward(context, params, n_heads)
        ids.append(int(np.argmax(logits[-1])))
    return ids
`,
        hints: ['`params["wte"][ids]` looks up one embedding row per token.', 'Slice the position table with `params["wpe"][:T]`.', 'In generation, `ids[-block_size:]` keeps at most block_size tokens; take `logits[-1]`.'],
        explain: `That's a complete GPT inference engine: the same computation (with more layers, wider matrices and trained weights) runs inside every LLM. The causality test confirms that the logits for earlier positions don't depend on later tokens — the property that makes parallel training and the KV cache possible.`,
        tests: [
          { name: 'logits vs reference', setup: 'params = _make_params(1)\nids = np.array([3, 1, 4, 1, 5])', call: 'gpt_forward(ids, params, 2)' },
          { name: 'shape', setup: 'params = _make_params(1)', code: 'out = gpt_forward(np.array([0, 2, 4]), params, 2)\nassert out.shape == (3, 11), out.shape' },
          { name: 'causal: later tokens do not change earlier logits', setup: 'params = _make_params(3)', code: 'a = gpt_forward(np.array([3, 1, 4, 1, 5]), params, 4)\nb = gpt_forward(np.array([3, 1, 4, 1, 9]), params, 4)\nassert np.allclose(a[:4], b[:4])\nassert not np.allclose(a[4], b[4])', hidden: true },
          { name: 'greedy generation', setup: 'params = _make_params(2)', call: 'generate_greedy([1, 2], params, 2, 4, 6)' },
          { name: 'generation longer than the context window', setup: 'params = _make_params(4)', call: 'generate_greedy([0], params, 2, 12, 6)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Karpathy — Let\'s build GPT: from scratch, in code, spelled out (video)', url: 'https://www.youtube.com/watch?v=kCc8FmEb1nY' },
      { title: 'Karpathy — nanoGPT', url: 'https://github.com/karpathy/nanoGPT' },
      { title: 'Karpathy — llm.c (GPT-2 training in raw C/CUDA)', url: 'https://github.com/karpathy/llm.c' },
      { title: 'Bengio et al. — A Neural Probabilistic Language Model (2003)', url: 'https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
